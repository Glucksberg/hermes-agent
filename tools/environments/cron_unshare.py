"""Fail-closed Linux network/PID namespace for guarded cron jobs.

Filesystem access is constrained with Landlock: system binaries and libraries
are readable, the configured workdir is the only writable tree, and host home,
Hermes home, mounts, and credentials remain outside the allowlist.
"""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from hermes_constants import get_hermes_home
from tools.environments.base import BaseEnvironment, _pipe_stdin


_LANDLOCK_CREATE_RULESET = 444
_LANDLOCK_ADD_RULE = 445
_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
_MS_NOSUID = 2
_MS_NODEV = 4
_MS_NOEXEC = 8

_ACCESS_EXECUTE = 1 << 0
_ACCESS_WRITE_FILE = 1 << 1
_ACCESS_READ_FILE = 1 << 2
_ACCESS_READ_DIR = 1 << 3
_ACCESS_REMOVE_DIR = 1 << 4
_ACCESS_REMOVE_FILE = 1 << 5
_ACCESS_MAKE_CHAR = 1 << 6
_ACCESS_MAKE_DIR = 1 << 7
_ACCESS_MAKE_REG = 1 << 8
_ACCESS_MAKE_SOCK = 1 << 9
_ACCESS_MAKE_FIFO = 1 << 10
_ACCESS_MAKE_BLOCK = 1 << 11
_ACCESS_MAKE_SYM = 1 << 12
_ACCESS_REFER = 1 << 13
_ACCESS_TRUNCATE = 1 << 14
_ACCESS_IOCTL_DEV = 1 << 15

_TRUSTED_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_TRUSTED_WORKER = _TRUSTED_SOURCE_ROOT / "tools" / "environments" / "cron_unshare.py"


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


def _syscall(number: int, *args) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(number, *args)
    if result < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return int(result)


def _landlock_abi() -> int:
    return _syscall(
        _LANDLOCK_CREATE_RULESET,
        ctypes.c_void_p(),
        ctypes.c_size_t(0),
        ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
    )


def _handled_access_for_abi(abi: int) -> int:
    access = (1 << 13) - 1
    if abi >= 2:
        access |= _ACCESS_REFER
    if abi >= 3:
        access |= _ACCESS_TRUNCATE
    if abi >= 5:
        access |= _ACCESS_IOCTL_DEV
    return access


def _add_landlock_path(ruleset_fd: int, path: Path, allowed: int) -> None:
    if not path.exists():
        return
    path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        attr = _PathBeneathAttr(
            allowed_access=allowed,
            parent_fd=path_fd,
            reserved=0,
        )
        _syscall(
            _LANDLOCK_ADD_RULE,
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(attr),
            ctypes.c_uint(0),
        )
    finally:
        os.close(path_fd)


def _install_landlock(workdir: Path) -> None:
    abi = _landlock_abi()
    if abi < 1:
        raise RuntimeError("Landlock is unavailable")
    handled = _handled_access_for_abi(abi)
    attr = _RulesetAttr(handled_access_fs=handled)
    ruleset_fd = _syscall(
        _LANDLOCK_CREATE_RULESET,
        ctypes.byref(attr),
        ctypes.sizeof(attr),
        ctypes.c_uint(0),
    )
    read_only = _ACCESS_EXECUTE | _ACCESS_READ_FILE | _ACCESS_READ_DIR
    try:
        for path in (
            "/bin",
            "/sbin",
            "/usr",
            "/lib",
            "/lib64",
            "/proc",
        ):
            _add_landlock_path(ruleset_fd, Path(path), read_only)
        # Do not expose all of /etc: mapped-root user namespaces can otherwise
        # read host credential material such as shadow or service key files.
        # Permit only the public runtime files needed by common dynamically
        # linked commands; networking itself remains disabled by the net ns.
        for path in (
            "/etc/ld.so.cache",
            "/etc/localtime",
            "/etc/os-release",
            "/etc/passwd",
            "/etc/group",
            "/etc/nsswitch.conf",
            "/etc/hosts",
            "/etc/resolv.conf",
        ):
            _add_landlock_path(ruleset_fd, Path(path), _ACCESS_READ_FILE)
        _add_landlock_path(ruleset_fd, Path("/etc/ssl/certs"), read_only)
        for path in ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom"):
            _add_landlock_path(
                ruleset_fd,
                Path(path),
                _ACCESS_READ_FILE | _ACCESS_WRITE_FILE,
            )
        _add_landlock_path(ruleset_fd, workdir, handled)

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        _syscall(
            _LANDLOCK_RESTRICT_SELF,
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint(0),
        )
    finally:
        os.close(ruleset_fd)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _mask_sensitive_paths(
    workdir: Path,
    host_home: Path,
    hermes_home: Path,
) -> None:
    """Overmount sensitive host trees, restoring only an enclosed workdir."""
    candidates = {
        Path("/home"),
        Path("/root"),
        Path("/mnt"),
        Path("/media"),
        Path("/run"),
        Path("/sys"),
        host_home,
        hermes_home,
    }
    existing = sorted(
        (path for path in candidates if path.is_absolute() and path.is_dir()),
        key=lambda path: len(path.parts),
    )
    masks: list[Path] = []
    for path in existing:
        if any(_is_within(path, parent) for parent in masks):
            continue
        masks.append(path)

    workdir_fd = os.open(workdir, os.O_PATH)
    os.set_inheritable(workdir_fd, True)
    libc = ctypes.CDLL(None, use_errno=True)
    for target in masks:
        result = libc.mount(
            ctypes.c_char_p(b"tmpfs"),
            ctypes.c_char_p(os.fsencode(target)),
            ctypes.c_char_p(b"tmpfs"),
            ctypes.c_ulong(_MS_NOSUID | _MS_NODEV | _MS_NOEXEC),
            ctypes.c_char_p(b"mode=755,size=1m"),
        )
        if result != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"failed to mask {target}: {os.strerror(err)}")
        if _is_within(workdir, target):
            workdir.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(f"/proc/self/fd/{workdir_fd}", workdir)

    # The PID namespace is only useful if /proc is remounted from inside it;
    # otherwise the inherited host procfs would still reveal sibling process
    # command lines and environments.
    result = libc.mount(
        ctypes.c_char_p(b"proc"),
        ctypes.c_char_p(b"/proc"),
        ctypes.c_char_p(b"proc"),
        ctypes.c_ulong(_MS_NOSUID | _MS_NODEV | _MS_NOEXEC),
        ctypes.c_void_p(),
    )
    if result != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"failed to mount private procfs: {os.strerror(err)}")


def _launcher_main(
    workdir: str,
    command: str,
    host_home: str,
    hermes_home: str,
) -> None:
    resolved = Path(workdir).resolve(strict=True)
    _mask_sensitive_paths(
        resolved,
        Path(host_home).resolve(strict=False),
        Path(hermes_home).resolve(strict=False),
    )
    # The worker starts from a trusted cwd. Do not enter the attacker-controlled
    # workdir until mount masking and Landlock are both active.
    _install_landlock(resolved)
    os.chdir(resolved)
    env = {
        "HOME": "/nonexistent",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TMPDIR": str(resolved),
    }
    os.execve(
        "/bin/bash",
        ["bash", "--noprofile", "--norc", "-c", command],
        env,
    )


class CronUnshareEnvironment(BaseEnvironment):
    """Execute commands with private network/PID state and Landlock FS rules."""

    _hermes_guarded_cron = True
    _snapshot_timeout = 20

    def __init__(self, workdir: str, timeout: int = 60):
        if sys.platform != "linux":
            raise RuntimeError("cron terminal_sandbox requires Linux")
        if not shutil.which("unshare", path="/usr/local/bin:/usr/bin:/bin"):
            raise RuntimeError("cron terminal_sandbox requires unshare")

        resolved = Path(workdir).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise RuntimeError(f"cron terminal_sandbox workdir is not a directory: {resolved}")
        self.workdir = resolved
        self._reject_trusted_bootstrap_overlap()
        self._reject_sensitive_workdir()
        self._sandbox_temp_dir = resolved / f".hermes-cron-tmp-{uuid.uuid4().hex[:10]}"
        self._sandbox_temp_dir.mkdir(mode=0o700)
        try:
            super().__init__(cwd=str(resolved), timeout=timeout, env={})
            self._strict_probe()
            self.init_session()
            if not self._snapshot_ready:
                raise RuntimeError("cron terminal_sandbox session bootstrap failed")
        except BaseException:
            self.cleanup()
            raise

    def _reject_sensitive_workdir(self) -> None:
        """Never expose an entire host/Hermes home as the writable workdir."""
        sensitive = {
            Path.home().resolve(strict=False),
            get_hermes_home().resolve(strict=False),
            *(Path(path) for path in ("/home", "/root", "/mnt", "/media", "/run", "/sys")),
        }
        for path in sensitive:
            try:
                path.relative_to(self.workdir)
            except ValueError:
                continue
            raise RuntimeError(
                f"cron terminal_sandbox workdir {self.workdir} would expose sensitive home {path}"
            )

    def _reject_trusted_bootstrap_overlap(self, source_root: Path | None = None) -> None:
        """Keep the writable tree disjoint from trusted launcher source."""
        try:
            workdir = Path(self.workdir).resolve(strict=True)
            trusted_root = (
                Path(source_root).resolve(strict=True)
                if source_root is not None
                else Path(_TRUSTED_SOURCE_ROOT).resolve(strict=True)
            )
        except OSError as exc:
            raise RuntimeError(
                f"cron terminal_sandbox trusted bootstrap unavailable: {exc}"
            ) from exc
        if _is_within(workdir, trusted_root) or _is_within(trusted_root, workdir):
            raise RuntimeError(
                "cron terminal_sandbox workdir overlaps trusted bootstrap: "
                f"{workdir} and {trusted_root}"
            )

    def get_temp_dir(self) -> str:
        return str(self._sandbox_temp_dir)

    def _unshare_args(self, command: str) -> list[str]:
        source_root = Path(_TRUSTED_SOURCE_ROOT)
        worker = source_root / "tools" / "environments" / "cron_unshare.py"
        try:
            source_root = source_root.resolve(strict=True)
            worker = worker.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                f"cron terminal_sandbox trusted bootstrap unavailable: {exc}"
            ) from exc
        if (
            not source_root.is_absolute()
            or not source_root.is_dir()
            or worker != Path(_TRUSTED_WORKER).resolve(strict=False)
            or worker != Path(__file__).resolve()
        ):
            raise RuntimeError(
                "cron terminal_sandbox trusted bootstrap unavailable: "
                "worker source root is invalid"
            )
        self._reject_trusted_bootstrap_overlap(source_root)

        # -I ignores PYTHON* environment settings and the user site; -S keeps
        # site.py (and therefore sitecustomize) from running. The only added
        # import root is this module's absolute, repository-owned source root.
        bootstrap = (
            "import sys;"
            f"sys.path.insert(0, {str(source_root)!r});"
            "from tools.environments.cron_unshare import _launcher_main as _launch;"
            "_launch(*sys.argv[1:])"
        )
        return [
            "unshare",
            "--user",
            "--map-root-user",
            "--mount",
            "--net",
            "--pid",
            "--fork",
            "--kill-child",
            sys.executable,
            "-I",
            "-S",
            "-c",
            bootstrap,
            str(self.workdir),
            command,
            str(Path.home().resolve(strict=False)),
            str(get_hermes_home().resolve(strict=False)),
        ]

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> subprocess.Popen:
        proc = subprocess.Popen(
            self._unshare_args(cmd_string),
            text=True,
            encoding="utf-8",
            errors="replace",
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
            },
            cwd="/",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            proc._hermes_pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pass
        if stdin_data is not None:
            _pipe_stdin(proc, stdin_data)
        return proc

    def _strict_probe(self) -> None:
        try:
            abi = _landlock_abi()
        except OSError as exc:
            raise RuntimeError(f"cron terminal_sandbox requires Landlock: {exc}") from exc
        if abi < 1:
            raise RuntimeError("cron terminal_sandbox requires Landlock")
        probe = self._run_bash(
            "test -d . && test ! -w / && test ! -w /usr && "
            "test ! -e /root/.ssh && test ! -e /mnt/c && "
            "test -z \"${HERMES_HOME:-}\" && test -z \"${OPENAI_API_KEY:-}\""
        )
        result = self._wait_for_process(probe, timeout=self._snapshot_timeout)
        if int(result.get("returncode", 1)) != 0:
            detail = str(result.get("output") or result.get("stdout") or "").strip()
            raise RuntimeError(
                "cron terminal_sandbox unshare/Landlock probe failed"
                + (f": {detail}" if detail else "")
            )

    def _kill_process(self, proc) -> None:
        pgid = getattr(proc, "_hermes_pgid", None)
        try:
            if pgid is None:
                pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass

    def cleanup(self) -> None:
        for path in (
            getattr(self, "_snapshot_path", None),
            getattr(self, "_cwd_file", None),
        ):
            if path:
                try:
                    Path(path).unlink()
                except OSError:
                    pass
        temp_dir = getattr(self, "_sandbox_temp_dir", None)
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit("internal cron sandbox launcher requires trusted bootstrap")
