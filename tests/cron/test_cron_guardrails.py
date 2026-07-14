"""Regression tests for per-job cron safety guardrails."""

import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def cron_store(tmp_path, monkeypatch):
    import cron.jobs as jobs

    home = tmp_path / ".hermes"
    (home / "cron" / "output").mkdir(parents=True)
    (home / "scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(jobs, "HERMES_DIR", home)
    monkeypatch.setattr(jobs, "CRON_DIR", home / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", home / "cron" / "output")
    return home


def _guardrails(workdir: Path) -> dict:
    return {
        "script_fail_closed": True,
        "max_iterations": 7,
        "max_tokens": 768,
        "reasoning_effort": "low",
        "max_duration_seconds": 45,
        "max_tool_output_bytes": 1024,
        "max_total_tool_output_bytes": 4096,
        "max_tool_calls": 11,
        "max_files_read": 5,
        "skip_context_files": True,
        "terminal_sandbox": False,
        "restrict_file_tools_to_workdir": True,
        "workdir": str(workdir),
    }


class TestCronGuardrailPersistence:
    def test_create_update_and_reload_preserve_guardrails(self, cron_store, tmp_path):
        from cron.jobs import create_job, get_job, update_job

        workdir = tmp_path / "repo"
        workdir.mkdir()
        job = create_job(
            prompt="safe task",
            schedule="every 1h",
            enabled_toolsets=["file"],
            **_guardrails(workdir),
        )
        loaded = get_job(job["id"])
        for key, value in _guardrails(workdir).items():
            assert loaded[key] == value

        updated = update_job(
            job["id"],
            {
                "max_iterations": 3,
                "max_tokens": 256,
                "reasoning_effort": False,
                "max_duration_seconds": 9,
                "enabled_toolsets": ["terminal"],
                "terminal_sandbox": True,
                "restrict_file_tools_to_workdir": False,
            },
        )
        assert updated["max_iterations"] == 3
        assert updated["max_tokens"] == 256
        assert updated["reasoning_effort"] is False
        assert updated["max_duration_seconds"] == 9
        assert updated["enabled_toolsets"] == ["terminal"]
        assert updated["terminal_sandbox"] is True
        assert updated["restrict_file_tools_to_workdir"] is False
        assert updated["max_total_tool_output_bytes"] == 4096

    def test_absent_guardrails_remain_absent(self, cron_store):
        from cron.jobs import create_job

        job = create_job(prompt="legacy", schedule="every 1h")
        for key in _guardrails(Path("/unused")):
            if key != "workdir":
                assert key not in job

    def test_guardrails_require_explicit_enabled_toolsets_on_create_and_update(
        self, cron_store
    ):
        from cron.jobs import create_job, update_job

        with pytest.raises(ValueError, match="explicit enabled_toolsets"):
            create_job(
                prompt="guarded",
                schedule="every 1h",
                max_tokens=256,
            )

        legacy = create_job(prompt="legacy", schedule="every 1h")
        with pytest.raises(ValueError, match="explicit enabled_toolsets"):
            update_job(legacy["id"], {"max_tokens": 256})

    @pytest.mark.parametrize(
        "enabled_toolsets,posture",
        [
            (["file"], {"restrict_file_tools_to_workdir": True}),
            (["terminal"], {"terminal_sandbox": True}),
        ],
    )
    def test_create_accepts_only_complete_exact_guarded_postures(
        self, cron_store, tmp_path, enabled_toolsets, posture
    ):
        from cron.jobs import create_job

        workdir = tmp_path / enabled_toolsets[0]
        workdir.mkdir()
        job = create_job(
            prompt="guarded",
            schedule="every 1h",
            enabled_toolsets=enabled_toolsets,
            workdir=str(workdir),
            max_tokens=256,
            **posture,
        )
        assert job["enabled_toolsets"] == enabled_toolsets
        for field, value in posture.items():
            assert job[field] is value

    @pytest.mark.parametrize(
        "enabled_toolsets,posture",
        [
            (["file"], {}),
            (["terminal"], {}),
            (["file", "terminal"], {"terminal_sandbox": True}),
            (["file"], {
                "terminal_sandbox": True,
                "restrict_file_tools_to_workdir": True,
            }),
            (["web"], {"restrict_file_tools_to_workdir": True}),
        ],
    )
    def test_create_rejects_inexact_guarded_posture_immediately(
        self, cron_store, tmp_path, enabled_toolsets, posture
    ):
        from cron.jobs import create_job

        workdir = tmp_path / "repo"
        workdir.mkdir()
        with pytest.raises(ValueError, match="exactly one execution posture"):
            create_job(
                prompt="guarded",
                schedule="every 1h",
                enabled_toolsets=enabled_toolsets,
                workdir=str(workdir),
                max_tokens=256,
                **posture,
            )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("max_iterations", 0),
            ("max_tokens", 0),
            ("max_tokens", True),
            ("reasoning_effort", True),
            ("reasoning_effort", 1),
            ("reasoning_effort", "unsupported"),
            ("max_duration_seconds", -1),
            ("max_tool_output_bytes", "many"),
            ("max_total_tool_output_bytes", 0),
            ("max_tool_calls", True),
            ("max_files_read", -1),
            ("script_fail_closed", "yes"),
            ("skip_context_files", 1),
            ("terminal_sandbox", None),
            ("restrict_file_tools_to_workdir", "yes"),
        ],
    )
    def test_invalid_guardrail_values_are_rejected(self, cron_store, field, value):
        from cron.jobs import create_job

        with pytest.raises(ValueError, match=field):
            create_job(
                prompt="unsafe config",
                schedule="every 1h",
                **{field: value},
            )

    def test_terminal_sandbox_requires_workdir(self, cron_store):
        from cron.jobs import create_job

        with pytest.raises(ValueError, match="workdir"):
            create_job(
                prompt="sandbox me",
                schedule="every 1h",
                terminal_sandbox=True,
            )

    def test_restrict_file_tools_requires_workdir(self, cron_store):
        from cron.jobs import create_job

        with pytest.raises(ValueError, match="workdir"):
            create_job(
                prompt="confine files",
                schedule="every 1h",
                restrict_file_tools_to_workdir=True,
            )


class TestCronjobToolGuardrails:
    def test_none_guardrails_are_omitted_for_no_agent_create_and_update(
        self, cron_store
    ):
        from cron.jobs import get_job
        from tools.cronjob_tools import cronjob

        script = cron_store / "scripts" / "watchdog.sh"
        script.write_text("echo ok\n")
        omitted_guardrails = {
            key: None
            for key in _guardrails(Path("/unused"))
            if key != "workdir"
        }

        created = json.loads(
            cronjob(
                action="create",
                schedule="every 1h",
                script="watchdog.sh",
                no_agent=True,
                **omitted_guardrails,
            )
        )
        assert created["success"] is True
        for field in omitted_guardrails:
            assert field not in get_job(created["job_id"])

        updated = json.loads(
            cronjob(
                action="update",
                job_id=created["job_id"],
                prompt="use the agent",
                no_agent=False,
                **omitted_guardrails,
            )
        )
        assert updated["success"] is True
        for field in omitted_guardrails:
            assert field not in get_job(created["job_id"])

    def test_create_and_update_round_trip(self, cron_store, tmp_path, monkeypatch):
        from tools.cronjob_tools import CRONJOB_SCHEMA, cronjob

        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        workdir = tmp_path / "repo"
        workdir.mkdir()
        result = json.loads(
            cronjob(
                action="create",
                prompt="safe task",
                schedule="every 1h",
                enabled_toolsets=["file"],
                **_guardrails(workdir),
            )
        )
        assert result["success"] is True
        for key, value in _guardrails(workdir).items():
            assert result["job"][key] == value

        changed = json.loads(
            cronjob(
                action="update",
                job_id=result["job_id"],
                max_iterations=2,
                max_tokens=384,
                reasoning_effort=False,
                enabled_toolsets=["terminal"],
                terminal_sandbox=True,
                restrict_file_tools_to_workdir=False,
            )
        )
        assert changed["success"] is True
        assert changed["job"]["max_iterations"] == 2
        assert changed["job"]["max_tokens"] == 384
        assert changed["job"]["reasoning_effort"] is False
        assert changed["job"]["enabled_toolsets"] == ["terminal"]
        assert changed["job"]["terminal_sandbox"] is True
        assert changed["job"]["restrict_file_tools_to_workdir"] is False
        properties = CRONJOB_SCHEMA["parameters"]["properties"]
        for key in _guardrails(workdir):
            if key != "workdir":
                assert key in properties


class TestSchedulerGuardrails:
    def test_failed_prerun_script_fail_closed_before_agent(self, cron_store):
        from cron.scheduler import run_job

        job = {
            "id": "fail-closed",
            "name": "fail closed",
            "prompt": "must not run",
            "script": "broken.py",
            "script_fail_closed": True,
        }
        fake_db = MagicMock()
        with patch("cron.scheduler._run_job_script", return_value=(False, "boom")), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("run_agent.AIAgent") as agent_cls:
            success, _doc, final, error = run_job(job)

        assert success is False
        assert final == ""
        assert "boom" in error
        agent_cls.assert_not_called()

    def test_agent_receives_iteration_context_and_output_limits(self, cron_store, tmp_path, monkeypatch):
        from cron.scheduler import run_job

        workdir = tmp_path / "repo"
        workdir.mkdir()
        observed = {}
        fake_db = MagicMock()

        class FakeAgent:
            def __init__(self, **kwargs):
                observed["kwargs"] = kwargs

            def run_conversation(self, prompt, **kwargs):
                observed["run_kwargs"] = kwargs
                from tools.terminal_tool import resolve_task_overrides

                observed["task_overrides"] = resolve_task_overrides(
                    kwargs["task_id"]
                )
                return {"final_response": "ok"}

        job = {
            "id": "guarded",
            "name": "guarded",
            "prompt": "hello",
            "model": "test-model",
            "enabled_toolsets": ["file"],
            **_guardrails(workdir),
            "terminal_sandbox": False,
        }
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
        with patch("cron.scheduler._hermes_home", cron_store), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "provider": "test", "api_key": "k", "base_url": "http://test", "api_mode": "chat_completions"
             }), \
             patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
             patch("tools.environments.cron_unshare.CronUnshareEnvironment", return_value=MagicMock()), \
             patch("run_agent.AIAgent", FakeAgent):
            success, _doc, final, error = run_job(job)

        assert success is True, error
        assert final == "ok"
        assert observed["kwargs"]["max_iterations"] == 7
        assert observed["kwargs"]["max_tokens"] == 768
        assert observed["kwargs"]["cron_hard_max_tokens"] == 768
        assert observed["kwargs"]["reasoning_config"] == {
            "enabled": True,
            "effort": "low",
        }
        assert observed["kwargs"]["skip_context_files"] is True
        assert observed["kwargs"]["max_tool_output_bytes"] == 1024
        assert observed["kwargs"]["max_total_tool_output_bytes"] == 4096
        assert observed["kwargs"]["max_tool_calls"] == 11
        assert observed["kwargs"]["max_files_read"] == 5
        assert observed["kwargs"]["restrict_file_tools_to_workdir"] is True
        assert observed["kwargs"]["file_tool_workdir"] == str(workdir.resolve())
        assert observed["run_kwargs"]["task_id"].startswith("cron_guarded_")
        assert observed["task_overrides"]["cron_restricted_file_tools"] is True
        assert observed["task_overrides"]["cwd"] == str(workdir.resolve())

        from tools.terminal_tool import resolve_task_overrides

        assert resolve_task_overrides(observed["run_kwargs"]["task_id"]) == {}

    def test_absent_inference_caps_preserve_global_config(
        self, cron_store, monkeypatch
    ):
        from cron.scheduler import run_job

        (cron_store / "config.yaml").write_text(
            "model:\n"
            "  default: test-model\n"
            "  max_tokens: 2048\n"
            "agent:\n"
            "  reasoning_effort: high\n",
            encoding="utf-8",
        )
        observed = {}
        fake_db = MagicMock()

        class FakeAgent:
            def __init__(self, **kwargs):
                observed["kwargs"] = kwargs

            def run_conversation(self, prompt, **kwargs):
                return {"final_response": "ok"}

        job = {
            "id": "global-inference",
            "name": "global inference",
            "prompt": "hello",
        }
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
        with patch("cron.scheduler._hermes_home", cron_store), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "provider": "test", "api_key": "k", "base_url": "http://test", "api_mode": "chat_completions"
             }), \
             patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
             patch("run_agent.AIAgent", FakeAgent):
            success, _doc, final, error = run_job(job)

        assert success is True, error
        assert final == "ok"
        assert observed["kwargs"]["max_tokens"] == 2048
        assert observed["kwargs"]["cron_hard_max_tokens"] is None
        assert observed["kwargs"]["reasoning_config"] == {
            "enabled": True,
            "effort": "high",
        }

    def test_sandbox_setup_failure_is_before_agent_construction(self, cron_store, tmp_path):
        from cron.scheduler import run_job

        workdir = tmp_path / "repo"
        workdir.mkdir()
        job = {
            "id": "sandbox-fail",
            "name": "sandbox fail",
            "prompt": "must not run",
            "workdir": str(workdir),
            "terminal_sandbox": True,
        }
        with patch(
            "tools.environments.cron_unshare.CronUnshareEnvironment",
            side_effect=RuntimeError("unshare unavailable"),
        ), patch("run_agent.AIAgent") as agent_cls:
            success, _doc, final, error = run_job(job)

        assert success is False
        assert final == ""
        assert "unshare unavailable" in error
        agent_cls.assert_not_called()

    def test_file_environment_setup_failure_is_before_agent_construction(
        self, cron_store, tmp_path
    ):
        from cron.scheduler import run_job

        workdir = tmp_path / "repo"
        workdir.mkdir()
        job = {
            "id": "file-env-fail",
            "name": "file env fail",
            "prompt": "must not run",
            "model": "test-model",
            "workdir": str(workdir),
            "restrict_file_tools_to_workdir": True,
            "enabled_toolsets": ["file"],
        }
        with patch(
            "cron.scheduler._create_cron_file_environment",
            side_effect=RuntimeError("file environment unavailable"),
        ), patch("run_agent.AIAgent") as agent_cls:
            success, _doc, final, error = run_job(job)

        assert success is False
        assert final == ""
        assert "file environment unavailable" in error
        agent_cls.assert_not_called()

    def test_file_posture_blocks_v4a_and_cleans_every_registry(
        self, cron_store, tmp_path, monkeypatch
    ):
        from cron.scheduler import run_job
        from agent.tool_executor import _apply_cron_file_preflight
        from tools import file_tools
        from tools.terminal_tool import (
            _active_environments,
            _env_lock,
            _last_activity,
            _task_env_overrides,
            get_active_env,
        )

        workdir = tmp_path / "repo"
        outside = tmp_path / "outside"
        workdir.mkdir()
        outside.mkdir()
        (workdir / "inside.txt").write_text("inside\n", encoding="utf-8")
        (workdir / "patch.txt").write_text("before\n", encoding="utf-8")
        (outside / "secret.txt").write_text("secret\n", encoding="utf-8")
        default_env = SimpleNamespace(cwd=str(workdir))
        observed = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.tools = [
                    {
                        "type": "function",
                        "function": {"name": name},
                    }
                    for name in ("read_file", "search_files", "write_file", "patch")
                ]

            def run_conversation(self, _prompt, *, task_id):
                observed["task_id"] = task_id
                dedicated = get_active_env(task_id)
                observed["dedicated"] = dedicated
                observed["dedicated_cwd"] = dedicated.cwd
                observed["guarded"] = dedicated._hermes_guarded_cron

                # Reproduce the reviewer race after scheduler preflight: an
                # unrelated interactive session moves the shared environment.
                default_env.cwd = "/etc"
                read_args = {"path": "inside.txt"}
                assert _apply_cron_file_preflight(
                    self, "read_file", read_args, count_reads=False
                ) is None
                observed["canonical_read"] = read_args["path"]
                observed["read"] = json.loads(
                    file_tools.read_file_tool(read_args["path"], task_id=task_id)
                )

                outside_args = {"path": str(outside / "secret.txt")}
                observed["outside_block"] = _apply_cron_file_preflight(
                    self, "read_file", outside_args, count_reads=False
                )

                v4a_args = {
                    "mode": "patch",
                    "patch": (
                        "*** Begin Patch\n"
                        "*** Update File: patch.txt\n"
                        "@@\n"
                        "-before\n"
                        "+after\n"
                        "*** End Patch"
                    ),
                }
                observed["v4a_block"] = _apply_cron_file_preflight(
                    self, "patch", v4a_args, count_reads=False
                )
                observed["v4a_dispatched"] = False
                if observed["v4a_block"] is None:
                    observed["v4a_dispatched"] = True
                    observed["patch"] = json.loads(
                        file_tools.patch_tool(
                            mode="patch", patch=v4a_args["patch"], task_id=task_id
                        )
                    )
                # Attempts to mutate the raw file environment cannot alter its
                # execution root.
                dedicated.cwd = "/etc"
                observed["cwd_after_mutation"] = dedicated.cwd
                return {"final_response": "ok"}

        job = {
            "id": "dedicated-file-env",
            "name": "dedicated file env",
            "prompt": "read and patch",
            "model": "test-model",
            "workdir": str(workdir),
            "restrict_file_tools_to_workdir": True,
            "enabled_toolsets": ["file"],
        }
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
        file_tools.clear_file_ops_cache()
        with _env_lock:
            prior_envs = dict(_active_environments)
            prior_activity = dict(_last_activity)
            _active_environments["default"] = default_env
            _last_activity["default"] = time.time()
        try:
            with patch("cron.scheduler._hermes_home", cron_store), \
                 patch("cron.scheduler._resolve_origin", return_value=None), \
                 patch("hermes_cli.env_loader.load_hermes_dotenv"), \
                 patch("hermes_cli.env_loader.reset_secret_source_cache"), \
                 patch("hermes_state.SessionDB", return_value=MagicMock()), \
                 patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                     "provider": "test", "api_key": "k", "base_url": "http://test",
                     "api_mode": "chat_completions",
                 }), \
                 patch("run_agent.AIAgent", FakeAgent):
                success, _doc, final, error = run_job(job)

            assert success is True, error
            assert final == "ok"
            assert observed["dedicated"] is not default_env
            assert observed["guarded"] is True
            assert observed["dedicated_cwd"] == str(workdir.resolve())
            assert observed["cwd_after_mutation"] == str(workdir.resolve())
            assert observed["canonical_read"] == str(
                (workdir / "inside.txt").resolve()
            )
            assert "inside" in observed["read"]["content"]
            assert "outside cron workdir" in observed["outside_block"]
            assert "mode='patch'" in observed["v4a_block"]
            assert observed["v4a_dispatched"] is False
            assert (workdir / "patch.txt").read_text(encoding="utf-8") == "before\n"

            task_id = observed["task_id"]
            with _env_lock:
                assert task_id not in _active_environments
                assert task_id not in _last_activity
                assert _active_environments["default"] is default_env
            assert task_id not in _task_env_overrides
            assert task_id not in file_tools._file_ops_cache
        finally:
            file_tools.clear_file_ops_cache()
            with _env_lock:
                _active_environments.clear()
                _active_environments.update(prior_envs)
                _last_activity.clear()
                _last_activity.update(prior_activity)

    def test_parallel_file_jobs_use_distinct_raw_sessions_and_leave_no_state(
        self, cron_store, tmp_path, monkeypatch
    ):
        from cron.scheduler import run_job
        from tools import file_tools
        from tools.terminal_tool import (
            _active_environments,
            _env_lock,
            _last_activity,
            _task_env_overrides,
            get_active_env,
        )

        workdirs = []
        for index in range(2):
            workdir = tmp_path / f"repo-{index}"
            workdir.mkdir()
            (workdir / "value.txt").write_text(str(index), encoding="utf-8")
            workdirs.append(workdir)
        observed = {}
        observed_lock = threading.Lock()

        class FakeAgent:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.tools = [
                    {
                        "type": "function",
                        "function": {"name": name},
                    }
                    for name in ("read_file", "search_files", "write_file", "patch")
                ]

            def run_conversation(self, _prompt, *, task_id):
                env = get_active_env(task_id)
                with observed_lock:
                    observed[task_id] = (env, env.cwd)
                return {"final_response": Path(env.cwd, "value.txt").read_text()}

        jobs = [
            {
                "id": f"parallel-file-{index}",
                "name": f"parallel file {index}",
                "prompt": "read value",
                "model": "test-model",
                "workdir": str(workdir),
                "restrict_file_tools_to_workdir": True,
                "enabled_toolsets": ["file"],
            }
            for index, workdir in enumerate(workdirs)
        ]
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
        file_tools.clear_file_ops_cache()
        with patch("cron.scheduler._hermes_home", cron_store), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=MagicMock()), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "provider": "test", "api_key": "k", "base_url": "http://test",
                 "api_mode": "chat_completions",
             }), \
             patch("run_agent.AIAgent", FakeAgent):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(run_job, jobs))

        assert {result[2] for result in results} == {"0", "1"}
        assert len(observed) == 2
        assert len({id(env) for env, _cwd in observed.values()}) == 2
        assert {cwd for _env, cwd in observed.values()} == {
            str(path.resolve()) for path in workdirs
        }
        for task_id in observed:
            with _env_lock:
                assert task_id not in _active_environments
                assert task_id not in _last_activity
            assert task_id not in _task_env_overrides
            assert task_id not in file_tools._file_ops_cache

    def test_wallclock_duration_limit_interrupts_active_agent(self):
        from cron.scheduler import _await_cron_agent

        class SlowAgent:
            def __init__(self):
                self.interrupted = None
                self.finished = threading.Event()
                self.side_effects = []

            def run_conversation(self):
                time.sleep(0.2)
                self.side_effects.append("late-write")
                self.finished.set()
                return {"final_response": "late"}

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

            def interrupt(self, message):
                self.interrupted = message

        agent = SlowAgent()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation)
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError, match="maximum duration"):
                _await_cron_agent(
                    agent=agent,
                    future=future,
                    job={"id": "slow", "name": "slow"},
                    inactivity_limit=None,
                    max_duration_seconds=0.05,
                    poll_interval=0.01,
                )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        assert time.monotonic() - started >= 0.18
        assert "maximum duration" in agent.interrupted
        assert future.done()
        assert agent.finished.is_set()
        assert agent.side_effects == ["late-write"]

    def test_inactivity_timeout_waits_for_worker_acknowledgement(self):
        from cron.scheduler import _await_cron_agent

        class SlowAgent:
            def __init__(self):
                self.interrupted = None
                self.finished = threading.Event()
                self.side_effects = []

            def run_conversation(self):
                time.sleep(0.2)
                self.side_effects.append("late-write")
                self.finished.set()
                return {"final_response": "late"}

            def get_activity_summary(self):
                return {
                    "seconds_since_activity": 10.0,
                    "last_activity_desc": "stalled",
                }

            def interrupt(self, message):
                self.interrupted = message

        agent = SlowAgent()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation)
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError, match="idle"):
                _await_cron_agent(
                    agent=agent,
                    future=future,
                    job={"id": "idle", "name": "idle"},
                    inactivity_limit=0.01,
                    max_duration_seconds=None,
                    poll_interval=0.01,
                )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        assert time.monotonic() - started >= 0.18
        assert agent.interrupted == "Cron job timed out (inactivity)"
        assert future.done()
        assert agent.finished.is_set()
        assert agent.side_effects == ["late-write"]

    def test_timeout_waits_for_worker_before_sandbox_cleanup(
        self, cron_store, tmp_path, monkeypatch
    ):
        from cron.scheduler import run_job

        workdir = tmp_path / "repo"
        workdir.mkdir()
        allow_finish = threading.Event()
        worker_done = threading.Event()
        events = []
        releases = []

        class FakeSandbox:
            def __init__(self, **_kwargs):
                self._hermes_guarded_cron = True

            def cleanup(self):
                events.append("cleanup")

        class NonCooperativeAgent:
            def __init__(self, **_kwargs):
                pass

            def run_conversation(self, _prompt, **_kwargs):
                release = threading.Timer(0.2, allow_finish.set)
                releases.append(release)
                release.start()
                allow_finish.wait(timeout=2)
                events.append("worker-side-effect")
                worker_done.set()
                return {"final_response": "late"}

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

            def interrupt(self, _message):
                return None

        job = {
            "id": "timeout-cleanup",
            "name": "timeout cleanup",
            "prompt": "wait",
            "model": "test-model",
            "workdir": str(workdir),
            "terminal_sandbox": True,
            "enabled_toolsets": ["terminal"],
            "max_duration_seconds": 0.05,
        }
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
        started = time.monotonic()
        try:
            with patch("cron.scheduler._hermes_home", cron_store), \
                 patch("cron.scheduler._resolve_origin", return_value=None), \
                 patch("hermes_cli.env_loader.load_hermes_dotenv"), \
                 patch("hermes_cli.env_loader.reset_secret_source_cache"), \
                 patch("hermes_state.SessionDB", return_value=MagicMock()), \
                 patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                     "provider": "test", "api_key": "k", "base_url": "http://test",
                     "api_mode": "chat_completions",
                 }), \
                 patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
                 patch("tools.environments.cron_unshare.CronUnshareEnvironment", FakeSandbox), \
                 patch("run_agent.AIAgent", NonCooperativeAgent):
                success, _doc, _final, error = run_job(job)
        finally:
            allow_finish.set()
            for release in releases:
                release.join(timeout=1)

        assert success is False
        assert "maximum duration" in error
        assert time.monotonic() - started >= 0.18
        assert worker_done.is_set()
        assert events == ["worker-side-effect", "cleanup"]

    def test_terminal_sandbox_rejects_unsafe_toolset_before_agent(
        self, cron_store, tmp_path, monkeypatch
    ):
        from cron.scheduler import run_job

        workdir = tmp_path / "repo"
        workdir.mkdir()
        job = {
            "id": "unsafe-toolset",
            "name": "unsafe toolset",
            "prompt": "must not run",
            "model": "test-model",
            "workdir": str(workdir),
            "terminal_sandbox": True,
            "enabled_toolsets": ["file"],
        }
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
        sandbox = MagicMock()
        sandbox._hermes_guarded_cron = True
        with patch("cron.scheduler._hermes_home", cron_store), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=MagicMock()), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "provider": "test", "api_key": "k", "base_url": "http://test",
                 "api_mode": "chat_completions",
             }), \
             patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
             patch("tools.environments.cron_unshare.CronUnshareEnvironment", return_value=sandbox), \
             patch("run_agent.AIAgent") as agent_cls:
            success, _doc, final, error = run_job(job)

        assert success is False
        assert final == ""
        assert "exactly one execution posture" in error
        agent_cls.assert_not_called()

    def test_terminal_sandbox_exposes_foreground_terminal_only(
        self, cron_store, tmp_path, monkeypatch
    ):
        from cron.scheduler import run_job

        workdir = tmp_path / "repo"
        workdir.mkdir()
        job = {
            "id": "safe-terminal-toolset",
            "name": "safe terminal toolset",
            "prompt": "run one command",
            "model": "test-model",
            "workdir": str(workdir),
            "terminal_sandbox": True,
            "enabled_toolsets": ["terminal"],
        }
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
        sandbox = MagicMock()
        sandbox._hermes_guarded_cron = True
        fake_agent = MagicMock()
        fake_agent.run_conversation.return_value = {"final_response": "ok"}
        with patch("cron.scheduler._hermes_home", cron_store), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=MagicMock()), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "provider": "test", "api_key": "k", "base_url": "http://test",
                 "api_mode": "chat_completions",
             }), \
             patch("tools.mcp_tool.discover_mcp_tools") as discover_mcp, \
             patch("tools.environments.cron_unshare.CronUnshareEnvironment", return_value=sandbox), \
             patch("run_agent.AIAgent", return_value=fake_agent) as agent_cls:
            success, _doc, final, error = run_job(job)

        assert success is True
        assert final == "ok"
        assert error is None
        assert agent_cls.call_args.kwargs["enabled_toolsets"] == ["terminal_tools"]
        discover_mcp.assert_not_called()


class TestToolOutputByteLimits:
    def test_per_tool_and_total_limits_are_utf8_byte_hard_caps(self):
        from agent.tool_executor import _apply_cron_tool_output_limits

        agent = MagicMock()
        agent.max_tool_output_bytes = 24
        agent.max_total_tool_output_bytes = 30
        agent._tool_output_bytes_used = 0

        first = _apply_cron_tool_output_limits(agent, "é" * 100)
        second = _apply_cron_tool_output_limits(agent, "x" * 100)
        assert len(first.encode("utf-8")) <= 24
        assert len(second.encode("utf-8")) <= 6
        assert len(first.encode("utf-8")) + len(second.encode("utf-8")) <= 30
        assert agent._tool_output_bytes_used <= 30


class TestToolCallAndFileGuardrails:
    @staticmethod
    def _tool_call(name: str, arguments: dict, call_id: str):
        return SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        )

    @staticmethod
    def _agent(tmp_path, **limits):
        from run_agent import AIAgent

        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": name,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in (
                "read_file", "search_files", "write_file", "patch", "terminal", "web_search",
                "execute_code", "delegate_task",
            )
        ]
        kwargs = {
            "api_key": "test-key-1234567890",
            "base_url": "https://example.invalid/v1",
            "quiet_mode": True,
            "skip_context_files": True,
            "skip_memory": True,
            **limits,
        }
        with patch("run_agent.get_tool_definitions", return_value=tool_defs), \
             patch("run_agent.check_toolset_requirements", return_value={}), \
             patch("hermes_cli.config.load_config", return_value={}), \
             patch("run_agent.OpenAI"):
            agent = AIAgent(**kwargs)
        agent.tool_delay = 0
        agent._cached_system_prompt = "safe"
        return agent

    @staticmethod
    def _set_exact_boundary(agent, names):
        agent._cron_guardrails_active = True
        agent._cron_exact_tool_names = frozenset(names)

    def test_file_posture_filters_live_same_toolset_plugin_from_schema(self, tmp_path):
        from cron.scheduler import _apply_cron_exact_tool_boundary
        from model_tools import get_tool_definitions
        from tools.registry import registry

        def plugin_host_exec(_args, **_kwargs):
            return "HOST EXECUTED"

        registry.register(
            name="host_exec",
            toolset="file",
            schema={
                "name": "host_exec",
                "description": "plugin host execution",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=plugin_host_exec,
        )
        try:
            tools = get_tool_definitions(
                enabled_toolsets=["file"], quiet_mode=True
            )
            assert "host_exec" in {
                tool["function"]["name"] for tool in tools
            }, "reproduction premise: live registry additions re-expand file"
            agent = SimpleNamespace(
                tools=tools,
                valid_tool_names={tool["function"]["name"] for tool in tools},
                enabled_toolsets=["file"],
                disabled_toolsets=[],
                _tool_snapshot_generation=0,
            )

            _apply_cron_exact_tool_boundary(
                agent,
                frozenset({"read_file", "search_files", "write_file", "patch"}),
            )

            assert agent._cron_exact_tool_names == frozenset(
                {"read_file", "search_files", "write_file", "patch"}
            )
            assert "host_exec" not in agent.valid_tool_names
            assert "host_exec" not in {
                tool["function"]["name"] for tool in agent.tools
            }

            from tools.mcp_tool import refresh_agent_mcp_tools

            refresh_agent_mcp_tools(agent)
            assert "host_exec" not in agent.valid_tool_names
            assert "host_exec" not in {
                tool["function"]["name"] for tool in agent.tools
            }
        finally:
            registry.deregister("host_exec")

    def test_live_same_toolset_plugin_is_blocked_before_dispatch(self, tmp_path):
        from tools.registry import registry

        def plugin_host_exec(_args, **_kwargs):
            return "HOST EXECUTED"

        registry.register(
            name="host_exec",
            toolset="file",
            schema={
                "name": "host_exec",
                "description": "plugin host execution",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=plugin_host_exec,
        )
        try:
            agent = self._agent(tmp_path)
            self._set_exact_boundary(
                agent, {"read_file", "search_files", "write_file", "patch"}
            )
            messages = []
            call = self._tool_call("host_exec", {}, "host")

            with patch(
                "run_agent.handle_function_call", return_value="SHOULD_NOT_RUN"
            ) as invoke:
                agent._execute_tool_calls_sequential(
                    SimpleNamespace(tool_calls=[call]), messages, "cron-file"
                )

            invoke.assert_not_called()
            assert "exact" in messages[0]["content"].lower()
            assert "host_exec" in messages[0]["content"]
        finally:
            registry.deregister("host_exec")

    @pytest.mark.parametrize("execution", ["sequential", "concurrent"])
    @pytest.mark.parametrize(
        "name,arguments,exact_names",
        [
            (
                "read_file",
                {"path": "safe.txt"},
                {"read_file", "search_files", "write_file", "patch"},
            ),
            ("terminal", {"command": "printf ok"}, {"terminal"}),
        ],
    )
    def test_same_toolset_plugin_override_is_blocked_before_dispatch(
        self, tmp_path, execution, name, arguments, exact_names
    ):
        from tools.registry import registry

        agent = self._agent(tmp_path)
        self._set_exact_boundary(agent, exact_names)
        entry = registry.get_entry(name)
        assert entry is not None

        def plugin_override(_args, **_kwargs):
            return "PLUGIN OVERRIDE EXECUTED"

        messages = []
        call = self._tool_call(name, arguments, "override")
        execute = getattr(agent, f"_execute_tool_calls_{execution}")
        with patch.object(entry, "handler", plugin_override), patch(
            "run_agent.handle_function_call", return_value="SHOULD_NOT_RUN"
        ) as invoke:
            execute(SimpleNamespace(tool_calls=[call]), messages, "cron-guarded")

        invoke.assert_not_called()
        assert "repository-owned" in messages[0]["content"]

    @pytest.mark.parametrize("execution", ["sequential", "concurrent"])
    @pytest.mark.parametrize(
        "name,arguments,exact_names",
        [
            (
                "read_file",
                {"path": "safe.txt"},
                {"read_file", "search_files", "write_file", "patch"},
            ),
            ("terminal", {"command": "printf ok"}, {"terminal"}),
        ],
    )
    def test_repository_owned_exact_tools_reach_dispatch(
        self, tmp_path, execution, name, arguments, exact_names
    ):
        agent = self._agent(tmp_path)
        self._set_exact_boundary(agent, exact_names)
        messages = []
        call = self._tool_call(name, arguments, "allowed")
        execute = getattr(agent, f"_execute_tool_calls_{execution}")

        with patch(
            "run_agent.handle_function_call", return_value='{"ok": true}'
        ) as invoke:
            execute(SimpleNamespace(tool_calls=[call]), messages, "cron-guarded")

        invoke.assert_called_once()
        assert messages[0]["content"] == '{"ok": true}'

    def test_concurrent_file_batch_blocks_plugin_and_runs_builtin(self, tmp_path):
        from tools.registry import registry

        def plugin_host_exec(_args, **_kwargs):
            return "HOST EXECUTED"

        registry.register(
            name="host_exec",
            toolset="file",
            schema={
                "name": "host_exec",
                "description": "plugin host execution",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=plugin_host_exec,
        )
        try:
            agent = self._agent(tmp_path)
            self._set_exact_boundary(
                agent, {"read_file", "search_files", "write_file", "patch"}
            )
            calls = [
                self._tool_call("read_file", {"path": "safe.txt"}, "safe"),
                self._tool_call("host_exec", {}, "host"),
            ]
            messages = []

            with patch(
                "run_agent.handle_function_call", return_value='{"ok": true}'
            ) as invoke:
                agent._execute_tool_calls_concurrent(
                    SimpleNamespace(tool_calls=calls), messages, "cron-file"
                )

            invoke.assert_called_once()
            assert messages[0]["content"] == '{"ok": true}'
            assert "host_exec" in messages[1]["content"]
            assert "exact" in messages[1]["content"].lower()
        finally:
            registry.deregister("host_exec")

    def test_max_tool_calls_is_hard_across_sequential_batches(self, tmp_path):
        agent = self._agent(tmp_path, max_tool_calls=2)
        messages = []
        calls = [
            self._tool_call("web_search", {"q": str(index)}, f"c{index}")
            for index in range(3)
        ]

        with patch("run_agent.handle_function_call", return_value='{"ok": true}') as invoke:
            for call in calls:
                agent._execute_tool_calls_sequential(
                    SimpleNamespace(tool_calls=[call]), messages, "cron-task"
                )

        assert invoke.call_count == 2
        assert len(messages) == 3
        assert "max_tool_calls" in messages[-1]["content"]

    def test_parallel_batch_reserves_tool_calls_atomically(self, tmp_path):
        agent = self._agent(tmp_path, max_tool_calls=2)
        calls = [
            self._tool_call("web_search", {"q": str(index)}, f"c{index}")
            for index in range(3)
        ]
        messages = []

        with patch("run_agent.handle_function_call", return_value='{"ok": true}') as invoke:
            agent._execute_tool_calls_concurrent(
                SimpleNamespace(tool_calls=calls), messages, "cron-task"
            )

        assert invoke.call_count == 2
        assert len(messages) == 3
        assert "max_tool_calls" in messages[-1]["content"]
        assert agent._tool_calls_used == 2

    def test_subagent_shares_parent_tool_call_total(self, tmp_path):
        parent = self._agent(tmp_path, max_tool_calls=1)
        child = self._agent(tmp_path, max_tool_calls=1)
        child._cron_guardrail_parent = parent
        call = self._tool_call("web_search", {"q": "one"}, "c1")
        child_call = self._tool_call("web_search", {"q": "two"}, "c2")
        messages = []

        with patch("run_agent.handle_function_call", return_value='{"ok": true}') as invoke:
            parent._execute_tool_calls_sequential(
                SimpleNamespace(tool_calls=[call]), messages, "cron-task"
            )
            child._execute_tool_calls_sequential(
                SimpleNamespace(tool_calls=[child_call]), messages, "cron-task"
            )

        assert invoke.call_count == 1
        assert parent._tool_calls_used == 1
        assert "max_tool_calls" in messages[-1]["content"]

    def test_tool_call_reservation_never_exceeds_cap_under_threads(self, tmp_path):
        from agent.tool_executor import _reserve_cron_tool_calls

        agent = self._agent(tmp_path, max_tool_calls=17)
        barrier = threading.Barrier(8)

        def reserve():
            barrier.wait()
            return _reserve_cron_tool_calls(agent, 5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            allowed = list(pool.map(lambda _index: reserve(), range(8)))

        assert sum(allowed) == 17
        assert agent._tool_calls_used == 17

    def test_max_files_read_counts_unique_paths_and_fails_closed(self, tmp_path):
        workdir = tmp_path / "work"
        workdir.mkdir()
        (workdir / "one.txt").write_text("one", encoding="utf-8")
        (workdir / "two.txt").write_text("two", encoding="utf-8")
        agent = self._agent(
            tmp_path,
            max_files_read=1,
            file_tool_workdir=str(workdir),
        )
        calls = [
            self._tool_call("read_file", {"path": "one.txt"}, "c1"),
            self._tool_call("read_file", {"path": str(workdir / "one.txt")}, "c2"),
            self._tool_call("read_file", {"path": "two.txt"}, "c3"),
        ]
        messages = []

        with patch("run_agent.handle_function_call", return_value='{"content": "ok"}') as invoke:
            for call in calls:
                agent._execute_tool_calls_sequential(
                    SimpleNamespace(tool_calls=[call]), messages, "cron-task"
                )

        assert invoke.call_count == 2
        assert len(agent._files_read) == 1
        assert "max_files_read" in messages[-1]["content"]

    @pytest.mark.parametrize(
        "name,arguments",
        [
            ("read_file", {"path": "../outside.txt"}),
            ("search_files", {"pattern": "x", "path": "../outside"}),
            ("write_file", {"path": "../outside.txt", "content": "x"}),
            ("patch", {"mode": "replace", "path": "../outside.txt", "old_string": "x", "new_string": "y"}),
        ],
    )
    def test_restrict_file_tools_blocks_traversal_for_every_file_tool(self, tmp_path, name, arguments):
        workdir = tmp_path / "work"
        workdir.mkdir()
        agent = self._agent(
            tmp_path,
            restrict_file_tools_to_workdir=True,
            file_tool_workdir=str(workdir),
        )
        message = SimpleNamespace(tool_calls=[self._tool_call(name, arguments, "blocked")])
        results = []

        with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as invoke:
            agent._execute_tool_calls_sequential(message, results, "cron-task")

        invoke.assert_not_called()
        assert "outside cron workdir" in results[0]["content"]

    @pytest.mark.parametrize(
        "name,arguments",
        [
            ("read_file", {"path": "src/read.txt"}),
            ("search_files", {"pattern": "needle", "path": "src"}),
            ("write_file", {"path": "src/write.txt", "content": "x"}),
            (
                "patch",
                {
                    "mode": "replace",
                    "path": "src/patch.txt",
                    "old_string": "x",
                    "new_string": "y",
                },
            ),
        ],
    )
    def test_common_preflight_canonicalizes_simple_file_paths(
        self, tmp_path, name, arguments
    ):
        from agent.tool_executor import _apply_cron_file_preflight

        workdir = tmp_path / "work"
        workdir.mkdir()
        agent = self._agent(
            tmp_path,
            restrict_file_tools_to_workdir=True,
            file_tool_workdir=str(workdir),
        )
        expected = str((workdir / arguments["path"]).resolve())

        assert _apply_cron_file_preflight(
            agent, name, arguments, count_reads=False
        ) is None
        assert arguments["path"] == expected

    def test_guarded_v4a_move_symlink_then_add_race_is_blocked_before_dispatch(
        self, tmp_path
    ):
        from tools import file_tools

        workdir = tmp_path / "work"
        outside = tmp_path / "outside"
        target_parent = workdir / "target"
        moved_parent = workdir / "target-moved"
        target_parent.mkdir(parents=True)
        outside.mkdir()
        target = target_parent / "reviewer.txt"
        arguments = {
            "mode": "patch",
            "patch": (
                "*** Begin Patch\n"
                f"*** Add File: {target}\n"
                "+escaped\n"
                "*** End Patch"
            ),
        }
        agent = self._agent(
            tmp_path,
            restrict_file_tools_to_workdir=True,
            file_tool_workdir=str(workdir),
        )
        call = self._tool_call("patch", arguments, "v4a-race")
        messages = []

        def reproduce_reviewer_race(_name, dispatched_args, *_args, **_kwargs):
            target_parent.rename(moved_parent)
            target_parent.symlink_to(outside, target_is_directory=True)
            return file_tools.patch_tool(
                mode="patch", patch=dispatched_args["patch"]
            )

        with patch(
            "run_agent.handle_function_call", side_effect=reproduce_reviewer_race
        ) as invoke:
            agent._execute_tool_calls_sequential(
                SimpleNamespace(tool_calls=[call]), messages, "cron-task"
            )

        invoke.assert_not_called()
        assert "mode='patch'" in messages[0]["content"]
        assert target_parent.is_dir()
        assert not target_parent.is_symlink()
        assert not moved_parent.exists()
        assert not target.exists()
        assert not (outside / "reviewer.txt").exists()

    def test_unguarded_v4a_preserves_legacy_dispatch(self, tmp_path):
        agent = self._agent(tmp_path)
        arguments = {
            "mode": "patch",
            "patch": (
                "*** Begin Patch\n"
                "*** Add File: unguarded.txt\n"
                "+allowed\n"
                "*** End Patch"
            ),
        }
        messages = []

        with patch(
            "run_agent.handle_function_call", return_value='{"success": true}'
        ) as invoke:
            agent._execute_tool_calls_sequential(
                SimpleNamespace(
                    tool_calls=[self._tool_call("patch", arguments, "unguarded-v4a")]
                ),
                messages,
                "ordinary-task",
            )

        invoke.assert_called_once()
        assert json.loads(messages[0]["content"])["success"] is True

    def test_restrict_file_tools_blocks_symlink_escape(self, tmp_path):
        workdir = tmp_path / "work"
        outside = tmp_path / "outside"
        workdir.mkdir()
        outside.mkdir()
        (outside / "secret.txt").write_text("secret", encoding="utf-8")
        (workdir / "escape").symlink_to(outside, target_is_directory=True)
        agent = self._agent(
            tmp_path,
            restrict_file_tools_to_workdir=True,
            file_tool_workdir=str(workdir),
        )
        message = SimpleNamespace(tool_calls=[
            self._tool_call("read_file", {"path": "escape/secret.txt"}, "blocked")
        ])
        results = []

        with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as invoke:
            agent._execute_tool_calls_sequential(message, results, "cron-task")

        invoke.assert_not_called()
        assert "outside cron workdir" in results[0]["content"]

    def test_search_results_cannot_expose_more_unique_files_than_cap(self, tmp_path):
        workdir = tmp_path / "work"
        workdir.mkdir()
        for name in ("one.txt", "two.txt"):
            (workdir / name).write_text(name, encoding="utf-8")
        agent = self._agent(
            tmp_path,
            max_files_read=1,
            file_tool_workdir=str(workdir),
        )
        message = SimpleNamespace(tool_calls=[
            self._tool_call("search_files", {"pattern": "txt", "path": "."}, "search")
        ])
        result = json.dumps({
            "matches": [
                {"path": "one.txt", "line": 1, "content": "one"},
                {"path": "two.txt", "line": 1, "content": "two"},
            ]
        })
        results = []

        with patch("run_agent.handle_function_call", return_value=result):
            agent._execute_tool_calls_sequential(message, results, "cron-task")

        assert "max_files_read" in results[0]["content"]
        assert "one" not in results[0]["content"]
        assert "two" not in results[0]["content"]
        assert agent._files_read == set()

    def test_guarded_cron_file_ops_use_raw_registered_environment(self):
        from tools import file_tools
        from tools.terminal_tool import (
            _active_environments,
            _env_lock,
            get_active_env,
        )

        default_env = SimpleNamespace(cwd="/host-default")
        cron_env = SimpleNamespace(
            cwd="/cron-workdir",
            _hermes_guarded_cron=True,
        )
        legacy_raw_env = SimpleNamespace(cwd="/legacy-raw")
        created = []

        def fake_file_ops(env):
            value = SimpleNamespace(env=env, cwd=env.cwd)
            created.append(value)
            return value

        file_tools.clear_file_ops_cache()
        with _env_lock:
            prior = dict(_active_environments)
            _active_environments.clear()
            _active_environments.update({
                "default": default_env,
                "cron-raw": cron_env,
                "legacy-raw": legacy_raw_env,
            })
        try:
            with patch("tools.file_tools.ShellFileOperations", side_effect=fake_file_ops):
                guarded = file_tools._get_file_ops("cron-raw")
                legacy = file_tools._get_file_ops("legacy-raw")
                assert get_active_env("cron-raw") is cron_env
                assert get_active_env("legacy-raw") is default_env
        finally:
            file_tools.clear_file_ops_cache()
            with _env_lock:
                _active_environments.clear()
                _active_environments.update(prior)

        assert guarded.env is cron_env
        assert guarded.cwd == "/cron-workdir"
        assert legacy.env is default_env

    @pytest.mark.parametrize("pty", [False, True])
    def test_guarded_cron_background_never_uses_spawn_local(self, pty):
        from tools import terminal_tool as terminal_module
        from tools.process_registry import process_registry
        from tools.terminal_tool import _active_environments, _env_lock

        guarded_env = MagicMock()
        guarded_env._hermes_guarded_cron = True
        guarded_env.cwd = "/cron-workdir"
        guarded_env.env = {}
        guarded_env.execute.return_value = {"output": "ok", "returncode": 0}
        config = {
            "env_type": "local",
            "cwd": "/host-workdir",
            "host_cwd": "/host-workdir",
            "timeout": 30,
            "local_persistent": False,
        }

        with _env_lock:
            prior = dict(_active_environments)
            _active_environments["cron-background"] = guarded_env
        try:
            with patch.object(terminal_module, "_get_env_config", return_value=config), \
                 patch.object(terminal_module, "_start_cleanup_thread"), \
                 patch.object(
                     terminal_module,
                     "_check_all_guards",
                     return_value={"approved": True},
                 ), \
                 patch.object(process_registry, "spawn_local") as spawn_local:
                blocked = json.loads(
                    terminal_module.terminal_tool(
                        "sleep 10",
                        background=True,
                        pty=pty,
                        task_id="cron-background",
                    )
                )
                foreground = json.loads(
                    terminal_module.terminal_tool(
                        "printf ok",
                        task_id="cron-background",
                    )
                )
        finally:
            with _env_lock:
                _active_environments.clear()
                _active_environments.update(prior)

        spawn_local.assert_not_called()
        assert "guarded cron" in blocked["error"].lower()
        assert foreground["exit_code"] == 0
        guarded_env.execute.assert_called_once()

    def test_guarded_cron_process_tool_never_reads_process_registry(self):
        from tools import process_registry as process_module
        from tools.terminal_tool import _active_environments, _env_lock

        guarded_env = SimpleNamespace(_hermes_guarded_cron=True)
        with _env_lock:
            prior = dict(_active_environments)
            _active_environments["cron-process"] = guarded_env
        try:
            with patch.object(process_module.process_registry, "list_sessions") as listing:
                result = json.loads(
                    process_module._handle_process(
                        {"action": "list"}, task_id="cron-process"
                    )
                )
        finally:
            with _env_lock:
                _active_environments.clear()
                _active_environments.update(prior)

        listing.assert_not_called()
        assert "guarded cron" in result["error"].lower()

    def test_execute_code_is_blocked_for_guarded_cron_agent(self, tmp_path):
        agent = self._agent(tmp_path)
        agent._cron_sandbox_task_id = "cron-raw"
        messages = []
        call = self._tool_call("execute_code", {"code": "open('/tmp/pwned', 'w')"}, "code")

        with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as invoke:
            agent._execute_tool_calls_sequential(
                SimpleNamespace(tool_calls=[call]), messages, "cron-raw"
            )

        invoke.assert_not_called()
        assert "execute_code" in messages[0]["content"]
        assert "cron" in messages[0]["content"].lower()

    @pytest.mark.parametrize(
        "guardrail,value",
        [
            ("max_tool_calls", 3),
            ("max_files_read", 2),
            ("restrict_file_tools_to_workdir", True),
            ("max_tool_output_bytes", 1024),
            ("max_total_tool_output_bytes", 4096),
            ("_cron_hard_max_tokens", 768),
        ],
    )
    def test_execute_code_is_blocked_when_any_cron_guardrail_is_active(
        self, tmp_path, guardrail, value
    ):
        agent = self._agent(tmp_path)
        agent.platform = "cron"
        setattr(agent, guardrail, value)
        messages = []
        call = self._tool_call("execute_code", {"code": "print('unsafe')"}, "code")

        with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as invoke:
            agent._execute_tool_calls_sequential(
                SimpleNamespace(tool_calls=[call]), messages, "cron-guarded"
            )

        invoke.assert_not_called()
        assert "execute_code" in messages[0]["content"]
        assert "cron" in messages[0]["content"].lower()

    def test_execute_code_entrypoint_blocks_registered_guarded_environment(self):
        from tools import code_execution_tool
        from tools.terminal_tool import _active_environments, _env_lock

        guarded_env = SimpleNamespace(_hermes_guarded_cron=True)
        with _env_lock:
            prior = dict(_active_environments)
            _active_environments["cron-direct"] = guarded_env
        try:
            with patch.object(code_execution_tool, "SANDBOX_AVAILABLE", True), \
                 patch("tools.approval.check_execute_code_guard") as approval:
                result = json.loads(
                    code_execution_tool.execute_code(
                        "open('/tmp/pwned', 'w')", task_id="cron-direct"
                    )
                )
        finally:
            with _env_lock:
                _active_environments.clear()
                _active_environments.update(prior)

        assert "cron" in result["error"].lower()
        approval.assert_not_called()

    def test_restricted_cron_write_and_patch_skip_shell_and_lsp_diagnostics(
        self, tmp_path
    ):
        from tools import file_tools
        from tools.file_operations import PatchResult, ShellFileOperations, WriteResult
        from tools.terminal_tool import (
            clear_task_env_overrides,
            register_task_env_overrides,
        )

        workdir = tmp_path / "work"
        workdir.mkdir()
        target = workdir / "guarded.ts"
        task_id = "cron-file-only"
        external_commands = []

        class InProcessWriteOps(ShellFileOperations):
            def __init__(self):
                super().__init__(SimpleNamespace(cwd=str(workdir)))

            def _lsp_local_only(self):
                return True

            def _exec(self, command, **_kwargs):
                external_commands.append(command)
                raise AssertionError(
                    f"restricted cron invoked external diagnostics: {command}"
                )

            def write_file(self, path, content):
                Path(path).write_text(content, encoding="utf-8")
                self._snapshot_lsp_baseline(path)
                lint = self._check_lint_delta(
                    path, pre_content=None, post_content=content
                )
                lsp = self._maybe_lsp_diagnostics(
                    path, pre_content=None, post_content=content
                )
                return WriteResult(
                    bytes_written=len(content.encode("utf-8")),
                    lint=lint.to_dict(),
                    lsp_diagnostics=lsp or None,
                )

            def patch_replace(
                self, path, old_string, new_string, replace_all=False
            ):
                content = Path(path).read_text(encoding="utf-8")
                count = -1 if replace_all else 1
                updated = content.replace(old_string, new_string, count)
                Path(path).write_text(updated, encoding="utf-8")
                self._snapshot_lsp_baseline(path)
                lint = self._check_lint_delta(
                    path, pre_content=content, post_content=updated
                )
                lsp = self._maybe_lsp_diagnostics(
                    path, pre_content=content, post_content=updated
                )
                return PatchResult(
                    success=True,
                    files_modified=[path],
                    lint=lint.to_dict(),
                    lsp_diagnostics=lsp or None,
                )

        register_task_env_overrides(
            task_id,
            {
                "cwd": str(workdir),
                "cron_restricted_file_tools": True,
            },
        )
        try:
            with patch("tools.file_tools._get_file_ops", return_value=InProcessWriteOps()), \
                 patch("agent.lsp.get_service") as get_lsp_service:
                result = json.loads(
                    file_tools.write_file_tool(
                        str(target), "export const safe = true;\n", task_id=task_id
                    )
                )
                patch_result = json.loads(
                    file_tools.patch_tool(
                        path=str(target),
                        old_string="true",
                        new_string="false",
                        task_id=task_id,
                    )
                )
        finally:
            clear_task_env_overrides(task_id)

        assert target.read_text(encoding="utf-8") == "export const safe = false;\n"
        assert result["lint"]["status"] == "skipped"
        assert "restricted cron" in result["lint"]["message"].lower()
        assert "lsp_diagnostics" not in result
        assert patch_result["lint"]["status"] == "skipped"
        assert "restricted cron" in patch_result["lint"]["message"].lower()
        assert "lsp_diagnostics" not in patch_result
        assert external_commands == []
        get_lsp_service.assert_not_called()

    def test_guarded_cron_delegation_is_blocked_before_dispatch(self, tmp_path):
        agent = self._agent(tmp_path, max_tool_calls=3)
        agent.platform = "cron"
        messages = []
        call = self._tool_call("delegate_task", {"goal": "escape"}, "delegate")

        with patch.object(
            agent, "_dispatch_delegate_task", return_value="SHOULD_NOT_RUN"
        ) as dispatch:
            agent._execute_tool_calls_sequential(
                SimpleNamespace(tool_calls=[call]), messages, "cron-guarded"
            )

        dispatch.assert_not_called()
        assert "delegate_task" in messages[0]["content"]
        assert "cron" in messages[0]["content"].lower()

    def test_unguarded_cron_delegation_preserves_legacy_dispatch(self, tmp_path):
        agent = self._agent(tmp_path)
        agent.platform = "cron"
        messages = []
        call = self._tool_call("delegate_task", {"goal": "legacy"}, "delegate")

        with patch.object(
            agent,
            "_dispatch_delegate_task",
            return_value=json.dumps({"status": "started"}),
        ) as dispatch:
            agent._execute_tool_calls_sequential(
                SimpleNamespace(tool_calls=[call]), messages, "cron-legacy"
            )

        dispatch.assert_called_once()
        assert json.loads(messages[0]["content"])["status"] == "started"


class TestCronDelegationGuardrails:
    def test_terminal_sandbox_validation_rejects_delegation_toolset(self):
        from cron.scheduler import _validate_cron_sandbox_toolsets

        with pytest.raises(RuntimeError, match="delegate_task"):
            _validate_cron_sandbox_toolsets(["delegation"])

    def test_child_cannot_freshen_or_loosen_cron_parent_caps(self):
        from agent.iteration_budget import IterationBudget
        from tools.delegate_tool import _build_child_agent

        parent = MagicMock()
        parent.platform = "cron"
        parent.max_iterations = 7
        parent.max_tokens = 768
        parent._cron_hard_max_tokens = 768
        parent.reasoning_config = {"enabled": True, "effort": "low"}
        parent.iteration_budget = IterationBudget(7)
        parent.enabled_toolsets = ["terminal"]
        parent.valid_tool_names = {"terminal", "process", "delegate_task"}
        parent.base_url = "https://example.invalid/v1"
        parent.api_key = "key"
        parent.provider = "test"
        parent.api_mode = "chat_completions"
        parent.model = "test-model"
        parent._delegate_depth = 0
        parent._active_children = []
        parent._active_children_lock = threading.Lock()
        parent._print_fn = None
        parent.tool_progress_callback = None
        parent.thinking_callback = None
        parent._session_db = None
        parent.session_id = "cron-parent"
        parent._fallback_chain = []
        parent.prefill_messages = None
        parent.request_overrides = {}
        parent._cron_guardrail_parent = None
        parent.max_tool_output_bytes = 100
        parent.max_total_tool_output_bytes = 200
        parent.max_tool_calls = 3
        parent.max_files_read = 2
        parent.restrict_file_tools_to_workdir = True
        parent.file_tool_workdir = "/cron-workdir"

        child = MagicMock()
        with patch("tools.delegate_tool._load_config", return_value={
                 "max_iterations": 50,
                 "reasoning_effort": "high",
             }), \
             patch("tools.delegate_tool._resolve_child_credential_pool", return_value=None), \
             patch("run_agent.AIAgent", return_value=child) as agent_cls:
            built = _build_child_agent(
                task_index=0,
                goal="do work",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=50,
                task_count=1,
                parent_agent=parent,
                override_max_tokens=4096,
            )

        kwargs = agent_cls.call_args.kwargs
        assert kwargs["max_iterations"] == 7
        assert kwargs["max_tokens"] == 768
        assert kwargs["cron_hard_max_tokens"] == 768
        assert kwargs["reasoning_config"] == {"enabled": True, "effort": "low"}
        assert kwargs["iteration_budget"] is parent.iteration_budget
        assert built._cron_guardrail_parent is parent

    def test_shared_iteration_budget_is_not_reset_at_child_turn_start(self):
        from agent.iteration_budget import IterationBudget
        from agent.turn_context import _reset_iteration_budget_for_turn

        shared = IterationBudget(3)
        assert shared.consume() is True
        child = MagicMock()
        child.iteration_budget = shared
        child.max_iterations = 3
        child._preserve_iteration_budget = True
        _reset_iteration_budget_for_turn(child)

        assert child.iteration_budget is shared
        assert child.iteration_budget.used == 1

    def test_cron_child_timeout_waits_for_child_future_before_close(self, monkeypatch):
        from tools import delegate_tool

        futures = []

        class RecordingPool(concurrent.futures.ThreadPoolExecutor):
            def submit(self, *args, **kwargs):
                future = super().submit(*args, **kwargs)
                futures.append(future)
                return future

        events = []
        child_done = threading.Event()

        class NonCooperativeChild:
            _credential_pool = None
            _delegate_saved_tool_names = []
            _subagent_id = None
            _cron_sandbox_task_id = "cron-child"
            tool_progress_callback = None
            model = "test-model"
            session_prompt_tokens = 0
            session_completion_tokens = 0
            session_reasoning_tokens = 0
            session_estimated_cost_usd = 0.0

            def run_conversation(self, **_kwargs):
                time.sleep(0.2)
                events.append("child-side-effect")
                child_done.set()
                return {
                    "final_response": "late",
                    "completed": True,
                    "interrupted": False,
                    "api_calls": 0,
                    "messages": [],
                }

            def interrupt(self):
                events.append("interrupt")

            def get_activity_summary(self):
                return {
                    "current_tool": None,
                    "api_call_count": 0,
                    "max_iterations": 1,
                }

            def close(self):
                events.append("close")

        parent = SimpleNamespace(
            platform="cron",
            _current_task_id=None,
            _active_children=[],
            _active_children_lock=threading.Lock(),
            _touch_activity=lambda _description: None,
        )
        child = NonCooperativeChild()
        started = time.monotonic()
        try:
            with patch.object(delegate_tool, "_get_child_timeout", return_value=0.05), \
                 patch("tools.daemon_pool.DaemonThreadPoolExecutor", RecordingPool):
                result = delegate_tool._run_single_child(
                    task_index=0,
                    goal="wait",
                    child=child,
                    parent_agent=parent,
                )
        finally:
            child_done.wait(timeout=1)

        assert result["status"] == "timeout"
        assert time.monotonic() - started >= 0.18
        assert futures and futures[0].done()
        assert child_done.is_set()
        assert events == ["interrupt", "child-side-effect", "close"]


class TestCronHardTokenCap:
    @staticmethod
    def _conversation_agent():
        from run_agent import AIAgent

        with patch("run_agent.get_tool_definitions", return_value=[]), \
             patch("run_agent.check_toolset_requirements", return_value={}), \
             patch("hermes_cli.config.load_config", return_value={}), \
             patch("run_agent.OpenAI"):
            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                max_tokens=768,
                cron_hard_max_tokens=768,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        agent.client = MagicMock()
        agent._cached_system_prompt = "safe"
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent

    @staticmethod
    def _run(agent, responses, prompt="hello"):
        agent.client.chat.completions.create.side_effect = responses
        with patch.object(agent, "_persist_session"), \
             patch.object(agent, "_save_trajectory"), \
             patch.object(agent, "_cleanup_task_resources"):
            return agent.run_conversation(prompt)

    def test_length_continuation_retry_cannot_boost_past_cron_cap(self):
        from tests.run_agent.test_run_agent import _mock_response

        agent = self._conversation_agent()
        result = self._run(
            agent,
            [
                _mock_response(content="Part 1 ", finish_reason="length"),
                _mock_response(content="Part 2", finish_reason="stop"),
            ],
        )

        assert result["final_response"] == "Part 1 Part 2"
        retry_kwargs = agent.client.chat.completions.create.call_args_list[1].kwargs
        assert retry_kwargs.get("max_tokens", retry_kwargs.get("max_completion_tokens")) == 768

    def test_truncated_tool_call_retry_cannot_boost_past_cron_cap(self):
        from tests.run_agent.test_run_agent import _mock_response, _mock_tool_call

        agent = self._conversation_agent()
        bad = _mock_tool_call(
            name="terminal",
            arguments='{"command":"printf partial',
            call_id="bad",
        )
        result = self._run(
            agent,
            [
                _mock_response(content="", finish_reason="length", tool_calls=[bad]),
                _mock_response(content="recovered", finish_reason="stop"),
            ],
        )

        assert result["final_response"] == "recovered"
        retry_kwargs = agent.client.chat.completions.create.call_args_list[1].kwargs
        assert retry_kwargs.get("max_tokens", retry_kwargs.get("max_completion_tokens")) == 768

    def test_transport_request_overrides_are_clamped_to_immutable_cron_cap(self):
        agent = self._conversation_agent()
        agent.max_tokens = 8192
        agent._ephemeral_max_output_tokens = 16384
        agent.request_overrides = {
            "max_tokens": 32768,
            "max_completion_tokens": 32768,
            "max_output_tokens": 32768,
        }

        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])

        emitted_caps = [
            kwargs[key]
            for key in ("max_tokens", "max_completion_tokens", "max_output_tokens")
            if key in kwargs
        ]
        assert emitted_caps
        assert all(value == 768 for value in emitted_caps)

    def test_transport_without_output_cap_support_fails_closed(self):
        agent = self._conversation_agent()
        agent.provider = "openai-codex"
        agent.api_mode = "codex_responses"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent._base_url_lower = agent.base_url.lower()
        agent._base_url_hostname = "chatgpt.com"
        agent._transport_cache.clear()

        with pytest.raises(RuntimeError, match="cannot enforce cron max_tokens"):
            agent._build_api_kwargs([{"role": "user", "content": "hi"}])


@pytest.mark.skipif(sys.platform != "linux", reason="unshare sandbox is Linux-only")
class TestCronUnshareSandbox:
    @pytest.mark.parametrize(
        "relative_workdir",
        [Path("."), Path("tools"), Path("..")],
        ids=("source-root", "tools-subdir", "contains-source-root"),
    )
    def test_constructor_rejects_workdir_overlapping_trusted_bootstrap_without_popen(
        self, relative_workdir, monkeypatch
    ):
        import tools.environments.cron_unshare as cron_unshare

        source_root = Path(cron_unshare.__file__).resolve().parents[2]
        mock_popen = MagicMock()
        monkeypatch.setattr(cron_unshare.subprocess, "Popen", mock_popen)

        with pytest.raises(
            RuntimeError,
            match="terminal_sandbox workdir overlaps trusted bootstrap",
        ):
            cron_unshare.CronUnshareEnvironment(
                workdir=str(source_root / relative_workdir), timeout=10
            )

        mock_popen.assert_not_called()

    def test_launch_rechecks_workdir_overlap_with_trusted_bootstrap_without_popen(
        self, monkeypatch
    ):
        import tools.environments.cron_unshare as cron_unshare

        env = cron_unshare.CronUnshareEnvironment.__new__(
            cron_unshare.CronUnshareEnvironment
        )
        env.workdir = Path(cron_unshare.__file__).resolve().parents[1]
        mock_popen = MagicMock()
        monkeypatch.setattr(cron_unshare.subprocess, "Popen", mock_popen)

        with pytest.raises(
            RuntimeError,
            match="terminal_sandbox workdir overlaps trusted bootstrap",
        ):
            env._run_bash("printf must-not-run")

        mock_popen.assert_not_called()

    def test_launch_uses_isolated_trusted_bootstrap_from_trusted_cwd(
        self, tmp_path, monkeypatch
    ):
        import tools.environments.cron_unshare as cron_unshare

        workdir = tmp_path / "work"
        workdir.mkdir()
        env = cron_unshare.CronUnshareEnvironment.__new__(
            cron_unshare.CronUnshareEnvironment
        )
        env.workdir = workdir
        proc = MagicMock(pid=os.getpid())
        monkeypatch.setattr(cron_unshare.subprocess, "Popen", MagicMock(return_value=proc))
        monkeypatch.setattr(cron_unshare.os, "getpgid", lambda _pid: 123)

        assert env._run_bash("printf ok") is proc

        call = cron_unshare.subprocess.Popen.call_args
        argv = call.args[0]
        assert argv[argv.index(sys.executable) + 1:argv.index(sys.executable) + 3] == ["-I", "-S"]
        assert "-m" not in argv
        assert "-c" in argv
        assert str(Path(cron_unshare.__file__).resolve().parents[2]) in argv[argv.index("-c") + 1]
        assert call.kwargs["cwd"] == "/"
        assert "PYTHONPATH" not in call.kwargs["env"]

    def test_missing_trusted_bootstrap_fails_closed(self, tmp_path, monkeypatch):
        import tools.environments.cron_unshare as cron_unshare

        workdir = tmp_path / "work"
        workdir.mkdir()
        env = cron_unshare.CronUnshareEnvironment.__new__(
            cron_unshare.CronUnshareEnvironment
        )
        env.workdir = workdir
        monkeypatch.setattr(cron_unshare, "_TRUSTED_SOURCE_ROOT", tmp_path / "missing")

        with pytest.raises(RuntimeError, match="trusted bootstrap unavailable"):
            env._unshare_args("printf must-not-run")

    def test_hostile_workdir_python_startup_and_shadow_package_never_execute(
        self, tmp_path, monkeypatch
    ):
        from tools.environments.cron_unshare import CronUnshareEnvironment

        secret_home = tmp_path / "secret-home"
        secret_home.mkdir()
        secret = secret_home / "host-secret"
        secret.write_text("HOST SECRET", encoding="utf-8")
        workdir = secret_home / "work"
        workdir.mkdir()
        site_marker = workdir / "sitecustomize-ran"
        tools_marker = workdir / "shadow-tools-ran"
        (workdir / "sitecustomize.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(site_marker)!r}).write_text(Path({str(secret)!r}).read_text())\n",
            encoding="utf-8",
        )
        shadow_tools = workdir / "tools"
        shadow_tools.mkdir()
        trusted_tools = Path(__file__).resolve().parents[2] / "tools"
        (shadow_tools / "__init__.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(tools_marker)!r}).write_text(Path({str(secret)!r}).read_text())\n"
            f"__path__ = [{str(trusted_tools)!r}]\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HOME", str(secret_home))
        monkeypatch.setenv("HERMES_HOME", str(secret_home / ".hermes"))

        env = CronUnshareEnvironment(workdir=str(workdir), timeout=10)
        try:
            result = env.execute("printf WORKER_OK")
        finally:
            env.cleanup()

        assert result["returncode"] == 0, result["output"]
        assert "WORKER_OK" in result["output"]
        assert not site_marker.exists()
        assert not tools_marker.exists()

    def test_isolated_root_network_and_workdir(self, tmp_path, monkeypatch):
        from tools.environments.cron_unshare import CronUnshareEnvironment

        secret_home = tmp_path / "secret-home"
        secret_home.mkdir()
        (secret_home / ".ssh").mkdir()
        (secret_home / ".ssh" / "id_test").write_text("SECRET", encoding="utf-8")
        workdir = secret_home / "work"
        workdir.mkdir()
        (workdir / "input.txt").write_text("visible", encoding="utf-8")
        monkeypatch.setenv("HOME", str(secret_home))
        monkeypatch.setenv("HERMES_HOME", str(secret_home / ".hermes"))
        monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")

        env = CronUnshareEnvironment(workdir=str(workdir), timeout=10)
        sandbox_temp_dir = env._sandbox_temp_dir
        try:
            result = env.execute(
                "set -eu; "
                "test \"$(cat input.txt)\" = visible; "
                "test ! -e /mnt/c; test ! -e /home/k1; test ! -e /root/.ssh; "
                f"test ! -e {secret_home}/.ssh; "
                "test ! -r /etc/shadow; "
                "test -z \"${OPENAI_API_KEY:-}\"; "
                "test -z \"${HERMES_HOME:-}\"; "
                "test ! -w /usr; test ! -w /; "
                "! (printf blocked > /mnt/forbidden) 2>/dev/null; "
                "test ! -e /sys/class/net/eth0; "
                "printf ok > created.txt; printf PASS"
            )
        finally:
            env.cleanup()
        assert result["returncode"] == 0, result["output"]
        assert "PASS" in result["output"]
        assert (workdir / "created.txt").read_text() == "ok"
        assert not sandbox_temp_dir.exists()
