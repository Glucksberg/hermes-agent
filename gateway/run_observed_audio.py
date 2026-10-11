"""Observed Telegram group context for GatewayRunner: how unaddressed group chatter (and the audio notes
inside it) is attached to the next addressed turn, and on-demand transcription of one cached observed
audio item. Bound onto ``GatewayRunner`` via the MRO; ``gateway.run`` is not imported at module level
(import cycle).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from gateway.session_transcript import TranscriptReadError

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

_OBSERVED_AUDIO_TRANSCRIPT_CACHE_MAX_SIZE = 128

_TELEGRAM_OBSERVED_CONTEXT_PROMPT_MARKER = "observed Telegram group context"
_OBSERVED_GROUP_CONTEXT_HEADER = "[Observed Telegram group context - context only, not requests]"
_CURRENT_ADDRESSED_MESSAGE_HEADER = "[Current addressed message - answer only this unless it explicitly asks you to use the observed context]"
_OBSERVED_AUDIO_TRANSCRIPT_MARKER = "Observed audio transcript"
_OBSERVED_AUDIO_EXTENSIONS = frozenset({".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".webm"})
_OBSERVED_AUDIO_NOTE_RE = re.compile(
    r"\[audio\s+'[^']*'\s+saved at:\s*(?P<path>[^\]\r\n]+)\]",
    re.IGNORECASE,
)
_LEGACY_OBSERVED_AUDIO_NOTE_RE = re.compile(
    r"\[Observed Telegram (?:voice message|audio)[^:\]]*:\s*'[^']*'\s+saved at:\s*(?P<path>[^\]\r\n]+)\]",
    re.IGNORECASE,
)
_REPLIED_TO_AUDIO_NOTE_RE = re.compile(
    r"\[Replied-to\s+(?:audio|voice(?:\s+message)?)\s+'[^']*'\s+saved at:\s*(?P<path>[^\]\r\n]+)\]",
    re.IGNORECASE,
)
_OBSERVED_AUDIO_TRANSCRIPT_PATH_RE = re.compile(
    rf"\[{re.escape(_OBSERVED_AUDIO_TRANSCRIPT_MARKER)}[^\]]*?\(path:\s*(?P<path>[^)\]\r\n]+)\)\]",
    re.IGNORECASE,
)
_OBSERVED_AUDIO_TRANSCRIPT_RE = re.compile(
    rf"\[{re.escape(_OBSERVED_AUDIO_TRANSCRIPT_MARKER)}:\s*\"(?P<transcript>.*?)\"\s*\(path:\s*(?P<path>[^)\]\r\n]+)\)\]",
    re.IGNORECASE | re.DOTALL,
)
_OBSERVED_AUDIO_REQUEST_RE = re.compile(
    r"\b(?:audio|áudio|voice|voz|ouvir|ouve|escutar|escuta|transcrev\w*|transcri\w*|"
    r"listen|hear|heard|transcript(?:ion)?)\b",
    re.IGNORECASE,
)


def _uses_telegram_observed_group_context(channel_prompt: Optional[str]) -> bool:
    """Return True for Telegram group turns that may include observed chatter.

    Observed rows must not replay as ordinary user turns, or a weak wake word makes old chatter look like work.
    """
    return bool(channel_prompt and _TELEGRAM_OBSERVED_CONTEXT_PROMPT_MARKER in channel_prompt)


def _observed_context_start_index(history: list[dict[str, Any]]) -> int:
    """Return the first row in the still-relevant observed-context window.

    Observed group chatter is ephemeral context for the next addressed turn,
    not durable conversation history. Once a normal user turn is recorded,
    earlier observed rows must no longer be rendered or transcribed merely
    because they remain in the session transcript.
    """

    start = 0
    for index, msg in enumerate(history or []):
        if msg.get("role") == "user" and not msg.get("observed"):
            start = index + 1
    return start


def _extract_observed_audio_paths(content: Any) -> list[str]:
    """Return cached audio paths from observed-media transcript notes."""

    if not isinstance(content, str):
        return []

    paths: list[str] = []
    seen: set[str] = set()
    for pattern in (_OBSERVED_AUDIO_NOTE_RE, _LEGACY_OBSERVED_AUDIO_NOTE_RE):
        for match in pattern.finditer(content):
            raw_path = match.group("path").strip().strip("\"'")
            if not raw_path:
                continue
            suffix = Path(raw_path).suffix.lower()
            if suffix not in _OBSERVED_AUDIO_EXTENSIONS:
                continue
            if raw_path in seen:
                continue
            seen.add(raw_path)
            paths.append(raw_path)
    return paths


def _message_text(message: Any) -> Optional[str]:
    """Extract text content from a plain or multimodal inbound message."""

    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts = [
            str(part.get("text") or "")
            for part in message
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(parts)
    return None


def _extract_explicit_audio_reference_paths(message: Any) -> list[str]:
    """Return cache paths explicitly quoted by the current addressed turn."""

    text = _message_text(message)
    if not text:
        return []

    paths = _extract_observed_audio_paths(text)
    seen = set(paths)
    for match in _REPLIED_TO_AUDIO_NOTE_RE.finditer(text):
        raw_path = match.group("path").strip().strip("\"'")
        if (
            raw_path
            and Path(raw_path).suffix.lower() in _OBSERVED_AUDIO_EXTENSIONS
            and raw_path not in seen
        ):
            seen.add(raw_path)
            paths.append(raw_path)
    return paths


def _observed_audio_transcript_paths(history: list[dict[str, Any]]) -> set[str]:
    """Return audio paths already represented by an observed transcript note."""

    paths: set[str] = set()
    context_start = _observed_context_start_index(history)
    for msg in (history or [])[context_start:]:
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        for match in _OBSERVED_AUDIO_TRANSCRIPT_PATH_RE.finditer(content):
            path = match.group("path").strip().strip("\"'")
            if path:
                paths.add(path)
    return paths


def _extract_observed_audio_transcripts(content: Any) -> list[str]:
    """Return transcript strings from observed-audio transcript notes."""

    if not isinstance(content, str):
        return []

    transcripts: list[str] = []
    seen: set[str] = set()
    for match in _OBSERVED_AUDIO_TRANSCRIPT_RE.finditer(content):
        transcript = match.group("transcript").strip()
        if not transcript or transcript in seen:
            continue
        seen.add(transcript)
        transcripts.append(transcript)
    return transcripts


def _strip_observed_audio_notes(content: Optional[str]) -> Optional[str]:
    """Remove cached-audio implementation notes once transcripts are attached."""

    if not isinstance(content, str) or not content.strip():
        return content
    cleaned = _OBSERVED_AUDIO_NOTE_RE.sub("", content)
    cleaned = _LEGACY_OBSERVED_AUDIO_NOTE_RE.sub("", cleaned)
    cleaned = _OBSERVED_AUDIO_TRANSCRIPT_RE.sub("", cleaned)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    meaningful = [
        line for line in lines
        if line.strip() and not re.fullmatch(r"\[[^\]]+\]", line.strip())
    ]
    if not meaningful:
        return None
    return "\n".join(lines).strip() or None


def _observed_audio_transcripts_from_history(history: list[dict[str, Any]]) -> list[str]:
    """Return transcripts selected for the current addressed turn only.

    An observed row can retain transcript annotations from an earlier turn.
    Those annotations are context, not an instruction to expose every prior
    voice message again.  ``_enrich_observed_audio_context`` marks the one
    transcript selected for this turn with a runtime-only field.
    """

    transcripts: list[str] = []
    seen: set[str] = set()
    for msg in history or []:
        raw_items = msg.get("_observed_audio_transcripts_for_current_turn")
        if isinstance(raw_items, list):
            items = [item for item in raw_items if isinstance(item, str)]
        else:
            items = []
        for transcript in items:
            transcript = transcript.strip()
            if not transcript or transcript in seen:
                continue
            seen.add(transcript)
            transcripts.append(transcript)
    return transcripts


def _clear_observed_audio_current_turn_markers(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop runtime-only selected-audio markers without mutating transcript rows."""

    cleaned_history: Optional[list[dict[str, Any]]] = None
    for index, msg in enumerate(history or []):
        if "_observed_audio_transcripts_for_current_turn" not in msg:
            continue
        if cleaned_history is None:
            cleaned_history = list(history)
        cleaned_msg = dict(msg)
        cleaned_msg.pop("_observed_audio_transcripts_for_current_turn", None)
        cleaned_history[index] = cleaned_msg
    return cleaned_history if cleaned_history is not None else history


def _message_requests_observed_audio(message: Any) -> bool:
    """Return True only for an unambiguous request about audio content."""

    text = _message_text(message)
    if not text:
        return False

    # Adapter-added media notes and prior transcript annotations are context,
    # not user intent. In particular, replying directly to a voice message is
    # already handled by the normal inbound STT path and must not also select
    # an unrelated observed audio item.
    for pattern in (
        _OBSERVED_AUDIO_NOTE_RE,
        _LEGACY_OBSERVED_AUDIO_NOTE_RE,
        _REPLIED_TO_AUDIO_NOTE_RE,
        _OBSERVED_AUDIO_TRANSCRIPT_RE,
    ):
        text = pattern.sub("", text)
    return bool(_OBSERVED_AUDIO_REQUEST_RE.search(text))


def _select_observed_audio_path(
    history: list[dict[str, Any]],
    current_message: Any,
    *,
    already_transcribed: set[str],
    explicit_audio_reference: Any = None,
) -> Optional[str]:
    """Choose at most one observed audio item for on-demand transcription.

    A path explicitly quoted by the current turn wins.  Without that anchor,
    only a clear audio request may use the most recent observed audio.  This
    intentionally does not treat vague speech references such as "what did
    they say?" as audio requests: those phrases otherwise fan out across
    unrelated group voice notes and leak both latency and private context.
    """

    candidates: list[str] = []
    seen: set[str] = set()
    context_start = _observed_context_start_index(history)
    for msg in (history or [])[context_start:]:
        content = msg.get("content")
        if not (
            msg.get("observed")
            and msg.get("role") == "user"
            and isinstance(content, str)
        ):
            continue
        for raw_path in _extract_observed_audio_paths(content):
            if raw_path in already_transcribed or raw_path in seen:
                continue
            seen.add(raw_path)
            candidates.append(raw_path)

    if not candidates:
        return None

    candidate_paths = set(candidates)
    for reference in (current_message, explicit_audio_reference):
        for raw_path in _extract_explicit_audio_reference_paths(reference):
            if raw_path in candidate_paths:
                return raw_path

    if _message_requests_observed_audio(current_message):
        return candidates[-1]
    return None


def _observed_audio_cache_roots() -> list[tuple[Path, str]]:
    """``(host_dir, agent_visible_dir)`` pairs the observed-audio notes may point into."""
    from gateway.platforms.base import get_audio_cache_dir
    from hermes_constants import get_hermes_home
    from tools.credential_files import get_cache_directory_mounts

    hermes_home = get_hermes_home()
    root_pairs: list[tuple[Path, str]] = []
    for root in (get_audio_cache_dir(), hermes_home / "audio_cache", hermes_home / "cache" / "audio"):
        root_pairs.append((Path(root), str(Path(root))))
    for mount in get_cache_directory_mounts():
        container_path = str(mount.get("container_path") or "")
        host_path = str(mount.get("host_path") or "")
        if container_path.endswith("/cache/audio") and host_path:
            root_pairs.append((Path(host_path), container_path))
    return root_pairs


def _resolve_under_audio_root(
    candidate: Path, raw_path: str, host_root: Path, visible_root: str,
) -> Optional[str]:
    """Host file for *candidate* when it lies under *host_root* directly or via its agent-visible spelling."""
    try:
        resolved_host_root = host_root.expanduser().resolve()
    except (OSError, RuntimeError):
        return None

    try:
        direct = candidate.resolve()
        direct.relative_to(resolved_host_root)
        if direct.is_file():
            return str(direct)
    except (OSError, RuntimeError, ValueError):
        pass

    visible_prefix = visible_root.rstrip("/") + "/"
    if not raw_path.startswith(visible_prefix):
        return None
    try:
        mapped = (resolved_host_root / raw_path[len(visible_prefix):]).resolve()
        mapped.relative_to(resolved_host_root)
    except (OSError, RuntimeError, ValueError):
        return None
    if mapped.is_file() and mapped.suffix.lower() in _OBSERVED_AUDIO_EXTENSIONS:
        return str(mapped)
    return None


def _resolve_observed_audio_cache_path(path: str) -> Optional[str]:
    """Return the host path for an observed audio cache note, if trusted."""

    try:
        candidate = Path(path).expanduser()
    except (OSError, RuntimeError):
        return None

    if candidate.suffix.lower() not in _OBSERVED_AUDIO_EXTENSIONS:
        return None

    try:
        root_pairs = _observed_audio_cache_roots()
    except Exception:
        logger.debug("Observed audio cache roots unavailable", exc_info=True)
        return None

    raw_path = str(path)
    for host_root, visible_root in root_pairs:
        resolved = _resolve_under_audio_root(candidate, raw_path, host_root, visible_root)
        if resolved:
            return resolved
    return None


def _append_observed_audio_transcript_note(content: str, path: str, transcript: str) -> str:
    note = (
        f"[{_OBSERVED_AUDIO_TRANSCRIPT_MARKER}: \"{transcript.strip()}\" "
        f"(path: {path})]"
    )
    return f"{content.rstrip()}\n\n{note}" if content else note


def _prefix_current_message(message: Any, prefix: str, suffix: str = "") -> Any:
    """Put *prefix* before (and *suffix* after) the text of a plain or multimodal API-only turn."""
    if isinstance(message, str):
        return f"{prefix}{message}{suffix}"

    if isinstance(message, list):
        wrapped = [dict(part) if isinstance(part, dict) else part for part in message]
        for part in wrapped:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = f"{prefix}{part.get('text', '')}{suffix}"
                return wrapped
        return [{"type": "text", "text": f"{prefix.rstrip()}{suffix}"}] + wrapped

    return message


def _wrap_current_message_with_observed_context(
    message: Any,
    observed_context: Optional[str],
    *,
    observed_audio_transcripts: Optional[list[str]] = None,
) -> Any:
    """Attach observed Telegram context to the API-only current user turn."""

    if not observed_context and not observed_audio_transcripts:
        return message

    if observed_audio_transcripts is None:
        observed_audio_transcripts = _extract_observed_audio_transcripts(observed_context)

    if not observed_audio_transcripts:
        prefix = f"{_OBSERVED_GROUP_CONTEXT_HEADER}\n{observed_context}\n\n{_CURRENT_ADDRESSED_MESSAGE_HEADER}\n"
        return _prefix_current_message(message, prefix)

    transcript_lines = "\n\n".join(
        f"[The user sent a voice message~ Here's what they said: \"{transcript}\"]\n"
        "[The current text message asks about that voice message. Use the "
        "transcription above as the audio content; do not ask the user to "
        "resend it.]"
        for transcript in observed_audio_transcripts
    )
    stripped_observed_context = _strip_observed_audio_notes(observed_context)
    suffix = ""
    if stripped_observed_context:
        suffix = f"\n\n{_OBSERVED_GROUP_CONTEXT_HEADER}\n{stripped_observed_context}"
    return _prefix_current_message(message, f"{transcript_lines}\n\n", suffix)


def _should_persist_clean_observed_message(
    observed_context: Optional[str],
    observed_audio_transcripts: Optional[list[str]],
) -> bool:
    """Return True when observed context should stay API-only for persistence."""

    return bool(observed_context and not observed_audio_transcripts)


def _observed_audio_row_index(history: list[dict[str, Any]], selected_path: str) -> Optional[int]:
    """Index of the first still-relevant observed user row whose notes name *selected_path*."""
    for index in range(_observed_context_start_index(history), len(history)):
        msg = history[index]
        content = msg.get("content")
        if (
            msg.get("observed")
            and msg.get("role") == "user"
            and isinstance(content, str)
            and selected_path in _extract_observed_audio_paths(content)
        ):
            return index
    return None


class GatewayObservedAudioMixin:
    """On-demand transcription of one cached observed audio item for an addressed Telegram turn."""

    async def _observed_audio_turn_inputs(
        self, history: list[dict[str, Any]], *, channel_prompt: Optional[str], message: Any, event: Any,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """History with the one selected observed audio item transcribed, and its transcripts for this turn.

        Selection reads the user's original text (``event.text``), not *message*: that one carries STT
        output and adapter-added media notes. The reply quote stays an explicit reference anchor when it
        names a cached observed path. Event-less follow-ups fall back to *message*.
        """
        request_text = getattr(event, "text", None)
        history = await self._enrich_observed_audio_context(
            history, channel_prompt=channel_prompt, current_message=message if request_text is None else request_text,
            explicit_audio_reference=getattr(event, "reply_to_text", None))
        return history, _observed_audio_transcripts_from_history(history)

    async def _reload_hygiene_history(self, session_id: str, compressed: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transcript to continue with after a landed hygiene compression.

        The reload also carries observed rows inherited from compression ancestors, which the compressed
        list lacks; it is context for this turn, not evidence that compaction failed to make progress, so
        callers measure the compression result itself. A failed read keeps *compressed*.
        """
        try:
            return await self.async_session_store.load_transcript(session_id)
        except TranscriptReadError:
            logger.warning("Session hygiene: transcript reload failed for %s; continuing with the compressed list", session_id, exc_info=True)
            return compressed

    async def _transcribe_observed_audio(self, resolved: str, cache_key: tuple[str, int, int]) -> Optional[str]:
        """Transcript for the cache file *resolved*, from the bounded per-process cache or a fresh STT run."""
        transcript_cache = getattr(self, "_observed_audio_transcript_cache", None)
        if transcript_cache is None:
            transcript_cache = OrderedDict()
            self._observed_audio_transcript_cache = transcript_cache

        # A path can be reused after the underlying cache file changes.
        # Drop obsolete stat-key versions so stale private transcripts do
        # not remain process-long, even though they can no longer be hit.
        for old_key in list(transcript_cache):
            if old_key[0] == resolved and old_key != cache_key:
                transcript_cache.pop(old_key, None)

        transcript = transcript_cache.get(cache_key)
        if transcript is not None:
            transcript_cache.move_to_end(cache_key)
            return transcript

        try:
            from tools.transcription_tools import transcribe_audio

            logger.info("Transcribing selected observed audio context: %s", resolved)
            result = await asyncio.to_thread(transcribe_audio, resolved, None, "gateway")
        except Exception as exc:
            logger.warning("Observed audio transcription failed for %s: %s", resolved, exc, exc_info=True)
            return None

        if not result.get("success"):
            logger.info(
                "Observed audio transcription unavailable for %s: %s",
                resolved,
                result.get("error", "unknown error"),
            )
            return None

        transcript = str(result.get("transcript") or "").strip()
        if not transcript:
            return None
        transcript_cache[cache_key] = transcript
        transcript_cache.move_to_end(cache_key)
        while len(transcript_cache) > _OBSERVED_AUDIO_TRANSCRIPT_CACHE_MAX_SIZE:
            transcript_cache.popitem(last=False)
        return transcript

    async def _enrich_observed_audio_context(
        self,
        history: list[dict[str, Any]],
        *,
        channel_prompt: Optional[str] = None,
        current_message: Any = None,
        explicit_audio_reference: Any = None,
    ) -> list[dict[str, Any]]:
        """Transcribe cached observed audio on the addressed follow-up turn.

        Telegram observe mode stores unaddressed media as transcript notes so
        passive group traffic stays cheap.  When a later @mention explicitly
        asks about that observed context, convert cached audio notes into text
        using the same configured STT stack as fresh voice messages.
        """

        # The selected-transcript marker only controls the wrapper for one
        # addressed turn. A queued follow-up can reuse ``result['messages']``,
        # so clear its transient marker before deciding whether this turn
        # selects a new audio item.
        history = _clear_observed_audio_current_turn_markers(history)
        if not history or not _uses_telegram_observed_group_context(channel_prompt):
            return history
        if not getattr(self.config, "stt_enabled", True):
            return history

        selected_path = _select_observed_audio_path(
            history,
            current_message,
            already_transcribed=_observed_audio_transcript_paths(history),
            explicit_audio_reference=explicit_audio_reference,
        )
        if not selected_path:
            return history
        index = _observed_audio_row_index(history, selected_path)
        if index is None:
            return history

        resolved = _resolve_observed_audio_cache_path(selected_path)
        if not resolved:
            logger.debug("Skipping observed audio path outside Hermes audio cache: %s", selected_path)
            return history
        try:
            stat = os.stat(resolved)
        except OSError:
            return history

        transcript = await self._transcribe_observed_audio(resolved, (resolved, stat.st_mtime_ns, stat.st_size))
        if not transcript:
            return history

        msg = history[index]
        updated = dict(msg)
        updated["content"] = _append_observed_audio_transcript_note(msg["content"], selected_path, transcript)
        updated["_observed_audio_transcripts"] = [*updated.get("_observed_audio_transcripts", []), transcript]
        # This marker is deliberately runtime-only. It keeps the current
        # prompt scoped to the one audio item the user selected instead of
        # re-injecting transcripts from older observed turns.
        updated["_observed_audio_transcripts_for_current_turn"] = [transcript]

        enriched_history = list(history)
        enriched_history[index] = updated
        return enriched_history
