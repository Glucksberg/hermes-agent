"""Turn wiring for observed Telegram audio context and observed-row replay boundaries.

``_observed_audio_turn_inputs`` selects from what the user typed (``event.text``), never from the
STT-enriched turn message; the hygiene reload keeps the compressed list when the read fails; and
the durable repair never folds observed rows into a neighbour.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.agent_runtime_helpers import _merge_consecutive_users
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway.session_transcript import TranscriptReadError

CHANNEL_PROMPT = "You see observed Telegram group context."
AUDIO_NOTE = "[audio 'voice.ogg' saved at: /cache/voice.ogg]"
HISTORY = [{"role": "user", "content": f"[Alice|1]\n{AUDIO_NOTE}", "observed": True}]


def _runner(load_transcript: AsyncMock | None = None) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(_store=runner.session_store, load_transcript=load_transcript)
    runner._enrich_observed_audio_context = AsyncMock(side_effect=lambda history, **_kw: history)
    return runner


@pytest.mark.asyncio
async def test_selection_reads_the_event_text_and_reply_quote() -> None:
    runner = _runner()
    event = SimpleNamespace(text="what is in that audio?", reply_to_text="quoted")

    history, transcripts = await runner._observed_audio_turn_inputs(
        HISTORY, channel_prompt=CHANNEL_PROMPT, message="[voice transcript from STT]", event=event)

    runner._enrich_observed_audio_context.assert_awaited_once_with(
        HISTORY, channel_prompt=CHANNEL_PROMPT, current_message="what is in that audio?",
        explicit_audio_reference="quoted")
    assert history == HISTORY and transcripts == []


@pytest.mark.asyncio
async def test_an_empty_event_text_does_not_fall_back_to_the_stt_enriched_message() -> None:
    runner = _runner()

    await runner._observed_audio_turn_inputs(
        HISTORY, channel_prompt=CHANNEL_PROMPT, message="ouça o áudio", event=SimpleNamespace(text="", reply_to_text=None))

    assert runner._enrich_observed_audio_context.await_args.kwargs["current_message"] == ""


@pytest.mark.asyncio
async def test_an_event_less_followup_selects_from_the_turn_message() -> None:
    runner = _runner()

    await runner._observed_audio_turn_inputs(HISTORY, channel_prompt=CHANNEL_PROMPT, message="continue", event=None)

    kwargs = runner._enrich_observed_audio_context.await_args.kwargs
    assert kwargs["current_message"] == "continue" and kwargs["explicit_audio_reference"] is None


@pytest.mark.asyncio
async def test_hygiene_reload_returns_the_stored_transcript_with_inherited_observed_rows() -> None:
    stored = [{"role": "user", "content": "inherited", "observed": True}, {"role": "assistant", "content": "summary"}]
    runner = _runner(AsyncMock(return_value=stored))

    assert await runner._reload_hygiene_history("sid", [{"role": "assistant", "content": "summary"}]) == stored


@pytest.mark.asyncio
async def test_hygiene_reload_keeps_the_compressed_list_when_the_read_fails() -> None:
    runner = _runner(AsyncMock(side_effect=TranscriptReadError("sid")))
    compressed = [{"role": "assistant", "content": "summary"}]

    assert await runner._reload_hygiene_history("sid", compressed) is compressed


def test_consecutive_observed_rows_are_never_merged() -> None:
    messages = [
        {"role": "user", "content": "first", "observed": True, "message_id": "tg-1"},
        {"role": "user", "content": "second", "observed": True, "message_id": "tg-2"},
        {"role": "user", "content": "addressed"},
    ]

    merged, repairs = _merge_consecutive_users([dict(message) for message in messages])

    assert repairs == 0
    assert [message["content"] for message in merged] == ["first", "second", "addressed"]


def test_plain_consecutive_users_still_merge() -> None:
    merged, repairs = _merge_consecutive_users(
        [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}])

    assert repairs == 1
    assert [message["content"] for message in merged] == ["one\n\ntwo"]
