"""Observed gateway rows (unaddressed group chatter) across in-place compaction are bounded.

Observed rows never reach the compressor's summary, so the live set keeps at most
``MAX_OBSERVED_CONTEXT_MESSAGES`` of them (the newest) whichever commit path runs: the
watermark path re-inserts them ahead of the compacted set, the proved-coverage path clones the
ones the compressor never held after it. Either way each survivor is live exactly once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_state import MAX_OBSERVED_CONTEXT_MESSAGES, SessionDB

OVER_CAP = MAX_OBSERVED_CONTEXT_MESSAGES + 5
SUMMARY = [{"role": "assistant", "content": "compacted summary"}]


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    d = SessionDB(tmp_path / "state.db")
    d.create_session("sess1", source="test")
    return d


def _seed_observed(db: SessionDB, count: int = OVER_CAP) -> None:
    for index in range(count):
        db.append_message(
            "sess1", role="user", content=f"observed-{index:03d}", observed=True,
            platform_message_id=f"tg-{index:03d}")


def _live(db: SessionDB) -> list[dict]:
    return db.get_messages_as_conversation("sess1")


def _observed_contents(messages: list[dict]) -> list[str]:
    return [message["content"] for message in messages if message.get("observed")]


EXPECTED_NEWEST = [f"observed-{index:03d}" for index in range(5, OVER_CAP)]


def test_watermark_path_keeps_newest_observed_ahead_of_the_compacted_set(db: SessionDB) -> None:
    _seed_observed(db)
    db.append_message("sess1", role="user", content="ordinary turn")

    count = db.archive_and_compact("sess1", SUMMARY, watermark=db.get_active_message_watermark("sess1"))

    live = _live(db)
    assert _observed_contents(live) == EXPECTED_NEWEST
    assert [message["content"] for message in live[-1:]] == ["compacted summary"]
    assert count == len(live) == MAX_OBSERVED_CONTEXT_MESSAGES + 1


def test_watermark_path_clones_observed_rows_that_arrive_mid_compression_once(db: SessionDB) -> None:
    _seed_observed(db, 3)
    watermark = db.get_active_message_watermark("sess1")
    db.append_message("sess1", role="user", content="observed-late", observed=True, platform_message_id="tg-late")

    db.archive_and_compact("sess1", SUMMARY, watermark=watermark)

    live = _live(db)
    assert _observed_contents(live) == ["observed-000", "observed-001", "observed-002", "observed-late"]
    assert live[-1]["message_id"] == "tg-late"


def test_proved_coverage_path_caps_the_observed_rows_it_clones(db: SessionDB) -> None:
    _seed_observed(db)
    ordinary = [
        db.append_message("sess1", role="user", content="ordinary question"),
        db.append_message("sess1", role="assistant", content="ordinary answer"),
    ]

    db.archive_and_compact("sess1", SUMMARY, covered_ids=ordinary)

    live = _live(db)
    assert _observed_contents(live) == EXPECTED_NEWEST
    assert live[0]["content"] == "compacted summary", "unseen rows follow the compacted set, as before"
    assert db.get_session("sess1")["message_count"] == len(live) == MAX_OBSERVED_CONTEXT_MESSAGES + 1


def test_proved_coverage_path_leaves_a_small_observed_set_untouched(db: SessionDB) -> None:
    _seed_observed(db, 3)
    ordinary = [db.append_message("sess1", role="user", content="ordinary question")]

    db.archive_and_compact("sess1", SUMMARY, covered_ids=ordinary)

    assert _observed_contents(_live(db)) == ["observed-000", "observed-001", "observed-002"]
