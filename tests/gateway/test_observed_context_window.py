"""Observed group chatter stays inside the prompt window across every way a session is shortened.

Observed rows are context only until the next addressed turn: ``_build_gateway_agent_history`` renders just
the observed rows after the last non-observed user row. Compaction and compression splits carry observed
rows over by hand, so each path must leave the still-pending ones AFTER the compacted/handoff block (before
it, the carried addressed turn would spend them) and must not revive spent ones. A DB with
``O1 O2 U1 A1 O3 O4`` has ``O3, O4`` pending; the same must hold afterwards, and one more addressed turn
spends them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.run import _build_gateway_agent_history
from hermes_state import SessionDB

CHANNEL_PROMPT = "You see observed Telegram group context."
SUMMARY = {"role": "assistant", "content": "SUMMARY"}
PENDING = "O3\nO4"


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    d = SessionDB(tmp_path / "state.db")
    d.create_session("s", source="test")
    return d


def _seed(db: SessionDB) -> dict[str, int]:
    ids = {}
    for name, role, observed in (
        ("O1", "user", True), ("O2", "user", True), ("U1", "user", False),
        ("A1", "assistant", False), ("O3", "user", True), ("O4", "user", True),
    ):
        ids[name] = db.append_message("s", role=role, content=name, observed=observed)
    return ids


def _context(history: list[dict]) -> str | None:
    return _build_gateway_agent_history(history, channel_prompt=CHANNEL_PROMPT)[1]


def _live(db: SessionDB, session_id: str = "s") -> list[dict]:
    return db.get_messages_as_conversation(session_id, repair_alternation=True)


def _compacted() -> list[dict]:
    return [dict(SUMMARY), {"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}]


def test_without_compaction_only_the_rows_after_the_last_addressed_turn_are_pending(db: SessionDB) -> None:
    _seed(db)

    assert _context(_live(db)) == PENDING


def test_watermark_compaction_keeps_pending_rows_pending_and_spent_rows_spent(db: SessionDB) -> None:
    _seed(db)

    db.archive_and_compact("s", _compacted(), watermark=db.get_active_message_watermark("s"))

    live = _live(db)
    assert [m["content"] for m in live] == ["SUMMARY", "U1", "A1", "O3", "O4"]
    assert _context(live) == PENDING


def test_proved_coverage_compaction_keeps_pending_rows_pending_and_spent_rows_spent(db: SessionDB) -> None:
    ids = _seed(db)

    db.archive_and_compact("s", _compacted(), covered_ids=[ids["U1"], ids["A1"]])

    live = _live(db)
    assert [m["content"] for m in live] == ["SUMMARY", "U1", "A1", "O3", "O4"]
    assert _context(live) == PENDING


def test_rotation_keeps_pending_rows_pending_and_spent_rows_spent(db: SessionDB) -> None:
    _seed(db)
    db.publish_compression_child(
        parent_session_id="s", child_session_id="child", source="test", messages=_compacted(),
        require_compression_lease=False)

    live = db.merge_ancestor_observed_messages("child", _live(db, "child"))

    assert [m["content"] for m in live] == ["SUMMARY", "U1", "A1", "O3", "O4"]
    assert _context(live) == PENDING


def test_rotation_carries_a_concurrent_observed_tail_row_once(db: SessionDB) -> None:
    _seed(db)
    watermark = db.get_active_message_watermark("s")
    db.append_message("s", role="user", content="late", observed=True)
    db.publish_compression_child(
        parent_session_id="s", child_session_id="child", source="test", messages=_compacted(),
        require_compression_lease=False, watermark=watermark)

    live = db.merge_ancestor_observed_messages("child", _live(db, "child"))

    contents = [m["content"] for m in live]
    assert contents[:3] == ["SUMMARY", "U1", "A1"]
    assert sorted(contents[3:]) == ["O3", "O4", "late"], "each pending row exactly once, none repeated"
    assert set(_context(live).split("\n")) == {"O3", "O4", "late"}


def test_an_addressed_tail_row_that_lands_during_compression_spends_the_rows_before_it(db: SessionDB) -> None:
    _seed(db)
    watermark = db.get_active_message_watermark("s")
    db.append_message("s", role="user", content="late steer")
    db.publish_compression_child(
        parent_session_id="s", child_session_id="child", source="test", messages=_compacted(),
        require_compression_lease=False, watermark=watermark)

    live = db.merge_ancestor_observed_messages("child", _live(db, "child"))

    assert [m["content"] for m in live] == ["SUMMARY", "U1", "A1", "late steer"]


@pytest.mark.parametrize("path", ["watermark", "covered", "rotation"])
def test_one_more_addressed_turn_spends_the_carried_rows(db: SessionDB, path: str) -> None:
    ids = _seed(db)
    session_id = "s"
    if path == "watermark":
        db.archive_and_compact("s", _compacted(), watermark=db.get_active_message_watermark("s"))
    elif path == "covered":
        db.archive_and_compact("s", _compacted(), covered_ids=[ids["U1"], ids["A1"]])
    else:
        session_id = "child"
        db.publish_compression_child(
            parent_session_id="s", child_session_id="child", source="test", messages=_compacted(),
            require_compression_lease=False)
    db.append_message(session_id, role="user", content="U2")

    live = db.merge_ancestor_observed_messages(session_id, _live(db, session_id))

    assert _context(live) is None
