"""Observed gateway rows (unaddressed group chatter, e.g. Telegram media) across compaction and
compression splits for SessionDB.

Observed rows bypass model-generated summaries, so they are carried over by hand, under two rules:

* **Window.** An observed row is context only until the next addressed (non-observed user) turn
  (``gateway.run_observed_audio._observed_context_start_index``). Only rows after the session's last
  addressed turn are carried; spent ones are archived with the rest, so they never re-enter the prompt.
* **Placement.** Carried rows go AFTER the compacted set / the child's handoff: that keeps them after the
  last addressed turn that the compacted set ends with, which is what keeps them inside the window.

One global cap (``hermes_state.MAX_OBSERVED_CONTEXT_MESSAGES``, late-imported from the facade) covers the
in-place flow and the split-session flow. Bound onto ``SessionDB`` via the MRO.
"""

from __future__ import annotations

from typing import Any

from hermes_state_common import _placeholders

_LAST_ADDRESSED_USER_SQL = (
    "SELECT COALESCE(MAX(id), 0) FROM messages "
    "WHERE session_id = ? AND active = 1 AND observed = 0 AND role = 'user'")


def _same_observed_message(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """True when *right* is the compacted copy of the stored observed message *left*."""
    if not right.get("observed"):
        return False
    left_id = left.get("message_id") or left.get("platform_message_id")
    right_id = right.get("message_id") or right.get("platform_message_id")
    if left_id and right_id:
        return left_id == right_id
    return left.get("role") == right.get("role") and left.get("content") == right.get("content")


class SessionObservedMixin:
    def _with_retained_observed(
        self, conn, session_id: str, compacted_messages: list[dict[str, Any]], watermark: int | None = None,
    ) -> list[dict[str, Any]]:
        """*compacted_messages* followed by the newest in-window observed rows of *session_id*.

        The in-place archive deactivates every live row, so the observed rows still in the window are
        re-inserted after the compacted set (reusing the compacted copy when one exists). Only rows at or
        below *watermark* are handled here: the concurrent-append path clones the newer ones itself.
        Observed dicts in *compacted_messages* that are not retained are dropped.
        """
        from hermes_state import MAX_OBSERVED_CONTEXT_MESSAGES

        ceiling = () if watermark is None else (int(watermark),)
        bound = "" if watermark is None else " AND id <= ?"
        last_addressed = conn.execute(f"{_LAST_ADDRESSED_USER_SQL}{bound}", (session_id, *ceiling)).fetchone()[0]
        rows = conn.execute(
            "SELECT role, content, timestamp, platform_message_id FROM ("
            "  SELECT id AS message_row_id, role, content, timestamp, platform_message_id "
            f"  FROM messages WHERE session_id = ? AND active = 1 AND observed = 1 AND id > ?{bound} "
            "  ORDER BY id DESC LIMIT ?"
            ") ORDER BY message_row_id",
            (session_id, last_addressed, *ceiling, MAX_OBSERVED_CONTEXT_MESSAGES),
        ).fetchall()
        compacted_observed = [message for message in compacted_messages if message.get("observed")]
        retained = []
        for row in rows:
            candidate: dict[str, Any] = {
                "role": row["role"], "content": self._decode_content(row["content"]),
                "timestamp": row["timestamp"], "observed": True,
            }
            if row["platform_message_id"]:
                candidate["message_id"] = row["platform_message_id"]
            retained.append(next(
                (message for message in compacted_observed if _same_observed_message(candidate, message)),
                candidate))
        return [*(message for message in compacted_messages if not message.get("observed")), *retained]

    def _bound_unseen_observed(self, conn, session_id: str, row_ids: list[int]) -> list[int]:
        """*row_ids* minus the observed rows that are spent or beyond the newest ``MAX_OBSERVED_CONTEXT_MESSAGES``.

        For the proved-coverage path, which clones every row the compressor never held (all observed
        rows, on Telegram) after the compacted set: the window and the cap must hold there too. Dropped
        rows are archived with the rest of the live set.
        """
        from hermes_state import MAX_OBSERVED_CONTEXT_MESSAGES

        if not row_ids:
            return row_ids
        last_addressed = conn.execute(_LAST_ADDRESSED_USER_SQL, (session_id,)).fetchone()[0]
        observed = {int(row["id"]) for row in conn.execute(
            f"SELECT id FROM messages WHERE observed = 1 AND id IN ({_placeholders(row_ids)}) ORDER BY id",
            row_ids).fetchall()}
        keep = set(sorted(row_id for row_id in observed if row_id > last_addressed)[-MAX_OBSERVED_CONTEXT_MESSAGES:])
        return [row_id for row_id in row_ids if row_id not in observed or row_id in keep]

    def _ancestor_observed_rows(self, ancestor_ids: list[str], boundaries: list[float]) -> list[Any]:
        """Newest in-window observed rows of the compression ancestors, oldest first.

        *boundaries[i]* is when ancestor ``i`` ended (= when its child's handoff was written). A row is
        spent when its own session has a later addressed turn, or when any later session of the lineage
        has an addressed turn after its handoff: the addressed turn already consumed it.
        """
        from hermes_state import MAX_OBSERVED_CONTEXT_MESSAGES

        rows: list[Any] = []
        for index, session_id in enumerate(ancestor_ids):
            # Ancestor ``later`` was published at ``boundaries[later - 1]``; the tip is checked by the caller.
            if any(self._has_addressed_turn_after(ancestor_ids[later], boundaries[later - 1])
                   for later in range(index + 1, len(ancestor_ids))):
                continue
            last_addressed = self._read_one(_LAST_ADDRESSED_USER_SQL, (session_id,))[0]
            rows += self._read_all(
                "SELECT id, role, content, timestamp, platform_message_id FROM messages "
                "WHERE session_id = ? AND observed = 1 AND active = 1 AND id > ?", (session_id, last_addressed))
        return sorted(rows, key=lambda row: row["id"])[-MAX_OBSERVED_CONTEXT_MESSAGES:]

    def _has_addressed_turn_after(self, session_id: str, handoff_ended_at: float) -> bool:
        """True when *session_id* holds an addressed user turn newer than its handoff."""
        return self._read_one(
            "SELECT 1 FROM messages WHERE session_id = ? AND active = 1 AND observed = 0 AND role = 'user' "
            "AND timestamp > ? LIMIT 1", (session_id, handoff_ended_at)) is not None

    def merge_ancestor_observed_messages(self, session_id: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """*messages* (the transcript of *session_id*) with its compression ancestors' in-window observed rows.

        When context compression creates a new session, observed messages stored in the parent session
        (e.g. unaddressed media/audio in a Telegram group) are no longer accessible via the new session's
        transcript. They are spliced in right after the child's handoff block (rows stamped no later than
        the moment the parent ended; anything newer is post-rotation), so they stay after the last
        addressed turn the handoff carries and before every later row: the first addressed turn after
        the split still sees them, and any later addressed turn spends them.

        Only the verified compression lineage counts (each hop's parent ended ``compression``), so
        branches, resets and delegate-subagent parents that share a ``parent_session_id`` link never
        leak in. At most ``MAX_OBSERVED_CONTEXT_MESSAGES`` rows are spliced.
        """
        lineage = self._resume_lineage_ids(session_id)
        ancestor_ids = lineage[:-1]
        if not ancestor_ids:
            return messages
        boundaries = [float((self.get_session(ancestor) or {}).get("ended_at") or 0.0) for ancestor in ancestor_ids]
        # The tip's addressed turns after its handoff spend everything; so do those of every later ancestor.
        if self._has_addressed_turn_after(lineage[-1], boundaries[-1]):
            return messages
        rows = self._ancestor_observed_rows(ancestor_ids, boundaries)
        if not rows:
            return messages
        # A concurrent-tail clone already sits in the child with its parent's timestamp: don't repeat it.
        in_child = {(message.get("timestamp"), message.get("content")) for message in messages if message.get("observed")}
        carried = []
        for row in rows:
            if (row["timestamp"], self._decode_content(row["content"])) in in_child:
                continue
            message: dict[str, Any] = {
                "role": row["role"], "content": self._decode_content(row["content"]), "observed": True}
            if row["platform_message_id"]:
                message["message_id"] = row["platform_message_id"]
            carried.append(message)
        split = next(
            (index for index, message in enumerate(messages)
             if isinstance(message.get("timestamp"), (int, float)) and message["timestamp"] > boundaries[-1]),
            len(messages))
        return [*messages[:split], *carried, *messages[split:]]
