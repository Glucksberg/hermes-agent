"""Observed gateway rows (unaddressed group chatter, e.g. Telegram media) across compaction and
compression splits for SessionDB. Observed rows bypass model-generated summaries, so retaining them
without a bound would rebuild an oversized live context: one global cap
(``hermes_state.MAX_OBSERVED_CONTEXT_MESSAGES``, late-imported from the facade) covers the in-place
flow and the split-session flow. Bound onto ``SessionDB`` via the MRO.
"""

from __future__ import annotations

from typing import Any

from hermes_state_common import _placeholders


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
        """*compacted_messages* with the newest live observed rows of *session_id* kept ahead of them.

        Observed rows are never summarized away, but the in-place archive deactivates every live row, so
        they are re-inserted (reusing the compacted copy when one exists). Only rows at or below
        *watermark* are handled here: the concurrent-append path clones the newer ones itself. Observed
        dicts in *compacted_messages* that are not among the retained rows are dropped.
        """
        from hermes_state import MAX_OBSERVED_CONTEXT_MESSAGES

        bound = "" if watermark is None else " AND id <= ?"
        rows = conn.execute(
            "SELECT role, content, timestamp, platform_message_id FROM ("
            "  SELECT id AS message_row_id, role, content, timestamp, platform_message_id "
            f"  FROM messages WHERE session_id = ? AND active = 1 AND observed = 1{bound} "
            "  ORDER BY id DESC LIMIT ?"
            ") ORDER BY message_row_id",
            (session_id, *(() if watermark is None else (int(watermark),)), MAX_OBSERVED_CONTEXT_MESSAGES),
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
        return [*retained, *(message for message in compacted_messages if not message.get("observed"))]

    def _drop_observed_beyond_cap(self, conn, row_ids: list[int]) -> list[int]:
        """*row_ids* minus the observed rows older than the newest ``MAX_OBSERVED_CONTEXT_MESSAGES`` of them.

        For the proved-coverage path, which clones every row the compressor never held (all observed
        rows, on Telegram) after the compacted set: the cap must hold there too. Dropped rows are
        archived with the rest of the live set.
        """
        from hermes_state import MAX_OBSERVED_CONTEXT_MESSAGES

        if len(row_ids) <= MAX_OBSERVED_CONTEXT_MESSAGES:
            return row_ids
        excess = {int(row["id"]) for row in conn.execute(
            f"SELECT id FROM messages WHERE observed = 1 AND id IN ({_placeholders(row_ids)}) "
            "ORDER BY id DESC LIMIT -1 OFFSET ?", [*row_ids, MAX_OBSERVED_CONTEXT_MESSAGES]).fetchall()}
        return [row_id for row_id in row_ids if row_id not in excess]

    def get_ancestor_observed_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Return observed=True messages from compression-split ancestor sessions.

        When context compression creates a new session, observed messages stored in the parent session
        (e.g. unaddressed media/audio in a Telegram group) are no longer accessible via the new session's
        transcript. This method fetches those messages so callers can surface them as context.

        Only the verified compression lineage counts (each hop's parent ended ``compression``), so
        branches, resets and delegate-subagent parents that share a ``parent_session_id`` link never
        leak in. The newest ``MAX_OBSERVED_CONTEXT_MESSAGES`` rows are selected globally across those
        ancestors, then returned in stable chronological order.
        """
        from hermes_state import MAX_OBSERVED_CONTEXT_MESSAGES

        ancestor_ids = self._resume_lineage_ids(session_id)[:-1]
        if not ancestor_ids:
            return []
        rows = self._read_all(
            "SELECT role, content, platform_message_id FROM ("
            "  SELECT id AS message_row_id, role, content, platform_message_id "
            f"  FROM messages WHERE session_id IN ({_placeholders(ancestor_ids)}) AND observed = 1 AND active = 1 "
            "  ORDER BY id DESC LIMIT ?"
            ") ORDER BY message_row_id",
            (*ancestor_ids, MAX_OBSERVED_CONTEXT_MESSAGES))
        result = []
        for row in rows:
            message: dict[str, Any] = {
                "role": row["role"], "content": self._decode_content(row["content"]), "observed": True}
            if row["platform_message_id"]:
                message["message_id"] = row["platform_message_id"]
            result.append(message)
        return result
