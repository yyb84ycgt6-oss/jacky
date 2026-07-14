#!/usr/bin/env python3
"""
Off Grid conversation dataset ingest — the data-centre end of the workstation loop.

The Off Grid mobile app (Storage -> Backup -> Export data) exports the user's
conversations and projects as one JSON file:

    {
      "format": "offgrid-backup",
      "version": 1,
      "exportedAt": "...",
      "conversations": [{ "id", "title", "modelId", "messages": [...], ... }],
      "projects": [...]
    }

This module ingests those files into a local SQLite dataset and serves
full-text search over every message ever exported. The merge contract matches
the app exactly: same conversation id -> the copy with the newer updatedAt
wins; unparseable timestamps never overwrite; corrupt entries are skipped and
counted, never fatal to the rest of the file.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("DatasetIngest")

BACKUP_FORMAT = "offgrid-backup"
MAX_SUPPORTED_VERSION = 1


class BackupValidationError(ValueError):
    """Raised when a payload is structurally not an Off Grid backup."""


def _parse_iso(ts) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp; None when unparseable (never overwrites)."""
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _valid_conversation(entry) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and isinstance(entry.get("title"), str)
        and isinstance(entry.get("updatedAt"), str)
        and isinstance(entry.get("messages"), list)
    )


def parse_offgrid_backup(raw) -> Tuple[Dict, int]:
    """Validate a backup payload (JSON text or already-decoded dict).

    Returns (payload, skipped_count). Structural problems raise
    BackupValidationError; corrupt individual conversations are skipped.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raise BackupValidationError("Body is not valid JSON.")

    if not isinstance(raw, dict) or raw.get("format") != BACKUP_FORMAT:
        raise BackupValidationError("This is not an Off Grid backup.")

    version = raw.get("version")
    if not isinstance(version, int) or version > MAX_SUPPORTED_VERSION:
        raise BackupValidationError(
            f"Unsupported backup version {version!r} (max {MAX_SUPPORTED_VERSION})."
        )

    conversations = raw.get("conversations")
    if not isinstance(conversations, list):
        raise BackupValidationError("Backup is missing its conversations array.")

    valid = [c for c in conversations if _valid_conversation(c)]
    skipped = len(conversations) - len(valid)

    return (
        {
            "format": BACKUP_FORMAT,
            "version": version,
            "exportedAt": raw.get("exportedAt", ""),
            "conversations": valid,
        },
        skipped,
    )


class DatasetStore:
    """SQLite-backed store for ingested conversation datasets.

    Full-text search uses FTS5 when the local SQLite build has it, and falls
    back to LIKE matching when it does not. The fallback is a capability of
    the store, not the caller's problem.
    """

    def __init__(self, db_path: str = "jacky_datasets.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.fts_enabled = self._detect_fts()
        self._init_db()
        log.info(f"DatasetStore ready at {db_path} (fts={self.fts_enabled})")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _detect_fts(self) -> bool:
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE VIRTUAL TABLE t USING fts5(c)")
            conn.close()
            return True
        except sqlite3.OperationalError:
            return False

    def _init_db(self):
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    model_id TEXT,
                    project_id TEXT,
                    created_at TEXT,
                    updated_at TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    ingested_at TEXT NOT NULL,
                    branch TEXT NOT NULL DEFAULT 'main'
                )
                """
            )
            # Databases created before branches existed get the column added.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(conversations)")}
            if "branch" not in cols:
                conn.execute(
                    "ALTER TABLE conversations ADD COLUMN branch TEXT NOT NULL DEFAULT 'main'"
                )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    conversation_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    role TEXT,
                    content TEXT NOT NULL DEFAULT '',
                    timestamp INTEGER,
                    PRIMARY KEY (conversation_id, idx)
                )
                """
            )
            if self.fts_enabled:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
                    USING fts5(conversation_id UNINDEXED, content)
                    """
                )
            conn.commit()
        finally:
            conn.close()

    def _replace_messages(self, conn, conv_id: str, messages: List[Dict]):
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        if self.fts_enabled:
            conn.execute("DELETE FROM messages_fts WHERE conversation_id = ?", (conv_id,))
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                content = ""
            role = msg.get("role") if isinstance(msg.get("role"), str) else None
            ts = msg.get("timestamp") if isinstance(msg.get("timestamp"), (int, float)) else None
            conn.execute(
                "INSERT INTO messages (conversation_id, idx, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                (conv_id, idx, role, content, ts),
            )
            if self.fts_enabled and content:
                conn.execute(
                    "INSERT INTO messages_fts (conversation_id, content) VALUES (?, ?)",
                    (conv_id, content),
                )

    def ingest(self, payload: Dict, branch: str = "main") -> Dict:
        """Upsert every conversation in a parsed payload. Newer updatedAt wins.

        branch names the dataset room shelf this import belongs to ('main',
        'claude-exports', 'game-chat', ...). Re-importing a conversation into
        a different branch moves it there when the copy is newer.
        """
        if not isinstance(branch, str) or not branch.strip():
            branch = "main"
        branch = branch.strip()
        added = updated = unchanged = 0
        with self._lock:
            conn = self._connect()
            try:
                for conv in payload["conversations"]:
                    row = conn.execute(
                        "SELECT updated_at FROM conversations WHERE id = ?", (conv["id"],)
                    ).fetchone()
                    if row is not None:
                        current = _parse_iso(row["updated_at"])
                        incoming = _parse_iso(conv["updatedAt"])
                        if current is None or incoming is None or incoming <= current:
                            unchanged += 1
                            continue
                        updated += 1
                    else:
                        added += 1
                    messages = conv.get("messages", [])
                    conn.execute(
                        """
                        INSERT INTO conversations
                            (id, title, model_id, project_id, created_at, updated_at, message_count, ingested_at, branch)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            title = excluded.title,
                            model_id = excluded.model_id,
                            project_id = excluded.project_id,
                            created_at = excluded.created_at,
                            updated_at = excluded.updated_at,
                            message_count = excluded.message_count,
                            ingested_at = excluded.ingested_at,
                            branch = excluded.branch
                        """,
                        (
                            conv["id"],
                            conv["title"],
                            conv.get("modelId"),
                            conv.get("projectId"),
                            conv.get("createdAt"),
                            conv["updatedAt"],
                            len(messages),
                            datetime.utcnow().isoformat(),
                            branch,
                        ),
                    )
                    self._replace_messages(conn, conv["id"], messages)
                conn.commit()
            finally:
                conn.close()
        result = {"added": added, "updated": updated, "unchanged": unchanged}
        log.info(f"Ingest applied: {result}")
        return result

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """Full-text search across every ingested message."""
        if not query or not query.strip():
            return []
        limit = max(1, min(int(limit), 100))
        conn = self._connect()
        try:
            if self.fts_enabled:
                # Quote the query so user text is treated as terms, not FTS syntax.
                quoted = '"' + query.replace('"', '""') + '"'
                rows = conn.execute(
                    """
                    SELECT f.conversation_id, snippet(messages_fts, 1, '[', ']', '...', 12) AS snip,
                           c.title, c.updated_at
                    FROM messages_fts f
                    JOIN conversations c ON c.id = f.conversation_id
                    WHERE messages_fts MATCH ?
                    ORDER BY c.updated_at DESC
                    LIMIT ?
                    """,
                    (quoted, limit),
                ).fetchall()
            else:
                like = f"%{query}%"
                rows = conn.execute(
                    """
                    SELECT m.conversation_id, substr(m.content, 1, 120) AS snip,
                           c.title, c.updated_at
                    FROM messages m
                    JOIN conversations c ON c.id = m.conversation_id
                    WHERE m.content LIKE ?
                    ORDER BY c.updated_at DESC
                    LIMIT ?
                    """,
                    (like, limit),
                ).fetchall()
            return [
                {
                    "conversation_id": r["conversation_id"],
                    "title": r["title"],
                    "updated_at": r["updated_at"],
                    "snippet": r["snip"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def stats(self) -> Dict:
        """Totals for the dashboard: how much of the personal dataset is here."""
        conn = self._connect()
        try:
            convs = conn.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()["n"]
            msgs = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
            latest = conn.execute(
                "SELECT MAX(updated_at) AS latest FROM conversations"
            ).fetchone()["latest"]
            branches = {
                r["branch"]: r["n"]
                for r in conn.execute(
                    "SELECT branch, COUNT(*) AS n FROM conversations GROUP BY branch"
                )
            }
            return {
                "conversations": convs,
                "messages": msgs,
                "latest_updated_at": latest,
                "branches": branches,
                "fts_enabled": self.fts_enabled,
            }
        finally:
            conn.close()

    def labeled_examples(self, label_by: str = "project_id", role: str = "user",
                         branch: Optional[str] = None,
                         min_per_label: int = 2) -> List[Tuple[str, str]]:
        """(text, label) pairs for router training, straight from the datasets.

        label_by: 'project_id', 'model_id', or 'branch' — whichever grouping
        the router should learn to tell apart. Labels rarer than min_per_label
        are dropped so a one-off cannot become a class.
        """
        if label_by not in ("project_id", "model_id", "branch"):
            raise ValueError("label_by must be project_id, model_id, or branch")
        conn = self._connect()
        try:
            sql = (
                f"SELECT m.content AS text, c.{label_by} AS label "
                "FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                f"WHERE c.{label_by} IS NOT NULL AND c.{label_by} != '' "
                "AND m.content != '' AND m.role = ?"
            )
            params: List = [role]
            if branch is not None:
                sql += " AND c.branch = ?"
                params.append(branch)
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        counts: Dict[str, int] = {}
        for r in rows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        return [(r["text"], r["label"]) for r in rows if counts[r["label"]] >= min_per_label]
