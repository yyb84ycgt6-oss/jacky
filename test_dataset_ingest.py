#!/usr/bin/env python3
"""Tests for dataset_ingest — the Off Grid backup ingest + search layer.

Drives the real DatasetStore against real temporary SQLite files. The only
thing varied is the FTS5 capability flag, exercised through the same public
API so both search paths are proven.
"""

import json

import pytest

from dataset_ingest import (
    BackupValidationError,
    DatasetStore,
    parse_offgrid_backup,
)


def make_backup(conversations=None, version=1, fmt="offgrid-backup"):
    return {
        "format": fmt,
        "version": version,
        "exportedAt": "2026-07-14T10:00:00.000Z",
        "conversations": conversations if conversations is not None else [],
        "projects": [],
    }


def make_conversation(conv_id="conv-1", title="Trip planning",
                      updated_at="2026-07-01T10:00:00.000Z", messages=None):
    return {
        "id": conv_id,
        "title": title,
        "modelId": "smollm2-360m",
        "createdAt": "2026-06-01T10:00:00.000Z",
        "updatedAt": updated_at,
        "messages": messages if messages is not None else [
            {"role": "user", "content": "Where should we camp?", "timestamp": 1751364000000},
            {"role": "assistant", "content": "Somewhere off grid.", "timestamp": 1751364060000},
        ],
    }


# ---------------------------------------------------------------------------
# parse_offgrid_backup
# ---------------------------------------------------------------------------

class TestParse:
    def test_accepts_json_text(self):
        payload, skipped = parse_offgrid_backup(json.dumps(make_backup([make_conversation()])))
        assert len(payload["conversations"]) == 1
        assert skipped == 0

    def test_accepts_decoded_dict(self):
        payload, skipped = parse_offgrid_backup(make_backup([make_conversation()]))
        assert payload["conversations"][0]["id"] == "conv-1"
        assert skipped == 0

    def test_rejects_invalid_json_text(self):
        with pytest.raises(BackupValidationError, match="not valid JSON"):
            parse_offgrid_backup("{nope")

    def test_rejects_non_dict(self):
        with pytest.raises(BackupValidationError, match="not an Off Grid backup"):
            parse_offgrid_backup(json.dumps([1, 2, 3]))

    def test_rejects_wrong_format(self):
        with pytest.raises(BackupValidationError, match="not an Off Grid backup"):
            parse_offgrid_backup(make_backup(fmt="other-app"))

    def test_rejects_newer_version(self):
        with pytest.raises(BackupValidationError, match="Unsupported backup version"):
            parse_offgrid_backup(make_backup(version=2))

    def test_rejects_non_int_version(self):
        with pytest.raises(BackupValidationError, match="Unsupported backup version"):
            parse_offgrid_backup(make_backup(version="1"))

    def test_rejects_missing_conversations_array(self):
        raw = make_backup()
        raw["conversations"] = "not-a-list"
        with pytest.raises(BackupValidationError, match="conversations array"):
            parse_offgrid_backup(raw)

    def test_skips_and_counts_corrupt_entries(self):
        corrupt = [
            "a string",
            {"id": 42, "title": "t", "updatedAt": "x", "messages": []},
            {"id": "ok-but-no-title", "updatedAt": "x", "messages": []},
            {"id": "no-messages", "title": "t", "updatedAt": "x"},
        ]
        payload, skipped = parse_offgrid_backup(make_backup(corrupt + [make_conversation()]))
        assert [c["id"] for c in payload["conversations"]] == ["conv-1"]
        assert skipped == 4


# ---------------------------------------------------------------------------
# DatasetStore ingest
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return DatasetStore(db_path=str(tmp_path / "datasets.db"))


@pytest.fixture
def like_store(tmp_path, monkeypatch):
    """A store built without FTS5, forcing the LIKE fallback path."""
    monkeypatch.setattr(DatasetStore, "_detect_fts", lambda self: False)
    return DatasetStore(db_path=str(tmp_path / "datasets-like.db"))


class TestIngest:
    def test_fresh_ingest_adds(self, store):
        payload, _ = parse_offgrid_backup(make_backup([make_conversation()]))
        result = store.ingest(payload)
        assert result == {"added": 1, "updated": 0, "unchanged": 0}
        assert store.stats()["conversations"] == 1
        assert store.stats()["messages"] == 2

    def test_reingest_identical_is_unchanged(self, store):
        payload, _ = parse_offgrid_backup(make_backup([make_conversation()]))
        store.ingest(payload)
        result = store.ingest(payload)
        assert result == {"added": 0, "updated": 0, "unchanged": 1}
        assert store.stats()["messages"] == 2  # no duplicate rows

    def test_newer_copy_replaces_title_and_messages(self, store):
        old, _ = parse_offgrid_backup(make_backup([make_conversation()]))
        store.ingest(old)
        newer, _ = parse_offgrid_backup(make_backup([make_conversation(
            title="Trip planning (revised)",
            updated_at="2026-07-10T10:00:00.000Z",
            messages=[{"role": "user", "content": "Changed plan entirely.", "timestamp": 1}],
        )]))
        result = store.ingest(newer)
        assert result == {"added": 0, "updated": 1, "unchanged": 0}
        hits = store.search("Changed plan")
        assert len(hits) == 1
        assert hits[0]["title"] == "Trip planning (revised)"
        assert store.stats()["messages"] == 1  # old messages replaced, not appended

    def test_older_copy_never_overwrites(self, store):
        current, _ = parse_offgrid_backup(make_backup([make_conversation(
            updated_at="2026-07-10T10:00:00.000Z")]))
        store.ingest(current)
        stale, _ = parse_offgrid_backup(make_backup([make_conversation(
            title="Stale", updated_at="2026-07-01T10:00:00.000Z")]))
        result = store.ingest(stale)
        assert result == {"added": 0, "updated": 0, "unchanged": 1}
        assert store.search("camp")[0]["title"] == "Trip planning"

    def test_equal_timestamp_keeps_local(self, store):
        payload, _ = parse_offgrid_backup(make_backup([make_conversation()]))
        store.ingest(payload)
        same_time, _ = parse_offgrid_backup(make_backup([make_conversation(title="Tie")]))
        assert store.ingest(same_time)["unchanged"] == 1

    def test_unparseable_incoming_timestamp_never_overwrites(self, store):
        payload, _ = parse_offgrid_backup(make_backup([make_conversation()]))
        store.ingest(payload)
        garbage, _ = parse_offgrid_backup(make_backup([make_conversation(
            title="Garbage time", updated_at="not-a-date")]))
        assert store.ingest(garbage)["unchanged"] == 1

    def test_unparseable_stored_timestamp_never_overwrites(self, store):
        first, _ = parse_offgrid_backup(make_backup([make_conversation(updated_at="not-a-date")]))
        store.ingest(first)
        second, _ = parse_offgrid_backup(make_backup([make_conversation(
            title="Valid time", updated_at="2026-07-10T10:00:00.000Z")]))
        assert store.ingest(second)["unchanged"] == 1

    def test_malformed_messages_are_normalized(self, store):
        conv = make_conversation(messages=[
            "not a dict",
            {"role": 42, "content": 99, "timestamp": "soon"},
            {"role": "user", "content": "real one", "timestamp": 5},
        ])
        payload, _ = parse_offgrid_backup(make_backup([conv]))
        store.ingest(payload)
        # Non-dict skipped; bad fields normalized; the real message searchable.
        assert store.stats()["messages"] == 2
        assert len(store.search("real one")) == 1


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:
    def _seed(self, s):
        payload, _ = parse_offgrid_backup(make_backup([
            make_conversation(),
            make_conversation(conv_id="conv-2", title="Recipes",
                              updated_at="2026-07-02T10:00:00.000Z",
                              messages=[{"role": "user", "content": "Bread without an oven?",
                                         "timestamp": 1}]),
        ]))
        s.ingest(payload)

    def test_finds_by_message_content(self, store):
        self._seed(store)
        hits = store.search("oven")
        assert len(hits) == 1
        assert hits[0]["conversation_id"] == "conv-2"
        assert hits[0]["title"] == "Recipes"
        assert "oven" in hits[0]["snippet"].lower()

    def test_empty_query_returns_nothing(self, store):
        self._seed(store)
        assert store.search("") == []
        assert store.search("   ") == []

    def test_fts_syntax_characters_do_not_crash(self, store):
        self._seed(store)
        assert store.search('camp" OR 1=1 --') == []
        assert store.search("bread AND oven*") == []

    def test_limit_is_clamped(self, store):
        self._seed(store)
        assert store.search("camp", limit=0) is not None
        assert store.search("camp", limit=9999) is not None

    def test_like_fallback_ingest_and_search(self, like_store):
        assert like_store.fts_enabled is False
        self._seed(like_store)
        hits = like_store.search("oven")
        assert len(hits) == 1
        assert hits[0]["title"] == "Recipes"
        assert like_store.stats()["fts_enabled"] is False


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_empty_store(self, store):
        stats = store.stats()
        assert stats["conversations"] == 0
        assert stats["messages"] == 0
        assert stats["latest_updated_at"] is None

    def test_counts_and_latest(self, store):
        payload, _ = parse_offgrid_backup(make_backup([
            make_conversation(),
            make_conversation(conv_id="conv-2", updated_at="2026-07-05T10:00:00.000Z"),
        ]))
        store.ingest(payload)
        stats = store.stats()
        assert stats["conversations"] == 2
        assert stats["messages"] == 4
        assert stats["latest_updated_at"] == "2026-07-05T10:00:00.000Z"
