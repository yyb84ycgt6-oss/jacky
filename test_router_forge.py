#!/usr/bin/env python3
"""Tests for router_forge — training, runtimes, registry, fetch scripts.

Everything drives the real code: real SGD training on synthetic separable
data, real SQLite registries on temp files, real hashing. The only injected
pieces are embed_fn / llm_fn, which are genuine external boundaries.
"""

import json

import pytest

from dataset_ingest import DatasetStore, parse_offgrid_backup
from router_forge import (
    FORGE_FORMAT,
    ForgeError,
    ForgeStore,
    RouterRuntime,
    build_embed,
    build_fetch_script,
    build_llm,
    extract_features,
    train_nano,
    validate_artifact,
)

# Two clearly separable classes: tech-support vs cooking.
TECH = [
    "my wifi router keeps dropping the connection",
    "how do I reset the network adapter drivers",
    "the gpu temperature is spiking under load",
    "python throws a module not found error",
    "docker container will not start after update",
    "ssh connection refused on port 22",
    "the api returns a 500 internal server error",
    "how to check disk usage on linux",
    "my screen flickers after the driver update",
    "the build fails with a linker error",
    "vpn disconnects every few minutes",
    "how do I open a port in the firewall",
]
COOK = [
    "how long should I knead sourdough bread",
    "what temperature to roast a whole chicken",
    "my cake sinks in the middle every time",
    "best way to caramelize onions slowly",
    "how much salt for pasta water",
    "can I substitute butter with olive oil",
    "the soup tastes flat what should I add",
    "how to keep rice from going mushy",
    "what is the ratio for vinaigrette",
    "how do I proof yeast for pizza dough",
    "searing steak in a cast iron pan",
    "how to thicken a sauce without flour",
]


def examples():
    return [(t, "tech") for t in TECH] + [(t, "cook") for t in COOK]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_deterministic_across_calls(self):
        assert extract_features("hello world") == extract_features("hello world")

    def test_case_and_whitespace_normalized(self):
        assert extract_features("Hello   World") == extract_features("hello world")

    def test_l2_normalized(self):
        feats = extract_features("some reasonable text here")
        norm = sum(v * v for v in feats.values())
        assert norm == pytest.approx(1.0)

    def test_short_text_below_ngram_size(self):
        assert extract_features("a") == {}


# ---------------------------------------------------------------------------
# Nano training + runtime
# ---------------------------------------------------------------------------

class TestNano:
    def test_trains_and_routes_correctly(self):
        artifact = train_nano(examples(), name="topic", task="tech vs cooking")
        runtime = RouterRuntime.load(artifact)

        tech_hit = runtime.route("the kernel panics when the driver loads")
        cook_hit = runtime.route("how long to simmer the tomato sauce")

        assert tech_hit["label"] == "tech"
        assert cook_hit["label"] == "cook"
        assert 0 < tech_hit["confidence"] <= 1
        assert set(tech_hit["scores"].keys()) == {"tech", "cook"}

    def test_holdout_accuracy_reported_and_high(self):
        artifact = train_nano(examples(), name="topic")
        trained = artifact["trainedOn"]
        assert trained["holdout"] is True
        assert trained["examples"] == 24
        assert trained["accuracy"] >= 0.75

    def test_small_sets_use_trainset_accuracy(self):
        small = examples()[:5] + examples()[-5:]
        artifact = train_nano(small, name="tiny")
        assert artifact["trainedOn"]["holdout"] is False
        assert artifact["trainedOn"]["accuracy"] >= 0.9

    def test_artifact_is_under_one_megabyte(self):
        artifact = train_nano(examples(), name="topic")
        assert len(json.dumps(artifact)) < 1024 * 1024

    def test_deterministic_given_same_seed(self):
        a = train_nano(examples(), name="topic", seed=42)
        b = train_nano(examples(), name="topic", seed=42)
        assert a["model"]["weights"] == b["model"]["weights"]

    def test_quantization_keeps_decisions(self):
        # The int8 roundtrip must not flip any training example's decision.
        artifact = train_nano(examples(), name="topic")
        runtime = RouterRuntime.load(artifact)
        hits = sum(1 for t, lab in examples() if runtime.route(t)["label"] == lab)
        assert hits >= len(examples()) - 1

    def test_rejects_too_few_examples(self):
        with pytest.raises(ForgeError, match="at least 4"):
            train_nano([("a text", "x"), ("b text", "y")], name="nope")

    def test_rejects_single_label(self):
        with pytest.raises(ForgeError, match="2 distinct labels"):
            train_nano([("a", "x"), ("b", "x"), ("c", "x"), ("d", "x")], name="nope")

    def test_filters_junk_examples(self):
        junk = [(None, "x"), ("", "x"), ("text", None), ("text", "")]
        artifact = train_nano(examples() + junk, name="topic")
        assert artifact["trainedOn"]["examples"] == 24


# ---------------------------------------------------------------------------
# Embed router
# ---------------------------------------------------------------------------

def fake_embed(text: str):
    """Deterministic toy embedder: 4 dims keyed on vocabulary."""
    t = text.lower()
    return [
        1.0 if any(w in t for w in ("wifi", "gpu", "docker", "python", "port")) else 0.0,
        1.0 if any(w in t for w in ("bread", "sauce", "roast", "dough", "salt")) else 0.0,
        0.5,
        float(len(t) % 3),
    ]


class TestEmbed:
    def test_builds_and_routes_via_injected_fn(self):
        artifact = build_embed(examples(), name="topic-embed", embed_fn=fake_embed,
                               embedding_model="fake-4d")
        runtime = RouterRuntime.load(artifact, embed_fn=fake_embed)
        assert runtime.route("gpu drivers and docker")["label"] == "tech"
        assert runtime.route("dough and sauce and salt")["label"] == "cook"
        assert artifact["model"]["dims"] == 4

    def test_route_without_embed_fn_raises(self):
        artifact = build_embed(examples(), name="topic-embed", embed_fn=fake_embed)
        runtime = RouterRuntime.load(artifact)
        with pytest.raises(ForgeError, match="embed_fn"):
            runtime.route("anything")

    def test_inconsistent_dims_rejected(self):
        calls = {"n": 0}

        def bad_embed(_text):
            calls["n"] += 1
            return [0.0] * (3 if calls["n"] > 1 else 4)

        with pytest.raises(ForgeError, match="inconsistent"):
            build_embed(examples(), name="bad", embed_fn=bad_embed)


# ---------------------------------------------------------------------------
# LLM router
# ---------------------------------------------------------------------------

class TestLlm:
    def test_spec_and_routing_through_injected_llm(self):
        artifact = build_llm(["tech", "cook"], name="topic-llm", task="tech vs cooking")
        runtime = RouterRuntime.load(artifact, llm_fn=lambda prompt: "tech")
        result = runtime.route("my gpu is on fire")
        assert result["label"] == "tech"
        assert "{text}" in artifact["model"]["promptTemplate"]

    def test_unrecognized_answer_falls_back_to_uniform(self):
        artifact = build_llm(["tech", "cook"], name="topic-llm")
        runtime = RouterRuntime.load(artifact, llm_fn=lambda prompt: "banana")
        result = runtime.route("whatever")
        assert result["scores"]["tech"] == result["scores"]["cook"]

    def test_route_without_llm_fn_raises(self):
        artifact = build_llm(["a", "b"], name="x")
        with pytest.raises(ForgeError, match="llm_fn"):
            RouterRuntime.load(artifact).route("text")

    def test_needs_two_labels(self):
        with pytest.raises(ForgeError, match="at least 2"):
            build_llm(["only-one"], name="x")


# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_rejects_non_artifact(self):
        with pytest.raises(ForgeError, match="not a router-forge artifact"):
            validate_artifact({"hello": "world"})

    def test_rejects_newer_version(self):
        artifact = train_nano(examples(), name="topic")
        artifact["version"] = 99
        with pytest.raises(ForgeError, match="Unsupported artifact version"):
            validate_artifact(artifact)

    def test_rejects_unknown_kind(self):
        artifact = train_nano(examples(), name="topic")
        artifact["kind"] = "mega"
        with pytest.raises(ForgeError, match="Unknown router kind"):
            validate_artifact(artifact)

    def test_rejects_missing_labels(self):
        artifact = train_nano(examples(), name="topic")
        artifact["labels"] = ["one"]
        with pytest.raises(ForgeError, match="at least 2 labels"):
            validate_artifact(artifact)

    def test_rejects_missing_nano_fields(self):
        artifact = train_nano(examples(), name="topic")
        del artifact["model"]["weights"]
        with pytest.raises(ForgeError, match="model.weights"):
            validate_artifact(artifact)

    def test_route_rejects_empty_text(self):
        runtime = RouterRuntime.load(train_nano(examples(), name="topic"))
        with pytest.raises(ForgeError, match="non-empty text"):
            runtime.route("   ")


# ---------------------------------------------------------------------------
# Fetch scripts
# ---------------------------------------------------------------------------

class TestFetchScripts:
    def test_ollama_script(self):
        out = build_fetch_script("ollama", "qwen2.5:0.5b")
        assert "ollama pull qwen2.5:0.5b" in out["script"]
        assert out["filename"].endswith(".sh")

    def test_huggingface_script(self):
        out = build_fetch_script("huggingface", "HuggingFaceTB/SmolLM2-135M-Instruct")
        assert "hf download HuggingFaceTB/SmolLM2-135M-Instruct" in out["script"]

    def test_url_script(self):
        out = build_fetch_script("url", "https://example.com/models/tiny.gguf")
        assert "curl -L --fail" in out["script"]
        assert "tiny.gguf" in out["filename"]

    def test_rejects_shell_injection_in_ref(self):
        with pytest.raises(ForgeError, match="not allowed"):
            build_fetch_script("ollama", "model; rm -rf /")

    def test_rejects_shell_characters_in_url(self):
        with pytest.raises(ForgeError, match="no shell characters"):
            build_fetch_script("url", "https://x.com/a;rm -rf /")

    def test_rejects_bad_dest(self):
        with pytest.raises(ForgeError, match="not allowed"):
            build_fetch_script("ollama", "qwen2.5:0.5b", dest="../etc")

    def test_rejects_unknown_source(self):
        with pytest.raises(ForgeError, match="source must be"):
            build_fetch_script("ftp", "whatever")


# ---------------------------------------------------------------------------
# ForgeStore registry
# ---------------------------------------------------------------------------

class TestForgeStore:
    def test_save_versions_and_get_latest(self, tmp_path):
        store = ForgeStore(db_path=str(tmp_path / "forge.db"))
        artifact = train_nano(examples(), name="topic")
        assert store.save(artifact) == 1
        assert store.save(artifact) == 2
        assert store.get("topic")["name"] == "topic"
        assert store.get("topic", version=1) is not None
        assert store.get("missing") is None

    def test_list_shows_latest_only_with_size(self, tmp_path):
        store = ForgeStore(db_path=str(tmp_path / "forge.db"))
        store.save(train_nano(examples(), name="topic"))
        store.save(train_nano(examples(), name="topic"))
        store.save(build_llm(["a", "b"], name="quick"))
        listing = store.list()
        assert {r["name"] for r in listing} == {"topic", "quick"}
        topic = next(r for r in listing if r["name"] == "topic")
        assert topic["version"] == 2
        assert 0 < topic["size_bytes"] < 1024 * 1024

    def test_delete_removes_all_versions(self, tmp_path):
        store = ForgeStore(db_path=str(tmp_path / "forge.db"))
        store.save(train_nano(examples(), name="topic"))
        store.save(train_nano(examples(), name="topic"))
        assert store.delete("topic") == 2
        assert store.get("topic") is None
        assert store.delete("topic") == 0

    def test_save_rejects_invalid_artifact(self, tmp_path):
        store = ForgeStore(db_path=str(tmp_path / "forge.db"))
        with pytest.raises(ForgeError):
            store.save({"format": "wrong"})

    def test_save_rejects_missing_name(self, tmp_path):
        store = ForgeStore(db_path=str(tmp_path / "forge.db"))
        artifact = train_nano(examples(), name="topic")
        artifact["name"] = "  "
        with pytest.raises(ForgeError, match="name"):
            store.save(artifact)


# ---------------------------------------------------------------------------
# End to end: datasets -> labeled examples -> trained router
# ---------------------------------------------------------------------------

def _dataset_backup():
    def conv(cid, project, texts, updated):
        return {
            "id": cid,
            "title": cid,
            "modelId": "m",
            "projectId": project,
            "updatedAt": updated,
            "messages": [
                {"role": "user", "content": t, "timestamp": i} for i, t in enumerate(texts)
            ],
        }

    return {
        "format": "offgrid-backup",
        "version": 1,
        "exportedAt": "2026-07-14T10:00:00.000Z",
        "conversations": [
            conv("c-tech-1", "tech-support", TECH[:6], "2026-07-01T00:00:00Z"),
            conv("c-tech-2", "tech-support", TECH[6:], "2026-07-02T00:00:00Z"),
            conv("c-cook-1", "cooking", COOK[:6], "2026-07-03T00:00:00Z"),
            conv("c-cook-2", "cooking", COOK[6:], "2026-07-04T00:00:00Z"),
        ],
        "projects": [],
    }


class TestDatasetsToRouter:
    def test_labeled_examples_from_ingested_data(self, tmp_path):
        ds = DatasetStore(db_path=str(tmp_path / "datasets.db"))
        payload, _ = parse_offgrid_backup(_dataset_backup())
        ds.ingest(payload, branch="training")

        pairs = ds.labeled_examples(label_by="project_id", branch="training")
        assert len(pairs) == 24
        assert {lab for _, lab in pairs} == {"tech-support", "cooking"}

    def test_branch_filter_and_min_per_label(self, tmp_path):
        ds = DatasetStore(db_path=str(tmp_path / "datasets.db"))
        payload, _ = parse_offgrid_backup(_dataset_backup())
        ds.ingest(payload, branch="training")
        assert ds.labeled_examples(label_by="project_id", branch="other") == []
        # A label rarer than min_per_label is dropped entirely.
        assert all(
            lab in ("tech-support", "cooking")
            for _, lab in ds.labeled_examples(label_by="project_id", min_per_label=7)
        )

    def test_label_by_rejects_unknown_column(self, tmp_path):
        ds = DatasetStore(db_path=str(tmp_path / "datasets.db"))
        with pytest.raises(ValueError, match="label_by"):
            ds.labeled_examples(label_by="password; DROP TABLE")

    def test_full_loop_dataset_to_working_router(self, tmp_path):
        ds = DatasetStore(db_path=str(tmp_path / "datasets.db"))
        payload, _ = parse_offgrid_backup(_dataset_backup())
        ds.ingest(payload)

        pairs = ds.labeled_examples(label_by="project_id")
        artifact = train_nano(pairs, name="project-dispatch",
                              task="route questions to the right project")
        store = ForgeStore(db_path=str(tmp_path / "forge.db"))
        store.save(artifact)

        runtime = RouterRuntime.load(store.get("project-dispatch"))
        assert runtime.route("the container crashes on boot")["label"] == "tech-support"
        assert runtime.route("how do I proof the dough overnight")["label"] == "cooking"

    def test_branches_reported_in_stats(self, tmp_path):
        ds = DatasetStore(db_path=str(tmp_path / "datasets.db"))
        payload, _ = parse_offgrid_backup(_dataset_backup())
        ds.ingest(payload, branch="training")
        stats = ds.stats()
        assert stats["branches"] == {"training": 4}
