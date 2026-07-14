#!/usr/bin/env python3
"""
Router Forge — generate router artifacts at any size and attach them to anything.

A router is a tiny decision function: route(text) -> {label, confidence, scores}.
The labels are yours; the forge does not care whether they name models, memory
branches, tools, bots, squads, or game moves. Three kinds, one artifact format:

  nano   under 1 MB. Hashed character n-grams -> logistic regression, int8
         quantized. Pure stdlib to train and to run. The same artifact runs in
         Python here and in a ~100 line TypeScript runtime on the phone.
  embed  centroid-per-label over sentence embeddings (inject any embed_fn;
         an Ollama adapter is provided). Artifact stays small; the embedding
         model is whatever the host already has.
  llm    a prompt spec executed by any LLM (Ollama local or the cloud
         waterfall). No training, instant to forge, the heavyweight option.

Artifact envelope (format "router-forge", version 1):
  { format, version, kind, name, task, labels, trainedOn, model }

The forge also generates fetch scripts for any public model (Ollama, Hugging
Face, direct URL) so new capability can be pulled onto whichever tier needs it.
Scripts are TEXT the user runs; the server never executes them.
"""

import base64
import json
import logging
import math
import random
import re
import sqlite3
import struct
import threading
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("RouterForge")

FORGE_FORMAT = "router-forge"
FORGE_VERSION = 1

DEFAULT_HASH_DIM = 16384
DEFAULT_NGRAMS = (2, 3, 4)


class ForgeError(ValueError):
    """Raised for invalid artifacts or unusable training input. User-presentable."""


# ---------------------------------------------------------------------------
# Feature hashing — FNV-1a, chosen because it is trivial to replicate
# byte-for-byte in any language (the phone runtime must hash identically).
# ---------------------------------------------------------------------------

def _fnv1a(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def extract_features(text: str, hash_dim: int = DEFAULT_HASH_DIM,
                     ngrams: Sequence[int] = DEFAULT_NGRAMS) -> Dict[int, float]:
    """Hashed char n-gram counts, L2 normalized. Sparse {index: value}."""
    normalized = " ".join(text.lower().split())
    counts: Dict[int, float] = {}
    for n in ngrams:
        if len(normalized) < n:
            continue
        for i in range(len(normalized) - n + 1):
            idx = _fnv1a(normalized[i:i + n].encode("utf-8")) % hash_dim
            counts[idx] = counts.get(idx, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in counts.values()))
    if norm > 0:
        for idx in counts:
            counts[idx] /= norm
    return counts


# ---------------------------------------------------------------------------
# Nano router: one-vs-rest logistic regression, SGD, pure stdlib.
# ---------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _train_ovr(features: List[Dict[int, float]], label_ids: List[int],
               n_labels: int, hash_dim: int, epochs: int, lr: float,
               l2: float, seed: int) -> Tuple[List[List[float]], List[float]]:
    weights = [[0.0] * hash_dim for _ in range(n_labels)]
    biases = [0.0] * n_labels
    order = list(range(len(features)))
    rng = random.Random(seed)
    for epoch in range(epochs):
        rng.shuffle(order)
        rate = lr / (1.0 + 0.1 * epoch)
        for i in order:
            feats = features[i]
            for c in range(n_labels):
                w = weights[c]
                z = biases[c] + sum(w[idx] * val for idx, val in feats.items())
                grad = _sigmoid(z) - (1.0 if label_ids[i] == c else 0.0)
                for idx, val in feats.items():
                    w[idx] -= rate * (grad * val + l2 * w[idx])
                biases[c] -= rate * grad
    return weights, biases


def _predict_ids(features: Dict[int, float], weights: List[List[float]],
                 biases: List[float]) -> List[float]:
    return [biases[c] + sum(weights[c][idx] * val for idx, val in features.items())
            for c in range(len(weights))]


def _softmax(scores: List[float]) -> List[float]:
    peak = max(scores)
    exps = [math.exp(s - peak) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def train_nano(examples: List[Tuple[str, str]], name: str, task: str = "",
               hash_dim: int = DEFAULT_HASH_DIM, ngrams: Sequence[int] = DEFAULT_NGRAMS,
               epochs: int = 12, lr: float = 0.5, l2: float = 1e-5,
               seed: int = 777) -> Dict:
    """Train a nano router from (text, label) pairs. Returns the artifact dict."""
    examples = [(t, lab) for t, lab in examples
                if isinstance(t, str) and t.strip() and isinstance(lab, str) and lab]
    if len(examples) < 4:
        raise ForgeError("Need at least 4 usable (text, label) examples to train.")
    labels = sorted({lab for _, lab in examples})
    if len(labels) < 2:
        raise ForgeError("Need at least 2 distinct labels to train a router.")

    label_index = {lab: i for i, lab in enumerate(labels)}
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)

    feats_all = [extract_features(t, hash_dim, ngrams) for t, _ in shuffled]
    ids_all = [label_index[lab] for _, lab in shuffled]

    # Holdout accuracy when there is enough data; final model trains on all of it.
    holdout = len(shuffled) >= 20
    if holdout:
        cut = max(1, len(shuffled) // 5)
        w, b = _train_ovr(feats_all[cut:], ids_all[cut:], len(labels),
                          hash_dim, epochs, lr, l2, seed)
        hits = sum(
            1 for f, y in zip(feats_all[:cut], ids_all[:cut])
            if max(range(len(labels)), key=lambda c: _predict_ids(f, w, b)[c]) == y
        )
        accuracy = hits / cut
    weights, biases = _train_ovr(feats_all, ids_all, len(labels),
                                 hash_dim, epochs, lr, l2, seed)
    if not holdout:
        hits = sum(
            1 for f, y in zip(feats_all, ids_all)
            if max(range(len(labels)), key=lambda c: _predict_ids(f, weights, biases)[c]) == y
        )
        accuracy = hits / len(shuffled)

    # int8 symmetric quantization per class row.
    scales: List[float] = []
    quantized = bytearray()
    for c in range(len(labels)):
        peak = max(abs(v) for v in weights[c]) or 1.0
        scale = peak / 127.0
        scales.append(scale)
        quantized.extend(
            struct.pack(f"{hash_dim}b",
                        *(max(-127, min(127, round(v / scale))) for v in weights[c]))
        )

    return {
        "format": FORGE_FORMAT,
        "version": FORGE_VERSION,
        "kind": "nano",
        "name": name,
        "task": task,
        "labels": labels,
        "trainedOn": {
            "examples": len(shuffled),
            "accuracy": round(accuracy, 4),
            "holdout": holdout,
        },
        "model": {
            "hashDim": hash_dim,
            "ngrams": list(ngrams),
            "weights": base64.b64encode(bytes(quantized)).decode("ascii"),
            "scales": scales,
            "biases": biases,
        },
    }


# ---------------------------------------------------------------------------
# Embed router: centroid per label over injected embeddings.
# ---------------------------------------------------------------------------

def build_embed(examples: List[Tuple[str, str]], name: str,
                embed_fn: Callable[[str], List[float]], task: str = "",
                embedding_model: str = "unknown") -> Dict:
    """Build a centroid router. embed_fn is injected so any embedder works."""
    examples = [(t, lab) for t, lab in examples
                if isinstance(t, str) and t.strip() and isinstance(lab, str) and lab]
    labels = sorted({lab for _, lab in examples})
    if len(labels) < 2:
        raise ForgeError("Need at least 2 distinct labels to build an embed router.")

    sums: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}
    dims = None
    for text, lab in examples:
        vec = embed_fn(text)
        if dims is None:
            dims = len(vec)
        if len(vec) != dims:
            raise ForgeError("embed_fn returned vectors of inconsistent dimensions.")
        acc = sums.setdefault(lab, [0.0] * dims)
        for i, v in enumerate(vec):
            acc[i] += v
        counts[lab] = counts.get(lab, 0) + 1

    centroids = {}
    for lab in labels:
        mean = [v / counts[lab] for v in sums[lab]]
        norm = math.sqrt(sum(v * v for v in mean)) or 1.0
        centroids[lab] = [v / norm for v in mean]

    return {
        "format": FORGE_FORMAT,
        "version": FORGE_VERSION,
        "kind": "embed",
        "name": name,
        "task": task,
        "labels": labels,
        "trainedOn": {"examples": len(examples), "accuracy": None, "holdout": False},
        "model": {"embeddingModel": embedding_model, "dims": dims, "centroids": centroids},
    }


def ollama_embed_fn(model: str = "nomic-embed-text",
                    host: str = "http://localhost:11434") -> Callable[[str], List[float]]:
    """Adapter: embeddings from a local Ollama. Import requests lazily so the
    forge stays stdlib-only unless this adapter is actually used."""
    import requests  # noqa: PLC0415

    def embed(text: str) -> List[float]:
        resp = requests.post(f"{host}/api/embeddings",
                             json={"model": model, "prompt": text}, timeout=30)
        resp.raise_for_status()
        return resp.json()["embedding"]

    return embed


_minilm_models: Dict[str, object] = {}


def minilm_embed_fn(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Callable[[str], List[float]]:
    """Adapter: sentence-transformers MiniLM, the same 384-dim family the
    phone bundles as all-MiniLM-L6-v2-Q8_0. normalize_embeddings=True gives
    the mean-pooled, L2-normalized vectors (same math as the reference
    tokenize -> mean_pooling -> F.normalize pipeline). The heavy torch
    dependency is imported lazily and the model instance is cached."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError:
        raise ForgeError(
            "sentence-transformers is not installed. Run: pip install -U sentence-transformers"
        )
    if model_name not in _minilm_models:
        _minilm_models[model_name] = SentenceTransformer(model_name)
    model = _minilm_models[model_name]

    def embed(text: str) -> List[float]:
        vec = model.encode(text, normalize_embeddings=True)
        return [float(v) for v in vec]

    return embed


def resolve_embed_fn(embedding_model: str) -> Callable[[str], List[float]]:
    """The one mapping from an artifact's embeddingModel string to an adapter.

    Names containing 'minilm' (any case) run through sentence-transformers;
    everything else is treated as a local Ollama embedding model. Train and
    route must both resolve through here so the decision exists once."""
    name = (embedding_model or "").strip() or "nomic-embed-text"
    if "minilm" in name.lower():
        return minilm_embed_fn(model_name=name)
    return ollama_embed_fn(model=name)


# ---------------------------------------------------------------------------
# LLM router: a prompt spec any engine can execute.
# ---------------------------------------------------------------------------

def build_llm(labels: List[str], name: str, task: str = "") -> Dict:
    """Forge an LLM router spec. No training; the intelligence is rented."""
    labels = [lab for lab in labels if isinstance(lab, str) and lab]
    if len(labels) < 2:
        raise ForgeError("Need at least 2 labels to build an llm router.")
    prompt = (
        "You are a router. Classify the user text into exactly one label.\n"
        f"Task: {task or 'route the text'}\n"
        f"Labels: {', '.join(labels)}\n"
        "Answer with the single label only, nothing else.\n"
        "Text: {text}"
    )
    return {
        "format": FORGE_FORMAT,
        "version": FORGE_VERSION,
        "kind": "llm",
        "name": name,
        "task": task,
        "labels": labels,
        "trainedOn": {"examples": 0, "accuracy": None, "holdout": False},
        "model": {"promptTemplate": prompt},
    }


# ---------------------------------------------------------------------------
# Unified runtime — the single seam every caller depends on.
# ---------------------------------------------------------------------------

class RouterRuntime:
    """Loads any artifact and answers route(text). Callers never branch on kind."""

    def __init__(self, artifact: Dict,
                 embed_fn: Optional[Callable[[str], List[float]]] = None,
                 llm_fn: Optional[Callable[[str], str]] = None):
        self.artifact = artifact
        self.kind = artifact["kind"]
        self.labels = artifact["labels"]
        self._embed_fn = embed_fn
        self._llm_fn = llm_fn
        if self.kind == "nano":
            m = artifact["model"]
            raw = base64.b64decode(m["weights"])
            dim = m["hashDim"]
            self._weights = []
            for c in range(len(self.labels)):
                row = struct.unpack(f"{dim}b", raw[c * dim:(c + 1) * dim])
                scale = m["scales"][c]
                self._weights.append([v * scale for v in row])
            self._biases = m["biases"]
            self._ngrams = tuple(m["ngrams"])
            self._hash_dim = dim

    @classmethod
    def load(cls, artifact: Dict, embed_fn=None, llm_fn=None) -> "RouterRuntime":
        validate_artifact(artifact)
        return cls(artifact, embed_fn=embed_fn, llm_fn=llm_fn)

    def route(self, text: str) -> Dict:
        if not isinstance(text, str) or not text.strip():
            raise ForgeError("route() needs non-empty text.")
        if self.kind == "nano":
            feats = extract_features(text, self._hash_dim, self._ngrams)
            raw = _predict_ids(feats, self._weights, self._biases)
        elif self.kind == "embed":
            if self._embed_fn is None:
                raise ForgeError("This embed router needs an embed_fn to run.")
            vec = self._embed_fn(text)
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            unit = [v / norm for v in vec]
            cents = self.artifact["model"]["centroids"]
            raw = [sum(a * b for a, b in zip(unit, cents[lab])) for lab in self.labels]
        else:  # llm
            if self._llm_fn is None:
                raise ForgeError("This llm router needs an llm_fn to run.")
            prompt = self.artifact["model"]["promptTemplate"].replace("{text}", text)
            answer = self._llm_fn(prompt).strip().lower()
            raw = [1.0 if lab.lower() in answer else 0.0 for lab in self.labels]
            if not any(raw):
                raw = [1.0 / len(self.labels)] * len(self.labels)
        probs = _softmax(raw)
        best = max(range(len(self.labels)), key=lambda i: probs[i])
        return {
            "label": self.labels[best],
            "confidence": round(probs[best], 4),
            "scores": {lab: round(p, 4) for lab, p in zip(self.labels, probs)},
        }


def validate_artifact(artifact) -> None:
    """Structural validation. Raises ForgeError with a user-presentable message."""
    if not isinstance(artifact, dict) or artifact.get("format") != FORGE_FORMAT:
        raise ForgeError("This is not a router-forge artifact.")
    version = artifact.get("version")
    if not isinstance(version, int) or version > FORGE_VERSION:
        raise ForgeError(f"Unsupported artifact version {version!r}.")
    if artifact.get("kind") not in ("nano", "embed", "llm"):
        raise ForgeError("Unknown router kind.")
    labels = artifact.get("labels")
    if not isinstance(labels, list) or len(labels) < 2:
        raise ForgeError("Artifact must carry at least 2 labels.")
    if not isinstance(artifact.get("model"), dict):
        raise ForgeError("Artifact is missing its model payload.")
    if artifact["kind"] == "nano":
        m = artifact["model"]
        for key in ("hashDim", "ngrams", "weights", "scales", "biases"):
            if key not in m:
                raise ForgeError(f"Nano artifact is missing model.{key}.")


# ---------------------------------------------------------------------------
# Fetch-script generator — pull any public model onto any tier.
# The forge emits SCRIPT TEXT for the user to run; nothing is executed here.
# ---------------------------------------------------------------------------

_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]*$")


def build_fetch_script(source: str, ref: str, dest: str = "models") -> Dict:
    """Generate a download script for a public model.

    source: 'ollama' (ollama pull), 'huggingface' (hf download), 'url' (curl).
    ref is validated against a strict character allowlist because it is
    interpolated into shell text.
    """
    if source == "url":
        if not ref.startswith(("https://", "http://")) or any(c in ref for c in " '\"`$;|&<>"):
            raise ForgeError("url source needs a plain http(s) URL with no shell characters.")
    elif not _SAFE_REF.match(ref or ""):
        raise ForgeError("Model reference contains characters that are not allowed.")
    if not _SAFE_REF.match(dest or ""):
        raise ForgeError("Destination contains characters that are not allowed.")

    if source == "ollama":
        script = (
            "#!/usr/bin/env sh\n"
            f"# Pull {ref} into local Ollama and smoke-test it.\n"
            f"ollama pull {ref}\n"
            f"ollama run {ref} 'Say ready.' --verbose\n"
        )
        filename = f"fetch-ollama-{ref.replace('/', '-').replace(':', '-')}.sh"
    elif source == "huggingface":
        script = (
            "#!/usr/bin/env sh\n"
            f"# Download {ref} from Hugging Face into ./{dest}.\n"
            "pip install -q -U huggingface_hub\n"
            f"hf download {ref} --local-dir {dest}/{ref.split('/')[-1]}\n"
        )
        filename = f"fetch-hf-{ref.replace('/', '-')}.sh"
    elif source == "url":
        target = ref.rstrip("/").split("/")[-1] or "model.bin"
        script = (
            "#!/usr/bin/env sh\n"
            f"# Download {ref} into ./{dest}.\n"
            f"mkdir -p {dest}\n"
            f"curl -L --fail -o {dest}/{target} '{ref}'\n"
        )
        filename = f"fetch-url-{target}.sh"
    else:
        raise ForgeError("source must be one of: ollama, huggingface, url.")
    return {"filename": filename, "script": script}


# ---------------------------------------------------------------------------
# Registry — named, versioned router artifacts in SQLite.
# ---------------------------------------------------------------------------

class ForgeStore:
    """SQLite registry of forged routers. Same shape as DatasetStore."""

    def __init__(self, db_path: str = "jacky_forge.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS routers (
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    task TEXT,
                    labels TEXT NOT NULL,
                    artifact TEXT NOT NULL,
                    accuracy REAL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (name, version)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()
        log.info(f"ForgeStore ready at {db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, artifact: Dict) -> int:
        """Store an artifact under its name; returns the new version number."""
        validate_artifact(artifact)
        name = artifact.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ForgeError("Artifact needs a non-empty name to be stored.")
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT MAX(version) AS v FROM routers WHERE name = ?", (name,)
                ).fetchone()
                version = (row["v"] or 0) + 1
                conn.execute(
                    "INSERT INTO routers (name, version, kind, task, labels, artifact, accuracy, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        name,
                        version,
                        artifact["kind"],
                        artifact.get("task", ""),
                        json.dumps(artifact["labels"]),
                        json.dumps(artifact),
                        (artifact.get("trainedOn") or {}).get("accuracy"),
                        datetime.utcnow().isoformat(),
                    ),
                )
                conn.commit()
                return version
            finally:
                conn.close()

    def get(self, name: str, version: Optional[int] = None) -> Optional[Dict]:
        conn = self._connect()
        try:
            if version is None:
                row = conn.execute(
                    "SELECT artifact FROM routers WHERE name = ? ORDER BY version DESC LIMIT 1",
                    (name,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT artifact FROM routers WHERE name = ? AND version = ?",
                    (name, version),
                ).fetchone()
            return json.loads(row["artifact"]) if row else None
        finally:
            conn.close()

    def list(self) -> List[Dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT r.name, r.version, r.kind, r.task, r.labels, r.accuracy, r.created_at,
                       LENGTH(r.artifact) AS size_bytes
                FROM routers r
                JOIN (SELECT name, MAX(version) AS v FROM routers GROUP BY name) latest
                  ON latest.name = r.name AND latest.v = r.version
                ORDER BY r.name
                """
            ).fetchall()
            return [
                {
                    "name": r["name"],
                    "version": r["version"],
                    "kind": r["kind"],
                    "task": r["task"],
                    "labels": json.loads(r["labels"]),
                    "accuracy": r["accuracy"],
                    "created_at": r["created_at"],
                    "size_bytes": r["size_bytes"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def delete(self, name: str) -> int:
        """Delete every version of a named router. Returns rows removed."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM routers WHERE name = ?", (name,))
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()
