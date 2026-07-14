# Router Forge

Generate router artifacts at any size and attach them to anything. A router is
a tiny decision function: `route(text) -> {label, confidence, scores}`. The
labels are yours - models, memory branches, tools, bots, squads, tiers, game
moves. The forge does not care what they mean.

## The three sizes

| Kind | Size | How it decides | When to use |
|---|---|---|---|
| `nano` | 50-300 KB (under 1 MB) | hashed char n-grams -> logistic regression, int8 | everywhere; runs in pure Python or ~100 lines of TS on the phone |
| `embed` | few KB + host's embedder | cosine vs per-label centroids (Ollama embeddings) | when nano accuracy is not enough and an embedder is nearby |
| `llm` | 1 KB prompt spec | any LLM classifies (local Ollama or cloud waterfall) | instant to forge, heavyweight to run, best for rare/complex routing |

All three share one artifact format (`router-forge` v1) and one runtime:
`RouterRuntime.load(artifact).route(text)`.

## Architecture

```
iPhone (Off Grid app)          PC data centre (jacky)              Cloud
  export conversations   -->   /api/datasets/ingest?branch=...     reserve
  (Backup -> Export data)      SQLite Dataset Room + FTS search
                               |
                               v
                               /api/forge/train  <- your datasets
                               ForgeStore (versioned registry)
                               |
       nano artifact  <--------+  /api/forge/<name>/export
       (runs offline           |
        on the phone)          +-> jacky's own routing (tier dispatch,
                                   squads, pods, memory branches)
```

The loop: conversations flow in, routers flow out. Every dataset you archive
makes the next router smarter.

## Quickstart (tonight)

Start the server:

```sh
python serve.py        # production, or: python jacky_api.py (dev)
```

1. Archive your chats into a branch of the Dataset Room:

```sh
curl -X POST "http://localhost:5000/api/datasets/ingest?branch=main" \
  -H "Content-Type: application/json" \
  -d @offgrid-backup-2026-07-14.json
```

2. Forge a nano router from those datasets (labels = your project names):

```sh
curl -X POST http://localhost:5000/api/forge/train \
  -H "Content-Type: application/json" \
  -d '{"name": "project-dispatch", "kind": "nano",
       "task": "route a question to the right project",
       "fromDatasets": {"labelBy": "project_id"}}'
```

Or from raw examples, attached to anything you can label:

```sh
curl -X POST http://localhost:5000/api/forge/train \
  -H "Content-Type: application/json" \
  -d '{"name": "tier-dispatch", "kind": "nano",
       "task": "which tier answers this",
       "examples": [
         {"text": "what time is it", "label": "local-tiny"},
         {"text": "summarize this repo and plan a refactor", "label": "cloud"},
         {"text": "write a short poem", "label": "pc-ollama"}
       ]}'
```

3. Test-drive it live:

```sh
curl -X POST http://localhost:5000/api/forge/project-dispatch/route \
  -H "Content-Type: application/json" \
  -d '{"text": "my docker container will not start"}'
# -> {"label": "tech-support", "confidence": 0.93, "scores": {...}}
```

4. Export the artifact and attach it anywhere:

```sh
curl http://localhost:5000/api/forge/project-dispatch/export > router.json
```

The artifact is one JSON file, usually well under 1 MB. Any process that can
hash bytes and multiply numbers can run it.

5. Pull any public model onto any tier (script generator - you run the script,
the server never executes anything):

```sh
curl -X POST http://localhost:5000/api/forge/fetch-script \
  -H "Content-Type: application/json" \
  -d '{"source": "ollama", "ref": "qwen2.5:0.5b"}'

curl -X POST http://localhost:5000/api/forge/fetch-script \
  -H "Content-Type: application/json" \
  -d '{"source": "huggingface", "ref": "HuggingFaceTB/SmolLM2-135M-Instruct"}'
```

## API reference

| Route | Method | Body / params | Does |
|---|---|---|---|
| `/api/forge/train` | POST | `{name, kind, task, examples\|fromDatasets, labels}` | forge + store a router (rate limited) |
| `/api/forge/routers` | GET | - | registry: latest version of every router |
| `/api/forge/<name>/route` | POST | `{text}`, `?version=` | live test-drive |
| `/api/forge/<name>/export` | GET | `?version=` | download the artifact JSON |
| `/api/forge/<name>` | DELETE | - | remove all versions |
| `/api/forge/fetch-script` | POST | `{source, ref, dest}` | generate a model download script |
| `/api/datasets/ingest` | POST | backup JSON, `?branch=` | archive conversations into a branch |
| `/api/datasets/search` | GET | `?q=` | full-text search the Dataset Room |
| `/api/datasets/stats` | GET | - | totals incl. per-branch counts |

`fromDatasets` options: `labelBy` (`project_id`, `model_id`, `branch`),
`role` (default `user`), `branch` (filter), `minPerLabel` (default 2).

## Environment

- `JACKY_DATASETS_DB` - Dataset Room path (default `jacky_datasets.db`)
- `JACKY_FORGE_DB` - router registry path (default `jacky_forge.db`)
- `SAS_ACCESS_TOKEN` - when set, all `/api/*` routes require login (unchanged)

Windows + WSL note: run the server inside WSL next to Ollama
(`http://localhost:11434`); embed routers use it automatically.

## Adding a new router to an app function

1. Forge and iterate via the API until accuracy satisfies you.
2. Export the artifact, version-control it next to the consumer.
3. Load it behind the app's existing decision seam (one service dispatches;
   callers never branch on router kind).
4. Gate on confidence: below your threshold, fall back to the rule-based
   behavior that was there before. Routers augment, they do not break.

## Roadmap

- TypeScript nano runtime in the Off Grid app (same artifact, contract-tested
  against a golden fixture so Python and TS can never drift)
- Confidence-gated tier dispatch inside jacky's ask pipeline
- Memory-branch routing over ECPS seeds
- LLM-router benchmarking across the cloud waterfall
