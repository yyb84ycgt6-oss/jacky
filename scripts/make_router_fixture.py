#!/usr/bin/env python3
"""Generate router-fixture-v1.json - the cross-language golden fixture.

The fixture carries two small artifacts (nano + embed) and a set of routing
cases with the exact outputs the Python runtime produces. The TypeScript
runtime on the phone replays every case and must match: labels exactly,
scores within 1e-4. One file guards both languages against drift.

Deterministic by construction: fixed seed, fixed examples, no timestamps,
sorted keys. Regenerating must produce byte-identical JSON.

Usage: python3 scripts/make_router_fixture.py [output_path]
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router_forge import RouterRuntime, build_embed, extract_features, train_nano

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

# Case texts pin the normalization contract: plain ASCII, mixed case with
# messy whitespace, an accented string, and an emoji string.
CASE_TEXTS = [
    "the container crashes on boot with a kernel error",
    "how long do I let the dough rise overnight",
    "  My   GPU\tDriver\n keeps   CRASHING  ",
    "sauté the onions until caramelized, chéf",
    "fix the wifi 📡 connection please 🤖",
    "kernel",
]

FIXTURE_EMBED_DIMS = 8


def fixture_embed_fn(text):
    """Deterministic fixture embedder both languages can implement: the same
    hashed char n-gram features as nano, taken dense at 8 dims. No model."""
    feats = extract_features(text, hash_dim=FIXTURE_EMBED_DIMS)
    return [feats.get(i, 0.0) for i in range(FIXTURE_EMBED_DIMS)]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "router-fixture-v1.json"
    )
    examples = [(t, "tech") for t in TECH] + [(t, "cook") for t in COOK]

    nano = train_nano(examples, name="fixture-nano", task="tech vs cooking",
                      hash_dim=4096, seed=777)
    embed = build_embed(examples, name="fixture-embed", task="tech vs cooking",
                        embed_fn=fixture_embed_fn,
                        embedding_model=f"fixture-dense-{FIXTURE_EMBED_DIMS}")

    nano_rt = RouterRuntime.load(nano)
    embed_rt = RouterRuntime.load(embed, embed_fn=fixture_embed_fn)

    fixture = {
        "format": "router-forge-fixture",
        "version": 1,
        "artifacts": {"nano": nano, "embed": embed},
        "cases": {
            "nano": [{"text": t, "expected": nano_rt.route(t)} for t in CASE_TEXTS],
            "embed": [{"text": t, "expected": embed_rt.route(t)} for t in CASE_TEXTS],
        },
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, sort_keys=True, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"fixture written to {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    main()
