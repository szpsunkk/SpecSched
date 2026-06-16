#!/usr/bin/env python3
"""Create a small ShareGPT-style prompt trace for smoke tests.

The generated prompts are synthetic and are intended only to verify that the
experiment pipeline starts, writes CSV/JSON artifacts, and can be processed.
They are not a substitute for the paper trace.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SHORT = [
    "Summarize cache eviction in one sentence.",
    "Define speculative decoding for a systems paper.",
    "List two causes of GPU queueing delay.",
]

MEDIUM = [
    "Explain how a small draft model and a larger verifier can cooperate during "
    "language model serving. Mention why batching can change latency.",
    "Describe a multi-tenant serving workload with mixed prompt lengths, shared "
    "GPU memory, and service-level objectives for token latency.",
    "Compare fixed-depth speculation with an online scheduler that adapts draft "
    "length to request pressure and cache residency.",
]

LONG = [
    "Write a technical paragraph about key-value cache pressure in transformer "
    "serving. Include discussion of prompt prefill, decode steps, cache blocks, "
    "eviction, re-prefill overhead, and how these effects interact with "
    "speculative decoding under concurrent requests. Use precise systems "
    "terminology and avoid marketing language.",
    "Describe an experiment for evaluating speculative decoding in a shared GPU "
    "server. The experiment should include heterogeneous prompt lengths, several "
    "concurrent clients, a no-speculation baseline, fixed draft depths, an "
    "adaptive scheduler, and metrics such as goodput, tail token latency, cache "
    "evictions, and target forward passes.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/sharegpt_trace.json")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pools = [SHORT, MEDIUM, LONG]
    rows = []
    for idx in range(args.rows):
        pool = pools[idx % len(pools)]
        text = rng.choice(pool)
        rows.append({"id": idx, "prompt": text})
    rng.shuffle(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out} with {len(rows)} synthetic prompts")


if __name__ == "__main__":
    main()
