#!/usr/bin/env python3
"""Generate a clean local-draft config for the SpecSched-v2 full sweep."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "configs" / "paper3090.json"
DST = ROOT / "configs" / "paper3090_v2.json"


def main() -> None:
    cfg = json.loads(SRC.read_text(encoding="utf-8"))
    cfg["experiment"]["name"] = "paper3090_v2"
    cfg["paths"]["results_root"] = "${PROJECT_ROOT}/results/local_v2"
    cfg["paths"]["legacy_results_roots"] = []
    cfg["execution"]["skip_existing"] = False
    DST.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(DST)


if __name__ == "__main__":
    main()
