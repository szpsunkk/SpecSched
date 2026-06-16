#!/usr/bin/env python3
"""Generate the edge-draft paper config from the local paper config."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "configs" / "paper3090.json"
DST = ROOT / "configs" / "paper3090_edge.json"


def main() -> None:
    cfg = json.loads(SRC.read_text(encoding="utf-8"))
    cfg["experiment"]["name"] = "paper3090_edge"
    cfg["paths"]["results_root"] = "${PROJECT_ROOT}/results/edge_full"
    cfg["models"].update(
        {
            "draft_label": "Qwen2.5-0.5B-Instruct-edge",
            "draft_backend": "edge",
            "draft_remote": "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct",
            "edge_hosts": "100.66.37.19,100.107.23.95,100.73.217.47",
            "edge_user": "skk",
            "edge_password": "111",
        }
    )
    DST.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(DST)


if __name__ == "__main__":
    main()
