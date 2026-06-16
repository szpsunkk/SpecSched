"""Run a focused multi-seed validation from a configured paper matrix."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

try:
    from .config_utils import expand_seed_placeholders, load_config as load_config_file, validate_schema
except ImportError:
    from config_utils import expand_seed_placeholders, load_config as load_config_file, validate_schema


ROOT = Path(__file__).resolve().parents[2]


def load_multiseed_config(path: Path) -> dict:
    os.environ.setdefault("PROJECT_ROOT", str(ROOT))
    cfg = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if cfg.get("schema_version") != 1:
        raise ValueError(f"unsupported config schema_version={cfg.get('schema_version')!r}")
    return cfg


def selected_experiments(base_cfg: dict, multi_cfg: dict) -> List[dict]:
    include = set(str(x) for x in multi_cfg.get("include_experiments", []))
    ns_filter = {int(x) for x in multi_cfg.get("Ns", [])}
    policy_filter = [str(x) for x in multi_cfg.get("policies", [])]
    experiments: List[dict] = []

    for exp in base_cfg["experiments"]:
        keys = {str(exp.get("name", "")), str(exp.get("tag", "")), str(exp.get("kind", ""))}
        if include and not (include & keys):
            continue
        out_exp = copy.deepcopy(exp)
        if ns_filter:
            out_exp["Ns"] = [int(n) for n in out_exp["Ns"] if int(n) in ns_filter]
        if policy_filter:
            allowed = set(out_exp["policies"])
            missing = [policy for policy in policy_filter if policy not in allowed]
            if missing:
                raise ValueError(f"{out_exp['name']} does not define policies: {missing}")
            out_exp["policies"] = policy_filter
        if out_exp.get("Ns"):
            experiments.append(out_exp)

    if not experiments:
        raise ValueError("multi-seed selection produced no experiments")
    return experiments


def write_seed_config(base_cfg: dict, multi_cfg: dict, seed: int) -> Path:
    cfg = copy.deepcopy(base_cfg)
    cfg["experiment"]["seed"] = int(seed)
    cfg["experiment"]["name"] = f"{base_cfg['experiment']['name']}_tpds_seed{seed}"
    cfg["paths"]["results_root"] = str(Path(multi_cfg["results_root"]) / f"seed_{seed}")
    cfg["paths"]["legacy_results_roots"] = []
    cfg["execution"]["skip_existing"] = bool(multi_cfg.get("skip_existing", True))
    cfg["experiments"] = selected_experiments(base_cfg, multi_cfg)
    cfg = expand_seed_placeholders(cfg, seed)

    namespace = str(multi_cfg.get("generated_config_namespace", "")).strip()
    out_dir = Path(multi_cfg["generated_config_dir"])
    if namespace:
        out_dir = out_dir / namespace
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg['experiment']['name']}.json"
    out_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return out_path


def run(cmd: Iterable[str]) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"[multiseed] {printable}", flush=True)
    subprocess.run(list(cmd), cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tpds_multiseed_main.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    args = parser.parse_args()

    multi_cfg = load_multiseed_config(Path(args.config).resolve())
    base_loaded = load_config_file(Path(multi_cfg["base_config"]).resolve(), ROOT)
    base_cfg = json.loads(os.path.expandvars(base_loaded.raw_text))
    validate_schema(base_cfg)
    seeds = [int(seed) for seed in multi_cfg["seeds"]]

    generated = [write_seed_config(base_cfg, multi_cfg, seed) for seed in seeds]
    for config_path in generated:
        if not args.process_only:
            cmd = [sys.executable, "experiments/real_matrix/run_configured_paper.py", "--config", str(config_path)]
            if args.dry_run:
                cmd.append("--dry-run")
            run(cmd)
        if not args.dry_run:
            run([sys.executable, "experiments/real_matrix/process_configured_results.py", "--config", str(config_path)])

    if not args.dry_run:
        run([sys.executable, "experiments/real_matrix/process_multiseed_validation.py", "--config", str(Path(args.config).resolve())])


if __name__ == "__main__":
    main()
