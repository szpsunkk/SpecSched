# SpecSched

SpecSched is a research prototype for cache-coupled speculative scheduling in
multi-tenant LLM serving. This public repository contains the code, configs, and
reproduction scripts needed to rerun the experiments. It intentionally excludes
model checkpoints, generated results, private traces, logs, and paper build
artifacts.

## Repository Layout

```text
configs/                    Experiment configurations.
experiments/scheduler/       Scheduler engine, policies, KV policies, workload logic.
experiments/real_matrix/     Config-driven experiment runner and processors.
experiments/motivation/      Edge/cloud motivation benchmark code.
scripts/                     Plotting, audits, and helper scripts.
main/                        Paper LaTeX source only, no compiled PDFs.
data/                        Local trace location, ignored by Git.
results/                     Local raw/processed/figure outputs, ignored by Git.
paper_assets/                Local generated paper figures/tables, ignored by Git.
```

## Setup

Use a Python environment with CUDA-enabled PyTorch for GPU experiments:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The configs use local Hugging Face snapshot paths under `models/` by default.
Place or symlink the required model snapshots there, or edit the `models`
section of the JSON config you run.

## Data

The public repo does not include traces. For a quick smoke test, generate a
synthetic ShareGPT-style trace:

```bash
python scripts/make_synthetic_trace.py --out data/sharegpt_trace.json --rows 256
```

For paper-scale experiments, replace `data/sharegpt_trace.json` with a
ShareGPT/LMSYS-style JSON list where each row has a `prompt` field.

## Quick Smoke Test

The smoke config uses tiny parameters and validates startup, raw artifacts,
processed artifacts, and CSV schemas. Performance numbers from this run are not
meaningful.

```bash
make real-paper-smoke
```

Expected outputs:

```text
results/smoke/raw/
results/smoke/processed/
```

## Paper-Scale Runs

Dry-run the configured paper matrix:

```bash
make real-paper-dry-run CONFIG=configs/paper3090_v3_kcap3.json
```

Run the main configured experiment:

```bash
make real-paper CONFIG=configs/paper3090_v3_kcap3.json
make real-paper-process CONFIG=configs/paper3090_v3_kcap3.json
make real-paper-figures CONFIG=configs/paper3090_v3_kcap3.json
```

Three-seed validation:

```bash
make real-paper-tpds-multiseed CONFIG=configs/tpds_multiseed_main.json
make real-paper-tpds-multiseed-process CONFIG=configs/tpds_multiseed_main.json
```

Open-loop heterogeneous Poisson validation:

```bash
make real-paper-heterog-poisson-multiseed CONFIG=configs/tpds_multiseed_heterog_poisson.json
make real-paper-heterog-poisson-multiseed-process CONFIG=configs/tpds_multiseed_heterog_poisson.json
```

## Output Policy

All generated artifacts stay local:

```text
results/raw/                 Raw CSV/JSON logs.
results/processed/           Processed summaries.
results/figures/             Configured-run figures.
paper_assets/figures/        Manuscript-ready figures.
paper_assets/tables/         Manuscript-ready tables.
```

These paths are ignored by Git so that the repository remains code-only.

## Reproducibility Notes

- Every configured experiment has an explicit seed.
- `run_configured_paper.py` writes `effective_config.json`,
  `run_manifest.json`, and `run_status.{jsonl,csv}` for each run.
- Model paths and trace paths are intentionally configurable and should not be
  hard-coded in scripts.
- Full paper-scale runs require substantial GPU time and local model snapshots.

## Remote

The intended public remote is:

```bash
git remote add origin git@github.com:szpsunkk/SpecSched.git
```
