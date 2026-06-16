# Configured Paper Experiment Framework

This framework implements the experiment matrix described in `ex.md` without
changing the experiment design.

## Layout

- Config: `configs/paper3090.json`
- Raw results: `results/raw/`
- Processed results: `results/processed/`
- Figures: `results/figures/`
- Logs: `results/raw/logs/`

## Commands

Dry-run the configured matrix:

```bash
make real-paper-dry-run
```

Run the complete configured experiment:

```bash
make real-paper
```

Run in the background:

```bash
make real-paper-bg
```

Rebuild processed CSVs from existing raw results:

```bash
make real-paper-process
```

Regenerate figures from processed CSVs:

```bash
make real-paper-figures
```

Audit headline paper claims against processed CSVs:

```bash
make real-paper-claim-audit
```

Run the full TPDS manuscript readiness check:

```bash
make tpds-readiness
```

Build an anonymous submission PDF if the TPDS submission system requests a
blind-review manuscript:

```bash
make tpds-anonymous
```

Run both manuscript checks before uploading:

```bash
make tpds-submission-check
```

Build a compact upload bundle under `dist/tpds_submission/` and
`dist/tpds_submission.zip`:

```bash
make tpds-package
```

The cover-letter draft, upload checklist, and submission metadata are in
`main/cover_letter_tpds.md`, `main/submission_checklist_tpds.md`, and
`main/submission_metadata_tpds.json`.

Run the focused TPDS multi-seed validation:

```bash
make real-paper-tpds-multiseed
```

or in the background:

```bash
make real-paper-tpds-multiseed-bg
```

Run the open-loop Poisson smoke test for the A1b experiment path:

```bash
make real-paper-poisson-smoke
```

Dry-run or launch the focused heterogeneous/open-loop A1 validation:

```bash
make real-paper-heterog-poisson-dry-run
make real-paper-heterog-poisson
make real-paper-heterog-poisson-process
```

For the three-seed version used to test whether the BestFixed confidence
interval closes:

```bash
make real-paper-heterog-poisson-multiseed-dry-run
make real-paper-heterog-poisson-multiseed-bg
make real-paper-heterog-poisson-multiseed-process
```

During a long run, use the partial processor only for internal diagnostics:

```bash
make real-paper-heterog-poisson-multiseed-process-partial
```

It writes available aggregates to `processed_partial/` and
`tables_partial/`. These files are intentionally separated from the official
`processed/` and `tables/` outputs because they may contain only one seed or a
subset of loads. Do not cite partial outputs in `main.tex` or use them for
headline claims. Partial processing also writes
`processed_partial/partial_diagnostics.csv`, which records completed policies,
the current best fixed-k row, and the current SpecSched-vs-BestFixed gap when
both are available. This file is only for monitoring a background run. The
strict `*-process` targets still fail until all configured `(seed, N, policy)`
rows are present.

Audit the remaining TPDS reviewer-risk gaps from `improve_cc.md` against saved
results:

```bash
make real-paper-tpds-gap-audit
```

Use a different config:

```bash
CONFIG=configs/paper3090.json make real-paper
```

## Reproducibility

Every scheduler subprocess receives the fixed `seed` from the config. The raw
run manifest records the config path, config hash, result directories, and seed.
Each `(experiment, N)` cell writes a dedicated scheduler CSV, request CSV, SLO
threshold CSV, NoSpec request CSV, calibration CSV, JSONL status row, and log.

The TPDS validation config `configs/tpds_multiseed_main.json` reuses the main
seed-7 matrix from `results/local_v3_kcap3/processed/` and runs two additional
focused seeds for the main experiment at `N=16,32,64`. The post-processor writes
`results/tpds_multiseed_main/processed/main_multiseed_summary.csv`, paired
goodput lifts in `main_multiseed_paired_lifts.csv`, and a LaTeX table with
mean/std values. It fails if any configured `(seed, N, policy)` row is missing,
so incomplete runs cannot be silently reported.

`make real-paper-claim-audit` checks the headline values cited in the paper
against processed CSVs: multi-seed lifts, the targeted SpecServe comparison,
SLO-sensitivity gains, and quality-check completion rows.
`make real-paper-tpds-gap-audit` is stricter and diagnostic: it reports whether
the saved results actually close the `improve_cc.md` gaps, including paired
confidence intervals versus BestFixed, SpecServe-Proxy sweep coverage,
EvictV2-vs-LRU coverage, and whether the main results include open-loop
Poisson arrivals. It also consumes focused audit outputs such as
`results/verify_latency_fit/processed/verify_latency_fit_metrics.json` and
`results/estimator_audit/processed/posalpha_estimator_audit.json` when those
files exist. It intentionally reports open gaps when the corresponding
experiments have not been run.
`make tpds-readiness` additionally compiles `main/main.tex`, checks the PDF page
count, LaTeX log, IEEE-style abstract constraints (one paragraph, 100--200
words, no math or citations), style constraints, included figures, and the claim
audit.

Focused reviewer-risk audits can be regenerated without touching the running
paper experiments:

```bash
python experiments/scheduler/scripts/verify_latency_fit.py \
  --target models/models--Qwen--Qwen2.5-7B-Instruct \
  --batch-sizes 1,2,4,8 --draft-ks 1,3,5,7 --context-lens 64,128,256,512
make real-paper-focused-audits
```

The latency perturbation audit is an offline ranking-stability check over the
saved latency-fit grid. It is not a measured SpecSched goodput degradation,
because the current online scheduler does not read fitted latency coefficients.
The PosAlpha audit is also a narrow targeted check. It shows whether replacing
the scalar EMA with position-level acceptance estimates changes the saved
single-seed matrix, but it does not replace a future Bayes/non-stationary
estimator study.

## Arrival Models

Configured experiments default to `arrival_model: "closed_loop"`, preserving
the existing paper matrix. To run open-loop experiments, set
`arrival_model: "poisson"` and `arrival_rate_rps` in either `common` or an
individual experiment. The configured runner forwards these fields to
`experiments/scheduler/run_scheduler.py`, and the summary CSV records
`arrival_model` and `arrival_rate_rps` for each row. The smoke config
`configs/smoke_poisson_paper.json` verifies this path without making any
performance claim.

For bursty open-loop experiments, set `arrival_model: "bursty"` and configure
`burst_size` plus `burst_period_s` in either `common` or an individual
experiment. Requests are released in fixed-size bursts separated by the
configured period, and the summary CSV records the burst parameters for later
claim audits. `configs/smoke_bursty_paper.json` verifies this D1.2 plumbing;
it is not a multi-seed bursty-arrival result for the manuscript.

For multi-tier SLO experiments, set `tier_names`, `tier_weights`,
`tier_tpot_mults`, and `tier_ttft_mults` in `common`. The scheduler assigns
requests to tiers, applies the tier multipliers after adaptive NoSpec
calibration, and the processor writes `tier_fairness_summary.csv` with
per-tier goodput, per-tier SLO attainment, and a Jain-index row per
policy/load cell. `configs/smoke_multitier_paper.json` verifies this D1.3
plumbing; it is not a multi-seed fairness result for the manuscript.

For production-style replay, set `arrival_model: "trace"` and
`arrival_trace` to a JSON, JSONL, or CSV file containing an arrival-time field
(`arrival_s`, `arrival_time`, `timestamp`, `time`, or `start_time`) and
optionally an output-length field (`max_new_tokens`, `output_length`,
`completion_tokens`, `response_tokens`, or `generated_tokens`). The runner
uses the arrival trace for timing and output lengths while drawing prompt text
from the configured prompt pool. `configs/smoke_trace_replay_paper.json`
verifies this plumbing with `data/synthetic_production_trace_smoke.jsonl`;
it is a code-path smoke test, not production-trace evidence for the paper.

The focused A1 config `configs/paper3090_heterog_poisson.json` runs only the
main beta-1.5 experiment for `N=16,32,64` under Poisson arrivals and writes to
`results/heterog_poisson/`. The multi-seed wrapper
`configs/tpds_multiseed_heterog_poisson.json` writes to
`results/tpds_multiseed_heterog_poisson/`. These directories are intentionally
separate from the current paper figures. Do not copy their numbers into
`main/main.tex` until the run is complete and the paired-lift audit shows the
claim is supported.

## Supplemental Multi-Seed Matrices

The supplemental validation configs target reviewer-risk gaps without rerunning
the full paper sweep:

- `configs/tpds_multiseed_second_model.json` runs Qwen2.5-1.5B-Instruct target
  with Qwen2.5-0.5B-Instruct draft at `N=16` across seeds 7, 42, and 123. It is
  a model-pair sensitivity check, not a claim that the smaller target benefits
  from speculation.
- `configs/tpds_multiseed_kv_pressure.json` runs the KV-pressure policies at
  `N=64` for 512 and 1024 vLLM blocks across seeds 7, 42, and 123. The
  multi-seed processor groups rows by `configured_kv_blocks`, so do not use the
  generic main-only processor for this matrix.
- `configs/tpds_multiseed_edge_practical.json` repeats the small real
  Jetson/Tailscale edge matrix across seeds. It depends on the edge hosts being
  reachable and should be run only when no local GPU experiment is active.

Use the generic supplemental entry points:

```bash
CONFIG=configs/tpds_multiseed_second_model.json make real-paper-supplement-multiseed-bg
CONFIG=configs/tpds_multiseed_kv_pressure.json make real-paper-supplement-multiseed-bg
CONFIG=configs/tpds_multiseed_edge_practical.json make real-paper-supplement-multiseed-bg

CONFIG=configs/tpds_multiseed_kv_pressure.json make real-paper-supplement-multiseed-status
CONFIG=configs/tpds_multiseed_kv_pressure.json make real-paper-supplement-multiseed-process
```

## 3090Ti OOM Handling

The configured runner isolates each `(experiment, N)` cell in a separate Python
process. This preserves the shared NoSpec SLO calibration for all policies at
the same `N`, while avoiding long-lived CUDA allocator state across the whole
paper sweep. KV-cache replay after preemption is chunked by
`cache_replay_chunk_tokens` in the config to avoid a large activation spike
during re-prefill.

The production config uses `max_batch=4` to reduce the SpecSched N=64 batched
verify peak memory on a 24GB 3090Ti. `max_batch` changes batching efficiency
for batched-verify policies, so it is part of `hyperparam_hash`; rows produced
with a different `max_batch` are treated as incompatible and that full `(tag,
N)` cell is rerun.

## Legacy Results

New runs write raw files under `results/raw/`. The processor also reads the
legacy flat result directory listed in `paths.legacy_results_roots` so partial
CSV files from earlier runs can be included or skipped during resume checks.
When both new and legacy files contain the same `(tag, N, policy)` cell, the new
`results/raw/` row takes precedence.

`SpecSched-NoEvict` disables victim selection, so KV pressure is handled by
waiting or recomputation instead of evicting another request. `SpecSched-LRU`
uses plain LRU eviction and is the KV-sweep baseline.
