# SpecSched TPDS Artifact

This artifact supports result-level reproduction for the TPDS
submission, *SpecSched: Cache-Coupled Speculative Scheduling for
Multi-Tenant LLM Serving*.

## Scope

The artifact contains saved experiment outputs and the scripts needed
to verify the paper claims and rebuild figures from those outputs. It
does not contain model checkpoints, full trace datasets, submission
PDFs, manuscript source bundles, cover letters, or private
submission-preparation notes.

Full end-to-end reruns require the hardware and software environment
listed in the manuscript: an RTX 3090 Ti class GPU, Jetson devices for
the real edge-cloud diagnostic, Qwen2.5-7B-Instruct and
Qwen2.5-0.5B-Instruct checkpoints, trace files, CUDA/PyTorch, and the
configured vLLM path.

## Quick Check

From the extracted artifact root, run:

```bash
make tpds-results-reproduce
```

This command rebuilds the paper figures and generated tables from
saved CSV/JSON outputs, audits headline claims against processed
results, and runs the artifact completeness check. It does not rerun
GPU experiments.

For a lighter check that skips figure regeneration, run:

```bash
make tpds-artifact-check
python scripts/audit_paper_claims.py
```

These commands verify that saved
CSV/JSON outputs, figure manifests, required fields, quality checks,
and headline manuscript claims are internally consistent.

Expected result: the JSON reports finish with `status: pass`.

## Optional Result-Level Docker Check

The artifact includes `Dockerfile.artifact` for reviewers who want a
clean Python environment for the saved-output reproduction path:

```bash
docker build -f Dockerfile.artifact -t specsched-tpds-artifact .
docker run --rm -v "$PWD":/workspace specsched-tpds-artifact
```

The container installs only the dependencies needed to rebuild figures
and tables from saved CSV/JSON outputs and to run the claim audits. It
does not include CUDA, PyTorch, vLLM, model checkpoints, traces, or
Jetson access, so it is not a full GPU rerun environment.

## Key Files

- `configs/paper3090_v3_kcap3.json`: full fixed-seed paper matrix.
- `configs/tpds_multiseed_main.json`: three-seed main-load validation.
- `configs/tpds_multiseed_eagle2_compare.json`: three-seed evaluated
  baseline map for TETRIS-Proxy, EAGLE-2-Proxy, and SpecServe-Proxy.
- `configs/paper3090_specserve_targeted.json`: three-load
  SpecServe-Proxy comparison.
- `configs/tpds_multiseed_heterog_poisson.json` and
  `configs/paper3090_heterog_poisson.json`: open-loop Poisson
  validation scaffold with per-load request counts and arrival rates.
- `configs/smoke_trace_replay_paper.json` and
  `data/synthetic_production_trace_smoke.jsonl`: tiny trace-replay
  smoke fixture for JSONL arrivals with output-length metadata. This
  checks trace-replay plumbing only; it is not an Azure, BurstGPT, or
  WildChat performance result.
- `configs/paper3090_evictv2_targeted.json` and
  `configs/paper3090_evict_guarded_targeted.json`: targeted eviction
  sensitivity runs used to audit cache-policy robustness.
- `configs/paper3090_second_model_targeted.json`: targeted
  second-model sanity check.
- `configs/tpds_multiseed_second_model_n32.json` and
  `configs/tpds_multiseed_pythia_model_n32.json`: three-seed $N=32$
  model-pair boundary checks.
- `configs/tpds_multiseed_second_model_kv_pressure.json`,
  `configs/tpds_multiseed_second_model_slo_sensitivity.json`,
  `configs/tpds_multiseed_pythia_model_kv_pressure.json`, and
  `configs/tpds_multiseed_pythia_model_slo_sensitivity.json`:
  three-seed $N=32$ model-pair pressure checks for KV budgets and SLO
  multipliers.
- `configs/tpds_multiseed_edge_practical.json`: three-seed real
  Jetson/Tailscale edge-draft diagnostic.
- `configs/scheduler_overhead_micro.json`: lightweight CPU-side
  scheduler-overhead microbenchmark for $N$ and $k_{\max}$ scaling.
- `results/local_v3_kcap3/`: raw and processed outputs for the main
  scheduler, SLO, KV, and ablation matrices.
- `results/tpds_multiseed_main/`: raw and processed outputs for the
  three-seed validation.
- `results/eagle2_compare/`: raw and processed outputs for the
  TETRIS-Proxy, EAGLE-2-Proxy, and SpecServe-Proxy evaluated-baseline
  map.
- `results/tpds_multiseed_heterog_poisson/`: raw and processed outputs
  for the three-seed open-loop Poisson validation.
- `results/specserve_targeted/`: raw and processed outputs for the
  three-load SpecServe-Proxy sweep.
- `results/evictv2_targeted/` and `results/evict_guarded_targeted/`:
  targeted eviction-policy audit outputs.
- `results/second_model_targeted/`: targeted second-model sanity-check
  outputs.
- `results/tpds_multiseed_second_model_n32/` and
  `results/tpds_multiseed_pythia_model_n32/`: processed and raw outputs
  for the $N=32$ model-pair boundary checks.
- `results/tpds_multiseed_second_model_kv_pressure/`,
  `results/tpds_multiseed_second_model_slo_sensitivity/`,
  `results/tpds_multiseed_pythia_model_kv_pressure/`, and
  `results/tpds_multiseed_pythia_model_slo_sensitivity/`: processed
  and raw outputs for the auxiliary model-pair pressure checks.
- `results/tpds_multiseed_edge_practical/`: processed and raw outputs
  for the real Jetson/Tailscale edge-draft diagnostic.
- `results/scheduler_overhead_micro/`: raw and processed outputs for
  the scheduler-overhead microbenchmark.
- `results/verify_latency_fit/`: verification-latency fit and
  perturbation-audit outputs.
- `results/estimator_audit/`: acceptance-estimator targeted audit
  outputs.
- `results/smoke_poisson_openloop/`: tiny flow test proving that the
  configured runner can replay open-loop Poisson workloads with more
  requests than active clients and per-load arrival rates.
- Trace-replay smoke results are generated only when
  `make real-paper-trace-smoke` is run locally. The packaged synthetic
  trace fixture documents the supported schema without creating a
  production-trace evidence claim.
- `paper_assets/figures/`: regenerated PDF/PNG figures.
- `paper_assets/tables/plot_manifest.json`: figure/table generation
  manifest used by the artifact checks.
- `main/figures/`: figure PDFs included by the manuscript.
- `scripts/audit_paper_claims.py`: verifies the main numeric claims
  against processed CSV files.
- `scripts/generate_scheduler_overhead_table.py`: rebuilds the
  scheduler-overhead table from saved microbenchmark CSVs.
- `scripts/reproduce_tpds_results.sh`: one-command result-level
  reproduction from saved outputs.
- `scripts/verify_tpds_artifact.py`: checks artifact completeness.
- `main/artifact_reproducibility_tpds.md`: longer reproducibility note.

## Rebuilding Figures From Saved Outputs

The plotting script reads processed CSV files and writes figures under
`paper_assets/figures/` and tables under `paper_assets/tables/`:

```bash
python scripts/plot_results.py
python scripts/audit_paper_claims.py
```

The artifact package check used before submission extracts this zip,
runs `make tpds-results-reproduce`, and verifies that the artifact
manifest hashes match the packaged files. The extracted-artifact
reproduction rebuilds plots and generated tables from saved outputs,
then reruns claim and completeness checks.

## Optional Trace-Replay Smoke

The scheduler includes a result-neutral trace-arrival path for JSON,
JSONL, or CSV arrival traces. The packaged synthetic fixture can be
dry-run without launching model inference:

```bash
python experiments/real_matrix/run_configured_paper.py \
  --config configs/smoke_trace_replay_paper.json --dry-run
```

The command should show `--arrival-model trace` and the packaged
`data/synthetic_production_trace_smoke.jsonl` file. This verifies
configuration plumbing only and does not create a production-trace
performance claim.

## Optional GPU Rerun: Verification-Latency Fit

The manuscript notes that profile-error audits are needed when moving
to unseen batch, draft-length, or context-length regimes. To collect
that audit data on the local GPU, run:

```bash
python experiments/scheduler/scripts/verify_latency_fit.py \
  --target models/models--Qwen--Qwen2.5-7B-Instruct \
  --batch-sizes 1,2,4,8 \
  --draft-ks 1,3,5,7 \
  --context-lens 64,128,256,512
```

The script writes raw timing samples to
`results/verify_latency_fit/raw/verify_latency_measurements.csv` and
MAE/RMSE/MAPE/R2 fit outputs to
`results/verify_latency_fit/processed/`.
