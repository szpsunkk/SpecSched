# Scheduler Phase-1 Report

This document is the landing page for the rewritten scheduler benchmark.

## Scope

- Real local execution on RTX 3090 Ti (no simulator, no profile-CSV
  back-fitting).
- Target verifier: `Qwen2.5-7B-Instruct` (full run) or `Qwen2.5-1.5B-Instruct`
  (smoke).
- Draft model: `Qwen2.5-0.5B-Instruct`.
- Workload: ShareGPT prompts, balanced over short/medium/long buckets.
- Policies: `NoSpec`, `FixedK-{1,3,5,7}`, `IndepOpt`, `SpecSched`, plus
  `SpecSched-NoBatch / NoEvict / NoSLO` ablations.

## What changed vs the previous report

| Issue in previous code | Fix in Phase 1 |
|---|---|
| Verification did `use_cache=False` and re-prefilled the prompt every step | `core/speculative.py` keeps and crops target `past_key_values` |
| Multi-client was serial FCFS | `core/engine.py` runs admission + batched scheduling |
| Single prompt length; IndepOpt degenerated to one bucket | `core/workload.py` length-bucketed sampling |
| No SLO | `core/slo.py` + goodput / attainment columns in CSV |
| No KV budget pressure | `core/kv_cache.py` block budget + eviction registry |
| No SpecSched, no ablations | `policies/registry.py` implements all four |

## Artifacts

- `results/scheduler_summary.csv` — per (policy, N) summary
- `results/scheduler_requests.csv` — per-request raw rows
- `results/scheduler_calibration.csv` — calibration table
- `figures/scheduler_*.{pdf,png}`

## Interpretation rules

- **Main metrics**: `goodput_tok_s` and `slo_attainment`. Use these in the main
  table; raw `throughput_tok_s` is a secondary metric.
- **Mechanism metrics**: `target_forwards_per_token` and
  `tokens_per_target_forward` for verifier pressure; `acceptance_rate` for
  draft quality; `kv_evictions` and `reprefills` for cache thrashing.
- **Tail metrics**: `p99_tpot_ms` and `p99_ttft_ms` are the per-token and
  first-token tail latencies that justify the SLO-aware framing.

## Outstanding / Phase-2 work

- Replace the per-request loop in `_batched_target_verify` with a single
  cross-request padded forward. The current path is correct but doesn't yet
  amortize verify across the batch on the GPU.
- Add TETRIS as a stronger speculative baseline.
- Add vLLM as the `NoSpec` reference for an apples-to-apples PagedAttention
  comparison.
- Wire `core/workload.make_poisson_workload` and `make_bursty_workload` into
  the driver for an open-loop arrivals figure.
- Implement an Offline Oracle (exhaustive search) for small N as the figure
  upper bound.

## Latest smoke

A short smoke (1.5B target, N=2, max_new=32) confirms:

- Speculative pipeline is end-to-end correct (matches pure greedy).
- SpecSched and ablations exercise speculation (acceptance > 0,
  target_forwards_per_token < 1).
- KV manager / SLO accounting / metrics emit as expected.

The full 7B-target run is required before drawing paper-level conclusions.
