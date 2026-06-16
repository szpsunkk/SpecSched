# Configured Paper Experiments

This directory is the active paper experiment framework. All experiment
parameters live in `configs/`; this folder only contains the runner, processor,
plotter, and smoke test.

## Commands

```bash
make real-paper-smoke
make real-paper-poisson-smoke
make real-paper-dry-run
make real-paper
make real-paper-bg
make real-paper-process
make real-paper-figures
make real-paper-claim-audit
make real-paper-tpds-gap-audit
make tpds-readiness
make real-paper-tpds-multiseed-dry-run
make real-paper-tpds-multiseed-bg
make real-paper-tpds-multiseed-process-partial
make real-paper-tpds-multiseed-status
make real-paper-heterog-poisson-dry-run
make real-paper-heterog-poisson-multiseed-bg
make real-paper-heterog-poisson-multiseed-process-partial
make real-paper-heterog-poisson-multiseed-status
```

## Data Contract

- Raw CSV/JSON: `results/raw/`
- Processed CSV/JSON: `results/processed/`
- Figures: `results/figures/`

The full matrix is `configs/paper3090.json`. The smoke matrix is
`configs/smoke_paper.json`. The focused multi-seed validation is
`configs/tpds_multiseed_main.json`. The open-loop Poisson smoke matrix is
`configs/smoke_poisson_paper.json`. The focused heterogeneous/open-loop A1
validation configs are `configs/paper3090_heterog_poisson.json` and
`configs/tpds_multiseed_heterog_poisson.json`. The non-stationary
BestFixed-transfer GPU protocol is `configs/paper3090_transfer_gpu.json`;
`make real-paper-transfer-smoke` checks the trace/runner/processor path, while
`make real-paper-transfer-gpu` starts the seed-7 measured replay. The follow-up
three-seed transfer replay is configured by
`configs/tpds_multiseed_transfer_gpu.json` and should be run only after the
single-seed replay completes, so the jobs do not compete for the same GPU.

Use the `*-process-partial` targets only for progress diagnostics during a
long-running multi-seed experiment. They write to `processed_partial/` and
`tables_partial/`, not the paper's official `processed/` and `tables/`
directories. Paper claims and TPDS gap audits should use the strict process
targets after every configured seed, load, and policy has completed.
