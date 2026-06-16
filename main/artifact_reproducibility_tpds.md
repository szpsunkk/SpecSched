# TPDS Artifact And Reproducibility Note

This note is for the authors during submission. Do not upload it as a
local-only helper unless the TPDS form explicitly asks for an artifact
or reproducibility statement.

## Artifact Scope

The project archive contains the code and data products needed to
rebuild the paper figures and generated tables from saved experiment
outputs:

- configured experiment files under `configs/`
- raw CSV/JSON logs under `results/raw/` and archived result roots
- processed CSV/JSON summaries under `results/processed/`
- plotting scripts under `scripts/`
- generated paper figures and tables under `paper_assets/`
- manuscript figures under `main/figures/`
- submission checks and claim-audit scripts under `scripts/`

The main reproducibility path is result-level reproduction from saved
CSV/JSON outputs. Full end-to-end reruns require the listed model
checkpoints, trace files, CUDA/PyTorch stack, an RTX 3090 Ti class GPU,
and Jetson devices for the real edge-cloud diagnostic.

## Light-Weight Checks

This result-level reproduction command does not rerun the full GPU
experiment:

```bash
make tpds-results-reproduce
```

It rebuilds paper figures and generated tables from saved CSV/JSON
outputs, audits headline claims against processed results, and checks
artifact completeness.

The result-level target expands to the figure/table rebuild and audit
commands in `scripts/reproduce_tpds_results.sh`. Related local checks
used during preparation are:

```bash
python scripts/plot_results.py --results-root results/local_v3_kcap3
python scripts/plot_edge_results.py --results-root results/tpds_multiseed_edge_practical
python scripts/generate_scheduler_overhead_table.py
python scripts/audit_paper_claims.py
make tpds-artifact-check
make tpds-upload-check
```

The plotting commands rebuild paper assets from saved logs. The claim
audit checks the headline numbers cited in the manuscript against
processed CSV files. The artifact check verifies the saved configs,
raw run manifests, processed summaries, generated figure/table
manifest, manuscript figure PDFs, and upload-package exclusion of
local-only notes. The upload check additionally verifies the packaged
source bundles and extracts the artifact zip to rerun
`make tpds-results-reproduce` in the packaged environment.

The result-level artifact also includes `Dockerfile.artifact` for a
clean saved-output reproduction environment:

```bash
docker build -f Dockerfile.artifact -t specsched-tpds-artifact .
docker run --rm -v "$PWD":/workspace specsched-tpds-artifact
```

This container intentionally contains only Python, Make, pandas, and
matplotlib. It supports figure/table rebuilds and claim audits from
saved CSV/JSON outputs. It is not a CUDA/PyTorch/vLLM image and does
not rerun the GPU or Jetson experiments.

## Optional Artifact Bundle

If TPDS asks for a supplementary artifact, build a result-level bundle
with:

```bash
make tpds-artifact-package
make tpds-artifact-package-check
```

The bundle is written to `dist/tpds_artifact.zip`. It contains
configs, saved raw/processed CSV/JSON outputs, plotting and audit
scripts, generated figures/tables, this reproducibility note, and the
result-level Dockerfile. It does not contain model checkpoints,
submission PDFs, manuscript source bundles, cover letters, or local
reviewer-risk notes.

The bundle also includes `configs/smoke_trace_replay_paper.json` and
`data/synthetic_production_trace_smoke.jsonl` to document the
trace-arrival replay interface. This fixture is a plumbing smoke test
for JSONL arrivals with output-length metadata. It is not a production
trace result and should not be cited as Azure, BurstGPT, or WildChat
evidence.

## Full Rerun Path

The full local scheduler matrix is configured by:

```bash
CONFIG=configs/paper3090_v3_kcap3.json make real-paper-bg
```

The focused multi-seed TPDS validation is configured by:

```bash
make real-paper-tpds-multiseed-bg
```

The real edge-cloud diagnostic uses Jetson devices and the edge configs
under `configs/`. It should be treated as hardware-dependent. The
manuscript reports it as a boundary case for SSH/Tailscale-based remote
drafting rather than as a claim that edge-cloud speculation improves
goodput.

The trace replay plumbing can be checked without model inference:

```bash
python experiments/real_matrix/run_configured_paper.py \
  --config configs/smoke_trace_replay_paper.json --dry-run
```

This command should emit a scheduler command containing
`--arrival-model trace` and the packaged synthetic trace path. A real
production-trace experiment still requires an external trace dataset
and a full GPU rerun.

## Logged Reproducibility Fields

Raw and processed logs record:

- random seed
- model paths and model labels
- config hash and effective config snapshot
- SLO mode and beta
- concurrency level
- policy name
- KV budget and block size
- maximum batch size
- peak GPU allocation
- run status, command line, exit code, and elapsed time

These fields are intended to make each reported cell traceable to a
concrete scheduler invocation.

## Suggested Submission Text

Use this wording only if it matches the exact artifact made available
with the submission:

"The artifact contains the configured experiment framework, saved raw
CSV/JSON logs, processed summaries, plotting scripts, and generated
figures and tables used by the manuscript. The saved outputs support
result-level reproduction of the paper figures, generated tables, and
claim-audit checks. The bundle includes a synthetic trace-replay smoke
fixture to document the arrival-trace interface, but it does not include
production-trace performance results. Full end-to-end reruns require
the listed GPU/Jetson hardware, model checkpoints, trace files, and
CUDA/PyTorch environment."
