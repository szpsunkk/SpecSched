# Scheduler Engine

This directory contains the scheduler engine used by the configured paper
framework in `experiments/real_matrix/`.

The user-facing paper entry point is not here; run:

```bash
make real-paper-smoke
make real-paper-bg
```

`run_scheduler.py` remains a low-level driver called by
`run_configured_paper.py`. Keep experiment parameters in `configs/` rather than
hard-coding them here.
