# Data

This public repository does not include the paper traces or generated results.

For a quick pipeline check, generate a synthetic prompt trace:

```bash
python scripts/make_synthetic_trace.py --out data/sharegpt_trace.json --rows 256
```

For paper-scale runs, replace `data/sharegpt_trace.json` with a ShareGPT/LMSYS-
style JSON list. Each row should contain a `prompt` string:

```json
[
  {"prompt": "Explain speculative decoding in one paragraph."}
]
```

The configured runners tokenize these prompts locally and write all raw and
processed outputs under `results/`.
