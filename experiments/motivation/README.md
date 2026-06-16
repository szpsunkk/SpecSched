# Motivation Experiment: Real Edge-Cloud Serving Benchmark

This folder contains the real serving benchmark used as the paper's motivation
experiment. It is intentionally separated from the simulator/profiling code so
that later paper experiments can be added without mixing result sources.

## What This Experiment Shows

The experiment compares:

- `single`: cloud-only target model serving.
- `multi_edge_k3`: Jetson edge draft model plus cloud target verification,
  with fixed speculative length `k=3`.

The main finding is a trade-off:

- Multi-model collaboration reduces cloud target invocations from `1.000` to
  `0.422` calls per generated token.
- Each target invocation yields `2.37` tokens on average instead of `1.00`.
- In the current Jetson + SSH + unoptimized verification implementation, the
  edge drafting and communication overhead dominates, so end-to-end throughput
  is lower than single-model serving.

This is a motivation result: it demonstrates both the potential benefit of
multi-model collaboration and the need for later scheduling/system
optimization.

## Directory Layout

```text
experiments/motivation/
├── README.md
├── report.md
├── serve_benchmark.py       # cloud-side real serving benchmark
├── edge_draft_worker.py     # worker uploaded and executed on Jetson
├── plot_serving.py          # IEEE-style split figures
├── scripts/
│   ├── run_smoke.sh
│   ├── run_edge_benchmark.sh
│   └── plot.sh
├── results/
│   ├── serving_benchmark.csv
│   └── serving_requests.csv
└── figures/
    └── figure2*_serving_*.{png,pdf}
```

## Hardware Assumptions

Cloud:

- GPU: NVIDIA GeForce RTX 3090 Ti
- Target model: Qwen2.5-7B-Instruct

Edge:

- Device: Jetson
- IP: `192.168.3.108`
- User: `jetson`
- Draft model: Qwen2.5-0.5B in the Jetson Hugging Face cache

## Reproduce

From the repository root:

```bash
export JETSON_PASSWORD='<your-jetson-password>'
bash experiments/motivation/scripts/run_smoke.sh
bash experiments/motivation/scripts/run_edge_benchmark.sh
bash experiments/motivation/scripts/plot.sh
```

The full benchmark writes:

- `experiments/motivation/results/serving_benchmark.csv`
- `experiments/motivation/results/serving_requests.csv`

The plot script writes split figures to:

- `experiments/motivation/figures/`

## Important Methodological Boundary

This is a real serving benchmark, not a simulator. It runs actual model forward
passes on the cloud GPU and actual draft generation on Jetson. However, it is
not an optimized production serving stack:

- no vLLM or TensorRT-LLM;
- no continuous batching;
- no optimized RPC stack;
- no full target KV-cache reuse in speculative verification.

Use the result to motivate the need for system-level scheduling and
implementation optimization. Do not claim that this current implementation
improves end-to-end throughput over single-model serving.
