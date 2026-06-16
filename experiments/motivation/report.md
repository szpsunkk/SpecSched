# 真实 Serving Benchmark 实验报告

实验日期：2026-05-21  
实验目标：比较单模型推理与云边多模型协同推理在多客户端 serving 场景下的性能差异，并分析多模型协同的收益与代价。

## 1. 实验真实性说明

本实验为真实 serving benchmark，不使用 simulator 结果，不使用 profile CSV 推算，不对数据做平滑或趋势修正。

数据来源：

- 汇总结果：`experiments/motivation/results/serving_benchmark.csv`
- 请求级原始结果：`experiments/motivation/results/serving_requests.csv`
- 绘图脚本：`experiments/motivation/plot_serving.py`
- benchmark 脚本：`experiments/motivation/serve_benchmark.py`
- Jetson 边缘 worker：`experiments/motivation/edge_draft_worker.py`

生成的图均直接读取 `serving_benchmark.csv`，没有混入 simulator、adaptive policy 或手工构造数据。

## 2. 实验设置

### 2.1 硬件与模型

Cloud 端：

- 设备：NVIDIA GeForce RTX 3090 Ti
- Target model：Qwen2.5-7B-Instruct
- 模型路径：`models/models--Qwen--Qwen2.5-7B-Instruct`

Edge 端：

- 设备：Jetson，IP `192.168.3.108`
- Draft model：Qwen2.5-0.5B
- 运行方式：云端通过 SSH 启动 Jetson 常驻 draft worker，Jetson 实际执行 draft token generation

### 2.2 Workload

- Trace：`data/sharegpt_trace.json`
- Prompt 长度上限：128 tokens
- 每个请求生成长度：8 tokens
- 每个并发点请求数：32
- 并发客户端数：`N = 4, 8, 16, 32`

### 2.3 对比方法

`single`：

单模型 serving。云端 Qwen2.5-7B-Instruct autoregressive decoding，每生成一个 token 调用一次 target model。

`multi_edge_k3`：

云边多模型协同 serving。Jetson 上的 Qwen2.5-0.5B 生成 `k=3` 个 draft tokens，云端 Qwen2.5-7B-Instruct 对 draft tokens 做 verification。该方法不是 adaptive policy，固定 `k=3`。

## 3. 实验指标

主要端到端 serving 指标：

- Throughput：单位时间完成的生成 token 数，单位 tok/s。
- Mean latency：请求从进入系统到完成生成的平均延迟。
- P99 TPOT：99 分位 token latency，反映尾延迟。
- P95 queueing delay：客户端请求在 server 侧排队等待的延迟。

协同推理机制指标：

- Acceptance rate：draft token 被 target 接受的比例。
- Target invocations/token：每生成一个 token 需要调用 target model 的次数，越低越好。
- Tokens/target invocation：每次 target verification 平均产出的 token 数，越高越好。

## 4. 实验结果

### 4.1 端到端 serving 性能

| N | Method | Throughput (tok/s) | Mean latency (s) | P99 TPOT (ms/token) | Acceptance | Target calls/token | Tokens/target call |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | single | 46.41 | 0.66 | 89.2 | 0.000 | 1.000 | 1.00 |
| 4 | multi_edge_k3 | 8.64 | 3.52 | 574.0 | 0.552 | 0.422 | 2.37 |
| 8 | single | 46.67 | 1.22 | 171.9 | 0.000 | 1.000 | 1.00 |
| 8 | multi_edge_k3 | 8.59 | 6.54 | 1088.7 | 0.552 | 0.422 | 2.37 |
| 16 | single | 46.58 | 2.10 | 343.6 | 0.000 | 1.000 | 1.00 |
| 16 | multi_edge_k3 | 8.65 | 10.95 | 2098.0 | 0.552 | 0.422 | 2.37 |
| 32 | single | 46.57 | 2.84 | 680.5 | 0.000 | 1.000 | 1.00 |
| 32 | multi_edge_k3 | 8.54 | 14.82 | 3697.5 | 0.552 | 0.422 | 2.37 |

### 4.2 相对性能

| N | Multi throughput / Single | Multi P99 TPOT / Single | Target call reduction |
|---:|---:|---:|---:|
| 4 | 0.186x | 6.44x | 57.8% |
| 8 | 0.184x | 6.33x | 57.8% |
| 16 | 0.186x | 6.11x | 57.8% |
| 32 | 0.183x | 5.43x | 57.8% |

## 5. 图说明

本次生成的是拆分图，不包含合并大图。

- `figure2a_serving_throughput.{png,pdf}`：真实 serving throughput。该图显示当前实现下 multi-edge 协同方法端到端吞吐低于 single model。
- `figure2b_serving_speedup.{png,pdf}`：multi-edge throughput 相对 single model 的比例。该比例约为 `0.18x`。
- `figure2c_serving_latency.{png,pdf}`：请求级平均延迟。客户端数增加时，multi-edge 方法延迟显著上升。
- `figure2d_serving_p99_tpot.{png,pdf}`：P99 TPOT。该图说明多客户端下尾延迟被排队和云边交互进一步放大。
- `figure2e_serving_queue.{png,pdf}`：P95 queueing delay。该图体现并发客户端增加后 server 侧排队压力增大。
- `figure2f_serving_acceptance.{png,pdf}`：draft acceptance rate。multi-edge 方法 acceptance rate 为 `0.552`。
- `figure2g_serving_target_calls.{png,pdf}`：每生成一个 token 所需 target 调用次数。multi-edge 方法从 `1.000` 降到 `0.422`，说明协同推理确实减少了云端 target 调用。
- `figure2h_serving_verify_yield.{png,pdf}`：每次 target 调用平均产出的 token 数。multi-edge 方法为 `2.37 tokens/call`，single model 为 `1.00 tokens/call`。

## 6. 结果分析

### 6.1 多模型协同的端到端代价

在当前真实 serving 实现中，`multi_edge_k3` 的端到端 throughput 明显低于 `single`。例如在 `N=4` 时，single model throughput 为 `46.41 tok/s`，而 multi-edge 为 `8.64 tok/s`；在 `N=32` 时，single model throughput 为 `46.57 tok/s`，multi-edge 为 `8.54 tok/s`。

该结果说明，当前 Jetson draft、SSH 云边通信、以及未优化的 verification 路径带来的开销超过了 speculative decoding 减少 target 调用的收益。因此，这组真实实验不能支持“当前 multi-edge serving 端到端性能优于 single model”的结论。

### 6.2 多客户端对协同推理性能的影响

随着客户端数从 `N=4` 增加到 `N=32`，single model 的 P99 TPOT 从 `89.2 ms/token` 增加到 `680.5 ms/token`，multi-edge 的 P99 TPOT 从 `574.0 ms/token` 增加到 `3697.5 ms/token`。

这说明多客户端并发会显著放大 serving 系统中的排队延迟和尾延迟。对于 multi-edge 协同推理，边缘 draft、云边通信和云端 verification 串联在一起，使得每个请求的服务时间更长，因此在并发增加时更容易形成排队积压。

### 6.3 多模型协同的真实收益

尽管当前 multi-edge 方法端到端性能较差，但它仍然体现了 speculative collaborative inference 的机制收益：

- Draft acceptance rate 为 `0.552`，说明 Jetson draft 生成的 token 有超过一半被云端 target 接受。
- Target calls/token 从 single model 的 `1.000` 降低到 `0.422`。
- Tokens/target call 从 `1.00` 提升到 `2.37`。
- 这等价于云端 target 调用次数减少约 `57.8%`。

因此，该实验可以支持如下结论：多模型协同推理能够减少云端大模型 target invocation，并提高单次 verification 的 token 产出效率；但在真实云边 serving 中，该收益能否转化为端到端吞吐提升，取决于边缘 draft 速度、云边通信延迟、verification 实现和调度机制。

## 7. 论文中建议使用的表述

可以写入论文的客观表述：

> We conduct a real serving benchmark with a cloud-side Qwen2.5-7B-Instruct target model and an edge-side Qwen2.5-0.5B draft model deployed on Jetson. The multi-model collaborative setting reduces target invocations from 1.00 to 0.422 per generated token and improves the verification yield from 1.00 to 2.37 tokens per target invocation. This confirms that speculative collaboration can reduce cloud target computation. However, under the current edge-cloud serving implementation, the additional edge drafting and communication overhead dominates the saved target computation, leading to lower end-to-end throughput and higher tail latency than single-model serving. The result highlights that multi-model collaboration must be jointly optimized with serving-system scheduling and communication overhead, especially under high client concurrency.

中文表述：

> 真实 serving benchmark 表明，云边多模型协同推理能够显著减少云端大模型的 target 调用次数：每生成一个 token 所需的 target 调用次数从单模型的 1.000 降低到 0.422，每次 target verification 平均产出的 token 数从 1.00 提升到 2.37。这说明 speculative collaboration 在计算结构上具有减少云端大模型调用的潜力。然而，在当前 Jetson 边缘 draft、SSH 云边通信和未优化 verification 的真实实现中，边缘 draft 与通信开销超过了节省的 target 计算，因此端到端吞吐低于单模型，并且在高并发客户端下尾延迟显著放大。该结果说明，多模型协同推理的性能收益并不是自动成立的，需要与 serving 调度、通信路径和 KV-cache/verification 机制联合优化。

## 8. 实验边界与注意事项

本实验应该作为真实系统现象报告，而不是理论最优性能上界。

需要在论文中明确的边界：

- 当前 serving loop 是 closed-loop 多客户端 benchmark，云端 target server 按 FCFS 处理请求。
- 当前实现没有使用 vLLM、TensorRT-LLM 或 continuous batching。
- 当前 edge-cloud 通信通过 SSH JSONL worker 实现，不代表最优 RPC 性能。
- 当前 speculative verification 没有做充分的 KV-cache 复用优化。
- 因此，multi-edge 的负向端到端结果主要反映当前真实实现和设备配置下的系统开销，而不是否定多模型协同推理机制本身。

如果论文需要证明多模型协同在端到端 serving 上优于单模型，需要进一步实现并重新实测：

- 更低延迟的 edge-cloud RPC；
- batch verification；
- target KV-cache 复用；
- 更强的 edge draft 设备或更小的 draft 模型；
- 面向多客户端的 joint scheduling/admission control。
