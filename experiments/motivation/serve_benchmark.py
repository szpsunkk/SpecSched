"""Real serving benchmark for Figure 2.

This script executes model forward passes instead of reading simulator/profile
tables. It supports:

* single: cloud target model only.
* multi_edge_k: edge Jetson draft model over SSH + cloud target verification.
* multi_local_k: local draft model + cloud target verification, for debugging.

The serving loop is a closed-loop N-client benchmark: up to N client requests
are outstanding; one cloud GPU target server processes requests FCFS. Request
latency includes queueing behind other clients, which is the quantity Figure 2
needs to expose as N increases.
"""
import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import deque

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "../.."))
if EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENT_DIR)
COMMON_DIR = os.path.join(ROOT, "experiments", "common")
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)


def resolve_snapshot(path):
    path = os.path.expanduser(path)
    snapshots = os.path.join(path, "snapshots")
    if os.path.isdir(snapshots):
        names = sorted(
            d for d in os.listdir(snapshots)
            if os.path.isdir(os.path.join(snapshots, d))
        )
        if names:
            return os.path.join(snapshots, names[0])
    return path


def percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def load_target(path):
    model_path = resolve_snapshot(path)
    print(f"[cloud] loading target {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    print(f"[cloud] target params={sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)
    return tok, model


def load_prompts(trace_path, tokenizer, prompt_len, count, seed):
    import random

    with open(trace_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    rng = random.Random(seed)
    rng.shuffle(rows)
    prompts = []
    for row in rows:
        text = row.get("prompt", "")
        if not text:
            continue
        ids = tokenizer(
            text,
            truncation=True,
            max_length=prompt_len,
            return_tensors="pt",
        ).input_ids[0].to(dtype=torch.long)
        if ids.numel() >= 2:
            prompts.append(ids.cpu())
        if len(prompts) >= count:
            break
    if len(prompts) < count:
        raise RuntimeError(f"only loaded {len(prompts)} prompts, need {count}")
    return prompts


@torch.inference_mode()
def warmup_target(model, vocab_size):
    ids = torch.randint(0, vocab_size, (1, 64), device="cuda")
    for _ in range(2):
        _ = model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()


@torch.inference_mode()
def single_generate(target, prompt_ids, max_new_tokens):
    ids = prompt_ids.to("cuda").view(1, -1)
    out = target(input_ids=ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1).view(1, 1)
    generated = []
    for step in range(max_new_tokens):
        generated.append(int(next_tok.item()))
        if step + 1 == max_new_tokens:
            break
        out = target(input_ids=next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1).view(1, 1)
    torch.cuda.synchronize()
    return {
        "tokens": len(generated),
        "accepted_draft": 0,
        "drafted": 0,
        "draft_elapsed_ms": 0.0,
        "target_forwards": max_new_tokens,
        "draft_forwards": 0,
    }


class LocalDraftClient:
    def __init__(self, path):
        model_path = resolve_snapshot(path)
        print(f"[cloud] loading local draft {model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        print(f"[cloud] local draft params={sum(p.numel() for p in self.model.parameters())/1e6:.1f}M", flush=True)

    @torch.inference_mode()
    def draft(self, input_ids, k):
        t0 = time.perf_counter()
        ids = torch.tensor(input_ids, dtype=torch.long, device="cuda").view(1, -1)
        out = self.model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1).view(1, 1)
        tokens = []
        for step in range(k):
            tokens.append(int(next_tok.item()))
            if step + 1 == k:
                break
            out = self.model(input_ids=next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1).view(1, 1)
        torch.cuda.synchronize()
        return tokens, (time.perf_counter() - t0) * 1000.0

    def close(self):
        pass


class EdgeDraftClient:
    def __init__(self, worker, remote_model, host, user, password):
        from ssh_jetson import open_ssh, put_file

        if not password:
            raise RuntimeError("edge password is empty; set JETSON_PASSWORD or pass --edge-password")
        self.request_id = 0
        self.host = host
        self.client = open_ssh(host=host, user=user, password=password, timeout=30)
        self.remote_dir = f"/home/{user}/specsched"
        self.remote_worker = f"{self.remote_dir}/edge_draft_worker.py"
        self.client.exec_command(f"mkdir -p {self.remote_dir}")
        put_file(self.client, worker, self.remote_worker)
        cmd = (
            f"cd {self.remote_dir} && "
            f"USE_TF=0 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 "
            f"python3 {self.remote_worker} --model {remote_model}"
        )
        print(f"[edge] starting worker on {host}", flush=True)
        self.stdin, self.stdout, self.stderr = self.client.exec_command(cmd, get_pty=False, timeout=None)
        self.stdin.channel.settimeout(None)
        self.stdout.channel.settimeout(None)
        self._wait_until_ready()

    def _read_stderr_available(self):
        chunks = []
        while self.stderr.channel.recv_stderr_ready():
            chunks.append(self.stderr.channel.recv_stderr(4096).decode("utf-8", "replace"))
        return "".join(chunks)

    def _wait_until_ready(self):
        deadline = time.time() + 180
        buffer = ""
        while time.time() < deadline:
            if self.stderr.channel.recv_stderr_ready():
                text = self.stderr.channel.recv_stderr(4096).decode("utf-8", "replace")
                buffer += text
                print(text, end="", flush=True)
                if "[edge] ready" in buffer:
                    return
            if self.stdout.channel.exit_status_ready():
                raise RuntimeError("edge worker exited before ready: " + buffer)
            time.sleep(0.2)
        raise TimeoutError("edge worker did not become ready within 180s")

    def draft(self, input_ids, k):
        self.request_id += 1
        req = {"id": self.request_id, "input_ids": input_ids, "k": k}
        t0 = time.perf_counter()
        self.stdin.write(json.dumps(req) + "\n")
        self.stdin.flush()
        line = self.stdout.readline()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not line:
            raise RuntimeError("edge worker closed stdout: " + self._read_stderr_available())
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"edge worker error: {resp}")
        return resp["tokens"], elapsed_ms

    def close(self):
        try:
            self.stdin.write(json.dumps({"cmd": "stop"}) + "\n")
            self.stdin.flush()
        except Exception:
            pass
        try:
            self.client.close()
        except Exception:
            pass


class MultiEdgeDraftClient:
    def __init__(self, worker, remote_model, hosts, user, password):
        if not hosts:
            raise RuntimeError("at least one edge host is required")
        self.clients = [
            EdgeDraftClient(worker, remote_model, host, user, password)
            for host in hosts
        ]
        self.next_client = 0

    def draft(self, input_ids, k):
        client = self.clients[self.next_client]
        self.next_client = (self.next_client + 1) % len(self.clients)
        return client.draft(input_ids, k)

    def close(self):
        for client in self.clients:
            client.close()


@torch.inference_mode()
def verify_and_append(target, current, draft_tokens, remaining):
    draft = torch.tensor(draft_tokens, dtype=torch.long, device="cuda")
    seq = torch.cat([current, draft], dim=0).view(1, -1)
    out = target(input_ids=seq, use_cache=False)
    logits = out.logits[0]
    base = current.numel() - 1

    accepted = 0
    append_tokens = []
    for j, tok in enumerate(draft_tokens):
        pred = int(logits[base + j].argmax(dim=-1).item())
        if pred == tok:
            accepted += 1
            append_tokens.append(tok)
        else:
            append_tokens.append(pred)
            break
    if accepted == len(draft_tokens):
        bonus = int(logits[base + len(draft_tokens)].argmax(dim=-1).item())
        append_tokens.append(bonus)

    append_tokens = append_tokens[:remaining]
    new_tokens = torch.tensor(append_tokens, dtype=torch.long, device="cuda")
    current = torch.cat([current, new_tokens], dim=0)
    torch.cuda.synchronize()
    return current, len(append_tokens), accepted


@torch.inference_mode()
def speculative_generate(target, draft_client, prompt_ids, max_new_tokens, k):
    current = prompt_ids.to("cuda")
    generated = 0
    accepted_total = 0
    drafted_total = 0
    draft_elapsed_ms = 0.0
    target_forwards = 0
    draft_forwards = 0

    while generated < max_new_tokens:
        step_k = min(k, max_new_tokens - generated)
        draft_tokens, elapsed_ms = draft_client.draft(current.detach().cpu().tolist(), step_k)
        draft_elapsed_ms += elapsed_ms
        drafted_total += len(draft_tokens)
        draft_forwards += len(draft_tokens)
        current, added, accepted = verify_and_append(
            target,
            current,
            draft_tokens,
            max_new_tokens - generated,
        )
        generated += added
        accepted_total += accepted
        target_forwards += 1

    return {
        "tokens": generated,
        "accepted_draft": accepted_total,
        "drafted": drafted_total,
        "draft_elapsed_ms": draft_elapsed_ms,
        "target_forwards": target_forwards,
        "draft_forwards": draft_forwards,
    }


def run_closed_loop(mode, n_clients, prompts, target, max_new_tokens, k, draft_client):
    total_requests = len(prompts)
    pending = deque()
    next_idx = 0
    wall_start = time.perf_counter()
    for _ in range(min(n_clients, total_requests)):
        pending.append((next_idx, wall_start))
        next_idx += 1

    rows = []
    while pending:
        req_idx, arrival = pending.popleft()
        prompt_ids = prompts[req_idx]
        service_start = time.perf_counter()
        torch.cuda.synchronize()
        if mode == "single":
            result = single_generate(target, prompt_ids, max_new_tokens)
        else:
            result = speculative_generate(target, draft_client, prompt_ids, max_new_tokens, k)
        torch.cuda.synchronize()
        finish = time.perf_counter()
        service_s = finish - service_start
        latency_s = finish - arrival
        queue_s = max(0.0, service_start - arrival)
        rows.append({
            "mode": mode,
            "N": n_clients,
            "request_id": req_idx,
            "prompt_len": int(prompt_ids.numel()),
            "generated_tokens": result["tokens"],
            "latency_s": latency_s,
            "service_s": service_s,
            "queue_s": queue_s,
            "tpot_ms": latency_s * 1000.0 / max(1, result["tokens"]),
            "service_tpot_ms": service_s * 1000.0 / max(1, result["tokens"]),
            "accepted_draft": result["accepted_draft"],
            "drafted": result["drafted"],
            "acceptance_rate": (
                result["accepted_draft"] / result["drafted"] if result["drafted"] else 0.0
            ),
            "draft_elapsed_ms": result["draft_elapsed_ms"],
            "target_forwards": result["target_forwards"],
            "draft_forwards": result["draft_forwards"],
        })
        if next_idx < total_requests:
            pending.append((next_idx, finish))
            next_idx += 1

    wall_s = time.perf_counter() - wall_start
    return rows, wall_s


def summarize(mode, n_clients, rows, wall_s, target_model, draft_backend, k):
    tokens = sum(r["generated_tokens"] for r in rows)
    lat = [r["latency_s"] for r in rows]
    tpot = [r["tpot_ms"] for r in rows]
    queue = [r["queue_s"] for r in rows]
    drafted = sum(r["drafted"] for r in rows)
    accepted = sum(r["accepted_draft"] for r in rows)
    return {
        "mode": mode,
        "N": n_clients,
        "requests": len(rows),
        "tokens": tokens,
        "wall_s": wall_s,
        "throughput_tok_s": tokens / wall_s if wall_s > 0 else float("nan"),
        "mean_latency_s": statistics.mean(lat),
        "p50_latency_s": percentile(lat, 0.50),
        "p95_latency_s": percentile(lat, 0.95),
        "p99_latency_s": percentile(lat, 0.99),
        "mean_tpot_ms": statistics.mean(tpot),
        "p50_tpot_ms": percentile(tpot, 0.50),
        "p95_tpot_ms": percentile(tpot, 0.95),
        "p99_tpot_ms": percentile(tpot, 0.99),
        "mean_queue_s": statistics.mean(queue),
        "p95_queue_s": percentile(queue, 0.95),
        "acceptance_rate": accepted / drafted if drafted else 0.0,
        "target_forwards": sum(r["target_forwards"] for r in rows),
        "draft_forwards": sum(r["draft_forwards"] for r in rows),
        "target_model": target_model,
        "draft_backend": draft_backend,
        "k": k if mode != "single" else 0,
    }


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=f"{ROOT}/models/models--Qwen--Qwen2.5-7B-Instruct")
    parser.add_argument("--draft-local", default=f"{ROOT}/models/models--Qwen--Qwen2.5-0.5B-Instruct")
    parser.add_argument("--draft-remote", default="~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B")
    parser.add_argument("--trace", default=f"{ROOT}/data/sharegpt_trace.json")
    parser.add_argument("--out", default=f"{EXPERIMENT_DIR}/results/serving_benchmark.csv")
    parser.add_argument("--request-out", default=f"{EXPERIMENT_DIR}/results/serving_requests.csv")
    parser.add_argument("--draft-backend", choices=["edge", "local"], default="edge")
    parser.add_argument("--modes", default="single,multi")
    parser.add_argument("--Ns", default="4,8,16,32")
    parser.add_argument("--requests-per-N", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--edge-host", default=os.environ.get("JETSON_HOST", "192.168.3.108"))
    parser.add_argument(
        "--edge-hosts",
        default=os.environ.get("JETSON_HOSTS", ""),
        help="Comma-separated Jetson hosts. Overrides --edge-host when set.",
    )
    parser.add_argument("--edge-user", default=os.environ.get("JETSON_USER", "jetson"))
    parser.add_argument("--edge-password", default=os.environ.get("JETSON_PASSWORD", ""))
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    target_path = resolve_snapshot(args.target)
    tokenizer, target = load_target(args.target)
    warmup_target(target, tokenizer.vocab_size)

    ns = [int(x) for x in args.Ns.split(",") if x.strip()]
    max_requests = max(max(ns), args.requests_per_N)
    prompts = load_prompts(args.trace, tokenizer, args.prompt_len, max_requests, args.seed)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    summary_rows = []
    request_rows = []
    draft_client = None
    try:
        if "multi" in modes:
            if args.draft_backend == "edge":
                hosts = [h.strip() for h in args.edge_hosts.split(",") if h.strip()]
                if not hosts:
                    hosts = [args.edge_host]
                draft_client = MultiEdgeDraftClient(
                    worker=f"{EXPERIMENT_DIR}/edge_draft_worker.py",
                    remote_model=args.draft_remote,
                    hosts=hosts,
                    user=args.edge_user,
                    password=args.edge_password,
                )
            else:
                draft_client = LocalDraftClient(args.draft_local)

        for n_clients in ns:
            requests = max(n_clients, args.requests_per_N)
            run_prompts = prompts[:requests]
            for mode in modes:
                label = "single" if mode == "single" else f"multi_{args.draft_backend}_k{args.k}"
                print(
                    f"[bench] mode={label} N={n_clients} requests={requests} "
                    f"prompt_len={args.prompt_len} max_new={args.max_new_tokens}",
                    flush=True,
                )
                rows, wall_s = run_closed_loop(
                    "single" if mode == "single" else "multi",
                    n_clients,
                    run_prompts,
                    target,
                    args.max_new_tokens,
                    args.k,
                    draft_client,
                )
                for row in rows:
                    row["mode"] = label
                summary = summarize(
                    label,
                    n_clients,
                    rows,
                    wall_s,
                    target_path,
                    args.draft_backend,
                    args.k,
                )
                summary_rows.append(summary)
                request_rows.extend(rows)
                print(
                    f"[bench] {label} N={n_clients}: "
                    f"throughput={summary['throughput_tok_s']:.2f} tok/s, "
                    f"p99_tpot={summary['p99_tpot_ms']:.1f} ms/token, "
                    f"accept={summary['acceptance_rate']:.3f}",
                    flush=True,
                )
    finally:
        if draft_client is not None:
            draft_client.close()

    summary_fields = [
        "mode", "N", "requests", "tokens", "wall_s", "throughput_tok_s",
        "mean_latency_s", "p50_latency_s", "p95_latency_s", "p99_latency_s",
        "mean_tpot_ms", "p50_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
        "mean_queue_s", "p95_queue_s", "acceptance_rate", "target_forwards",
        "draft_forwards", "target_model", "draft_backend", "k",
    ]
    request_fields = [
        "mode", "N", "request_id", "prompt_len", "generated_tokens",
        "latency_s", "service_s", "queue_s", "tpot_ms", "service_tpot_ms",
        "accepted_draft", "drafted", "acceptance_rate", "draft_elapsed_ms",
        "target_forwards", "draft_forwards",
    ]
    write_csv(args.out, summary_rows, summary_fields)
    write_csv(args.request_out, request_rows, request_fields)
    print(f"[bench] wrote {args.out}", flush=True)
    print(f"[bench] wrote {args.request_out}", flush=True)


if __name__ == "__main__":
    main()
