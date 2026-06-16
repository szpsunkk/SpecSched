#!/usr/bin/env python3
"""Profile and fit the padded verification latency model.

This script is an auditable entry point for improve_cc.md A4. It first builds
a target KV cache for each (batch, context length) point, then measures only
the cached verification forward over k+1 new tokens. It fits a linear model:

    latency_ms ~= c0 + c1 * B * (k + 1) + c2 * B * L + c3 * B * L * (k + 1)

The raw measurements and processed fit metrics are written as CSV/JSON so the
paper can report MAE/RMSE/MAPE/R2 only after this script has been run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

try:
    from transformers.cache_utils import DynamicCache  # noqa: E402
except Exception:  # pragma: no cover
    DynamicCache = None  # type: ignore

ROOT = Path(__file__).resolve().parents[3]


def resolve_snapshot(path: str) -> str:
    path = os.path.expanduser(path)
    snaps = os.path.join(path, "snapshots")
    if os.path.isdir(snaps):
        for d in sorted(os.listdir(snaps)):
            full = os.path.join(snaps, d)
            if os.path.isdir(full):
                return full
    return path


def parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def load_model(path: str):
    model_path = resolve_snapshot(path)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    return tok, mdl, model_path


def normalize_cache(past):
    if past is None or DynamicCache is None or isinstance(past, DynamicCache):
        return past
    if isinstance(past, tuple) and hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(past)
    return past


def build_features(batch_size: int, draft_k: int, context_len: int) -> list[float]:
    tokens = batch_size * (draft_k + 1)
    context = batch_size * context_len
    interaction = batch_size * context_len * (draft_k + 1)
    return [1.0, float(tokens), float(context), float(interaction)]


def mat_vec(features: list[list[float]], coeffs: torch.Tensor) -> list[float]:
    x = torch.tensor(features, dtype=torch.float64)
    return (x @ coeffs).tolist()


def fit_linear(rows: list[dict]) -> tuple[list[float], list[dict], dict]:
    features = [
        build_features(int(r["batch_size"]), int(r["draft_k"]), int(r["context_len"]))
        for r in rows
    ]
    y = torch.tensor([float(r["latency_ms"]) for r in rows], dtype=torch.float64).view(-1, 1)
    x = torch.tensor(features, dtype=torch.float64)
    coeffs = torch.linalg.lstsq(x, y).solution.view(-1)
    preds = mat_vec(features, coeffs)

    y_vals = [float(v) for v in y.view(-1).tolist()]
    mean_y = sum(y_vals) / max(1, len(y_vals))
    abs_err = [abs(p - t) for p, t in zip(preds, y_vals)]
    sq_err = [(p - t) ** 2 for p, t in zip(preds, y_vals)]
    ape = [abs(p - t) / max(abs(t), 1e-9) * 100.0 for p, t in zip(preds, y_vals)]
    ss_res = sum(sq_err)
    ss_tot = sum((t - mean_y) ** 2 for t in y_vals)
    metrics = {
        "count": len(rows),
        "mae_ms": sum(abs_err) / max(1, len(abs_err)),
        "rmse_ms": math.sqrt(sum(sq_err) / max(1, len(sq_err))),
        "mape_pct": sum(ape) / max(1, len(ape)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "coef_intercept": float(coeffs[0]),
        "coef_tokens": float(coeffs[1]),
        "coef_context": float(coeffs[2]),
        "coef_interaction": float(coeffs[3]),
    }

    pred_rows = []
    for row, pred, truth in zip(rows, preds, y_vals):
        pred_rows.append({
            **row,
            "pred_latency_ms": pred,
            "abs_error_ms": abs(pred - truth),
            "ape_pct": abs(pred - truth) / max(abs(truth), 1e-9) * 100.0,
        })
    return [float(c) for c in coeffs.tolist()], pred_rows, metrics


@torch.inference_mode()
def build_past(model, input_ids: torch.Tensor):
    out = model(input_ids=input_ids, use_cache=True)
    torch.cuda.synchronize()
    return normalize_cache(out.past_key_values)


@torch.inference_mode()
def cached_verify_once(model, verify_ids: torch.Tensor, past) -> float:
    t0 = time.perf_counter()
    model(input_ids=verify_ids, past_key_values=past, use_cache=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


@torch.inference_mode()
def measure_case(model, batch_size: int, draft_k: int, context_len: int, repeats: int, warmup: int) -> dict:
    device = next(model.parameters()).device
    verify_len = draft_k + 1
    # Use random IDs so token content does not require tokenizer-specific text.
    vocab = int(getattr(model.config, "vocab_size", 32000))
    context_ids = torch.randint(0, vocab, (batch_size, context_len), device=device)
    verify_ids = torch.randint(0, vocab, (batch_size, verify_len), device=device)

    for _ in range(warmup):
        past = build_past(model, context_ids)
        cached_verify_once(model, verify_ids, past)
        del past
        torch.cuda.empty_cache()

    samples = []
    for _ in range(repeats):
        past = build_past(model, context_ids)
        samples.append(cached_verify_once(model, verify_ids, past))
        del past
        torch.cuda.empty_cache()

    samples_sorted = sorted(samples)
    return {
        "batch_size": batch_size,
        "draft_k": draft_k,
        "context_len": context_len,
        "tokens": batch_size * (draft_k + 1),
        "verify_len": verify_len,
        "repeats": repeats,
        "latency_ms": sum(samples) / len(samples),
        "p50_latency_ms": samples_sorted[len(samples_sorted) // 2],
        "min_latency_ms": min(samples),
        "max_latency_ms": max(samples),
        "profile_mode": "cached_verify",
    }


def iter_grid(batch_sizes: Iterable[int], draft_ks: Iterable[int], context_lens: Iterable[int]):
    for batch_size in batch_sizes:
        for draft_k in draft_ks:
            for context_len in context_lens:
                yield batch_size, draft_k, context_len


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="HF target model path")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--draft-ks", default="1,2,3,5,7")
    parser.add_argument("--context-lens", default="64,128,256,512")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "verify_latency_fit"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    processed_dir = out_dir / "processed"

    _, model, model_path = load_model(args.target)
    rows = []
    for batch_size, draft_k, context_len in iter_grid(
        parse_ints(args.batch_sizes),
        parse_ints(args.draft_ks),
        parse_ints(args.context_lens),
    ):
        try:
            row = measure_case(model, batch_size, draft_k, context_len, args.repeats, args.warmup)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            row = {
                "batch_size": batch_size,
                "draft_k": draft_k,
                "context_len": context_len,
                "tokens": batch_size * (draft_k + 1),
                "verify_len": draft_k + 1,
                "repeats": args.repeats,
                "latency_ms": "",
                "p50_latency_ms": "",
                "min_latency_ms": "",
                "max_latency_ms": "",
                "profile_mode": "cached_verify",
                "status": "oom",
            }
        rows.append(row)
        if row.get("status") == "oom":
            print(
                f"[verify-fit][OOM] B={batch_size} k={draft_k} L={context_len}",
                flush=True,
            )
            continue
        print(
            f"[verify-fit] B={batch_size} k={draft_k} L={context_len} "
            f"lat={row['latency_ms']:.3f} ms",
            flush=True,
        )

    fit_rows = [row for row in rows if row.get("latency_ms") not in ("", None)]
    if not fit_rows:
        raise RuntimeError("No successful latency measurements; reduce the grid or batch sizes.")
    _, pred_rows, metrics = fit_linear(fit_rows)
    manifest = {
        "target_model": model_path,
        "seed": args.seed,
        "batch_sizes": parse_ints(args.batch_sizes),
        "draft_ks": parse_ints(args.draft_ks),
        "context_lens": parse_ints(args.context_lens),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "profile_mode": "cached_verify",
        "oom_cases": sum(1 for row in rows if row.get("status") == "oom"),
        "metrics": metrics,
    }

    write_csv(raw_dir / "verify_latency_measurements.csv", rows)
    write_csv(processed_dir / "verify_latency_fit.csv", pred_rows)
    (processed_dir / "verify_latency_fit_metrics.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
