"""Real 3090Ti scheduler benchmark — Phase 1.

Implements NoSpec, FixedK-k, IndepOpt, SpecSched, and three SpecSched
ablations (NoBatch / NoEvict / NoSLO) on top of the corrected speculative
kernel and the multi-tenant scheduling engine in `core/`.

Key changes vs the previous version:
    - Target KV-cache is reused across verification steps (no more full-seq
      re-prefill per verify; the speculative win is no longer cancelled).
    - True multi-tenant scheduling (admission + batching + KV eviction) lives
      in core/engine.py; this driver only orchestrates calibration and runs.
    - SLO-aware goodput and attainment are first-class metrics.
    - Workload mixes short/medium/long prompts so IndepOpt has real bucket
      structure to exploit.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Mapping, Optional, Sequence, Union

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

EXP_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from core.engine import Engine, EngineConfig  # noqa: E402
from core.kv_cache import KVManager  # noqa: E402
from core.metrics import record_to_row, summarize  # noqa: E402
from core.slo import SLOConfig, make_slo  # noqa: E402
from core.speculative import (  # noqa: E402
    ReqState, draft_propose, prefill_draft, prefill_target, target_verify,
)
from core.workload import (  # noqa: E402
    bucket_names, bucket_of, load_prompts, make_closed_loop_workload,
    make_bursty_workload, make_poisson_workload, make_trace_replay_workload,
)
from edge_client import MultiEdgeDraftClient  # noqa: E402
from kv_policies import policies as kv_policies  # noqa: E402
from policies.registry import (  # noqa: E402
    SpecSchedParams, eagle2_proxy, fixed_k, indep_opt, nospec, specsched,
    specsched_lowload_fixed, specsched_no_batch, specsched_no_slo,
    specsched_pressure_aimd, specsched_regime_aware, specserve_proxy, tetris_proxy,
)


def resolve_snapshot(path: str) -> str:
    path = os.path.expanduser(path)
    snaps = os.path.join(path, "snapshots")
    if os.path.isdir(snaps):
        for d in sorted(os.listdir(snaps)):
            full = os.path.join(snaps, d)
            if os.path.isdir(full):
                return full
    return path


def load_model(path: str, label: str):
    model_path = resolve_snapshot(path)
    print(f"[{label}] loading {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    return tok, mdl, model_path


@torch.inference_mode()
def warmup(model):
    ids = torch.randint(0, 1000, (1, 32), device="cuda")
    for _ in range(2):
        model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()


# -------- Calibration -------------------------------------------------------

def calibrate(
    target, draft, prompts, max_new_tokens, candidate_ks, out_csv,
) -> Dict[str, int]:
    """Measure throughput for each (bucket, k) and pick best k per bucket.

    Also returns reference (TTFT, TPOT) at k=0 for SLO preset scaling.
    """
    rows: List[dict] = []
    print(f"[calib] prompts={len(prompts)} ks={candidate_ks}", flush=True)
    for idx, ids in enumerate(prompts):
        bucket = bucket_of(int(ids.numel()))
        for k in candidate_ks:
            state = ReqState(
                req_id=idx, prompt_ids=ids, arrival_s=0.0,
                max_new_tokens=max_new_tokens, slo_ttft_ms=None,
                slo_tpot_ms=None, bucket=bucket,
            )
            t0 = time.perf_counter()
            state.target_past, first_tok, ttft_ms = prefill_target(target, ids)
            state.draft_past, _ = prefill_draft(draft, ids, idx) if draft is not None else (None, 0.0)
            state.generated.append(first_tok)
            state.last_token = first_tok
            state.target_forwards += 1
            while len(state.generated) < max_new_tokens:
                remaining = max_new_tokens - len(state.generated)
                if k > 0 and draft is not None:
                    proposals, _ = draft_propose(draft, state, min(k, remaining))
                else:
                    proposals = []
                target_verify(target, draft, state, proposals, remaining)
            elapsed = time.perf_counter() - t0
            rows.append({
                "calib_id": idx,
                "prompt_len": int(ids.numel()),
                "bucket": bucket,
                "k": k,
                "tokens": len(state.generated),
                "service_s": elapsed,
                "tokens_per_s": len(state.generated) / elapsed if elapsed > 0 else 0.0,
                "accepted": state.accepted_total,
                "drafted": state.drafted_total,
                "acceptance": state.accepted_total / state.drafted_total if state.drafted_total else 0.0,
                "target_forwards": state.target_forwards,
                "ttft_ms": ttft_ms,
                "tpot_ms": elapsed * 1000.0 / max(1, len(state.generated)),
            })
            print(
                f"[calib] id={idx:02d} bucket={bucket} k={k} "
                f"tok/s={rows[-1]['tokens_per_s']:.2f} accept={rows[-1]['acceptance']:.3f}",
                flush=True,
            )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Per-bucket best k by mean tok/s
    by_bucket: Dict[str, Dict[int, List[float]]] = {}
    for r in rows:
        by_bucket.setdefault(r["bucket"], {}).setdefault(r["k"], []).append(r["tokens_per_s"])
    bucket_k: Dict[str, int] = {}
    for bucket, ks in by_bucket.items():
        best = max(ks, key=lambda kk: sum(ks[kk]) / len(ks[kk]))
        bucket_k[bucket] = int(best)
    for name in bucket_names():
        bucket_k.setdefault(name, 0)
    print(f"[calib] IndepOpt bucket_k = {bucket_k}", flush=True)
    # ex.md §3.3: IndepOpt must yield a non-trivial k (target 3-7). Surface a
    # loud warning when buckets collapse to k=0 — usually because draft latency
    # is too high relative to target step latency.
    nonzero = sum(1 for v in bucket_k.values() if v > 0)
    if nonzero == 0:
        print(
            "[calib][WARN] IndepOpt selected k=0 for every bucket -- it will "
            "degenerate to NoSpec. Check draft/target latency ratio (ex.md §3.3).",
            flush=True,
        )
    elif nonzero < len(bucket_k):
        zero_buckets = sorted(b for b, v in bucket_k.items() if v == 0)
        print(
            f"[calib][WARN] IndepOpt selected k=0 for buckets {zero_buckets}; "
            "Observation 4 will be partial.",
            flush=True,
        )
    return bucket_k, rows


def reference_ttft_tpot(calib_rows: List[dict]) -> tuple:
    """Use NoSpec (k=0) calibration rows to set SLO reference TTFT/TPOT."""
    nospec = [r for r in calib_rows if r["k"] == 0]
    if not nospec:
        return None, None
    ttft = sum(r["ttft_ms"] for r in nospec) / len(nospec)
    tpot = sum(r["tpot_ms"] for r in nospec) / len(nospec)
    return ttft, tpot


def speculation_ok_detector(
    calib_rows: List[dict],
    kmax: int,
    threshold_margin: float,
) -> Dict[str, object]:
    """Profile-time break-even detector for safe speculative fallback.

    The detector uses calibration rows already collected by the benchmark. It
    compares mean NoSpec throughput against the best speculative calibration
    throughput. If speculation fails to clear the configured margin, deployment
    should use NoSpec for SpecSched-Safe on this target/draft pair.
    """
    nospec = [float(row["tokens_per_s"]) for row in calib_rows if int(row["k"]) == 0]
    spec = [float(row["tokens_per_s"]) for row in calib_rows if int(row["k"]) > 0]
    accept = [float(row["acceptance"]) for row in calib_rows if int(row["k"]) > 0]
    best_by_k: Dict[int, List[float]] = {}
    for row in calib_rows:
        k = int(row["k"])
        if k > 0:
            best_by_k.setdefault(k, []).append(float(row["tokens_per_s"]))
    nospec_tput = statistics.mean(nospec) if nospec else 0.0
    best_spec_k = 0
    best_spec_tput = 0.0
    for k, vals in best_by_k.items():
        mean = statistics.mean(vals) if vals else 0.0
        if mean > best_spec_tput:
            best_spec_tput = mean
            best_spec_k = k
    ratio = best_spec_tput / nospec_tput if nospec_tput > 0 else 0.0
    ok = ratio >= (1.0 + threshold_margin)
    alpha_hat = statistics.mean(accept) if accept else 0.0
    break_even_ratio = 1.0 + (1.0 / max(alpha_hat * max(1, kmax), 1e-9))
    print(
        f"[detector] nospec_tput={nospec_tput:.2f} best_spec_tput={best_spec_tput:.2f} "
        f"best_k={best_spec_k} ratio={ratio:.3f} margin={threshold_margin:.3f} "
        f"alpha={alpha_hat:.3f} analytic_break_even={break_even_ratio:.3f} "
        f"speculation_ok={int(ok)}",
        flush=True,
    )
    return {
        "speculation_ok": bool(ok),
        "decision": "speculate" if ok else "fallback_nospec",
        "nospec_tput": nospec_tput,
        "best_spec_tput": best_spec_tput,
        "best_spec_k": best_spec_k,
        "tput_ratio": ratio,
        "margin": threshold_margin,
        "alpha_hat": alpha_hat,
        "analytic_break_even_ratio": break_even_ratio,
    }


def specsched_calib_hints(
    calib_rows: List[dict],
    fixed_ks: List[int],
    near_best_ratio: float = 0.90,
) -> Dict[str, object]:
    by_bucket: Dict[str, Dict[int, List[float]]] = {}
    accept_by_bucket: Dict[str, List[float]] = {}
    for row in calib_rows:
        bucket = str(row["bucket"])
        k = int(row["k"])
        by_bucket.setdefault(bucket, {}).setdefault(k, []).append(float(row["tokens_per_s"]))
        if k > 0:
            accept_by_bucket.setdefault(bucket, []).append(float(row["acceptance"]))

    preferred_caps: List[int] = []
    for vals in by_bucket.values():
        means = {k: sum(v) / len(v) for k, v in vals.items() if v}
        if not means:
            continue
        best_k = max(means, key=means.get)
        best_tput = means[best_k]
        k1_tput = means.get(1)
        if k1_tput is not None and best_tput > 0 and k1_tput / best_tput >= near_best_ratio:
            preferred_caps.append(1)
        else:
            preferred_caps.append(int(best_k))

    low_pressure_k_cap = 3
    if preferred_caps and sum(1 for k in preferred_caps if k <= 1) >= math.ceil(len(preferred_caps) / 2):
        low_pressure_k_cap = 1
    if 1 in set(int(k) for k in fixed_ks):
        low_pressure_k_cap = min(low_pressure_k_cap, 1)
    print(
        f"[calib] SpecSched low_pressure_k_cap={low_pressure_k_cap} "
        f"preferred_caps={preferred_caps} fixed_ks={fixed_ks}",
        flush=True,
    )
    bucket_accept0 = {
        bucket: sum(vals) / len(vals)
        for bucket, vals in accept_by_bucket.items()
        if vals
    }
    return {"low_pressure_k_cap": low_pressure_k_cap, "bucket_accept0": bucket_accept0}


# -------- Run one (policy, N) cell -----------------------------------------

def build_policy(name: str, fixed_ks, bucket_k, kmax: int, specsched_hints: Optional[Dict[str, int]] = None):
    raw_hints = dict(specsched_hints or {})
    if name == "NoSpec":
        return nospec(), False
    if name == "SpecSched-Safe" and not bool(raw_hints.get("detector_speculation_ok", True)):
        return nospec(), False
    hints = {
        key: value
        for key, value in raw_hints.items()
        if key in getattr(SpecSchedParams, "__dataclass_fields__", {})
    }
    if name.startswith("FixedK-"):
        k = int(name.split("-", 1)[1])
        return fixed_k(k), True
    if name == "IndepOpt":
        return indep_opt(bucket_k), True
    if name == "TETRIS-Proxy":
        return tetris_proxy(kmax=kmax), True
    if name == "EAGLE-2-Proxy":
        return eagle2_proxy(kmax=kmax, conf_threshold=0.45), True
    if name == "SpecServe-Proxy":
        return specserve_proxy(kmax=kmax), True
    if name == "SpecSched":
        params = SpecSchedParams(kmax=kmax, bucket_k0=bucket_k, use_slo=True, homogenize_k=True, **hints)
        return specsched(params), True
    if name == "SpecSched-Safe":
        params = SpecSchedParams(kmax=kmax, bucket_k0=bucket_k, use_slo=True, homogenize_k=True, **hints)
        return specsched(params), True
    if name == "SpecSched-PosAlpha":
        params = SpecSchedParams(
            kmax=kmax,
            bucket_k0=bucket_k,
            use_slo=True,
            homogenize_k=True,
            use_position_alpha=True,
            **hints,
        )
        return specsched(params), True
    if name == "SpecSched-EvictV2":
        params = SpecSchedParams(kmax=kmax, bucket_k0=bucket_k, use_slo=True, homogenize_k=True, **hints)
        return specsched(params), True
    if name == "SpecSched-EvictGuarded":
        params = SpecSchedParams(kmax=kmax, bucket_k0=bucket_k, use_slo=True, homogenize_k=True, **hints)
        return specsched(params), True
    if name == "SpecSched-EvictCalibrated":
        params = SpecSchedParams(kmax=kmax, bucket_k0=bucket_k, use_slo=True, homogenize_k=True, **hints)
        return specsched(params), True
    if name == "SpecSched-GuardedK":
        params = SpecSchedParams(
            kmax=kmax,
            bucket_k0=bucket_k,
            use_slo=True,
            homogenize_k=True,
            adaptive_k_guard=True,
            **hints,
        )
        return specsched(params), True
    if name == "SpecSched-QueueAware":
        params = SpecSchedParams(
            kmax=kmax,
            bucket_k0=bucket_k,
            use_slo=True,
            homogenize_k=True,
            queue_aware=True,
            queue_k_guard=True,
            **hints,
        )
        return specsched(params), True
    if name == "SpecSched-RegimeAware":
        params = SpecSchedParams(
            kmax=kmax,
            bucket_k0=bucket_k,
            use_slo=True,
            homogenize_k=True,
            queue_aware=True,
            queue_k_guard=True,
            regime_aware=True,
            **hints,
        )
        return specsched_regime_aware(params), True
    if name == "SpecSched-AIMD":
        params = SpecSchedParams(
            kmax=kmax,
            bucket_k0=bucket_k,
            use_slo=True,
            homogenize_k=True,
            aimd_controller=True,
            **hints,
        )
        return specsched_regime_aware(params), True
    if name == "SpecSched-AdmissionAware":
        params = SpecSchedParams(
            kmax=kmax,
            bucket_k0=bucket_k,
            use_slo=True,
            homogenize_k=True,
            admission_aware=True,
            **hints,
        )
        return specsched_regime_aware(params), True
    if name == "SpecSched-Backpressure":
        params = SpecSchedParams(
            kmax=kmax,
            bucket_k0=bucket_k,
            use_slo=True,
            homogenize_k=True,
            backpressure_controller=True,
            **hints,
        )
        return specsched_regime_aware(params), True
    if name == "SpecSched-PressureAIMD":
        params = SpecSchedParams(
            kmax=kmax,
            bucket_k0=bucket_k,
            use_slo=True,
            homogenize_k=True,
            pressure_aimd_controller=True,
            **hints,
        )
        return specsched_pressure_aimd(params), True
    if name == "SpecSched-LowLoadFixed":
        params = SpecSchedParams(kmax=kmax, bucket_k0=bucket_k, use_slo=True, homogenize_k=True, **hints)
        return specsched_lowload_fixed(params), True
    if name == "SpecSched-NoBatch":
        params = SpecSchedParams(kmax=kmax, bucket_k0=bucket_k, use_slo=True, homogenize_k=False, **hints)
        return specsched_no_batch(params), True
    if name == "SpecSched-NoSLO":
        params = SpecSchedParams(kmax=kmax, bucket_k0=bucket_k, use_slo=False, homogenize_k=True, **hints)
        return specsched_no_slo(params), True
    if name == "SpecSched-NoEvict":
        params = SpecSchedParams(kmax=kmax, bucket_k0=bucket_k, use_slo=True, homogenize_k=True, **hints)
        return specsched(params), True
    if name == "SpecSched-LRU":
        params = SpecSchedParams(kmax=kmax, bucket_k0=bucket_k, use_slo=True, homogenize_k=True, **hints)
        return specsched(params), True
    if name == "SpecSched-Longest":
        params = SpecSchedParams(kmax=kmax, bucket_k0=bucket_k, use_slo=True, homogenize_k=True, **hints)
        return specsched(params), True
    if name == "SpecSched-Shortest":
        params = SpecSchedParams(kmax=kmax, bucket_k0=bucket_k, use_slo=True, homogenize_k=True, **hints)
        return specsched(params), True
    if name == "SpecSched-H2O":
        params = SpecSchedParams(kmax=kmax, bucket_k0=bucket_k, use_slo=True, homogenize_k=True, **hints)
        return specsched(params), True
    if name == "SpecSched-vLLM":
        params = SpecSchedParams(kmax=kmax, bucket_k0=bucket_k, use_slo=True, homogenize_k=True, **hints)
        return specsched(params), True
    raise ValueError(f"unknown policy {name}")


def kv_policy_for(name: str):
    if name == "SpecSched":
        return kv_policies.get("spec_aware")
    if name == "SpecSched-NoEvict":
        return kv_policies.get("no_evict")
    if name == "SpecSched-LRU":
        return kv_policies.get("lru")
    if name == "SpecSched-Longest":
        return kv_policies.get("longest")
    if name == "SpecSched-Shortest":
        return kv_policies.get("shortest")
    if name == "SpecSched-H2O":
        return kv_policies.get("h2o")
    if name == "SpecSched-vLLM":
        return kv_policies.get("vllm")
    if name == "SpecSched-EvictV2":
        return kv_policies.get("spec_aware_v2")
    if name == "SpecSched-EvictGuarded":
        return kv_policies.get("spec_aware_guarded")
    if name == "SpecSched-EvictCalibrated":
        return kv_policies.get("spec_aware_calibrated")
    if name == "SpecSched-Backpressure":
        return kv_policies.get("lru")
    if name == "SpecSched-PressureAIMD":
        return kv_policies.get("lru")
    if name.startswith("SpecSched"):
        return kv_policies.get("spec_aware")
    return kv_policies.get("lru")


def run_cell(
    policy_name: str,
    n_clients: int,
    request_count: int,
    prompts,
    target, draft,
    target_path: str, draft_path: Optional[str],
    max_new_tokens,
    fixed_ks, bucket_k, kmax: int,
    specsched_hints: Dict[str, object],
    kv_total_blocks: int, block_tokens: int,
    slo: SLOConfig,
    slo_by_bucket: Optional[Dict[str, SLOConfig]],
    workload_slo_ttft_ms: Optional[Union[float, Sequence[Optional[float]], Mapping[str, float]]],
    workload_slo_tpot_ms: Optional[Union[float, Sequence[Optional[float]], Mapping[str, float]]],
    max_batch: int,
    max_admit_per_step: int,
    cache_replay_chunk_tokens: int,
    log_every_steps: int,
    arrival_model: str,
    arrival_rate_rps: float,
    arrival_trace: str,
    burst_size: int,
    burst_period_s: float,
    tiers: Optional[Sequence[str]],
    tier_names_text: str,
    tier_weights_text: str,
    tier_tpot_mults_text: str,
    tier_ttft_mults_text: str,
    tier_tpot_mult: Mapping[str, float],
    tier_ttft_mult: Mapping[str, float],
    seed: int,
    deterministic: bool,
    pressure_aimd_min_extra_gain: float,
):
    policy, uses_draft = build_policy(policy_name, fixed_ks, bucket_k, kmax, specsched_hints)
    kv_policy = kv_policy_for(policy_name)
    kv = KVManager(total_blocks=kv_total_blocks, block_tokens=block_tokens, eviction_policy=kv_policy)
    setattr(policy, "_kv_manager", kv)

    cfg = EngineConfig(
        max_batch=(1 if policy_name == "SpecSched-NoBatch" else max_batch),
        batch_verify=(policy_name != "SpecSched-NoBatch"),
        use_speculation=uses_draft,
        use_draft=uses_draft,
        block_tokens=block_tokens,
        cache_replay_chunk_tokens=cache_replay_chunk_tokens,
        log_every_steps=log_every_steps,
        arrival_model=arrival_model,
        arrival_rate_rps=arrival_rate_rps,
        max_admit_per_step=max_admit_per_step,
    )
    engine = Engine(
        target_model=target,
        draft_model=draft if uses_draft else None,
        kv_manager=kv,
        cfg=cfg,
        policy=policy,
    )

    workload_prompts = prompts[:request_count]
    if isinstance(max_new_tokens, Sequence) and not isinstance(max_new_tokens, (str, bytes)):
        workload_max_new = list(max_new_tokens)[:request_count]
    else:
        workload_max_new = int(max_new_tokens)
    if workload_slo_ttft_ms is not None or workload_slo_tpot_ms is not None:
        ttft_by_bucket = workload_slo_ttft_ms
        tpot_by_bucket = workload_slo_tpot_ms
    elif slo_by_bucket:
        ttft_by_bucket = {k: v.ttft_ms for k, v in slo_by_bucket.items()}
        tpot_by_bucket = {k: v.tpot_ms for k, v in slo_by_bucket.items()}
    else:
        ttft_by_bucket = slo.ttft_ms
        tpot_by_bucket = slo.tpot_ms
    wl = make_workload(
        workload_prompts,
        n_clients,
        workload_max_new,
        ttft_by_bucket,
        tpot_by_bucket,
        arrival_model=arrival_model,
        arrival_rate_rps=arrival_rate_rps,
        arrival_trace=arrival_trace,
        burst_size=burst_size,
        burst_period_s=burst_period_s,
        tiers=tiers,
        seed=seed,
    )
    if (
        workload_slo_ttft_ms is None
        and workload_slo_tpot_ms is None
        and any(req.tier != "default" for req in wl.requests)
    ):
        # Trace and bursty replay can reorder prompts and can also override
        # tier labels from the trace file. Bind tier multipliers after the
        # workload is materialized so each request's actual bucket/tier pair
        # receives the SLO threshold recorded in the request-level CSV.
        for req in wl.requests:
            if req.slo_ttft_ms is not None:
                req.slo_ttft_ms = float(req.slo_ttft_ms) * float(tier_ttft_mult.get(str(req.tier), 1.0))
            if req.slo_tpot_ms is not None:
                req.slo_tpot_ms = float(req.slo_tpot_ms) * float(tier_tpot_mult.get(str(req.tier), 1.0))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = engine.run(wl)
    peak_gpu_mb = float("nan")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6

    summary = summarize(
        policy=policy_name,
        n_clients=n_clients,
        records=result.records,
        wall_s=result.wall_s,
        kv_stats=result.kv_stats,
        slo=slo,
        target_model=target_path,
        draft_model=draft_path if uses_draft else "",
        arrival_model=wl.arrival_model,
        rate_rps=wl.rate_rps,
        extras={
            "max_batch": cfg.max_batch,
            "max_admit_per_step": max_admit_per_step,
            "block_tokens": block_tokens,
            "kv_blocks": kv_total_blocks,
            "kmax": kmax,
            "slo_mode": "bucket" if slo_by_bucket else "global",
            "seed": seed,
            "request_count": request_count,
            "arrival_trace": arrival_trace,
            "trace_duration_s": wl.meta.get("duration_s", ""),
            "trace_rows": wl.meta.get("trace_rows", ""),
            "burst_size": wl.meta.get("burst_size", ""),
            "burst_period_s": wl.meta.get("burst_period_s", ""),
            "tier_names": tier_names_text,
            "tier_weights": tier_weights_text,
            "tier_tpot_mults": tier_tpot_mults_text,
            "tier_ttft_mults": tier_ttft_mults_text,
            "deterministic": int(deterministic),
            "peak_gpu_mb": peak_gpu_mb,
            "pressure_aimd_min_extra_gain": pressure_aimd_min_extra_gain,
            "detector_speculation_ok": specsched_hints.get("detector_speculation_ok", ""),
            "detector_decision": specsched_hints.get("detector_decision", ""),
            "detector_nospec_tput": specsched_hints.get("detector_nospec_tput", ""),
            "detector_best_spec_tput": specsched_hints.get("detector_best_spec_tput", ""),
            "detector_best_spec_k": specsched_hints.get("detector_best_spec_k", ""),
            "detector_tput_ratio": specsched_hints.get("detector_tput_ratio", ""),
            "detector_margin": specsched_hints.get("detector_margin", ""),
            "detector_alpha_hat": specsched_hints.get("detector_alpha_hat", ""),
            "detector_analytic_break_even_ratio": specsched_hints.get("detector_analytic_break_even_ratio", ""),
        },
    )
    req_rows = [record_to_row(r, policy_name, n_clients) for r in result.records]
    return summary, req_rows


# -------- Main --------------------------------------------------------------

def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def output_lengths_for(prompts, default_max_new: int, output_lengths: List[int]) -> List[int]:
    """Map each prompt's length bucket to an output length.

    Accepts any number of values in `output_lengths` (>=1). The smallest is
    used for the `short` bucket, the largest for `long`, the value closest to
    the middle of the sorted list for `medium`. If `output_lengths` is empty,
    every request uses `default_max_new`.
    """
    if not output_lengths:
        return [int(default_max_new) for _ in prompts]
    vals = sorted(int(x) for x in output_lengths)
    short = vals[0]
    long = vals[-1]
    medium = vals[(len(vals) - 1) // 2]
    by_bucket = {"short": short, "medium": medium, "long": long}
    return [by_bucket.get(bucket_of(int(ids.numel())), medium) for ids in prompts]


def parse_csv_floats(text: str) -> List[float]:
    if not text:
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def tiers_for_requests(count: int, tier_names: Sequence[str], tier_weights: Sequence[float]) -> List[str]:
    if count <= 0:
        return []
    names = [str(name).strip() for name in tier_names if str(name).strip()]
    if not names:
        return ["default"] * count
    weights = [float(w) for w in tier_weights[: len(names)]]
    if len(weights) < len(names):
        weights.extend([1.0] * (len(names) - len(weights)))
    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        weights = [1.0] * len(names)
        total = float(len(names))
    raw = [count * max(0.0, w) / total for w in weights]
    base = [int(math.floor(x)) for x in raw]
    remainder = count - sum(base)
    order = sorted(range(len(names)), key=lambda i: raw[i] - base[i], reverse=True)
    for idx in order[:remainder]:
        base[idx] += 1
    tiers: List[str] = []
    for name, n in zip(names, base):
        tiers.extend([name] * n)
    return tiers[:count] + [names[-1]] * max(0, count - len(tiers))


def tier_slo_maps(
    tiers: Sequence[str],
    base_ttft_ms: Optional[Union[float, Mapping[str, float]]],
    base_tpot_ms: Optional[Union[float, Mapping[str, float]]],
    tier_tpot_mult: Mapping[str, float],
    tier_ttft_mult: Mapping[str, float],
) -> tuple[List[Optional[float]], List[Optional[float]]]:
    def base_value(value, bucket: str):
        if isinstance(value, Mapping):
            return value.get(bucket)
        return value

    ttft_by_req: List[Optional[float]] = []
    tpot_by_req: List[Optional[float]] = []
    # Tier SLO is applied after bucket SLO calibration: a tight tier uses a
    # multiplier below 1, while relaxed tiers use larger multipliers.
    for tier in tiers:
        ttft_mult = float(tier_ttft_mult.get(tier, 1.0))
        tpot_mult = float(tier_tpot_mult.get(tier, 1.0))
        # Placeholder bucket is resolved by caller when base is already scalar.
        if isinstance(base_ttft_ms, Mapping) or isinstance(base_tpot_ms, Mapping):
            ttft_by_req.append(None)
            tpot_by_req.append(None)
        else:
            ttft_by_req.append(None if base_ttft_ms is None else float(base_ttft_ms) * ttft_mult)
            tpot_by_req.append(None if base_tpot_ms is None else float(base_tpot_ms) * tpot_mult)
    return ttft_by_req, tpot_by_req


def apply_tier_slo(
    prompts: Sequence[torch.Tensor],
    tiers: Sequence[str],
    base_ttft_ms: Optional[Union[float, Mapping[str, float]]],
    base_tpot_ms: Optional[Union[float, Mapping[str, float]]],
    tier_tpot_mult: Mapping[str, float],
    tier_ttft_mult: Mapping[str, float],
) -> tuple[Optional[Union[float, List[Optional[float]], Mapping[str, float]]], Optional[Union[float, List[Optional[float]], Mapping[str, float]]]]:
    if not tiers or set(tiers) == {"default"}:
        return base_ttft_ms, base_tpot_ms

    def base_value(value, bucket: str):
        if isinstance(value, Mapping):
            return value.get(bucket)
        return value

    ttft: List[Optional[float]] = []
    tpot: List[Optional[float]] = []
    for ids, tier in zip(prompts, tiers):
        bucket = bucket_of(int(ids.numel()))
        tier = str(tier)
        ttft_base = base_value(base_ttft_ms, bucket)
        tpot_base = base_value(base_tpot_ms, bucket)
        ttft.append(None if ttft_base is None else float(ttft_base) * float(tier_ttft_mult.get(tier, 1.0)))
        tpot.append(None if tpot_base is None else float(tpot_base) * float(tier_tpot_mult.get(tier, 1.0)))
    return ttft, tpot


def make_workload(
    prompts,
    n_clients: int,
    max_new_tokens: Union[int, Sequence[int]],
    slo_ttft_ms: Optional[Union[float, Mapping[str, float]]],
    slo_tpot_ms: Optional[Union[float, Mapping[str, float]]],
    arrival_model: str,
    arrival_rate_rps: float,
    arrival_trace: str,
    burst_size: int,
    burst_period_s: float,
    tiers: Optional[Sequence[str]],
    seed: int,
):
    if arrival_model == "closed_loop":
        return make_closed_loop_workload(
            prompts,
            n_clients,
            max_new_tokens,
            slo_ttft_ms,
            slo_tpot_ms,
            tiers=tiers,
        )
    if arrival_model == "poisson":
        if isinstance(max_new_tokens, int):
            poisson_max_new: Union[int, Sequence[int]] = max_new_tokens
        else:
            poisson_max_new = list(max_new_tokens)
        return make_poisson_workload(
            prompts,
            rate_rps=arrival_rate_rps,
            max_new_tokens=poisson_max_new,
            slo_ttft_ms=slo_ttft_ms,
            slo_tpot_ms=slo_tpot_ms,
            seed=seed,
            tiers=tiers,
        )
    if arrival_model == "bursty":
        if isinstance(max_new_tokens, int):
            bursty_max_new: Union[int, Sequence[int]] = max_new_tokens
        else:
            bursty_max_new = list(max_new_tokens)
        return make_bursty_workload(
            prompts,
            burst_size=burst_size,
            burst_period_s=burst_period_s,
            max_new_tokens=bursty_max_new,
            slo_ttft_ms=slo_ttft_ms,
            slo_tpot_ms=slo_tpot_ms,
            seed=seed,
            tiers=tiers,
        )
    if arrival_model == "trace":
        if isinstance(max_new_tokens, int):
            trace_max_new: Union[int, Sequence[int]] = max_new_tokens
        else:
            trace_max_new = list(max_new_tokens)
        if not arrival_trace:
            raise ValueError("--arrival-trace is required when --arrival-model=trace")
        return make_trace_replay_workload(
            prompts,
            trace_path=arrival_trace,
            max_new_tokens=trace_max_new,
            slo_ttft_ms=slo_ttft_ms,
            slo_tpot_ms=slo_tpot_ms,
            seed=seed,
            tiers=tiers,
        )
    raise ValueError(f"unsupported arrival_model={arrival_model!r}")


def median_by_bucket(rows: List[dict], metric: str) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["bucket"]), []).append(float(row[metric]))
    global_vals = [float(row[metric]) for row in rows]
    fallback = statistics.median(global_vals) if global_vals else float("nan")
    return {
        bucket: statistics.median(vals) if vals else fallback
        for bucket, vals in grouped.items()
    }


def bucket_thresholds(
    n: int,
    pilot_req_rows: List[dict],
    tpot_mult: float,
    ttft_mult: float,
) -> Dict[str, SLOConfig]:
    tpot = median_by_bucket(pilot_req_rows, "tpot_ms")
    ttft = median_by_bucket(pilot_req_rows, "ttft_ms")
    out: Dict[str, SLOConfig] = {}
    for bucket in bucket_names():
        if bucket in tpot or bucket in ttft:
            ref_tpot = tpot.get(bucket, statistics.median([float(r["tpot_ms"]) for r in pilot_req_rows]))
            ref_ttft = ttft.get(bucket, statistics.median([float(r["ttft_ms"]) for r in pilot_req_rows]))
            out[bucket] = SLOConfig(
                name=f"adaptive-bucket-{tpot_mult:g}x",
                ttft_ms=ref_ttft * ttft_mult,
                tpot_ms=ref_tpot * tpot_mult,
            )
    return out


def threshold_rows_for_requests(
    n: int,
    beta: float,
    pilot_req_rows: List[dict],
    slo_by_bucket: Dict[str, SLOConfig],
    tier_tpot_mult: Optional[Mapping[str, float]] = None,
    tier_ttft_mult: Optional[Mapping[str, float]] = None,
) -> List[dict]:
    med_tpot = median_by_bucket(pilot_req_rows, "tpot_ms")
    med_ttft = median_by_bucket(pilot_req_rows, "ttft_ms")
    rows = []
    for row in pilot_req_rows:
        bucket = str(row["bucket"])
        tier = str(row.get("tier", "default"))
        slo = slo_by_bucket[bucket]
        tpot_mult = float((tier_tpot_mult or {}).get(tier, 1.0))
        ttft_mult = float((tier_ttft_mult or {}).get(tier, 1.0))
        rows.append({
            "request_id": row["req_id"],
            "class": bucket,
            "tier": tier,
            "N": n,
            "median_tpot_nospec": med_tpot.get(bucket, ""),
            "median_ttft_nospec": med_ttft.get(bucket, ""),
            "beta": beta,
            "tier_tpot_mult": tpot_mult,
            "tier_ttft_mult": ttft_mult,
            "tau_slo": slo.tpot_ms * tpot_mult,
            "tau_ttft": slo.ttft_ms * ttft_mult,
            "unit": "ms/token",
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default=f"{ROOT}/models/models--Qwen--Qwen2.5-7B-Instruct")
    p.add_argument("--draft", default=f"{ROOT}/models/models--Qwen--Qwen2.5-0.5B-Instruct")
    p.add_argument("--draft-backend", choices=["local", "edge"], default="local")
    p.add_argument("--draft-remote", default="~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct")
    p.add_argument("--edge-hosts", default=os.environ.get("JETSON_HOSTS", "100.66.37.19,100.107.23.95,100.73.217.47"))
    p.add_argument("--edge-user", default=os.environ.get("JETSON_USER", "skk"))
    p.add_argument("--edge-password", default=os.environ.get("JETSON_PASSWORD", "111"))
    p.add_argument("--edge-request-timeout-s", type=float, default=float(os.environ.get("JETSON_REQUEST_TIMEOUT_S", "600")))
    p.add_argument("--trace", default=f"{ROOT}/data/sharegpt_trace.json")
    p.add_argument("--out", default=f"{EXP_DIR}/results/scheduler_summary.csv")
    p.add_argument("--request-out", default=f"{EXP_DIR}/results/scheduler_requests.csv")
    p.add_argument("--calib-out", default=f"{EXP_DIR}/results/scheduler_calibration.csv")
    p.add_argument("--slo-thresholds-out", default="")
    p.add_argument("--nospec-request-out", default="")
    p.add_argument("--Ns", default="4,8,16,32")
    p.add_argument("--fixed-ks", default="1,3,5,7")
    p.add_argument("--policies", default=(
        "NoSpec,FixedK,IndepOpt,SpecSched,SpecSched-NoBatch,SpecSched-NoEvict,SpecSched-NoSLO"
    ))
    p.add_argument("--requests-per-N", type=int, default=32)
    p.add_argument("--calib-requests", type=int, default=6)
    p.add_argument(
        "--calib-max-new-tokens",
        type=int,
        default=0,
        help="Optional shorter generation length for calibration; 0 uses --max-new-tokens.",
    )
    p.add_argument("--prompt-hard-max", type=int, default=384)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--output-lengths",
        default="",
        help="Optional bucketed output lengths, e.g. 64,128,256 for short/medium/long.",
    )
    p.add_argument("--kmax", type=int, default=7)
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument(
        "--max-admit-per-step",
        type=int,
        default=0,
        help="Limit newly arrived requests admitted per scheduler epoch; 0 admits all ready requests.",
    )
    p.add_argument(
        "--backpressure-pressure-threshold",
        type=float,
        default=2.0,
        help="Open-loop pressure threshold for SpecSched-Backpressure.",
    )
    p.add_argument(
        "--backpressure-delay-threshold-s",
        type=float,
        default=0.25,
        help="Oldest ready-pending delay threshold for SpecSched-Backpressure.",
    )
    p.add_argument(
        "--backpressure-cost",
        type=float,
        default=0.45,
        help="Penalty weight applied to extra draft tokens under open-loop backpressure.",
    )
    p.add_argument(
        "--backpressure-alpha-floor",
        type=float,
        default=0.78,
        help="Acceptance floor below which SpecSched-Backpressure caps k under pressure.",
    )
    p.add_argument(
        "--backpressure-urgency-cap",
        type=float,
        default=1.15,
        help="SLO urgency cap used by SpecSched-Backpressure marginal utility.",
    )
    p.add_argument(
        "--backpressure-opportunistic-k",
        type=int,
        default=2,
        help="Maximum opportunistic k when open-loop backlog is low.",
    )
    p.add_argument(
        "--backpressure-opportunistic-alpha",
        type=float,
        default=0.78,
        help="Acceptance threshold for opportunistic Backpressure drafting.",
    )
    p.add_argument(
        "--backpressure-opportunistic-pressure",
        type=float,
        default=0.25,
        help="Pressure-ratio ceiling for opportunistic Backpressure drafting.",
    )
    p.add_argument(
        "--backpressure-opportunistic-delay-ratio",
        type=float,
        default=0.25,
        help="Delay-ratio ceiling for opportunistic Backpressure drafting.",
    )
    p.add_argument(
        "--backpressure-opportunistic-kv-pressure",
        type=float,
        default=0.70,
        help="KV pressure ceiling for opportunistic Backpressure drafting.",
    )
    p.add_argument(
        "--backpressure-fastpath-kv-pressure",
        type=float,
        default=0.88,
        help="KV pressure threshold for FixedK-1 fast path in SpecSched-Backpressure.",
    )
    p.add_argument(
        "--pressure-aimd-pressure-threshold",
        type=float,
        default=1.0,
        help="Post-admission ready-backlog pressure threshold for SpecSched-PressureAIMD.",
    )
    p.add_argument(
        "--pressure-aimd-delay-threshold-s",
        type=float,
        default=0.25,
        help="Post-admission oldest-ready delay threshold for SpecSched-PressureAIMD.",
    )
    p.add_argument(
        "--pressure-aimd-add",
        type=float,
        default=0.20,
        help="Additive increase for the open-loop AIMD draft cap.",
    )
    p.add_argument(
        "--pressure-aimd-mult",
        type=float,
        default=0.70,
        help="Multiplicative decrease for the open-loop AIMD draft cap.",
    )
    p.add_argument(
        "--pressure-aimd-min",
        type=float,
        default=1.0,
        help="Minimum AIMD draft cap for SpecSched-PressureAIMD.",
    )
    p.add_argument(
        "--pressure-aimd-max",
        type=float,
        default=3.0,
        help="Maximum AIMD draft cap for SpecSched-PressureAIMD.",
    )
    p.add_argument(
        "--pressure-aimd-recovery-pressure",
        type=float,
        default=0.25,
        help="Ready-backlog pressure below which SpecSched-PressureAIMD additively recovers.",
    )
    p.add_argument(
        "--pressure-aimd-recovery-delay-ratio",
        type=float,
        default=0.25,
        help="Pending-delay ratio below which SpecSched-PressureAIMD additively recovers.",
    )
    p.add_argument(
        "--pressure-aimd-delay-gain",
        type=float,
        default=0.35,
        help="Bounded pending-delay multiplier in SpecSched-PressureAIMD marginal utility.",
    )
    p.add_argument(
        "--pressure-aimd-cost",
        type=float,
        default=0.30,
        help="Penalty for extra draft tokens under arrival pressure in SpecSched-PressureAIMD.",
    )
    p.add_argument(
        "--pressure-aimd-alpha-floor",
        type=float,
        default=0.70,
        help="Acceptance floor used by SpecSched-PressureAIMD pressure penalty.",
    )
    p.add_argument(
        "--pressure-aimd-extra-budget-frac",
        type=float,
        default=0.50,
        help="Fraction of AIMD extra draft capacity exposed as per-batch extra-token budget.",
    )
    p.add_argument(
        "--pressure-aimd-stable-pressure",
        type=float,
        default=0.25,
        help="Ready-backlog pressure that switches SpecSched-PressureAIMD to admission-stable batching.",
    )
    p.add_argument(
        "--pressure-aimd-stable-delay-ratio",
        type=float,
        default=0.25,
        help="Pending-delay ratio that switches SpecSched-PressureAIMD to admission-stable batching.",
    )
    p.add_argument(
        "--pressure-aimd-active-pressure",
        type=float,
        default=2.0,
        help="Active/max_batch ratio that switches SpecSched-PressureAIMD to admission-stable batching.",
    )
    p.add_argument(
        "--pressure-aimd-min-extra-gain",
        type=float,
        default=0.85,
        help="Minimum batch-level marginal gain required before SpecSched-PressureAIMD releases extra drafts.",
    )
    p.add_argument(
        "--speculation-ok-margin",
        type=float,
        default=0.0,
        help="Calibration throughput margin required before SpecSched-Safe uses drafting.",
    )
    p.add_argument("--cache-replay-chunk-tokens", type=int, default=32)
    p.add_argument("--log-every-steps", type=int, default=0)
    p.add_argument("--kv-blocks", type=int, default=512)
    p.add_argument("--block-tokens", type=int, default=16)
    p.add_argument(
        "--arrival-model",
        default="closed_loop",
        choices=["closed_loop", "poisson", "bursty", "trace"],
        help="Request arrival process for benchmark replay.",
    )
    p.add_argument(
        "--arrival-rate-rps",
        type=float,
        default=0.0,
        help="Poisson arrival rate in requests/s when --arrival-model=poisson.",
    )
    p.add_argument(
        "--arrival-trace",
        default="",
        help="JSON/JSONL/CSV arrival trace used when --arrival-model=trace.",
    )
    p.add_argument(
        "--burst-size",
        type=int,
        default=4,
        help="Requests released per burst when --arrival-model=bursty.",
    )
    p.add_argument(
        "--burst-period-s",
        type=float,
        default=1.0,
        help="Seconds between burst releases when --arrival-model=bursty.",
    )
    p.add_argument(
        "--tier-names",
        default="",
        help="Comma-separated tenant tiers for multi-tier SLO smoke runs.",
    )
    p.add_argument(
        "--tier-weights",
        default="",
        help="Comma-separated tier request weights matching --tier-names.",
    )
    p.add_argument(
        "--tier-tpot-mults",
        default="",
        help="Comma-separated TPOT SLO multipliers per tier after adaptive calibration.",
    )
    p.add_argument(
        "--tier-ttft-mults",
        default="",
        help="Comma-separated TTFT SLO multipliers per tier after adaptive calibration.",
    )
    p.add_argument("--slo", default="medium", choices=["off", "tight", "medium", "loose", "relaxed", "adaptive"])
    p.add_argument(
        "--adaptive-slo-tpot-mult",
        type=float,
        default=1.5,
        help="For --slo adaptive, TPOT SLO = this multiplier * NoSpec median TPOT for each N.",
    )
    p.add_argument(
        "--adaptive-slo-ttft-mult",
        type=float,
        default=2.0,
        help="For --slo adaptive, TTFT SLO = this multiplier * NoSpec median TTFT for each N.",
    )
    p.add_argument(
        "--adaptive-slo-scope",
        default="bucket",
        choices=["global", "bucket"],
        help="For --slo adaptive, compute one threshold per N or per (N,bucket).",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable strict PyTorch deterministic kernels. This slows local draft performance.",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = not args.deterministic
    torch.backends.cudnn.deterministic = bool(args.deterministic)
    torch.use_deterministic_algorithms(bool(args.deterministic), warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    ns = parse_int_list(args.Ns)
    fixed_ks = parse_int_list(args.fixed_ks)
    output_lengths = parse_int_list(args.output_lengths) if args.output_lengths else []
    tier_names = [s.strip() for s in args.tier_names.split(",") if s.strip()]
    tier_weights = parse_csv_floats(args.tier_weights)
    tier_tpot_vals = parse_csv_floats(args.tier_tpot_mults)
    tier_ttft_vals = parse_csv_floats(args.tier_ttft_mults)
    tier_tpot_mult = {
        name: (tier_tpot_vals[i] if i < len(tier_tpot_vals) else 1.0)
        for i, name in enumerate(tier_names)
    }
    tier_ttft_mult = {
        name: (tier_ttft_vals[i] if i < len(tier_ttft_vals) else 1.0)
        for i, name in enumerate(tier_names)
    }
    tier_names_text = ",".join(tier_names)
    tier_weights_text = ",".join(str(x) for x in tier_weights)
    tier_tpot_mults_text = ",".join(str(tier_tpot_mult.get(name, 1.0)) for name in tier_names)
    tier_ttft_mults_text = ",".join(str(tier_ttft_mult.get(name, 1.0)) for name in tier_names)
    candidate_ks = sorted(set([0] + fixed_ks + [args.kmax]))
    policies_raw = [s.strip() for s in args.policies.split(",") if s.strip()]
    needs_draft = any(pol != "NoSpec" for pol in policies_raw)

    target_tok, target, target_path = load_model(args.target, "target")
    warmup(target)
    draft_tok = draft = draft_path = None
    if needs_draft:
        if args.draft_backend == "edge":
            hosts = [h.strip() for h in args.edge_hosts.split(",") if h.strip()]
            draft = MultiEdgeDraftClient(
                remote_model=args.draft_remote,
                hosts=hosts,
                user=args.edge_user,
                password=args.edge_password,
                request_timeout_s=args.edge_request_timeout_s,
            )
            draft_path = f"edge:{args.draft_remote}@{','.join(hosts)}"
        else:
            draft_tok, draft, draft_path = load_model(args.draft, "draft")
            if target_tok.vocab_size != draft_tok.vocab_size:
                raise RuntimeError(
                    f"vocab mismatch: target={target_tok.vocab_size} draft={draft_tok.vocab_size}"
                )
            warmup(draft)

    max_requests = max(max(ns), args.requests_per_N)
    total_needed = args.calib_requests + max_requests
    prompts = load_prompts(args.trace, target_tok, total_needed, args.prompt_hard_max, args.seed)
    calib_prompts = prompts[: args.calib_requests]
    bench_prompts = prompts[args.calib_requests :]
    calib_max_new_tokens = int(args.calib_max_new_tokens) if args.calib_max_new_tokens else args.max_new_tokens

    try:
        bucket_k, calib_rows = calibrate(
            target, draft if needs_draft else None,
            calib_prompts, calib_max_new_tokens, candidate_ks, args.calib_out,
        )
        specsched_hints = specsched_calib_hints(calib_rows, fixed_ks)
        detector = speculation_ok_detector(calib_rows, args.kmax, args.speculation_ok_margin)
        specsched_hints.update({
            "detector_speculation_ok": int(bool(detector["speculation_ok"])),
            "detector_decision": detector["decision"],
            "detector_nospec_tput": detector["nospec_tput"],
            "detector_best_spec_tput": detector["best_spec_tput"],
            "detector_best_spec_k": detector["best_spec_k"],
            "detector_tput_ratio": detector["tput_ratio"],
            "detector_margin": detector["margin"],
            "detector_alpha_hat": detector["alpha_hat"],
            "detector_analytic_break_even_ratio": detector["analytic_break_even_ratio"],
            "backpressure_pressure_threshold": args.backpressure_pressure_threshold,
            "backpressure_delay_threshold_s": args.backpressure_delay_threshold_s,
            "backpressure_cost": args.backpressure_cost,
            "backpressure_alpha_floor": args.backpressure_alpha_floor,
            "backpressure_urgency_cap": args.backpressure_urgency_cap,
            "backpressure_opportunistic_k": args.backpressure_opportunistic_k,
            "backpressure_opportunistic_alpha": args.backpressure_opportunistic_alpha,
            "backpressure_opportunistic_pressure": args.backpressure_opportunistic_pressure,
            "backpressure_opportunistic_delay_ratio": args.backpressure_opportunistic_delay_ratio,
            "backpressure_opportunistic_kv_pressure": args.backpressure_opportunistic_kv_pressure,
            "backpressure_fastpath_kv_pressure": args.backpressure_fastpath_kv_pressure,
            "pressure_aimd_pressure_threshold": args.pressure_aimd_pressure_threshold,
            "pressure_aimd_delay_threshold_s": args.pressure_aimd_delay_threshold_s,
            "pressure_aimd_add": args.pressure_aimd_add,
            "pressure_aimd_mult": args.pressure_aimd_mult,
            "pressure_aimd_min": args.pressure_aimd_min,
            "pressure_aimd_max": args.pressure_aimd_max,
            "pressure_aimd_recovery_pressure": args.pressure_aimd_recovery_pressure,
            "pressure_aimd_recovery_delay_ratio": args.pressure_aimd_recovery_delay_ratio,
            "pressure_aimd_delay_gain": args.pressure_aimd_delay_gain,
            "pressure_aimd_cost": args.pressure_aimd_cost,
            "pressure_aimd_alpha_floor": args.pressure_aimd_alpha_floor,
            "pressure_aimd_extra_budget_frac": args.pressure_aimd_extra_budget_frac,
            "pressure_aimd_stable_pressure": args.pressure_aimd_stable_pressure,
            "pressure_aimd_stable_delay_ratio": args.pressure_aimd_stable_delay_ratio,
            "pressure_aimd_active_pressure": args.pressure_aimd_active_pressure,
            "pressure_aimd_min_extra_gain": args.pressure_aimd_min_extra_gain,
            "speculation_ok_margin": args.speculation_ok_margin,
        })
        ref_ttft, ref_tpot = reference_ttft_tpot(calib_rows)
        base_slo = None
        if args.slo != "adaptive":
            base_slo = make_slo(args.slo, ref_ttft, ref_tpot)
            print(f"[slo] {base_slo}", flush=True)
        else:
            print(
                f"[slo] adaptive tpot_mult={args.adaptive_slo_tpot_mult} "
                f"ttft_mult={args.adaptive_slo_ttft_mult}",
                flush=True,
            )

        expanded: List[str] = []
        for name in policies_raw:
            if name == "FixedK":
                expanded.extend([f"FixedK-{k}" for k in fixed_ks])
            else:
                expanded.append(name)

        summary_rows: List[dict] = []
        request_rows: List[dict] = []
        threshold_rows: List[dict] = []
        nospec_pilot_rows: List[dict] = []
        for n in ns:
            request_count = n if args.arrival_model == "closed_loop" else max(n, args.requests_per_N)
            cell_prompts = bench_prompts[:request_count]
            cell_max_new = output_lengths_for(cell_prompts, args.max_new_tokens, output_lengths)
            cell_tiers = tiers_for_requests(request_count, tier_names, tier_weights)
            if args.slo == "adaptive":
                print(f"[slo] pilot NoSpec N={n} for adaptive threshold", flush=True)
                pilot_slo = SLOConfig("off", None, None)
                _, pilot_req_rows = run_cell(
                    policy_name="NoSpec",
                    n_clients=n,
                    request_count=request_count,
                    prompts=cell_prompts,
                    target=target,
                    draft=draft,
                    target_path=target_path,
                    draft_path=draft_path,
                    max_new_tokens=cell_max_new,
                    fixed_ks=fixed_ks,
                    bucket_k=bucket_k,
                    kmax=args.kmax,
                    specsched_hints=specsched_hints,
                    kv_total_blocks=args.kv_blocks,
                    block_tokens=args.block_tokens,
                    slo=pilot_slo,
                    slo_by_bucket=None,
                    workload_slo_ttft_ms=None,
                    workload_slo_tpot_ms=None,
                    max_batch=args.max_batch,
                    max_admit_per_step=args.max_admit_per_step,
                    cache_replay_chunk_tokens=args.cache_replay_chunk_tokens,
                    log_every_steps=args.log_every_steps,
                    arrival_model=args.arrival_model,
                    arrival_rate_rps=args.arrival_rate_rps,
                    arrival_trace=args.arrival_trace,
                    burst_size=args.burst_size,
                    burst_period_s=args.burst_period_s,
                    tiers=cell_tiers,
                    tier_names_text=tier_names_text,
                    tier_weights_text=tier_weights_text,
                    tier_tpot_mults_text=tier_tpot_mults_text,
                    tier_ttft_mults_text=tier_ttft_mults_text,
                    tier_tpot_mult=tier_tpot_mult,
                    tier_ttft_mult=tier_ttft_mult,
                    seed=args.seed,
                    deterministic=bool(args.deterministic),
                    pressure_aimd_min_extra_gain=args.pressure_aimd_min_extra_gain,
                )
                for row in pilot_req_rows:
                    copied = dict(row)
                    copied["pilot"] = "NoSpec"
                    nospec_pilot_rows.append(copied)
                median_tpot = statistics.median(float(r["tpot_ms"]) for r in pilot_req_rows)
                median_ttft = statistics.median(float(r["ttft_ms"]) for r in pilot_req_rows)
                if args.adaptive_slo_scope == "bucket":
                    slo_by_bucket = bucket_thresholds(
                        n,
                        pilot_req_rows,
                        args.adaptive_slo_tpot_mult,
                        args.adaptive_slo_ttft_mult,
                    )
                    threshold_rows.extend(
                        threshold_rows_for_requests(
                            n,
                            args.adaptive_slo_tpot_mult,
                            pilot_req_rows,
                            slo_by_bucket,
                            tier_tpot_mult,
                            tier_ttft_mult,
                        )
                    )
                    slo = SLOConfig(
                        name=f"adaptive-bucket-{args.adaptive_slo_tpot_mult:g}x",
                        ttft_ms=median_ttft * args.adaptive_slo_ttft_mult,
                        tpot_ms=median_tpot * args.adaptive_slo_tpot_mult,
                    )
                else:
                    slo_by_bucket = None
                    slo = SLOConfig(
                        name=f"adaptive-{args.adaptive_slo_tpot_mult:g}x",
                        ttft_ms=median_ttft * args.adaptive_slo_ttft_mult,
                        tpot_ms=median_tpot * args.adaptive_slo_tpot_mult,
                    )
                    threshold_rows.extend([
                        {
                            "request_id": row["req_id"],
                            "class": row["bucket"],
                            "tier": row.get("tier", "default"),
                            "N": n,
                            "median_tpot_nospec": median_tpot,
                            "median_ttft_nospec": median_ttft,
                            "beta": args.adaptive_slo_tpot_mult,
                            "tier_tpot_mult": float(tier_tpot_mult.get(str(row.get("tier", "default")), 1.0)),
                            "tier_ttft_mult": float(tier_ttft_mult.get(str(row.get("tier", "default")), 1.0)),
                            "tau_slo": slo.tpot_ms * float(tier_tpot_mult.get(str(row.get("tier", "default")), 1.0)),
                            "tau_ttft": slo.ttft_ms * float(tier_ttft_mult.get(str(row.get("tier", "default")), 1.0)),
                            "unit": "ms/token",
                        }
                        for row in pilot_req_rows
                    ])
                print(
                    f"[slo] N={n} median_ttft={median_ttft:.1f}ms "
                    f"median_tpot={median_tpot:.1f}ms -> {slo}",
                    flush=True,
                )
            else:
                slo = base_slo
                slo_by_bucket = None
            if args.arrival_model in {"bursty", "trace"}:
                run_slo_ttft, run_slo_tpot = None, None
            else:
                run_slo_ttft, run_slo_tpot = apply_tier_slo(
                    cell_prompts,
                    cell_tiers,
                    {k: v.ttft_ms for k, v in slo_by_bucket.items()} if slo_by_bucket else slo.ttft_ms,
                    {k: v.tpot_ms for k, v in slo_by_bucket.items()} if slo_by_bucket else slo.tpot_ms,
                    tier_tpot_mult,
                    tier_ttft_mult,
                )
            for name in expanded:
                print(
                    f"[bench] policy={name} N={n} requests={request_count} "
                    f"max_new={args.max_new_tokens} kv_blocks={args.kv_blocks}",
                    flush=True,
                )
                summary, req_rows = run_cell(
                    policy_name=name,
                    n_clients=n,
                    request_count=request_count,
                    prompts=cell_prompts,
                    target=target,
                    draft=draft,
                    target_path=target_path,
                    draft_path=draft_path,
                    max_new_tokens=cell_max_new,
                    fixed_ks=fixed_ks,
                    bucket_k=bucket_k,
                    kmax=args.kmax,
                    specsched_hints=specsched_hints,
                    kv_total_blocks=args.kv_blocks,
                    block_tokens=args.block_tokens,
                    slo=slo,
                    slo_by_bucket=slo_by_bucket,
                    workload_slo_ttft_ms=run_slo_ttft,
                    workload_slo_tpot_ms=run_slo_tpot,
                    max_batch=args.max_batch,
                    max_admit_per_step=args.max_admit_per_step,
                    cache_replay_chunk_tokens=args.cache_replay_chunk_tokens,
                    log_every_steps=args.log_every_steps,
                    arrival_model=args.arrival_model,
                    arrival_rate_rps=args.arrival_rate_rps,
                    arrival_trace=args.arrival_trace,
                    burst_size=args.burst_size,
                    burst_period_s=args.burst_period_s,
                    tiers=cell_tiers,
                    tier_names_text=tier_names_text,
                    tier_weights_text=tier_weights_text,
                    tier_tpot_mults_text=tier_tpot_mults_text,
                    tier_ttft_mults_text=tier_ttft_mults_text,
                    tier_tpot_mult=tier_tpot_mult,
                    tier_ttft_mult=tier_ttft_mult,
                    seed=args.seed,
                    deterministic=bool(args.deterministic),
                    pressure_aimd_min_extra_gain=args.pressure_aimd_min_extra_gain,
                )
                summary_rows.append(summary)
                request_rows.extend(req_rows)
                # Persist after every completed cell. Long real runs on 24GB GPUs can
                # still hit OOM at the largest N; keep all completed cells auditable.
                write_csv(args.out, summary_rows)
                write_csv(args.request_out, request_rows)
                if args.slo_thresholds_out:
                    write_csv(args.slo_thresholds_out, threshold_rows)
                if args.nospec_request_out:
                    write_csv(args.nospec_request_out, nospec_pilot_rows)
                print(
                    f"[bench] {name} N={n}: tput={summary['throughput_tok_s']:.2f} tok/s "
                    f"goodput={summary['goodput_tok_s']:.2f} tok/s "
                    f"attain={summary['slo_attainment']:.2f} "
                    f"p99_tpot={summary['p99_tpot_ms']:.1f}ms "
                    f"accept={summary['acceptance_rate']:.3f} "
                    f"tgt/tok={summary['target_forwards_per_token']:.3f} "
                    f"evict={summary['kv_evictions']}",
                    flush=True,
                )

        write_csv(args.out, summary_rows)
        write_csv(args.request_out, request_rows)
        if args.slo_thresholds_out:
            write_csv(args.slo_thresholds_out, threshold_rows)
        if args.nospec_request_out:
            write_csv(args.nospec_request_out, nospec_pilot_rows)
        print(f"[bench] wrote {args.out}", flush=True)
        print(f"[bench] wrote {args.request_out}", flush=True)
        if args.slo_thresholds_out:
            print(f"[bench] wrote {args.slo_thresholds_out}", flush=True)
        if args.nospec_request_out:
            print(f"[bench] wrote {args.nospec_request_out}", flush=True)
    finally:
        if needs_draft and draft is not None and hasattr(draft, "close"):
            draft.close()
        del target
        if draft is not None and not hasattr(draft, "close"):
            del draft
        torch.cuda.empty_cache()


def write_csv(path: str, rows: List[dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


if __name__ == "__main__":
    main()
