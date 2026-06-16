"""Aggregation: per-request records -> summary row with SLO goodput."""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict
from typing import Iterable, List, Optional, Sequence

from core.engine import RequestRecord
from core.slo import SLOConfig, attains


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else float("nan")


def _record_attains(record: RequestRecord, fallback: SLOConfig) -> bool:
    if record.slo_ttft_ms is not None or record.slo_tpot_ms is not None:
        return attains(
            record.ttft_ms,
            record.tpot_ms,
            SLOConfig("per-request", record.slo_ttft_ms, record.slo_tpot_ms),
        )
    return attains(record.ttft_ms, record.tpot_ms, fallback)


def _verifier_capacity(extras: Optional[dict]) -> int:
    if not extras:
        return 1
    max_batch = max(1, int(extras.get("max_batch", 1)))
    kmax = max(0, int(extras.get("kmax", 0)))
    return max(1, max_batch * (kmax + 1))


def _verifier_rho(n_clients: int, mean_selected_k: float, extras: Optional[dict]) -> float:
    return n_clients * mean_selected_k / _verifier_capacity(extras)


def summarize(
    policy: str,
    n_clients: int,
    records: List[RequestRecord],
    wall_s: float,
    kv_stats: dict,
    slo: SLOConfig,
    target_model: str,
    draft_model: Optional[str],
    arrival_model: str,
    rate_rps: float,
    extras: Optional[dict] = None,
) -> dict:
    tokens = sum(r.generated for r in records)
    lat = [r.e2e_latency_s for r in records]
    tpot = [r.tpot_ms for r in records]
    ttft = [r.ttft_ms for r in records]
    queue = [r.queue_s for r in records]
    drafted = sum(r.drafted for r in records)
    accepted = sum(r.accepted for r in records)
    target_forwards = sum(r.target_forwards for r in records)
    draft_forwards = sum(r.draft_forwards for r in records)
    reprefills = sum(r.reprefills for r in records)
    service_target_ms = sum(r.service_target_ms for r in records)
    service_draft_ms = sum(r.service_draft_ms for r in records)
    service_other_ms = sum(r.service_other_ms for r in records)
    service_total_ms = service_target_ms + service_draft_ms + service_other_ms

    attain_flags = [_record_attains(r, slo) for r in records]
    good_tokens = sum(r.generated for r, ok in zip(records, attain_flags) if ok)
    attainment = (sum(1 for ok in attain_flags if ok) / max(1, len(records)))

    row = {
        "policy": policy,
        "N": n_clients,
        "requests": len(records),
        "tokens": tokens,
        "wall_s": wall_s,
        "throughput_tok_s": tokens / wall_s if wall_s > 0 else float("nan"),
        "goodput_tok_s": good_tokens / wall_s if wall_s > 0 else float("nan"),
        "slo_attainment": attainment,
        "slo_name": slo.name,
        "slo_ttft_ms": slo.ttft_ms if slo.enabled else float("nan"),
        "slo_tpot_ms": slo.tpot_ms if slo.enabled else float("nan"),
        "mean_e2e_s": _mean(lat),
        "p50_e2e_s": percentile(lat, 0.5),
        "p95_e2e_s": percentile(lat, 0.95),
        "p99_e2e_s": percentile(lat, 0.99),
        "mean_ttft_ms": _mean(ttft),
        "p95_ttft_ms": percentile(ttft, 0.95),
        "p99_ttft_ms": percentile(ttft, 0.99),
        "mean_tpot_ms": _mean(tpot),
        "p95_tpot_ms": percentile(tpot, 0.95),
        "p99_tpot_ms": percentile(tpot, 0.99),
        "mean_queue_s": _mean(queue),
        "p95_queue_s": percentile(queue, 0.95),
        "acceptance_rate": accepted / drafted if drafted else 0.0,
        "accepted_draft": accepted,
        "drafted": drafted,
        "target_forwards": target_forwards,
        "draft_forwards": draft_forwards,
        "target_forwards_per_token": target_forwards / max(tokens, 1),
        "tokens_per_target_forward": tokens / max(target_forwards, 1),
        "service_target_ms": service_target_ms,
        "service_draft_ms": service_draft_ms,
        "service_other_ms": service_other_ms,
        "service_target_frac": service_target_ms / service_total_ms if service_total_ms > 0 else float("nan"),
        "service_draft_frac": service_draft_ms / service_total_ms if service_total_ms > 0 else float("nan"),
        "service_other_frac": service_other_ms / service_total_ms if service_total_ms > 0 else float("nan"),
        "reprefills": reprefills,
        "reprefills_per_1k_tokens": reprefills * 1000.0 / max(tokens, 1),
        "kv_evictions": kv_stats.get("evictions", 0),
        "kv_evictions_per_1k_tokens": kv_stats.get("evictions", 0) * 1000.0 / max(tokens, 1),
        "kv_blocks_used": kv_stats.get("live_blocks", 0),
        "kv_blocks_total": kv_stats.get("total_blocks", 0),
        "kv_pressure_events": kv_stats.get("pressure_events", 0),
        "kv_max_live_blocks": kv_stats.get("max_live_blocks", 0),
        "kv_mean_live_blocks": kv_stats.get("mean_live_blocks", 0.0),
        "mean_selected_k": kv_stats.get("mean_selected_k", 0.0),
        "selected_count": kv_stats.get("selected_count", 0),
        "scheduler_steps": kv_stats.get("scheduler_steps", 0),
        "scheduler_cpu_total_us": kv_stats.get("scheduler_cpu_total_us", 0.0),
        "scheduler_cpu_mean_us": kv_stats.get("scheduler_cpu_mean_us", 0.0),
        "scheduler_cpu_frac": kv_stats.get("scheduler_cpu_frac", 0.0),
        "scheduler_bookkeeping_mean_us": kv_stats.get("scheduler_bookkeeping_mean_us", 0.0),
        "scheduler_bookkeeping_p95_us": kv_stats.get("scheduler_bookkeeping_p95_us", 0.0),
        "scheduler_bookkeeping_p99_us": kv_stats.get("scheduler_bookkeeping_p99_us", 0.0),
        "scheduler_policy_mean_us": kv_stats.get("scheduler_policy_mean_us", 0.0),
        "scheduler_policy_p95_us": kv_stats.get("scheduler_policy_p95_us", 0.0),
        "scheduler_policy_p99_us": kv_stats.get("scheduler_policy_p99_us", 0.0),
        "scheduler_kv_commit_mean_us": kv_stats.get("scheduler_kv_commit_mean_us", 0.0),
        "scheduler_kv_commit_p95_us": kv_stats.get("scheduler_kv_commit_p95_us", 0.0),
        "scheduler_kv_commit_p99_us": kv_stats.get("scheduler_kv_commit_p99_us", 0.0),
        "scheduler_cleanup_mean_us": kv_stats.get("scheduler_cleanup_mean_us", 0.0),
        "scheduler_cleanup_p95_us": kv_stats.get("scheduler_cleanup_p95_us", 0.0),
        "scheduler_cleanup_p99_us": kv_stats.get("scheduler_cleanup_p99_us", 0.0),
        # Verifier load factor rho_d (ex.md §3.1):
        #     rho_d = N * mean(k_i) / C_d
        # C_d is the verifier capacity per scheduling step measured in tokens:
        # one verify forward processes (k+1) tokens for at most `max_batch`
        # requests in parallel, so C_d = max_batch * (kmax + 1). When rho_d > 1
        # the verifier is the bottleneck; that is the regime Observation 4 wants.
        "verifier_load_rho": _verifier_rho(
            n_clients,
            kv_stats.get("mean_selected_k", 0.0),
            extras,
        ),
        "verifier_capacity_C_d": _verifier_capacity(extras),
        "verifier_utilization_proxy": min(
            1.0,
            _verifier_rho(
                n_clients,
                kv_stats.get("mean_selected_k", 0.0),
                extras,
            ),
        ),
        "arrival_model": arrival_model,
        "arrival_rate_rps": rate_rps,
        "target_model": target_model,
        "draft_model": draft_model or "",
    }
    if extras:
        for k, v in extras.items():
            row[f"x_{k}"] = v
    return row


def record_to_row(rec: RequestRecord, policy: str, n_clients: int) -> dict:
    d = asdict(rec)
    d["policy"] = policy
    d["N"] = n_clients
    return d
