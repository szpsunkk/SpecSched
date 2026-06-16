"""ShareGPT-derived workload with explicit length bucketing and arrival models.

The previous experiment hard-truncated every prompt to a single length, which
collapsed all requests into one IndepOpt bucket and hid the cross-bucket
behavior the paper wants to study. This module instead samples a balanced mix
of short / medium / long prompts and exposes deterministic closed-loop arrivals
plus Poisson / bursty open-loop arrivals.
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Union

import torch


LENGTH_BUCKETS = (
    ("short", 16, 64),
    ("medium", 64, 192),
    ("long", 192, 512),
)


@dataclass
class Request:
    req_id: int
    prompt_ids: torch.Tensor  # 1-D cpu long tensor
    bucket: str
    arrival_s: float
    max_new_tokens: int
    tier: str = "default"
    slo_ttft_ms: Optional[float] = None
    slo_tpot_ms: Optional[float] = None
    phase: str = ""
    source_id: str = ""


@dataclass
class Workload:
    requests: List[Request]
    arrival_model: str
    rate_rps: float = 0.0
    seed: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class TraceArrival:
    arrival_s: float
    max_new_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    prompt_bucket: Optional[str] = None
    tier: Optional[str] = None
    phase: str = ""
    source_id: str = ""


def _bucket_of(length: int) -> str:
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= length < hi:
            return name
    return "long"


def _tier_at(tiers: Optional[Sequence[str]], i: int) -> str:
    if not tiers:
        return "default"
    return str(tiers[i])


def _pick_slo(value, bucket: str, i: int):
    if isinstance(value, Mapping):
        return value.get(bucket)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[i]
    return value


def _tokenize_all(rows: Sequence[dict], tokenizer, hard_max: int) -> List[torch.Tensor]:
    out = []
    for row in rows:
        text = row.get("prompt", "")
        if not text:
            continue
        ids = tokenizer(
            text,
            truncation=True,
            max_length=hard_max,
            return_tensors="pt",
        ).input_ids[0].to(dtype=torch.long).cpu()
        if ids.numel() >= 4:
            out.append(ids)
    return out


def _balanced_sample(prompts: List[torch.Tensor], count: int, rng: random.Random) -> List[torch.Tensor]:
    by_bucket = {name: [] for name, _, _ in LENGTH_BUCKETS}
    for ids in prompts:
        by_bucket[_bucket_of(int(ids.numel()))].append(ids)
    for bucket in by_bucket.values():
        rng.shuffle(bucket)

    per_bucket = math.ceil(count / len(LENGTH_BUCKETS))
    picked: List[torch.Tensor] = []
    for name, _, _ in LENGTH_BUCKETS:
        pool = by_bucket[name]
        if len(pool) >= per_bucket:
            picked.extend(pool[:per_bucket])
        else:
            picked.extend(pool)
    rng.shuffle(picked)

    if len(picked) < count:
        leftover = [p for p in prompts if not any(p is x for x in picked)]
        rng.shuffle(leftover)
        picked.extend(leftover[: count - len(picked)])

    return picked[:count]


def load_prompts(
    trace_path: str,
    tokenizer,
    count: int,
    hard_max: int,
    seed: int,
) -> List[torch.Tensor]:
    with open(trace_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    rng = random.Random(seed)
    rng.shuffle(rows)
    tokenized = _tokenize_all(rows, tokenizer, hard_max)
    if len(tokenized) < count:
        raise RuntimeError(
            f"trace yielded only {len(tokenized)} usable prompts (need {count})"
        )
    return _balanced_sample(tokenized, count, rng)


def make_closed_loop_workload(
    prompts: Sequence[torch.Tensor],
    n_clients: int,
    max_new_tokens: Union[int, Sequence[int]],
    slo_ttft_ms: Optional[Union[float, Mapping[str, float]]],
    slo_tpot_ms: Optional[Union[float, Mapping[str, float]]],
    tiers: Optional[Sequence[str]] = None,
) -> Workload:
    """All N clients are admitted at t=0; the engine handles in-flight scheduling."""
    def pick_output(i: int) -> int:
        if isinstance(max_new_tokens, int):
            return max_new_tokens
        return int(max_new_tokens[i])

    requests = []
    for i, ids in enumerate(prompts):
        bucket = _bucket_of(int(ids.numel()))
        requests.append(
            Request(
                req_id=i,
                prompt_ids=ids,
                bucket=bucket,
                arrival_s=0.0,
                max_new_tokens=pick_output(i),
                tier=_tier_at(tiers, i),
                slo_ttft_ms=_pick_slo(slo_ttft_ms, bucket, i),
                slo_tpot_ms=_pick_slo(slo_tpot_ms, bucket, i),
            )
        )
    return Workload(
        requests=requests,
        arrival_model="closed_loop",
        meta={"n_clients": n_clients},
    )


def make_poisson_workload(
    prompts: Sequence[torch.Tensor],
    rate_rps: float,
    max_new_tokens: Union[int, Sequence[int]],
    slo_ttft_ms: Optional[Union[float, Mapping[str, float]]],
    slo_tpot_ms: Optional[Union[float, Mapping[str, float]]],
    seed: int,
    tiers: Optional[Sequence[str]] = None,
) -> Workload:
    rng = random.Random(seed)
    t = 0.0
    requests = []
    def pick_output(i: int) -> int:
        if isinstance(max_new_tokens, int):
            return max_new_tokens
        return int(max_new_tokens[i])

    for i, ids in enumerate(prompts):
        t += rng.expovariate(rate_rps) if rate_rps > 0 else 0.0
        bucket = _bucket_of(int(ids.numel()))
        requests.append(
            Request(
                req_id=i,
                prompt_ids=ids,
                bucket=bucket,
                arrival_s=t,
                max_new_tokens=pick_output(i),
                tier=_tier_at(tiers, i),
                slo_ttft_ms=_pick_slo(slo_ttft_ms, bucket, i),
                slo_tpot_ms=_pick_slo(slo_tpot_ms, bucket, i),
            )
        )
    return Workload(
        requests=requests,
        arrival_model="poisson",
        rate_rps=rate_rps,
        seed=seed,
    )


def _first_present(row: Mapping[str, object], names: Sequence[str]) -> object:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _as_float(value: object, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field} value {value!r}") from exc


def _as_optional_int(value: object, field: str) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field} value {value!r}") from exc
    return max(1, parsed)


def _load_trace_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("requests", "records", "trace", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"unsupported trace format in {path}")


def load_arrival_trace(trace_path: str, limit: Optional[int] = None) -> list[TraceArrival]:
    """Load production-style arrival/output metadata from JSON, JSONL, or CSV.

    Supported arrival fields include `arrival_s`, `arrival_time`,
    `timestamp`, `time`, and `start_time`. Supported output-length fields
    include `max_new_tokens`, `output_length`, `completion_tokens`,
    `response_tokens`, and `generated_tokens`. The function preserves the
    trace ordering after sorting by arrival time and normalizes the first
    arrival to zero.
    """
    path = Path(trace_path).expanduser()
    rows = _load_trace_rows(path)
    arrivals: list[TraceArrival] = []
    for idx, row in enumerate(rows):
        arrival_value = _first_present(
            row,
            ("arrival_s", "arrival_time", "timestamp", "time", "start_time"),
        )
        if arrival_value is None:
            raise ValueError(
                f"trace row {idx} has no arrival field "
                "(expected arrival_s/arrival_time/timestamp/time/start_time)"
            )
        output_value = _first_present(
            row,
            ("max_new_tokens", "output_length", "completion_tokens", "response_tokens", "generated_tokens"),
        )
        prompt_value = _first_present(row, ("prompt_tokens", "prompt_length", "input_tokens"))
        bucket_value = _first_present(row, ("prompt_bucket", "bucket", "length_bucket"))
        tier_value = _first_present(row, ("tier", "slo_tier", "tenant_tier"))
        phase_value = _first_present(row, ("phase", "segment", "workload_phase"))
        source_value = _first_present(row, ("id", "request_id", "conversation_id", "session_id"))
        arrivals.append(
            TraceArrival(
                arrival_s=_as_float(arrival_value, "arrival"),
                max_new_tokens=_as_optional_int(output_value, "output length"),
                prompt_tokens=_as_optional_int(prompt_value, "prompt length"),
                prompt_bucket=None if bucket_value is None else str(bucket_value),
                tier=None if tier_value is None else str(tier_value),
                phase="" if phase_value is None else str(phase_value),
                source_id="" if source_value is None else str(source_value),
            )
        )
    arrivals.sort(key=lambda item: item.arrival_s)
    if limit is not None:
        arrivals = arrivals[:limit]
    if not arrivals:
        raise ValueError(f"trace {trace_path} has no usable arrivals")
    base = arrivals[0].arrival_s
    return [
        TraceArrival(
            arrival_s=max(0.0, item.arrival_s - base),
            max_new_tokens=item.max_new_tokens,
            prompt_tokens=item.prompt_tokens,
            prompt_bucket=item.prompt_bucket,
            tier=item.tier,
            phase=item.phase,
            source_id=item.source_id,
        )
        for item in arrivals
    ]


def _prompt_for_trace(
    pools: Mapping[str, list[torch.Tensor]],
    fallback: list[torch.Tensor],
    traced: TraceArrival,
    rng: random.Random,
) -> torch.Tensor:
    bucket = (traced.prompt_bucket or "").strip()
    if bucket in pools and pools[bucket]:
        return pools[bucket].pop()
    if fallback:
        return fallback.pop()
    nonempty = [pool for pool in pools.values() if pool]
    if nonempty:
        return rng.choice(nonempty).pop()
    raise RuntimeError("no prompts remain for trace replay")


def make_trace_replay_workload(
    prompts: Sequence[torch.Tensor],
    trace_path: str,
    max_new_tokens: Union[int, Sequence[int]],
    slo_ttft_ms: Optional[Union[float, Mapping[str, float]]],
    slo_tpot_ms: Optional[Union[float, Mapping[str, float]]],
    seed: int,
    tiers: Optional[Sequence[str]] = None,
) -> Workload:
    arrivals = load_arrival_trace(trace_path, limit=len(prompts))
    if len(arrivals) < len(prompts):
        raise RuntimeError(
            f"arrival trace yielded only {len(arrivals)} rows (need {len(prompts)})"
        )

    rng = random.Random(seed)
    prompt_pool = list(prompts)
    rng.shuffle(prompt_pool)
    pools = {name: [] for name, _, _ in LENGTH_BUCKETS}
    for ids in prompt_pool:
        pools[_bucket_of(int(ids.numel()))].append(ids)
    for pool in pools.values():
        rng.shuffle(pool)
    fallback = list(prompt_pool)
    rng.shuffle(fallback)

    def pick_output(i: int, traced: TraceArrival) -> int:
        if traced.max_new_tokens is not None:
            return traced.max_new_tokens
        if isinstance(max_new_tokens, int):
            return max_new_tokens
        return int(max_new_tokens[i])

    requests = []
    for i, traced in enumerate(arrivals[: len(prompt_pool)]):
        ids = _prompt_for_trace(pools, fallback, traced, rng)
        bucket = _bucket_of(int(ids.numel()))
        requests.append(
            Request(
                req_id=i,
                prompt_ids=ids,
                bucket=bucket,
                arrival_s=traced.arrival_s,
                max_new_tokens=pick_output(i, traced),
                tier=traced.tier or _tier_at(tiers, i),
                slo_ttft_ms=_pick_slo(slo_ttft_ms, bucket, i),
                slo_tpot_ms=_pick_slo(slo_tpot_ms, bucket, i),
                phase=traced.phase,
                source_id=traced.source_id,
            )
        )
    duration_s = requests[-1].arrival_s - requests[0].arrival_s if len(requests) > 1 else 0.0
    rate_rps = (len(requests) - 1) / duration_s if duration_s > 0 else 0.0
    return Workload(
        requests=requests,
        arrival_model="trace",
        rate_rps=rate_rps,
        seed=seed,
        meta={
            "trace_path": str(Path(trace_path).expanduser()),
            "trace_rows": len(arrivals),
            "duration_s": duration_s,
        },
    )


def make_bursty_workload(
    prompts: Sequence[torch.Tensor],
    burst_size: int,
    burst_period_s: float,
    max_new_tokens: Union[int, Sequence[int]],
    slo_ttft_ms: Optional[Union[float, Mapping[str, float]]],
    slo_tpot_ms: Optional[Union[float, Mapping[str, float]]],
    seed: int,
    tiers: Optional[Sequence[str]] = None,
) -> Workload:
    rng = random.Random(seed)
    prompt_pool = list(prompts)
    rng.shuffle(prompt_pool)

    def pick_output(i: int) -> int:
        if isinstance(max_new_tokens, int):
            return max_new_tokens
        return int(max_new_tokens[i])

    requests = []
    for i, ids in enumerate(prompt_pool):
        burst_idx = i // max(burst_size, 1)
        bucket = _bucket_of(int(ids.numel()))
        requests.append(
            Request(
                req_id=i,
                prompt_ids=ids,
                bucket=bucket,
                arrival_s=burst_idx * burst_period_s,
                max_new_tokens=pick_output(i),
                tier=_tier_at(tiers, i),
                slo_ttft_ms=_pick_slo(slo_ttft_ms, bucket, i),
                slo_tpot_ms=_pick_slo(slo_tpot_ms, bucket, i),
            )
        )
    return Workload(
        requests=requests,
        arrival_model="bursty",
        seed=seed,
        meta={"burst_size": burst_size, "burst_period_s": burst_period_s},
    )


def bucket_of(length: int) -> str:
    return _bucket_of(length)


def bucket_names() -> Iterable[str]:
    return [name for name, _, _ in LENGTH_BUCKETS]
