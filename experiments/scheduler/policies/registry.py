"""Scheduling policies for the speculative-decoding engine.

Each policy implements the signature
    fn(active: List[ReqState], step_idx: int, cfg: EngineConfig) -> StepDecision

so the engine can swap them without code changes. Policies:

    nospec       : always k=0
    fixed_k(K)   : every request uses the same K
    indep_opt    : per-request bucket-K from calibration table
    specsched    : the paper's joint draft-length + SLO-aware policy
    specsched_no_batch  : SpecSched, batch=1 (force per-request stepping)
    specsched_no_evict  : SpecSched, KV eviction victim selection disabled
    specsched_no_slo    : SpecSched, SLO virtual queue disabled
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from core.engine import EngineConfig, PolicyFn, StepDecision
from core.speculative import ReqState
from core.workload import bucket_of


def nospec() -> PolicyFn:
    def fn(active: List[ReqState], step_idx: int, cfg: EngineConfig) -> StepDecision:
        sel = [(s.req_id, 0) for s in active[: cfg.max_batch]]
        return StepDecision(selected=sel)
    return fn


def fixed_k(k: int) -> PolicyFn:
    def fn(active: List[ReqState], step_idx: int, cfg: EngineConfig) -> StepDecision:
        sel = [(s.req_id, k) for s in active[: cfg.max_batch]]
        return StepDecision(selected=sel)
    return fn


def indep_opt(bucket_k: Dict[str, int]) -> PolicyFn:
    def fn(active: List[ReqState], step_idx: int, cfg: EngineConfig) -> StepDecision:
        sel = []
        for s in active[: cfg.max_batch]:
            sel.append((s.req_id, int(bucket_k.get(s.bucket, 0))))
        return StepDecision(selected=sel)
    return fn


def tetris_proxy(kmax: int) -> PolicyFn:
    """TETRIS-inspired proxy baseline.

    The TETRIS paper optimizes draft-token selection inside a batch using
    acceptance likelihoods. This lightweight proxy is not a reimplementation of
    the full algorithm; it approximates the serving-side behavior available in
    this engine by selecting the requests with highest observed acceptance and
    assigning larger k to those requests, while homogenizing k for verification
    batching. The name is intentionally suffixed with "Proxy" in result CSVs.
    """
    kmax = max(1, int(kmax))

    def accept_rate(s: ReqState) -> float:
        if s.drafted_total <= 0:
            return 0.5
        return s.accepted_total / max(1, s.drafted_total)

    def fn(active: List[ReqState], step_idx: int, cfg: EngineConfig) -> StepDecision:
        if not active:
            return StepDecision()
        ranked = sorted(active, key=accept_rate, reverse=True)
        batch = ranked[: cfg.max_batch]
        mean_accept = sum(accept_rate(s) for s in batch) / max(1, len(batch))
        k = max(1, min(kmax, round(kmax * mean_accept)))
        return StepDecision(selected=[(s.req_id, k) for s in batch])

    return fn


def specserve_proxy(kmax: int, target_accept: float = 0.6) -> PolicyFn:
    """SpecServe-style adaptive speculation under SLO slack.

    This proxy keeps the engine interface unchanged while adding a recent
    speculation-control baseline. It chooses a per-request draft length from
    the observed acceptance rate and remaining TPOT slack, then lets the engine
    execute the resulting variable-length batch.
    """
    kmax = max(1, int(kmax))
    target_accept = max(0.05, min(0.95, float(target_accept)))

    def accept_rate(s: ReqState) -> float:
        if s.drafted_total <= 0:
            return target_accept
        return max(0.05, min(0.95, s.accepted_total / max(1, s.drafted_total)))

    def slo_cap(s: ReqState) -> int:
        if not s.slo_tpot_ms or s.first_token_s is None:
            return kmax
        generated = max(1, len(s.generated))
        elapsed_ms = max(0.0, s.scheduler_now_s - s.arrival_s) * 1000.0
        current_tpot = elapsed_ms / generated
        slack_ms = float(s.slo_tpot_ms) - current_tpot
        if slack_ms <= 0:
            return 1
        # Use the measured TPOT target as a conservative local time scale.
        return max(1, min(kmax, int(1 + slack_ms / max(1.0, float(s.slo_tpot_ms) / 4.0))))

    def fn(active: List[ReqState], step_idx: int, cfg: EngineConfig) -> StepDecision:
        selected = []
        for s in active[: cfg.max_batch]:
            alpha = accept_rate(s)
            # Leviathan-style optimum increases with acceptance probability.
            gamma_accept = max(1, int(math.ceil(-1.0 / math.log(alpha))))
            remaining = max(0, s.max_new_tokens - len(s.generated))
            k = min(kmax, slo_cap(s), gamma_accept, remaining)
            selected.append((s.req_id, max(0, int(k))))
        return StepDecision(selected=selected)

    return fn


def eagle2_proxy(kmax: int, conf_threshold: float = 0.45) -> PolicyFn:
    """EAGLE-2-style dynamic draft truncation proxy.

    EAGLE-2 truncates a draft tree when the draft-side confidence falls below
    a calibrated threshold. This scheduler-level proxy uses the engine's
    observable per-position acceptance EMA as that confidence signal.
    """
    kmax = max(1, int(kmax))
    conf_threshold = max(0.0, min(1.0, float(conf_threshold)))

    def fn(active: List[ReqState], step_idx: int, cfg: EngineConfig) -> StepDecision:
        selected = []
        for s in active[: cfg.max_batch]:
            remaining = max(0, s.max_new_tokens - len(s.generated))
            if remaining <= 0:
                selected.append((s.req_id, 0))
                continue
            k = 0
            for ell in range(min(kmax, remaining)):
                alpha_ell = s.alpha_pos[ell] if ell < len(s.alpha_pos) else 0.5
                if alpha_ell < conf_threshold:
                    break
                k += 1
            selected.append((s.req_id, max(1, min(kmax, remaining, k))))
        return StepDecision(selected=selected)

    return fn


@dataclass
class SpecSchedParams:
    kmax: int = 7
    kmin: int = 1
    target_accept: float = 0.6
    use_slo: bool = True
    homogenize_k: bool = True  # group requests so the batched-verify path triggers
    bucket_k0: Optional[Dict[str, int]] = None  # initial per-bucket k from calibration
    bucket_accept0: Optional[Dict[str, float]] = None
    epsilon_h: float = 0.25
    epsilon_h_safe: float = 0.05
    r_max: float = 0.25
    rho_prefill_threshold: float = 0.05
    pressure_threshold: float = 0.90
    severe_pressure_threshold: float = 0.96
    high_load_factor: float = 4.0
    k_safe: int = 1
    k_pressure: int = 2
    max_slo_boost: float = 3.0
    enable_overload_hook: bool = True
    use_position_alpha: bool = False
    low_pressure_k_cap: Optional[int] = None
    adaptive_k_guard: bool = False
    marginal_gain_floor: float = 1e-4
    queue_aware: bool = False
    queue_age_boost: float = 0.35
    queue_pressure_threshold: float = 1.25
    queue_k_guard: bool = False
    regime_aware: bool = False
    open_loop_queue_boost: float = 0.20
    open_loop_guard_threshold: float = 1.0
    aimd_controller: bool = False
    aimd_add: float = 0.25
    aimd_mult: float = 0.65
    aimd_pressure_threshold: float = 4.0
    aimd_delay_threshold_s: float = 0.25
    aimd_recovery_threshold: float = 1.5
    pending_delay_boost: float = 0.25
    admission_aware: bool = False
    ready_delay_boost: float = 0.45
    ready_pressure_threshold: float = 2.0
    backpressure_controller: bool = False
    backpressure_pressure_threshold: float = 2.0
    backpressure_delay_threshold_s: float = 0.25
    backpressure_cost: float = 0.45
    backpressure_alpha_floor: float = 0.78
    backpressure_urgency_cap: float = 1.15
    backpressure_opportunistic_k: int = 2
    backpressure_opportunistic_alpha: float = 0.78
    backpressure_opportunistic_pressure: float = 0.25
    backpressure_opportunistic_delay_ratio: float = 0.25
    backpressure_opportunistic_kv_pressure: float = 0.70
    backpressure_fastpath_kv_pressure: float = 0.88
    pressure_aimd_controller: bool = False
    pressure_aimd_pressure_threshold: float = 1.0
    pressure_aimd_delay_threshold_s: float = 0.25
    pressure_aimd_add: float = 0.20
    pressure_aimd_mult: float = 0.70
    pressure_aimd_min: float = 1.0
    pressure_aimd_max: float = 3.0
    pressure_aimd_recovery_pressure: float = 0.25
    pressure_aimd_recovery_delay_ratio: float = 0.25
    pressure_aimd_delay_gain: float = 0.35
    pressure_aimd_cost: float = 0.30
    pressure_aimd_alpha_floor: float = 0.70
    pressure_aimd_extra_budget_frac: float = 0.50
    pressure_aimd_stable_pressure: float = 0.25
    pressure_aimd_stable_delay_ratio: float = 0.25
    pressure_aimd_active_pressure: float = 2.0
    pressure_aimd_min_extra_gain: float = 0.85


def specsched(params: SpecSchedParams) -> PolicyFn:
    """SpecSched core with cache-resident admission and overload guardrails.

    The production policy still keeps the original lightweight engine contract
    (one StepDecision per scheduling step), but avoids the high-load feedback
    loop where SLO urgency repeatedly selects evicted requests and forces
    expensive re-prefills. Candidate selection discounts non-resident requests,
    bounds how many of them can enter one batch, and caps k when recent
    re-prefills or KV pressure indicate overload.
    """

    bucket_prior = dict(params.bucket_k0 or {})

    def _accept_ema(s: ReqState) -> float:
        if s.drafted_total > 0:
            return max(0.05, min(0.95, s.accepted_total / max(1, s.drafted_total)))
        if (params.backpressure_controller or params.pressure_aimd_controller) and params.bucket_accept0:
            prior_alpha = params.bucket_accept0.get(s.bucket)
            if prior_alpha is not None:
                return max(0.05, min(0.95, float(prior_alpha)))
        prior = int(bucket_prior.get(s.bucket, 0) or 0)
        if prior > 0:
            return max(0.05, min(0.95, prior / max(1, params.kmax)))
        return max(0.05, min(0.95, params.target_accept))

    def _resident(s: ReqState, kv_manager) -> bool:
        if kv_manager is None:
            return bool(s.cache_valid)
        return bool(s.cache_valid and kv_manager.cache_resident(s.req_id))

    def _slo_urgency(s: ReqState) -> float:
        if not params.use_slo or not s.slo_tpot_ms or s.first_token_s is None:
            return 1.0
        generated = max(1, len(s.generated))
        elapsed_ms = max(0.0, s.scheduler_now_s - s.arrival_s) * 1000.0
        tpot_so_far = elapsed_ms / generated
        ratio = tpot_so_far / max(1e-6, float(s.slo_tpot_ms))
        # Bound urgency so already-far-late requests cannot monopolize the
        # scheduler and repeatedly trigger re-prefills under pressure.
        return max(0.5, min(params.max_slo_boost, 0.5 + ratio))

    def _queue_age_ratio(s: ReqState) -> float:
        if not params.queue_aware:
            return 0.0
        now = max(0.0, float(s.scheduler_now_s))
        age_ms = max(0.0, now - float(s.arrival_s)) * 1000.0
        if s.first_token_s is None and s.slo_ttft_ms:
            return age_ms / max(1.0, float(s.slo_ttft_ms))
        if s.slo_tpot_ms:
            generated = max(1, len(s.generated))
            return (age_ms / generated) / max(1.0, float(s.slo_tpot_ms))
        return 0.0

    def _queue_boost(s: ReqState) -> float:
        ratio = _queue_age_ratio(s)
        if ratio <= 0.0:
            return 1.0
        return 1.0 + params.queue_age_boost * min(2.0, ratio)

    def _open_loop(cfg: EngineConfig) -> bool:
        return str(getattr(cfg, "arrival_model", "closed_loop")) in ("poisson", "bursty")

    def _queue_pressure(cfg: EngineConfig) -> float:
        active = max(0, int(getattr(cfg, "active_count", 0)))
        pending = max(0, int(getattr(cfg, "pending_count", 0)))
        return (active + pending) / max(1, int(getattr(cfg, "max_batch", 1)))

    def _ready_pressure(cfg: EngineConfig) -> float:
        active = max(0, int(getattr(cfg, "active_count", 0)))
        ready = max(0, int(getattr(cfg, "pre_admit_pending_count", 0)))
        return (active + ready) / max(1, int(getattr(cfg, "max_batch", 1)))

    def _open_loop_pressure(cfg: EngineConfig) -> float:
        post_ready = max(0, int(getattr(cfg, "post_admit_ready_count", 0)))
        if post_ready > 0:
            max_admit = int(getattr(cfg, "max_admit_per_step", 0) or 0)
            return post_ready / max(1, max_admit if max_admit > 0 else int(getattr(cfg, "max_batch", 1)))
        ready = max(0, int(getattr(cfg, "pre_admit_pending_count", 0)))
        max_admit = int(getattr(cfg, "max_admit_per_step", 0) or 0)
        if max_admit > 0:
            # Backpressure should react to requests that remain queued after
            # admission capacity is consumed, not to all active residents.
            return max(0, ready - max_admit) / max(1, max_admit)
        pending = max(0, int(getattr(cfg, "pending_count", 0)))
        return pending / max(1, int(getattr(cfg, "max_batch", 1)))

    def _oldest_open_loop_delay_s(cfg: EngineConfig) -> float:
        post_delay = max(0.0, float(getattr(cfg, "oldest_post_admit_ready_s", 0.0)))
        if post_delay > 0.0:
            return post_delay
        max_admit = int(getattr(cfg, "max_admit_per_step", 0) or 0)
        ready = max(0, int(getattr(cfg, "pre_admit_pending_count", 0)))
        if max_admit > 0:
            if ready <= max_admit:
                return 0.0
            return max(0.0, float(getattr(cfg, "oldest_ready_pending_s", 0.0)))
        return max(0.0, float(getattr(cfg, "oldest_pending_s", 0.0)))

    def _pending_delay_ratio(cfg: EngineConfig, s: Optional[ReqState] = None) -> float:
        delay_s = max(
            0.0,
            float(getattr(cfg, "oldest_ready_pending_s", 0.0)),
            float(getattr(cfg, "oldest_pending_s", 0.0)),
        )
        if s is not None and s.slo_ttft_ms:
            scale_s = max(1e-3, float(s.slo_ttft_ms) / 1000.0)
        else:
            scale_s = max(1e-3, params.aimd_delay_threshold_s)
        return delay_s / scale_s

    def _regime_queue_boost(s: ReqState, cfg: EngineConfig) -> float:
        if not params.regime_aware:
            return _queue_boost(s)
        if not _open_loop(cfg):
            return 1.0
        ratio = _queue_age_ratio(s)
        pressure = _queue_pressure(cfg)
        if ratio <= 0.0 or pressure < params.queue_pressure_threshold:
            return 1.0
        gain = params.open_loop_queue_boost * min(2.0, ratio) * min(2.0, pressure / params.queue_pressure_threshold)
        return 1.0 + gain

    def _pending_utility_boost(s: ReqState, cfg: EngineConfig) -> float:
        if params.backpressure_controller:
            return 1.0
        if not (params.aimd_controller or params.admission_aware) or not _open_loop(cfg):
            return 1.0
        pressure = _ready_pressure(cfg) if params.admission_aware else _queue_pressure(cfg)
        delay_ratio = _pending_delay_ratio(cfg, s)
        threshold = params.ready_pressure_threshold if params.admission_aware else params.queue_pressure_threshold
        boost = params.ready_delay_boost if params.admission_aware else params.pending_delay_boost
        if pressure < threshold and delay_ratio < 1.0:
            return 1.0
        return 1.0 + boost * min(2.0, delay_ratio) * min(2.0, pressure / max(1e-6, threshold))

    def _aimd_cap(fn, base_cap: int, cfg: EngineConfig, overload: bool, kv_pressure: float) -> int:
        if not params.aimd_controller or not _open_loop(cfg):
            return base_cap
        control = float(getattr(fn, "_aimd_cap", float(base_cap)))
        pressure = _queue_pressure(cfg)
        delay_s = max(0.0, float(getattr(cfg, "oldest_pending_s", 0.0)))
        congested = (
            overload
            or kv_pressure >= params.pressure_threshold
            or pressure >= params.aimd_pressure_threshold
            or delay_s >= params.aimd_delay_threshold_s
        )
        recovered = (
            pressure <= params.aimd_recovery_threshold
            and delay_s <= 0.5 * params.aimd_delay_threshold_s
            and not overload
            and kv_pressure < params.pressure_threshold
        )
        if congested:
            control = max(float(params.kmin), control * params.aimd_mult)
        elif recovered:
            control = min(float(params.kmax), control + params.aimd_add)
        control = max(float(params.kmin), min(float(base_cap), control))
        setattr(fn, "_aimd_cap", control)
        return max(1, min(base_cap, int(round(control))))

    def _pressure(kv_manager) -> float:
        if kv_manager is None:
            return 0.0
        used = kv_manager.total_blocks - kv_manager.free_blocks()
        return used / max(1, kv_manager.total_blocks)

    def _k_cap(active_count: int, kv_pressure: float, overload: bool, max_batch: int) -> int:
        if overload or kv_pressure >= params.severe_pressure_threshold:
            return max(0, min(params.kmax, params.k_safe))
        if kv_pressure >= params.pressure_threshold:
            return max(1, min(params.kmax, params.k_pressure))
        batch = max(1, int(max_batch))
        low_cap = params.low_pressure_k_cap
        if low_cap is None:
            low_cap = 3
        low_cap = max(1, min(params.kmax, int(low_cap)))
        if active_count <= 2 * batch:
            return low_cap
        if active_count <= 4 * batch:
            return low_cap
        if active_count >= int(params.high_load_factor * batch):
            return max(1, min(params.kmax, params.k_pressure))
        return params.kmax

    def _guarded_cap(s: ReqState, base_cap: int, kv_pressure: float, overload: bool) -> int:
        if not params.adaptive_k_guard or base_cap <= 1:
            return base_cap
        alpha = _accept_ema(s)
        urgency = _slo_urgency(s)
        if overload or kv_pressure >= params.pressure_threshold:
            return min(base_cap, params.k_pressure)
        if alpha < 0.72 and urgency < 1.25:
            return 1
        if alpha < 0.82 and urgency < 1.6:
            return min(base_cap, 2)
        return base_cap

    def _queue_cap(s: ReqState, base_cap: int, cfg: EngineConfig) -> int:
        if not params.queue_aware or not params.queue_k_guard or base_cap <= 1:
            return base_cap
        if params.regime_aware and not _open_loop(cfg):
            return base_cap
        pressure = _queue_pressure(cfg)
        if params.regime_aware and pressure < params.queue_pressure_threshold:
            return base_cap
        ratio = _queue_age_ratio(s)
        alpha = _accept_ema(s)
        threshold = params.open_loop_guard_threshold if params.regime_aware else params.queue_pressure_threshold
        if ratio >= threshold and alpha < 0.80:
            return 1
        if ratio >= threshold * 1.5:
            return min(base_cap, 1)
        return base_cap

    def _backpressure_cap(s: ReqState, base_cap: int, cfg: EngineConfig, kv_pressure: float) -> int:
        if not params.backpressure_controller or not _open_loop(cfg) or base_cap <= 1:
            return base_cap
        pressure = _open_loop_pressure(cfg)
        pressure_ratio = pressure / max(1e-6, params.backpressure_pressure_threshold)
        delay_ratio = _oldest_open_loop_delay_s(cfg) / max(1e-6, params.backpressure_delay_threshold_s)
        alpha = _accept_ema(s)
        cap = base_cap
        if (
            pressure_ratio <= params.backpressure_opportunistic_pressure
            and delay_ratio <= params.backpressure_opportunistic_delay_ratio
            and kv_pressure <= params.backpressure_opportunistic_kv_pressure
            and alpha >= params.backpressure_opportunistic_alpha
        ):
            cap = max(cap, min(params.kmax, int(params.backpressure_opportunistic_k)))
        if pressure_ratio >= 2.0 or delay_ratio >= 2.0:
            return 1
        if (pressure_ratio >= 1.0 or delay_ratio >= 1.0) and alpha < params.backpressure_alpha_floor:
            return 1
        if pressure_ratio >= 1.0 and delay_ratio >= 0.5:
            return min(cap, 2)
        return cap

    def _prefix_accept(s: ReqState, upto: int) -> float:
        if not params.use_position_alpha:
            return _accept_ema(s) ** max(1, upto)
        prod = 1.0
        fallback = _accept_ema(s)
        for ell in range(max(1, upto)):
            alpha = s.alpha_pos[ell] if ell < len(s.alpha_pos) else fallback
            prod *= max(0.05, min(0.95, alpha))
        return prod

    def _marginal_gain(s: ReqState, current_k: int, resident: bool, eps_h: float, overload: bool) -> float:
        cache_factor = 1.0 if resident else eps_h
        penalty = 1.0 + (2.0 if not resident else 0.0) + (2.0 if overload and not resident else 0.0)
        urgency = _slo_urgency(s)
        if params.backpressure_controller:
            urgency = min(urgency, params.backpressure_urgency_cap)
        return urgency * cache_factor * _prefix_accept(s, current_k + 1) / penalty

    def _backpressure_gain(s: ReqState, current_k: int, base_gain: float, cfg: EngineConfig) -> float:
        if not params.backpressure_controller or not _open_loop(cfg):
            return base_gain
        pressure_excess = max(
            0.0,
            _open_loop_pressure(cfg) / max(1e-6, params.backpressure_pressure_threshold) - 1.0,
        )
        delay_ratio = _oldest_open_loop_delay_s(cfg) / max(1e-6, params.backpressure_delay_threshold_s)
        alpha_gap = max(0.0, params.backpressure_alpha_floor - _accept_ema(s)) / max(
            1e-6,
            params.backpressure_alpha_floor,
        )
        token_position = (current_k + 1) / max(1, params.kmax)
        cost = (
            params.backpressure_cost
            * (pressure_excess + delay_ratio)
            * (1.0 + alpha_gap)
            * (1.0 + token_position)
        )
        return base_gain - cost

    def _pressure_aimd_state(fn, base_cap: int, cfg: EngineConfig, overload: bool, kv_pressure: float) -> tuple:
        if not params.pressure_aimd_controller or not _open_loop(cfg):
            return float(base_cap), 0.0, 0.0, 0.0, False, True
        pressure = _open_loop_pressure(cfg)
        delay_ratio = _oldest_open_loop_delay_s(cfg) / max(1e-6, params.pressure_aimd_delay_threshold_s)
        active_pressure = max(0, int(getattr(cfg, "active_count", 0))) / max(1, int(getattr(cfg, "max_batch", 1)))
        control = float(getattr(fn, "_pressure_aimd_cap", min(float(base_cap), params.pressure_aimd_max)))
        congested = (
            overload
            or kv_pressure >= params.pressure_threshold
            or pressure >= params.pressure_aimd_pressure_threshold
            or delay_ratio >= 1.0
            or active_pressure >= params.pressure_aimd_active_pressure
        )
        recovered = (
            pressure <= params.pressure_aimd_recovery_pressure
            and delay_ratio <= params.pressure_aimd_recovery_delay_ratio
            and active_pressure <= 1.0
            and not overload
            and kv_pressure < params.pressure_threshold
        )
        if congested:
            control = max(params.pressure_aimd_min, control * params.pressure_aimd_mult)
        elif recovered:
            control = min(params.pressure_aimd_max, control + params.pressure_aimd_add)
        control = max(params.pressure_aimd_min, min(float(base_cap), control))
        setattr(fn, "_pressure_aimd_cap", control)
        return control, pressure, delay_ratio, active_pressure, congested, recovered

    def _pressure_aimd_extra_budget(
        fn,
        control: float,
        batch_size: int,
        pressure: float,
        delay_ratio: float,
        congested: bool,
        recovered: bool,
        batch_extra_gain: float,
    ) -> int:
        if not params.pressure_aimd_controller:
            return max(0, int(batch_size * max(0.0, control - 1.0)))
        pressure_scale = 1.0 / (1.0 + max(0.0, pressure / max(1e-6, params.pressure_aimd_pressure_threshold)))
        delay_scale = 1.0 / (1.0 + max(0.0, delay_ratio))
        budget_float = (
            max(0.0, control - 1.0)
            * max(1, batch_size)
            * max(0.0, params.pressure_aimd_extra_budget_frac)
            * min(pressure_scale, delay_scale)
        )
        batch = max(1, int(batch_size))
        max_extra_depth = max(0, int(math.ceil(control)) - 1)
        if max_extra_depth <= 0:
            setattr(fn, "_pressure_aimd_credit", 0.0)
            return 0
        credit = float(getattr(fn, "_pressure_aimd_credit", 0.0))
        gain_ok = batch_extra_gain >= params.pressure_aimd_min_extra_gain
        if congested:
            credit = max(0.0, credit * params.pressure_aimd_mult)
        elif recovered and gain_ok:
            credit = min(float(batch * max_extra_depth), credit + max(0.0, budget_float))
        else:
            credit = max(0.0, credit * params.pressure_aimd_mult - max(0.0, budget_float))
        full_layers = min(max_extra_depth, int(math.floor(credit / batch)))
        if full_layers > 0:
            credit -= float(full_layers * batch)
        setattr(fn, "_pressure_aimd_credit", credit)
        return full_layers * batch

    def _pressure_aimd_gain(s: ReqState, current_k: int, base_gain: float, pressure: float, delay_ratio: float) -> float:
        if not params.pressure_aimd_controller:
            return base_gain
        alpha = _accept_ema(s)
        age_ratio = _queue_age_ratio(s)
        delay_credit = 1.0 + params.pressure_aimd_delay_gain * min(2.0, max(delay_ratio, age_ratio))
        alpha_gap = max(0.0, params.pressure_aimd_alpha_floor - alpha) / max(
            1e-6,
            params.pressure_aimd_alpha_floor,
        )
        token_position = (current_k + 1) / max(1, params.kmax)
        pressure_excess = max(0.0, pressure / max(1e-6, params.pressure_aimd_pressure_threshold) - 1.0)
        cost = params.pressure_aimd_cost * (pressure_excess + max(0.0, delay_ratio - 1.0)) * (
            1.0 + alpha_gap + token_position
        )
        return base_gain * delay_credit - cost


    def fn(active: List[ReqState], step_idx: int, cfg: EngineConfig) -> StepDecision:
        if not active:
            return StepDecision()

        kv_manager = getattr(fn, "_kv_manager", None)
        kv_pressure = _pressure(kv_manager)
        rho_prefill = kv_manager.reprefill_rate() if kv_manager is not None else 0.0
        overload = (
            params.enable_overload_hook
            and (rho_prefill > params.rho_prefill_threshold or kv_pressure >= params.severe_pressure_threshold)
        )
        eps_h = params.epsilon_h_safe if overload else params.epsilon_h

        if (
            params.backpressure_controller
            and _open_loop(cfg)
            and kv_pressure >= params.backpressure_fastpath_kv_pressure
        ):
            selected = []
            for s in active[: cfg.max_batch]:
                remaining = max(0, s.max_new_tokens - len(s.generated))
                selected.append((s.req_id, min(params.kmin, remaining)))
            return StepDecision(selected=selected)

        stable_open_loop = False
        if params.pressure_aimd_controller and _open_loop(cfg):
            pressure_now = _open_loop_pressure(cfg)
            delay_now = _oldest_open_loop_delay_s(cfg) / max(1e-6, params.pressure_aimd_delay_threshold_s)
            active_pressure = max(0, int(getattr(cfg, "active_count", 0))) / max(1, int(getattr(cfg, "max_batch", 1)))
            stable_open_loop = (
                pressure_now >= params.pressure_aimd_stable_pressure
                or delay_now >= params.pressure_aimd_stable_delay_ratio
                or active_pressure >= params.pressure_aimd_active_pressure
                or kv_pressure >= params.pressure_threshold
            )

        def pre_score(s: ReqState) -> tuple:
            resident = _resident(s, kv_manager)
            cache_factor = 1.0 if resident else eps_h
            score = _slo_urgency(s) * _regime_queue_boost(s, cfg) * _accept_ema(s) * cache_factor
            return (-score, s.req_id)

        ranked = active[:] if stable_open_loop else sorted(active, key=pre_score)
        has_resident = any(_resident(s, kv_manager) for s in ranked)
        if overload and has_resident:
            reprefill_budget = 0
        else:
            reprefill_budget = max(1, int(params.r_max * max(1, cfg.max_batch)))

        if stable_open_loop:
            candidates = ranked[: cfg.max_batch]
        else:
            candidates: List[ReqState] = []
            remaining_reprefills = reprefill_budget
            for s in ranked:
                if len(candidates) >= cfg.max_batch:
                    break
                resident = _resident(s, kv_manager)
                if not resident:
                    if remaining_reprefills <= 0:
                        continue
                    remaining_reprefills -= 1
                candidates.append(s)
        if not candidates:
            # Avoid deadlock if every live request is currently non-resident.
            candidates = ranked[:1]

        k_cap = _k_cap(len(active), kv_pressure, overload, cfg.max_batch)
        k_cap = _aimd_cap(fn, k_cap, cfg, overload, kv_pressure)
        if params.pressure_aimd_controller and _open_loop(cfg) and not overload and kv_pressure < params.pressure_threshold:
            k_cap = max(k_cap, min(params.kmax, int(math.ceil(params.pressure_aimd_max))))
        (
            aimd_float_cap,
            aimd_pressure,
            aimd_delay_ratio,
            aimd_active_pressure,
            aimd_congested,
            aimd_recovered,
        ) = _pressure_aimd_state(
            fn,
            k_cap,
            cfg,
            overload,
            kv_pressure,
        )
        if params.pressure_aimd_controller:
            k_cap = max(1, min(k_cap, int(math.ceil(aimd_float_cap))))
        ks = {s.req_id: 0 for s in candidates}
        caps = {}
        for s in candidates:
            remaining = max(0, s.max_new_tokens - len(s.generated))
            guarded_cap = _guarded_cap(s, k_cap, kv_pressure, overload)
            guarded_cap = _queue_cap(s, guarded_cap, cfg)
            guarded_cap = _backpressure_cap(s, guarded_cap, cfg, kv_pressure)
            if params.pressure_aimd_controller and _open_loop(cfg):
                alpha = _accept_ema(s)
                if aimd_congested and alpha < params.pressure_aimd_alpha_floor:
                    guarded_cap = min(guarded_cap, params.kmin)
                elif aimd_delay_ratio < params.pressure_aimd_recovery_delay_ratio and aimd_pressure <= params.pressure_aimd_recovery_pressure:
                    guarded_cap = min(guarded_cap, max(params.kmin, int(math.ceil(aimd_float_cap))))
            caps[s.req_id] = max(0, min(guarded_cap, remaining))

        extra_budget = 0
        if params.pressure_aimd_controller:
            extra_gains = []
            for s in candidates:
                current = min(params.kmin, caps[s.req_id])
                if current >= caps[s.req_id]:
                    continue
                resident = _resident(s, kv_manager)
                gain = _marginal_gain(s, current, resident, eps_h, overload)
                gain = _pressure_aimd_gain(s, current, gain, aimd_pressure, aimd_delay_ratio)
                gain *= _pending_utility_boost(s, cfg)
                extra_gains.append(gain)
            batch_extra_gain = sum(extra_gains) / len(extra_gains) if extra_gains else 0.0
            extra_budget = _pressure_aimd_extra_budget(
                fn,
                aimd_float_cap,
                len(candidates),
                aimd_pressure,
                aimd_delay_ratio,
                aimd_congested,
                aimd_recovered,
                batch_extra_gain,
            )
            if stable_open_loop and extra_budget <= 0:
                selected = []
                for s in candidates:
                    remaining = max(0, s.max_new_tokens - len(s.generated))
                    selected.append((s.req_id, min(params.kmin, remaining)))
                return StepDecision(selected=selected)
            token_budget = len(candidates) + extra_budget
        elif params.backpressure_controller:
            token_budget = max(0, sum(caps.values()))
        else:
            token_budget = max(0, len(candidates) * k_cap)
        for s in candidates:
            resident = _resident(s, kv_manager)
            min_k = 0 if (overload and not resident) else min(params.kmin, caps[s.req_id])
            if min_k > 0 and token_budget >= min_k:
                ks[s.req_id] = min_k
                token_budget -= min_k

        if params.pressure_aimd_controller and candidates:
            full_layers = max(0, int(extra_budget) // max(1, len(candidates)))
            for _ in range(full_layers):
                if token_budget < len(candidates):
                    break
                eligible = [s for s in candidates if ks[s.req_id] < caps[s.req_id]]
                if len(eligible) != len(candidates):
                    break
                for s in candidates:
                    ks[s.req_id] += 1
                    token_budget -= 1

        while token_budget > 0:
            best = None
            best_gain = params.marginal_gain_floor
            for s in candidates:
                cur = ks[s.req_id]
                if cur >= caps[s.req_id]:
                    continue
                resident = _resident(s, kv_manager)
                gain = _marginal_gain(s, cur, resident, eps_h, overload)
                gain = _backpressure_gain(s, cur, gain, cfg)
                gain = _pressure_aimd_gain(s, cur, gain, aimd_pressure, aimd_delay_ratio)
                gain *= _pending_utility_boost(s, cfg)
                if gain > best_gain:
                    best = s
                    best_gain = gain
            if best is None:
                break
            ks[best.req_id] += 1
            token_budget -= 1

        selected = [(s.req_id, int(ks[s.req_id])) for s in candidates]
        if params.homogenize_k and len(selected) > 1 and all(k > 0 for _, k in selected):
            sorted_ks = sorted(k for _, k in selected)
            median_index = (len(sorted_ks) - 1) // 2 if (params.backpressure_controller or params.pressure_aimd_controller) else len(sorted_ks) // 2
            median_k = sorted_ks[median_index]
            median_k = min(k_cap, max(params.kmin, median_k))
            median_k = min(median_k, min(caps.values()) if caps else median_k)
            selected = [(rid, median_k) for rid, _ in selected]

        return StepDecision(selected=selected)

    return fn


def specsched_no_batch(params: SpecSchedParams) -> PolicyFn:
    """Ablation: SpecSched without padding-aware batching (max_batch=1)."""
    base = specsched(params)

    def fn(active, step_idx, cfg):
        setattr(base, "_kv_manager", getattr(fn, "_kv_manager", None))
        forced = EngineConfig(
            max_batch=1,
            batch_verify=False,
            use_speculation=cfg.use_speculation,
            use_draft=cfg.use_draft,
            block_tokens=cfg.block_tokens,
            cache_replay_chunk_tokens=cfg.cache_replay_chunk_tokens,
            log_every_steps=cfg.log_every_steps,
            arrival_model=cfg.arrival_model,
            arrival_rate_rps=cfg.arrival_rate_rps,
            pending_count=cfg.pending_count,
            in_flight_count=cfg.in_flight_count,
            active_count=cfg.active_count,
            oldest_pending_s=cfg.oldest_pending_s,
            max_admit_per_step=cfg.max_admit_per_step,
            pre_admit_pending_count=cfg.pre_admit_pending_count,
            oldest_ready_pending_s=cfg.oldest_ready_pending_s,
            post_admit_ready_count=cfg.post_admit_ready_count,
            oldest_post_admit_ready_s=cfg.oldest_post_admit_ready_s,
        )
        return base(active, step_idx, forced)
    return fn


def specsched_no_slo(params: SpecSchedParams) -> PolicyFn:
    """Ablation: SpecSched without SLO virtual queue."""
    no_slo_params = SpecSchedParams(
        kmax=params.kmax,
        kmin=params.kmin,
        target_accept=params.target_accept,
        use_slo=False,
        homogenize_k=params.homogenize_k,
        bucket_k0=params.bucket_k0,
        bucket_accept0=params.bucket_accept0,
        epsilon_h=params.epsilon_h,
        epsilon_h_safe=params.epsilon_h_safe,
        r_max=params.r_max,
        rho_prefill_threshold=params.rho_prefill_threshold,
        pressure_threshold=params.pressure_threshold,
        severe_pressure_threshold=params.severe_pressure_threshold,
        high_load_factor=params.high_load_factor,
        k_safe=params.k_safe,
        k_pressure=params.k_pressure,
        max_slo_boost=params.max_slo_boost,
        enable_overload_hook=params.enable_overload_hook,
        use_position_alpha=params.use_position_alpha,
        low_pressure_k_cap=params.low_pressure_k_cap,
        adaptive_k_guard=params.adaptive_k_guard,
        marginal_gain_floor=params.marginal_gain_floor,
        queue_aware=params.queue_aware,
        queue_age_boost=params.queue_age_boost,
        queue_pressure_threshold=params.queue_pressure_threshold,
        queue_k_guard=params.queue_k_guard,
        regime_aware=params.regime_aware,
        open_loop_queue_boost=params.open_loop_queue_boost,
        open_loop_guard_threshold=params.open_loop_guard_threshold,
        aimd_controller=params.aimd_controller,
        aimd_add=params.aimd_add,
        aimd_mult=params.aimd_mult,
        aimd_pressure_threshold=params.aimd_pressure_threshold,
        aimd_delay_threshold_s=params.aimd_delay_threshold_s,
        aimd_recovery_threshold=params.aimd_recovery_threshold,
        pending_delay_boost=params.pending_delay_boost,
        admission_aware=params.admission_aware,
        ready_delay_boost=params.ready_delay_boost,
        ready_pressure_threshold=params.ready_pressure_threshold,
        backpressure_controller=params.backpressure_controller,
        backpressure_pressure_threshold=params.backpressure_pressure_threshold,
        backpressure_delay_threshold_s=params.backpressure_delay_threshold_s,
        backpressure_cost=params.backpressure_cost,
        backpressure_alpha_floor=params.backpressure_alpha_floor,
        backpressure_urgency_cap=params.backpressure_urgency_cap,
        backpressure_opportunistic_k=params.backpressure_opportunistic_k,
        backpressure_opportunistic_alpha=params.backpressure_opportunistic_alpha,
        backpressure_opportunistic_pressure=params.backpressure_opportunistic_pressure,
        backpressure_opportunistic_delay_ratio=params.backpressure_opportunistic_delay_ratio,
        backpressure_opportunistic_kv_pressure=params.backpressure_opportunistic_kv_pressure,
        backpressure_fastpath_kv_pressure=params.backpressure_fastpath_kv_pressure,
        pressure_aimd_controller=params.pressure_aimd_controller,
        pressure_aimd_pressure_threshold=params.pressure_aimd_pressure_threshold,
        pressure_aimd_delay_threshold_s=params.pressure_aimd_delay_threshold_s,
        pressure_aimd_add=params.pressure_aimd_add,
        pressure_aimd_mult=params.pressure_aimd_mult,
        pressure_aimd_min=params.pressure_aimd_min,
        pressure_aimd_max=params.pressure_aimd_max,
        pressure_aimd_recovery_pressure=params.pressure_aimd_recovery_pressure,
        pressure_aimd_recovery_delay_ratio=params.pressure_aimd_recovery_delay_ratio,
        pressure_aimd_delay_gain=params.pressure_aimd_delay_gain,
        pressure_aimd_cost=params.pressure_aimd_cost,
        pressure_aimd_alpha_floor=params.pressure_aimd_alpha_floor,
        pressure_aimd_extra_budget_frac=params.pressure_aimd_extra_budget_frac,
        pressure_aimd_stable_pressure=params.pressure_aimd_stable_pressure,
        pressure_aimd_stable_delay_ratio=params.pressure_aimd_stable_delay_ratio,
        pressure_aimd_active_pressure=params.pressure_aimd_active_pressure,
        pressure_aimd_min_extra_gain=params.pressure_aimd_min_extra_gain,
    )
    return specsched(no_slo_params)


def specsched_lowload_fixed(params: SpecSchedParams) -> PolicyFn:
    """SpecSched with a low-load FixedK-1 guard.

    The multiseed audit shows the strongest BestFixed competitor is usually
    FixedK-1 in low-pressure regimes. This candidate removes SpecSched's
    ranking/homogenization overhead when the active set is small and KV pressure
    is mild, then falls back to the normal SpecSched policy as load increases.
    """
    base = specsched(params)

    def fn(active: List[ReqState], step_idx: int, cfg: EngineConfig) -> StepDecision:
        kv_manager = getattr(fn, "_kv_manager", None)
        setattr(base, "_kv_manager", kv_manager)
        if not active:
            return StepDecision()
        kv_pressure = 0.0
        if kv_manager is not None:
            used = kv_manager.total_blocks - kv_manager.free_blocks()
            kv_pressure = used / max(1, kv_manager.total_blocks)
        decision = base(active, step_idx, cfg)
        if len(active) <= 4 * max(1, cfg.max_batch) and kv_pressure < params.pressure_threshold:
            by_id = {s.req_id: s for s in active}
            selected = []
            for req_id, _ in decision.selected:
                state = by_id.get(req_id)
                if state is None:
                    continue
                remaining = max(0, state.max_new_tokens - len(state.generated))
                selected.append((req_id, min(1, remaining)))
            return StepDecision(selected=selected)
        return decision

    return fn


def specsched_regime_aware(params: SpecSchedParams) -> PolicyFn:
    """Run base SpecSched for closed-loop loads and queue-aware control for open-loop loads."""
    base_params = SpecSchedParams(
        kmax=params.kmax,
        kmin=params.kmin,
        target_accept=params.target_accept,
        use_slo=params.use_slo,
        homogenize_k=params.homogenize_k,
        bucket_k0=params.bucket_k0,
        bucket_accept0=params.bucket_accept0,
        epsilon_h=params.epsilon_h,
        epsilon_h_safe=params.epsilon_h_safe,
        r_max=params.r_max,
        rho_prefill_threshold=params.rho_prefill_threshold,
        pressure_threshold=params.pressure_threshold,
        severe_pressure_threshold=params.severe_pressure_threshold,
        high_load_factor=params.high_load_factor,
        k_safe=params.k_safe,
        k_pressure=params.k_pressure,
        max_slo_boost=params.max_slo_boost,
        enable_overload_hook=params.enable_overload_hook,
        use_position_alpha=params.use_position_alpha,
        low_pressure_k_cap=params.low_pressure_k_cap,
        adaptive_k_guard=params.adaptive_k_guard,
        marginal_gain_floor=params.marginal_gain_floor,
        aimd_add=params.aimd_add,
        aimd_mult=params.aimd_mult,
        aimd_pressure_threshold=params.aimd_pressure_threshold,
        aimd_delay_threshold_s=params.aimd_delay_threshold_s,
        aimd_recovery_threshold=params.aimd_recovery_threshold,
        pending_delay_boost=params.pending_delay_boost,
        admission_aware=params.admission_aware,
        ready_delay_boost=params.ready_delay_boost,
        ready_pressure_threshold=params.ready_pressure_threshold,
        backpressure_controller=params.backpressure_controller,
        backpressure_pressure_threshold=params.backpressure_pressure_threshold,
        backpressure_delay_threshold_s=params.backpressure_delay_threshold_s,
        backpressure_cost=params.backpressure_cost,
        backpressure_alpha_floor=params.backpressure_alpha_floor,
        backpressure_urgency_cap=params.backpressure_urgency_cap,
        backpressure_opportunistic_k=params.backpressure_opportunistic_k,
        backpressure_opportunistic_alpha=params.backpressure_opportunistic_alpha,
        backpressure_opportunistic_pressure=params.backpressure_opportunistic_pressure,
        backpressure_opportunistic_delay_ratio=params.backpressure_opportunistic_delay_ratio,
        backpressure_opportunistic_kv_pressure=params.backpressure_opportunistic_kv_pressure,
        backpressure_fastpath_kv_pressure=params.backpressure_fastpath_kv_pressure,
        pressure_aimd_controller=params.pressure_aimd_controller,
        pressure_aimd_pressure_threshold=params.pressure_aimd_pressure_threshold,
        pressure_aimd_delay_threshold_s=params.pressure_aimd_delay_threshold_s,
        pressure_aimd_add=params.pressure_aimd_add,
        pressure_aimd_mult=params.pressure_aimd_mult,
        pressure_aimd_min=params.pressure_aimd_min,
        pressure_aimd_max=params.pressure_aimd_max,
        pressure_aimd_recovery_pressure=params.pressure_aimd_recovery_pressure,
        pressure_aimd_recovery_delay_ratio=params.pressure_aimd_recovery_delay_ratio,
        pressure_aimd_delay_gain=params.pressure_aimd_delay_gain,
        pressure_aimd_cost=params.pressure_aimd_cost,
        pressure_aimd_alpha_floor=params.pressure_aimd_alpha_floor,
        pressure_aimd_extra_budget_frac=params.pressure_aimd_extra_budget_frac,
        pressure_aimd_stable_pressure=params.pressure_aimd_stable_pressure,
        pressure_aimd_stable_delay_ratio=params.pressure_aimd_stable_delay_ratio,
        pressure_aimd_active_pressure=params.pressure_aimd_active_pressure,
        pressure_aimd_min_extra_gain=params.pressure_aimd_min_extra_gain,
    )
    open_params = SpecSchedParams(
        kmax=params.kmax,
        kmin=params.kmin,
        target_accept=params.target_accept,
        use_slo=params.use_slo,
        homogenize_k=params.homogenize_k,
        bucket_k0=params.bucket_k0,
        bucket_accept0=params.bucket_accept0,
        epsilon_h=params.epsilon_h,
        epsilon_h_safe=params.epsilon_h_safe,
        r_max=params.r_max,
        rho_prefill_threshold=params.rho_prefill_threshold,
        pressure_threshold=params.pressure_threshold,
        severe_pressure_threshold=params.severe_pressure_threshold,
        high_load_factor=params.high_load_factor,
        k_safe=params.k_safe,
        k_pressure=params.k_pressure,
        max_slo_boost=params.max_slo_boost,
        enable_overload_hook=params.enable_overload_hook,
        use_position_alpha=params.use_position_alpha,
        low_pressure_k_cap=params.low_pressure_k_cap,
        adaptive_k_guard=params.adaptive_k_guard,
        marginal_gain_floor=params.marginal_gain_floor,
        queue_aware=True,
        queue_age_boost=params.queue_age_boost,
        queue_pressure_threshold=params.queue_pressure_threshold,
        queue_k_guard=True,
        regime_aware=True,
        open_loop_queue_boost=params.open_loop_queue_boost,
        open_loop_guard_threshold=params.open_loop_guard_threshold,
        aimd_controller=params.aimd_controller,
        aimd_add=params.aimd_add,
        aimd_mult=params.aimd_mult,
        aimd_pressure_threshold=params.aimd_pressure_threshold,
        aimd_delay_threshold_s=params.aimd_delay_threshold_s,
        aimd_recovery_threshold=params.aimd_recovery_threshold,
        pending_delay_boost=params.pending_delay_boost,
        admission_aware=params.admission_aware,
        ready_delay_boost=params.ready_delay_boost,
        ready_pressure_threshold=params.ready_pressure_threshold,
        backpressure_controller=params.backpressure_controller,
        backpressure_pressure_threshold=params.backpressure_pressure_threshold,
        backpressure_delay_threshold_s=params.backpressure_delay_threshold_s,
        backpressure_cost=params.backpressure_cost,
        backpressure_alpha_floor=params.backpressure_alpha_floor,
        backpressure_urgency_cap=params.backpressure_urgency_cap,
        backpressure_opportunistic_k=params.backpressure_opportunistic_k,
        backpressure_opportunistic_alpha=params.backpressure_opportunistic_alpha,
        backpressure_opportunistic_pressure=params.backpressure_opportunistic_pressure,
        backpressure_opportunistic_delay_ratio=params.backpressure_opportunistic_delay_ratio,
        backpressure_opportunistic_kv_pressure=params.backpressure_opportunistic_kv_pressure,
        backpressure_fastpath_kv_pressure=params.backpressure_fastpath_kv_pressure,
        pressure_aimd_controller=params.pressure_aimd_controller,
        pressure_aimd_pressure_threshold=params.pressure_aimd_pressure_threshold,
        pressure_aimd_delay_threshold_s=params.pressure_aimd_delay_threshold_s,
        pressure_aimd_add=params.pressure_aimd_add,
        pressure_aimd_mult=params.pressure_aimd_mult,
        pressure_aimd_min=params.pressure_aimd_min,
        pressure_aimd_max=params.pressure_aimd_max,
        pressure_aimd_recovery_pressure=params.pressure_aimd_recovery_pressure,
        pressure_aimd_recovery_delay_ratio=params.pressure_aimd_recovery_delay_ratio,
        pressure_aimd_delay_gain=params.pressure_aimd_delay_gain,
        pressure_aimd_cost=params.pressure_aimd_cost,
        pressure_aimd_alpha_floor=params.pressure_aimd_alpha_floor,
        pressure_aimd_extra_budget_frac=params.pressure_aimd_extra_budget_frac,
        pressure_aimd_stable_pressure=params.pressure_aimd_stable_pressure,
        pressure_aimd_stable_delay_ratio=params.pressure_aimd_stable_delay_ratio,
        pressure_aimd_active_pressure=params.pressure_aimd_active_pressure,
        pressure_aimd_min_extra_gain=params.pressure_aimd_min_extra_gain,
    )
    base = specsched(base_params)
    open_loop = specsched(open_params)

    def fn(active: List[ReqState], step_idx: int, cfg: EngineConfig) -> StepDecision:
        kv_manager = getattr(fn, "_kv_manager", None)
        is_open_loop = str(getattr(cfg, "arrival_model", "closed_loop")) in ("poisson", "bursty")
        target = open_loop if is_open_loop else base
        setattr(target, "_kv_manager", kv_manager)
        return target(active, step_idx, cfg)

    return fn


def specsched_pressure_aimd(params: SpecSchedParams) -> PolicyFn:
    """Open-loop controller with stateful arrival-pressure AIMD and marginal delay utility."""
    pressure_params = SpecSchedParams(
        kmax=params.kmax,
        kmin=params.kmin,
        target_accept=params.target_accept,
        use_slo=params.use_slo,
        homogenize_k=params.homogenize_k,
        bucket_k0=params.bucket_k0,
        bucket_accept0=params.bucket_accept0,
        epsilon_h=params.epsilon_h,
        epsilon_h_safe=params.epsilon_h_safe,
        r_max=params.r_max,
        rho_prefill_threshold=params.rho_prefill_threshold,
        pressure_threshold=params.pressure_threshold,
        severe_pressure_threshold=params.severe_pressure_threshold,
        high_load_factor=params.high_load_factor,
        k_safe=params.k_safe,
        k_pressure=params.k_pressure,
        max_slo_boost=params.max_slo_boost,
        enable_overload_hook=params.enable_overload_hook,
        use_position_alpha=params.use_position_alpha,
        low_pressure_k_cap=params.low_pressure_k_cap,
        adaptive_k_guard=params.adaptive_k_guard,
        marginal_gain_floor=params.marginal_gain_floor,
        queue_aware=True,
        queue_age_boost=params.queue_age_boost,
        queue_pressure_threshold=params.queue_pressure_threshold,
        queue_k_guard=True,
        regime_aware=True,
        open_loop_queue_boost=params.open_loop_queue_boost,
        open_loop_guard_threshold=params.open_loop_guard_threshold,
        aimd_controller=False,
        pending_delay_boost=params.pending_delay_boost,
        backpressure_controller=False,
        pressure_aimd_controller=True,
        pressure_aimd_pressure_threshold=params.pressure_aimd_pressure_threshold,
        pressure_aimd_delay_threshold_s=params.pressure_aimd_delay_threshold_s,
        pressure_aimd_add=params.pressure_aimd_add,
        pressure_aimd_mult=params.pressure_aimd_mult,
        pressure_aimd_min=params.pressure_aimd_min,
        pressure_aimd_max=params.pressure_aimd_max,
        pressure_aimd_recovery_pressure=params.pressure_aimd_recovery_pressure,
        pressure_aimd_recovery_delay_ratio=params.pressure_aimd_recovery_delay_ratio,
        pressure_aimd_delay_gain=params.pressure_aimd_delay_gain,
        pressure_aimd_cost=params.pressure_aimd_cost,
        pressure_aimd_alpha_floor=params.pressure_aimd_alpha_floor,
        pressure_aimd_extra_budget_frac=params.pressure_aimd_extra_budget_frac,
        pressure_aimd_stable_pressure=params.pressure_aimd_stable_pressure,
        pressure_aimd_stable_delay_ratio=params.pressure_aimd_stable_delay_ratio,
        pressure_aimd_active_pressure=params.pressure_aimd_active_pressure,
        pressure_aimd_min_extra_gain=params.pressure_aimd_min_extra_gain,
    )
    return specsched_regime_aware(pressure_params)


# specsched_no_evict is realized at engine-construction time by passing the
# 'no_evict' KV policy instead of 'spec_aware'.
