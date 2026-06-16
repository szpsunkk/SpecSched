"""Multi-tenant speculative-decoding engine with KV-aware admission.

Architecture
------------
Time advances in *scheduling steps*. Each step:
    1. ADMIT: pull as many newly-arrived requests as the KV manager will accept;
       run target prefill (and draft prefill if speculation is enabled). Each
       admitted request commits its first target token immediately so it has a
       valid `last_token` going into the first speculative round.
    2. SCHEDULE: ask the policy to pick a batch of in-flight requests and a
       per-request draft length k_i (k_i may be 0 -> NoSpec step). The batch
       size is bounded by `max_batch`.
    3. DRAFT: each batched request takes k_i sequential draft forwards (drafts
       are still per-request because draft KV-caches differ; this matches what
       real systems do).
    4. VERIFY: one target forward per request in the batch. We support a
       "batched verification" mode that stacks B requests' (last_token + drafts)
       into a single padded forward when `batch_verify=True` AND every k_i is
       equal. Otherwise we fall back to per-request verification (NoBatch
       ablation also forces this path).
    5. COMMIT + EVICT: extend each request's KV reservation; the KV manager
       evicts on pressure using the configured policy.

The scheduling step is also the unit at which SpecSched runs its SLO virtual
queue and per-request draft-length adaptation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from core.kv_cache import KVManager
from core.speculative import (
    ReqState,
    _is_remote_draft,
    batched_target_verify,
    draft_propose,
    free_state,
    normalize_cache,
    prefill_draft,
    prefill_target,
    target_verify,
)
from core.workload import Request, Workload


@dataclass
class EngineConfig:
    max_batch: int = 8
    batch_verify: bool = True   # stack equal-k requests into one target forward
    use_speculation: bool = True
    use_draft: bool = True
    block_tokens: int = 16
    cache_replay_chunk_tokens: int = 32
    log_every_steps: int = 0
    arrival_model: str = "closed_loop"
    arrival_rate_rps: float = 0.0
    pending_count: int = 0
    in_flight_count: int = 0
    active_count: int = 0
    oldest_pending_s: float = 0.0
    max_admit_per_step: int = 0
    pre_admit_pending_count: int = 0
    oldest_ready_pending_s: float = 0.0
    post_admit_ready_count: int = 0
    oldest_post_admit_ready_s: float = 0.0


@dataclass
class StepDecision:
    selected: List[Tuple[int, int]] = field(default_factory=list)  # (req_id, k)


# Policy interface: given list of in-flight ReqState and current step, return
# StepDecision. May also mutate state (e.g. SLO virtual queue). Policies are in
# the `policies/` package.
PolicyFn = Callable[[List[ReqState], int, EngineConfig], StepDecision]


@dataclass
class RequestRecord:
    req_id: int
    bucket: str
    tier: str
    phase: str
    source_id: str
    prompt_len: int
    arrival_s: float
    admit_s: float
    first_token_s: float
    finish_s: float
    generated: int
    target_forwards: int
    draft_forwards: int
    accepted: int
    drafted: int
    reprefills: int
    service_target_ms: float
    service_draft_ms: float
    service_other_ms: float
    queue_s: float
    ttft_ms: float
    tpot_ms: float
    e2e_latency_s: float
    slo_ttft_ms: Optional[float] = None
    slo_tpot_ms: Optional[float] = None


@dataclass
class EngineResult:
    records: List[RequestRecord]
    wall_s: float
    kv_stats: dict


def _timing_summary_ns(values: List[int]) -> Dict[str, float]:
    if not values:
        return {
            "mean_us": 0.0,
            "p50_us": 0.0,
            "p95_us": 0.0,
            "p99_us": 0.0,
        }
    ordered = sorted(values)

    def pct(q: float) -> float:
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        if lo == hi:
            return ordered[lo] / 1000.0
        frac = pos - lo
        return (ordered[lo] * (1.0 - frac) + ordered[hi] * frac) / 1000.0

    return {
        "mean_us": (sum(values) / len(values)) / 1000.0,
        "p50_us": pct(0.50),
        "p95_us": pct(0.95),
        "p99_us": pct(0.99),
    }


class Engine:
    def __init__(
        self,
        target_model,
        draft_model,
        kv_manager: KVManager,
        cfg: EngineConfig,
        policy: PolicyFn,
    ):
        self.target = target_model
        self.draft = draft_model
        self.kv = kv_manager
        self.cfg = cfg
        self.policy = policy

    # -- helpers ---------------------------------------------------------

    def _ready_pending_stats(self, pending: List[Request], now: float) -> Tuple[int, float]:
        ready = 0
        oldest = 0.0
        for req in pending:
            if req.arrival_s > now:
                break
            if ready == 0:
                oldest = max(0.0, now - req.arrival_s)
            ready += 1
        return ready, oldest

    def _admit(self, pending: List[Request], now: float, step_idx: int, in_flight: Dict[int, ReqState]) -> None:
        i = 0
        admitted = 0
        max_admit = int(getattr(self.cfg, "max_admit_per_step", 0) or 0)
        while i < len(pending):
            if max_admit > 0 and admitted >= max_admit:
                break
            req = pending[i]
            if req.arrival_s > now:
                break
            ok, evicted = self.kv.admit_or_evict(req.req_id, int(req.prompt_ids.numel()), step_idx)
            self._mark_evicted(evicted, in_flight, step_idx)
            if not ok:
                break
            state = ReqState(
                req_id=req.req_id,
                prompt_ids=req.prompt_ids,
                arrival_s=req.arrival_s,
                max_new_tokens=req.max_new_tokens,
                slo_ttft_ms=req.slo_ttft_ms,
                slo_tpot_ms=req.slo_tpot_ms,
                bucket=req.bucket,
                tier=req.tier,
                phase=req.phase,
                source_id=req.source_id,
            )
            t_admit = time.perf_counter()
            state.admit_s = t_admit
            state.target_past, first_tok, _ = prefill_target(self.target, req.prompt_ids)
            if self.cfg.use_draft and self.draft is not None:
                state.draft_past, _ = prefill_draft(self.draft, req.prompt_ids, req.req_id)
                state.service_other_ms += float(getattr(self.draft, "last_overhead_ms", 0.0))
            state.generated.append(first_tok)
            state.last_token = first_tok
            state.first_token_s = time.perf_counter()
            state.target_forwards += 1  # the prefill itself
            in_flight[req.req_id] = state
            self._sync_kv_entry(state, step_idx)
            pending.pop(i)
            admitted += 1
            if state.max_new_tokens <= 1:
                state.finish_s = state.first_token_s

    def _mark_evicted(self, evicted: List[int], in_flight: Dict[int, ReqState], step_idx: int) -> None:
        for victim_id in evicted:
            victim = in_flight.get(victim_id)
            if victim is None or victim.finished():
                continue
            free_state(victim)
            victim.cache_valid = False
            victim.reprefill_count += 1
            self.kv.mark_reprefill(victim_id, step_idx)

    def _sync_kv_entry(self, state: ReqState, step_idx: int, accepted: int = 0, drafted: int = 0) -> None:
        generated = max(1, len(state.generated))
        elapsed_ms = max(0.0, state.scheduler_now_s - state.arrival_s) * 1000.0
        slo_tpot = float(state.slo_tpot_ms or 0.0)
        deficit = max(0.0, elapsed_ms / generated - slo_tpot) if slo_tpot > 0 else 0.0
        self.kv.touch(
            state.req_id,
            step_idx,
            accepted=accepted,
            drafted=drafted,
            remaining_tokens=state.max_new_tokens - len(state.generated),
            virtual_queue_ms=deficit,
            slo_tpot_ms=slo_tpot,
        )

    def _ensure_cache(self, state: ReqState, step_idx: int, in_flight: Dict[int, ReqState]) -> bool:
        if state.cache_valid and state.target_past is not None:
            state.target_past = normalize_cache(state.target_past)
            if self.cfg.use_draft and self.draft is not None and not _is_remote_draft(self.draft):
                state.draft_past = normalize_cache(state.draft_past)
            return True
        if len(state.generated) <= 0:
            state.target_past, first_tok, _ = prefill_target(self.target, state.prompt_ids)
            state.generated.append(first_tok)
            state.last_token = first_tok
            state.target_forwards += 1
            if self.cfg.use_draft and self.draft is not None:
                state.draft_past, _ = prefill_draft(self.draft, state.prompt_ids, state.req_id)
                state.service_other_ms += float(getattr(self.draft, "last_overhead_ms", 0.0))
            state.cache_valid = True
            ok, evicted = self.kv.admit_or_evict(
                state.req_id,
                int(state.prompt_ids.numel()),
                step_idx,
            )
            self._mark_evicted(evicted, in_flight, step_idx)
            state.cache_valid = ok
            if not ok:
                free_state(state)
            return ok

        state.target_past, _, _ = prefill_target(self.target, state.prompt_ids)
        if self.cfg.use_draft and self.draft is not None:
            state.draft_past, _ = prefill_draft(self.draft, state.prompt_ids, state.req_id)
            state.service_other_ms += float(getattr(self.draft, "last_overhead_ms", 0.0))
        self._replay_generated_tail(state)
        ok, evicted = self.kv.admit_or_evict(
            state.req_id,
            int(state.prompt_ids.numel()) + len(state.generated) - 1,
            step_idx,
        )
        self._mark_evicted(evicted, in_flight, step_idx)
        state.cache_valid = ok
        if not ok:
            free_state(state)
        return ok

    def _step_per_request(
        self,
        decisions: List[Tuple[int, int]],
        in_flight: Dict[int, ReqState],
        step_idx: int,
    ) -> None:
        for req_id, k in decisions:
            state = in_flight.get(req_id)
            if state is None or state.finished():
                continue
            if not self._ensure_cache(state, step_idx, in_flight):
                continue
            remaining = state.max_new_tokens - len(state.generated)
            if remaining <= 0:
                state.finish_s = time.perf_counter()
                continue
            k_eff = min(k, remaining) if k > 0 else 0
            proposals: List[int] = []
            if k_eff > 0 and self.cfg.use_speculation and self.draft is not None:
                proposals, _ = draft_propose(self.draft, state, k_eff)
            accepted, appended, _ = target_verify(
                self.target, self.draft, state, proposals, remaining
            )
            now = time.perf_counter()
            self._sync_kv_entry(state, step_idx, accepted=accepted, drafted=len(proposals))
            ok, evicted = self.kv.grow(state.req_id, len(appended), step_idx)
            self._mark_evicted(evicted, in_flight, step_idx)
            if not ok:
                # KV pressure: free this request and force a re-prefill on the
                # next scheduling round. Generation continues after rebuild.
                free_state(state)
                self.kv.release(state.req_id)
                state.reprefill_count += 1
                self.kv.mark_reprefill(state.req_id, step_idx)
                state.cache_valid = False
            if state.first_token_s is None:
                state.first_token_s = now
            if len(state.generated) >= state.max_new_tokens:
                state.finish_s = now

    def _step_batched_verify(
        self,
        decisions: List[Tuple[int, int]],
        in_flight: Dict[int, ReqState],
        step_idx: int,
    ) -> None:
        if not decisions:
            return
        if self.draft is None or not self.cfg.use_speculation:
            self._step_per_request(decisions, in_flight, step_idx)
            return

        by_k: Dict[int, List[Tuple[int, int]]] = {}
        for rid, k in decisions:
            by_k.setdefault(int(k), []).append((rid, int(k)))

        if len(by_k) > 1:
            for k, group in sorted(by_k.items(), key=lambda item: (-len(item[1]), item[0])):
                if k <= 0 or len(group) <= 1:
                    self._step_per_request(group, in_flight, step_idx)
                else:
                    self._step_batched_verify(group, in_flight, step_idx)
            return

        # Per-request draft (drafts share no KV across requests).
        k = next(iter(by_k))
        if k <= 0:
            self._step_per_request(decisions, in_flight, step_idx)
            return

        per_req_proposals: Dict[int, List[int]] = {}
        skipped: set[int] = set()
        for req_id, _ in decisions:
            state = in_flight[req_id]
            if not self._ensure_cache(state, step_idx, in_flight):
                skipped.add(req_id)
                continue
            remaining = state.max_new_tokens - len(state.generated)
            k_eff = min(k, remaining)
            if k_eff <= 0:
                per_req_proposals[req_id] = []
                continue
            proposals, _ = draft_propose(self.draft, state, k_eff)
            per_req_proposals[req_id] = proposals

        # If the picked batch has heterogenous remaining-token counts, drop those
        # whose proposals are shorter than k from the batched verify and run
        # them on the per-request path.
        runnable = [
            (rid, k_) for (rid, k_) in decisions
            if rid not in skipped
            and in_flight[rid].cache_valid
            and in_flight[rid].target_past is not None
        ]
        eligible = [(rid, k_) for (rid, k_) in runnable if len(per_req_proposals[rid]) == k]
        spillover = [(rid, k_) for (rid, k_) in runnable if len(per_req_proposals[rid]) != k]

        if eligible:
            self._batched_target_verify(eligible, per_req_proposals, in_flight, step_idx)
        if spillover:
            # Use the per-request commit path for these (no double-draft; just verify with stored proposals).
            for rid, _ in spillover:
                state = in_flight[rid]
                if not state.cache_valid or state.target_past is None:
                    continue
                proposals = per_req_proposals[rid]
                remaining = state.max_new_tokens - len(state.generated)
                accepted, appended, _ = target_verify(
                    self.target, self.draft, state, proposals, remaining
                )
                self._post_commit(state, len(appended), accepted, len(proposals), step_idx, in_flight)

    @torch.inference_mode()
    def _batched_target_verify(
        self,
        decisions: List[Tuple[int, int]],
        per_req_proposals: Dict[int, List[int]],
        in_flight: Dict[int, ReqState],
        step_idx: int,
    ) -> None:
        # Remote-draft path doesn't expose the target's DynamicCache so we
        # cannot pack the verify forward across requests. Fall back per-request.
        if self.draft is not None and _is_remote_draft(self.draft):
            for rid, _ in decisions:
                state = in_flight[rid]
                proposals = per_req_proposals[rid]
                remaining = state.max_new_tokens - len(state.generated)
                accepted, appended, _ = target_verify(
                    self.target, self.draft, state, proposals, remaining
                )
                self._post_commit(state, len(appended), accepted, len(proposals), step_idx, in_flight)
            torch.cuda.synchronize()
            return

        states = [in_flight[rid] for rid, _ in decisions]
        proposals_per_req = [per_req_proposals[rid] for rid, _ in decisions]
        remaining_per_req = [s.max_new_tokens - len(s.generated) for s in states]

        try:
            accepted_list, appended_list, _ = batched_target_verify(
                self.target,
                self.draft,
                states,
                proposals_per_req,
                remaining_per_req,
            )
        except Exception as exc:
            # If the batched path is incompatible (e.g. transformers version
            # without DynamicCache APIs, or unexpected past type), fall back to
            # per-request verify so the run still completes auditable rows.
            print(
                f"[engine] batched_target_verify fell back to per-request: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            for rid, _ in decisions:
                state = in_flight[rid]
                proposals = per_req_proposals[rid]
                remaining = state.max_new_tokens - len(state.generated)
                accepted, appended, _ = target_verify(
                    self.target, self.draft, state, proposals, remaining
                )
                self._post_commit(state, len(appended), accepted, len(proposals), step_idx, in_flight)
            torch.cuda.synchronize()
            return

        # Commit each row through the standard accounting path (KV grow, eviction,
        # finish bookkeeping). Generated tokens were already extended inside
        # batched_target_verify, so _post_commit only needs the per-row counts.
        for state, accepted, appended, proposals in zip(
            states, accepted_list, appended_list, proposals_per_req
        ):
            if state.cache_valid and state.target_past is not None:
                self._post_commit(state, len(appended), accepted, len(proposals), step_idx, in_flight)
            elif len(state.generated) >= state.max_new_tokens:
                state.finish_s = time.perf_counter()
        torch.cuda.synchronize()

    def _post_commit(
        self,
        state: ReqState,
        appended: int,
        accepted: int,
        drafted: int,
        step_idx: int,
        in_flight: Dict[int, ReqState],
    ) -> None:
        now = time.perf_counter()
        self._sync_kv_entry(state, step_idx, accepted=accepted, drafted=drafted)
        ok, evicted = self.kv.grow(state.req_id, appended, step_idx)
        self._mark_evicted(evicted, in_flight, step_idx)
        if not ok:
            # treat as preemption + re-prefill
            free_state(state)
            self.kv.release(state.req_id)
            state.reprefill_count += 1
            self.kv.mark_reprefill(state.req_id, step_idx)
            state.cache_valid = False
        if state.first_token_s is None:
            state.first_token_s = now
        if len(state.generated) >= state.max_new_tokens:
            state.finish_s = now

    @torch.inference_mode()
    def _replay_generated_tail(self, state: ReqState) -> None:
        """Rebuild KV cache after preemption without a large activation spike."""
        tail = state.generated[:-1]
        if not tail:
            return
        chunk = max(1, int(self.cfg.cache_replay_chunk_tokens))
        for start in range(0, len(tail), chunk):
            piece = tail[start:start + chunk]
            ids = torch.tensor([piece], dtype=torch.long, device="cuda")
            out = self.target(input_ids=ids, past_key_values=state.target_past, use_cache=True)
            state.target_past = normalize_cache(out.past_key_values)
            if self.cfg.use_draft and self.draft is not None and not _is_remote_draft(self.draft):
                out_d = self.draft(input_ids=ids, past_key_values=state.draft_past, use_cache=True)
                state.draft_past = normalize_cache(out_d.past_key_values)

    # -- main loop -------------------------------------------------------

    def run(self, workload: Workload) -> EngineResult:
        pending = sorted(workload.requests, key=lambda r: (r.arrival_s, r.req_id))
        in_flight: Dict[int, ReqState] = {}
        records: List[RequestRecord] = []

        wall_start = time.perf_counter()
        step_idx = 0
        selected_k_sum = 0
        selected_count = 0
        nonzero_k_count = 0
        scheduler_timing_ns: Dict[str, List[int]] = {
            "bookkeeping": [],
            "policy": [],
            "kv_commit": [],
            "cleanup": [],
        }

        while pending or in_flight:
            now = time.perf_counter() - wall_start
            ready_before_admit, oldest_ready_before_admit = self._ready_pending_stats(pending, now)
            self._admit(pending, now, step_idx, in_flight)
            ready_after_admit, oldest_ready_after_admit = self._ready_pending_stats(pending, now)

            if not in_flight:
                if pending:
                    sleep_s = max(0.0, pending[0].arrival_s - (time.perf_counter() - wall_start))
                    if sleep_s > 0:
                        time.sleep(min(sleep_s, 0.05))
                continue

            t_book0 = time.perf_counter_ns()
            active = [s for s in in_flight.values() if not s.finished()]
            for state in active:
                state.scheduler_now_s = now
            self.cfg.pending_count = len(pending)
            self.cfg.in_flight_count = len(in_flight)
            self.cfg.active_count = len(active)
            self.cfg.pre_admit_pending_count = ready_before_admit
            self.cfg.oldest_ready_pending_s = oldest_ready_before_admit
            self.cfg.post_admit_ready_count = ready_after_admit
            self.cfg.oldest_post_admit_ready_s = oldest_ready_after_admit
            self.cfg.oldest_pending_s = max(
                0.0,
                now - pending[0].arrival_s,
            ) if pending else 0.0
            scheduler_timing_ns["bookkeeping"].append(time.perf_counter_ns() - t_book0)
            if not active:
                # drain finished
                t_cleanup0 = time.perf_counter_ns()
                for state in list(in_flight.values()):
                    if state.finished():
                        records.append(_record_from_state(state, wall_start))
                        if self.draft is not None and _is_remote_draft(self.draft):
                            self.draft.release(state.req_id)
                        free_state(state)
                        self.kv.release(state.req_id)
                        in_flight.pop(state.req_id)
                scheduler_timing_ns["cleanup"].append(time.perf_counter_ns() - t_cleanup0)
                continue

            t_policy0 = time.perf_counter_ns()
            decision = self.policy(active, step_idx, self.cfg)
            scheduler_timing_ns["policy"].append(time.perf_counter_ns() - t_policy0)
            if not decision.selected:
                step_idx += 1
                continue
            selected_k_sum += sum(int(k) for _, k in decision.selected)
            selected_count += len(decision.selected)
            nonzero_k_count += sum(1 for _, k in decision.selected if int(k) > 0)
            prev_lens = {
                rid: len(in_flight[rid].generated)
                for rid, _ in decision.selected
                if rid in in_flight
            }

            if self.cfg.batch_verify and self.cfg.use_speculation and self.draft is not None:
                self._step_batched_verify(decision.selected, in_flight, step_idx)
            else:
                self._step_per_request(decision.selected, in_flight, step_idx)
            tokens_this_step = sum(
                max(0, len(in_flight[rid].generated) - before)
                for rid, before in prev_lens.items()
                if rid in in_flight
            )
            t_kv0 = time.perf_counter_ns()
            self.kv.note_tokens_generated(step_idx, tokens_this_step)
            scheduler_timing_ns["kv_commit"].append(time.perf_counter_ns() - t_kv0)

            # collect finished
            t_cleanup0 = time.perf_counter_ns()
            for state in list(in_flight.values()):
                if state.finished():
                    records.append(_record_from_state(state, wall_start))
                    if self.draft is not None and _is_remote_draft(self.draft):
                        self.draft.release(state.req_id)
                    free_state(state)
                    self.kv.release(state.req_id)
                    in_flight.pop(state.req_id)
            scheduler_timing_ns["cleanup"].append(time.perf_counter_ns() - t_cleanup0)

            step_idx += 1
            if self.cfg.log_every_steps and step_idx % self.cfg.log_every_steps == 0:
                print(
                    f"[engine] step={step_idx} in_flight={len(in_flight)} "
                    f"pending={len(pending)} finished={len(records)}",
                    flush=True,
                )

        wall_s = time.perf_counter() - wall_start
        stats = self.kv.stats()
        total_scheduler_ns = sum(sum(values) for values in scheduler_timing_ns.values())
        stats.update({
            "scheduler_steps": step_idx,
            "selected_k_sum": selected_k_sum,
            "selected_count": selected_count,
            "nonzero_k_count": nonzero_k_count,
            "mean_selected_k": selected_k_sum / max(1, selected_count),
            "scheduler_cpu_total_us": total_scheduler_ns / 1000.0,
            "scheduler_cpu_mean_us": (total_scheduler_ns / max(1, step_idx)) / 1000.0,
            "scheduler_cpu_frac": (total_scheduler_ns / 1e9) / max(wall_s, 1e-9),
        })
        for phase, values in scheduler_timing_ns.items():
            for key, value in _timing_summary_ns(values).items():
                stats[f"scheduler_{phase}_{key}"] = value
        return EngineResult(records=records, wall_s=wall_s, kv_stats=stats)


def _record_from_state(state: ReqState, wall_start: float) -> RequestRecord:
    arrival_abs = wall_start + state.arrival_s
    admit_s = (state.admit_s - arrival_abs) if state.admit_s else 0.0
    ttft_ms = ((state.first_token_s - arrival_abs) * 1000.0) if state.first_token_s else float("inf")
    finish = state.finish_s if state.finish_s else time.perf_counter()
    e2e = finish - arrival_abs
    generated = len(state.generated)
    tpot_ms = (e2e * 1000.0 / max(1, generated))
    return RequestRecord(
        req_id=state.req_id,
        bucket=state.bucket,
        tier=state.tier,
        phase=state.phase,
        source_id=state.source_id,
        prompt_len=int(state.prompt_ids.numel()),
        arrival_s=state.arrival_s,
        admit_s=max(0.0, admit_s),
        first_token_s=state.first_token_s - wall_start if state.first_token_s else float("nan"),
        finish_s=finish - wall_start,
        generated=generated,
        target_forwards=state.target_forwards,
        draft_forwards=state.draft_forwards,
        accepted=state.accepted_total,
        drafted=state.drafted_total,
        reprefills=state.reprefill_count,
        service_target_ms=state.service_target_ms,
        service_draft_ms=state.service_draft_ms,
        service_other_ms=state.service_other_ms,
        queue_s=max(0.0, admit_s),
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_latency_s=e2e,
        slo_ttft_ms=state.slo_ttft_ms,
        slo_tpot_ms=state.slo_tpot_ms,
    )
