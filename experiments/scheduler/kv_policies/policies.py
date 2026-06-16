"""Eviction policies plugged into core.kv_cache.KVManager."""
from __future__ import annotations

from typing import Dict, List

from core.kv_cache import CacheEntry, EvictionDecision


def _victim_one(entries: Dict[int, CacheEntry], need_blocks: int, key_fn) -> EvictionDecision:
    if not entries:
        return EvictionDecision()
    victims: List[int] = []
    freed = 0
    for entry in sorted(entries.values(), key=key_fn):
        victims.append(entry.req_id)
        freed += entry.blocks
        if freed >= need_blocks:
            break
    return EvictionDecision(victims=victims)


def lru(entries, need_blocks, current_step):
    return _victim_one(entries, need_blocks, key_fn=lambda e: e.last_use_step)


def longest_context(entries, need_blocks, current_step):
    return _victim_one(entries, need_blocks, key_fn=lambda e: -e.tokens)


def shortest_context(entries, need_blocks, current_step):
    return _victim_one(entries, need_blocks, key_fn=lambda e: e.tokens)


def h2o_like(entries, need_blocks, current_step):
    """H2O-inspired: keep entries with high recent usage (proxy: small idle gap),
    evict ones that look 'cold' — combine idle gap + length penalty."""
    def score(e: CacheEntry):
        idle = current_step - e.last_use_step
        return (-idle, -e.tokens)  # most idle first; tie-break by largest

    return _victim_one(entries, need_blocks, key_fn=score)


def spec_aware(entries, need_blocks, current_step):
    """Speculation-aware eviction with per-block future-value scoring.

    Lower scores are evicted first. The policy keeps cache for requests that
    are likely to amortize their KV through accepted drafts, have meaningful
    remaining output, or are already close to their TPOT budget. Large idle
    contexts receive a pressure-relief discount so the policy can still free
    enough blocks under tight budgets.
    """
    def score(e: CacheEntry):
        idle = current_step - e.last_use_step
        blocks = max(1, e.blocks)
        accept = max(0.0, min(1.0, e.accept_rate_ema))
        remaining = max(1, e.remaining_tokens)
        future_value = accept * min(remaining, 64) / 64.0
        slo_risk = min(3.0, e.virtual_queue_ms / max(1.0, e.slo_tpot_ms)) if e.slo_tpot_ms else 0.0
        prompt_cost = min(2.0, e.prompt_len / 512.0)
        idle_discount = min(2.0, idle / 32.0)
        size_discount = min(2.0, blocks / 32.0)
        keep_value = 1.2 * future_value + 0.9 * slo_risk + 0.4 * prompt_cost
        evict_value = idle_discount + 0.8 * size_discount + (1.0 - accept)
        per_block_score = (keep_value - evict_value) / blocks
        return (per_block_score, e.last_use_step)

    return _victim_one(entries, need_blocks, key_fn=score)


def spec_aware_v2(entries, need_blocks, current_step):
    """Spec-aware eviction variant with stronger re-prefill amortization.

    This is an experimental variant: keep requests with high future accepted
    draft value and high SLO risk, but evict large idle contexts when their
    remaining-token amortization is low. Ranking remains per block so one large
    cold context can be preferred over many small active contexts.
    """
    if not entries:
        return EvictionDecision()
    max_blocks = max(1, max(e.blocks for e in entries.values()))
    max_prompt = max(1, max(e.prompt_len for e in entries.values()))
    max_remaining = max(1, max(e.remaining_tokens for e in entries.values()))

    def score(e: CacheEntry):
        idle = max(0, current_step - e.last_use_step)
        blocks = max(1, e.blocks)
        accept = max(0.0, min(1.0, e.accept_rate_ema))
        remaining = max(1, e.remaining_tokens)
        future_value = accept * (remaining / max_remaining) ** 0.5
        slo_risk = min(3.0, e.virtual_queue_ms / max(1.0, e.slo_tpot_ms)) if e.slo_tpot_ms else 0.0
        prefill_amort = (e.prompt_len / max_prompt) / remaining
        idle_pressure = min(3.0, idle / 16.0)
        size_pressure = blocks / max_blocks
        keep_value = 1.5 * future_value + 0.8 * slo_risk + 0.6 * prefill_amort
        evict_value = 1.2 * idle_pressure + 1.0 * size_pressure + 0.6 * (1.0 - accept)
        return ((keep_value - evict_value) / blocks, e.last_use_step)

    return _victim_one(entries, need_blocks, key_fn=score)


def spec_aware_guarded(entries, need_blocks, current_step):
    """Spec-aware eviction with a pressure guard.

    The default score is intentionally conservative. Under severe pressure,
    however, keeping many large contexts can create re-prefill cascades. This
    variant switches to a larger-context pressure score only when the victim
    set itself indicates high cache pressure: not enough cold low-value entries
    are available to satisfy the requested blocks. It remains a static policy
    entry point for controlled experiments, not the paper's default.
    """
    if not entries:
        return EvictionDecision()

    max_blocks = max(1, max(e.blocks for e in entries.values()))
    max_prompt = max(1, max(e.prompt_len for e in entries.values()))
    max_remaining = max(1, max(e.remaining_tokens for e in entries.values()))

    def base_score(e: CacheEntry):
        idle = max(0, current_step - e.last_use_step)
        blocks = max(1, e.blocks)
        accept = max(0.0, min(1.0, e.accept_rate_ema))
        remaining = max(1, e.remaining_tokens)
        future_value = accept * min(remaining, 64) / 64.0
        slo_risk = min(2.5, e.virtual_queue_ms / max(1.0, e.slo_tpot_ms)) if e.slo_tpot_ms else 0.0
        prompt_cost = min(2.0, e.prompt_len / max_prompt)
        idle_discount = min(2.5, idle / 24.0)
        size_discount = blocks / max_blocks
        keep_value = 1.15 * future_value + 0.85 * slo_risk + 0.45 * prompt_cost
        evict_value = idle_discount + 0.9 * size_discount + 0.7 * (1.0 - accept)
        return ((keep_value - evict_value) / blocks, e.last_use_step)

    # Estimate whether a few low-value victims can satisfy pressure. If not,
    # promote size relief to avoid evicting many small active requests.
    trial = sorted(entries.values(), key=base_score)
    freed = 0
    victims_needed = 0
    for entry in trial:
        freed += entry.blocks
        victims_needed += 1
        if freed >= need_blocks:
            break
    severe = victims_needed >= 3 or need_blocks >= max(4, int(0.12 * max_blocks))

    if not severe:
        return _victim_one(entries, need_blocks, key_fn=base_score)

    def guarded_score(e: CacheEntry):
        idle = max(0, current_step - e.last_use_step)
        blocks = max(1, e.blocks)
        accept = max(0.0, min(1.0, e.accept_rate_ema))
        remaining = max(1, e.remaining_tokens)
        future_value = accept * (remaining / max_remaining) ** 0.5
        slo_risk = min(2.5, e.virtual_queue_ms / max(1.0, e.slo_tpot_ms)) if e.slo_tpot_ms else 0.0
        prefill_cost = min(2.0, e.prompt_len / max_prompt)
        size_relief = blocks / max_blocks
        idle_relief = min(2.5, idle / 16.0)
        low_remaining_relief = 1.0 / max(1.0, remaining / 32.0)
        keep_value = 1.35 * future_value + 0.95 * slo_risk + 0.55 * prefill_cost
        evict_value = 1.45 * size_relief + 1.05 * idle_relief + 0.65 * low_remaining_relief + 0.7 * (1.0 - accept)
        return ((keep_value - evict_value) / blocks, e.last_use_step)

    return _victim_one(entries, need_blocks, key_fn=guarded_score)


def spec_aware_calibrated(entries, need_blocks, current_step):
    """Conservative hybrid eviction with an LRU guardrail.

    LRU is a strong default in the current KV sweep. This calibrated variant
    therefore treats LRU as the fallback path and only deviates when a victim is
    clearly low-value: large enough to reduce multi-victim cascades, idle enough
    to be cold, and unlikely to amortize a re-prefill through accepted drafts.
    The guardrail is intentionally strict so uncertain cells remain comparable
    to LRU instead of regressing under high load.
    """
    if not entries:
        return EvictionDecision()

    fallback = lru(entries, need_blocks, current_step)
    if not fallback.victims:
        return fallback

    max_blocks = max(1, max(e.blocks for e in entries.values()))
    max_remaining = max(1, max(e.remaining_tokens for e in entries.values()))

    def candidate_score(e: CacheEntry):
        idle = max(0, current_step - e.last_use_step)
        blocks = max(1, e.blocks)
        accept = max(0.0, min(1.0, e.accept_rate_ema))
        remaining = max(1, e.remaining_tokens)
        size_relief = blocks / max_blocks
        idle_relief = min(2.0, idle / 24.0)
        low_accept_relief = 1.0 - accept
        low_remaining_relief = 1.0 - min(1.0, remaining / max_remaining)
        slo_risk = min(2.0, e.virtual_queue_ms / max(1.0, e.slo_tpot_ms)) if e.slo_tpot_ms else 0.0
        relief = 1.4 * size_relief + 1.0 * idle_relief + 0.7 * low_accept_relief + 0.35 * low_remaining_relief
        keep = 0.9 * accept + 0.7 * slo_risk + 0.4 * (remaining / max_remaining)
        return (relief - keep, e.last_use_step)

    # Only consider entries with a clear relief signal. This keeps ambiguous
    # high-load cases on LRU and avoids replacing one good victim with several
    # slightly-scored victims.
    ranked = sorted(entries.values(), key=candidate_score, reverse=True)
    picked: List[int] = []
    freed = 0
    for entry in ranked:
        idle = max(0, current_step - entry.last_use_step)
        accept = max(0.0, min(1.0, entry.accept_rate_ema))
        large = entry.blocks >= max(4, int(0.20 * max_blocks))
        cold = idle >= 4
        low_value = accept <= 0.72 or entry.remaining_tokens <= 48
        if not (large and cold and low_value):
            continue
        picked.append(entry.req_id)
        freed += entry.blocks
        if freed >= need_blocks:
            break

    if freed < need_blocks:
        return fallback

    lru_freed = sum(entries[vid].blocks for vid in fallback.victims if vid in entries)
    if len(picked) > len(fallback.victims) or freed < lru_freed:
        return fallback
    return EvictionDecision(victims=picked)


def vllm_preemption(entries, need_blocks, current_step):
    """vLLM default preemption: pick the most recently admitted in-flight
    request, free its whole KV, and let the engine recompute (re-prefill) it
    on the next admission round.

    vLLM's recompute policy preempts the youngest request first (LIFO) so older
    requests retain their KV and make progress; the preempted one effectively
    restarts. We approximate "youngest" with (-req_id, -last_use_step) which is
    correct under closed-loop (all arrivals at t=0; larger req_id = later
    admit) and is a reasonable proxy under open-loop.
    """
    def score(e: CacheEntry):
        return (-e.req_id, -e.last_use_step)

    return _victim_one(entries, need_blocks, key_fn=score)


def no_evict(entries, need_blocks, current_step):
    """Disable victim selection; callers must wait or recompute instead."""
    return EvictionDecision()


REGISTRY = {
    "lru": lru,
    "longest": longest_context,
    "shortest": shortest_context,
    "h2o": h2o_like,
    "spec_aware": spec_aware,
    "spec_aware_v2": spec_aware_v2,
    "spec_aware_guarded": spec_aware_guarded,
    "spec_aware_calibrated": spec_aware_calibrated,
    "vllm": vllm_preemption,
    "no_evict": no_evict,
}


def get(name: str):
    if name not in REGISTRY:
        raise ValueError(f"unknown eviction policy {name!r}; pick from {list(REGISTRY)}")
    return REGISTRY[name]
