"""Per-request KV-cache block table with a pluggable eviction policy.

We do not implement a paged-attention kernel. Instead we model KV-cache budget
at the block granularity that papers use to discuss admission/eviction:
    - Each request owns its HuggingFace `past_key_values` tuple in GPU memory.
    - We *count* its KV usage in fixed-size blocks (block_tokens tokens each).
    - A KV manager enforces a global block budget across all live requests.
    - On budget pressure, an eviction policy chooses victims; victims are
      preempted, their KV is freed, and they re-prefill on the next admission.

This is the same accounting vLLM uses for PagedAttention budgeting, just
without the paged attention kernel — sufficient for fairly comparing
speculation-aware vs speculation-oblivious eviction policies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class CacheEntry:
    req_id: int
    blocks: int
    tokens: int
    last_use_step: int
    prompt_len: int
    accept_rate_ema: float = 0.0  # speculation quality signal, used by spec-aware
    pending_drafts: int = 0
    remaining_tokens: int = 1
    virtual_queue_ms: float = 0.0
    slo_tpot_ms: float = 0.0


@dataclass
class EvictionDecision:
    victims: List[int] = field(default_factory=list)


# Policy signature: (entries, blocks_needed, current_step) -> EvictionDecision
EvictionPolicy = Callable[[Dict[int, CacheEntry], int, int], EvictionDecision]


class KVManager:
    """Block-budget KV-cache manager.

    Block budget is a hard cap on the sum of `blocks` across live entries. The
    target model's actual `past_key_values` is held externally; this class just
    bookkeeps how many blocks each request occupies and which to evict.
    """

    def __init__(self, total_blocks: int, block_tokens: int, eviction_policy: EvictionPolicy):
        if total_blocks <= 0 or block_tokens <= 0:
            raise ValueError("total_blocks and block_tokens must be positive")
        self.total_blocks = total_blocks
        self.block_tokens = block_tokens
        self.policy = eviction_policy
        self.entries: Dict[int, CacheEntry] = {}
        self.eviction_count = 0
        self.reprefill_count = 0
        self.pressure_events = 0
        self.max_live_blocks = 0
        self.live_block_samples = 0
        self.live_block_sum = 0
        self._window_capacity_steps = 200
        self._reprefill_events_window: List[Tuple[int, int]] = []
        self._tokens_generated_window: List[Tuple[int, int]] = []

    def _sample(self) -> None:
        used = self._used()
        self.max_live_blocks = max(self.max_live_blocks, used)
        self.live_block_samples += 1
        self.live_block_sum += used

    def _used(self) -> int:
        return sum(e.blocks for e in self.entries.values())

    def free_blocks(self) -> int:
        return self.total_blocks - self._used()

    def cache_resident(self, req_id: int) -> bool:
        return req_id in self.entries

    def _trim_window(self, current_step: int) -> None:
        lo = current_step - self._window_capacity_steps
        self._reprefill_events_window = [
            item for item in self._reprefill_events_window if item[0] >= lo
        ]
        self._tokens_generated_window = [
            item for item in self._tokens_generated_window if item[0] >= lo
        ]

    def note_tokens_generated(self, current_step: int, n_tokens: int) -> None:
        if n_tokens > 0:
            self._tokens_generated_window.append((current_step, int(n_tokens)))
        self._trim_window(current_step)

    def reprefill_rate(self) -> float:
        reprefills = sum(count for _, count in self._reprefill_events_window)
        tokens = sum(count for _, count in self._tokens_generated_window)
        return reprefills / max(1, tokens)

    def blocks_for(self, tokens: int) -> int:
        return (tokens + self.block_tokens - 1) // self.block_tokens

    def admit_or_evict(
        self,
        req_id: int,
        prompt_len: int,
        current_step: int,
    ) -> Tuple[bool, List[int]]:
        """Try to admit `req_id`'s prompt; evict victims if needed.

        Returns (admitted, evicted_ids). If admitted is False, caller should
        defer the request to the next scheduling round.
        """
        if req_id in self.entries:
            return True, []
        need = self.blocks_for(prompt_len)
        if need > self.total_blocks:
            raise RuntimeError(
                f"prompt of {prompt_len} tokens exceeds total KV budget "
                f"({self.total_blocks} blocks * {self.block_tokens} tokens)"
            )

        evicted: List[int] = []
        if self.free_blocks() < need:
            self.pressure_events += 1
        while self.free_blocks() < need:
            decision = self.policy(self.entries, need - self.free_blocks(), current_step)
            if not decision.victims:
                return False, evicted
            for vid in decision.victims:
                if vid in self.entries:
                    self.entries.pop(vid)
                    self.eviction_count += 1
                    evicted.append(vid)
            if not evicted:
                return False, evicted

        self.entries[req_id] = CacheEntry(
            req_id=req_id,
            blocks=need,
            tokens=prompt_len,
            last_use_step=current_step,
            prompt_len=prompt_len,
        )
        self._sample()
        return True, evicted

    def grow(self, req_id: int, extra_tokens: int, current_step: int) -> Tuple[bool, List[int]]:
        """Reserve room for `extra_tokens` more for an already-admitted request."""
        entry = self.entries.get(req_id)
        if entry is None:
            return False, []
        new_tokens = entry.tokens + extra_tokens
        new_blocks = self.blocks_for(new_tokens)
        delta = new_blocks - entry.blocks
        evicted: List[int] = []
        if delta > 0 and self.free_blocks() < delta:
            self.pressure_events += 1
        while self.free_blocks() < delta:
            decision = self.policy(
                {k: v for k, v in self.entries.items() if k != req_id},
                delta - self.free_blocks(),
                current_step,
            )
            if not decision.victims:
                return False, evicted
            for vid in decision.victims:
                if vid in self.entries and vid != req_id:
                    self.entries.pop(vid)
                    self.eviction_count += 1
                    evicted.append(vid)
            if not evicted:
                return False, evicted
        entry.blocks = new_blocks
        entry.tokens = new_tokens
        entry.last_use_step = current_step
        self._sample()
        return True, evicted

    def touch(
        self,
        req_id: int,
        current_step: int,
        accepted: int = 0,
        drafted: int = 0,
        remaining_tokens: Optional[int] = None,
        virtual_queue_ms: Optional[float] = None,
        slo_tpot_ms: Optional[float] = None,
    ) -> None:
        entry = self.entries.get(req_id)
        if entry is None:
            return
        entry.last_use_step = current_step
        if drafted > 0:
            rate = accepted / drafted
            entry.accept_rate_ema = 0.7 * entry.accept_rate_ema + 0.3 * rate
        if remaining_tokens is not None:
            entry.remaining_tokens = max(1, int(remaining_tokens))
        if virtual_queue_ms is not None:
            entry.virtual_queue_ms = max(0.0, float(virtual_queue_ms))
        if slo_tpot_ms is not None:
            entry.slo_tpot_ms = max(0.0, float(slo_tpot_ms))

    def release(self, req_id: int) -> None:
        self.entries.pop(req_id, None)

    def mark_reprefill(self, req_id: int, current_step: Optional[int] = None) -> None:
        self.reprefill_count += 1
        if current_step is not None:
            self._reprefill_events_window.append((current_step, 1))
            self._trim_window(current_step)

    def stats(self) -> dict:
        return {
            "evictions": self.eviction_count,
            "reprefills": self.reprefill_count,
            "live_blocks": self._used(),
            "total_blocks": self.total_blocks,
            "block_tokens": self.block_tokens,
            "pressure_events": self.pressure_events,
            "max_live_blocks": self.max_live_blocks,
            "mean_live_blocks": self.live_block_sum / max(1, self.live_block_samples),
            "reprefill_rate_recent": self.reprefill_rate(),
        }
