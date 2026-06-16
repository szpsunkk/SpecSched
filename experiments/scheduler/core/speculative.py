"""Correct speculative-decoding primitives with target KV-cache reuse.

INVARIANT held between speculative rounds for every ReqState:
    target_past, draft_past  encode  prompt + generated[:-1]
    last_token  ==  generated[-1]

A speculative round (k drafts):
    1. draft_propose(k): k forwards through the draft model, returning
       [p_1..p_k]. Leaves draft_past at length L_d + k where L_d is the
       length entering the round.
    2. target_verify(proposals): one target forward over [last_token, p_1..p_k]
       (length k+1). Reads k+1 logits, accepts a prefix, then either
       bonus-extends (if all k accepted) or replaces the first-mismatch with
       target's argmax. Reconciles target_past and draft_past back to invariant.

Greedy verification (matches argmax target distribution).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover - older transformers
    DynamicCache = None  # type: ignore


def _is_remote_draft(draft_model) -> bool:
    return bool(getattr(draft_model, "remote_draft", False))


def normalize_cache(past):
    if past is None or DynamicCache is None or isinstance(past, DynamicCache):
        return past
    if isinstance(past, tuple) and hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(past)
    return past


@dataclass
class ReqState:
    req_id: int
    prompt_ids: torch.Tensor
    arrival_s: float
    max_new_tokens: int
    slo_ttft_ms: Optional[float]
    slo_tpot_ms: Optional[float]
    bucket: str
    tier: str = "default"
    phase: str = ""
    source_id: str = ""
    generated: List[int] = field(default_factory=list)
    target_past: Optional[object] = None
    draft_past: Optional[object] = None
    last_token: Optional[int] = None
    admit_s: Optional[float] = None
    first_token_s: Optional[float] = None
    finish_s: Optional[float] = None
    target_forwards: int = 0
    draft_forwards: int = 0
    accepted_total: int = 0
    drafted_total: int = 0
    alpha_pos: List[float] = field(default_factory=lambda: [0.6] * 16)
    alpha_pos_count: List[int] = field(default_factory=lambda: [0] * 16)
    reprefill_count: int = 0
    service_target_ms: float = 0.0
    service_draft_ms: float = 0.0
    service_other_ms: float = 0.0
    last_step_idx: int = -1
    cache_valid: bool = True
    scheduler_now_s: float = 0.0

    def finished(self) -> bool:
        return self.finish_s is not None


def _seq_len(past) -> int:
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return past.get_seq_length()
    return past[0][0].shape[-2]


def _crop(past, new_len: int):
    if past is None:
        return None
    if hasattr(past, "crop"):
        past.crop(new_len)
        return past
    cropped = []
    for k_layer, v_layer in past:
        cropped.append(
            (k_layer[:, :, :new_len, :].contiguous(), v_layer[:, :, :new_len, :].contiguous())
        )
    return tuple(cropped)


def update_position_acceptance(state: ReqState, drafted: int, accepted: int) -> None:
    """Update per-draft-position acceptance estimates from one verify step."""
    if drafted <= 0:
        return
    need = max(0, drafted - len(state.alpha_pos))
    if need:
        state.alpha_pos.extend([0.6] * need)
        state.alpha_pos_count.extend([0] * need)
    gamma = 0.3
    observed = min(drafted, accepted + (1 if accepted < drafted else 0))
    for ell in range(observed):
        z = 1.0 if ell < accepted else 0.0
        state.alpha_pos[ell] = (1.0 - gamma) * state.alpha_pos[ell] + gamma * z
        state.alpha_pos_count[ell] += 1


@torch.inference_mode()
def prefill_target(target_model, prompt_ids: torch.Tensor) -> Tuple[object, int, float]:
    t0 = time.perf_counter()
    ids = prompt_ids.to("cuda").view(1, -1)
    init_past = DynamicCache() if DynamicCache is not None else None
    out = target_model(input_ids=ids, past_key_values=init_past, use_cache=True)
    first = int(out.logits[0, -1].argmax().item())
    torch.cuda.synchronize()
    return normalize_cache(out.past_key_values), first, (time.perf_counter() - t0) * 1000.0


@torch.inference_mode()
def prefill_draft(draft_model, prompt_ids: torch.Tensor, req_id: Optional[int] = None) -> Tuple[object, float]:
    if _is_remote_draft(draft_model):
        if req_id is None:
            raise RuntimeError("remote draft prefill requires req_id")
        return draft_model.prefill(int(req_id), prompt_ids.cpu().tolist())
    t0 = time.perf_counter()
    ids = prompt_ids.to("cuda").view(1, -1)
    init_past = DynamicCache() if DynamicCache is not None else None
    out = draft_model(input_ids=ids, past_key_values=init_past, use_cache=True)
    torch.cuda.synchronize()
    return normalize_cache(out.past_key_values), (time.perf_counter() - t0) * 1000.0


@torch.inference_mode()
def draft_propose(draft_model, state: ReqState, k: int) -> Tuple[List[int], float]:
    if k <= 0 or state.last_token is None or state.draft_past is None:
        return [], 0.0
    if _is_remote_draft(draft_model):
        proposals, elapsed = draft_model.draft(state.req_id, int(state.last_token), k)
        state.draft_forwards += len(proposals)
        state.service_draft_ms += elapsed
        state.service_other_ms += float(getattr(draft_model, "last_overhead_ms", 0.0))
        state.draft_past = True
        return proposals, elapsed
    t0 = time.perf_counter()
    proposals: List[int] = []
    cur = torch.tensor([[state.last_token]], dtype=torch.long, device="cuda")
    past = normalize_cache(state.draft_past)
    for _ in range(k):
        out = draft_model(input_ids=cur, past_key_values=past, use_cache=True)
        past = normalize_cache(out.past_key_values)
        nxt = int(out.logits[0, -1].argmax().item())
        proposals.append(nxt)
        cur = torch.tensor([[nxt]], dtype=torch.long, device="cuda")
        state.draft_forwards += 1
    state.draft_past = past
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000.0
    state.service_draft_ms += elapsed
    return proposals, elapsed


@torch.inference_mode()
def target_verify(
    target_model,
    draft_model,
    state: ReqState,
    proposals: List[int],
    remaining: int,
) -> Tuple[int, List[int], float]:
    """One verification forward + commit + cache reconciliation."""
    t0 = time.perf_counter()
    prompt_len = int(state.prompt_ids.numel())

    if not proposals:
        cur = torch.tensor([[state.last_token]], dtype=torch.long, device="cuda")
        out = target_model(input_ids=cur, past_key_values=state.target_past, use_cache=True)
        state.target_past = normalize_cache(out.past_key_values)
        nxt = int(out.logits[0, -1].argmax().item())
        state.generated.append(nxt)
        state.last_token = nxt
        state.target_forwards += 1
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0
        state.service_target_ms += elapsed
        return 0, [nxt], elapsed

    # Capture old cache lengths BEFORE the forward (DynamicCache is mutated in place).
    old_target_len = _seq_len(state.target_past)

    seq_ids = [state.last_token] + list(proposals)
    cur = torch.tensor([seq_ids], dtype=torch.long, device="cuda")
    out = target_model(input_ids=cur, past_key_values=state.target_past, use_cache=True)
    logits = out.logits[0]
    new_past = normalize_cache(out.past_key_values)
    state.target_forwards += 1

    accepted = 0
    appended: List[int] = []
    for j, prop in enumerate(proposals):
        pred = int(logits[j].argmax().item())
        if pred == prop:
            accepted += 1
            appended.append(prop)
        else:
            appended.append(pred)
            break
    if accepted == len(proposals):
        bonus = int(logits[len(proposals)].argmax().item())
        appended.append(bonus)

    if remaining < len(appended):
        appended = appended[:remaining]

    desired_target_len = old_target_len + len(appended)
    if _seq_len(new_past) > desired_target_len:
        new_past = _crop(new_past, desired_target_len)
    state.target_past = new_past

    state.generated.extend(appended)
    state.last_token = appended[-1]
    update_position_acceptance(state, len(proposals), accepted)
    state.accepted_total += accepted
    state.drafted_total += len(proposals)

    if _is_remote_draft(draft_model):
        desired_draft_len = prompt_len + len(state.generated) - 1
        draft_model.crop(state.req_id, desired_draft_len)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0
        state.service_target_ms += elapsed
        return accepted, appended, elapsed

    desired_draft_len = prompt_len + len(state.generated) - 1
    current_draft_len = _seq_len(state.draft_past)
    if current_draft_len > desired_draft_len:
        state.draft_past = _crop(state.draft_past, desired_draft_len)
    elif current_draft_len < desired_draft_len:
        missing = desired_draft_len - current_draft_len
        tail = state.generated[-(missing + 1):-1]
        ids = torch.tensor([tail], dtype=torch.long, device="cuda")
        out = draft_model(input_ids=ids, past_key_values=state.draft_past, use_cache=True)
        state.draft_past = normalize_cache(out.past_key_values)
        state.draft_forwards += len(tail)

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000.0
    state.service_target_ms += elapsed
    return accepted, appended, elapsed


@torch.inference_mode()
def target_step_no_spec(target_model, state: ReqState) -> Tuple[List[int], float]:
    accepted, appended, elapsed = target_verify(target_model, None, state, [], remaining=1024)
    return appended, elapsed


def free_state(state: ReqState) -> None:
    state.target_past = None
    state.draft_past = None


# ---------------------------------------------------------------------------
# Padding-aware batched target verification.
# ---------------------------------------------------------------------------
#
# The per-request `target_verify` does one forward of length k+1 per request.
# When the policy chooses the same k for every request in a batch, we can pack
# B requests' (last_token + drafts) into a single (B, k+1) target forward by:
#
#   1. Front-padding each request's past_key_values to S_max = max_i S_i;
#   2. Concatenating along batch_dim into one DynamicCache;
#   3. Passing per-row `position_ids` and an attention_mask that hides the
#      padding region [0, S_max - S_i) for row i;
#   4. After the forward, the cache has shape [B, H, S_max + k+1, D]. We then
#      split it back into per-request DynamicCaches, crop each to its true
#      length, and reassign to the corresponding ReqState.
#
# Requires `transformers.cache_utils.DynamicCache`. Falls back to per-request
# verify if the cache class is unavailable or any request's past is not a
# DynamicCache instance.


def _pad_front_cache(past: "DynamicCache", target_len: int) -> "DynamicCache":
    new_cache = DynamicCache()
    for li in range(len(past.key_cache)):
        k = past.key_cache[li]
        v = past.value_cache[li]
        S = k.shape[-2]
        pad = target_len - S
        if pad > 0:
            k = torch.nn.functional.pad(k, (0, 0, pad, 0))
            v = torch.nn.functional.pad(v, (0, 0, pad, 0))
        new_cache.update(k, v, li)
    return new_cache


def _stack_caches(caches: List["DynamicCache"]) -> "DynamicCache":
    batched = DynamicCache()
    n_layers = len(caches[0].key_cache)
    for li in range(n_layers):
        bk = torch.cat([c.key_cache[li] for c in caches], dim=0)
        bv = torch.cat([c.value_cache[li] for c in caches], dim=0)
        batched.update(bk, bv, li)
    return batched


def _extract_row_cache(
    batched: "DynamicCache",
    row_idx: int,
    *,
    old_seq_len: int,
    kept_new_tokens: int,
    padded_seq_len: int,
) -> "DynamicCache":
    """Return one request's unpadded cache after a padded batched verify.

    The batched cache layout for each row is:
        [front padding][old true cache][k+1 verified tokens]

    If only part of the verified sequence is committed, we must keep the true
    old cache plus the committed prefix of the new tokens. Taking the last
    `old_seq_len + kept_new_tokens` positions would incorrectly drop old
    context whenever there was front padding or an early reject.
    """
    out = DynamicCache()
    for li in range(len(batched.key_cache)):
        full_k = batched.key_cache[li][row_idx:row_idx + 1]  # (1, H, S_total, D)
        full_v = batched.value_cache[li][row_idx:row_idx + 1]
        pad = padded_seq_len - old_seq_len
        if pad < 0:
            raise RuntimeError(
                f"_extract_row_cache: old_seq_len {old_seq_len} > padded_seq_len {padded_seq_len}"
            )
        old_start = pad
        old_end = padded_seq_len
        new_start = padded_seq_len
        new_end = padded_seq_len + kept_new_tokens
        old_k = full_k[:, :, old_start:old_end, :]
        old_v = full_v[:, :, old_start:old_end, :]
        new_k = full_k[:, :, new_start:new_end, :]
        new_v = full_v[:, :, new_start:new_end, :]
        out.update(
            torch.cat([old_k, new_k], dim=-2).contiguous(),
            torch.cat([old_v, new_v], dim=-2).contiguous(),
            li,
        )
    return out


@torch.inference_mode()
def batched_target_verify(
    target_model,
    draft_model,
    states: List[ReqState],
    proposals_per_req: List[List[int]],
    remaining_per_req: List[int],
) -> Tuple[List[int], List[List[int]], float]:
    """One padded target forward over B requests with identical proposal length.

    Returns (accepted_per_req, appended_per_req, elapsed_ms).

    Pre-conditions:
        - len(states) == len(proposals_per_req) >= 1
        - all proposals have the same length k > 0
        - every state.target_past is a DynamicCache (raises otherwise)
        - target_model is a local HF causal LM (remote draft path uses the
          per-request fallback in engine; remote target is not supported)
    """
    assert len(states) == len(proposals_per_req) == len(remaining_per_req)
    assert len(states) >= 1, "batched_target_verify needs at least one state"
    k = len(proposals_per_req[0])
    assert k > 0, "batched_target_verify requires k > 0"
    for p in proposals_per_req:
        assert len(p) == k, "all proposals must share the same length"
    if DynamicCache is None:
        raise RuntimeError("DynamicCache unavailable; falling back to per-request path")
    for s in states:
        s.target_past = normalize_cache(s.target_past)
        if not isinstance(s.target_past, DynamicCache):
            raise RuntimeError(
                "batched_target_verify requires target_past to be a DynamicCache; "
                f"got {type(s.target_past).__name__}"
            )

    B = len(states)
    t0 = time.perf_counter()
    device = next(target_model.parameters()).device

    seq_lens = [_seq_len(s.target_past) for s in states]
    S_max = max(seq_lens)

    # 1. Pad each past to S_max, then stack along batch dim.
    padded = [_pad_front_cache(s.target_past, S_max) for s in states]
    batched_past = _stack_caches(padded)

    # 2. Build (B, k+1) input: [last_token, p1..pk] per row.
    seq_ids = [[s.last_token] + list(props) for s, props in zip(states, proposals_per_req)]
    input_ids = torch.tensor(seq_ids, dtype=torch.long, device=device)

    # 3. Per-row position_ids (each row starts at its own true S_i).
    pos_rows = [list(range(S, S + k + 1)) for S in seq_lens]
    position_ids = torch.tensor(pos_rows, dtype=torch.long, device=device)

    # 4. Attention mask: 1 on real-past slots [S_max-S_i, S_max) and on the
    #    k+1 new positions; 0 on the front-pad slots.
    attn_mask = torch.zeros(B, S_max + k + 1, dtype=torch.long, device=device)
    for i, S in enumerate(seq_lens):
        attn_mask[i, S_max - S:] = 1

    cache_position = torch.arange(S_max, S_max + k + 1, dtype=torch.long, device=device)

    out = target_model(
        input_ids=input_ids,
        past_key_values=batched_past,
        attention_mask=attn_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=True,
    )
    logits = out.logits  # (B, k+1, V)
    new_batched_past = normalize_cache(out.past_key_values)  # cache now has S_max + k+1 length
    for s in states:
        s.target_forwards += 1

    # 5. Per-row accept/reject, with the same logic as `target_verify`.
    accepted_list: List[int] = []
    appended_list: List[List[int]] = []
    new_seq_lens: List[int] = []
    for i, (s, proposals, remaining) in enumerate(zip(states, proposals_per_req, remaining_per_req)):
        row_logits = logits[i]
        accepted = 0
        appended: List[int] = []
        for j, prop in enumerate(proposals):
            pred = int(row_logits[j].argmax().item())
            if pred == prop:
                accepted += 1
                appended.append(prop)
            else:
                appended.append(pred)
                break
        if accepted == len(proposals):
            bonus = int(row_logits[len(proposals)].argmax().item())
            appended.append(bonus)
        if remaining < len(appended):
            appended = appended[:remaining]

        s.generated.extend(appended)
        s.last_token = appended[-1]
        update_position_acceptance(s, len(proposals), accepted)
        s.accepted_total += accepted
        s.drafted_total += len(proposals)

        accepted_list.append(accepted)
        appended_list.append(appended)
        new_seq_lens.append(seq_lens[i] + len(appended))

    # 6. Slice the batched cache back into per-request DynamicCaches.
    for i, s in enumerate(states):
        s.target_past = _extract_row_cache(
            new_batched_past,
            i,
            old_seq_len=seq_lens[i],
            kept_new_tokens=len(appended_list[i]),
            padded_seq_len=S_max,
        )

    # 7. Reconcile each request's draft cache (per-request path; local draft only).
    if draft_model is not None and not _is_remote_draft(draft_model):
        for s, appended in zip(states, appended_list):
            prompt_len = int(s.prompt_ids.numel())
            desired_draft_len = prompt_len + len(s.generated) - 1
            current_draft_len = _seq_len(s.draft_past)
            if current_draft_len > desired_draft_len:
                s.draft_past = _crop(s.draft_past, desired_draft_len)
            elif current_draft_len < desired_draft_len:
                missing = desired_draft_len - current_draft_len
                tail = s.generated[-(missing + 1):-1]
                if tail:
                    ids = torch.tensor([tail], dtype=torch.long, device=device)
                    out_d = draft_model(
                        input_ids=ids, past_key_values=s.draft_past, use_cache=True
                    )
                    s.draft_past = normalize_cache(out_d.past_key_values)
                    s.draft_forwards += len(tail)
    elif draft_model is not None and _is_remote_draft(draft_model):
        for s in states:
            prompt_len = int(s.prompt_ids.numel())
            desired_draft_len = prompt_len + len(s.generated) - 1
            draft_model.crop(s.req_id, desired_draft_len)

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000.0
    # Per-request service_target_ms records the share of the batched forward.
    share = elapsed / B
    for s in states:
        s.service_target_ms += share
    return accepted_list, appended_list, elapsed
