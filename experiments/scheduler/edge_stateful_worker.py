"""Stateful JSONL draft worker for scheduler experiments."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_snapshot(path: str) -> str:
    path = os.path.expanduser(path)
    snapshots = os.path.join(path, "snapshots")
    if os.path.isdir(snapshots):
        names = sorted(d for d in os.listdir(snapshots) if os.path.isdir(os.path.join(snapshots, d)))
        if names:
            return os.path.join(snapshots, names[0])
    return path


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def seq_len(past) -> int:
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return past.get_seq_length()
    return past[0][0].shape[-2]


def crop(past, new_len: int):
    if past is None:
        return None
    if hasattr(past, "crop"):
        past.crop(new_len)
        return past
    cropped = []
    for k_layer, v_layer in past:
        cropped.append((k_layer[:, :, :new_len, :].contiguous(), v_layer[:, :, :new_len, :].contiguous()))
    return tuple(cropped)


def load_model(path: str):
    model_path = resolve_snapshot(path)
    log(f"[edge] loading {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to("cuda").eval()
    log(f"[edge] loaded params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return tok, model


@torch.inference_mode()
def warmup(model, vocab_size: int) -> None:
    ids = torch.randint(0, vocab_size, (1, 32), device="cuda")
    for _ in range(2):
        _ = model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()


@torch.inference_mode()
def prefill(model, states: Dict[str, object], state_id: str, input_ids) -> float:
    t0 = time.perf_counter()
    ids = torch.tensor(input_ids, dtype=torch.long, device="cuda").view(1, -1)
    out = model(input_ids=ids, use_cache=True)
    states[state_id] = out.past_key_values
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


@torch.inference_mode()
def draft(model, states: Dict[str, object], state_id: str, last_token: int, k: int):
    t0 = time.perf_counter()
    if state_id not in states:
        raise RuntimeError(f"unknown state_id {state_id}")
    past = states[state_id]
    cur = torch.tensor([[last_token]], dtype=torch.long, device="cuda")
    tokens = []
    for _ in range(k):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1, :].argmax().item())
        tokens.append(nxt)
        cur = torch.tensor([[nxt]], dtype=torch.long, device="cuda")
    states[state_id] = past
    torch.cuda.synchronize()
    return tokens, (time.perf_counter() - t0) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    tok, model = load_model(args.model)
    warmup(model, tok.vocab_size)
    states: Dict[str, object] = {}
    log("[edge] ready")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = {}
        try:
            req = json.loads(line)
            cmd = req.get("cmd", "draft")
            state_id = str(req.get("state_id", req.get("id", "")))
            if cmd == "stop":
                resp = {"ok": True, "stopped": True}
                print(json.dumps(resp), flush=True)
                break
            if cmd == "prefill":
                elapsed = prefill(model, states, state_id, req["input_ids"])
                resp = {"ok": True, "id": req.get("id"), "edge_elapsed_ms": elapsed, "seq_len": seq_len(states[state_id])}
            elif cmd == "draft":
                tokens, elapsed = draft(model, states, state_id, int(req["last_token"]), int(req["k"]))
                resp = {"ok": True, "id": req.get("id"), "tokens": tokens, "edge_elapsed_ms": elapsed, "seq_len": seq_len(states[state_id])}
            elif cmd == "crop":
                if state_id in states:
                    states[state_id] = crop(states[state_id], int(req["seq_len"]))
                resp = {"ok": True, "id": req.get("id"), "seq_len": seq_len(states.get(state_id))}
            elif cmd == "release":
                states.pop(state_id, None)
                resp = {"ok": True, "id": req.get("id"), "released": state_id}
            else:
                raise RuntimeError(f"unknown cmd {cmd}")
        except Exception as exc:
            resp = {"ok": False, "id": req.get("id"), "error": repr(exc)}
        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
