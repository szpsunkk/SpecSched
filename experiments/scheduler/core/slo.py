"""SLO budget and goodput accounting.

SLO definition (matches mainstream LLM-serving literature):
    - TTFT SLO: first token must arrive within `slo_ttft_ms` after arrival.
    - TPOT SLO: average per-token latency over the whole generation must stay
      under `slo_tpot_ms`.
A request counts toward SLO attainment iff BOTH constraints are met.
Goodput counts only tokens from SLO-attaining requests.

Presets (tight / medium / loose) are intentionally relative to the measured
single-request service time so they remain meaningful across hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


PRESETS = {
    "tight":  (1.5, 1.5),   # ttft_mul, tpot_mul over single-request reference
    "medium": (3.0, 3.0),
    "loose":  (6.0, 6.0),
    # Used on a single RTX 3090 Ti for higher-N real sweeps. The original
    # paper used A100-class hardware; this preset keeps SLO-goodput from
    # collapsing to all-zero while still penalizing overload at large N.
    "relaxed": (20.0, 20.0),
}


@dataclass
class SLOConfig:
    name: str
    ttft_ms: Optional[float]
    tpot_ms: Optional[float]

    @property
    def enabled(self) -> bool:
        return self.ttft_ms is not None and self.tpot_ms is not None


def make_slo(
    name: str,
    ref_ttft_ms: Optional[float],
    ref_tpot_ms: Optional[float],
) -> SLOConfig:
    if name in (None, "", "off", "none"):
        return SLOConfig("off", None, None)
    if name not in PRESETS:
        raise ValueError(f"unknown SLO preset {name!r}; pick from {list(PRESETS)} or 'off'")
    if ref_ttft_ms is None or ref_tpot_ms is None:
        raise ValueError("SLO presets require reference TTFT/TPOT from calibration")
    ttft_mul, tpot_mul = PRESETS[name]
    return SLOConfig(name=name, ttft_ms=ref_ttft_ms * ttft_mul, tpot_ms=ref_tpot_ms * tpot_mul)


def attains(ttft_ms: float, tpot_ms: float, slo: SLOConfig) -> bool:
    if not slo.enabled:
        return True
    return ttft_ms <= slo.ttft_ms and tpot_ms <= slo.tpot_ms
