"""Deterministic coverage walk over per-parameter level spaces.

Coverage contract: every eligible parameter's levels get measured —
worst case down to the resolution floor for floats, full enumeration
for ints. Retirement (converged params) only accelerates; it never
licenses skipping.

Selection rule (one line): the next level for a parameter is the
candidate FARTHEST from everything already measured on that parameter;
ties go to the candidate closest to the range center, then to the
higher level. On a uniform grid this yields midpoint -> edges ->
quarters — the low-discrepancy prefix computed directly instead of
sampled.

No sampler, no RNG, no seed: the walk is a pure function of
(bounds, floor, retirement, progress).
"""

from typing import Dict, List, Optional, Set, Tuple

# name -> (lower, upper, is_int)
Bounds = Dict[str, Tuple[float, float, bool]]


class CoverageWalk:
    """Round-robin farthest-point walk over (param, level) cells."""

    def __init__(self, bounds: Bounds):
        self.bounds: Bounds = dict(bounds)
        # Resolution floor per float param (initial: range / 4).
        # Int params enumerate and have no floor.
        self._floors: Dict[str, float] = {
            name: (hi - lo) / 4.0
            for name, (lo, hi, is_int) in self.bounds.items()
            if not is_int and hi > lo
        }
        self._measured: Dict[str, List[float]] = {n: [] for n in self.bounds}
        self._rr = 0  # round-robin position

    # ── candidates ───────────────────────────────────────────────────

    def _candidates(self, name: str) -> List[float]:
        lo, hi, is_int = self.bounds[name]
        meas = self._measured[name]
        if is_int:
            return [float(v) for v in range(int(round(lo)), int(round(hi)) + 1)
                    if float(v) not in meas]
        # Floats: bounds plus midpoints of all gaps (measured ∪ bounds).
        anchors = sorted(meas + [lo, hi])
        cands = {round(lo, 6), round(hi, 6)}
        for a, b in zip(anchors, anchors[1:]):
            cands.add(round((a + b) / 2.0, 6))
        return [c for c in cands if c not in meas]

    def _distance(self, level: float, meas: List[float], span: float) -> float:
        if not meas:
            return span
        return min(abs(level - m) for m in meas)

    def _next_level(self, name: str) -> Optional[float]:
        lo, hi, is_int = self.bounds[name]
        meas = self._measured[name]
        cands = self._candidates(name)
        # Resolution floor: stop splitting when a new float level would
        # sit closer than the floor to an existing measurement. Ints
        # enumerate fully regardless of floor.
        if not is_int and meas:
            floor = self._floors.get(name, 0.0)
            cands = [c for c in cands
                     if self._distance(c, meas, hi - lo) >= floor]
        if not cands:
            return None
        center = (lo + hi) / 2.0

        def rank(lv: float) -> Tuple[float, float, float]:
            # farthest from measured, closest to center, higher
            return (self._distance(lv, meas, hi - lo), -abs(lv - center), lv)

        return max(cands, key=rank)

    # ── public API ───────────────────────────────────────────────────

    def next_probe(self, skip: Set[str]) -> Optional[Tuple[str, float]]:
        """Next (param, level) to measure, or None when the walk is done.

        `skip` holds converged param names (retirement).
        """
        eligible = [n for n in self.bounds if n not in skip]
        if not eligible:
            return None
        for _ in range(len(eligible)):
            name = eligible[self._rr % len(eligible)]
            self._rr += 1
            level = self._next_level(name)
            if level is not None:
                self._measured[name].append(level)
                return name, level
        return None

    def can_probe(self, skip: Set[str]) -> bool:
        """Pure check: would next_probe return a cell? (no mutation)"""
        return any(
            self._next_level(n) is not None
            for n in self.bounds if n not in skip
        )

    def refine(self) -> None:
        """Halve resolution floors — next pass measures the new midpoints."""
        for name in self._floors:
            self._floors[name] /= 2.0

    def status(self) -> Dict[str, Dict]:
        """Per-param walk state for logging/status endpoints."""
        out = {}
        for name, (lo, hi, is_int) in self.bounds.items():
            floor = self._floors.get(name)
            if is_int:
                total = int(round(hi)) - int(round(lo)) + 1
            elif floor:
                total = int(round((hi - lo) / floor)) + 1
            else:
                total = None
            out[name] = {
                'step': floor if not is_int else 1.0,
                'levels_total': total,
                'levels_measured': len(self._measured[name]),
                'levels': list(self._measured[name]),
            }
        return out
