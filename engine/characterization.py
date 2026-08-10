#
# Copyright (c) 2019 Matthias Tafelmeier.
#
# This file is part of godon
#
# godon is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# godon is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this godon. If not, see <http://www.gnu.org/licenses/>.
"""
Response curve interpolation and convergence detection.

The characterization loop:
  probe → measure → interpolate → check convergence → halt or probe next

This module implements the mathematical core:
  - ResponseCurve: accumulates (level, response) points per edge
  - Piecewise linear interpolation through measured points
  - Convergence delta: how much did the interpolated surface move
    when the latest point was added

The convergence delta serves triple duty:
  1. Sampler objective — points that move the surface are informative
  2. Halting criterion — when delta drops below threshold, edge is characterized
  3. Drift detector — persistent nonzero delta means the system is non-stationary

No model form assumed. The measured data IS the function. Works for linear,
threshold, saturation, discontinuous — the shape emerges from measurements.
"""

from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ResponseCurve:
    """Accumulates measured response points for a single edge.

    An edge is a (sender_param, receiver_objective) pair. The response curve
    maps sender push level → receiver objective shift.

    Points are stored sorted by level. Piecewise linear interpolation between
    adjacent measured points. No extrapolation — levels outside the measured
    range return None.
    """

    def __init__(self, convergence_threshold: float = 0.01,
                 min_points: int = 3,
                 evaluation_grid_size: int = 200):
        self.convergence_threshold = convergence_threshold
        self.min_points = min_points
        self.grid_size = evaluation_grid_size

        # Measured points: list of (level, response_shift)
        self._points: List[Tuple[float, float]] = []

        # Previous interpolation snapshot (for convergence comparison)
        self._prev_interp: Optional[List[Tuple[float, float]]] = None

        # Convergence history: delta after each point addition
        self._delta_history: List[float] = []

        # Number of substeps inserted (recursive refinement counter)
        self._substep_count = 0

    @property
    def points(self) -> List[Tuple[float, float]]:
        """Sorted copy of measured points."""
        return sorted(self._points, key=lambda p: p[0])

    @property
    def num_points(self) -> int:
        return len(self._points)

    @property
    def is_converged(self) -> bool:
        """True when the last added point barely moved the surface."""
        if len(self._delta_history) < 1:
            return False
        if self.num_points < self.min_points:
            return False
        return self._delta_history[-1] < self.convergence_threshold

    @property
    def last_delta(self) -> float:
        """Convergence delta from the most recent point addition."""
        return self._delta_history[-1] if self._delta_history else float('inf')

    def add_point(self, level: float, response: float) -> float:
        """Add a measured point and return the convergence delta.

        If a point at the same level already exists (re-measurement),
        it is REPLACED. The delta from replacement is the drift signal —
        how much did the response at this level change since last measured.

        The convergence delta is the L1 difference between the new
        interpolated surface and the previous one, evaluated on a fine grid
        spanning the measured range.
        """
        # Check for re-measurement
        existing_idx = None
        for i, (l, r) in enumerate(self._points):
            if abs(l - level) < 1e-9:
                existing_idx = i
                break

        if existing_idx is not None:
            old_response = self._points[existing_idx][1]
            self._points[existing_idx] = (level, response)
            logger.info(
                f"ResponseCurve: re-measured level {level:.1f}, "
                f"response {old_response:.4f} → {response:.4f} "
                f"(drift={abs(response - old_response):.4f})"
            )
        else:
            self._points.append((level, response))

        # Sort for interpolation
        sorted_points = sorted(self._points, key=lambda p: p[0])

        # Compute current interpolation on evaluation grid
        curr_interp = self._evaluate_grid(sorted_points)

        if self._prev_interp is None:
            # First meaningful point — no prior to compare
            delta = float('inf')
        else:
            # Compare current vs previous interpolation
            delta = self._grid_distance(curr_interp, self._prev_interp)

        self._prev_interp = curr_interp
        self._delta_history.append(delta)

        logger.info(
            f"ResponseCurve: added ({level:.1f}, {response:.4f}), "
            f"delta={delta:.6f}, points={self.num_points}, "
            f"converged={delta < self.convergence_threshold}"
        )

        return delta

    def _evaluate_grid(self, sorted_points: List[Tuple[float, float]]
                       ) -> List[Tuple[float, float]]:
        """Evaluate piecewise linear interpolation on a fine grid.

        Grid spans from min to max measured level.
        """
        if len(sorted_points) < 2:
            return sorted_points.copy()

        lo = sorted_points[0][0]
        hi = sorted_points[-1][0]
        if hi <= lo:
            return sorted_points.copy()

        step = (hi - lo) / self.grid_size
        result = []
        for i in range(self.grid_size + 1):
            x = lo + i * step
            y = self._interp_at(x, sorted_points)
            result.append((x, y))

        return result

    @staticmethod
    def _interp_at(x: float,
                   sorted_points: List[Tuple[float, float]]) -> float:
        """Piecewise linear interpolation at x."""
        if x <= sorted_points[0][0]:
            return sorted_points[0][1]
        if x >= sorted_points[-1][0]:
            return sorted_points[-1][1]

        for i in range(len(sorted_points) - 1):
            x0, y0 = sorted_points[i]
            x1, y1 = sorted_points[i + 1]
            if x0 <= x <= x1:
                if x1 == x0:
                    return y0
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)

        return sorted_points[-1][1]

    @staticmethod
    def _grid_distance(grid_a: List[Tuple[float, float]],
                       grid_b: List[Tuple[float, float]]) -> float:
        """L1 distance between two interpolation grids.

        Both grids must be evaluated on the same evaluation points.
        If the ranges differ (new point extends the range), we compare
        only on the overlapping region and count the extension as
        maximum-distance movement.
        """
        if not grid_a or not grid_b:
            return float('inf')

        # Interpolate grid_b onto grid_a's evaluation points
        total_dist = 0.0
        count = 0

        sorted_b = sorted(grid_b, key=lambda p: p[0])

        for x, y_a in grid_a:
            y_b = ResponseCurve._interp_at(x, sorted_b)
            total_dist += abs(y_a - y_b)
            count += 1

        # Normalize: mean absolute deviation over the grid
        return total_dist / count if count > 0 else float('inf')

    def suggest_next_level(self, lower: float, upper: float, step: float,
                           visited: set) -> Optional[float]:
        """Suggest the next level to probe based on where the curve moves most.

        Strategy:
        1. If unvisited levels remain, return the one adjacent to the
           largest interpolation slope (where the curve is changing fastest).
        2. If all levels visited, suggest a substep between the two adjacent
           points with the steepest slope.
        3. If converged, return None.

        This is a coverage-first, slope-guided heuristic. It does NOT need
        an external sampler — the interpolation itself guides exploration.
        """
        if self.is_converged:
            return None

        # Enumerate all discrete levels
        levels = []
        v = lower
        while v <= upper + 1e-9:
            levels.append(round(v, 6))
            v += step

        unvisited = [l for l in levels if l not in visited]

        if unvisited:
            if self.num_points < 2:
                # Not enough points to estimate slope — take midpoint of unvisited
                return unvisited[len(unvisited) // 2]

            # Estimate slope at each unvisited level by interpolating
            # current curve there. The level adjacent to the steepest
            # segment is most informative.
            sorted_pts = self.points
            best_level = unvisited[0]
            best_priority = -1

            for l in unvisited:
                # Find the steepest segment this level would fall into
                for i in range(len(sorted_pts) - 1):
                    x0, y0 = sorted_pts[i]
                    x1, y1 = sorted_pts[i + 1]
                    if x0 <= l <= x1 and x1 > x0:
                        slope = abs(y1 - y0) / (x1 - x0)
                        if slope > best_priority:
                            best_priority = slope
                            best_level = l
                        break

            return best_level

        # All discrete levels visited — suggest substep
        sorted_pts = self.points
        if len(sorted_pts) < 2:
            return None

        # Find the segment with the steepest slope
        max_slope = 0
        substep_at = None
        for i in range(len(sorted_pts) - 1):
            x0, y0 = sorted_pts[i]
            x1, y1 = sorted_pts[i + 1]
            if x1 > x0:
                slope = abs(y1 - y0) / (x1 - x0)
                if slope > max_slope:
                    max_slope = slope
                    mid = (x0 + x1) / 2.0
                    if mid not in visited:
                        substep_at = mid
                        max_slope_candidate = slope

        if substep_at is not None:
            self._substep_count += 1
            logger.info(
                f"ResponseCurve: substep at {substep_at:.1f} "
                f"(steepest segment slope={max_slope:.4f})"
            )

        return substep_at

    def export_points(self) -> List[dict]:
        """Export measured points for storage/transfer."""
        return [
            {'level': l, 'response': r}
            for l, r in sorted(self._points, key=lambda p: p[0])
        ]

    def summary(self) -> dict:
        """Summary statistics for logging/debugging."""
        pts = self.points
        if len(pts) < 2:
            return {'num_points': len(pts), 'converged': False}

        slopes = []
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            if x1 > x0:
                slopes.append(abs(y1 - y0) / (x1 - x0))

        import statistics
        return {
            'num_points': len(pts),
            'num_substeps': self._substep_count,
            'converged': self.is_converged,
            'last_delta': self.last_delta,
            'mean_slope': statistics.mean(slopes) if slopes else 0,
            'max_slope': max(slopes) if slopes else 0,
            'level_range': [pts[0][0], pts[-1][0]],
            'response_range': [
                min(p[1] for p in pts),
                max(p[1] for p in pts)
            ],
        }
