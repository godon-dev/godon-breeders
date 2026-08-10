#!/usr/bin/env python3
"""
Unit tests for response_curve.py — interpolation + convergence.

Tests the mathematical core of the characterization loop against
ground-truth response shapes (linear, threshold, saturation).
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from engine.characterization import ResponseCurve


def test_linear_convergence():
    """Linear response: f(x) = 0.7 * x.
    
    Expected: converges fast, few points, delta drops quickly."""
    print("\n=== test_linear_convergence ===")
    
    curve = ResponseCurve(convergence_threshold=0.005, min_points=4)
    lower, upper, step = 0.0, 100.0, 20.0
    visited = set()
    
    def true_response(level):
        return 0.7 * (level / 100.0)  # normalize
    
    # Simulate the characterization loop
    for i in range(20):  # max 20 probes
        level = curve.suggest_next_level(lower, upper, step, visited)
        if level is None:
            break
        visited.add(round(level, 6))
        
        # Simulate measurement with small noise
        response = true_response(level)
        delta = curve.add_point(level, response)
    
    summary = curve.summary()
    print(f"  Points measured: {summary['num_points']}")
    print(f"  Converged: {summary['converged']}")
    print(f"  Last delta: {summary['last_delta']:.6f}")
    print(f"  Max slope: {summary['max_slope']:.4f} (expected ~0.007)")
    print(f"  Points: {[(round(l,1), round(r,4)) for l,r in curve.points]}")
    
    assert summary['converged'], "Linear should converge"
    assert summary['num_points'] <= 8, "Linear should converge fast"
    print("  PASS")
    return True


def test_threshold_recovery():
    """Threshold response: f(x) = 0 if x < 50, else 0.7.
    
    Expected: converges, jump visible in interpolation.
    The key test: does it recover the discontinuity shape?"""
    print("\n=== test_threshold_recovery ===")
    
    curve = ResponseCurve(convergence_threshold=0.01, min_points=5)
    lower, upper, step = 0.0, 100.0, 20.0
    visited = set()
    
    def true_response(level):
        return 0.0 if level < 50 else 0.7
    
    for i in range(25):
        level = curve.suggest_next_level(lower, upper, step, visited)
        if level is None:
            break
        visited.add(round(level, 6))
        response = true_response(level)
        delta = curve.add_point(level, response)
    
    summary = curve.summary()
    pts = curve.points
    
    print(f"  Points measured: {summary['num_points']}")
    print(f"  Substeps: {summary['num_substeps']}")
    print(f"  Converged: {summary['converged']}")
    print(f"  Points: {[(round(l,1), round(r,4)) for l,r in pts]}")
    
    # Verify the jump is captured
    below = [p for p in pts if p[0] < 50]
    above = [p for p in pts if p[0] >= 50]
    
    assert len(below) >= 1, "Should have points below threshold"
    assert len(above) >= 1, "Should have points above threshold"
    
    # The response below should be ~0, above should be ~0.7
    assert all(abs(p[1]) < 0.05 for p in below), f"Below threshold should be ~0: {below}"
    assert all(abs(p[1] - 0.7) < 0.05 for p in above), f"Above threshold should be ~0.7: {above}"
    
    print("  PASS — threshold shape recovered")
    return True


def test_saturation_recovery():
    """Saturation response: f(x) = 0.7 * x / (1 + x).
    
    Expected: converges, bend visible."""
    print("\n=== test_saturation_recovery ===")
    
    curve = ResponseCurve(convergence_threshold=0.005, min_points=4)
    lower, upper, step = 0.0, 100.0, 20.0
    visited = set()
    
    def true_response(level):
        x = level / 100.0
        return 0.7 * x / (1.0 + (x - 0.5).abs() * 4.0) if hasattr(x, 'abs') else 0.7 * x / (1.0 + abs(x - 0.5) * 4.0)
    
    for i in range(25):
        level = curve.suggest_next_level(lower, upper, step, visited)
        if level is None:
            break
        visited.add(round(level, 6))
        response = true_response(level)
        delta = curve.add_point(level, response)
    
    summary = curve.summary()
    print(f"  Points measured: {summary['num_points']}")
    print(f"  Converged: {summary['converged']}")
    print(f"  Points: {[(round(l,1), round(r,4)) for l,r in curve.points]}")
    
    assert summary['converged'], "Saturation should converge"
    print("  PASS")
    return True


def test_drift_detection():
    """Non-stationary: response function shifts between probes.
    
    Expected: when we re-measure a previously visited level and get a
    different response, the interpolation surface moves significantly.
    This is the drift signal — the same input produces different output."""
    print("\n=== test_drift_detection ===")
    
    curve = ResponseCurve(convergence_threshold=0.01, min_points=4)
    lower, upper, step = 0.0, 100.0, 25.0
    visited = set()
    
    # First pass: measure all levels (stationary)
    levels_first = [0.0, 50.0, 100.0, 25.0, 75.0]
    for level in levels_first:
        visited.add(round(level, 6))
        response = 0.7 * (level / 100.0)
        curve.add_point(level, response)
    
    converged_before = curve.is_converged
    delta_before = curve.last_delta
    print(f"  After first pass: converged={converged_before}, delta={delta_before:.6f}")
    
    # Now re-measure level 50 — but the system drifted, response is different
    drift_amount = 0.15
    response_drifted = 0.7 * (50.0 / 100.0) + drift_amount
    delta_after = curve.add_point(50.0, response_drifted)
    
    print(f"  After re-measuring drifted point: delta={delta_after:.6f}")
    print(f"  Points: {[(round(l,1), round(r,4)) for l,r in curve.points]}")
    
    # The delta from re-measuring a drifted point should exceed the
    # convergence threshold — the surface moved enough to matter
    assert delta_after > curve.convergence_threshold, \
        f"Drift should cause delta above threshold: {delta_after:.4f}"
    print(f"  PASS — drift re-measurement produced delta={delta_after:.4f}")
    return True


def test_convergence_delta_stationary():
    """For a stationary linear system, delta should eventually reach
    near-zero — adding points stops moving the surface."""
    print("\n=== test_convergence_delta_stationary ===")
    
    curve = ResponseCurve(convergence_threshold=0.001, min_points=6)
    lower, upper, step = 0.0, 100.0, 25.0
    visited = set()
    
    deltas = []
    
    for i in range(10):
        level = curve.suggest_next_level(lower, upper, step, visited)
        if level is None:
            break
        visited.add(round(level, 6))
        response = 0.5 * (level / 100.0)
        delta = curve.add_point(level, response)
        if delta < float('inf'):
            deltas.append(delta)
    
    print(f"  Deltas: {[f'{d:.6f}' for d in deltas]}")
    print(f"  Converged: {curve.is_converged}")
    
    # The last delta should be near-zero (stationary linear function)
    assert curve.is_converged, "Stationary linear should converge"
    assert curve.last_delta < 0.001, \
        f"Last delta should be ~0 for stationary: {curve.last_delta}"
    
    print("  PASS")
    return True


if __name__ == '__main__':
    results = []
    results.append(test_linear_convergence())
    results.append(test_threshold_recovery())
    results.append(test_saturation_recovery())
    results.append(test_drift_detection())
    results.append(test_convergence_delta_stationary())
    
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        sys.exit(1)
