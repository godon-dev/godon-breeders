#!/usr/bin/env python3
"""
Unit tests for probe_coordinator.py — schedule, step derivation, refinement.

Tests the coordinator logic without a real DB or causal service.
Mocks the shared_db_fn and causal HTTP calls.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Stub Windmill runtime imports before importing coordinator
import types
import logging
if 'f' not in sys.modules:
    f_mod = types.ModuleType('f')
    breeder_mod = types.ModuleType('f.breeder')
    shared_mod = types.ModuleType('f.breeder.shared')
    otel_mod = types.ModuleType('f.breeder.shared.otel_logging')
    engine_mod = types.ModuleType('f.breeder.engine')
    otel_mod.get_logger = lambda name: logging.getLogger(name)
    f_mod.breeder = breeder_mod
    breeder_mod.shared = shared_mod
    shared_mod.otel_logging = otel_mod
    breeder_mod.engine = engine_mod
    sys.modules['f'] = f_mod
    sys.modules['f.breeder'] = breeder_mod
    sys.modules['f.breeder.shared'] = shared_mod
    sys.modules['f.breeder.shared.otel_logging'] = otel_mod
    sys.modules['f.breeder.engine'] = engine_mod

# Also make characterization importable as f.breeder.engine.characterization
import importlib
try:
    char_mod = importlib.import_module('engine.characterization')
    sys.modules['f.breeder.engine.characterization'] = char_mod
except ImportError:
    pass

from engine.probe_coordinator import ProbeCoordinator


def _noop_db(fn, desc):
    """DB mock that does nothing — lease ops silently succeed."""
    pass


def _config(params=None, **overrides):
    """Build a minimal breeder config with given params."""
    params = params or {
        'param_0': {'constraints': [{'lower': 0.0, 'upper': 100.0}]},
        'param_1': {'constraints': [{'lower': 0.0, 'upper': 100.0}]},
        'param_2': {'constraints': [{'lower': 0.0, 'upper': 100.0}]},
    }
    cfg = {
        'breeder': {'type': 'bench_generic', 'uuid': 'test-1'},
        'settings': {'generic': params},
        'interference_detection': {
            'group': 'test',
            'hold_params': {name: 50.0 for name in params},
            'push_block_size': 3,
            'pause_block_size': 3,
            'cooldown_trials': 1,
            'min_optimize_trials': 2,
            'convergence_threshold': 0.02,
            'refinement_depth': 3,
        },
    }
    cfg['interference_detection'].update(overrides)
    return cfg


def _make_coordinator(config=None, params=None, **overrides):
    """Create a coordinator with mocked DB."""
    config = config or _config(params=params, **overrides)
    coord = ProbeCoordinator(
        breeder_id='test-sender-1',
        config=config,
        shared_db_fn=_noop_db,
        collect_upper_bounds_fn=lambda settings: _fake_upper_bounds(settings),
    )
    coord._initialized = True
    return coord


def _fake_upper_bounds(settings):
    """Mimic the worker's _collect_upper_bounds."""
    gen = settings.get('generic', settings)
    bounds = []
    for name, cfg in gen.items():
        if not isinstance(cfg, dict) or 'constraints' not in cfg:
            continue
        for c in cfg['constraints']:
            bounds.append({
                'name': name,
                'lower': c.get('lower', 0.0),
                'upper': c.get('upper', 100.0),
                'step': c.get('step', 20.0),
                'range': c.get('upper', 100.0) - c.get('lower', 0.0),
                'is_int': isinstance(c.get('lower', 0), int) and isinstance(c.get('upper', 100), int),
            })
    return bounds


# ─── Step Derivation ──────────────────────────────────────────────

def test_step_derivation_float():
    """Float param: range 0-100, threshold 0.02 → step = 25."""
    print("\n=== test_step_derivation_float ===")
    coord = _make_coordinator()
    step = coord._derive_initial_step(0.0, 100.0, is_int=False)
    assert step == 25.0, f"Expected 25.0, got {step}"
    print(f"  step={step}")
    print("  PASS")
    return True


def test_step_derivation_int():
    """Int param: range 0-10, step raw=2.5 → snapped to int."""
    print("\n=== test_step_derivation_int ===")
    coord = _make_coordinator()
    step = coord._derive_initial_step(0.0, 10.0, is_int=True)
    # round(2.5) = 2 in Python (banker's rounding), step=2 is valid
    assert step == 2.0, f"Expected 2.0, got {step}"
    print(f"  step={step}")
    print("  PASS")
    return True


def test_step_derivation_int_small_range():
    """Int param: range 0-3, step raw=0.75 → snapped to 1."""
    print("\n=== test_step_derivation_int_small_range ===")
    coord = _make_coordinator()
    step = coord._derive_initial_step(0.0, 3.0, is_int=True)
    assert step == 1.0, f"Expected 1.0, got {step}"
    print(f"  step={step}")
    print("  PASS")
    return True


def test_step_derivation_degenerate():
    """Range 0-0 → step 0."""
    print("\n=== test_step_derivation_degenerate ===")
    coord = _make_coordinator()
    step = coord._derive_initial_step(50.0, 50.0)
    assert step == 0.0, f"Expected 0.0, got {step}"
    print(f"  step={step}")
    print("  PASS")
    return True


# ─── Probe Schedule ───────────────────────────────────────────────

def test_schedule_basic():
    """3 params, range 0-100, step 25 → 5 levels per param = 15 probes."""
    print("\n=== test_schedule_basic ===")
    coord = _make_coordinator()
    coord._build_probe_schedule()
    assert len(coord._probe_schedule) == 15, \
        f"Expected 15 probes, got {len(coord._probe_schedule)}"
    
    params_in_schedule = set(p['param_name'] for p in coord._probe_schedule)
    assert params_in_schedule == {'param_0', 'param_1', 'param_2'}
    
    param_0_levels = [p['level'] for p in coord._probe_schedule if p['param_name'] == 'param_0']
    assert param_0_levels == [0.0, 25.0, 50.0, 75.0, 100.0], \
        f"Expected [0, 25, 50, 75, 100], got {param_0_levels}"
    
    print(f"  {len(coord._probe_schedule)} probes, levels={param_0_levels}")
    print("  PASS")
    return True


def test_schedule_int_param():
    """Int param: range 0-10, derived step → integer levels."""
    print("\n=== test_schedule_int_param ===")
    params = {
        'param_0': {'constraints': [{'lower': 0, 'upper': 10}]},
    }
    coord = _make_coordinator(params=params)
    coord._build_probe_schedule()
    
    levels = [p['level'] for p in coord._probe_schedule]
    # Coordinator derives step: range/4 = 2.5, snapped to int 2
    # → levels [0, 2, 4, 6, 8, 10]
    assert all(isinstance(l, int) for l in levels), "All levels should be int"
    assert levels == [0, 2, 4, 6, 8, 10], f"Expected [0, 2, 4, 6, 8, 10], got {levels}"
    print(f"  levels={levels}")
    print("  PASS")
    return True


def test_schedule_config_uses_neutral():
    """Each probe config has all params at neutral except the probed one."""
    print("\n=== test_schedule_config_uses_neutral ===")
    coord = _make_coordinator()
    coord._build_probe_schedule()
    
    for probe in coord._probe_schedule:
        cfg = probe['config']
        for param_name, value in cfg.items():
            if param_name == probe['param_name']:
                assert value == probe['level'], \
                    f"Probed param should be at probe level"
            else:
                assert value == 50.0, \
                    f"Non-probed param should be neutral (50.0), got {value}"
    
    print("  All configs correct")
    print("  PASS")
    return True


def test_schedule_custom_threshold():
    """Tighter threshold → same initial step (threshold affects refinement, not coarse)."""
    print("\n=== test_schedule_custom_threshold ===")
    coord = _make_coordinator(convergence_threshold=0.001)
    coord._build_probe_schedule()
    # Coarse pass is always 4 segments regardless of threshold
    assert len(coord._probe_schedule) == 15
    print(f"  threshold=0.001 → {len(coord._probe_schedule)} probes (same coarse)")
    print("  PASS")
    return True


# ─── Refinement ───────────────────────────────────────────────────

def test_refinement_halves_step():
    """After coarse pass, refinement generates midpoints at half step."""
    print("\n=== test_refinement_halves_step ===")
    coord = _make_coordinator()
    coord._build_probe_schedule()
    
    # Simulate: all coarse probes for param_0 done
    coord._probe_idx = 5  # past all param_0 coarse probes (indices 0-4)
    
    # Generate first refinement pass
    coord._generate_halved_levels('param_0')
    
    param_0_all = [p['level'] for p in coord._probe_schedule if p['param_name'] == 'param_0']
    refinement = [p for p in coord._probe_schedule 
                  if p['param_name'] == 'param_0' and p.get('is_refinement')]
    
    print(f"  coarse: {[0.0, 25.0, 50.0, 75.0, 100.0]}")
    print(f"  after refinement 1: {sorted(param_0_all)}")
    
    # Should have midpoints at 12.5, 37.5, 62.5, 87.5
    assert 12.5 in param_0_all
    assert 37.5 in param_0_all
    assert 62.5 in param_0_all
    assert 87.5 in param_0_all
    assert len(refinement) == 4
    
    print("  PASS")
    return True


def test_refinement_depth_cap():
    """refinement_depth=2 allows 2 bisection passes, then stops."""
    print("\n=== test_refinement_depth_cap ===")
    coord = _make_coordinator(refinement_depth=2)
    coord._build_probe_schedule()
    
    # Simulate coarse pass done for param_0
    coord._probe_idx = 5
    
    # Pass 1
    coord._check_param_complete('param_0')
    n_after_pass1 = len(coord._probe_schedule)
    assert coord._refinement_passes.get('param_0') == 1
    
    # Simulate refinement probes done
    coord._probe_idx = n_after_pass1
    
    # Pass 2
    coord._check_param_complete('param_0')
    n_after_pass2 = len(coord._probe_schedule)
    assert coord._refinement_passes.get('param_0') == 2
    
    # Simulate all done
    coord._probe_idx = n_after_pass2
    
    # Pass 3 — should NOT generate (depth=2)
    coord._check_param_complete('param_0')
    assert len(coord._probe_schedule) == n_after_pass2, \
        "Should not generate beyond refinement_depth"
    
    print(f"  pass 1: {n_after_pass1} probes")
    print(f"  pass 2: {n_after_pass2} probes")
    print(f"  pass 3: capped (no new probes)")
    print("  PASS")
    return True


def test_refinement_skipped_if_converged():
    """Converged params don't trigger refinement."""
    print("\n=== test_refinement_skipped_if_converged ===")
    coord = _make_coordinator()
    coord._build_probe_schedule()
    coord._probe_idx = 5
    
    # Mark param_0 as converged
    coord._converged_params.add('param_0')
    coord._check_param_complete('param_0')
    
    param_0_probes = [p for p in coord._probe_schedule if p['param_name'] == 'param_0']
    assert len(param_0_probes) == 5, \
        "Converged param should not get refinement probes"
    
    print("  PASS")
    return True


def test_refinement_halves_twice():
    """Two refinement passes produce 3 resolution levels."""
    print("\n=== test_refinement_halves_twice ===")
    coord = _make_coordinator(refinement_depth=3)
    coord._build_probe_schedule()
    
    # Coarse done
    coord._probe_idx = 5
    
    # Pass 1: step 25→12.5
    coord._generate_halved_levels('param_0')
    coord._refinement_passes['param_0'] = 1
    pass1_levels = sorted(set(
        p['level'] for p in coord._probe_schedule if p['param_name'] == 'param_0'
    ))
    
    # Pass 2: step 12.5→6.25
    coord._generate_halved_levels('param_0')
    coord._refinement_passes['param_0'] = 2
    pass2_levels = sorted(set(
        p['level'] for p in coord._probe_schedule if p['param_name'] == 'param_0'
    ))
    
    print(f"  pass 0 (coarse): 5 levels")
    print(f"  pass 1: {len(pass1_levels)} levels")
    print(f"  pass 2: {len(pass2_levels)} levels")
    
    # Each pass should add more levels
    assert len(pass2_levels) > len(pass1_levels)
    
    # Check specific levels exist
    assert 6.25 in pass2_levels
    assert 12.5 in pass2_levels
    
    print("  PASS")
    return True


# ─── Timeout Deskew ───────────────────────────────────────────────

def test_timeout_no_history():
    """No trial duration history → floor 2s."""
    print("\n=== test_timeout_no_history ===")
    coord = _make_coordinator()
    timeout = coord._causal_timeout()
    assert timeout == 2.0, f"Expected 2.0, got {timeout}"
    print(f"  timeout={timeout}")
    print("  PASS")
    return True


def test_timeout_scales_with_trial_duration():
    """Fast trials → short budget. Slow trials → long budget."""
    print("\n=== test_timeout_scales_with_trial_duration ===")
    coord = _make_coordinator()
    
    # Simulate slow trials (45s each)
    for _ in range(5):
        coord._record_trial_duration(45.0)
    
    timeout = coord._causal_timeout()
    assert timeout == 90.0, f"Expected 90.0 (2×45), got {timeout}"
    
    # Simulate fast trials (1s each)
    coord._trial_duration_history = []
    for _ in range(5):
        coord._record_trial_duration(1.0)
    
    timeout = coord._causal_timeout()
    # Budget = 2×1 = 2, but no RTT history so return budget
    assert timeout == 2.0, f"Expected 2.0, got {timeout}"
    
    print(f"  slow trials (45s) → timeout {90.0}s")
    print(f"  fast trials (1s) → timeout {2.0}s")
    print("  PASS")
    return True


def test_timeout_tightened_by_rtt():
    """Causal consistently fast RTT → tightened timeout."""
    print("\n=== test_timeout_tightened_by_rtt ===")
    coord = _make_coordinator()
    
    # Slow trials but causal is consistently fast
    for _ in range(5):
        coord._record_trial_duration(45.0)
    for _ in range(5):
        coord._causal_rtt_history.append(0.003)  # 3ms
    
    timeout = coord._causal_timeout()
    # Budget = 90, tightened = 10×0.003 = 0.03, floor 2.0
    assert timeout == 2.0, f"Expected 2.0 (floor), got {timeout}"
    
    # If causal is moderately slow
    coord._causal_rtt_history = [0.5, 0.5, 0.5]  # 500ms RTT
    timeout = coord._causal_timeout()
    # Budget = 90, tightened = 10×0.5 = 5.0
    assert timeout == 5.0, f"Expected 5.0, got {timeout}"
    
    print(f"  fast causal (3ms) → timeout 2.0s (floor)")
    print(f"  moderate causal (500ms) → timeout 5.0s")
    print("  PASS")
    return True


def test_timeout_never_below_floor():
    """Timeout always ≥ 2s regardless of history."""
    print("\n=== test_timeout_never_below_floor ===")
    coord = _make_coordinator()
    
    # Extremely fast everything
    coord._trial_duration_history = [0.001]
    coord._causal_rtt_history = [0.0001]
    
    timeout = coord._causal_timeout()
    assert timeout >= 2.0, f"Should never go below 2.0, got {timeout}"
    print(f"  timeout={timeout}")
    print("  PASS")
    return True


# ─── Convergence Skip ─────────────────────────────────────────────

def test_skip_to_next_param():
    """When param converged, _skip_to_next_param advances past its levels."""
    print("\n=== test_skip_to_next_param ===")
    coord = _make_coordinator()
    coord._build_probe_schedule()
    
    # param_0 occupies indices 0-4, param_1 starts at 5
    assert coord._probe_schedule[0]['param_name'] == 'param_0'
    assert coord._probe_schedule[5]['param_name'] == 'param_1'
    
    coord._probe_idx = 2  # mid-way through param_0
    coord._skip_to_next_param('param_0')
    
    assert coord._probe_idx == 5, \
        f"Should skip to index 5 (param_1), got {coord._probe_idx}"
    
    print(f"  skipped param_0 from idx 2 → {coord._probe_idx}")
    print("  PASS")
    return True


if __name__ == '__main__':
    results = []
    results.append(test_step_derivation_float())
    results.append(test_step_derivation_int())
    results.append(test_step_derivation_int_small_range())
    results.append(test_step_derivation_degenerate())
    results.append(test_schedule_basic())
    results.append(test_schedule_int_param())
    results.append(test_schedule_config_uses_neutral())
    results.append(test_schedule_custom_threshold())
    results.append(test_refinement_halves_step())
    results.append(test_refinement_depth_cap())
    results.append(test_refinement_skipped_if_converged())
    results.append(test_refinement_halves_twice())
    results.append(test_timeout_no_history())
    results.append(test_timeout_scales_with_trial_duration())
    results.append(test_timeout_tightened_by_rtt())
    results.append(test_timeout_never_below_floor())
    results.append(test_skip_to_next_param())
    
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        sys.exit(1)
