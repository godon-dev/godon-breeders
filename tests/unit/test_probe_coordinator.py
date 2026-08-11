#!/usr/bin/env python3
"""
Unit tests for probe_coordinator.py — char studies, step derivation, refinement.

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


# ─── Characterization Study Init ───────────────────────────────────

def test_char_init_basic():
    """3 params → 3 studies, each with derived step."""
    print("\n=== test_char_init_basic ===")
    coord = _make_coordinator()
    coord._init_characterization()

    assert len(coord._char_studies) == 3, \
        f"Expected 3 char studies, got {len(coord._char_studies)}"
    assert len(coord._param_order) == 3

    for name in coord._param_order:
        step = coord._char_steps[name]
        assert step == 25.0, f"{name}: expected step 25.0, got {step}"

    print(f"  {len(coord._char_studies)} studies, step=25.0")
    print("  PASS")
    return True


def test_char_init_int_param():
    """Int param: step snapped to integer."""
    print("\n=== test_char_init_int_param ===")
    params = {
        'param_0': {'constraints': [{'lower': 0, 'upper': 10}]},
    }
    coord = _make_coordinator(params=params)
    coord._init_characterization()

    assert len(coord._char_studies) == 1
    step = coord._char_steps['param_0']
    assert step == 2.0, f"Expected int step 2.0, got {step}"
    print(f"  step={step}")
    print("  PASS")
    return True


def test_char_ask_returns_probe():
    """Ask returns a probe dict with config, level, param_name."""
    print("\n=== test_char_ask_returns_probe ===")
    coord = _make_coordinator()
    coord._init_characterization()

    probe = coord._ask_next_probe()
    assert probe is not None
    assert 'param_name' in probe
    assert 'level' in probe
    assert 'config' in probe
    assert probe['param_name'] in coord._param_order

    print(f"  param={probe['param_name']} level={probe['level']}")
    print("  PASS")
    return True


def test_char_ask_uses_neutral():
    """Probe config has all params at neutral except the probed one."""
    print("\n=== test_char_ask_uses_neutral ===")
    coord = _make_coordinator()
    coord._init_characterization()

    probe = coord._ask_next_probe()
    cfg = probe['config']
    for param_name, value in cfg.items():
        if param_name == probe['param_name']:
            assert value == probe['level']
        else:
            assert value == 50.0, \
                f"Non-probed param should be neutral (50.0), got {value}"

    print("  All configs correct")
    print("  PASS")
    return True


def test_char_ask_stepped_level():
    """Level is on the discrete grid (lower + k*step)."""
    print("\n=== test_char_ask_stepped_level ===")
    coord = _make_coordinator()
    coord._init_characterization()

    seen_levels = set()
    for _ in range(10):
        probe = coord._ask_next_probe()
        if probe and probe['param_name'] == 'param_0':
            seen_levels.add(round(probe['level'], 2))

    step = coord._char_steps['param_0']
    lower = coord._param_bounds['param_0']['lower']
    for level in seen_levels:
        remainder = (level - lower) % step
        assert remainder < 0.01 or abs(remainder - step) < 0.01, \
            f"Level {level} not on grid (step={step}, lower={lower})"

    print(f"  param_0 levels: {sorted(seen_levels)}")
    print("  PASS")
    return True


# ─── Characterization Study Tell ────────────────────────────────────

def test_char_tell_feeds_delta():
    """Telling delta records it in the study as a COMPLETE trial."""
    print("\n=== test_char_tell_feeds_delta ===")
    coord = _make_coordinator()
    coord._init_characterization()

    probe = coord._ask_next_probe()
    param = probe['param_name']

    from optuna.trial import TrialState
    coord._tell_char_study(param, delta=0.5)

    study = coord._char_studies[param]
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    assert len(completed) == 1
    assert completed[0].values == [0.5]

    print(f"  told delta=0.5, study has {len(completed)} complete trial")
    print("  PASS")
    return True


def test_char_tell_fail_on_none():
    """Telling None (causal unavailable) marks trial as FAIL."""
    print("\n=== test_char_tell_fail_on_none ===")
    coord = _make_coordinator()
    coord._init_characterization()

    probe = coord._ask_next_probe()
    param = probe['param_name']

    from optuna.trial import TrialState
    coord._tell_char_study(param, delta=None)

    study = coord._char_studies[param]
    failed = [t for t in study.trials if t.state == TrialState.FAIL]
    assert len(failed) == 1

    print(f"  told None, study has {len(failed)} fail trial")
    print("  PASS")
    return True


# ─── Coverage Guard ────────────────────────────────────────────────

def test_coverage_guard_cycles():
    """_select_next_param cycles through all params."""
    print("\n=== test_coverage_guard_cycles ===")
    coord = _make_coordinator()
    coord._init_characterization()

    seen = []
    for _ in range(6):
        param = coord._select_next_param()
        seen.append(param)

    assert len(set(seen[:3])) == 3, f"First cycle should cover all params: {seen[:3]}"
    assert seen[:3] == seen[3:6], f"Cycles should repeat: {seen}"

    print(f"  cycle: {seen[:3]} → {seen[3:6]}")
    print("  PASS")
    return True


def test_coverage_guard_skips_converged():
    """Converged params are skipped."""
    print("\n=== test_coverage_guard_skips_converged ===")
    coord = _make_coordinator()
    coord._init_characterization()

    coord._converged_params.add('param_0')

    seen = set()
    for _ in range(4):
        param = coord._select_next_param()
        if param:
            seen.add(param)

    assert 'param_0' not in seen
    assert seen == {'param_1', 'param_2'}

    print(f"  converged=param_0, selected from: {seen}")
    print("  PASS")
    return True


def test_coverage_guard_all_converged():
    """When all params converged, _select_next_param returns None."""
    print("\n=== test_coverage_guard_all_converged ===")
    coord = _make_coordinator()
    coord._init_characterization()

    for p in coord._param_order:
        coord._converged_params.add(p)

    result = coord._select_next_param()
    assert result is None

    print("  all converged → None")
    print("  PASS")
    return True


# ─── Refinement ────────────────────────────────────────────────────

def test_refinement_creates_new_study():
    """Refinement creates new study at halved step."""
    print("\n=== test_refinement_creates_new_study ===")
    coord = _make_coordinator()
    coord._init_characterization()

    original_step = coord._char_steps['param_0']
    original_study = coord._char_studies['param_0']

    coord._refine_study('param_0')

    new_step = coord._char_steps['param_0']
    new_study = coord._char_studies['param_0']

    assert new_step == original_step / 2.0
    assert new_study is not original_study
    assert coord._refinement_level['param_0'] == 1

    print(f"  step {original_step}→{new_step}, new study created")
    print("  PASS")
    return True


def test_refinement_depth_cap():
    """refinement_depth limits passes, then accepts best effort (marks converged)."""
    print("\n=== test_refinement_depth_cap ===")
    coord = _make_coordinator(refinement_depth=2)
    coord._init_characterization()

    coord._refine_study('param_0')
    assert coord._refinement_level['param_0'] == 1

    coord._refine_study('param_0')
    assert coord._refinement_level['param_0'] == 2

    coord._refine_study('param_0')
    assert 'param_0' in coord._converged_params

    print(f"  depth=2, after 3 calls → converged")
    print("  PASS")
    return True


def test_count_discrete_levels():
    """Step 25, range 0-100 → 5 levels."""
    print("\n=== test_count_discrete_levels ===")
    coord = _make_coordinator()
    coord._init_characterization()

    n = coord._count_discrete_levels('param_0')
    assert n == 5, f"Expected 5 levels, got {n}"

    print(f"  step=25, range 0-100 → {n} levels")
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


if __name__ == '__main__':
    results = []
    results.append(test_step_derivation_float())
    results.append(test_step_derivation_int())
    results.append(test_step_derivation_int_small_range())
    results.append(test_step_derivation_degenerate())
    results.append(test_char_init_basic())
    results.append(test_char_init_int_param())
    results.append(test_char_ask_returns_probe())
    results.append(test_char_ask_uses_neutral())
    results.append(test_char_ask_stepped_level())
    results.append(test_char_tell_feeds_delta())
    results.append(test_char_tell_fail_on_none())
    results.append(test_coverage_guard_cycles())
    results.append(test_coverage_guard_skips_converged())
    results.append(test_coverage_guard_all_converged())
    results.append(test_refinement_creates_new_study())
    results.append(test_refinement_depth_cap())
    results.append(test_count_discrete_levels())
    results.append(test_timeout_no_history())
    results.append(test_timeout_scales_with_trial_duration())
    results.append(test_timeout_tightened_by_rtt())
    results.append(test_timeout_never_below_floor())

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        sys.exit(1)
