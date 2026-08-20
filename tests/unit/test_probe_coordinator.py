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


def _ensure_real_optuna():
    """Restore real optuna — multiple test files mock it in sys.modules."""
    import importlib
    for mod in ['optuna.samplers', 'optuna.trial', 'optuna.storages', 'optuna']:
        if mod in sys.modules:
            obj = sys.modules[mod]
            if not hasattr(obj, '__name__') or obj.__name__ != 'optuna':
                del sys.modules[mod]
    if 'optuna' not in sys.modules or getattr(sys.modules.get('optuna'), '__name__', '') != 'optuna':
        import optuna
        sys.modules['optuna'] = optuna
        sys.modules['optuna.storages'] = optuna.storages
        sys.modules['optuna.trial'] = optuna.trial
        sys.modules['optuna.samplers'] = optuna.samplers


def _make_coordinator(config=None, params=None, **overrides):
    """Create a coordinator with mocked DB."""
    _ensure_real_optuna()
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


def test_step_derivation_int():
    """Int param: range 0-10, step raw=2.5 → snapped to int."""
    print("\n=== test_step_derivation_int ===")
    coord = _make_coordinator()
    step = coord._derive_initial_step(0.0, 10.0, is_int=True)
    # round(2.5) = 2 in Python (banker's rounding), step=2 is valid
    assert step == 2.0, f"Expected 2.0, got {step}"
    print(f"  step={step}")
    print("  PASS")


def test_step_derivation_int_small_range():
    """Int param: range 0-3, step raw=0.75 → snapped to 1."""
    print("\n=== test_step_derivation_int_small_range ===")
    coord = _make_coordinator()
    step = coord._derive_initial_step(0.0, 3.0, is_int=True)
    assert step == 1.0, f"Expected 1.0, got {step}"
    print(f"  step={step}")
    print("  PASS")


def test_step_derivation_degenerate():
    """Range 0-0 → step 0."""
    print("\n=== test_step_derivation_degenerate ===")
    coord = _make_coordinator()
    step = coord._derive_initial_step(50.0, 50.0)
    assert step == 0.0, f"Expected 0.0, got {step}"
    print(f"  step={step}")
    print("  PASS")


# ─── Characterization Study (single study, all params) ────────────

def test_char_init_basic():
    """3 params → ONE study with all params as dimensions."""
    print("\n=== test_char_init_basic ===")
    coord = _make_coordinator()
    coord._init_characterization()

    assert coord._char_study is not None, "study should be created"
    assert len(coord._param_names) == 3
    assert coord._char_step == 25.0

    print(f"  1 study, {len(coord._param_names)} params, step={coord._char_step}")
    print("  PASS")


def test_char_ask_picks_param_and_level():
    """Ask returns a probe with param_name from params, level on grid."""
    print("\n=== test_char_ask_picks_param_and_level ===")
    coord = _make_coordinator()
    coord._init_characterization()

    probe = coord._ask_next_probe()
    assert probe is not None
    assert probe['param_name'] in coord._param_names
    assert probe['level'] is not None
    assert probe['config'] is not None
    # The probed param should be at probe level, others at neutral
    for name, val in probe['config'].items():
        if name == probe['param_name']:
            assert val == probe['level']
        else:
            assert val == 50.0

    print(f"  param={probe['param_name']} level={probe['level']}")
    print("  PASS")


def test_char_ask_samples_different_params():
    """Multiple asks should eventually sample different params (startup randomness)."""
    print("\n=== test_char_ask_samples_different_params ===")
    coord = _make_coordinator()
    coord._init_characterization()

    seen_params = set()
    for _ in range(10):
        probe = coord._ask_next_probe()
        if probe:
            seen_params.add(probe['param_name'])
            coord._tell_char_study(probe['param_name'], {'delta': 0.5})

    # With 3 params and n_startup=9, all 3 should appear
    assert len(seen_params) >= 2, f"Expected >=2 params, got {seen_params}"

    print(f"  params seen: {seen_params}")
    print("  PASS")


def test_char_tell_feeds_delta():
    """Telling delta records it in the study as COMPLETE."""
    print("\n=== test_char_tell_feeds_delta ===")
    coord = _make_coordinator()
    coord._init_characterization()

    probe = coord._ask_next_probe()
    coord._tell_char_study(probe['param_name'], {'delta': 0.5})

    from optuna.trial import TrialState
    complete = [t for t in coord._char_study.trials if t.state == TrialState.COMPLETE]
    assert len(complete) == 1
    assert complete[0].values == [0.5]

    print(f"  told delta=0.5, 1 complete trial")
    print("  PASS")


def test_char_tell_catches_infinity():
    """INFINITY-replacement value is caught and replaced with 1.0."""
    print("\n=== test_char_tell_catches_infinity ===")
    coord = _make_coordinator()
    coord._init_characterization()

    probe = coord._ask_next_probe()
    # Simulate causal's INFINITY replacement (f64::MAX/2), old-causal shape
    coord._tell_char_study(probe['param_name'], {'delta': 8.988e+307})

    from optuna.trial import TrialState
    complete = [t for t in coord._char_study.trials if t.state == TrialState.COMPLETE]
    assert len(complete) == 1
    # Should be 1.0, not 8.988e+307
    assert complete[0].values == [1.0], f"Expected [1.0], got {complete[0].values}"

    print(f"  caught INFINITY → replaced with 1.0")
    print("  PASS")


def test_char_tell_fail_on_none():
    """None delta (causal unavailable) marks trial as FAIL."""
    print("\n=== test_char_tell_fail_on_none ===")
    coord = _make_coordinator()
    coord._init_characterization()

    probe = coord._ask_next_probe()
    coord._tell_char_study(probe['param_name'], None)

    from optuna.trial import TrialState
    failed = [t for t in coord._char_study.trials if t.state == TrialState.FAIL]
    assert len(failed) == 1

    print(f"  told None → 1 FAIL trial")
    print("  PASS")


def test_char_tell_prefers_z():
    """When causal returns z, the study is told z — not delta."""
    print("\n=== test_char_tell_prefers_z ===")
    coord = _make_coordinator()
    coord._init_characterization()

    probe = coord._ask_next_probe()
    coord._tell_char_study(probe['param_name'], {
        'shift': 0.37, 'shift_bar': 0.012, 'z': 8.3,
        'drift': False, 'delta': 0.04, 'converged': False,
    })

    from optuna.trial import TrialState
    complete = [t for t in coord._char_study.trials if t.state == TrialState.COMPLETE]
    assert len(complete) == 1
    assert complete[0].values == [8.3], \
        f"Expected z=8.3 as objective, got {complete[0].values}"

    print(f"  told z=8.3 → objective 8.3")
    print("  PASS")


def test_process_probe_result_returns_dict():
    """_process_probe_result returns the causal result dict."""
    print("\n=== test_process_probe_result_returns_dict ===")
    from unittest.mock import patch
    coord = _make_coordinator()
    coord._init_characterization()

    probe = {'param_name': 'param_0', 'level': 50.0, 'param_idx': 0}

    with patch.object(coord, '_query_causal_probe_result',
                      return_value={'shift': 0.04, 'shift_bar': 0.02, 'z': 1.5,
                                    'drift': False, 'delta': 0.015, 'converged': False}):
        result = coord._process_probe_result(probe)
        assert isinstance(result, dict)
        assert result['delta'] == 0.015
        assert result['z'] == 1.5

    # Causal unavailable → None
    with patch.object(coord, '_query_causal_probe_result', return_value=None):
        result = coord._process_probe_result(probe)
        assert result is None

    print("  causal available → result dict (z + delta)")
    print("  causal unavailable → None")
    print("  PASS")


def test_refinement_halves_step():
    """Refinement creates new study at halved step."""
    print("\n=== test_refinement_halves_step ===")
    coord = _make_coordinator()
    coord._init_characterization()

    original_step = coord._char_step
    coord._refine_study()

    assert coord._char_step == original_step / 2.0
    assert coord._refinement_level == 1

    print(f"  step {original_step}→{coord._char_step}")
    print("  PASS")


def test_refinement_depth_cap():
    """Refinement depth limits passes, then accepts (marks all converged)."""
    print("\n=== test_refinement_depth_cap ===")
    coord = _make_coordinator(refinement_depth=2)
    coord._init_characterization()

    coord._refine_study()
    assert coord._refinement_level == 1
    assert len(coord._converged_params) == 0

    coord._refine_study()
    assert coord._refinement_level == 2
    assert len(coord._converged_params) == 0

    coord._refine_study()
    # All params should now be converged
    assert len(coord._converged_params) == len(coord._param_names)

    print(f"  depth=2, after 3 calls → all converged")
    print("  PASS")


def test_count_param_levels():
    """Step 25, range 0-100 → 5 levels."""
    print("\n=== test_count_param_levels ===")
    coord = _make_coordinator()
    coord._init_characterization()

    n = coord._count_param_levels('param_0')
    assert n == 5, f"Expected 5 levels, got {n}"

    print(f"  step=25, range 0-100 → {n} levels")
    print("  PASS")

# ─── Timeout Deskew ───────────────────────────────────────────────

def test_timeout_no_history():
    """No trial duration history → floor 2s."""
    print("\n=== test_timeout_no_history ===")
    coord = _make_coordinator()
    timeout = coord._causal_timeout()
    assert timeout == 2.0, f"Expected 2.0, got {timeout}"
    print(f"  timeout={timeout}")
    print("  PASS")


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


# ─── State Machine Integration ────────────────────────────────────

class _FakeTrial:
    """Minimal trial stand-in for decide_trial()."""
    def __init__(self, number=0):
        self.number = number
        self._attrs = {}
    def set_user_attr(self, key, value):
        self._attrs[key] = value
    @property
    def user_attrs(self):
        return self._attrs


def test_push_pause_round():
    """Full push/pause round: ask → push N → pause N → tell delta."""
    print("\n=== test_push_pause_round ===")
    from unittest.mock import patch
    coord = _make_coordinator()
    coord._init_characterization()

    # Patch causal to return a delta
    with patch.object(coord, '_query_causal_probe_result',
                      return_value={'shift': 0.05, 'delta': 0.03, 'converged': False}):
        # Simulate acquiring lease and entering PROBE_PUSH
        coord.state = coord.PROBE_PUSH
        coord._push_count = 0

        # Push block (3 trials)
        decisions = []
        for i in range(coord.push_block_size):
            t = _FakeTrial(number=i)
            d = coord._handle_probe_push(t)
            decisions.append(d)

        assert all(d['mode'] == 'impulse' for d in decisions)
        assert decisions[0]['probe_param'] is not None
        assert decisions[0]['probe_level'] is not None
        assert coord.state == coord.PROBE_PAUSE

        # Pause block (3 trials)
        coord._pause_count = 0
        pause_decisions = []
        for i in range(coord.pause_block_size):
            t = _FakeTrial(number=i + coord.push_block_size)
            d = coord._handle_probe_pause(t)
            pause_decisions.append(d)

        assert all(d['mode'] == 'hold' for d in pause_decisions)

        # After pause completes, should be back in PROBE_PUSH
        assert coord.state == coord.PROBE_PUSH

        # The char study should have 1 complete trial (delta told)
        from optuna.trial import TrialState
        complete = [t for t in coord._char_study.trials if t.state == TrialState.COMPLETE]
        assert len(complete) == 1
        assert complete[0].values == [0.03]

    print(f"  push={coord.push_block_size}, pause={coord.pause_block_size}")
    print(f"  delta=0.03 told to study, 1 complete trial")
    print("  PASS")


def test_exhaustion_triggers_refinement():
    """Visiting all discrete combinations triggers refinement."""
    print("\n=== test_exhaustion_triggers_refinement ===")
    coord = _make_coordinator(params={
        'param_0': {'constraints': [{'lower': 0.0, 'upper': 100.0}]},
    })
    coord._init_characterization()

    original_step = coord._char_step
    n_levels = coord._count_param_levels('param_0')

    # Ask + tell all levels
    for i in range(n_levels):
        probe = coord._ask_next_probe()
        assert probe is not None
        coord._tell_char_study('param_0', {'delta': 0.5})

    # Should have triggered refinement
    assert coord._char_step == original_step / 2.0, \
        f"Expected step {original_step/2.0}, got {coord._char_step}"
    assert coord._refinement_level == 1

    print(f"  {n_levels} levels exhausted → step {original_step}→{coord._char_step}")
    print("  PASS")


def test_convergence_all_params_done():
    """When all params converged, _ask_next_probe still works but DONE triggers via state."""
    print("\n=== test_convergence_all_params_done ===")
    coord = _make_coordinator(params={
        'param_0': {'constraints': [{'lower': 0.0, 'upper': 100.0}]},
    })
    coord._init_characterization()

    coord._converged_params.add('param_0')

    # With all params converged, the coordinator's DONE check in
    # _handle_probe_push triggers when _ask_next_probe returns None.
    # But ask itself still works (TPE can still sample). The DONE
    # logic is in the state machine, not in ask.
    probe = coord._ask_next_probe()
    # Probe is not None — study still gives trials. Convergence is
    # checked by the state machine via _converged_params.
    assert probe is not None

    print("  converged param → study still samples (state machine handles DONE)")
    print("  PASS")


def test_process_probe_result_marks_converged():
    """When causal says converged=True, param is added to converged set."""
    print("\n=== test_process_probe_result_marks_converged ===")
    from unittest.mock import patch
    coord = _make_coordinator()
    coord._init_characterization()

    probe = {'param_name': 'param_1', 'level': 75.0, 'param_idx': 1}
    assert 'param_1' not in coord._converged_params

    with patch.object(coord, '_query_causal_probe_result',
                      return_value={'shift': 0.001, 'delta': 0.005, 'converged': True}):
        coord._process_probe_result(probe)

    assert 'param_1' in coord._converged_params

    print("  converged=True → param_1 in converged set")
    print("  PASS")


def test_get_char_status():
    """get_char_status returns structured per-param state."""
    print("\n=== test_get_char_status ===")
    coord = _make_coordinator()
    coord._init_characterization()

    # Do one probe to have some state
    probe = coord._ask_next_probe()
    coord._tell_char_study(probe['param_name'], {'delta': 0.5})

    status = coord.get_char_status()

    assert status['state'] == coord.state
    assert status['converged_count'] == 0
    assert status['params_total'] == 3
    assert status['combinations_explored'] == 1

    for name in coord._param_names:
        s = status['params'][name]
        assert 'converged' in s
        assert 'step' in s
        assert 'levels_total' in s

    print(f"  {status['converged_count']}/{status['params_total']} converged")
    print(f"  {status['combinations_explored']}/{status['combinations_total']} explored")
    print("  PASS")


def test_ask_after_all_converged():
    """When all params converged, study still samples (DONE handled by state machine)."""
    print("\n=== test_ask_after_all_converged ===")
    coord = _make_coordinator()
    coord._init_characterization()

    for p in coord._param_names:
        coord._converged_params.add(p)

    probe = coord._ask_next_probe()
    # Study still samples — convergence is checked by the state machine
    # via _converged_params in _handle_probe_push, not by the study.
    assert probe is not None
    assert len(coord._converged_params) == len(coord._param_names)

    print("  all converged → study still samples, state machine handles DONE")
    print("  PASS")



if __name__ == '__main__':
    test_fns = [
        test_step_derivation_float,
        test_step_derivation_int,
        test_step_derivation_int_small_range,
        test_step_derivation_degenerate,
        test_char_init_basic,
        test_char_init_int_param,
        test_char_ask_returns_probe,
        test_char_ask_uses_neutral,
        test_char_ask_stepped_level,
        test_char_tell_feeds_delta,
        test_char_tell_fail_on_none,
        test_coverage_guard_cycles,
        test_coverage_guard_skips_converged,
        test_coverage_guard_all_converged,
        test_refinement_creates_new_study,
        test_refinement_depth_cap,
        test_count_discrete_levels,
        test_timeout_no_history,
        test_timeout_scales_with_trial_duration,
        test_timeout_tightened_by_rtt,
        test_timeout_never_below_floor,
        test_push_pause_round,
        test_exhaustion_triggers_refinement,
        test_convergence_all_params_done,
        test_char_tell_prefers_z,
        test_process_probe_result_returns_dict,
        test_process_probe_result_marks_converged,
        test_get_char_status,
        test_ask_after_all_converged,
    ]
    passed = 0
    for fn in test_fns:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {fn.__name__}: {e}")
    total = len(test_fns)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        sys.exit(1)
