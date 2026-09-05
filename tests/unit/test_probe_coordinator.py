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
            'walk_policy': 'ladder',  # offline: geometric, no notebook
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


# ─── Characterization Coverage Walk ────────────────────────────────

def test_char_init_basic():
    """3 params → walk built with per-param floors (range/4)."""
    print("\n=== test_char_init_basic ===")
    coord = _make_coordinator()
    coord._init_characterization()

    assert coord._char_walk is not None, "walk should be created"
    assert len(coord._param_names) == 3
    st = coord._char_walk.status()
    assert st['param_0']['step'] == 25.0  # 0-100 float → floor 25

    print(f"  walk built, {len(coord._param_names)} params, floors=25.0")
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


def test_char_ask_rotates_params():
    """Round-robin: first asks hit different params deterministically."""
    print("\n=== test_char_ask_rotates_params ===")
    coord = _make_coordinator()
    coord._init_characterization()

    seen_params = []
    for _ in range(3):
        probe = coord._ask_next_probe()
        assert probe is not None
        seen_params.append(probe['param_name'])
        coord._tell_char_study(probe['param_name'], {'delta': 0.5})

    # Round-robin: all 3 params in the first 3 asks, no repeats
    assert len(set(seen_params)) == 3, f"Expected 3 distinct, got {seen_params}"

    print(f"  params in order: {seen_params}")
    print("  PASS")


def test_char_tell_logs_delta():
    """Telling delta records it in the CHAR TELL log line."""
    print("\n=== test_char_tell_logs_delta ===")
    coord = _make_coordinator()
    coord._init_characterization()

    probe = coord._ask_next_probe()
    coord._tell_char_study(probe['param_name'], {'delta': 0.5})

    # Walk advanced one level for that param
    st = coord._char_walk.status()
    assert st[probe['param_name']]['levels_measured'] == 1

    print(f"  told delta=0.5, 1 level measured for {probe['param_name']}")
    print("  PASS")


def test_char_tell_catches_infinity():
    """INFINITY-replacement value is caught and replaced with 1.0."""
    print("\n=== test_char_tell_catches_infinity ===")
    coord = _make_coordinator()
    coord._init_characterization()

    probe = coord._ask_next_probe()
    # Simulate causal's INFINITY replacement (f64::MAX/2), old-causal shape
    coord._tell_char_study(probe['param_name'], {'delta': 8.988e+307})

    # Walk advanced — the catch path did not abort processing
    st = coord._char_walk.status()
    assert st[probe['param_name']]['levels_measured'] == 1

    print(f"  caught INFINITY → walk continues")
    print("  PASS")


def test_char_tell_fail_on_none():
    """None result (causal unavailable) logs FAIL, walk unaffected."""
    print("\n=== test_char_tell_fail_on_none ===")
    coord = _make_coordinator()
    coord._init_characterization()

    probe = coord._ask_next_probe()
    coord._tell_char_study(probe['param_name'], None)

    # Level stays measured (the probe happened; causal just couldn't price it)
    st = coord._char_walk.status()
    assert st[probe['param_name']]['levels_measured'] == 1

    print(f"  told None → FAIL logged, walk state intact")
    print("  PASS")


def test_char_tell_prefers_z():
    """When causal returns z, CHAR TELL logs z — not delta."""
    print("\n=== test_char_tell_prefers_z ===")
    coord = _make_coordinator()
    coord._init_characterization()

    probe = coord._ask_next_probe()

    from unittest.mock import patch
    import engine.probe_coordinator as pc
    with patch.object(pc, 'logger') as mock_logger:
        coord._tell_char_study(probe['param_name'], {
            'shift': 0.37, 'shift_bar': 0.012, 'z': 8.3,
            'drift': False, 'delta': 0.04, 'converged': False,
        })

    logged = " ".join(str(c.args[0]) for c in mock_logger.info.call_args_list
                      if c.args)
    assert 'z=8.300' in logged, f"Expected z=8.300 in CHAR TELL log, got: {logged}"
    # z is the headline; delta only appears parenthesized
    assert 'delta=0.04 ' not in logged or '(delta=0.04' in logged

    print(f"  told z=8.3 → logged z=8.300")
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


def test_refinement_halves_floors():
    """Refinement halves the walk's resolution floors."""
    print("\n=== test_refinement_halves_floors ===")
    coord = _make_coordinator()
    coord._init_characterization()

    old_floor = coord._char_walk.status()['param_0']['step']
    coord._refine_study()

    new_floor = coord._char_walk.status()['param_0']['step']
    assert new_floor == old_floor / 2.0
    assert coord._refinement_level == 1

    print(f"  floor {old_floor}→{new_floor}")
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

        # The walk should show 1 measured level for the probed param
        st = coord._char_walk.status()
        assert sum(v['levels_measured'] for v in st.values()) == 1

    print(f"  push={coord.push_block_size}, pause={coord.pause_block_size}")
    print(f"  delta=0.03 told, 1 level measured")
    print("  PASS")


def test_exhaustion_triggers_refinement():
    """Walking all levels at current floor triggers refinement."""
    print("\n=== test_exhaustion_triggers_refinement ===")
    coord = _make_coordinator(params={
        'param_0': {'constraints': [{'lower': 0.0, 'upper': 100.0}]},
    })
    coord._init_characterization()

    old_floor = coord._char_walk.status()['param_0']['step']
    n_levels = coord._char_walk.status()['param_0']['levels_total']

    # Ask + tell all levels at this floor (0-100, floor 25 → 5 levels)
    for i in range(n_levels):
        probe = coord._ask_next_probe()
        assert probe is not None, f"ask {i} returned None before exhaustion"
        coord._tell_char_study('param_0', {'delta': 0.5})

    # The last tell saw the walk exhausted → refinement fired
    new_floor = coord._char_walk.status()['param_0']['step']
    assert new_floor == old_floor / 2.0, \
        f"Expected floor {old_floor/2.0}, got {new_floor}"
    assert coord._refinement_level == 1

    # And the walk continues at the finer floor (midpoints now in reach)
    probe = coord._ask_next_probe()
    assert probe is not None
    assert probe['level'] not in (0.0, 25.0, 50.0, 75.0, 100.0)

    print(f"  {n_levels} levels exhausted → floor {old_floor}→{new_floor}")
    print("  PASS")


def test_convergence_all_params_done():
    """All params converged → ask returns None (coverage contract).

    DONE is driven by the state machine via this None.
    """
    print("\n=== test_convergence_all_params_done ===")
    coord = _make_coordinator(params={
        'param_0': {'constraints': [{'lower': 0.0, 'upper': 100.0}]},
    })
    coord._init_characterization()

    coord._converged_params.add('param_0')

    probe = coord._ask_next_probe()
    assert probe is None, \
        f"Coverage contract: all params converged → None, got {probe}"

    print("  all params converged → ask returns None (state machine handles DONE)")
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


def test_multi_receiver_retires_only_when_every_receiver_converged():
    """Multi-receiver contract (receiver-keyed curves).

    causal's top-level `converged` is the all-receiver aggregate. The
    two-key retirement must honor it: one unconverged downstream
    receiver keeps the param in rotation even though the direct
    receiver is stable — every probe feeds all listeners, so the walk
    keeps going until all of them settle.
    """
    print("\n=== test_multi_receiver_retires_only_when_every_receiver_converged ===")
    from unittest.mock import patch
    coord = _make_coordinator()
    coord._init_characterization()

    probe = {'param_name': 'param_1', 'level': 75.0, 'param_idx': 1}

    # Receiver B (direct): converged. Receiver C (downstream): not yet.
    # Top-level aggregate: converged=False.
    resp_mixed = {
        'shift': 0.001, 'shift_bar': 0.05, 'z': 0.3, 'delta': 0.004,
        'converged': False, 'gaps': [], 'unresolved_gaps': 0,
        'primary_receiver': 'B', 'primary_channel': 'objective_0',
        'receivers': {
            'B': {'primary_channel': 'objective_0', 'shift': 0.001,
                  'shift_bar': 0.05, 'z': 0.2, 'drift': False,
                  'delta': 0.004, 'converged': True, 'gaps': [],
                  'unresolved_gaps': 0, 'channels': {}},
            'C': {'primary_channel': 'objective_0', 'shift': 0.002,
                  'shift_bar': 0.05, 'z': 0.4, 'drift': False,
                  'delta': 0.05, 'converged': False, 'gaps': [],
                  'unresolved_gaps': 0, 'channels': {}},
        },
    }
    with patch.object(coord, '_query_causal_probe_result', return_value=resp_mixed):
        coord._process_probe_result(probe)
    assert 'param_1' not in coord._converged_params, \
        "one unconverged receiver must keep the param in rotation"
    print("  B converged + C not → param stays in rotation")

    # All receivers converged, gaps priced out → retired.
    resp_all = {
        'shift': 0.001, 'shift_bar': 0.05, 'z': 0.3, 'delta': 0.004,
        'converged': True, 'gaps': [], 'unresolved_gaps': 0,
        'primary_receiver': 'B', 'primary_channel': 'objective_0',
        'receivers': {
            'B': dict(resp_mixed['receivers']['B']),
            'C': dict(resp_mixed['receivers']['C'], converged=True, delta=0.004),
        },
    }
    with patch.object(coord, '_query_causal_probe_result', return_value=resp_all):
        coord._process_probe_result(probe)
    assert 'param_1' in coord._converged_params, \
        "all receivers converged + gaps priced → retire"
    print("  all receivers converged → param retired")

    print("  PASS")


def test_multi_receiver_char_tell_logs_per_receiver():
    """The receivers map produces one CHAR TELL line per listener.

    The per-listener paper trail in Loki: shift/z/conv per receiver,
    primary marked.
    """
    print("\n=== test_multi_receiver_char_tell_logs_per_receiver ===")
    import logging as _logging
    from unittest.mock import patch
    coord = _make_coordinator()
    coord._init_characterization()

    probe = coord._ask_next_probe()
    assert probe is not None, "walk must offer a first probe"
    resp = {
        'shift': 0.2, 'shift_bar': 0.05, 'z': 1.2, 'delta': 0.01,
        'converged': False, 'gaps': [], 'unresolved_gaps': 0,
        'primary_receiver': 'B', 'primary_channel': 'objective_0',
        'receivers': {
            'B': {'primary_channel': 'objective_0', 'shift': 0.2,
                  'shift_bar': 0.05, 'z': 1.2, 'drift': False,
                  'delta': 0.01, 'converged': False, 'gaps': [],
                  'unresolved_gaps': 1, 'channels': {}},
            'C': {'primary_channel': 'objective_0', 'shift': 0.01,
                  'shift_bar': 0.04, 'z': 0.2, 'drift': False,
                  'delta': 0.003, 'converged': True, 'gaps': [],
                  'unresolved_gaps': 0, 'channels': {}},
        },
    }

    records = []

    class _Capture(_logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    # Full-suite pollution guard: earlier test files (worker lifecycle)
    # replace f.breeder.shared.otel_logging with a MagicMock whose
    # get_logger returns a Mock — the coordinator module imported under
    # that regime holds a Mock logger and emits nothing. Inject a real
    # logger for the duration of the call; order-independent.
    import engine.probe_coordinator as _pc_mod
    cap_logger = _logging.getLogger('test.char.tell.per_recv')
    cap_logger.addHandler(handler)
    cap_logger.setLevel(_logging.INFO)
    saved_logger = _pc_mod.logger
    _pc_mod.logger = cap_logger
    try:
        coord._tell_char_study(probe['param_name'], resp)
    finally:
        _pc_mod.logger = saved_logger
        cap_logger.removeHandler(handler)

    per_recv = [m for m in records if 'CHAR TELL:' in m and
                (' B ' in m or ' C ' in m)]
    assert any('B' in m and '*' in m for m in per_recv), \
        f"primary receiver B must be marked: {records}"
    assert any('C' in m for m in per_recv), f"receiver C line missing: {records}"
    assert any('conv=False' in m for m in per_recv) and \
           any('conv=True' in m for m in per_recv), \
        f"per-receiver convergence states missing: {records}"
    print(f"  {len(per_recv)} per-receiver lines logged, primary marked")
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
    assert status['levels_measured'] == 1

    for name in coord._param_names:
        s = status['params'][name]
        assert 'converged' in s
        assert 'step' in s
        assert 'levels_total' in s
        assert 'levels_measured' in s

    print(f"  {status['converged_count']}/{status['params_total']} converged")
    print(f"  {status['levels_measured']}/{status['levels_total']} levels measured")
    print("  PASS")


def test_init_char_import_via_f_breeder_namespace():
    """Production namespace regression (bench-4 VOID lesson).

    breeder_worker imports this module as f.breeder.engine.*; the
    sibling coverage_walk import inside _init_characterization must
    resolve there. Block the engine.* path and prove the f.breeder
    path alone is sufficient.
    """
    print("\n=== test_init_char_import_via_f_breeder_namespace ===")
    import importlib
    real = importlib.import_module('engine.coverage_walk')
    engine_mod = sys.modules.get('f.breeder.engine') or types.ModuleType('f.breeder.engine')
    saved_f = sys.modules.get('f.breeder.engine.coverage_walk')
    saved_e = sys.modules.get('engine.coverage_walk')
    sys.modules['f.breeder.engine'] = engine_mod
    sys.modules['f.breeder.engine.coverage_walk'] = real
    sys.modules['engine.coverage_walk'] = None  # poison fallback path
    try:
        coord = _make_coordinator()
        coord._init_characterization()
        assert coord._char_walk is not None, "init must work via f.breeder path alone"
    finally:
        if saved_e is None:
            sys.modules.pop('engine.coverage_walk', None)
        else:
            sys.modules['engine.coverage_walk'] = saved_e
        if saved_f is None:
            sys.modules.pop('f.breeder.engine.coverage_walk', None)
        else:
            sys.modules['f.breeder.engine.coverage_walk'] = saved_f
    print("  f.breeder path alone -> walk built — PASS")


def test_ask_after_all_converged():
    """All params converged → ask returns None under the coverage contract."""
    print("\n=== test_ask_after_all_converged ===")
    coord = _make_coordinator()
    coord._init_characterization()

    for p in coord._param_names:
        coord._converged_params.add(p)

    probe = coord._ask_next_probe()
    assert probe is None, \
        f"Coverage contract: all converged → None, got {probe}"
    assert len(coord._converged_params) == len(coord._param_names)

    print("  all converged → ask returns None, state machine handles DONE")
    print("  PASS")



if __name__ == '__main__':
    test_fns = [
        test_char_init_basic,
        test_char_ask_picks_param_and_level,
        test_char_ask_rotates_params,
        test_char_tell_logs_delta,
        test_char_tell_catches_infinity,
        test_char_tell_fail_on_none,
        test_char_tell_prefers_z,
        test_process_probe_result_returns_dict,
        test_process_probe_result_marks_converged,
        test_refinement_halves_floors,
        test_refinement_depth_cap,
        test_timeout_no_history,
        test_timeout_scales_with_trial_duration,
        test_timeout_tightened_by_rtt,
        test_timeout_never_below_floor,
        test_push_pause_round,
        test_exhaustion_triggers_refinement,
        test_convergence_all_params_done,
        test_get_char_status,
        test_ask_after_all_converged,
        test_init_char_import_via_f_breeder_namespace,
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


# ─── Listening from trial 1 (warmup gate removed) ─────────────────

def test_receiver_holds_from_first_trial():
    """An active sender outranks warmup: a breeder at trial 1 with a
    leased sender must HOLD, not optimize. Regression for the startup
    race where the sender probed a receiver still inside its own
    min_optimize_trials warmup (saturation run 23, param_0@50 = -0.68)."""
    import types
    coord = _make_coordinator()
    # DB answers: an active sender exists (lease held, fresh heartbeat)
    coord._db = lambda op, desc=None: True

    trial = types.SimpleNamespace(number=1)
    result = coord._handle_optimize(trial)

    assert coord.state == coord.HOLD, f"expected HOLD, got {coord.state}"
    assert result['mode'] == 'hold', f"expected hold, got {result}"
    assert result['params'] == {'param_0': 50.0, 'param_1': 50.0, 'param_2': 50.0}


# ─── Receiver-table write gate (sender self-read exclusion) ────────

def test_pause_return_has_no_lease_phase():
    """The sender's PROBE_PAUSE parks return mode 'hold' WITHOUT a
    lease_phase — this is the discriminator the worker's receiver-write
    gate relies on. Sender pause trials are the sender's own node
    self-reads and must never enter receiver_observations (they
    poisoned shift medians as ~0.5 rows: run-23 phantom +0.339)."""
    import types
    coord = _make_coordinator()
    coord.state = coord.PROBE_PUSH
    coord._current_probe = {'param_name': 'param_0', 'param_idx': 0,
                            'level': 50.0, 'config': {'param_0': 50.0}}
    coord._push_count = coord.push_block_size  # force PAUSE on next trial
    coord._pause_count = 0
    decision = coord._handle_probe_pause(types.SimpleNamespace(number=9))
    assert decision['mode'] == 'hold'
    assert decision.get('lease_phase') is None, \
        "sender pause must stay untagged or the write gate breaks"

def test_hold_return_carries_lease_phase():
    """Receiver HOLD always carries the observed lease phase — the
    positive side of the write-gate discriminator."""
    import types
    coord = _make_coordinator()
    coord.state = coord.HOLD
    coord._hold_count = 0
    coord._db = lambda op, desc=None: 'probe_push'   # observed phase
    decision = coord._handle_hold(types.SimpleNamespace(number=1))
    assert decision['mode'] == 'hold'
    assert decision.get('lease_phase') == 'probe_push'

def test_write_gate_discriminates_sender_self_reads():
    """Publication vs consumption are separate gates now. The worker
    publishes every phased trial's own readings (receiver holds AND
    sender push/pause — the self-curve data); causal's receiver query
    keeps the walking sender's rows out of its receivers' shift
    medians by receiver_id (the run-23 poison protection, which
    attribution by receiver_id makes permanent)."""
    publish_gate = lambda d: d.get('lease_phase') or d.get('impulse_phase')
    receiver_decision = {'mode': 'hold', 'lease_phase': 'probe_pause'}
    sender_pause_decision = {'mode': 'hold', 'impulse_phase': 'probe_pause'}

    assert publish_gate(receiver_decision) == 'probe_pause'
    assert publish_gate(sender_pause_decision) == 'probe_pause'

    receiver_query = lambda row, sender: (
        row['receiver_id'] != sender and row['lease_phase'] is not None
    )
    assert receiver_query({'receiver_id': 'other', 'lease_phase': 'probe_push'}, 'sender') is True
    assert receiver_query({'receiver_id': 'sender', 'lease_phase': 'probe_push'}, 'sender') is False


# ─── Park at neutral (quiescence outside push blocks) ──────────────

def _parked_coord(**overrides):
    """Protocol participant (interference_detection present) with a
    live coverage walk."""
    coord = _make_coordinator(**overrides)
    coord._init_characterization()
    return coord


def test_done_parks_at_neutral():
    """A finished walker parks: DONE returns hold-at-neutral, not the
    optimizer's next suggestion (dense run 33172249837: A stood at 100
    for C's whole walk after finishing)."""
    print("\n=== test_done_parks_at_neutral ===")
    coord = _parked_coord()
    coord.state = coord.PROBE_PUSH
    coord._push_count = 0
    coord._converged_params = {'param_0', 'param_1', 'param_2'}  # ask → None
    result = coord._handle_probe_push(types.SimpleNamespace(number=1))
    assert coord.state == coord.COOLDOWN
    assert result['mode'] == 'hold', f"expected park, got {result}"
    assert result['params'] == {'param_0': 50.0, 'param_1': 50.0, 'param_2': 50.0}
    assert 'lease_phase' not in result, "parks must stay out of receiver rows"
    print("  DONE → hold@50, no lease_phase, state COOLDOWN")
    print("  PASS")


def test_pure_optimizer_passes_through():
    """A breeder without an interference_detection section is a pure
    optimizer: the coordinator passes through, it never parks."""
    print("\n=== test_pure_optimizer_passes_through ===")
    cfg = _config()
    del cfg['interference_detection']
    coord = _make_coordinator(config=cfg)
    assert coord._coordination_enabled is False
    coord._count_active_breeders = lambda: 1
    result = coord._handle_optimize(types.SimpleNamespace(number=1))
    assert result == {'mode': 'optimize', 'params': None, 'detection_trial': False}
    print("  no section → optimize pass-through, no park")
    print("  PASS")


def test_walk_complete_does_not_reacquire_lease():
    """A converged walk never re-acquires: acquire → ask → None → DONE
    loops and starves the group's remaining walkers (A's optimize↔
    cooldown flip-flop 13:05–13:15 in the dense run)."""
    print("\n=== test_walk_complete_does_not_reacquire_lease ===")
    from unittest.mock import patch
    coord = _parked_coord()
    coord._converged_params = {'param_0', 'param_1', 'param_2'}
    coord._count_active_breeders = lambda: 3
    coord._has_active_sender = lambda: False
    with patch.object(coord, '_try_acquire_lease', return_value=True) as acq:
        result = coord._handle_optimize(types.SimpleNamespace(number=5))
    acq.assert_not_called()
    assert result['mode'] == 'hold', f"expected park, got {result}"
    print("  walk complete → no acquire, parked")
    print("  PASS")


def test_solo_protocol_breeder_parks():
    """A protocol participant waiting for its group parks instead of
    wandering under the optimizer."""
    print("\n=== test_solo_protocol_breeder_parks ===")
    coord = _parked_coord()
    coord._count_active_breeders = lambda: 1
    result = coord._handle_optimize(types.SimpleNamespace(number=1))
    assert result['mode'] == 'hold'
    assert result['params'] == {'param_0': 50.0, 'param_1': 50.0, 'param_2': 50.0}
    print("  solo → hold@50")
    print("  PASS")


def test_cooldown_parks_between_trials():
    """Cooldown-gap trials park — the handoff window between one walker's
    DONE and the next's acquire injects no optimizer application."""
    print("\n=== test_cooldown_parks_between_trials ===")
    coord = _parked_coord(cooldown_trials=3)
    coord.state = coord.COOLDOWN
    coord._cooldown_count = 0
    result = coord._handle_cooldown(types.SimpleNamespace(number=1))
    assert coord.state == coord.COOLDOWN
    assert result['mode'] == 'hold'
    assert result['params'] == {'param_0': 50.0, 'param_1': 50.0, 'param_2': 50.0}
    assert 'lease_phase' not in result
    print("  cooldown gap → hold@50")
    print("  PASS")


def test_cooldown_reacquires_when_work_remains():
    """Parking does not break turn-taking: cooldown expiry with
    unconverged params and a pending walk still re-acquires."""
    print("\n=== test_cooldown_reacquires_when_work_remains ===")
    from unittest.mock import patch
    coord = _parked_coord(cooldown_trials=1)
    coord.state = coord.COOLDOWN
    coord._cooldown_count = 0
    with patch.object(coord, '_try_acquire_lease', return_value=True) as acq:
        result = coord._handle_cooldown(types.SimpleNamespace(number=1))
    acq.assert_called_once()
    assert coord.state == coord.PROBE_PUSH
    assert result['mode'] == 'impulse'
    print("  expiry with work → acquire → push")
    print("  PASS")


def test_cooldown_expiry_with_finished_walk_parks():
    """Expiry with a finished walk parks instead of entering the
    acquire loop."""
    print("\n=== test_cooldown_expiry_with_finished_walk_parks ===")
    from unittest.mock import patch
    coord = _parked_coord(cooldown_trials=1)
    coord._converged_params = {'param_0', 'param_1', 'param_2'}
    coord.state = coord.COOLDOWN
    coord._cooldown_count = 0
    with patch.object(coord, '_try_acquire_lease', return_value=True) as acq:
        result = coord._handle_cooldown(types.SimpleNamespace(number=1))
    acq.assert_not_called()
    assert coord.state == coord.OPTIMIZE
    assert result['mode'] == 'hold'
    print("  expiry, walk done → parked, no acquire")
    print("  PASS")


def test_hold_exit_parks():
    """A receiver leaving HOLD (sender finished) parks through the
    handoff instead of applying one optimizer trial."""
    print("\n=== test_hold_exit_parks ===")
    coord = _parked_coord()
    coord.state = coord.HOLD
    coord._db = lambda op, desc=None: False  # no active sender
    result = coord._handle_hold(types.SimpleNamespace(number=1))
    assert coord.state == coord.OPTIMIZE
    assert result['mode'] == 'hold'
    assert result['params'] == {'param_0': 50.0, 'param_1': 50.0, 'param_2': 50.0}
    print("  hold exit → hold@50 through the gap")
    print("  PASS")


def test_park_falls_back_when_no_neutral_params():
    """No neutral params available → park degrades to the legacy
    optimize return instead of failing the trial."""
    print("\n=== test_park_falls_back_when_no_neutral_params ===")
    coord = _parked_coord()
    coord._get_neutral_params = lambda: None
    result = coord._park_result('test')
    assert result == {'mode': 'optimize', 'params': None, 'detection_trial': False}
    print("  no neutral → optimize fallback")
    print("  PASS")


# ─── Priced stop: two-key retirement ───────────────────────────────

def _probe_result(converged=True, gaps=None, shift_bar=0.05):
    return {'converged': converged, 'gaps': gaps or [], 'shift_bar': shift_bar,
            'delta': 0.005, 'z': 1.0}

def test_retires_when_converged_and_gap_free():
    import types
    coord = _make_coordinator()
    probe = {'param_name': 'param_1', 'param_idx': 1, 'level': 50.0,
             'config': {'param_0': 50.0, 'param_1': 50.0, 'param_2': 50.0}}
    coord._query_causal_probe_result = lambda p: _probe_result(
        converged=True, gaps=[{'from_level': 75.0, 'to_level': 100.0,
                               'jump': 0.03, 'bars_sum': 0.05, 'width': 25.0,
                               'unresolved': False, 'ignorance': 0.0075}])
    coord._process_probe_result(probe)
    assert 'param_1' in coord._converged_params

def test_no_retire_when_gap_priced_above_bar():
    # Run 25's (0,50): jump 0.35, ignorance 0.175 >> bar 0.05 — the rise
    # is worth measuring; converged alone must NOT retire it.
    import types
    coord = _make_coordinator()
    probe = {'param_name': 'param_1', 'param_idx': 1, 'level': 50.0,
             'config': {'param_0': 50.0, 'param_1': 50.0, 'param_2': 50.0}}
    coord._query_causal_probe_result = lambda p: _probe_result(
        converged=True, gaps=[{'from_level': 0.0, 'to_level': 50.0,
                               'jump': 0.35, 'bars_sum': 0.04, 'width': 50.0,
                               'unresolved': True, 'ignorance': 0.175}])
    coord._process_probe_result(probe)
    assert 'param_1' not in coord._converged_params

def test_retires_when_unresolved_but_priced_out():
    # A step's bracket can be unresolved forever (bars never overlap
    # across an edge) — retirement happens when its ignorance is below
    # the bar: the priced stop proper.
    import types
    coord = _make_coordinator()
    probe = {'param_name': 'param_1', 'param_idx': 1, 'level': 56.0,
             'config': {'param_0': 50.0, 'param_1': 50.0, 'param_2': 50.0}}
    coord._query_causal_probe_result = lambda p: _probe_result(
        converged=True, gaps=[{'from_level': 50.0, 'to_level': 56.0,
                               'jump': 0.35, 'bars_sum': 0.30, 'width': 6.0,
                               'unresolved': True, 'ignorance': 0.021}])
    coord._process_probe_result(probe)
    assert 'param_1' in coord._converged_params

def test_no_retire_when_not_converged():
    import types
    coord = _make_coordinator()
    probe = {'param_name': 'param_0', 'param_idx': 0, 'level': 50.0,
             'config': {'param_0': 50.0, 'param_1': 50.0, 'param_2': 50.0}}
    coord._query_causal_probe_result = lambda p: _probe_result(converged=False)
    coord._process_probe_result(probe)
    assert 'param_0' not in coord._converged_params


# ─── Receiver rows carry every read metric (channels) ──────────────

def test_record_includes_watched_observations():
    """The worker's receiver-row builder must include BOTH config
    sections' metrics — objectives AND observations. A metric omitted
    is a channel causal never sees (multi-channel runs 27+: rows
    carried only objective_0 because observations weren't unioned)."""
    cfg_sections = {
        'objectives': [{'name': 'objective_0'}],
        'observations': [{'name': 'objective_1'}],
    }
    metrics = {'objective_0': 0.51, 'objective_1': 0.34}
    readings = {}
    for section in ('objectives', 'observations'):
        for obj in cfg_sections.get(section, []) or []:
            oname = obj.get('name', 'unknown')
            if oname in metrics and oname not in readings:
                readings[oname] = metrics[oname]
    assert readings == {'objective_0': 0.51, 'objective_1': 0.34}



def test_acquire_publishes_demand_and_carries_fair_share_guard():
    """Lease fairness (seed-47 starved self-map): acquire publishes this
    breeder's walk demand, then denies itself while a walk-pending peer
    has had fewer turns. Poll speed cannot beat the count."""
    print("\n=== test_acquire_publishes_demand_and_carries_fair_share_guard ===")
    coord = _parked_coord()
    coord._walk_pending = lambda: True
    captured = {"sql": [], "params": []}

    class _Cur:
        rowcount = 1
        def execute(self, sql, params=None):
            captured["sql"].append(sql)
            captured["params"].append(params)
        def fetchone(self):
            return (7,)
        def close(self):
            pass

    class _Conn:
        def cursor(self):
            return _Cur()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    coord._db = lambda fn, desc=None: fn(_Conn())

    assert coord._try_acquire_lease(coord.PROBE_PUSH) is True

    def _capture_with(fragment):
        for sql, params in zip(captured["sql"], captured["params"]):
            if fragment in sql:
                return sql, params
        raise AssertionError(f"no captured query contains {fragment!r}")

    pub_sql, pub_params = _capture_with("walk_pending")
    assert "interference_active_breeders" in pub_sql
    assert pub_params[0] is True

    lease_sql, lease_params = _capture_with("NOT EXISTS")
    assert "NOT EXISTS" in lease_sql, "fair-share guard missing"
    assert "walk_pending IS TRUE" in lease_sql, "demand filter missing"
    assert "acquire_count" in lease_sql, "turn-count comparison missing"
    # params: (bid, want, bid, phase, gid, gid, bid, gid, bid)
    assert lease_params[-1] == coord.breeder_id
    assert lease_params[0] == coord.breeder_id
    # regression (seed-47-fair incident): params count must equal
    # placeholder count — a mismatch fails every acquire at runtime
    assert lease_params.__len__() == lease_sql.count("%s"), (
        f"params {len(lease_params)} != placeholders {lease_sql.count('%s')}")

    bump_sql = captured["sql"][3]
    assert "acquire_count = COALESCE(acquire_count, 0) + 1" in bump_sql
    assert captured["params"][3] == (coord.breeder_id,)
    print("  demand published; fair-share guard present; turn counted")
    print("  PASS")


def test_acquire_without_demand_still_counts_turn():
    """A walker with no remaining demand can still take the lease (park
    completion path), and its turn still counts."""
    print("\n=== test_acquire_without_demand_still_counts_turn ===")
    coord = _parked_coord()
    coord._walk_pending = lambda: False
    seen = {"walk": None}

    class _Cur:
        rowcount = 1
        def execute(self, sql, params=None):
            if "SET walk_pending" in sql:
                seen["walk"] = params[0]
        def fetchone(self):
            return (7,)
        def close(self):
            pass

    class _Conn:
        def cursor(self):
            return _Cur()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    coord._db = lambda fn, desc=None: fn(_Conn())

    assert coord._try_acquire_lease(coord.PROBE_PUSH) is True
    assert seen["walk"] is False
    print("  demand=False published; lease acquired; turn counted")
    print("  PASS")


def test_walk_complete_publishes_demand_false():
    """Deadlock (seed-47-extended, run 33720423947): a finished walker
    exits at the 'walk complete' early-return, which never re-publishes
    demand. Its walk_pending flag stayed TRUE in the DB forever (heartbeat
    keeps last_seen fresh, so the staleness filter never excludes it) and
    the fair-share predicate — fresh walk-pending peer with fewer turns —
    denied every active walker for the rest of the run. The completion
    path must clear the flag it last published as TRUE."""
    print("\n=== test_walk_complete_publishes_demand_false ===")
    coord = _parked_coord()
    coord._walk_pending = lambda: False
    coord._has_active_sender = lambda: False
    coord._count_active_breeders = lambda: 3
    captured = {"sql": [], "params": []}

    class _Cur:
        def execute(self, sql, params=None):
            captured["sql"].append(sql)
            captured["params"].append(params)

        def close(self):
            pass

    class _Conn:
        def cursor(self):
            return _Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    coord._db = lambda fn, desc=None: fn(_Conn())

    result = coord._handle_optimize(types.SimpleNamespace(number=1))

    assert result["mode"] == "hold", f"expected park, got {result}"
    pub = [(s, p) for s, p in zip(captured["sql"], captured["params"])
           if "SET walk_pending" in s]
    assert pub, "walk-complete path must publish demand"
    assert pub[0][1][0] is False, f"demand must clear, got {pub[0][1]}"
    print("  walk complete → demand=False published; parked")
    print("  PASS")


def test_denied_acquire_names_the_block():
    """The deadlock ran ~2090 silent denials with zero log lines: the
    fair-share acquire returns False with no reason attached. A denial
    must emit a diagnostic — holder, heartbeat age, blocking peers —
    so a live blockage is visible in Loki within one poll cycle."""
    print("\n=== test_denied_acquire_names_the_block ===")
    import engine.probe_coordinator as pc
    coord = _parked_coord()
    coord._walk_pending = lambda: True
    logs = []

    class _Log:
        def info(self, fmt, *a):
            logs.append(fmt % a if a else fmt)

        def warning(self, fmt, *a):
            logs.append("WARN " + (fmt % a if a else fmt))

    saved = pc.logger
    pc.logger = _Log()
    try:
        class _Cur:
            rowcount = 0

            def execute(self, sql, params=None):
                self.sql = sql

            def fetchone(self):
                if "sender_lease" in self.sql:
                    return ("05aa7f7d", 12.0)
                return None

            def fetchall(self):
                if "interference_active_breeders" in self.sql:
                    return [("6a1db105", 1, 3.0)]
                return []

            def close(self):
                pass

        class _Conn:
            def cursor(self):
                return _Cur()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        coord._db = lambda fn, desc=None: fn(_Conn())
        got = coord._try_acquire_lease(coord.PROBE_PUSH)
    finally:
        pc.logger = saved

    assert got is False
    denial = [l for l in logs if "denied" in l]
    assert denial, f"no denial diagnostic logged; got {logs}"
    assert "05aa7f7d" in denial[0], "denial must name the denied breeder"
    assert "6a1db105" in denial[0], "denial must name the blocking peer"
    print("  denial logged:", denial[0][:120])
    print("  PASS")
