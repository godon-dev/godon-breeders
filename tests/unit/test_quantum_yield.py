"""Quantum yield + role ledger tests — the anti-monopoly rung.

Seed-50 replay: one breeder held the lease for its entire walk while
three followers held to cap, never walking. These tests pin the cure:
a full slice with a pending peer yields the lease (at a completed
probe-cycle boundary only), hold trials stop consuming the iteration
budget, and a needier pending peer defers the acquire.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import pytest

from engine.probe_coordinator import ProbeCoordinator


def _config(**overrides):
    params = {'param_0': {'constraints': [{'lower': 0.0, 'upper': 100.0}]}}
    cfg = {
        'breeder': {'type': 'bench_generic', 'uuid': 'quantum-1'},
        'settings': {'generic': params},
        'interference_detection': {
            'group': 'bench-characterization',
            'hold_params': {'param_0': 50.0},
            'push_block_size': 10,
            'pause_block_size': 10,
            'cooldown_trials': 5,
            'convergence_threshold': 0.005,
            'refinement_depth': 3,
            'walk_policy': 'ladder',
            'quantum_cycles': 2,
        },
    }
    cfg['interference_detection'].update(overrides)
    return cfg


def _coordinator(**overrides):
    return ProbeCoordinator(
        breeder_id='B4',
        config=_config(**overrides),
        shared_db_fn=lambda op, label=None: None,
        collect_upper_bounds_fn=lambda settings: [
            {'name': 'param_0', 'lower': 0.0, 'upper': 100.0,
             'is_int': False, 'idx': 0},
        ],
        compute_neutral_params_fn=lambda: {'param_0': 50.0},
    )


class _WalkStub:
    """Stands in for the walk: always has a next probe."""

    def next_probe(self, skip):
        return 'param_0', 25.0

    def can_probe(self, skip):
        return True

    def refine(self):
        pass

    def status(self):
        return {'param_0': {'step': 25.0, 'levels_total': None,
                            'levels_measured': 3, 'levels': [0.0, 50.0, 100.0]}}


def _mid_cycle_coord(peer_pending, quantum=2):
    """A coordinator completing the LAST pause trial of a cycle."""
    coord = _coordinator(**{'quantum_cycles': quantum})
    coord._char_walk = _WalkStub()
    coord._live_peer_count = lambda: 3 if peer_pending else 0
    released = []
    coord._release_lease = lambda: released.append(1)
    coord._process_probe_result = lambda probe: {
        'converged': False, 'shift_bar': 0.02, 'gaps': []}
    coord.state = coord.PROBE_PAUSE
    coord._pause_count = 9   # pause_block_size 10 → this trial completes the cycle
    coord._push_count = 0
    coord._current_probe = {'param_name': 'param_0', 'level': 25.0, 'config': {}}
    coord._stretch_cycles = quantum - 1  # slice is up after this cycle
    coord._round_push_start = None
    coord._round_pause_end = None
    return coord, released


class TestQuantumYield:
    def test_full_slice_with_pending_peer_yields(self):
        coord, released = _mid_cycle_coord(peer_pending=True)
        res = coord._handle_probe_pause(trial=None)
        assert coord.state == coord.COOLDOWN, "yielded walker cools down, then re-acquires"
        assert released, "the lease must be released so peers can take the mic"
        assert coord._stretch_cycles == 0, "slice counter resets on yield"
        assert res.get('mode') != 'impulse', "no further push trials this stretch"

    def test_yield_keys_on_peer_existence_not_their_flag(self):
        """Seed-52 catch-22: peers in HOLD never publish walk_pending
        (they only publish it when attempting an acquire, which they
        can't do while a sender is active). The yield must key on their
        EXISTENCE — a served slice with live peers yields."""
        coord, released = _mid_cycle_coord(peer_pending=True)
        res = coord._handle_probe_pause(trial=None)
        assert coord.state == coord.COOLDOWN, "live peers exist -> yield the mic"
        assert released
    def test_unfilled_slice_does_not_yield(self):
        coord, released = _mid_cycle_coord(peer_pending=True, quantum=5)
        coord._stretch_cycles = 1  # slice not yet full
        res = coord._handle_probe_pause(trial=None)
        assert coord.state == coord.PROBE_PUSH
        assert not released


class TestRoleLedger:
    def test_walk_trials_gate_the_iteration_cap(self):
        import sys
        from unittest.mock import MagicMock

        stubs = {name: MagicMock() for name in [
            'f', 'f.breeder', 'f.breeder.engine',
            'f.breeder.engine.probe_coordinator',
            'f.breeder.engine.breeder_metrics_client',
            'f.breeder.engine.communication',
            'f.breeder.engine.coverage_walk',
            'f.breeder.engine.walk_policy',
            'f.breeder.strains', 'f.breeder.strains.bench_generic',
            'f.breeder.strains.bench_generic.strain',
            'wmill']}
        saved = {k: sys.modules.get(k) for k in stubs}
        sys.modules.update(stubs)
        try:
            from engine.breeder_worker import BreederWorker
            import types
            w = BreederWorker.__new__(BreederWorker)  # skip heavy init
            w.config = {'run': {'completion_criteria': {
                'iterations': {'min': 10, 'max': 120}}},
                'interference_detection': {}}
            w.study = types.SimpleNamespace(trials=list(range(500)))
            w._own_trials = 0
            w._check_time_budget = lambda cc: False
            w._check_shutdown_requested = lambda: False

            assert w._should_continue() is True, \
                "holding 500 trials must not stop a breeder that never worked"
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        w._own_trials = 130  # cap 120 exceeded by actual walking
        assert w._should_continue() is False, \
            "130 own-work trials over a 120 cap stops the breeder"


class TestNeedDefer:
    def test_defers_to_needier_pending_peer(self):
        coord = _coordinator()
        coord._walk_pending_peers = lambda: ["PEER"]
        coord._walk_transport = lambda method, url, payload=None: {
            "curves": [
                {"sender_id": "PEER", "receiver_id": "X", "param": "param_0",
                 "channel": "objective_0", "state": {
                     "converged": False, "num_points": 3,
                     "points": [[0.0, 0.0, 0.02], [50.0, 0.0, 0.02], [100.0, 1.0, 0.02]],
                     "gaps": [{"from_level": 50.0, "to_level": 100.0, "jump": 1.0,
                               "bars_sum": 0.04, "unresolved": True, "ignorance": 0.5}]}}]}

        assert coord._defer_to_needier_peer(), "needier pending peer → defer"

    def test_no_defer_when_own_need_is_higher(self):
        coord = _coordinator()
        coord._walk_transport = lambda method, url, payload=None: {
            "curves": [
                {"sender_id": "B4", "receiver_id": "X", "param": "param_0",
                 "channel": "objective_0", "state": {
                     "converged": False, "num_points": 3,
                     "points": [[0.0, 0.0, 0.02], [50.0, 0.0, 0.02], [100.0, 1.0, 0.02]],
                     "gaps": [{"from_level": 50.0, "to_level": 100.0, "jump": 1.0,
                               "bars_sum": 0.04, "unresolved": True, "ignorance": 0.9}]}}]}
        # Peer transport serves an empty notebook (no outstanding prices).
        def empty_peer(method, url, payload=None):
            return {"curves": []}
        # _defer_to_needier_peer walks pending peers via its peer query;
        # stub the peers to a single one with empty need.
        coord._walk_pending_peers = lambda: ["PEER"]
        coord._walk_transport = lambda method, url, payload=None: (
            {"curves": [
                {"sender_id": "PEER", "receiver_id": "X", "param": "param_0",
                 "channel": "objective_0", "state": {
                     "converged": True, "num_points": 3,
                     "points": [[0.0, 0.0, 0.02], [50.0, 0.0, 0.02], [100.0, 0.0, 0.02]],
                     "gaps": []}}]}
            if "PEER" in url else
            {"curves": [
                {"sender_id": "B4", "receiver_id": "X", "param": "param_0",
                 "channel": "objective_0", "state": {
                     "converged": False, "num_points": 3,
                     "points": [[0.0, 0.0, 0.02], [50.0, 0.0, 0.02], [100.0, 1.0, 0.02]],
                     "gaps": [{"from_level": 50.0, "to_level": 100.0, "jump": 1.0,
                               "bars_sum": 0.04, "unresolved": True, "ignorance": 0.9}]}}]})

        assert not coord._defer_to_needier_peer(), "own need higher → take the mic"
