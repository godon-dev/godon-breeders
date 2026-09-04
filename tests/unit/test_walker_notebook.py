"""Seed-48 regression tests — the thin-map incident, pinned as behavior.

The incident: node-4's tail self-walk (a) retired while its OWN curve
carried an unresolved +0.996 bracket (self-curve exempt from the
verdict), and (b) re-walked the {50, 100, 0} prefix forever because the
walk plan lived in RAM and died at invocation boundaries. These tests
pin the cures: the notebook walk continues past the prefix and steers
into the fat self bracket; the two-key check prices each gap against
its own bars.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.probe_coordinator import ProbeCoordinator


def _config():
    params = {
        'param_0': {'constraints': [{'lower': 0.0, 'upper': 100.0}]},
    }
    return {
        'breeder': {'type': 'bench_generic', 'uuid': 'replay-1'},
        'settings': {'generic': params},
        'interference_detection': {
            'group': 'bench-characterization',
            'hold_params': {'param_0': 50.0},
            'push_block_size': 10,
            'pause_block_size': 10,
            'cooldown_trials': 5,
            'convergence_threshold': 0.005,
            'refinement_depth': 3,
            # two-key tests below don't exercise the walk; keep them
            # offline unless the notebook transport is injected
            'walk_policy': 'ladder',
        },
    }


# NOTE: the coordinator constructor is the DI seam for tests — if the
# constructor does not accept walk_transport, build via config instead.
def _coordinator_via_config(walk_transport=None, overrides=None):
    cfg = _config()
    if walk_transport is not None:
        cfg['interference_detection']['walk_transport'] = walk_transport
    cfg['interference_detection'].update(overrides or {})
    return ProbeCoordinator(
        breeder_id='B4',
        config=cfg,
        shared_db_fn=lambda op, label=None: None,
        collect_upper_bounds_fn=lambda settings: [
            {'name': 'param_0', 'lower': 0.0, 'upper': 100.0,
             'is_int': False, 'idx': 0},
        ],
        compute_neutral_params_fn=lambda: {'param_0': 50.0},
    )


def _seed48_view():
    """The notebook page as it stood when node-4's walk went thin:
    three banked levels everywhere, receiver curves priced out, and the
    walker's own curve carrying the fat unresolved step at (50, 100)."""
    return {
        "sender_id": "B4",
        "param": "param_0",
        "refinement_level": 0,
        "curves": [
            {"receiver_id": "R1", "channel": "objective_0",
             "converged": True, "levels": [0.0, 50.0, 100.0],
             "gaps": [
                 {"from_level": 0.0, "to_level": 50.0, "jump": 0.0063,
                  "bars_sum": 0.0375, "unresolved": False, "ignorance": 0.003},
                 {"from_level": 50.0, "to_level": 100.0, "jump": 0.0012,
                  "bars_sum": 0.0439, "unresolved": False, "ignorance": 0.0006},
             ]},
            {"receiver_id": "SELF", "channel": "objective_1",
             "converged": True, "levels": [0.0, 50.0, 100.0],
             "gaps": [
                 {"from_level": 50.0, "to_level": 100.0, "jump": 1.0043,
                  "bars_sum": 0.0412, "unresolved": True, "ignorance": 0.502},
             ]},
        ],
    }


class _Notebook:
    def __init__(self, view):
        self.view = view
        self.levels = list(view["curves"][-1]["levels"])

    def __call__(self, method, url, payload=None):
        if method == "GET":
            return self.view
        if method == "POST":
            return {"refinement_level": 1}
        raise AssertionError(method)


def test_notebook_walk_continues_past_prefix_into_self_bracket():
    """The exact thin-map moment: anchors banked, receivers priced out,
    self bracket fat. The old walk retired here at 3 points; the notebook
    walk must steer to (50,100)/2 = 75.0."""
    nb = _Notebook(_seed48_view())
    coord = _coordinator_via_config(walk_transport=nb)
    coord._init_characterization()
    probe = coord._ask_next_probe()
    assert probe is not None, "walk must NOT retire with its own bracket fat"
    assert probe['level'] == 75.0


def test_two_key_prices_each_gap_against_its_own_bars():
    """A quiet listener's gap (ignorance above its OWN bars, below a loud
    primary's bar) must block retirement — the shared-bar defect."""
    coord = _coordinator_via_config()
    coord._init_characterization()
    coord._converged_params = set()

    result = {
        'converged': True,
        'shift_bar': 0.08,  # loud primary listener's bar
        'gaps': [
            # quiet listener: price above ITS OWN fuzz, below the primary's
            {'from_level': 0.0, 'to_level': 50.0, 'jump': 0.2,
             'bars_sum': 0.03, 'unresolved': True, 'ignorance': 0.05},
            # genuinely priced out
            {'from_level': 50.0, 'to_level': 100.0, 'jump': 0.02,
             'bars_sum': 0.05, 'unresolved': True, 'ignorance': 0.02},
        ],
    }
    coord._query_causal_probe_result = lambda probe: result
    coord._process_probe_result({'param_name': 'param_0', 'level': 50.0})
    assert 'param_0' not in coord._converged_params, \
        "gap priced above its own bars must keep the param in rotation"


def test_two_key_releases_fully_priced_out_param():
    coord = _coordinator_via_config()
    coord._init_characterization()
    coord._converged_params = set()

    result = {
        'converged': True,
        'shift_bar': 0.02,
        'gaps': [
            {'from_level': 0.0, 'to_level': 50.0, 'jump': 0.0063,
             'bars_sum': 0.0375, 'unresolved': False, 'ignorance': 0.003},
        ],
    }
    coord._query_causal_probe_result = lambda probe: result
    coord._process_probe_result({'param_name': 'param_0', 'level': 50.0})
    assert 'param_0' in coord._converged_params, \
        "stability + fully priced out = the honest stop"


def test_notebook_unreachable_fails_loudly():
    """No silent rigid-ladder fallback: an unreachable notebook kills the
    init — the invocation dies visibly and the next one retries."""
    import pytest
    coord = _coordinator_via_config(overrides={'walk_policy': 'notebook'})
    with pytest.raises(Exception):
        coord._init_characterization()
    assert coord._char_walk is None
