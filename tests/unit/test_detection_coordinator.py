#
# Copyright (c) 2019 Matthias Tafelmeier.
#
# Tests for the DetectionCoordinator state machine.
#

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Mock heavy imports
sys.modules['wmill'] = MagicMock()
sys.modules['optuna'] = MagicMock()
sys.modules['optuna.storages'] = MagicMock()
sys.modules['optuna.trial'] = MagicMock()
from optuna.trial import TrialState
sys.modules['optuna.samplers'] = MagicMock()
sys.modules['prometheus_api_client'] = MagicMock()
sys.modules['prometheus_client'] = MagicMock()
sys.modules['psycopg2'] = MagicMock()

from engine.detection_coordinator import DetectionCoordinator


def _base_config():
    return {
        'breeder': {'uuid': 'test-uuid', 'name': 'test', 'type': 'greenhouse'},
        'objectives': [{'name': 'growth', 'direction': 'maximize'}],
        'detection': {'warmup_trials': 3, 'impulses_per_round': 3, 'recover_trials': 2},
    }


def _mock_db(return_value=True):
    """Mock _with_shared_db that returns a fixed value."""
    db = MagicMock(return_value=return_value)
    return db


def _mock_study(n_complete=0):
    """Create a mock study with n_complete complete trials."""
    study = MagicMock()
    trials = []
    for i in range(n_complete):
        t = MagicMock()
        t.state = TrialState.COMPLETE
        t.values = [0.5 + i * 0.01]
        t.user_attrs = {'effectuation_params': '{"heating": 20.0, "light": 300}'}
        trials.append(t)
    for i in range(5):
        t = MagicMock()
        t.state = TrialState.FAIL
        t.user_attrs = {}
        trials.append(t)
    study.trials = trials
    return study


def _create_coordinator(breeder_id='test-uuid', n_complete=0, db_return=True):
    config = _base_config()
    db = _mock_db(db_return)
    collect_fn = MagicMock(return_value=[
        {'name': 'heating', 'upper': 30, 'lower': 10, 'range': 20, 'is_int': False},
        {'name': 'light', 'upper': 1000, 'lower': 0, 'range': 1000, 'is_int': False},
    ])
    coord = DetectionCoordinator(
        breeder_id=breeder_id,
        config=config,
        shared_db_fn=db,
        collect_upper_bounds_fn=collect_fn,
    )
    return coord, db


class TestWarmup:
    def test_returns_optimize_during_warmup(self):
        coord, _ = _create_coordinator()
        study = _mock_study(n_complete=1)  # Less than warmup_target=3
        trial = MagicMock()
        with patch.object(coord, '_any_active_round', return_value=False):
            decision = coord.decide_trial(trial, study)
        assert decision['mode'] == 'optimize'
        assert decision['params'] is None
        assert coord.state == DetectionCoordinator.WARMUP

    def test_transitions_to_sender_after_warmup(self):
        coord, db = _create_coordinator()
        study = _mock_study(n_complete=3)  # Equals warmup_target
        trial = MagicMock()
        # db returns True for try_start_round
        with patch.object(coord, '_any_active_round', return_value=False):
            decision = coord.decide_trial(trial, study)
        assert decision['mode'] == 'optimize'  # Last warmup trial
        assert coord.state == DetectionCoordinator.SENDER_PING

    def test_transitions_to_receiver_if_cannot_start(self):
        coord, db = _create_coordinator(db_return=False)
        study = _mock_study(n_complete=3)
        trial = MagicMock()
        # Need _any_active_round to return True
        with patch.object(coord, '_any_active_round', return_value=True):
            decision = coord.decide_trial(trial, study)
        assert coord.state == DetectionCoordinator.RECEIVER_HOLD


class TestSenderPing:
    def test_ping_returns_impulse_with_extreme_params(self):
        coord, _ = _create_coordinator()
        coord.state = DetectionCoordinator.SENDER_PING
        coord._baseline_params = {'heating': 20.0, 'light': 300.0}
        trial = MagicMock()
        decision = coord.decide_trial(trial, MagicMock())
        assert decision['mode'] == 'impulse'
        assert decision['impulse_phase'] == 'ping'
        assert decision['params'] is not None
        # Heating should be at upper bound (30 * 1.0)
        assert decision['params']['heating'] == 30.0

    def test_ping_increments_counter(self):
        coord, _ = _create_coordinator()
        coord.state = DetectionCoordinator.SENDER_PING
        coord._baseline_params = {'heating': 20.0}
        trial = MagicMock()
        coord.decide_trial(trial, MagicMock())
        assert coord._ping_count == 1

    def test_ping_transitions_to_listen(self):
        coord, _ = _create_coordinator()
        coord.state = DetectionCoordinator.SENDER_PING
        coord._baseline_params = {'heating': 20.0}
        trial = MagicMock()
        coord.decide_trial(trial, MagicMock())
        assert coord.state == DetectionCoordinator.SENDER_LISTEN


class TestSenderListen:
    def test_listen_returns_baseline_params(self):
        coord, _ = _create_coordinator()
        coord.state = DetectionCoordinator.SENDER_LISTEN
        coord._baseline_params = {'heating': 20.0, 'light': 300.0}
        trial = MagicMock()
        decision = coord.decide_trial(trial, MagicMock())
        assert decision['mode'] == 'impulse'
        assert decision['impulse_phase'] == 'listen'
        assert decision['params']['heating'] == 20.0  # Baseline, not extreme

    def test_listen_transitions_back_to_ping_if_round_not_done(self):
        coord, _ = _create_coordinator()
        coord.state = DetectionCoordinator.SENDER_LISTEN
        coord._ping_count = 1  # Less than impulses_per_round=3
        coord._baseline_params = {'heating': 20.0}
        coord.decide_trial(MagicMock(), MagicMock())
        assert coord.state == DetectionCoordinator.SENDER_PING

    def test_listen_transitions_to_done_if_round_complete(self):
        coord, _ = _create_coordinator()
        coord.state = DetectionCoordinator.SENDER_LISTEN
        coord._ping_count = 3  # Equals impulses_per_round
        coord._baseline_params = {'heating': 20.0}
        coord.decide_trial(MagicMock(), MagicMock())
        assert coord.state == DetectionCoordinator.SENDER_DONE


class TestSenderDone:
    def test_completes_round_and_enters_recover(self):
        coord, _ = _create_coordinator()
        coord.state = DetectionCoordinator.SENDER_DONE
        decision = coord.decide_trial(MagicMock(), MagicMock())
        assert decision['mode'] == 'optimize'
        assert coord.state == DetectionCoordinator.RECOVER
        assert coord._recover_count == 0


class TestRecover:
    def test_optimizes_during_recovery(self):
        coord, _ = _create_coordinator()
        coord.state = DetectionCoordinator.RECOVER
        coord._recover_count = 0
        decision = coord.decide_trial(MagicMock(), MagicMock())
        assert decision['mode'] == 'optimize'
        assert coord._recover_count == 1

    def test_becomes_sender_after_recovery(self):
        coord, db = _create_coordinator()
        coord.state = DetectionCoordinator.RECOVER
        coord._recover_count = 1  # One more and it's done (target=2)
        coord._baseline_params = {'heating': 20.0}
        study = _mock_study(n_complete=5)
        decision = coord.decide_trial(MagicMock(), study)
        assert coord.state == DetectionCoordinator.SENDER_PING
        assert coord._ping_count == 0


class TestReceiverHold:
    def test_returns_hold_with_baseline_params(self):
        coord, _ = _create_coordinator()
        coord.state = DetectionCoordinator.RECEIVER_HOLD
        coord._baseline_params = {'heating': 20.0}
        decision = coord.decide_trial(MagicMock(), MagicMock())
        assert decision['mode'] == 'hold'
        assert decision['params']['heating'] == 20.0

    def test_enters_recover_when_sender_finishes(self):
        coord, _ = _create_coordinator()
        coord.state = DetectionCoordinator.RECEIVER_HOLD
        coord._baseline_params = {'heating': 20.0}
        with patch.object(coord, '_any_active_round', return_value=False):
            decision = coord.decide_trial(MagicMock(), MagicMock())
        assert coord.state == DetectionCoordinator.RECOVER


class TestGuardrailFail:
    def test_ping_fail_triggers_aimd_backoff(self):
        coord, _ = _create_coordinator()
        coord.state = DetectionCoordinator.SENDER_PING
        coord._ping_count = 2
        coord._baseline_params = {'heating': 20.0}
        # Generate impulse params first
        coord._get_impulse_params()
        original_scale = coord._impulse_scale
        coord.on_guardrail_fail({'heating': 30.0})
        assert coord._impulse_scale == original_scale * 0.5
        assert coord._ping_count == 1  # Decremented

    def test_listen_fail_does_not_trigger_backoff(self):
        coord, _ = _create_coordinator()
        coord.state = DetectionCoordinator.SENDER_LISTEN
        coord._impulse_scale = 1.0
        coord.on_guardrail_fail({'heating': 20.0})
        assert coord._impulse_scale == 1.0  # Unchanged


class TestStateCleanup:
    def test_cleans_up_stale_rounds_on_init(self):
        coord, db = _create_coordinator()
        trial = MagicMock()
        study = _mock_study(n_complete=0)
        coord.decide_trial(trial, study)
        # Should have called db at least once for cleanup
        assert db.call_count >= 1
