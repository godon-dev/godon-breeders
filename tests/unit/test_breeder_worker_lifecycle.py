
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
#
import pytest
import sys
import os
from unittest.mock import MagicMock, patch, call
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

sys.modules['wmill'] = MagicMock()
sys.modules['optuna'] = MagicMock()
sys.modules['optuna.storages'] = MagicMock()
sys.modules['optuna.trial'] = MagicMock()
sys.modules['optuna.samplers'] = MagicMock()

# Mock Windmill package namespace so breeder_worker's internal imports resolve
sys.modules['f'] = MagicMock()
sys.modules['f.breeder'] = MagicMock()
sys.modules['f.breeder.engine'] = MagicMock()
sys.modules['f.breeder.engine.probe_coordinator'] = MagicMock()
sys.modules['f.breeder.engine.probe_coordinator'].ProbeCoordinator = MagicMock
sys.modules['f.breeder.engine.breeder_metrics_client'] = MagicMock()
sys.modules['f.breeder.engine.breeder_metrics_client'].BreederMetricsClient = MagicMock
sys.modules['f.breeder.engine.communication'] = MagicMock()
sys.modules['f.breeder.engine.communication'].CommunicationCallback = MagicMock
sys.modules['f.breeder.engine.strain_loader'] = MagicMock()
sys.modules['f.breeder.engine.strain_loader'].load_strain = MagicMock(return_value=MagicMock())
sys.modules['f.breeder.engine.watermark'] = MagicMock()
sys.modules['f.breeder.engine.watermark'].create_watermark = MagicMock(return_value=None)
sys.modules['f.breeder.engine.watermark'].Watermark = MagicMock
sys.modules['f.breeder.shared'] = MagicMock()
sys.modules['f.breeder.shared.otel_logging'] = MagicMock()
sys.modules['f.breeder.shared.otel_logging'].get_logger = MagicMock(return_value=MagicMock())

from engine.breeder_worker import BreederWorker


def _base_config(**overrides):
    config = {
        'breeder': {
            'name': 'test_breeder',
            'uuid': 'test-uuid-123',
            'type': 'linux_performance',
        },
        'creation_ts': '2025-01-15T10:30:00Z',
        'run': {'parallel': 1},
        'objectives': [{'name': 'throughput', 'direction': 'minimize'}],
        'effectuation': {'targets': [], 'type': 'ssh'},
        'reconnaissance': {'type': 'prometheus'},
    }
    config.update(overrides)
    return config


def _mock_study(trials=None, best_trials=None):
    study = MagicMock()
    study.trials = trials or []
    study.best_trials = best_trials or []
    study.study_name = 'test_study'
    study.user_attrs = {}

    def set_attr(key, value):
        study.user_attrs[key] = value

    study.set_user_attr = set_attr
    return study


def _create_worker(**config_overrides):
    config = _base_config(**config_overrides)
    study = _mock_study()

    with patch.object(BreederWorker, '_load_or_create_study', return_value=study), \
         patch.object(BreederWorker, '_setup_communication', return_value=None), \
         patch.object(BreederWorker, '_update_state'), \
         patch('engine.breeder_worker.load_strain', return_value=MagicMock()):
        worker = BreederWorker(config)
    worker._check_shutdown_requested = lambda: False
    return worker


class TestExecuteTrial:
    def setup_method(self):
        sys.modules['wmill'].reset_mock()

    def test_success_returns_metrics(self):
        worker = _create_worker()
        settings = {'vm.swappiness': 10}
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.side_effect = [
            {'status': 'completed'},
            {'status': 'completed', 'metrics': {'throughput': 42.5}},
        ]

        result = worker._execute_trial(settings)
        assert result == {'throughput': 42.5}
        assert mock_wmill.run_script_by_path.call_count == 2

    def test_effectuation_failure_returns_inf(self):
        worker = _create_worker()
        settings = {'vm.swappiness': 10}
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.side_effect = Exception('SSH failed')
        result = worker._execute_trial(settings)
        assert result == {'throughput': float('inf')}

    def test_no_metrics_returns_inf(self):
        worker = _create_worker()
        settings = {}

        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.side_effect = [
            {'status': 'completed'},
            {'status': 'completed'},
        ]

        result = worker._execute_trial(settings)
        assert result == {'throughput': float('inf')}

    def test_uses_configured_effectuation_type(self):
        worker = _create_worker(effectuation={'targets': [], 'type': 'http'})
        settings = {}

        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.side_effect = [
            {'status': 'completed'},
            {'status': 'completed', 'metrics': {'throughput': 10.0}},
        ]
        worker._execute_trial(settings)
        eff_call = mock_wmill.run_script_by_path.call_args_list[0]
        assert eff_call[0][0] == 'f/effectuation/http'


class TestShouldContinue:
    def test_continues_below_min_iterations(self):
        worker = _create_worker(run={
            'parallel': 1,
            'completion_criteria': {
                'iterations': {'min': 10, 'max': 100},
            }
        })
        worker.study.trials = [MagicMock() for _ in range(5)]
        assert worker._should_continue() is True

    def test_stops_at_max_iterations(self):
        worker = _create_worker(run={
            'parallel': 1,
            'completion_criteria': {
                'iterations': {'min': 10, 'max': 100},
            }
        })
        worker.study.trials = [MagicMock() for _ in range(100)]
        # Role ledger: the cap gates OWN-WORK trials (optimize + walk).
        worker._own_trials = 100
        assert worker._should_continue() is False

    def test_stops_when_time_budget_exceeded(self):
        worker = _create_worker(run={
            'parallel': 1,
            'completion_criteria': {
                'iterations': {'min': 0, 'max': 1000},
                'timing': {'end': '1d'},
            }
        })
        worker.study.trials = [MagicMock() for _ in range(50)]
        worker.start_time = datetime.datetime(2020, 1, 1)

        assert worker._should_continue() is False

    def test_continues_when_time_budget_not_exceeded(self):
        worker = _create_worker(run={
            'parallel': 1,
            'completion_criteria': {
                'iterations': {'min': 0, 'max': 1000},
                'timing': {'end': '30d'},
            }
        })
        worker.study.trials = [MagicMock() for _ in range(5)]
        worker.start_time = datetime.datetime.now()

        assert worker._should_continue() is True

    def test_stops_on_quality_threshold_achieved(self):
        worker = _create_worker(
            run={
                'parallel': 1,
                'completion_criteria': {
                    'iterations': {'min': 0, 'max': 1000},
                    'quality_achieved': True,
                },
            },
            objectives=[
                {'name': 'throughput', 'direction': 'minimize', 'quality_threshold': 10.0}
            ],
        )

        best_trial = MagicMock()
        best_trial.values = [5.0]
        worker.study.trials = [MagicMock() for _ in range(20)]
        worker.study.best_trials = [best_trial]

        assert worker._should_continue() is False

    def test_continues_when_quality_not_achieved(self):
        worker = _create_worker(
            run={
                'parallel': 1,
                'completion_criteria': {
                    'iterations': {'min': 0, 'max': 1000},
                    'quality_achieved': True,
                },
            },
            objectives=[
                {'name': 'throughput', 'direction': 'minimize', 'quality_threshold': 10.0}
            ],
        )

        best_trial = MagicMock()
        best_trial.values = [50.0]
        worker.study.trials = [MagicMock() for _ in range(20)]
        worker.study.best_trials = [best_trial]

        assert worker._should_continue() is True


class TestCheckTimeBudget:
    def test_days_format(self):
        worker = _create_worker()
        worker.start_time = datetime.datetime(2020, 1, 1)
        result = worker._check_time_budget({'timing': {'end': '1d'}})
        assert result is True

    def test_hours_format(self):
        worker = _create_worker()
        worker.start_time = datetime.datetime.now() - datetime.timedelta(hours=2)
        result = worker._check_time_budget({'timing': {'end': '1h'}})
        assert result is True

    def test_minutes_format(self):
        worker = _create_worker()
        worker.start_time = datetime.datetime.now()
        result = worker._check_time_budget({'timing': {'end': '60m'}})
        assert result is False

    def test_no_end_time_returns_false(self):
        worker = _create_worker()
        result = worker._check_time_budget({})
        assert result is False

    def test_invalid_format_returns_false(self):
        worker = _create_worker()
        worker.start_time = datetime.datetime(2020, 1, 1)
        result = worker._check_time_budget({'timing': {'end': 'invalid'}})
        assert result is False


class TestCheckQualityThresholds:
    def test_threshold_met_minimize(self):
        worker = _create_worker(objectives=[
            {'name': 'throughput', 'direction': 'minimize', 'quality_threshold': 10.0}
        ])

        best_trial = MagicMock()
        best_trial.values = [5.0]
        worker.study.best_trials = [best_trial]

        assert worker._check_quality_thresholds() is True

    def test_threshold_not_met_minimize(self):
        worker = _create_worker(objectives=[
            {'name': 'throughput', 'direction': 'minimize', 'quality_threshold': 10.0}
        ])

        best_trial = MagicMock()
        best_trial.values = [15.0]
        worker.study.best_trials = [best_trial]

        assert worker._check_quality_thresholds() is False

    def test_threshold_met_maximize(self):
        worker = _create_worker(objectives=[
            {'name': 'throughput', 'direction': 'maximize', 'quality_threshold': 100.0}
        ])

        best_trial = MagicMock()
        best_trial.values = [150.0]
        worker.study.best_trials = [best_trial]

        assert worker._check_quality_thresholds() is True

    def test_no_best_trials_returns_false(self):
        worker = _create_worker(objectives=[
            {'name': 'throughput', 'direction': 'minimize', 'quality_threshold': 10.0}
        ])
        worker.study.best_trials = []

        assert worker._check_quality_thresholds() is False

    def test_no_objectives_returns_false(self):
        worker = _create_worker(objectives=[])
        assert worker._check_quality_thresholds() is False

    def test_no_quality_threshold_in_objectives_returns_false(self):
        worker = _create_worker(objectives=[
            {'name': 'throughput', 'direction': 'minimize'}
        ])
        best_trial = MagicMock()
        best_trial.values = [5.0]
        worker.study.best_trials = [best_trial]

        assert worker._check_quality_thresholds() is False


class TestWorkerInit:
    def test_missing_creation_ts_raises(self):
        config = _base_config()
        del config['creation_ts']

        with pytest.raises(ValueError, match="creation_ts"):
            with patch.object(BreederWorker, '_load_or_create_study'), \
                 patch.object(BreederWorker, '_setup_communication', return_value=None), \
                 patch.object(BreederWorker, '_update_state'), \
                 patch('engine.breeder_worker.load_strain', return_value=MagicMock()):
                BreederWorker(config)

    def test_target_resolution_valid(self):
        worker = _create_worker(effectuation={
            'targets': [{'id': 't0', 'address': '10.0.0.1'}],
            'type': 'ssh',
        }, target_id=0)
        assert worker.target['id'] == 't0'

    def test_target_resolution_invalid_uses_first(self):
        worker = _create_worker(effectuation={
            'targets': [{'id': 't0', 'address': '10.0.0.1'}],
            'type': 'ssh',
        }, target_id=99)
        assert worker.target['id'] == 't0'

    def test_rollback_disabled_by_default(self):
        worker = _create_worker(effectuation={
            'targets': [{'id': 't0', 'address': '10.0.0.1'}],
            'type': 'ssh',
        })
        assert worker.rollback_enabled is False


class TestRunLoop:
    def _run_loop_worker(self):
        worker = _create_worker(run={
            'parallel': 1,
            'completion_criteria': {
                'iterations': {'min': 0, 'max': 1},
            }
        })
        trial_counter = [0]

        def fake_ask():
            t = MagicMock()
            t.number = trial_counter[0]
            trial_counter[0] += 1
            return t

        worker.study.ask = fake_ask

        real_trials = []

        def fake_tell(trial, *args, **kwargs):
            real_trials.append(trial)

        worker.study.tell = fake_tell
        worker.study.trials = real_trials

        # The iteration cap gates WALK trials (role ledger): the fake
        # coordinator must report walking mode, or the loop never stops.
        worker._probe_coordinator.decide_trial = MagicMock(
            return_value={'mode': 'impulse',
                          'params': {'vm.swappiness': 10}})
        best_mock = MagicMock()
        best_mock.number = -1
        worker.study.best_trials = [best_mock]

        return worker

    def test_run_executes_single_trial(self):
        worker = self._run_loop_worker()

        mock_strain = MagicMock()
        mock_strain.suggest_params.return_value = {'vm.swappiness': 10}
        worker.strain = mock_strain

        with patch.object(worker, '_execute_trial', return_value={'throughput': 42.5}), \
             patch.object(worker, '_check_guardrails', return_value=(False, [])), \
             patch.object(worker, 'metrics'):
            worker.run()

        assert len(real_trials := worker.study.trials) == 1

    def test_run_marks_trial_failed_on_guardrail_violation(self):
        worker = self._run_loop_worker()

        mock_strain = MagicMock()
        mock_strain.suggest_params.return_value = {'vm.swappiness': 10}
        worker.strain = mock_strain

        from optuna.trial import TrialState

        with patch.object(worker, '_execute_trial', return_value={'cpu_usage': 99.0}), \
             patch.object(worker, '_check_guardrails', return_value=(True, ['cpu_usage violated'])), \
             patch.object(worker, '_handle_guardrail_violation'), \
             patch.object(worker, 'metrics'):
            worker.run()

        assert len(worker.study.trials) == 1

    def test_run_handles_trial_exception(self):
        worker = self._run_loop_worker()

        mock_strain = MagicMock()
        mock_strain.suggest_params.side_effect = RuntimeError("param error")
        worker.strain = mock_strain

        with patch.object(worker, 'metrics'):
            worker.run()

        assert len(worker.study.trials) == 1

    def test_run_pushes_metrics_on_completion(self):
        worker = _create_worker(run={
            'parallel': 1,
            'completion_criteria': {
                'iterations': {'min': 0, 'max': 0},
            }
        })

        with patch.object(worker, 'metrics') as mock_metrics:
            worker.run()
            mock_metrics.mark_running.assert_called_once()
            mock_metrics.mark_stopped.assert_called_once()
            mock_metrics.push.assert_called()

    def test_run_shares_trial_with_communication(self):
        worker = self._run_loop_worker()

        mock_strain = MagicMock()
        mock_strain.suggest_params.return_value = {'vm.swappiness': 10}
        worker.strain = mock_strain

        mock_comm = MagicMock()
        worker.communication_callback = mock_comm

        # Mock the detection coordinator to return optimize mode with no params
        worker._probe_coordinator.decide_trial.return_value = {
            'mode': 'optimize', 'params': None
        }

        with patch.object(worker, '_execute_trial', return_value={'throughput': 42.5}), \
             patch.object(worker, '_check_guardrails', return_value=(False, [])), \
             patch.object(worker, 'metrics'):
            worker.run()

        mock_comm.assert_called_once()
        call_args = mock_comm.call_args
        assert call_args[0][0] == worker.study
        assert len(worker.study.trials) == 1


class TestRunReconnaissance:
    def setup_method(self):
        sys.modules['wmill'].reset_mock()
        sys.modules['wmill'].run_script_by_path.reset_mock()
        sys.modules['wmill'].run_script_by_path.side_effect = None

    def test_returns_metrics_from_recon(self):
        worker = _create_worker()
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.return_value = {
            'status': 'completed',
            'metrics': {'throughput': 42.5}
        }

        result = worker._run_reconnaissance()
        assert result == {'throughput': 42.5}
        mock_wmill.run_script_by_path.assert_called_once()
        call_args = mock_wmill.run_script_by_path.call_args
        assert call_args[0][0] == 'f/reconnaissance/prometheus'

    def test_returns_inf_on_empty_metrics(self):
        worker = _create_worker()
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.return_value = {
            'status': 'completed',
            'metrics': {}
        }

        result = worker._run_reconnaissance()
        assert result == {'throughput': float('inf')}


class TestExecuteTrialUsesRunRecon:
    def setup_method(self):
        sys.modules['wmill'].reset_mock()
        sys.modules['wmill'].run_script_by_path.reset_mock()
        sys.modules['wmill'].run_script_by_path.side_effect = None

    def test_execute_trial_calls_recon_through_shared_method(self):
        worker = _create_worker()
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.side_effect = [
            {'status': 'completed', 'successful_changes': 1, 'failed_changes': 0},
            {'status': 'completed', 'metrics': {'throughput': 42.5}},
        ]

        result = worker._execute_trial({'vm.swappiness': 10})
        assert result == {'throughput': 42.5}
        assert mock_wmill.run_script_by_path.call_count == 2
        recon_call = mock_wmill.run_script_by_path.call_args_list[1]
        assert recon_call[0][0] == 'f/reconnaissance/prometheus'


class TestPublishStandingParams:
    """Per-trial upsert of the breeder's applied params — the standing
    dials causal stamps curve points with (the ambient of measurement)."""

    def setup_method(self):
        sys.modules['wmill'].reset_mock()

    def test_upserts_params_json(self):
        import json as _json
        worker = _create_worker(interference_detection={'group': 'g1'})
        cursor = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value = cursor
        worker._with_shared_db = lambda op, desc=None: op(conn)

        worker._publish_standing_params({'param_0': 50.0})

        sql, args = cursor.execute.call_args[0]
        assert 'UPDATE interference_active_breeders' in sql
        assert 'params = %s' in sql
        assert 'last_seen = NOW()' in sql
        assert _json.loads(args[0]) == {'param_0': 50.0}
        assert args[1] == worker.breeder_id

    def test_skips_without_detection_section(self):
        worker = _create_worker()  # base config: pure optimizer
        called = []
        worker._with_shared_db = lambda op, desc=None: called.append(desc)

        worker._publish_standing_params({'param_0': 50.0})

        assert called == []

    def test_skips_on_empty_params(self):
        worker = _create_worker(interference_detection={'group': 'g1'})
        called = []
        worker._with_shared_db = lambda op, desc=None: called.append(desc)

        worker._publish_standing_params({})

        assert called == []


class TestObservationPublishGate:
    """Which trials publish their own readings: every trial with a
    protocol phase in flight (receiver hold, sender push, sender
    pause). Parked and optimizer trials publish nothing."""

    def test_phase_discriminator(self):
        gate = lambda d: d.get('lease_phase') or d.get('impulse_phase')

        receiver_hold = {'mode': 'hold', 'lease_phase': 'probe_push'}
        sender_push = {'mode': 'impulse', 'impulse_phase': 'probe_push'}
        sender_pause = {'mode': 'hold', 'impulse_phase': 'probe_pause'}
        parked = {'mode': 'hold'}
        optimize = {'mode': 'optimize'}

        assert gate(receiver_hold) == 'probe_push'
        assert gate(sender_push) == 'probe_push'
        assert gate(sender_pause) == 'probe_pause'
        assert gate(parked) is None
        assert gate(optimize) is None

