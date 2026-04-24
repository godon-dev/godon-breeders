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


class TestObserveOnly:
    def setup_method(self):
        sys.modules['wmill'].reset_mock()
        sys.modules['wmill'].run_script_by_path.reset_mock()
        sys.modules['wmill'].run_script_by_path.side_effect = None

    def test_calls_recon_without_effectuation(self):
        worker = _create_worker()
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.return_value = {
            'status': 'completed',
            'metrics': {'throughput': 30.0}
        }

        mock_trial = MagicMock()
        mock_trial.number = 99
        worker.study.ask.return_value = mock_trial

        result = worker._observe_only('choreo-123')

        assert result == {'throughput': 30.0}
        assert mock_wmill.run_script_by_path.call_count == 1
        call_path = mock_wmill.run_script_by_path.call_args[0][0]
        assert call_path == 'f/reconnaissance/prometheus'

    def test_records_trial_with_phase_marker(self):
        worker = _create_worker()
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.return_value = {
            'status': 'completed',
            'metrics': {'throughput': 30.0}
        }

        mock_trial = MagicMock()
        mock_trial.number = 42
        worker.study.ask.return_value = mock_trial

        worker._observe_only('choreo-123')

        mock_trial.set_user_attr.assert_any_call('phase', 'observe_only')
        mock_trial.set_user_attr.assert_any_call('choreography_id', 'choreo-123')
        worker.study.tell.assert_called_once()
        tell_args = worker.study.tell.call_args
        assert tell_args[0][0] == mock_trial
        assert tell_args[1]['values'] == [30.0]

    def test_works_without_choreography_id(self):
        worker = _create_worker()
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.return_value = {
            'status': 'completed',
            'metrics': {'throughput': 10.0}
        }

        mock_trial = MagicMock()
        worker.study.ask.return_value = mock_trial

        result = worker._observe_only()

        assert result == {'throughput': 10.0}
        mock_trial.set_user_attr.assert_any_call('choreography_id', '')


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


class TestGetCurrentPhaseMode:
    def test_returns_active_when_no_choreography(self):
        worker = _create_worker(interference_detection={'mode': 'active'})
        with patch.object(worker, '_get_active_choreography', return_value=None):
            mode, cid = worker._get_current_phase_mode()
        assert mode == 'active'
        assert cid is None

    def test_returns_observe_only_when_breeder_is_targeted(self):
        worker = _create_worker(interference_detection={'mode': 'active'})
        claim = {
            'choreography_id': 'choreo-1',
            'current_phase': 1,
            'phases': [
                {'observe_breeder': None},
                {'observe_breeder': worker.breeder_id},
                {'observe_breeder': None},
            ]
        }
        with patch.object(worker, '_get_active_choreography', return_value=claim):
            mode, cid = worker._get_current_phase_mode()
        assert mode == 'observe_only'
        assert cid == 'choreo-1'

    def test_returns_active_when_other_breeder_is_targeted(self):
        worker = _create_worker(interference_detection={'mode': 'active'})
        claim = {
            'choreography_id': 'choreo-1',
            'current_phase': 1,
            'phases': [
                {'observe_breeder': None},
                {'observe_breeder': 'other-breeder-id'},
                {'observe_breeder': None},
            ]
        }
        with patch.object(worker, '_get_active_choreography', return_value=claim):
            mode, cid = worker._get_current_phase_mode()
        assert mode == 'active'
        assert cid == 'choreo-1'

    def test_returns_active_when_phase_index_out_of_range(self):
        worker = _create_worker(interference_detection={'mode': 'active'})
        claim = {
            'choreography_id': 'choreo-1',
            'current_phase': 99,
            'phases': [{'observe_breeder': None}]
        }
        with patch.object(worker, '_get_active_choreography', return_value=claim):
            mode, cid = worker._get_current_phase_mode()
        assert mode == 'active'


class TestRunLoopObserveOnly:
    def _observe_worker(self, max_trials=2):
        worker = _create_worker(
            interference_detection={'mode': 'active'},
            run={
                'parallel': 1,
                'completion_criteria': {
                    'iterations': {'min': 0, 'max': max_trials},
                }
            }
        )
        return worker

    def test_observe_only_skips_ask_tell(self):
        call_count = [0]

        def should_continue_side_effect():
            call_count[0] += 1
            return call_count[0] <= 1

        worker = self._observe_worker(max_trials=1)

        with patch.object(worker, '_should_continue', side_effect=should_continue_side_effect), \
             patch.object(worker, '_get_current_phase_mode',
                          return_value=('observe_only', 'choreo-1')), \
             patch.object(worker, '_observe_only',
                          return_value={'throughput': 30.0}) as mock_observe, \
             patch.object(worker, '_update_state'), \
             patch.object(worker, 'metrics'):
            worker.run()

        mock_observe.assert_called_once_with('choreo-1')

    def test_alternates_between_active_and_observe(self):
        modes = ['active', 'observe_only', 'observe_only', 'active']
        mode_idx = [0]

        def next_mode(*args, **kwargs):
            idx = min(mode_idx[0], len(modes) - 1)
            m = modes[idx]
            mode_idx[0] += 1
            return m, 'choreo-1' if m == 'observe_only' else None

        call_count = [0]

        def should_continue_side_effect():
            call_count[0] += 1
            return call_count[0] <= 4

        worker = self._observe_worker(max_trials=4)

        mock_strain = MagicMock()
        mock_strain.suggest_params.return_value = {'vm.swappiness': 10}
        worker.strain = mock_strain

        trial_counter = [0]

        def fake_ask():
            t = MagicMock()
            t.number = trial_counter[0]
            trial_counter[0] += 1
            return t

        worker.study.ask = fake_ask
        worker.study.trials = []
        worker.study.best_trials = [MagicMock(number=-1)]

        with patch.object(worker, '_should_continue', side_effect=should_continue_side_effect), \
             patch.object(worker, '_get_current_phase_mode', side_effect=next_mode), \
             patch.object(worker, '_execute_trial', return_value={'throughput': 42.5}), \
             patch.object(worker, '_observe_only', return_value={'throughput': 30.0}), \
             patch.object(worker, '_check_guardrails', return_value=(False, [])), \
             patch.object(worker, '_update_state'), \
             patch.object(worker, 'metrics'):
            worker.run()

        assert worker._observe_only.call_count == 2
        assert worker._execute_trial.call_count == 2
