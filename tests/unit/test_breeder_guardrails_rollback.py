import pytest
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock
import json

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
        'objectives': [{'name': 'test_obj', 'direction': 'minimize'}],
        'effectuation': {'targets': []},
    }
    config.update(overrides)
    return config


def _create_worker(**config_overrides):
    config = _base_config(**config_overrides)
    with patch.object(BreederWorker, '_load_or_create_study') as mock_study, \
         patch.object(BreederWorker, '_setup_communication', return_value=None), \
         patch.object(BreederWorker, '_update_state'), \
         patch('engine.breeder_worker.load_strain', return_value=MagicMock()):
        mock_study.return_value = MagicMock()
        return BreederWorker(config)


def _mock_study_with_attrs():
    study = MagicMock()
    study.user_attrs = {}

    def set_attr(key, value):
        study.user_attrs[key] = value

    study.set_user_attr = set_attr
    study.study_name = 'test_study'
    return study


class TestCheckGuardrails:
    def test_no_guardrails_config_returns_false(self):
        worker = _create_worker()
        violated, violations = worker._check_guardrails({'anything': 42.0})
        assert violated is False
        assert violations == []

    def test_empty_guardrails_list_returns_false(self):
        worker = _create_worker(guardrails=[])
        violated, violations = worker._check_guardrails({'cpu_usage': 95.0})
        assert violated is False

    def test_no_violation_when_within_limits(self):
        worker = _create_worker(guardrails=[
            {'name': 'cpu_usage', 'hard_limit': 90.0},
        ])
        violated, violations = worker._check_guardrails({'cpu_usage': 75.0})
        assert violated is False
        assert violations == []

    def test_single_violation(self):
        worker = _create_worker(guardrails=[
            {'name': 'cpu_usage', 'hard_limit': 90.0},
        ])
        violated, violations = worker._check_guardrails({'cpu_usage': 95.0})
        assert violated is True
        assert len(violations) == 1
        assert 'cpu_usage' in violations[0]
        assert '95.0' in violations[0]

    def test_multiple_violations(self):
        worker = _create_worker(guardrails=[
            {'name': 'cpu_usage', 'hard_limit': 90.0},
            {'name': 'memory_usage', 'hard_limit': 85.0},
        ])
        violated, violations = worker._check_guardrails({
            'cpu_usage': 95.0,
            'memory_usage': 90.0,
        })
        assert violated is True
        assert len(violations) == 2

    def test_inf_metric_treated_as_violation(self):
        worker = _create_worker(guardrails=[
            {'name': 'cpu_usage', 'hard_limit': 90.0},
        ])
        violated, violations = worker._check_guardrails({'cpu_usage': float('inf')})
        assert violated is True

    def test_missing_hard_limit_skipped(self):
        worker = _create_worker(guardrails=[
            {'name': 'cpu_usage'},
        ])
        violated, violations = worker._check_guardrails({'cpu_usage': 95.0})
        assert violated is False

    def test_metric_not_in_results_skipped(self):
        worker = _create_worker(guardrails=[
            {'name': 'cpu_usage', 'hard_limit': 90.0},
        ])
        violated, violations = worker._check_guardrails({'other_metric': 50.0})
        assert violated is False

    def test_non_numeric_hard_limit_skipped(self):
        worker = _create_worker(guardrails=[
            {'name': 'cpu_usage', 'hard_limit': 'high'},
        ])
        violated, violations = worker._check_guardrails({'cpu_usage': 95.0})
        assert violated is False


class TestRollbackStateManagement:
    def _create_worker_with_rollback(self, strategy='standard', consecutive_failures=3):
        config = _base_config(
            effectuation={
                'targets': [
                    {
                        'id': 't1',
                        'address': '10.0.0.1',
                        'rollback': {
                            'enabled': True,
                            'strategy': strategy,
                        },
                    }
                ],
            },
            rollback_strategies={
                'standard': {
                    'consecutive_failures': consecutive_failures,
                    'target_state': 'previous',
                },
            },
        )
        study = _mock_study_with_attrs()

        with patch('engine.breeder_worker.load_strain', return_value=MagicMock()):
            worker = BreederWorker.__new__(BreederWorker)
            worker.config = config
            worker.study = study
            worker.breeder_id = 'test-breeder-id'
            worker.breeder_uuid = 'test-uuid'
            worker.worker_id = 'test_worker_test-uuid'
            worker.target_id = 0
            worker.target = config['effectuation']['targets'][0]
            worker.rollback_config = config['effectuation']['targets'][0]['rollback']
            worker.rollback_enabled = True
            worker.breeder_type = 'linux_performance'
            worker.metrics = MagicMock()

        return worker

    def test_init_rollback_state_creates_initial_state(self):
        worker = self._create_worker_with_rollback()
        worker._init_rollback_state()

        state = json.loads(worker.study.user_attrs['rollback_state_target_0'])
        assert state['state'] == 'normal'
        assert state['consecutive_failures'] == 0
        assert state['last_successful_params'] is None
        assert state['version'] == 0

    def test_init_rollback_state_idempotent(self):
        worker = self._create_worker_with_rollback()
        worker._init_rollback_state()
        worker._init_rollback_state()

        state = json.loads(worker.study.user_attrs['rollback_state_target_0'])
        assert state['version'] == 0

    def test_get_rollback_state_returns_initialized(self):
        worker = self._create_worker_with_rollback()
        worker._init_rollback_state()
        state = worker._get_rollback_state()
        assert state['state'] == 'normal'

    def test_get_rollback_state_auto_inits(self):
        worker = self._create_worker_with_rollback()
        state = worker._get_rollback_state()
        assert state['state'] == 'normal'
        assert 'rollback_state_target_0' in worker.study.user_attrs

    def test_update_rollback_state_increments_version(self):
        worker = self._create_worker_with_rollback()
        worker._init_rollback_state()

        new_state = worker._get_rollback_state()
        new_state['consecutive_failures'] = 1
        worker._update_rollback_state(new_state)

        stored = json.loads(worker.study.user_attrs['rollback_state_target_0'])
        assert stored['version'] == 1
        assert stored['consecutive_failures'] == 1

    def test_check_needs_rollback_below_threshold(self):
        worker = self._create_worker_with_rollback(consecutive_failures=3)
        worker._init_rollback_state()

        state = worker._get_rollback_state()
        state['consecutive_failures'] = 2
        worker._update_rollback_state(state)

        assert worker._check_needs_rollback() is False

    def test_check_needs_rollback_at_threshold(self):
        worker = self._create_worker_with_rollback(consecutive_failures=3)
        worker._init_rollback_state()

        state = worker._get_rollback_state()
        state['consecutive_failures'] = 3
        worker._update_rollback_state(state)

        assert worker._check_needs_rollback() is True

    def test_check_needs_rollback_disabled(self):
        worker = self._create_worker_with_rollback()
        worker.rollback_enabled = False
        assert worker._check_needs_rollback() is False

    def test_handle_guardrail_violation_increments_failures(self):
        worker = self._create_worker_with_rollback(consecutive_failures=3)
        worker._init_rollback_state()

        worker._handle_guardrail_violation({'param': 'value'})
        state = worker._get_rollback_state()
        assert state['consecutive_failures'] == 1
        assert state['state'] == 'normal'

    def test_handle_guardrail_violation_triggers_rollback_state(self):
        worker = self._create_worker_with_rollback(consecutive_failures=2)
        worker._init_rollback_state()

        state = worker._get_rollback_state()
        state['consecutive_failures'] = 2
        worker._update_rollback_state(state)

        worker._handle_guardrail_violation({'param': 'value'})
        state = worker._get_rollback_state()
        assert state['consecutive_failures'] == 3
        assert state['state'] == 'needs_rollback'

    def test_handle_guardrail_violation_noop_when_disabled(self):
        worker = self._create_worker_with_rollback()
        worker.rollback_enabled = False
        worker._handle_guardrail_violation({'param': 'value'})
        assert 'rollback_state_target_0' not in worker.study.user_attrs

    def test_handle_successful_trial_resets_failures(self):
        worker = self._create_worker_with_rollback()
        worker._init_rollback_state()

        state = worker._get_rollback_state()
        state['consecutive_failures'] = 5
        state['state'] = 'needs_rollback'
        worker._update_rollback_state(state)

        worker._handle_successful_trial({'param': 'new_value'})
        state = worker._get_rollback_state()
        assert state['consecutive_failures'] == 0
        assert state['state'] == 'normal'
        assert state['last_successful_params'] == {'param': 'new_value'}

    def test_handle_successful_trial_noop_when_disabled(self):
        worker = self._create_worker_with_rollback()
        worker.rollback_enabled = False
        worker._handle_successful_trial({'param': 'value'})
        assert 'rollback_state_target_0' not in worker.study.user_attrs


class TestExecuteRollback:
    def setup_method(self):
        sys.modules['wmill'].reset_mock()

    def _create_worker_for_rollback(self, target_state='previous', on_failure='stop',
                                    last_successful_params=None, best_trials=None):
        config = _base_config(
            effectuation={
                'targets': [
                    {
                        'id': 't1',
                        'address': '10.0.0.1',
                        'rollback': {
                            'enabled': True,
                            'strategy': 'standard',
                        },
                    }
                ],
                'type': 'ssh',
            },
            rollback_strategies={
                'standard': {
                    'consecutive_failures': 3,
                    'target_state': target_state,
                    'on_failure': on_failure,
                },
            },
        )
        study = _mock_study_with_attrs()
        study.best_trials = best_trials or []

        with patch('engine.breeder_worker.load_strain', return_value=MagicMock()):
            worker = BreederWorker.__new__(BreederWorker)
            worker.config = config
            worker.study = study
            worker.breeder_id = 'test-breeder-id'
            worker.breeder_uuid = 'test-uuid'
            worker.worker_id = 'test_worker_test-uuid'
            worker.target_id = 0
            worker.target = config['effectuation']['targets'][0]
            worker.rollback_config = config['effectuation']['targets'][0]['rollback']
            worker.rollback_enabled = True
            worker.breeder_type = 'linux_performance'
            worker.metrics = MagicMock()

        worker._init_rollback_state()
        if last_successful_params:
            state = worker._get_rollback_state()
            state['last_successful_params'] = last_successful_params
            worker._update_rollback_state(state)

        return worker

    def test_rollback_to_previous_success(self):
        worker = self._create_worker_for_rollback(
            last_successful_params={'net.core.somaxconn': 4096}
        )
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.return_value = {'status': 'completed'}
        result = worker._execute_rollback()
        assert result is True

        state = worker._get_rollback_state()
        assert state['state'] == 'completed'
        assert state['consecutive_failures'] == 0

    def test_rollback_to_best_success(self):
        best_trial = MagicMock()
        best_trial.params = {'vm.swappiness': 10}
        worker = self._create_worker_for_rollback(
            target_state='best', best_trials=[best_trial]
        )
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.return_value = {'status': 'completed'}
        result = worker._execute_rollback()
        assert result is True

    def test_rollback_to_best_no_best_trial_fails(self):
        worker = self._create_worker_for_rollback(target_state='best', best_trials=[])
        result = worker._execute_rollback()
        assert result is False

    def test_rollback_to_baseline_empty_params(self):
        worker = self._create_worker_for_rollback(target_state='baseline')
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.return_value = {'status': 'completed'}
        result = worker._execute_rollback()
        assert result is True

    def test_rollback_to_previous_no_params_fails(self):
        worker = self._create_worker_for_rollback(last_successful_params=None)
        result = worker._execute_rollback()
        assert result is False

    def test_rollback_unknown_target_state_fails(self):
        worker = self._create_worker_for_rollback(target_state='invalid_state')
        result = worker._execute_rollback()
        assert result is False

    def test_rollback_effectuation_error_on_failure_stop(self):
        worker = self._create_worker_for_rollback(
            on_failure='stop', last_successful_params={'p': 1}
        )
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.side_effect = Exception('SSH failed')
        with pytest.raises(Exception, match='SSH failed'):
            worker._execute_rollback()

        state = worker._get_rollback_state()
        assert state['state'] == 'failed'

    def test_rollback_effectuation_error_on_failure_continue(self):
        worker = self._create_worker_for_rollback(
            on_failure='continue', last_successful_params={'p': 1}
        )
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.side_effect = Exception('SSH failed')
        result = worker._execute_rollback()
        assert result is False

        state = worker._get_rollback_state()
        assert state['state'] == 'failed'

    def test_rollback_effectuation_error_on_failure_skip_target(self):
        worker = self._create_worker_for_rollback(
            on_failure='skip_target', last_successful_params={'p': 1}
        )
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.side_effect = Exception('SSH failed')
        result = worker._execute_rollback()
        assert result is False

        state = worker._get_rollback_state()
        assert state['state'] == 'skip_target'

    def test_rollback_metrics_pushed_on_success(self):
        worker = self._create_worker_for_rollback(last_successful_params={'p': 1})
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.return_value = {'status': 'completed'}
        worker._execute_rollback()
        worker.metrics.inc_rollback.assert_called_with('success')
        worker.metrics.push.assert_called()

    def test_rollback_metrics_pushed_on_failure(self):
        worker = self._create_worker_for_rollback(
            on_failure='continue', last_successful_params={'p': 1}
        )
        mock_wmill = sys.modules['wmill']
        mock_wmill.run_script_by_path.side_effect = Exception('fail')
        worker._execute_rollback()
        worker.metrics.inc_rollback.assert_called_with('failed')
        worker.metrics.push.assert_called()
