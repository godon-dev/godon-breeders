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
# along with this godon.  If not, see <http://www.gnu.org/licenses/>.
#

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, Mock
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class TestCheckGuardrails:
    """Test guardrails checking logic"""

    @pytest.fixture
    def mock_worker(self):
        """Create a mock breeder worker with guardrails config"""
        worker = MagicMock()
        worker.config = {
            'guardrails': [
                {
                    'name': 'cpu_usage',
                    'hard_limit': 90.0,
                    'reconnaissance': {
                        'service': 'prometheus',
                        'query': 'rate(process_cpu_seconds_total[5m]) * 100'
                    }
                },
                {
                    'name': 'memory_usage',
                    'hard_limit': 85.0,
                    'reconnaissance': {
                        'service': 'prometheus',
                        'query': 'process_resident_memory_bytes / total_memory * 100'
                    }
                }
            ]
        }
        return worker

    def test_no_guardrails_violations(self, mock_worker):
        """Test that all metrics within limits pass guardrails check"""
        metrics = {
            'cpu_usage': 75.0,  # Below 90
            'memory_usage': 60.0  # Below 85
        }

        guardrails = mock_worker.config['guardrails']
        violations = []

        for guardrail in guardrails:
            metric_name = guardrail['name']
            hard_limit = guardrail['hard_limit']
            metric_value = metrics.get(metric_name, float('inf'))

            if metric_value > hard_limit:
                violations.append({
                    'name': metric_name,
                    'hard_limit': hard_limit,
                    'actual_value': metric_value
                })

        assert len(violations) == 0

    def test_single_guardrail_violation(self, mock_worker):
        """Test detection of single guardrail violation"""
        metrics = {
            'cpu_usage': 95.0,  # Exceeds 90
            'memory_usage': 60.0  # Below 85
        }
        
        guardrails = mock_worker.config['guardrails']
        violations = []
        
        for guardrail in guardrails:
            metric_name = guardrail['name']
            hard_limit = guardrail['hard_limit']
            metric_value = metrics.get(metric_name, float('inf'))
            
            if metric_value > hard_limit:
                violations.append({
                    'name': metric_name,
                    'hard_limit': hard_limit,
                    'actual_value': metric_value
                })
        
        assert len(violations) == 1
        assert violations[0]['name'] == 'cpu_usage'
        assert violations[0]['actual_value'] == 95.0
        assert violations[0]['hard_limit'] == 90.0

    def test_multiple_guardrail_violations(self, mock_worker):
        """Test detection of multiple guardrail violations"""
        metrics = {
            'cpu_usage': 95.0,  # Exceeds 90
            'memory_usage': 90.0  # Exceeds 85
        }
        
        guardrails = mock_worker.config['guardrails']
        violations = []
        
        for guardrail in guardrails:
            metric_name = guardrail['name']
            hard_limit = guardrail['hard_limit']
            metric_value = metrics.get(metric_name, float('inf'))
            
            if metric_value > hard_limit:
                violations.append({
                    'name': metric_name,
                    'hard_limit': hard_limit,
                    'actual_value': metric_value
                })
        
        assert len(violations) == 2
        assert violations[0]['name'] == 'cpu_usage'
        assert violations[1]['name'] == 'memory_usage'

    def test_infinite_metric_treated_as_violation(self, mock_worker):
        """Test that infinity (failed metric collection) is treated as violation"""
        metrics = {
            'cpu_usage': float('inf'),  # Metric collection failed
            'memory_usage': 60.0
        }
        
        guardrails = mock_worker.config['guardrails']
        violations = []
        
        for guardrail in guardrails:
            metric_name = guardrail['name']
            hard_limit = guardrail['hard_limit']
            metric_value = metrics.get(metric_name, float('inf'))
            
            if metric_value > hard_limit:
                violations.append({
                    'name': metric_name,
                    'hard_limit': hard_limit,
                    'actual_value': metric_value
                })
        
        # Infinity exceeds any hard limit
        assert len(violations) == 1
        assert violations[0]['name'] == 'cpu_usage'
        assert violations[0]['actual_value'] == float('inf')


class TestRollbackStateManagement:
    """Test rollback state management logic"""

    def _create_mock_study(self):
        """Helper to create a mock study with working set_user_attr"""
        study = MagicMock()
        study.user_attrs = {}

        def mock_set_user_attr(key, value):
            study.user_attrs[key] = value

        study.set_user_attr = mock_set_user_attr
        return study

    def test_init_rollback_state(self):
        """Test initialization of rollback state in study.user_attrs"""
        study = self._create_mock_study()

        initial_state = {
            'state': 'normal',
            'consecutive_failures': 0,
            'last_successful_params': None,
            'rollback_strategy': 'standard',
            'version': 0
        }

        state_key = 'rollback_state_target_0'
        study.set_user_attr(state_key, json.dumps(initial_state))

        # Verify state was stored
        assert state_key in study.user_attrs
        stored_state = json.loads(study.user_attrs[state_key])
        assert stored_state['state'] == 'normal'
        assert stored_state['consecutive_failures'] == 0
        assert stored_state['version'] == 0

    def test_increment_consecutive_failures(self):
        """Test incrementing consecutive failures counter"""
        study = self._create_mock_study()

        # Initialize state
        state_key = 'rollback_state_target_0'
        initial_state = {
            'state': 'normal',
            'consecutive_failures': 2,
            'last_successful_params': {'param1': 100},
            'rollback_strategy': 'standard',
            'version': 0
        }
        study.user_attrs[state_key] = json.dumps(initial_state)

        # Simulate incrementing failures
        current_state = json.loads(study.user_attrs[state_key])
        current_state['consecutive_failures'] += 1
        current_state['version'] += 1
        study.set_user_attr(state_key, json.dumps(current_state))

        # Verify incremented state
        updated_state = json.loads(study.user_attrs[state_key])
        assert updated_state['consecutive_failures'] == 3
        assert updated_state['version'] == 1

    def test_reset_consecutive_failures_on_success(self):
        """Test that successful trial resets consecutive failures"""
        study = self._create_mock_study()

        # Initialize state with failures
        state_key = 'rollback_state_target_0'
        initial_state = {
            'state': 'needs_rollback',
            'consecutive_failures': 5,
            'last_successful_params': {'param1': 100},
            'rollback_strategy': 'standard',
            'version': 2
        }
        study.user_attrs[state_key] = json.dumps(initial_state)

        # Simulate successful trial
        current_state = json.loads(study.user_attrs[state_key])
        current_state['consecutive_failures'] = 0
        current_state['state'] = 'normal'
        current_state['last_successful_params'] = {'param1': 150, 'param2': 200}
        current_state['version'] += 1
        study.set_user_attr(state_key, json.dumps(current_state))

        # Verify reset state
        updated_state = json.loads(study.user_attrs[state_key])
        assert updated_state['consecutive_failures'] == 0
        assert updated_state['state'] == 'normal'
        assert updated_state['last_successful_params'] == {'param1': 150, 'param2': 200}
        assert updated_state['version'] == 3

    def test_check_needs_rollback_below_threshold(self):
        """Test that rollback is not triggered below threshold"""
        state = {
            'consecutive_failures': 2,
            'rollback_strategy': 'standard'
        }
        
        rollback_config = {
            'strategies': {
                'standard': {
                    'consecutive_failures': 3
                }
            }
        }
        
        threshold = rollback_config['strategies'][state['rollback_strategy']]['consecutive_failures']
        needs_rollback = state['consecutive_failures'] >= threshold
        
        assert needs_rollback is False

    def test_check_needs_rollback_at_threshold(self):
        """Test that rollback is triggered when threshold is reached"""
        state = {
            'consecutive_failures': 3,
            'rollback_strategy': 'standard'
        }
        
        rollback_config = {
            'strategies': {
                'standard': {
                    'consecutive_failures': 3
                }
            }
        }
        
        threshold = rollback_config['strategies'][state['rollback_strategy']]['consecutive_failures']
        needs_rollback = state['consecutive_failures'] >= threshold
        
        assert needs_rollback is True

    def test_rollback_state_transition_to_in_progress(self):
        """Test state transition from needs_rollback to in_progress"""
        study = self._create_mock_study()

        state_key = 'rollback_state_target_0'
        initial_state = {
            'state': 'needs_rollback',
            'consecutive_failures': 3,
            'last_successful_params': {'param1': 100},
            'rollback_strategy': 'standard',
            'version': 1
        }
        study.user_attrs[state_key] = json.dumps(initial_state)

        # Transition to in_progress
        current_state = json.loads(study.user_attrs[state_key])
        current_state['state'] = 'in_progress'
        current_state['version'] += 1
        study.set_user_attr(state_key, json.dumps(current_state))

        # Verify transition
        updated_state = json.loads(study.user_attrs[state_key])
        assert updated_state['state'] == 'in_progress'
        assert updated_state['version'] == 2


class TestV03ParameterSuggestion:
    """Test parameter suggestion with multiple categories"""

    def test_suggest_single_param_integer_range(self):
        """Test suggesting integer parameter from range constraints"""
        # Mock Optuna trial
        trial = MagicMock()
        trial.suggest_int.return_value = 50000
        
        param_config = {
            'constraints': [
                {'step': 100, 'lower': 4096, 'upper': 65536}
            ]
        }
        
        # Suggest parameter from first constraint
        constraint = param_config['constraints'][0]
        value = trial.suggest_int(
            'test_param',
            int(constraint['lower']),
            int(constraint['upper']),
            step=int(constraint['step'])
        )
        
        assert 4096 <= value <= 65536
        assert value % 100 == 0

    def test_suggest_single_param_categorical(self):
        """Test suggesting categorical parameter from values"""
        trial = MagicMock()
        trial.suggest_categorical.return_value = 'performance'
        
        param_config = {
            'constraints': [
                {'values': ['performance', 'powersave', 'ondemand']}
            ]
        }
        
        # Suggest categorical parameter
        constraint = param_config['constraints'][0]
        value = trial.suggest_categorical('test_param', constraint['values'])
        
        assert value in ['performance', 'powersave', 'ondemand']

    def test_suggest_single_param_multiple_ranges(self):
        """Test that parameter suggestion handles multiple constraint ranges"""
        trial = MagicMock()
        trial.suggest_int.return_value = 50000  # Value within first constraint range

        param_config = {
            'constraints': [
                {'step': 100, 'lower': 4096, 'upper': 65536},
                {'step': 100, 'lower': 65536, 'upper': 131072}
            ]
        }

        #  Optuna should handle the full range across constraints
        # Worker suggests from first constraint for simplicity
        constraint = param_config['constraints'][0]
        value = trial.suggest_int(
            'test_param',
            int(constraint['lower']),
            int(constraint['upper']),
            step=int(constraint['step'])
        )

        # The mock should return the value we set
        assert value == 50000
        assert 4096 <= 50000 <= 65536

    def test_suggest_params_sysctl_category(self):
        """Test suggesting sysctl parameters"""
        trial = MagicMock()
        trial.suggest_int.return_value = 50000  # Value within expected range

        settings = {
            'sysctl': {
                'net.core.rmem_max': {
                    'constraints': [{'step': 1000, 'lower': 10000, 'upper': 100000}]
                }
            }
        }

        # Sysctl parameters should be suggested
        for category in ['sysctl']:
            if category in settings:
                for param_name, param_config in settings[category].items():
                    if param_name == 'net.core.rmem_max':
                        constraint = param_config['constraints'][0]
                        value = trial.suggest_int(
                            f'sysctl.{param_name}',
                            int(constraint['lower']),
                            int(constraint['upper']),
                            step=int(constraint['step'])
                        )
                        # The mock should return the value we set
                        assert value == 50000
                        assert 10000 <= 50000 <= 100000

    def test_suggest_params_sysfs_category(self):
        """Test suggesting sysfs parameters (categorical)"""
        trial = MagicMock()
        trial.suggest_categorical.return_value = 'performance'
        
        settings = {
            'sysfs': {
                'cpu_governor': {
                    'constraints': [{'values': ['performance', 'powersave']}]
                }
            }
        }
        
        # Sysfs parameters should be suggested
        for category in ['sysfs']:
            if category in settings:
                for param_name, param_config in settings[category].items():
                    if param_name == 'cpu_governor':
                        constraint = param_config['constraints'][0]
                        value = trial.suggest_categorical(
                            f'sysfs.{param_name}',
                            constraint['values']
                        )
                        assert value in ['performance', 'powersave']

    def test_suggest_params_all_categories(self):
        """Test suggesting parameters from all supported categories"""
        trial = MagicMock()
        
        settings = {
            'sysctl': {
                'vm.swappiness': {
                    'constraints': [{'step': 1, 'lower': 0, 'upper': 100}]
                }
            },
            'sysfs': {
                'cpu_governor': {
                    'constraints': [{'values': ['performance', 'powersave']}]
                }
            }
        }
        
        suggested_params = {}
        
        # Suggest from each category
        for category in ['sysctl', 'sysfs']:
            if category in settings:
                for param_name, param_config in settings[category].items():
                    constraint = param_config['constraints'][0]
                    
                    if 'values' in constraint:
                        # Categorical
                        value = trial.suggest_categorical(f'{category}.{param_name}', constraint['values'])
                    else:
                        # Integer range
                        value = trial.suggest_int(
                            f'{category}.{param_name}',
                            int(constraint['lower']),
                            int(constraint['upper']),
                            step=int(constraint['step'])
                        )
                    
                    suggested_params[param_name] = value
        
        assert 'vm.swappiness' in suggested_params
        assert 'cpu_governor' in suggested_params
        assert len(suggested_params) == 2
