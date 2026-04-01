import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

sys.modules['optuna'] = MagicMock()

from strains.linux_performance.strain import suggest_params, _suggest_single_param, validate_config


class TestSuggestSingleParam:
    def test_categorical_param(self):
        trial = MagicMock()
        trial.suggest_categorical.return_value = 'performance'

        result = _suggest_single_param(
            trial, 'cpu_governor',
            [{'values': ['performance', 'powersave']}],
            category='sysfs',
        )
        assert result == 'performance'
        trial.suggest_categorical.assert_called_once_with('cpu_governor', ['performance', 'powersave'])

    def test_integer_range_param(self):
        trial = MagicMock()
        trial.suggest_int.return_value = 50000

        result = _suggest_single_param(
            trial, 'net.core.rmem_max',
            [{'step': 100, 'lower': 4096, 'upper': 65536}],
            category='sysctl',
        )
        assert result == 50000
        trial.suggest_int.assert_called_once_with('net.core.rmem_max', 4096, 65536, step=100)

    def test_float_range_param(self):
        trial = MagicMock()
        trial.suggest_float.return_value = 2.5

        result = _suggest_single_param(
            trial, 'min_freq_ghz',
            [{'step': 0.1, 'lower': 1.0, 'upper': 3.0}],
            category='cpufreq',
        )
        assert result == 2.5
        trial.suggest_float.assert_called_once_with('min_freq_ghz', 1.0, 3.0, step=0.1)

    def test_empty_constraints_raises(self):
        trial = MagicMock()
        with pytest.raises(ValueError, match="constraints must be a non-empty list"):
            _suggest_single_param(trial, 'param', [], category='sysctl')

    def test_non_list_constraints_raises(self):
        trial = MagicMock()
        with pytest.raises(ValueError, match="constraints must be a non-empty list"):
            _suggest_single_param(trial, 'param', "not_a_list", category='sysctl')

    def test_invalid_constraint_format_raises(self):
        trial = MagicMock()
        with pytest.raises(ValueError, match="constraint must have either"):
            _suggest_single_param(trial, 'param', [{'unknown_key': 1}], category='sysctl')


class TestSuggestParams:
    def test_sysctl_category(self):
        trial = MagicMock()
        trial.suggest_int.return_value = 50000

        settings = {
            'sysctl': {
                'net.core.rmem_max': {
                    'constraints': [{'step': 100, 'lower': 4096, 'upper': 65536}]
                }
            }
        }

        result = suggest_params(trial, settings)
        assert 'net.core.rmem_max' in result
        assert result['net.core.rmem_max'] == 50000

    def test_sysfs_category(self):
        trial = MagicMock()
        trial.suggest_categorical.return_value = 'performance'

        settings = {
            'sysfs': {
                'cpu_governor': {
                    'constraints': [{'values': ['performance', 'powersave']}]
                }
            }
        }

        result = suggest_params(trial, settings)
        assert result['cpu_governor'] == 'performance'

    def test_ethtool_category(self):
        trial = MagicMock()
        trial.suggest_categorical.return_value = 'on'

        settings = {
            'ethtool': {
                'eth0': {
                    'tso': {
                        'constraints': [{'values': ['on', 'off']}]
                    }
                }
            }
        }

        result = suggest_params(trial, settings)
        assert 'eth0_tso' in result

    def test_multiple_categories(self):
        trial = MagicMock()
        trial.suggest_int.return_value = 50000
        trial.suggest_categorical.return_value = 'performance'

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

        result = suggest_params(trial, settings)
        assert 'vm.swappiness' in result
        assert 'cpu_governor' in result

    def test_skips_missing_constraints(self):
        trial = MagicMock()

        settings = {
            'sysctl': {
                'some_param': {
                    'no_constraints_key': True
                }
            }
        }

        result = suggest_params(trial, settings)
        assert result == {}

    def test_empty_settings(self):
        trial = MagicMock()
        result = suggest_params(trial, {})
        assert result == {}

    def test_unknown_category_skipped(self):
        trial = MagicMock()

        settings = {
            'unknown_category': {
                'param': {'constraints': [{'values': ['a']}]}
            }
        }

        result = suggest_params(trial, settings)
        assert result == {}

    def test_ethtool_invalid_interface_config_skipped(self):
        trial = MagicMock()

        settings = {
            'ethtool': {
                'eth0': 'not_a_dict'
            }
        }

        result = suggest_params(trial, settings)
        assert result == {}


class TestValidateConfig:
    def test_delegates_to_preflight(self):
        config = {'settings': {'sysctl': {}}}
        with patch('strains.linux_performance.strain.preflight') as mock_preflight:
            mock_preflight.main.return_value = {'result': 'SUCCESS'}
            result = validate_config(config)
            mock_preflight.main.assert_called_once_with(config)
            assert result['result'] == 'SUCCESS'
