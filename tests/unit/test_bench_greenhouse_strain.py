import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

sys.modules['optuna'] = MagicMock()

from strains.bench_greenhouse.strain import suggest_params, _suggest_single, validate_config


class TestSuggestSingleParam:
    def test_categorical_param(self):
        trial = MagicMock()
        trial.suggest_categorical.return_value = 'performance'

        result = _suggest_single(
            trial, 'qdisc',
            [{'values': ['fq', 'fq_codel', 'codel']}],
        )
        assert result == 'performance'
        trial.suggest_categorical.assert_called_once_with('qdisc', ['fq', 'fq_codel', 'codel'])

    def test_integer_range_param(self):
        trial = MagicMock()
        trial.suggest_int.return_value = 60

        result = _suggest_single(
            trial, 'sim_steps',
            [{'step': 10, 'lower': 10, 'upper': 200}],
        )
        assert result == 60
        trial.suggest_int.assert_called_once_with('sim_steps', 10, 200, step=10)

    def test_float_range_param(self):
        trial = MagicMock()
        trial.suggest_float.return_value = 0.5

        result = _suggest_single(
            trial, 'shading',
            [{'step': 0.05, 'lower': 0.0, 'upper': 1.0}],
        )
        assert result == 0.5
        trial.suggest_float.assert_called_once_with('shading', 0.0, 1.0, step=0.05)

    def test_empty_constraints_raises(self):
        trial = MagicMock()
        with pytest.raises(ValueError, match="constraints must be a non-empty list"):
            _suggest_single(trial, 'param', [])

    def test_non_list_constraints_raises(self):
        trial = MagicMock()
        with pytest.raises(ValueError, match="constraints must be a non-empty list"):
            _suggest_single(trial, 'param', "not_a_list")

    def test_invalid_constraint_format_raises(self):
        trial = MagicMock()
        with pytest.raises(ValueError, match="constraint must have either"):
            _suggest_single(trial, 'param', [{'unknown_key': 1}])


class TestSuggestParams:
    def test_global_params(self):
        trial = MagicMock()
        trial.suggest_float.return_value = 0.3
        trial.suggest_int.return_value = 60

        settings = {
            'greenhouse': {
                'zones': 2,
                'shading': {
                    'constraints': [{'step': 0.05, 'lower': 0.0, 'upper': 1.0}]
                },
                'sim_steps': {
                    'constraints': [{'step': 10, 'lower': 10, 'upper': 200}]
                },
            }
        }

        result = suggest_params(trial, settings)
        assert 'shading' in result
        assert 'sim_steps' in result

    def test_per_zone_params_expanded(self):
        trial = MagicMock()
        trial.suggest_float.return_value = 22.0

        settings = {
            'greenhouse': {
                'zones': 3,
                'heating_setpoints': {
                    'constraints': [{'step': 0.5, 'lower': 5.0, 'upper': 40.0}]
                },
            }
        }

        result = suggest_params(trial, settings)
        assert 'heating_setpoints' in result
        assert len(result['heating_setpoints']) == 3
        assert trial.suggest_float.call_count == 3

        call_names = [c[0][0] for c in trial.suggest_float.call_args_list]
        assert call_names == ['heating_setpoints_0', 'heating_setpoints_1', 'heating_setpoints_2']

    def test_vent_openings_per_zone(self):
        trial = MagicMock()
        trial.suggest_float.return_value = 0.3

        settings = {
            'greenhouse': {
                'zones': 2,
                'vent_openings': {
                    'constraints': [{'step': 0.05, 'lower': 0.0, 'upper': 1.0}]
                },
            }
        }

        result = suggest_params(trial, settings)
        assert len(result['vent_openings']) == 2

    def test_mixed_global_and_per_zone(self):
        trial = MagicMock()
        trial.suggest_float.return_value = 0.5
        trial.suggest_int.return_value = 60

        settings = {
            'greenhouse': {
                'zones': 2,
                'heating_setpoints': {
                    'constraints': [{'step': 0.5, 'lower': 5.0, 'upper': 40.0}]
                },
                'shading': {
                    'constraints': [{'step': 0.05, 'lower': 0.0, 'upper': 1.0}]
                },
            }
        }

        result = suggest_params(trial, settings)
        assert len(result['heating_setpoints']) == 2
        assert 'shading' in result
        assert isinstance(result['shading'], float)

    def test_skips_missing_constraints(self):
        trial = MagicMock()

        settings = {
            'greenhouse': {
                'zones': 2,
                'some_param': {
                    'no_constraints_key': True
                },
            }
        }

        result = suggest_params(trial, settings)
        assert result == {}

    def test_empty_settings(self):
        trial = MagicMock()
        result = suggest_params(trial, {})
        assert result == {}

    def test_zones_key_skipped(self):
        trial = MagicMock()

        settings = {
            'greenhouse': {
                'zones': 4,
            }
        }

        result = suggest_params(trial, settings)
        assert result == {}


class TestValidateConfig:
    def test_delegates_to_preflight(self):
        config = {'settings': {'greenhouse': {'zones': 2}}}
        mock_preflight = MagicMock()
        mock_preflight.main.return_value = {'result': 'SUCCESS'}
        parent_mod = sys.modules['f.breeder.strains.bench_greenhouse']
        with patch.object(parent_mod, 'preflight', mock_preflight, create=True):
            result = validate_config(config)
            mock_preflight.main.assert_called_once_with(config)
            assert result['result'] == 'SUCCESS'
