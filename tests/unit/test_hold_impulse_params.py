#
# Copyright (c) 2019 Matthias Tafelmeier.
#
# Tests for hold and impulse param format correctness.
#
import pytest
import json
import sys
import types
from unittest.mock import MagicMock, patch
from optuna.trial import TrialState

otel_mock = types.ModuleType('f.breeder.shared.otel_logging')
otel_mock.get_logger = lambda name: type('Logger', (), {
    'info': lambda *a, **kw: None, 'warning': lambda *a, **kw: None, 'error': lambda *a, **kw: None,
})()
sys.modules['f.breeder.shared.otel_logging'] = otel_mock
wmill_mock = types.ModuleType('wmill')
wmill_mock.run_script_by_path = MagicMock()
sys.modules['wmill'] = wmill_mock
psycopg2_mock = types.ModuleType('psycopg2')
sys.modules['psycopg2'] = psycopg2_mock

from engine.breeder_worker import BreederWorker


def _greenhouse_settings():
    return {
        'greenhouse': {
            'zones': 2,
            'heating_setpoints': {'constraints': [{'step': 0.5, 'lower': 5.0, 'upper': 40.0}]},
            'vent_openings': {'constraints': [{'step': 0.05, 'lower': 0.0, 'upper': 1.0}]},
            'co2_injection': {'constraints': [{'step': 0.5, 'lower': 0.0, 'upper': 20.0}]},
        }
    }


def _make_trial(state=TrialState.COMPLETE, user_attrs=None, flat_params=None):
    trial = MagicMock()
    trial.state = state
    trial.user_attrs = user_attrs or {}
    trial.params = flat_params or {}
    return trial


def _make_worker_with_trials(trials):
    """Create a worker with a real list for study.trials."""
    config = {
        'breeder': {'type': 'bench_greenhouse', 'uuid': 'test-uuid', 'name': 'test'},
        'creation_ts': '2025-01-15T10:30:00Z',
        'settings': _greenhouse_settings(),
        'effectuation': {'type': 'http', 'targets': []},
        'run': {'parallel': 1},
        'objectives': [{'name': 'growth_rate', 'direction': 'maximize'}],
        'reconnaissance': {'type': 'prometheus'},
        'interference_detection': {'mode': 'active'},
    }
    study = MagicMock()
    study.trials = list(trials)
    with patch.object(BreederWorker, '_load_or_create_study', return_value=study), \
         patch.object(BreederWorker, '_setup_communication', return_value=None), \
         patch.object(BreederWorker, '_update_state'), \
         patch.object(BreederWorker, '_register_interference_breeder'), \
         patch('engine.breeder_worker.load_strain', return_value=MagicMock()):
        worker = BreederWorker(config)
    worker.study = study
    return worker


class TestHoldParamsFormat:
    def test_reads_stashed_effectuation_params(self):
        nested = {'heating_setpoints': [11.0, 13.5], 'co2_injection': 3.5}
        trial = _make_trial(user_attrs={'effectuation_params': json.dumps(nested)})
        worker = _make_worker_with_trials([trial])

        # Debug
        print(f"study type: {type(worker.study)}")
        print(f"trials type: {type(worker.study.trials)}")
        print(f"trials len: {len(worker.study.trials)}")
        if worker.study.trials:
            t = worker.study.trials[0]
            print(f"trial.state={t.state}, type={type(t.state)}")
            print(f"COMPLETE={TrialState.COMPLETE}, type={type(TrialState.COMPLETE)}")
            print(f"match={t.state == TrialState.COMPLETE}")
            print(f"user_attrs={t.user_attrs}")
            print(f"get result={t.user_attrs.get('effectuation_params')}")

        result = worker._get_last_successful_params()
        assert result == nested

    def test_per_zone_stays_as_list(self):
        nested = {'heating_setpoints': [38.0, 39.0], 'vent_openings': [1.0, 1.0]}
        trial = _make_trial(user_attrs={'effectuation_params': json.dumps(nested)})
        worker = _make_worker_with_trials([trial])

        result = worker._get_last_successful_params()
        assert isinstance(result['heating_setpoints'], list)
        assert isinstance(result['vent_openings'], list)

    def test_fallback_to_flat_params_without_stash(self):
        trial = _make_trial(flat_params={'co2_injection': 3.5})
        worker = _make_worker_with_trials([trial])

        result = worker._get_last_successful_params()
        assert result == {'co2_injection': 3.5}

    def test_returns_none_no_completed_trials(self):
        worker = _make_worker_with_trials([])
        assert worker._get_last_successful_params() is None


class TestImpulseParamsFormat:
    def test_uses_strain_format_template(self):
        template = {'heating_setpoints': [11.0, 13.5], 'co2_injection': 3.5, 'vent_openings': [0.5, 0.6]}
        trial = _make_trial(user_attrs={'effectuation_params': json.dumps(template)})
        worker = _make_worker_with_trials([trial])

        result = worker._generate_impulse_params(_greenhouse_settings())
        assert result is not None
        assert isinstance(result['heating_setpoints'], list)

    def test_overrides_top3_to_upper_bounds(self):
        template = {'heating_setpoints': [11.0, 13.5], 'co2_injection': 3.5, 'vent_openings': [0.5, 0.6]}
        trial = _make_trial(user_attrs={'effectuation_params': json.dumps(template)})
        worker = _make_worker_with_trials([trial])

        result = worker._generate_impulse_params(_greenhouse_settings())
        assert result is not None
        assert result['heating_setpoints'] == [40.0, 40.0]
        assert result['co2_injection'] == 20.0

    def test_returns_none_without_prior_trial(self):
        worker = _make_worker_with_trials([])
        result = worker._generate_impulse_params(_greenhouse_settings())
        assert result is None


class TestCollectUpperBounds:
    def test_finds_nested_constraints(self):
        worker = _make_worker_with_trials([])
        results = worker._collect_upper_bounds(_greenhouse_settings())
        names = [r['name'] for r in results]
        assert 'heating_setpoints' in names
        assert 'co2_injection' in names
        assert 'vent_openings' in names

    def test_computes_range(self):
        worker = _make_worker_with_trials([])
        results = worker._collect_upper_bounds(_greenhouse_settings())
        for r in results:
            assert r['range'] == r['upper'] - r['lower']

    def test_handles_empty_settings(self):
        worker = _make_worker_with_trials([])
        results = worker._collect_upper_bounds({})
        assert results == []
