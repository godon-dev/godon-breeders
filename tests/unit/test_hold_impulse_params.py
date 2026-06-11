#
# Copyright (c) 2019 Matthias Tafelmeier.
#
# This file is part of godon
#
# godon is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
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


def _greenhouse_config():
    return {
        'breeder': {'type': 'bench_greenhouse', 'uuid': 'test-uuid'},
        'settings': {
            'greenhouse': {
                'zones': 2,
                'heating_setpoints': {'constraints': [{'step': 0.5, 'lower': 5.0, 'upper': 40.0}]},
                'vent_openings': {'constraints': [{'step': 0.05, 'lower': 0.0, 'upper': 1.0}]},
                'co2_injection': {'constraints': [{'step': 0.5, 'lower': 0.0, 'upper': 20.0}]},
            }
        },
        'effectuation': {'type': 'http', 'targets': []},
        'run': {'parallel': 1},
        'interference_detection': {'mode': 'active'},
    }


def _make_worker(config=None):
    config = config or _greenhouse_config()
    with patch('engine.breeder_worker.OptunaStrain'), \
         patch('engine.breeder_worker.create_watermark', return_value=None), \
         patch('engine.breeder_worker.MetricsPusher'):
        worker = BreederWorker(config, breeder_id='test-uuid')
    worker.study = MagicMock()
    return worker


class TestHoldParamsFormat:
    """Hold mode must return effectuation-format params."""

    def test_reads_stashed_effectuation_params(self):
        worker = _make_worker()
        nested = {'heating_setpoints': [11.0, 13.5], 'co2_injection': 3.5}
        trial = MagicMock()
        trial.state = TrialState.COMPLETE
        trial.user_attrs = {'effectuation_params': json.dumps(nested)}
        trial.params = {'heating_setpoints_0': 11.0, 'co2_injection': 3.5}
        worker.study.trials = [trial]

        result = worker._get_last_successful_params()
        assert result == nested

    def test_per_zone_stays_as_list(self):
        worker = _make_worker()
        nested = {'heating_setpoints': [38.0, 39.0], 'vent_openings': [1.0, 1.0]}
        trial = MagicMock()
        trial.state = TrialState.COMPLETE
        trial.user_attrs = {'effectuation_params': json.dumps(nested)}
        worker.study.trials = [trial]

        result = worker._get_last_successful_params()
        assert isinstance(result['heating_setpoints'], list)
        assert isinstance(result['vent_openings'], list)

    def test_fallback_to_flat_params_without_stash(self):
        worker = _make_worker()
        trial = MagicMock()
        trial.state = TrialState.COMPLETE
        trial.user_attrs = {}
        trial.params = {'co2_injection': 3.5}
        worker.study.trials = [trial]

        result = worker._get_last_successful_params()
        assert result == {'co2_injection': 3.5}

    def test_returns_none_no_completed_trials(self):
        worker = _make_worker()
        worker.study.trials = []
        assert worker._get_last_successful_params() is None


class TestImpulseParamsFormat:
    """Impulse mode must produce effectuation-format params using strain template."""

    def test_uses_strain_format_template(self):
        worker = _make_worker()
        template = {'heating_setpoints': [11.0, 13.5], 'co2_injection': 3.5, 'vent_openings': [0.5, 0.6]}
        trial = MagicMock()
        trial.state = TrialState.COMPLETE
        trial.user_attrs = {'effectuation_params': json.dumps(template)}
        worker.study.trials = [trial]

        result = worker._generate_impulse_params(_greenhouse_config()['settings'])
        assert result is not None
        # heating_setpoints must stay as list
        assert isinstance(result['heating_setpoints'], list)

    def test_overrides_top3_to_upper_bounds(self):
        worker = _make_worker()
        template = {'heating_setpoints': [11.0, 13.5], 'co2_injection': 3.5, 'vent_openings': [0.5, 0.6]}
        trial = MagicMock()
        trial.state = TrialState.COMPLETE
        trial.user_attrs = {'effectuation_params': json.dumps(template)}
        worker.study.trials = [trial]

        result = worker._generate_impulse_params(_greenhouse_config()['settings'])
        assert result is not None
        # Top 3 by range: heating (35), co2 (20), vent (1)
        # heating and co2 should be at upper bounds
        assert result['heating_setpoints'] == [40.0, 40.0]
        assert result['co2_injection'] == 20.0

    def test_returns_none_without_prior_trial(self):
        worker = _make_worker()
        worker.study.trials = []
        result = worker._generate_impulse_params(_greenhouse_config()['settings'])
        assert result is None


class TestCollectUpperBounds:
    """_collect_upper_bounds walks config tree generically."""

    def test_finds_nested_constraints(self):
        worker = _make_worker()
        results = worker._collect_upper_bounds(_greenhouse_config()['settings'])
        names = [r['name'] for r in results]
        assert 'heating_setpoints' in names
        assert 'co2_injection' in names
        assert 'vent_openings' in names

    def test_computes_range(self):
        worker = _make_worker()
        results = worker._collect_upper_bounds(_greenhouse_config()['settings'])
        for r in results:
            assert r['range'] == r['upper'] - r['lower']

    def test_handles_empty_settings(self):
        worker = _make_worker()
        results = worker._collect_upper_bounds({})
        assert results == []
