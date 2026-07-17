
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

# Mock Windmill package namespace for breeder_worker internal imports
for mod_path in ['f', 'f.breeder', 'f.breeder.engine',
                 'f.breeder.engine.detection_coordinator',
                 'f.breeder.engine.breeder_metrics_client',
                 'f.breeder.engine.communication',
                 'f.breeder.engine.strain_loader',
                 'f.breeder.engine.watermark']:
    parts = mod_path.split('.')
    parent = '.'.join(parts[:-1]) if len(parts) > 1 else None
    mock_mod = types.ModuleType(mod_path)
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], parts[-1], mock_mod)
    sys.modules[mod_path] = mock_mod

sys.modules['f.breeder.engine.detection_coordinator'].DetectionCoordinator = MagicMock
sys.modules['f.breeder.engine.breeder_metrics_client'].BreederMetricsClient = MagicMock
sys.modules['f.breeder.engine.communication'].CommunicationCallback = MagicMock
sys.modules['f.breeder.engine.strain_loader'].load_strain = MagicMock(return_value=MagicMock())
sys.modules['f.breeder.engine.watermark'].create_watermark = MagicMock(return_value=None)
sys.modules['f.breeder.engine.watermark'].Watermark = MagicMock
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


def _make_trial(state=1, user_attrs=None, flat_params=None):
    """Create a mock trial. state defaults to 1 (TrialState.COMPLETE)."""
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

        result = worker._compute_neutral_params()
        assert result is not None
        assert isinstance(result['heating_setpoints'], list)

    def test_overrides_to_midpoints(self):
        template = {'heating_setpoints': [11.0, 13.5], 'co2_injection': 3.5, 'vent_openings': [0.5, 0.6]}
        trial = _make_trial(user_attrs={'effectuation_params': json.dumps(template)})
        worker = _make_worker_with_trials([trial])

        result = worker._compute_neutral_params()
        assert result is not None
        # heating_setpoints midpoint of [5, 40] = 22.5
        assert result['heating_setpoints'] == [22.5, 22.5]
        # co2_injection midpoint of [0, 20] = 10.0
        assert result['co2_injection'] == 10.0

    def test_returns_none_without_prior_trial(self):
        worker = _make_worker_with_trials([])
        result = worker._compute_neutral_params()
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
