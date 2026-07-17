
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

sys.path.insert(0, '.')

# Mock Windmill imports before anything else
sys.modules['wmill'] = MagicMock()
sys.modules['prometheus_api_client'] = MagicMock()
sys.modules['prometheus_api_client.exceptions'] = MagicMock()
sys.modules['prometheus_client'] = MagicMock()
sys.modules['psycopg2'] = MagicMock()

# Mock f.breeder.shared.otel_logging
fake_f = MagicMock()
fake_breeder = MagicMock()
fake_breeder_shared = MagicMock()
fake_f.breeder = fake_breeder
fake_breeder.shared = fake_breeder_shared
sys.modules['f'] = fake_f
sys.modules['f.breeder'] = fake_breeder
sys.modules['f.breeder.shared'] = fake_breeder_shared

fake_otel = MagicMock()
fake_otel.get_logger = lambda name: MagicMock()
sys.modules['f.breeder.shared.otel_logging'] = fake_otel

# Import the module (not individual functions) for patch.object
import reconnaissance.http as http_mod
from reconnaissance.http import _gather_single_metric, _aggregate_samples, _compute_cv, _sprt_sample_stable, main as recon_main


class TestHttpGetWithRetry:
    @patch('reconnaissance.http.requests.get')
    def test_succeeds_on_first_attempt(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {'growth_rate': 0.85}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        from reconnaissance.http import _http_get_with_retry
        result = _http_get_with_retry('http://localhost:8090/metrics/json')
        assert result == {'growth_rate': 0.85}
        mock_get.assert_called_once()

    @patch('reconnaissance.http.time.sleep')
    @patch('reconnaissance.http.requests.get')
    def test_retries_on_connection_error(self, mock_get, mock_sleep):
        from requests.exceptions import ConnectionError
        from reconnaissance.http import _http_get_with_retry

        mock_response = MagicMock()
        mock_response.json.return_value = {'growth_rate': 0.85}
        mock_response.raise_for_status.return_value = None
        mock_get.side_effect = [
            ConnectionError("refused"),
            mock_response,
        ]

        result = _http_get_with_retry('http://localhost:8090/metrics/json', max_retries=3, initial_delay=1)
        assert result == {'growth_rate': 0.85}
        assert mock_get.call_count == 2

    @patch('reconnaissance.http.time.sleep')
    @patch('reconnaissance.http.requests.get')
    def test_raises_after_exhausted_retries(self, mock_get, mock_sleep):
        from requests.exceptions import ConnectionError
        from reconnaissance.http import _http_get_with_retry

        mock_get.side_effect = ConnectionError("refused")

        with pytest.raises(Exception, match="failed after 2 retries"):
            _http_get_with_retry('http://localhost:8090/metrics/json', max_retries=2, initial_delay=1)


class TestAggregateSamples:
    def test_median_aggregation(self):
        result = _aggregate_samples([10.0, 11.0, 12.0, 100.0, 9.0], method='median')
        assert result == 11.0

    def test_filters_none_values(self):
        result = _aggregate_samples([10.0, None, 12.0, None, 11.0], method='median')
        assert result == 11.0

    def test_all_none_returns_inf(self):
        result = _aggregate_samples([None, None], method='median')
        assert result == float('inf')


class TestGatherSingleMetric:
    @patch.object(http_mod, 'time')
    @patch.object(http_mod, '_http_get_with_retry')
    def test_single_sample_no_stabilization(self, mock_http, mock_time):
        mock_http.return_value = {'growth_rate': 0.85}

        recon_config = {
            'service': 'http',
            'path': '/metrics/json',
            'key': 'growth_rate',
            'stabilization_seconds': 0,
            'samples': 1,
            'interval': 0,
            'aggregation': 'median',
        }

        result = _gather_single_metric('http://localhost:8090', 'growth_rate', recon_config)
        assert result['value'] == 0.85
        mock_time.sleep.assert_not_called()

    @patch.object(http_mod, 'time')
    @patch.object(http_mod, '_http_get_with_retry')
    def test_with_stabilization_wait(self, mock_http, mock_time):
        mock_http.return_value = {'growth_rate': 0.9}

        recon_config = {
            'service': 'http',
            'path': '/metrics/json',
            'key': 'growth_rate',
            'stabilization_seconds': 5,
            'samples': 1,
            'interval': 0,
            'aggregation': 'median',
        }

        result = _gather_single_metric('http://localhost:8090', 'growth_rate', recon_config)
        assert result['value'] == 0.9
        mock_time.sleep.assert_any_call(5)

    @patch.object(http_mod, 'time')
    @patch.object(http_mod, '_http_get_with_retry')
    def test_multiple_samples_with_interval(self, mock_http, mock_time):
        mock_http.return_value = {'energy': 12.5}

        recon_config = {
            'service': 'http',
            'path': '/metrics/json',
            'key': 'energy',
            'stabilization_seconds': 0,
            'samples': 3,
            'interval': 2,
            'aggregation': 'median',
        }

        result = _gather_single_metric('http://localhost:8090', 'energy', recon_config)
        assert result['value'] == 12.5
        assert mock_http.call_count == 3

    @patch.object(http_mod, '_http_get_with_retry')
    def test_missing_key_returns_none_sample(self, mock_http):
        mock_http.return_value = {'other_key': 42.0}

        recon_config = {
            'service': 'http',
            'path': '/metrics/json',
            'key': 'growth_rate',
            'stabilization_seconds': 0,
            'samples': 1,
            'interval': 0,
            'aggregation': 'median',
        }

        result = _gather_single_metric('http://localhost:8090', 'growth_rate', recon_config)
        assert result['value'] == float('inf')

    @patch.object(http_mod, '_http_get_with_retry')
    def test_http_failure_returns_inf(self, mock_http):
        mock_http.side_effect = Exception("Connection refused")

        recon_config = {
            'service': 'http',
            'path': '/metrics/json',
            'key': 'growth_rate',
            'stabilization_seconds': 0,
            'samples': 1,
            'interval': 0,
            'aggregation': 'median',
        }

        result = _gather_single_metric('http://localhost:8090', 'growth_rate', recon_config)
        assert result['value'] == float('inf')

    def test_unsupported_service_returns_inf(self):
        recon_config = {
            'service': 'unsupported',
            'path': '/metrics/json',
            'key': 'growth_rate',
        }

        result = _gather_single_metric('http://localhost:8090', 'growth_rate', recon_config)
        assert result['value'] == float('inf')


class TestComputeCV:
    def test_low_cv_stable_signal(self):
        cv = _compute_cv([10.0, 10.1, 9.9, 10.0, 10.05])
        assert cv < 0.02

    def test_single_sample_returns_inf(self):
        cv = _compute_cv([10.0])
        assert cv == float('inf')


class TestSprtSampleStable:
    def test_stable_with_3_samples(self):
        assert _sprt_sample_stable([10.0, 10.01, 9.99]) is True

    def test_too_few_samples(self):
        assert _sprt_sample_stable([10.0, 10.0]) is False


class TestAdaptiveSampling:
    @patch.object(http_mod, 'time')
    @patch.object(http_mod, '_http_get_with_retry')
    def test_stops_early_when_stable(self, mock_http, mock_time):
        mock_http.side_effect = [{'temp': 100.0}] * 3 + [{'temp': 200.0}] * 20

        recon_config = {
            'service': 'http',
            'path': '/metrics/json',
            'key': 'temp',
            'stabilization_seconds': 0,
            'samples': 3,
            'max_samples': 20,
            'interval': 0,
            'aggregation': 'median',
            'cv_threshold': 0.05,
        }

        result = _gather_single_metric('http://localhost:8090', 'temp', recon_config)
        assert result['value'] == 100.0
        assert mock_http.call_count == 3

    @patch.object(http_mod, 'time')
    @patch.object(http_mod, '_http_get_with_retry')
    def test_keeps_sampling_when_noisy(self, mock_http, mock_time):
        mock_http.side_effect = [
            {'temp': 80.0}, {'temp': 120.0}, {'temp': 60.0},
            {'temp': 140.0}, {'temp': 50.0}, {'temp': 130.0},
            {'temp': 70.0}, {'temp': 110.0}, {'temp': 90.0},
            {'temp': 100.0},
        ]

        recon_config = {
            'service': 'http',
            'path': '/metrics/json',
            'key': 'temp',
            'stabilization_seconds': 0,
            'samples': 3,
            'max_samples': 20,
            'interval': 0,
            'aggregation': 'median',
            'cv_threshold': 0.01,
        }

        result = _gather_single_metric('http://localhost:8090', 'temp', recon_config)
        assert mock_http.call_count > 3
        assert mock_http.call_count <= 20

    @patch.object(http_mod, 'time')
    @patch.object(http_mod, '_http_get_with_retry')
    def test_respects_max_samples(self, mock_http, mock_time):
        mock_http.side_effect = [{'temp': float(i)} for i in range(100)]

        recon_config = {
            'service': 'http',
            'path': '/metrics/json',
            'key': 'temp',
            'stabilization_seconds': 0,
            'samples': 3,
            'max_samples': 8,
            'interval': 0,
            'aggregation': 'median',
            'cv_threshold': 0.001,
        }

        result = _gather_single_metric('http://localhost:8090', 'temp', recon_config)
        assert mock_http.call_count == 8

    @patch.object(http_mod, 'time')
    @patch.object(http_mod, '_http_get_with_retry')
    def test_backward_compat_no_max_samples(self, mock_http, mock_time):
        mock_http.return_value = {'temp': 42.0}

        recon_config = {
            'service': 'http',
            'path': '/metrics/json',
            'key': 'temp',
            'stabilization_seconds': 0,
            'samples': 5,
            'interval': 0,
            'aggregation': 'median',
        }

        result = _gather_single_metric('http://localhost:8090', 'temp', recon_config)
        assert result['value'] == 42.0
        assert mock_http.call_count == 5

    @patch.object(http_mod, 'time')
    @patch.object(http_mod, '_http_get_with_retry')
    def test_backward_compat_single_sample(self, mock_http, mock_time):
        mock_http.return_value = {'temp': 42.0}

        recon_config = {
            'service': 'http',
            'path': '/metrics/json',
            'key': 'temp',
            'stabilization_seconds': 0,
            'samples': 1,
            'interval': 0,
            'aggregation': 'median',
        }

        result = _gather_single_metric('http://localhost:8090', 'temp', recon_config)
        assert result['value'] == 42.0
        assert mock_http.call_count == 1


class TestHttpReconnaissanceMain:
    @patch.object(http_mod, '_gather_single_metric')
    def test_gathers_objective_metrics(self, mock_gather):
        mock_gather.return_value = {'value': 0.85, 'noise_cv': None}

        context = {
            'reconnaissance': {
                'http': {'url': 'http://greenhouse:8090'}
            },
            'objectives': [
                {
                    'name': 'growth_rate',
                    'reconnaissance': {
                        'service': 'http',
                        'path': '/metrics/json',
                        'key': 'growth_rate',
                    }
                }
            ],
            'guardrails': []
        }

        result = recon_main(context, [], {})
        assert result['status'] == 'completed'
        assert result['metrics']['growth_rate'] == 0.85

    @patch.object(http_mod, '_gather_single_metric')
    def test_empty_objectives_and_guardrails(self, mock_gather):
        context = {
            'reconnaissance': {'http': {'url': 'http://localhost:8090'}},
            'objectives': [],
            'guardrails': []
        }

        result = recon_main(context, [], {})
        assert result['status'] == 'completed'
        assert result['metrics'] == {}
