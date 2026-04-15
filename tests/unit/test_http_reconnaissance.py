import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from reconnaissance.http import _http_get_with_retry, _aggregate_samples, _gather_single_metric, main as recon_main


class TestHttpGetWithRetry:
    @patch('reconnaissance.http.requests.get')
    def test_succeeds_on_first_attempt(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {'growth_rate': 0.85}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = _http_get_with_retry('http://localhost:8090/metrics/json')
        assert result == {'growth_rate': 0.85}
        mock_get.assert_called_once()

    @patch('reconnaissance.http.time.sleep')
    @patch('reconnaissance.http.requests.get')
    def test_retries_on_connection_error(self, mock_get, mock_sleep):
        from requests.exceptions import ConnectionError

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
    def test_retries_on_timeout(self, mock_get, mock_sleep):
        from requests.exceptions import Timeout

        mock_response = MagicMock()
        mock_response.json.return_value = {'growth_rate': 0.85}
        mock_response.raise_for_status.return_value = None
        mock_get.side_effect = [
            Timeout("timeout"),
            Timeout("timeout"),
            mock_response,
        ]

        result = _http_get_with_retry('http://localhost:8090/metrics/json', max_retries=3, initial_delay=1)
        assert mock_get.call_count == 3

    @patch('reconnaissance.http.time.sleep')
    @patch('reconnaissance.http.requests.get')
    def test_raises_after_exhausted_retries(self, mock_get, mock_sleep):
        from requests.exceptions import ConnectionError

        mock_get.side_effect = ConnectionError("refused")

        with pytest.raises(Exception, match="failed after 2 retries"):
            _http_get_with_retry('http://localhost:8090/metrics/json', max_retries=2, initial_delay=1)

    @patch('reconnaissance.http.requests.get')
    def test_non_retryable_error_raises_immediately(self, mock_get):
        mock_get.side_effect = RuntimeError("unexpected")

        with pytest.raises(RuntimeError, match="unexpected"):
            _http_get_with_retry('http://localhost:8090/metrics/json')

    @patch('reconnaissance.http.time.sleep')
    @patch('reconnaissance.http.requests.get')
    def test_exponential_backoff_delay(self, mock_get, mock_sleep):
        from requests.exceptions import ConnectionError

        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_get.side_effect = [
            ConnectionError("fail"),
            ConnectionError("fail"),
            mock_response,
        ]

        _http_get_with_retry('http://localhost:8090/metrics/json', max_retries=3, initial_delay=5)
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert sleep_calls == [5, 10]


class TestAggregateSamples:
    def test_median_aggregation(self):
        result = _aggregate_samples([10.0, 11.0, 12.0, 100.0, 9.0], method='median')
        assert result == 11.0

    def test_mean_aggregation(self):
        result = _aggregate_samples([10.0, 20.0, 30.0], method='mean')
        assert result == 20.0

    def test_min_aggregation(self):
        result = _aggregate_samples([15.0, 10.0, 25.0], method='min')
        assert result == 10.0

    def test_max_aggregation(self):
        result = _aggregate_samples([15.0, 10.0, 25.0], method='max')
        assert result == 25.0

    def test_filters_none_values(self):
        result = _aggregate_samples([10.0, None, 12.0, None, 11.0], method='median')
        assert result == 11.0

    def test_all_none_returns_inf(self):
        result = _aggregate_samples([None, None], method='median')
        assert result == float('inf')

    def test_empty_list_returns_inf(self):
        result = _aggregate_samples([], method='median')
        assert result == float('inf')

    def test_unknown_method_defaults_to_median(self):
        result = _aggregate_samples([10.0, 15.0, 20.0], method='bogus')
        assert result == 15.0


class TestGatherSingleMetric:
    @patch('reconnaissance.http.time.sleep')
    @patch('reconnaissance.http._http_get_with_retry')
    def test_single_sample_no_stabilization(self, mock_http, mock_sleep):
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
        assert result == 0.85
        mock_sleep.assert_not_called()

    @patch('reconnaissance.http.time.sleep')
    @patch('reconnaissance.http._http_get_with_retry')
    def test_with_stabilization_wait(self, mock_http, mock_sleep):
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
        assert result == 0.9
        mock_sleep.assert_any_call(5)

    @patch('reconnaissance.http.time.sleep')
    @patch('reconnaissance.http._http_get_with_retry')
    def test_multiple_samples_with_interval(self, mock_http, mock_sleep):
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
        assert result == 12.5
        assert mock_http.call_count == 3
        interval_calls = [c for c in mock_sleep.call_args_list if c[0][0] == 2]
        assert len(interval_calls) == 2

    @patch('reconnaissance.http._http_get_with_retry')
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
        assert result == float('inf')

    @patch('reconnaissance.http._http_get_with_retry')
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
        assert result == float('inf')

    def test_unsupported_service_returns_inf(self):
        recon_config = {
            'service': 'unsupported',
            'path': '/metrics/json',
            'key': 'growth_rate',
        }

        result = _gather_single_metric('http://localhost:8090', 'growth_rate', recon_config)
        assert result == float('inf')


class TestHttpReconnaissanceMain:
    @patch('reconnaissance.http._gather_single_metric')
    def test_gathers_objective_metrics(self, mock_gather):
        mock_gather.return_value = 0.85

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

    @patch('reconnaissance.http._gather_single_metric')
    def test_gathers_guardrail_metrics(self, mock_gather):
        mock_gather.return_value = 38.5

        context = {
            'reconnaissance': {
                'http': {'url': 'http://greenhouse:8090'}
            },
            'objectives': [],
            'guardrails': [
                {
                    'name': 'max_temp',
                    'reconnaissance': {
                        'service': 'http',
                        'path': '/metrics/json',
                        'key': 'max_temp',
                    }
                }
            ]
        }

        result = recon_main(context, [], {})
        assert result['metrics']['max_temp'] == 38.5

    @patch('reconnaissance.http._gather_single_metric')
    def test_per_objective_url_override(self, mock_gather):
        mock_gather.return_value = 10.0

        context = {
            'reconnaissance': {
                'http': {'url': 'http://default:8090'}
            },
            'objectives': [
                {
                    'name': 'growth_rate',
                    'reconnaissance': {
                        'service': 'http',
                        'url': 'http://custom:8090',
                        'path': '/metrics/json',
                        'key': 'growth_rate',
                    }
                }
            ],
            'guardrails': []
        }

        recon_main(context, [], {})
        mock_gather.assert_called_once_with('http://custom:8090', 'growth_rate', context['objectives'][0]['reconnaissance'])

    def test_empty_objectives_and_guardrails(self):
        context = {
            'reconnaissance': {'http': {'url': 'http://localhost:8090'}},
            'objectives': [],
            'guardrails': []
        }

        result = recon_main(context, [], {})
        assert result['status'] == 'completed'
        assert result['metrics'] == {}

    @patch('reconnaissance.http._gather_single_metric')
    def test_multiple_objectives(self, mock_gather):
        mock_gather.side_effect = [0.9, 12.5, 3.2]

        context = {
            'reconnaissance': {
                'http': {'url': 'http://greenhouse:8090'}
            },
            'objectives': [
                {'name': 'growth_rate', 'reconnaissance': {'service': 'http', 'path': '/m', 'key': 'growth_rate'}},
                {'name': 'trial_energy_kwh', 'reconnaissance': {'service': 'http', 'path': '/m', 'key': 'trial_energy_kwh'}},
                {'name': 'trial_water_liters', 'reconnaissance': {'service': 'http', 'path': '/m', 'key': 'trial_water_liters'}},
            ],
            'guardrails': []
        }

        result = recon_main(context, [], {})
        assert result['metrics']['growth_rate'] == 0.9
        assert result['metrics']['trial_energy_kwh'] == 12.5
        assert result['metrics']['trial_water_liters'] == 3.2
