
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
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from unittest.mock import MagicMock, patch

# Mock Windmill and external deps before imports
sys.modules['wmill'] = MagicMock()
sys.modules['prometheus_api_client'] = MagicMock()
sys.modules['prometheus_api_client.exceptions'] = MagicMock()
sys.modules['psycopg2'] = MagicMock()

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

import reconnaissance.prometheus as prom_mod
from reconnaissance.prometheus import extract_scalar_value, aggregate_samples, _gather_single_metric


class TestExtractScalarValue:
    def test_extract_valid_scalar(self):
        query_result = {
            'resultType': 'scalar',
            'result': [1234567890, '42.5']
        }
        value = extract_scalar_value(query_result)
        assert value == 42.5

    def test_extract_nan_value(self):
        query_result = {
            'resultType': 'scalar',
            'result': [1234567890, 'NaN']
        }
        value = extract_scalar_value(query_result)
        assert value is None

    def test_invalid_result_format_empty(self):
        query_result = {
            'resultType': 'scalar',
            'result': []
        }
        with pytest.raises(ValueError, match="Invalid scalar result format"):
            extract_scalar_value(query_result)


class TestAggregateSamples:
    def test_median_aggregation_filters_outliers(self):
        samples = [10.0, 11.0, 12.0, 100.0, 9.0]
        result = aggregate_samples(samples, method='median')
        assert result == 11.0

    def test_aggregation_filters_none_values(self):
        samples = [10.0, None, 12.0, None, 11.0]
        result = aggregate_samples(samples, method='median')
        assert result == 11.0

    def test_aggregation_with_all_none_returns_inf(self):
        samples = [None, None, None]
        result = aggregate_samples(samples, method='median')
        assert result == float('inf')


class TestGatherSingleMetric:
    @patch.object(prom_mod, 'prometheus_query_with_retry')
    @patch.object(prom_mod, 'time')
    def test_gather_metric_single_sample_no_stabilization(self, mock_time, mock_query):
        mock_query.return_value = {
            'resultType': 'scalar',
            'result': [1234567890, '42.5']
        }

        prom_conn = MagicMock()
        recon_config = {
            'service': 'prometheus',
            'query': 'rate(http_requests_total[5m])',
            'stabilization_seconds': 0,
            'samples': 1,
            'interval': 0,
            'aggregation': 'median'
        }

        result = _gather_single_metric(prom_conn, 'test_metric', recon_config)
        assert result == 42.5
        mock_query.assert_called_once()

    @patch.object(prom_mod, 'prometheus_query_with_retry')
    @patch.object(prom_mod, 'time')
    def test_gather_metric_with_stabilization_wait(self, mock_time, mock_query):
        mock_query.return_value = {
            'resultType': 'scalar',
            'result': [1234567890, '100.0']
        }

        prom_conn = MagicMock()
        recon_config = {
            'service': 'prometheus',
            'query': 'rate(cpu_usage[5m])',
            'stabilization_seconds': 30,
            'samples': 1,
            'interval': 0,
            'aggregation': 'median'
        }

        result = _gather_single_metric(prom_conn, 'cpu_metric', recon_config)
        mock_time.sleep.assert_any_call(30)
        assert result == 100.0

    @patch.object(prom_mod, 'prometheus_query_with_retry')
    @patch.object(prom_mod, 'time')
    def test_gather_metric_multiple_samples_with_interval(self, mock_time, mock_query):
        mock_query.return_value = {
            'resultType': 'scalar',
            'result': [1234567890, '50.0']
        }

        prom_conn = MagicMock()
        recon_config = {
            'service': 'prometheus',
            'query': 'rate(memory_usage[5m])',
            'stabilization_seconds': 0,
            'samples': 3,
            'interval': 5,
            'aggregation': 'median'
        }

        result = _gather_single_metric(prom_conn, 'memory_metric', recon_config)
        assert mock_query.call_count == 3
        assert result == 50.0

    @patch.object(prom_mod, 'prometheus_query_with_retry')
    def test_gather_metric_with_nan_samples(self, mock_query):
        mock_query.side_effect = [
            {'resultType': 'scalar', 'result': [1234567890, 'NaN']},
            {'resultType': 'scalar', 'result': [1234567890, '100.0']},
            {'resultType': 'scalar', 'result': [1234567890, '200.0']},
        ]

        prom_conn = MagicMock()
        recon_config = {
            'service': 'prometheus',
            'query': 'rate(metric[5m])',
            'stabilization_seconds': 0,
            'samples': 3,
            'interval': 0,
            'aggregation': 'median'
        }

        result = _gather_single_metric(prom_conn, 'test_metric', recon_config)
        assert result == 150.0  # Median of [100.0, 200.0]

    @patch.object(prom_mod, 'prometheus_query_with_retry')
    def test_gather_metric_all_nan_returns_inf(self, mock_query):
        mock_query.return_value = {
            'resultType': 'scalar',
            'result': [1234567890, 'NaN']
        }

        prom_conn = MagicMock()
        recon_config = {
            'service': 'prometheus',
            'query': 'rate(metric[5m])',
            'stabilization_seconds': 0,
            'samples': 3,
            'interval': 0,
            'aggregation': 'median'
        }

        result = _gather_single_metric(prom_conn, 'test_metric', recon_config)
        assert result == float('inf')

    @patch.object(prom_mod, 'prometheus_query_with_retry')
    def test_gather_metric_query_failure(self, mock_query):
        mock_query.side_effect = Exception("Connection error")

        prom_conn = MagicMock()
        recon_config = {
            'service': 'prometheus',
            'query': 'rate(metric[5m])',
            'stabilization_seconds': 0,
            'samples': 1,
            'interval': 0,
            'aggregation': 'median'
        }

        result = _gather_single_metric(prom_conn, 'test_metric', recon_config)
        assert result == float('inf')

    @patch.object(prom_mod, 'prometheus_query_with_retry')
    def test_gather_metric_unsupported_service(self, mock_query):
        prom_conn = MagicMock()
        recon_config = {
            'service': 'unsupported_service',
            'query': 'some query'
        }

        result = _gather_single_metric(prom_conn, 'test_metric', recon_config)
        assert result == float('inf')
        mock_query.assert_not_called()


class TestPrometheusQueryWithRetry:
    _MockPromExc = type('PrometheusApiClientException', (Exception,), {})

    def test_succeeds_on_first_attempt(self):
        from reconnaissance.prometheus import prometheus_query_with_retry

        prom_conn = MagicMock()
        expected = {'resultType': 'scalar', 'result': [1, '42.0']}
        prom_conn.custom_query.return_value = expected

        with patch.object(prom_mod, 'time'), \
             patch.object(prom_mod, 'PrometheusApiClientException', self._MockPromExc):
            result = prometheus_query_with_retry(prom_conn, 'up')

        assert result == expected
        prom_conn.custom_query.assert_called_once_with('up')

    def test_retries_on_connection_error(self):
        from reconnaissance.prometheus import prometheus_query_with_retry
        from requests.exceptions import ConnectionError

        prom_conn = MagicMock()
        prom_conn.custom_query.side_effect = [
            ConnectionError("refused"),
            {'resultType': 'scalar', 'result': [1, '42.0']},
        ]

        with patch.object(prom_mod, 'time'), \
             patch.object(prom_mod, 'PrometheusApiClientException', self._MockPromExc):
            result = prometheus_query_with_retry(prom_conn, 'up', max_retries=3, initial_delay=1)

        assert result['result'][1] == '42.0'
        assert prom_conn.custom_query.call_count == 2

    def test_raises_after_exhausted_retries(self):
        from reconnaissance.prometheus import prometheus_query_with_retry
        from requests.exceptions import ConnectionError

        prom_conn = MagicMock()
        prom_conn.custom_query.side_effect = ConnectionError("refused")

        with patch.object(prom_mod, 'time'), \
             patch.object(prom_mod, 'PrometheusApiClientException', self._MockPromExc), \
             pytest.raises(Exception, match="failed after 2 retries"):
            prometheus_query_with_retry(prom_conn, 'up', max_retries=2, initial_delay=1)


class TestReconnaissanceMain:
    def test_gathers_objective_metrics(self):
        from reconnaissance.prometheus import main as recon_main

        with patch.object(prom_mod, 'PrometheusConnect') as mock_prom_cls, \
             patch.object(prom_mod, '_gather_single_metric', return_value=42.5) as mock_gather:
            mock_prom_cls.return_value = MagicMock()

            context = {
                'reconnaissance': {
                    'prometheus': {'url': 'http://prom:9090'}
                },
                'objectives': [
                    {
                        'name': 'throughput',
                        'reconnaissance': {'service': 'prometheus', 'query': 'rate(http_total[5m])'}
                    }
                ],
                'guardrails': []
            }

            result = recon_main(context, [], {})
            assert result['status'] == 'completed'
            assert result['metrics']['throughput'] == 42.5

    def test_empty_objectives_and_guardrails(self):
        from reconnaissance.prometheus import main as recon_main

        with patch.object(prom_mod, 'PrometheusConnect') as mock_prom_cls:
            mock_prom_cls.return_value = MagicMock()

            context = {
                'reconnaissance': {
                    'prometheus': {'url': 'http://prom:9090'}
                },
                'objectives': [],
                'guardrails': []
            }

            result = recon_main(context, [], {})
            assert result['status'] == 'completed'
            assert result['metrics'] == {}
