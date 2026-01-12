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

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from reconnaissance.prometheus import extract_scalar_value, aggregate_samples, _gather_single_metric
from unittest.mock import MagicMock, patch
import time


class TestExtractScalarValue:
    """Test Prometheus scalar value extraction logic"""
    
    def test_extract_valid_scalar(self):
        """Test extraction of valid scalar value from Prometheus response"""
        query_result = {
            'resultType': 'scalar',
            'result': [1234567890, '42.5']
        }
        
        value = extract_scalar_value(query_result)
        assert value == 42.5
    
    def test_extract_zero_value(self):
        """Test extraction of zero value"""
        query_result = {
            'resultType': 'scalar',
            'result': [1234567890, '0']
        }
        
        value = extract_scalar_value(query_result)
        assert value == 0.0
    
    def test_extract_nan_value(self):
        """Test that NaN values are properly handled and return None"""
        query_result = {
            'resultType': 'scalar',
            'result': [1234567890, 'NaN']
        }
        
        value = extract_scalar_value(query_result)
        assert value is None
    
    def test_extract_none_value(self):
        """Test that None values are properly handled"""
        query_result = {
            'resultType': 'scalar',
            'result': [1234567890, None]
        }
        
        value = extract_scalar_value(query_result)
        assert value is None
    
    def test_invalid_result_format_empty(self):
        """Test that empty result raises ValueError"""
        query_result = {
            'resultType': 'scalar',
            'result': []
        }
        
        with pytest.raises(ValueError, match="Invalid scalar result format"):
            extract_scalar_value(query_result)
    
    def test_invalid_result_format_single_element(self):
        """Test that single-element result raises ValueError"""
        query_result = {
            'resultType': 'scalar',
            'result': [1234567890]
        }
        
        with pytest.raises(ValueError, match="Invalid scalar result format"):
            extract_scalar_value(query_result)


class TestAggregateSamples:
    """Test sample aggregation logic"""
    
    def test_median_aggregation_filters_outliers(self):
        """Test that median aggregation filters out extreme values"""
        import statistics
        
        samples = [10.0, 11.0, 12.0, 100.0, 9.0]  # 100 is extreme outlier
        result = aggregate_samples(samples, method='median')
        
        assert result == 11.0  # Median of [9.0, 10.0, 11.0, 12.0, 100.0]
    
    def test_median_aggregation_even_count(self):
        """Test median aggregation with even number of samples"""
        samples = [10.0, 12.0, 14.0, 16.0]
        result = aggregate_samples(samples, method='median')
        
        # Median of even count is average of middle two
        assert result == 13.0  # (12 + 14) / 2
    
    def test_mean_aggregation(self):
        """Test mean aggregation includes all values"""
        samples = [10.0, 20.0, 30.0, 40.0]
        result = aggregate_samples(samples, method='mean')
        
        assert result == 25.0  # (10 + 20 + 30 + 40) / 4
    
    def test_min_aggregation(self):
        """Test min aggregation returns smallest value"""
        samples = [15.0, 10.0, 25.0, 5.0]
        result = aggregate_samples(samples, method='min')
        
        assert result == 5.0
    
    def test_max_aggregation(self):
        """Test max aggregation returns largest value"""
        samples = [15.0, 10.0, 25.0, 5.0]
        result = aggregate_samples(samples, method='max')
        
        assert result == 25.0
    
    def test_aggregation_filters_none_values(self):
        """Test that None values are filtered before aggregation"""
        samples = [10.0, None, 12.0, None, 11.0]
        result = aggregate_samples(samples, method='median')
        
        assert result == 11.0  # Median of [10.0, 11.0, 12.0]
    
    def test_aggregation_with_all_none_returns_inf(self):
        """Test that all-None samples return infinity"""
        samples = [None, None, None]
        result = aggregate_samples(samples, method='median')
        
        assert result == float('inf')
    
    def test_aggregation_empty_list(self):
        """Test that empty sample list returns infinity"""
        samples = []
        result = aggregate_samples(samples, method='median')
        
        assert result == float('inf')
    
    def test_unknown_aggregation_method_defaults_to_median(self):
        """Test that unknown aggregation methods default to median"""
        samples = [10.0, 15.0, 20.0]
        result = aggregate_samples(samples, method='unknown_method')
        
        # Should default to median
        assert result == 15.0

class TestGatherSingleMetric:
    """Test _gather_single_metric function"""

    @patch('reconnaissance.prometheus.prometheus_query_with_retry')
    @patch('time.sleep')
    def test_gather_metric_single_sample_no_stabilization(self, mock_sleep, mock_query):
        """Test gathering a single metric without stabilization wait"""
        # Setup mock
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

        # Verify result
        assert result == 42.5
        # Verify no stabilization sleep was called
        mock_sleep.assert_not_called()
        # Verify query was called once
        mock_query.assert_called_once()

    @patch('reconnaissance.prometheus.prometheus_query_with_retry')
    @patch('time.sleep')
    def test_gather_metric_with_stabilization_wait(self, mock_sleep, mock_query):
        """Test that stabilization_seconds triggers a wait before sampling"""
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

        # Verify stabilization sleep was called before query
        assert mock_sleep.call_count >= 1
        # First call should be for 30 seconds
        mock_sleep.assert_any_call(30)
        assert result == 100.0

    @patch('reconnaissance.prometheus.prometheus_query_with_retry')
    @patch('time.sleep')
    def test_gather_metric_multiple_samples_with_interval(self, mock_sleep, mock_query):
        """Test gathering multiple samples with interval between them"""
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

        # Verify query was called 3 times
        assert mock_query.call_count == 3
        # Verify interval sleeps (should be called between samples, not after last)
        # 3 samples = 2 intervals between them
        interval_calls = [call for call in mock_sleep.call_args_list if call[0][0] == 5]
        assert len(interval_calls) == 2
        assert result == 50.0

    @patch('reconnaissance.prometheus.prometheus_query_with_retry')
    def test_gather_metric_with_nan_samples(self, mock_query):
        """Test aggregation when some samples return NaN"""
        # Mock returns mixed valid and NaN values
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

        # Should filter out NaN and aggregate remaining values
        # Median of [100.0, 200.0] = 150.0
        assert result == 150.0

    @patch('reconnaissance.prometheus.prometheus_query_with_retry')
    def test_gather_metric_all_nan_returns_inf(self, mock_query):
        """Test that all-NaN samples return infinity"""
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

        # All NaN should return infinity
        assert result == float('inf')

    @patch('reconnaissance.prometheus.prometheus_query_with_retry')
    def test_gather_metric_query_failure(self, mock_query):
        """Test that query failures return infinity"""
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

        # Failed queries should return infinity
        assert result == float('inf')

    @patch('reconnaissance.prometheus.prometheus_query_with_retry')
    def test_gather_metric_unsupported_service(self, mock_query):
        """Test that unsupported reconnaissance service returns infinity"""
        prom_conn = MagicMock()

        recon_config = {
            'service': 'unsupported_service',
            'query': 'some query'
        }

        result = _gather_single_metric(prom_conn, 'test_metric', recon_config)

        # Unsupported service should return infinity
        assert result == float('inf')
        # Query should not be called for unsupported service
        mock_query.assert_not_called()
