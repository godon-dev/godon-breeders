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

from reconnaissance.prometheus import extract_scalar_value, aggregate_samples


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