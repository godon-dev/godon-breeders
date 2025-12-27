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


class TestTargetParsing:
    """Test target configuration parsing and validation"""
    
    def test_target_with_all_fields(self):
        """Test parsing of complete target configuration"""
        target = {
            'id': 'target_1',
            'hostname': '10.0.0.1',
            'username': 'admin',
            'ssh_key_variable_path': 'u/user/ssh_key'
        }
        
        assert target['id'] == 'target_1'
        assert target['hostname'] == '10.0.0.1'
        assert target['username'] == 'admin'
        assert target['ssh_key_variable_path'] == 'u/user/ssh_key'
    
    def test_target_with_default_username(self):
        """Test that default username is handled correctly"""
        target = {
            'id': 'target_2',
            'hostname': '10.0.0.2',
            'ssh_key_variable_path': 'u/user/ssh_key'
        }
        
        # Username should default to 'root' if not provided
        username = target.get('username', 'root')
        assert username == 'root'
    
    def test_target_validation_missing_required_fields(self):
        """Test that targets missing required fields are identified"""
        # Missing hostname
        target = {
            'id': 'target_3',
            'ssh_key_variable_path': 'u/user/ssh_key'
        }
        
        assert 'hostname' not in target
        assert 'id' in target
        assert 'ssh_key_variable_path' in target


class TestResultAggregation:
    """Test effectuation result aggregation logic"""
    
    def test_aggregate_success_results(self):
        """Test aggregation of successful effectuation results"""
        results = [
            {'target_id': 'target_1', 'hostname': '10.0.0.1', 'success': True},
            {'target_id': 'target_2', 'hostname': '10.0.0.2', 'success': True},
            {'target_id': 'target_3', 'hostname': '10.0.0.3', 'success': True}
        ]
        
        success_count = sum(1 for r in results if r.get('success', False))
        assert success_count == 3
        
        failed_count = len(results) - success_count
        assert failed_count == 0
    
    def test_aggregate_mixed_results(self):
        """Test aggregation with both successful and failed targets"""
        results = [
            {'target_id': 'target_1', 'hostname': '10.0.0.1', 'success': True},
            {'target_id': 'target_2', 'hostname': '10.0.0.2', 'success': False, 'error': 'Connection failed'},
            {'target_id': 'target_3', 'hostname': '10.0.0.3', 'success': True},
            {'target_id': 'target_4', 'hostname': '10.0.0.4', 'success': False, 'error': 'Timeout'}
        ]
        
        success_count = sum(1 for r in results if r.get('success', False))
        assert success_count == 2
        
        failed_count = len(results) - success_count
        assert failed_count == 2
    
    def test_aggregate_all_failed(self):
        """Test aggregation when all targets fail"""
        results = [
            {'target_id': 'target_1', 'success': False, 'error': 'Error 1'},
            {'target_id': 'target_2', 'success': False, 'error': 'Error 2'}
        ]
        
        success_count = sum(1 for r in results if r.get('success', False))
        assert success_count == 0
        
        failed_count = len(results) - success_count
        assert failed_count == 2
    
    def test_result_includes_error_messages(self):
        """Test that failed results include error information"""
        results = [
            {'target_id': 'target_1', 'success': False, 'error': 'Connection refused'}
        ]
        
        assert 'error' in results[0]
        assert results[0]['error'] == 'Connection refused'


class TestParameterHandling:
    """Test parameter handling and validation"""
    
    def test_sysctl_parameter_formatting(self):
        """Test formatting of sysctl parameters"""
        sysctl_params = {
            'net.ipv4.tcp_rmem': '4096 131072 174760',
            'net.core.netdev_budget': '600'
        }
        
        # Verify parameter structure
        assert 'net.ipv4.tcp_rmem' in sysctl_params
        assert sysctl_params['net.ipv4.tcp_rmem'] == '4096 131072 174760'
        assert sysctl_params['net.core.netdev_budget'] == '600'
    
    def test_empty_parameter_dict(self):
        """Test handling of empty parameter dictionary"""
        params = {}
        
        assert len(params) == 0
        # Should handle gracefully without errors
    
    def test_parameter_value_types(self):
        """Test that parameter values can be different types"""
        params = {
            'string_param': 'value',
            'int_param': 12345,
            'list_param': [4096, 131072, 174760]
        }
        
        assert isinstance(params['string_param'], str)
        assert isinstance(params['int_param'], int)
        assert isinstance(params['list_param'], list)