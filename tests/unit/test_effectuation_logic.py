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
# but WITHOUT ANY WARRANTY; without even the implied warranty
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this godon.  If not, see <http://www.gnu.org/licenses/>.
#

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import effectuation.ssh as ssh_module
import effectuation.http as http_module


class TestSSHEffectuation:
    def test_single_target_success(self):
        with patch.object(ssh_module, 'wmill') as mock_wmill:
            mock_wmill.run_script_by_path.return_value = {'success': True}

            context = {
                'effectuation': {
                    'type': 'ssh',
                    'playbook_path': 'f/test/playbook',
                }
            }
            targets = [
                {'id': 't1', 'address': '10.0.0.1', 'username': 'admin',
                 'ssh_key_variable_path': 'u/user/key'}
            ]
            settings = {'net.ipv4.tcp_rmem': '4096 131072 174760'}

            result = ssh_module.main(context, targets, settings)

            assert result['status'] == 'completed'
            assert result['successful_changes'] == 1
            assert result['failed_changes'] == 0
            assert result['results'][0]['success'] is True

            mock_wmill.run_script_by_path.assert_called_once()
            call_args = mock_wmill.run_script_by_path.call_args
            assert call_args[0][0] == 'f/test/playbook'
            assert call_args[1]['args']['params'] == settings
            assert call_args[1]['args']['username'] == 'admin'

    def test_multiple_targets_mixed_results(self):
        with patch.object(ssh_module, 'wmill') as mock_wmill:
            mock_wmill.run_script_by_path.side_effect = [
                {'success': True},
                {'success': False, 'error': 'Connection refused'},
                {'success': True},
            ]

            context = {'effectuation': {'type': 'ssh'}}
            targets = [
                {'id': 't1', 'address': '10.0.0.1', 'ssh_key_variable_path': 'u/key'},
                {'id': 't2', 'address': '10.0.0.2', 'ssh_key_variable_path': 'u/key'},
                {'id': 't3', 'address': '10.0.0.3', 'ssh_key_variable_path': 'u/key'},
            ]
            settings = {'vm.swappiness': 10}

            result = ssh_module.main(context, targets, settings)

            assert result['successful_changes'] == 2
            assert result['failed_changes'] == 1
            assert result['results'][1]['success'] is False
            assert 'Connection refused' in result['results'][1]['error']

    def test_target_exception_handled(self):
        with patch.object(ssh_module, 'wmill') as mock_wmill:
            mock_wmill.run_script_by_path.side_effect = Exception('SSH timeout')

            context = {'effectuation': {'type': 'ssh'}}
            targets = [
                {'id': 't1', 'address': '10.0.0.1', 'ssh_key_variable_path': 'u/key'},
            ]
            settings = {'net.core.somaxconn': 4096}

            result = ssh_module.main(context, targets, settings)

            assert result['successful_changes'] == 0
            assert result['failed_changes'] == 1
            assert 'SSH timeout' in result['results'][0]['error']

    def test_default_username_is_root(self):
        with patch.object(ssh_module, 'wmill') as mock_wmill:
            mock_wmill.run_script_by_path.return_value = {'success': True}

            context = {'effectuation': {'type': 'ssh'}}
            targets = [
                {'id': 't1', 'address': '10.0.0.1', 'ssh_key_variable_path': 'u/key'},
            ]
            settings = {}

            ssh_module.main(context, targets, settings)

            call_args = mock_wmill.run_script_by_path.call_args
            assert call_args[1]['args']['username'] == 'root'

    def test_stabilization_wait(self):
        with patch.object(ssh_module, 'wmill') as mock_wmill, \
             patch('effectuation.ssh.time') as mock_time:
            mock_wmill.run_script_by_path.return_value = {'success': True}

            context = {
                'effectuation': {
                    'type': 'ssh',
                    'stabilization_seconds': 30,
                }
            }
            targets = [{'id': 't1', 'address': '10.0.0.1', 'ssh_key_variable_path': 'u/key'}]
            settings = {}

            ssh_module.main(context, targets, settings)

            mock_time.sleep.assert_called_with(30)

    def test_no_stabilization_by_default(self):
        with patch.object(ssh_module, 'wmill') as mock_wmill, \
             patch('effectuation.ssh.time') as mock_time:
            mock_wmill.run_script_by_path.return_value = {'success': True}

            context = {'effectuation': {'type': 'ssh'}}
            targets = [{'id': 't1', 'address': '10.0.0.1', 'ssh_key_variable_path': 'u/key'}]
            settings = {}

            ssh_module.main(context, targets, settings)

            mock_time.sleep.assert_not_called()

    def test_default_playbook_path(self):
        with patch.object(ssh_module, 'wmill') as mock_wmill:
            mock_wmill.run_script_by_path.return_value = {'success': True}

            context = {'effectuation': {'type': 'ssh'}}
            targets = [{'id': 't1', 'address': '10.0.0.1', 'ssh_key_variable_path': 'u/key'}]
            settings = {}

            ssh_module.main(context, targets, settings)

            call_args = mock_wmill.run_script_by_path.call_args
            assert call_args[0][0] == 'f/breeder/strains/linux_performance/effectuate_settings'


class TestHTTPEffectuation:
    def test_single_target_success(self):
        with patch.object(http_module, 'wmill') as mock_wmill, \
             patch.object(http_module, 'requests') as mock_requests:
            mock_wmill.get_variable.return_value = 'test-token'
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_response.json.return_value = {'applied': True}
            mock_response.content = b'{"applied": true}'
            mock_requests.request.return_value = mock_response

            context = {
                'effectuation': {
                    'type': 'http',
                    'endpoint_config': {
                        'path': '/api/v1/config',
                        'method': 'POST',
                    }
                }
            }
            targets = [
                {'id': 't1', 'url': 'https://api.example.com',
                 'auth_type': 'bearer', 'auth_variable_path': 'u/auth/token'}
            ]
            settings = {'tcp_rmem': '4096 131072 174760'}

            result = http_module.main(context, targets, settings)

            assert result['status'] == 'completed'
            assert result['successful_changes'] == 1
            assert result['results'][0]['success'] is True
            assert result['results'][0]['status_code'] == 200

            req_call = mock_requests.request.call_args
            assert req_call[1]['url'] == 'https://api.example.com/api/v1/config'
            assert req_call[1]['method'] == 'POST'
            assert req_call[1]['json'] == settings

    def test_target_http_failure(self):
        with patch.object(http_module, 'wmill') as mock_wmill, \
             patch.object(http_module, 'requests') as mock_requests:
            mock_response = MagicMock()
            mock_response.ok = False
            mock_response.status_code = 503
            mock_response.text = 'Service Unavailable'
            mock_requests.request.return_value = mock_response

            context = {
                'effectuation': {
                    'type': 'http',
                    'endpoint_config': {'path': '/api/config', 'method': 'PUT'}
                }
            }
            targets = [{'id': 't1', 'url': 'https://api.example.com'}]
            settings = {'cpu_governor': 'performance'}

            result = http_module.main(context, targets, settings)

            assert result['successful_changes'] == 0
            assert result['results'][0]['success'] is False
            assert result['results'][0]['status_code'] == 503


class TestContractCompliance:
    def test_ssh_accepts_standard_signature(self):
        with patch.object(ssh_module, 'wmill') as mock_wmill:
            mock_wmill.run_script_by_path.return_value = {'success': True}

            context = {'effectuation': {'type': 'ssh'}}
            targets = [{'id': 't1', 'address': '10.0.0.1', 'ssh_key_variable_path': 'u/key'}]
            settings = {'param': 'value'}

            result = ssh_module.main(context, targets, settings)

            assert 'status' in result
            assert 'successful_changes' in result
            assert 'failed_changes' in result
            assert 'results' in result

    def test_http_accepts_standard_signature(self):
        with patch.object(http_module, 'wmill'), \
             patch.object(http_module, 'requests') as mock_requests:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_response.content = None
            mock_requests.request.return_value = mock_response

            context = {
                'effectuation': {
                    'type': 'http',
                    'endpoint_config': {'path': '/api/config', 'method': 'POST'}
                }
            }
            targets = [{'id': 't1', 'url': 'https://api.example.com'}]
            settings = {'param': 'value'}

            result = http_module.main(context, targets, settings)

            assert 'status' in result
            assert 'successful_changes' in result
            assert 'failed_changes' in result
            assert 'results' in result
