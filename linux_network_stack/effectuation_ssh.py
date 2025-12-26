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

import tempfile
import os
import logging
from typing import Dict, Any, List
import wmill
from fabric import Connection, Config

logger = logging.getLogger(__name__)


def build_sysctl_command(sysctl_params: Dict[str, str]) -> str:
    """Build a single bash command that applies all sysctl parameters"""
    commands = []
    
    for param_name, param_value in sysctl_params.items():
        # Handle special sysctl format for tcp_rmem/tcp_wmem
        if param_name in ['net.ipv4.tcp_rmem', 'net.ipv4.tcp_wmem']:
            if isinstance(param_value, list):
                param_value = ' '.join(map(str, param_value))
            commands.append(f"sudo sysctl -w {param_name}='{param_value}'")
        else:
            commands.append(f"sudo sysctl -w {param_name}={param_value}")
    
    # Combine all commands with && so they run sequentially
    return ' && '.join(commands)


def apply_sysctl_to_target(hostname: str, username: str, ssh_key: str, sysctl_params: Dict[str, str]) -> Dict[str, Any]:
    """Apply all sysctl parameters to a single target via one SSH connection"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.key', delete=False) as key_file:
        key_file.write(ssh_key)
        key_file_path = key_file.name
    
    try:
        os.chmod(key_file_path, 0o600)
        
        # Create Fabric connection with proper config
        config = Config({
            'connect_kwargs': {
                'key_filename': key_file_path,
                'timeout': 10,
            },
            'run': {
                'hide': True,  # Capture output instead of printing
                'warn': True,  # Don't raise exceptions on command failure
            }
        })
        
        conn = Connection(
            host=hostname,
            user=username,
            config=config
        )
        
        # Build single command with all sysctl parameters
        sysctl_command = build_sysctl_command(sysctl_params)
        
        logger.info(f"Executing batched sysctl command on {hostname}")
        logger.debug(f"Command: {sysctl_command}")
        
        # Execute all parameters in one SSH call
        result = conn.run(sysctl_command, timeout=60)
        
        conn.close()
        
        # Parse results - assume success if no exception
        results = []
        for param_name, param_value in sysctl_params.items():
            results.append({
                'parameter': param_name,
                'value': str(param_value),
                'success': result.return_code == 0,
                'stdout': result.stdout.strip() if result.stdout else '',
                'stderr': result.stderr.strip() if result.stderr else ''
            })
        
        return {'success': result.return_code == 0, 'results': results}
        
    except Exception as e:
        logger.error(f"Fabric connection error to {hostname}: {e}")
        return {'success': False, 'error': str(e), 'results': []}
    
    finally:
        os.unlink(key_file_path)


def main(targets: List[Dict[str, Any]], sysctl_params: Dict[str, str]) -> Dict[str, Any]:
    """
    Apply network parameters to targets via SSH using Fabric (batched)
    
    Args:
        targets: List of target configurations with:
            - id: target identifier
            - hostname: target hostname/IP
            - username: SSH username
            - ssh_key_variable_path: Windmill variable path to SSH key
        sysctl_params: Dictionary of sysctl parameters to apply
    
    Returns:
        Dictionary with aggregated results and success status
    """
    logger.info(f"Starting batched SSH effectuation for {len(targets)} targets")
    logger.info(f"Parameters: {list(sysctl_params.keys())}")
    
    all_results = []
    
    for target in targets:
        target_id = target.get('id')
        hostname = target.get('hostname')
        username = target.get('username', 'root')
        ssh_key_path = target.get('ssh_key_variable_path')
        
        logger.info(f"Processing target {target_id}: {hostname}")
        
        try:
            # Get SSH key from Windmill secrets
            ssh_key = wmill.get_variable(ssh_key_path)
            
            # Apply all sysctl commands in one SSH connection
            target_result = apply_sysctl_to_target(hostname, username, ssh_key, sysctl_params)
            
            if target_result.get('success'):
                for result in target_result.get('results', []):
                    all_results.append({
                        'target_id': target_id,
                        'hostname': hostname,
                        **result
                    })
            else:
                all_results.append({
                    'target_id': target_id,
                    'hostname': hostname,
                    'success': False,
                    'error': target_result.get('error', 'Unknown error')
                })
                
        except Exception as e:
            logger.error(f"Failed to process target {target_id}: {e}", exc_info=True)
            all_results.append({
                'target_id': target_id,
                'hostname': hostname,
                'success': False,
                'error': f"Target processing failed: {str(e)}"
            })
    
    # Wait for network stabilization
    logger.info("Waiting for network stabilization (5 seconds)")
    import time
    time.sleep(5)
    
    # Aggregate results
    success_count = sum(1 for r in all_results if r.get('success', False))
    total_count = len(all_results)
    
    summary = {
        'status': 'completed',
        'targets_count': len(targets),
        'parameters_count': len(sysctl_params),
        'successful_changes': success_count,
        'failed_changes': total_count - success_count,
        'results': all_results
    }
    
    logger.info(f"Effectuation completed: {success_count}/{total_count} successful")
    
    return summary
