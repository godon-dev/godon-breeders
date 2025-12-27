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

import logging
from typing import Dict, Any, List
import wmill

logger = logging.getLogger(__name__)


def main(targets: List[Dict[str, Any]], sysctl_params: Dict[str, str]) -> Dict[str, Any]:
    """
    Apply network parameters to targets via SSH using Windmill Ansible playbook
    
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
    logger.info(f"Starting SSH effectuation for {len(targets)} targets")
    logger.info(f"Parameters: {list(sysctl_params.keys())}")
    
    all_results = []
    playbook_path = "f/breeder/linux_network_stack/effectuation_ssh"
    
    for target in targets:
        target_id = target.get('id')
        hostname = target.get('hostname')
        username = target.get('username', 'root')
        ssh_key_path = target.get('ssh_key_variable_path')
        
        logger.info(f"Processing target {target_id}: {hostname}")
        
        try:
            # Execute Ansible playbook via Windmill
            result = wmill.run_flow(
                playbook_path,
                {
                    "target_hostname": hostname,
                    "username": username,
                    "ssh_key_variable": ssh_key_path,
                    "sysctl_params": sysctl_params
                }
            )
            
            if result.get('success'):
                all_results.append({
                    'target_id': target_id,
                    'hostname': hostname,
                    'success': True,
                    'applied_count': result.get('applied_count', 0),
                    'applied_params': result.get('applied_params', [])
                })
            else:
                all_results.append({
                    'target_id': target_id,
                    'hostname': hostname,
                    'success': False,
                    'error': result.get('error', 'Unknown error')
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
