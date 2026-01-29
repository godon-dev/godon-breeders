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
import time
from typing import Dict, Any, List
import wmill

logger = logging.getLogger(__name__)


def main(targets: List[Dict[str, Any]], playbook_path: str, playbook_vars: Dict[str, Any], stabilization_seconds: int = 0) -> Dict[str, Any]:
    """
    Apply configuration to targets via SSH using Windmill Ansible playbook
    
    Args:
        targets: List of target configurations with:
            - id: target identifier
            - hostname: target hostname/IP
            - username: SSH username
            - ssh_key_variable_path: Windmill variable path to SSH key
        playbook_path: Windmill path to the Ansible playbook to execute
        playbook_vars: Variables to pass to the Ansible playbook
        stabilization_seconds: Optional wait time after applying changes
    
    Returns:
        Dictionary with aggregated results and success status
    """
    logger.info(f"Starting SSH effectuation for {len(targets)} targets")
    logger.info(f"Playbook: {playbook_path}")
    logger.info(f"Variables: {list(playbook_vars.keys())}")
    
    all_results = []
    
    for target in targets:
        target_id = target.get('id')
        address = target.get('address')
        username = target.get('username', 'root')
        ssh_key_path = target.get('ssh_key_variable_path')
        
        logger.info(f"Processing target {target_id}: {address}")
        
        try:
            # Build variables for this target
            # Note: playbook_vars contains all configuration parameters (e.g., qdisc, cpu_governor)
            # The Ansible playbook expects them wrapped in a 'params' dict, not as separate kwargs
            target_vars = {
                'target_hostname': address,
                'username': username,
                'ssh_key_variable': ssh_key_path,
                'params': playbook_vars  # Wrap configuration parameters in 'params' dict
            }

            # Execute Ansible playbook via Windmill (playbooks are scripts, not flows)
            # Pass arguments via args parameter, not as kwargs
            result = wmill.run_script_by_path(playbook_path, args=target_vars)
            
            if result.get('success'):
                all_results.append({
                    'target_id': target_id,
                    'address': address,
                    'success': True,
                    'result': result
                })
            else:
                all_results.append({
                    'target_id': target_id,
                    'address': address,
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                })
                
        except Exception as e:
            logger.error(f"Failed to process target {target_id}: {e}", exc_info=True)
            all_results.append({
                'target_id': target_id,
                'address': address,
                'success': False,
                'error': f"Target processing failed: {str(e)}"
            })
    
    # Optional stabilization wait
    if stabilization_seconds > 0:
        logger.info(f"Waiting {stabilization_seconds}s for system stabilization")
        time.sleep(stabilization_seconds)
    
    # Aggregate results
    success_count = sum(1 for r in all_results if r.get('success', False))
    total_count = len(all_results)
    
    summary = {
        'status': 'completed',
        'targets_count': len(targets),
        'successful_changes': success_count,
        'failed_changes': total_count - success_count,
        'results': all_results
    }
    
    logger.info(f"Effectuation completed: {success_count}/{total_count} successful")
    
    return summary