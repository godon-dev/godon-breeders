#extra_requirements:
#opentelemetry-api
#opentelemetry-sdk
#opentelemetry-exporter-otlp

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

import time
from typing import Dict, Any, List
import wmill

from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)


def main(context: Dict[str, Any], targets: List[Dict[str, Any]], settings: Dict[str, Any]) -> Dict[str, Any]:
    effectuation_config = context.get('effectuation', {})
    playbook_path = effectuation_config.get('playbook_path',
        "f/breeder/strains/linux_performance/effectuate_settings")
    stabilization_seconds = effectuation_config.get('stabilization_seconds', 0)

    logger.info(f"SSH effectuation for {len(targets)} targets via playbook {playbook_path}")
    logger.info(f"Settings: {list(settings.keys())}")

    all_results = []

    for target in targets:
        target_id = target.get('id')
        address = target.get('address')
        username = target.get('username', 'root')
        ssh_key_path = target.get('ssh_key_variable_path')

        logger.info(f"Processing target {target_id}: {address}")

        try:
            target_vars = {
                'target_hostname': address,
                'username': username,
                'ssh_key_variable': ssh_key_path,
                'params': settings
            }

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

    if stabilization_seconds > 0:
        logger.info(f"Waiting {stabilization_seconds}s for system stabilization")
        time.sleep(stabilization_seconds)

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