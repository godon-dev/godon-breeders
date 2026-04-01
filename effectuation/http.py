#extra_requirements:
#opentelemetry-api
#opentelemetry-sdk
#opentelemetry-exporter-otlp

#
# DRAFT - Not yet integrated with any breeder, no examples available
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
import requests
from typing import Dict, Any, List
import wmill

from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)


def main(
    context: Dict[str, Any],
    targets: List[Dict[str, Any]],
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    effectuation_config = context.get('effectuation', {})
    endpoint_config = effectuation_config.get('endpoint_config', {})
    stabilization_seconds = effectuation_config.get('stabilization_seconds', 0)

    logger.info(f"HTTP effectuation for {len(targets)} targets")
    logger.info(f"Endpoint: {endpoint_config.get('method', 'POST')} {endpoint_config.get('path', '')}")
    logger.info(f"Settings: {list(settings.keys())}")

    all_results = []

    for target in targets:
        target_id = target.get('id')
        base_url = target.get('url')
        auth_type = target.get('auth_type', 'none')
        auth_variable_path = target.get('auth_variable_path')

        logger.info(f"Processing target {target_id}: {base_url}")

        try:
            headers = dict(endpoint_config.get('headers', {}))
            headers['Content-Type'] = 'application/json'

            if auth_variable_path:
                auth_creds = wmill.get_variable(auth_variable_path)
                if auth_type == 'bearer':
                    headers['Authorization'] = f"Bearer {auth_creds}"
                elif auth_type == 'basic':
                    import base64
                    username = auth_creds.get('username', '')
                    password = auth_creds.get('password', '')
                    creds = base64.b64encode(f"{username}:{password}".encode()).decode()
                    headers['Authorization'] = f"Basic {creds}"
                elif auth_type == 'api_key':
                    key_name = auth_creds.get('header_name', 'X-API-Key')
                    headers[key_name] = auth_creds.get('api_key', '')

            full_url = f"{base_url.rstrip('/')}{endpoint_config.get('path', '')}" if base_url else endpoint_config.get('path', '')
            method = endpoint_config.get('method', 'POST').upper()
            timeout = endpoint_config.get('timeout_seconds', 30)

            response = requests.request(
                method=method,
                url=full_url,
                json=settings,
                headers=headers,
                timeout=timeout
            )

            if response.ok:
                all_results.append({
                    'target_id': target_id,
                    'url': full_url,
                    'success': True,
                    'status_code': response.status_code,
                    'response': response.json() if response.content else None
                })
            else:
                all_results.append({
                    'target_id': target_id,
                    'url': full_url,
                    'success': False,
                    'status_code': response.status_code,
                    'error': f"HTTP {response.status_code}: {response.text[:500]}"
                })

        except requests.exceptions.Timeout:
            logger.error(f"Timeout for target {target_id}")
            all_results.append({
                'target_id': target_id,
                'url': base_url,
                'success': False,
                'error': 'Request timed out'
            })
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for target {target_id}: {e}")
            all_results.append({
                'target_id': target_id,
                'url': base_url,
                'success': False,
                'error': f"Connection failed: {str(e)[:200]}"
            })
        except Exception as e:
            logger.error(f"Failed to process target {target_id}: {e}", exc_info=True)
            all_results.append({
                'target_id': target_id,
                'url': base_url,
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

    logger.info(f"HTTP effectuation completed: {success_count}/{total_count} successful")

    return summary
