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

from prometheus_api_client import PrometheusConnect
import urllib3
import logging
import wmill
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def main(config: Dict[str, Any], targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Gather network metrics from Prometheus for linux_network_stack breeder
    
    Args:
        config: Breeder configuration with objectives and reconnaissance settings
        targets: List of target systems (unused for Prometheus, but kept for interface consistency)
    
    Returns:
        Dictionary with metrics for each objective (tcp_rtt, tcp_delivery_rate_bytes, etc.)
    """
    logger.info("Starting network reconnaissance via Prometheus")
    
    # Get Prometheus URL from Windmill resources
    prometheus_url = wmill.get_resource('u/user/prometheus')
    logger.info(f"Connecting to Prometheus: {prometheus_url}")
    
    # Create Prometheus connection
    prom_conn = PrometheusConnect(
        url=prometheus_url,
        retry=urllib3.util.retry.Retry(total=3, raise_on_status=True, backoff_factor=0.5),
        disable_ssl=True
    )
    
    metric_data = {}
    
    for objective in config.get('objectives', []):
        objective_name = objective.get('name')
        logger.info(f"Gathering metric: {objective_name}")
        
        recon_config = objective.get('reconaissance', {})
        recon_service = recon_config.get('service')
        
        if recon_service == 'prometheus':
            try:
                recon_query = recon_config.get('query')
                logger.debug(f"Executing Prometheus query: {recon_query}")
                
                query_result = prom_conn.custom_query(recon_query)
                
                if query_result.get('resultType') != 'scalar':
                    raise ValueError(f"Query must return scalar result, got: {query_result.get('resultType')}")
                
                # Extract scalar value: [timestamp, value]
                result = query_result.get('result', [])
                if len(result) < 2:
                    raise ValueError(f"Invalid scalar result format: {result}")
                
                value = result[1]
                
                if value == "NaN" or value is None:
                    logger.warning(f"Query returned NaN for {objective_name}")
                    metric_data[objective_name] = float('inf')  # Penalize failed metrics
                else:
                    metric_data[objective_name] = float(value)
                    logger.info(f"Metric {objective_name}: {value}")
                    
            except Exception as e:
                logger.error(f"Failed to gather metric {objective_name}: {e}")
                metric_data[objective_name] = float('inf')  # Penalize failed reconnaissance
                
        else:
            logger.error(f"Unsupported reconnaissance service: {recon_service}")
            metric_data[objective_name] = float('inf')
    
    logger.info(f"Reconnaissance completed with {len(metric_data)} metrics")
    
    return {
        'status': 'completed',
        'metrics': metric_data
    }
