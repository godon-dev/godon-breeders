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
import time
import statistics
from typing import Dict, Any, List, Optional
from requests.exceptions import ConnectionError, Timeout
from prometheus_api_client.exceptions import PrometheusError

logger = logging.getLogger(__name__)


def extract_scalar_value(query_result: Dict[str, Any]) -> Optional[float]:
    """Extract scalar value from Prometheus query result"""
    result = query_result.get('result', [])
    if len(result) < 2:
        raise ValueError(f"Invalid scalar result format: {result}")
    
    value = result[1]
    
    if value == "NaN" or value is None:
        return None
    
    return float(value)


def aggregate_samples(samples: List[float], method: str = 'median') -> float:
    """
    Aggregate multiple samples using specified method
    
    Args:
        samples: List of sample values (may contain None)
        method: Aggregation method ('median', 'mean', 'min', 'max')
        
    Returns:
        Aggregated value, or float('inf') if no valid samples exist
    """
    valid_samples = [s for s in samples if s is not None and s != float('inf')]
    
    if not valid_samples:
        return float('inf')
    
    if method == 'median':
        return statistics.median(valid_samples)
    elif method == 'mean':
        return statistics.mean(valid_samples)
    elif method == 'min':
        return min(valid_samples)
    elif method == 'max':
        return max(valid_samples)
    else:
        # Default to median for unknown methods
        return statistics.median(valid_samples)


def prometheus_query_with_retry(prom_conn, query: str, max_retries: int = 3, initial_delay: int = 5) -> Dict[str, Any]:
    """
    Execute Prometheus query with exponential backoff retry logic
    
    Args:
        prom_conn: Prometheus connection object
        query: PromQL query string
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (will be doubled each retry)
    
    Returns:
        Query result from Prometheus
    
    Raises:
        Exception: If all retries are exhausted
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            result = prom_conn.custom_query(query)
            return result
            
        except (ConnectionError, Timeout, PrometheusError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                logger.warning(f"Query failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Query failed after {max_retries} retries: {e}")
                
        except Exception as e:
            # Non-retryable error
            logger.error(f"Non-retryable query error: {e}")
            raise
    
    # All retries exhausted
    raise Exception(f"Prometheus query failed after {max_retries} retries: {last_exception}")


def main(config: Dict[str, Any], targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Gather metrics from Prometheus for linux_performance breeder
    
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
                
                # Get stabilization and sampling configuration
                stabilization_seconds = recon_config.get('stabilization_seconds', 120)
                samples = recon_config.get('samples', 1)
                interval = recon_config.get('interval', 0)
                
                # Wait for system stabilization after parameter changes
                logger.info(f"Waiting {stabilization_seconds}s for network stabilization")
                time.sleep(stabilization_seconds)
                logger.info("Stabilization period completed")
                
                logger.info(f"Collecting {samples} samples with {interval}s interval")
                logger.debug(f"Executing Prometheus query: {recon_query}")
                
                sample_values = []
                
                for i in range(samples):
                    query_result = prometheus_query_with_retry(prom_conn, recon_query)
                    
                    if query_result.get('resultType') != 'scalar':
                        raise ValueError(f"Query must return scalar result, got: {query_result.get('resultType')}")
                    
                    value = extract_scalar_value(query_result)
                    sample_values.append(value)
                    
                    if value is not None:
                        logger.debug(f"Sample {i+1}/{samples}: {value}")
                    else:
                        logger.debug(f"Sample {i+1}/{samples}: NaN")
                    
                    # Wait between samples (but not after the last one)
                    if i < samples - 1 and interval > 0:
                        time.sleep(interval)
                
                # Get aggregation method
                aggregation_method = recon_config.get('aggregation', 'median')
                
                # Aggregate samples
                final_value = aggregate_samples(sample_values, aggregation_method)
                
                if final_value == float('inf'):
                    logger.warning(f"All samples returned NaN for {objective_name}")
                    metric_data[objective_name] = final_value
                else:
                    metric_data[objective_name] = final_value
                    logger.info(f"Metric {objective_name}: {final_value} (using {aggregation_method} of {len([s for s in sample_values if s is not None])} samples)")
                    
            except Exception as e:
                logger.error(f"Failed to gather metric {objective_name}: {e}")
                metric_data[objective_name] = float('inf')
                
        else:
            logger.error(f"Unsupported reconnaissance service: {recon_service}")
            metric_data[objective_name] = float('inf')
    
    logger.info(f"Reconnaissance completed with {len(metric_data)} metrics")
    
    return {
        'status': 'completed',
        'metrics': metric_data
    }
