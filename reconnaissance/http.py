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

import requests
import time
import statistics
from typing import Dict, Any, List
from requests.exceptions import ConnectionError, Timeout

from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)


def _http_get_with_retry(url: str, max_retries: int = 3, initial_delay: int = 2, timeout: int = 30) -> Dict[str, Any]:
    last_exception = None

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (ConnectionError, Timeout, requests.exceptions.HTTPError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                logger.warning(f"HTTP GET failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"HTTP GET failed after {max_retries} retries: {e}")
        except Exception as e:
            logger.error(f"Non-retryable HTTP error: {e}")
            raise

    raise Exception(f"HTTP GET failed after {max_retries} retries: {last_exception}")


def _aggregate_samples(samples: List[float], method: str = 'median') -> float:
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
        return statistics.median(valid_samples)


def _compute_cv(samples: List[float]) -> float:
    finite = [s for s in samples if s is not None and s != float('inf')]
    if len(finite) < 2:
        return float('inf')
    med = statistics.median(finite)
    if med == 0.0:
        mean_abs = statistics.mean([abs(v) for v in finite])
        if mean_abs == 0.0:
            return 0.0
        mad = statistics.median([abs(v - med) for v in finite])
        return mad / mean_abs
    mad = statistics.median([abs(v - med) for v in finite])
    return mad / abs(med)


def _sprt_sample_stable(samples: List[float], threshold_cv: float = 0.05) -> bool:
    if len(samples) < 3:
        return False
    cv = _compute_cv(samples)
    return cv < threshold_cv


def _gather_single_metric(base_url: str, metric_name: str, recon_config: Dict[str, Any]) -> Dict[str, Any]:
    recon_service = recon_config.get('service')

    if recon_service != 'http':
        logger.error(f"Unsupported reconnaissance service: {recon_service}")
        return {'value': float('inf'), 'noise_cv': None}

    try:
        path = recon_config.get('path', '')
        key = recon_config.get('key')
        url = f"{base_url.rstrip('/')}{path}"

        stabilization_seconds = recon_config.get('stabilization_seconds', 2)
        min_samples = recon_config.get('samples', 1)
        max_samples = recon_config.get('max_samples', min_samples)
        interval = recon_config.get('interval', 0)
        timeout = recon_config.get('timeout_seconds', 30)
        cv_threshold = recon_config.get('cv_threshold', 0.05)

        if stabilization_seconds > 0:
            logger.info(f"Waiting {stabilization_seconds}s for stabilization")
            time.sleep(stabilization_seconds)

        logger.info(f"Collecting {min_samples}-{max_samples} samples from {url} (key={key})")

        sample_values = []

        def _take_sample(idx):
            data = _http_get_with_retry(url, timeout=timeout)
            logger.info(f"[DEBUG] _take_sample data={data} type={type(data)} key={key}")
            if key not in data:
                logger.warning(f"Key '{key}' not found in response. Available keys: {list(data.keys())}")
                return None
            value = data[key]
            if value is not None:
                value = float(value)
            logger.info(f"[DEBUG] _take_sample value={value}")
            return value

        for i in range(min_samples):
            sample_values.append(_take_sample(i))
            if i < min_samples - 1 and interval > 0:
                time.sleep(interval)

        while len(sample_values) < max_samples:
            if _sprt_sample_stable(sample_values, cv_threshold):
                logger.info(f"Sample stability reached at {len(sample_values)} samples (CV < {cv_threshold})")
                break
            sample_values.append(_take_sample(len(sample_values)))
            if interval > 0:
                time.sleep(interval)

        aggregation_method = recon_config.get('aggregation', 'median')
        final_value = _aggregate_samples(sample_values, aggregation_method)
        noise_cv = _compute_cv(sample_values)

        if final_value == float('inf'):
            logger.warning(f"All samples returned invalid values for {metric_name}")
        else:
            valid_count = len([s for s in sample_values if s is not None])
            logger.info(f"Metric {metric_name}: {final_value} (using {aggregation_method} of {valid_count} samples, CV={noise_cv:.4f})")

        return {'value': final_value, 'noise_cv': round(noise_cv, 6) if noise_cv != float('inf') else None}

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Failed to gather metric {metric_name}: {e}")
        return {'value': float('inf'), 'noise_cv': None}


def main(context: Dict[str, Any], targets: List[Dict[str, Any]], settings: Dict[str, Any] = None) -> Dict[str, Any]:
    logger.info("Starting HTTP reconnaissance")

    global_recon_config = context.get('reconnaissance', {})
    global_http_config = global_recon_config.get('http', {})
    global_url = global_http_config.get('url', 'http://localhost:8090')
    logger.info(f"Default HTTP URL: {global_url}")

    metric_data = {}
    metric_noise = {}

    for objective in context.get('objectives', []):
        objective_name = objective.get('name')
        logger.info(f"Gathering objective metric: {objective_name}")

        recon_config = objective.get('reconnaissance', {})
        base_url = recon_config.get('url', global_url)

        if recon_config.get('url'):
            logger.info(f"Using per-objective HTTP URL: {base_url}")

        result = _gather_single_metric(base_url, objective_name, recon_config)
        metric_data[objective_name] = result['value']
        if result.get('noise_cv') is not None:
            metric_noise[objective_name] = result['noise_cv']

    for guardrail in context.get('guardrails', []):
        guardrail_name = guardrail.get('name')
        logger.info(f"Gathering guardrail metric: {guardrail_name}")

        recon_config = guardrail.get('reconnaissance', {})
        base_url = recon_config.get('url', global_url)

        if recon_config.get('url'):
            logger.info(f"Using per-guardrail HTTP URL: {base_url}")

        result = _gather_single_metric(base_url, guardrail_name, recon_config)
        metric_data[guardrail_name] = result['value']
        if result.get('noise_cv') is not None:
            metric_noise[guardrail_name] = result['noise_cv']

    logger.info(f"HTTP reconnaissance completed with {len(metric_data)} metrics")

    return {
        'status': 'completed',
        'metrics': metric_data,
        'metric_noise': metric_noise,
    }
