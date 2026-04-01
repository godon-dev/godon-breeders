"""
Breeder Metrics Client

Thin wrapper around prometheus_client for Godon breeders.
Simplifies pushing metrics to Prometheus Push Gateway.

Dependencies:
    pip install prometheus_client

Usage:
    from f.breeder.engine.breeder_metrics_client import BreederMetricsClient

    metrics = BreederMetricsClient(breeder_id='abc-123', worker_id='worker_1', breeder_type='linux_performance')
    metrics.mark_running()
    metrics.inc_trial('complete', value=0.85)
    metrics.push()
"""

import os
from typing import Optional
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway

from f.breeder.shared.otel_logging import get_logger

logger = get_logger(__name__)


class BreederMetricsClient:

    def __init__(self, breeder_id: str, worker_id: str, breeder_type: str,
                 pushgateway_url: Optional[str] = None):
        self.breeder_id = breeder_id
        self.worker_id = worker_id
        self.breeder_type = breeder_type

        self.enabled = os.getenv("PUSH_METRICS_ENABLED", "true").lower() == "true"
        self.pushgateway_url = pushgateway_url or os.getenv("PUSH_GATEWAY_URL", "http://pushgateway:9091")

        if not self.enabled:
            logger.info("Prometheus metrics pushing disabled via PUSH_METRICS_ENABLED=false")
            return

        self.registry = CollectorRegistry()
        self._init_metrics()

        logger.debug(f"Initialized {self.__class__.__name__} for {breeder_id}/{worker_id}")

    def _init_metrics(self):
        self._worker_status = Gauge(
            'godon_breeder_worker_status',
            'Breeder worker running status',
            ['breeder_id', 'worker_id', 'breeder_type', 'status'],
            registry=self.registry
        )

        self._trial_count = Counter(
            'godon_breeder_trials_total',
            'Total trials executed',
            ['breeder_id', 'worker_id', 'breeder_type', 'state'],
            registry=self.registry
        )

        self._best_value = Gauge(
            'godon_breeder_best_value',
            'Best objective value achieved',
            ['breeder_id', 'worker_id', 'breeder_type'],
            registry=self.registry
        )

        self._last_trial_value = Gauge(
            'godon_breeder_last_trial_value',
            'Most recent trial value',
            ['breeder_id', 'worker_id', 'breeder_type'],
            registry=self.registry
        )

        self._total_trials = Gauge(
            'godon_breeder_total_trials',
            'Total number of trials in study',
            ['breeder_id', 'worker_id', 'breeder_type'],
            registry=self.registry
        )

        self._trial_duration = Histogram(
            'godon_breeder_trial_duration_seconds',
            'Trial execution time',
            ['breeder_id', 'worker_id', 'breeder_type'],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800],
            registry=self.registry
        )

        self._effectuation_count = Counter(
            'godon_breeder_effectuation_total',
            'Effectuation executions',
            ['breeder_id', 'worker_id', 'breeder_type', 'status'],
            registry=self.registry
        )

        self._guardrail_violations = Counter(
            'godon_breeder_guardrail_violations_total',
            'Safety guardrail violations',
            ['breeder_id', 'worker_id', 'breeder_type', 'guardrail_name'],
            registry=self.registry
        )

        self._rollback_count = Counter(
            'godon_breeder_rollbacks_total',
            'Number of rollbacks performed',
            ['breeder_id', 'worker_id', 'breeder_type', 'status'],
            registry=self.registry
        )

        self._trials_shared = Counter(
            'godon_breeder_trials_shared_total',
            'Trials shared with other breeders',
            ['breeder_id', 'worker_id', 'breeder_type', 'strategy'],
            registry=self.registry
        )

    def push(self) -> bool:
        if not self.enabled:
            return False

        try:
            push_to_gateway(
                self.pushgateway_url,
                job=f'breeder_{self.breeder_id}',
                registry=self.registry
            )
            logger.debug(f"Pushed metrics to {self.pushgateway_url}")
            return True
        except Exception as e:
            logger.warning(f"Failed to push metrics to {self.pushgateway_url}: {e}")
            return False

    def mark_running(self):
        if not self.enabled:
            return
        self._worker_status.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type,
            status='running'
        ).set(1)
        self._worker_status.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type,
            status='stopped'
        ).set(0)

    def mark_stopped(self):
        if not self.enabled:
            return
        self._worker_status.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type,
            status='running'
        ).set(0)
        self._worker_status.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type,
            status='stopped'
        ).set(1)

    def inc_trial(self, state: str, value: Optional[float] = None):
        if not self.enabled:
            return

        self._trial_count.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type,
            state=state
        ).inc()

        if value is not None:
            self._last_trial_value.labels(
                breeder_id=self.breeder_id,
                worker_id=self.worker_id,
                breeder_type=self.breeder_type
            ).set(value)

    def set_best_value(self, value: float):
        if not self.enabled:
            return
        self._best_value.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type
        ).set(value)

    def set_total_trials(self, count: int):
        if not self.enabled:
            return
        self._total_trials.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type
        ).set(count)

    def observe_trial_duration(self, duration_seconds: float):
        if not self.enabled:
            return
        self._trial_duration.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type
        ).observe(duration_seconds)

    def inc_effectuation(self, status: str):
        if not self.enabled:
            return
        self._effectuation_count.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type,
            status=status
        ).inc()

    def inc_guardrail_violation(self, guardrail_name: str):
        if not self.enabled:
            return
        self._guardrail_violations.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type,
            guardrail_name=guardrail_name
        ).inc()

    def inc_rollback(self, status: str):
        if not self.enabled:
            return
        self._rollback_count.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type,
            status=status
        ).inc()

    def inc_trial_shared(self, strategy: str):
        if not self.enabled:
            return
        self._trials_shared.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type,
            strategy=strategy
        ).inc()
