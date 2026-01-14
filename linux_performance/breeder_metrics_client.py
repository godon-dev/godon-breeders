"""
Breeder Metrics Client

Thin wrapper around prometheus_client for Godon breeders.
Simplifies pushing metrics to Prometheus Push Gateway.

Dependencies:
    pip install prometheus_client

Usage:
    from f.breeder.linux_performance.breeder_metrics_client import BreederMetricsClient

    metrics = BreederMetricsClient(breeder_id='abc-123', worker_id='worker_1', breeder_type='linux_performance')
    metrics.mark_running()
    metrics.inc_trial('complete', value=0.85)
    metrics.push()
"""

import os
import logging
from typing import Optional
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway

logger = logging.getLogger(__name__)


class BreederMetricsClient:
    """
    Prometheus metrics client for Godon breeders.

    Wraps prometheus_client to provide a simple API for breeders
    to push metrics to Prometheus Push Gateway.
    """

    def __init__(self, breeder_id: str, worker_id: str, breeder_type: str,
                 pushgateway_url: Optional[str] = None):
        """
        Initialize metrics client for a breeder worker.

        Args:
            breeder_id: Unique breeder identifier (UUID)
            worker_id: Unique worker identifier
            breeder_type: Type of breeder (e.g., 'linux_performance')
            pushgateway_url: Push Gateway URL (default from env or http://pushgateway:9091)
        """
        self.breeder_id = breeder_id
        self.worker_id = worker_id
        self.breeder_type = breeder_type

        # Check if metrics pushing is enabled
        self.enabled = os.getenv("PUSH_METRICS_ENABLED", "true").lower() == "true"
        self.pushgateway_url = pushgateway_url or os.getenv("PUSH_GATEWAY_URL", "http://pushgateway:9091")

        if not self.enabled:
            logger.info("Prometheus metrics pushing disabled via PUSH_METRICS_ENABLED=false")
            return

        # Create Prometheus registry and metrics
        self.registry = CollectorRegistry()
        self._init_metrics()

        logger.debug(f"Initialized {self.__class__.__name__} for {breeder_id}/{worker_id}")

    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        # Worker status: 1=running, 0=stopped
        self._worker_status = Gauge(
            'godon_breeder_worker_status',
            'Breeder worker running status',
            ['breeder_id', 'worker_id', 'breeder_type', 'status'],
            registry=self.registry
        )

        # Trial counters
        self._trial_count = Counter(
            'godon_breeder_trials_total',
            'Total trials executed',
            ['breeder_id', 'worker_id', 'breeder_type', 'state'],
            registry=self.registry
        )

        # Best value achieved
        self._best_value = Gauge(
            'godon_breeder_best_value',
            'Best objective value achieved',
            ['breeder_id', 'worker_id', 'breeder_type'],
            registry=self.registry
        )

        # Last trial value
        self._last_trial_value = Gauge(
            'godon_breeder_last_trial_value',
            'Most recent trial value',
            ['breeder_id', 'worker_id', 'breeder_type'],
            registry=self.registry
        )

        # Total trials in study
        self._total_trials = Gauge(
            'godon_breeder_total_trials',
            'Total number of trials in study',
            ['breeder_id', 'worker_id', 'breeder_type'],
            registry=self.registry
        )

        # Trial duration histogram
        self._trial_duration = Histogram(
            'godon_breeder_trial_duration_seconds',
            'Trial execution time',
            ['breeder_id', 'worker_id', 'breeder_type'],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800],
            registry=self.registry
        )

        # Effectuation (applying parameters) results
        self._effectuation_count = Counter(
            'godon_breeder_effectuation_total',
            'Effectuation executions',
            ['breeder_id', 'worker_id', 'breeder_type', 'status'],
            registry=self.registry
        )

        # Guardrail violations
        self._guardrail_violations = Counter(
            'godon_breeder_guardrail_violations_total',
            'Safety guardrail violations',
            ['breeder_id', 'worker_id', 'breeder_type', 'guardrail_name'],
            registry=self.registry
        )

        # Rollback counter
        self._rollback_count = Counter(
            'godon_breeder_rollbacks_total',
            'Number of rollbacks performed',
            ['breeder_id', 'worker_id', 'breeder_type', 'status'],
            registry=self.registry
        )

        # Cooperation metrics
        self._trials_shared = Counter(
            'godon_breeder_trials_shared_total',
            'Trials shared with other breeders',
            ['breeder_id', 'worker_id', 'breeder_type', 'strategy'],
            registry=self.registry
        )

    def push(self) -> bool:
        """
        Push all metrics to Push Gateway.

        Returns:
            True if push succeeded, False otherwise
        """
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

    # Status methods
    def mark_running(self):
        """Mark worker as running (call at start of run())"""
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
        """Mark worker as stopped (call at end of run())"""
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

    # Trial methods
    def inc_trial(self, state: str, value: Optional[float] = None):
        """
        Increment trial counter.

        Args:
            state: Trial state ('complete', 'failed', 'running', 'pruned')
            value: Trial value (optional, updates best/last value gauges)
        """
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
        """Set best objective value"""
        if not self.enabled:
            return
        self._best_value.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type
        ).set(value)

    def set_total_trials(self, count: int):
        """Set total number of trials in study"""
        if not self.enabled:
            return
        self._total_trials.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type
        ).set(count)

    def observe_trial_duration(self, duration_seconds: float):
        """Record trial execution duration"""
        if not self.enabled:
            return
        self._trial_duration.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type
        ).observe(duration_seconds)

    # Effectuation methods
    def inc_effectuation(self, status: str):
        """
        Increment effectuation counter.

        Args:
            status: 'success' or 'failure'
        """
        if not self.enabled:
            return
        self._effectuation_count.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type,
            status=status
        ).inc()

    # Guardrail methods
    def inc_guardrail_violation(self, guardrail_name: str):
        """Increment guardrail violation counter"""
        if not self.enabled:
            return
        self._guardrail_violations.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type,
            guardrail_name=guardrail_name
        ).inc()

    # Rollback methods
    def inc_rollback(self, status: str):
        """
        Increment rollback counter.

        Args:
            status: 'success' or 'failed'
        """
        if not self.enabled:
            return
        self._rollback_count.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type,
            status=status
        ).inc()

    # Cooperation methods
    def inc_trial_shared(self, strategy: str):
        """
        Increment shared trials counter.

        Args:
            strategy: Sharing strategy ('probabilistic', 'best', 'worst', 'extremes')
        """
        if not self.enabled:
            return
        self._trials_shared.labels(
            breeder_id=self.breeder_id,
            worker_id=self.worker_id,
            breeder_type=self.breeder_type,
            strategy=strategy
        ).inc()
