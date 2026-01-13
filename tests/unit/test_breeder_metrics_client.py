"""
Unit tests for BreederMetricsClient

Tests the Prometheus metrics wrapper without requiring actual
Prometheus Push Gateway or Windmill infrastructure.
"""

import sys
import os
from unittest.mock import MagicMock, patch, call
import pytest

# Mock prometheus_client before importing the code under test
sys.modules['prometheus_client'] = MagicMock()

from linux_performance.breeder_metrics_client import BreederMetricsClient


class TestBreederMetricsClientInitialization:
    """Test BreederMetricsClient initialization and configuration"""

    @patch('linux_performance.breeder_metrics_client.push_to_gateway')
    @patch('linux_performance.breeder_metrics_client.CollectorRegistry')
    def test_initialization_enabled(self, mock_registry, mock_push):
        """Test client initializes correctly when enabled"""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        client = BreederMetricsClient(
            breeder_id='test-breeder-123',
            worker_id='test-worker-1',
            breeder_type='linux_performance',
            pushgateway_url='http://test-pushgateway:9091'
        )

        assert client.breeder_id == 'test-breeder-123'
        assert client.worker_id == 'test-worker-1'
        assert client.breeder_type == 'linux_performance'
        assert client.pushgateway_url == 'http://test-pushgateway:9091'
        assert client.enabled is True

        # Verify registry was created
        mock_registry.assert_called_once()

    @patch.dict(os.environ, {'PUSH_METRICS_ENABLED': 'false'})
    @patch('linux_performance.breeder_metrics_client.CollectorRegistry')
    def test_initialization_disabled(self, mock_registry):
        """Test client respects PUSH_METRICS_ENABLED=false"""
        import os

        client = BreederMetricsClient(
            breeder_id='test-breeder-123',
            worker_id='test-worker-1',
            breeder_type='linux_performance'
        )

        assert client.enabled is False

        # Registry should not be created when disabled
        mock_registry.assert_not_called()


class TestMetricCreation:
    """Test that Prometheus metrics are created with correct configuration"""

    @patch('linux_performance.breeder_metrics_client.push_to_gateway')
    @patch('linux_performance.breeder_metrics_client.CollectorRegistry')
    @patch('linux_performance.breeder_metrics_client.Gauge')
    @patch('linux_performance.breeder_metrics_client.Counter')
    @patch('linux_performance.breeder_metrics_client.Histogram')
    def test_worker_status_metric_created(self, mock_histogram, mock_counter, mock_gauge, mock_registry, mock_push):
        """Test worker status Gauge metric is created"""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_gauge_instance = MagicMock()
        mock_gauge.return_value = mock_gauge_instance

        client = BreederMetricsClient(
            breeder_id='test-breeder-123',
            worker_id='test-worker-1',
            breeder_type='linux_performance'
        )

        # Verify Gauge was created for worker status
        assert mock_gauge.call_count >= 1  # At least one Gauge created

        # Check that worker status gauge was created with correct labels
        gauge_calls = [str(call) for call in mock_gauge.call_args_list]
        assert any('godon_breeder_worker_status' in str(call) for call in gauge_calls)

    @patch('linux_performance.breeder_metrics_client.push_to_gateway')
    @patch('linux_performance.breeder_metrics_client.CollectorRegistry')
    @patch('linux_performance.breeder_metrics_client.Counter')
    def test_trial_counter_metric_created(self, mock_counter, mock_registry, mock_push):
        """Test trial counter is created"""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        client = BreederMetricsClient(
            breeder_id='test-breeder-123',
            worker_id='test-worker-1',
            breeder_type='linux_performance'
        )

        # Verify Counter was created
        counter_calls = [str(call) for call in mock_counter.call_args_list]
        assert any('godon_breeder_trials_total' in str(call) for call in counter_calls)


class TestMetricMethods:
    """Test that metric methods correctly update metrics"""

    @patch('linux_performance.breeder_metrics_client.push_to_gateway')
    @patch('linux_performance.breeder_metrics_client.CollectorRegistry')
    @patch('linux_performance.breeder_metrics_client.Counter')
    def test_inc_trial(self, mock_counter, mock_registry, mock_push):
        """Test inc_trial increments counter"""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_counter_instance = MagicMock()
        mock_counter.return_value = mock_counter_instance

        client = BreederMetricsClient(
            breeder_id='test-breeder-123',
            worker_id='test-worker-1',
            breeder_type='linux_performance'
        )

        # Call inc_trial
        client.inc_trial('complete', value=0.85)

        # Verify counter was incremented
        mock_counter_instance.labels.assert_called_once()
        mock_counter_instance.labels.return_value.inc.assert_called_once()

    @patch('linux_performance.breeder_metrics_client.push_to_gateway')
    @patch('linux_performance.breeder_metrics_client.CollectorRegistry')
    @patch('linux_performance.breeder_metrics_client.Gauge')
    def test_set_best_value(self, mock_gauge, mock_registry, mock_push):
        """Test set_best_value updates gauge"""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_gauge_instance = MagicMock()
        mock_gauge.return_value = mock_gauge_instance

        client = BreederMetricsClient(
            breeder_id='test-breeder-123',
            worker_id='test-worker-1',
            breeder_type='linux_performance'
        )

        # Set best value
        client.set_best_value(0.123)

        # Verify gauge was set
        mock_gauge_instance.labels.assert_called_once()
        mock_gauge_instance.labels.return_value.set.assert_called_with(0.123)

    @patch('linux_performance.breeder_metrics_client.push_to_gateway')
    @patch('linux_performance.breeder_metrics_client.CollectorRegistry')
    @patch('linux_performance.breeder_metrics_client.Gauge')
    def test_mark_running_stopped(self, mock_gauge, mock_registry, mock_push):
        """Test mark_running and mark_stopped"""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_gauge_instance = MagicMock()
        mock_gauge.return_value = mock_gauge_instance

        client = BreederMetricsClient(
            breeder_id='test-breeder-123',
            worker_id='test-worker-1',
            breeder_type='linux_performance'
        )

        # Mark as running
        client.mark_running()

        # Verify running=1, stopped=0
        assert mock_gauge_instance.labels.call_count == 2  # running and stopped

        # Mark as stopped
        client.mark_stopped()

        # Should have 4 more calls (running=0, stopped=1)
        assert mock_gauge_instance.labels.call_count == 4


class TestPushToGateway:
    """Test that metrics are pushed to Push Gateway correctly"""

    @patch('linux_performance.breeder_metrics_client.push_to_gateway')
    @patch('linux_performance.breeder_metrics_client.CollectorRegistry')
    def test_push_calls_pushgateway(self, mock_registry, mock_push):
        """Test push() calls prometheus push_to_gateway"""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        client = BreederMetricsClient(
            breeder_id='test-breeder-123',
            worker_id='test-worker-1',
            breeder_type='linux_performance',
            pushgateway_url='http://test-pushgateway:9091'
        )

        # Push metrics
        result = client.push()

        # Verify push_to_gateway was called
        mock_push.assert_called_once_with(
            'http://test-pushgateway:9091',
            job='breeder_test-breeder-123',
            registry=mock_registry_instance
        )
        assert result is True

    @patch('linux_performance.breeder_metrics_client.push_to_gateway')
    @patch('linux_performance.breeder_metrics_client.CollectorRegistry')
    def test_push_disabled(self, mock_registry, mock_push):
        """Test push() does nothing when disabled"""
        # Create client with PUSH_METRICS_ENABLED=false via env
        import os
        with patch.dict(os.environ, {'PUSH_METRICS_ENABLED': 'false'}):
            client = BreederMetricsClient(
                breeder_id='test-breeder-123',
                worker_id='test-worker-1',
                breeder_type='linux_performance'
            )

            # Push metrics
            result = client.push()

            # Verify push_to_gateway was NOT called
            mock_push.assert_not_called()
            assert result is False

    @patch('linux_performance.breeder_metrics_client.push_to_gateway')
    @patch('linux_performance.breeder_metrics_client.CollectorRegistry')
    def test_push_handles_errors(self, mock_registry, mock_push):
        """Test push() handles Push Gateway errors gracefully"""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        # Simulate Push Gateway error
        mock_push.side_effect = Exception("Connection refused")

        client = BreederMetricsClient(
            breeder_id='test-breeder-123',
            worker_id='test-worker-1',
            breeder_type='linux_performance'
        )

        # Push should not raise exception
        result = client.push()

        # Should return False on error
        assert result is False


class TestRollbackCounter:
    """Test rollback counter functionality"""

    @patch('linux_performance.breeder_metrics_client.push_to_gateway')
    @patch('linux_performance.breeder_metrics_client.CollectorRegistry')
    @patch('linux_performance.breeder_metrics_client.Counter')
    def test_inc_rollback_success(self, mock_counter, mock_registry, mock_push):
        """Test rollback counter increments for successful rollback"""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_counter_instance = MagicMock()
        mock_counter.return_value = mock_counter_instance

        client = BreederMetricsClient(
            breeder_id='test-breeder-123',
            worker_id='test-worker-1',
            breeder_type='linux_performance'
        )

        # Increment rollback success
        client.inc_rollback('success')

        # Verify counter was incremented
        mock_counter_instance.labels.assert_called_once()
        labels_call = mock_counter_instance.labels.call_args
        assert 'success' in str(labels_call)

    @patch('linux_performance.breeder_metrics_client.push_to_gateway')
    @patch('linux_performance.breeder_metrics_client.CollectorRegistry')
    @patch('linux_performance.breeder_metrics_client.Counter')
    def test_inc_rollback_failed(self, mock_counter, mock_registry, mock_push):
        """Test rollback counter increments for failed rollback"""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_counter_instance = MagicMock()
        mock_counter.return_value = mock_counter_instance

        client = BreederMetricsClient(
            breeder_id='test-breeder-123',
            worker_id='test-worker-1',
            breeder_type='linux_performance'
        )

        # Increment rollback failed
        client.inc_rollback('failed')

        # Verify counter was incremented
        mock_counter_instance.labels.assert_called_once()
        labels_call = mock_counter_instance.labels.call_args
        assert 'failed' in str(labels_call)


class TestMetricLabels:
    """Test that metrics have correct labels"""

    @patch('linux_performance.breeder_metrics_client.push_to_gateway')
    @patch('linux_performance.breeder_metrics_client.CollectorRegistry')
    @patch('linux_performance.breeder_metrics_client.Gauge')
    def test_metrics_include_common_labels(self, mock_gauge, mock_registry, mock_push):
        """Test all metrics include breeder_id, worker_id, breeder_type labels"""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_gauge_instance = MagicMock()
        mock_gauge.return_value = mock_gauge_instance

        client = BreederMetricsClient(
            breeder_id='test-breeder-abc',
            worker_id='test-worker-xyz',
            breeder_type='linux_performance'
        )

        # Set best value (triggers labels())
        client.set_best_value(0.5)

        # Verify labels include breeder_id, worker_id, breeder_type
        labels_call = mock_gauge_instance.labels.call_args
        args, kwargs = labels_call

        # Check keyword arguments (label names and values)
        assert 'breeder_id' in kwargs
        assert kwargs['breeder_id'] == 'test-breeder-abc'
        assert 'worker_id' in kwargs
        assert kwargs['worker_id'] == 'test-worker-xyz'
        assert 'breeder_type' in kwargs
        assert kwargs['breeder_type'] == 'linux_performance'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
