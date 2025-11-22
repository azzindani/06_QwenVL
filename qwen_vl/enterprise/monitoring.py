"""Monitoring and metrics for document processing service."""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from threading import Lock


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collect and expose metrics for monitoring."""

    def __init__(self):
        """Initialize metrics collector."""
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()

    # Counters (monotonically increasing)

    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name
            value: Value to add
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def get_counter(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)

    # Gauges (can go up or down)

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Set a gauge metric.

        Args:
            name: Metric name
            value: Current value
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def get_gauge(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0.0)

    # Histograms (distributions)

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Observe a histogram value.

        Args:
            name: Metric name
            value: Observed value
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)
            # Keep last 1000 observations
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]

    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Get histogram statistics.

        Returns:
            Dict with count, sum, avg, min, max, p50, p95, p99
        """
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])

        if not values:
            return {
                "count": 0,
                "sum": 0,
                "avg": 0,
                "min": 0,
                "max": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0,
            }

        sorted_values = sorted(values)
        count = len(values)

        return {
            "count": count,
            "sum": sum(values),
            "avg": sum(values) / count,
            "min": min(values),
            "max": max(values),
            "p50": sorted_values[int(count * 0.50)],
            "p95": sorted_values[int(count * 0.95)] if count > 1 else sorted_values[0],
            "p99": sorted_values[int(count * 0.99)] if count > 1 else sorted_values[0],
        }

    def _make_key(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create metric key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    # Prometheus export

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        # Counters
        for key, value in self._counters.items():
            name = key.split("{")[0]
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{key} {value}")

        # Gauges
        for key, value in self._gauges.items():
            name = key.split("{")[0]
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{key} {value}")

        # Histograms
        for key, values in self._histograms.items():
            name = key.split("{")[0]
            if values:
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count{key[len(name):]} {len(values)}")
                lines.append(f"{name}_sum{key[len(name):]} {sum(values)}")

        return "\n".join(lines)

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics as dictionary.

        Returns:
            Dict of all metrics
        """
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                k: self.get_histogram_stats(k.split("{")[0])
                for k in self._histograms.keys()
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


class RequestTimer:
    """Context manager for timing requests."""

    def __init__(
        self,
        collector: MetricsCollector,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize request timer.

        Args:
            collector: Metrics collector
            metric_name: Histogram metric name
            labels: Optional labels
        """
        self.collector = collector
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        self.collector.observe_histogram(
            self.metric_name,
            duration_ms,
            self.labels,
        )

        # Also increment request counter
        status = "error" if exc_type else "success"
        labels = {**(self.labels or {}), "status": status}
        self.collector.increment_counter(
            f"{self.metric_name}_total",
            labels=labels,
        )


# Pre-defined metrics
def setup_default_metrics(collector: MetricsCollector) -> None:
    """Set up default application metrics."""
    # Initialize counters
    collector.increment_counter("requests_total", 0)
    collector.increment_counter("errors_total", 0)

    # Initialize gauges
    collector.set_gauge("model_loaded", 0)
    collector.set_gauge("active_jobs", 0)
    collector.set_gauge("gpu_utilization_percent", 0)
    collector.set_gauge("gpu_memory_used_mb", 0)


# Global metrics collector
_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
        setup_default_metrics(_collector)
    return _collector


if __name__ == "__main__":
    print("=" * 60)
    print("MONITORING METRICS TEST")
    print("=" * 60)

    collector = MetricsCollector()

    # Test counter
    collector.increment_counter("requests", labels={"task": "ocr"})
    collector.increment_counter("requests", labels={"task": "ocr"})
    print(f"  OCR requests: {collector.get_counter('requests', {'task': 'ocr'})}")

    # Test gauge
    collector.set_gauge("active_jobs", 5)
    print(f"  Active jobs: {collector.get_gauge('active_jobs')}")

    # Test histogram
    for ms in [100, 150, 200, 250, 300]:
        collector.observe_histogram("request_duration_ms", ms)

    stats = collector.get_histogram_stats("request_duration_ms")
    print(f"  Duration p50: {stats['p50']}ms, p95: {stats['p95']}ms")

    # Test timer
    with RequestTimer(collector, "test_request", {"endpoint": "/test"}):
        time.sleep(0.01)

    print("=" * 60)
