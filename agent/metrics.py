"""
Metrics Module for Autonomous Research Assistant

Provides structured metrics for monitoring:
- Counters: Track occurrences (e.g., API calls, errors)
- Gauges: Track current values (e.g., active requests, queue size)
- Histograms: Track distributions (e.g., response times, document counts)
"""

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from agent.logger import get_logger

logger = get_logger("metrics")


@dataclass
class Metric:
    """Base metric data structure"""

    name: str
    value: float | int
    timestamp: str
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Thread-safe metrics collector"""

    _instance: "MetricsCollector | None" = None
    _lock: Lock = Lock()

    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._lock = Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._metric_history: list[Metric] = []
        self._max_history: int = 10000  # Keep last 10k metrics

        # Initialize default metrics
        self._initialize_default_metrics()

        self._initialized = True

    def _initialize_default_metrics(self) -> None:
        """Initialize default application metrics"""
        self._counters.update(
            {
                "research_requests_total": 0,
                "research_completed_total": 0,
                "research_failed_total": 0,
                "api_calls_total": 0,
                "api_errors_total": 0,
                "web_searches_total": 0,
                "web_scrapes_total": 0,
                "arxiv_searches_total": 0,
                "documents_processed_total": 0,
                "embeddings_created_total": 0,
                "vector_store_operations_total": 0,
            }
        )

        self._gauges.update(
            {
                "active_research_requests": 0,
                "vector_store_size": 0,
                "total_documents_stored": 0,
            }
        )

    def increment_counter(
        self, name: str, value: int = 1, labels: dict[str, str] | None = None
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name
            value: Increment value (default: 1)
            labels: Optional labels for the metric
        """
        with self._lock:
            self._counters[name] += value
            metric = Metric(
                name=name,
                value=self._counters[name],
                timestamp=datetime.utcnow().isoformat(),
                labels=labels or {},
                metadata={"type": "counter", "increment": value},
            )
            self._record_metric(metric)
            logger.debug(f"Counter {name} incremented by {value} to {self._counters[name]}")

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """
        Set a gauge metric.

        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels for the metric
        """
        with self._lock:
            self._gauges[name] = value
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.utcnow().isoformat(),
                labels=labels or {},
                metadata={"type": "gauge"},
            )
            self._record_metric(metric)
            logger.debug(f"Gauge {name} set to {value}")

    def record_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """
        Record a histogram value.

        Args:
            name: Metric name
            value: Value to record
            labels: Optional labels for the metric
        """
        with self._lock:
            self._histograms[name].append(value)
            # Keep only last 1000 values per histogram
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]

            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.utcnow().isoformat(),
                labels=labels or {},
                metadata={"type": "histogram", "count": len(self._histograms[name])},
            )
            self._record_metric(metric)
            logger.debug(f"Histogram {name} recorded value {value}")

    def _record_metric(self, metric: Metric) -> None:
        """Record metric to history"""
        self._metric_history.append(metric)
        if len(self._metric_history) > self._max_history:
            self._metric_history = self._metric_history[-self._max_history :]

    def get_counter(self, name: str) -> int:
        """Get current counter value"""
        with self._lock:
            return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        """Get current gauge value"""
        with self._lock:
            return self._gauges.get(name, 0.0)

    def get_histogram_stats(self, name: str) -> dict[str, float]:
        """
        Get histogram statistics.

        Returns:
            Dictionary with min, max, mean, median, p95, p99
        """
        with self._lock:
            values = self._histograms.get(name, [])
            if not values:
                return {
                    "count": 0,
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "median": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                }

            sorted_values = sorted(values)
            count = len(sorted_values)
            return {
                "count": count,
                "min": min(sorted_values),
                "max": max(sorted_values),
                "mean": sum(sorted_values) / count,
                "median": sorted_values[count // 2],
                "p95": sorted_values[int(count * 0.95)] if count > 0 else 0.0,
                "p99": sorted_values[int(count * 0.99)] if count > 0 else 0.0,
            }

    def get_all_metrics(self) -> dict[str, Any]:
        """
        Get all current metrics.

        Returns:
            Dictionary with counters, gauges, and histogram stats
        """
        with self._lock:
            histograms = {name: self.get_histogram_stats(name) for name in self._histograms}

            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": histograms,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def export_metrics(self, filepath: Path | str | None = None) -> None:
        """
        Export metrics to JSON file.

        Args:
            filepath: Optional file path (defaults to logs/metrics.json)
        """
        if filepath is None:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            filepath = log_dir / "metrics.json"

        filepath = Path(filepath)

        metrics_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "metrics": self.get_all_metrics(),
            "recent_history": [
                {
                    "name": m.name,
                    "value": m.value,
                    "timestamp": m.timestamp,
                    "labels": m.labels,
                    "metadata": m.metadata,
                }
                for m in self._metric_history[-1000:]  # Last 1000 metrics
            ],
        }

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2, default=str)

        logger.info(f"Metrics exported to {filepath}")

    @contextmanager
    def timer(self, metric_name: str, labels: dict[str, str] | None = None):
        """
        Context manager for timing operations.

        Args:
            metric_name: Name of the timing metric
            labels: Optional labels

        Example:
            with metrics.timer("operation_duration"):
                # operation code
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_histogram(metric_name, duration, labels)
            logger.debug(f"Operation {metric_name} took {duration:.3f}s")


# Global metrics instance
_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance"""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
