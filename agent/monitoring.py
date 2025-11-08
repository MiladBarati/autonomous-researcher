"""
Performance Monitoring Module for Autonomous Research Assistant

Provides performance monitoring capabilities:
- Execution time tracking
- Memory usage monitoring
- Resource utilization tracking
- Performance decorators and context managers
"""

import functools
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

from agent.logger import get_logger
from agent.metrics import get_metrics

logger = get_logger("monitoring")

F = TypeVar("F", bound=Callable[..., Any])


def get_memory_usage() -> dict[str, float]:
    """
    Get current memory usage in MB.

    Returns:
        Dictionary with memory usage statistics
    """
    try:
        # Try psutil first (works on all platforms)
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Bytes to MB
    except ImportError:
        # Fallback to resource module (Linux/Mac only, not available on Windows)
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            memory_mb = usage.ru_maxrss / 1024.0  # Convert KB to MB (Linux/Mac)
        except (ImportError, AttributeError):
            # If both fail (e.g., on Windows), return 0
            memory_mb = 0.0

    return {
        "memory_mb": memory_mb,
        "memory_gb": memory_mb / 1024.0,
    }


@contextmanager
def performance_monitor(operation_name: str, log_level: str = "info"):
    """
    Context manager for monitoring operation performance.

    Args:
        operation_name: Name of the operation being monitored
        log_level: Logging level ('info', 'debug', 'warning')

    Example:
        with performance_monitor("web_search"):
            # operation code
    """
    metrics = get_metrics()
    start_time = time.time()
    start_memory = get_memory_usage()

    try:
        yield
    finally:
        duration = time.time() - start_time
        end_memory = get_memory_usage()
        memory_delta = end_memory["memory_mb"] - start_memory["memory_mb"]

        # Record metrics
        metrics.record_histogram(f"{operation_name}_duration_seconds", duration)
        metrics.record_histogram(f"{operation_name}_memory_delta_mb", memory_delta)
        metrics.set_gauge(f"{operation_name}_last_duration_seconds", duration)
        metrics.set_gauge(f"{operation_name}_last_memory_mb", end_memory["memory_mb"])

        # Log performance
        log_message = (
            f"Performance: {operation_name} - "
            f"Duration: {duration:.3f}s, "
            f"Memory: {end_memory['memory_mb']:.2f}MB "
            f"(Δ{memory_delta:+.2f}MB)"
        )

        if log_level == "debug":
            logger.debug(log_message)
        elif log_level == "warning":
            logger.warning(log_message)
        else:
            logger.info(log_message)


def monitor_performance(operation_name: str | None = None, log_level: str = "info"):
    """
    Decorator for monitoring function performance.

    Args:
        operation_name: Optional operation name (defaults to function name)
        log_level: Logging level ('info', 'debug', 'warning')

    Example:
        @monitor_performance("search_web")
        def search_web(query: str):
            # function code
    """

    def decorator(func: F) -> F:
        name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with performance_monitor(name, log_level):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


class PerformanceTracker:
    """Class for tracking performance across multiple operations"""

    def __init__(self, operation_name: str):
        """
        Initialize performance tracker.

        Args:
            operation_name: Name of the operation being tracked
        """
        self.operation_name = operation_name
        self.start_time: float | None = None
        self.start_memory: dict[str, float] | None = None
        self.checkpoints: list[dict[str, Any]] = []

    def start(self) -> None:
        """Start tracking performance"""
        self.start_time = time.time()
        self.start_memory = get_memory_usage()
        self.checkpoints = []
        logger.debug(f"Started performance tracking for {self.operation_name}")

    def checkpoint(self, checkpoint_name: str) -> None:
        """
        Record a checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint
        """
        if self.start_time is None:
            logger.warning("Performance tracker not started, call start() first")
            return

        current_time = time.time()
        elapsed = current_time - self.start_time
        current_memory = get_memory_usage()

        checkpoint_data = {
            "name": checkpoint_name,
            "elapsed_seconds": elapsed,
            "memory_mb": current_memory["memory_mb"],
            "timestamp": time.time(),
        }

        self.checkpoints.append(checkpoint_data)
        logger.debug(
            f"Checkpoint {checkpoint_name}: {elapsed:.3f}s, "
            f"Memory: {current_memory['memory_mb']:.2f}MB"
        )

    def finish(self) -> dict[str, Any]:
        """
        Finish tracking and return performance summary.

        Returns:
            Dictionary with performance summary
        """
        if self.start_time is None:
            logger.warning("Performance tracker not started")
            return {}

        total_duration = time.time() - self.start_time
        end_memory = get_memory_usage()
        start_memory_mb = self.start_memory["memory_mb"] if self.start_memory else 0.0
        memory_delta = end_memory["memory_mb"] - start_memory_mb

        # Record metrics
        metrics = get_metrics()
        metrics.record_histogram(f"{self.operation_name}_total_duration_seconds", total_duration)
        metrics.set_gauge(f"{self.operation_name}_total_memory_mb", end_memory["memory_mb"])

        summary = {
            "operation": self.operation_name,
            "total_duration_seconds": total_duration,
            "start_memory_mb": start_memory_mb,
            "end_memory_mb": end_memory["memory_mb"],
            "memory_delta_mb": memory_delta,
            "checkpoints": self.checkpoints,
            "checkpoint_count": len(self.checkpoints),
        }

        logger.info(
            f"Performance Summary: {self.operation_name} - "
            f"Total: {total_duration:.3f}s, "
            f"Memory: {end_memory['memory_mb']:.2f}MB "
            f"(Δ{memory_delta:+.2f}MB), "
            f"Checkpoints: {len(self.checkpoints)}"
        )

        return summary

    def __enter__(self) -> "PerformanceTracker":
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit"""
        self.finish()


def track_api_call(api_name: str):
    """
    Decorator for tracking API calls with performance monitoring.

    Args:
        api_name: Name of the API being called

    Example:
        @track_api_call("tavily_search")
        def search(query: str):
            # API call code
    """

    def decorator(func: F) -> F:
        metrics = get_metrics()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics.increment_counter("api_calls_total", labels={"api": api_name})
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_histogram(f"{api_name}_duration_seconds", duration)
                logger.debug(f"API call {api_name} completed in {duration:.3f}s")
                return result
            except Exception as e:
                metrics.increment_counter("api_errors_total", labels={"api": api_name})
                duration = time.time() - start_time
                metrics.record_histogram(f"{api_name}_error_duration_seconds", duration)
                logger.error(f"API call {api_name} failed after {duration:.3f}s: {e}")
                raise

        return wrapper  # type: ignore[return-value]

    return decorator
