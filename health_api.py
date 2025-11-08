"""
Health Check API for Autonomous Research Assistant

Provides health check endpoints for monitoring and observability.
Can run alongside the Streamlit application.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
except ImportError:
    print("FastAPI and uvicorn are required for health check API")
    print("Install with: pip install fastapi uvicorn")
    sys.exit(1)

from agent.logger import get_logger

# Lazy imports for metrics and monitoring to avoid dependency issues
logger = get_logger("health_api")

app = FastAPI(
    title="Autonomous Research Assistant Health API",
    description="Health check and monitoring endpoints",
    version="0.1.0",
)


def check_dependencies() -> dict[str, Any]:
    """
    Check if required dependencies are available.

    Returns:
        Dictionary with dependency status
    """
    try:
        from config import Config

        dependencies = {
            "groq_api_key": Config.GROQ_API_KEY is not None,
            "tavily_api_key": Config.TAVILY_API_KEY is not None,
            "chroma_db": Path(Config.CHROMA_PERSIST_DIR).exists(),
            "logs_directory": Path("logs").exists(),
        }

        return {
            "status": "healthy" if all(dependencies.values()) else "degraded",
            "dependencies": dependencies,
        }
    except ImportError as e:
        return {
            "status": "unhealthy",
            "dependencies": {},
            "error": f"Config import failed: {str(e)}",
        }


def check_vector_store() -> dict[str, Any]:
    """
    Check vector store status.

    Returns:
        Dictionary with vector store status
    """
    try:
        # Lazy import to avoid dependency issues at startup
        from agent.rag import RAGPipeline

        rag = RAGPipeline()
        stats = rag.get_collection_stats()
        return {
            "status": "healthy",
            "collection_name": rag.collection_name,
            "stats": stats,
        }
    except ImportError as e:
        logger.warning(f"Vector store check failed - import error: {e}")
        return {
            "status": "unhealthy",
            "error": f"Import error: {str(e)}",
            "note": "This may be due to missing dependencies. Try: pip install -e .",
        }
    except Exception as e:
        logger.warning(f"Vector store check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint"""
    return {
        "service": "Autonomous Research Assistant Health API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    """
    Basic health check endpoint.

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "autonomous-research-assistant",
    }


@app.get("/health/ready")
async def readiness() -> dict[str, Any]:
    """
    Readiness check endpoint.

    Checks if the service is ready to accept requests.

    Returns:
        Readiness status
    """
    try:
        from config import Config

        # Check configuration
        Config.validate()

        # Check dependencies
        deps = check_dependencies()

        # Check vector store
        vector_store = check_vector_store()

        is_ready = deps["status"] in ["healthy", "degraded"] and vector_store["status"] == "healthy"

        status_code = 200 if is_ready else 503

        response = {
            "status": "ready" if is_ready else "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            "dependencies": deps,
            "vector_store": vector_store,
        }

        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=response)

        return response
    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        ) from e


@app.get("/health/live")
async def liveness() -> dict[str, Any]:
    """
    Liveness check endpoint.

    Checks if the service is alive and responsive.

    Returns:
        Liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """
    Get current metrics.

    Returns:
        All current metrics (counters, gauges, histograms)
    """
    try:
        from agent.metrics import get_metrics

        metrics_collector = get_metrics()
        return metrics_collector.get_all_metrics()
    except ImportError as e:
        logger.error(f"Failed to import metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={"error": f"Metrics module not available: {str(e)}"},
        ) from e
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)}) from e


@app.get("/metrics/summary")
async def metrics_summary() -> dict[str, Any]:
    """
    Get metrics summary.

    Returns:
        Summary of key metrics
    """
    try:
        from agent.metrics import get_metrics
        from agent.monitoring import get_memory_usage

        metrics_collector = get_metrics()
        all_metrics = metrics_collector.get_all_metrics()

        # Extract key metrics
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "counters": {
                "research_requests_total": all_metrics["counters"].get(
                    "research_requests_total", 0
                ),
                "research_completed_total": all_metrics["counters"].get(
                    "research_completed_total", 0
                ),
                "research_failed_total": all_metrics["counters"].get("research_failed_total", 0),
                "api_calls_total": all_metrics["counters"].get("api_calls_total", 0),
                "api_errors_total": all_metrics["counters"].get("api_errors_total", 0),
            },
            "gauges": {
                "active_research_requests": all_metrics["gauges"].get(
                    "active_research_requests", 0
                ),
                "total_documents_stored": all_metrics["gauges"].get("total_documents_stored", 0),
            },
            "performance": {
                "research_duration_seconds": all_metrics["histograms"].get(
                    "research_total_duration_seconds", {}
                ),
                "api_call_duration_seconds": all_metrics["histograms"].get(
                    "api_calls_duration_seconds", {}
                ),
            },
            "system": {
                "memory_usage_mb": get_memory_usage()["memory_mb"],
            },
        }

        return summary
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)}) from e


@app.get("/status")
async def status() -> dict[str, Any]:
    """
    Get comprehensive service status.

    Returns:
        Complete service status including health, metrics, and system info
    """
    try:
        from agent.metrics import get_metrics
        from agent.monitoring import get_memory_usage

        metrics_collector = get_metrics()
        deps = check_dependencies()
        vector_store = check_vector_store()
        memory = get_memory_usage()

        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "health": {
                "status": "healthy",
                "dependencies": deps,
                "vector_store": vector_store,
            },
            "metrics": {
                "counters": {
                    "research_requests": metrics_collector.get_counter("research_requests_total"),
                    "research_completed": metrics_collector.get_counter("research_completed_total"),
                    "research_failed": metrics_collector.get_counter("research_failed_total"),
                    "api_calls": metrics_collector.get_counter("api_calls_total"),
                    "api_errors": metrics_collector.get_counter("api_errors_total"),
                },
                "gauges": {
                    "active_requests": metrics_collector.get_gauge("active_research_requests"),
                    "documents_stored": metrics_collector.get_gauge("total_documents_stored"),
                },
            },
            "system": {
                "memory_mb": memory["memory_mb"],
                "memory_gb": memory["memory_gb"],
            },
        }
    except Exception as e:
        logger.error(f"Failed to get status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)}) from e


def main() -> None:
    """Run the health check API server"""
    port = int(os.getenv("HEALTH_API_PORT", "8080"))
    host = os.getenv("HEALTH_API_HOST", "0.0.0.0")

    logger.info(f"Starting health check API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
