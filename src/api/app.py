"""
FastAPI application for RAG system.

Endpoints:
- POST /query: Answer natural language questions
- GET /healthz: Health check
- GET /metrics: Prometheus metrics
"""

import time
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import structlog

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.generation.generator import RAGGenerator
from src.api.models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    ErrorResponse
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Prometheus metrics
QUERY_COUNTER = Counter('rag_queries_total', 'Total number of queries', ['status'])
QUERY_LATENCY = Histogram('rag_query_latency_seconds', 'Query latency in seconds')
RETRIEVAL_LATENCY = Histogram('rag_retrieval_latency_seconds', 'Retrieval latency in seconds')
GENERATION_LATENCY = Histogram('rag_generation_latency_seconds', 'Generation latency in seconds')
ACTIVE_REQUESTS = Gauge('rag_active_requests', 'Number of active requests')
ERROR_COUNTER = Counter('rag_errors_total', 'Total number of errors', ['error_type'])

# Global state
generator: Optional[RAGGenerator] = None
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global generator
    
    logger.info("starting_rag_service")
    
    try:
        # Initialize RAG generator
        generator = RAGGenerator()
        logger.info("rag_generator_initialized")
        yield
    except Exception as e:
        logger.error("startup_failed", error=str(e))
        raise
    finally:
        logger.info("shutting_down_rag_service")


# Create FastAPI app
app = FastAPI(
    title="Disney RAG System",
    description="RAG system for answering questions about Disney parks using reviews",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with latency."""
    start_time = time.time()
    
    ACTIVE_REQUESTS.inc()
    
    try:
        response = await call_next(request)
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms
        )
        
        return response
    except Exception as e:
        logger.error(
            "request_failed",
            method=request.method,
            path=request.url.path,
            error=str(e)
        )
        raise
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Answer a natural language query using RAG.
    
    Args:
        request: Query request with parameters
    
    Returns:
        QueryResponse with answer, citations, and metadata
    """
    if not generator:
        ERROR_COUNTER.labels(error_type="generator_not_initialized").inc()
        raise HTTPException(status_code=503, detail="RAG generator not initialized")
    
    query_start = time.time()
    
    try:
        logger.info(
            "query_received",
            query=request.query,
            top_k=request.top_k,
            enable_filtering=request.enable_filtering
        )
        
        # Generate answer
        with QUERY_LATENCY.time():
            result = generator.generate(
                query=request.query,
                enable_filtering=request.enable_filtering,
                return_contexts=request.return_contexts
            )
        
        # Record metrics
        QUERY_COUNTER.labels(status="success").inc()
        RETRIEVAL_LATENCY.observe(result['metrics']['retrieval_latency_ms'] / 1000)
        GENERATION_LATENCY.observe(result['metrics']['generation_latency_ms'] / 1000)
        
        query_latency_ms = (time.time() - query_start) * 1000
        
        logger.info(
            "query_completed",
            query=request.query,
            num_citations=len(result['citations']),
            num_contexts=result['metrics']['num_contexts'],
            total_latency_ms=query_latency_ms
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        QUERY_COUNTER.labels(status="error").inc()
        ERROR_COUNTER.labels(error_type="query_processing").inc()
        
        logger.error(
            "query_failed",
            query=request.query,
            error=str(e)
        )
        
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with service status
    """
    uptime = time.time() - start_time
    
    components = {
        "generator": "healthy" if generator else "not_initialized",
        "api": "healthy"
    }
    
    status = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        components=components,
        uptime_seconds=uptime
    )


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Metrics in Prometheus format
    """
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Disney RAG System",
        "version": "1.0.0",
        "endpoints": {
            "query": "POST /query",
            "health": "GET /healthz",
            "metrics": "GET /metrics"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    ERROR_COUNTER.labels(error_type="unhandled").inc()
    
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc)
    )
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=500
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

