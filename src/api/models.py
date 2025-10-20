"""
Pydantic models for API request/response validation.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    
    query: str = Field(..., description="Natural language query", min_length=1)
    top_k: int = Field(10, description="Number of chunks to retrieve", ge=1, le=50)
    enable_filtering: bool = Field(True, description="Enable metadata filtering")
    return_contexts: bool = Field(False, description="Include retrieved contexts in response")


class Context(BaseModel):
    """Model for retrieved context chunk."""
    
    review_id: str
    chunk_text: str
    park: str
    country: str
    season: str
    rating: int
    score: float
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    
    query: str
    answer: str
    citations: List[str]
    filters_applied: Dict
    contexts: Optional[List[Context]] = None
    metrics: Dict


class IngestRequest(BaseModel):
    """Request model for /ingest endpoint (future use)."""
    
    csv_path: str = Field(..., description="Path to CSV file")
    rebuild_indices: bool = Field(True, description="Rebuild FAISS/BM25 indices")


class IngestResponse(BaseModel):
    """Response model for /ingest endpoint."""
    
    status: str
    num_reviews: int
    num_chunks: int
    message: str


class HealthResponse(BaseModel):
    """Response model for /healthz endpoint."""
    
    status: str
    version: str
    components: Dict[str, str]
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Response model for error cases."""
    
    error: str
    detail: Optional[str] = None
    status_code: int

