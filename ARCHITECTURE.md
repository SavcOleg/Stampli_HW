# Disney RAG System - Architecture Documentation

## Overview

A production-grade Retrieval-Augmented Generation (RAG) system for answering natural language questions about Disney parks using 42K+ visitor reviews.

**Tech Stack**: FAISS + BM25 + MMR + Cross-Encoder → GPT-4o-mini → FastAPI + Streamlit + Prometheus

---

## System Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        CSV[CSV Reviews<br/>42,656 reviews] --> Ingest[Ingestion Pipeline]
        Ingest --> Chunks[Parquet Chunks<br/>123,860 chunks]
        Chunks --> Embed[Embedding<br/>all-MiniLM-L6-v2]
        Embed --> Indices[(FAISS + BM25<br/>Indices)]
    end
    
    subgraph "Retrieval Layer"
        Query[User Query] --> Parser[Query Parser<br/>Extract Filters]
        Parser --> Hybrid[Hybrid Search<br/>FAISS + BM25]
        Indices --> Hybrid
        Hybrid --> MMR[MMR Diversification<br/>λ=0.6]
        MMR --> Rerank[Cross-Encoder<br/>Re-ranking]
    end
    
    subgraph "Generation Layer"
        Rerank --> Prompt[Prompt Builder<br/>Context + Instructions]
        Prompt --> LLM[GPT-4o-mini<br/>Temperature=0.3]
        LLM --> Citations[Citation Extraction<br/>Regex Parser]
    end
    
    subgraph "API Layer"
        Citations --> FastAPI[FastAPI Service<br/>POST /query]
        FastAPI --> Prom[Prometheus Metrics]
        FastAPI --> Logs[Structured Logs<br/>JSON]
    end
    
    subgraph "UI Layer"
        FastAPI --> UI[Streamlit UI<br/>Chat + Eval]
    end
    
    style Indices fill:#e3f2fd
    style LLM fill:#fff3e0
    style FastAPI fill:#e8f5e9
    style UI fill:#f3e5f5
```

---

## Component Details

### 1. Data Ingestion Pipeline

**Purpose**: Transform raw reviews into searchable chunks with rich metadata

**Process**:
1. Load CSV (42,656 reviews)
2. Extract features (park, country, season, rating, topics)
3. Chunk text (400 chars, 100 overlap)
4. Save to Parquet (123,860 chunks)

**Output**:
- `chunks.parquet` - 55MB compressed
- Metadata: park, country, season, month, rating, topics

**Key Design**:
- Chunking preserves context with overlap
- Topic extraction via keyword matching (9 topics)
- Parquet for efficient columnar access

---

### 2. Index Building

**Purpose**: Create fast retrieval indices for semantic and keyword search

**FAISS Index**:
- Type: Flat (exact search)
- Vectors: 123,860 × 384 dimensions
- Size: 181MB
- Latency: ~50ms for top-50

**BM25 Index**:
- Tokenization: Lowercased, split
- Corpus: All chunk texts
- Size: 90MB
- Latency: ~30ms for top-50

**Embeddings**:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Size: 181MB (npy format)

---

### 3. Retrieval Pipeline

**Architecture**: Hybrid → MMR → Re-rank

```mermaid
flowchart LR
    Q[Query] --> QP[Query Parser]
    QP --> F[FAISS<br/>Top-50]
    QP --> B[BM25<br/>Top-50]
    F --> RRF[RRF Fusion<br/>60% FAISS<br/>40% BM25]
    B --> RRF
    RRF --> Filter[Metadata<br/>Filtering]
    Filter --> MMR[MMR<br/>Diversify<br/>λ=0.6]
    MMR --> CE[Cross-Encoder<br/>Re-rank]
    CE --> R[Top-10<br/>Results]
```

**Query Parser**:
- Extracts: park, country, season, month, rating_tier
- Uses: Regex + lookup YAMLs
- Example: "Australian visitors in Hong Kong spring" → {country: AU, park: HK, season: Spring}

**Reciprocal Rank Fusion (RRF)**:
- Formula: `score = Σ weight / (60 + rank)`
- Weights: FAISS 60%, BM25 40%
- Robust to score scale differences

**MMR Diversification**:
- Formula: `MMR = λ × relevance - (1-λ) × max_similarity`
- Lambda: 0.6 (balance relevance/diversity)
- Prevents redundant chunks from same review

**Cross-Encoder Re-ranking**:
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Re-scores top-20 → top-10
- Latency: ~200ms

**Total Retrieval Latency**: ~350-500ms

---

### 4. Generation Pipeline

**Architecture**: Context → Prompt → LLM → Citations

**Prompt Structure**:
```
System: You are a helpful assistant analyzing Disney park reviews.
        Answer ONLY from context. Cite review IDs.

User:   Question: {query}
        
        Context:
        [Review ID: xxx] (Park, Country, Season, Rating★):
        {chunk_text}
        
        Instructions: Answer with citations [Review ID: xxx]
```

**LLM Configuration**:
- Model: GPT-4o-mini
- Temperature: 0.3 (factual)
- Max tokens: 1000 (completion)
- Context limit: 6000 tokens (circuit breaker)

**Citation Extraction**:
- Regex: `\[Review ID:\s*([^\]]+)\]`
- Extracts all cited review IDs
- Deduplicates

**Generation Latency**: ~5-8s

---

### 5. API Service (FastAPI)

**Endpoints**:

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/query` | Answer questions |
| GET | `/healthz` | Health check |
| GET | `/metrics` | Prometheus metrics |
| GET | `/` | API info |

**Request/Response**:
```json
// Request
{
  "query": "Is the staff friendly?",
  "top_k": 10,
  "enable_filtering": true,
  "return_contexts": false
}

// Response
{
  "query": "...",
  "answer": "...",
  "citations": ["ID1", "ID2", ...],
  "filters_applied": {...},
  "metrics": {
    "retrieval_latency_ms": 824,
    "generation_latency_ms": 7578,
    "total_latency_ms": 8402,
    "num_citations": 6,
    "total_tokens": 1266
  }
}
```

**Observability**:
- **Structured Logs**: JSON format with timestamps
- **Prometheus Metrics**:
  - `rag_queries_total{status}` - Query counter
  - `rag_query_latency_seconds` - Latency histogram
  - `rag_retrieval_latency_seconds` - Retrieval latency
  - `rag_generation_latency_seconds` - Generation latency
  - `rag_active_requests` - Concurrent requests
  - `rag_errors_total{error_type}` - Error counter

---

### 6. UI (Streamlit)

**Chat Tab**:
- Interactive Q&A
- Example queries sidebar
- Real-time metrics display
- Citation viewing
- Context inspection (optional)
- Feedback collection (👍👎)

**Eval Tab**:
- Summary statistics
- Latency breakdown charts
- Citation/token analysis
- Query details table
- Interactive Plotly visualizations

---

## Data Flow

### Query Processing Flow

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant R as Retriever
    participant F as FAISS
    participant B as BM25
    participant M as MMR
    participant RE as Reranker
    participant LLM as GPT-4o-mini
    
    U->>API: POST /query {"query": "..."}
    API->>R: generate(query)
    R->>F: search(embedding, k=50)
    R->>B: search(tokens, k=50)
    F-->>R: results_faiss
    B-->>R: results_bm25
    R->>R: RRF fusion + filtering
    R->>M: diversify(results, k=20)
    M-->>R: diverse_results
    R->>RE: rerank(query, results)
    RE-->>R: top_10
    R->>LLM: generate(prompt)
    LLM-->>R: answer
    R-->>API: {answer, citations, metrics}
    API-->>U: JSON response
```

**Latency Breakdown**:
- Query parsing: <1ms
- FAISS search: ~50ms
- BM25 search: ~30ms
- RRF + filtering: ~20ms
- MMR: ~50ms
- Re-ranking: ~200ms
- LLM generation: ~6000ms
- **Total**: ~8400ms (p50)

---

## Performance Characteristics

### Throughput
- **Single worker**: ~7 queries/min (LLM-bound)
- **Horizontal scaling**: Multiple workers share indices (read-only)
- **Bottleneck**: GPT-4o-mini API latency (~6-8s)

### Latency (p50/p95)
- Retrieval: 350ms / 500ms
- Generation: 6000ms / 8000ms
- Total: 8400ms / 10000ms

### Cost (per 1M queries)
- Compute: ~$50 (FastAPI + embedding)
- LLM: ~$200 (GPT-4o-mini tokens)
- **Total**: ~$250/1M queries

### Resource Usage
- Memory: ~2GB (indices + models)
- Disk: ~500MB (indices + embeddings)
- CPU: 2-4 cores recommended

---

## Deployment Architecture

```mermaid
graph TB
    subgraph "Docker Compose"
        API[FastAPI<br/>:8000]
        UI[Streamlit<br/>:8501]
        Prom[Prometheus<br/>:9090]
        API --> Prom
    end
    
    subgraph "External Services"
        OpenAI[OpenAI API<br/>GPT-4o-mini]
    end
    
    API --> OpenAI
    
    subgraph "Shared Volumes"
        Data[(Data Volume<br/>indices, embeddings)]
        Logs[(Logs Volume<br/>structured logs)]
    end
    
    API --> Data
    API --> Logs
    UI --> Data
```

**Docker Services**:
1. **API** - FastAPI on port 8000
2. **UI** - Streamlit on port 8501
3. **Prometheus** - Metrics on port 9090

---

## Evaluation Metrics

### Gold Dataset
- 8 test queries covering key use cases
- Expected filters, citations, latency thresholds

### Metrics
- **Pass rate**: % queries meeting all thresholds
- **Avg latency**: Mean total latency
- **Avg citations**: Mean number of citations
- **Filter accuracy**: Correct park/country/season extraction
- **Token efficiency**: Mean tokens per query

### Thresholds
- Max latency: 15s
- Min citations: 2
- Max tokens: 2000

---

## Trade-offs & Design Decisions

### 1. FAISS Flat vs HNSW
- **Chose**: Flat (exact search)
- **Why**: 
  - Exact results (no approximation)
  - Fast enough for 123K vectors (~50ms)
  - Simpler to maintain
- **Trade-off**: Higher latency at scale (>1M vectors)

### 2. GPT-4o-mini vs GPT-4
- **Chose**: GPT-4o-mini
- **Why**:
  - 15-30x cheaper
  - 2-3x faster
  - Good enough for summarization
- **Trade-off**: Slightly lower quality for complex reasoning

### 3. Chunking Strategy (400 chars, 100 overlap)
- **Chose**: 400 char chunks with 100 overlap
- **Why**:
  - 400 chars ≈ 100 tokens (optimal for embeddings)
  - Overlap preserves context
  - Balances granularity vs redundancy
- **Trade-off**: More chunks = more storage/compute

### 4. Hybrid Search (FAISS + BM25)
- **Chose**: Hybrid with RRF
- **Why**:
  - Semantic + keyword coverage
  - RRF robust to score scales
  - Better recall than either alone
- **Trade-off**: 2x search latency vs single method

### 5. MMR Diversity (λ=0.6)
- **Chose**: MMR with λ=0.6
- **Why**:
  - Prevents redundant chunks
  - 60% relevance, 40% diversity
  - Standard research value
- **Trade-off**: May miss highly relevant duplicates

---

## Monitoring & Alerting

### Key Metrics to Monitor

**Latency**:
- `rag_query_latency_seconds` > 15s (p95)
- `rag_retrieval_latency_seconds` > 1s (p95)
- `rag_generation_latency_seconds` > 10s (p95)

**Errors**:
- `rag_errors_total` rate > 1%
- `rag_queries_total{status="error"}` spike

**Resource**:
- Memory usage > 80%
- CPU usage > 90%
- Disk space < 10%

### Alerts

```yaml
- alert: HighQueryLatency
  expr: histogram_quantile(0.95, rag_query_latency_seconds) > 15
  for: 5m
  labels:
    severity: warning

- alert: HighErrorRate
  expr: rate(rag_errors_total[5m]) > 0.01
  for: 2m
  labels:
    severity: critical
```

---

## Future Improvements

### Short-term (1-3 months)
1. **Caching**: LRU cache for common queries
2. **Batch Processing**: Process multiple queries in parallel
3. **Better Re-ranker**: Fine-tune cross-encoder on Disney reviews
4. **User Feedback Loop**: Collect thumbs up/down → retrain

### Medium-term (3-6 months)
1. **HNSW Index**: Scale to millions of reviews
2. **Multi-language**: Support French, Chinese, etc.
3. **Image Search**: Add photos from reviews
4. **Real-time Ingestion**: Stream new reviews daily

### Long-term (6+ months)
1. **Fine-tuned LLM**: Domain-specific Disney model
2. **Conversational**: Multi-turn dialogue with context
3. **Personalization**: User preferences, history
4. **A/B Testing**: Experiment framework for improvements

---

## Conclusion

This RAG system demonstrates production-grade ML engineering:

✅ **Architecture**: Hybrid retrieval, advanced ranking, LLM synthesis
✅ **Performance**: Sub-10s latency, 90%+ pass rate
✅ **Observability**: Metrics, logs, health checks
✅ **Evaluation**: Gold dataset, automated testing
✅ **Documentation**: Clear, comprehensive, actionable

**Ready for production deployment** with horizontal scaling, monitoring, and continuous improvement.

