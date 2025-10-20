# Design Log - Disney RAG System

*Documented design decisions and trade-offs throughout implementation*

## Phase 0: Bootstrap & Foundation

**Date**: 2025-10-20
**Status**: Completed

### Project Structure Decisions

#### Modular Architecture
**Decision**: Separate codebase into distinct modules (`ingestion`, `retrieval`, `generation`, `api`, `monitoring`, `ui`)

**Rationale**:
- **Separation of Concerns**: Each module has a single responsibility
- **Testability**: Easy to mock and test components in isolation
- **Maintainability**: Changes in one module don't cascade
- **Scalability**: Modules can be extracted into microservices if needed

**Trade-offs**:
- ✅ Pro: Clean boundaries, easier to reason about
- ✅ Pro: Multiple developers can work in parallel
- ⚠️ Con: More boilerplate (but manageable with good tooling)

#### Data Directory Structure
**Decision**: Organize data by processing stage (`raw`, `processed`, `indices`, `cache`, `lookup`)

**Rationale**:
- **Data Lineage**: Clear progression from raw to processed to indices
- **Debugging**: Easy to inspect intermediate outputs
- **Cleanup**: Can safely delete processed/cache without losing raw data
- **Gitignore**: Exclude large generated files while keeping configs

#### Dependency Management
**Decision**: Pin all dependency versions in `requirements.txt`

**Rationale**:
- **Reproducibility**: Same versions across all environments
- **Stability**: Avoid breaking changes from dependency updates
- **Security**: Easier to track CVEs for specific versions

**Key Dependencies**:
- `faiss-cpu==1.7.4`: CPU-only version (GPU unnecessary for 50K vectors)
- `sentence-transformers==2.2.2`: Stable, well-tested embedding library
- `openai==1.3.0`: Latest stable API with function calling support
- `fastapi==0.104.1`: Modern, fast, with automatic OpenAPI docs
- `streamlit==1.28.1`: Rapid UI development with minimal code

**Trade-offs**:
- ✅ Pro: Predictable builds
- ⚠️ Con: Manual updates needed (but intentional for home task)

#### Makefile Commands
**Decision**: Use Makefile instead of shell scripts or task runners

**Rationale**:
- **Standard**: Works on all Unix systems, no extra dependencies
- **Self-Documenting**: `make help` shows all commands
- **Composable**: Commands can depend on others
- **IDE Support**: Most IDEs can run make targets

**Commands Structure**:
- Data pipeline: `ingest` → `build-index`
- Services: `serve` (API), `serve-ui` (Streamlit)
- Quality: `test`, `lint`, `eval`
- Deployment: `docker-build`, `docker-up`, `docker-down`

#### Environment Configuration
**Decision**: `.env` file for secrets + YAML for lookups

**Rationale**:
- **Security**: `.env` not committed to git
- **Flexibility**: Easy to change configs without code changes
- **Best Practice**: 12-factor app methodology
- **Type Safety**: YAML provides structure for complex configs

**Configuration Layers**:
1. `.env` - secrets and environment-specific values
2. `config/*.yaml` - lookup tables and business logic
3. Code defaults - sensible fallbacks

### Technology Choices

#### Why FAISS over Pinecone/Weaviate?
**Decision**: Use local FAISS indices

**Rationale**:
- **Local-First**: No external dependencies for demo
- **Cost**: Free for development and evaluation
- **Speed**: Low latency for 50K vectors
- **Control**: Full control over indexing and search

**Trade-offs**:
- ✅ Pro: Simple deployment, no vendor lock-in
- ✅ Pro: Works offline
- ⚠️ Con: Doesn't scale to 1M+ vectors (but noted in docs)
- ⚠️ Con: No built-in backup/replication

#### Why FastAPI over Flask?
**Decision**: Use FastAPI for REST API

**Rationale**:
- **Performance**: Async/await support, faster than Flask
- **Type Safety**: Pydantic models for validation
- **Documentation**: Automatic OpenAPI/Swagger UI
- **Modern**: Built for Python 3.7+ features

#### Why Streamlit over React/Vue?
**Decision**: Use Streamlit for UI

**Rationale**:
- **Speed**: Build UI in hours, not days
- **Python-First**: No JavaScript needed
- **Prototyping**: Perfect for ML demos and evaluation
- **Built-in Features**: Charts, file uploads, forms out of box

**Trade-offs**:
- ✅ Pro: Rapid development
- ⚠️ Con: Less customizable than React (but sufficient for evaluation)

#### Why Prometheus over CloudWatch/Datadog?
**Decision**: Use Prometheus for metrics

**Rationale**:
- **Open Source**: Free, self-hosted
- **Standard**: De facto standard for metrics
- **Integration**: Easy to add Grafana dashboards
- **Pull Model**: Metrics scraped, not pushed (less coupling)

### Versioning Strategy

**Decision**: Track three version numbers (`data_version`, `index_version`, `prompt_version`)

**Rationale**:
- **Data Versioning**: Track which dataset version was used
- **Index Versioning**: Detect stale indices (need rebuild)
- **Prompt Versioning**: A/B test prompt changes

**Example Scenario**:
- User reports bad answers → check prompt version
- Index seems slow → check if index version matches data version
- New dataset → increment data version, rebuild required

### Files Created

```
.
├── .env.example           # Environment template
├── .gitignore            # Exclude artifacts and secrets
├── Makefile              # Operational commands
├── README.md             # Quick start guide
├── requirements.txt      # Pinned dependencies
├── DESIGN_LOG.md         # This file
├── src/                  # Application code
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── api/
│   ├── monitoring/
│   └── ui/
├── tests/                # Test suites
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── eval/                 # Evaluation framework
├── data/                 # Data pipeline
│   ├── raw/
│   ├── processed/
│   ├── indices/
│   ├── cache/
│   ├── feedback/
│   └── lookup/
├── logs/                 # Application logs
└── config/               # Application config
```

### Validation Results

✅ **Step 0 Complete**:
- Dependencies installed (faiss-cpu, sentence-transformers, fastapi, streamlit, etc.)
- All directories created and properly structured
- Core imports working: `import faiss, streamlit, fastapi` ✓
- CSV data loaded: 42,657 rows (~42K reviews)
- Context7 insight: FAISS HNSW with efSearch=64, M=32 provides good balance

## Phase 1: Data Ingestion & Feature Engineering

**Date**: 2025-10-20
**Status**: Completed

### Implementation Details

#### Feature Extraction
- **Temporal Features**: Extracted month, season, year from `Year_Month` (e.g., "2019-4" → April, Spring, 2019)
- **Geographic Normalization**: Mapped reviewer locations to standard country names (e.g., "UAE" → "United Arab Emirates")
- **Park Normalization**: Mapped branch codes to clean names (e.g., "Disneyland_HongKong" → "Hong Kong")
- **Rating Tiers**: Categorized ratings (1-2: negative, 3: neutral, 4-5: positive)
- **Topic Extraction**: Keyword-based detection for 9 topics (staff, food, crowded, rides, value, weather, wait_times, cleanliness, size)

#### Chunking Strategy
- **Chunk Size**: 400 characters (optimal for embeddings)
- **Overlap**: 100 characters (preserves context across chunks)
- **Smart Boundaries**: Breaks at sentence or word boundaries when possible
- **Metadata Preservation**: All features copied to each chunk

### Results

✅ **Validation Results**:
- Input: 42,656 reviews from CSV
- Output: 123,860 chunks (~2.90 chunks/review)
- File size: 55.55 MB (compressed Parquet with Snappy)
- Average chunk length: 305 characters

**Data Distribution**:
- **Parks**: Paris (51.5K), California (50.3K), Hong Kong (22K)
- **Ratings**: 1★ (6.6K), 2★ (8.9K), 3★ (17.5K), 4★ (33.4K), 5★ (57.5K)
- **Top Countries**: USA (39.9K), UK (37.4K), Australia (12.2K)
- **Seasons**: Well balanced across Spring, Summer, Fall, Winter

**Columns in chunks.parquet**:
- Identifiers: `review_id`, `chunk_id`, `chunk_index`, `total_chunks`
- Text: `review_text`, `chunk_text`
- Metadata: `park`, `country`, `rating`, `rating_tier`, `year`, `month`, `month_name`, `season`, `year_month`
- Topics: `topics` (array of detected topics)

### Design Decisions

**Why Parquet over CSV?**
- Columnar format → faster reads for specific columns
- Built-in compression → 2-3x smaller file size
- Type preservation → no string→int conversions needed
- Arrow compatibility → efficient data interchange

**Why 400-char chunks with 100-char overlap?**
- 400 chars ≈ 100 tokens (optimal for embeddings)
- 100-char overlap preserves context across boundaries
- Avoids splitting key phrases mid-sentence

**Why keyword-based topics vs ML?**
- Simple, interpretable, no training needed
- Fast processing (42K reviews in ~2 seconds)
- Good enough for metadata filtering
- Can upgrade to BERT-based later if needed

## Phase 2: Index Building (FAISS + BM25)

**Date**: 2025-10-20
**Status**: Completed

### Implementation Details

#### Embeddings Generation
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim vectors)
- **Processing**: Batch encoding with 32 samples/batch
- **Output**: 123,860 embeddings in ~3 minutes
- **File**: `embeddings.npy` (181MB, float32)

#### FAISS Indices
- **Flat Index (Exact Search)**: 
  - IndexFlatL2 for exact nearest neighbor search
  - Fast for datasets <200K vectors
  - Size: 181MB
  - Search latency: <50ms per query
  
- **HNSW Index (Approximate Search)**:
  - Attempted with M=16, efConstruction=100, efSearch=32
  - Hit memory/segfault issue during build (123K vectors is borderline)
  - Decision: Use Flat index (exact search is fine for prototype)

#### BM25 Index
- **Algorithm**: BM25Okapi for lexical/keyword matching
- **Tokenization**: Simple lowercase + whitespace split
- **Size**: 90.43MB (compressed with pickle)
- **Performance**: 54ms avg latency, 18.4 queries/sec

#### Metadata Storage
- **Format**: Python pickle of dictionary (idx → chunk data)
- **Contents**: chunk_id, review_id, text, park, country, season, rating, topics
- **Size**: 49.45MB
- **Purpose**: Map FAISS/BM25 indices back to full chunk information

### Results

✅ **Validation Results**:
- Embeddings: 123,860 vectors × 384 dims = 181MB ✓
- FAISS Flat: 123,860 vectors indexed ✓
- BM25: 123,860 documents indexed ✓
- Metadata: 123,860 chunks mapped ✓
- Total index size: 421MB (181 + 181 + 90 + 50)

**Performance Tests**:
- FAISS search: <50ms per query (exact results)
- BM25 search: 54ms per query (keyword matching)
- Combined hybrid search estimate: ~100ms

### Design Decisions

**Why skip HNSW after segfault?**
- Flat index provides **exact** results (no approximation)
- For 123K vectors, Flat search is still fast (<50ms)
- HNSW benefits show at 1M+ vectors
- Production version can use HNSW or move to Pinecone/Weaviate

**Why BM25 + FAISS hybrid?**
- BM25: catches exact keyword matches ("staff friendly")
- FAISS: catches semantic similarity ("helpful employees")
- Hybrid: best of both worlds via Reciprocal Rank Fusion

**Why pickle for BM25 and metadata?**
- Simple, fast serialization for Python objects
- BM25Okapi doesn't have native save/load
- Metadata dict is complex (nested arrays, mixed types)
- Alternative: Use SQLite for metadata (can add later)

### Files Created

```
data/processed/
  embeddings.npy              181MB (123,860 × 384 float32)
  embeddings_metadata.pkl     105B  (model info)

data/indices/
  faiss_flat.index            181MB (exact NN search)
  bm25.pkl                    90MB  (keyword search)
  metadata.pkl                49MB  (chunk mappings)
```

## Phase 3: Retrieval Pipeline

**Date**: 2025-10-20
**Status**: Completed

### Implementation Details

#### Query Parser
- **Intents Extracted**: Geographic (park, country), temporal (season, month), rating tier
- **Normalization**: Uses lookup YAMLs for consistent mapping
- **Example**: "What do Australian visitors say about Hong Kong in spring?" → 
  - park: "Hong Kong", country: "Australia", season: "Spring"

#### Hybrid Search (FAISS + BM25)
- **Algorithm**: Reciprocal Rank Fusion (RRF)
- **Weights**: FAISS 60%, BM25 40%
- **Process**:
  1. Query both indices for top-50 candidates
  2. Merge using RRF formula: score = Σ weight / (60 + rank)
  3. Apply metadata filters from query parser
  4. Return top candidates

#### MMR Diversification
- **Algorithm**: Maximal Marginal Relevance  
- **Formula**: MMR = λ × relevance - (1-λ) × max_similarity
- **Lambda**: 0.6 (balance relevance vs diversity)
- **Purpose**: Prevents redundant results from same review
- **Input**: 50 hybrid results → Output: 20 diverse results

#### Cross-Encoder Re-ranker
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Purpose**: Fine-grained relevance scoring on query-document pairs
- **Process**: Re-scores top-20 MMR results, returns top-10
- **Latency**: ~200ms for 20 documents

### Results

✅ **Validation Results**:
- Query parser: Correctly extracts filters from all 4 key queries ✓
- Hybrid search: Returns relevant chunks with RRF scoring ✓
- MMR: Diversifies results (tested with 50→20 reduction) ✓
- Re-ranker: Improves relevance order (scores 4-9 range) ✓
- End-to-end: Working pipeline with all components ✓

**Example Query**: "What do visitors from Australia say about Hong Kong?"
- Parsed filters: park="Hong Kong", country="Australia"
- Pipeline: 100 candidates → 28 filtered → 20 diverse → 3 final
- Top result: Relevant review from Australian about Hong Kong with rerank_score=5.88

### Design Decisions

**Why Reciprocal Rank Fusion?**
- Simple, effective fusion method
- No training required (unlike learned fusion)
- Robust to score scale differences between FAISS and BM25
- Well-studied in IR literature

**Why MMR with λ=0.6?**
- λ=0.6 means 60% relevance, 40% diversity
- Prevents showing 10 chunks from the same review
- Standard value from research (λ typically 0.5-0.7)

**Why cross-encoder after MMR?**
- Cross-encoders are slow (~10ms per pair)
- MMR reduces candidates from 50 to 20 (saves compute)
- Re-ranking 20 docs takes ~200ms (acceptable)
- Re-ranking 50 docs would take ~500ms (too slow)

**Why enable/disable flags?**
- MMR and re-ranking add latency (~300ms total)
- For speed-critical use cases, can disable
- Allows A/B testing different configurations
- Trade-off: speed vs quality

### Performance

**Latency Breakdown** (estimated):
- Query parsing: <1ms
- Hybrid search: 50-100ms (FAISS 50ms + BM25 54ms in parallel)
- MMR: ~50ms (50 candidates, similarity computations)
- Re-ranker: ~200ms (20 documents)
- **Total**: ~350ms p50, ~500ms p95

### Files Created

```
src/retrieval/
  query_parser.py       - Intent extraction and filtering
  hybrid_search.py      - FAISS + BM25 with RRF fusion
  mmr.py                - MMR diversification algorithm
  reranker.py           - Cross-encoder re-ranking
  retriever.py          - Main orchestrator (full pipeline)
```

## Phase 4: LLM Generation

**Date**: 2025-10-20
**Status**: Completed

### Implementation Details

#### Prompt Builder
- **System Prompt**: Clear instructions for review analysis with citation requirements
- **Context Format**: `[Review ID: xxx] (Park, Country, Season, Rating★): text`
- **Guidelines**: Answer only from context, cite all claims, summarize patterns
- **Token Awareness**: Tracks context size for circuit breaker

#### LLM Client
- **Model**: `gpt-4o-mini` (cost-effective, fast)
- **Guardrails**:
  - Max context tokens: 6000 (circuit breaker)
  - Max completion tokens: 1000
  - Temperature: 0.3 (low for factual responses)
  - Retry logic: 3 attempts with exponential backoff
- **Token Counting**: Uses tiktoken for accurate counting
- **Metadata**: Returns prompt_tokens, completion_tokens, latency

#### RAG Generator
- **Pipeline**: 
  1. Retrieve top-k chunks (default k=10)
  2. Build prompt with contexts
  3. Generate with GPT-4o-mini
  4. Extract citations using regex
  5. Return answer + metadata
- **Citation Extraction**: Regex pattern `\[Review ID:\s*([^\]]+)\]`
- **Error Handling**: Graceful fallback if no results or LLM fails

### Results

✅ **Validation Results**:
- LLM client: Successfully connected to OpenAI API ✓
- Token counting: Accurate with tiktoken ✓
- Prompt building: Clean format with metadata ✓
- Citation extraction: Works with regex ✓
- End-to-end RAG: Full pipeline operational ✓

**Example Query**: "What do visitors from Australia say about Disneyland in Hong Kong?"

**Performance**:
- Retrieval: 1,032ms
- Generation: 6,815ms
- **Total: 7,847ms** (~8 seconds p50)
- Tokens: 979 (prompt + completion)
- Citations: 5 review IDs cited

**Answer Quality**:
- ✅ Balanced positive and negative feedback
- ✅ All claims cited with review IDs
- ✅ Mentioned specific patterns (families from Melbourne, comparisons to Anaheim)
- ✅ Geographic/temporal context preserved
- ✅ Factual, no hallucination

### Design Decisions

**Why GPT-4o-mini over GPT-4?**
- 15-30x cheaper ($0.15/1M vs $5/1M input tokens)
- 2-3x faster latency (~5s vs ~15s)
- Sufficient quality for summarization task
- Can upgrade to GPT-4 for complex analysis if needed

**Why temperature=0.3?**
- Need consistency and factuality
- 0.0 too deterministic (repetitive)
- 0.3 allows slight variation while staying grounded
- Higher temps (0.7+) risk hallucination

**Why max_tokens=1000?**
- Typical answer: 200-400 tokens
- 1000 provides buffer for detailed answers
- Prevents runaway generation costs
- Users prefer concise answers anyway

**Why 6000 token circuit breaker?**
- GPT-4o-mini context: 128K tokens
- 6000 = ~10 chunks × 600 tokens/chunk
- Leaves headroom for system prompt + safety margin
- Prevents expensive API calls on huge contexts

**Why extract citations with regex?**
- Simple, reliable pattern matching
- No need for complex NLP
- Fast (<1ms overhead)
- Works well with structured format

### Cost Analysis

**Per Query** (assuming 10 chunks):
- Prompt tokens: ~800-1000
- Completion tokens: ~300-500
- Cost: ~$0.0002 (0.02 cents)

**At Scale**:
- 1M queries/month: ~$200/month
- Well within production budget
- Can optimize with caching if needed

### Files Created

```
src/generation/
  __init__.py           - Module init
  prompt_builder.py     - Prompt construction with context
  llm_client.py         - OpenAI client with guardrails
  generator.py          - Main RAG orchestrator
```

## Phase 5: FastAPI Service

**Date**: 2025-10-20
**Status**: Completed

### Implementation Details

#### API Endpoints

**POST /query**
- Input: `QueryRequest` (query, top_k, enable_filtering, return_contexts)
- Output: `QueryResponse` (answer, citations, metrics, optional contexts)
- Validation: Pydantic models with constraints
- Error handling: Graceful fallback with HTTP 500

**GET /healthz**
- Health check with component status
- Returns uptime, version, component health
- Used by load balancers / orchestrators

**GET /metrics**
- Prometheus metrics in text format
- Custom metrics: queries_total, latencies, errors
- Standard metrics: active_requests, etc.

**GET /**
- Root endpoint with API documentation
- Lists available endpoints

#### Observability

**Structured Logging (structlog)**
- JSON-formatted logs for easy parsing
- ISO timestamps
- Request/response logging with latency
- Error logging with context

**Prometheus Metrics**
- `rag_queries_total{status}` - Counter of queries
- `rag_query_latency_seconds` - Histogram of total latency
- `rag_retrieval_latency_seconds` - Histogram of retrieval latency
- `rag_generation_latency_seconds` - Histogram of generation latency
- `rag_active_requests` - Gauge of concurrent requests
- `rag_errors_total{error_type}` - Counter of errors

**Middleware**
- Request logging with latency
- Active request tracking
- CORS for cross-origin requests
- Global exception handler

#### Pydantic Models

- Type validation for all request/response
- Field constraints (min/max, required)
- Automatic OpenAPI schema generation
- Error messages for invalid input

### Results

✅ **Validation Results**:
- Health endpoint: Returns 200 OK ✓
- Query endpoint: Processes requests successfully ✓
- Metrics endpoint: Exports Prometheus metrics ✓
- Structured logs: JSON format with timestamps ✓
- Error handling: Graceful with proper HTTP codes ✓

**Example Query**: "Is the staff friendly?"
```json
{
  "query": "Is the staff friendly?",
  "answer": "Mixed reviews... [with citations]",
  "citations": ["3429205", "392945847", ...],
  "metrics": {
    "retrieval_latency_ms": 824,
    "generation_latency_ms": 7578,
    "total_latency_ms": 8403,
    "num_citations": 6,
    "total_tokens": 1266
  }
}
```

### Design Decisions

**Why FastAPI over Flask?**
- Native async support (better concurrency)
- Automatic OpenAPI/Swagger docs
- Built-in request validation with Pydantic
- Better performance (ASGI vs WSGI)
- Modern, actively maintained

**Why Prometheus metrics?**
- Industry standard for observability
- Native Kubernetes integration
- Rich query language (PromQL)
- Excellent alerting support
- Many dashboard tools (Grafana)

**Why structlog over stdlib logging?**
- Structured (JSON) logs easier to parse
- Better for log aggregation (ELK, Datadog)
- Contextual information preserved
- Type-safe logging

**Why separate /healthz from /metrics?**
- Health checks don't need auth
- Different access patterns
- Health checks are lightweight
- Metrics can be expensive to generate

**Why Pydantic validation?**
- Catches bad input early
- Clear error messages
- Type safety
- Auto-generates API docs
- Reduces boilerplate

### Performance

**Latency** (from test):
- Health check: <10ms
- Query endpoint: ~8.4s (825ms retrieval + 7.6s generation)
- Metrics endpoint: <50ms

**Throughput**:
- Single worker: ~7 queries/min (limited by LLM)
- Can scale horizontally with multiple workers
- Each worker shares indices (read-only)

### Files Created

```
src/api/
  __init__.py       - Module init
  models.py         - Pydantic request/response models
  app.py            - FastAPI application with endpoints

test_api.py         - API test script
```

## Phase 6: Streamlit UI

**Date**: 2025-10-20
**Status**: Completed

### Implementation Details

#### Chat Tab
- **Interactive Q&A**: Text input for natural language queries
- **Example Queries**: Pre-built queries in sidebar for quick testing
- **Real-time Results**: Displays answer, citations, metrics immediately
- **Chat History**: Maintains conversation history with timestamps
- **Contexts Display**: Optional view of retrieved chunks with scores
- **Feedback**: Thumbs up/down for user feedback collection
- **Settings**: Adjustable top_k, filtering, context display

#### Eval Tab
- **Summary Statistics**: Total queries, avg latency, avg citations, avg tokens
- **Latency Analysis**: 
  - Stacked bar chart (retrieval vs generation)
  - Line chart showing trend over time
- **Citation Analysis**: Bar chart of citations per query
- **Token Analysis**: Stacked bar chart of prompt vs completion tokens
- **Query Details Table**: Comprehensive view of all queries with metrics

#### UI Features
- **Modern Design**: Clean, professional interface with custom CSS
- **Responsive Layout**: Wide layout for better data visualization
- **Interactive Charts**: Plotly for beautiful, interactive visualizations
- **Cached Loading**: Generator loaded once and cached for performance
- **Error Handling**: Graceful error display with helpful messages

### Results

✅ **Validation Results**:
- Streamlit app structure created ✓
- Chat tab with query interface ✓
- Eval tab with metrics visualization ✓
- Example queries in sidebar ✓
- Interactive Plotly charts ✓
- Session state management ✓

**Features**:
- 💬 Chat interface with example queries
- 📊 Real-time metrics (latency, citations, tokens)
- 📈 Visualization charts (latency breakdown, token usage)
- 📚 Citation display with review IDs
- 👍👎 Feedback collection
- ⚙️ Configurable settings (top_k, filtering)

### Design Decisions

**Why Streamlit over React?**
- Rapid prototyping (Python-native)
- No frontend expertise needed
- Built-in state management
- Great for ML/data apps
- Easy deployment

**Why two tabs (Chat + Eval)?**
- Separation of concerns
- Chat for end-users
- Eval for analysts/developers
- Clean, organized interface

**Why Plotly over Matplotlib?**
- Interactive charts
- Beautiful out-of-the-box styling
- Hover tooltips
- Zoom, pan capabilities
- Better for web apps

**Why session state?**
- Persist chat history
- Maintain settings across reruns
- Enable multi-query analysis
- Required for stateful UI

### Files Created

```
src/ui/
  __init__.py       - Module init
  app.py            - Main Streamlit application (Chat + Eval tabs)
```

### Next Steps

- [x] Phase 0: Bootstrap completed
- [x] Phase 1: Data ingestion and feature engineering completed  
- [x] Phase 2: Index building completed (FAISS Flat + BM25)
- [x] Phase 3: Retrieval pipeline completed (hybrid, MMR, re-ranking)
- [x] Phase 4: LLM generation completed (GPT-4o-mini synthesis)
- [x] Phase 5: FastAPI service completed
- [x] Phase 6: Streamlit UI completed
- [ ] Phase 7: Evaluation framework
- [ ] Phase 8: Documentation
- [ ] Phase 9: Docker
- [ ] Phase 10: Testing
- [ ] Phase 11: Final validation

---

*This log is continuously updated as design decisions are made.*

