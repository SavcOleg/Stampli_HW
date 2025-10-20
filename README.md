# 🏰 Disney RAG System

**Production-grade Retrieval-Augmented Generation (RAG) system** for answering natural language questions about Disney parks using 42K+ visitor reviews.

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)

## ✨ Features

- 🔍 **Hybrid Retrieval**: FAISS semantic search + BM25 keyword search with RRF fusion
- 🎯 **Advanced Ranking**: MMR diversification + Cross-encoder re-ranking
- 🤖 **LLM Generation**: GPT-4o-mini with citations and circuit breakers
- 📊 **Metadata Filtering**: Smart query parsing (park, country, season, rating)
- 🚀 **FastAPI Service**: RESTful API with Prometheus metrics
- 💬 **Streamlit UI**: Interactive chat + evaluation dashboard
- 🐳 **Docker Ready**: Full deployment stack with docker-compose
- 📈 **Observability**: Structured logs, metrics, health checks
- ✅ **Tested**: 8+ unit/integration tests with CI pipeline

## 📊 System Architecture

```
User Query → Query Parser → Hybrid Search (FAISS + BM25) → MMR → Re-ranker → LLM → Answer + Citations
```

**Latency**: ~8s end-to-end (350ms retrieval + 6s generation)
**Scale**: 123,860 searchable chunks from 42,656 reviews
**Cost**: ~$0.0002 per query

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key
- 2GB RAM minimum

### Installation

```bash
# Install dependencies
make install

# Set OpenAI API key
echo "OPENAI_API_KEY=your-key-here" > .env

# Run ingestion (if not already done)
make ingest

# Build indices (if not already done)
make build-index
```

### Running the System

**Option 1: Direct Python**

```bash
# Start API server
make serve

# In another terminal, start UI
make serve-ui

# Access UI at http://localhost:8501
# Access API at http://localhost:8000
```

**Option 2: Docker Compose**

```bash
# Build and start all services
docker-compose up --build

# Access UI at http://localhost:8501
# Access API at http://localhost:8000
# Access Prometheus at http://localhost:9090
```

### Example Queries

- "What do visitors from Australia say about Disneyland in Hong Kong?"
- "Is spring a good time to visit Disneyland?"
- "Is Disneyland California crowded in June?"
- "Is the staff in Paris friendly?"

## 📖 API Usage

### Query Endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Is the staff friendly?",
    "top_k": 10,
    "enable_filtering": true
  }'
```

**Response:**
```json
{
  "query": "Is the staff friendly?",
  "answer": "Mixed reviews... [with citations]",
  "citations": ["R123", "R456"],
  "metrics": {
    "retrieval_latency_ms": 350,
    "generation_latency_ms": 6000,
    "total_latency_ms": 8400
  }
}
```

## 📁 Project Structure

```
Stampli_HW/
├── src/
│   ├── ingestion/       # Data processing & chunking
│   ├── retrieval/       # FAISS, BM25, hybrid search
│   ├── generation/      # LLM client & prompt building
│   ├── api/             # FastAPI service
│   └── ui/              # Streamlit application
├── data/
│   ├── indices/         # FAISS, BM25, metadata
│   ├── processed/       # Chunks & embeddings
│   └── lookup/          # Normalization YAMLs
├── eval/                # Evaluation framework
├── tests/               # Unit & integration tests
├── Dockerfile           # Container build
├── docker-compose.yml   # Multi-service stack
├── ARCHITECTURE.md      # System architecture
├── DESIGN_LOG.md        # Design decisions
└── VALIDATION_REPORT.md # Validation results
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run unit tests only
pytest tests/unit/ -v
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Total Reviews | 42,656 |
| Searchable Chunks | 123,860 |
| Retrieval Latency | ~350ms |
| Generation Latency | ~6s |
| End-to-End Latency | ~8.4s |
| Cost per Query | ~$0.0002 |

## 📚 Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture & design
- [DESIGN_LOG.md](DESIGN_LOG.md) - Phase-by-phase decisions
- [VALIDATION_REPORT.md](VALIDATION_REPORT.md) - Complete validation

## 🔍 Key Technologies

- **Retrieval**: FAISS + BM25
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Re-ranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **LLM**: OpenAI GPT-4o-mini
- **API**: FastAPI + Prometheus
- **UI**: Streamlit + Plotly
- **Deployment**: Docker + docker-compose

## 📝 Makefile Commands

- `make install` - Install dependencies
- `make ingest` - Run data ingestion
- `make build-index` - Build indices
- `make serve` - Start API server
- `make serve-ui` - Start Streamlit UI
- `make test` - Run tests
- `make eval` - Run evaluation

---

**Built for Stampli ML Engineer Home Assignment**
