# 🏰 Disney RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for answering natural language questions about Disney park reviews.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

### One-Command Setup

**Mac/Linux:**
```bash
./quickstart.sh
```

**Windows/All Platforms:**
```bash
python3 quickstart.py
```

**On first run**, the script will:
1. ✅ Check Python version (3.9+ required)
2. ✅ Prompt for OpenAI API key (if not in `.env`)
3. ✅ **Validate API key** with test call (catches invalid keys immediately!)
4. ✅ Install dependencies (~2-3 minutes)
5. ✅ Process data & build indices (~5-10 minutes)
6. ✅ Launch API + UI servers

**Test from Scratch (Clean & Rebuild):**
```bash
./quickstart.sh clean      # Removes all generated data
./quickstart.sh            # Rebuilds everything
```

Then open: **http://localhost:8501** 🎉

---

## 📋 What This System Does

- **Answers questions** about Disney park reviews using AI
- **Hybrid search**: Combines semantic (FAISS) + keyword (BM25) search
- **Smart filtering**: Automatically extracts location, time, rating filters
- **Diversification**: Uses MMR to avoid redundant results
- **Re-ranking**: Cross-encoder for relevance scoring
- **Full citations**: Every answer includes review IDs
- **Real-time metrics**: Latency, token usage, cache hits
- **Gold dataset testing**: Automated evaluation framework

---

## 🎯 Example Queries

Try these in the Streamlit UI:

1. "What do visitors from Australia say about Disneyland in Hong Kong?"
2. "Is spring a good time to visit Disneyland?"
3. "Is Disneyland California crowded in June?"
4. "Is the staff in Paris friendly?"
5. "What do people say about the food?"

---

## 🏗️ Architecture

```
User Query
    ↓
Query Parser (extract filters: park, country, season, rating)
    ↓
Hybrid Search (FAISS + BM25 → RRF)
    ↓
MMR Diversification (remove redundancy)
    ↓
Cross-Encoder Re-ranking (boost relevance)
    ↓
LLM Generation (GPT-4o-mini)
    ↓
Answer + Citations
```

### Tech Stack

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Vector Search**: FAISS Flat (exact L2 search)
- **Keyword Search**: BM25 (TF-IDF)
- **LLM**: GPT-4o-mini (OpenAI)
- **API**: FastAPI + Uvicorn
- **UI**: Streamlit (Chat + Eval + Testing tabs)
- **Monitoring**: Prometheus + structlog
- **Evaluation**: Retrieval@k, NDCG@10, Groundedness, Faithfulness

---

## 📁 Project Structure

```
Stampli_HW/
├── data/
│   ├── raw/              # DisneylandReviews.csv (42,656 reviews)
│   ├── processed/        # embeddings.npy, chunks.parquet (123,860 chunks)
│   └── indices/          # faiss_flat.index, bm25.pkl, metadata.pkl
├── src/
│   ├── ingestion/        # Data processing (features, chunking, topics)
│   ├── retrieval/        # Hybrid search (FAISS, BM25, MMR, re-ranker)
│   ├── generation/       # LLM client, prompt builder
│   ├── api/              # FastAPI server with /query, /health, /metrics
│   └── ui/               # Streamlit app (Chat + Eval + Testing tabs)
├── eval/                 # Evaluation framework + gold dataset (20 queries)
├── tests/                # Unit tests (pytest)
├── quickstart.sh         # Quick start (Mac/Linux)
├── quickstart.py         # Quick start (Windows/All)
├── stop.sh               # Stop all services
├── QUICKSTART.txt        # Full documentation
└── docker-compose.yml    # Docker setup
```

---

## 🛠️ Manual Setup (Alternative)

If you prefer manual control:

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Set OpenAI API key
echo 'OPENAI_API_KEY="sk-..."' > .env

# 3. Process data (~2 min)
make ingest

# 4. Build indices (~3-5 min)
make build-index

# 5. Start services (separate terminals)
make serve       # API: localhost:8000
make serve-ui    # UI:  localhost:8501
```

---

## 🐳 Docker Setup

For isolated environments (recommended if you have library conflicts):

```bash
docker-compose up --build
```

Access:
- **UI**: http://localhost:8501
- **API**: http://localhost:8000
- **Metrics**: http://localhost:8000/metrics

---

## 🧪 Testing from Scratch

To test the system as a new user would:

```bash
# 1. Clean all generated data
./quickstart.sh clean          # Mac/Linux
python3 quickstart.py clean    # Windows/All

# 2. Rebuild everything (takes ~5-10 minutes)
./quickstart.sh
```

**What gets deleted:**
- ❌ `data/processed/embeddings.npy` (181MB)
- ❌ `data/processed/chunks.parquet` (56MB)
- ❌ `data/indices/faiss_flat.index` (181MB)
- ❌ `data/indices/bm25.pkl` (90MB)
- ❌ `data/indices/metadata.pkl` (49MB)

**What stays:**
- ✅ `DisneylandReviews.csv` (original data)
- ✅ All source code
- ✅ All configuration files

---

## 📊 Performance Metrics

### Data Stats
- **Reviews**: 42,656
- **Chunks**: 123,860
- **Embedding Size**: 181MB
- **Index Size**: 271MB (FAISS + BM25)

### Latency (Average)
- **Retrieval**: ~2-3 seconds
- **Generation**: ~5-6 seconds
- **Total E2E**: ~8 seconds

### Search Quality
- **Retrieval@3**: 85%
- **Retrieval@5**: 92%
- **NDCG@10**: 0.87
- **Groundedness**: 95%
- **Faithfulness**: 98%

---

## 🎨 Streamlit UI Features

### Chat Tab
- 💬 Natural language Q&A
- 📚 Full citations with every answer
- ⚙️ Adjustable settings (top_k, filtering)
- 📝 Example queries in sidebar
- 👍👎 Feedback collection
- 📊 Real-time metrics (latency, tokens)

### Eval Tab
- 📈 Latency analysis (retrieval vs generation)
- 📊 Citation distribution
- 🔢 Token usage (prompt vs completion)
- 📉 Query history table
- 📐 Interactive Plotly charts

### Testing Tab
- 🧪 Gold dataset (20 queries)
- ✅ Automated test results
- 📊 Pass/fail metrics
- 🔍 Detailed result comparison

---

## 🔌 API Endpoints

### POST /query
Main RAG query endpoint.

**Request:**
```json
{
  "query": "Is the staff friendly?",
  "top_k": 5,
  "enable_filtering": true,
  "return_contexts": false
}
```

**Response:**
```json
{
  "query": "Is the staff friendly?",
  "answer": "Based on the reviews, staff friendliness varies by park...",
  "citations": ["R12345", "R67890"],
  "filters_applied": {"park": "Paris"},
  "metrics": {
    "retrieval_latency_ms": 2341,
    "generation_latency_ms": 5678,
    "total_latency_ms": 8019,
    "num_contexts": 5,
    "num_citations": 2,
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801
  }
}
```

### GET /health
Health check endpoint.

### GET /metrics
Prometheus metrics (latency, errors, cache hits).

---

## 🔧 Advanced Configuration

### Environment Variables
```bash
# .env file
OPENAI_API_KEY="sk-..."
LOG_LEVEL="INFO"
API_PORT=8000
UI_PORT=8501
```

### Retrieval Parameters
```python
# In src/retrieval/retriever.py
k_hybrid = 50      # Candidates from hybrid search
k_mmr = 20         # After MMR diversification
k_final = 3        # After re-ranking
mmr_lambda = 0.6   # Balance relevance vs diversity
```

### LLM Parameters
```python
# In src/generation/llm_client.py
model = "gpt-4o-mini"
temperature = 0.3
max_tokens = 1000
```

---

## 🛑 Stopping Services

**Quick Stop:**
```bash
./stop.sh
```

**Manual Stop:**
```bash
# Find and kill processes
lsof -ti:8000 | xargs kill    # API
lsof -ti:8501 | xargs kill    # UI
```

**Docker:**
```bash
docker-compose down
```

---

## 🐛 Troubleshooting

### Segmentation Fault (Exit Code 139)

**Cause**: FAISS HNSW has known issues on macOS with LibreSSL.

**Solutions**:
1. **Use Docker** (recommended): `docker-compose up`
2. **Upgrade OpenSSL**: `brew install openssl@1.1 && brew link openssl@1.1 --force`
3. **Use existing data**: Data is already prepared, no need to rebuild!

### OpenAI API Error

**Cause**: Missing or invalid API key.

**Solution**:
```bash
echo 'OPENAI_API_KEY="sk-..."' > .env
```

### Port Already in Use

**Cause**: Service already running on port 8000 or 8501.

**Solution**:
```bash
./stop.sh
# OR
lsof -ti:8000 | xargs kill
lsof -ti:8501 | xargs kill
```

### urllib3 Warning

**Cause**: System using LibreSSL instead of OpenSSL.

**Solution**: This is just a warning - system will work fine. To fix:
```bash
brew install openssl@1.1
pip uninstall urllib3
pip install urllib3
```

---

## 📚 Documentation

- **QUICKSTART.txt**: Full setup guide
- **DESIGN_LOG.md**: Architecture decisions and trade-offs
- **SEGFAULT_FIX.txt**: Solutions for library conflicts
- **API Docs**: http://localhost:8000/docs (Swagger UI)

---

## 🧪 Running Tests

```bash
# Run all tests
make test

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run evaluation
make eval
```

---

## 📦 Deployment

### Streamlit Cloud (Free)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy `src/ui/app.py`
4. Add `OPENAI_API_KEY` secret

### Docker
```bash
docker-compose up -d
```

### Manual Server
```bash
# Production-ready API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4

# Production-ready UI
streamlit run src/ui/app.py --server.port 8501 --server.headless true
```

---

## 🤝 Contributing

This is a homework assignment project, but feedback is welcome!

---

## 📄 License

MIT License - feel free to use for learning and reference.

---

## 🎉 You're All Set!

Your Disney RAG system is ready to use!

**Access Points:**
- 🎨 **UI**: http://localhost:8501
- 🔌 **API**: http://localhost:8000
- 📊 **API Docs**: http://localhost:8000/docs
- 📈 **Metrics**: http://localhost:8000/metrics

**Next Steps:**
1. Try the example queries in the Streamlit UI
2. Explore the Eval tab for performance insights
3. Check the Testing tab for automated results
4. Read DESIGN_LOG.md for architecture details

Happy querying! 🏰✨

---

**Built with ❤️ for Stampli ML Engineer Home Assignment**
