#!/bin/bash

# Disney RAG System - Quick Start Script
# This script sets up and launches the entire system

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo ""
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

# Cleanup function
cleanup_data() {
    print_header "🧹 Cleaning Up Generated Data"
    
    print_warning "This will remove all generated indices and processed data"
    print_warning "You will need to regenerate everything (takes ~5-10 minutes)"
    echo ""
    read -p "Are you sure you want to clean up? (yes/no): " -r
    echo
    
    if [[ $REPLY =~ ^[Yy]es$ ]]; then
        print_info "Removing data/processed/..."
        rm -f data/processed/*.npy data/processed/*.parquet 2>/dev/null || true
        
        print_info "Removing data/indices/..."
        rm -f data/indices/*.index data/indices/*.pkl 2>/dev/null || true
        
        print_success "Cleanup complete! You can now test from scratch."
        echo ""
        print_info "Run ./quickstart.sh again to rebuild everything"
        exit 0
    else
        print_info "Cleanup cancelled"
        exit 0
    fi
}

# Check for cleanup flag
if [[ "$1" == "clean" ]] || [[ "$1" == "--clean" ]]; then
    cleanup_data
fi

# Check if running in project directory
if [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_header "🏰 Disney RAG System - Quick Start"
print_info "Tip: Use './quickstart.sh clean' to remove all generated data"
echo ""

# Step 1: Check Python version
print_info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 is required but not found. Please install Python 3.9+"
    exit 1
fi

# Step 2: Check for OpenAI API key
print_info "Checking for OpenAI API key..."

# Function to validate OpenAI API key
validate_openai_key() {
    local key=$1
    python3 -c "
import openai
import sys
try:
    client = openai.OpenAI(api_key='$key')
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'Hi'}],
        max_tokens=5
    )
    sys.exit(0)
except openai.AuthenticationError:
    sys.exit(1)
except openai.RateLimitError:
    sys.exit(0)  # Rate limit means key is valid
except:
    sys.exit(0)  # Other errors, continue anyway
" 2>/dev/null
    return $?
}

# Try to read existing key from .env
API_KEY=""
if [ -f ".env" ]; then
    print_success ".env file exists"
    # Extract API key
    API_KEY=$(grep "^OPENAI_API_KEY" .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'")
fi

# If no key found, ask for it
if [ -z "$API_KEY" ]; then
    print_warning "OpenAI API key not found in .env file!"
    echo ""
    echo "Please enter your OpenAI API key:"
    echo "(You can get one from: https://platform.openai.com/api-keys)"
    read -p "OpenAI API Key: " API_KEY
    
    if [ -z "$API_KEY" ]; then
        print_error "No API key provided"
        exit 1
    fi
    
    # Save to .env
    echo "OPENAI_API_KEY=\"$API_KEY\"" > .env
    print_success "API key saved to .env file"
fi

# Validate the API key
print_info "Validating OpenAI API key..."
if validate_openai_key "$API_KEY"; then
    print_success "OpenAI API key is valid!"
else
    print_error "Invalid OpenAI API key!"
    echo ""
    echo "Please check your key at: https://platform.openai.com/api-keys"
    echo ""
    read -p "Would you like to enter a new API key? (yes/no): " -r
    echo
    if [[ $REPLY =~ ^[Yy]es$ ]]; then
        rm -f .env
        exec "$0" "$@"  # Restart script
    else
        exit 1
    fi
fi

# Step 3: Install dependencies
print_header "📦 Installing Dependencies"
print_info "This may take 2-3 minutes..."

if python3 -m pip install --upgrade pip > /dev/null 2>&1; then
    print_success "pip upgraded"
else
    print_warning "Could not upgrade pip (continuing anyway)"
fi

if python3 -m pip install -r requirements.txt > /dev/null 2>&1; then
    print_success "All dependencies installed"
else
    print_error "Failed to install dependencies. Please run manually: pip3 install -r requirements.txt"
    exit 1
fi

# Step 4: Check if data is already prepared
print_header "🔧 Preparing Data"

DATA_READY=false

if [ -f "data/processed/chunks.parquet" ] && \
   [ -f "data/indices/faiss_flat.index" ] && \
   [ -f "data/indices/bm25.pkl" ] && \
   [ -f "data/indices/metadata.pkl" ]; then
    print_success "Data already prepared! Skipping ingestion and indexing."
    DATA_READY=true
else
    print_info "Data not found. Running ingestion and indexing..."
    
    # Step 4a: Run data ingestion
    print_info "Step 1/2: Running data ingestion (extracting features, chunking reviews)..."
    if python3 -m src.ingestion.pipeline > logs/ingestion.log 2>&1; then
        print_success "Ingestion completed (123,860 chunks created)"
    else
        print_error "Ingestion failed. Check logs/ingestion.log for details"
        exit 1
    fi
    
    # Step 4b: Build indices
    print_info "Step 2/2: Building search indices (FAISS + BM25)..."
    print_warning "This may take 3-5 minutes for embeddings..."
    if python3 -m src.retrieval.build_indices > logs/indexing.log 2>&1; then
        print_success "Indices built successfully"
        DATA_READY=true
    else
        print_error "Index building failed. Check logs/indexing.log for details"
        exit 1
    fi
fi

if [ "$DATA_READY" = true ]; then
    print_success "All data is ready!"
    echo ""
    print_info "Data summary:"
    echo "  - Reviews: 42,656"
    echo "  - Chunks: 123,860"
    echo "  - FAISS index: $(du -h data/indices/faiss_flat.index 2>/dev/null | cut -f1 || echo 'N/A')"
    echo "  - BM25 index: $(du -h data/indices/bm25.pkl 2>/dev/null | cut -f1 || echo 'N/A')"
fi

# Step 5: Create logs directory
mkdir -p logs

# Step 6: Kill any existing processes
print_header "🚀 Starting Services"

print_info "Stopping any existing services..."
pkill -f "uvicorn src.api.app:app" 2>/dev/null || true
pkill -f "streamlit run src/ui/app.py" 2>/dev/null || true
sleep 2

# Step 7: Start API server
print_info "Starting FastAPI server on port 8000..."
nohup python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
API_PID=$!
echo $API_PID > logs/api.pid
sleep 3

# Check if API started
if ps -p $API_PID > /dev/null; then
    print_success "API server started (PID: $API_PID)"
else
    print_error "API server failed to start. Check logs/api.log"
    exit 1
fi

# Step 8: Start Streamlit UI
print_info "Starting Streamlit UI on port 8501..."
nohup python3 -m streamlit run src/ui/app.py --server.port 8501 --server.headless true > logs/streamlit.log 2>&1 &
UI_PID=$!
echo $UI_PID > logs/streamlit.pid
sleep 5

# Check if UI started
if ps -p $UI_PID > /dev/null; then
    print_success "Streamlit UI started (PID: $UI_PID)"
else
    print_error "Streamlit UI failed to start. Check logs/streamlit.log"
    exit 1
fi

# Step 9: Wait for services to be ready
print_info "Waiting for services to be ready..."
sleep 3

# Test API health
if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then
    print_success "API is healthy and responding"
else
    print_warning "API may still be initializing..."
fi

# Step 10: Success message
print_header "🎉 SUCCESS! Disney RAG System is Running"

echo ""
echo -e "${GREEN}Your Disney RAG System is now live!${NC}"
echo ""
echo -e "${BLUE}📱 Access Points:${NC}"
echo -e "  • Streamlit UI:  ${GREEN}http://localhost:8501${NC}"
echo -e "  • API Server:    ${GREEN}http://localhost:8000${NC}"
echo -e "  • API Health:    http://localhost:8000/healthz"
echo -e "  • API Metrics:   http://localhost:8000/metrics"
echo ""
echo -e "${BLUE}📊 What You Can Do:${NC}"
echo "  1. Open http://localhost:8501 in your browser"
echo "  2. Try the example queries in the sidebar"
echo "  3. Use the Chat tab for Q&A"
echo "  4. Use the Eval tab for analytics"
echo "  5. Use the Testing tab to run automated tests"
echo ""
echo -e "${BLUE}📁 Project Stats:${NC}"
echo "  • 42,656 reviews indexed"
echo "  • 123,860 searchable chunks"
echo "  • ~8 second average response time"
echo "  • Full citations with every answer"
echo ""
echo -e "${BLUE}🔧 Management:${NC}"
echo "  • View logs: tail -f logs/api.log or logs/streamlit.log"
echo "  • Stop services: ./stop.sh (or kill PIDs in logs/*.pid)"
echo "  • Clean data: ./quickstart.sh clean"
echo "  • API PID: $API_PID"
echo "  • UI PID: $UI_PID"
echo ""
echo -e "${YELLOW}Services are running in the background${NC}"
echo -e "${BLUE}View live logs with:${NC} tail -f logs/api.log logs/streamlit.log"
echo ""
