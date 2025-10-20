#!/bin/bash

# Disney RAG System - Stop Script

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}🛑 Stopping Disney RAG System...${NC}"
echo ""

# Stop API server
if [ -f logs/api.pid ]; then
    API_PID=$(cat logs/api.pid)
    if ps -p $API_PID > /dev/null 2>&1; then
        kill $API_PID 2>/dev/null
        echo -e "${GREEN}✅ API server stopped (PID: $API_PID)${NC}"
    else
        echo -e "${BLUE}ℹ️  API server not running${NC}"
    fi
    rm logs/api.pid
else
    # Try to find and kill by process name
    pkill -f "uvicorn src.api.app:app" 2>/dev/null && echo -e "${GREEN}✅ API server stopped${NC}"
fi

# Stop Streamlit UI
if [ -f logs/streamlit.pid ]; then
    UI_PID=$(cat logs/streamlit.pid)
    if ps -p $UI_PID > /dev/null 2>&1; then
        kill $UI_PID 2>/dev/null
        echo -e "${GREEN}✅ Streamlit UI stopped (PID: $UI_PID)${NC}"
    else
        echo -e "${BLUE}ℹ️  Streamlit UI not running${NC}"
    fi
    rm logs/streamlit.pid
else
    # Try to find and kill by process name
    pkill -f "streamlit run src/ui/app.py" 2>/dev/null && echo -e "${GREEN}✅ Streamlit UI stopped${NC}"
fi

echo ""
echo -e "${GREEN}All services stopped successfully!${NC}"
echo ""

