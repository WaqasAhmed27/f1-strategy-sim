#!/bin/bash

# Railway startup script for F1 Prediction System

echo "🏎️ Starting F1 Prediction System..."

# Set environment variables
export PYTHONPATH="/app/src"
export FASTAPI_ENV="production"

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "📦 Installing dependencies..."
    pip install -r web/backend/requirements.txt
    pip install -e .
fi

# Start the application
echo "🚀 Starting FastAPI server..."
uvicorn web.backend.main:app --host 0.0.0.0 --port $PORT
