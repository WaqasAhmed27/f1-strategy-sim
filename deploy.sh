#!/bin/bash

# F1 Prediction System Deployment Script

set -e

echo "🏎️ F1 Prediction System Deployment"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data models configs

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check if the API is responding
echo "🔍 Checking API health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is healthy!"
else
    echo "❌ API health check failed"
    exit 1
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Services:"
echo "  - F1 Prediction API: http://localhost:8000"
echo "  - Web Dashboard: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "🔧 Management commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop services: docker-compose down"
echo "  - Restart services: docker-compose restart"
echo "  - Update services: docker-compose up --build -d"
echo ""
echo "🏁 Ready to predict F1 races!"

