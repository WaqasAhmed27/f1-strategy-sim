# Multi-stage Docker build for F1 Prediction System

# Stage 1: Build Frontend
FROM node:18-alpine AS frontend-build
WORKDIR /app/frontend
COPY web/frontend/package*.json ./
RUN npm install
COPY web/frontend/ ./
RUN npm run build

# Stage 2: Build Backend
FROM python:3.11-slim AS backend-build
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY web/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . ./

# Install the f1sim package
RUN pip install -e .

# Stage 3: Production Image
FROM python:3.11-slim AS production
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from backend build
COPY --from=backend-build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-build /usr/local/bin /usr/local/bin

# Copy application code
COPY . ./
COPY --from=frontend-build /app/frontend/build ./web/frontend/build

# Install the f1sim package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 f1user && chown -R f1user:f1user /app
USER f1user

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["uvicorn", "web.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

