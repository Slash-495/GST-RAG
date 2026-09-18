# ==============================================================================
# Production Dockerfile for GST RAG FastAPI Backend
# Optimized for Ubuntu Server Deployment
# ==============================================================================

# Use official lightweight Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables:
# - PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
# - PYTHONUNBUFFERED: Keeps stdout/stderr unbuffered for immediate real-time logging
# - PORT: Default listening port
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory inside the container
WORKDIR /app

# Install system dependencies:
# - build-essential: C/C++ compiler toolchain if wheels require compilation
# - libgomp1: OpenMP runtime library essential for faiss-cpu on Linux
# - curl: Required for Docker container health check probes
# - ca-certificates: Secure HTTPS communication with APIs (Google Gemini, Cohere, Supabase, AWS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy only requirements first to optimize Docker layer caching
COPY requirements.txt .

# Install Python dependencies without storing cached wheel files
RUN pip install --no-cache-dir -r requirements.txt

# Create a dedicated non-root application user for secure production execution
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

# Copy the rest of the application codebase
COPY --chown=appuser:appuser . .

# Switch to the non-root user
USER appuser

# Expose port 8000 for external traffic
EXPOSE 8000

# Health check to ensure the FastAPI server is responding properly
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
