# Multi-stage Dockerfile for viewTag Video Search Engine
# Build: docker build -t viewtag .
# Build with GPU: docker build --build-arg ENABLE_GPU=1 -t viewtag:gpu .
# Run: docker run -p 8000:8000 -v $(pwd)/qdrant_data:/app/qdrant_data -v $(pwd)/.env:/app/.env viewtag

# =============================================================================
# BASE STAGE - System dependencies
# =============================================================================
FROM python:3.11-slim AS base

# Install system dependencies
# - ffmpeg: video processing (decord backend)
# - libgl1, libglib2.0-0: OpenCV / decord requirements
# - curl: health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# =============================================================================
# BUILD STAGE - Install Python dependencies
# =============================================================================
FROM base AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install package in editable mode
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e .

# =============================================================================
# GPU STAGE (optional) - Install CUDA-enabled PyTorch
# =============================================================================
FROM builder AS gpu-builder

ARG ENABLE_GPU=0

# Install CUDA-enabled PyTorch if GPU enabled
RUN if [ "$ENABLE_GPU" = "1" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio; \
    fi

# =============================================================================
# RUNTIME STAGE - Minimal production image
# =============================================================================
FROM base AS runtime

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy GPU-enabled torch if built
ARG ENABLE_GPU=0
COPY --from=gpu-builder /opt/venv /opt/venv

# Copy application code
COPY src/ ./src/

# Create directories for data and set permissions
RUN mkdir -p /app/qdrant_data \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
# Uses settings from .env (mounted at runtime) or defaults
CMD ["viewtag-api"]