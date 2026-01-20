# Use an official PyTorch image as parent
# 'runtime' is smaller; use 'devel' if you need to compile custom CUDA kernels
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (e.g., for OpenCV or Matplotlib if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/
COPY tests/ ./tests/

# Install dependencies
RUN uv pip install --system .
RUN uv pip install --system pytest

# Default command
CMD ["pytest"]
