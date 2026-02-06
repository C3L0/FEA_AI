# Use an official NVIDIA CUDA image with Python support
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (GMSH, OpenMPI for FEniCS, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libglu1-mesa \
    libxcursor1 \
    libxinerama1 \
    libxft2 \
    libfltk1.3 \
    libgmsh-dev \
    gmsh \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set work directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies using uv
# We sync the environment based on the pyproject.toml and uv.lock
RUN uv sync --frozen

# Expose Streamlit port
EXPOSE 8501

# Command to launch the inference app by default
ENTRYPOINT ["uv", "run", "streamlit", "run", "src/fea_gnn/app.py", "--server.address=0.0.0.0"]
