# H. pylori RL Simulator Docker Image

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including OpenGL/OSMesa for 3D rendering
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libosmesa6 \
    libosmesa6-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    freeglut3-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/backend
COPY ../configs /app/configs

# Create necessary directories
RUN mkdir -p /app/data /app/checkpoints /app/logs /app/reports

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health')"

# Run API server
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

