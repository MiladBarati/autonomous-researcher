# Dockerfile for Autonomous Research Assistant
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and necessary source files for installation
COPY pyproject.toml README.md ./
COPY agent/ ./agent/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Install Playwright browsers (required for web scraping)
RUN playwright install --with-deps chromium

# Copy remaining application code
COPY . .

# Create directories for persistent data
RUN mkdir -p /app/chroma_db /app/logs

# Expose Streamlit port
EXPOSE 8501

# Health check (check if Streamlit port is accessible)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 8501)); s.close()" || exit 1

# Default command (can be overridden in docker-compose)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
