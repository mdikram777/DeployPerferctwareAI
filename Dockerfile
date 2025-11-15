# Multi-stage build to reduce image size
# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# Install PyTorch CPU-only first (much smaller: ~500MB vs ~2GB for full PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt && \
    pip cache purge

# Stage 2: Runtime image
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Clean up unnecessary files to reduce image size
RUN find /root/.local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -name "*.pyc" -delete && \
    find /root/.local -name "*.pyo" -delete

# Copy application code
COPY . .

# Create streamlit config
RUN mkdir -p .streamlit && \
    echo '[server]' > .streamlit/config.toml && \
    echo 'headless = true' >> .streamlit/config.toml && \
    echo 'port = 7860' >> .streamlit/config.toml && \
    echo 'enableCORS = false' >> .streamlit/config.toml && \
    echo 'enableXsrfProtection = false' >> .streamlit/config.toml && \
    echo '' >> .streamlit/config.toml && \
    echo '[browser]' >> .streamlit/config.toml && \
    echo 'gatherUsageStats = false' >> .streamlit/config.toml && \
    echo '' >> .streamlit/config.toml && \
    echo '[theme]' >> .streamlit/config.toml && \
    echo 'primaryColor = "#FF6B6B"' >> .streamlit/config.toml && \
    echo 'backgroundColor = "#FFFFFF"' >> .streamlit/config.toml && \
    echo 'secondaryBackgroundColor = "#F0F2F6"' >> .streamlit/config.toml && \
    echo 'textColor = "#262730"' >> .streamlit/config.toml

# Expose port
EXPOSE ${PORT:-7860}

# Set environment variables
ENV STREAMLIT_SERVER_PORT=${PORT:-7860}
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-7860}/_stcore/health || exit 1

# Run the application
CMD sh -c "streamlit run display.py --server.port=\${PORT:-7860} --server.address=0.0.0.0"