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

# Stage 2: Runtime image with aggressive cleanup
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Aggressive cleanup to reduce image size
RUN find /root/.local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -name "*.pyc" -delete && \
    find /root/.local -name "*.pyo" -delete && \
    find /root/.local -name "*.dist-info" -type d -exec sh -c 'rm -rf "$1"/RECORD "$1"/INSTALLER 2>/dev/null || true' _ {} \; && \
    find /root/.local -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -name "test" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -name "*.md" -delete && \
    find /root/.local -name "*.txt" -path "*/test*" -delete && \
    find /root/.local -name "*.rst" -delete && \
    find /root/.local -name "LICENSE*" -delete && \
    find /root/.local -name "*.so" -exec strip {} \; 2>/dev/null || true && \
    rm -rf /root/.cache && \
    rm -rf /tmp/* /var/tmp/*

# Copy only essential application code (exclude large files via .dockerignore)
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

# Final cleanup
RUN rm -rf /tmp/* /var/tmp/* && \
    find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

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