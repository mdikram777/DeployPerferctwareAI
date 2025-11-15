# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories and config file
RUN mkdir -p .streamlit

# Create streamlit config file
RUN echo '[server]' > .streamlit/config.toml && \
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

# Expose port (Railway uses PORT env variable)
EXPOSE ${PORT:-7860}

# Set environment variables
ENV STREAMLIT_SERVER_PORT=${PORT:-7860}
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

# Run the application (Railway will set PORT env variable)
CMD sh -c "streamlit run display.py --server.port=\${PORT:-7860} --server.address=0.0.0.0"
