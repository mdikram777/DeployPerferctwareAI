#!/bin/bash

# Load environment variables
export HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

# Activate virtual environment
source perfectware_env_311/bin/activate

# Run the app
streamlit run display.py --server.headless true --server.port 8501
