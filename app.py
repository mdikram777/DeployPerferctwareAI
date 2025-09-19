import subprocess
import sys
import os

# Set environment variables for Hugging Face Spaces
os.environ["STREAMLIT_SERVER_PORT"] = "7860"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# Run the main Streamlit app
if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "main.py",
        "--server.port=7860",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ])
