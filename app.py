import subprocess
import sys
import os

# Get port from environment variable (Railway sets PORT, default to 7860)
port = os.environ.get("PORT", "7860")

# Set environment variables
os.environ["STREAMLIT_SERVER_PORT"] = port
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# Run the main Streamlit app
if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "display.py",
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ])
