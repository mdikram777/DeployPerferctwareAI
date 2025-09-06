#!/usr/bin/env python3
"""
Setup script for Perfectware AI Product Assistant
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Perfectware AI Product Assistant")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 11:
        print(f"❌ Python 3.11+ required. Current version: {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Check if Ollama is installed
    if not run_command("ollama --version", "Checking Ollama installation"):
        print("⚠️  Ollama not found. Please install Ollama from https://ollama.ai")
        print("   After installation, run: ollama pull llama3.2")
        return False
    
    # Check if llama3.2 model is available
    if not run_command("ollama list | grep llama3.2", "Checking llama3.2 model"):
        print("📥 Downloading llama3.2 model...")
        if not run_command("ollama pull llama3.2", "Downloading llama3.2 model"):
            return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Place your product catalog PDFs in the current directory")
    print("2. Run: python main.py *.pdf (to process PDFs)")
    print("3. Run: streamlit run display.py (to start the web app)")
    print("\n💡 Tips:")
    print("- Make sure Ollama is running: ollama serve")
    print("- The app works best with 5-6 product catalog PDFs")
    print("- PDFs should contain product images and descriptions")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 