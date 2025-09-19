# Perfectware AI - Hugging Face Spaces Deployment Guide

## 🚀 Quick Deploy to Hugging Face Spaces

Your project is now ready for Hugging Face Spaces deployment! Here's how to deploy it:

### Step 1: Create Hugging Face Space

1. **Go to [Hugging Face Spaces](https://huggingface.co/new-space)**
2. **Fill in the details:**
   - **Space name**: `perfectware-ai` (or your preferred name)
   - **License**: MIT
   - **SDK**: Docker
   - **Hardware**: CPU Basic (free tier)
   - **Visibility**: Public (or Private if you prefer)

3. **Click "Create Space"**

### Step 2: Configure Environment Variables

1. **Go to your Space Settings**
2. **Navigate to "Repository secrets"**
3. **Add these secrets:**
   ```
   HUGGINGFACE_API_TOKEN=your_huggingface_token_here
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Step 3: Upload Your Code

**Option A: Using Git (Recommended)**
```bash
# Clone your HF Space repository
git clone https://huggingface.co/spaces/mdikram777/perfectware-ai
cd perfectware-ai

# Copy your project files
cp -r /path/to/your/aiperfect/* .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

**Option B: Using HF Hub**
```bash
# Install huggingface_hub
pip install huggingface_hub

# Upload files
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='.',
    repo_id='mdikram777/perfectware-ai',
    repo_type='space'
)
"
```

### Step 4: Deploy!

1. **Push your code** to the Space repository
2. **Hugging Face will automatically:**
   - Build the Docker image
   - Install all dependencies
   - Start your Streamlit app
   - Make it available at `https://huggingface.co/spaces/mdikram777/perfectware-ai`

## 📁 Project Structure for HF Spaces

```
perfectware-ai/
├── Dockerfile              # Docker configuration
├── app.py                  # Entry point for HF Spaces
├── main.py                 # Main Streamlit application
├── display.py              # Display utilities
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── .dockerignore           # Docker ignore file
├── extracted_content/      # Product images
├── perfectware_products_index/  # FAISS index files
└── product_list.json       # Product data
```

## 🔧 Configuration Files

### Dockerfile
- Uses Python 3.11 slim image
- Installs all dependencies
- Configures Streamlit for port 7860
- Sets up health checks

### app.py
- Entry point for Hugging Face Spaces
- Configures environment variables
- Launches Streamlit with proper settings

### .streamlit/config.toml
- Headless mode for server deployment
- Port 7860 (HF Spaces standard)
- Custom theme configuration

## 🎯 Why Hugging Face Spaces?

✅ **Free Tier Benefits:**
- 16GB RAM (sufficient for your AI models)
- 50GB persistent storage
- Automatic HTTPS
- Custom domain support

✅ **AI-Optimized:**
- Built for ML/AI applications
- Pre-installed ML libraries
- GPU support (paid tiers)
- Model caching

✅ **Easy Deployment:**
- Git-based deployment
- Automatic builds
- Environment variable management
- Built-in monitoring

## 🐛 Troubleshooting

### Common Issues:

1. **Build Fails**
   - Check Dockerfile syntax
   - Ensure all dependencies are in requirements.txt
   - Verify file paths are correct

2. **App Doesn't Start**
   - Check environment variables are set
   - Verify port 7860 is used
   - Check logs in HF Spaces interface

3. **Memory Issues**
   - Consider upgrading to paid tier
   - Optimize model loading
   - Use smaller models if needed

4. **Slow Loading**
   - First load downloads models (5-10 minutes)
   - Subsequent loads are much faster
   - Consider model caching

### Debug Commands:

```bash
# Test locally with Docker
docker build -t perfectware-ai .
docker run -p 7860:7860 -e HUGGINGFACE_API_TOKEN=your_token -e GROQ_API_KEY=your_key perfectware-ai

# Check logs
docker logs <container_id>
```

## 🚀 Success!

Once deployed, your app will be available at:
`https://huggingface.co/spaces/mdikram777/perfectware-ai`

**Features:**
- ✅ Public access
- ✅ HTTPS enabled
- ✅ Mobile responsive
- ✅ Auto-updates on git push
- ✅ Built-in monitoring

## 📞 Support

- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Docker Docs**: https://docs.docker.com/
- **Streamlit Docs**: https://docs.streamlit.io/

---

**Ready to deploy?** Follow the steps above and your AI assistant will be live in minutes! 🎉
