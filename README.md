# Perfectware AI Product Assistant

An intelligent product recommendation system for tiles, sanitary ware, and bathroom accessories using AI-powered search and chat functionality.

## Features

- **AI-Powered Search**: Semantic search through product catalogs using embeddings
- **Chat Interface**: Conversational AI assistant with memory
- **Image Recognition**: Product image analysis and matching
- **PDF Processing**: Automatic extraction of product data from PDF catalogs
- **Enhanced Recommendations**: Multi-factor ranking system for better results

## 🚀 Live Demo

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/mdikram777/perfectware-ai)

## Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mdikram777/DeployPerferctwareAI.git
   cd DeployPerferctwareAI
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Create a `.env` file in the root directory
   - Add your API tokens:
   ```bash
   HUGGINGFACE_API_TOKEN=your_huggingface_token_here
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

## 🐳 Deployment on Hugging Face Spaces

### Prerequisites
- GitHub repository with the code
- Hugging Face account (free)
- Docker (for local testing)

### Steps to Deploy

1. **Go to [Hugging Face Spaces](https://huggingface.co/new-space)**

2. **Create New Space**
   - **Space name**: `perfectware-ai`
   - **License**: MIT
   - **SDK**: Docker
   - **Hardware**: CPU Basic (free) or upgrade if needed

3. **Configure Environment Variables**
   - Go to Settings → Repository secrets
   - Add: `HUGGINGFACE_API_TOKEN` = `your_huggingface_token_here`
   - Add: `GROQ_API_KEY` = `your_groq_api_key_here`

4. **Deploy**
   - Push your code to the space repository
   - Hugging Face will automatically build and deploy using Docker

### Important Notes for Hugging Face Spaces

- **Port**: Uses port 7860 (automatically configured)
- **Memory**: Free tier has 16GB RAM (sufficient for this app)
- **Storage**: 50GB persistent storage
- **Build Time**: First deployment takes 5-10 minutes
- **Auto-deploy**: Updates automatically when you push to main branch

## File Structure

```
├── display.py              # Main Streamlit application
├── main.py                 # PDF processing and data extraction
├── requirements.txt        # Python dependencies
├── product_list.json       # Structured product data
├── perfectware_products_index/  # FAISS vector database
├── extracted_content/      # Processed images and text
└── README.md              # This file
```

## How It Works

1. **PDF Processing**: `main.py` extracts product information from PDF catalogs
2. **Vector Database**: Creates embeddings using sentence transformers
3. **Web Interface**: `display.py` provides the Streamlit UI
4. **AI Search**: Uses semantic similarity to find relevant products
5. **Chat Memory**: Maintains conversation context for better recommendations

## Customization

- **Product Data**: Update `product_list.json` or add new PDFs
- **Styling**: Modify CSS in `display.py`
- **AI Model**: Change the embedding model in `main.py`
- **Search Logic**: Adjust ranking algorithms in `enhanced_product_search()`

## Troubleshooting

- **Memory Issues**: Consider using smaller models or upgrading Render plan
- **Slow Startup**: Models are downloaded on first run
- **Search Not Working**: Ensure FAISS index files are present
- **PDF Processing**: Check if PDF files are accessible and readable

## Support

For issues or questions, please create an issue in the GitHub repository.