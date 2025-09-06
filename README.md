# Perfectware AI Product Assistant

An intelligent product recommendation system for tiles, sanitary ware, and bathroom accessories using AI-powered search and chat functionality.

## Features

- **AI-Powered Search**: Semantic search through product catalogs using embeddings
- **Chat Interface**: Conversational AI assistant with memory
- **Image Recognition**: Product image analysis and matching
- **PDF Processing**: Automatic extraction of product data from PDF catalogs
- **Enhanced Recommendations**: Multi-factor ranking system for better results

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
   - Add your Hugging Face API token:
   ```bash
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
   ```
   - Get your free token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

5. **Run the application**
   ```bash
   streamlit run display.py
   ```

## Deployment on Render

### Prerequisites
- GitHub repository with the code
- Render account (free tier available)

### Steps to Deploy

1. **Go to [Render Dashboard](https://dashboard.render.com/)**

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select the repository: `DeployPerferctwareAI`

3. **Configure the Service**
   - **Name**: `perfectware-ai-assistant`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run display.py --server.port=$PORT --server.address=0.0.0.0`
   - **Python Version**: `3.11`

4. **Environment Variables**
   - In the Render dashboard, go to "Environment" tab
   - Add: `HUGGINGFACEHUB_API_TOKEN` = `your_huggingface_token_here`
   - Get your free token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

### Important Notes for Render

- **Port Configuration**: The app uses `$PORT` environment variable (required by Render)
- **Memory Requirements**: The app uses ML models, so consider upgrading if you hit memory limits
- **Startup Time**: First deployment may take 5-10 minutes due to model downloads
- **Data Persistence**: The FAISS index and extracted content are included in the repository

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