# Quick Start Guide - Perfectware AI Product Assistant

## 🚀 Get Started in 5 Minutes

### Step 1: Environment Setup
```bash
# Create Python 3.11 environment
conda create -n perfectware python=3.11
conda activate perfectware

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama and download model
ollama serve
# In another terminal:
ollama pull llama3.2
```

### Step 3: Test Your Setup
```bash
# Run the test script
python test_setup.py
```

### Step 4: Try the Demo
```bash
# Run the demo to see the system in action
python demo.py
```

### Step 5: Process Your PDFs
```bash
# Place your product catalog PDFs in the directory
# Then run:
python main.py *.pdf
```

### Step 6: Launch Web App
```bash
# Start the Streamlit web application
streamlit run display.py
```

## 📋 What's New in This Version

### Enhanced Features:
- ✅ **Image Context Integration**: Products now include image captions for better recommendations
- ✅ **Improved UI**: Modern, responsive design with better product cards
- ✅ **PDF Uploader**: Upload additional catalogs directly in the web interface
- ✅ **Enhanced Search**: Better semantic search with image context boosting
- ✅ **Professional Recommendations**: AI provides structured, sales-focused responses

### Technical Improvements:
- ✅ **RecursiveCharacterTextSplitter**: Better text processing for PDFs
- ✅ **Image Caption Extraction**: Automatically captures text near images
- ✅ **Base64 Image Storage**: Images are stored for web display
- ✅ **Error Handling**: Robust error handling throughout the system
- ✅ **Performance Optimization**: Faster search and processing

## 🎯 Key Features

### 1. Smart Product Search
- Natural language queries
- Semantic matching
- Image context boosting
- Category-specific optimization

### 2. AI-Powered Recommendations
- Professional sales consultant persona
- Context-aware responses
- Structured product recommendations
- Company information integration

### 3. Modern Web Interface
- Beautiful, responsive design
- Product cards with images
- Real-time search
- PDF upload functionality

### 4. PDF Processing
- Automatic image extraction
- Text and image linking
- Product information extraction
- Duplicate removal

## 🔧 Troubleshooting

### Common Issues:

1. **"Ollama not found"**
   ```bash
   # Install Ollama first
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **"Model not found"**
   ```bash
   # Download the required model
   ollama pull llama3.2
   ```

3. **"Import errors"**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

4. **"Memory issues"**
   - Close other applications
   - Process PDFs in smaller batches
   - Use smaller embedding models

### Performance Tips:
- Process 5-6 PDFs at once for best results
- Ensure PDFs contain extractable text and images
- Use SSD storage for faster processing
- Close unnecessary applications during processing

## 📱 Usage Examples

### Search Queries:
- "Anti-skid outdoor tiles for garden area"
- "Modern bathroom sink with faucet"
- "Ceramic wall tiles for kitchen backsplash"
- "Luxury bathroom accessories"
- "Durable floor tiles for commercial use"

### Expected Results:
- Relevant product recommendations
- Detailed specifications
- Application suggestions
- Quality highlights
- Professional sales pitch

## 🎨 Customization

### Company Information:
Edit `COMPANY_INFO` in both `main.py` and `display.py`:
```python
COMPANY_INFO = {
    "name": "Your Company Name",
    "established": "Year",
    "experience": "X+ years",
    "location": "Your Location",
    "specialties": ["product1", "product2"],
    "target_customers": ["customer1", "customer2"],
    "values": ["value1", "value2"],
    "tagline": "Your Tagline"
}
```

### UI Customization:
- Modify CSS in `display.py`
- Update color schemes
- Add custom components
- Adjust layouts

## 📞 Support

If you encounter issues:
1. Run `python test_setup.py` to diagnose problems
2. Check the troubleshooting section above
3. Verify all dependencies are installed
4. Ensure Ollama is running and accessible

## 🎉 Success Indicators

You'll know everything is working when:
- ✅ `python test_setup.py` shows all green checkmarks
- ✅ `python demo.py` runs without errors
- ✅ `streamlit run display.py` opens the web interface
- ✅ You can search for products and get AI recommendations
- ✅ Product cards display with images and information

---

**Ready to revolutionize your product recommendations? Start with the demo and then process your own catalogs!** 🚀 