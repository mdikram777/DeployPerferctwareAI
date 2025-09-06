import streamlit as st
# from langchain_community.llms import HuggingFaceHub  # Deprecated
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import base64
import re
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Debug: Check if token is loaded
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if token:
    print(f"✅ Token loaded successfully: {token[:10]}...")
else:
    print("❌ Token not found in environment variables")

# Company information
COMPANY_INFO = {
    "name": "Perfectware Buildings Private Limited",
    "established": "2008",
    "experience": "15+ years",
    "location": "Tamil Nadu",
    "specialties": ["tiles", "sanitary ware", "bathroom accessories"],
    "target_customers": ["homeowners", "contractors", "interior designers"],
    "values": ["quality", "customer satisfaction", "innovation", "durability", "affordability"],
    "tagline": "Your Trusted Partner for Premium Tiles & Sanitary Solutions"
}

# Helper functions for chat history and enhanced search
def initialize_chat_history():
    """Initialize chat history in session state if not exists"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def get_conversation_context(history: list) -> str:
    """Extract context from conversation history"""
    if not history:
        return "No previous conversation."
    
    context = "Previous conversation:\n"
    for i, msg in enumerate(history[-5:]):  # Keep last 5 exchanges for context
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role}: {msg['content']}\n"
    
    return context

def enhanced_product_search(query: str, vs, k: int = 10, history: list = None) -> list:
    """Enhanced product search with history context and better ranking"""
    if not vs or embeddings is None:
        return []
    
    try:
        # Get more results initially
        docs = vs.similarity_search(query, k=15)
        
        # If we have history, use it to refine results
        if history and len(history) > 0:
            # Extract previous product mentions to avoid repetition
            previous_products = set()
            for msg in history:
                if msg["role"] == "assistant":
                    # Simple pattern to extract product names (could be improved)
                    products = re.findall(r'\b([A-Z][a-zA-Z\s]{3,}(?:\s+(?:HL|FP|VC|Décor|Decor)\s*\d*[A-Z]*)?)\b', msg["content"])
                    previous_products.update(products)
            
            # Filter out previously mentioned products
            if previous_products:
                filtered_docs = []
                for doc in docs:
                    product_name = doc.metadata.get("product_name", "")
                    if product_name and product_name not in previous_products:
                        filtered_docs.append(doc)
                docs = filtered_docs[:k]  # Still keep up to k results
        
        # Enhanced scoring with multiple factors
        scored_docs = []
        query_embedding = embeddings.embed_query(query)
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for doc in docs:
            score = 1.0
            metadata = doc.metadata
            
            # 1. Semantic similarity (original score)
            doc_embedding = embeddings.embed_query(doc.page_content)
            semantic_similarity = cosine_similarity(
                [query_embedding], [doc_embedding]
            )[0][0]
            score *= (1 + semantic_similarity)
            
            # 2. Image availability boost
            if metadata.get("has_image", False):
                score *= 1.5
            
            # 3. Keyword matching in product name
            product_name = metadata.get("product_name", "").lower()
            if product_name:
                name_words = set(product_name.split())
                common_words = query_words.intersection(name_words)
                if common_words:
                    score *= (1 + len(common_words) * 0.3)
            
            # 4. Feature relevance
            features = metadata.get("features", [])
            feature_text = ' '.join(features).lower()
            feature_matches = sum(1 for word in query_words if word in feature_text)
            if feature_matches > 0:
                score *= (1 + feature_matches * 0.2)
            
            # 5. Application relevance
            applications = metadata.get("applications", [])
            app_text = ' '.join(applications).lower()
            app_matches = sum(1 for word in query_words if word in app_text)
            if app_matches > 0:
                score *= (1 + app_matches * 0.2)
            
            scored_docs.append((doc, score))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]
    
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

# Initialize models
@st.cache_resource
def load_models():
    # Check if token exists
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        st.warning("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
        return None, None
    
    # Use Hugging Face Hub API (free tier available)
    # Note: HuggingFace API may have connectivity issues, so we'll rely on fallback
    try:
        from langchain_huggingface import HuggingFaceEndpoint
        
        llm = HuggingFaceEndpoint(
            repo_id="microsoft/DialoGPT-small",
            temperature=0.7,
            huggingfacehub_api_token=token
        )
        # Test the model to ensure it works
        test_response = llm.invoke("test")
        print("✅ AI model is working")
    except Exception as e:
        print(f"⚠️ AI model unavailable: {str(e)}")
        print("🔄 Using fallback response system")
        llm = None
    
    try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )
        print("✅ Embeddings model loaded successfully")
    except Exception as e:
        print(f"⚠️ Embeddings model unavailable: {str(e)}")
        print("🔄 Using fallback embeddings system")
        # Create a simple fallback embeddings class
        class FallbackEmbeddings:
            def embed_query(self, text):
                # Simple hash-based embedding as fallback
                import hashlib
                hash_obj = hashlib.md5(text.encode())
                # Convert to 384-dimensional vector (same as sentence-transformers)
                hash_bytes = hash_obj.digest()
                # Repeat and pad to 384 dimensions
                embedding = []
                for i in range(384):
                    embedding.append(hash_bytes[i % len(hash_bytes)] / 255.0)
                return embedding
            
            def embed_documents(self, texts):
                return [self.embed_query(text) for text in texts]
        
        embeddings = FallbackEmbeddings()
    
    return llm, embeddings

llm, embeddings = load_models()

# Enhanced page configuration
st.set_page_config(
    page_title="Perfectware Buildings - AI Product Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ENHANCED CSS WITH BETTER IMAGE HANDLING
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Company Header */
    .company-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .company-name {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .company-tagline {
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 400;
        margin-bottom: 1rem;
    }
    
    .company-stats {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    
    .stat-item {
        background: rgba(255,255,255,0.15);
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .stat-number {
        font-size: 1.2rem;
        font-weight: 600;
        display: block;
    }
    
    .stat-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    /* Enhanced Product Cards */
    .product-card {
        background: white;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .product-header {
        display: flex;
        gap: 1.5rem;
        padding: 1.5rem;
        align-items: flex-start;
    }
    
    .product-image-container {
        flex: 0 0 250px;
        height: 250px;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        position: relative;
        background: #f8f9fa;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .product-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.3s ease;
    }
    
    .product-image-container:hover .product-image {
        transform: scale(1.05);
    }
    
    .no-image-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 100%;
        color: #6c757d;
        background: linear-gradient(135deg, #f1f3f5 0%, #e9ecef 100%);
        padding: 1rem;
        text-align: center;
    }
    
    .no-image-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.7;
    }
    
    .product-text-content {
        flex: 1;
    }
    
    .product-name {
        font-size: 1.8rem;
        font-weight: 700;
        color: #000 !important;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }
    
    .product-size {
        color: #667eea;
        font-weight: 500;
        margin-bottom: 1rem;
        font-size: 1rem;
    }
    
    .product-category-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    .product-description {
        color: #5a6c7d;
        line-height: 1.6;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
    
    .product-features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        font-size: 0.9rem;
        border-left: 3px solid #667eea;
    }
    
    .image-caption {
        font-size: 0.85rem;
        color: #6c757d;
        font-style: italic;
        margin: 0.5rem 1.5rem 1.5rem;
        padding: 0.8rem;
        background: #f9f9f9;
        border-radius: 8px;
        border-left: 3px solid #adb5bd;
    }
    
    /* PDF uploader styling */
    .uploaded-pdf {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .pdf-icon {
        font-size: 2rem;
        color: #e74c3c;
    }
    
    /* Search bar styling */
    .search-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 5px 25px rgba(0,0,0,0.08);
    }
    
    .search-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .search-subtitle {
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 1.5rem;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Recommendation Section */
    .recommendation-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border-left: 5px solid #667eea;
    }
    
    .recommendation-title {
        color: #2c3e50;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .recommendation-content {
        color: #495057;
        line-height: 1.7;
        font-size: 1rem;
    }
    
    .recommended-product {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 3px 15px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
    }
    
    .recommended-product-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .recommended-product-image {
        width: 120px;
        height: 120px;
        border-radius: 10px;
        object-fit: cover;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .recommended-product-name {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
    }
    
    .recommended-features {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
    }
    
    .recommended-feature {
        background: #f1f3f9;
        padding: 0.6rem;
        border-radius: 8px;
        font-size: 0.9rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .company-name {
            font-size: 2rem;
        }
        
        .company-stats {
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }
        
        .product-header {
            flex-direction: column;
            gap: 1.5rem;
        }
        
        .product-image-container {
            flex: 0 0 auto;
            width: 100%;
            height: 300px;
        }
        
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load vector store with error handling
@st.cache_resource
def load_vector_store():
    try:
        if os.path.exists("perfectware_products_index"):
            if embeddings is None:
                st.warning("⚠️ Embeddings not available, using fallback search")
                return None
            
            vs = FAISS.load_local(
                "perfectware_products_index", 
                embeddings=embeddings, 
                allow_dangerous_deserialization=True
            )
            print("✅ Vector store loaded successfully")
            return vs
        else:
            st.error("❌ No product catalog index found. Please process your product catalog PDFs first using main.py")
            return None
    except Exception as e:
        st.error(f"❌ Error loading product catalog: {e}")
        return None

vs = load_vector_store()

# UPDATED PROMPT TEMPLATE WITH IMAGE CONTEXT
prompt_template = """
You are a professional sales consultant for {company_name}, {tagline}.

Company Information:
• Established: {established} with {experience} of experience
• Location: {location}
• Specialties: {specialties}
• Target Customers: {customers}
• Values: {values}

Product Information:
{context}

Customer Inquiry: {question}

IMPORTANT: If the customer asks about topics unrelated to tiles, sanitary ware, or bathroom accessories, politely redirect them by saying: "I'd be happy to help you with tile and bathroom product recommendations! Could you please ask about tiles, sanitary ware, or bathroom accessories instead? For example, you could ask about specific tile types, sizes, colors, or applications."

As a knowledgeable product consultant:
1. Provide warm, professional greeting
2. Recommend specific products matching the query
3. Highlight key features and benefits
4. Mention size and technical specifications
5. Include applications and styling suggestions
6. Reference image context when available: [Image Context: {image_context}]
7. Always include product names exactly as they appear in the catalog
8. For additional matching products, provide brief explanations of why they were suggested

Structure your response with:
- Product recommendations first
- Detailed specifications
- Suggested applications
- Quality and durability highlights
- Brief explanations for additional matching products

Professional Recommendation:
"""

@st.cache_resource
def create_qa_chain():
    if llm is None:
        return None
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )

qa_chain = create_qa_chain()



# FUNCTION TO DISPLAY PRODUCT CARDS
def display_product(doc):
    metadata = doc.metadata
    image_base64 = metadata.get("image_base64", None)
    caption = metadata.get("image_caption", "")
    product_name = metadata.get("product_name", "")
    
    # Prepare image HTML
    if image_base64:
        image_html = f"<img src='data:image/jpeg;base64,{image_base64}' class='product-image' />"
    else:
        image_html = """
        <div class='no-image-placeholder'>
            <div class='no-image-icon'>🖼️</div>
            <div>Image Not Available</div>
        </div>
        """
    
    # Prepare caption HTML (removed div tags)
    caption_html = ""
    if caption:
        caption_text = caption[:250] + ('...' if len(caption) > 250 else '')
        caption_html = f"**Image Context:** {caption_text}"
    
    # Prepare features HTML (removed div tags)
    features = metadata.get('features', [])[:6]
    features_html = ' • '.join(features) if features else ""
    
    st.markdown(f"""
    <div class="product-card">
        <div class="product-header">
            <div class="product-image-container">
                {image_html}
            </div>
            <div class="product-text-content">
                <h3 class="product-name">{product_name}</h3>
                <div class="product-size"><strong>Size:</strong> {metadata.get('size', 'N/A')}</div>
                <div class="product-category-badge">{metadata.get('product_category', '').replace('_', ' ').title()}</div>
                <p class="product-description">{metadata.get('description', '')[:300] + ('...' if len(metadata.get('description', '')) > 300 else '')}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display caption and features without div tags
    if caption_html:
        st.markdown(caption_html)
    
    if features_html:
        st.markdown(f"**Features:** {features_html}")

# EXTRACT RECOMMENDED PRODUCTS FROM RESPONSE
def extract_recommended_products(response, all_docs):
    """Extract recommended products with better name matching"""
    recommended_docs = []
    response_lower = response.lower()
    
    # Look for product names mentioned in the response
    for doc in all_docs:
        product_name = doc.metadata.get("product_name", "")
        if not product_name:
            continue
            
        # Check various forms of the product name
        name_variations = [
            product_name,
            product_name.lower(),
            product_name.replace(' ', ''),
            ' '.join(product_name.split()[:2])  # First two words
        ]
        
        for variation in name_variations:
            if variation.lower() in response_lower:
                recommended_docs.append(doc)
                break
    
    # Remove duplicates while preserving order
    seen_names = set()
    unique_recommended = []
    for doc in recommended_docs:
        name = doc.metadata.get("product_name", "")
        if name not in seen_names:
            seen_names.add(name)
            unique_recommended.append(doc)
    
    return unique_recommended[:3]  # Limit to top 3 recommendations

# MAIN APPLICATION LOGIC
def main():
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize vector store
    if 'vector_store' not in st.session_state:
        try:
            st.session_state.vector_store = load_vector_store()
        except Exception as e:
            st.error(f"Vector store unavailable: {str(e)}")
            st.session_state.vector_store = None
    
    # Initialize query_input if not exists
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""
    
    # Company header
    st.markdown(f"""
    <div class="company-header">
        <h1 class="company-name">{COMPANY_INFO['name']}</h1>
        <div class="company-tagline">{COMPANY_INFO['tagline']}</div>
        <div class="company-stats">
            <div class="stat-item">
                <span class="stat-number">{COMPANY_INFO['experience']}</span>
                <span class="stat-label">Years Experience</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">{len(COMPANY_INFO['specialties'])}</span>
                <span class="stat-label">Product Categories</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">{len(COMPANY_INFO['values'])}</span>
                <span class="stat-label">Core Values</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # WhatsApp-like Chat Interface
    st.markdown("### 💬 Perfectware AI Assistant")
    
    # Create chat container
    chat_container = st.container()

    with chat_container:
        # Display chat history in WhatsApp-like format
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    # User message on the right
                    col1, col2 = st.columns([1, 2])
                    with col2:
                st.markdown(f"""
                        <div style="background-color: #DCF8C6; padding: 15px; border-radius: 15px; margin: 10px 0; text-align: right; color: #000; border: 1px solid #B8E6B8;">
                            <strong style="color: #2E7D32;">You</strong><br>
                            <div style="color: #000; margin-top: 5px;">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
                else:
                    # AI message on the left - UNIFIED MESSAGE WITH IMAGES
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        # Start AI message container
                        ai_message_html = f"""
                        <div style="background-color: #F5F5F5; padding: 15px; border-radius: 15px; margin: 10px 0; color: #000; border: 1px solid #E0E0E0; max-width: 100%; overflow: hidden;">
                            <strong style="color: #1976D2;">Perfectware AI</strong><br>
                            <div style="color: #000; margin-top: 10px; word-wrap: break-word; font-size: 14px; line-height: 1.6;">
                        """
                        
                        # Add the text content with proper markdown rendering
                        text_content = message['content']
                        # Convert markdown to HTML for proper rendering
                        import re
                        # Convert **text** to <strong>text</strong>
                        text_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text_content)
                        # Convert *text* to <em>text</em>
                        text_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text_content)
                        # Convert line breaks
                        formatted_content = text_content.replace('\n', '<br>')
                        ai_message_html += formatted_content
                        
                        # Add product images inline if this is the latest AI message
                        if i == len(st.session_state.chat_history) - 1 and message['role'] == 'assistant':
                            if 'current_docs' in st.session_state and st.session_state.current_docs:
                                ai_message_html += '<br><br><strong style="color: #1976D2;">🖼️ Product Images:</strong><br><br>'
                                
                                # Add images inline in the message
                                for doc_idx, doc in enumerate(st.session_state.current_docs[:5]):  # Show up to 5 products
                                    metadata = doc.metadata
                                    image_base64 = metadata.get("image_base64", None)
                                    product_name = metadata.get("product_name", f"Product {doc_idx + 1}")
                                    size = metadata.get("size", "N/A")
                                    description = metadata.get("description", "")
                                    features = metadata.get('features', [])
                                    
                                    # Add product container
                                    ai_message_html += f'<div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 15px 0; background: white;"><div style="display: flex; gap: 15px; align-items: flex-start;">'
                                    
                                    # Add image
                                    if image_base64:
                                        ai_message_html += f'<div style="flex: 0 0 150px;"><img src="data:image/jpeg;base64,{image_base64}" style="width: 150px; height: 150px; object-fit: cover; border-radius: 8px; border: 1px solid #eee;" /></div>'
                                    else:
                                        ai_message_html += '<div style="flex: 0 0 150px; height: 150px; background: #f8f9fa; border: 1px solid #ddd; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #6c757d;"><div style="text-align: center;"><div style="font-size: 2rem; margin-bottom: 5px;">🖼️</div><div style="font-size: 0.8rem;">No Image</div></div></div>'
                                    
                                    # Add product details
                                    ai_message_html += f'<div style="flex: 1;"><h4 style="margin: 0 0 8px 0; color: #2c3e50; font-size: 1.1rem;">{product_name}</h4><p style="margin: 0 0 8px 0; color: #667eea; font-weight: 500;">Size: {size}</p><p style="margin: 0 0 10px 0; color: #5a6c7d; font-size: 0.9rem; line-height: 1.4;">{description[:150] + ("..." if len(description) > 150 else "")}</p>'
                                    
                                    # Add features if available
                                    if features:
                                        ai_message_html += f'<div style="font-size: 0.85rem; color: #495057;"><strong>Features:</strong> {" • ".join(features[:3])}</div>'
                                    
                                    ai_message_html += '</div></div></div>'
                        
                        # Close the AI message container
                        ai_message_html += """
                            </div>
                        </div>
                        """
                        
                        # Display the complete message
                        try:
                            st.markdown(ai_message_html, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error rendering HTML: {e}")
                            # Fallback: display as plain text for debugging
                            st.text(ai_message_html[:500] + "..." if len(ai_message_html) > 500 else ai_message_html)
        else:
            # Welcome message
            col1, col2 = st.columns([2, 1])
            with col1:
    st.markdown("""
                <div style="background-color: #F5F5F5; padding: 15px; border-radius: 15px; margin: 10px 0; color: #000; border: 1px solid #E0E0E0; max-width: 100%; overflow: hidden;">
                    <strong style="color: #1976D2;">Perfectware AI</strong><br>
                    <div style="color: #000; margin-top: 5px; word-wrap: break-word;">👋 Hello! I'm your Perfectware AI assistant. I'm here to help you find the perfect tiles for your project. What are you looking for today?</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Input field positioned below chat messages (like messaging app)
    st.markdown("---")
    st.markdown("### 💬 Type your message")
    
    # Create a persistent input area with unique key
    query = st.text_input(
        "Type your message here...", 
        value=st.session_state.query_input,
        placeholder="E.g.: I need onyx tiles for my bathroom...",
        key="main_query",  # Fixed key
        label_visibility="collapsed"
    )
    
    # Add send and clear buttons
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("📤 Send", key="send_btn", type="primary"):
            pass  # The query will be processed below
    with col2:
        if st.button("🗑️ Clear", key="clear_btn"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Process query if provided
    if query and query.strip():  # Only process non-empty queries
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query.strip()})
        
        # Clear the input after processing
        st.session_state.query_input = ""
        
        with st.spinner("🔍 Searching our catalog and preparing recommendations..."):
            try:
                # Check if vector store is available
                if st.session_state.vector_store is None:
                    st.error("Vector store not loaded. Please restart the application.")
                    return
                
                # Perform enhanced search with history context
                docs = enhanced_product_search(
                    query, 
                    st.session_state.vector_store, 
                    k=5,  # Get 5 most relevant products
                    history=st.session_state.chat_history
                )
                
                # Generate AI response
                if qa_chain is not None:
                    try:
                # Prepare context with image captions
                context = []
                image_contexts = []
                for doc in docs:
                    context.append(doc.page_content)
                    # Add image context to prompt
                    if "image_caption" in doc.metadata:
                        image_contexts.append(doc.metadata['image_caption'])
                
                        # Get conversation context for the prompt
                        conversation_context = get_conversation_context(st.session_state.chat_history)
                
                # Generate response
                        response = qa_chain.invoke({
                    "context": "\n\n".join(context),
                    "question": query,
                    "company_name": COMPANY_INFO['name'],
                    "tagline": COMPANY_INFO['tagline'],
                    "established": COMPANY_INFO['established'],
                    "experience": COMPANY_INFO['experience'],
                    "location": COMPANY_INFO['location'],
                    "specialties": ", ".join(COMPANY_INFO['specialties']),
                    "customers": ", ".join(COMPANY_INFO['target_customers']),
                    "values": ", ".join(COMPANY_INFO['values']),
                            "image_context": " | ".join(image_contexts[:3]),  # Limit to top 3
                            "history": conversation_context  # Add conversation history to prompt
                        })
                        
                        if hasattr(response, 'content'):
                            response_text = response.content
                        elif isinstance(response, str):
                            response_text = response
                        else:
                            response_text = str(response)
                        
                        # Store the simple AI response in chat history (plain text only)
                        # Remove any HTML tags from the response before storing
                        import re
                        clean_response = re.sub(r'<[^>]+>', '', response_text)
                        st.session_state.chat_history.append({"role": "assistant", "content": clean_response})
                        
                        # Store current search results for image display  
                        st.session_state.current_docs = docs
                        
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                        # Fallback response
                        response_text = "I apologize, but I'm having trouble generating a response right now. Please try again."
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                else:
                    # Fallback response when AI model is not available
                    # Analyze query intent for better fallback
                    query_lower = query.lower()
                    sales_pitch = ""
                    context_advice = ""
                    
                    if any(word in query_lower for word in ["bathroom", "bath", "shower", "toilet"]):
                        sales_pitch = "These bathroom tiles are perfect for creating a luxurious, water-resistant environment. They're designed to withstand moisture while maintaining their beauty."
                        context_advice = "Perfect for bathroom walls and floors with excellent water resistance and easy maintenance."
                    elif any(word in query_lower for word in ["kitchen", "cook", "food"]):
                        sales_pitch = "These kitchen tiles combine style with functionality. They're stain-resistant, easy to clean, and perfect for high-traffic cooking areas."
                        context_advice = "Ideal for kitchen backsplashes and floors with superior durability and stain resistance."
                    elif any(word in query_lower for word in ["outdoor", "garden", "patio", "balcony"]):
                        sales_pitch = "These outdoor tiles are built to withstand weather conditions while maintaining their aesthetic appeal. They're frost-resistant and anti-skid for safety."
                        context_advice = "Perfect for outdoor spaces with excellent weather resistance and anti-skid properties."
                    elif any(word in query_lower for word in ["commercial", "office", "business", "retail"]):
                        sales_pitch = "These commercial-grade tiles are designed for high-traffic areas. They're our most durable options, perfect for offices, retail spaces, and commercial buildings."
                        context_advice = "Professional-grade tiles perfect for commercial spaces with high durability and easy maintenance."
                    else:
                        sales_pitch = "These versatile tiles are perfect for any project. With our 15+ years of experience, I can confidently say these will exceed your expectations."
                        context_advice = "Versatile tiles suitable for various applications in your home or commercial space."
                    
                    # Get product information for fallback
                    product_names = [doc.metadata.get('product_name', 'Premium Tile') for doc in docs[:3]]
                    product_descriptions = [doc.metadata.get('description', 'High-quality tile option') for doc in docs[:3]]
                    
                    # Create a more AI-like response with sales personality
                    response_text = f"""
                    Hello! Thank you for your interest in our premium tile collection. I'm excited to help you find the perfect tiles for your project!

                    {sales_pitch}

                    Based on your search for "{query}", I've carefully analyzed our extensive catalog and selected these exceptional options that would be perfect for your needs:

                    **🌟 Top Recommendations:**

                    **{product_names[0] if len(product_names) > 0 else 'Premium Tile Collection'}**
                    {product_descriptions[0][:200] + '...' if len(product_descriptions) > 0 and product_descriptions[0] else 'A stunning choice that combines elegance with durability.'}
                    {context_advice}

                    **{product_names[1] if len(product_names) > 1 else 'Designer Series'}**
                    {product_descriptions[1][:200] + '...' if len(product_descriptions) > 1 and product_descriptions[1] else 'Perfect for creating a sophisticated atmosphere in any space.'}
                    This option offers exceptional value and timeless appeal.

                    **{product_names[2] if len(product_names) > 2 else 'Classic Collection'}**
                    {product_descriptions[2][:200] + '...' if len(product_descriptions) > 2 and product_descriptions[2] else 'Timeless beauty that will enhance your space for years to come.'}
                    A reliable choice that never goes out of style.

                    **Why these tiles are perfect for you:**
                    ✨ {context_advice}
                    ✨ Premium quality materials from trusted manufacturers
                    ✨ 15+ years of industry expertise backing every recommendation
                    ✨ Easy maintenance and long-lasting beauty
                    ✨ Perfect sizing and installation support available

                    **Additional Options to Consider:**
                    """
                    
                    # Add additional products
                    for i, doc in enumerate(docs[3:6]):  # Show 3 additional products
                        metadata = doc.metadata
                        product_name = metadata.get("product_name", "")
                        description = metadata.get("description", "")
                        size = metadata.get('size', 'N/A')
                        features = metadata.get('features', [])
                        
                        if product_name:
                            explanations = []
                            if "onyx" in query_lower and "onyx" in product_name.lower():
                                explanations.append("perfect onyx option")
                            if size != 'N/A':
                                explanations.append(f"available in {size}")
                            
                            explanation = ", ".join(explanations) if explanations else "matches your requirements"
                            
                            response_text += f"""
                    **{product_name}** - {explanation}
                    {description[:150] + '...' if len(description) > 150 else description}
                    Size: {size}
                    Key Features: {' • '.join(features[:3]) if features else 'Premium quality'}
                    
                    """
                    
                    response_text += """
                    **How can I help you further?**
                    As your Perfectware consultant, I'd love to assist you with:
                    • **Detailed specifications and sizing information** - Get exact dimensions and technical details
                    • **Installation tips and professional guidance** - Expert advice for your project
                    • **Color coordination and design advice** - Help you choose the perfect color scheme
                    • **Pricing and availability details** - Transparent pricing and stock information

                    What specific aspect would you like to explore further? I'm here to make your tile selection process smooth and successful!
                    """
                    
                    # Store the simple AI response in chat history (plain text only)
                    # Remove any HTML tags from the response before storing
                    import re
                    clean_response = re.sub(r'<[^>]+>', '', response_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": clean_response})
                    
                    # Store current search results for image display  
                    st.session_state.current_docs = docs
                
                # AI response is already displayed in chat history above
                # No need for separate display
                
                # All products are now displayed in the chat history above
                # No need for separate display sections
                    
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
        
        # Don't rerun to avoid infinite loop
    
    # PDF UPLOAD SECTION
    with st.sidebar.expander("📁 Manage Product Catalogs", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload product catalogs (PDF)",
            type="pdf",
            accept_multiple_files=True,
            help="Upload 5-6 product catalog PDFs"
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} catalog(s) uploaded successfully!")
            for file in uploaded_files:
                st.markdown(f"""
                <div class="uploaded-pdf">
                    <div class="pdf-icon">📄</div>
                    <div>
                        <div><strong>{file.name}</strong></div>
                        <div>{file.size//1024} KB</div>
                    </div>
                    </div>
                """, unsafe_allow_html=True)
    
    # Search section
    st.markdown("""
    <div class="search-container">
        <div class="search-title">Find Perfect Products</div>
        <div class="search-subtitle">
            Search our extensive catalog of tiles, sanitary ware, and bathroom accessories. 
            Describe your requirements or project needs.
                </div>
                """, unsafe_allow_html=True)
                
    # Predefined prompts section
    st.markdown("### 💡 Quick Questions You Can Ask")
    col1, col2, col3 = st.columns(3)
    
    
    query = ""
    
        with col1:
        if st.button("🏠 Bathroom wall tiles", key="prompt1"):
            st.session_state.query_input = "bathroom wall tiles"
            st.rerun()
        if st.button("🚿 Anti-skid floor tiles", key="prompt2"):
            st.session_state.query_input = "anti-skid floor tiles"
            st.rerun()
        if st.button("🎨 Decorative tiles", key="prompt3"):
            st.session_state.query_input = "decorative tiles"
            st.rerun()
    
        with col2:
        if st.button("🏢 Commercial tiles", key="prompt4"):
            st.session_state.query_input = "commercial tiles"
            st.rerun()
        if st.button("🌿 Outdoor tiles", key="prompt5"):
            st.session_state.query_input = "outdoor tiles"
            st.rerun()
        if st.button("💎 Premium tiles", key="prompt6"):
            st.session_state.query_input = "premium tiles"
            st.rerun()
    
    with col3:
        if st.button("🛁 Bathroom accessories", key="prompt7"):
            st.session_state.query_input = "bathroom accessories"
            st.rerun()
        if st.button("📏 Large format tiles", key="prompt8"):
            st.session_state.query_input = "large format tiles"
            st.rerun()
        if st.button("🔍 Help me choose", key="prompt9"):
            st.session_state.query_input = "help me choose tiles for my project"
            st.rerun()
    
    # Input field has been moved to below chat messages for better UX
    

if __name__ == "__main__":
    main()