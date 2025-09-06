import os
import io
import base64
import uuid
import hashlib
import glob
import re
import shutil
import json
import unicodedata
from fuzzywuzzy import fuzz
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image, ImageEnhance
from langchain_community.llms import HuggingFaceHub
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Initialize models
try:
    llm_text = HuggingFaceHub(
        repo_id="gpt2",
        model_kwargs={"temperature": 0.7, "max_length": 512},
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
except Exception as e:
    print(f"Warning: AI model unavailable: {str(e)}")
    llm_text = None

try:
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Warning: Embeddings model unavailable: {str(e)}")
    embeddings = None

# Company information
COMPANY_INFO = {
    "name": "Perfectware Buildings Private Limited",
    "established": "2008",
    "experience": "15+ years",
    "location": "Tamil Nadu",
    "specialties": ["tiles", "sanitary ware", "bathroom accessories"],
    "target_customers": ["homeowners", "contractors", "interior designers"],
    "values": ["quality", "customer satisfaction", "innovation", "durability", "affordability"]
}

# -------------------------
# Utility / parsing helpers
# -------------------------
def debug_elements(elements):
    """Debug function to understand element structure"""
    print(f"\n=== ELEMENT DEBUG INFO ===")
    print(f"Total elements: {len(elements)}")
    element_types = {}
    for idx, element in enumerate(elements):
        elem_type = getattr(element, 'category', type(element).__name__)
        element_types[elem_type] = element_types.get(elem_type, 0) + 1
        
        # Print first few elements for debugging
        if idx < 5:
            text_preview = getattr(element, 'text', '')[:100] if hasattr(element, 'text') else 'No text'
            print(f"  Element {idx}: {elem_type} - {text_preview}...")
    
    print("Element types found:")
    for elem_type, count in element_types.items():
        print(f"  {elem_type}: {count}")

def clean_extracted_text(text: str) -> str:
    """Clean extracted text to improve parsing"""
    if not text:
        return ""
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # remove standalone page numbers
    text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[~—–•¢]', '', text)
    text = re.sub(r'\s+e\s+', ' ', text)
    return text.strip()

def normalize_name_for_match(s: str) -> tuple:
    """Normalize product names, handling HL/FP/VC spacing variants, always returning 2 values."""
    if not s:
        return "", ""

    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\([^)]*\)", "", s)  # remove parentheses
    s = re.sub(r"[‐-–—−]", "-", s)   # normalize dashes
    s = re.sub(r"\s+", " ", s).strip()

    # Spaced version (keeps HL 01 A)
    spaced = re.sub(r"(\d+)([A-Z])\b", r"\1 \2", s)

    # Collapsed version (HL01A, FP01, etc.)
    collapsed = re.sub(r"\b(HL|FP|VC)\s*([0-9]+)\s*([A-Z]?)\b", r"\1\2\3", spaced, flags=re.I)
    collapsed = re.sub(r"\s+", "", collapsed)

    return spaced.lower(), collapsed.lower()
# Known product names from your catalog
# Replace your existing PRODUCT_NAMES list with this updated version
PRODUCT_NAMES = [
    # Original series
    "Carrara Golden", "Invisible Grey", "Onyx Ivory", "Fiesta Crema", "Crystal Gold", 
    "Earlin Azul", "Emperor Vedre", "Gemstone Azul", "Onyx Terquoise", "Onyx Blue",
    "Hermes Taupe", "Ambrosia Decor", "Linosa Taupe", "Olio Grey", "Gemstone Crema",
    "Corten", "Terrazzo Grey", "Delphi Gris", "Ebony White", "Linea Beige",
    "Travertino Beige", "Kraft Gris", "Quad Grey", "Mosaic", "Natic Beige Décor",
    "Lyno Grey", "Palazio Beige", "Royal Botticino", "Satuario Gold", "Venato Bianco",
    "Oyster Beige", "Rainbo Grey", "Terra Grey", "Fabriano Beige", "Bowmore Beige",
    "Tankery Grey", "Valina Azul", "Patch Multi Gris", "Argo Green", "Malfoy Grey",
    "Kyro Beige", "Akira Beige", "Ambra Grey", "Botticino Classico", "Breccia Natural",
    "Pulpis Gris", "Dyno Taupe", "Legno Gris", "Terro Beige", "York Taupe",
    "Tranco Satuario", "Nova Satuario", "Gardenia Blush", "Sega White", "Ivanik Sparkle",
    "Aloise Brown", "Pietrus Brown", "Pietrus Grey", "Edith Decor", "Severin Caramel",
    "Severin Grey", "Avana Brown", "Avana Natural", "Caja Brown", "Ciaz Grey",
    "Cassis Decor", "Crag Brown", "Estuco Cotto", "Estuco Beige", "Hexon Multi",
    "Hexon Brown", "Hexon Grey", "Matone Dark Grey",
    
    # HL series - ALL variants
    "AMAY LIGHT", "AMAY HL 01",
    "CELIA HL 01", "CELIA HL 02", "CELIA BLACK",
    "AMUSE HL 01", "AMUSE DARK", 
    "ANZIO STATUARIO HL 02", 
    "CLAP WOOD HL 01", "CLAP WOOD DARK", "CLAP WOOD LIGHT",  # ADDED MISSING
    "DECAY HL 01", "DECAY DARK",
    "DEMURE HL 01 A", "DEMURE HL 01 B",
    "DROPLET HL 01", "DROPLET DARK",
    "EZRA HL 02 A", "EZRA HL 02 B", 
    "FLORUIT HL 01", 
    "JOYER DARK",
    "LAMELLA HL 01", "LAMELLA WOOD", 
    "PLATANO HL 01 A", "PLATANO HL 01 B",
    "SURFO DARK", "SURFO HL 01 A", 
    "ACCULE HL 01", 
    "AZZARO BEIGE HL 01", "AZZARO BEIGE",
    "FIONA BROWN HL 01", "FIONA LIGHT HL 01",

    # Spectra series
    "SPECTRA SALT", "SPECTRA CHEESE", "SPECTRA PEANUT", "SPECTRA MUSHROOM",
    "SPECTRA CUMIN", "SPECTRA PEPPER", "SPECTRA MUSTARD", "SPECTRA TANGO",
    "SPECTRA BERRY", "SPECTRA CHILLI", "SPECTRA HERBS", "SPECTRA WINE",

    # Messa series
    "MESSA PEANUT", "MESSA SALT", "MESSA CHEESE",

    # Neo series - ALL variants (ADDED MISSING)
    "NEO BLUE DECOR 1", "NEO BLUE DECOR 2", "NEO BLUE DECOR 3", 
    "NEO NAVY DECOR", "NEO COCOA DECOR", "NEO BEIGE DECOR 1", "NEO BEIGE",

    # Garden / Soil / Folk / Motif
    "GARDEN GEO CAMO", "GARDEN LOOP BEACH", "SOIL PETALS WHITE", 
    "FOLK ORIGIN", "MOTIF HERBS",

    # GRAN series
    "GRAN BAMPTON BLUE HG", "GRAN ESPLENDOR WHITE FP", "GRAN AFFRESCO BLACK HG",
    "GRAN ARKANA LIGHT GREY FP", "GRAN LICORICE WHITE FP", "GRAN RELIANCE STATUARIO FP",

    # BON series - ALL variants (ADDED MISSING)
    "BON ARGOS GREY FP", "BON SATUARIO FP", "BON SANTORO NERO CR",
    "BON EUREKA BEIGE FP", "BON IRISH GREY FP", "BON ARGOS DECOR FP",
    "BON ADRIA PEARL FP",
    "BON BOTTICHINO FP", "BON OCEAN ARC-A FP", "BON PORTORO NERO FP",
    "BON VERSILIA NATURAL FP", "BON DRESDEN DECOR", "BON EVIAN DECOR",
    "BON ENIGMA WOOD VC", "BON STARK WOOD VC", "BON NUBE GREY DARK VC",

    # NUEVA series - ALL variants (ADDED MISSING)
    "NUEVA GARNER BEIGE FP", "NUEVA EMPERADOR GOLD FP",
    "NUEVA BALTIC BLACK FP", "NUEVA BRECCIA BROWN FP", "NUEVA NILE BROWN FP",
    "NUEVA EVOQUE GREY LIGHT FP", "NUEVA ONYX FP", "NUEVA ONIXZER FP",
    "NUEVA PORTORO GOLD FP", "NUEVA WOODSTOCK FP", 
    "NUEVA THAMES WOOD FP", "NUEVA ZODIAC WOOD FP"  # ADDED MISSING
]

# Precompute normalized product names for faster lookup
NORMALIZED_PRODUCTS = set()
NORMALIZED_PRODUCTS_NOSPACE = set()
for name in PRODUCT_NAMES:
    n1, n2 = normalize_name_for_match(name)
    NORMALIZED_PRODUCTS.add(n1.lower())
    NORMALIZED_PRODUCTS_NOSPACE.add(n2.lower())


def is_likely_product_name(line: str) -> bool:
    line = line.strip()
    if not line or len(line) < 2:
        return False
    
    low = line.lower()
    
    # === PRIORITY: Reject obvious feature/section lines first ===
    feature_line_patterns = [
        r'^features?\s*:',  # "Features:" or "Feature:"
        r'^application\s+areas?\s*:',  # "Application Areas:"
        r'^size\s*:',  # "Size:"
        r'^description\s*:',  # "Description:"
        r'^key\s+features?\s*:',  # "Key Features:"
        r'^product\s+features?\s*:',  # "Product Features:"
        r'^technical\s+specifications?\s*:',  # "Technical Specifications:"
    ]
    
    for pattern in feature_line_patterns:
        if re.match(pattern, low):
            print(f"DEBUG: Rejected feature line: {line}")
        return False
    
    # Reject lines that start with common feature descriptors
    feature_start_words = [
        "polish finish", "matt finish", "glossy finish", "full polish", 
        "hd full polish", "matt vc finish", "scs matt finish",
        "easy to clean", "stain proof", "water resistant", "scratch resistant",
        "abrasion resistant", "impact resistant", "hard surface", "durable",
        "horizontal motif", "light enhancing", "water repellent"
    ]
    
    for feature_start in feature_start_words:
        if low.startswith(feature_start):
            print(f"DEBUG: Rejected feature description: {line}")
        return False
    
    # Reject if line has too many consecutive feature keywords
    feature_keywords = [
        "polish", "finish", "resistant", "proof", "clean", "stain", "water", 
        "scratch", "abrasion", "impact", "hard", "surface", "durable", "matt", 
        "glossy", "repellent", "enhancing", "motif"
    ]
    feature_word_count = sum(1 for keyword in feature_keywords if keyword in low)
    total_words = len(line.split())
    if feature_word_count >= 3 and total_words <= 8:  # High density of feature words
        print(f"DEBUG: Rejected high-density feature line: {line}")
        return False
    
    # Skip obvious non-product text
    if any(phrase in low for phrase in ["tiles present", "practicality", "page", "index"]):
        return False

    # Skip descriptive sentences
    description_triggers = [
        " is ", " are ", " offers ", " delivers ", " designed ", " presents ",
        " showcases ", " brings ", " highlights ", " enhances ", " provides ",
        " adds ", " gives ", " infuses ", " exudes ", " enriches ", " features ",
        " has ", " have ", " uses ", " utilizes ", " includes ", " contains ",
        " ensures ", " makes ", " creates ", " ideal for ", " perfect for ",
        " suitable for ", " designed for "
    ]
    if any(t in low for t in description_triggers):
        return False
        
    # Keep original length limit - you were right about 60
    if len(line) > 60:
        return False
        
    # Skip dimensions/measurements
    if (line.replace(' ', '').replace('x', '').replace('X', '').replace('mm', '').isdigit() or
        re.match(r'^\d+\s*[xX×]\s*\d+', line)):
        return False

    # === PRIORITY CHECK: Missing products that must always pass ===
    missing_products = [
        "CLAP WOOD HL 01", "CLAP WOOD DARK", "CLAP WOOD LIGHT",
        "NEO COCOA DECOR",
        "BON ENIGMA WOOD VC", "BON STARK WOOD VC",
        "NUEVA THAMES WOOD FP", "NUEVA ZODIAC WOOD FP"
    ]
    
    for missing_product in missing_products:
        n1_missing, n2_missing = normalize_name_for_match(missing_product)
        n1_line, n2_line = normalize_name_for_match(line)
        
        # Exact match
        if n1_line == n1_missing or n2_line == n2_missing:
            print(f"DEBUG: Force-accept missing product (exact): {line}")
        return True
    
        # Partial match for compound names
        if (n1_missing in n1_line and len(n1_missing) > 5) or (n1_line in n1_missing and len(n1_line) > 5):
            print(f"DEBUG: Force-accept missing product (partial): {line}")
        return True
    
        # Fuzzy match for missing products
        if fuzz.ratio(n1_line, n1_missing) >= 80 or fuzz.ratio(n2_line, n2_missing) >= 80:
            print(f"DEBUG: Force-accept missing product (fuzzy): {line}")
        return True
    
    # Skip common non-product phrases AFTER checking missing products
    skip_phrases = [
        "description", "size", "features", "key features", "application areas",
        "applications", "application", "image", "diagram", "outdoor", "vitrified",
        "tile", "mm", "anti-skid", "frost resistance", "high flexural strength",
        "present", "delivering", "using", "effects", "projects", "spaces"
    ]
    # Only skip if the line consists primarily of these phrases
    skip_word_count = sum(1 for phrase in skip_phrases if phrase in low)
    total_words = len(line.split())
    if skip_word_count > 0 and total_words <= 3:  # Only skip short phrases
    return False

    # 1) Exact match with known products
    n1, n2 = normalize_name_for_match(line)
    if n1 in NORMALIZED_PRODUCTS or n2 in NORMALIZED_PRODUCTS_NOSPACE:
        return True

    # 2) Enhanced fuzzy matching for multi-word products
    words = line.split()
    if len(words) <= 8:
        for name in PRODUCT_NAMES:
            p1, p2 = normalize_name_for_match(name)
            
            if n1 == p1 or n2 == p2:
                return True
            
            if n1.startswith(p1) or p1.startswith(n1):
                return True
            if n2.startswith(p2) or p2.startswith(n2):
                return True
            
            if len(words) >= 2:
                if fuzz.token_set_ratio(p1, n1) >= 85 or fuzz.partial_ratio(p2, n2) >= 85:
                    return True
                if fuzz.token_sort_ratio(p1, n1) >= 85:
                    return True

    # 3) Enhanced regex patterns for product names
    enhanced_patterns = [
        r'\b(CLAP|NEO|BON|NUEVA)\s+(WOOD|COCOA|ENIGMA|STARK|THAMES|ZODIAC)\b',
        r'\bCLAP\s+WOOD\s+(HL\s*\d+|DARK|LIGHT)\b',
        r'\bNEO\s+COCOA\s+DECOR\b', 
        r'\bBON\s+(ENIGMA|STARK)\s+WOOD\s+VC\b',
        r'\bNUEVA\s+(THAMES|ZODIAC)\s+WOOD\s+FP\b',
    ]
    for pattern in enhanced_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            print(f"DEBUG: Found product pattern: '{line}' matches '{pattern}'")
            return True

    # 4) Standard product indicators
    product_indicators = [
        r'\b(Golden|Grey|Gray|Beige|White|Black|Blue|Green|Red|Brown|Taupe|Azul|Gris|Ivory|Dark|Light|Navy|Cocoa|Pearl|Salt|Cheese|Peanut|Mushroom|Cumin|Pepper|Mustard|Tango|Berry|Chilli|Herbs|Wine|Cotto|Natural|Gold|Caramel)\b',
        r'\b(AMAY|CELIA|AMUSE|ANZIO|CLAP|DECAY|DEMURE|DROPLET|EZRA|FLORUIT|JOYER|LAMELLA|PLATANO|SURFO|ACCULE|AZZARO|FIONA|SPECTRA|MESSA|NEO|GARDEN|SOIL|FOLK|MOTIF|GRAN|BON|NUEVA)\b',
        r'\b(WOOD|THAMES|ZODIAC|ENIGMA|STARK|MARBLE|STONE|CERAMIC)\b',
        r'\b(D[eé]cor|Decor|HL|VC|FP|HG|CR|POLISH|CARVING|PUNCH)\b',
    ]
    for pattern in product_indicators:
        if re.search(pattern, line, re.IGNORECASE):
            return True

    # 5) Fallback for capitalized multi-word product names
    if 2 <= len(words) <= 6 and 5 <= len(line) <= 60:  # Back to 60 as you suggested
        capitalized_words = [w for w in words if len(w) > 1 and (w[0].isupper() or w.isupper())]
        if len(capitalized_words) >= max(1, len(words)//2):
            combined_text = ' '.join(words).lower()
            known_indicators = ['wood', 'decor', 'hl', 'fp', 'vc', 'hg', 'cr', 'dark', 'light',
                                'neo', 'bon', 'nueva', 'clap', 'gran', 'cocoa', 'enigma',
                                'stark', 'thames', 'zodiac']
            if any(indicator in combined_text for indicator in known_indicators):
                return True

    return False

def extract_all_text_from_elements(elements: list) -> str:
    """Extract and combine all text from elements with enhanced handling of split names"""
    all_text = []
    
    for element in elements:
        elem_text = ""
        if hasattr(element, 'text') and element.text:
            elem_text = element.text.strip()
        elif hasattr(element, 'get_text') and callable(getattr(element, 'get_text')):
            try:
                elem_text = element.get_text().strip()
            except:
                elem_text = ""
        
        if elem_text:
            all_text.append(elem_text)
    
    combined_text = '\n'.join(all_text)
    cleaned_text = clean_extracted_text(combined_text)
    
    # POST-PROCESS: Try to reconstruct split product names with enhanced patterns
    lines = cleaned_text.split('\n')
    reconstructed_lines = []
    
    i = 0
    while i < len(lines):
        current_line = lines[i].strip()
        
        if i < len(lines) - 1:
            next_line = lines[i + 1].strip()
            
            # Enhanced split patterns for missing products
            split_patterns = [
                # "CLAP" + "WOOD" → CLAP WOOD
                (r'^(CLAP|NEO|BON|NUEVA)\s*$', r'^(WOOD|COCOA|ENIGMA|STARK|THAMES|ZODIAC)'),
                # "CLAP WOOD" + "HL 01" or "DARK" or "LIGHT"
                (r'^(CLAP\s+WOOD|NEO\s+COCOA|BON\s+ENIGMA|BON\s+STARK|NUEVA\s+THAMES|NUEVA\s+ZODIAC)\s*$', 
                 r'^(HL\s*\d+|DARK|LIGHT|DECOR|WOOD|FP|VC)\s*$'),
                # "BON ENIGMA" + "WOOD VC"
                (r'^(BON)\s+(ENIGMA|STARK)\s*$', r'^(WOOD)\s+(VC)\s*$'),
                # "NUEVA THAMES" + "WOOD FP" 
                (r'^(NUEVA)\s+(THAMES|ZODIAC)\s*$', r'^(WOOD)\s+(FP)\s*$'),
                # Generic pattern for compound product names
                (r'^([A-Z]+)\s+([A-Z]+)\s*$', r'^([A-Z]+)\s+(HL|FP|VC|DECOR|DARK|LIGHT)\s*$'),
            ]
            
            matched = False
            for first_pattern, second_pattern in split_patterns:
                if re.match(first_pattern, current_line, re.IGNORECASE) and re.match(second_pattern, next_line, re.IGNORECASE):
                    # Check if this would create a known missing product
                    combined_line = f"{current_line} {next_line}".strip()
                    missing_products = [
                        "CLAP WOOD HL 01", "CLAP WOOD DARK", "CLAP WOOD LIGHT",
                        "NEO COCOA DECOR", "BON ENIGMA WOOD VC", "BON STARK WOOD VC",
                        "NUEVA THAMES WOOD FP", "NUEVA ZODIAC WOOD FP"
                    ]
                    
                    # Check if the combined line matches any missing product
                    is_missing_product = False
                    for missing in missing_products:
                        if fuzz.ratio(combined_line.lower(), missing.lower()) >= 70:
                            is_missing_product = True
                            break
                    
                    if is_missing_product or any(keyword in combined_line.upper() for keyword in ['CLAP WOOD', 'NEO COCOA', 'ENIGMA WOOD', 'STARK WOOD', 'THAMES WOOD', 'ZODIAC WOOD']):
                        reconstructed_lines.append(combined_line)
                        print(f"DEBUG: Reconstructed missing product: '{current_line}' + '{next_line}' = '{combined_line}'")
                        i += 2
                        matched = True
                        break
                    else:
                        # Standard reconstruction
                        reconstructed_lines.append(combined_line)
                        print(f"DEBUG: Standard reconstruction: '{current_line}' + '{next_line}' = '{combined_line}'")
                        i += 2
                        matched = True
                        break
            
            if not matched:
                reconstructed_lines.append(current_line)
                i += 1
        else:
            reconstructed_lines.append(current_line)
            i += 1
    
    return '\n'.join(reconstructed_lines)


def parse_product_sections(text: str) -> List[Dict]:
    """Parse structured product sections with enhanced missing product detection"""
    products = []
    current_product = {}
    current_section = None
    section_headers = ["Description", "Size", "Features", "Application Areas"]
    last_product_name = None
    skip_next_name_as_caption = False

    # Enhanced skip keywords
    SKIP_NAME_KEYWORDS = {
        "Chromatic Properties", "Fire Resistance", "Eco Friendly & Recyclable",
        "Easy To Clean", "Heavy Load Bearing", "High Mechanical Resistance",
        "Garden", "Exit Area", "Industrial Area", "Pavement", "Parking",
        "Railway", "Metro", "Gazebos", "Key Features", "Application Areas",
        "Product Features", "Technical Specifications"
    }
    
    DESCRIPTION_START_TRIGGERS = [
        " is ", " are ", " offers ", " delivers ", " designed ", " presents ",
        " showcases ", " brings ", " highlights ", " enhances ", " provides ",
        " adds ", " gives ", " infuses ", " exudes ", " enriches ",
        "with ", "featuring ", "classic ", "a bold ", "a deep ", "a sophisticated ",
        "marble ", "wood ", "cocoa ", "gloss ", "warm appearance"
    ]

    def is_false_product_name(line: str, last_name: str) -> bool:
        if line in SKIP_NAME_KEYWORDS:
            return True
        if any(trigger in line.lower() for trigger in DESCRIPTION_START_TRIGGERS):
            return True
        
        # Be more lenient with missing product patterns
        missing_patterns = ['clap wood', 'neo cocoa', 'enigma wood', 'stark wood', 'thames wood', 'zodiac wood']
        if any(pattern in line.lower() for pattern in missing_patterns):
            return False
        
        # Only reject if it's an obvious description continuation, not a variant
        if last_name and len(line.split()) > 6:
            prev_words = last_name.lower().split()
            curr_words = line.lower().split()
            # If current line starts with same 2+ words AND has description triggers, it's likely a description
            if len(prev_words) >= 2 and len(curr_words) >= 2 and prev_words[:2] == curr_words[:2]:
                if any(trigger in line.lower() for trigger in DESCRIPTION_START_TRIGGERS):
                    return True
        return False

    def is_legitimate_variant(current_name: str, last_name: str) -> bool:
        """Enhanced variant detection for missing products"""
        if not last_name or not current_name:
            return True
            
        # Normalize names for comparison
        def normalize_variant(name):
            return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', '', name.lower().strip()))
        
        curr_norm = normalize_variant(current_name)
        last_norm = normalize_variant(last_name)
        
        # Exact match after normalization - definitely duplicate
        if curr_norm == last_norm:
            return False
        
        # Special handling for missing products - always allow them
        missing_indicators = ['clap wood', 'neo cocoa', 'enigma wood', 'stark wood', 'thames wood', 'zodiac wood']
        if any(indicator in curr_norm for indicator in missing_indicators):
            print(f"DEBUG: Allowing missing product variant: {current_name}")
            return True
        
        # Check if they're different variants of the same base product
        base_products = [
            'CLAP WOOD', 'NEO COCOA', 'BON ENIGMA WOOD', 'BON STARK WOOD',
            'NUEVA THAMES WOOD', 'NUEVA ZODIAC WOOD', 'DEMURE HL', 
            'PLATANO HL', 'SURFO HL'
        ]
        
        for base in base_products:
            base_norm = normalize_variant(base)
            if (curr_norm.startswith(base_norm) and 
                last_norm.startswith(base_norm) and 
                curr_norm != last_norm):
                return True
        
        # Allow different variants by default
        return True

    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if skip_next_name_as_caption:
            skip_next_name_as_caption = False
            i += 1
            continue

        if line.lower().startswith("image"):
            skip_next_name_as_caption = True
            i += 1
            continue

        # Enhanced product name detection
        if is_likely_product_name(line):
            # Normalize name: strip punctuation, multiple spaces
            clean_name = re.sub(r'^[\.\-\•\s]+', '', line).strip()
            clean_name = re.sub(r'\s+', ' ', clean_name)

            # Don't skip missing products even if they seem like false names
            missing_indicators = ['clap wood', 'neo cocoa', 'enigma wood', 'stark wood', 'thames wood', 'zodiac wood']
            is_missing_product = any(indicator in clean_name.lower() for indicator in missing_indicators)
            
            if not is_missing_product and is_false_product_name(clean_name, last_product_name):
                if current_product.get('name'):
                    current_product['description'] = (
                        current_product.get('description', '') + " " + clean_name
                    ).strip()
                i += 1
                continue

            # Check if it's a legitimate variant vs actual duplicate
            if not is_legitimate_variant(clean_name, last_product_name):
                i += 1
                continue

            # Save current product before starting new one
            if current_product.get('name'):
                products.append(current_product)
                current_product = {}

            current_product = {'name': clean_name, 'text': clean_name}
            last_product_name = clean_name
            current_section = None
            print(f"DEBUG: Added product: {clean_name}")
            i += 1
            continue

        # Rest of the parsing logic remains the same...
        # Handle section headers
        header_found = False
        for header in section_headers:
            if line.lower().startswith(header.lower()):
                current_section = header
                content = line[len(header):].lstrip(': ').strip()
                if content:
                    current_product.setdefault(header.lower(), []).append(content)
                header_found = True
                break
        if header_found:
            i += 1
            continue

        # Handle section separators
        if re.match(r'^-{3,}\s*$', line) or line.strip() == "---":
            if current_product.get('name'):
                products.append(current_product)
                current_product = {}
                last_product_name = None
            current_section = None
            i += 1
            continue

        # Handle content for current section
        if current_section and current_product.get('name'):
            if current_section in ["Features", "Application Areas"]:
                if line.startswith('•') or line.startswith('-'):
                    current_product.setdefault(current_section.lower(), []).append(line[1:].strip())
                else:
                    items = [item.strip() for item in re.split(r'[;,•]', line) if item.strip()]
                    if items:
                        current_product.setdefault(current_section.lower(), []).extend(items)
            else:
                content = current_product.get(current_section.lower(), "")
                if isinstance(content, list):
                    content = " ".join(content)
                content = (content + " " + line).strip()
                current_product[current_section.lower()] = content
        elif current_product.get('name'):
            current_product['text'] += "\n" + line

        i += 1

    # Add final product
    if current_product.get('name'):
        products.append(current_product)

    print(f"DEBUG: Total products parsed: {len(products)}")
    return products


def extract_size_from_text(text: str) -> str:
    size_patterns = [
        r'Size:\s*([^\n]+)',
        r'(\d+\s*[xX×]\s*\d+(?:\s*[xX×]\s*\d+)?(?:\s*mm|cm|inches?)?)',
    ]
    for pattern in size_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""

def extract_features(text: str) -> list:
    features = []
    
    # Look for explicit features section
    features_match = re.search(r'Features?:\s*(.*?)(?=Application|Size:|$)', text, re.DOTALL | re.IGNORECASE)
    if features_match:
        features_text = features_match.group(1)
        # Extract bullet points
        bullet_features = re.findall(r'[•\-]\s*([^\n•]+)', features_text)
        features.extend([f.strip() for f in bullet_features if f.strip()])
    
    # Also look for common feature keywords in the description
    feature_keywords = [
        'anti-skid', 'slip-resistant', 'frost resistant', 'water resistant',
        'durable', 'glazed', 'vitrified', 'ceramic', 'porcelain',
        'decorative', 'textured', 'smooth', 'matte', 'glossy', 'polish'
    ]
    text_lower = text.lower()
    for keyword in feature_keywords:
        if keyword in text_lower and keyword.title() not in features:
            features.append(keyword.title())
    
    return features[:10]  # Limit to top 10 features

def extract_applications(text: str) -> list:
    applications = []
    
    # Look for explicit application section
    app_match = re.search(r'Application Areas?:\s*(.*?)(?=Features|Size:|$)', text, re.DOTALL | re.IGNORECASE)
    if app_match:
        app_text = app_match.group(1)
        # Extract bullet points
        bullet_apps = re.findall(r'[•\-]\s*([^\n•]+)', app_text)
        applications.extend([a.strip() for a in bullet_apps if a.strip()])
    
    # Also look for common application keywords
    app_keywords = [
        'outdoor', 'indoor', 'garden', 'parking', 'industrial',
        'residential', 'commercial', 'bathroom', 'kitchen',
        'floor', 'wall', 'swimming pool', 'lobby', 'bedroom'
    ]
    text_lower = text.lower()
    for keyword in app_keywords:
        if keyword in text_lower and keyword.title() not in applications:
            applications.append(keyword.title())
    
    return applications[:10]  # Limit to top 10 applications

def image2base64(img_path: str) -> str:
    """Convert image to base64 string"""
    try:
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            return ""
        with Image.open(img_path) as image:
            if image.width > 800 or image.height > 800:
                image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""

def create_empty_product(name: str, index: int) -> dict:
    """Create a basic product structure"""
    return {
        'name': name,
        'description': f"Tile product: {name}",
        'size': "",
        'features': [],
        'applications': [],
        'text': f"{name}\nTile product: {name}",
        'element_index': index,
        'image_path': None,
        'image_index': None
    }

def assign_images_to_products(products: List[Dict], image_files: List[str]) -> List[Dict]:
    """Assign images to products by strict sequence order (1st product -> 1st image ...).
       This replaces the previous proximity heuristic which caused many wrong assignments."""
    # Assign in-order where possible
    for idx, product in enumerate(products):
        if idx < len(image_files):
            product['image_path'] = image_files[idx]
            product['image_index'] = idx
            print(f"✓ Assigned image {idx} to: {product['name']}")
        else:
            # no corresponding image available for this product index
            product['image_path'] = None
            product['image_index'] = None

    # For any remaining images (images > products), create new product placeholders
    for img_idx in range(len(products), len(image_files)):
        img_path = image_files[img_idx]
        name = f"Product {len(products) + 1}"
        product = create_empty_product(name, len(products))
        product['image_path'] = img_path
        product['image_index'] = img_idx
        products.append(product)
        print(f"✓ Created product for unassigned image: {name}")

    return products

def extract_products_improved(elements: list, image_files: list, extracted_image_dir: str) -> list:
    """
    Improved product extraction with structured parsing
    """
    products = []

    debug_elements(elements)

    # Extract all text from elements
    all_text = extract_all_text_from_elements(elements)
    print(f"Extracted text length: {len(all_text)} characters")
    print(f"Text preview: {all_text[:500]}...")

    # Parse structured product sections
    product_data = parse_product_sections(all_text)
    print(f"Found {len(product_data)} product sections")
    
    # Create product objects
    for idx, data in enumerate(product_data):
        name = data.get('name', f"Product {idx+1}")
        description = data.get('description', "")
        size = data.get('size', "")
        features = data.get('features', [])
        applications = data.get('application areas', [])
        
        # If any of description/size are lists, join them
        if isinstance(description, list):
            description = " ".join(description)
        if isinstance(size, list):
            size = " ".join(size)

        # Extract size if not explicitly found
        if not size:
            size = extract_size_from_text(description)
        
        # Extract features if not explicitly found
        if not features:
            features = extract_features(description)
        
        # Extract applications if not explicitly found
        if not applications:
            applications = extract_applications(description)
        
        # Create text representation
        text_parts = [name]
        if description:
            text_parts.append(f"Description: {description}")
        if size:
            text_parts.append(f"Size: {size}")
        if features:
            text_parts.append(f"Features: {', '.join(features)}")
        if applications:
            text_parts.append(f"Applications: {', '.join(applications)}")
        
        product = {
            'name': name,
            'description': description,
            'size': size,
            'features': features,
            'applications': applications,
            'text': "\n".join(text_parts),
            'element_index': idx,
            'image_path': None,
            'image_index': None
        }
        products.append(product)
        print(f"Parsed product {idx + 1}: {name}")

    # Assign images in strict order
    products = assign_images_to_products(products, image_files)
    
    print(f"Final product count: {len(products)}")
    return products

# ----------------------------------------
# document creation and vector store
# ----------------------------------------
def determine_product_category(text, product_name):
    """Determine product category based on text content and product name"""
    category = "tile"
    text_lower = (text or "").lower()

    if any(word in text_lower for word in ['outdoor', 'garden', 'parking']):
        category = "outdoor_tile"
    elif any(word in text_lower for word in ['vitrified', 'glazed']):
        category = "vitrified_tile"
    elif any(word in text_lower for word in ['ceramic', 'porcelain']):
        category = "ceramic_tile"
    elif any(word in text_lower for word in ['decor', 'decorative']):
        category = "decorative_tile"
    elif any(word in text_lower for word in ['wall']):
        category = "wall_tile"
    elif any(word in text_lower for word in ['floor']):
        category = "floor_tile"

    name_lower = (product_name or "").lower()
    if 'decor' in name_lower:
        category = "decorative_tile"
    if 'outdoor' in name_lower:
        category = "outdoor_tile"

    return category

def create_enhanced_documents(products: List[Dict], pdf_name: str) -> List[Document]:
    """Create enhanced Document objects from extracted products"""
    documents = []
    
    for product in products:
        category = determine_product_category(product['text'], product['name'])

        enhanced_text = f"""Product Name: {product['name']}
Category: {category}
Description: {product['description']}
Size: {product['size']}
Key Features: {', '.join(product['features'][:5])}
Application Areas: {', '.join(product.get('applications', [])[:5])}

Complete Product Information:
{product['text']}
"""
        
        metadata = {
            "id": str(uuid.uuid4()),
            "type": "product",
            "product_name": product['name'],
            "product_category": category,
            "description": product['description'][:200],
            "size": product['size'],
            "features": product['features'],
            "applications": product.get('applications', []),
            "pdf_source": pdf_name,
            "company": COMPANY_INFO["name"],
            "has_image": product.get('image_path') is not None,
            "element_index": product.get('element_index', 0),
            "image_index": product.get('image_index', None)
        }

        image_path = product.get('image_path')
        if image_path and os.path.exists(image_path):
            try:
                base64_image = image2base64(image_path)
                if base64_image:
                    metadata["image_base64"] = base64_image
                    metadata["image_path"] = image_path
                    metadata["image_caption"] = f"Product image for {product['name']}"
                    print(f"✓ Image processed for: {product['name']}")
                else:
                    print(f"✗ Failed to encode image for: {product['name']}")
            except Exception as e:
                print(f"✗ Error processing image for {product['name']}: {e}")
        else:
            print(f"⚠ No image found for: {product['name']}")
        
        documents.append(Document(
            page_content=enhanced_text.strip(),
            metadata=metadata
        ))
    
    return documents

def process_pdf(pdf_path: str, output_dir: str):
    """Extract elements from PDF using unstructured with improved image handling"""
    print(f"Processing PDF: {os.path.basename(pdf_path)}...")

    # Create unique output directory for this PDF
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_dir = os.path.join(output_dir, pdf_name)
    if os.path.exists(pdf_output_dir):
        shutil.rmtree(pdf_output_dir)
    os.makedirs(pdf_output_dir, exist_ok=True)

    elements = partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=500,
        extract_image_block_output_dir=pdf_output_dir
    )

    # Get extracted images with better sorting
    image_pattern_files = (
        glob.glob(os.path.join(pdf_output_dir, '*.jpg')) +
        glob.glob(os.path.join(pdf_output_dir, '*.jpeg')) +
        glob.glob(os.path.join(pdf_output_dir, '*.png'))
    )

    def extract_image_number(filepath):
        filename = os.path.basename(filepath)
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0

    image_files = sorted(image_pattern_files, key=extract_image_number)
    print(f"Extracted {len(image_files)} image files to {pdf_output_dir}")

    # Extract products using improved method
    products = extract_products_improved(elements, image_files, pdf_output_dir)
    print(f"Found {len(products)} products in PDF")

    return products, pdf_output_dir

def process_multiple_pdfs(pdf_paths: List[str]) -> List[Document]:
    """Process multiple PDF files with enhanced error handling"""
    all_documents = []
    base_output_dir = "extracted_content"
    os.makedirs(base_output_dir, exist_ok=True)
    
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"⚠ Warning: PDF not found: {pdf_path}")
            continue
            
        try:
            print(f"\n{'='*50}")
            print(f"Processing: {os.path.basename(pdf_path)}")
            print(f"{'='*50}")
            
            products, pdf_output_dir = process_pdf(pdf_path, base_output_dir)
            print(f"✓ Extracted {len(products)} products")
            
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            documents = create_enhanced_documents(products, pdf_name)
            all_documents.extend(documents)
            
            with_images = sum(1 for d in documents if d.metadata.get('has_image'))
            print(f"✓ Generated {len(documents)} documents ({with_images} with images)")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_documents

def initialize_chat_history():
    """Initialize chat history in session state if not exists"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def get_conversation_context(history: List[Dict]) -> str:
    """Extract context from conversation history"""
    if not history:
        return "No previous conversation."
    
    context = "Previous conversation:\n"
    for i, msg in enumerate(history[-5:]):  # Keep last 5 exchanges for context
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role}: {msg['content']}\n"
    
    return context

def enhanced_product_search(query: str, vs, k: int = 10, history: List[Dict] = None) -> list:
    """Enhanced product search with history context and better ranking"""
    if not vs:
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

def create_vector_store(pdf_paths: List[str]):
    """Create enhanced vector store with improved deduplication"""
    print(f"\n{'='*60}")
    print(f"🏢 {COMPANY_INFO['name']} - Product Catalog Processing")
    print(f"{'='*60}")
    print(f"Processing {len(pdf_paths)} PDF files...")
    
    all_documents = process_multiple_pdfs(pdf_paths)
    
    if not all_documents:
        raise ValueError("No documents were created from the provided PDFs")
    
    print(f"\nDeduplicating {len(all_documents)} documents...")
    unique_docs = {}
    seen_names = set()
    
    for doc in all_documents:
        product_name = doc.metadata.get('product_name', '').strip()
        if not product_name:
            continue
            
        # Normalize product name for comparison
        normalized_name = re.sub(r'[^a-zA-Z0-9]', '', product_name).lower()
        
        # Skip duplicates
        if normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)

        # Create unique key
        key = f"{normalized_name}_{doc.metadata.get('product_category', '')}"
                unique_docs[key] = doc
    
    final_documents = list(unique_docs.values())
    removed_count = len(all_documents) - len(final_documents)
    print(f"✓ Final dataset: {len(final_documents)} unique products (removed {removed_count} duplicates)")
    
    print("\nCreating vector store...")
    vs = FAISS.from_documents(final_documents, embeddings)
    vs.save_local("perfectware_products_index")
    print("✓ Vector store saved successfully!")
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("📊 PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    categories = {}
    images_count = 0
    
    for doc in final_documents:
        cat = doc.metadata.get('product_category', 'uncategorized')
        categories[cat] = categories.get(cat, 0) + 1
        if doc.metadata.get('has_image'):
            images_count += 1
    
    print(f"📦 Total Products: {len(final_documents)}")
    print(f"🖼️  With Images: {images_count} ({images_count/len(final_documents)*100:.1f}%)")
    print(f"📂 Categories:")
    for cat, count in sorted(categories.items()):
        print(f"   • {cat.replace('_', ' ').title()}: {count}")
    
    print(f"\n📋 Product Names Found:")
    for doc in final_documents[:20]:  # Show first 20 products
        name = doc.metadata.get('product_name', 'Unknown')
        has_image = doc.metadata.get('has_image', False)
        print(f"   • {name}: {'✓' if has_image else '✗'}")
    
    if len(final_documents) > 20:
        print(f"   ... and {len(final_documents) - 20} more products")

    # Save product list for reference
    with open("product_list.json", "w") as f:
        json.dump([doc.metadata['product_name'] for doc in final_documents], f, indent=2)
    
    return vs

# ---------------------------
# CLI entrypoint
# ---------------------------
if __name__ == "__main__":
    import sys
    
    print(f"\n{'='*60}")
    print(f"🏢 {COMPANY_INFO['name']}")
    print(f"📅 Established: {COMPANY_INFO['established']} ({COMPANY_INFO['experience']} experience)")
    print(f"📍 Location: {COMPANY_INFO['location']}")
    print(f"🛍️ Specialties: {', '.join(COMPANY_INFO['specialties'])}")
    print(f"{'='*60}\n")
    
    if len(sys.argv) > 1:
        pdf_files = sys.argv[1:]
    else:
        pdf_files = glob.glob("*.pdf")
        if pdf_files:
            print(f"Found PDF files: {', '.join(pdf_files)}")
        else:
            print("No product catalog PDFs found. Please provide PDF file paths as arguments.")
            print("Usage: python main.py <catalog1.pdf> <catalog2.pdf> ...")
            sys.exit(1)
    
    valid_pdfs = [pdf for pdf in pdf_files if os.path.exists(pdf)]
    if not valid_pdfs:
        print("No valid PDF files found!")
        sys.exit(1)
    
    print(f"Processing {len(valid_pdfs)} PDF files...")
    try:
        create_vector_store(valid_pdfs)
        print(f"\n🎉 Processing completed successfully!")
        print(f"💡 Run your Streamlit app to test the enhanced catalog!")
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)