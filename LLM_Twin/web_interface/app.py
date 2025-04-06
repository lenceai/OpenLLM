"""
Streamlit web interface for the LLM Twin project.
"""

import os
import sys
import logging
import requests
import json
import time
from pathlib import Path
import streamlit as st
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_PORT, API_HOST, STREAMLIT_PORT
from data_collection.linkedin_crawler import LinkedInCrawler
from data_collection.data_warehouse import DataWarehouse
from feature_pipeline.text_preprocessing import TextPreprocessor
from feature_pipeline.embedding_generator import EmbeddingGenerator
from feature_pipeline.vector_db import VectorDB
from training_pipeline.model_fine_tuner import ModelFineTuner
from training_pipeline.model_registry import ModelRegistry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("WebInterface")

# API URL
API_URL = f"http://{API_HOST}:{API_PORT}"

# Page configuration
st.set_page_config(
    page_title="LLM Twin",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2196F3;
        margin-bottom: 1.5rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-success {
        background-color: #E8F5E9;
        border-left: 0.5rem solid #4CAF50;
    }
    .status-warning {
        background-color: #FFF8E1;
        border-left: 0.5rem solid #FFC107;
    }
    .status-error {
        background-color: #FFEBEE;
        border-left: 0.5rem solid #F44336;
    }
    .generated-content {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 0.5rem solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def check_api_status():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/")
        return response.status_code == 200
    except:
        return False

def fetch_models():
    """Fetch available models from the API."""
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json()["models"]
        return []
    except:
        return []

def generate_text(query, model_id=None, use_rag=True):
    """Generate text using the API."""
    try:
        data = {
            "query": query,
            "model_id": model_id,
            "use_rag": use_rag
        }
        response = requests.post(f"{API_URL}/generate", json=data)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error generating text: {str(e)}")
        return None

# State initialization
if "data_collected" not in st.session_state:
    st.session_state.data_collected = False
if "embeddings_created" not in st.session_state:
    st.session_state.embeddings_created = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "generation_history" not in st.session_state:
    st.session_state.generation_history = []

# App header
st.markdown('<p class="main-header">🤖 LLM Twin</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Personalize content creation with your digital footprint</p>', unsafe_allow_html=True)

# Sidebar - Configuration and Status
with st.sidebar:
    st.header("Configuration")
    
    # Data sources
    st.subheader("Data Sources")
    
    # LinkedIn
    linkedin_url = st.text_input("LinkedIn Profile URL", "https://www.linkedin.com/in/username/")
    linkedin_username = st.text_input("LinkedIn Username (optional)")
    linkedin_password = st.text_input("LinkedIn Password (optional)", type="password")
    
    # GitHub
    github_url = st.text_input("GitHub Profile URL", "https://github.com/username")
    
    # Medium
    medium_url = st.text_input("Medium Profile URL", "https://medium.com/@username")
    
    # Substack
    substack_url = st.text_input("Substack URL", "https://username.substack.com")
    
    # Actions
    st.subheader("Actions")
    
    collect_data_btn = st.button("1. Collect Data")
    create_embeddings_btn = st.button("2. Create Embeddings")
    train_model_btn = st.button("3. Train Model")
    
    # System Status
    st.subheader("System Status")
    
    # Data collection status
    if st.session_state.data_collected:
        st.markdown('<div class="status-box status-success">Data Collection: Complete ✅</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-warning">Data Collection: Not Started ⏳</div>', unsafe_allow_html=True)
    
    # Embeddings status
    if st.session_state.embeddings_created:
        st.markdown('<div class="status-box status-success">Embeddings: Created ✅</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-warning">Embeddings: Not Created ⏳</div>', unsafe_allow_html=True)
    
    # Model training status
    if st.session_state.model_trained:
        st.markdown('<div class="status-box status-success">Model: Trained ✅</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-warning">Model: Not Trained ⏳</div>', unsafe_allow_html=True)
    
    # API status
    api_running = check_api_status()
    if api_running:
        st.markdown('<div class="status-box status-success">API: Running ✅</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-error">API: Not Running ❌</div>', unsafe_allow_html=True)
        st.info("To start the API, run: `python -m inference_pipeline.api`")

# Main panel - Content Generation
st.header("Content Generation")

# Input for content generation
gen_col1, gen_col2 = st.columns([3, 1])

with gen_col1:
    user_prompt = st.text_area("Enter your prompt:", height=100, 
                            placeholder="e.g., Write a LinkedIn post about artificial intelligence trends in 2023")

with gen_col2:
    # Get available models
    models = fetch_models()
    model_options = [model["model_id"] for model in models]
    
    if model_options:
        selected_model = st.selectbox("Select Model:", model_options)
    else:
        selected_model = None
        st.info("No fine-tuned models available yet")
    
    use_rag = st.checkbox("Use RAG", value=True)
    
    generate_btn = st.button("Generate Content", disabled=not api_running or selected_model is None)

# Generation result
if generate_btn and user_prompt:
    with st.spinner("Generating content..."):
        result = generate_text(user_prompt, selected_model, use_rag)
        
        if result:
            # Add to history
            st.session_state.generation_history.append({
                "prompt": user_prompt,
                "result": result,
                "timestamp": time.time()
            })
            
            # Display result
            st.subheader("Generated Content:")
            for i, text in enumerate(result["generated_texts"]):
                st.markdown(f'<div class="generated-content">{text}</div>', unsafe_allow_html=True)
                
                # Copy button
                if st.button(f"Copy to Clipboard", key=f"copy_{i}"):
                    st.info("Content copied to clipboard!")

# Generation History
if st.session_state.generation_history:
    st.header("Generation History")
    
    for i, item in enumerate(reversed(st.session_state.generation_history)):
        with st.expander(f"Prompt: {item['prompt'][:50]}...", expanded=i==0):
            st.write("**Prompt:**")
            st.write(item["prompt"])
            st.write("**Generated Content:**")
            for j, text in enumerate(item["result"]["generated_texts"]):
                st.markdown(f'<div class="generated-content">{text}</div>', unsafe_allow_html=True)

# Handle data collection
if collect_data_btn:
    with st.spinner("Collecting data..."):
        try:
            # Initialize data warehouse
            warehouse = DataWarehouse()
            warehouse.connect()
            
            # LinkedIn
            if linkedin_url and linkedin_url != "https://www.linkedin.com/in/username/":
                st.info("Collecting data from LinkedIn...")
                crawler = LinkedInCrawler(linkedin_url)
                credentials = None
                if linkedin_username and linkedin_password:
                    credentials = {"username": linkedin_username, "password": linkedin_password}
                linkedin_data = crawler.run(credentials)
                if linkedin_data:
                    warehouse.store_data(linkedin_data, "linkedin")
                    st.success(f"Collected {len(linkedin_data)} items from LinkedIn")
            
            # TODO: Add other crawlers (GitHub, Medium, Substack)
            
            st.session_state.data_collected = True
            st.success("Data collection completed! Please refresh the page to see updated status.")
            
        except Exception as e:
            st.error(f"Error collecting data: {str(e)}")

# Handle embeddings creation
if create_embeddings_btn:
    if not st.session_state.data_collected:
        st.error("Please collect data first!")
    else:
        with st.spinner("Creating embeddings..."):
            try:
                # Get data from warehouse
                warehouse = DataWarehouse()
                warehouse.connect()
                data = warehouse.get_all_data()
                
                if not data:
                    st.error("No data found in the warehouse!")
                else:
                    # Preprocess data
                    preprocessor = TextPreprocessor()
                    processed_data = preprocessor.process_batch(data)
                    
                    # Generate embeddings
                    embedder = EmbeddingGenerator()
                    embedded_data = embedder.process_items(processed_data)
                    
                    # Store in vector DB
                    vector_db = VectorDB()
                    vector_db.connect()
                    vector_db.store_embeddings(embedded_data)
                    
                    st.session_state.embeddings_created = True
                    st.success("Embeddings created! Please refresh the page to see updated status.")
                
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")

# Handle model training
if train_model_btn:
    if not st.session_state.embeddings_created:
        st.error("Please create embeddings first!")
    else:
        with st.spinner("Training model... This may take a while."):
            try:
                # Get data from warehouse
                warehouse = DataWarehouse()
                warehouse.connect()
                data = warehouse.get_all_data()
                
                if not data:
                    st.error("No data found in the warehouse!")
                else:
                    # Fine-tune model
                    fine_tuner = ModelFineTuner()
                    model_path = fine_tuner.fine_tune(data)
                    
                    # Register model
                    registry = ModelRegistry()
                    model_id = registry.register_model(
                        model_path, 
                        model_name=f"LLM Twin Model {time.strftime('%Y-%m-%d')}",
                        description="Fine-tuned on user data"
                    )
                    
                    st.session_state.model_trained = True
                    st.success(f"Model trained and registered with ID: {model_id}")
                    st.info("Please start the API to use the model for generation.")
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")

# Run the Streamlit app
if __name__ == "__main__":
    # This code will be executed when the script is run directly
    # streamlit run app.py
    pass 