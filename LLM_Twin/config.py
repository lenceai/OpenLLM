"""
Configuration settings for the LLM Twin project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = "llm_twin_db"
MONGODB_RAW_COLLECTION = "raw_data"

# Vector DB configuration
VECTOR_DB_PATH = DATA_DIR / "vector_db"
VECTOR_DIMENSION = 768  # Default for most embedding models

# Model configuration
BASE_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Can be changed to any compatible model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Training configuration
LEARNING_RATE = 3e-5
BATCH_SIZE = 4
NUM_EPOCHS = 3
MAX_LENGTH = 512
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Inference configuration
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 40
NUM_RETURN_SEQUENCES = 1
REPETITION_PENALTY = 1.1

# RAG configuration
NUM_CHUNKS_TO_RETRIEVE = 5

# Web interface configuration
STREAMLIT_PORT = 8501

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000 