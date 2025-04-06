"""
Embedding generator module for converting text to vector embeddings.
"""

import os
import logging
import sys
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL_NAME

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("EmbeddingGenerator")

class EmbeddingGenerator:
    """Generate embeddings for text using a pre-trained model."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the pre-trained model to use.
        """
        self.model_name = model_name
        self.model = None
        self.logger = logger
    
    def load_model(self):
        """Load the embedding model."""
        if self.model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name)
                self.logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
                raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for.
            
        Returns:
            Numpy array containing the embedding vector.
        """
        if not text:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        if self.model is None:
            self.load_model()
        
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for.
            batch_size: Batch size for processing.
            
        Returns:
            List of numpy arrays containing the embedding vectors.
        """
        if not texts:
            return []
        
        if self.model is None:
            self.load_model()
        
        try:
            self.logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {str(e)}")
            # Return empty embeddings
            return [np.zeros(self.model.get_sentence_embedding_dimension()) for _ in texts]
    
    def process_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of data items, adding embeddings.
        
        Args:
            items: List of dictionaries containing data to process.
            
        Returns:
            List of dictionaries with embeddings added.
        """
        if not items:
            return []
        
        # Extract text content
        texts = [item.get('content', '') for item in items]
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts)
        
        # Add embeddings to items
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            # Convert numpy array to list for JSON serialization
            item['embedding'] = embedding.tolist()
        
        self.logger.info(f"Added embeddings to {len(items)} items")
        
        return items 