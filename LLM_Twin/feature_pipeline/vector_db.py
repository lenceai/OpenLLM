"""
Vector database module for storing and retrieving embeddings.
"""

import os
import logging
import sys
import json
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path
import qdrant_client
from qdrant_client.http import models as rest
from qdrant_client.http.models import PointStruct

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VECTOR_DB_PATH, VECTOR_DIMENSION

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("VectorDB")

class VectorDB:
    """Vector database for storing and retrieving embeddings."""
    
    def __init__(self, 
                 collection_name: str = "llm_twin_embeddings",
                 path: Optional[Union[str, Path]] = VECTOR_DB_PATH,
                 vector_size: int = VECTOR_DIMENSION):
        """
        Initialize the vector database.
        
        Args:
            collection_name: Name of the collection to store embeddings.
            path: Path to store the vector database.
            vector_size: Dimension of the embedding vectors.
        """
        self.collection_name = collection_name
        self.path = Path(path) if path else None
        self.vector_size = vector_size
        self.client = None
        self.logger = logger
    
    def connect(self) -> bool:
        """
        Connect to the vector database.
        
        Returns:
            True if connection was successful, False otherwise.
        """
        try:
            # Create directory for the database if it doesn't exist
            if self.path:
                os.makedirs(self.path, exist_ok=True)
                self.client = qdrant_client.QdrantClient(path=str(self.path))
                self.logger.info(f"Connected to local vector DB at: {self.path}")
            else:
                # Use in-memory database if no path is provided
                self.client = qdrant_client.QdrantClient(":memory:")
                self.logger.info("Connected to in-memory vector DB")
            
            # Check if collection exists, if not create it
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest.VectorParams(
                        size=self.vector_size,
                        distance=rest.Distance.COSINE
                    )
                )
                self.logger.info(f"Created collection: {self.collection_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to vector DB: {str(e)}")
            return False
    
    def store_embeddings(self, items: List[Dict[str, Any]]) -> bool:
        """
        Store embeddings in the vector database.
        
        Args:
            items: List of dictionaries containing embeddings.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.client:
            if not self.connect():
                return False
        
        try:
            # Convert items to points
            points = []
            
            for i, item in enumerate(items):
                # Get embedding
                embedding = item.get('embedding')
                if not embedding:
                    continue
                
                # Convert list to numpy array if needed
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                # Create payload (metadata)
                payload = {k: v for k, v in item.items() if k != 'embedding'}
                
                # Create point
                point = PointStruct(
                    id=item.get('id', i),
                    vector=embedding.tolist(),
                    payload=payload
                )
                
                points.append(point)
            
            # Store points in the database
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                self.logger.info(f"Stored {len(points)} embedding points in vector DB")
                return True
            else:
                self.logger.warning("No valid embedding points to store")
                return False
            
        except Exception as e:
            self.logger.error(f"Error storing embeddings: {str(e)}")
            return False
    
    def search_similar(self, query_embedding: Union[List[float], np.ndarray], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding to search for.
            top_k: Number of similar embeddings to return.
            
        Returns:
            List of dictionaries containing similar documents.
        """
        if not self.client:
            if not self.connect():
                return []
        
        try:
            # Convert list to numpy array if needed
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            
            # Search for similar embeddings
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            
            # Convert search results to dictionaries
            results = []
            for score_point in search_result:
                item = {
                    'id': score_point.id,
                    'score': score_point.score,
                    'embedding': query_embedding.tolist(),
                    **score_point.payload
                }
                results.append(item)
            
            self.logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching for similar embeddings: {str(e)}")
            return []
    
    def delete_collection(self) -> bool:
        """
        Delete the collection.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.client:
            if not self.connect():
                return False
        
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting collection: {str(e)}")
            return False 