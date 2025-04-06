"""
Data warehouse module for storing collected data in MongoDB.
"""

from typing import Dict, List, Any, Optional
import logging
import sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MONGODB_URI, MONGODB_DB, MONGODB_RAW_COLLECTION

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("DataWarehouse")

class DataWarehouse:
    """MongoDB-based data warehouse for storing collected data."""
    
    def __init__(self, uri: str = MONGODB_URI, db_name: str = MONGODB_DB, collection_name: str = MONGODB_RAW_COLLECTION):
        """
        Initialize the data warehouse.
        
        Args:
            uri: MongoDB connection URI.
            db_name: Database name.
            collection_name: Collection name for raw data.
        """
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.logger = logger
    
    def connect(self) -> bool:
        """
        Connect to MongoDB.
        
        Returns:
            True if connection was successful, False otherwise.
        """
        try:
            self.client = MongoClient(self.uri)
            
            # Check if the connection is established
            self.client.admin.command('ping')
            
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            
            self.logger.info(f"Connected to MongoDB: {self.uri}")
            self.logger.info(f"Using database: {self.db_name}")
            self.logger.info(f"Using collection: {self.collection_name}")
            
            return True
            
        except ConnectionFailure:
            self.logger.error(f"Failed to connect to MongoDB: {self.uri}")
            return False
        except Exception as e:
            self.logger.error(f"Error connecting to MongoDB: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None
            self.logger.info("Disconnected from MongoDB")
    
    def store_data(self, data: List[Dict[str, Any]], source_identifier: str = "") -> int:
        """
        Store data in MongoDB.
        
        Args:
            data: List of dictionaries containing data to store.
            source_identifier: Optional identifier for the data source.
            
        Returns:
            Number of documents inserted.
        """
        if not self.client:
            if not self.connect():
                return 0
        
        try:
            # Add source identifier if provided
            if source_identifier:
                for item in data:
                    item["source_id"] = source_identifier
            
            # Insert data
            result = self.collection.insert_many(data)
            count = len(result.inserted_ids)
            
            self.logger.info(f"Inserted {count} documents into MongoDB")
            return count
            
        except Exception as e:
            self.logger.error(f"Error storing data in MongoDB: {str(e)}")
            return 0
    
    def get_all_data(self, query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve data from MongoDB.
        
        Args:
            query: Optional query for filtering results.
            
        Returns:
            List of dictionaries containing retrieved data.
        """
        if not self.client:
            if not self.connect():
                return []
        
        try:
            # Use empty query if none provided
            query = query or {}
            
            # Retrieve data
            cursor = self.collection.find(query)
            data = list(cursor)
            
            self.logger.info(f"Retrieved {len(data)} documents from MongoDB")
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving data from MongoDB: {str(e)}")
            return []
    
    def get_data_by_source(self, source: str) -> List[Dict[str, Any]]:
        """
        Retrieve data by source.
        
        Args:
            source: Source name (e.g., 'linkedin', 'github').
            
        Returns:
            List of dictionaries containing retrieved data.
        """
        query = {"source": source}
        return self.get_all_data(query)
    
    def clear_collection(self) -> bool:
        """
        Clear the collection.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.client:
            if not self.connect():
                return False
        
        try:
            self.collection.delete_many({})
            self.logger.info(f"Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing collection: {str(e)}")
            return False 