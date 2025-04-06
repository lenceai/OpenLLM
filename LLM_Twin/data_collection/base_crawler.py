"""
Base crawler that defines the interface for all platform-specific crawlers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class BaseCrawler(ABC):
    """Base class for all platform-specific crawlers."""
    
    def __init__(self, url: str):
        """
        Initialize the crawler with a URL.
        
        Args:
            url: The URL to crawl.
        """
        self.url = url
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data = []
    
    @abstractmethod
    def authenticate(self, credentials: Optional[Dict[str, str]] = None) -> bool:
        """
        Authenticate with the platform if required.
        
        Args:
            credentials: Optional dictionary containing authentication credentials.
            
        Returns:
            True if authentication was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def crawl(self) -> List[Dict[str, Any]]:
        """
        Crawl the platform and collect data.
        
        Returns:
            A list of dictionaries containing the collected data.
        """
        pass
    
    @abstractmethod
    def standardize(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Standardize the raw data to a common format.
        
        Args:
            raw_data: The raw data collected from the platform.
            
        Returns:
            A list of dictionaries with standardized data.
        """
        pass
    
    def clean(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean the standardized data.
        
        Args:
            data: The standardized data.
            
        Returns:
            A list of dictionaries with cleaned data.
        """
        cleaned_data = []
        for item in data:
            # Remove empty fields
            cleaned_item = {k: v for k, v in item.items() if v}
            
            # Add to cleaned data if it contains actual content
            if cleaned_item.get("content"):
                cleaned_data.append(cleaned_item)
        
        return cleaned_data
    
    def run(self, credentials: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Run the complete crawling pipeline.
        
        Args:
            credentials: Optional dictionary containing authentication credentials.
            
        Returns:
            A list of dictionaries with cleaned, standardized data.
        """
        self.logger.info(f"Starting crawler for URL: {self.url}")
        
        # Authenticate if credentials are provided
        if credentials:
            auth_result = self.authenticate(credentials)
            if not auth_result:
                self.logger.error("Authentication failed")
                return []
        
        # Crawl the platform
        self.logger.info("Crawling data...")
        raw_data = self.crawl()
        self.logger.info(f"Collected {len(raw_data)} raw data items")
        
        # Standardize the data
        self.logger.info("Standardizing data...")
        standardized_data = self.standardize(raw_data)
        
        # Clean the data
        self.logger.info("Cleaning data...")
        cleaned_data = self.clean(standardized_data)
        self.logger.info(f"Final dataset contains {len(cleaned_data)} cleaned items")
        
        return cleaned_data 