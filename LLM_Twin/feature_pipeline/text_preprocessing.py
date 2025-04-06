"""
Text preprocessing module for cleaning and chunking text data.
"""

import re
from typing import List, Dict, Any, Tuple
import logging
import sys
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("TextPreprocessing")

class TextPreprocessor:
    """Text preprocessing for cleaning and chunking text data."""
    
    def __init__(self, 
                 chunk_size: int = 512, 
                 chunk_overlap: int = 50,
                 min_chunk_length: int = 100):
        """
        Initialize the text preprocessor.
        
        Args:
            chunk_size: Target size (in characters) for each text chunk.
            chunk_overlap: Overlap (in characters) between consecutive chunks.
            min_chunk_length: Minimum chunk length to keep.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.logger = logger
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and normalizing whitespace.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Replace newlines with spaces
        text = re.sub(r'\n+', ' ', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters except punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"-]', '', text)
        
        # Trim leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text_by_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks based on sentences, targeting chunk_size with overlap.
        
        Args:
            text: Text to split into chunks.
            
        Returns:
            List of text chunks.
        """
        if not text:
            return []
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_chunk_len = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence would exceed the chunk size,
            # finalize the current chunk and start a new one
            if current_chunk and current_chunk_len + sentence_len + 1 > self.chunk_size:
                # Only add the chunk if it meets the minimum length
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_length:
                    chunks.append(chunk_text)
                
                # Start a new chunk with overlap (keep some sentences from the end)
                overlap_chars = 0
                overlap_sentences = []
                
                # Add sentences from the end until we reach the desired overlap
                for i in range(len(current_chunk) - 1, -1, -1):
                    sentence_to_add = current_chunk[i]
                    overlap_chars += len(sentence_to_add) + 1  # +1 for space
                    overlap_sentences.insert(0, sentence_to_add)
                    
                    if overlap_chars >= self.chunk_overlap:
                        break
                
                current_chunk = overlap_sentences
                current_chunk_len = overlap_chars
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_chunk_len += sentence_len + 1  # +1 for space
        
        # Add the final chunk if it's not empty and meets minimum length
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_length:
                chunks.append(chunk_text)
        
        return chunks
    
    def process_item(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single data item (document) into multiple chunks.
        
        Args:
            item: Dictionary containing data to process.
            
        Returns:
            List of dictionaries, each representing a chunk of the original document.
        """
        processed_items = []
        
        # Get all text content
        title = item.get('title', '')
        content = item.get('content', '')
        
        # Clean the content
        cleaned_content = self.clean_text(content)
        
        # Chunk the cleaned content
        if cleaned_content:
            chunks = self.chunk_text_by_sentences(cleaned_content)
            
            # Create a new item for each chunk
            for i, chunk in enumerate(chunks):
                chunk_item = item.copy()
                
                # Replace content with the chunk
                chunk_item['content'] = chunk
                
                # Add chunk metadata
                chunk_item['chunk_id'] = i
                chunk_item['total_chunks'] = len(chunks)
                
                # If there's a title, add it to the first chunk
                if title and i == 0:
                    chunk_item['content'] = f"{title}\n\n{chunk}"
                
                processed_items.append(chunk_item)
        
        return processed_items
    
    def process_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of data items.
        
        Args:
            items: List of dictionaries containing data to process.
            
        Returns:
            List of processed items, with each original item potentially split into multiple chunks.
        """
        all_processed_items = []
        
        for item in items:
            processed_items = self.process_item(item)
            all_processed_items.extend(processed_items)
        
        self.logger.info(f"Processed {len(items)} items into {len(all_processed_items)} chunks")
        
        return all_processed_items 