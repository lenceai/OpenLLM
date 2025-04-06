"""
RAG generator module for text generation with retrieval-augmented generation.
"""

import os
import logging
import sys
import torch
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASE_MODEL_NAME,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    NUM_RETURN_SEQUENCES,
    REPETITION_PENALTY,
    NUM_CHUNKS_TO_RETRIEVE
)
from feature_pipeline.embedding_generator import EmbeddingGenerator
from feature_pipeline.vector_db import VectorDB

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("RAGGenerator")

class RAGGenerator:
    """Text generator with retrieval-augmented generation."""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 base_model_name: str = BASE_MODEL_NAME,
                 max_new_tokens: int = MAX_NEW_TOKENS,
                 temperature: float = TEMPERATURE,
                 top_p: float = TOP_P,
                 top_k: int = TOP_K,
                 num_return_sequences: int = NUM_RETURN_SEQUENCES,
                 repetition_penalty: float = REPETITION_PENALTY,
                 num_chunks_to_retrieve: int = NUM_CHUNKS_TO_RETRIEVE):
        """
        Initialize the RAG generator.
        
        Args:
            model_path: Path to the fine-tuned model.
            base_model_name: Name of the base model.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Temperature for sampling.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            num_return_sequences: Number of sequences to return.
            repetition_penalty: Penalty for token repetition.
            num_chunks_to_retrieve: Number of chunks to retrieve from vector DB.
        """
        self.model_path = Path(model_path)
        self.base_model_name = base_model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_return_sequences = num_return_sequences
        self.repetition_penalty = repetition_penalty
        self.num_chunks_to_retrieve = num_chunks_to_retrieve
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDB()
        self.logger = logger
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        self.logger.info(f"Loading fine-tuned model from: {self.model_path}")
        
        try:
            # Load the base model with quantization
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # Ensure the tokenizer has a pad token
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load the model with 4-bit quantization
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Load the fine-tuned model adapter
            self.model = PeftModel.from_pretrained(model, self.model_path)
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from vector DB.
        
        Args:
            query: Query to retrieve context for.
            
        Returns:
            List of dictionaries containing relevant documents.
        """
        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Retrieve similar documents
        similar_docs = self.vector_db.search_similar(
            query_embedding=query_embedding,
            top_k=self.num_chunks_to_retrieve
        )
        
        return similar_docs
    
    def format_prompt_with_context(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Format the prompt with retrieved context.
        
        Args:
            query: User query.
            context_docs: List of context documents.
            
        Returns:
            Formatted prompt string.
        """
        # Format context string
        context_str = ""
        for i, doc in enumerate(context_docs):
            content = doc.get('content', '')
            source = doc.get('source', 'unknown')
            
            context_str += f"[Document {i+1} from {source}]\n{content}\n\n"
        
        # Format full prompt
        prompt = f"""
Below is some context information:
{context_str}

Based on the information above, please respond to the following:
{query}

Response:
"""
        
        return prompt
    
    def generate_text(self, prompt: str) -> List[str]:
        """
        Generate text based on the prompt.
        
        Args:
            prompt: Prompt text.
            
        Returns:
            List of generated text strings.
        """
        if not self.model or not self.tokenizer:
            self.load_model()
        
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    num_return_sequences=self.num_return_sequences,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=True
                )
            
            # Decode the generated text
            generated_texts = []
            for output in outputs:
                # Only keep the generated part (exclude the input prompt)
                generated_text = self.tokenizer.decode(output[len(inputs.input_ids[0]):], skip_special_tokens=True)
                generated_texts.append(generated_text)
            
            return generated_texts
            
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            return []
    
    def generate_with_rag(self, query: str) -> List[str]:
        """
        Generate text with retrieval-augmented generation.
        
        Args:
            query: User query.
            
        Returns:
            List of generated text strings.
        """
        # Retrieve context
        context_docs = self.retrieve_context(query)
        
        if not context_docs:
            self.logger.warning("No context found in vector DB, generating without RAG")
            return self.generate_text(query)
        
        # Format prompt with context
        prompt = self.format_prompt_with_context(query, context_docs)
        
        # Generate text
        generated_texts = self.generate_text(prompt)
        
        return generated_texts 