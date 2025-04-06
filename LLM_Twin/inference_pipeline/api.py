"""
REST API for the Inference Pipeline.
"""

import os
import logging
import sys
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_HOST, API_PORT
from training_pipeline.model_registry import ModelRegistry
from inference_pipeline.rag_generator import RAGGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("API")

# Create FastAPI app
app = FastAPI(title="LLM Twin API", description="API for the LLM Twin project")

# Model registry
model_registry = ModelRegistry()

# RAG generator (to be initialized when needed)
rag_generator = None

# Pydantic models for request/response
class GenerateRequest(BaseModel):
    """Request model for text generation."""
    query: str = Field(..., description="The user's query")
    model_id: str = Field(None, description="ID of the model to use (or None for latest)")
    use_rag: bool = Field(True, description="Whether to use RAG")
    
class GenerateResponse(BaseModel):
    """Response model for text generation."""
    generated_texts: List[str] = Field(..., description="Generated text responses")
    model_id: str = Field(..., description="ID of the model used")
    
class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    models: List[Dict[str, Any]] = Field(..., description="List of available models")

# Initialize the RAG generator with the specified model
def get_rag_generator(model_id: str = None):
    """
    Get the RAG generator for the specified model.
    
    Args:
        model_id: ID of the model to use (or None for latest).
        
    Returns:
        RAG generator instance.
    """
    global rag_generator
    
    # Get model info
    if model_id:
        model_info = model_registry.get_model(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    else:
        model_info = model_registry.get_latest_model()
        if not model_info:
            raise HTTPException(status_code=404, detail="No models available")
    
    model_path = model_info["model_path"]
    base_model_name = model_info.get("metadata", {}).get("base_model")
    
    # Create RAG generator if it doesn't exist or if the model has changed
    if not rag_generator or rag_generator.model_path != Path(model_path):
        logger.info(f"Initializing RAG generator with model: {model_path}")
        rag_generator = RAGGenerator(model_path=model_path, base_model_name=base_model_name)
    
    return rag_generator, model_info["model_id"]

@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the LLM Twin API"}

@app.get("/models", response_model=ModelInfoResponse, tags=["Models"])
async def list_models():
    """List available models."""
    models = model_registry.list_models()
    return {"models": models}

@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate(request: GenerateRequest):
    """
    Generate text based on the query.
    
    Args:
        request: GenerateRequest object.
        
    Returns:
        GenerateResponse object.
    """
    try:
        # Get RAG generator and model ID
        generator, model_id = get_rag_generator(request.model_id)
        
        # Generate text
        if request.use_rag:
            generated_texts = generator.generate_with_rag(request.query)
        else:
            generated_texts = generator.generate_text(request.query)
        
        return GenerateResponse(generated_texts=generated_texts, model_id=model_id)
        
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def run_api():
    """Run the API server."""
    uvicorn.run(app, host=API_HOST, port=API_PORT)

if __name__ == "__main__":
    run_api() 