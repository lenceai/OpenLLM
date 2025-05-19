#!/usr/bin/env python3
# server.py

import os
import subprocess
import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import uvicorn
from typing import List, Optional, Dict, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Llama 4 Scout Inference Server")

# Paths to executables and models
BASE_DIR = Path(os.getcwd())
LLAMA_CPP_PATH = BASE_DIR / "llama.cpp" / "build" / "bin" / "main"
MODEL_PATH = BASE_DIR / "models" / "unsloth-llama4-scout" / "Llama-4-Scout-17B-16E-Instruct-unsloth-IQ2_XXS.gguf"

# Check if model and executable exist
if not LLAMA_CPP_PATH.exists():
    logger.error(f"llama.cpp executable not found at {LLAMA_CPP_PATH}")
    logger.error("Please run the download_model.py script first")

if not MODEL_PATH.exists():
    # Try to find any GGUF model in the directory
    model_dir = BASE_DIR / "models" / "unsloth-llama4-scout"
    if model_dir.exists():
        gguf_files = list(model_dir.glob("*.gguf"))
        if gguf_files:
            MODEL_PATH = gguf_files[0]
            logger.info(f"Using found model: {MODEL_PATH}")
        else:
            logger.error(f"No GGUF model found in {model_dir}")
            logger.error("Please run the download_model.py script first")
    else:
        logger.error(f"Model directory not found at {model_dir}")
        logger.error("Please run the download_model.py script first")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    repeat_penalty: Optional[float] = 1.1
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n_gpu_layers: Optional[int] = 35
    n_threads: Optional[int] = 8
    ctx_size: Optional[int] = 4096

class ChatResponse(BaseModel):
    response: str
    usage: Dict[str, Any]

def format_prompt(messages: List[ChatMessage]) -> str:
    """Format messages in llama chat format."""
    system_prompt = "You are a helpful, honest, and concise assistant."
    
    # Check if there's a system message
    if messages and messages[0].role == "system":
        system_prompt = messages[0].content
        messages = messages[1:]
    
    formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n"
    
    for i, msg in enumerate(messages):
        if msg.role == "user":
            formatted_prompt += f"{msg.content}\n"
            if i < len(messages) - 1 and messages[i+1].role == "assistant":
                formatted_prompt += "<|assistant|>\n"
        elif msg.role == "assistant":
            formatted_prompt += f"{msg.content}\n"
            if i < len(messages) - 1 and messages[i+1].role == "user":
                formatted_prompt += "<|user|>\n"
    
    # If the last message is from the user, add the assistant prefix
    if messages and messages[-1].role == "user":
        formatted_prompt += "<|assistant|>\n"
    
    return formatted_prompt

@app.get("/")
async def root():
    return {"status": "ok", "model": str(MODEL_PATH.name)}

@app.post("/health")
async def health_check():
    if not LLAMA_CPP_PATH.exists():
        return {"status": "error", "message": "llama.cpp executable not found"}
    if not MODEL_PATH.exists():
        return {"status": "error", "message": "Model file not found"}
    return {"status": "ok", "model": str(MODEL_PATH.name)}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    formatted_prompt = format_prompt(request.messages)
    
    # Count tokens in prompt (estimate)
    prompt_tokens = len(formatted_prompt.split())
    
    cmd = [
        str(LLAMA_CPP_PATH),
        "-m", str(MODEL_PATH),
        "--ctx-size", str(request.ctx_size),
        "--temp", str(request.temperature),
        "--top_p", str(request.top_p),
        "--top_k", str(request.top_k),
        "--repeat-penalty", str(request.repeat_penalty),
        "--presence-penalty", str(request.presence_penalty),
        "--frequency-penalty", str(request.frequency_penalty),
        "--n-predict", str(request.max_tokens),
        "--threads", str(request.n_threads),
        "--n-gpu-layers", str(request.n_gpu_layers),
        "-p", formatted_prompt
    ]
    
    logger.info(f"Running llama.cpp with {request.n_gpu_layers} GPU layers")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Unknown error"
            logger.error(f"llama.cpp error: {error_msg}")
            raise HTTPException(status_code=500, detail=f"llama.cpp error: {error_msg}")
        
        # Extract the assistant's response (everything after the last <|assistant|>)
        response_text = result.stdout
        assistant_tag = "<|assistant|>"
        if assistant_tag in response_text:
            response_text = response_text.split(assistant_tag)[-1].strip()
        
        # Remove any trailing user or system tags
        stop_tokens = ["<|user|>", "<|system|>", "<|eot_id|>"]
        for token in stop_tokens:
            if token in response_text:
                response_text = response_text.split(token)[0].strip()
        
        # Count tokens in response (estimate)
        completion_tokens = len(response_text.split())
        
        return {
            "response": response_text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        
    except Exception as e:
        logger.error(f"Error running llama.cpp: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu-info")
async def gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return {"status": "error", "message": "Failed to get GPU info"}
            
        lines = result.stdout.strip().split('\n')
        gpus = []
        
        for i, line in enumerate(lines):
            parts = line.split(', ')
            if len(parts) == 4:
                gpus.append({
                    "id": i,
                    "name": parts[0],
                    "memory_total": parts[1],
                    "memory_free": parts[2],
                    "memory_used": parts[3]
                })
        
        return {"status": "ok", "gpus": gpus}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
