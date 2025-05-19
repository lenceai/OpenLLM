#!/usr/bin/env python3
# download_model.py

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Make sure we use HF Transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from huggingface_hub import snapshot_download
    import torch
except ImportError:
    logger.error("Required packages not installed. Run: pip install huggingface_hub hf-transfer torch")
    sys.exit(1)

def main():
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU detected: {gpu_name} with {vram_gb:.2f} GB VRAM")
        
        if vram_gb < 20:
            logger.warning(f"Your GPU has only {vram_gb:.2f} GB VRAM, which might be insufficient for Llama 4.")
            proceed = input("Continue anyway? (y/n): ")
            if proceed.lower() != 'y':
                sys.exit(0)
    else:
        logger.warning("No GPU detected! This project requires an NVIDIA GPU.")
        proceed = input("Continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            sys.exit(0)
    
    # Download both the GGUF model for inference and the 4-bit model for fine-tuning
    logger.info("Downloading quantized Llama 4 Scout model for inference...")
    try:
        gguf_path = snapshot_download(
            repo_id="unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
            local_dir=models_dir / "unsloth-llama4-scout",
            allow_patterns=["*IQ2_XXS*"],  # This is the smallest quantized version
        )
        logger.info(f"Downloaded GGUF model to: {gguf_path}")
    except Exception as e:
        logger.error(f"Error downloading GGUF model: {e}")
        
    logger.info("Downloading 4-bit model for fine-tuning...")
    try:
        bnb_path = snapshot_download(
            repo_id="unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit",
            local_dir=models_dir / "unsloth-llama4-scout-4bit",
        )
        logger.info(f"Downloaded 4-bit model to: {bnb_path}")
    except Exception as e:
        logger.error(f"Error downloading 4-bit model: {e}")
    
    logger.info("Model download complete!")
    
    # Check if llama.cpp is already cloned
    if not Path("llama.cpp").exists():
        logger.info("Cloning llama.cpp...")
        os.system("git clone https://github.com/ggerganov/llama.cpp")
        
        # Build llama.cpp
        logger.info("Building llama.cpp...")
        os.makedirs("llama.cpp/build", exist_ok=True)
        os.chdir("llama.cpp/build")
        os.system("cmake .. -DLLAMA_CUBLAS=ON")
        os.system("make -j")
        os.chdir("../..")
        
        logger.info("llama.cpp build complete!")
    else:
        logger.info("llama.cpp already exists, skipping clone and build.")

if __name__ == "__main__":
    main()
