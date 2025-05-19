#!/usr/bin/env python3
# convert_model.py

import os
import sys
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    # Check if the fine-tuned model exists
    model_dir = Path("ft-llama4-scout/final")
    if not model_dir.exists():
        logger.error(f"Fine-tuned model not found at {model_dir}")
        logger.error("Please run finetune.py first")
        return

    # Step 1: Merge LoRA weights with base model
    logger.info("Step 1: Merging LoRA weights with base model...")
    try:
        # Import required libraries
        import torch
        from peft import AutoPeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("Required packages not installed. Run: pip install peft transformers torch")
        return
    
    try:
        # Load the fine-tuned model
        logger.info("Loading fine-tuned model...")
        model = AutoPeftModel.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        # Merge LoRA weights with the base model
        logger.info("Merging weights... (this may take a while and use significant memory)")
        merged_model = model.merge_and_unload()

        # Save the merged model
        merged_dir = Path("merged-model")
        merged_dir.mkdir(exist_ok=True)
        logger.info(f"Saving merged model to {merged_dir}...")
        merged_model.save_pretrained(merged_dir)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        tokenizer.save_pretrained(merged_dir)
        
        # Clear memory
        del model
        del merged_model
        torch.cuda.empty_cache()
        
        logger.info("Model merged successfully!")
    except Exception as e:
        logger.error(f"Error merging model: {e}")
        return

    # Step 2: Convert to GGUF format
    logger.info("Step 2: Converting to GGUF format...")
    
    # Check if llama.cpp exists
    llama_cpp_dir = Path("llama.cpp")
    if not llama_cpp_dir.exists():
        logger.error("llama.cpp directory not found")
        logger.error("Please run download_model.py first to clone and build llama.cpp")
        return

    # Check if convert.py exists
    convert_script = llama_cpp_dir / "convert.py"
    if not convert_script.exists():
        logger.error(f"convert.py not found at {convert_script}")
        return

    # Prepare paths
    model_path = merged_dir
    output_path = Path("ft-llama4-scout-gguf")
    output_path.mkdir(exist_ok=True)
    
    # Run conversion
    cmd = [
        sys.executable,
        str(convert_script),
        "--outfile", str(output_path / "ft-llama4-scout.gguf"),
        "--outtype", "f16",
        str(model_path)
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Conversion successful!")
        logger.info(f"Output: {process.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Step 3: Quantize the model
    logger.info("Step 3: Quantizing the model...")
    
    # Check if quantize exists
    quantize_exe = llama_cpp_dir / "build" / "bin" / "quantize"
    if not quantize_exe.exists():
        logger.error(f"quantize executable not found at {quantize_exe}")
        return
    
    input_file = output_path / "ft-llama4-scout.gguf"
    output_file = output_path / "ft-llama4-scout-q4_k_m.gguf"
    
    cmd = [
        str(quantize_exe),
        str(input_file),
        str(output_file),
        "q4_k_m"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Quantization successful!")
        if process.stdout:
            logger.info(f"Output: {process.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Quantization failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    logger.info(f"Conversion complete! The quantized model is saved at: {output_file}")
    logger.info("To use this model:")
    logger.info(f"1. Update the MODEL_PATH in server.py to point to: {output_file}")
    logger.info("2. Restart the server")

if __name__ == "__main__":
    main()
