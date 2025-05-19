#!/usr/bin/env python3
# finetune.py

import os
import logging
import sys
import json
from pathlib import Path
import torch
from datasets import load_dataset
import transformers
from transformers import TrainingArguments
from trl import SFTTrainer
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Check if we have enough GPU memory
def check_gpu_memory():
    if not torch.cuda.is_available():
        logger.error("No GPU available. Cannot proceed with fine-tuning.")
        return False
        
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Available VRAM: {vram_gb:.2f} GB")
    
    if vram_gb < 20:
        logger.error(f"Not enough VRAM for fine-tuning. Available: {vram_gb:.2f} GB, Recommended: at least 24GB")
        return False
        
    return True

# Format the dataset in the required format
def format_instruction_dataset(example):
    """Format data into the instruction format expected by Llama models."""
    if isinstance(example, dict):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        response = example.get("output", "") or example.get("response", "")
        
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction
            
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response}
        ]
        
        # Add system prompt if present
        if "system" in example:
            messages.insert(0, {"role": "system", "content": example["system"]})
            
        return {"messages": messages}
    else:
        logger.warning(f"Unexpected example format: {type(example)}")
        return {"messages": []}

def main():
    # Check GPU
    if not check_gpu_memory():
        logger.error("Insufficient GPU resources. Exiting.")
        return
    
    # Create output directory
    output_dir = Path("ft-llama4-scout")
    output_dir.mkdir(exist_ok=True)
    
    # Try to import unsloth after GPU check
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth not installed. Run: pip install unsloth")
        return
    
    # Set training parameters based on available VRAM
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Adjust parameters based on available VRAM
    if vram_gb >= 20:
        max_seq_length = 2048
        per_device_batch_size = 1
        gradient_accumulation_steps = 4
    else:
        max_seq_length = 1024
        per_device_batch_size = 1
        gradient_accumulation_steps = 8
    
    # Load the model with Unsloth's optimizations
    logger.info("Loading model...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit",
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            # Trust remote code since Unsloth may need custom code
            trust_remote_code=True,
            device_map="auto",
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Add LoRA adapters
    logger.info("Adding LoRA adapters...")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,       # Rank of the adapters (lower to save VRAM)
            # Target all important modules for tuning
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            use_gradient_checkpointing=True,
            random_state=42,
            use_rslora=False,  # Set to True for better results but slower training
            bf16=True,
        )
    except Exception as e:
        logger.error(f"Error setting up LoRA: {e}")
        return
    
    # Check for dataset or use default
    data_path = Path("data/finetune.json")
    
    if data_path.exists():
        logger.info(f"Using custom dataset from {data_path}")
        try:
            dataset = load_dataset("json", data_files=str(data_path), split="train")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return
    else:
        logger.info("No custom dataset found. Using a public dataset for demonstration.")
        try:
            dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
        except Exception:
            try:
                # Fall back to another dataset if the first one fails
                dataset = load_dataset("yahma/alpaca-cleaned", split="train")
            except Exception as e:
                logger.error(f"Error loading public dataset: {e}")
                return
    
    # Format the dataset
    logger.info("Formatting dataset...")
    try:
        formatted_dataset = dataset.map(format_instruction_dataset)
    except Exception as e:
        logger.error(f"Error formatting dataset: {e}")
        return
    
    # Show an example of the formatted data
    if len(formatted_dataset) > 0:
        logger.info("Example of formatted data:")
        logger.info(json.dumps(formatted_dataset[0], indent=2))
    
    # Calculate appropriate training steps
    num_samples = len(formatted_dataset)
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps
    logger.info(f"Dataset size: {num_samples} samples")
    logger.info(f"Effective batch size: {effective_batch_size}")
    
    # Use a modest number of steps based on dataset size
    if num_samples < 100:
        max_steps = 20
    elif num_samples < 1000:
        max_steps = 100
    else:
        max_steps = 500
    
    logger.info(f"Training for {max_steps} steps")
    
    # Configure training parameters
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=5,
        save_steps=50,
        learning_rate=2e-4,
        weight_decay=0.01,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_total_limit=3,
        push_to_hub=False,
        fp16=False,
        bf16=True,  # Use bfloat16 precision
        max_steps=max_steps,
        report_to="none",
        # Disable trying to create a validation split since we're using max_steps
        do_eval=False,
    )
    
    # Create SFT trainer
    logger.info("Setting up trainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=formatted_dataset,
            tokenizer=tokenizer,
            peft_config=None,  # We already added LoRA adapters
            dataset_text_field="messages",
            max_seq_length=max_seq_length,
        )
    except Exception as e:
        logger.error(f"Error setting up trainer: {e}")
        return
    
    # Train the model
    logger.info("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return
    
    # Save the fine-tuned model
    logger.info("Saving model...")
    try:
        trainer.save_model(str(output_dir / "final"))
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return
    
    logger.info(f"Training complete! Model saved to {output_dir}/final")
    logger.info("To use this model for inference:")
    logger.info("1. Create a GGUF file using the convert_model.py script")
    logger.info("2. Update the MODEL_PATH in server.py to point to your new model")

if __name__ == "__main__":
    main()
