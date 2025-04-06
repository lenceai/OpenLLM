"""
Model fine-tuner module for fine-tuning a pre-trained LLM on user data.
"""

import os
import logging
import sys
import json
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASE_MODEL_NAME, 
    MODELS_DIR, 
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    MAX_LENGTH,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ModelFineTuner")

class ModelFineTuner:
    """Fine-tune a pre-trained LLM on user-specific data."""
    
    def __init__(self, 
                 base_model_name: str = BASE_MODEL_NAME,
                 models_dir: Path = MODELS_DIR,
                 learning_rate: float = LEARNING_RATE,
                 batch_size: int = BATCH_SIZE,
                 num_epochs: int = NUM_EPOCHS,
                 max_length: int = MAX_LENGTH,
                 lora_r: int = LORA_R,
                 lora_alpha: int = LORA_ALPHA,
                 lora_dropout: float = LORA_DROPOUT):
        """
        Initialize the model fine-tuner.
        
        Args:
            base_model_name: Name of the pre-trained model to fine-tune.
            models_dir: Directory to save fine-tuned models.
            learning_rate: Learning rate for fine-tuning.
            batch_size: Batch size for training.
            num_epochs: Number of training epochs.
            max_length: Maximum sequence length.
            lora_r: LoRA rank.
            lora_alpha: LoRA alpha.
            lora_dropout: LoRA dropout rate.
        """
        self.base_model_name = base_model_name
        self.models_dir = models_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.logger = logger
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
    
    def _format_data_for_training(self, items: List[Dict[str, Any]]) -> List[str]:
        """
        Format data for training.
        
        Args:
            items: List of dictionaries containing data to format.
            
        Returns:
            List of formatted strings for training.
        """
        formatted_data = []
        
        for item in items:
            # Extract relevant fields
            source = item.get('source', 'unknown')
            content_type = item.get('type', 'content')
            title = item.get('title', '')
            content = item.get('content', '')
            
            # Skip items without content
            if not content:
                continue
            
            # Format the training data with source and type information
            formatted_text = f"Source: {source}\nType: {content_type}\n"
            
            # Add title if available
            if title:
                formatted_text += f"Title: {title}\n"
            
            # Add content
            formatted_text += f"Content: {content}\n\n"
            
            formatted_data.append(formatted_text)
        
        return formatted_data
    
    def _prepare_dataset(self, formatted_data: List[str]) -> Dataset:
        """
        Prepare the dataset for training.
        
        Args:
            formatted_data: List of formatted strings for training.
            
        Returns:
            HuggingFace Dataset.
        """
        # Create dataset
        dataset = Dataset.from_dict({"text": formatted_data})
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        self.logger.info(f"Loading model: {self.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Ensure the tokenizer has a pad_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set up quantization configuration for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Set up LoRA configuration
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Create PEFT model
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print model information
        self.peft_model.print_trainable_parameters()
    
    def fine_tune(self, items: List[Dict[str, Any]]) -> str:
        """
        Fine-tune the model on user data.
        
        Args:
            items: List of dictionaries containing data to fine-tune on.
            
        Returns:
            Path to the saved model.
        """
        # Format data for training
        self.logger.info("Formatting data for training")
        formatted_data = self._format_data_for_training(items)
        
        if not formatted_data:
            self.logger.error("No valid data for fine-tuning")
            return ""
        
        # Load model and tokenizer if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # Prepare dataset
        self.logger.info("Preparing dataset")
        tokenized_dataset = self._prepare_dataset(formatted_data)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.models_dir / f"fine_tuned_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            save_total_limit=2,
            logging_steps=10,
            fp16=True,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            report_to="none"
        )
        
        # Set up data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        # Train the model
        self.logger.info("Starting fine-tuning")
        trainer.train()
        
        # Save the model and tokenizer
        self.logger.info(f"Saving fine-tuned model to {output_dir}")
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save model metadata
        model_metadata = {
            "base_model": self.base_model_name,
            "fine_tune_timestamp": timestamp,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "max_length": self.max_length,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "num_training_samples": len(formatted_data)
        }
        
        with open(output_dir / "model_metadata.json", "w") as f:
            json.dump(model_metadata, f, indent=4)
        
        self.logger.info("Fine-tuning completed successfully")
        
        return str(output_dir) 