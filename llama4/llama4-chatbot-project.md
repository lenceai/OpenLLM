# Llama 4 Scout Chatbot for RTX 3090

This project sets up a custom chatbot using Llama 4 Scout on an RTX 3090 GPU, including both inference and fine-tuning capabilities. We'll use aggressive quantization techniques to fit the model within the 24GB VRAM constraint.

## Table of Contents
1. [Requirements](#requirements)
2. [Project Setup](#project-setup)
3. [Download and Quantize the Model](#download-and-quantize-the-model)
4. [Inference Server](#inference-server)
5. [Web UI for Chatbot](#web-ui-for-chatbot)
6. [Fine-tuning the Model](#fine-tuning-the-model)
7. [Running the Complete System](#running-the-complete-system)

## Requirements

- NVIDIA RTX 3090 GPU (24GB VRAM)
- Ubuntu 20.04 or higher (Windows with WSL2 should also work)
- NVIDIA drivers and CUDA 12.1+
- At least 150GB of free disk space
- At least 32GB of system RAM

## Project Setup

First, let's set up our environment:

```bash
# Create a new directory for our project
mkdir llama4-chatbot
cd llama4-chatbot

# Create a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install basic dependencies
pip install --upgrade pip
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install huggingface_hub hf-transfer accelerate unsloth gradio fastapi uvicorn
```

## Download and Quantize the Model

We'll use Unsloth's optimized 1.78-bit quantized version of Llama 4 Scout, which should fit on your RTX 3090:

```python
# Create a download script: download_model.py
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

# Download the optimized quantized model
snapshot_download(
    repo_id="unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
    local_dir="models/unsloth-llama4-scout",
    allow_patterns=["*IQ2_XXS*"],  # Use the smallest quantized version
)
```

Next, we need to install llama.cpp for inference:

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake .. -DLLAMA_CUBLAS=ON -DLLAMA_AVX=ON -DLLAMA_AVX2=ON
make -j

# Move back to the project directory
cd ../..
```

## Inference Server

Create a simple inference server using FastAPI:

```python
# server.py
import os
import subprocess
import json
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import uvicorn
from typing import List, Optional

app = FastAPI()

# Path to the llama.cpp executable and model
LLAMA_CPP_PATH = os.path.join(os.getcwd(), "llama.cpp", "build", "bin", "main")
MODEL_PATH = os.path.join(os.getcwd(), "models", "unsloth-llama4-scout", "Llama-4-Scout-17B-16E-Instruct-unsloth-IQ2_XXS.gguf")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

def format_prompt(messages):
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

@app.post("/chat")
async def chat(request: ChatRequest):
    formatted_prompt = format_prompt(request.messages)
    
    cmd = [
        LLAMA_CPP_PATH,
        "-m", MODEL_PATH,
        "--ctx-size", "16384",
        "--temp", str(request.temperature),
        "--top_p", str(request.top_p),
        "--n-predict", str(request.max_tokens),
        "-n", str(32),  # number of threads - adjust based on your CPU
        "--n-gpu-layers", "35",  # Adjust based on your VRAM availability
        "-p", formatted_prompt
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"llama.cpp error: {result.stderr}")
        
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
                
        return {"response": response_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
```

## Web UI for Chatbot

Create a simple Gradio UI:

```python
# app.py
import gradio as gr
import requests
import json

# Server URL
SERVER_URL = "http://localhost:8000/chat"

def predict(message, history):
    # Convert history and new message to the format expected by our API
    messages = []
    
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    # Add the latest user message
    messages.append({"role": "user", "content": message})
    
    # Call our API
    response = requests.post(
        SERVER_URL,
        json={
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9
        }
    )
    
    if response.status_code != 200:
        return "Error: " + response.text
    
    return response.json()["response"]

# Create the Gradio interface
demo = gr.ChatInterface(
    predict,
    title="Llama 4 Scout Chatbot",
    description="A custom chatbot powered by Llama 4 Scout running on your RTX 3090",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(share=True)
```

## Fine-tuning the Model

For fine-tuning on an RTX 3090, we'll use Unsloth with QLoRA to minimize VRAM usage:

```python
# finetune.py
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Check if CUDA is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load the model (Unsloth optimized 4-bit quantized version)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,  # Load in 4-bit precision
    device_map="auto",
)

# 2. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,       # Rank of the adapters
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05
)

# 3. Prepare your dataset (replace with your own custom dataset)
# This is a simple example using the Anthropic Helpful Assistant dataset
dataset = load_dataset("teknium/openhermes", split="train")

# Format the dataset to match the input format expected by the model
def format_dataset(example):
    # You'll need to modify this function to match your dataset's structure
    messages = []
    if example.get("system_prompt"):
        messages.append({"role": "system", "content": example["system_prompt"]})
    messages.append({"role": "user", "content": example["instruction"]})
    messages.append({"role": "assistant", "content": example["response"]})
    
    return {"messages": messages}

formatted_dataset = dataset.map(format_dataset)
   
# 4. Configure training parameters
training_args = TrainingArguments(
    output_dir="./ft-llama4-scout",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_torch",
    logging_steps=10,
    save_steps=200,
    learning_rate=2e-4,
    weight_decay=0.01,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_total_limit=3,
    fp16=False,
    bf16=False,  # Using 4-bit quantization instead
    max_steps=-1,
    report_to="none",
)

# 5. Create SFT trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    tokenizer=tokenizer,
    peft_config=None,  # We already added LoRA adapters
    dataset_text_field="messages",
    max_seq_length=2048,
)

# 6. Train the model
trainer.train()

# 7. Save the fine-tuned model
trainer.save_model("./ft-llama4-scout-final")

# 8. Save GGUF version of the fine-tuned model for inference
print("Training complete. To convert the model to GGUF format for inference, follow the conversion instructions in the README.")
```

### Converting Fine-tuned Model to GGUF Format

After fine-tuning, you'll need to convert the model to GGUF format for use with llama.cpp:

```bash
# Create a conversion script
# This is a multistep process:
# 1. Merge LoRA weights with the base model
# 2. Convert to GGUF format

# This would be done after fine-tuning
# See the project README for detailed instructions
```

## Running the Complete System

1. **Start the Inference Server**:
   ```bash
   python server.py
   ```

2. **Start the Gradio UI**:
   ```bash
   python app.py
   ```

3. **Access the Chatbot**:
   Open your browser and navigate to `http://localhost:7860`

4. **Fine-tune the Model** (optional):
   ```bash
   python finetune.py
   ```

## Additional Notes

- **Memory Optimization**: You may need to adjust `--n-gpu-layers` in the server.py file depending on your specific VRAM usage. Start with a lower number and increase gradually.
- **Context Length**: The default context length is set to 16384 tokens, but you can adjust this based on your needs and available VRAM.
- **Performance**: Inference speed may be limited due to the quantization and hardware constraints. Expect around 5-10 tokens per second.
- **Custom Dataset**: Replace the example dataset in the fine-tuning script with your own data in the appropriate format.
