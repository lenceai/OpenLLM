#!/usr/bin/env python3
# app.py

import gradio as gr
import requests
import json
import logging
import os
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Server URL
SERVER_URL = "http://localhost:8000/chat"
HEALTH_URL = "http://localhost:8000/health"
GPU_INFO_URL = "http://localhost:8000/gpu-info"

# Check server connection
def check_server_connection():
    try:
        response = requests.get(HEALTH_URL)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                return True, f"Connected to server. Model: {data.get('model', 'unknown')}"
        return False, f"Server error: {response.text}"
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to server. Make sure server.py is running."
    except Exception as e:
        return False, f"Error: {str(e)}"

# Get GPU info
def get_gpu_info():
    try:
        response = requests.get(GPU_INFO_URL)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok" and data.get("gpus"):
                gpu = data["gpus"][0]  # Assuming we're using the first GPU
                return (
                    f"GPU: {gpu['name']}\n"
                    f"Memory: {gpu['memory_used']} / {gpu['memory_total']} used"
                )
        return "GPU info not available"
    except Exception:
        return "GPU info not available"

def predict(
    message: str, 
    history: List[Tuple[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    n_gpu_layers: int
) -> str:
    """Send message to API and get response"""
    # Convert history and new message to the format expected by our API
    messages = []
    
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    # Add the latest user message
    messages.append({"role": "user", "content": message})
    
    # Call our API
    try:
        response = requests.post(
            SERVER_URL,
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n_gpu_layers": n_gpu_layers
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Server error: {response.text}")
            return f"Error: The server returned status code {response.status_code}. Please check the server logs."
        
        data = response.json()
        return data["response"]
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the server. Make sure server.py is running on port 8000."
    except Exception as e:
        logger.error(f"Error calling API: {str(e)}")
        return f"Error: {str(e)}"

# Check connection on startup
server_ok, server_message = check_server_connection()
if not server_ok:
    logger.warning(server_message)

# Custom CSS
css = """
.container { max-width: 800px; margin: auto; }
.status-connected { color: green; }
.status-disconnected { color: red; }
.gpu-info { font-family: monospace; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Llama 4 Scout Chatbot")
    
    with gr.Row():
        with gr.Column(scale=3):
            server_status = gr.Markdown(
                f"**Server Status**: {'<span class=\"status-connected\">Connected</span>' if server_ok else '<span class=\"status-disconnected\">Disconnected</span>'}"
            )
        with gr.Column(scale=2):
            gpu_info_text = gr.Markdown(f"<div class='gpu-info'>{get_gpu_info() if server_ok else 'GPU info not available'}</div>")
            refresh_btn = gr.Button("Refresh Status")
    
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(placeholder="Type your message here...", lines=3)
            clear = gr.Button("Clear")
        
        with gr.Column():
            gr.Markdown("### Model Parameters")
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.1, label="Top P")
            max_tokens = gr.Slider(64, 1024, value=512, step=64, label="Max Tokens")
            n_gpu_layers = gr.Slider(1, 45, value=35, step=1, label="GPU Layers")
    
    def refresh_status():
        server_ok, server_message = check_server_connection()
        status_html = f"**Server Status**: {'<span class=\"status-connected\">Connected</span>' if server_ok else '<span class=\"status-disconnected\">Disconnected</span>'}"
        gpu_html = f"<div class='gpu-info'>{get_gpu_info() if server_ok else 'GPU info not available'}</div>"
        return status_html, gpu_html
    
    refresh_btn.click(refresh_status, outputs=[server_status, gpu_info_text])
    
    msg.submit(
        predict, 
        [msg, chatbot, temperature, top_p, max_tokens, n_gpu_layers], 
        chatbot
    ).then(
        lambda: "", 
        None, 
        msg
    )
    
    clear.click(lambda: None, None, chatbot)
    
    gr.Markdown("""
    ## About
    
    This chatbot is powered by Llama 4 Scout running locally on your RTX 3090 GPU.
    
    - Adjust the temperature for more or less creative responses
    - Increase or decrease GPU layers if you experience memory issues
    - The model has been quantized to fit within your GPU's 24GB of VRAM
    
    Visit the GitHub repository for more information and customization options.
    """)

if __name__ == "__main__":
    demo.launch(share=False)
