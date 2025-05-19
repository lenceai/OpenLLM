# README.md

# Llama 4 Scout Chatbot on RTX 3090

This project sets up a custom Llama 4 Scout chatbot that can run on an RTX 3090 GPU. It includes code for both inference and fine-tuning.

## Important Notes Before Starting

The RTX 3090 has 24GB of VRAM, which is the minimum required to run a heavily quantized version of Llama 4 Scout. Here's what to expect:

- **Performance**: Inference speed will be around 5-15 tokens per second with the quantized model
- **Quality**: The heavily quantized version maintains reasonable quality but may not match the full model
- **Fine-tuning**: We use QLoRA to make fine-tuning possible with limited VRAM
- **Storage**: You'll need about 150GB of free disk space

## Quick Start Guide

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/llama4-chatbot
   cd llama4-chatbot
   ```

2. **Set up the environment**
   ```bash
   conda create -n llama4 python=3.11
   conda activate llama4
   python -m pip install --upgrade pip 
   pip install -r requirements.txt
   ```

3. **Download and prepare the model**
   ```bash
   python download-script.py
   ```

4. **Start the inference server**
   ```bash
   python server.py
   ```

5. **Start the web UI**
   ```bash
   python app.py
   ```

## Directory Structure

```
llama4-chatbot/
├── app.py                   # Gradio web UI
├── server.py                # FastAPI inference server
├── finetune.py              # Fine-tuning script
├── scripts/
│   ├── download_model.py    # Downloads model
│   └── convert_model.py     # Converts fine-tuned model to GGUF
├── models/                  # Models will be stored here
├── data/                    # Place your training data here
├── venv/                    # Virtual environment
└── README.md                # This file
```

## Troubleshooting

### Common Issues and Solutions

1. **Out of Memory (OOM) Errors**
   - Reduce the number of GPU layers with `--n-gpu-layers`
   - Reduce context size with `--ctx-size`
   - Ensure no other GPU processes are running

2. **Slow Inference Speed**
   - Check GPU utilization with `nvidia-smi`
   - Adjust thread count to match your CPU cores
   - Consider using more aggressive quantization

3. **Model Loading Errors**
   - Ensure you have the correct GGUF file path
   - Verify CUDA and GPU drivers are properly installed

## Fine-tuning Guide

To fine-tune the model with your own data:

1. **Prepare your dataset**
   - Create a JSON file with your training examples
   - Format it as described in the documentation
   - Place it in the `data/` directory

2. **Modify finetune.py**
   - Update the dataset loading code to use your data
   - Adjust hyperparameters as needed

3. **Run the fine-tuning**
   ```bash
   python finetune.py
   ```

4. **Convert to GGUF format**
   ```bash
   python scripts/convert_model.py
   ```

5. **Use your fine-tuned model**
   - Update the model path in `server.py`
   - Restart the server

## Advanced Configuration

For advanced users, you can modify these parameters:

- **Prompt Format**: Edit the `format_prompt` function in `server.py`
- **Inference Parameters**: Adjust temperature, top_p, etc. in both `server.py` and `app.py`
- **QLoRA Settings**: Modify rank, alpha, and target modules in `finetune.py`

## Acknowledgements

This project uses:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient inference
- [Unsloth](https://github.com/unsloth-ai/unsloth) for optimized fine-tuning
- [Gradio](https://gradio.app/) for the web interface
- [FastAPI](https://fastapi.tiangolo.com/) for the API server

## License

This project is licensed under the MIT License - see the LICENSE file for details.
