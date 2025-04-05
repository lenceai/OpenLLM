# Import necessary libraries
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import gc

# Clear CUDA cache to free up memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU memory cleared")

# Use a pipeline with quantization
def use_pipeline():
    clear_gpu_memory()
    
    # Define the messages for the pipeline
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    
    # Create a text-generation pipeline with quantization
    try:
        from transformers import BitsAndBytesConfig, AutoModelForCausalLM
        
        # 8-bit quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        print("Loading model with 8-bit quantization...")
        # First load the model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", 
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config
        )
        
        # Then create the pipeline with the loaded model
        pipe = pipeline(
            "text-generation", 
            model=model,
            device_map="auto"
        )
    except (ImportError, Exception) as e:
        print(f"Error with 8-bit quantization: {str(e)}")
        print("Falling back to 4-bit quantization...")
        
        try:
            from transformers import BitsAndBytesConfig, AutoModelForCausalLM
            
            # 4-bit quantization config (more memory efficient)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # First load the model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct", 
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config
            )
            
            # Then create the pipeline with the loaded model
            pipe = pipeline(
                "text-generation", 
                model=model,
                device_map="auto"
            )
        except Exception as e:
            print(f"Error with quantization: {str(e)}")
            print("Falling back to CPU...")
            pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", device="cpu")
    
    # Generate a response
    response = pipe(messages, max_length=200)
    print("Pipeline response:", response)

# Load model directly with quantization
def load_model_directly():
    clear_gpu_memory()
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    try:
        # Try loading with 4-bit quantization (requires bitsandbytes)
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        print("Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Model loaded with 4-bit quantization.")
    except ImportError:
        print("BitsAndBytes library not found. Install with: pip install bitsandbytes>=0.39.0")
        print("Falling back to loading on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.float16,
            device_map="cpu"
        )
    
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

# Execute the functions
if __name__ == "__main__":
    # First make sure requirements are installed
    print("Note: For optimal performance, install these libraries:")
    print("pip install accelerate>=0.26.0 bitsandbytes>=0.39.0")
    
    print("Running pipeline...")
    use_pipeline()
    
    print("\nLoading model directly...")
    model, tokenizer = load_model_directly() 