# LLM Twin MVP

A personalized content creation system that fine-tunes an open-source LLM on your digital footprint and uses RAG to generate personalized posts.

## Overview

LLM Twin creates a "digital twin" by:
1. Collecting your content from LinkedIn, Medium, Substack, and GitHub
2. Fine-tuning an open-source LLM (Llama variant) on your content
3. Using Retrieval-Augmented Generation (RAG) to generate personalized posts

## Project Structure

```
LLM_Twin/
├── data_collection/      # Crawlers and data standardization
├── feature_pipeline/     # Preprocessing and embedding generation
├── training_pipeline/    # Fine-tuning module and model registry
├── inference_pipeline/   # RAG module and REST API
├── web_interface/        # User configuration and content generation UI
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── config.py             # Configuration settings
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/OpenLLM.git
   cd OpenLLM/LLM_Twin
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure your environment:
   ```
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

4. Run the application:
   ```
   cd web_interface
   streamlit run app.py
   ```

## Usage

1. Enter your social media URLs in the web interface
2. Trigger data collection to crawl your content
3. Initiate fine-tuning (this may take some time depending on hardware)
4. Generate personalized content by entering prompts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 