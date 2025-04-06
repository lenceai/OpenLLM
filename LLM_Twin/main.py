"""
Main entry point for the LLM Twin application.
"""

import os
import sys
import logging
import argparse
import time
import subprocess
from pathlib import Path
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Main")

def run_api():
    """Run the inference API server."""
    logger.info("Starting API server...")
    from inference_pipeline.api import run_api
    run_api()

def run_web_interface():
    """Run the web interface."""
    logger.info("Starting web interface...")
    subprocess.run(["streamlit", "run", "web_interface/app.py"])

def print_welcome_message():
    """Print a welcome message with instructions."""
    welcome_message = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║                        🤖 LLM TWIN MVP                        ║
    ║                                                               ║
    ║   Personalize content creation with your digital footprint    ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    The application is starting up...
    
    • API server: http://localhost:8000
    • Web interface: http://localhost:8501
    
    Use the web interface to:
    1. Configure your data sources (LinkedIn, GitHub, Medium, Substack)
    2. Collect data from your digital footprint
    3. Create embeddings for Retrieval-Augmented Generation (RAG)
    4. Fine-tune an LLM on your data
    5. Generate personalized content
    
    Press Ctrl+C to shut down the application.
    """
    print(welcome_message)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LLM Twin application")
    parser.add_argument("--api-only", action="store_true", help="Run only the API server")
    parser.add_argument("--web-only", action="store_true", help="Run only the web interface")
    args = parser.parse_args()
    
    print_welcome_message()
    
    try:
        if args.api_only:
            # Run only the API server
            run_api()
        elif args.web_only:
            # Run only the web interface
            run_web_interface()
        else:
            # Run both in separate threads
            api_thread = threading.Thread(target=run_api)
            web_thread = threading.Thread(target=run_web_interface)
            
            # Start threads
            api_thread.daemon = True
            web_thread.daemon = True
            
            api_thread.start()
            # Wait a bit for API to start
            time.sleep(2)
            web_thread.start()
            
            # Wait for threads to finish (this keeps the main thread alive)
            api_thread.join()
            web_thread.join()
    
    except KeyboardInterrupt:
        logger.info("Shutting down the application...")
    except Exception as e:
        logger.error(f"Error running the application: {str(e)}")

if __name__ == "__main__":
    main() 