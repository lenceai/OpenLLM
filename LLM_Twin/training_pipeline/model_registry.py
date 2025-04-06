"""
Model registry for tracking and managing fine-tuned models.
"""

import os
import logging
import sys
import json
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ModelRegistry")

class ModelRegistry:
    """Registry for tracking and managing fine-tuned models."""
    
    def __init__(self, models_dir: Path = MODELS_DIR):
        """
        Initialize the model registry.
        
        Args:
            models_dir: Directory containing fine-tuned models.
        """
        self.models_dir = models_dir
        self.registry_file = models_dir / "model_registry.json"
        self.registry = {}
        self.logger = logger
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load existing registry if available
        self._load_registry()
    
    def _load_registry(self):
        """Load the model registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    self.registry = json.load(f)
                self.logger.info(f"Loaded model registry with {len(self.registry)} models")
            except Exception as e:
                self.logger.error(f"Error loading model registry: {str(e)}")
                self.registry = {}
        else:
            self.logger.info("No existing model registry found. Creating new registry.")
            self.registry = {}
    
    def _save_registry(self):
        """Save the model registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=4)
            self.logger.info(f"Saved model registry with {len(self.registry)} models")
        except Exception as e:
            self.logger.error(f"Error saving model registry: {str(e)}")
    
    def register_model(self, model_path: Path, model_name: Optional[str] = None, description: str = "") -> str:
        """
        Register a fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model.
            model_name: Optional custom name for the model.
            description: Optional description of the model.
            
        Returns:
            Model ID.
        """
        model_path = Path(model_path)
        
        # Check if model exists
        if not model_path.exists():
            self.logger.error(f"Model path does not exist: {model_path}")
            return ""
        
        # Read model metadata
        metadata_file = model_path / "model_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading model metadata: {str(e)}")
                metadata = {}
        else:
            metadata = {}
        
        # Generate model ID (timestamp if not already in path name)
        if "fine_tuned_" in model_path.name:
            model_id = model_path.name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"fine_tuned_{timestamp}"
        
        # Generate model name if not provided
        if not model_name:
            model_name = f"Model {model_id}"
        
        # Create registry entry
        registry_entry = {
            "model_id": model_id,
            "model_name": model_name,
            "model_path": str(model_path),
            "description": description,
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        # Add to registry
        self.registry[model_id] = registry_entry
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Registered model with ID: {model_id}")
        
        return model_id
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get a model from the registry.
        
        Args:
            model_id: ID of the model to get.
            
        Returns:
            Dictionary containing model information.
        """
        if model_id in self.registry:
            return self.registry[model_id]
        else:
            self.logger.error(f"Model not found in registry: {model_id}")
            return {}
    
    def get_latest_model(self) -> Dict[str, Any]:
        """
        Get the latest model from the registry.
        
        Returns:
            Dictionary containing model information.
        """
        if not self.registry:
            self.logger.error("No models in registry")
            return {}
        
        # Get latest model based on registration time
        latest_model_id = max(
            self.registry.keys(),
            key=lambda model_id: self.registry[model_id].get("registered_at", "")
        )
        
        return self.registry[latest_model_id]
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models in the registry.
        
        Returns:
            List of dictionaries containing model information.
        """
        return list(self.registry.values())
    
    def delete_model(self, model_id: str, delete_files: bool = True) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: ID of the model to delete.
            delete_files: Whether to delete the model files.
            
        Returns:
            True if successful, False otherwise.
        """
        if model_id not in self.registry:
            self.logger.error(f"Model not found in registry: {model_id}")
            return False
        
        # Delete model files if requested
        if delete_files:
            model_path = Path(self.registry[model_id]["model_path"])
            if model_path.exists():
                try:
                    shutil.rmtree(model_path)
                    self.logger.info(f"Deleted model files at: {model_path}")
                except Exception as e:
                    self.logger.error(f"Error deleting model files: {str(e)}")
                    return False
        
        # Remove from registry
        del self.registry[model_id]
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Deleted model with ID: {model_id}")
        
        return True
    
    def update_model(self, model_id: str, model_name: Optional[str] = None, description: Optional[str] = None) -> bool:
        """
        Update model information in the registry.
        
        Args:
            model_id: ID of the model to update.
            model_name: New name for the model.
            description: New description for the model.
            
        Returns:
            True if successful, False otherwise.
        """
        if model_id not in self.registry:
            self.logger.error(f"Model not found in registry: {model_id}")
            return False
        
        # Update model information
        if model_name:
            self.registry[model_id]["model_name"] = model_name
        
        if description:
            self.registry[model_id]["description"] = description
        
        # Update timestamp
        self.registry[model_id]["updated_at"] = datetime.now().isoformat()
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Updated model with ID: {model_id}")
        
        return True 