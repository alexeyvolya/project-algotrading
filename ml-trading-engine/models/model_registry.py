import os
import joblib
import yaml
from datetime import datetime

class ModelRegistry:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.registry_path = self.config['data']['model_registry_path']
        os.makedirs(self.registry_path, exist_ok=True)
    
    def save_model(self, model, metadata=None):
        """Save model with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"model_{timestamp}.joblib"
        model_path = os.path.join(self.registry_path, model_filename)
        
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = metadata or {}
        metadata.update({
            'timestamp': timestamp,
            'path': model_path
        })
        
        metadata_path = os.path.join(self.registry_path, f"metadata_{timestamp}.yaml")
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        return model_path
    
    def load_latest_model(self):
        """Load the most recent model"""
        model_files = [f for f in os.listdir(self.registry_path) if f.startswith('model_') and f.endswith('.joblib')]
        
        if not model_files:
            return None
        
        # Sort by timestamp (assuming filename format)
        latest_model = sorted(model_files, reverse=True)[0]
        model_path = os.path.join(self.registry_path, latest_model)
        
        return joblib.load(model_path)
    
    def list_models(self):
        """List all saved models"""
        return [f for f in os.listdir(self.registry_path) if f.startswith('model_') and f.endswith('.joblib')]