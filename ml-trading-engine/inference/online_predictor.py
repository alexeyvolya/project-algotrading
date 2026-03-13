import pandas as pd
import numpy as np
import joblib
import yaml
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model_registry import ModelRegistry

class OnlinePredictor:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.registry = ModelRegistry(config_path)
        self.model = self.registry.load_latest_model()
        
        if self.model is None:
            raise ValueError("No trained model found in registry")
    
    def predict(self, features_df):
        """Make predictions on feature data"""
        # Ensure features are in correct order (same as training)
        feature_cols = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target']]
        X = features_df[feature_cols]
        
        # Get probability predictions
        probas = self.model.predict_proba(X)
        
        # Return probability of positive class (price increase)
        return probas[:, 1]
    
    def generate_signal(self, probability):
        """Generate trading signal based on probability"""
        if probability > 0.60:
            return 'LONG'
        elif probability < 0.40:
            return 'SHORT'
        else:
            return 'NO_TRADE'