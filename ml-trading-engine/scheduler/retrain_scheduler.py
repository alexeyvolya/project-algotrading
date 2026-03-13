import schedule
import time
from datetime import datetime, timedelta
import yaml
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_ingestion.historical_loader import HistoricalLoader
from feature_pipeline.feature_engineering import FeatureEngineering
from models.train_model import ModelTrainer
from models.model_registry import ModelRegistry

class RetrainScheduler:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_loader = HistoricalLoader(config_path)
        self.feature_engineer = FeatureEngineering(config_path)
        self.trainer = ModelTrainer(config_path)
        self.registry = ModelRegistry(config_path)
    
    def retrain_model(self):
        """Retrain the model with latest data"""
        print(f"Starting model retraining at {datetime.now()}")
        
        # Load recent data (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = self.data_loader.load_data(start_date, end_date)
        features_df = self.feature_engineer.engineer_features(data)
        
        # Train model
        model_path = f"models/registry/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model, importance = self.trainer.train_and_save(features_df, model_path)
        
        # Save to registry
        self.registry.save_model(model, {'importance': importance})
        
        print("Model retraining completed")
    
    def start_scheduler(self):
        """Start the retraining scheduler (every 24 hours)"""
        schedule.every(24).hours.do(self.retrain_model)
        
        print("Retraining scheduler started. Model will retrain every 24 hours.")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute