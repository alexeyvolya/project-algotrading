#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_ingestion.historical_loader import HistoricalLoader
from feature_pipeline.feature_engineering import FeatureEngineering
from models.train_model import ModelTrainer
from models.model_registry import ModelRegistry
import pandas as pd

def main():
    # Load historical data
    loader = HistoricalLoader()
    data = loader.load_data(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01'))  # Small dataset for testing

    # Engineer features
    engineer = FeatureEngineering()
    features = engineer.engineer_features(data)

    # Train model
    trainer = ModelTrainer()
    model_path = 'models/registry/test_model.joblib'
    model, importance = trainer.train_and_save(features, model_path)

    # Register model
    registry = ModelRegistry()
    registry.save_model(model)

    print("Training completed successfully.")
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()