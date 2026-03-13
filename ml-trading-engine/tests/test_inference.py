import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.online_predictor import OnlinePredictor
from models.train_model import ModelTrainer
from models.model_registry import ModelRegistry

class TestOnlinePredictor:
    def test_predict(self):
        # First, train a model
        trainer = ModelTrainer()
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='1min')
        data = {
            'open': 90 + np.random.randn(100) * 0.1,
            'high': 90.1 + np.random.randn(100) * 0.1,
            'low': 89.9 + np.random.randn(100) * 0.1,
            'close': 90 + np.random.randn(100) * 0.1,
            'volume': np.random.randint(1000, 10000, 100),
            'log_return': np.random.randn(100) * 0.01,
            'rolling_volatility': np.random.randn(100) * 0.1,
            'momentum_5': np.random.randn(100) * 0.1,
            'sma_20': 90 + np.random.randn(100) * 0.1,
            'ema_20': 90 + np.random.randn(100) * 0.1,
            'rsi': 50 + np.random.randn(100) * 10,
            'atr': 0.1 + np.random.randn(100) * 0.01,
            'macd': np.random.randn(100) * 0.01,
            'macd_signal': np.random.randn(100) * 0.01,
            'macd_diff': np.random.randn(100) * 0.01,
            'bb_high': 90.2 + np.random.randn(100) * 0.1,
            'bb_low': 89.8 + np.random.randn(100) * 0.1,
            'bb_middle': 90 + np.random.randn(100) * 0.1,
            'volume_imbalance': np.random.randn(100) * 0.1,
            'volume_sma_20': np.random.randint(1000, 10000, 100),
            'volume_ratio': 1 + np.random.randn(100) * 0.1,
            'realized_vol': np.random.randn(100) * 0.1,
            'vol_breakout': np.random.randn(100) * 0.1
        }
        df = pd.DataFrame(data, index=dates)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.joblib')
            model, _ = trainer.train_and_save(df, model_path)
            
            # Mock registry
            registry = ModelRegistry()
            registry.save_model(model, {'test': True})
            
            predictor = OnlinePredictor()
            # Override model loading for test
            predictor.model = model
            
            # Test prediction
            features_df = df.tail(1)
            probabilities = predictor.predict(features_df)
            assert len(probabilities) == 1
            assert 0 <= probabilities[0] <= 1
    
    def test_generate_signal(self):
        predictor = OnlinePredictor()
        predictor.model = None  # Mock
        
        signal = predictor.generate_signal(0.7)
        assert signal == 'LONG'
        
        signal = predictor.generate_signal(0.3)
        assert signal == 'SHORT'
        
        signal = predictor.generate_signal(0.5)
        assert signal == 'NO_TRADE'