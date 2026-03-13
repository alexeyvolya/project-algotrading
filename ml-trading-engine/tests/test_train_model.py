import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.train_model import ModelTrainer

class TestModelTrainer:
    def test_create_target(self):
        trainer = ModelTrainer()
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=50, freq='1min')
        data = {
            'open': 90 + np.random.randn(50) * 0.1,
            'high': 90.1 + np.random.randn(50) * 0.1,
            'low': 89.9 + np.random.randn(50) * 0.1,
            'close': 90 + np.random.randn(50) * 0.1,
            'volume': np.random.randint(1000, 10000, 50),
            'log_return': np.random.randn(50) * 0.01,
            'rolling_volatility': np.random.randn(50) * 0.1
        }
        df = pd.DataFrame(data, index=dates)
        
        result = trainer.create_target(df)
        assert 'target' in result.columns
        assert len(result) < len(df)  # Some rows dropped
    
    def test_prepare_features(self):
        trainer = ModelTrainer()
        # Create sample data with target
        dates = pd.date_range('2023-01-01', periods=50, freq='1min')
        data = {
            'open': 90 + np.random.randn(50) * 0.1,
            'high': 90.1 + np.random.randn(50) * 0.1,
            'low': 89.9 + np.random.randn(50) * 0.1,
            'close': 90 + np.random.randn(50) * 0.1,
            'volume': np.random.randint(1000, 10000, 50),
            'log_return': np.random.randn(50) * 0.01,
            'rolling_volatility': np.random.randn(50) * 0.1,
            'target': np.random.randint(0, 2, 50)
        }
        df = pd.DataFrame(data, index=dates)
        
        X, y = trainer.prepare_features(df)
        assert 'open' not in X.columns
        assert 'target' not in X.columns
        assert len(X) == len(y)