import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from feature_pipeline.feature_engineering import FeatureEngineering

class TestFeatureEngineering:
    def test_engineer_features(self):
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='1min')
        data = {
            'open': 90 + np.random.randn(100) * 0.1,
            'high': 90.1 + np.random.randn(100) * 0.1,
            'low': 89.9 + np.random.randn(100) * 0.1,
            'close': 90 + np.random.randn(100) * 0.1,
            'volume': np.random.randint(1000, 10000, 100)
        }
        df = pd.DataFrame(data, index=dates)
        
        engineer = FeatureEngineering()
        features = engineer.engineer_features(df)
        
        assert isinstance(features, pd.DataFrame)
        assert 'log_return' in features.columns
        assert 'rsi' in features.columns
        assert 'macd' in features.columns