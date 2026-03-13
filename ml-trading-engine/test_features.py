#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from feature_pipeline.feature_engineering import FeatureEngineering

def main():
    # Create sample OHLCV data
    dates = pd.date_range('2023-01-01', periods=100, freq='1min')
    np.random.seed(42)
    data = {
        'open': 90 + np.random.randn(100) * 0.1,
        'high': 90.1 + np.random.randn(100) * 0.1,
        'low': 89.9 + np.random.randn(100) * 0.1,
        'close': 90 + np.random.randn(100) * 0.1,
        'volume': np.random.randint(1000, 10000, 100)
    }
    df = pd.DataFrame(data, index=dates)

    # Engineer features
    engineer = FeatureEngineering()
    features_df = engineer.engineer_features(df)

    print("Feature engineering successful.")
    print(f"Original shape: {df.shape}")
    print(f"Features shape: {features_df.shape}")
    print("Sample features:")
    print(features_df.head())

if __name__ == "__main__":
    main()