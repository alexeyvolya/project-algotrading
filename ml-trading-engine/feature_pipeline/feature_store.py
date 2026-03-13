import pandas as pd
import os
import yaml
from datetime import datetime

class FeatureStore:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.store_path = self.config['data']['feature_store_path']
        os.makedirs(self.store_path, exist_ok=True)
    
    def save_features(self, df, symbol, date):
        """Save engineered features to parquet"""
        filename = f"{symbol}_features_{date.strftime('%Y%m%d')}.parquet"
        filepath = os.path.join(self.store_path, filename)
        df.to_parquet(filepath)
        return filepath
    
    def load_features(self, symbol, date):
        """Load features from parquet"""
        filename = f"{symbol}_features_{date.strftime('%Y%m%d')}.parquet"
        filepath = os.path.join(self.store_path, filename)
        
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        else:
            return None
    
    def get_latest_features(self, symbol, lookback_days=30):
        """Get latest features for inference"""
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=lookback_days)
        
        all_features = []
        current_date = start_date
        
        while current_date <= end_date:
            df = self.load_features(symbol, current_date.date())
            if df is not None:
                all_features.append(df)
            current_date += pd.Timedelta(days=1)
        
        if all_features:
            return pd.concat(all_features).tail(100)  # Last 100 rows for inference
        else:
            return None