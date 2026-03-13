import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import yaml

class HistoricalLoader:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.symbol = self.config['trading']['symbol']
        self.data_path = self.config['data']['historical_data_path']
        os.makedirs(self.data_path, exist_ok=True)
        
    def generate_historical_data(self, start_date, end_date, timeframe='1m'):
        """Generate simulated historical OHLCV data"""
        current_date = start_date
        data = []
        
        price = 90.0  # Starting USDRUB price
        
        while current_date <= end_date:
            # Generate daily candles if timeframe is 1m, but for simplicity, generate minute data
            for hour in range(24):
                for minute in range(60):
                    # Random walk
                    price_change = np.random.normal(0, 0.0001)  # Small changes
                    price *= (1 + price_change)
                    
                    high = price * (1 + abs(np.random.normal(0, 0.0002)))
                    low = price * (1 - abs(np.random.normal(0, 0.0002)))
                    open_price = price
                    close = price * (1 + np.random.normal(0, 0.0001))
                    volume = np.random.randint(100, 1000)
                    
                    data.append({
                        'timestamp': current_date + timedelta(hours=hour, minutes=minute),
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'close': close,
                        'volume': volume
                    })
                    
                    price = close
            
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    
    def load_data(self, start_date, end_date):
        """Load historical data, generate if not exists"""
        filename = f"{self.symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.data_path, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        else:
            df = self.generate_historical_data(start_date, end_date)
            df.to_csv(filepath)
        
        return df