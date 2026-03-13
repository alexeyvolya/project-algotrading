import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yaml

class MarketStream:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.symbol = self.config['trading']['symbol']
        self.timeframe = self.config['trading']['timeframe']
        
        # Simulate initial price
        self.current_price = 90.0  # USDRUB around 90
        self.last_timestamp = datetime.now()
        
    def generate_candle(self):
        """Generate a simulated OHLCV candle"""
        # Random walk with some volatility
        price_change = np.random.normal(0, 0.001)  # 0.1% std dev
        self.current_price *= (1 + price_change)
        
        # Generate OHLC
        high = self.current_price * (1 + abs(np.random.normal(0, 0.0005)))
        low = self.current_price * (1 - abs(np.random.normal(0, 0.0005)))
        open_price = self.last_price if hasattr(self, 'last_price') else self.current_price
        close = self.current_price
        
        # Volume
        volume = np.random.randint(1000, 10000)
        
        self.last_price = close
        self.last_timestamp += timedelta(minutes=1)
        
        return {
            'timestamp': self.last_timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }
    
    def stream_data(self):
        """Generator that yields new candles"""
        while True:
            candle = self.generate_candle()
            yield pd.DataFrame([candle])
            time.sleep(1)  # Simulate real-time streaming