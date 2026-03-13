import pandas as pd
import numpy as np
import ta
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
import yaml

class FeatureEngineering:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def add_price_features(self, df):
        """Add price-based features"""
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Rolling volatility (20-period)
        df['rolling_volatility'] = df['log_return'].rolling(window=20).std()
        
        # Price momentum (5-period)
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        return df
    
    def add_technical_indicators(self, df):
        """Add technical indicators using ta library"""
        # RSI
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()
        
        # ATR
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr.average_true_range()
        
        # MACD
        macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        
        return df
    
    def add_microstructure_features(self, df):
        """Add microstructure features"""
        # Volume imbalance
        df['volume_imbalance'] = (df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Rolling volume averages
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        return df
    
    def add_volatility_features(self, df):
        """Add volatility regime features"""
        # Realized volatility (daily)
        df['realized_vol'] = df['log_return'].rolling(window=1440).std()  # Assuming 1m data, 1440 min = 1 day
        
        # Volatility breakout signal
        df['vol_breakout'] = (df['close'] - df['close'].shift(1)).abs() / df['atr']
        
        return df
    
    def engineer_features(self, df):
        """Apply all feature engineering"""
        df = df.copy()
        df = self.add_price_features(df)
        df = self.add_technical_indicators(df)
        df = self.add_microstructure_features(df)
        df = self.add_volatility_features(df)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df