import pandas as pd
import time
from datetime import datetime
import yaml
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_ingestion.market_stream import MarketStream
from feature_pipeline.feature_engineering import FeatureEngineering
from feature_pipeline.feature_store import FeatureStore
from inference.online_predictor import OnlinePredictor
from execution.exchange_connector import ExchangeConnector
from execution.order_manager import OrderManager
from risk.risk_manager import RiskManager

class TradingLoop:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.market_stream = MarketStream(config_path)
        self.feature_engineer = FeatureEngineering(config_path)
        self.feature_store = FeatureStore(config_path)
        self.predictor = OnlinePredictor(config_path)
        self.exchange = ExchangeConnector(config_path)
        self.order_manager = OrderManager(self.exchange, config_path)
        self.risk_manager = RiskManager(config_path)
        
        self.recent_data = []  # Store recent market data for feature calculation
    
    def run_live_trading(self):
        """Main live trading loop"""
        print("Starting live trading loop...")
        
        for candle_df in self.market_stream.stream_data():
            try:
                # Step 1: Ingest new market data
                self.recent_data.append(candle_df)
                
                # Keep only recent data (last 100 candles)
                if len(self.recent_data) > 100:
                    self.recent_data = self.recent_data[-100:]
                
                # Combine recent data
                if len(self.recent_data) >= 20:  # Need minimum data for features
                    market_data = pd.concat(self.recent_data)
                    
                    # Step 2: Compute features
                    features_df = self.feature_engineer.engineer_features(market_data)
                    
                    if not features_df.empty:
                        # Step 3: Perform ML inference
                        latest_features = features_df.tail(1)
                        probabilities = self.predictor.predict(latest_features)
                        signal = self.predictor.generate_signal(probabilities[0])
                        
                        # Step 4: Apply risk management
                        if signal != 'NO_TRADE':
                            current_price = latest_features['close'].iloc[0]
                            atr = latest_features['atr'].iloc[0]
                            stop_loss = self.risk_manager.calculate_atr_stop_loss(current_price, atr)
                            
                            risk_amount = self.risk_manager.calculate_risk_amount(self.exchange.get_balance())
                            position_size = self.risk_manager.calculate_position_size(
                                risk_amount, abs(current_price - stop_loss)
                            )
                            
                            # Validate trade
                            valid, reason = self.risk_manager.validate_trade(
                                signal, position_size, current_price, stop_loss, self.exchange.get_balance()
                            )
                            
                            if valid:
                                # Step 5: Send order to execution engine
                                side = 'BUY' if signal == 'LONG' else 'SELL'
                                order = self.order_manager.place_market_order(side, position_size, current_price)
                                
                                print(f"Executed {signal} order: {order}")
                            else:
                                print(f"Trade rejected: {reason}")
                        
                        # Step 6: Log current state
                        print(f"Portfolio value: ${self.exchange.get_balance():.2f}, Signal: {signal}")
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                continue
        
        print("Trading loop stopped.")

if __name__ == "__main__":
    loop = TradingLoop()
    loop.run_live_trading()