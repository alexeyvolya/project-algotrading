import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_ingestion.historical_loader import HistoricalLoader
from feature_pipeline.feature_engineering import FeatureEngineering
from inference.online_predictor import OnlinePredictor
from execution.exchange_connector import ExchangeConnector
from execution.order_manager import OrderManager
from risk.risk_manager import RiskManager
from backtesting.performance_metrics import PerformanceMetrics

class BacktestEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.start_date = pd.to_datetime(self.config['backtest']['start_date'])
        self.end_date = pd.to_datetime(self.config['backtest']['end_date'])
        self.initial_capital = self.config['backtest']['initial_capital']
        
        # Initialize components
        self.data_loader = HistoricalLoader(config_path)
        self.feature_engineer = FeatureEngineering(config_path)
        self.predictor = OnlinePredictor(config_path)
        self.exchange = ExchangeConnector(config_path)
        self.order_manager = OrderManager(self.exchange, config_path)
        self.risk_manager = RiskManager(config_path)
        self.metrics = PerformanceMetrics()
        
        # Reset exchange balance
        self.exchange.balance = self.initial_capital
    
    def run_backtest(self):
        """Run the backtesting simulation"""
        # Load historical data
        data = self.data_loader.load_data(self.start_date, self.end_date)
        
        # Engineer features
        features_df = self.feature_engineer.engineer_features(data)
        
        trades = []
        portfolio_values = []
        
        for i in range(len(features_df)):
            current_data = features_df.iloc[i:i+1]
            
            # Get prediction
            probabilities = self.predictor.predict(current_data)
            signal = self.predictor.generate_signal(probabilities[0])
            
            # Execute trade if signal
            if signal != 'NO_TRADE':
                self.execute_trade(signal, current_data, trades)
            
            # Record portfolio value
            portfolio_values.append({
                'timestamp': current_data.index[0],
                'value': self.exchange.get_balance()
            })
        
        # Calculate performance metrics
        portfolio_df = pd.DataFrame(portfolio_values)
        performance = self.metrics.calculate_metrics(portfolio_df, trades)
        
        return performance, trades, portfolio_df
    
    def execute_trade(self, signal, current_data, trades):
        """Execute a trade in backtest"""
        current_price = current_data['close'].iloc[0]
        atr = current_data['atr'].iloc[0]
        
        # Calculate stop loss
        stop_loss = self.risk_manager.calculate_atr_stop_loss(current_price, atr)
        
        # Calculate position size
        risk_amount = self.risk_manager.calculate_risk_amount(self.exchange.get_balance())
        position_size = self.risk_manager.calculate_position_size(risk_amount, abs(current_price - stop_loss))
        
        # Validate trade
        valid, reason = self.risk_manager.validate_trade(
            signal, position_size, current_price, stop_loss, self.exchange.get_balance()
        )
        
        if not valid:
            return
        
        # Place order
        side = 'BUY' if signal == 'LONG' else 'SELL'
        order = self.order_manager.place_market_order(side, position_size, current_price)
        
        # Record trade
        trade = {
            'timestamp': current_data.index[0],
            'signal': signal,
            'entry_price': order['execution_price'],
            'quantity': position_size,
            'stop_loss': stop_loss
        }
        
        trades.append(trade)