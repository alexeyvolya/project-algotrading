import pandas as pd
import numpy as np
from datetime import datetime
import yaml

class RiskManager:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.risk_per_trade = self.config['trading']['risk_per_trade']
        self.max_drawdown = self.config['trading']['max_drawdown']
        self.max_exposure = self.config['trading']['max_exposure']
        
        self.portfolio_value = 100000  # Initial capital
        self.peak_value = self.portfolio_value
        self.current_drawdown = 0.0
        
    def calculate_atr_stop_loss(self, current_price, atr):
        """Calculate ATR-based stop loss"""
        return current_price - (atr * 2)  # 2 ATR stop loss
    
    def calculate_position_size(self, risk_amount, stop_loss_distance):
        """Calculate position size based on risk"""
        position_size = risk_amount / stop_loss_distance
        return min(position_size, self.max_exposure)
    
    def update_drawdown(self, current_value):
        """Update drawdown calculation"""
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        
        return self.current_drawdown
    
    def check_risk_limits(self, current_value):
        """Check if risk limits are breached"""
        drawdown = self.update_drawdown(current_value)
        
        if drawdown > self.max_drawdown:
            return False, "Max drawdown exceeded"
        
        return True, "OK"
    
    def calculate_risk_amount(self, portfolio_value):
        """Calculate risk amount per trade"""
        return portfolio_value * self.risk_per_trade
    
    def validate_trade(self, side, quantity, entry_price, stop_loss_price, portfolio_value):
        """Validate trade against risk parameters"""
        # Check position size
        risk_amount = self.calculate_risk_amount(portfolio_value)
        stop_distance = abs(entry_price - stop_loss_price)
        
        position_size = self.calculate_position_size(risk_amount, stop_distance)
        if position_size <= 0:
            return False, "Invalid position size"
        
        # Check drawdown
        ok, reason = self.check_risk_limits(portfolio_value)
        if not ok:
            return False, reason
        
        return True, "Trade approved"