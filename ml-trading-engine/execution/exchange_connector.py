import pandas as pd
import numpy as np
from datetime import datetime
import yaml

class ExchangeConnector:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.slippage = self.config['execution']['slippage']
        self.commission = self.config['execution']['commission']
        
        # Simulated market state
        self.current_price = 90.0
        self.positions = []  # List of open positions
        self.balance = 100000  # Starting balance
        
    def get_current_price(self):
        """Get current market price"""
        return self.current_price
    
    def place_order(self, side, quantity, price=None):
        """Place a market order"""
        if price is None:
            price = self.current_price
        
        # Apply slippage
        if side == 'BUY':
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)
        
        # Calculate commission
        commission_amount = execution_price * quantity * self.commission
        
        # Update balance
        if side == 'BUY':
            total_cost = execution_price * quantity + commission_amount
            self.balance -= total_cost
        else:
            total_revenue = execution_price * quantity - commission_amount
            self.balance += total_revenue
        
        # Record position
        position = {
            'timestamp': datetime.now(),
            'side': side,
            'quantity': quantity,
            'price': execution_price,
            'commission': commission_amount
        }
        
        self.positions.append(position)
        
        return position
    
    def close_position(self, position_index, current_price):
        """Close a position"""
        if position_index >= len(self.positions):
            return None
        
        position = self.positions[position_index]
        
        # Calculate PnL
        if position['side'] == 'BUY':
            pnl = (current_price - position['price']) * position['quantity']
        else:
            pnl = (position['price'] - current_price) * position['quantity']
        
        # Apply commission
        commission = current_price * position['quantity'] * self.commission
        pnl -= commission
        
        # Update balance
        self.balance += pnl
        
        # Remove position
        closed_position = self.positions.pop(position_index)
        closed_position['pnl'] = pnl
        closed_position['close_price'] = current_price
        
        return closed_position
    
    def get_positions(self):
        """Get current open positions"""
        return self.positions
    
    def get_balance(self):
        """Get current balance"""
        return self.balance