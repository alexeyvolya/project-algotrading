import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class OrderManager:
    def __init__(self, exchange_connector, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.exchange = exchange_connector
        self.max_exposure = self.config['trading']['max_exposure']
        
        self.active_orders = []
        self.order_history = []
    
    def calculate_position_size(self, risk_amount, stop_loss_distance):
        """Calculate position size based on risk management"""
        position_size = risk_amount / stop_loss_distance
        return min(position_size, self.max_exposure)
    
    def place_market_order(self, side, quantity, price=None):
        """Place a market order through exchange"""
        order = {
            'order_id': len(self.order_history),
            'timestamp': datetime.now(),
            'type': 'market',
            'side': side,
            'quantity': quantity,
            'price': price,
            'status': 'pending'
        }
        
        # Execute order
        execution = self.exchange.place_order(side, quantity, price)
        
        order.update({
            'status': 'filled',
            'execution_price': execution['price'],
            'commission': execution['commission']
        })
        
        self.active_orders.append(order)
        self.order_history.append(order)
        
        return order
    
    def place_limit_order(self, side, quantity, limit_price):
        """Place a limit order (simplified)"""
        order = {
            'order_id': len(self.order_history),
            'timestamp': datetime.now(),
            'type': 'limit',
            'side': side,
            'quantity': quantity,
            'limit_price': limit_price,
            'status': 'pending'
        }
        
        # For simulation, assume it fills immediately if price is favorable
        current_price = self.exchange.get_current_price()
        
        if (side == 'BUY' and limit_price >= current_price) or (side == 'SELL' and limit_price <= current_price):
            execution = self.exchange.place_order(side, quantity, limit_price)
            order.update({
                'status': 'filled',
                'execution_price': execution['price'],
                'commission': execution['commission']
            })
        else:
            order['status'] = 'cancelled'
        
        self.order_history.append(order)
        
        if order['status'] == 'filled':
            self.active_orders.append(order)
        
        return order
    
    def cancel_order(self, order_id):
        """Cancel a pending order"""
        for i, order in enumerate(self.active_orders):
            if order['order_id'] == order_id and order['status'] == 'pending':
                order['status'] = 'cancelled'
                self.active_orders.pop(i)
                return True
        return False
    
    def get_active_orders(self):
        """Get active orders"""
        return self.active_orders
    
    def get_order_history(self):
        """Get order history"""
        return self.order_history