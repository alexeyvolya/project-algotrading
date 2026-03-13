import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from execution.exchange_connector import ExchangeConnector
from execution.order_manager import OrderManager

class TestExchangeConnector:
    def test_get_current_price(self):
        exchange = ExchangeConnector()
        price = exchange.get_current_price()
        assert isinstance(price, float)
        assert price > 0
    
    def test_place_order(self):
        exchange = ExchangeConnector()
        order = exchange.place_order('BUY', 1000, 90.0)
        assert 'price' in order
        assert 'commission' in order
        assert order['quantity'] == 1000
    
    def test_get_balance(self):
        exchange = ExchangeConnector()
        balance = exchange.get_balance()
        assert balance == 100000  # Initial

class TestOrderManager:
    def test_place_market_order(self):
        exchange = ExchangeConnector()
        order_manager = OrderManager(exchange)
        order = order_manager.place_market_order('BUY', 1000, 90.0)
        assert order['status'] == 'filled'
        assert 'execution_price' in order
    
    def test_place_limit_order(self):
        exchange = ExchangeConnector()
        order_manager = OrderManager(exchange)
        order = order_manager.place_limit_order('BUY', 1000, 89.9)
        assert order['status'] in ['filled', 'cancelled']
    
    def test_get_active_orders(self):
        exchange = ExchangeConnector()
        order_manager = OrderManager(exchange)
        orders = order_manager.get_active_orders()
        assert isinstance(orders, list)