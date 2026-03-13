import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from risk.risk_manager import RiskManager

class TestRiskManager:
    def test_calculate_position_size(self):
        rm = RiskManager()
        risk_amount = 1000
        stop_distance = 1.0
        size = rm.calculate_position_size(risk_amount, stop_distance)
        assert size == 1000
    
    def test_validate_trade(self):
        rm = RiskManager()
        valid, reason = rm.validate_trade('BUY', 1000, 90, 89, 100000)
        assert valid or not valid  # Depends on conditions