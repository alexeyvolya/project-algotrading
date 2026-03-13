import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_ingestion.market_stream import MarketStream
from data_ingestion.historical_loader import HistoricalLoader

class TestMarketStream:
    def test_generate_candle(self):
        stream = MarketStream()
        candle = stream.generate_candle()
        assert isinstance(candle, pd.DataFrame)
        assert len(candle) == 1
        assert all(col in candle.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])

class TestHistoricalLoader:
    def test_load_data(self):
        loader = HistoricalLoader()
        start = pd.Timestamp('2023-01-01')
        end = pd.Timestamp('2023-01-02')
        data = loader.load_data(start, end)
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])