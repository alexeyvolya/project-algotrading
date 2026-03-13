#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backtesting.backtest_engine import BacktestEngine

def main():
    engine = BacktestEngine()
    performance, trades, portfolio = engine.run_backtest()

    print("Backtesting completed.")
    print("Performance Metrics:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()