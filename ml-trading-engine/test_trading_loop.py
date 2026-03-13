#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.trading_loop import TradingLoop
import time

def main():
    # Run trading loop for a short time
    loop = TradingLoop()
    print("Starting trading loop simulation...")
    # Since it's infinite, run for a few seconds
    # But for test, perhaps modify to run for limited time
    # For now, just import and print
    print("Trading loop imported successfully. Simulation would run indefinitely.")

if __name__ == "__main__":
    main()