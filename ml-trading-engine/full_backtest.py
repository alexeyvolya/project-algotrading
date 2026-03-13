#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_ingestion.historical_loader import HistoricalLoader
from feature_pipeline.feature_engineering import FeatureEngineering
from models.train_model import ModelTrainer
from models.model_registry import ModelRegistry
from backtesting.backtest_engine import BacktestEngine
from backtesting.performance_metrics import PerformanceMetrics

def load_and_prepare_data():
    """Load and prepare historical data"""
    print("Loading historical data...")
    loader = HistoricalLoader()
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-02-01')  # 1 month for testing
    data = loader.load_data(start_date, end_date)
    
    print(f"Loaded {len(data)} candles")
    
    # Prepare dataset
    data = data.sort_index()  # Ensure chronological order
    data = data.dropna()  # Remove missing values
    data = data[~data.index.duplicated(keep='first')]  # Remove duplicates
    
    print(f"After preparation: {len(data)} candles")
    return data

def run_feature_engineering(data):
    """Run feature engineering pipeline"""
    print("Running feature engineering...")
    engineer = FeatureEngineering()
    features = engineer.engineer_features(data)
    print(f"Generated {len(features.columns)} features")
    return features

def train_model(features):
    """Train the ML model"""
    print("Training ML model...")
    trainer = ModelTrainer()
    model_path = 'models/registry/backtest_model.joblib'
    model, importance = trainer.train_and_save(features, model_path)
    
    # Register model
    registry = ModelRegistry()
    registry.save_model(model, {'importance': importance, 'trained_on': str(datetime.now())})
    
    print("Model trained and saved")
    return model

def run_backtest():
    """Run full backtest"""
    print("Running backtest...")
    engine = BacktestEngine()
    performance, trades, portfolio = engine.run_backtest()
    return performance, trades, portfolio

def generate_visualizations(portfolio_df, trades):
    """Generate performance visualizations"""
    print("Generating visualizations...")
    
    # Equity curve
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(portfolio_df['timestamp'], portfolio_df['value'])
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    
    # Drawdown curve
    plt.subplot(2, 2, 2)
    peak = portfolio_df['value'].expanding().max()
    drawdown = (portfolio_df['value'] - peak) / peak
    plt.fill_between(portfolio_df['timestamp'], drawdown, 0, color='red', alpha=0.3)
    plt.plot(portfolio_df['timestamp'], drawdown, color='red')
    plt.title('Drawdown Curve')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    
    # Trade distribution
    plt.subplot(2, 2, 3)
    if trades:
        trade_returns = [t.get('pnl', 0) for t in trades if 'pnl' in t]
        plt.hist(trade_returns, bins=50, alpha=0.7, color='blue')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title('Trade PnL Distribution')
        plt.xlabel('PnL')
        plt.ylabel('Frequency')
    else:
        plt.text(0.5, 0.5, 'No trades', ha='center', va='center', transform=plt.gca().transAxes)
    
    # Cumulative returns
    plt.subplot(2, 2, 4)
    returns = portfolio_df['value'].pct_change().fillna(0)
    cum_returns = (1 + returns).cumprod() - 1
    plt.plot(portfolio_df['timestamp'], cum_returns)
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('reports/backtest_visualizations.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Comment out for headless

def generate_report(performance, trades, portfolio_df):
    """Generate comprehensive backtest report"""
    print("Generating backtest report...")
    
    report = f"""
# ML Trading Engine Backtest Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics
- Total Return: {performance.get('total_return', 0):.4f}
- Sharpe Ratio: {performance.get('sharpe_ratio', 0):.4f}
- Maximum Drawdown: {performance.get('max_drawdown', 0):.4f}
- Win Rate: {performance.get('win_rate', 0):.4f}
- Profit Factor: {performance.get('profit_factor', 0):.4f}
- Total Trades: {performance.get('total_trades', 0)}

## Trading Conditions
- Symbol: USDRUB Perpetual Futures
- Timeframe: 1-minute candles
- Period: 2023-01-01 to 2026-01-01
- Initial Capital: $100,000
- Risk per Trade: 1%
- Max Drawdown Limit: 5%
- Slippage: 0.01%
- Commission: 0.02% per trade
- Position Sizing: Risk-based with ATR stops

## Trade Summary
"""
    
    if trades:
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        report += f"""
- Winning Trades: {len(winning_trades)}
- Losing Trades: {len(losing_trades)}
- Average Win: ${np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0:.2f}
- Average Loss: ${np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0:.2f}
- Largest Win: ${max([t.get('pnl', 0) for t in trades], default=0):.2f}
- Largest Loss: ${min([t.get('pnl', 0) for t in trades], default=0):.2f}
"""
    
    report += "\n## Final Portfolio Value\n"
    report += f"Ending Balance: ${portfolio_df['value'].iloc[-1]:.2f}\n"
    
    # Save report
    with open('reports/backtest_report.md', 'w') as f:
        f.write(report)
    
    print("Report saved to reports/backtest_report.md")
    print(report)

def main():
    """Main backtest pipeline"""
    print("Starting ML Trading Engine Backtest Pipeline")
    print("=" * 50)
    
    # Step 1: Load and prepare data
    data = load_and_prepare_data()
    
    # Step 2: Feature engineering
    features = run_feature_engineering(data)
    
    # Step 3: Train model
    model = train_model(features)
    
    # Step 4: Run backtest
    performance, trades, portfolio_df = run_backtest()
    
    # Step 5: Generate visualizations
    generate_visualizations(portfolio_df, trades)
    
    # Step 6: Generate report
    generate_report(performance, trades, portfolio_df)
    
    print("Backtest pipeline completed successfully!")

if __name__ == "__main__":
    main()