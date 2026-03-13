# ML Trading Engine

A production-style algorithmic trading engine for perpetual futures on the USDRUB pair, built with Python and designed to mimic quantitative hedge fund architectures.

## System Architecture

The system follows a modular, professional quant trading architecture with the following components:

### Data Ingestion
- **market_stream.py**: Streams real-time market data (OHLCV candles)
- **historical_loader.py**: Loads and manages historical market data with timestamp normalization

### Feature Pipeline
- **feature_engineering.py**: Advanced feature engineering including price features, technical indicators, microstructure features, and volatility regime features
- **feature_store.py**: Local feature store using Parquet format for consistent training/inference features

### Machine Learning Pipeline
- **train_model.py**: XGBoost classification model training with time-series split and walk-forward validation
- **model_registry.py**: Model versioning and storage system

### Inference Service
- **online_predictor.py**: Real-time inference service with trading signal generation

### Execution Engine
- **exchange_connector.py**: Simulated exchange connector with order execution, PnL tracking, and slippage/commission modeling
- **order_manager.py**: Order management system with market and limit orders

### Risk Management
- **risk_manager.py**: Professional risk controls including position sizing, ATR-based stops, drawdown protection, and exposure limits

### Backtesting Framework
- **backtest_engine.py**: Realistic backtesting with historical replay and order simulation
- **performance_metrics.py**: Comprehensive metrics including Sharpe ratio, max drawdown, win rate, and profit factor

### Scheduler
- **retrain_scheduler.py**: Automated model retraining every 24 hours

### Trading Loop
- **trading_loop.py**: Live trading orchestration loop

## Features

### Market Data
- Streaming OHLCV candles
- Historical data loading with normalization
- Minute-level granularity support

### Feature Engineering
- **Price Features**: Log returns, rolling volatility, momentum, moving averages
- **Technical Indicators**: RSI, ATR, MACD, Bollinger Bands
- **Microstructure**: Volume imbalance, rolling volume averages
- **Volatility Regime**: Realized volatility, breakout signals

### Machine Learning
- XGBoost classifier for price direction prediction
- Time-series aware training with walk-forward validation
- Feature importance analysis
- Model registry with versioning

### Risk Management
- Fixed risk percentage per trade
- ATR-based stop losses
- Maximum drawdown protection
- Position size limits

### Trading Rules
- LONG if probability > 0.60
- SHORT if probability < 0.40
- NO TRADE otherwise

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

First, train the initial model:

```python
from data_ingestion.historical_loader import HistoricalLoader
from feature_pipeline.feature_engineering import FeatureEngineering
from models.train_model import ModelTrainer
from models.model_registry import ModelRegistry

# Load data
loader = HistoricalLoader()
data = loader.load_data(pd.Timestamp('2023-01-01'), pd.Timestamp('2024-01-01'))

# Engineer features
engineer = FeatureEngineering()
features = engineer.engineer_features(data)

# Train model
trainer = ModelTrainer()
model, importance = trainer.train_and_save(features, 'models/registry/model_v1.joblib')

# Register model
registry = ModelRegistry()
registry.save_model(model)
```

### Running Backtesting

```python
from backtesting.backtest_engine import BacktestEngine

engine = BacktestEngine()
performance, trades, portfolio = engine.run_backtest()

print("Performance Metrics:")
for metric, value in performance.items():
    print(f"{metric}: {value}")
```

### Starting Live Trading

```python
from src.trading_loop import TradingLoop

loop = TradingLoop()
loop.run_live_trading()
```

### Automated Retraining

```python
from scheduler.retrain_scheduler import RetrainScheduler

scheduler = RetrainScheduler()
scheduler.start_scheduler()
```

## Configuration

The system is configured via `config/config.yaml`:

- Trading parameters (symbol, risk settings)
- Data paths and model registry
- ML hyperparameters
- Execution settings (slippage, commissions)
- Backtesting parameters

## Running in GitHub Codespaces

1. Open the project in GitHub Codespaces
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model first
4. Run backtesting to validate
5. Start the live trading loop

The system is designed to run efficiently in cloud environments with minimal resource requirements.

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computing
- xgboost: Machine learning
- scikit-learn: ML utilities
- ta: Technical analysis
- joblib: Model serialization
- pyyaml: Configuration
- schedule: Task scheduling
- matplotlib: Plotting (for analysis)

## Architecture Principles

- **Modularity**: Each component has a single responsibility
- **Testability**: Components can be tested independently
- **Scalability**: Architecture supports additional assets and strategies
- **Risk Management**: Conservative risk controls prevent catastrophic losses
- **Reproducibility**: Configuration-driven with versioned models
- **Monitoring**: Comprehensive logging and performance tracking