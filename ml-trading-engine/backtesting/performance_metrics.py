import pandas as pd
import numpy as np
from datetime import datetime

class PerformanceMetrics:
    def calculate_metrics(self, portfolio_df, trades):
        """Calculate comprehensive performance metrics"""
        if portfolio_df.empty:
            return {}
        
        # Basic returns
        initial_value = portfolio_df['value'].iloc[0]
        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Sharpe ratio (assuming daily returns, risk-free rate = 0)
        daily_returns = portfolio_df.set_index('timestamp')['value'].pct_change().dropna()
        if len(daily_returns) > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = portfolio_df['value'].expanding().max()
        drawdown = (portfolio_df['value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate and profit factor
        if trades:
            pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p < 0]
            
            win_rate = len(winning_trades) / len(pnls) if pnls else 0
            gross_profit = sum(winning_trades)
            gross_loss = abs(sum(losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            largest_win = max(pnls, default=0)
            largest_loss = min(pnls, default=0)
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
            largest_win = 0
            largest_loss = 0
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            sortino_ratio = daily_returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades) if trades else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }