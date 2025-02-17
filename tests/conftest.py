"""Test configuration and shared fixtures."""

import os
import pytest
import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, List
import sqlite3
from backtest.simulator import MarketSim

@pytest.fixture
def sample_stock_data() -> pd.DataFrame:
    """Create sample SPY price data."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'SPY': [450.0, 452.0, 451.0, 453.0, 455.0,
                454.0, 456.0, 458.0, 457.0, 460.0]
    })
    df.set_index('Date', inplace=True)
    return df

@pytest.fixture
def sample_orders(sample_stock_data) -> pd.DataFrame:
    """Create sample trading orders."""
    dates = sample_stock_data.index
    return pd.DataFrame({
        'Symbol': ['SPY'] * len(dates),
        'Order': ['BUY'] + ['HOLD'] * (len(dates)-2) + ['SELL'],
        'Shares': [100] + [0] * (len(dates)-2) + [100]
    }, index=dates)

@pytest.fixture
def mock_db_connections(tmp_path, sample_stock_data) -> Dict[str, str]:
    """Create temporary database files with sample data."""
    daily_db = tmp_path / "daily.db"
    intraday_db = tmp_path / "intraday.db"
    
    # Create daily database with SPY data
    conn = sqlite3.connect(daily_db)
    df = sample_stock_data.reset_index()
    df.columns = ['date', 'close']  # Match expected schema
    df['open'] = df['close'] * 0.999
    df['high'] = df['close'] * 1.002
    df['low'] = df['close'] * 0.998
    df['volume'] = np.random.randint(50000000, 100000000, len(df))
    
    df.to_sql('SPY', conn, if_exists='replace', index=False)
    conn.close()
    
    # Create empty intraday database
    intraday_db.touch()
    
    return {
        'daily_db': str(daily_db),
        'intra_day_db': str(intraday_db)
    }

@pytest.fixture
def market_sim(mock_db_connections) -> MarketSim:
    """Create MarketSim instance for testing."""
    return MarketSim(mock_db_connections, verbose=False)

@pytest.fixture
def portfolio_data() -> pd.DataFrame:
    """Create sample portfolio value data."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    start_value = 100000.0
    returns = np.random.normal(0.001, 0.01, len(dates))  # Realistic daily returns
    portfolio_values = start_value * np.exp(np.cumsum(returns))
    
    portfolio = pd.DataFrame({
        'port_val': portfolio_values
    }, index=dates)
    return portfolio

@pytest.fixture
def risk_metrics() -> Dict[str, float]:
    """Return realistic risk metrics for SPY-like returns."""
    return {
        'annualized_return': 0.10,  # 10% annual return
        'annualized_volatility': 0.15,  # 15% volatility
        'sharpe_ratio': 0.67,  # (10% - 0% risk-free) / 15%
        'max_drawdown': -0.10,  # 10% maximum drawdown
        'sortino_ratio': 1.0,
        'calmar_ratio': 1.0
    }

@pytest.fixture
def market_impact() -> float:
    """Return realistic market impact for SPY."""
    return 0.0005  # 0.05% market impact (SPY is highly liquid)

@pytest.fixture
def commission() -> float:
    """Return standard commission for trading."""
    return 9.95  # $9.95 per trade

@pytest.fixture
def trading_params() -> Dict[str, float]:
    """Return standard trading parameters."""
    return {
        'starting_value': 100000.0,
        'commission': 9.95,
        'market_impact': 0.0005,
        'risk_free_rate': 0.02,  # 2% risk-free rate
    }

@pytest.fixture
def spy_volatility() -> float:
    """Return realistic SPY volatility."""
    return 0.15  # 15% annualized volatility

@pytest.fixture
def spy_correlation() -> float:
    """Return SPY correlation with market."""
    return 1.0  # SPY represents the market

@pytest.fixture
def spy_beta() -> float:
    """Return SPY beta."""
    return 1.0  # SPY beta is 1.0 by definition