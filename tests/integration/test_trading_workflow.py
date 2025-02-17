"""Integration tests for the complete trading workflow."""

import pytest
import pandas as pd
import numpy as np
import datetime as dt
import sqlite3
import os
from backtest.strategies import Policy
from backtest.simulator import MarketSim
from backtest.PortfolioStats import PortfolioStats

pytestmark = [pytest.mark.integration, pytest.mark.slow]  # Mark all tests as integration tests and slow

@pytest.fixture
def setup_test_database(tmp_path):
    """Create and populate a test database with SPY data."""
    db_path = tmp_path / "test_daily.db"
    
    # Create test database with realistic SPY data
    conn = sqlite3.connect(db_path)
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    # Generate realistic SPY prices with some volatility
    # Start with SPY at 450 and add random walk with drift
    base_price = 450.0
    daily_returns = np.random.normal(0.0005, 0.01, len(dates))  # Realistic SPY daily returns
    prices = base_price * np.exp(np.cumsum(daily_returns))
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(50000000, 100000000, len(dates))  # Typical SPY volume
    })
    
    df.to_sql('SPY', conn, if_exists='replace', index=False)
    conn.close()
    
    return {
        'daily_db': str(db_path),
        'intra_day_db': str(tmp_path / "test_intraday.db")
    }

@pytest.mark.slow
def test_end_to_end_buy_and_hold(setup_test_database):
    """Test end-to-end buy and hold strategy execution."""
    # Initialize components
    policy = Policy(setup_test_database)
    start_val = 100000.0
    
    # Generate and execute buy and hold strategy
    bh_orders = policy.buy_and_hold(start_val)
    results = policy.evaluate_policy(
        bh_orders,
        sv=start_val,
        name="Buy and Hold",
        commission=9.95,
        impact=0.005
    )
    
    # Verify results structure and basic properties
    assert isinstance(results, pd.DataFrame)
    assert 'port_val' in results.columns
    assert len(results) > 0
    assert results['port_val'].iloc[0] == start_val
    
    # Get performance metrics
    metrics = policy._qs(name="Buy and Hold")
    assert isinstance(metrics, pd.DataFrame)
    assert 'sharpeRatio' in metrics.columns
    assert 'maxDrawDown' in metrics.columns
    
    # Verify realistic value ranges
    assert -0.5 <= metrics['maxDrawDown'].iloc[0] <= 0  # Max drawdown should be negative but not extreme
    assert -3 <= metrics['sharpeRatio'].iloc[0] <= 3  # Realistic Sharpe ratio range

@pytest.mark.slow
def test_strategy_comparison_workflow(setup_test_database):
    """Test comparing multiple trading strategies."""
    policy = Policy(setup_test_database)
    start_val = 100000.0
    
    # Generate orders for different strategies
    bh_orders = policy.buy_and_hold(start_val)
    opt_orders = policy.optimal_policy(start_val, D=1)
    
    # Compare strategies with realistic transaction costs
    comparison = policy.eval_multiple_orders(
        orders=[bh_orders, opt_orders],
        names=['Buy&Hold', 'Optimal'],
        sv=start_val,
        commission=9.95,  # Standard commission
        impact=0.005  # Typical market impact for SPY
    )
    
    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) >= 2
    assert all(col in comparison.columns for col in [
        'Stock', 'StartBalance', 'EndBalance',
        'sharpeRatio', 'maxDrawDown'
    ])
    
    # Verify realistic metric ranges
    assert all(comparison['maxDrawDown'] <= 0)  # All drawdowns should be non-positive
    assert all(abs(comparison['sharpeRatio']) < 5)  # Realistic Sharpe ratio range

@pytest.mark.slow
def test_portfolio_analysis_pipeline(setup_test_database):
    """Test the complete portfolio analysis pipeline."""
    policy = Policy(setup_test_database)
    start_val = 100000.0
    
    # Generate and execute strategy
    orders = policy.buy_and_hold(start_val)
    portfolio = policy.evaluate_policy(
        orders,
        sv=start_val,
        commission=9.95,
        impact=0.005
    )
    
    # Analyze portfolio performance
    stats = PortfolioStats(portfolio, risk_free_rate=0.02)  # 2% risk-free rate
    metrics = stats._portfolio_stats()
    
    assert isinstance(metrics, pd.DataFrame)
    assert all(col in metrics.columns for col in [
        'cumulativeReturns',
        'annualizedReturns',
        'sharpeRatio',
        'maxDrawDown'
    ])
    
    # Verify realistic metric ranges for SPY
    assert -0.5 <= metrics['cumulativeReturns'].iloc[0] <= 0.5  # Reasonable return range
    assert -0.5 <= metrics['annualizedReturns'].iloc[0] <= 0.5  # Reasonable annualized return
    assert -3 <= metrics['sharpeRatio'].iloc[0] <= 3  # Reasonable Sharpe ratio
    assert -0.5 <= metrics['maxDrawDown'].iloc[0] <= 0  # Reasonable drawdown range

@pytest.mark.slow
def test_market_impact_analysis(setup_test_database):
    """Test the impact of different transaction costs on strategy performance."""
    policy = Policy(setup_test_database)
    start_val = 100000.0
    orders = policy.buy_and_hold(start_val)
    
    # Test different impact levels
    impact_levels = [0.001, 0.005, 0.01]  # Typical range for SPY
    results = []
    for impact in impact_levels:
        performance = policy.evaluate_policy(
            orders,
            sv=start_val,
            commission=9.95,
            impact=impact,
            name=f"Impact_{impact}"
        )
        metrics = policy._qs(name=f"Impact_{impact}")
        results.append(metrics)
    
    combined_results = pd.concat(results)
    
    # Higher impact should lead to lower returns
    returns = combined_results['cumulativeReturns']
    assert returns.iloc[0] >= returns.iloc[-1]  # Returns should decrease with higher impact
    
    # Verify impact effects are within reasonable ranges
    return_differences = np.diff(returns)
    assert all(abs(diff) < 0.05 for diff in return_differences)  # Impact differences should be modest

@pytest.mark.slow
def test_risk_management_workflow(setup_test_database):
    """Test risk management aspects of the trading system."""
    policy = Policy(setup_test_database)
    start_val = 100000.0
    
    # Test strategy with different position sizes
    position_sizes = [0.25, 0.5, 1.0]  # Different allocation levels
    results = []
    for size_factor in position_sizes:
        orders = policy.buy_and_hold(start_val * size_factor)
        policy.evaluate_policy(
            orders,
            sv=start_val * size_factor,
            name=f"Size_{size_factor}",
            commission=9.95,
            impact=0.005
        )
        metrics = policy._qs(name=f"Size_{size_factor}")
        results.append(metrics)
    
    combined_results = pd.concat(results)
    
    # Verify risk metrics scale appropriately with position size
    volatilities = combined_results['annualizedVolatility']
    assert all(volatilities.diff().dropna() >= 0)  # Volatility should increase with position size
    
    # Verify Sharpe ratios are similar (within reason) across position sizes
    sharpe_ratios = combined_results['sharpeRatio']
    assert max(sharpe_ratios) - min(sharpe_ratios) < 0.5  # Sharpe ratios shouldn't vary too much