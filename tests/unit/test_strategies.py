"""Unit tests for the Policy class."""

import pytest
import pandas as pd
import numpy as np
import datetime as dt
from backtest.strategies import Policy

pytestmark = pytest.mark.unit  # Mark all tests in this file as unit tests

@pytest.fixture
def mock_market_data():
    """Create mock SPY market data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    return pd.DataFrame({
        'SPY': [
            450.0, 452.0, 451.0, 453.0, 455.0,
            454.0, 456.0, 458.0, 457.0, 460.0
        ]
    }, index=dates)

@pytest.fixture
def initialized_policy(mock_db_connections, mock_market_data, monkeypatch):
    """Create a Policy instance with initialized market data."""
    policy = Policy(mock_db_connections)
    
    def mock_prices(*args, **kwargs):
        return mock_market_data
    
    monkeypatch.setattr(policy.marketsim, 'prices', mock_prices)
    
    # Initialize with sample orders
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    orders = pd.DataFrame({
        'Symbol': ['SPY'] * 10,
        'Order': ['BUY'] + ['HOLD'] * 8 + ['SELL'],
        'Shares': [100] + [0] * 8 + [100]
    }, index=dates)
    
    policy._initialize_params(orders)
    return policy

def test_policy_initialization(mock_db_connections):
    """Test Policy class initialization."""
    policy = Policy(mock_db_connections)
    assert policy.marketsim is not None
    assert policy.orders is None
    assert policy.intra_day_flag is False
    assert policy.stock is None
    assert policy.start_date is None
    assert policy.end_date is None

def test_initialize_params_with_orders(initialized_policy):
    """Test parameter initialization with orders."""
    assert initialized_policy.stock == 'SPY'
    assert initialized_policy.trade_size == 100
    assert not initialized_policy.intra_day_flag
    assert isinstance(initialized_policy.stock_prices, pd.DataFrame)
    assert 'SPY' in initialized_policy.stock_prices.columns

def test_determine_max_shares(initialized_policy):
    """Test maximum shares calculation."""
    price = 450.0  # Realistic SPY price
    cash = 100000.0
    
    max_shares = initialized_policy.determine_max_shares(price, cash)
    assert max_shares == int(cash // price)
    
    # Test error cases
    with pytest.raises(ValueError):
        initialized_policy.determine_max_shares(-450.0, cash)
    with pytest.raises(ValueError):
        initialized_policy.determine_max_shares(price, -cash)

def test_buy_and_hold_strategy(initialized_policy):
    """Test buy-and-hold strategy generation."""
    sv = 100000.0
    bh_orders = initialized_policy.buy_and_hold(sv)
    
    assert isinstance(bh_orders, pd.DataFrame)
    assert list(bh_orders.columns) == ['Symbol', 'Order', 'Shares']
    
    # Verify strategy structure
    assert bh_orders['Order'].iloc[0] == 'BUY'
    assert bh_orders['Order'].iloc[-1] == 'SELL'
    assert (bh_orders['Order'].iloc[1:-1] == 'HOLD').all()
    assert (bh_orders['Symbol'] == 'SPY').all()

def test_optimal_policy_strategy(initialized_policy):
    """Test optimal policy strategy generation."""
    sv = 100000.0
    look_ahead = 1
    
    opt_orders = initialized_policy.optimal_policy(sv, D=look_ahead)
    
    assert isinstance(opt_orders, pd.DataFrame)
    assert list(opt_orders.columns) == ['Symbol', 'Order', 'Shares']
    assert (opt_orders['Symbol'] == 'SPY').all()
    assert all(order in ['BUY', 'SELL', 'HOLD'] for order in opt_orders['Order'])

def test_strategy_evaluation(initialized_policy, mock_market_data, monkeypatch):
    """Test strategy evaluation functionality."""
    def mock_compute_portvals(*args, **kwargs):
        return {
            'portfolio': pd.DataFrame({
                'port_val': [100000.0, 100500.0, 101000.0, 101500.0, 102000.0]
            }, index=mock_market_data.index[:5])
        }
    
    monkeypatch.setattr(initialized_policy.marketsim, 'compute_portvals', mock_compute_portvals)
    
    sv = 100000.0
    bh_orders = initialized_policy.buy_and_hold(sv)
    performance = initialized_policy.evaluate_policy(bh_orders, sv)
    
    assert isinstance(performance, pd.DataFrame)
    assert 'port_val' in performance.columns
    assert performance['port_val'].iloc[0] == sv
    assert performance['port_val'].iloc[-1] > sv  # Strategy should show profit

@pytest.mark.slow
def test_multiple_strategy_evaluation(initialized_policy):
    """Test evaluation of multiple strategies."""
    sv = 100000.0
    
    # Generate orders for different strategies
    bh_orders = initialized_policy.buy_and_hold(sv)
    opt_orders = initialized_policy.optimal_policy(sv)
    
    result = initialized_policy.eval_multiple_orders(
        orders=[bh_orders, opt_orders],
        names=['Buy&Hold', 'Optimal'],
        sv=sv
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) >= 2  # At least two strategies
    assert 'Buy&Hold' in result.index
    assert 'Optimal' in result.index

def test_error_handling(initialized_policy):
    """Test error handling in Policy class."""
    # Test invalid starting value
    with pytest.raises(ValueError):
        initialized_policy.buy_and_hold(-100000)
    
    # Test invalid look-ahead period
    with pytest.raises(ValueError):
        initialized_policy.optimal_policy(100000, D=0)
    
    # Test invalid orders DataFrame
    with pytest.raises(ValueError):
        initialized_policy._initialize_params(pd.DataFrame())

def test_performance_metrics(initialized_policy):
    """Test performance metrics calculation."""
    sv = 100000.0
    bh_orders = initialized_policy.buy_and_hold(sv)
    
    # Evaluate strategy
    performance = initialized_policy.evaluate_policy(bh_orders, sv, name="Buy&Hold")
    metrics = initialized_policy._qs(name="Buy&Hold")
    
    assert isinstance(metrics, pd.DataFrame)
    required_metrics = [
        'cumulativeReturns', 'averageDailyReturns', 'stdDailyReturns',
        'annualizedReturns', 'annualizedVolatility', 'maxDrawDown',
        'sharpeRatio', 'calmarRatio', 'sorintinoRatio'
    ]
    assert all(metric in metrics.columns for metric in required_metrics)

@pytest.mark.slow
def test_strategy_comparison(initialized_policy):
    """Test strategy comparison functionality."""
    sv = 100000.0
    commission = 9.95
    impact = 0.005
    
    # Generate and evaluate multiple strategies
    bh_orders = initialized_policy.buy_and_hold(sv)
    opt_orders = initialized_policy.optimal_policy(sv)
    
    comparison = initialized_policy.eval_multiple_orders(
        orders=[bh_orders, opt_orders],
        names=['Buy&Hold', 'Optimal'],
        sv=sv,
        commission=commission,
        impact=impact
    )
    
    assert isinstance(comparison, pd.DataFrame)
    assert comparison.shape[0] >= 2
    assert all(col in comparison.columns for col in [
        'Stock', 'Days', 'StartDate', 'EndDate',
        'StartBalance', 'EndBalance', 'sharpeRatio'
    ])

def test_trading_frequency(initialized_policy):
    """Test different trading frequencies."""
    sv = 100000.0
    strategies = {
        'daily': initialized_policy.buy_and_hold(sv),
        'optimal': initialized_policy.optimal_policy(sv, D=1)
    }
    
    for name, orders in strategies.items():
        # Count actual trades (excluding HOLD)
        trades = orders[orders['Order'].isin(['BUY', 'SELL'])]
        assert len(trades) > 0, f"Strategy {name} should have at least some trades"
        assert len(trades) <= len(orders), f"Strategy {name} cannot have more trades than days"