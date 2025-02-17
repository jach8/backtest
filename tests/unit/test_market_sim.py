"""Unit tests for the MarketSim class."""

import pytest
import pandas as pd
import numpy as np
import datetime as dt
from backtest.simulator import MarketSim

pytestmark = pytest.mark.unit  # Mark all tests in this file as unit tests

def test_market_sim_initialization(mock_db_connections):
    """Test MarketSim initialization."""
    sim = MarketSim(mock_db_connections, verbose=True)
    assert sim.verbose
    assert isinstance(sim.tracking, dict)
    assert len(sim.tracking) == 0

@pytest.fixture
def sample_orders():
    """Create sample trading orders for SPY."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    orders = pd.DataFrame({
        'Symbol': ['SPY'] * 10,
        'Order': ['BUY'] + ['HOLD'] * 8 + ['SELL'],
        'Shares': [100] + [0] * 8 + [100]
    }, index=dates)
    return orders

@pytest.fixture
def mock_price_data():
    """Create mock SPY price data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
    return pd.DataFrame({
        'SPY': [450.0, 452.0, 451.0, 453.0, 455.0]
    }, index=dates)

def test_get_close_prices(market_sim, mock_price_data, monkeypatch):
    """Test retrieval of closing prices."""
    def mock_read_sql(*args, **kwargs):
        return mock_price_data[['SPY']]
    
    monkeypatch.setattr(pd, 'read_sql_query', mock_read_sql)
    
    prices = market_sim.get_close(
        stocks=['SPY'],
        start=dt.datetime(2024, 1, 1),
        end=dt.datetime(2024, 1, 5)
    )
    
    pd.testing.assert_frame_equal(prices, mock_price_data)

def test_setup_simulation(market_sim, sample_orders, mock_price_data, monkeypatch):
    """Test simulation setup with initial portfolio."""
    def mock_prices(*args, **kwargs):
        return mock_price_data
    
    monkeypatch.setattr(market_sim, 'prices', mock_prices)
    
    start_val = 100000.0
    market_sim._MarketSim__SetupSim(sample_orders, start_val)
    
    assert 'trades' in market_sim.tracking
    assert 'holdings' in market_sim.tracking
    assert 'stock_prices' in market_sim.tracking
    assert market_sim.tracking['trades'].index.equals(mock_price_data.index)
    assert market_sim.tracking['trades']['Cash'].iloc[0] == start_val

def test_buy_order_execution(market_sim, mock_price_data):
    """Test execution of buy orders."""
    start_val = 100000.0
    shares = 100
    price = 450.0
    commission = 9.95
    impact = 0.005
    
    market_sim.tracking = {
        'trades': pd.DataFrame(0, index=mock_price_data.index, columns=['SPY', 'Cash']),
        'holdings': pd.DataFrame(0, index=mock_price_data.index, columns=['SPY', 'Cash']),
        'stock_prices': mock_price_data.copy()
    }
    market_sim.tracking['trades']['Cash'].iloc[0] = start_val
    market_sim.tracking['holdings']['Cash'].iloc[0] = start_val
    
    date = mock_price_data.index[0]
    market_sim._MarketSim__BuyOrder(date, 'SPY', shares, price, commission, impact)
    
    expected_cost = shares * price * (1 + impact) + commission
    assert market_sim.tracking['trades'].loc[date, 'SPY'] == shares
    assert np.isclose(market_sim.tracking['trades'].loc[date, 'Cash'], start_val - expected_cost)

def test_sell_order_execution(market_sim, mock_price_data):
    """Test execution of sell orders."""
    initial_shares = 100
    price = 450.0
    commission = 9.95
    impact = 0.005
    
    market_sim.tracking = {
        'trades': pd.DataFrame(0, index=mock_price_data.index, columns=['SPY', 'Cash']),
        'holdings': pd.DataFrame(0, index=mock_price_data.index, columns=['SPY', 'Cash'])
    }
    date = mock_price_data.index[0]
    market_sim.tracking['trades'].loc[date, 'SPY'] = initial_shares
    market_sim.tracking['holdings'].loc[date, 'SPY'] = initial_shares
    
    market_sim._MarketSim__SellOrder(date, 'SPY', initial_shares, price, commission, impact)
    
    expected_proceeds = initial_shares * price * (1 - impact) - commission
    assert market_sim.tracking['trades'].loc[date, 'SPY'] == 0
    assert np.isclose(market_sim.tracking['trades'].loc[date, 'Cash'], expected_proceeds)

def test_hold_order_execution(market_sim, mock_price_data):
    """Test execution of hold orders."""
    date = mock_price_data.index[0]
    price = 450.0
    
    market_sim.tracking = {
        'trades': pd.DataFrame(0, index=mock_price_data.index, columns=['SPY', 'Cash']),
        'holdings': pd.DataFrame(0, index=mock_price_data.index, columns=['SPY', 'Cash'])
    }
    
    initial_state = market_sim.tracking['trades'].copy()
    market_sim._MarketSim__HoldOrder(date, 'SPY', price)
    
    pd.testing.assert_frame_equal(market_sim.tracking['trades'], initial_state)

def test_compute_portfolio_values(market_sim, sample_orders, mock_price_data, monkeypatch):
    """Test portfolio value computation."""
    def mock_prices(*args, **kwargs):
        return mock_price_data
    
    monkeypatch.setattr(market_sim, 'prices', mock_prices)
    
    start_val = 100000.0
    result = market_sim.compute_portvals(sample_orders, start_val)
    
    assert 'portfolio' in result
    assert 'port_val' in result['portfolio'].columns
    assert len(result['portfolio']) == len(mock_price_data)
    assert result['portfolio']['port_val'].iloc[0] == start_val

def test_insufficient_funds(market_sim, mock_price_data):
    """Test handling of insufficient funds for buying."""
    start_val = 1000.0  # Small starting value
    shares = 100
    price = 450.0  # Total cost would exceed starting value
    
    market_sim.tracking = {
        'trades': pd.DataFrame(0, index=mock_price_data.index, columns=['SPY', 'Cash']),
        'holdings': pd.DataFrame(0, index=mock_price_data.index, columns=['SPY', 'Cash'])
    }
    market_sim.tracking['trades']['Cash'].iloc[0] = start_val
    market_sim.tracking['holdings']['Cash'].iloc[0] = start_val
    
    date = mock_price_data.index[0]
    market_sim._MarketSim__BuyOrder(date, 'SPY', shares, price)
    
    # Verify no trade was executed
    assert market_sim.tracking['trades'].loc[date, 'SPY'] == 0
    assert market_sim.tracking['trades'].loc[date, 'Cash'] == start_val

def test_market_impact(market_sim, mock_price_data):
    """Test market impact on trade prices."""
    start_val = 100000.0
    shares = 100
    price = 450.0
    impact = 0.01  # 1% market impact
    commission = 9.95
    
    market_sim.tracking = {
        'trades': pd.DataFrame(0, index=mock_price_data.index, columns=['SPY', 'Cash']),
        'holdings': pd.DataFrame(0, index=mock_price_data.index, columns=['SPY', 'Cash'])
    }
    market_sim.tracking['trades']['Cash'].iloc[0] = start_val
    market_sim.tracking['holdings']['Cash'].iloc[0] = start_val
    
    date = mock_price_data.index[0]
    
    # Test buy with market impact
    market_sim._MarketSim__BuyOrder(date, 'SPY', shares, price, commission, impact)
    expected_cost = shares * price * (1 + impact) + commission
    actual_cost = start_val - market_sim.tracking['trades'].loc[date, 'Cash']
    assert np.isclose(actual_cost, expected_cost, rtol=1e-10)

def test_portfolio_tracking(market_sim, mock_price_data):
    """Test portfolio value tracking over time."""
    start_val = 100000.0
    shares = 100
    price = 450.0
    
    market_sim.tracking = {
        'trades': pd.DataFrame(0, index=mock_price_data.index, columns=['SPY', 'Cash']),
        'holdings': pd.DataFrame(0, index=mock_price_data.index, columns=['SPY', 'Cash']),
        'stock_prices': mock_price_data.copy()
    }
    market_sim.tracking['trades']['Cash'].iloc[0] = start_val
    market_sim.tracking['holdings']['Cash'].iloc[0] = start_val
    
    # Execute a buy order
    date = mock_price_data.index[0]
    market_sim._MarketSim__BuyOrder(date, 'SPY', shares, price)
    
    # Verify portfolio tracking
    market_sim.tracking['portfolio'] = market_sim.tracking['stock_prices'] * market_sim.tracking['trades'].cumsum()
    market_sim.tracking['portfolio']['port_val'] = market_sim.tracking['portfolio'].sum(axis=1)
    
    # Portfolio value should reflect both cash and stock positions
    expected_portfolio_value = start_val  # Total value should remain same minus transaction costs
    assert np.isclose(market_sim.tracking['portfolio']['port_val'].iloc[0], expected_portfolio_value, rtol=0.01)