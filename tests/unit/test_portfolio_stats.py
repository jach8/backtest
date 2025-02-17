"""Unit tests for the PortfolioStats class."""

import pytest
import pandas as pd
import numpy as np
from backtest.PortfolioStats import PortfolioStats

pytestmark = pytest.mark.unit  # Mark all tests in this file as unit tests

def test_portfolio_stats_initialization(portfolio_data):
    """Test PortfolioStats initialization."""
    stats = PortfolioStats(portfolio_data, risk_free_rate=0.02)
    assert stats.rf == 0.02
    assert stats.N == 360
    assert stats.port.equals(portfolio_data)

def test_returns_calculation(portfolio_data):
    """Test daily returns calculation."""
    stats = PortfolioStats(portfolio_data, risk_free_rate=0.0)
    returns = stats._returns(portfolio_data)
    
    # Manual calculation for verification
    expected_returns = portfolio_data.pct_change().iloc[1:]
    pd.testing.assert_frame_equal(returns, expected_returns)

def test_annualized_returns(portfolio_data):
    """Test annualized returns calculation."""
    stats = PortfolioStats(portfolio_data, risk_free_rate=0.02)
    returns = stats._returns(portfolio_data)
    ann_returns = stats._annualized_returns(returns['port_val'])
    
    # Verify the calculation
    expected_return = returns['port_val'].mean() * 360 - 0.02
    assert np.isclose(ann_returns, expected_return)

def test_annualized_volatility(portfolio_data):
    """Test annualized volatility calculation."""
    stats = PortfolioStats(portfolio_data, risk_free_rate=0.0)
    returns = stats._returns(portfolio_data)
    volatility = stats._annualized_volatility(returns['port_val'])
    
    # Verify the calculation
    expected_volatility = returns['port_val'].std() * np.sqrt(360)
    assert np.isclose(volatility, expected_volatility)

def test_cumulative_returns(portfolio_data):
    """Test cumulative returns calculation."""
    stats = PortfolioStats(portfolio_data, risk_free_rate=0.0)
    returns = stats._returns(portfolio_data)
    cum_returns = stats._cumulative_returns(returns['port_val'])
    
    # Verify the calculation
    expected_cum_returns = (1 + returns['port_val']).cumprod() - 1
    pd.testing.assert_series_equal(cum_returns, expected_cum_returns)

def test_max_drawdown(portfolio_data):
    """Test maximum drawdown calculation."""
    stats = PortfolioStats(portfolio_data, risk_free_rate=0.0)
    returns = stats._returns(portfolio_data)
    max_dd = stats._max_drawdown(returns['port_val'])
    
    # Manual calculation
    comp_ret = (returns['port_val'] + 1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    expected_dd = ((comp_ret / peak) - 1).min()
    
    assert np.isclose(max_dd, expected_dd)

def test_sharpe_ratio(portfolio_data):
    """Test Sharpe ratio calculation."""
    stats = PortfolioStats(portfolio_data, risk_free_rate=0.02)
    returns = stats._returns(portfolio_data)
    sharpe = stats._sharpe_ratio(returns['port_val'])
    
    # Manual calculation
    mean_return = returns['port_val'].mean() * 360 - 0.02
    std_dev = returns['port_val'].std() * np.sqrt(360)
    expected_sharpe = mean_return / std_dev if std_dev != 0 else 0
    
    assert np.isclose(sharpe, expected_sharpe)

def test_sortino_ratio(portfolio_data):
    """Test Sortino ratio calculation."""
    stats = PortfolioStats(portfolio_data, risk_free_rate=0.02)
    returns = stats._returns(portfolio_data)
    sortino = stats._sortino_ratio(returns['port_val'])
    
    # Manual calculation
    mean_return = returns['port_val'].mean() * 360 - 0.02
    neg_returns = returns['port_val'][returns['port_val'] < 0]
    if not neg_returns.empty:
        neg_std = neg_returns.std() * np.sqrt(360)
        expected_sortino = mean_return / neg_std if neg_std != 0 else float('inf')
        assert np.isclose(sortino, expected_sortino)
    else:
        assert np.isinf(sortino)

def test_calmar_ratio(portfolio_data):
    """Test Calmar ratio calculation."""
    stats = PortfolioStats(portfolio_data, risk_free_rate=0.0)
    returns = stats._returns(portfolio_data)
    rbar = returns['port_val'].mean() * 360
    calmar = stats._calmar_ratio(returns['port_val'], rbar)
    
    # Manual calculation
    mdd = abs(stats._max_drawdown(returns['port_val']))
    if mdd != 0:
        expected_calmar = rbar / mdd
        assert np.isclose(calmar, expected_calmar)
    else:
        assert np.isinf(calmar)

def test_empty_portfolio():
    """Test error handling for empty portfolio."""
    empty_df = pd.DataFrame(columns=['port_val'])
    
    with pytest.raises(ValueError, match="Return dataframe cannot be empty"):
        stats = PortfolioStats(empty_df, risk_free_rate=0.0)
        stats._portfolio_stats()

def test_zero_volatility_portfolio():
    """Test handling of portfolio with zero volatility."""
    constant_value = pd.DataFrame({
        'port_val': [100.0] * 10
    }, index=pd.date_range('2024-01-01', periods=10))
    
    stats = PortfolioStats(constant_value, risk_free_rate=0.0)
    returns = stats._returns(constant_value)
    
    # Sharpe ratio should be 0 for zero volatility
    assert stats._sharpe_ratio(returns['port_val']) == 0

def test_all_negative_returns():
    """Test handling of portfolio with all negative returns."""
    declining_value = pd.DataFrame({
        'port_val': [100.0, 99.0, 98.0, 97.0, 96.0]
    }, index=pd.date_range('2024-01-01', periods=5))
    
    stats = PortfolioStats(declining_value, risk_free_rate=0.0)
    returns = stats._returns(declining_value)
    
    # Should still calculate valid metrics
    dd = stats._max_drawdown(returns['port_val'])
    ann_ret = stats._annualized_returns(returns['port_val'])
    
    assert dd < 0
    assert ann_ret < 0

def test_portfolio_stats_output_format(portfolio_data):
    """Test the format of portfolio statistics output."""
    stats = PortfolioStats(portfolio_data, risk_free_rate=0.0)
    result = stats._portfolio_stats(name="Test Portfolio")
    
    expected_columns = {
        'cumulativeReturns', 'averageDailyReturns', 'stdDailyReturns',
        'annualizedReturns', 'annualizedVolatility', 'maxDrawDown',
        'sharpeRatio', 'calmarRatio', 'sorintinoRatio'
    }
    
    assert isinstance(result, pd.DataFrame)
    assert result.index[0] == "Test Portfolio"
    assert set(result.columns) >= expected_columns  # Using >= for subset comparison

def test_risk_free_rate_impact(portfolio_data):
    """Test impact of different risk-free rates."""
    rf_rates = [0.0, 0.02, 0.05]
    sharpe_ratios = []
    
    for rf in rf_rates:
        stats = PortfolioStats(portfolio_data, risk_free_rate=rf)
        returns = stats._returns(portfolio_data)
        sharpe = stats._sharpe_ratio(returns['port_val'])
        sharpe_ratios.append(sharpe)
    
    # Higher risk-free rate should result in lower Sharpe ratio
    assert all(sharpe_ratios[i] > sharpe_ratios[i+1] for i in range(len(sharpe_ratios)-1))