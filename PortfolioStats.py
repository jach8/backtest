"""Portfolio statistics calculation module."""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Union

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class PortfolioStats:
    """Calculate and manage portfolio statistics."""

    def __init__(self, portfolio_df: pd.DataFrame, risk_free_rate: float = 0.0) -> None:
        """
        Initialize PortfolioStats with portfolio data.

        Args:
            portfolio_df (pd.DataFrame): DataFrame with portfolio values
            risk_free_rate (float): Annual risk-free rate

        Raises:
            ValueError: If portfolio_df is empty
        """
        if portfolio_df.empty:
            raise ValueError("Return dataframe cannot be empty")
            
        self.port = portfolio_df
        self.rf = risk_free_rate  # Annual risk-free rate
        self.N = 360  # Trading days in a year

    def _returns(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns.

        Args:
            portfolio_df (pd.DataFrame): Portfolio value data

        Returns:
            pd.DataFrame: Daily returns
        """
        return portfolio_df.pct_change().iloc[1:]

    def _annualized_returns(self, returns: pd.Series) -> float:
        """
        Calculate annualized returns.

        Args:
            returns (pd.Series): Daily returns

        Returns:
            float: Annualized returns
        """
        return returns.mean() * self.N - self.rf

    def _annualized_volatility(self, returns: pd.Series) -> float:
        """
        Calculate annualized volatility.

        Args:
            returns (pd.Series): Daily returns

        Returns:
            float: Annualized volatility
        """
        return returns.std() * np.sqrt(self.N)

    def _cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns.

        Args:
            returns (pd.Series): Daily returns

        Returns:
            pd.Series: Cumulative returns
        """
        return (1 + returns).cumprod() - 1

    def _max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            returns (pd.Series): Daily returns

        Returns:
            float: Maximum drawdown
        """
        comp_ret = (returns + 1).cumprod()
        peak = comp_ret.expanding(min_periods=1).max()
        drawdown = ((comp_ret / peak) - 1)
        return drawdown.min()

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns (pd.Series): Daily returns

        Returns:
            float: Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
            
        sigma = self._annualized_volatility(returns)
        if np.isclose(sigma, 0.0):
            logger.warning("Volatility is zero; Sharpe Ratio might be meaningless.")
            return 0.0
            
        return self._annualized_returns(returns) / sigma

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sortino ratio.

        Args:
            returns (pd.Series): Daily returns

        Returns:
            float: Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
            
        downside_returns = returns[returns < 0]
        if downside_returns.empty:
            return float('inf')
        
        downside_std = downside_returns.std() * np.sqrt(self.N)
        if np.isclose(downside_std, 0.0):
            return float('inf')
            
        return self._annualized_returns(returns) / downside_std

    def _calmar_ratio(self, returns: pd.Series, rbar: float) -> float:
        """
        Calculate Calmar ratio.

        Args:
            returns (pd.Series): Daily returns
            rbar (float): Average return

        Returns:
            float: Calmar ratio
        """
        if len(returns) < 2:
            return 0.0
            
        mdd = abs(self._max_drawdown(returns))
        if np.isclose(mdd, 0.0):
            return float('inf')
            
        return rbar / mdd

    def _portfolio_stats(self, name: str = "Portfolio") -> pd.DataFrame:
        """
        Calculate portfolio statistics.

        Args:
            name (str): Portfolio name

        Returns:
            pd.DataFrame: Portfolio statistics with standardized column names
        """
        returns = self._returns(self.port)
        if returns.empty:
            return pd.DataFrame({
                'Status': ['Empty portfolio'],
                'Value': [0.0]
            }, index=[name])
            
        rbar = returns['port_val'].mean() * self.N
        sr = self._sharpe_ratio(returns['port_val'])
        sortino = self._sortino_ratio(returns['port_val'])
        calmar = self._calmar_ratio(returns['port_val'], rbar)
        mdd = self._max_drawdown(returns['port_val'])
        vol = self._annualized_volatility(returns['port_val'])
        cr = self._cumulative_returns(returns['port_val']).iloc[-1]

        return pd.DataFrame({
            'averageDailyReturns': returns['port_val'].mean(),
            'stdDailyReturns': returns['port_val'].std(),
            'annualizedReturns': rbar,
            'annualizedVolatility': vol,
            'sharpeRatio': sr,
            'sorintinoRatio': sortino,
            'calmarRatio': calmar,
            'maxDrawDown': mdd,
            'cumulativeReturns': cr
        }, index=[name])