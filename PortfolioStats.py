import numpy as np
import pandas as pd
from scipy import optimize as spo
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioStats:
    """
    This class should take the Strategies Class for portfolio analysis.
    """
    def __init__(self, allocation: dict, risk_free_rate: float):
        self.allocation = allocation
        self.rf = risk_free_rate
        self.N = 360
        
        # Error check for allocation
        if not all(0 <= v <= 1 for v in allocation.values()):
            logger.error("Allocation values must be between 0 and 1.")
            raise ValueError("Allocation values must be between 0 and 1.")
        
        if abs(sum(allocation.values()) - 1) > 1e-6:  # Using small tolerance for float comparison
            logger.warning("The sum of allocation weights is not 1. Adjusting weights.")
            total = sum(allocation.values())
            self.allocation = {k: v / total for k, v in allocation.items()}

    def _annualized_returns(self, ret: pd.Series) -> float:
        """ Return the Annualized Returns of the Portfolio """
        if ret.empty:
            logger.error("Empty return series provided.")
            raise ValueError("Return series cannot be empty.")
        return ret.mean() * self.N - self.rf

    def _annualized_volatility(self, ret: pd.Series) -> float:
        """ Return the Annualized Volatility of the Portfolio """
        if ret.empty:
            logger.error("Empty return series provided for volatility calculation.")
            raise ValueError("Return series cannot be empty.")
        return ret.std() * np.sqrt(self.N)

    def _cumulative_returns(self, ret: pd.Series) -> pd.Series:
        """ Return the Cumulative Returns of the Portfolio """
        return (1 + ret).cumprod() - 1

    def _max_drawdown(self, ret: pd.Series) -> float:
        """ Return the Maximum Drawdown of the Portfolio 
            Maximum Drawdown = (Peak - Trough) / Peak
        """
        comp_ret = (ret + 1).cumprod()
        peak = comp_ret.expanding(min_periods=1).max()
        dd = (comp_ret / peak) - 1
        return dd.min()

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """ Return the Annualized Sharpe Ratio of the Portfolio 
            Sharpe Ratio = (Rp - Rf) / σp
                - Rp = Annual Expected Returns of the asset 
                - Rf = Risk Free Rate of Return
                - σp = Annualized Standard Deviation of returns       
            ---------------------------------------------------
            Sharpe Ratio Higher than 2.0 is rated as very good 
                         Higher than 3.0 is Excellent 
                         Lower than 1.0 is very BAD
            ---------------------------------------------------
            - Adding assets to diversify a portfolio can increase the ratio
            - Abnormal distribution of returns can result in a flawed ratio (very high)
        """
        mean = returns.mean() * self.N - self.rf
        sigma = returns.std() * np.sqrt(self.N)
        if sigma == 0:
            logger.warning("Volatility is zero, Sharpe Ratio might be meaningless.")
            return 0
        return mean / sigma

    def _sortino_ratio(self, ret: pd.Series) -> float:
        """
        Return the Sortino Ratio of the Portfolio 
            Sortino Ratio = (Rp - Rf) / σn
                - Rp = Annual Expected Returns of the asset 
                - Rf = Risk Free Rate of Return
                - σn = Negative Standard Deviation of returns
        """
        mu = ret.mean() * self.N - self.rf
        negative_returns = ret[ret < 0]
        if negative_returns.empty:
            logger.warning("No negative returns found, Sortino Ratio might be meaningless.")
            return float('inf')
        std_neg = negative_returns.std() * np.sqrt(self.N)
        if std_neg == 0:
            logger.warning("Standard deviation of negative returns is zero, Sortino Ratio might be meaningless.")
            return float('inf')
        return mu / std_neg

    def _calmar_ratio(self, ret: pd.Series, rbar: float) -> float:
        """ Risk/Reward ratio the Calmar Ratio of the Portfolio 
            Calmar Ratio = (Rbar) / MDD
                - Rbar = Annualized Returns 
                - MDD = Maximum Drawdown
        """
        mdd = abs(self._max_drawdown(ret))
        if mdd == 0:
            logger.warning("Maximum drawdown is zero, Calmar Ratio might be meaningless.")
            return float('inf')
        return rbar / mdd

    def _optimize_portfolio(self) -> dict:
        """ Optimize the Portfolio """
        # This method assumes `prices` and `_returns` are defined elsewhere in the class
        if not hasattr(self, 'prices'):
            logger.error("Price data not available for optimization.")
            raise AttributeError("Price data not set in the instance.")
        
        def f(x: np.ndarray) -> float:
            """optimize based on the Sharpe ratio"""
            returns = self._returns(self._normalize(self.prices)).dot(x)
            return -self._sharpe_ratio(returns)

        constraints = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)})
        bounds = tuple((0.0, 1.0) for _ in range(len(self.allocation)))
        
        result = spo.minimize(f, list(self.allocation.values()), method='SLSQP', bounds=bounds, constraints=constraints)
        new_weight = np.round(result.x, 4)
        return dict(zip(self.allocation.keys(), new_weight))

    def __stats(self, returns: pd.DataFrame, strategy_name: str = None) -> pd.DataFrame:
        """ Return Portfolio Statistics

        Args:
            returns (pd.DataFrame): Returns of the Portfolio
            strategy_name (str, optional): Name of the Strategy. Defaults to None.

        Returns:
            pd.DataFrame: Portfolio Statistics

        """
        if returns.empty:
            logger.error("Empty returns dataframe provided for stats calculation.")
            raise ValueError("Return dataframe cannot be empty.")
        
        rp = self._annualized_returns(returns.iloc[:, 0])  # Assuming single column for returns
        cum_ret = self._cumulative_returns(returns.iloc[:, 0]).iloc[-1]
        sigma = self._annualized_volatility(returns.iloc[:, 0])
        sr = self._sharpe_ratio(returns.iloc[:, 0])
        mdd = self._max_drawdown(returns.iloc[:, 0])
        cr = self._calmar_ratio(returns.iloc[:, 0], rp)
        sortino = self._sortino_ratio(returns.iloc[:, 0])
        
        d = {
            'start_date': returns.index[0],
            'end_date': returns.index[-1],
            "Cumulative Returns": cum_ret,
            'Annualized Returns': rp,
            'Annualized Volatility': sigma,
            'Max Drawdown': mdd,
            'Sharpe Ratio': sr,
            'Calmar Ratio': cr,
            'Sortino Ratio': sortino
        }
        
        if strategy_name is None:
            return pd.DataFrame(d, index=[0])
        else:
            return pd.DataFrame(d, index=[strategy_name])

    
    def _stock_stats(self) -> pd.DataFrame:
        """ Return Portfolio Statistics """
        returns = self._returns(self.prices)
        return self.__stats(returns)
        
    def _portfolio_stats(self) -> pd.DataFrame:
        """ Return Portfolio Statistics """
        returns = self._returns(self.prices).dot(list(self.allocation.values()))
        return self.__stats(returns, port = True)
    
    
    def _optimized_portfolio_stats(self) -> pd.DataFrame:
        """ Return Portfolio Statistics """
        returns = self._returns(self.prices).dot(list(self._optimize_portfolio().values()))
        return self.__stats(returns, port = True)
    
if __name__ == '__main__':
    print('\n(12) The bewildered spirit soul, under the influence of the three modes of material nature, thinks himself to be the doer of activities, which are in actuality carried out by nature.')
    print('\n(13) One who is not envious but who is a kind friend to all living entities, who does not think himself a proprietor, who is free from false ego and equal both in happiness and distress, who is always satisfied and engaged in devotional service with determination and whose mind and intelligence are in agreement with Me—he is very dear to Me. \n')

    
    connections = {
        ##### Price Report ###########################
        'daily_db': 'data/prices/stocks.db', 
        'intraday_db': 'data/prices/stocks_intraday.db',
        'ticker_path': 'data/stocks/tickers.json',
        'stock_names': 'data/stocks/stock_names.db'
    }
    
    stocks = ['spy', 'qqq', 'iwm', 'amd', 'nvda']
    alloc = [(1/len(stocks)) for _ in range(len(stocks))]
    benchmark = ['spy']
    risk_free_rate = 0.00
    
    allocations = dict(zip(stocks, alloc))
    p = PorfolioStats(connections, allocations, risk_free_rate)
    print("Stock Stats:\n")
    print(p._stock_stats())
    print("\nPortfolio Stats:\n")
    print('Equal Weight: ', allocations)
    print(p._portfolio_stats())
    print("\nOptimized Portfolio:\n")
    print(p._optimize_portfolio())
    print(p._optimized_portfolio_stats())
    


