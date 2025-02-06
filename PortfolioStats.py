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
    
    Args:
        port (pd.Series): Portfolio Performance DataFrame
        risk_free_rate (float): Risk Free Rate of Return
    """
    def __init__(self, port: pd.DataFrame, risk_free_rate: float):
        self.port = port.copy()
        self.rf = risk_free_rate
        self.N = 360

    def _annualized_returns(self, ret: pd.Series) -> float:
        """ Return the Annualized Returns of the Portfolio """
        if ret.empty:
            logger.error("Empty return series provided.")
            raise ValueError("Return series cannot be empty.")
        return ret.mean() * self.N - self.rf

    def _returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """ Return the Daily Returns of the Portfolio """
        return prices.pct_change().iloc[1:]

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


    def __stats(self, returns: pd.Series, strategy_name: str = None) -> pd.DataFrame:
        """ Return Portfolio Statistics

        Args:
            returns (pd.Series): Returns of the Portfolio
            strategy_name (str, optional): Name of the Strategy. Defaults to None.

        Returns:
            pd.DataFrame: Portfolio Statistics

        """
        if returns.empty:
            logger.error("Empty returns dataframe provided for stats calculation.")
            raise ValueError("Return dataframe cannot be empty.")
        
        rp = self._annualized_returns(returns)  # Assuming single column for returns
        cum_ret = self._cumulative_returns(returns).iloc[-1]
        sigma = self._annualized_volatility(returns)
        sr = self._sharpe_ratio(returns)
        mdd = self._max_drawdown(returns)
        cr = self._calmar_ratio(returns, rp)
        sortino = self._sortino_ratio(returns)
        adr = returns.mean()
        sddr = returns.std()
        d = {
            "cumulativeReturns": cum_ret,
            'averageDailyReturns': adr,
            'stdDailyReturns': sddr,
            'annualizedReturns': rp,
            'annualizedVolatility': sigma,
            'maxDrawDown': mdd,
            'sharpeRatio': sr,
            'calmarRatio': cr,
            'sorintinoRatio': sortino
        }
        
        if strategy_name is None:
            return pd.DataFrame(d, index=[0])
        else:
            return pd.DataFrame(d, index=[strategy_name])
        
    def _portfolio_stats(self, name: str = "Portfolio") -> pd.DataFrame:
        """ Return Portfolio Statistics """
        returns = self._returns(self.port)
        return self.__stats(returns, strategy_name = name)
    



if __name__ == '__main__':
    print('\n(12) The bewildered spirit soul, under the influence of the three modes of material nature, thinks himself to be the doer of activities, which are in actuality carried out by nature.')
    print('\n(13) One who is not envious but who is a kind friend to all living entities, who does not think himself a proprietor, who is free from false ego and equal both in happiness and distress, who is always satisfied and engaged in devotional service with determination and whose mind and intelligence are in agreement with Me—he is very dear to Me. \n')
    connections = {}
    pre = '../'
    with open('config.env') as f:
        for line in f:
            name, path = line.strip().split('=')
            connections[name.lower()] = pre + path
      
      
    from simulator import MarketSim
    ms = MarketSim(connections, verbose = True)
    orders = pd.read_csv('orders/DTLearn.csv', index_col='Date', parse_dates=True, na_values=['nan']).sort_index()
    orders = pd.read_csv('orders/additional_orders/orders2.csv', index_col='Date', parse_dates=True, na_values=['nan']).sort_index()
    d = ms.compute_portvals(orders, 1000000, commission = 9.95, impact = 0.005)  
    port = d['portfolio']['port_val']
    
    ms.close_all()
    
    p = PortfolioStats(port, 0.00)
    print(p._portfolio_stats())
        
    