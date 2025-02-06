"""
To Do: 

    - Integrate this into the Policy Class. 


Calculate Portfolio Statistics 


"""
import datetime as dt 
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import scipy.optimize as spo

from strategies import Policy 

class PorfolioStats:
    """
    
    This should take the Strategies Class
    """
    def __init__(self, allocation, risk_free_rate):
        self.allocation = allocation
        self.rf = risk_free_rate
        self.prices = self.__get_prices()["2024-01-01":]
        self.N = 360
        
    def __get_prices(self):
        """ Return the Normalized Prices of the Stocks """
        self.stocks = list(self.allocation.keys())
        # return self._normalize(self.get_close(stocks).dropna())
        return self.get_close(self.stocks).dropna()  
    
    def _annualized_returns(self, ret):
        """ Return the Annualized Returns of the Portfolio """
        return ret.mean() * self.N - self.rf
    
    def _annualized_volatility(self, ret):
        """ Return the Annualized Volatility of the Portfolio """
        return ret.std() * np.sqrt(self.N)
    
    def _cumulative_returns(self, ret):
        """ Return the Cumulative Returns of the Portfolio """
        return (1 + ret).cumprod() - 1
    
    def _max_drawdown(self, ret):
        """ Return the Maximum Drawdown of the Portfolio 
            Maximum Drawdown = (Peak - Trough) / Peak
        """
        comp_ret = (ret + 1).cumprod()
        peak = comp_ret.expanding(min_periods=1).max()
        dd = (comp_ret / peak) - 1
        return dd.min()
    
    def _sharpe_ratio(self, returns):
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
        return mean / sigma
    
    def _sortino_ratio(self, ret):
        """
        Return the Sorinto Ratio of the Portfolio 
            Sortino Ratio = (Rp - Rf) / σn
                - Rp = Annual Expected Returns of the asset 
                - Rf = Risk Free Rate of Return
                - σn = Negative Standard Deviation of returns
        """
        mu = ret.mean() * self.N - self.rf
        std_neg = ret[ret < 0].std() * np.sqrt(self.N)
        return mu / std_neg
        
        
    def _calmar_ratio(self, ret, rbar):
        """ Risk/Reward ratio the Calmar Ratio of the Portfolio 
            Calmar Ratio = (Rbar) / MDD
                - Rbar = Annualized Returns 
                - MDD = Maximum Drawdown
        """
        return rbar / abs(self._max_drawdown(ret))
    
    def _optimize_portfolio(self):
        """ Optimize the Portfolio """
        def f(x):
            """optimize based on cumulative returns"""
            returns = self._returns(self._normalize(self.prices)).dot(x)
            return -self._cumulative_returns(returns).iloc[-1]
        
        def f(x):
            """optimize based on the sharpe ratio"""
            returns = self._returns(self._normalize(self.prices)).dot(x)
            return -self._sharpe_ratio(returns)
        
        constraints = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)})
        bounds = tuple((0.0, 1.0) for x in range(len(self.allocation)))
        result = spo.minimize(f, list(self.allocation.values()), method='SLSQP', bounds=bounds, constraints=constraints)
        new_weight = np.round(result.x, 4)
        return dict(zip(self.allocation.keys(), new_weight))
    
    def __stats(self, returns, port = None):
        rp = self._annualized_returns(returns)
        cum_ret = self._cumulative_returns(returns).iloc[-1]
        sigma = self._annualized_volatility(returns)
        sr = self._sharpe_ratio(returns)
        mdd = self._max_drawdown(returns)
        cr = self._calmar_ratio(returns, rp)
        sortino = self._sortino_ratio(returns)
        d = {
            'start_date': self.prices.index[0],
            'end_date': self.prices.index[-1],
            "Cumulative Returns": cum_ret,
            'Annualized Returns': rp,
            'Annualized Volatility': sigma,
            'Max Drawdown': mdd,
            'Sharpe Ratio': sr,
            'Calmar Ratio': cr,
            'Sortino Ratio': sortino
            }
        if port == None:
            return pd.DataFrame(d)
        else: 
            return pd.DataFrame(d, index=[port])
    
    def _stock_stats(self):
        """ Return Portfolio Statistics """
        returns = self._returns(self.prices)
        return self.__stats(returns)
        
    def _portfolio_stats(self):
        """ Return Portfolio Statistics """
        returns = self._returns(self.prices).dot(list(self.allocation.values()))
        return self.__stats(returns, port = True)
    
    
    def _optimized_portfolio_stats(self):
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
    


