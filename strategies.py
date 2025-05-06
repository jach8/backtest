"""Trading strategy implementations."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from .simulator import MarketSim
from .config import BacktestConfig

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Policy:
    """Trading policy implementation."""

    def __init__(self, db_connections: Dict[str, str], verbose = False) -> None:
        """
        Initialize Policy.

        Args:
            db_connections (Dict[str, str]): Database connections
            verbose
        """
        self.marketsim = MarketSim(db_connections, verbose)
        self.orders = None
        self.intra_day_flag = False
        self.stock = None
        self.start_date = None
        self.end_date = None
        self.stock_prices = None
        self.trade_size = None

    def _initialize_params(self, orders: pd.DataFrame) -> None:
        """
        Initialize parameters from orders.

        Args:
            orders (pd.DataFrame): Trading orders
        """
        if orders.empty:
            raise ValueError("Orders DataFrame cannot be empty")

        stock = orders['Symbol'].iloc[0]
        start_date = orders.index.min()
        end_date = orders.index.max()
        self.orders = orders
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date

        # Get price data
        self.stock_prices = self.marketsim.get_close([stock], start_date, end_date)
        
        # Set trade size from first non-zero trade
        non_zero_trades = orders[orders['Shares'] > 0]
        self.trade_size = non_zero_trades['Shares'].iloc[0] if not non_zero_trades.empty else 100

    def determine_max_shares(self, price: float, cash: float) -> int:
        """
        Calculate maximum shares that can be bought.

        Args:
            price (float): Current price
            cash (float): Available cash

        Returns:
            int: Maximum shares
        """
        if price <= 0:
            raise ValueError("Price must be positive")
        if cash <= 0:
            raise ValueError("Cash must be positive")
        return int(cash // price)

    def buy_and_hold(self, sv: float) -> pd.DataFrame:
        """
        Generate orders for a buy-and-hold strategy.

        Args:
            sv (float): Starting value

        Returns:
            pd.DataFrame: Orders DataFrame
        """
        if sv <= 0:
            raise ValueError("Starting value must be positive")

        if self.stock_prices is None or self.stock is None:
            raise ValueError("Strategy not properly initialized")

        current_stock_price = self.stock_prices[self.stock].iloc[0]
        max_shares = self.determine_max_shares(current_stock_price, sv * 0.99)  # Use 90% of cash

        orders = pd.DataFrame(index=self.stock_prices.index)
        orders['Symbol'] = self.stock
        orders['Order'] = 'HOLD'
        orders['Shares'] = 0

        # Buy on first day
        orders.iloc[0, orders.columns.get_loc('Order')] = 'BUY'
        orders.iloc[0, orders.columns.get_loc('Shares')] = max_shares

        # Sell on last day
        orders.iloc[-1, orders.columns.get_loc('Order')] = 'SELL'
        orders.iloc[-1, orders.columns.get_loc('Shares')] = max_shares

        return orders

    def optimal_policy(self, D: int = 1) -> pd.DataFrame:
        """
        Generate orders for the optimal policy (maximum return).

        Args:
            sv (float): Starting value
            D (int): Look-ahead days

        Returns:
            pd.DataFrame: Orders DataFrame
        """
        if D < 1:
            raise ValueError("Starting value must be positive and D must be >= 1")

        if self.stock_prices is None or self.stock is None:
            raise ValueError("Strategy not properly initialized")

        D = 1

        # df = pd.DataFrame(index=self.stock_prices.index)
        # price_series = self.stock_prices[self.stock]
        # future_prices = price_series.shift(-D)

        # df['Symbol'] = self.stock
        # df['Shares'] = self.trade_size if self.trade_size else 100
        # df['Order'] = np.where(future_prices > price_series, 'BUY', 'SELL')

        # # # Close out open trades 
        # df2 = df.copy().shift(D).dropna()
        # df2['Order'] = np.where(df2['Order'] == 'BUY', 'SELL', 'BUY')
        # df2['Shares'] = df2['Shares']
        # return pd.concat([df, df2], axis=0).dropna().sort_index()

        df = pd.DataFrame(index=self.stock_prices.index)
        df['Symbol'] = self.stock
        df['Order'] = 'HOLD'
        df['Shares'] = 0

        price_series = self.stock_prices[self.stock]
        for i in range(len(price_series) - D):
            curr_price = price_series.iloc[i]
            future_price = price_series.iloc[i + D]
            
            # Close any existing position
            if i > 0 and df.iloc[i-1]['Order'] == 'BUY':
                df.iloc[i, df.columns.get_loc('Order')] = 'SELL'
                df.iloc[i, df.columns.get_loc('Shares')] = self.trade_size

            # Open new position if profitable
            if future_price > curr_price:
                df.iloc[i, df.columns.get_loc('Order')] = 'BUY'
                df.iloc[i, df.columns.get_loc('Shares')] = self.trade_size

        # Close any remaining position on last day
        if df.iloc[-2]['Order'] == 'BUY':
            df.iloc[-1, df.columns.get_loc('Order')] = 'SELL'
            df.iloc[-1, df.columns.get_loc('Shares')] = self.trade_size

        return df

    def eval_multiple_orders(self, 
            orders: List[pd.DataFrame], 
            names: List[str],
            sv: float,
            commission: float = 9.95,
            impact: float = 0.005
        ) -> pd.DataFrame:
        """
        Evaluate multiple trading policies.

        Args:
            orders (List[pd.DataFrame]): List of trading orders
            names (List[str]): Strategy names
            sv (float): Starting value
            commission (float): Trading commission
            impact (float): Market impact

        Returns:
            pd.DataFrame: Performance comparison
        """
        if len(orders) != len(names):
            raise ValueError("Number of orders must match number of names")

        results = pd.DataFrame()
        self.list_eval = {}
        for order_df, name in zip(orders, names):
            self._initialize_params(order_df)
            portvals = self.marketsim.compute_portvals(order_df, sv, commission, impact)
            
            trade_days = len(portvals['portfolio'])
            end_balance = portvals['portfolio']['port_val'].iloc[-1]
            
            results.loc[name, 'Stock'] = self.stock
            results.loc[name, 'Days'] = trade_days
            results.loc[name, 'StartDate'] = self.start_date.strftime('%Y-%m-%d')
            results.loc[name, 'EndDate'] = self.end_date.strftime('%Y-%m-%d')
            results.loc[name, 'StartBalance'] = sv
            results.loc[name, 'EndBalance'] = end_balance
            results.loc[name, 'Return'] = ((end_balance - sv) / sv) * 100
            results.loc[name, 'Commission'] = commission
            results.loc[name, 'Impact'] = impact
            self.list_eval[name] = portvals

        # Add the buy and hold and optimal policy to the results
        self.list_eval['Buy and Hold'] = self.marketsim.compute_portvals(self.buy_and_hold(sv), startval = sv, impact = 0, commission = 0 )    
        results.loc['Buy and Hold', 'Stock'] = self.stock
        results.loc['Buy and Hold', 'Days'] = trade_days
        results.loc['Buy and Hold', 'StartDate'] = self.start_date.strftime('%Y-%m-%d')
        results.loc['Buy and Hold', 'EndDate'] = self.end_date.strftime('%Y-%m-%d')
        results.loc['Buy and Hold', 'StartBalance'] = self.marketsim.params.get('startval', sv)
        results.loc['Buy and Hold', 'EndBalance'] = self.list_eval['Buy and Hold']['portfolio']['port_val'].iloc[-1]
        results.loc['Buy and Hold', 'Return'] = ((results.loc['Buy and Hold', 'EndBalance'] - sv) / sv) * 100 
        results.loc['Buy and Hold', 'Commission'] = self.marketsim.params.get('commission', commission)
        results.loc['Buy and Hold', 'Impact'] = self.marketsim.params.get('impact', impact)
        
        # Optimal policy
        self.list_eval['Optimal Policy'] = self.marketsim.compute_portvals(self.optimal_policy(D=1), startval = sv, impact = 0, commission = 0 )
        results.loc['Optimal Policy', 'Stock'] = self.stock
        results.loc['Optimal Policy', 'Days'] = trade_days
        results.loc['Optimal Policy', 'StartDate'] = self.start_date.strftime('%Y-%m-%d')
        results.loc['Optimal Policy', 'EndDate'] = self.end_date.strftime('%Y-%m-%d')
        results.loc['Optimal Policy', 'StartBalance'] = self.marketsim.params.get('startval', sv)
        results.loc['Optimal Policy', 'EndBalance'] = self.list_eval['Optimal Policy']['portfolio']['port_val'].iloc[-1]
        results.loc['Optimal Policy', 'Return'] = ((results.loc['Optimal Policy', 'EndBalance'] - sv) / sv) * 100 
        results.loc['Optimal Policy', 'Commission'] = self.marketsim.params.get('commission', commission)
        results.loc['Optimal Policy', 'Impact'] = self.marketsim.params.get('impact', impact)

        return results

    def _qs(self, name: str = "Strategy", portvals: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate quick statistics for current strategy.

        Args:
            name (str): Strategy name
            portvals (pd.DataFrame): Portfolio values

        Returns:
            pd.DataFrame: Strategy statistics
        """
        from .PortfolioStats import PortfolioStats
        if hasattr(self, '_portvals'):
            stats = PortfolioStats(self._portvals)
            return stats._portfolio_stats(name=name)
        else:
            if portvals is None:
                raise ValueError("No portfolio values available, Provide them or Run evaluate_policy first.")
            stats = PortfolioStats(portvals)
            return stats._portfolio_stats(name=name)
        
    

    ##### Updated Methods for Backtesting with a Configuration Dictionary #####
    ### Example Config Dictionary:
    # config = {
    #     'initial_capital': 100000,    
    #     'commission': 9.95,
    #     'impact': 0.005,
    #     'slippage': 0.001,
    #     'lookahead': 1,
    #     'intra_day': False,
    #     'start_date': '2023-01-01',
    #     'end_date': '2023-12-31',
    #     'stock': 'AAPL',
    #     'trade_size': 100,
    #     'price_data': None
    # }
    ####

    def evaluate_policy_with_config(self,
            orders: pd.DataFrame,
            config: BacktestConfig,
            name: Optional[str] = None
        ) -> pd.DataFrame:
        """
        Evaluate a trading policy using BacktestConfig parameters.
        This is an alternative to evaluate_policy that uses the BacktestConfig class
        for more sophisticated backtesting features.

        Args:
            orders (pd.DataFrame): Trading orders
            config (BacktestConfig): Backtesting configuration parameters
            name (str, optional): Strategy name for results

        Returns:
            pd.DataFrame: Portfolio performance metrics
        """
        self._initialize_params(orders)
        portvals = self.marketsim.compute_portvals_with_config(orders, config)
        return portvals['portfolio']

    def eval_multiple_orders_with_config(self,
            orders: List[pd.DataFrame],
            names: List[str],
            config: BacktestConfig
        ) -> pd.DataFrame:
        """
        Evaluate multiple trading policies using BacktestConfig parameters.
        This is an alternative to eval_multiple_orders that uses the BacktestConfig class
        for more sophisticated backtesting features.

        Args:
            orders (List[pd.DataFrame]): List of trading orders
            names (List[str]): Strategy names
            config (BacktestConfig): Backtesting configuration parameters

        Returns:
            pd.DataFrame: Performance comparison of strategies
        """
        if len(orders) != len(names):
            raise ValueError("Number of orders must match number of names")

        results = pd.DataFrame()
        self.list_eval = {}
        for order_df, name in zip(orders, names):
            self._initialize_params(order_df)
            portvals = self.marketsim.compute_portvals_with_config(order_df, config)
            
            trade_days = len(portvals['portfolio'])
            end_balance = portvals['portfolio']['port_val'].iloc[-1]
            
            results.loc[name, 'Stock'] = self.stock
            results.loc[name, 'Days'] = trade_days
            results.loc[name, 'StartDate'] = self.start_date.strftime('%Y-%m-%d')
            results.loc[name, 'EndDate'] = self.end_date.strftime('%Y-%m-%d')
            results.loc[name, 'StartBalance'] = config.initial_capital
            results.loc[name, 'EndBalance'] = end_balance
            results.loc[name, 'Return'] = (end_balance - config.initial_capital) / config.initial_capital
            results.loc[name, 'Commission'] = config.commission
            results.loc[name, 'Impact'] = config.impact
            self.list_eval[name] = portvals
        return results

if __name__ == "__main__":
    ## Just get the buy and hold, and optimal policy 
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from bin.main import get_path
    policy = Policy(get_path())
    
    
    import pickle
    
    orders_dict = pickle.load(open('backtest/test_orders.pkl', 'rb'))
    orders = []
    names = []
    for name, order in orders_dict.items():
        orders.append(order)
        names.append(name)

    res = policy.eval_multiple_orders(orders, names, 100000)
    res = res.round(2)
    
    print('\n\n',res, '\n\n')
