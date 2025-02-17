import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, List, Union, Optional
from tqdm import tqdm
import logging
import sys
import time

# Database Managment and Market Simulator 
from dbm import DBManager
from simulator import MarketSim
from PortfolioStats import PortfolioStats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Policy:
    """
    Policy class for generating orders for the order book. Each method will return the orders for the strategy.

    The orders are in a DataFrame with the following columns:
        1. Date: The date the order was placed
        2. Symbol: The stock symbol
        3. Order: BUY or SELL
        4. Shares: The number of shares to buy or sell

    Methods:
        1. Buy and Hold strategy: Generates orders for a simple buy-and-hold approach.
        2. Optimal Policy: Generates orders for the maximum possible return.

    Args:
        connections (Optional[Dict[str, str]]): Dictionary of database connections.
        orders (Optional[pd.DataFrame]): Orders DataFrame, defaults to None.
    """

    def __init__(self, connections: Optional[Dict[str, str]] = None, orders: Optional[pd.DataFrame] = None) -> None:
        self.marketsim: MarketSim = MarketSim(connections, verbose=False) if connections else None
        self.orders: Optional[pd.DataFrame] = None
        self.intra_day_flag: bool = False
        self.stock: Optional[str] = None
        self.start_date: Optional[dt.datetime] = None
        self.end_date: Optional[dt.datetime] = None
        self.trade_size: Optional[int] = None
        self.stock_prices: Optional[pd.DataFrame] = None

        if orders is not None:
            self._initialize_params(orders)

    def _initialize_params(self, orders: pd.DataFrame) -> None:
        """
        Initialize the parameters for the policy. Creates the following attributes:
            1. orders: Orders DataFrame
            2. intra_day_flag: Flag for intra-day trading
            3. stock: Stock ticker
            4. start_date: Start date
            5. end_date: End date
            6. trade_size: Number of shares to trade
            7. stock_prices: Stock price data
    
        Args:
            orders (pd.DataFrame): Orders DataFrame.
        """
        if not isinstance(orders, pd.DataFrame):
            raise ValueError("orders must be a pandas DataFrame")
        required_columns = ['Symbol', 'Order', 'Shares']
        if not all(col in orders.columns for col in required_columns):
            raise ValueError(f"orders DataFrame must contain columns: {required_columns}")

        self.orders = orders.copy()
        self.intra_day_flag = False if orders.index[0].hour == 0 else True
        stock = orders['Symbol'].iloc[0]
        size = orders['Shares'].iloc[0]
        dtes = self.get_dates(stock, orders.index.min(), orders.index.max(), shift=1)[1:]
        sd, ed = dtes[0], dtes[-1]
        self.stock = stock
        self.start_date = sd
        self.end_date = ed
        self.trade_size = size
        self.stock_prices = self.get_data(stock, sd, ed)
        self.policies = {}
        logger.info("Policy parameters initialized for stock=%s, start_date=%s, end_date=%s", stock, sd, ed)

    def determine_max_shares(self, price: float, cash: float) -> int:
        """
        Determine the maximum number of shares that can be bought.

        Args:
            price (float): Price of the stock.
            cash (float): Available cash.

        Returns:
            int: Maximum number of shares.
        """
        if price <= 0 or cash < 0:
            raise ValueError("Price must be positive and cash must be non-negative")
        return int(np.floor(cash / price))

    def get_data(self, stock: str, sd: dt.datetime, ed: dt.datetime) -> pd.DataFrame:
        """
        Get the stock price data.

        Args:
            stock (str): Stock ticker.
            sd (dt.datetime): Start date.
            ed (dt.datetime): End date.

        Returns:
            pd.DataFrame: Stock price data.
        """
        try:
            stock_prices = self.marketsim.prices([stock], start_date=sd, end_date=ed, intra_day=self.intra_day_flag)
            if stock_prices.empty:
                raise ValueError(f"No data found for stock={stock}, start_date={sd}, end_date={ed}")
            return stock_prices.sort_index(ascending=True)
        except Exception as e:
            logger.error("Error fetching data for stock=%s: %s", stock, e)
            raise

    def buy_and_hold(self, sv: float) -> pd.DataFrame:
        """
        Generate orders for a buy-and-hold strategy.

        Args:
            sv (float): Starting value of the account.

        Returns:
            pd.DataFrame: Orders DataFrame.
        """
        if sv <= 0:
            raise ValueError("Starting value must be positive")
        current_stock_price = self.stock_prices[self.stock].iloc[0]
        self.trade_size = self.determine_max_shares(current_stock_price, sv)
        df = pd.DataFrame(columns=['Symbol', 'Order', 'Shares'], index=self.stock_prices.index)
        df['Symbol'] = self.stock.upper()
        df['Shares'] = 0
        df.iloc[0] = [self.stock.upper(), 'BUY', self.trade_size]
        df.iloc[-1] = [self.stock.upper(), 'SELL', self.trade_size]
        df['Order'] = df['Order'].fillna('HOLD')
        logger.info("Buy-and-Hold strategy generated with trade_size=%d", self.trade_size)
        return df

    def optimal_policy(self, sv: float, D: int = 1) -> pd.DataFrame:
        """
        Generate orders for the optimal policy (maximum return).

        Args:
            sv (float): Starting value of the account.
            D (int): Look-ahead days.

        Returns:
            pd.DataFrame: Orders DataFrame.
        """
        if sv <= 0 or D < 1:
            raise ValueError("Starting value must be positive and D must be >= 1")
        df = pd.DataFrame(columns=['Symbol', 'Order', 'Shares'], index=self.stock_prices.index)
        prices = self.stock_prices[self.stock]
        df['Symbol'] = self.stock.upper()
        df['Order'] = np.where(prices.shift(-D) > prices, 'BUY', 'SELL')
        df['Shares'] = self.determine_max_shares(self.stock_prices[self.stock].iloc[0], sv)
        df2 = df.shift(D).dropna()
        df2['Order'] = np.where(df2['Order'] == 'BUY', 'SELL', 'BUY')
        optimal_orders = pd.concat([df, df2]).sort_index(ascending=True)
        logger.info("Optimal policy generated with look-ahead=%d days", D)
        return optimal_orders

    def get_dates(self, ticker: str, sd: dt.datetime, ed: dt.datetime, shift: Optional[int] = None) -> List[dt.datetime]:
        """
        Get the dates for a given ticker.

        Args:
            ticker (str): Stock ticker.
            sd (dt.datetime): Start date.
            ed (dt.datetime): End date.
            shift (Optional[int]): Shift the dates by a certain number of days.

        Returns:
            List[dt.datetime]: List of dates.
        """
        prices = self.marketsim.prices([ticker], intra_day=self.intra_day_flag)
        # prices = self.stock_prices.copy()
        if shift is None:
            return prices[sd:ed].index.tolist()
        else:
            pr = prices.reset_index()
            pr['Date'] = pd.to_datetime(pr['Date'])
            start = pr[pr['Date'] >= sd].index[0] - shift
            end = pr[pr['Date'] <= ed].index[-1] + shift
            return pr.loc[start:end, 'Date'].tolist()

    def evaluate_policy(self, 
                        orders: pd.DataFrame,
                        sv: float, 
                        name:str = "port",
                        commission: float = 9.95,
                        impact: float = 0.005
        ) -> pd.DataFrame:
        """
        Evaluate a trading policy.

        Args:
            orders (pd.DataFrame): Orders DataFrame with columns ['Symbol', 'Order', 'Shares'] and a datetime index.
            sv (float): Starting value of the account.
            name (str): Name of the strategy.
            commission (float): Commission per trade.
            impact (float): Market impact factor.

        Returns:
            pd.DataFrame: Portfolio value over time.
        """
        if not isinstance(orders, pd.DataFrame):
            raise ValueError("orders must be a pandas DataFrame")
        required_columns = ['Symbol', 'Order', 'Shares']
        if not all(col in orders.columns for col in required_columns):
            raise ValueError(f"orders DataFrame must contain columns: {required_columns}")
        if not isinstance(sv, (int, float)) or sv <= 0:
            raise ValueError("sv (starting value) must be a positive number")
        if not isinstance(commission, (int, float)) or commission < 0:
            raise ValueError("commission must be a non-negative number")
        if not isinstance(impact, (int, float)) or impact < 0:
            raise ValueError("impact must be a non-negative number")

        logger.info("Evaluating policy with starting value=$%.2f, commission=$%.2f, impact=%.3f", sv, commission, impact)
        try:
            tracking_dict = self.marketsim.compute_portvals(orders, sv, commission, impact)
            self.policies[name] = tracking_dict.copy()
            self.stock = orders['Symbol'].iloc[0].lower()
            logger.info("Policy evaluation completed successfully")
            return tracking_dict['portfolio']
        except Exception as e:
            logger.error("Error during policy evaluation: %s", e)
            raise

    def _qs(self, rf_rate: float = 0.0, name: str = 'port') -> pd.DataFrame:
        """
        Return a DataFrame of the portfolio stats.

        Args:
            perf (pd.DataFrame): DataFrame of the portfolio value with a column named 'port_val'.
            rf_rate (float): Risk-free rate, default is 0.0.
            name (str): Name of the portfolio, default is 'port'.

        Returns:
            pd.DataFrame: Portfolio statistics.
        """
        # Get the portfolio values from the policies
        perf = self.policies[name]['portfolio']
        
        if not isinstance(perf, pd.DataFrame):
            raise ValueError("perf must be a pandas DataFrame")
        if 'port_val' not in perf.columns:
            raise ValueError(f"The DataFrame must have a column named 'port_val', only found: {list(perf.columns)}")
        if not isinstance(rf_rate, (int, float)):
            raise ValueError("rf_rate must be a number")
        if not isinstance(name, str):
            raise ValueError("name must be a string")

        # Compute the portfolio statistics, Risk Free Rate of return 
        perf_pct = perf['port_val']
        stats = PortfolioStats(port = perf_pct, risk_free_rate = rf_rate)
        stat_df = stats._portfolio_stats(name=name)
        out = pd.DataFrame({
            "Stock": self.stock.upper(),
            "Days": perf.shape[0],
            "StartDate": perf.index[0],
            "EndDate": perf.index[-1],
            "StartBalance": perf['port_val'].iloc[0],
            "EndBalance": perf['port_val'].iloc[-1],
            
        }, index=[name])
        out = pd.concat([out, stat_df], axis=1)

        
        
        logger.info("Computed portfolio stats for %s", name)
        return out

    def performance(self, orders: pd.DataFrame, sv: float, name: str = "Strategy", commission: float = 0.95, impact: float = 0.005) -> pd.DataFrame:
        """
        Evaluate the performance of the strategy.

        Args:
            orders (pd.DataFrame): Orders DataFrame.
            sv (float): Starting account balance.
            name (str): Name of the strategy.
            commission (float): Commission per trade.
            impact (float): Market impact per trade.

        Returns:
            pd.DataFrame: Performance metrics for the strategy.
        """
        if not isinstance(orders, pd.DataFrame):
            raise ValueError("orders must be a pandas DataFrame")
        if not isinstance(sv, (int, float)) or sv <= 0:
            raise ValueError("sv (starting value) must be a positive number")
        if not isinstance(name, str):
            raise ValueError("name must be a string")

        self._initialize_params(orders)
        pdf = self.__strat_perf(sv)
        strategy = self.evaluate_policy(orders, sv, commission=commission, impact=impact)
        sdf = self._qs(strategy, name=name)
        logger.info("Performance evaluation completed for strategy: %s", name)
        return pd.concat([pdf, sdf])

    def __strat_perf(self, sv: float) -> pd.DataFrame:
        """
        Internal method for evaluating the performance of buy-and-hold and optimal strategies.

        Args:
            sv (float): Starting account value.

        Returns:
            pd.DataFrame: Performance metrics for buy-and-hold and optimal strategies.
        """
        if not isinstance(sv, (int, float)) or sv <= 0:
            raise ValueError("sv (starting value) must be a positive number")

        ################################################################################################
        lodf = []
        # Buy and Hold
        bh = self.buy_and_hold(sv)
        bhdf = self.evaluate_policy(bh, sv=sv, commission=0, impact=0, name = "Buy and Hold")
        b = self._qs(rf_rate=0.0, name="Buy and Hold")
        
        # Optimal Policy
        op = self.optimal_policy(sv)
        optdf = self.evaluate_policy(op, sv=sv, commission=0, impact=0, name = "Optimal Policy")
        a = self._qs(rf_rate = 0.0, name="Optimal Policy")

        lodf.append(pd.concat([b, a], axis=0))
        return pd.concat(lodf)

    def eval_multiple_orders(self, orders: List[pd.DataFrame], names: List[str], sv: float, 
                             commission: float = 0.95, impact: float = 0.005) -> pd.DataFrame:
        """
        Evaluate multiple strategies.

        Args:
            orders (List[pd.DataFrame]): List of orders DataFrames.
            names (List[str]): List of strategy names.
            sv (float): Starting account balance.
            commission (float): Commission per trade.
            impact (float): Market impact per trade.

        Returns:
            pd.DataFrame: Performance metrics for all strategies.
        """
        if not isinstance(orders, list) or not all(isinstance(o, pd.DataFrame) for o in orders):
            raise ValueError("orders must be a list of pandas DataFrames")
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError("names must be a list of strings")
        if len(orders) != len(names):
            raise ValueError("orders and names must have the same length")
        if not isinstance(sv, (int, float)) or sv <= 0:
            raise ValueError("sv (starting value) must be a positive number")

        self._initialize_params(orders[0])
        pdf = self.__strat_perf(sv)
        strats = [self.evaluate_policy(o, sv, commission=commission, impact=impact, name = names[j]) for j, o in enumerate(orders)]
        pdfs = [self._qs(rf_rate = 0.02, name=names[x]) for x in range(len(orders))]
        # Concatenate the performance metrics. 
        sdf = pd.concat(pdfs)
        self.list_eval = dict(zip(names, strats))
        logger.info("Evaluated %d strategies", len(orders))
        return pd.concat([pdf, sdf])

if __name__ == "__main__":
    connections = {}
    pre = '../'
    with open('config.env') as f:
        for line in f:
            name, path = line.strip().split('=')
            connections[name.lower()] = pre + path

    P = Policy(connections)
    orders1 = pd.read_csv('orders/DTLearn.csv', index_col='Date', parse_dates=True, na_values=['nan']).sort_index()
    orders2 = pd.read_csv('orders/RTLearn.csv', index_col='Date', parse_dates=True, na_values=['nan']).sort_index()
    df = P.eval_multiple_orders(orders=[orders1, orders2], names=['DTLEARN', 'RTLEARN'], sv=10_000)
    
    print(df)
    print()
    print(P.policies.keys())
