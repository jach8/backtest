"""
#### To Do: 2/5/2025

Areas for Improvement:
    Single Responsibility Principle: The class handles too many responsibilities (database access, order execution, portfolio tracking). 
    This could be split into smaller, more focused classes (e.g., PortfolioManager, OrderExecutor).

    Extensibility: Adding new features (e.g., short selling, margin accounts) would require significant changes. A modular design would make this easier.

    Input Validation: The code assumes well-formed inputs (e.g., orders DataFrame). Adding validation checks would improve robustness.

    Performance: For large datasets, repeatedly updating DataFrames in a loop could be inefficient. Consider vectorized operations or batch updates.

    Error Handling: While present, it could be more granular (e.g., handling specific database errors).

Suggestions for Version 2 (V2):
    Split into smaller classes:
        PortfolioManager: Handles trades, holdings, and portfolio value computation.

        OrderExecutor: Handles order execution logic (buy, sell, etc.).

        MarketSim: Orchestrates the simulation and interacts with the database.

    Add support for advanced features:
        Short selling, limit orders, and margin accounts.
        Options Trading

    Multi-asset portfolios (e.g., stocks, bonds).

Enhance extensibility:
    Use a strategy pattern for order execution to make adding new order types easier.

Improve performance:
    Use vectorized operations where possible to reduce loop overhead.

Input validation:
    Validate the orders DataFrame (e.g., check for required columns, data types).

"""


import datetime as dt
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from tqdm import tqdm
import logging
import time

# Import the database manager (assumed to be implemented elsewhere)
from .dbm import DBManager
# from dbm import DBManager
from .config import BacktestConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketSim(DBManager):
    """
    A class for simulating market orders.

    This class simulates market operations such as buying, selling, and holding stocks.
    It uses a database connection manager (DBManager) to query stock price information.

    Args:
        connections (Dict[str, str]): The database connections to query stock price information.
        verbose (bool): If True, prints detailed logs of operations. Default is False.

    Attributes:
        verbose (bool): Flag to enable verbose logging.
        tracking (Dict[str, pd.DataFrame]): Tracks trades, holdings, stock prices, and portfolio values.

    Methods:
        prices: Return the prices for the given symbols.
        compute_portvals: Compute the portfolio value based on orders.
    """

    def __init__(self, connections: Dict[str, str], verbose: bool = False) -> None:
        """
        Initialize the MarketSim class, this will also create an empty dict, to store the tracking information.
        
        Args: 
            connections: Dict[str, str]: The database connections to query stock price information.
            verbose: bool: If True, prints detailed logs of operations. Default is False.
        
        Returns:
            None
        
        
        """
        super().__init__(connections)
        self.verbose: bool = verbose
        self.tracking: Dict[str, pd.DataFrame] = {}
        logger.info("MarketSim initialized with verbose=%s", verbose)
        # If self.verbose is True set the logging level to DEBUG
        if self.verbose == True: 
            logger.setLevel(logging.DEBUG)
        
    def get_close(self, stocks: List[str], start: dt.datetime = None, end: dt.datetime = None, intra_day_flag: bool= False) -> pd.DataFrame:
        """
        Get the close prices for the specified stocks.

        Args:
            stocks (List[str]): List of stock symbols.
            start (dt.datetime): Start date for the price query.
            end (dt.datetime): End date for the price query.
            intra_day_flag (bool): If True, retrieves intra-day prices. Default is False.

        Returns:
            pd.DataFrame: DataFrame containing the close prices for the specified stocks.
        """
        table_names = ', '.join([f'"{stock}"' for stock in stocks])
        if start is not None and end is not None:
            logger.debug("Fetching close prices for stocks: %s, start=%s, end=%s", stocks, start, end)
        else:
            logger.debug("Fetching close prices for stocks: %s", stocks)
        
        out = []
        if intra_day_flag: 
            con = 'intraday_db'
        else:
            con = 'daily_db'
        with self.get_connection(con) as conn:
            for stock in stocks:
                q = f'''select date(date) as "Date", close as "{stock}" from {stock} order by date(date) asc'''
                df = pd.read_sql_query(q, conn, parse_dates=['Date'], index_col='Date')
                out.append(df)
        out = pd.concat(out, axis=1)
        if start is not None and end is not None:
            return out[start:end]
        return out

    def prices(self, syms: Union[str, List[str]], start_date: Optional[dt.datetime] = None, 
                end_date: Optional[dt.datetime] = None, intra_day: bool = False) -> pd.DataFrame:
        """
        Return the prices for the specified symbols.

        Args:
            syms (Union[str, List[str]]): Stock symbol(s) to retrieve prices for.
            start_date (Optional[datetime]): Start date for the price query.
            end_date (Optional[datetime]): End date for the price query.
            intra_day (bool): If True, retrieves intra-day prices. Default is False.

        Returns:
            pd.DataFrame: DataFrame containing the prices for the specified symbols.
        """
        if isinstance(syms, str):
            syms = [syms]

        logger.debug("Fetching prices for symbols: %s, intra_day=%s", syms, intra_day)
        try:
            if not intra_day:
                out = self.get_close(stocks=syms, start=start_date, end=end_date)
            else:
                out = self.get_close(stocks=syms, start=start_date, end=end_date, intra_day_flag=True)

            if start_date is None:
                return out
            return out[start_date:end_date]
        except Exception as e:
            logger.error("Error fetching prices: %s", e)
            raise

    def __SetupSim(self, orders: pd.DataFrame, start_val: float) -> None:
        """
        Initialize the simulation.
        This will also initialize the tracking dictionary, which includes:
            - trades: Dataframe to track the number of shares bought or sold
            - holdings: Dataframe to track the number of shares held
            - stock_prices: Closing Prices for a stock. 
        
        Args:
            orders (pd.DataFrame): The orders DataFrame with columns ['Symbol', 'Order', 'Shares'].
            start_val (float): The starting value of the portfolio.
        
        Returns: 
            None
        """
        logger.debug("Setting up simulation with start_val=%.2f", start_val)
        orders.index = pd.to_datetime(orders.index)
        intra_day_flag = False if orders.index[0].time() == dt.time(0, 0) else True
        orders = orders.sort_index()
        stocks = list(set(orders['Symbol']))
        stock_prices = self.prices(stocks, orders.index[0], orders.index[-1], intra_day=intra_day_flag)

        # Add Cash Column to stock prices
        stock_prices['Cash'] = 1.0

        # Create trades DataFrame to track the number of shares bought or sold
        trades = stock_prices.copy() * 0.0
        trades["Cash"] = [start_val] + [0.0] * (len(trades) - 1)

        # Create Holdings DataFrame to track the number of shares held
        holdings = trades.copy()
        holdings.iloc[0] = trades.iloc[0]

        if self.verbose:
            logger.debug("Starting Balance: $%.2f", start_val)

        self.tracking = {'trades': trades, 'holdings': holdings, 'stock_prices': stock_prices}
        logger.debug("Simulation setup completed")

    def __check_cash(self, date: dt.datetime, cost: float) -> bool:
        """
        Check if there is enough cash to make the trade.

        Args:
            date (dt.datetime): The date of the trade.
            cost (float): The cost of the trade.

        Returns:
            bool: True if there is enough cash, False otherwise.
        """
        cash = self.__get_current_cash(date)
        logger.debug("Checking cash: Available=$%.2f, Required=$%.2f", cash, cost)
        return cash >= cost

    def __check_shares(self, date: dt.datetime, stock: str, shares: int) -> bool:
        """
        Check if there are enough shares to sell.

        Args:
            date (dt.datetime): The date of the trade.
            stock (str): The stock symbol.
            shares (int): The number of shares to sell.

        Returns:
            bool: True if there are enough shares, False otherwise.
        """
        available_shares = self.__get_current_shares(date, stock)
        logger.debug("Checking shares for %s: Available=%d, Required=%d", stock, available_shares, shares)
        return available_shares >= shares

    def __get_current_cash(self, date: dt.datetime) -> float:
        """
        Get the current cash in the account.
        
        Args:
            date (dt.datetime): The date of the trade.
            
        Returns:
            float: The current cash balance.
    
        """
        return self.tracking['holdings'].cumsum().loc[date, 'Cash']

    def __get_current_shares(self, date: dt.datetime, stock: str) -> int:
        """
        Get the current shares of a stock in the account.
        
        Args:
            date (dt.datetime): The date of the trade.
            stock (str): The stock symbol.
        
        Returns:
            int: The current number of shares of the stock.
        """
        return int(self.tracking['holdings'].cumsum().loc[date, stock])

    def __BuyOrder(self, date: dt.datetime, stock: str, shares: int, stock_price: float, 
                   commission: float = 5.50, impact: float = 0.005) -> None:
        """
        Execute a Buy Order. This function will update the tracking Dictionary with the new trade: 
            Trades: 
                - Subtract the cost of the trade from the cash balance.
                - Add the number of shares to the stock.
            Holdings:
                - Subtract the cost of the trade from the cash balance.
                - Add the number of shares to the stock.
        
        Args:
            date (dt.datetime): The date of the trade.
            stock (str): The stock symbol.
            shares (int): The number of shares to buy.
            stock_price (float): The price of the stock.
            commission (float): The commission for the trade. Default is 5.50.
            impact (float): The market impact of the trade. Default is 0.005.
        
        Returns:
            None
        """
        cost = shares * stock_price * (1 + impact) + commission
        if not self.__check_cash(date, cost):
            logger.debug("Insufficient Funds to Buy %d shares of %s @ %.2f", shares, stock.upper(), stock_price)
            return
        self.tracking['trades'].loc[date, 'Cash'] -= cost
        self.tracking['trades'].loc[date, stock] += shares
        self.tracking['holdings'].loc[date, 'Cash'] -= cost
        self.tracking['holdings'].loc[date, stock] += shares
        logger.debug("Buy Order executed: %d shares of %s @ %.2f", shares, stock.upper(), stock_price)

    def __SellOrder(self, date: dt.datetime, stock: str, shares: int, stock_price: float, 
                    commission: float = 5.50, impact: float = 0.005) -> None:
        """
        Execute a Sell Order. This function will update the tracking Dictionary with the new trade:
            Trades:
                - Add the cost of the trade to the cash balance.
                - Subtract the number of shares from the stock.
            Holdings:
                - Add the cost of the trade to the cash balance.
                - Subtract the number of shares from the stock.    
        
        Args:
            date (dt.datetime): The date of the trade.
            stock (str): The stock symbol.
            shares (int): The number of shares to sell.
            stock_price (float): The price of the stock.
            commission (float): The commission for the trade. Default is 5.50.
            impact (float): The market impact of the trade. Default is 0.005.
        
        Returns:
            None
        """
        if not self.__check_shares(date, stock, shares):
            logger.debug("Insufficient Shares to Sell %d shares of %s", shares, stock.upper())
            return
        cost = shares * stock_price * (1 - impact) - commission
        self.tracking['trades'].loc[date, 'Cash'] += cost
        self.tracking['trades'].loc[date, stock] -= shares
        self.tracking['holdings'].loc[date, 'Cash'] += cost
        self.tracking['holdings'].loc[date, stock] -= shares
        logger.debug("Sell Order executed: %d shares of %s @ %.2f", shares, stock.upper(), stock_price)

    def __HoldOrder(self, date: dt.datetime, stock: str, stock_price: float) -> None:
        """
        Execute a Hold Order.
        This function will update the tracking Dictionary with the new trade:
            Trades:
                - No change in the cash balance or shares.
            Holdings:
                - No change in the cash balance or shares.
            
        Args:
            date (dt.datetime): The date of the trade.
            stock (str): The stock symbol.
            stock_price (float): The price of the stock.
        
        """
        self.tracking['trades'].loc[date, stock] = 0
        logger.debug("Hold Order executed for %s", stock)

    def __VerboseOrder(self, date: dt.datetime, stock: str, order: str, shares: int) -> None:
        """Print the Order details if verbose is enabled."""
        if self.verbose:
            cash = self.__get_current_cash(date)
            logger.debug("%s Order: %d shares of %s, Cash Balance: $%.2f", order, shares, stock.upper(), cash)

    def __ExecuteOrder(self, date: dt.datetime, stock: str, order: str, shares: int, stock_price: float, 
                       commission: float = 5.50, impact: float = 0.005) -> None:
        """
        Execute an Order (Buy, Sell, or Hold). This will execute the order. 
        
        Args:
            date (dt.datetime): The date of the trade.
            stock (str): The stock symbol.
            order (str): The order type (BUY, SELL, or HOLD).
            shares (int): The number of shares to buy or sell.
            stock_price (float): The price of the stock.
            commission (float): The commission for the trade. Default is 5.50.
            impact (float): The market impact of the trade. Default is 0.005.
        
        Returns:
            None
        
        """
        logger.debug("Executing %s order for %s: %d shares", order, stock, shares)
        if order == 'BUY':
            self.__BuyOrder(date, stock, shares, stock_price, commission, impact)
        elif order == 'SELL':
            self.__SellOrder(date, stock, shares, stock_price, commission, impact)
        else:
            self.__HoldOrder(date, stock, stock_price)
        self.__VerboseOrder(date, stock, order, shares)

    def compute_portvals(self, orders: pd.DataFrame, startval: float, commission: float = 5.50, impact: float = 0.005) -> Dict[str, pd.DataFrame]:
        """
        Compute the portfolio value. This function will iterate through the orders dataframe and execute the orders. 
        This function also creates two new entries in the tracking dictionary:
            1. Orders: The orders DataFrame
            2. Portfolio: The portfolio value DataFrame
        

        Args:
            orders (pd.DataFrame): The orders DataFrame with columns ['Symbol', 'Order', 'Shares'].
            startval (float): The starting value of the portfolio.
            commission (float): The commission for each trade. Default is 5.50.
            impact (float): The market impact of each trade. Default is 0.005.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing the tracking information
                example:
                    {
                    'trades': pd.DataFrame, 
                    'holdings': pd.DataFrame, 
                    'stock_prices': pd.DataFrame, 
                    'portfolio': pd.DataFrame
                    'orders': pd.DataFrame
                    }
        """
        self.__SetupSim(orders, startval)
        self.tracking['orders'] = orders
        self.params = {'startval': startval, 'commission': commission, 'impact': impact}
        for date, row in orders.iterrows():
            stock = row['Symbol']
            order = row['Order']
            shares = row['Shares']
            stock_price = self.tracking['stock_prices'].loc[date, stock]
            self.__ExecuteOrder(date, stock, order, shares, stock_price, commission, impact)
            self.tracking['portfolio'] = self.tracking['stock_prices'] * self.tracking['trades'].cumsum()
            self.tracking['portfolio']['port_val'] = self.tracking['portfolio'].sum(axis=1)
            
        logger.info(f"Portfolio value computation completed, final value: ${self.tracking['portfolio']['port_val'].iloc[-1]:,.2f}")
        # CLose connections
        return self.tracking


    def compute_portvals_with_config(self, orders: pd.DataFrame, config: BacktestConfig) -> Dict[str, pd.DataFrame]:
        """
        Compute portfolio values using BacktestConfig parameters.
        This is an alternative to compute_portvals that uses the BacktestConfig class
        for more sophisticated backtesting features.

        Args:
            orders (pd.DataFrame): Trading orders DataFrame
            config (BacktestConfig): Backtesting configuration parameters

        Returns:
            Dict[str, pd.DataFrame]: Portfolio tracking information
        """
        return self.compute_portvals(
            orders=orders,
            startval=config.initial_capital,
            commission=config.commission,
            impact=config.get_impact()
        )

"""
if __name__ == "__main__":
    ms = MarketSim(connections, verbose = True)
    orders = pd.read_csv('orders/additional_orders/orders2.csv', index_col='Date', parse_dates=True, na_values=['nan']).sort_index()
    orders = pd.read_csv('orders/DTLearn.csv', index_col='Date', parse_dates=True, na_values=['nan']).sort_index()
    d = ms.compute_portvals(orders, 1000000, commission = 9.95, impact = 0.005)
    #  $1,010,884.73    
    # for i, k in d.items():
    #     print(f'\n\n{i}\n\n')
    #     print(k)
"""



