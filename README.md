# Backtest: A Trading Strategy Backtesting Framework

## Table of Contents
- [Backtest: A Trading Strategy Backtesting Framework](#backtest-a-trading-strategy-backtesting-framework)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
    - [Option 1: Install from source](#option-1-install-from-source)
  - [Prerequisites](#prerequisites)
    - [Python Version](#python-version)
    - [Required Dependencies (installed automatically)](#required-dependencies-installed-automatically)
    - [Database Requirements](#database-requirements)
  - [Project Structure](#project-structure)
  - [Configuration](#configuration)
    - [Database Setup](#database-setup)
    - [Configuration File](#configuration-file)
  - [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Example Output](#example-output)
    - [Strategy Performance Metrics](#strategy-performance-metrics)
  - [Development](#development)
  - [Extending the Framework](#extending-the-framework)
    - [Add New Strategies](#add-new-strategies)
    - [Support for Multi-Stock Portfolios](#support-for-multi-stock-portfolios)
    - [Advanced Order Types](#advanced-order-types)
  - [License](#license)
  - [Contributing](#contributing)

## Overview
Backtest is a Python-based framework for simulating market trading strategies and evaluating their performance. It allows users to:
- Generate trading orders for different strategies (e.g., Buy-and-Hold, Optimal Policy)
- Simulate the execution of these strategies using historical stock price data
- Evaluate the performance of strategies with detailed metrics such as cumulative return, Sharpe ratio, and daily return statistics

The project is designed to be modular and extensible, making it suitable for both educational purposes and advanced trading strategy research.

## Features
- **Strategy Generation:**
  - Buy-and-Hold: A simple baseline strategy that buys at the start and sells at the end
  - Optimal Policy: A theoretical strategy that maximizes returns based on look-ahead price data (for benchmarking)

- **Market Simulation:**
  - Simulates trading strategies using historical stock price data
  - Accounts for trading costs, including commission and market impact

- **Performance Evaluation:**
  - Computes key metrics such as cumulative return, average daily return, standard deviation of returns, and Sharpe ratio
  - Compares multiple strategies side by side

- **Database Integration:**
  - Fetches historical stock price data from a database (via DBManager)
  - Supports both daily and intra-day price data

## Installation

### Option 1: Install from source
1. Clone the repository:
```bash
git clone https://github.com/jach8/backtest.git
cd backtest
```

2. Create a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
# Install required packages
pip install -r requirements.txt

# For development setup (includes testing tools)
pip install -e .[dev]
```

## Prerequisites
### Python Version
- Python 3.8 or higher

### Required Dependencies (installed automatically)
Core dependencies:
- numpy: Numerical computations
- pandas: Data manipulation and analysis
- scipy: Scientific computing
- matplotlib: Data visualization
- tqdm: Progress bars
- typing: Type hints
- datetime: Date and time utilities

Development dependencies (optional):
- pytest: Testing framework
- pytest-cov: Code coverage reporting
- black: Code formatting
- isort: Import sorting
- mypy: Static type checking
- pylint: Code analysis

### Database Requirements
- A configured database with historical stock price data
- A `config.env` file specifying database connections (see [Configuration](#configuration))

## Project Structure
```
backtest/
├── __init__.py
├── dbm.py              # Database management for querying stock prices
├── simulator.py        # Market simulation for executing orders
├── strategies.py       # Trading strategy implementations
├── PortfolioStats.py   # Portfolio statistics and analysis
├── main.py            # Command-line interface
├── tests/             # Test suite
│   ├── unit/         # Unit tests
│   └── integration/  # Integration tests
├── setup.py           # Package configuration
└── README.md          # Project documentation
```

## Configuration
### Database Setup
Ensure your database contains historical stock price data with tables for each stock symbol (e.g., AAPL, GOOG) containing:
- `date`: The date of the price (or datetime for intra-day data)
- `close`: The closing price (or intra-day price)

### Configuration File
Create a `config.env` file in your project directory:
```
daily_price_db=path/to/daily_price_db.sqlite
intraday_price_db=path/to/intraday_price_db.sqlite
```

## Usage
```python
from backtest import Policy

# Initialize with database connections
connections = {   
    "daily_db": "data/stocks.db", 
    "intraday_db": "data/stocks_intraday.db",
    "ticker_path": "data/tickers.json",
    "stock_names": "data/stock_names.db"
}

# Create policy instance
policy = Policy(connections)

# Generate and evaluate Buy-and-Hold strategy
sv = 100_000  # Starting value
bh_orders = policy.buy_and_hold(sv)
bh_performance = policy.evaluate_policy(
    bh_orders, 
    sv=sv, 
    commission=9.95, 
    impact=0.005
)
metrics = policy._qs(bh_performance)
print(metrics)

# Evaluate multiple strategies
results = policy.eval_multiple_orders(
    orders=[orders1, orders2], 
    names=['Strategy1', 'Strategy2'], 
    sv=100_000
)
```

## Command Line Interface
The package installs a command-line tool for managing price data:

```bash
# Basic usage
backtest-price-data --help

# Update price data
backtest-price-data update --database daily_db --symbols AAPL GOOG
```

## Example Output
### Strategy Performance Metrics
```
| Strategy      | Stock | Days | StartDate  | EndDate    | StartBal  | EndBal    | CumRet% | DailyRet% | StdDev | Sharpe |
|--------------|-------|------|------------|------------|-----------|-----------|---------|-----------|---------|--------|
| Buy and Hold | SPY   | 252  | 2023-01-01 | 2023-12-31 | 100,000   | 110,500   | 10.50   | 0.04      | 0.12    | 1.20   |
| Optimal      | SPY   | 252  | 2023-01-01 | 2023-12-31 | 100,000   | 125,000   | 25.00   | 0.10      | 0.15    | 1.50   |
```

## Development
Install development dependencies:
```bash
pip install backtest[dev]
```

Development tools included:
- pytest: Testing framework
- pytest-cov: Code coverage reporting
- black: Code formatting
- isort: Import sorting
- mypy: Static type checking
- pylint: Code analysis

Run tests:
```bash
pytest
pytest --cov=backtest  # With coverage
```

## Extending the Framework
### Add New Strategies
1. Add strategy implementation to `strategies.py`
2. Return orders in standard format: Date, Symbol, Order, Shares
3. Add evaluation method to Policy class

### Support for Multi-Stock Portfolios
Extend the Policy class to handle multiple stocks simultaneously

### Advanced Order Types
Add support for limit orders, stop-loss orders, etc., by extending the MarketSim class

## License
MIT License. See the LICENSE file for details.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Submit a pull request

For questions or feedback, please open an issue on GitHub.
