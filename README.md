# Market Simulation and Strategy Evaluation

## Table of Contents
- [Market Simulation and Strategy Evaluation](#market-simulation-and-strategy-evaluation)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
    - [Strategy Generation:](#strategy-generation)
    - [Market Simulation:](#market-simulation)
    - [Performance Evaluation:](#performance-evaluation)
    - [Database Integration:](#database-integration)
    - [Extensibility:](#extensibility)
  - [Prerequisites](#prerequisites)
  - [Project Structure](#project-structure)
  - [Configuration](#configuration)
    - [Database Setup:](#database-setup)
    - [Configuration File:](#configuration-file)
  - [Usage](#usage)
    - [Initialize the Policy Class](#initialize-the-policy-class)
    - [Generate Orders for a Strategy](#generate-orders-for-a-strategy)
      - [Buy-and-Hold Strategy](#buy-and-hold-strategy)
      - [Optimal Policy](#optimal-policy)
    - [Evaluate a Strategy](#evaluate-a-strategy)
    - [Evaluate Multiple Strategies](#evaluate-multiple-strategies)
  - [Example Output](#example-output)
    - [Strategy Performance Metrics](#strategy-performance-metrics)
  - [Extending the Framework](#extending-the-framework)
    - [Add New Strategies:](#add-new-strategies)
    - [Support for Multi-Stock Portfolios:](#support-for-multi-stock-portfolios)
    - [Advanced Order Types:](#advanced-order-types)
  - [License](#license)
  - [Contributing](#contributing)
  - [Contact](#contact)

## Overview
This project provides a Python-based framework for simulating market trading strategies and evaluating their performance. It allows users to:
- Generate trading orders for different strategies (e.g., Buy-and-Hold, Optimal Policy).
- Simulate the execution of these strategies using historical stock price data.
- Evaluate the performance of the strategies with detailed metrics such as cumulative return, Sharpe ratio, and daily return statistics.

The project is designed to be modular and extensible, making it suitable for both educational purposes and advanced trading strategy research.

## Features
### Strategy Generation:
- **Buy-and-Hold**: A simple baseline strategy that buys at the start and sells at the end.
- **Optimal Policy**: A theoretical strategy that maximizes returns based on look-ahead price data (for benchmarking).

### Market Simulation:
- Simulates trading strategies using historical stock price data.
- Accounts for trading costs, including commission and market impact.

### Performance Evaluation:
- Computes key metrics such as cumulative return, average daily return, standard deviation of returns, and Sharpe ratio.
- Compares multiple strategies side by side.

### Database Integration:
- Fetches historical stock price data from a database (via DBManager).
- Supports both daily and intra-day price data.

### Extensibility:
- Easily add new trading strategies.
- Modular design for integrating with other systems (e.g., custom databases, advanced simulators).

## Prerequisites
Before using this project, ensure you have the following:

- **Python 3.8+**

- **Required Libraries**:
  - pandas
  - numpy
  - tqdm
  - logging

- **Database**:
  - A configured database with historical stock price data.
  - A `config.env` file specifying the database connections (see [Configuration](#configuration)).

Install the required Python libraries using:
```bash
pip install pandas numpy tqdm
```

## Project Structure
```
market_simulation/
│
├── dbm.py               # Database manager (DBManager) for querying stock prices
├── msim.py              # Market simulation (MarketSim) for executing orders
├── policy.py            # Policy class for generating and evaluating strategies
├── config.env           # Configuration file for database connections
├── bin/
│   ├── orders/          # Sample order files (e.g., DTLearn.csv, RTLearn.csv)
│   └── main.py          # Utility scripts (optional)
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```

## Configuration
### Database Setup:
Ensure your database contains historical stock price data.

The database should have tables for each stock symbol (e.g., AAPL, GOOG) with columns:
- `date`: The date of the price (or datetime for intra-day data).
- `close`: The closing price (or intra-day price).

### Configuration File:
Create a `config.env` file in the project root with your database connections:
```
daily_price_db=path/to/daily_price_db.sqlite
intraday_price_db=path/to/intraday_price_db.sqlite
```
Replace `path/to/` with the actual file paths to your database files.

## Usage
### Initialize the Policy Class
```python
from policy import Policy

# Load database connections
connections = {}
with open('config.env') as f:
    for line in f:
        name, path = line.strip().split('=')
        connections[name.lower()] = path

# Initialize Policy
policy = Policy(connections)
```

### Generate Orders for a Strategy
#### Buy-and-Hold Strategy
```python
# Load sample orders
orders = pd.read_csv('bin/orders/DTLearn.csv', index_col='Date', parse_dates=True, na_values=['nan']).sort_index()

# Initialize policy with orders
policy = Policy(connections, orders)

# Generate Buy-and-Hold orders
sv = 100_000  # Starting value
bh_orders = policy.buy_and_hold(sv)
print(bh_orders)
```

#### Optimal Policy
```python
# Generate Optimal Policy orders
optimal_orders = policy.optimal_policy(sv)
print(optimal_orders)
```

### Evaluate a Strategy
```python
# Evaluate Buy-and-Hold strategy
bh_performance = policy.evaluate_policy(bh_orders, sv=sv, commission=9.95, impact=0.005)
print(bh_performance)

# Compute performance metrics
metrics = policy._qs(bh_performance)
print(metrics)
```

### Evaluate Multiple Strategies
```python
# Load multiple order files
orders1 = pd.read_csv('bin/orders/DTLearn.csv', index_col='Date', parse_dates=True, na_values=['nan']).sort_index()
orders2 = pd.read_csv('bin/orders/RTLearn.csv', index_col='Date', parse_dates=True, na_values=['nan']).sort_index()

# Evaluate multiple strategies
results = policy.eval_multiple_orders(orders=[orders1, orders2], names=['DTLearn', 'RTLearn'], sv=100_000)
print(results)
```

## Example Output
### Strategy Performance Metrics
| Strategy      | Stock | Days | StartDate   | EndDate     | StartBalance | EndBalance | CumReturn% | AvgDailyRet% | StdDevRet | Sharpe |
|---------------|-------|------|-------------|-------------|--------------|------------|------------|--------------|-----------|--------|
| Buy and Hold  | SPY   | 252  | 2023-01-01  | 2023-12-31  | 100,000.00   | 110,500.00 | 10.50      | 0.04         | 0.12      | 1.20   |
| Optimal Policy| SPY   | 252  | 2023-01-01  | 2023-12-31  | 100,000.00   | 125,000.00 | 25.00      | 0.10         | 0.15      | 1.50   |
| DTLearn       | SPY   | 252  | 2023-01-01  | 2023-12-31  | 100,000.00   | 115,000.00 | 15.00      | 0.06         | 0.13      | 1.30   |

## Extending the Framework
### Add New Strategies:
- Create new methods in the `Policy` class (e.g., `mean_reversion_strategy`).
- Return orders in the standard format: Date, Symbol, Order, Shares.

### Support for Multi-Stock Portfolios:
- Extend the `Policy` class to handle multiple stocks simultaneously.

### Advanced Order Types:
- Add support for limit orders, stop-loss orders, etc., by extending the `MarketSim` class.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Contact
For questions or feedback, please open an issue or send a direct message.


