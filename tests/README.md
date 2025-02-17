# Backtesting System Test Suite

This directory contains the comprehensive test suite for the backtesting system. The tests are organized into unit tests and integration tests to ensure code quality and proper functionality.

## Test Structure

```
tests/
├── conftest.py              # Shared test fixtures and configuration
├── unit/                    # Unit tests for individual components
│   ├── test_dbm.py         # Database manager tests
│   ├── test_market_sim.py  # Market simulator tests
│   ├── test_portfolio_stats.py  # Portfolio statistics tests
│   └── test_strategies.py  # Trading strategy tests
├── integration/            # Integration tests
│   └── test_trading_workflow.py  # End-to-end workflow tests
└── README.md              # This file
```

## Running Tests

The test suite can be run using the provided `run_tests.py` script in the root directory:

```bash
# Run all tests with coverage
./run_tests.py --all

# Run only unit tests
./run_tests.py --unit

# Run only integration tests
./run_tests.py --integration

# Generate HTML coverage report
./run_tests.py --coverage --html
```

## Test Categories

### Unit Tests
- `test_dbm.py`: Tests the database connection pool management and SQL operations
- `test_market_sim.py`: Tests market simulation, order execution, and portfolio tracking
- `test_portfolio_stats.py`: Tests portfolio performance metrics calculations
- `test_strategies.py`: Tests trading strategy generation and evaluation

### Integration Tests
- `test_trading_workflow.py`: Tests the complete trading pipeline from data retrieval to performance analysis

## Test Coverage Requirements

The test suite is configured to maintain a minimum of 80% code coverage. Coverage reports are generated automatically when running tests with the `--coverage` flag.

## Fixtures

Common test fixtures are defined in `conftest.py`:
- `mock_db_connections`: Provides temporary database connections for testing
- `sample_stock_data`: Generates realistic SPY price data
- `sample_orders`: Creates sample trading orders
- `portfolio_data`: Provides sample portfolio values
- `risk_metrics`: Defines expected risk metric values for validation

## Best Practices

1. All tests use SPY ETF data instead of individual stocks for consistency
2. Each test function focuses on a single aspect or functionality
3. Use appropriate markers (`unit` or `integration`) for test categorization
4. Include both positive and negative test cases
5. Test edge cases and error conditions

## Adding New Tests

When adding new tests:

1. Place unit tests in the appropriate file under `unit/`
2. Place integration tests in `integration/`
3. Use the existing fixtures from `conftest.py` when possible
4. Add test markers:
   ```python
   @pytest.mark.unit  # For unit tests
   @pytest.mark.integration  # For integration tests
   ```
5. Follow the naming convention: `test_*` for test functions

## Test Configuration

The testing framework is configured in `pytest.ini`:
- Test discovery patterns
- Marker definitions
- Coverage settings
- Console output formatting
- Logging configuration

## Continuous Integration

The test suite is designed to be run in CI environments. Required coverage thresholds and test categories are enforced through pytest configuration.

## Troubleshooting

If tests fail:

1. Check the test logs for specific error messages
2. Verify database connections and fixtures
3. Ensure all dependencies are installed (`pytest`, `pytest-cov`)
4. Check coverage reports for untested code paths

## Dependencies

- pytest
- pytest-cov
- pandas
- numpy
- sqlite3

Install development dependencies:
```bash
pip install -e ".[dev]"