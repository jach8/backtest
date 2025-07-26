import os
import logging
from typing import Dict
from dbm import DBManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_connections(connections: Dict[str, str]) -> bool:
    """
    Validates the connections dictionary to ensure it contains all required database configurations.
    
    Args:
        connections (Dict[str, str]): A dictionary mapping database names to their file paths.
            Required structure:
            {
                'daily_db': 'path/to/daily/database.db',
                'intra_day_db': 'path/to/intraday/database.db'
            }
    
    Returns:
        bool: True if the connections dictionary is valid, False otherwise.
    
    Raises:
        ValueError: If the connections dictionary is invalid or missing required databases.
    """
    if not isinstance(connections, dict):
        raise ValueError("Connections must be a dictionary")
    
    # Check for required database configurations
    required_dbs = {'daily_db', 'intraday_db'}
    missing_dbs = required_dbs - set(connections.keys())
    
    if missing_dbs:
        raise ValueError(f"Missing required database configurations: {missing_dbs}")
    
    # Validate each database path
    for db_name, db_path in connections.items():
        if not isinstance(db_path, str):
            raise ValueError(f"Database path for {db_name} must be a string")
        
        # Check if the path exists
        if not os.path.exists(db_path):
            raise ValueError(f"Database file not found: {db_path}")
        
        # Check if the path is a file
        if not os.path.isfile(db_path):
            raise ValueError(f"Database path is not a file: {db_path}")
        
        # Check if the file is readable
        if not os.access(db_path, os.R_OK):
            raise ValueError(f"Database file is not readable: {db_path}")
    
    logger.info("Connections validation successful")
    return True

def initialize_price_data(connections: Dict[str, str], pool_size: int = 5) -> DBManager:
    """
    Initializes the price data system with the provided database connections.
    
    Args:
        connections (Dict[str, str]): A dictionary mapping database names to their file paths.
        pool_size (int, optional): The size of the connection pool. Defaults to 5.
    
    Returns:
        DBManager: An initialized database manager instance.
    
    Raises:
        ValueError: If the connections configuration is invalid.
    """
    try:
        # Validate the connections dictionary
        if validate_connections(connections):
            # Create and return a new DBManager instance
            db_manager = DBManager(connections, pool_size=pool_size)
            logger.info("Successfully initialized price data system")
            return db_manager
    except ValueError as e:
        logger.error("Failed to initialize price data system: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error during initialization: %s", str(e))
        raise


"""
Example Usage: 
if __name__ == "__main__":    
    try:
        test_connections = {
            'daily_db': 'bin/data/stocks.db',
            'intra_day_db': 'bin/data/stocks_intraday.db'
        }
        
        db_manager = initialize_price_data(test_connections)
        
        # Test a connection to each database
        for db_name in ['daily_db', 'intra_day_db']:
            with db_manager.get_connection(db_name) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                logger.info(f"Tables in {db_name}: {tables}")
                
    except ValueError as e:
        logger.error("Validation error: %s", str(e))
    except Exception as e:
        logger.error("Error: %s", str(e))
        
"""