"""Database management module for backtesting system."""

import sqlite3
import logging
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DBManager:
    """Manages database connections and operations."""
    
    def __init__(self, connections: Dict[str, str], pool_size: int = 5) -> None:
        """
        Initialize DBManager with connection information.
        
        Args:
            connections (Dict[str, str]): Dictionary mapping database names to paths
            pool_size (int): Maximum number of connections per database
            
        Raises:
            TypeError: If connections is None
            ValueError: If connections is empty
        """
        if connections is None:
            raise TypeError("Connections dictionary cannot be None")
        if not connections:
            raise ValueError("Connections dictionary cannot be empty")
            
        self.connections = connections
        self.pool_size = pool_size
        required_connections = ['daily_db', 'intraday_db']
        self.pool: Dict[str, List[sqlite3.Connection]] = {
            db_name: [] for db_name in connections if db_name in required_connections
        }
        
        logger.debug(f"DBManager initialized with connections: {self.connections.keys()} "
                   f"and pool size: {self.pool_size}")

    def get_connection(self, db_name: str) -> sqlite3.Connection:
        """
        Get a database connection from the pool or create a new one.
        
        Args:
            db_name (str): Name of the database to connect to
            
        Returns:
            sqlite3.Connection: Database connection
            
        Raises:
            ValueError: If database name is not found
            RuntimeError: If pool is full
            sqlite3.Error: If connection fails
        """
        if db_name not in self.connections:
            logger.error(f"Database {db_name} not found in the connections")
            raise ValueError(f"Database {db_name} not found in the connections")

        # Try to get an existing connection from the pool
        if self.pool[db_name]:
            conn = self.pool[db_name].pop()
            logger.debug(f"Reusing existing connection for database: {db_name}")
            return conn

        # Create new connection if pool isn't full
        if len(self.pool[db_name]) < self.pool_size:
            conn = self._create_connection(db_name)
            logger.debug(f"Created new connection for database: {db_name}")
            return conn
            
        logger.error(f"Connection pool for database {db_name} is full")
        raise RuntimeError(f"Connection pool for database {db_name} is full")

    def return_connection(self, db_name: str, conn: sqlite3.Connection) -> None:
        """
        Return a connection to the pool.
        
        Args:
            db_name (str): Name of the database
            conn (sqlite3.Connection): Connection to return
        """
        if len(self.pool[db_name]) < self.pool_size:
            self.pool[db_name].append(conn)
            logger.debug(f"Connection returned to pool for database: {db_name}")
        else:
            logger.debug(f"Pool for database {db_name} is full; connection closed")
            conn.close()

    def _create_connection(self, db_name: str) -> sqlite3.Connection:
        """
        Create a new database connection.
        
        Args:
            db_name (str): Name of the database to connect to
            
        Returns:
            sqlite3.Connection: New database connection
            
        Raises:
            sqlite3.Error: If connection fails
        """
        try:
            return sqlite3.connect(self.connections[db_name])
        except sqlite3.Error as e:
            logger.error(f"SQLite error occurred while connecting to {db_name}: {str(e)}")
            raise

    def close_all(self) -> None:
        """Close all database connections in the pool."""
        for db_name, connections in self.pool.items():
            while connections:
                conn = connections.pop()
                conn.close()
                logger.debug(f"Closed connection to database: {db_name}")
        logger.info("All connections cleared")

    def __del__(self) -> None:
        """Clean up resources when object is destroyed."""
        logger.debug("DBManager object is being destroyed; closing all connections")
        try:
            self.close_all()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __enter__(self) -> 'DBManager':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Optional[type],
                exc_val: Optional[Exception],
                exc_tb: Optional[Any]) -> None:
        """
        Context manager exit.
        
        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        self.close_all()
        if exc_type is not None:
            logger.error(f"An error occurred: {str(exc_val)}")
            
    @contextmanager
    def connection(self, db_name: str) -> sqlite3.Connection:
        """
        Context manager for getting and returning database connections.
        
        Args:
            db_name (str): Name of the database to connect to
            
        Yields:
            sqlite3.Connection: Database connection
            
        Example:
            with db_manager.connection('daily_db') as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM stocks")
        """
        conn = self.get_connection(db_name)
        try:
            yield conn
        finally:
            self.return_connection(db_name, conn)