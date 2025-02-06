import sqlite3
from contextlib import contextmanager
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DBManager:
    """
    A database connection pooling manager for SQLite databases.

    This class manages a pool of database connections, allowing for efficient reuse of connections.
    It also provides context manager support for handling connections safely.

    Args:
        connections (Dict[str, str]): A dictionary of database names mapped to their file paths.
            Example:
                connections = {
                    'price_db': 'path_to_price_db',
                    'user_db': 'path_to_user_db'
                }
        pool_size (int): The maximum number of connections to maintain in the pool for each database.
            Default is 5.
    """

    def __init__(self, connections: Dict[str, str], pool_size: int = 5) -> None:
        self.connections: Dict[str, str] = connections
        self.pool_size: int = pool_size
        self.pool: Dict[str, list[sqlite3.Connection]] = {db_name: [] for db_name in connections}
        logger.info("DBManager initialized with connections: %s and pool size: %d", self.connections.keys(), pool_size)

    def _get_connection_from_pool(self, db_name: str) -> sqlite3.Connection:
        """
        Retrieves an available connection from the pool or creates a new one if the pool is not full.

        Args:
            db_name (str): The name of the database.

        Returns:
            sqlite3.Connection: An available connection from the pool.

        Raises:
            ValueError: If the database name is not found in the connections dictionary.
        """
        if db_name not in self.connections:
            logger.error("Database %s not found in the connections", db_name)
            raise ValueError(f"Database {db_name} not found in the connections")

        # Check if there is an available connection in the pool
        if self.pool[db_name]:
            conn = self.pool[db_name].pop(0)
            logger.info("Reusing existing connection for database: %s", db_name)
            return conn

        # If the pool is not full, create a new connection
        if len(self.pool[db_name]) < self.pool_size:
            conn = sqlite3.connect(self.connections[db_name])
            logger.info("Created new connection for database: %s", db_name)
            return conn

        # If the pool is full, wait for a connection to become available (not implemented here)
        logger.error("Connection pool for database %s is full", db_name)
        raise RuntimeError(f"Connection pool for database {db_name} is full")

    def _return_connection_to_pool(self, db_name: str, conn: sqlite3.Connection) -> None:
        """
        Returns a connection to the pool for reuse.

        Args:
            db_name (str): The name of the database.
            conn (sqlite3.Connection): The connection to return to the pool.
        """
        if len(self.pool[db_name]) < self.pool_size:
            self.pool[db_name].append(conn)
            logger.info("Connection returned to pool for database: %s", db_name)
        else:
            conn.close()
            logger.info("Pool for database %s is full; connection closed", db_name)

    @contextmanager
    def get_connection(self, db_name: str) -> Any:
        """
        Context manager for retrieving a database connection from the pool.

        This method provides a connection to the specified database and ensures it is returned
        to the pool after use.

        Args:
            db_name (str): The name of the database to connect to.

        Yields:
            sqlite3.Connection: A connection object for the specified database.

        Raises:
            ValueError: If the specified database name is not found in the connections dictionary.
        """
        conn = None
        try:
            conn = self._get_connection_from_pool(db_name)
            yield conn
        except sqlite3.Error as e:
            logger.error("SQLite error occurred while connecting to %s: %s", db_name, e)
            raise
        finally:
            if conn:
                self._return_connection_to_pool(db_name, conn)

    def close_all(self) -> None:
        """
        Closes all connections in the pool and clears the connections dictionary.

        This method should be called explicitly or is automatically invoked when the object
        is destroyed or used as a context manager.
        """
        for db_name, connections in self.pool.items():
            for conn in connections:
                try:
                    conn.close()
                    logger.info("Closed connection to database: %s", db_name)
                except sqlite3.Error as e:
                    logger.error("Error closing connection to %s: %s", db_name, e)
            connections.clear()
        self.pool.clear()
        logger.info("All connections cleared")

    def __del__(self) -> None:
        """
        Destructor to ensure all connections are closed when the object is garbage collected.
        """
        logger.info("DBManager object is being destroyed, closing all connections")
        self.close_all()

    def __enter__(self) -> 'DBManager':
        """
        Enables the use of the DBManager as a context manager.

        Returns:
            DBManager: The current instance of DBManager.
        """
        logger.info("Entering DBManager context")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """
        Exits the DBManager context, ensuring all connections are closed.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
            exc_val: The instance of the exception that caused the context to be exited.
            exc_tb: A traceback object encoding the stack trace.

        Returns:
            bool: False, indicating that any exception should be re-raised.
        """
        logger.info("Exiting DBManager context, closing all connections")
        self.close_all()
        return False


if __name__ == "__main__":
    # Example: Using DBManager with connection pooling
    connections = {}
    pre = '../'
    with open('config.env') as f:
        for line in f:
            name, path = line.strip().split('=')
            connections[name.lower()] = pre + path
            
    
    db_manager = DBManager(connections, pool_size=3)

    try:
        # Reusing connections from the pool
        for _ in range(5):
            with db_manager.get_connection('daily_price_db') as conn:
                cursor = conn.cursor()
                logger.info("Executing query on price_db: SELECT * FROM aapl")
                cursor.execute("SELECT * FROM aapl")
                print(len(cursor.fetchall()))

            with db_manager.get_connection('intra_day_db') as conn:
                cursor = conn.cursor()
                logger.info("Executing query on user_db: select * from tem")
                cursor.execute("SELECT * from tem")
                print(len(cursor.fetchall()))
    except Exception as e:
        logger.error("An error occurred: %s", e)
    finally:
        db_manager.close_all()