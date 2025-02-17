"""Unit tests for the DBManager class."""

import pytest
import sqlite3
from backtest.dbm import DBManager

pytestmark = pytest.mark.unit  # Mark all tests in this file as unit tests

def test_dbmanager_initialization(mock_db_connections):
    """Test DBManager initialization with valid connections."""
    pool_size = 3
    db_manager = DBManager(mock_db_connections, pool_size=pool_size)
    
    assert db_manager.pool_size == pool_size
    assert set(db_manager.connections.keys()) == set(mock_db_connections.keys())
    assert all(len(connections) == 0 for connections in db_manager.pool.values())

def test_dbmanager_invalid_connections():
    """Test DBManager initialization with invalid connections."""
    with pytest.raises(TypeError):
        DBManager(None)
    
    with pytest.raises(ValueError):
        DBManager({})

def test_connection_pool_management(mock_db_connections):
    """Test connection pool creation and management."""
    pool_size = 2
    db_manager = DBManager(mock_db_connections, pool_size=pool_size)
    connections = []
    
    # Get first connection
    with db_manager.get_connection('daily_db') as conn1:
        assert isinstance(conn1, sqlite3.Connection)
        connections.append(conn1)
        assert len(db_manager.pool['daily_db']) == 0
    
    # Connection should be returned to pool
    assert len(db_manager.pool['daily_db']) == 1
    
    # Get second connection
    with db_manager.get_connection('daily_db') as conn2:
        assert isinstance(conn2, sqlite3.Connection)
        connections.append(conn2)
        # First connection should be reused
        assert id(conn1) == id(conn2)

def test_connection_pool_overflow(mock_db_connections):
    """Test behavior when pool size is exceeded."""
    pool_size = 1
    db_manager = DBManager(mock_db_connections, pool_size=pool_size)
    
    # Fill up the pool
    with db_manager.get_connection('daily_db') as conn1:
        assert isinstance(conn1, sqlite3.Connection)
        
        # Try to get another connection when pool is full
        with pytest.raises(RuntimeError, match="Connection pool for database daily_db is full"):
            with db_manager.get_connection('daily_db'):
                pass

def test_invalid_database_name(mock_db_connections):
    """Test error handling for invalid database names."""
    db_manager = DBManager(mock_db_connections)
    
    with pytest.raises(ValueError, match="Database nonexistent_db not found in the connections"):
        with db_manager.get_connection('nonexistent_db'):
            pass

def test_context_manager(mock_db_connections):
    """Test DBManager as a context manager."""
    with DBManager(mock_db_connections) as db_manager:
        with db_manager.get_connection('daily_db') as conn:
            assert isinstance(conn, sqlite3.Connection)
            # Verify connection is active
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            assert cursor.fetchone() == (1,)

def test_close_all(mock_db_connections):
    """Test explicit closing of all connections."""
    db_manager = DBManager(mock_db_connections)
    
    # Create some connections
    with db_manager.get_connection('daily_db'):
        pass
    with db_manager.get_connection('intra_day_db'):
        pass
    
    # Close all connections
    db_manager.close_all()
    
    # Verify all pools are empty
    assert all(len(connections) == 0 for connections in db_manager.pool.values())

def test_connection_reuse(mock_db_connections):
    """Test that connections are reused from the pool."""
    db_manager = DBManager(mock_db_connections, pool_size=1)
    
    # First use creates a connection
    with db_manager.get_connection('daily_db') as conn1:
        conn1_id = id(conn1)
    
    # Second use should reuse the same connection
    with db_manager.get_connection('daily_db') as conn2:
        conn2_id = id(conn2)
    
    assert conn1_id == conn2_id

def test_concurrent_database_access(mock_db_connections):
    """Test accessing different databases concurrently."""
    db_manager = DBManager(mock_db_connections)
    
    with db_manager.get_connection('daily_db') as daily_conn:
        with db_manager.get_connection('intra_day_db') as intra_conn:
            assert isinstance(daily_conn, sqlite3.Connection)
            assert isinstance(intra_conn, sqlite3.Connection)
            assert daily_conn != intra_conn

def test_connection_error_handling(mock_db_connections, monkeypatch):
    """Test error handling for connection failures."""
    def mock_connect(*args, **kwargs):
        raise sqlite3.Error("Connection failed")
    
    monkeypatch.setattr(sqlite3, 'connect', mock_connect)
    db_manager = DBManager(mock_db_connections)
    
    with pytest.raises(sqlite3.Error, match="Connection failed"):
        with db_manager.get_connection('daily_db'):
            pass