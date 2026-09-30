"""
Database Connection Manager.
Handles PostgreSQL database connections with automatic detection, transparent logging,
and local relational warehouse fallback (olist_warehouse.db).
"""

import os
from sqlalchemy import create_engine, text
from src.utils.config import DATABASE_URL, SQLITE_DB_PATH, SQLITE_DB_URI
from src.utils.logger import get_logger

logger = get_logger("DatabaseManager")


def get_db_engine():
    """
    Returns an active database engine.
    Tests PostgreSQL connection first; if unavailable, uses the local SQLite relational warehouse.
    Returns (engine, backend_name).
    """
    # Attempt PostgreSQL
    try:
        engine = create_engine(DATABASE_URL, echo=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Connected to PostgreSQL backend: {engine.url.render_as_string(hide_password=True)}")
        return engine, "postgresql"
    except Exception as e:
        logger.info(f"PostgreSQL backend not reachable ({type(e).__name__}). Using local relational warehouse: {SQLITE_DB_PATH.name}")
        engine = create_engine(SQLITE_DB_URI, echo=False)
        return engine, "sqlite"



def get_connection():
    """Returns a raw connection object from the active engine."""
    engine, backend = get_db_engine()
    return engine.connect(), backend

