"""Data schemas and loading utilities for recommender system."""

from datetime import date, datetime
from pathlib import Path
from typing import Optional, Dict, Union

import polars as pl

from .config import DATA_DIR, TRANSACTIONS_PATH, ITEMS_PATH, USERS_PATH


class DataSchema:
    """Data schemas for the recommender system."""

    @staticmethod
    def transactions_schema() -> Dict[str, pl.DataType]:
        """Schema for transactions table.
        
        Returns:
            Dictionary mapping column names to polars data types.
        """
        return {
            "customer_id": pl.Int64,
            "item_id": pl.Int64,
            "created_at": pl.Datetime,
            "order_id": pl.Int64,
        }

    @staticmethod
    def items_schema() -> Dict[str, pl.DataType]:
        """Schema for items table.
        
        Returns:
            Dictionary mapping column names to polars data types.
        """
        return {
            "item_id": pl.Int64,
            "brand": pl.Utf8,
            "age_group": pl.Utf8,
            "category": pl.Utf8,
        }

    @staticmethod
    def users_schema() -> Dict[str, pl.DataType]:
        """Schema for users table.
        
        Returns:
            Dictionary mapping column names to polars data types.
        """
        return {
            "customer_id": pl.Int64,
            "date_of_birth": pl.Date,
        }


def load_transactions(path: Optional[Union[Path, str]] = None) -> pl.LazyFrame:
    """Load transactions data as a LazyFrame.
    
    Args:
        path: Path or glob pattern to the transactions parquet file(s). 
              If None, uses default pattern from config.
        
    Returns:
        LazyFrame with transactions data (from all matching files).
    """
    if path is None:
        path = TRANSACTIONS_PATH
    # scan_parquet handles glob patterns automatically
    return pl.scan_parquet(path)


def load_items(path: Optional[Union[Path, str]] = None) -> pl.LazyFrame:
    """Load items data as a LazyFrame.
    
    Args:
        path: Path or glob pattern to the items parquet file(s).
              If None, uses default pattern from config.
        
    Returns:
        LazyFrame with items data (from all matching files).
    """
    if path is None:
        path = ITEMS_PATH
    # scan_parquet handles glob patterns automatically
    return pl.scan_parquet(path)


def load_users(path: Optional[Union[Path, str]] = None) -> pl.LazyFrame:
    """Load users data as a LazyFrame.
    
    Args:
        path: Path or glob pattern to the users parquet file(s).
              If None, uses default path from config.
        
    Returns:
        LazyFrame with users data.
    """
    if path is None:
        path = USERS_PATH
    return pl.scan_parquet(path)


def validate_transactions(df: pl.LazyFrame) -> None:
    """Validate transactions data schema.
    
    Args:
        df: Transactions LazyFrame to validate.
        
    Raises:
        ValueError: If required columns are missing.
    """
    required_cols = {"customer_id", "item_id", "created_at"}
    actual_cols = set(df.columns)
    
    missing = required_cols - actual_cols
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_items(df: pl.LazyFrame) -> None:
    """Validate items data schema.
    
    Args:
        df: Items LazyFrame to validate.
        
    Raises:
        ValueError: If required columns are missing.
    """
    required_cols = {"item_id", "brand", "age_group", "category"}
    actual_cols = set(df.columns)
    
    missing = required_cols - actual_cols
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_users(df: pl.LazyFrame) -> None:
    """Validate users data schema.
    
    Args:
        df: Users LazyFrame to validate.
        
    Raises:
        ValueError: If required columns are missing.
    """
    required_cols = {"customer_id", "date_of_birth"}
    actual_cols = set(df.columns)
    
    missing = required_cols - actual_cols
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
