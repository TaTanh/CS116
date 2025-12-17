"""Utility to list and explore available parquet files in the dataset directory."""

from pathlib import Path
from typing import List, Dict, Union
import polars as pl

from .config import DATA_DIR


def list_parquet_files() -> List[Path]:
    """List all parquet files in the dataset directory.
    
    Returns:
        List of Path objects for .parquet files.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")
    
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    return sorted(parquet_files)


def inspect_parquet_file(file_path: Union[Path, str]) -> Dict:
    """Inspect a parquet file and return basic information.
    
    Args:
        file_path: Path to the parquet file.
        
    Returns:
        Dictionary with file info: name, shape, columns, schema.
    """
    df = pl.read_parquet(file_path)
    
    return {
        "file_name": Path(file_path).name,
        "shape": df.shape,
        "columns": df.columns,
        "schema": {col: str(dtype) for col, dtype in df.schema.items()},
        "null_counts": {col: df[col].null_count() for col in df.columns},
        "sample": df.head(3),
    }


def explore_dataset() -> None:
    """Print information about all parquet files in the dataset directory."""
    print(f"Dataset directory: {DATA_DIR}")
    print("=" * 70)
    
    parquet_files = list_parquet_files()
    
    if not parquet_files:
        print("No parquet files found!")
        return
    
    print(f"\nFound {len(parquet_files)} parquet file(s):\n")
    
    for file_path in parquet_files:
        print(f"\n{'=' * 70}")
        print(f"FILE: {file_path.name}")
        print("=" * 70)
        
        try:
            info = inspect_parquet_file(file_path)
            
            print(f"\nShape: {info['shape'][0]:,} rows × {info['shape'][1]} columns")
            print(f"\nColumns: {', '.join(info['columns'])}")
            
            print("\nSchema:")
            for col, dtype in info['schema'].items():
                null_count = info['null_counts'][col]
                null_pct = (null_count / info['shape'][0] * 100) if info['shape'][0] > 0 else 0
                print(f"  - {col:20s} {dtype:15s} (nulls: {null_count:,} = {null_pct:.1f}%)")
            
            print("\nSample data:")
            print(info['sample'])
            
        except Exception as e:
            print(f"Error reading file: {e}")
    
    print("\n" + "=" * 70)


def load_any_parquet(file_name: str) -> pl.LazyFrame:
    """Load any parquet file from the dataset directory by name.
    
    Args:
        file_name: Name of the parquet file (with or without .parquet extension).
        
    Returns:
        LazyFrame with the data.
    """
    if not file_name.endswith('.parquet'):
        file_name += '.parquet'
    
    file_path = DATA_DIR / file_name
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Available files: {[f.name for f in list_parquet_files()]}"
        )
    
    return pl.scan_parquet(file_path)
