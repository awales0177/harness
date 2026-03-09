"""Write operations for ETL pipeline."""

from .write_table import write_table
from .write_manifest import write_manifest

__all__ = ["write_table", "write_manifest"]
