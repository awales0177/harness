from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, InitVar
from datetime import datetime
from typing import Any, Dict, Generator, List, Literal, Optional, TYPE_CHECKING
import json
import re
import uuid

try:
    from dataframe_viewer import displayDF
except ImportError:
    displayDF = None

from pretty_print_rich import (
    pretty_print_etl_context,
    pretty_print_etl_context_init,
    pretty_print_source_info,
    pretty_print_submit,
    pretty_print_table,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STATUSES = frozenset({"passed", "failed", "warning"})
VALID_ORIGIN_TYPES = frozenset({"source", "iso", "language", "regex", "metadata", "cleaned", "date"})
VALID_STAGES = frozenset({"bronze", "silver", "gold"})
DELIVERED_VERSION_PATTERN = re.compile(r"^v\d+$")

OriginType = Literal["source", "iso", "language", "regex", "metadata", "cleaned", "date"]
Stage = Literal["bronze", "silver", "gold"]
Status = Literal["passed", "failed", "warning"]


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass
class DataValidationResult:
    """A single validation rule outcome."""

    rule_id: str
    passed_or_failed: Status
    message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.passed_or_failed not in VALID_STATUSES:
            raise ValueError(
                f"passed_or_failed must be one of {VALID_STATUSES!r}, "
                f"got {self.passed_or_failed!r}"
            )


@dataclass
class DataQuality:
    """Data quality metrics for a table."""

    null_value_count: Optional[int] = None
    empty_row_count: Optional[int] = None


@dataclass
class Schema:
    """Describes a single column in a table."""

    column_name: str
    origin_type: OriginType
    function_created_by: Optional[str] = None
    data_type: Optional[str] = None

    def __post_init__(self) -> None:
        if self.origin_type not in VALID_ORIGIN_TYPES:
            raise ValueError(
                f"origin_type must be one of {VALID_ORIGIN_TYPES!r}, "
                f"got {self.origin_type!r}"
            )


@dataclass
class FunctionExecution:
    """Records a single function run with timing metadata."""

    run_id: str
    function_name: str
    run_order: int
    function_engines: Optional[List[Dict[str, str]]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.start_time is None:
            self.start_time = datetime.now()

    @property
    def duration_seconds(self) -> Optional[float]:
        """Elapsed seconds, or None if the run hasn't finished."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


# ---------------------------------------------------------------------------
# Source extraction (pure function — no coupling to TableContext)
# ---------------------------------------------------------------------------

@dataclass
class SourceInfo:
    """Extracted metadata from a source DataFrame."""

    schema: List[Schema]
    num_of_columns: int
    num_of_rows: int
    data_quality: DataQuality


def extract_source_info(df: "DataFrame") -> SourceInfo:
    """Pull schema, row/column counts, and null metrics from a PySpark DataFrame.

    Performs a single Spark action (one pass over the data).
    """
    from pyspark.sql import functions as F

    schema = [
        Schema(column_name=f.name, origin_type="source", data_type=str(f.dataType))
        for f in df.schema.fields
    ]
    num_of_columns = len(df.columns)

    agg_exprs = [F.count("*").alias("row_count")]

    if df.columns:
        agg_exprs += [
            F.sum(F.col(c).isNull().cast("long")).alias(f"null_{i}")
            for i, c in enumerate(df.columns)
        ]
        all_null_expr = F.least(*[F.col(c).isNull().cast("long") for c in df.columns])
        agg_exprs.append(F.sum(all_null_expr).alias("empty_row_count"))

    row = df.agg(*agg_exprs).collect()[0]

    num_of_rows = row["row_count"]

    if df.columns:
        null_value_count = sum(row[f"null_{i}"] for i in range(num_of_columns))
        empty_row_count = int(row["empty_row_count"] or 0)
    else:
        null_value_count = 0
        empty_row_count = 0

    return SourceInfo(
        schema=schema,
        num_of_columns=num_of_columns,
        num_of_rows=num_of_rows,
        data_quality=DataQuality(
            null_value_count=null_value_count,
            empty_row_count=empty_row_count,
        ),
    )


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def update_schema_origin(
    schema: List[Schema],
    column_names: List[str],
    origin_type: OriginType,
) -> None:
    """Set *origin_type* on matching Schema entries; append new ones if absent.

    Mutates *schema* in-place.
    """
    index = {s.column_name: s for s in schema}
    for name in column_names:
        if name in index:
            index[name].origin_type = origin_type
        else:
            schema.append(Schema(column_name=name, origin_type=origin_type))


# ---------------------------------------------------------------------------
# Operation namespaces
# ---------------------------------------------------------------------------

class ReadOperations:
    """Read operations available on a TableContext."""

    def __init__(self, table: TableContext) -> None:
        self._table = table

    def load_data(self):
        from read.load_data import load_data

        engines = [{"engine": "pandas", "version": "2.0.0"}]
        with self._table.function_tracker("load_data", function_engines=engines):
            return load_data()


class TransformOperations:
    """Transform operations available on a TableContext."""

    def __init__(self, table: TableContext) -> None:
        self._table = table

    def _run(
        self,
        function_name: str,
        import_path: str,
        data,
        engines: List[Dict[str, str]],
        mark_columns: Optional[List[str]],
        origin_type: OriginType,
    ):
        """Generic helper: import *import_path*, run it, update schema, return result."""
        module_path, fn_name = import_path.rsplit(".", 1)
        import importlib
        fn = getattr(importlib.import_module(module_path), fn_name)
        with self._table.function_tracker(function_name, function_engines=engines):
            result = fn(data)
        if mark_columns:
            update_schema_origin(self._table.schema, mark_columns, origin_type)
        return result

    _SPARK = [{"engine": "spark", "version": "3.5.0"}]
    _PANDAS = [{"engine": "pandas", "version": "2.0.0"}]
    _OPUS = [{"engine": "opus", "version": "2.0.0"}]

    def clean_data(self, data, cleaned_columns: Optional[List[str]] = None):
        return self._run("clean_data", "transform.clean_data.clean_data", data, self._PANDAS, cleaned_columns, "cleaned")

    def transform_data(self, data, translation_columns: Optional[List[str]] = None):
        return self._run("transform_data", "transform.transform_data.transform_data", data, self._SPARK, translation_columns, "language")

    def transform_iso(self, data, country_iso_columns: Optional[List[str]] = None):
        return self._run("transform_iso", "transform.iso_data.iso_data", data, self._SPARK, country_iso_columns, "iso")

    def transform_regex(self, data, regex_columns: Optional[List[str]] = None):
        return self._run("transform_regex", "transform.regex_data.regex_data", data, self._SPARK, regex_columns, "regex")

    def transform_metadata(self, data, metadata_columns: Optional[List[str]] = None):
        return self._run("transform_metadata", "transform.metadata_data.metadata_data", data, self._SPARK, metadata_columns, "metadata")

    def transform_date(self, data, date_columns: Optional[List[str]] = None):
        return self._run("transform_date", "transform.date_data.date_data", data, self._SPARK, date_columns, "date")


class WriteOperations:
    """Write operations available on a TableContext."""

    def __init__(self, table: TableContext) -> None:
        self._table = table

    def table(self, data, format: str = "iceberg"):
        from write.write_table import write_table

        engines = [{"engine": "pyspark", "version": "2.0.0"}]
        with self._table.function_tracker("write_table", function_engines=engines):
            self._table.table_type = format
            return write_table(data, format=format)


# ---------------------------------------------------------------------------
# TableContext
# ---------------------------------------------------------------------------

@dataclass
class TableContext:
    """Represents a single table within an ETL pipeline run."""

    # --- product / output fields ---
    table_name: str
    table_type: Optional[str] = None
    s3_path: Optional[str] = None
    stage: Optional[Stage] = None
    data_quality: Optional[DataQuality] = None
    num_of_columns: Optional[int] = None
    num_of_rows: Optional[int] = None
    schema: List[Schema] = field(default_factory=list)
    data_validation_results: List[DataValidationResult] = field(default_factory=list)
    data_domains: List[str] = field(default_factory=list)
    sample_data: List[Dict[str, Any]] = field(default_factory=list)
    functions_ran: List[FunctionExecution] = field(default_factory=list)
    etl_comments: Optional[str] = None

    # --- source / input fields ---
    source_s3_path: Optional[str] = None
    source_data_quality: Optional[DataQuality] = None
    source_num_of_columns: Optional[int] = None
    source_num_of_rows: Optional[int] = None

    # Accepted at construction time, extracted, then discarded — never stored.
    source_df: InitVar[Optional["DataFrame"]] = None

    def __post_init__(self, source_df: Optional["DataFrame"]) -> None:
        if not self.table_name or not self.table_name.strip():
            raise ValueError("table_name cannot be empty")
        if self.stage is not None and self.stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES!r}, got {self.stage!r}")

        # Operation namespaces — plain attributes so dataclass machinery ignores them.
        self.read = ReadOperations(self)
        self.transform = TransformOperations(self)
        self.write = WriteOperations(self)

        if source_df is not None:
            self.ingest_source(source_df)

    # ------------------------------------------------------------------
    # Source DataFrame ingestion (call once; the DataFrame is never stored)
    # ------------------------------------------------------------------

    def ingest_source(self, df: "DataFrame") -> None:
        """Extract metadata from *df* and populate source fields.

        The DataFrame itself is **never stored** on the instance; only the
        derived scalar values are kept.  Call this once after construction.

        Example::

            table = TableContext(table_name="orders", stage="bronze")
            table.ingest_source(source_df)
        """
        try:
            info = extract_source_info(df)
        except Exception as exc:
            print(f"\n⚠️  Warning: Failed to extract source information: {exc}")
            pretty_print_source_info(self)
            return

        # Populate source fields only if they haven't been set manually.
        if not self.schema:
            self.schema = info.schema
        if self.source_num_of_columns is None:
            self.source_num_of_columns = info.num_of_columns
        if self.source_num_of_rows is None:
            self.source_num_of_rows = info.num_of_rows
        if self.source_data_quality is None:
            self.source_data_quality = info.data_quality

        pretty_print_source_info(self)

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def copy_schema(self) -> List[Schema]:
        """Return a deep copy of this table's schema list."""
        return [
            Schema(
                column_name=s.column_name,
                origin_type=s.origin_type,
                function_created_by=s.function_created_by,
                data_type=s.data_type,
            )
            for s in self.schema
        ]

    def _columns_by_origin(self, origin_type: OriginType) -> List[str]:
        return [s.column_name for s in self.schema if s.origin_type == origin_type]

    # ------------------------------------------------------------------
    # DataFrame display
    # ------------------------------------------------------------------

    def displayDF(self, df: "DataFrame", max_rows: int = 100) -> None:
        """Display *df* using the custom pretty-printer.

        Example::

            df = customer.transform.clean_data(source_df)
            customer.display(df)
        """
        if displayDF is None:
            raise ImportError(
                "dataframe_viewer is not available. "
                "Ensure dataframe_viewer.py is on your Python path."
            )
        displayDF(
            df,
            max_rows=max_rows,
            table_name=self.table_name,
            stage=self.stage,
            translation_columns=self._columns_by_origin("language") or None,
            country_iso_columns=self._columns_by_origin("iso") or None,
            regex_columns=self._columns_by_origin("regex") or None,
            metadata_columns=self._columns_by_origin("metadata") or None,
            cleaned_columns=self._columns_by_origin("cleaned") or None,
            date_columns=self._columns_by_origin("date") or None,
        )

    # ------------------------------------------------------------------
    # Function tracking
    # ------------------------------------------------------------------

    @contextmanager
    def function_tracker(
        self,
        function_name: str,
        function_engines: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[FunctionExecution, None, None]:
        """Context manager that records a function execution with timing.

        Example::

            with table.function_tracker("clean_data", function_engines=[...]):
                ...
        """
        execution = FunctionExecution(
            run_id=str(uuid.uuid4()),
            function_name=function_name,
            run_order=len(self.functions_ran) + 1,
            start_time=datetime.now(),
            function_engines=function_engines,
        )
        self.functions_ran.append(execution)
        try:
            yield execution
        finally:
            execution.end_time = datetime.now()

    def get_functions_by_name(self, function_name: str) -> List[FunctionExecution]:
        """Return all executions whose name matches *function_name*."""
        return [e for e in self.functions_ran if e.function_name == function_name]

    def get_functions_by_run_id(self, run_id: str) -> List[FunctionExecution]:
        """Return all executions whose run_id matches *run_id*."""
        return [e for e in self.functions_ran if e.run_id == run_id]

    def update_function_end_time(self, run_id: str, end_time: Optional[datetime] = None) -> bool:
        """Stamp *end_time* on the execution identified by *run_id*.

        Returns True if found, False otherwise.
        """
        end_time = end_time or datetime.now()
        for execution in self.functions_ran:
            if execution.run_id == run_id:
                execution.end_time = end_time
                return True
        return False


# ---------------------------------------------------------------------------
# ETLContext
# ---------------------------------------------------------------------------

@dataclass
class ETLContext:
    """Top-level container for a single ETL pipeline run."""

    pipeline_id: Optional[str] = None
    source_dataset_ids: List[str] = field(default_factory=list)
    periodicity: Optional[str] = None
    expected_delivery: Optional[str] = None
    delivered_version: Optional[str] = None
    etl_platform: Optional[str] = None
    github_link: Optional[str] = None
    data_agreement_id: Optional[str] = None
    data_model_id: Optional[str] = None
    etl_comments: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    tables: List[TableContext] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.delivered_version is not None:
            if not DELIVERED_VERSION_PATTERN.match(self.delivered_version):
                raise ValueError(
                    "delivered_version must match 'v<number>' (e.g. 'v1', 'v12'), "
                    f"got {self.delivered_version!r}"
                )
        if self.start_time is None:
            self.start_time = datetime.now()
        pretty_print_etl_context_init(self)

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def add_table(self, table: TableContext) -> None:
        """Add *table*, replacing any existing table with the same name."""
        existing = self.get_table_by_name(table.table_name)
        if existing:
            idx = self.tables.index(existing)
            self.tables[idx] = table
        else:
            self.tables.append(table)
            idx = len(self.tables) - 1
        pretty_print_table(table, idx + 1)

    def get_table_by_name(self, table_name: str) -> Optional[TableContext]:
        return next((t for t in self.tables if t.table_name == table_name), None)

    def get_table_by_s3_path(self, s3_path: str) -> Optional[TableContext]:
        return next((t for t in self.tables if t.s3_path == s3_path), None)

    def get_tables_by_type(self, table_type: str) -> List[TableContext]:
        return [t for t in self.tables if t.table_type == table_type]

    def update_table_type(self, table_name: str, table_type: str) -> bool:
        """Set *table_type* on the named table. Returns True if found."""
        table = self.get_table_by_name(table_name)
        if table:
            table.table_type = table_type
            return True
        return False

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def update_end_time(self, end_time: Optional[datetime] = None) -> None:
        self.end_time = end_time or datetime.now()

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self) -> None:
        """Submit ETL context metrics to monitoring/observability systems."""
        if self.end_time is None:
            self.update_end_time()
        pretty_print_submit(self)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return _serialize(self)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Return a JSON string representation.

        Args:
            indent: Indentation level for pretty-printing (None = compact).
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def pretty_print(self) -> None:
        """Print a human-readable summary of the entire ETL context."""
        pretty_print_etl_context(self)


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

_SKIP_TYPES = (ReadOperations, TransformOperations, WriteOperations)


def _serialize(obj: Any) -> Any:
    """Recursively convert *obj* to a JSON-safe structure.

    Operation namespace objects are skipped to avoid circular references.
    datetime values are ISO-8601 strings.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, _SKIP_TYPES):
        return None
    if hasattr(obj, "__dataclass_fields__"):
        return {
            k: _serialize(v)
            for k, v in vars(obj).items()
            if not isinstance(v, _SKIP_TYPES)
        }
    if isinstance(obj, list):
        return [_serialize(item) for item in obj if not isinstance(item, _SKIP_TYPES)]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items() if not isinstance(v, _SKIP_TYPES)}
    return obj