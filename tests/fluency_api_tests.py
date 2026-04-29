"""
Test suite for etl_context.py

Run with:
    pytest test_etl_context.py -v
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# We import the module under test. The PySpark + pretty_print imports inside
# the module are conditionally used, so we patch them at the module boundary.
# ---------------------------------------------------------------------------

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root so fluency_api can be imported when running pytest from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub out heavy / unavailable dependencies before importing the module.
for mod in [
    "pretty_print_rich",
    "dataframe_viewer",
    "pyspark",
    "pyspark.sql",
    "pyspark.sql.functions",
]:
    sys.modules.setdefault(mod, MagicMock())

# Provide the specific names imported by the module.
sys.modules["pretty_print_rich"].pretty_print_etl_context = MagicMock()
sys.modules["pretty_print_rich"].pretty_print_etl_context_init = MagicMock()
sys.modules["pretty_print_rich"].pretty_print_source_info = MagicMock()
sys.modules["pretty_print_rich"].pretty_print_submit = MagicMock()
sys.modules["pretty_print_rich"].pretty_print_table = MagicMock()

from fluency_api import (  # noqa: E402  (import after sys.modules patching)
    DataQuality,
    DataValidationResult,
    ETLContext,
    FunctionExecution,
    Schema,
    SourceInfo,
    TableContext,
    DELIVERED_VERSION_PATTERN,
    extract_source_info,
    update_schema_origin,
    _serialize,
    _schema_column_key,
)


# ===========================================================================
# Helpers
# ===========================================================================

def make_table(name: str = "orders", stage: str = "bronze") -> TableContext:
    return TableContext(table_name=name, stage=stage)


def make_etl(**kwargs) -> ETLContext:
    defaults = dict(data_product_id="pipe-1", product_delivery_version="v1")
    defaults.update(kwargs)
    return ETLContext(**defaults)


# ===========================================================================
# DataValidationResult
# ===========================================================================

class TestDataValidationResult:
    def test_valid_passed(self):
        r = DataValidationResult(rule_id="r1", passed_or_failed="passed")
        assert r.rule_id == "r1"
        assert r.passed_or_failed == "passed"

    def test_valid_failed(self):
        r = DataValidationResult(rule_id="r2", passed_or_failed="failed", message="oops")
        assert r.message == "oops"

    def test_valid_warning(self):
        r = DataValidationResult(rule_id="r3", passed_or_failed="warning")
        assert r.passed_or_failed == "warning"

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="passed_or_failed"):
            DataValidationResult(rule_id="r4", passed_or_failed="unknown")

    def test_optional_message_defaults_none(self):
        r = DataValidationResult(rule_id="r5", passed_or_failed="passed")
        assert r.message is None


# ===========================================================================
# Schema
# ===========================================================================

class TestSchema:
    def test_valid_origin_types(self):
        for origin in ("source", "iso", "language", "regex", "metadata", "cleaned", "date"):
            s = Schema(original_column_name="col", origin_type=origin)
            assert s.origin_type == origin

    def test_invalid_origin_type_raises(self):
        with pytest.raises(ValueError, match="origin_type"):
            Schema(original_column_name="col", origin_type="bad_type")

    def test_optional_fields_default_none(self):
        s = Schema(original_column_name="col", origin_type="source")
        assert s.function_created_by is None
        assert s.data_type is None


# ===========================================================================
# FunctionExecution
# ===========================================================================

class TestFunctionExecution:
    def test_start_time_set_on_init(self):
        before = datetime.now()
        fe = FunctionExecution(run_id="x", function_name="fn", run_order=1)
        after = datetime.now()
        assert before <= fe.start_time <= after

    def test_duration_none_without_end_time(self):
        fe = FunctionExecution(run_id="x", function_name="fn", run_order=1)
        assert fe.duration_seconds is None

    def test_duration_calculated(self):
        fe = FunctionExecution(
            run_id="x",
            function_name="fn",
            run_order=1,
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 12, 0, 5),
        )
        assert fe.duration_seconds == pytest.approx(5.0)

    def test_duration_zero_same_start_end(self):
        t = datetime(2024, 1, 1, 12, 0, 0)
        fe = FunctionExecution(run_id="x", function_name="fn", run_order=1, start_time=t, end_time=t)
        assert fe.duration_seconds == pytest.approx(0.0)


# ===========================================================================
# DataQuality
# ===========================================================================

class TestDataQuality:
    def test_defaults_none(self):
        dq = DataQuality()
        assert dq.null_value_count is None
        assert dq.empty_row_count is None

    def test_set_values(self):
        dq = DataQuality(null_value_count=3, empty_row_count=1)
        assert dq.null_value_count == 3
        assert dq.empty_row_count == 1


# ===========================================================================
# update_schema_origin
# ===========================================================================

class TestUpdateSchemaOrigin:
    def test_updates_existing_column(self):
        schema = [Schema(original_column_name="col1", origin_type="source")]
        update_schema_origin(schema, ["col1"], "cleaned")
        assert schema[0].origin_type == "cleaned"

    def test_appends_new_column(self):
        schema = [Schema(original_column_name="col1", origin_type="source")]
        update_schema_origin(schema, ["col2"], "language")
        names = [_schema_column_key(s) for s in schema]
        assert "col2" in names
        new = next(s for s in schema if _schema_column_key(s) == "col2")
        assert new.origin_type == "language"

    def test_mixed_existing_and_new(self):
        schema = [Schema(original_column_name="a", origin_type="source")]
        update_schema_origin(schema, ["a", "b"], "iso")
        assert len(schema) == 2
        assert all(s.origin_type == "iso" for s in schema)

    def test_empty_column_list_no_change(self):
        schema = [Schema(original_column_name="a", origin_type="source")]
        update_schema_origin(schema, [], "cleaned")
        assert schema[0].origin_type == "source"

    def test_empty_schema_appends(self):
        schema = []
        update_schema_origin(schema, ["new_col"], "regex")
        assert len(schema) == 1
        assert schema[0].original_column_name == "new_col"


# ===========================================================================
# TableContext construction & validation
# ===========================================================================

class TestTableContextConstruction:
    def test_basic_construction(self):
        t = make_table("my_table", "silver")
        assert t.table_name == "my_table"
        assert t.stage == "silver"

    def test_empty_table_name_raises(self):
        with pytest.raises(ValueError, match="table_name"):
            TableContext(table_name="")

    def test_whitespace_table_name_raises(self):
        with pytest.raises(ValueError, match="table_name"):
            TableContext(table_name="   ")

    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError, match="stage"):
            TableContext(table_name="t", stage="platinum")

    def test_valid_stages(self):
        for stage in ("bronze", "silver", "gold"):
            t = TableContext(table_name="t", stage=stage)
            assert t.stage == stage

    def test_none_stage_allowed(self):
        t = TableContext(table_name="t")
        assert t.stage is None

    def test_invalid_table_type_raises(self):
        with pytest.raises(ValueError, match="table_type"):
            TableContext(table_name="t", table_type="parquet")

    def test_valid_table_types(self):
        for tt in ("iceberg", "delta"):
            t = TableContext(table_name="t", table_type=tt)
            assert t.table_type == tt

    def test_none_table_type_allowed(self):
        t = TableContext(table_name="t")
        assert t.table_type is None

    def test_invalid_pipeline_name_raises(self):
        with pytest.raises(ValueError, match="pipeline_name"):
            TableContext(table_name="t", pipeline_name="ETL")

    def test_valid_pipeline_name_MD(self):
        t = TableContext(table_name="t", pipeline_name="MD")
        assert t.pipeline_name == "MD"

    def test_none_pipeline_name_allowed(self):
        assert TableContext(table_name="t").pipeline_name is None

    def test_operation_namespaces_attached(self):
        t = make_table()
        assert hasattr(t, "read")
        assert hasattr(t, "transform")
        assert hasattr(t, "write")

    def test_default_lists_empty(self):
        t = make_table()
        assert t.schema == []
        assert t.data_validation_results == []
        assert t.functions_ran == []
        assert t.data_tags == []
        assert t.sample_data == []


# ===========================================================================
# TableContext.ingest_source
# ===========================================================================

class TestIngestSource:
    def _make_mock_df(self, columns=("a", "b"), num_rows=10, null_count=2, empty_rows=0):
        """Return a MagicMock that passes through extract_source_info."""
        from pyspark.sql import functions as F

        # Build a fake schema
        field_a = MagicMock(); field_a.name = "a"; field_a.dataType = "StringType()"
        field_b = MagicMock(); field_b.name = "b"; field_b.dataType = "IntegerType()"
        fields = [field_a, field_b][: len(columns)]

        df = MagicMock()
        df.columns = list(columns)
        df.schema.fields = fields

        # Aggregate row
        row = MagicMock()
        row.__getitem__ = lambda self, key: (
            num_rows if key == "row_count" else
            empty_rows if key == "empty_row_count" else
            null_count // len(columns)  # null_N
        )
        df.agg.return_value.collect.return_value = [row]
        return df

    def test_source_fields_populated(self):
        t = make_table()
        df = self._make_mock_df(num_rows=50, null_count=4, empty_rows=1)
        with patch("fluency_api.extract_source_info") as mock_extract:
            mock_extract.return_value = SourceInfo(
                schema=[Schema(original_column_name="a", origin_type="source")],
                num_of_columns=2,
                num_of_rows=50,
                data_quality=DataQuality(null_value_count=4, empty_row_count=1),
            )
            t.ingest_source(df)

        assert t.source_num_of_columns == 2
        assert t.source_num_of_rows == 50
        assert t.source_data_quality.null_value_count == 4
        assert t.source_data_quality.empty_row_count == 1

    def test_existing_schema_not_overwritten(self):
        existing_schema = [Schema(original_column_name="x", origin_type="cleaned")]
        t = TableContext(table_name="t", schema=existing_schema)
        with patch("fluency_api.extract_source_info") as mock_extract:
            mock_extract.return_value = SourceInfo(
                schema=[Schema(original_column_name="a", origin_type="source")],
                num_of_columns=1,
                num_of_rows=5,
                data_quality=DataQuality(),
            )
            t.ingest_source(MagicMock())

        # Original schema preserved
        assert t.schema == existing_schema

    def test_failed_extraction_prints_warning(self, capsys):
        t = make_table()
        with patch("fluency_api.extract_source_info", side_effect=RuntimeError("boom")):
            t.ingest_source(MagicMock())  # Should not raise


# ===========================================================================
# TableContext — function_tracker
# ===========================================================================

class TestFunctionTracker:
    def test_records_execution(self):
        t = make_table()
        with t.function_tracker("my_fn"):
            pass
        assert len(t.functions_ran) == 1
        assert t.functions_ran[0].function_name == "my_fn"

    def test_run_order_increments(self):
        t = make_table()
        with t.function_tracker("fn1"):
            pass
        with t.function_tracker("fn2"):
            pass
        assert t.functions_ran[0].run_order == 1
        assert t.functions_ran[1].run_order == 2

    def test_end_time_set_after_context(self):
        t = make_table()
        with t.function_tracker("fn") as fe:
            assert fe.end_time is None
        assert fe.end_time is not None

    def test_end_time_set_even_on_exception(self):
        t = make_table()
        with pytest.raises(ValueError):
            with t.function_tracker("fn") as fe:
                raise ValueError("intentional")
        assert fe.end_time is not None

    def test_yields_function_execution_instance(self):
        t = make_table()
        with t.function_tracker("fn", function_engines=[{"engine": "spark"}]) as fe:
            assert isinstance(fe, FunctionExecution)
            assert fe.function_engines == [{"engine": "spark"}]

    def test_duration_positive(self):
        import time
        t = make_table()
        with t.function_tracker("fn") as fe:
            time.sleep(0.01)
        assert fe.duration_seconds > 0

    def test_get_functions_by_name(self):
        t = make_table()
        with t.function_tracker("alpha"):
            pass
        with t.function_tracker("beta"):
            pass
        with t.function_tracker("alpha"):
            pass
        results = t.get_functions_by_name("alpha")
        assert len(results) == 2
        assert all(r.function_name == "alpha" for r in results)

    def test_get_functions_by_run_id(self):
        t = make_table()
        with t.function_tracker("fn") as fe:
            target_id = fe.run_id
        with t.function_tracker("fn"):
            pass
        results = t.get_functions_by_run_id(target_id)
        assert len(results) == 1
        assert results[0].run_id == target_id

    def test_get_functions_by_name_no_match(self):
        t = make_table()
        assert t.get_functions_by_name("nonexistent") == []

    def test_get_functions_by_run_id_no_match(self):
        t = make_table()
        assert t.get_functions_by_run_id("fake-id") == []


# ===========================================================================
# TableContext — update_function_end_time
# ===========================================================================

class TestUpdateFunctionEndTime:
    def test_updates_end_time_by_run_id(self):
        t = make_table()
        with t.function_tracker("fn") as fe:
            rid = fe.run_id
        new_time = datetime(2024, 6, 1, 10, 0, 0)
        result = t.update_function_end_time(rid, new_time)
        assert result is True
        assert fe.end_time == new_time

    def test_returns_false_for_unknown_run_id(self):
        t = make_table()
        result = t.update_function_end_time("nonexistent-id")
        assert result is False

    def test_defaults_to_now_when_no_end_time_given(self):
        t = make_table()
        with t.function_tracker("fn") as fe:
            rid = fe.run_id
        before = datetime.now()
        t.update_function_end_time(rid)
        assert fe.end_time >= before


# ===========================================================================
# TableContext — copy_schema
# ===========================================================================

class TestCopySchema:
    def test_returns_deep_copy(self):
        t = make_table()
        t.schema = [Schema(original_column_name="col", origin_type="source", data_type="StringType")]
        copy = t.copy_schema()
        assert copy is not t.schema
        assert copy[0] is not t.schema[0]
        assert copy[0].original_column_name == "col"
        assert copy[0].origin_type == "source"

    def test_mutation_does_not_affect_original(self):
        t = make_table()
        t.schema = [Schema(original_column_name="col", origin_type="source")]
        copy = t.copy_schema()
        copy[0].origin_type = "cleaned"
        assert t.schema[0].origin_type == "source"

    def test_empty_schema_copy(self):
        t = make_table()
        assert t.copy_schema() == []


# ===========================================================================
# ETLContext construction & validation
# ===========================================================================

class TestETLContextConstruction:
    def test_basic_construction(self):
        etl = make_etl()
        assert etl.data_product_id == "pipe-1"
        assert etl.product_delivery_version == "v1"

    def test_start_time_set_automatically(self):
        before = datetime.now()
        etl = make_etl()
        after = datetime.now()
        assert before <= etl.start_time <= after

    def test_invalid_product_delivery_version_raises(self):
        with pytest.raises(ValueError, match="product_delivery_version"):
            ETLContext(product_delivery_version="version1")

    def test_valid_product_delivery_versions(self):
        for v in ("v1", "v10", "v999"):
            etl = ETLContext(product_delivery_version=v)
            assert etl.product_delivery_version == v

    def test_none_product_delivery_version_allowed(self):
        etl = ETLContext()
        assert etl.product_delivery_version is None

    def test_invalid_periodicity_raises(self):
        with pytest.raises(ValueError, match="periodicity"):
            ETLContext(periodicity="Daily")

    def test_valid_periodicity_values(self):
        for p in ("Feed", "One-Time"):
            etl = ETLContext(periodicity=p)
            assert etl.periodicity == p

    def test_none_periodicity_allowed(self):
        assert ETLContext().periodicity is None

    def test_invalid_fabric_raises(self):
        with pytest.raises(ValueError, match="fabric"):
            ETLContext(fabric="M")

    def test_valid_fabric_L(self):
        etl = ETLContext(fabric="L")
        assert etl.fabric == "L"

    def test_none_fabric_allowed(self):
        assert ETLContext().fabric is None

    def test_invalid_etl_platform_raises(self):
        with pytest.raises(ValueError, match="etl_platform"):
            ETLContext(etl_platform="airflow")

    def test_valid_etl_platforms(self):
        for p in ("kubeflow", "palantir"):
            etl = ETLContext(etl_platform=p)
            assert etl.etl_platform == p

    def test_none_etl_platform_allowed(self):
        assert ETLContext().etl_platform is None

    def test_tables_default_empty(self):
        etl = make_etl()
        assert etl.tables == []


# ===========================================================================
# ETLContext — table management
# ===========================================================================

class TestETLContextTableManagement:
    def test_add_table(self):
        etl = make_etl()
        t = make_table("orders")
        etl.add_table(t)
        assert etl.get_table_by_name("orders") is t

    def test_add_table_replaces_duplicate_name(self):
        etl = make_etl()
        t1 = make_table("orders")
        t2 = make_table("orders")
        etl.add_table(t1)
        etl.add_table(t2)
        assert len(etl.tables) == 1
        assert etl.get_table_by_name("orders") is t2

    def test_get_table_by_name_not_found(self):
        etl = make_etl()
        assert etl.get_table_by_name("missing") is None

    def test_get_table_by_s3_path(self):
        etl = make_etl()
        t = make_table("orders")
        t.table_s3_path = "s3://bucket/orders"
        etl.add_table(t)
        assert etl.get_table_by_s3_path("s3://bucket/orders") is t

    def test_get_table_by_s3_path_not_found(self):
        etl = make_etl()
        assert etl.get_table_by_s3_path("s3://missing") is None

    def test_get_tables_by_type(self):
        etl = make_etl()
        t1 = make_table("a"); t1.table_type = "iceberg"
        t2 = make_table("b"); t2.table_type = "delta"
        t3 = make_table("c"); t3.table_type = "iceberg"
        for t in (t1, t2, t3):
            etl.add_table(t)
        results = etl.get_tables_by_type("iceberg")
        assert len(results) == 2
        assert t2 not in results

    def test_update_table_type_success(self):
        etl = make_etl()
        etl.add_table(make_table("orders"))
        result = etl.update_table_type("orders", "delta")
        assert result is True
        assert etl.get_table_by_name("orders").table_type == "delta"

    def test_update_table_type_not_found(self):
        etl = make_etl()
        result = etl.update_table_type("ghost", "iceberg")
        assert result is False

    def test_update_table_type_invalid_raises(self):
        etl = make_etl()
        etl.add_table(make_table("orders"))
        with pytest.raises(ValueError, match="table_type"):
            etl.update_table_type("orders", "parquet")  # type: ignore[arg-type]


# ===========================================================================
# ETLContext — timing
# ===========================================================================

class TestETLContextTiming:
    def test_update_end_time_explicit(self):
        etl = make_etl()
        t = datetime(2024, 12, 31, 23, 59, 59)
        etl.update_end_time(t)
        assert etl.end_time == t

    def test_update_end_time_defaults_to_now(self):
        etl = make_etl()
        before = datetime.now()
        etl.update_end_time()
        assert etl.end_time >= before

    def test_submit_sets_end_time_if_missing(self):
        etl = make_etl()
        assert etl.end_time is None
        etl.submit()
        assert etl.end_time is not None


# ===========================================================================
# ETLContext — serialisation
# ===========================================================================

class TestSerialisation:
    def test_to_dict_returns_dict(self):
        etl = make_etl()
        d = etl.to_dict()
        assert isinstance(d, dict)

    def test_to_json_valid_json(self):
        etl = make_etl()
        raw = etl.to_json()
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_to_json_pretty_indent(self):
        etl = make_etl()
        raw = etl.to_json(indent=2)
        assert "\n" in raw

    def test_to_dict_contains_expected_keys(self):
        etl = make_etl(data_product_id="pipe-42")
        d = etl.to_dict()
        assert "data_product_id" in d
        assert d["data_product_id"] == "pipe-42"

    def test_to_dict_datetime_serialised_as_string(self):
        etl = make_etl()
        d = etl.to_dict()
        assert isinstance(d["start_time"], str)

    def test_to_dict_tables_are_refs_only(self):
        etl = make_etl()
        etl.add_table(make_table("orders"))
        d = etl.to_dict()
        assert len(d["tables"]) == 1
        ref = d["tables"][0]
        assert ref == {
            "table_name": "orders",
            "data_product_id": etl.data_product_id,
        }

    def test_operation_namespaces_not_in_table_refs(self):
        etl = make_etl()
        etl.add_table(make_table("t"))
        d = etl.to_dict()
        table_ref = d["tables"][0]
        assert set(table_ref.keys()) == {"table_name", "data_product_id"}
        assert "read" not in table_ref
        assert "transform" not in table_ref
        assert "write" not in table_ref


# ===========================================================================
# _serialize helper
# ===========================================================================

class TestSerializeHelper:
    def test_datetime_to_iso_string(self):
        dt = datetime(2024, 1, 15, 10, 30, 0)
        assert _serialize(dt) == "2024-01-15T10:30:00"

    def test_list_serialised(self):
        result = _serialize([1, "two", 3.0])
        assert result == [1, "two", 3.0]

    def test_dict_serialised(self):
        result = _serialize({"a": 1, "b": datetime(2024, 1, 1)})
        assert result["a"] == 1
        assert isinstance(result["b"], str)

    def test_primitive_passthrough(self):
        for val in (42, 3.14, "hello", True, None):
            assert _serialize(val) == val

    def test_nested_dataclass(self):
        dq = DataQuality(null_value_count=5, empty_row_count=2)
        result = _serialize(dq)
        assert result["null_value_count"] == 5
        assert result["empty_row_count"] == 2


# ===========================================================================
# DELIVERED_VERSION_PATTERN
# ===========================================================================

class TestDeliveredVersionPattern:
    @pytest.mark.parametrize("v", ["v1", "v10", "v100", "v999"])
    def test_valid_versions(self, v):
        assert DELIVERED_VERSION_PATTERN.match(v)

    @pytest.mark.parametrize("v", ["v", "version1", "1", "V1", "v1.0", "v-1", ""])
    def test_invalid_versions(self, v):
        assert not DELIVERED_VERSION_PATTERN.match(v)


# ===========================================================================
# Integration: full pipeline simulation (no Spark)
# ===========================================================================

class TestIntegration:
    def test_full_pipeline_roundtrip(self):
        etl = ETLContext(data_product_id="int-test", product_delivery_version="v3")

        table = TableContext(table_name="customers", stage="silver")
        table.schema = [
            Schema(original_column_name="id", origin_type="source", data_type="IntegerType"),
            Schema(original_column_name="name", origin_type="source", data_type="StringType"),
        ]

        with table.function_tracker("clean_data", function_engines=[{"engine": "pandas"}]):
            update_schema_origin(table.schema, ["name"], "cleaned")

        table.data_validation_results.append(
            DataValidationResult(rule_id="not_null_id", passed_or_failed="passed")
        )

        etl.add_table(table)
        etl.submit()

        payload = json.loads(etl.to_json())
        assert payload["data_product_id"] == "int-test"
        assert payload["product_delivery_version"] == "v3"
        assert len(payload["tables"]) == 1
        assert payload["tables"][0] == {
            "table_name": "customers",
            "data_product_id": "int-test",
        }

        t_dict = json.loads(json.dumps(_serialize(table)))
        assert t_dict["table_name"] == "customers"
        assert t_dict["stage"] == "silver"

        schema_origins = {
            (s["transformed_column_name"] or s["original_column_name"]): s["origin_type"]
            for s in t_dict["schema"]
        }
        assert schema_origins["id"] == "source"
        assert schema_origins["name"] == "cleaned"

        fn = t_dict["functions_ran"][0]
        assert fn["function_name"] == "clean_data"
        assert fn["end_time"] is not None

    def test_multiple_tables_independent(self):
        etl = make_etl()
        for name in ("orders", "products", "customers"):
            t = make_table(name)
            with t.function_tracker(f"process_{name}"):
                pass
            etl.add_table(t)

        assert len(etl.tables) == 3
        for name in ("orders", "products", "customers"):
            assert etl.get_table_by_name(name) is not None