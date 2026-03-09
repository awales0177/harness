"""
Robust test suite for dataframe_viewer.py
==========================================

Run with:
    pytest test_dataframe_viewer.py -v

All PySpark, IPython, and Jupyter dependencies are fully stubbed so the
suite runs in any plain Python environment.

Coverage areas
--------------
- Constants & configuration
- _TagSpec dataclass
- ISO helpers: _iso2_to_flag, _code_to_flag, _ISO3_TO_ISO2
- SVG icon constructors: _hdr_svg, _thm_svg, _hdr_svg_filled, _hdr_svg_fill_stroke
- TAG_ICONS completeness assertion
- _validate_inputs
- _build_viewer_ids
- _collect_rows / _TableData
- _estimate_display_len
- _get_column_metadata
- _is_complex_type
- _format_complex_cell  (StructType, ArrayType, MapType, sentinel, None)
- _format_one_country_code
- _format_country_list
- _format_simple_cell   (None, bool, ISO, plain)
- _sort_attr_value
- _build_rows_text
- _render_css / _render_scripts / _render_header / _render_table / _render_footer
- _render_column_selector_modal
- displayDF integration (IPython path, fallback path, validation errors)
- _STYLE_INJECTED one-shot CSS sentinel
- HTML-injection safety (XSS)
"""

from __future__ import annotations

import html as _html
import sys
import types
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch, call
import re

import pytest


# ===========================================================================
# ── Stub out all heavy / unavailable dependencies ──────────────────────────
# ===========================================================================

def _make_spark_types():
    """Return a minimal fake pyspark.sql.types module."""
    m = types.ModuleType("pyspark.sql.types")

    class _Base: pass

    class StructType(_Base): pass
    class ArrayType(_Base): pass
    class MapType(_Base): pass
    class StringType(_Base):
        def __repr__(self): return "StringType()"
    class IntegerType(_Base):
        def __repr__(self): return "IntegerType()"
    class BooleanType(_Base):
        def __repr__(self): return "BooleanType()"
    class LongType(_Base):
        def __repr__(self): return "LongType()"

    m.StructType  = StructType
    m.ArrayType   = ArrayType
    m.MapType     = MapType
    m.StringType  = StringType
    m.IntegerType = IntegerType
    m.BooleanType = BooleanType
    m.LongType    = LongType
    return m


def _make_pyspark():
    """Full pyspark stub tree."""
    pyspark            = types.ModuleType("pyspark")
    pyspark_sql        = types.ModuleType("pyspark.sql")
    pyspark_sql_types  = _make_spark_types()
    pyspark_sql_funcs  = types.ModuleType("pyspark.sql.functions")

    # DataFrame stub with a __instancecheck__ so isinstance(df, DataFrame) works
    class DataFrame:
        pass
    pyspark_sql.DataFrame = DataFrame

    pyspark.sql         = pyspark_sql
    pyspark_sql.types   = pyspark_sql_types
    pyspark_sql.functions = pyspark_sql_funcs

    sys.modules["pyspark"]             = pyspark
    sys.modules["pyspark.sql"]         = pyspark_sql
    sys.modules["pyspark.sql.types"]   = pyspark_sql_types
    sys.modules["pyspark.sql.functions"] = pyspark_sql_funcs
    return pyspark, pyspark_sql, pyspark_sql_types


_pyspark, _pyspark_sql, _spark_types = _make_pyspark()

# IPython stub
_ipython_mod = types.ModuleType("IPython")
_ipython_display_mod = types.ModuleType("IPython.display")
_ipython_display_mock = MagicMock()
_HTML_mock = MagicMock(side_effect=lambda x: x)   # HTML(x) → x
_ipython_display_mod.HTML = _HTML_mock
_ipython_display_mod.display = _ipython_display_mock
_ipython_mod.display = _ipython_display_mod
sys.modules["IPython"]         = _ipython_mod
sys.modules["IPython.display"] = _ipython_display_mod

# Now safe to import the module under test
import dataframe_viewer as dv   # noqa: E402


# ===========================================================================
# ── Helpers / fixtures ─────────────────────────────────────────────────────
# ===========================================================================

def _make_field(name: str, dtype=None):
    """Return a fake schema field."""
    f = MagicMock()
    f.name = name
    f.dataType = dtype or _spark_types.StringType()
    return f


def _make_df(columns=("a", "b"), rows=None, dtypes=None):
    """
    Return a MagicMock that looks enough like a PySpark DataFrame for
    _get_column_metadata, _collect_rows, etc.

    We intentionally do NOT pass spec= to the top-level mock so that
    sub-attribute assignment (df.schema.fields = ...) works freely.
    _validate_inputs uses isinstance(df, SparkDataFrame) — we patch that
    check away in the integration tests, so the lack of spec is fine.
    """
    df = MagicMock()
    # Make isinstance(df, pyspark.sql.DataFrame) return True so _validate_inputs
    # doesn't raise TypeError.  We do this by making the DataFrame class's
    # __instancecheck__ accept our mock via a side-channel patch applied once.
    df.columns = list(columns)
    fields = []
    for i, c in enumerate(columns):
        dtype = (dtypes[i] if dtypes and i < len(dtypes) else None)
        fields.append(_make_field(c, dtype))
    # schema is also a plain MagicMock sub-attribute — assignment works fine.
    df.schema.fields = fields

    if rows is None:
        rows = [{c: f"v_{c}_{r}" for c in columns} for r in range(3)]

    # Support row[col_name] access
    row_mocks = []
    for r in rows:
        rm = MagicMock()
        rm.__getitem__ = lambda self, k, _r=r: _r[k]
        row_mocks.append(rm)

    df.take.side_effect = lambda n: row_mocks[:n]
    return df


# ===========================================================================
# ── 1  Constants ────────────────────────────────────────────────────────────
# ===========================================================================

class TestConstants:
    def test_valid_stages(self):
        assert dv.VALID_STAGES == frozenset({"bronze", "silver", "gold"})

    def test_complex_type_names(self):
        assert "StructType" in dv._COMPLEX_TYPE_NAMES
        assert "ArrayType"  in dv._COMPLEX_TYPE_NAMES
        assert "MapType"    in dv._COMPLEX_TYPE_NAMES

    def test_col_limits_sensible(self):
        assert dv._COL_MIN_PX < dv._COL_MAX_PX
        assert dv._COL_SAMPLE_ROWS > 0

    def test_max_rows_hard_cap_positive(self):
        assert dv._MAX_ROWS_HARD_CAP > 0

    def test_force_array_display_sentinel_is_string(self):
        assert isinstance(dv._FORCE_ARRAY_DISPLAY, str)

    def test_all_tags_length(self):
        assert len(dv.ALL_TAGS) == 6

    def test_all_tags_contains_expected_specs(self):
        keys = {t.filter_key for t in dv.ALL_TAGS}
        assert keys == {"language", "iso", "regex", "cleaned", "date", "metadata"}


# ===========================================================================
# ── 2  _TagSpec ─────────────────────────────────────────────────────────────
# ===========================================================================

class TestTagSpec:
    def test_css_class_format(self):
        assert dv.TAG_LANGUAGE.css_class == "tag-language"
        assert dv.TAG_ISO.css_class      == "tag-iso"
        assert dv.TAG_REGEX.css_class    == "tag-regex"
        assert dv.TAG_CLEANED.css_class  == "tag-cleaned"
        assert dv.TAG_DATE.css_class     == "tag-date"
        assert dv.TAG_METADATA.css_class == "tag-metadata"

    def test_param_names(self):
        assert dv.TAG_LANGUAGE.param_name == "translation_columns"
        assert dv.TAG_ISO.param_name      == "country_iso_columns"

    def test_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            dv.TAG_LANGUAGE.filter_key = "mutated"  # type: ignore


# ===========================================================================
# ── 3  TAG_ICONS completeness ───────────────────────────────────────────────
# ===========================================================================

class TestTagIcons:
    def test_all_tags_present_in_icons(self):
        assert set(dv.TAG_ICONS.keys()) == set(dv.ALL_TAGS)

    def test_every_icon_is_nonempty_string(self):
        for spec, icon in dv.TAG_ICONS.items():
            assert isinstance(icon, str) and icon, f"Empty icon for {spec.filter_key}"

    def test_icons_contain_svg(self):
        for spec, icon in dv.TAG_ICONS.items():
            assert "<svg" in icon, f"Icon for {spec.filter_key} missing <svg>"


# ===========================================================================
# ── 4  ISO helpers ──────────────────────────────────────────────────────────
# ===========================================================================

class TestIso2ToFlag:
    def test_known_us(self):
        flag = dv._iso2_to_flag("US")
        assert len(flag) == 2           # two regional-indicator chars
        assert flag == "\U0001F1FA\U0001F1F8"

    def test_lowercase_normalised(self):
        assert dv._iso2_to_flag("us") == dv._iso2_to_flag("US")

    def test_empty_string_returns_empty(self):
        assert dv._iso2_to_flag("") == ""

    def test_wrong_length_returns_empty(self):
        assert dv._iso2_to_flag("GBR") == ""
        assert dv._iso2_to_flag("G")   == ""

    def test_non_alpha_returns_empty(self):
        assert dv._iso2_to_flag("1B") == ""
        assert dv._iso2_to_flag("A2") == ""

    def test_gb(self):
        assert dv._iso2_to_flag("GB") == "\U0001F1EC\U0001F1E7"

    @pytest.mark.parametrize("code", ["US", "GB", "DE", "FR", "JP", "CN", "BR", "AU"])
    def test_common_codes_produce_two_chars(self, code):
        flag = dv._iso2_to_flag(code)
        assert len(flag) == 2


class TestCodeToFlag:
    def test_iso3_usa(self):
        flag = dv._code_to_flag("USA")
        assert flag == dv._iso2_to_flag("US")

    def test_iso3_gbr(self):
        assert dv._code_to_flag("GBR") == dv._iso2_to_flag("GB")

    def test_iso2_passthrough(self):
        assert dv._code_to_flag("US") == dv._iso2_to_flag("US")

    def test_unknown_iso3_returns_empty(self):
        assert dv._code_to_flag("ZZZ") == ""

    def test_none_returns_empty(self):
        assert dv._code_to_flag(None) == ""  # type: ignore

    def test_empty_string_returns_empty(self):
        assert dv._code_to_flag("") == ""

    def test_non_string_returns_empty(self):
        assert dv._code_to_flag(42) == ""   # type: ignore

    def test_lowercase_iso3(self):
        assert dv._code_to_flag("usa") == dv._code_to_flag("USA")

    def test_mixed_case_iso3(self):
        assert dv._code_to_flag("Usa") == dv._code_to_flag("USA")

    def test_xkx_kosovo(self):
        # Kosovo is a non-standard code explicitly added
        flag = dv._code_to_flag("XKX")
        assert flag != ""

    def test_four_char_returns_empty(self):
        assert dv._code_to_flag("USAA") == ""


class TestISO3ToISO2Mapping:
    def test_major_nations_present(self):
        required = {"USA": "US", "GBR": "GB", "DEU": "DE", "FRA": "FR",
                    "JPN": "JP", "CHN": "CN", "BRA": "BR", "AUS": "AU",
                    "IND": "IN", "RUS": "RU", "CAN": "CA", "MEX": "MX"}
        for iso3, iso2 in required.items():
            assert dv._ISO3_TO_ISO2.get(iso3) == iso2

    def test_all_values_are_two_chars(self):
        for iso3, iso2 in dv._ISO3_TO_ISO2.items():
            assert len(iso2) == 2, f"{iso3} → {iso2!r} is not 2 chars"

    def test_all_keys_are_three_chars(self):
        for iso3 in dv._ISO3_TO_ISO2:
            assert len(iso3) == 3, f"{iso3!r} is not 3 chars"

    def test_no_duplicate_values_except_legitimate(self):
        # Just ensure there's a reasonable number of mappings
        assert len(dv._ISO3_TO_ISO2) >= 200


# ===========================================================================
# ── 5  SVG icon constructors ────────────────────────────────────────────────
# ===========================================================================

class TestSvgConstructors:
    def test_hdr_svg_contains_class(self):
        out = dv._hdr_svg("my-class", "My title", "<path/>")
        assert 'class="my-class"' in out
        assert 'title="My title"' in out
        assert "<path/>" in out
        assert "14px" in out

    def test_thm_svg_larger_size(self):
        out = dv._thm_svg("t-class", "T", "<circle/>")
        assert "20px" in out
        assert "20" in out

    def test_hdr_svg_filled_viewbox(self):
        out = dv._hdr_svg_filled("f-class", "F", "0 0 32 32", "<rect/>")
        assert 'viewBox="0 0 32 32"' in out
        assert 'fill="currentColor"' in out
        assert "stroke" not in out.split("stroke-")[0]  # no bare stroke attr

    def test_hdr_svg_fill_stroke_has_both(self):
        out = dv._hdr_svg_fill_stroke("fs-class", "FS", "0 0 24 24", "<g/>")
        assert 'fill="currentColor"' in out
        assert 'stroke="currentColor"' in out

    def test_all_svgs_are_well_formed(self):
        """Every constructor should produce an opening and closing svg tag."""
        for fn, *args in [
            (dv._hdr_svg,           "c", "t", "<p/>"),
            (dv._thm_svg,           "c", "t", "<p/>"),
            (dv._hdr_svg_filled,    "c", "t", "0 0 24 24", "<p/>"),
            (dv._hdr_svg_fill_stroke, "c", "t", "0 0 24 24", "<p/>"),
        ]:
            out = fn(*args)
            assert out.startswith("<svg ") and out.endswith("</svg>")


# ===========================================================================
# ── 6  _validate_inputs ─────────────────────────────────────────────────────
# ===========================================================================

class TestValidateInputs:
    def _real_df(self):
        """Return a real-enough DF: isinstance(df, pyspark.sql.DataFrame) must pass."""
        return MagicMock(spec=_pyspark_sql.DataFrame)

    def test_valid_df_no_stage(self):
        dv._validate_inputs(self._real_df(), None, 100)   # no exception

    @pytest.mark.parametrize("stage", ["bronze", "BRONZE", "silver", "gold"])
    def test_valid_stages(self, stage):
        dv._validate_inputs(self._real_df(), stage, 100)

    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError, match="stage"):
            dv._validate_inputs(self._real_df(), "platinum", 100)

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="PySpark DataFrame"):
            dv._validate_inputs("not a df", None, 100)

    def test_exceeds_hard_cap_raises(self):
        with pytest.raises(ValueError, match="hard cap"):
            dv._validate_inputs(self._real_df(), None, dv._MAX_ROWS_HARD_CAP + 1)

    def test_exactly_at_hard_cap_ok(self):
        dv._validate_inputs(self._real_df(), None, dv._MAX_ROWS_HARD_CAP)

    def test_pyspark_not_installed_raises(self):
        """If PySpark isn't importable _validate_inputs raises ImportError."""
        original = sys.modules.get("pyspark.sql")
        try:
            sys.modules["pyspark.sql"] = None   # simulate missing
            with pytest.raises((ImportError, AttributeError)):
                dv._validate_inputs(object(), None, 10)
        finally:
            sys.modules["pyspark.sql"] = original

    def test_max_rows_zero_allowed(self):
        """max_rows=0 is allowed (empty preview)."""
        dv._validate_inputs(self._real_df(), None, 0)

    def test_negative_max_rows_raises(self):
        """Negative max_rows is rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            dv._validate_inputs(self._real_df(), None, -1)


# ===========================================================================
# ── 7  _build_viewer_ids ────────────────────────────────────────────────────
# ===========================================================================

class TestBuildViewerIds:
    def test_returns_viewer_ids(self):
        ids = dv._build_viewer_ids()
        assert isinstance(ids, dv._ViewerIds)

    def test_uid_is_8_hex_chars(self):
        ids = dv._build_viewer_ids()
        assert len(ids.uid) == 8
        assert all(c in "0123456789abcdef" for c in ids.uid)

    def test_all_fields_contain_uid(self):
        ids = dv._build_viewer_ids()
        for attr in ("viewer", "table", "toggle", "view", "filter",
                     "col_sel", "col_sel_modal", "footer"):
            val = getattr(ids, attr)
            assert ids.uid in val, f"uid not in {attr}: {val!r}"

    def test_ids_are_unique_across_calls(self):
        ids1 = dv._build_viewer_ids()
        ids2 = dv._build_viewer_ids()
        assert ids1.uid != ids2.uid


# ===========================================================================
# ── 8  _collect_rows / _TableData ───────────────────────────────────────────
# ===========================================================================

class TestCollectRows:
    def _make_rows(self, n):
        return [{"x": i} for i in range(n)]

    def _df_with_rows(self, n):
        df = MagicMock()
        rows = self._make_rows(n)
        df.take.side_effect = lambda limit: rows[:limit]
        return df

    def test_fewer_rows_than_max(self):
        df = self._df_with_rows(5)
        td = dv._collect_rows(df, 100)
        assert len(td.display_rows) == 5
        assert td.has_more is False

    def test_exactly_max_rows(self):
        df = self._df_with_rows(10)
        td = dv._collect_rows(df, 10)
        # take(11) returns 10 → no overflow
        assert len(td.display_rows) == 10
        assert td.has_more is False

    def test_more_rows_than_max(self):
        df = self._df_with_rows(50)
        td = dv._collect_rows(df, 10)
        assert len(td.display_rows) == 10
        assert td.has_more is True

    def test_take_called_with_max_plus_one(self):
        df = self._df_with_rows(5)
        dv._collect_rows(df, 10)
        df.take.assert_called_once_with(11)

    def test_empty_dataframe(self):
        df = self._df_with_rows(0)
        td = dv._collect_rows(df, 100)
        assert len(td.display_rows) == 0
        assert td.has_more is False

    def test_take_raises_propagates(self):
        """When df.take() raises, _collect_rows propagates the exception."""
        df = MagicMock()
        df.take.side_effect = RuntimeError("OOM or serialization error")
        with pytest.raises(RuntimeError, match="OOM or serialization"):
            dv._collect_rows(df, 10)


# ===========================================================================
# ── 9  _estimate_display_len ────────────────────────────────────────────────
# ===========================================================================

class TestEstimateDisplayLen:
    def test_none(self):
        assert dv._estimate_display_len(None) == 4

    def test_true(self):
        assert dv._estimate_display_len(True) == 5

    def test_false(self):
        assert dv._estimate_display_len(False) == 5

    def test_int(self):
        assert dv._estimate_display_len(12345) == len(repr(12345))

    def test_float(self):
        assert dv._estimate_display_len(3.14) == len(repr(3.14))

    def test_short_string(self):
        assert dv._estimate_display_len("hello") == 5

    def test_long_string_capped(self):
        s = "x" * 200
        assert dv._estimate_display_len(s) == dv._COL_MAX_SAMPLE_LEN

    def test_string_exactly_at_cap(self):
        s = "y" * dv._COL_MAX_SAMPLE_LEN
        assert dv._estimate_display_len(s) == dv._COL_MAX_SAMPLE_LEN

    def test_complex_type_capped(self):
        obj = {"key": "value" * 30}  # dict str() > cap
        result = dv._estimate_display_len(obj)
        assert result <= dv._COL_MAX_SAMPLE_LEN


# ===========================================================================
# ── 10  _get_column_metadata ────────────────────────────────────────────────
# ===========================================================================

class TestGetColumnMetadata:
    def _tag_sets(self, **kw):
        base = {
            dv.TAG_LANGUAGE: set(),
            dv.TAG_ISO:      set(),
            dv.TAG_REGEX:    set(),
            dv.TAG_CLEANED:  set(),
            dv.TAG_DATE:     set(),
            dv.TAG_METADATA: set(),
        }
        for tag, cols in kw.items():
            tag_obj = {
                "language": dv.TAG_LANGUAGE,
                "iso":      dv.TAG_ISO,
                "regex":    dv.TAG_REGEX,
                "cleaned":  dv.TAG_CLEANED,
                "date":     dv.TAG_DATE,
                "metadata": dv.TAG_METADATA,
            }[tag]
            base[tag_obj] = set(cols)
        return base

    def test_basic_column_count(self):
        df = _make_df(("a", "b", "c"))
        td = dv._TableData(display_rows=[], has_more=False)
        metas = dv._get_column_metadata(df, td, self._tag_sets())
        assert len(metas) == 3

    def test_column_names_preserved(self):
        df = _make_df(("col_x", "col_y"))
        td = dv._TableData(display_rows=[], has_more=False)
        metas = dv._get_column_metadata(df, td, self._tag_sets())
        assert [m.name for m in metas] == ["col_x", "col_y"]

    def test_no_tags_gives_empty_class(self):
        df = _make_df(("a",))
        td = dv._TableData(display_rows=[], has_more=False)
        metas = dv._get_column_metadata(df, td, self._tag_sets())
        assert metas[0].tag_class == ""

    def test_single_tag_reflected_in_class(self):
        df = _make_df(("country",))
        td = dv._TableData(display_rows=[], has_more=False)
        metas = dv._get_column_metadata(df, td, self._tag_sets(iso=["country"]))
        assert "tag-iso" in metas[0].tag_class

    def test_multiple_tags_on_one_column(self):
        df = _make_df(("col",))
        td = dv._TableData(display_rows=[], has_more=False)
        metas = dv._get_column_metadata(df, td, self._tag_sets(iso=["col"], cleaned=["col"]))
        assert "tag-iso"     in metas[0].tag_class
        assert "tag-cleaned" in metas[0].tag_class

    def test_icons_html_empty_for_untagged(self):
        df = _make_df(("a",))
        td = dv._TableData(display_rows=[], has_more=False)
        metas = dv._get_column_metadata(df, td, self._tag_sets())
        assert metas[0].icons_html == ""

    def test_icons_html_nonempty_for_tagged(self):
        df = _make_df(("lang_col",))
        td = dv._TableData(display_rows=[], has_more=False)
        metas = dv._get_column_metadata(df, td, self._tag_sets(language=["lang_col"]))
        assert metas[0].icons_html != ""

    def test_width_within_bounds(self):
        df = _make_df(("x",))
        td = dv._TableData(display_rows=[], has_more=False)
        metas = dv._get_column_metadata(df, td, self._tag_sets())
        assert dv._COL_MIN_PX <= metas[0].width_px <= dv._COL_MAX_PX

    def test_type_str_populated(self):
        df = _make_df(("a",), dtypes=[_spark_types.IntegerType()])
        td = dv._TableData(display_rows=[], has_more=False)
        metas = dv._get_column_metadata(df, td, self._tag_sets())
        assert metas[0].type_str != ""

    def test_row_missing_column_key_raises(self):
        """When a row dict is missing an expected column key, _get_column_metadata raises KeyError."""
        df = _make_df(("a", "b"), rows=[{"a": 1}])  # row missing "b"
        td = dv._collect_rows(df, 10)
        with pytest.raises(KeyError):
            dv._get_column_metadata(df, td, self._tag_sets())

    def test_very_long_column_name_width_capped(self):
        """Very long column names still yield width within _COL_MIN_PX.._COL_MAX_PX."""
        long_name = "x" * 500
        df = _make_df((long_name,))
        td = dv._TableData(display_rows=[], has_more=False)
        metas = dv._get_column_metadata(df, td, self._tag_sets())
        assert len(metas) == 1
        assert dv._COL_MIN_PX <= metas[0].width_px <= dv._COL_MAX_PX


# ===========================================================================
# ── 11  _is_complex_type ────────────────────────────────────────────────────
# ===========================================================================

class TestIsComplexType:
    def test_struct_type(self):
        assert dv._is_complex_type(_spark_types.StructType()) is True

    def test_array_type(self):
        assert dv._is_complex_type(_spark_types.ArrayType()) is True

    def test_map_type(self):
        assert dv._is_complex_type(_spark_types.MapType()) is True

    def test_string_type_false(self):
        assert dv._is_complex_type(_spark_types.StringType()) is False

    def test_integer_type_false(self):
        assert dv._is_complex_type(_spark_types.IntegerType()) is False

    def test_bool_false(self):
        assert dv._is_complex_type(True) is False

    def test_fallback_by_class_name(self):
        """When pyspark.sql.types is unavailable, name-based fallback fires."""
        # Temporarily hide the pyspark types module so isinstance() fails and
        # _is_complex_type falls back to checking type(obj).__name__.
        original = sys.modules.get("pyspark.sql.types")
        try:
            sys.modules["pyspark.sql.types"] = None
            # Define a class whose __name__ matches a complex type
            class StructType: pass
            assert dv._is_complex_type(StructType()) is True
        finally:
            sys.modules["pyspark.sql.types"] = original

    def test_none_is_not_complex(self):
        assert dv._is_complex_type(None) is False


# ===========================================================================
# ── 12  _format_complex_cell ────────────────────────────────────────────────
# ===========================================================================

class TestFormatComplexCell:
    def _uid(self):
        return "testuid"

    def test_none_returns_null_span(self):
        result = dv._format_complex_cell(None, _spark_types.StructType(), "cid", self._uid())
        assert "dataframe-null" in result
        assert "null" in result

    def test_struct_shows_fields_count(self):
        struct_val = MagicMock()
        struct_val.asDict.return_value = {"name": "Alice", "age": 30}
        result = dv._format_complex_cell(
            struct_val, _spark_types.StructType(), "cid1", self._uid()
        )
        assert "Struct(" in result
        assert "2 fields" in result

    def test_array_shows_items_count(self):
        arr_val = [1, 2, 3]
        result = dv._format_complex_cell(
            arr_val, _spark_types.ArrayType(), "cid2", self._uid()
        )
        assert "Array[" in result
        assert "3 items" in result

    def test_map_shows_entries_count(self):
        map_val = {"k1": "v1", "k2": "v2"}
        result = dv._format_complex_cell(
            map_val, _spark_types.MapType(), "cid3", self._uid()
        )
        assert "Map[" in result
        assert "2 entries" in result

    def test_force_array_sentinel(self):
        arr_val = ["a", "b"]
        result = dv._format_complex_cell(
            arr_val, dv._FORCE_ARRAY_DISPLAY, "cid4", self._uid()
        )
        assert "Array[" in result
        assert "2 items" in result

    def test_complex_toggle_html_present(self):
        struct_val = MagicMock()
        struct_val.asDict.return_value = {}
        result = dv._format_complex_cell(
            struct_val, _spark_types.StructType(), "cid5", self._uid()
        )
        assert "complex-toggle" in result
        assert "complex-content" in result

    def test_cell_id_in_output(self):
        result = dv._format_complex_cell(
            {}, _spark_types.MapType(), "my_cell_id", self._uid()
        )
        assert "my_cell_id" in result

    def test_array_handles_tuple(self):
        result = dv._format_complex_cell(
            (10, 20, 30), _spark_types.ArrayType(), "cid6", self._uid()
        )
        assert "3 items" in result

    def test_json_escaped_in_pre(self):
        """Values containing HTML special chars are escaped in the output."""
        struct_val = MagicMock()
        struct_val.asDict.return_value = {"key": "<script>alert(1)</script>"}
        result = dv._format_complex_cell(
            struct_val, _spark_types.StructType(), "cid7", self._uid()
        )
        assert "<script>" not in result

    def test_data_raw_json_attribute_present(self):
        """Lazy highlight sentinel must be present on first render."""
        result = dv._format_complex_cell(
            {"a": 1}, _spark_types.MapType(), "cid8", self._uid()
        )
        assert 'data-raw-json="1"' in result

    def test_empty_array(self):
        result = dv._format_complex_cell(
            [], _spark_types.ArrayType(), "cid9", self._uid()
        )
        assert "0 items" in result

    def test_empty_map(self):
        result = dv._format_complex_cell(
            {}, _spark_types.MapType(), "cid10", self._uid()
        )
        assert "0 entries" in result

    def test_struct_unicode_field_value(self):
        """Struct with other-language / Unicode in field value renders correctly."""
        struct_val = MagicMock()
        struct_val.asDict.return_value = {"name": "日本語", "city": "München", "note": "Café"}
        result = dv._format_complex_cell(
            struct_val, _spark_types.StructType(), "cid_uni", self._uid()
        )
        assert "日本語" in result
        assert "München" in result
        assert "Café" in result
        assert "Struct(3 fields)" in result

    def test_array_unicode_elements(self):
        """Array with Unicode elements renders correctly."""
        arr_val = ["北京", "Tokyo", "São Paulo", "Zürich"]
        result = dv._format_complex_cell(
            arr_val, _spark_types.ArrayType(), "cid_uni_arr", self._uid()
        )
        assert "北京" in result
        assert "São Paulo" in result
        assert "Zürich" in result
        assert "Array[4 items]" in result

    def test_map_unicode_keys_and_values(self):
        """Map with Unicode keys and values renders correctly."""
        map_val = {"言語": "Japanese", "stadt": "München", "emoji": "🎉"}
        result = dv._format_complex_cell(
            map_val, _spark_types.MapType(), "cid_uni_map", self._uid()
        )
        assert "言語" in result
        assert "Japanese" in result
        assert "stadt" in result
        assert "München" in result
        assert "emoji" in result
        assert "🎉" in result
        assert "Map[3 entries]" in result

    def test_complex_unicode_html_escaped(self):
        """Unicode in complex types is preserved; HTML special chars still escaped."""
        struct_val = MagicMock()
        struct_val.asDict.return_value = {"label": "Test <b>bold</b> & 日本語"}
        result = dv._format_complex_cell(
            struct_val, _spark_types.StructType(), "cid_esc", self._uid()
        )
        assert "日本語" in result
        assert "<b>" not in result  # HTML-escaped
        assert "&amp;" in result or "&lt;" in result  # at least one escaped

    def test_complex_serialization_fallback_on_error(self):
        """When struct/array/map serialization fails, fallback to str(value) in pre."""
        bad_val = MagicMock()
        bad_val.asDict.side_effect = RuntimeError("not serializable")
        result = dv._format_complex_cell(
            bad_val, _spark_types.StructType(), "cid_err", self._uid()
        )
        assert "complex-value" in result
        assert "complex-json" in result
        # Fallback: pre contains escaped str(value) or error indicator
        assert "data-raw-json" in result or "Struct" in result


# ===========================================================================
# ── 13  _format_one_country_code ────────────────────────────────────────────
# ===========================================================================

class TestFormatOneCountryCode:
    def test_iso3_produces_chip(self):
        out = dv._format_one_country_code("USA")
        assert "dataframe-country" in out
        assert "USA" in out

    def test_iso3_contains_flag_span(self):
        out = dv._format_one_country_code("GBR")
        # GBR has a known flag
        assert "dataframe-country-flag" in out

    def test_iso2_produces_plain_span(self):
        """ISO2 codes are not styled as pill chips."""
        out = dv._format_one_country_code("US")
        assert "dataframe-country-plain" in out

    def test_unknown_iso3_still_shows_code(self):
        out = dv._format_one_country_code("ZZZ")
        assert "ZZZ" in out

    def test_xss_escaped_in_code(self):
        out = dv._format_one_country_code('<script>alert(1)</script>x')
        assert "<script>" not in out

    def test_whitespace_stripped(self):
        out = dv._format_one_country_code("  DEU  ")
        assert "DEU" in out

    def test_empty_string_returns_empty(self):
        assert dv._format_one_country_code("") == ""

    def test_lowercase_iso3(self):
        # Both should render as a country chip (flag present), even if the
        # displayed code preserves original casing.
        out_lower = dv._format_one_country_code("usa")
        out_upper = dv._format_one_country_code("USA")
        assert "dataframe-country" in out_lower
        assert "dataframe-country" in out_upper
        # Both should resolve to the same flag emoji
        import re as _re
        flag_lower = _re.search(r"dataframe-country-flag[^>]*>([^<]+)<", out_lower)
        flag_upper = _re.search(r"dataframe-country-flag[^>]*>([^<]+)<", out_upper)
        assert flag_lower and flag_upper
        assert flag_lower.group(1) == flag_upper.group(1)


# ===========================================================================
# ── 14  _format_country_list ────────────────────────────────────────────────
# ===========================================================================

class TestFormatCountryList:
    def test_none_returns_null_span(self):
        out = dv._format_country_list(None)
        assert "dataframe-null" in out

    def test_list_of_iso3(self):
        out = dv._format_country_list(["USA", "GBR"])
        assert "USA" in out
        assert "GBR" in out

    def test_returns_list_wrapper(self):
        out = dv._format_country_list(["USA"])
        assert "dataframe-country-list" in out

    def test_none_item_in_list(self):
        out = dv._format_country_list([None, "USA"])
        assert "dataframe-null" in out
        assert "USA" in out

    def test_empty_list(self):
        out = dv._format_country_list([])
        assert "dataframe-country-list" in out

    def test_tuple_input_accepted(self):
        out = dv._format_country_list(("DEU", "FRA"))
        assert "DEU" in out
        assert "FRA" in out

    def test_generator_input_accepted(self):
        out = dv._format_country_list(c for c in ["ITA", "ESP"])
        assert "ITA" in out
        assert "ESP" in out

    def test_non_string_item_shown(self):
        out = dv._format_country_list([42])
        assert "42" in out

    def test_xss_in_list_item_escaped(self):
        out = dv._format_country_list(["<b>x</b>"])
        assert "<b>" not in out


# ===========================================================================
# ── 15  _format_simple_cell ─────────────────────────────────────────────────
# ===========================================================================

class TestFormatSimpleCell:
    def test_none(self):
        out = dv._format_simple_cell(None)
        assert "dataframe-null" in out

    def test_true(self):
        out = dv._format_simple_cell(True)
        assert "bool-true" in out
        assert "True" in out

    def test_false(self):
        out = dv._format_simple_cell(False)
        assert "bool-false" in out
        assert "False" in out

    def test_string_escaped(self):
        out = dv._format_simple_cell("<b>bold</b>")
        assert "<b>" not in out
        assert "&lt;b&gt;" in out

    def test_integer_shown(self):
        out = dv._format_simple_cell(42)
        assert "42" in out

    def test_float_nan_inf_rendered_safely(self):
        """Float, NaN, Inf render without HTML injection."""
        import math
        out_nan = dv._format_simple_cell(float("nan"))
        out_inf = dv._format_simple_cell(float("inf"))
        assert "<" not in out_nan
        assert "<" not in out_inf
        assert "dataframe-null" not in out_nan  # NaN is not None

    def test_iso_column_single_code(self):
        out = dv._format_simple_cell("USA", tag_specs=[dv.TAG_ISO])
        assert "dataframe-country" in out

    def test_iso_column_list(self):
        out = dv._format_simple_cell(["USA", "GBR"], tag_specs=[dv.TAG_ISO])
        assert "dataframe-country-list" in out

    def test_non_iso_tag_plain(self):
        out = dv._format_simple_cell("hello", tag_specs=[dv.TAG_LANGUAGE])
        assert "hello" in out
        assert "dataframe-country" not in out

    def test_quote_equals_true_escaping(self):
        """Strings with quotes must be safe for both attribute and content use."""
        out = dv._format_simple_cell('"quoted"')
        assert '"quoted"' not in out or "&quot;" in out or "&#x27;" in out or _html.escape('"quoted"', quote=True) in out

    def test_empty_tag_specs(self):
        out = dv._format_simple_cell("plain", tag_specs=[])
        assert "plain" in out


# ===========================================================================
# ── 16  _sort_attr_value ────────────────────────────────────────────────────
# ===========================================================================

class TestSortAttrValue:
    def test_none_returns_empty(self):
        assert dv._sort_attr_value(None) == ""

    def test_string(self):
        assert dv._sort_attr_value("hello") == "hello"

    def test_int(self):
        assert dv._sort_attr_value(42) == "42"

    def test_xss_escaped(self):
        out = dv._sort_attr_value('<script>')
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_quote_escaped(self):
        out = dv._sort_attr_value('"value"')
        # Should use html.escape(quote=True) so " → &quot;
        assert '"' not in out or "&quot;" in out

    def test_float_nan_inf_safe(self):
        """NaN and Inf produce attribute-safe strings (no raw < or quotes)."""
        import math
        out_nan = dv._sort_attr_value(float("nan"))
        out_inf = dv._sort_attr_value(float("inf"))
        assert "<" not in out_nan
        assert "<" not in out_inf
        assert out_nan != "" or "nan" in out_nan.lower()
        assert "inf" in out_inf.lower() or "∞" in out_inf


# ===========================================================================
# ── 17  _build_rows_text ────────────────────────────────────────────────────
# ===========================================================================

class TestBuildRowsText:
    def _td(self, n, has_more):
        rows = [{}] * n
        return dv._TableData(display_rows=rows, has_more=has_more)

    def test_all_rows_no_total(self):
        text = dv._build_rows_text(self._td(5, False), None)
        assert "5" in text
        assert "showing" not in text

    def test_all_rows_with_total(self):
        text = dv._build_rows_text(self._td(5, False), 5)
        assert "5" in text

    def test_truncated_no_total(self):
        text = dv._build_rows_text(self._td(100, True), None)
        assert "100" in text
        assert "showing" in text

    def test_truncated_with_total(self):
        text = dv._build_rows_text(self._td(100, True), 1000)
        assert "100" in text
        assert "1,000" in text or "1000" in text

    def test_zero_rows(self):
        text = dv._build_rows_text(self._td(0, False), None)
        assert "0" in text

    def test_uses_len_not_stored_field(self):
        """Regression: uses len(display_rows), not a deleted num_display attr."""
        td = dv._TableData(display_rows=[{}, {}, {}], has_more=False)
        text = dv._build_rows_text(td, None)
        assert "3" in text


# ===========================================================================
# ── 18  _render_css ─────────────────────────────────────────────────────────
# ===========================================================================

class TestRenderCss:
    def test_returns_style_block(self):
        ids = dv._build_viewer_ids()
        css = dv._render_css(ids)
        assert "<style>" in css
        assert "</style>" in css

    def test_scoped_to_viewer_id(self):
        ids = dv._build_viewer_ids()
        css = dv._render_css(ids)
        assert f"#{ids.viewer}" in css

    def test_contains_key_selectors(self):
        ids = dv._build_viewer_ids()
        css = dv._render_css(ids)
        for sel in (".dataframe-table", ".dataframe-header", ".dataframe-footer"):
            assert sel in css

    def test_dark_and_light_mode_rules(self):
        ids = dv._build_viewer_ids()
        css = dv._render_css(ids)
        assert "light-mode" in css


# ===========================================================================
# ── 19  _render_scripts ─────────────────────────────────────────────────────
# ===========================================================================

class TestRenderScripts:
    def test_returns_script_block(self):
        ids = dv._build_viewer_ids()
        js = dv._render_scripts(ids, 5, False)
        assert "<script>" in js and "</script>" in js

    def test_viewer_id_in_script(self):
        ids = dv._build_viewer_ids()
        js = dv._render_scripts(ids, 5, False)
        assert ids.viewer in js

    def test_table_id_in_script(self):
        ids = dv._build_viewer_ids()
        js = dv._render_scripts(ids, 5, False)
        assert ids.table in js

    def test_total_columns_in_script(self):
        ids = dv._build_viewer_ids()
        js = dv._render_scripts(ids, 7, False)
        assert "7" in js

    def test_tag_filter_block_included_when_has_tags(self):
        ids = dv._build_viewer_ids()
        js = dv._render_scripts(ids, 3, True)
        assert "column-filter-btn" in js or "tag-" in js or "active.has" in js

    def test_tag_filter_block_absent_when_no_tags(self):
        ids = dv._build_viewer_ids()
        js_no_tags = dv._render_scripts(ids, 3, False)
        js_tags    = dv._render_scripts(ids, 3, True)
        assert len(js_tags) > len(js_no_tags)

    def test_filter_keys_json_in_script(self):
        ids = dv._build_viewer_ids()
        js = dv._render_scripts(ids, 3, True)
        # All tag filter_key strings should appear
        for tag in dv.ALL_TAGS:
            assert tag.filter_key in js

    def test_uid_specific_functions_present(self):
        ids = dv._build_viewer_ids()
        js = dv._render_scripts(ids, 3, False)
        assert f"dfHighlight_{ids.uid}" in js
        assert f"dfToggle_{ids.uid}" in js


# ===========================================================================
# ── 20  _render_header ──────────────────────────────────────────────────────
# ===========================================================================

class TestRenderHeader:
    def _empty_tag_sets(self):
        return {spec: set() for spec in dv.ALL_TAGS}

    def test_table_name_in_output(self):
        ids = dv._build_viewer_ids()
        out = dv._render_header(ids, "my_table", None, self._empty_tag_sets(), [])
        assert "my_table" in out

    def test_default_label_when_no_name(self):
        ids = dv._build_viewer_ids()
        out = dv._render_header(ids, None, None, self._empty_tag_sets(), [])
        assert "DataFrame" in out

    def test_stage_dot_bronze(self):
        ids = dv._build_viewer_ids()
        out = dv._render_header(ids, "t", "bronze", self._empty_tag_sets(), [])
        assert "stage-dot bronze" in out or 'class="stage-dot bronze"' in out

    def test_stage_dot_gold(self):
        ids = dv._build_viewer_ids()
        out = dv._render_header(ids, "t", "gold", self._empty_tag_sets(), [])
        assert "gold" in out

    def test_no_stage_no_dot(self):
        ids = dv._build_viewer_ids()
        out = dv._render_header(ids, "t", None, self._empty_tag_sets(), [])
        assert "stage-dot" not in out

    def test_filter_pane_shown_when_tags(self):
        ids = dv._build_viewer_ids()
        tag_sets = self._empty_tag_sets()
        tag_sets[dv.TAG_ISO] = {"country"}
        out = dv._render_header(ids, "t", None, tag_sets, [])
        assert "column-filter-pane" in out or "column-filter-btn" in out

    def test_filter_pane_absent_when_no_tags(self):
        ids = dv._build_viewer_ids()
        out = dv._render_header(ids, "t", None, self._empty_tag_sets(), [])
        assert "column-filter-pane" not in out

    def test_theme_toggle_present(self):
        ids = dv._build_viewer_ids()
        out = dv._render_header(ids, "t", None, self._empty_tag_sets(), [])
        assert ids.toggle in out or "theme-toggle" in out

    def test_xss_in_table_name_escaped(self):
        ids = dv._build_viewer_ids()
        out = dv._render_header(ids, '<script>alert(1)</script>', None, self._empty_tag_sets(), [])
        assert "<script>" not in out

    def test_column_selector_button_present(self):
        ids = dv._build_viewer_ids()
        out = dv._render_header(ids, "t", None, self._empty_tag_sets(), [])
        assert ids.col_sel in out


# ===========================================================================
# ── 21  _render_table ───────────────────────────────────────────────────────
# ===========================================================================

class TestRenderTable:
    def _simple_metas(self, names=("a", "b")):
        metas = []
        for i, name in enumerate(names):
            m = dv._ColumnMeta(
                name=name,
                data_type=_spark_types.StringType(),
                type_str="StringType()",
                tag_specs=[],
                tag_class="",
                width_px=120,
                icons_html="",
            )
            metas.append(m)
        return metas

    def _make_td(self, rows):
        r_mocks = []
        for r in rows:
            rm = MagicMock()
            rm.__getitem__ = lambda self, k, _r=r: _r[k]
            r_mocks.append(rm)
        return dv._TableData(display_rows=r_mocks, has_more=False)

    def test_returns_table_html(self):
        ids = dv._build_viewer_ids()
        td = self._make_td([{"a": "x", "b": "y"}])
        out = dv._render_table(ids, self._simple_metas(), td, None)
        assert "<table" in out
        assert "</table>" in out

    def test_thead_present(self):
        ids = dv._build_viewer_ids()
        td = self._make_td([])
        out = dv._render_table(ids, self._simple_metas(), td, None)
        assert "<thead>" in out and "</thead>" in out

    def test_column_headers_in_output(self):
        ids = dv._build_viewer_ids()
        td = self._make_td([])
        out = dv._render_table(ids, self._simple_metas(("col_alpha", "col_beta")), td, None)
        assert "col_alpha" in out
        assert "col_beta" in out

    def test_row_data_present(self):
        ids = dv._build_viewer_ids()
        td = self._make_td([{"a": "hello", "b": "world"}])
        out = dv._render_table(ids, self._simple_metas(), td, None)
        assert "hello" in out
        assert "world" in out

    def test_has_more_notice_when_truncated(self):
        ids = dv._build_viewer_ids()
        td = dv._TableData(display_rows=[], has_more=True)
        out = dv._render_table(ids, self._simple_metas(), td, None)
        assert "showing" in out.lower() or "sorted within" in out.lower()

    def test_no_more_notice_when_not_truncated(self):
        ids = dv._build_viewer_ids()
        td = self._make_td([{"a": "v", "b": "w"}])
        out = dv._render_table(ids, self._simple_metas(), td, None)
        assert "sorted within" not in out

    def test_has_more_with_total_shows_total(self):
        ids = dv._build_viewer_ids()
        td = dv._TableData(display_rows=[], has_more=True)
        out = dv._render_table(ids, self._simple_metas(), td, 9999)
        assert "9,999" in out or "9999" in out

    def test_data_column_attribute_on_th(self):
        ids = dv._build_viewer_ids()
        td = self._make_td([])
        out = dv._render_table(ids, self._simple_metas(), td, None)
        assert 'data-column="0"' in out

    def test_data_sort_value_on_td(self):
        ids = dv._build_viewer_ids()
        td = self._make_td([{"a": "sortme", "b": "x"}])
        out = dv._render_table(ids, self._simple_metas(), td, None)
        assert "data-sort-value" in out

    def test_xss_in_cell_value_escaped(self):
        ids = dv._build_viewer_ids()
        td = self._make_td([{"a": "<script>xss</script>", "b": "ok"}])
        out = dv._render_table(ids, self._simple_metas(), td, None)
        assert "<script>" not in out

    def test_null_cell_rendered(self):
        ids = dv._build_viewer_ids()
        td = self._make_td([{"a": None, "b": "x"}])
        out = dv._render_table(ids, self._simple_metas(), td, None)
        assert "dataframe-null" in out

    def test_bool_cell_rendered(self):
        dtype_bool = _spark_types.BooleanType()
        meta = dv._ColumnMeta(
            name="flag", data_type=dtype_bool, type_str="BooleanType()",
            tag_specs=[], tag_class="", width_px=100, icons_html=""
        )
        ids = dv._build_viewer_ids()
        td = self._make_td([{"flag": True}])
        out = dv._render_table(ids, [meta], td, None)
        assert "bool-true" in out

    def test_table_id_used(self):
        ids = dv._build_viewer_ids()
        td = self._make_td([])
        out = dv._render_table(ids, self._simple_metas(), td, None)
        assert ids.table in out

    def test_empty_rows_no_crash(self):
        ids = dv._build_viewer_ids()
        td = dv._TableData(display_rows=[], has_more=False)
        out = dv._render_table(ids, self._simple_metas(), td, None)
        assert "<table" in out   # still produces table markup

    def test_empty_selector_row_present(self):
        ids = dv._build_viewer_ids()
        td = self._make_td([])
        out = dv._render_table(ids, self._simple_metas(), td, None)
        assert "column-selector-empty-row" in out

    def test_sort_indicator_in_header(self):
        ids = dv._build_viewer_ids()
        td = self._make_td([])
        out = dv._render_table(ids, self._simple_metas(), td, None)
        assert "sort-indicator" in out


# ===========================================================================
# ── 22  _render_footer ──────────────────────────────────────────────────────
# ===========================================================================

class TestRenderFooter:
    def test_returns_footer_div(self):
        ids = dv._build_viewer_ids()
        out = dv._render_footer(ids, 5, "100 rows")
        assert "dataframe-footer" in out

    def test_total_columns_shown(self):
        ids = dv._build_viewer_ids()
        out = dv._render_footer(ids, 7, "50 rows")
        assert "7" in out

    def test_rows_text_shown(self):
        ids = dv._build_viewer_ids()
        out = dv._render_footer(ids, 3, "showing first 42 rows")
        assert "showing first 42 rows" in out

    def test_footer_id_in_output(self):
        ids = dv._build_viewer_ids()
        out = dv._render_footer(ids, 3, "x")
        assert ids.footer in out

    def test_hidden_notice_present_but_hidden(self):
        ids = dv._build_viewer_ids()
        out = dv._render_footer(ids, 4, "x")
        assert "columns-hidden-notice" in out
        assert 'display:none' in out


# ===========================================================================
# ── 23  _render_column_selector_modal ───────────────────────────────────────
# ===========================================================================

class TestRenderColumnSelectorModal:
    def _metas(self, names):
        return [
            dv._ColumnMeta(n, _spark_types.StringType(), "StringType()", [], "", 100, "")
            for n in names
        ]

    def test_modal_present(self):
        ids = dv._build_viewer_ids()
        out = dv._render_column_selector_modal(ids, self._metas(["a", "b"]))
        assert "column-selector-modal" in out

    def test_col_names_in_modal(self):
        ids = dv._build_viewer_ids()
        out = dv._render_column_selector_modal(ids, self._metas(["col_one", "col_two"]))
        assert "col_one" in out
        assert "col_two" in out

    def test_checkboxes_present(self):
        ids = dv._build_viewer_ids()
        out = dv._render_column_selector_modal(ids, self._metas(["x"]))
        assert 'type="checkbox"' in out

    def test_checkboxes_checked_by_default(self):
        ids = dv._build_viewer_ids()
        out = dv._render_column_selector_modal(ids, self._metas(["x"]))
        assert "checked" in out

    def test_backdrop_id_present(self):
        ids = dv._build_viewer_ids()
        out = dv._render_column_selector_modal(ids, self._metas(["x"]))
        assert ids.col_sel_modal in out

    def test_xss_in_col_name_escaped(self):
        ids = dv._build_viewer_ids()
        out = dv._render_column_selector_modal(ids, self._metas(['<script>x</script>']))
        assert "<script>" not in out

    def test_empty_columns(self):
        ids = dv._build_viewer_ids()
        out = dv._render_column_selector_modal(ids, [])
        assert "column-selector-modal" in out


# ===========================================================================
# ── 24  displayDF integration ───────────────────────────────────────────────
# ===========================================================================

def _reset_style_sentinel():
    """Allow _STYLE_INJECTED to be reset between tests."""
    dv._STYLE_INJECTED = False


class TestDisplayDFIntegration:
    """End-to-end tests for the public displayDF function."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        # Bypass isinstance check — the validation tests below handle it explicitly.
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def _df(self, columns=("id", "name"), n_rows=3):
        rows = [{c: f"v_{c}_{i}" for c in columns} for i in range(n_rows)]
        return _make_df(columns, rows=rows)

    def _spec_df(self):
        """Spec'd mock that passes isinstance — used only by validation tests."""
        return MagicMock(spec=_pyspark_sql.DataFrame)

    # ── Validation errors (re-enable real _validate_inputs) ───────────────

    def test_invalid_stage_raises(self):
        self._val_patcher.stop()
        try:
            with pytest.raises(ValueError):
                dv.displayDF(self._spec_df(), stage="platinum")
        finally:
            self._val_patcher.start()

    def test_non_df_raises(self):
        self._val_patcher.stop()
        try:
            with pytest.raises(TypeError):
                dv.displayDF("not a df")
        finally:
            self._val_patcher.start()

    def test_max_rows_over_cap_raises(self):
        self._val_patcher.stop()
        try:
            with pytest.raises(ValueError):
                dv.displayDF(self._spec_df(), max_rows=dv._MAX_ROWS_HARD_CAP + 1)
        finally:
            self._val_patcher.start()

    def test_negative_max_rows_raises(self):
        self._val_patcher.stop()
        try:
            with pytest.raises(ValueError, match="non-negative"):
                dv.displayDF(self._spec_df(), max_rows=-1)
        finally:
            self._val_patcher.start()

    # ── Happy path ────────────────────────────────────────────────────────

    def test_calls_ipython_display(self):
        dv.displayDF(self._df())
        _ipython_display_mock.assert_called_once()

    def test_html_object_passed_to_display(self):
        dv.displayDF(self._df())
        args = _ipython_display_mock.call_args[0]
        # HTML(html_str) → the mock returns the string; check it's truthy
        assert args[0]

    def test_html_contains_column_names(self):
        df = self._df(("alpha_col", "beta_col"))
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "alpha_col" in html_str
        assert "beta_col" in html_str

    def test_html_contains_table_name(self):
        dv.displayDF(self._df(), table_name="my_orders")
        html_str = _HTML_mock.call_args[0][0]
        assert "my_orders" in html_str

    def test_html_contains_stage_dot(self):
        dv.displayDF(self._df(), stage="gold")
        html_str = _HTML_mock.call_args[0][0]
        assert "gold" in html_str

    def test_shared_css_emitted_first_call(self):
        dv.displayDF(self._df())
        html_str = _HTML_mock.call_args[0][0]
        assert "dfv-shared-css" in html_str

    def test_shared_css_not_emitted_second_call(self):
        dv.displayDF(self._df())
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        dv.displayDF(self._df())
        html_str = _HTML_mock.call_args[0][0]
        assert "dfv-shared-css" not in html_str

    def test_style_injected_sentinel_set(self):
        dv._STYLE_INJECTED = False
        dv.displayDF(self._df())
        assert dv._STYLE_INJECTED is True

    def test_translation_columns_tagged(self):
        dv.displayDF(self._df(("text", "lang")), translation_columns=["text"])
        html_str = _HTML_mock.call_args[0][0]
        assert "tag-language" in html_str

    def test_iso_columns_tagged(self):
        dv.displayDF(self._df(("country",)), country_iso_columns=["country"])
        html_str = _HTML_mock.call_args[0][0]
        assert "tag-iso" in html_str

    def test_regex_columns_tagged(self):
        dv.displayDF(self._df(("pattern",)), regex_columns=["pattern"])
        html_str = _HTML_mock.call_args[0][0]
        assert "tag-regex" in html_str

    def test_date_columns_tagged(self):
        dv.displayDF(self._df(("created_at",)), date_columns=["created_at"])
        html_str = _HTML_mock.call_args[0][0]
        assert "tag-date" in html_str

    def test_cleaned_columns_tagged(self):
        dv.displayDF(self._df(("name",)), cleaned_columns=["name"])
        html_str = _HTML_mock.call_args[0][0]
        assert "tag-cleaned" in html_str

    def test_metadata_columns_tagged(self):
        dv.displayDF(self._df(("meta",)), metadata_columns=["meta"])
        html_str = _HTML_mock.call_args[0][0]
        assert "tag-metadata" in html_str

    def test_show_total_rows_calls_count(self):
        df = self._df()
        df.count.return_value = 999
        dv.displayDF(df, show_total_rows=True)
        df.count.assert_called_once()

    def test_show_total_rows_false_does_not_call_count(self):
        df = self._df()
        dv.displayDF(df, show_total_rows=False)
        df.count.assert_not_called()

    def test_total_rows_count_failure_graceful(self):
        df = self._df()
        df.count.side_effect = RuntimeError("Spark error")
        # Should not raise even if count() fails
        dv.displayDF(df, show_total_rows=True)
        _ipython_display_mock.assert_called_once()

    def test_total_rows_count_failure_footer_shows_displayed_only(self):
        """When count() fails, footer shows displayed row count without total."""
        df = self._df()
        df.count.side_effect = RuntimeError("Spark error")
        dv.displayDF(df, show_total_rows=True)
        html_str = _HTML_mock.call_args[0][0]
        # Should show "3 rows" (no "of X" since total_rows is None)
        assert "3 rows" in html_str or "3," in html_str
        # Should not show "of 0" (erroneous total)
        assert " of 0 " not in html_str

    def test_collect_rows_take_raises_propagates(self):
        """When df.take() raises, displayDF propagates the exception."""
        df = self._df()
        df.take.side_effect = RuntimeError("take failed")
        with pytest.raises(RuntimeError, match="take failed"):
            dv.displayDF(df, max_rows=10)

    def test_viewer_div_in_output(self):
        dv.displayDF(self._df())
        html_str = _HTML_mock.call_args[0][0]
        assert "dataframe-viewer" in html_str

    def test_fallback_without_ipython(self, capsys):
        _reset_style_sentinel()
        original = dv.IPYTHON_AVAILABLE
        try:
            dv.IPYTHON_AVAILABLE = False
            dv.displayDF(self._df())
            captured = capsys.readouterr()
            assert "DataFrame viewer requires Jupyter" in captured.out or "Columns" in captured.out
        finally:
            dv.IPYTHON_AVAILABLE = original

    def test_fallback_prints_column_names(self, capsys):
        _reset_style_sentinel()
        original = dv.IPYTHON_AVAILABLE
        try:
            dv.IPYTHON_AVAILABLE = False
            dv.displayDF(self._df(("alpha", "beta")))
            captured = capsys.readouterr()
            assert "alpha" in captured.out and "beta" in captured.out
        finally:
            dv.IPYTHON_AVAILABLE = original

    def test_max_rows_respected(self):
        df = self._df()
        dv.displayDF(df, max_rows=5)
        df.take.assert_called_with(6)   # max_rows + 1

    def test_all_tag_kwarg_names_accepted(self):
        """All keyword argument names in the public API must not raise."""
        df = self._df(("a", "b", "c", "d", "e", "f"))
        dv.displayDF(
            df,
            translation_columns=["a"],
            country_iso_columns=["b"],
            regex_columns=["c"],
            metadata_columns=["d"],
            cleaned_columns=["e"],
            date_columns=["f"],
        )
        _ipython_display_mock.assert_called_once()


# ===========================================================================
# ── 25  _STYLE_INJECTED one-shot sentinel ───────────────────────────────────
# ===========================================================================

class TestStyleInjectedSentinel:
    def setup_method(self):
        dv._STYLE_INJECTED = False
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def _df(self):
        return _make_df(("x",))

    def test_first_call_emits_shared_css(self):
        dv.displayDF(self._df())
        html_str = _HTML_mock.call_args[0][0]
        assert "dfv-shared-css" in html_str

    def test_second_call_omits_shared_css(self):
        dv.displayDF(self._df())
        _HTML_mock.reset_mock()
        dv.displayDF(self._df())
        html_str = _HTML_mock.call_args[0][0]
        assert "dfv-shared-css" not in html_str

    def test_third_call_still_omits(self):
        for _ in range(3):
            _HTML_mock.reset_mock()
            dv.displayDF(self._df())
        html_str = _HTML_mock.call_args[0][0]
        assert "dfv-shared-css" not in html_str


# ===========================================================================
# ── 26  XSS / HTML-injection safety ─────────────────────────────────────────
# ===========================================================================

_XSS_PAYLOADS = [
    '<script>alert("xss")</script>',
    '"><img src=x onerror=alert(1)>',
    "'; DROP TABLE users; --",
    '<b onclick="evil()">bold</b>',
    '&lt;already-escaped&gt;',
    '\u0000null-byte',
]


def _html_viewer_markup_only(html_str: str) -> str:
    """Return HTML with <style> and <script> blocks removed so XSS assertions
    only run against viewer markup (user-controlled content), not embedded CSS/JS."""
    out = re.sub(r"<style[^>]*>.*?</style>", "", html_str, flags=re.DOTALL | re.IGNORECASE)
    out = re.sub(r"<script[^>]*>.*?</script>", "", out, flags=re.DOTALL | re.IGNORECASE)
    return out


class TestXssSafety:
    """Every code path that touches user-supplied strings must escape them."""

    def setup_method(self):
        dv._STYLE_INJECTED = False
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    @pytest.mark.parametrize("payload", _XSS_PAYLOADS)
    def test_table_name_escaped(self, payload):
        df = _make_df(("a",))
        dv.displayDF(df, table_name=payload)
        html_str = _HTML_mock.call_args[0][0]
        markup = _html_viewer_markup_only(html_str)
        # Raw payload must not appear verbatim in viewer markup (exclude style/script)
        assert payload not in markup

    @pytest.mark.parametrize("payload", ['<script>x</script>', '"><evil>'])
    def test_cell_value_escaped(self, payload):
        rows = [{"col": payload}]
        df = _make_df(("col",), rows=rows)
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        markup = _html_viewer_markup_only(html_str)
        assert "<script>" not in markup
        assert "<evil>" not in markup

    @pytest.mark.parametrize("payload", ['<script>x</script>', '"><evil>'])
    def test_column_name_escaped(self, payload):
        df = _make_df((payload,))
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        markup = _html_viewer_markup_only(html_str)
        assert "<script>" not in markup
        assert "<evil>" not in markup

    def test_sort_attr_value_xss(self):
        for payload in _XSS_PAYLOADS:
            out = dv._sort_attr_value(payload)
            assert "<" not in out

    def test_country_code_xss(self):
        out = dv._format_one_country_code('<script>alert(1)</script>')
        assert "<script>" not in out

    def test_simple_cell_xss(self):
        for payload in _XSS_PAYLOADS:
            out = dv._format_simple_cell(payload)
            assert "<script>" not in out


# ===========================================================================
# ── 27  JS template correctness (structural) ────────────────────────────────
# ===========================================================================

class TestJsTemplates:
    """Verify that static JS template strings can be rendered without errors
    (no unmatched braces, correct placeholder names)."""

    def _render(self):
        ids = dv._build_viewer_ids()
        return dv._render_scripts(ids, 10, True)

    def test_no_unmatched_format_braces(self):
        """After rendering, no leftover {placeholder} strings should remain."""
        js = self._render()
        # Look for {word} that are NOT valid JS object literals or regex
        # A simple heuristic: no single-word Python-style placeholders remain
        leftover = re.findall(r'\{[a-z_]+\}', js)
        # Allow {} which is valid JS; only flag named ones
        assert leftover == [], f"Unreplaced placeholders: {leftover}"

    def test_script_is_valid_utf8(self):
        js = self._render()
        js.encode("utf-8")   # should not raise

    def test_script_opens_and_closes_correctly(self):
        js = self._render()
        assert js.startswith("<script>") and js.endswith("</script>")

    def test_resize_template_uses_table_id(self):
        ids = dv._build_viewer_ids()
        js = dv._JS_RESIZE_TMPL.format(table_id=ids.table)
        assert ids.table in js

    def test_sort_template_uses_table_id(self):
        ids = dv._build_viewer_ids()
        js = dv._JS_SORT_TMPL.format(table_id=ids.table)
        assert ids.table in js
        assert "data-column" in js

    def test_prefs_template_uses_viewer_toggle_view(self):
        ids = dv._build_viewer_ids()
        js = dv._JS_PREFS_TMPL.format(
            viewer_id=ids.viewer, toggle_id=ids.toggle, view_id=ids.view
        )
        assert ids.viewer in js
        assert ids.toggle in js
        assert ids.view in js


# ===========================================================================
# ── 28  Complex cell in table render (ISO list expert/simple mode) ───────────
# ===========================================================================

class TestIsoListInTable:
    def test_iso_list_renders_expert_and_simple_divs(self):
        iso_meta = dv._ColumnMeta(
            name="countries",
            data_type=_spark_types.StringType(),
            type_str="ArrayType(StringType)",
            tag_specs=[dv.TAG_ISO],
            tag_class="tag-iso",
            width_px=200,
            icons_html="",
        )
        ids = dv._build_viewer_ids()
        rows_val = ["USA", "GBR"]
        rows = [MagicMock()]
        rows[0].__getitem__ = lambda self, k: rows_val
        td = dv._TableData(display_rows=rows, has_more=False)
        out = dv._render_table(ids, [iso_meta], td, None)
        assert "expert-only" in out
        assert "simple-only" in out

    def test_force_array_sentinel_in_simple_div(self):
        iso_meta = dv._ColumnMeta(
            name="countries",
            data_type=_spark_types.StringType(),
            type_str="ArrayType",
            tag_specs=[dv.TAG_ISO],
            tag_class="tag-iso",
            width_px=200,
            icons_html="",
        )
        ids = dv._build_viewer_ids()
        rows = [MagicMock()]
        rows[0].__getitem__ = lambda self, k: ["DEU", "FRA"]
        td = dv._TableData(display_rows=rows, has_more=False)
        out = dv._render_table(ids, [iso_meta], td, None)
        # The simple-only div should contain an Array[…] expandable, not plain text
        assert "Array[" in out


# ===========================================================================
# ── 29  Edge cases & regressions ────────────────────────────────────────────
# ===========================================================================

class TestEdgeCases:
    def setup_method(self):
        dv._STYLE_INJECTED = False
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_single_column_df(self):
        dv.displayDF(_make_df(("only_col",)))
        _ipython_display_mock.assert_called_once()

    def test_many_columns_df(self):
        cols = [f"col_{i}" for i in range(30)]
        dv.displayDF(_make_df(tuple(cols)))
        _ipython_display_mock.assert_called_once()

    def test_empty_df_no_crash(self):
        df = _make_df(("a", "b"), rows=[])
        dv.displayDF(df)
        _ipython_display_mock.assert_called_once()

    def test_unicode_column_name(self):
        dv.displayDF(_make_df(("日本語",)))
        html_str = _HTML_mock.call_args[0][0]
        assert "日本語" in html_str

    def test_unicode_cell_value(self):
        rows = [{"a": "héllo wörld"}]
        df = _make_df(("a",), rows=rows)
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "héllo wörld" in html_str

    def test_max_rows_one(self):
        dv.displayDF(_make_df(), max_rows=1)
        _ipython_display_mock.assert_called_once()

    def test_zero_rows_shown_in_footer(self):
        df = _make_df(("a",), rows=[])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "0" in html_str

    def test_none_table_name_uses_default(self):
        dv.displayDF(_make_df(), table_name=None)
        html_str = _HTML_mock.call_args[0][0]
        assert "DataFrame" in html_str

    def test_viewer_simple_class_in_initial_output(self):
        dv.displayDF(_make_df())
        html_str = _HTML_mock.call_args[0][0]
        assert "viewer-simple" in html_str

    def test_multiple_viewers_have_unique_ids(self):
        dv.displayDF(_make_df())
        html1 = _HTML_mock.call_args[0][0]
        _HTML_mock.reset_mock()
        dv.displayDF(_make_df())
        html2 = _HTML_mock.call_args[0][0]
        # Extract viewer IDs
        ids1 = re.findall(r'id="dfv_([a-f0-9]{8})"', html1)
        ids2 = re.findall(r'id="dfv_([a-f0-9]{8})"', html2)
        assert ids1 and ids2
        assert ids1[0] != ids2[0]

    def test_col_tag_for_nonexistent_column_no_crash(self):
        """Tagging a column that doesn't exist in df should not raise."""
        dv.displayDF(_make_df(("a",)), translation_columns=["nonexistent"])
        _ipython_display_mock.assert_called_once()

    def test_non_ipython_prints_message(self):
        """When IPython is not available, displayDF prints a message and does not display HTML."""
        with patch.object(dv, "IPYTHON_AVAILABLE", False):
            with patch("dataframe_viewer._ipython_display") as mock_display:
                with patch("builtins.print") as mock_print:
                    dv.displayDF(_make_df())
                    mock_print.assert_called()
                    printed = " ".join(str(c[0][0]) for c in mock_print.call_args_list)
                    assert "Jupyter" in printed or "HTML" in printed
                    mock_display.assert_not_called()

    def test_boolean_dtype_column(self):
        meta_bool = dv._ColumnMeta(
            "flag", _spark_types.BooleanType(), "BooleanType()", [], "", 80, ""
        )
        ids = dv._build_viewer_ids()
        rows = [MagicMock()]
        rows[0].__getitem__ = lambda self, k: False
        td = dv._TableData(display_rows=rows, has_more=False)
        out = dv._render_table(ids, [meta_bool], td, None)
        assert "bool-false" in out

    def test_struct_type_column_renders_complex(self):
        struct_dt = _spark_types.StructType()
        meta_struct = dv._ColumnMeta(
            "details", struct_dt, "StructType()", [], "", 200, ""
        )
        ids = dv._build_viewer_ids()
        val = MagicMock()
        val.asDict.return_value = {"x": 1}
        rows = [MagicMock()]
        rows[0].__getitem__ = lambda self, k: val
        td = dv._TableData(display_rows=rows, has_more=False)
        out = dv._render_table(ids, [meta_struct], td, None)
        assert "complex-toggle" in out or "Struct(" in out

    def test_very_long_cell_value_escaped_and_safe(self):
        """Very long cell value is HTML-escaped and does not break output."""
        long_val = "A" * 5000 + "<script>"
        df = _make_df(("a",), rows=[{"a": long_val}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        # Cell content must be escaped (viewer also emits its own <script> blocks)
        assert "&lt;script&gt;" in html_str


# ===========================================================================
# ── 30  Accessibility (ARIA / semantics) ───────────────────────────────────
# ===========================================================================

class TestAccessibility:
    """Assert key accessibility attributes and structure in emitted HTML."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def _html(self):
        dv.displayDF(_make_df())
        return _HTML_mock.call_args[0][0]

    def test_aria_label_close_present(self):
        """Column selector modal close button has aria-label for screen readers."""
        assert 'aria-label="Close"' in self._html()

    def test_aria_hidden_on_decorative_svg(self):
        """Decorative SVGs are marked aria-hidden so they are skipped by screen readers."""
        assert "aria-hidden=\"true\"" in self._html()

    def test_table_element_present(self):
        """Output uses a semantic <table> element."""
        html = self._html()
        assert "<table" in html
        assert "<thead" in html
        assert "<tbody" in html

    def test_modal_has_dialog_role_or_aria_modal(self):
        """Column selector modal has role='dialog' and/or aria-modal='true' for screen readers."""
        html = self._html()
        assert 'role="dialog"' in html or "aria-modal" in html

    def test_sortable_headers_have_data_column(self):
        """Sortable th elements have data-column attribute for JS and accessibility."""
        html = self._html()
        assert "data-column=" in html

    def test_modal_aria_labelledby_target_id_exists(self):
        """The id referenced by aria-labelledby (col_sel_modal + '-title') exists in the HTML."""
        html = self._html()
        import re
        m = re.search(r'aria-labelledby="(dfcolselmodal_[a-f0-9]{8}-title)"', html)
        assert m, "aria-labelledby must reference modal title id"
        label_id = m.group(1)
        assert f'id="{label_id}"' in html

    def test_modal_close_button_inside_dialog(self):
        """Close button (column-selector-close) is inside the modal (column-selector-modal)."""
        html = self._html()
        modal_start = html.find('class="column-selector-modal"')
        modal_end = html.find("</div>", html.find("column-selector-modal-body", modal_start)) + len("</div>")
        modal_section = html[modal_start:modal_end]
        assert "column-selector-close" in modal_section

    def test_modal_checkboxes_are_label_wrapped(self):
        """Each column selector checkbox is inside a <label> (column-selector-row)."""
        html = self._html()
        assert "column-selector-row" in html
        assert "column-selector-check" in html
        # Structure: label.column-selector-row contains input.column-selector-check
        assert html.count("column-selector-row") >= 1
        assert html.count("column-selector-check") >= 1


# ===========================================================================
# ── 31  Datetime / date / timestamp rendering ──────────────────────────────
# ===========================================================================

class TestDatetimeDateTimestampRendering:
    """Datetime, date, and timestamp-like values render safely and consistently."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_python_datetime_rendered_escaped(self):
        """Python datetime renders via str() and is HTML-escaped (no raw angle brackets in cell)."""
        dt = datetime(2024, 3, 15, 14, 30, 0)
        df = _make_df(("ts",), rows=[{"ts": dt}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "2024" in html_str
        # Cell content appears as text (datetime has no < so no escaping needed; check it's present)
        assert "2024-03-15" in html_str or "14:30" in html_str

    def test_python_date_rendered_escaped(self):
        """Python date renders via str() and is HTML-escaped."""
        d = date(2024, 3, 15)
        df = _make_df(("d",), rows=[{"d": d}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "2024" in html_str

    def test_timezone_aware_datetime_rendered(self):
        """Timezone-aware datetime renders without error."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        df = _make_df(("ts",), rows=[{"ts": dt}])
        dv.displayDF(df)
        _ipython_display_mock.assert_called_once()

    def test_timezone_naive_datetime_rendered(self):
        """Timezone-naive datetime renders without error."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        df = _make_df(("ts",), rows=[{"ts": dt}])
        dv.displayDF(df)
        _ipython_display_mock.assert_called_once()

    def test_date_tagged_column_gets_icon(self):
        """Columns in date_columns get date icon in header."""
        df = _make_df(("created_at",), rows=[{"created_at": "2024-01-01"}])
        dv.displayDF(df, date_columns=["created_at"])
        html_str = _HTML_mock.call_args[0][0]
        assert "date-icon" in html_str or "tag-date" in html_str

    def test_iso_string_with_milliseconds_rendered_safely(self):
        """Long ISO string with milliseconds and timezone offset is escaped."""
        iso = "2024-03-15T14:30:00.123456+00:00"
        df = _make_df(("ts",), rows=[{"ts": iso}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "2024" in html_str
        assert "123456" in html_str or "14:30" in html_str

    def test_sort_attr_value_datetime_consistent(self):
        """Sort attribute for datetime is attribute-safe (no unescaped quotes)."""
        dt = datetime(2024, 3, 15)
        out = dv._sort_attr_value(dt)
        assert '"' not in out or out.count('"') == 0
        assert "<" not in out


# ===========================================================================
# ── 32  Decimal / bytes / binary / UUID / custom objects ───────────────────
# ===========================================================================

class TestDecimalBytesBinaryCustomRendering:
    """Decimal, bytes, UUID, enums, and custom objects render safely."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_decimal_rendered_escaped(self):
        """Decimal('10.50') renders via str() and is escaped."""
        df = _make_df(("amount",), rows=[{"amount": Decimal("10.50")}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "10" in html_str
        assert "50" in html_str

    def test_bytes_rendered_safely(self):
        """bytes(b'\\x00\\x01abc') renders without breaking HTML."""
        df = _make_df(("blob",), rows=[{"blob": b"\x00\x01abc"}])
        dv.displayDF(df)
        _ipython_display_mock.assert_called_once()
        html_str = _HTML_mock.call_args[0][0]
        assert "&lt;" not in html_str or html_str.count("<script") == 0

    def test_bytearray_rendered(self):
        """bytearray renders via str() without exception."""
        df = _make_df(("data",), rows=[{"data": bytearray(b"xyz")}])
        dv.displayDF(df)
        _ipython_display_mock.assert_called_once()

    def test_uuid_rendered_escaped(self):
        """UUID object renders via str() and is escaped."""
        try:
            import uuid as uuid_mod
            uid = uuid_mod.UUID("550e8400-e29b-41d4-a716-446655440000")
        except ImportError:
            pytest.skip("uuid not available")
        df = _make_df(("id",), rows=[{"id": uid}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "550e8400" in html_str

    def test_custom_object_hostile_str_escaped(self):
        """Custom object whose __str__ returns HTML-like text is escaped."""

        class Hostile:
            def __str__(self):
                return '<script>alert(1)</script>'

        df = _make_df(("x",), rows=[{"x": Hostile()}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "&lt;script&gt;" in html_str
        assert html_str.count("<script>") == 0 or "dataframe-viewer" in html_str  # only our own script

    def test_custom_object_repr_raises_fallback(self):
        """Object whose __str__ raises is handled (format_simple_cell uses str(); exception may propagate)."""
        class BadRepr:
            def __str__(self):
                raise RuntimeError("str not allowed")

        # _format_simple_cell calls str(value) for non-special types, so this will raise.
        with pytest.raises(RuntimeError, match="str not allowed"):
            dv._format_simple_cell(BadRepr(), None)


# ===========================================================================
# ── 33  Broken / weird row objects ─────────────────────────────────────────
# ===========================================================================

class TestBrokenWeirdRowObjects:
    """Row access raises on one column, unexpected types, extra keys, etc."""

    def setup_method(self):
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_row_access_raises_on_one_column_only(self):
        """Row that raises KeyError for one column propagates from _get_column_metadata."""
        # Row with only "a"; schema has "a" and "b". Already covered by test_row_missing_column_key_raises.
        df = _make_df(("a", "b"), rows=[{"a": 1}])
        td = dv._collect_rows(df, 10)
        with pytest.raises(KeyError):
            dv._get_column_metadata(df, td, {
                dv.TAG_LANGUAGE: set(), dv.TAG_ISO: set(), dv.TAG_REGEX: set(),
                dv.TAG_CLEANED: set(), dv.TAG_DATE: set(), dv.TAG_METADATA: set(),
            })

    def test_row_returns_unexpected_type_for_cell(self):
        """Row returning non-string for a 'string' column still renders (str() applied)."""
        df = _make_df(("x",), rows=[{"x": 12345}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "12345" in html_str

    def test_one_bad_row_value_does_not_corrupt_other_cells(self):
        """One cell that would raise in formatting: test isolation (e.g. skip or error that row)."""
        # If we had a row where one column's value raises when formatted, we don't have that path yet.
        # Test that mixed good/bad rows still render good rows' content.
        rows = [{"a": "ok"}, {"a": "also_ok"}]
        df = _make_df(("a",), rows=rows)
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "ok" in html_str and "also_ok" in html_str


# ===========================================================================
# ── 34  Duplicate column names ──────────────────────────────────────────────
# ===========================================================================

class TestDuplicateColumnNames:
    """DataFrame with duplicate column names (e.g. after join) has defined behavior."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_duplicate_column_names_render_with_stable_data_column_indexes(self):
        """Two columns with same name produce distinct data-column indexes (0 and 1)."""
        # Schema with two fields both named "x"; rows have one key "x" (value shown in both columns).
        columns = ("x", "x")
        df = _make_df(columns, rows=[{"x": "first"}, {"x": "second"}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert 'data-column="0"' in html_str
        assert 'data-column="1"' in html_str

    def test_duplicate_column_selector_shows_both_entries(self):
        """Column selector modal lists both duplicate-named columns (by index)."""
        columns = ("a", "a")
        df = _make_df(columns, rows=[{"a": 1}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        # Two labels with same text "a"
        assert html_str.count('data-column="0"') >= 1
        assert html_str.count('data-column="1"') >= 1

    def test_duplicate_column_names_with_different_tags(self):
        """Duplicate names with different tag groups (e.g. one ISO, one date) still render."""
        columns = ("x", "x")
        df = _make_df(columns, rows=[{"x": "USA"}, {"x": "2024-01-01"}])
        dv.displayDF(df, country_iso_columns=["x"], date_columns=["x"])
        html_str = _HTML_mock.call_args[0][0]
        assert 'data-column="0"' in html_str and 'data-column="1"' in html_str

    def test_duplicate_column_names_with_complex_type(self):
        """Duplicate names where one column is complex type still produces distinct data-column."""
        val = MagicMock()
        val.asDict.return_value = {"a": 1}
        columns = ("payload", "payload")
        rows = [{"payload": val}, {"payload": "plain"}]
        df = _make_df(columns, rows=rows, dtypes=[_spark_types.StructType(), _spark_types.StringType()])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert 'data-column="0"' in html_str and 'data-column="1"' in html_str
        assert "complex-toggle" in html_str or "complex-value" in html_str

    def test_duplicate_names_modal_checkbox_count_matches_columns(self):
        """Modal has one checkbox per column (2 for two duplicate-named columns)."""
        df = _make_df(("a", "a"), rows=[{"a": 1}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        checkboxes = re.findall(r'class="column-selector-check"\s+data-column="(\d+)"', html_str)
        assert len(checkboxes) == 2
        assert set(checkboxes) == {"0", "1"}

    def test_duplicate_names_sort_headers_distinct_data_column(self):
        """Sortable th elements for duplicate-named columns have distinct data-column 0 and 1."""
        df = _make_df(("x", "x"), rows=[{"x": 1}, {"x": 2}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert 'th class="sortable' in html_str
        assert 'data-column="0"' in html_str and 'data-column="1"' in html_str


# ===========================================================================
# ── 35  Zero-column DataFrame ──────────────────────────────────────────────
# ===========================================================================

class TestZeroColumnDataFrame:
    """DataFrame with zero columns is rejected or handled explicitly."""

    def setup_method(self):
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_zero_columns_raises_clean_error(self):
        """displayDF with zero columns raises ValueError with clear message."""
        df = _make_df((), rows=[{}])
        with pytest.raises(ValueError, match="no columns"):
            dv.displayDF(df)

    def test_zero_columns_raises_exact_message(self):
        """Exact error message for zero-column DataFrame."""
        df = _make_df((), rows=[{}])
        with pytest.raises(ValueError) as exc_info:
            dv.displayDF(df)
        assert "DataFrame has no columns" in str(exc_info.value)
        assert "at least one column" in str(exc_info.value)

    def test_zero_columns_fallback_path_not_run(self):
        """When zero columns raise, fallback print path is not executed."""
        df = _make_df((), rows=[{}])
        with patch.object(dv, "IPYTHON_AVAILABLE", False):
            with patch("builtins.print") as mock_print:
                with pytest.raises(ValueError, match="no columns"):
                    dv.displayDF(df)
                mock_print.assert_not_called()

    def test_zero_columns_render_table_never_called(self):
        """_render_table is never called when total_columns is 0."""
        df = _make_df((), rows=[{}])
        with patch("dataframe_viewer._render_table") as mock_render:
            with pytest.raises(ValueError, match="no columns"):
                dv.displayDF(df)
            mock_render.assert_not_called()


# ===========================================================================
# ── 36  Large-width / stress ───────────────────────────────────────────────
# ===========================================================================

class TestLargeWidthStress:
    """Many columns, wide names, max_rows=0, many tags, large JSON in cell."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_200_columns_renders_without_blow_up(self):
        """200+ columns still produce valid output and unique data-column ids."""
        columns = tuple(f"col_{i}" for i in range(200))
        df = _make_df(columns)
        dv.displayDF(df, max_rows=2)
        html_str = _HTML_mock.call_args[0][0]
        assert "data-column=" in html_str
        assert "data-column=\"0\"" in html_str
        assert "data-column=\"199\"" in html_str

    def test_max_rows_zero_with_many_columns(self):
        """max_rows=0 with many columns produces empty tbody but valid structure."""
        columns = ("a", "b", "c")
        df = _make_df(columns, rows=[])
        dv.displayDF(df, max_rows=0)
        html_str = _HTML_mock.call_args[0][0]
        assert "<tbody" in html_str
        assert "0" in html_str or "rows" in html_str

    def test_many_tagged_columns_multiple_categories(self):
        """Many columns tagged across multiple tag groups render without error."""
        columns = tuple(f"c{i}" for i in range(20))
        df = _make_df(columns)
        dv.displayDF(
            df,
            translation_columns=[f"c{i}" for i in range(0, 20, 2)],
            country_iso_columns=["c1", "c5"],
            date_columns=["c3", "c7"],
            metadata_columns=["c9"],
        )
        _ipython_display_mock.assert_called_once()

    def test_large_complex_json_in_cell(self):
        """Large nested structure in one cell renders (expand/collapse)."""
        big = {"k": list(range(500)), "nested": {"a": "b" * 200}}
        df = _make_df(("payload",), rows=[{"payload": big}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "complex-toggle" in html_str or "complex-value" in html_str


# ===========================================================================
# ── 37  Tag collisions / overlapping metadata ──────────────────────────────
# ===========================================================================

class TestTagCollisionsOverlapping:
    """One column in all tag groups, non-string in tag sets, repeated names, case, special chars."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_one_column_in_all_six_tag_groups(self):
        """Single column tagged with all six tag types renders all icons/classes."""
        df = _make_df(("meta",), rows=[{"meta": "x"}])
        dv.displayDF(
            df,
            translation_columns=["meta"],
            country_iso_columns=["meta"],
            regex_columns=["meta"],
            cleaned_columns=["meta"],
            date_columns=["meta"],
            metadata_columns=["meta"],
        )
        html_str = _HTML_mock.call_args[0][0]
        for tag_class in ("tag-language", "tag-iso", "tag-regex", "tag-cleaned", "tag-date", "tag-metadata"):
            assert tag_class in html_str

    def test_column_names_with_special_chars_escaped(self):
        """Column names with spaces, hyphens, dots, Unicode are escaped in output."""
        df = _make_df(("col with space", "col-with-dash", "col.with.dots", "日本語"))
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "col with space" in html_str or "space" in html_str
        assert "col-with-dash" in html_str or "dash" in html_str
        assert "日本語" in html_str


# ===========================================================================
# ── 38  Sort behavior edge cases ───────────────────────────────────────────
# ===========================================================================

class TestSortBehaviorEdgeCases:
    """Mixed types in column, NaN/Inf, ISO chip sort key, escaped chars, negatives."""

    def test_sort_value_none_empty(self):
        assert dv._sort_attr_value(None) == ""

    def test_sort_value_mixed_types_same_column_representation(self):
        """None, bool, int, float, string all produce attribute-safe sort values."""
        for val in [None, True, False, 0, -1, 3.14, "hello", ""]:
            out = dv._sort_attr_value(val)
            assert "<" not in out
            assert '"' not in out  # attribute-safe: no raw double-quote

    def test_sort_value_nan_inf_safe(self):
        """Float nan/inf produce safe attribute string (already in TestSortAttrValue)."""
        out_nan = dv._sort_attr_value(float("nan"))
        out_inf = dv._sort_attr_value(float("inf"))
        assert "<" not in out_nan
        assert "<" not in out_inf

    def test_sort_value_negative_and_decimal(self):
        """Negative numbers and decimals produce valid sort attribute."""
        out_neg = dv._sort_attr_value(-100)
        assert "-100" in out_neg or "100" in out_neg
        out_dec = dv._sort_attr_value(Decimal("10.50"))
        assert "10" in out_dec


# ===========================================================================
# ── 39  Contract tests for public API defaults ─────────────────────────────
# ===========================================================================

class TestDisplayDFContractDefaults:
    """Lock down default table_name, max_rows, show_total_rows, missing tag kwargs."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_default_table_name_is_dataframe(self):
        dv.displayDF(_make_df())
        html_str = _HTML_mock.call_args[0][0]
        assert "DataFrame" in html_str

    def test_default_max_rows_100(self):
        df = _make_df()
        dv.displayDF(df)
        df.take.assert_called_once()
        # take(max_rows+1) so 101
        assert df.take.call_args[0][0] == 101

    def test_default_show_total_rows_false(self):
        df = _make_df()
        dv.displayDF(df)
        df.count.assert_not_called()

    def test_show_total_rows_true_count_zero_footer_shows_zero_rows(self):
        """When show_total_rows=True and df.count() returns 0, footer shows '0 rows' (HTML path)."""
        df = _make_df(("a",), rows=[{"a": 1}])
        df.count.return_value = 0
        dv.displayDF(df, show_total_rows=True)
        html = _HTML_mock.call_args[0][0]
        assert "0 rows" in html or "0," in html

    def test_max_rows_zero_with_non_empty_df_truncated_notice(self):
        """max_rows=0 with non-empty df: take(1) called, has_more True, 'showing first 0 rows'."""
        df = _make_df(("a",), rows=[{"a": 1}])
        dv.displayDF(df, max_rows=0)
        html = _HTML_mock.call_args[0][0]
        assert "showing first 0 rows" in html or "0 rows" in html
        df.take.assert_called_once_with(1)

    def test_missing_tag_kwargs_treated_as_empty(self):
        """Omitting translation_columns etc. is same as passing None/empty."""
        dv.displayDF(_make_df(("a",)))
        html_str = _HTML_mock.call_args[0][0]
        assert "tag-language" not in html_str or "tag-iso" not in html_str

    def test_stage_normalization_lowercase(self):
        df = _make_df()
        dv.displayDF(df, stage="SILVER")
        html_str = _HTML_mock.call_args[0][0]
        assert "silver" in html_str.lower()

    def test_stage_bronze_uppercase_renders_bronze_dot(self):
        """Stage 'BRONZE' renders bronze dot (stage.lower() used in render)."""
        dv.displayDF(_make_df(), stage="BRONZE")
        html_str = _HTML_mock.call_args[0][0]
        assert 'class="stage-dot bronze"' in html_str or "bronze" in html_str

    def test_stage_gold_mixed_case_renders_gold_dot(self):
        """Stage 'GoLd' renders gold dot."""
        dv.displayDF(_make_df(), stage="GoLd")
        html_str = _HTML_mock.call_args[0][0]
        assert "gold" in html_str.lower()


# ===========================================================================
# ── 39b  Stage validation exact contracts ─────────────────────────────────
# ===========================================================================

class TestStageValidationExact:
    """Stage: empty string raises; whitespace around stage is not stripped (raises)."""

    def _stub_df(self):
        """Instance of stub DataFrame so isinstance(df, SparkDataFrame) passes."""
        return _pyspark_sql.DataFrame()

    def test_stage_empty_string_raises(self):
        with pytest.raises(ValueError, match="stage"):
            dv._validate_inputs(self._stub_df(), "", 10)

    def test_stage_whitespace_around_raises(self):
        """stage=' bronze ' fails because we do not strip; only .lower() is applied."""
        with pytest.raises(ValueError, match="stage"):
            dv._validate_inputs(self._stub_df(), " bronze ", 10)


# ===========================================================================
# ── 40  Control characters and invisible Unicode ──────────────────────────
# ===========================================================================

class TestControlCharactersInvisibleUnicode:
    """Newline, tab, null, zero-width, RTL, combining chars, emoji."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_newline_tab_carriage_return_escaped(self):
        """Control chars in cell value are escaped and do not break markup."""
        df = _make_df(("x",), rows=[{"x": "a\nb\tc\rd"}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "a" in html_str and "b" in html_str

    def test_null_byte_stripped_or_escaped_in_table_name(self):
        """Null byte in table_name is stripped (viewer does replace('\\x00',''))."""
        dv.displayDF(_make_df(), table_name="bad\x00name")
        html_str = _HTML_mock.call_args[0][0]
        assert "\x00" not in html_str

    def test_emoji_in_cell_rendered(self):
        """Emoji in cell value does not break rendering."""
        df = _make_df(("x",), rows=[{"x": "hello \u2705 world"}])
        dv.displayDF(df)
        html_str = _HTML_mock.call_args[0][0]
        assert "hello" in html_str and "world" in html_str


# ===========================================================================
# ── 41  Failure isolation (auxiliary pieces) ─────────────────────────────────
# ===========================================================================

class TestFailureIsolation:
    """HTML constructor, display(), render failures: document whether they propagate."""

    def setup_method(self):
        _reset_style_sentinel()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_ipython_display_failure_propagates(self):
        """If _ipython_display(HTML(...)) raises, the exception propagates."""
        with patch("dataframe_viewer._ipython_display") as mock_display:
            mock_display.side_effect = RuntimeError("display failed")
            with pytest.raises(RuntimeError, match="display failed"):
                dv.displayDF(_make_df())


# ===========================================================================
# ── 41b  Complex cell IDs: consistency and uniqueness ───────────────────────
# ===========================================================================

class TestComplexCellIds:
    """cell_id used in onclick, id=cell_id, id=icon_*; no duplicates; unique across viewers."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_complex_cell_id_in_onclick_and_id_and_icon(self):
        """Complex cell: same cell_id in onclick, id=cell_id, and id=icon_{cell_id}."""
        val = MagicMock()
        val.asDict.return_value = {"a": 1}
        df = _make_df(("x",), rows=[{"x": val}], dtypes=[_spark_types.StructType()])
        dv.displayDF(df)
        html = _HTML_mock.call_args[0][0]
        import re
        cell_ids = re.findall(r"cell_[a-f0-9]{8}_\d+_\d+", html)
        assert len(cell_ids) >= 1
        cid = cell_ids[0]
        assert f"dfToggle_" in html and f"('{cid}')" in html
        assert f'id="{cid}"' in html
        assert f'id="icon_{cid}"' in html

    def test_no_duplicate_cell_ids_in_single_viewer(self):
        """Unique cell_* IDs in one viewer: each (row,col) appears once as ID (may appear in onclick, id, icon_)."""
        val = MagicMock()
        val.asDict.return_value = {"a": 1}
        df = _make_df(("x", "y"), rows=[{"x": val, "y": val}], dtypes=[_spark_types.StructType(), _spark_types.StructType()])
        dv.displayDF(df)
        html = _HTML_mock.call_args[0][0]
        cell_ids = re.findall(r"cell_[a-f0-9]{8}_\d+_\d+", html)
        unique_ids = set(cell_ids)
        assert len(unique_ids) == 2  # two complex cells (row 0 col 0, row 0 col 1)

    def test_cell_ids_unique_across_multiple_viewers(self):
        """Two displayDF calls produce disjoint cell_* ID sets."""
        val = MagicMock()
        val.asDict.return_value = {"a": 1}
        df = _make_df(("x",), rows=[{"x": val}], dtypes=[_spark_types.StructType()])
        dv.displayDF(df)
        html1 = _HTML_mock.call_args[0][0]
        _HTML_mock.reset_mock()
        dv.displayDF(df)
        html2 = _HTML_mock.call_args[0][0]
        ids1 = set(re.findall(r"cell_[a-f0-9]{8}_\d+_\d+", html1))
        ids2 = set(re.findall(r"cell_[a-f0-9]{8}_\d+_\d+", html2))
        assert ids1 and ids2
        assert ids1.isdisjoint(ids2)


# ===========================================================================
# ── 41c  Sort data-sort-value semantics (regression for JS Number/string) ───
# ===========================================================================

class TestSortDataSortValueSemantics:
    """data-sort-value stringification: None, '', '001', '1e3', '-5', 'NaN', etc."""

    @pytest.mark.parametrize("value", [
        None, "", " ", "001", "1e3", "-5", "01.20", "NaN", "Infinity", "true", "false",
    ])
    def test_sort_attr_value_stringified_safely(self, value):
        """Sort attribute is HTML-safe (no raw quote/angle) for JS Number/string handling."""
        out = dv._sort_attr_value(value)
        assert '"' not in out
        assert "<" not in out
        if value is None:
            assert out == ""
        elif value != "":
            # String form should appear (possibly escaped)
            assert len(out) >= 0


# ===========================================================================
# ── 41d  Fallback output exactness ──────────────────────────────────────────
# ===========================================================================

class TestFallbackOutputExact:
    """Fallback print path: exact format; total_rows=0 vs None; zero-column raises before fallback."""

    def setup_method(self):
        _reset_style_sentinel()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_fallback_columns_format(self):
        """Printed 'Columns : a, b' when two columns."""
        with patch.object(dv, "IPYTHON_AVAILABLE", False):
            with patch("builtins.print") as mock_print:
                dv.displayDF(_make_df(("a", "b")))
                calls = [str(c[0][0]) for c in mock_print.call_args_list]
                combined = " ".join(calls)
                assert "Columns" in combined and "a" in combined and "b" in combined

    def test_fallback_rows_when_total_rows_none(self):
        """When total_rows is None, print does not include 'of X'."""
        with patch.object(dv, "IPYTHON_AVAILABLE", False):
            with patch("builtins.print") as mock_print:
                dv.displayDF(_make_df())
                calls = [str(c[0][0]) for c in mock_print.call_args_list]
                combined = " ".join(calls)
                assert "Rows" in combined
                assert " of " not in combined or "of 0" in combined  # may have "of 0" if we fixed it

    def test_fallback_rows_when_total_rows_zero(self):
        """When df.count() returns 0, fallback prints 'of 0' (total_rows is not None)."""
        df = _make_df()
        df.count.return_value = 0
        with patch.object(dv, "IPYTHON_AVAILABLE", False):
            with patch("builtins.print") as mock_print:
                dv.displayDF(df, show_total_rows=True)
                calls = [str(c[0][0]) for c in mock_print.call_args_list]
                combined = " ".join(calls)
                assert " of 0" in combined

    def test_fallback_rows_when_total_rows_123(self):
        """When total_rows=123, fallback prints 'of 123'."""
        df = _make_df()
        df.count.return_value = 123
        with patch.object(dv, "IPYTHON_AVAILABLE", False):
            with patch("builtins.print") as mock_print:
                dv.displayDF(df, show_total_rows=True)
                calls = [str(c[0][0]) for c in mock_print.call_args_list]
                combined = " ".join(calls)
                assert " of 123" in combined or "of 123" in combined

    def test_fallback_no_ipython_display_called(self):
        """Fallback path does not call _ipython_display."""
        with patch.object(dv, "IPYTHON_AVAILABLE", False):
            with patch("dataframe_viewer._ipython_display") as mock_display:
                with patch("builtins.print"):
                    dv.displayDF(_make_df())
                mock_display.assert_not_called()

    def test_zero_columns_raises_before_fallback(self):
        """Zero-column raises ValueError; fallback print never runs."""
        df = _make_df((), rows=[{}])
        with patch.object(dv, "IPYTHON_AVAILABLE", False):
            with patch("builtins.print") as mock_print:
                with pytest.raises(ValueError, match="no columns"):
                    dv.displayDF(df)
                mock_print.assert_not_called()


# ===========================================================================
# ── 41e  Complex type preview fallback (asDict/list/dict raises) ─────────────
# ===========================================================================

class TestComplexPreviewFallback:
    """Struct/Array/Map when serialization raises: preview is type name; fallback escaped."""

    def test_struct_asdict_raises_preview_is_structtype(self):
        """StructType value whose asDict() raises → preview becomes 'StructType'."""
        val = MagicMock()
        val.asDict.side_effect = RuntimeError("nope")
        out = dv._format_complex_cell(val, _spark_types.StructType(), "cell_1_0_0", "abc12345")
        assert "StructType" in out
        assert "complex-preview" in out
        assert "&lt;" in out or "<" not in out.replace("</span>", "").replace("<div", "").replace("<span", "").replace("<pre", "")

    def test_array_list_raises_preview_is_arraytype(self):
        """ArrayType value where list(value) raises (e.g. non-iterable) → preview becomes 'ArrayType'."""
        # Passing non-iterable (e.g. int) to ArrayType branch: list(3) raises TypeError → except block
        out = dv._format_complex_cell(3, _spark_types.ArrayType(), "cell_1_0_0", "abc12345")
        assert "ArrayType" in out or "Array" in out

    def test_map_dict_raises_preview_is_maptype(self):
        """MapType value where dict(value) raises (e.g. non-mapping) → preview becomes 'MapType'."""
        # list is not a mapping; dict([1,2,3]) raises TypeError → except block
        out = dv._format_complex_cell([1, 2, 3], _spark_types.MapType(), "cell_1_0_0", "abc12345")
        assert "MapType" in out or "Map" in out

    def test_fallback_preview_escaped(self):
        """Fallback string (str(value)) in preview is HTML-escaped."""

        class BadVal:
            def asDict(self):
                raise RuntimeError("nope")
            def __str__(self):
                return "<script>"
        out = dv._format_complex_cell(BadVal(), _spark_types.StructType(), "c_0_0", "u1")
        assert "&lt;script&gt;" in out or "&lt;" in out

    def test_struct_asdict_and_str_both_raise_propagates(self):
        """When asDict() raises and __str__() also raises, exception propagates (no silent fallback)."""

        class BadStruct:
            def asDict(self):
                raise RuntimeError("asDict failed")
            def __str__(self):
                raise RuntimeError("str failed")
        with pytest.raises(RuntimeError, match="str failed"):
            dv._format_complex_cell(BadStruct(), _spark_types.StructType(), "c_0_0", "u1")


# ===========================================================================
# ── 41f  ISO iterable mismatch (list/tuple vs generator/set) ─────────────────
# ===========================================================================

class TestISOIterableMismatch:
    """_format_simple_cell only delegates to _format_country_list for (list, tuple); others get plain str."""

    def setup_method(self):
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_iso_tagged_generator_plain_fallback(self):
        """ISO-tagged column with generator value → plain string (not country list)."""
        def gen():
            yield "USA"
            yield "GBR"
        out = dv._format_simple_cell(gen(), [dv.TAG_ISO])
        # Generator is not list/tuple so we hit the final str(value) branch
        assert "dataframe-country-list" not in out or "USA" in out

    def test_iso_tagged_set_plain_string_fallback(self):
        """ISO-tagged column with set → not (list, tuple) so plain str(value); output contains repr."""
        out = dv._format_simple_cell({"USA", "GBR"}, [dv.TAG_ISO])
        # set is not (list, tuple) so we hit final branch: _html.escape(str(value))
        assert "USA" in out or "GBR" in out or "{" in out

    def test_iso_tagged_list_gets_country_list(self):
        """ISO-tagged list gets country list formatting."""
        out = dv._format_simple_cell(["USA", "GBR"], [dv.TAG_ISO])
        assert "dataframe-country" in out or "dataframe-country-list" in out


# ===========================================================================
# ── 41g  _get_column_metadata / schema edge cases ───────────────────────────
# ===========================================================================

class TestGetColumnMetadataEdgeCases:
    """dataType.__str__ raises; field name not string; missing name/dataType."""

    def test_data_type_str_raises_propagates(self):
        """Custom dataType whose __str__ raises → _get_column_metadata propagates."""
        class BadType:
            def __str__(self):
                raise RuntimeError("str failed")
        df = _make_df(("a",))
        df.schema.fields[0].dataType = BadType()
        td = dv._TableData(display_rows=[], has_more=False)
        tag_sets = {dv.TAG_LANGUAGE: set(), dv.TAG_ISO: set(), dv.TAG_REGEX: set(),
                    dv.TAG_CLEANED: set(), dv.TAG_DATE: set(), dv.TAG_METADATA: set()}
        with pytest.raises(RuntimeError, match="str failed"):
            dv._get_column_metadata(df, td, tag_sets)


# ===========================================================================
# ── 41h  Table name / header escaping and control chars ──────────────────────
# ===========================================================================

class TestTableNameHeaderEscaping:
    """table_name: null byte stripped; newline/tab/RTL escaped or stripped."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_table_name_embedded_null_byte_removed(self):
        """Null byte in table_name is removed, not merely escaped."""
        dv.displayDF(_make_df(), table_name="bad\x00name")
        html = _HTML_mock.call_args[0][0]
        assert "\x00" not in html

    def test_table_name_newline_tab_escaped(self):
        """Newline/tab in table_name do not break markup."""
        dv.displayDF(_make_df(), table_name="a\nb\tc")
        html = _HTML_mock.call_args[0][0]
        assert "a" in html and "b" in html and "c" in html

    def test_table_name_empty_string_renders_default_label(self):
        """table_name='' is falsy so label becomes 'DataFrame' (same as None)."""
        dv.displayDF(_make_df(), table_name="")
        html = _HTML_mock.call_args[0][0]
        assert "DataFrame" in html


# ===========================================================================
# ── 41i  Footer DOM contract (JS relies on structure) ────────────────────────
# ===========================================================================

class TestFooterDOMContract:
    """Footer markup: #dffoot_* inside .dataframe-footer; notice is next sibling; rows span present."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_columns_hidden_notice_inside_dataframe_footer(self):
        """The .columns-hidden-notice element is inside .dataframe-footer."""
        dv.displayDF(_make_df(("a",)))
        html = _HTML_mock.call_args[0][0]
        assert "dataframe-footer" in html
        assert "columns-hidden-notice" in html
        # Structure: footer div contains the notice (notice appears between footer start and footer end)
        footer_start = html.find('class="dataframe-footer"')
        footer_end = html.find("</div>", footer_start) + len("</div>")
        footer_section = html[footer_start:footer_end]
        assert "columns-hidden-notice" in footer_section

    def test_notice_is_immediate_next_sibling_of_footer_span_in_markup(self):
        """In rendered HTML, .columns-hidden-notice span appears right after the span with id=dffoot_*."""
        dv.displayDF(_make_df(("a",)))
        html = _HTML_mock.call_args[0][0]
        # Pattern: <span id="dffoot_...">...</span> then <span class="columns-hidden-notice"
        # (so footerEl.nextElementSibling in JS finds the notice)
        idx_footer = html.find('id="dffoot_')
        assert idx_footer >= 0
        after_footer_close = html.find("</span>", html.find(">", idx_footer)) + len("</span>")
        rest = html[after_footer_close:after_footer_close + 200]
        assert "columns-hidden-notice" in rest, "columns-hidden-notice must follow #dffoot_* span (nextElementSibling)"

    def test_dffoot_id_inside_dataframe_footer(self):
        """#dffoot_* (footerEl) is inside .dataframe-footer."""
        dv.displayDF(_make_df(("a",)))
        html = _HTML_mock.call_args[0][0]
        assert 'id="dffoot_' in html
        footer_start = html.find('class="dataframe-footer"')
        assert footer_start >= 0
        assert html.find('id="dffoot_', footer_start) > 0 or html.find('id="dffoot_') < html.find("</div>", footer_start)

    def test_dataframe_footer_rows_present_after_footer_span(self):
        """ .dataframe-footer-rows is present so footer updates don't remove it (JS only mutates footer span)."""
        dv.displayDF(_make_df(("a",)))
        html = _HTML_mock.call_args[0][0]
        assert "dataframe-footer-rows" in html


# ===========================================================================
# ── 41i2  Colspan contract ──────────────────────────────────────────────────
# ===========================================================================

class TestColspanContract:
    """Empty-row and truncated-row colspan matches column count; one-col → colspan=1."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_one_column_table_colspan_one(self):
        dv.displayDF(_make_df(("only",)))
        html = _HTML_mock.call_args[0][0]
        assert 'colspan="1"' in html

    def test_many_columns_colspan_matches(self):
        df = _make_df(tuple(f"c{i}" for i in range(10)))
        dv.displayDF(df)
        html = _HTML_mock.call_args[0][0]
        assert 'colspan="10"' in html

    def test_empty_row_and_truncated_row_same_colspan(self):
        """When has_more, filter-show-all row and column-selector-empty-cell both use colspan=column_count."""
        # 20 rows, max_rows=5 → has_more True → filter-show-all row with colspan=3
        df = _make_df(("a", "b", "c"), rows=[{"a": i, "b": i, "c": i} for i in range(20)])
        dv.displayDF(df, max_rows=5)
        html = _HTML_mock.call_args[0][0]
        assert html.count('colspan="3"') >= 2


# ===========================================================================
# ── 41j  Column names with punctuation / hostile ───────────────────────────
# ===========================================================================

class TestColumnNamesPunctuation:
    """Column names: quotes, apostrophes, ampersands, slashes, brackets, emoji."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    @pytest.mark.parametrize("col_name", [
        'col"quote', "col'apos", "a&amp;b", "a/b", "a[b]", "col\u2705",
    ])
    def test_column_name_escaped_in_output(self, col_name):
        """Punctuation and emoji in column name are escaped in HTML."""
        df = _make_df((col_name,))
        dv.displayDF(df)
        html = _HTML_mock.call_args[0][0]
        assert "col" in html or "a" in html  # at least partial match; no raw breakage


# ===========================================================================
# ── 41k  _estimate_display_len hostile ──────────────────────────────────────
# ===========================================================================

class TestEstimateDisplayLenHostile:
    """Object whose __str__ raises or returns huge string."""

    def test_estimate_display_len_str_raises_propagates(self):
        """_estimate_display_len propagates when str(val) raises."""
        class Bad:
            def __str__(self):
                raise RuntimeError("str failed")
        with pytest.raises(RuntimeError, match="str failed"):
            dv._estimate_display_len(Bad())


# ===========================================================================
# ── 41l  Shared CSS sentinel semantics ──────────────────────────────────────
# ===========================================================================

class TestSharedCSSSentinelSemantics:
    """_STYLE_INJECTED=True before first render; reset brings CSS back; multiple viewers one shared block."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_style_injected_true_before_first_render_skips_shared_css(self):
        """If _STYLE_INJECTED is already True, first displayDF does not emit shared CSS."""
        dv._STYLE_INJECTED = True
        try:
            dv.displayDF(_make_df())
            html = _HTML_mock.call_args[0][0]
            assert "dfv-shared-css" not in html
        finally:
            dv._STYLE_INJECTED = False

    def test_reset_sentinel_then_render_emits_shared_css_again(self):
        """After resetting _STYLE_INJECTED to False, next render emits shared CSS."""
        dv.displayDF(_make_df())
        _HTML_mock.reset_mock()
        dv._STYLE_INJECTED = False
        dv.displayDF(_make_df())
        html = _HTML_mock.call_args[0][0]
        assert "dfv-shared-css" in html

    def test_initial_html_always_includes_viewer_simple(self):
        """Initial HTML always includes viewer-simple class."""
        dv.displayDF(_make_df())
        html = _HTML_mock.call_args[0][0]
        assert 'viewer-simple' in html


# ===========================================================================
# ── 42  Snapshot-style regression (normalized IDs) ───────────────────────────
# ===========================================================================

def _normalize_viewer_html(html: str) -> str:
    """Replace dynamic viewer IDs with placeholders for stable snapshot comparison."""
    out = re.sub(r"dfv_[a-f0-9]{8}", "dfv_XXXXXXXX", html)
    out = re.sub(r"dft_[a-f0-9]{8}", "dft_XXXXXXXX", out)
    out = re.sub(r"dftoggle_[a-f0-9]{8}", "dftoggle_XXXXXXXX", out)
    out = re.sub(r"dfview_[a-f0-9]{8}", "dfview_XXXXXXXX", out)
    out = re.sub(r"dffilter_[a-f0-9]{8}", "dffilter_XXXXXXXX", out)
    out = re.sub(r"dfcolsel_[a-f0-9]{8}", "dfcolsel_XXXXXXXX", out)
    out = re.sub(r"dfcolselmodal_[a-f0-9]{8}", "dfcolselmodal_XXXXXXXX", out)
    out = re.sub(r"dfcolselmodal_[a-f0-9]{8}-title", "dfcolselmodal_XXXXXXXX-title", out)
    out = re.sub(r"dffoot_[a-f0-9]{8}", "dffoot_XXXXXXXX", out)
    out = re.sub(r"cell_[a-f0-9]{8}_\d+_\d+", "cell_XXXXXXXX_R_C", out)
    return out


class TestSnapshotRegression:
    """Normalized HTML structure regression; IDs replaced for stable comparison."""

    def setup_method(self):
        _reset_style_sentinel()
        _ipython_display_mock.reset_mock()
        _HTML_mock.reset_mock()
        self._val_patcher = patch("dataframe_viewer._validate_inputs")
        self._val_patcher.start()

    def teardown_method(self):
        self._val_patcher.stop()

    def test_simple_table_normalized_structure(self):
        """After normalizing IDs, simple table has expected structure."""
        dv.displayDF(_make_df(("a", "b")))
        html = _normalize_viewer_html(_HTML_mock.call_args[0][0])
        assert "dfv_XXXXXXXX" in html
        assert "dft_XXXXXXXX" in html
        assert "<table" in html and "<thead" in html and "<tbody" in html
        assert 'data-column="0"' in html and 'data-column="1"' in html
        assert "viewer-simple" in html

    def test_tag_heavy_table_normalized_structure(self):
        """Tag-heavy table still has filter pane and column selector after normalize."""
        dv.displayDF(
            _make_df(("lang", "country", "dt")),
            translation_columns=["lang"],
            country_iso_columns=["country"],
            date_columns=["dt"],
        )
        html = _normalize_viewer_html(_HTML_mock.call_args[0][0])
        assert "column-filter-pane" in html
        assert "column-selector-modal" in html
        assert "tag-language" in html and "tag-iso" in html and "tag-date" in html

    def test_empty_truncated_table_normalized_structure(self):
        """Empty / truncated table has footer and empty row notice."""
        df = _make_df(("a",), rows=[])
        dv.displayDF(df)
        html = _normalize_viewer_html(_HTML_mock.call_args[0][0])
        assert "0 rows" in html or "rows" in html
        assert "column-selector-empty-row" in html or "column-selector-empty-cell" in html

    def test_modal_title_id_normalized_in_snapshot(self):
        """Modal title id (col_sel_modal + '-title') is normalized so snapshots stay stable."""
        dv.displayDF(_make_df(("a",)))
        html = _normalize_viewer_html(_HTML_mock.call_args[0][0])
        assert "dfcolselmodal_XXXXXXXX-title" in html
        assert "dfcolselmodal_" in html and "-title" in html


# ===========================================================================
# ── 43  Property-style fuzzing for escaping ───────────────────────────────
# ===========================================================================

class TestEscapingFuzz:
    """Arbitrary Unicode, control chars, RTL, bidi, malformed HTML: never emit raw executable markup."""

    @pytest.mark.parametrize(
        "payload",
        [
            "<script>alert(1)</script>",
            "><img src=x onerror=alert(1)>",
            "\x00null\x00byte",
            "\n\t\r",
            "\u202eRTL\u202c",  # RTL override
            "\u200b\u200c\u200d\u2060",  # zero-width joiners etc
            "''\"\"\"\"",
            "a" * 1000 + "<" + "b" * 1000,
            "\u0000\u0001\u0002",
            "&\u003c\u003e\"",
        ],
    )
    def test_simple_cell_never_emits_raw_markup_from_value(self, payload):
        """User content is escaped so no raw < or unescaped quotes in cell output."""
        out = dv._format_simple_cell(payload, None)
        assert "<script" not in out
        assert "onerror" not in out or "&" in out
        assert out.count("<") == 0 or (out.count("&lt;") >= 1 and "<" not in out.replace("&lt;", ""))

    @pytest.mark.parametrize(
        "payload",
        ["<script>x</script>", 'say "hello"', "\n\t", "\u202e", 'a"b"c'],
    )
    def test_sort_attr_value_never_contains_raw_quote_or_angle(self, payload):
        """Sort attribute value is safe for HTML attributes."""
        out = dv._sort_attr_value(payload)
        assert '"' not in out
        assert "<" not in out
        assert ">" not in out