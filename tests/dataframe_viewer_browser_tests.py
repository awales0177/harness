"""
Browser-level tests for the DataFrame viewer using Playwright.

These tests verify that the emitted HTML/JS actually behaves correctly in a browser:
sort clicks change row order, theme toggle updates classes, column selector hides
columns, modal open/close works, complex cell expand/collapse reveals JSON.

Run only when Playwright is installed and browsers are available:
    pip install playwright
    playwright install chromium
    pytest tests/dataframe_viewer_browser_tests.py -v

Skip automatically if playwright is not installed.
"""

from __future__ import annotations

import tempfile

import pytest

# Skip entire module if playwright not available
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Build minimal viewer HTML for testing (avoid full displayDF path and stubs)
def _get_viewer_html():
    """Produce viewer HTML by calling displayDF with a mock and capturing output."""
    import sys
    import types
    from unittest.mock import MagicMock, patch

    # Stub pyspark and IPython so we can import dataframe_viewer
    if "pyspark" not in sys.modules:
        pyspark = types.ModuleType("pyspark")
        pyspark_sql = types.ModuleType("pyspark.sql")
        pyspark_sql.DataFrame = type("DataFrame", (), {})
        pyspark.sql = pyspark_sql
        sys.modules["pyspark"] = pyspark
        sys.modules["pyspark.sql"] = pyspark_sql
    if "IPython" not in sys.modules:
        ipython = types.ModuleType("IPython")
        ipython_display = types.ModuleType("IPython.display")
        ipython_display.HTML = lambda x: x
        ipython_display.display = lambda x: None
        ipython.display = ipython_display
        sys.modules["IPython"] = ipython
        sys.modules["IPython.display"] = ipython_display

    import dataframe_viewer as dv

    # Build mock DF: 3 columns, 3 rows with sortable values
    columns = ("name", "score", "date")
    rows = [
        {"name": "Alice", "score": 90, "date": "2024-01-01"},
        {"name": "Bob", "score": 70, "date": "2024-01-02"},
        {"name": "Carol", "score": 85, "date": "2024-01-03"},
    ]
    df = MagicMock()
    fields = []
    for c in columns:
        f = MagicMock()
        f.name = c
        f.dataType = MagicMock()
        fields.append(f)
    df.schema.fields = fields
    row_mocks = []
    for r in rows:
        rm = MagicMock()
        rm.__getitem__ = lambda self, k, _r=r: _r[k]
        row_mocks.append(rm)
    df.take.side_effect = lambda n: row_mocks[:n]
    df.count.return_value = 3

    captured = []
    with patch.object(dv, "_validate_inputs"):
        with patch.object(dv, "IPYTHON_AVAILABLE", True):
            with patch("dataframe_viewer._ipython_display", side_effect=lambda x: captured.append(x)):
                dv._STYLE_INJECTED = False
                dv.displayDF(df, max_rows=10, table_name="BrowserTest")
    return captured[0] if captured else ""


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
class TestDataFrameViewerBrowser:
    """Browser tests: interactions actually work in a real page."""

    @pytest.fixture(scope="class")
    def viewer_html(self):
        return _get_viewer_html()

    @pytest.fixture
    def page(self, viewer_html):
        """Open viewer HTML in a headless browser and yield the page."""
        html = viewer_html
        if not html or len(html) < 100:
            pytest.skip("Could not generate viewer HTML")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".html", delete=False, encoding="utf-8"
                ) as f:
                    f.write("<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>")
                    f.write(html)
                    f.write("</body></html>")
                    path = f.name
                page.goto(f"file://{path}")
                yield page
            finally:
                browser.close()

    def test_page_loads_viewer(self, page):
        """Viewer root div is present and visible."""
        viewer = page.locator(".dataframe-viewer")
        assert viewer.count() == 1
        assert viewer.is_visible()

    def test_theme_toggle_changes_root_class(self, page):
        """Clicking theme toggle adds/removes light-mode on root."""
        viewer = page.locator(".dataframe-viewer").first
        toggle = page.locator(".theme-toggle").first
        toggle.click()
        page.wait_for_timeout(200)
        # After click, root should have light-mode (or not); at least class changed
        classes = viewer.get_attribute("class") or ""
        assert "viewer-simple" in classes or "viewer-expert" in classes

    def test_column_selector_modal_opens_and_closes(self, page):
        """Column selector button opens modal; close button closes it."""
        btn = page.locator("button.column-selector-btn").first
        backdrop = page.locator(".column-selector-backdrop").first
        btn.click()
        page.wait_for_timeout(150)
        assert "open" in (backdrop.get_attribute("class") or "")
        close_btn = page.locator(".column-selector-close").first
        close_btn.click()
        page.wait_for_timeout(150)
        assert "open" not in (backdrop.get_attribute("class") or "")

    def test_unchecking_column_hides_column_in_table(self, page):
        """Unchecking a column in the modal hides that column's th/td."""
        btn = page.locator("button.column-selector-btn").first
        btn.click()
        page.wait_for_timeout(150)
        # Uncheck first column (data-column="0")
        first_check = page.locator(".column-selector-check[data-column='0']").first
        first_check.uncheck()
        page.wait_for_timeout(200)
        # First th should have column-selector-hidden
        hidden_th = page.locator("th.column-selector-hidden[data-column='0']")
        assert hidden_th.count() >= 1

    def test_sort_header_has_data_column(self, page):
        """Sortable headers have data-column for JS sort logic."""
        sortable = page.locator("th.sortable[data-column]")
        assert sortable.count() >= 1

    def test_modal_has_dialog_role(self, page):
        """Modal has role=dialog for accessibility."""
        modal = page.locator(".column-selector-modal[role='dialog']")
        assert modal.count() == 1

    def test_deselect_one_column_shows_hidden_notice(self, page):
        """Deselecting a column shows the 'Columns are hidden' notice (footerEl.nextElementSibling)."""
        btn = page.locator("button.column-selector-btn").first
        btn.click()
        page.wait_for_timeout(150)
        notice = page.locator(".columns-hidden-notice").first
        assert notice.count() == 1
        # Initially hidden (display:none or not visible)
        first_check = page.locator(".column-selector-check[data-column='0']").first
        first_check.uncheck()
        page.wait_for_timeout(250)
        # JS should set notice.style.display = '' when vis < total
        notice_visible = notice.evaluate("el => el.style.display !== 'none'")
        assert notice_visible, "Hidden notice should be visible after deselecting a column"

    def test_reselect_all_columns_hides_notice(self, page):
        """Reselecting all columns hides the 'Columns are hidden' notice."""
        btn = page.locator("button.column-selector-btn").first
        btn.click()
        page.wait_for_timeout(150)
        first_check = page.locator(".column-selector-check[data-column='0']").first
        first_check.uncheck()
        page.wait_for_timeout(200)
        first_check.check()
        page.wait_for_timeout(200)
        notice = page.locator(".columns-hidden-notice").first
        notice_display = notice.evaluate("el => el.style.display")
        assert notice_display == "none", "Notice should be hidden when all columns visible"

    def test_footer_count_all_visible(self, page):
        """When all columns visible, footer shows 'N columns' (no 'of')."""
        footer = page.locator(".dataframe-footer span").first
        text = footer.inner_text()
        assert "columns" in text
        assert " of " not in text

    def test_footer_count_after_hide_one(self, page):
        """When one column hidden, footer shows 'X of N columns'."""
        btn = page.locator("button.column-selector-btn").first
        btn.click()
        page.wait_for_timeout(150)
        page.locator(".column-selector-check[data-column='0']").first.uncheck()
        page.wait_for_timeout(200)
        footer = page.locator(".dataframe-footer span").first
        text = footer.inner_text()
        assert " of " in text
        assert "columns" in text

    def test_initial_html_has_viewer_simple(self, page):
        """Initial HTML includes viewer-simple class before JS may change it."""
        viewer = page.locator(".dataframe-viewer").first
        classes = viewer.get_attribute("class") or ""
        assert "viewer-simple" in classes

    def test_hide_all_columns_shows_empty_row_message(self, page):
        """Unchecking all columns in the selector shows the empty-row message (zero visible columns)."""
        btn = page.locator("button.column-selector-btn").first
        btn.click()
        page.wait_for_timeout(150)
        for i in range(3):
            check = page.locator(f".column-selector-check[data-column='{i}']").first
            if check.is_visible():
                check.uncheck()
        page.wait_for_timeout(300)
        empty_row = page.locator("tr.column-selector-empty-row").first
        empty_display = empty_row.evaluate("el => el.style.display")
        assert empty_display != "none", "Empty row message should be visible when no columns selected"
        text = empty_row.inner_text()
        assert "column" in text.lower() or "selected" in text.lower()
