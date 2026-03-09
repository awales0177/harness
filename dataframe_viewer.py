"""
Custom DataFrame viewer with rounded corners and scrolling for Jupyter notebooks.
Supports expandable complex types (structs, arrays, maps).
"""

from __future__ import annotations

import html as _html
import io
import json
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STAGES = frozenset({"bronze", "silver", "gold"})

_COMPLEX_TYPE_NAMES = frozenset({"StructType", "ArrayType", "MapType"})

_COL_MIN_PX         = 80
_COL_MAX_PX         = 500
_COL_SAMPLE_ROWS    = 10
_COL_MAX_SAMPLE_LEN = 80   # chars — stops huge blobs from driving column width
_PX_PER_CHAR_HEADER = 8
_PX_PER_CHAR_VALUE  = 7
_COL_PADDING        = 24

# FIX (Perf #4): cap max_rows to prevent accidental driver OOM
_MAX_ROWS_HARD_CAP  = 10_000

# ---------------------------------------------------------------------------
# One-time stylesheet injection sentinel
# FIX (Perf #3): CSS is only emitted once per kernel session, not per call.
# ---------------------------------------------------------------------------
_STYLE_INJECTED: bool = False


# ---------------------------------------------------------------------------
# Tag specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _TagSpec:
    """Describes one column-tagging category (e.g. translation, date, …).

    filter_key  – used as the JS filter key and the suffix in the CSS class
                  ``tag-<filter_key>``.
    param_name  – matches the ``displayDF`` kwarg name (for docstrings).
    """
    filter_key: str
    param_name: str

    @property
    def css_class(self) -> str:
        return f"tag-{self.filter_key}"


TAG_LANGUAGE = _TagSpec("language", "translation_columns")
TAG_ISO      = _TagSpec("iso",      "country_iso_columns")
TAG_REGEX    = _TagSpec("regex",    "regex_columns")
TAG_CLEANED  = _TagSpec("cleaned",  "cleaned_columns")
TAG_DATE     = _TagSpec("date",     "date_columns")
TAG_METADATA = _TagSpec("metadata", "metadata_columns")

# Ordered list — iteration order controls button order in the filter pane
ALL_TAGS: List[_TagSpec] = [
    TAG_LANGUAGE, TAG_ISO, TAG_REGEX, TAG_CLEANED, TAG_DATE, TAG_METADATA,
]


# ---------------------------------------------------------------------------
# ISO3 → ISO2 for flag emoji (regional indicator symbols)
# ---------------------------------------------------------------------------

# ISO 3166-1 alpha-3 → alpha-2 (subset; add more as needed)
_ISO3_TO_ISO2: Dict[str, str] = {
    "AFG": "AF", "ALB": "AL", "DZA": "DZ", "ASM": "AS", "AND": "AD", "AGO": "AO",
    "AIA": "AI", "ATA": "AQ", "ATG": "AG", "ARG": "AR", "ARM": "AM", "ABW": "AW",
    "AUS": "AU", "AUT": "AT", "AZE": "AZ", "BHS": "BS", "BHR": "BH", "BGD": "BD",
    "BRB": "BB", "BLR": "BY", "BEL": "BE", "BLZ": "BZ", "BEN": "BJ", "BMU": "BM",
    "BTN": "BT", "BOL": "BO", "BES": "BQ", "BIH": "BA", "BWA": "BW", "BVT": "BV",
    "BRA": "BR", "IOT": "IO", "BRN": "BN", "BGR": "BG", "BFA": "BF", "BDI": "BI",
    "CPV": "CV", "KHM": "KH", "CMR": "CM", "CAN": "CA", "CYM": "KY", "CAF": "CF",
    "TCD": "TD", "CHL": "CL", "CHN": "CN", "CXR": "CX", "CCK": "CC", "COL": "CO",
    "COM": "KM", "COD": "CD", "COG": "CG", "COK": "CK", "CRI": "CR", "CIV": "CI",
    "HRV": "HR", "CUB": "CU", "CUW": "CW", "CYP": "CY", "CZE": "CZ", "DNK": "DK",
    "DJI": "DJ", "DMA": "DM", "DOM": "DO", "ECU": "EC", "EGY": "EG", "SLV": "SV",
    "GNQ": "GQ", "ERI": "ER", "EST": "EE", "SWZ": "SZ", "ETH": "ET", "FLK": "FK",
    "FRO": "FO", "FJI": "FJ", "FIN": "FI", "FRA": "FR", "GUF": "GF", "PYF": "PF",
    "ATF": "TF", "GAB": "GA", "GMB": "GM", "GEO": "GE", "DEU": "DE", "GHA": "GH",
    "GIB": "GI", "GRC": "GR", "GRL": "GL", "GRD": "GD", "GLP": "GP", "GUM": "GU",
    "GTM": "GT", "GGY": "GG", "GIN": "GN", "GNB": "GW", "GUY": "GY", "HTI": "HT",
    "HMD": "HM", "VAT": "VA", "HND": "HN", "HKG": "HK", "HUN": "HU", "ISL": "IS",
    "IND": "IN", "IDN": "ID", "IRN": "IR", "IRQ": "IQ", "IRL": "IE", "IMN": "IM",
    "ISR": "IL", "ITA": "IT", "JAM": "JM", "JPN": "JP", "JEY": "JE", "JOR": "JO",
    "KAZ": "KZ", "KEN": "KE", "KIR": "KI", "PRK": "KP", "KOR": "KR", "KWT": "KW",
    "KGZ": "KG", "LAO": "LA", "LVA": "LV", "LBN": "LB", "LSO": "LS", "LBR": "LR",
    "LBY": "LY", "LIE": "LI", "LTU": "LT", "LUX": "LU", "MAC": "MO", "MDG": "MG",
    "MWI": "MW", "MYS": "MY", "MDV": "MV", "MLI": "ML", "MLT": "MT", "MHL": "MH",
    "MTQ": "MQ", "MRT": "MR", "MUS": "MU", "MYT": "YT", "MEX": "MX", "FSM": "FM",
    "MDA": "MD", "MCO": "MC", "MNG": "MN", "MNE": "ME", "MSR": "MS", "MAR": "MA",
    "MOZ": "MZ", "MMR": "MM", "NAM": "NA", "NRU": "NR", "NPL": "NP", "NLD": "NL",
    "NCL": "NC", "NZL": "NZ", "NIC": "NI", "NER": "NE", "NGA": "NG", "NIU": "NU",
    "NFK": "NF", "MKD": "MK", "MNP": "MP", "NOR": "NO", "OMN": "OM", "PAK": "PK",
    "PLW": "PW", "PSE": "PS", "PAN": "PA", "PNG": "PG", "PRY": "PY", "PER": "PE",
    "PHL": "PH", "PCN": "PN", "POL": "PL", "PRT": "PT", "PRI": "PR", "QAT": "QA",
    "REU": "RE", "ROU": "RO", "RUS": "RU", "RWA": "RW", "BLM": "BL", "SHN": "SH",
    "KNA": "KN", "LCA": "LC", "MAF": "MF", "SPM": "PM", "VCT": "VC", "WSM": "WS",
    "SMR": "SM", "STP": "ST", "SAU": "SA", "SEN": "SN", "SRB": "RS", "SYC": "SC",
    "SLE": "SL", "SGP": "SG", "SXM": "SX", "SVK": "SK", "SVN": "SI", "SLB": "SB",
    "SOM": "SO", "ZAF": "ZA", "SGS": "GS", "SSD": "SS", "ESP": "ES", "LKA": "LK",
    "SDN": "SD", "SUR": "SR", "SJM": "SJ", "SWE": "SE", "CHE": "CH", "SYR": "SY",
    "TWN": "TW", "TJK": "TJ", "TZA": "TZ", "THA": "TH", "TLS": "TL", "TGO": "TG",
    "TKL": "TK", "TON": "TO", "TTO": "TT", "TUN": "TN", "TUR": "TR", "TKM": "TM",
    "TCA": "TC", "TUV": "TV", "UGA": "UG", "UKR": "UA", "ARE": "AE", "GBR": "GB",
    "USA": "US", "UMI": "UM", "URY": "UY", "UZB": "UZ", "VUT": "VU", "VEN": "VE",
    "VNM": "VN", "VGB": "VG", "VIR": "VI", "WLF": "WF", "ESH": "EH", "YEM": "YE",
    "ZMB": "ZM", "ZWE": "ZW",
    # Widely-used non-ISO-3166-1 codes included for practical coverage
    "XKX": "XK",  # Kosovo
}


def _iso2_to_flag(iso2: str) -> str:
    """Turn a two-letter ISO 3166-1 alpha-2 code into a flag emoji (regional indicators)."""
    if not iso2 or len(iso2) != 2 or not iso2[0].isalpha() or not iso2[1].isalpha():
        return ""
    a, b = iso2.upper()[0], iso2.upper()[1]
    return chr(0x1F1E6 + ord(a) - ord("A")) + chr(0x1F1E6 + ord(b) - ord("A"))


def _code_to_flag(code: str) -> str:
    """Turn ISO3 or ISO2 country code into flag emoji; empty string if unknown."""
    if not code or not isinstance(code, str):
        return ""
    raw = code.strip().upper()
    if len(raw) == 2:
        return _iso2_to_flag(raw)
    if len(raw) == 3:
        iso2 = _ISO3_TO_ISO2.get(raw)
        return _iso2_to_flag(iso2) if iso2 else ""
    return ""


# ---------------------------------------------------------------------------
# SVG icons
# ---------------------------------------------------------------------------

def _hdr_svg(css_class: str, title: str, body: str) -> str:
    attrs = (
        f'class="{css_class}" title="{title}" '
        'style="width:14px;height:14px;display:inline-block;vertical-align:middle" '
        'xmlns="http://www.w3.org/2000/svg" width="14" height="14" '
        'viewBox="0 0 24 24" fill="none" stroke="#000000" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
    )
    return f"<svg {attrs}>{body}</svg>"


def _thm_svg(css_class: str, title: str, body: str) -> str:
    attrs = (
        f'class="{css_class}" title="{title}" '
        'style="width:20px;height:20px;display:inline-block;vertical-align:middle" '
        'xmlns="http://www.w3.org/2000/svg" width="20" height="20" '
        'viewBox="0 0 24 24" fill="none" stroke="#000000" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
    )
    return f"<svg {attrs}>{body}</svg>"


def _hdr_svg_filled(css_class: str, title: str, view_box: str, body: str) -> str:
    """Header icon with custom viewBox and fill (no stroke), for 14x14 display."""
    attrs = (
        f'class="{css_class}" title="{title}" '
        'style="width:14px;height:14px;display:inline-block;vertical-align:middle" '
        'xmlns="http://www.w3.org/2000/svg" width="14" height="14" '
        f'viewBox="{view_box}" fill="currentColor"'
    )
    return f"<svg {attrs}>{body}</svg>"


def _hdr_svg_fill_stroke(css_class: str, title: str, view_box: str, body: str) -> str:
    """Header icon with custom viewBox, fill and stroke for 14x14 display."""
    attrs = (
        f'class="{css_class}" title="{title}" '
        'style="width:14px;height:14px;display:inline-block;vertical-align:middle" '
        'xmlns="http://www.w3.org/2000/svg" width="14" height="14" '
        f'viewBox="{view_box}" fill="currentColor" stroke="currentColor"'
    )
    return f"<svg {attrs}>{body}</svg>"


ICON_LANGUAGE = _hdr_svg("lang-icon", "Translation column",
    '<path d="M5 8l6 6"/><path d="M4 14l6-6 2-3"/><path d="M2 5h12"/>'
    '<path d="M7 2h1"/><path d="M22 22l-5-10-5 10"/><path d="M14 18h6"/>')

ICON_GLOBE = _hdr_svg("globe-icon", "Country ISO column",
    '<path d="M15 21v-4a2 2 0 012-2h4"/>'
    '<path d="M7 4v2a3 3 0 003 2h0a2 2 0 012 2 2 2 0 004 0 2 2 0 012-2h3"/>'
    '<path d="M3 11h2a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v4"/>'
    '<circle cx="12" cy="12" r="10"/>')

ICON_REGEX = _hdr_svg("regex-icon", "Regex column",
    '<path d="M17 3v10"/><path d="M12.67 5.5l8.66 5"/>'
    '<path d="M12.67 10.5l8.66-5"/>'
    '<path d="M9 17a2 2 0 00-2-2H5a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2v-2z"/>')

ICON_META = _hdr_svg("meta-icon", "Metadata column",
    '<path d="M12.22 2h-.44a2 2 0 00-2 2v.18a2 2 0 01-1 1.73l-.43.25a2 2 0 01-2 0'
    'l-.15-.08a2 2 0 00-2.73.73l-.22.38a2 2 0 00.73 2.73l.15.1a2 2 0 011 1.72v.51'
    'a2 2 0 01-1 1.74l-.15.09a2 2 0 00-.73 2.73l.22.38a2 2 0 002.73.73l.15-.08'
    'a2 2 0 012 0l.43.25a2 2 0 011 1.73V20a2 2 0 002 2h.44a2 2 0 002-2v-.18'
    'a2 2 0 011-1.73l.43-.25a2 2 0 012 0l.15.08a2 2 0 002.73-.73l.22-.39'
    'a2 2 0 00-.73-2.73l-.15-.08a2 2 0 01-1-1.74v-.5a2 2 0 011-1.74l.15-.09'
    'a2 2 0 00.73-2.73l-.22-.38a2 2 0 00-2.73-.73l-.15.08a2 2 0 01-2 0l-.43-.25'
    'a2 2 0 01-1-1.73V4a2 2 0 00-2-2z"/><circle cx="12" cy="12" r="3"/>')

# Clean column icon (slider/clean UI icon, fill + stroke)
_CLEAN_VIEWBOX = "0 0 24 24"
_CLEAN_PATHS = (
    '<path fill="currentColor" d="M5.50506 11.4096L6.03539 11.9399L5.50506 11.4096ZM3 14.9522H2.25H3ZM12.5904 18.4949L12.0601 17.9646L12.5904 18.4949ZM9.04776 21V21.75V21ZM11.4096 5.50506L10.8792 4.97473L11.4096 5.50506ZM13.241 17.8444C13.5339 18.1373 14.0088 18.1373 14.3017 17.8444C14.5946 17.5515 14.5946 17.0766 14.3017 16.7837L13.241 17.8444ZM7.21629 9.69832C6.9234 9.40543 6.44852 9.40543 6.15563 9.69832C5.86274 9.99122 5.86274 10.4661 6.15563 10.759L7.21629 9.69832ZM16.073 16.073C16.3659 15.7801 16.3659 15.3053 16.073 15.0124C15.7801 14.7195 15.3053 14.7195 15.0124 15.0124L16.073 16.073ZM18.4676 11.5559C18.1759 11.8499 18.1777 12.3248 18.4718 12.6165C18.7658 12.9083 19.2407 12.9064 19.5324 12.6124L18.4676 11.5559ZM6.03539 11.9399L11.9399 6.03539L10.8792 4.97473L4.97473 10.8792L6.03539 11.9399ZM6.03539 17.9646C5.18538 17.1146 4.60235 16.5293 4.22253 16.0315C3.85592 15.551 3.75 15.2411 3.75 14.9522H2.25C2.25 15.701 2.56159 16.3274 3.03 16.9414C3.48521 17.538 4.1547 18.2052 4.97473 19.0253L6.03539 17.9646ZM4.97473 10.8792C4.1547 11.6993 3.48521 12.3665 3.03 12.9631C2.56159 13.577 2.25 14.2035 2.25 14.9522H3.75C3.75 14.6633 3.85592 14.3535 4.22253 13.873C4.60235 13.3752 5.18538 12.7899 6.03539 11.9399L4.97473 10.8792ZM12.0601 17.9646C11.2101 18.8146 10.6248 19.3977 10.127 19.7775C9.64651 20.1441 9.33665 20.25 9.04776 20.25V21.75C9.79649 21.75 10.423 21.4384 11.0369 20.97C11.6335 20.5148 12.3008 19.8453 13.1208 19.0253L12.0601 17.9646ZM4.97473 19.0253C5.79476 19.8453 6.46201 20.5148 7.05863 20.97C7.67256 21.4384 8.29902 21.75 9.04776 21.75V20.25C8.75886 20.25 8.449 20.1441 7.9685 19.7775C7.47069 19.3977 6.88541 18.8146 6.03539 17.9646L4.97473 19.0253ZM17.9646 6.03539C18.8146 6.88541 19.3977 7.47069 19.7775 7.9685C20.1441 8.449 20.25 8.75886 20.25 9.04776H21.75C21.75 8.29902 21.4384 7.67256 20.97 7.05863C20.5148 6.46201 19.8453 5.79476 19.0253 4.97473L17.9646 6.03539ZM19.0253 4.97473C18.2052 4.1547 17.538 3.48521 16.9414 3.03C16.3274 2.56159 15.701 2.25 14.9522 2.25V3.75C15.2411 3.75 15.551 3.85592 16.0315 4.22253C16.5293 4.60235 17.1146 5.18538 17.9646 6.03539L19.0253 4.97473ZM11.9399 6.03539C12.7899 5.18538 13.3752 4.60235 13.873 4.22253C14.3535 3.85592 14.6633 3.75 14.9522 3.75V2.25C14.2035 2.25 13.577 2.56159 12.9631 3.03C12.3665 3.48521 11.6993 4.1547 10.8792 4.97473L11.9399 6.03539ZM14.3017 16.7837L7.21629 9.69832L6.15563 10.759L13.241 17.8444L14.3017 16.7837ZM15.0124 15.0124L12.0601 17.9646L13.1208 19.0253L16.073 16.073L15.0124 15.0124ZM19.5324 12.6124C20.1932 11.9464 20.7384 11.3759 21.114 10.8404C21.5023 10.2869 21.75 9.71511 21.75 9.04776H20.25C20.25 9.30755 20.1644 9.58207 19.886 9.979C19.5949 10.394 19.1401 10.8781 18.4676 11.5559L19.5324 12.6124Z"/>'
    '<path stroke="currentColor" stroke-width="1.5" stroke-linecap="round" d="M9 21H21"/>'
)
ICON_CLEAN = _hdr_svg_fill_stroke("clean-icon", "Cleaned column", _CLEAN_VIEWBOX, _CLEAN_PATHS)

# Expert view button (right of theme toggle): chips for ISO and boolean when active
ICON_EXPERT = _thm_svg("expert-view-icon", "Expert view (chips for ISO and boolean)",
    '<path d="M19 3V7M17 5H21M19 17V21M17 19H21'
    'M10 5L8.53 8.73C8.34 9.20 8.25 9.44 8.10 9.64C7.98 9.82 7.82 9.98 7.64 10.10'
    'C7.44 10.25 7.20 10.34 6.73 10.53L3 12L6.73 13.47'
    'C7.20 13.66 7.44 13.75 7.64 13.90C7.82 14.02 7.98 14.18 8.10 14.36'
    'C8.25 14.56 8.34 14.80 8.53 15.27L10 19L11.47 15.27'
    'C11.66 14.80 11.75 14.56 11.90 14.36C12.02 14.18 12.18 14.02 12.36 13.90'
    'C12.56 13.75 12.80 13.66 13.27 13.47L17 12L13.27 10.53'
    'C12.80 10.34 12.56 10.25 12.36 10.10C12.18 9.98 12.02 9.82 11.90 9.64'
    'C11.75 9.44 11.66 9.20 11.47 8.73L10 5Z" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')

ICON_DATE = _hdr_svg("date-icon", "Date column",
    '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>'
    '<line x1="16" y1="2" x2="16" y2="6"/>'
    '<line x1="8" y1="2" x2="8" y2="6"/>'
    '<line x1="3" y1="10" x2="21" y2="10"/>')

ICON_MOON = _thm_svg("theme-toggle-moon", "Switch to light mode",
    '<path d="M3.32 11.68C3.32 16.65 7.35 20.68 12.32 20.68'
    'C16.11 20.68 19.35 18.34 20.68 15.03'
    'C19.64 15.45 18.51 15.68 17.32 15.68'
    'C12.35 15.68 8.32 11.65 8.32 6.68'
    'C8.32 5.50 8.55 4.36 8.96 3.33'
    'C5.66 4.66 3.32 7.90 3.32 11.68Z"/>')

ICON_LAYERS = (
    '<svg class="column-selector-layers-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M20 10L12 5L4 10L12 15L20 10Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M20 14L12 19L4 14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)
ICON_SORT_UP = (
    '<svg class="sort-icon-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M5 15L10 9.84985C10.2563 9.57616 10.566 9.35814 10.9101 9.20898C11.2541 9.05983 11.625 8.98291 12 8.98291C12.375 8.98291 12.7459 9.05983 13.0899 9.20898C13.434 9.35814 13.7437 9.57616 14 9.84985L19 15" '
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)
ICON_SORT_DOWN = (
    '<svg class="sort-icon-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M19 9L14 14.1599C13.7429 14.4323 13.4329 14.6493 13.089 14.7976C12.7451 14.9459 12.3745 15.0225 12 15.0225C11.6255 15.0225 11.2549 14.9459 10.9109 14.7976C10.567 14.6493 10.2571 14.4323 10 14.1599L5 9" '
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)
ICON_WARNING = (
    '<svg class="columns-hidden-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M12 15H12.01M12 12V9M4.98207 19H19.0179C20.5615 19 21.5233 17.3256 20.7455 15.9923L13.7276 3.96153C12.9558 2.63852 11.0442 2.63852 10.2724 3.96153L3.25452 15.9923C2.47675 17.3256 3.43849 19 4.98207 19Z" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)
ICON_LAYERS_DESELECTED = (
    '<svg class="column-selector-layers-icon column-selector-layers-deselected" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M4 4L20 20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'
    '<path fill-rule="evenodd" clip-rule="evenodd" d="M7.32711 6.74132L3.47001 9.152C3.17763 9.33474 3.00001 9.65521 3.00001 10C3.00001 10.3448 3.17763 10.6653 3.47001 10.848L11.47 15.848C11.7943 16.0507 12.2057 16.0507 12.53 15.848L14.9323 14.3465L13.481 12.8952L12 13.8208L5.88681 10L8.77849 8.1927L7.32711 6.74132ZM15.2215 11.8073L18.1132 10L12 6.17925L10.5191 7.10484L9.06768 5.65346L11.47 4.152C11.7943 3.94933 12.2057 3.94933 12.53 4.152L20.53 9.152C20.8224 9.33474 21 9.65521 21 10C21 10.3448 20.8224 10.6653 20.53 10.848L16.6729 13.2587L15.2215 11.8073ZM15.9425 15.3567L12 17.8208L4.53001 13.152C4.06167 12.8593 3.44472 13.0017 3.15201 13.47C2.8593 13.9383 3.00168 14.5553 3.47001 14.848L11.47 19.848C11.7943 20.0507 12.2057 20.0507 12.53 19.848L17.3939 16.8081L15.9425 15.3567ZM19.1344 15.7202L17.6831 14.2688L19.47 13.152C19.9383 12.8593 20.5553 13.0017 20.848 13.47C21.1407 13.9383 20.9983 14.5553 20.53 14.848L19.1344 15.7202Z" fill="currentColor"/>'
    '</svg>'
)
ICON_COLUMNS = _thm_svg("column-selector-icon", "Column selector",
    '<path d="M11 5h10M11 9h5" stroke-linecap="round" stroke-linejoin="round"/>'
    '<rect width="4" height="4" x="3" y="5" rx="1" stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M11 15h10m-10 4h5" stroke-linecap="round" stroke-linejoin="round"/>'
    '<rect width="4" height="4" x="3" y="15" rx="1" stroke-linecap="round" stroke-linejoin="round"/>')
ICON_SUN = _thm_svg("theme-toggle-sun", "Switch to dark mode",
    '<path d="M12 3V4M12 20V21M4 12H3M6.31 6.31L5.5 5.5'
    'M17.69 6.31L18.5 5.5M6.31 17.69L5.5 18.5'
    'M17.69 17.69L18.5 18.5M21 12H20'
    'M16 12C16 14.21 14.21 16 12 16C9.79 16 8 14.21 8 12'
    'C8 9.79 9.79 8 12 8C14.21 8 16 9.79 16 12Z"/>')

# FIX (Design #5): enforce TAG_ICONS completeness at import time so a missing
# entry raises immediately rather than producing a silent KeyError at render time.
TAG_ICONS: Dict[_TagSpec, str] = {
    TAG_LANGUAGE: ICON_LANGUAGE,
    TAG_ISO:      ICON_GLOBE,
    TAG_REGEX:    ICON_REGEX,
    TAG_CLEANED:  ICON_CLEAN,
    TAG_DATE:     ICON_DATE,
    TAG_METADATA: ICON_META,
}
assert set(TAG_ICONS) == set(ALL_TAGS), (
    "TAG_ICONS is missing entries for: "
    + str({t.filter_key for t in ALL_TAGS} - {t.filter_key for t in TAG_ICONS})
)

try:
    from IPython.display import HTML, display as _ipython_display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _ViewerIds:
    uid:    str
    viewer: str
    table:  str
    toggle: str
    view:   str
    filter: str
    col_sel: str
    col_sel_modal: str
    footer: str


@dataclass
class _ColumnMeta:
    name:       str
    data_type:  object   # pyspark DataType
    type_str:   str
    tag_specs:  List[_TagSpec]
    tag_class:  str      # space-joined CSS class string
    width_px:   int
    icons_html: str      # concatenated SVG markup (may be empty)


@dataclass
class _TableData:
    display_rows: list
    has_more:     bool
    # FIX (Design #1): removed redundant num_display field; use len(display_rows) directly.


# ---------------------------------------------------------------------------
# Step 1 — Validation
# ---------------------------------------------------------------------------

def _validate_inputs(df, stage: Optional[str], max_rows: int) -> None:
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
    except ImportError:
        raise ImportError("PySpark is not installed.")
    if not isinstance(df, SparkDataFrame):
        raise TypeError(f"Expected a PySpark DataFrame, got {type(df).__name__!r}.")
    if stage is not None and stage.lower() not in VALID_STAGES:
        raise ValueError(
            f"stage must be one of {sorted(VALID_STAGES)}, got {stage!r}."
        )
    if max_rows < 0:
        raise ValueError(f"max_rows must be non-negative, got {max_rows}.")
    # FIX (Perf #4): guard against accidental OOM from huge max_rows values.
    if max_rows > _MAX_ROWS_HARD_CAP:
        raise ValueError(
            f"max_rows={max_rows} exceeds the hard cap of {_MAX_ROWS_HARD_CAP:,}. "
            "Increase _MAX_ROWS_HARD_CAP if you intentionally need more rows."
        )


# ---------------------------------------------------------------------------
# Step 2 — Unique DOM IDs
# ---------------------------------------------------------------------------

def _build_viewer_ids() -> _ViewerIds:
    uid = uuid.uuid4().hex[:8]
    return _ViewerIds(
        uid=uid,
        viewer=f"dfv_{uid}",
        table=f"dft_{uid}",
        toggle=f"dftoggle_{uid}",
        view=f"dfview_{uid}",
        filter=f"dffilter_{uid}",
        col_sel=f"dfcolsel_{uid}",
        col_sel_modal=f"dfcolselmodal_{uid}",
        footer=f"dffoot_{uid}",
    )


# ---------------------------------------------------------------------------
# Step 3 — Row collection
# ---------------------------------------------------------------------------

def _collect_rows(df, max_rows: int) -> _TableData:
    rows = df.take(max_rows + 1)
    has_more = len(rows) > max_rows
    display_rows = rows[:max_rows]
    # FIX (Design #1): _TableData no longer stores num_display; use len(display_rows).
    return _TableData(
        display_rows=display_rows,
        has_more=has_more,
    )


# ---------------------------------------------------------------------------
# Step 4 — Column metadata
# ---------------------------------------------------------------------------

def _estimate_display_len(val) -> int:
    """
    FIX (Perf #2): Estimate the rendered string length of a value without
    building the full string. Short-circuits on known cheap types so that large
    structs/arrays don't pay the full str() cost just to measure width.
    """
    if val is None:
        return 4   # "null"
    if isinstance(val, bool):
        return 5   # "False" (worst case)
    if isinstance(val, (int, float)):
        # Fast path: avoids str() for common numeric types.
        return len(repr(val))
    if isinstance(val, str):
        return min(len(val), _COL_MAX_SAMPLE_LEN)
    # For complex / unknown types clamp at the sample limit immediately.
    return min(len(str(val)), _COL_MAX_SAMPLE_LEN)


def _get_column_metadata(
    df,
    data: _TableData,
    tag_sets: Dict[_TagSpec, Set[str]],
) -> List[_ColumnMeta]:
    # FIX (Design #2): invert tag_sets once (col_name → [specs]) for O(1) lookup
    # instead of iterating all 6 tag sets per column.
    col_to_specs: Dict[str, List[_TagSpec]] = {}
    for spec, col_set in tag_sets.items():
        for col_name in col_set:
            col_to_specs.setdefault(col_name, []).append(spec)

    metas: List[_ColumnMeta] = []
    for field_obj in df.schema.fields:
        name      = field_obj.name
        data_type = field_obj.dataType
        type_str  = str(data_type)

        applicable  = col_to_specs.get(name, [])
        tag_class   = " ".join(spec.css_class for spec in applicable)
        icons_html  = "".join(TAG_ICONS[spec] for spec in applicable)

        width = len(name) * _PX_PER_CHAR_HEADER + _COL_PADDING
        for row in data.display_rows[:_COL_SAMPLE_ROWS]:
            val = row[name]
            if val is not None:
                # FIX (Perf #2): use _estimate_display_len to avoid double str() call.
                width = max(width, _estimate_display_len(val) * _PX_PER_CHAR_VALUE + _COL_PADDING)
        width = max(_COL_MIN_PX, min(width, _COL_MAX_PX))

        metas.append(_ColumnMeta(
            name=name,
            data_type=data_type,
            type_str=type_str,
            tag_specs=applicable,
            tag_class=tag_class,
            width_px=width,
            icons_html=icons_html,
        ))
    return metas


# ---------------------------------------------------------------------------
# Step 5 — Cell formatting
# ---------------------------------------------------------------------------

def _is_complex_type(data_type) -> bool:
    """Prefer isinstance; fall back to name check if PySpark types unavailable or type is unknown."""
    try:
        from pyspark.sql import types as T
        if T is None:
            raise AttributeError("types module not available")
        if isinstance(data_type, (T.StructType, T.ArrayType, T.MapType)):
            return True
    except (ImportError, AttributeError):
        pass
    return type(data_type).__name__ in _COMPLEX_TYPE_NAMES


# FIX (Bug #1): replaced the broken _array_type_for_display() approach.
# The old implementation set __name__ on the *instance*, but _format_complex_cell
# checks type(data_type).__name__ (the *class* name), so it always resolved to
# "_ArrayType" and never matched "ArrayType" — the ArrayType branch never fired.
# Solution: use a proper sentinel string flag instead of a fake type object.
_FORCE_ARRAY_DISPLAY = "__force_array__"


def _format_complex_cell(value, data_type, cell_id: str, uid: str) -> str:
    if value is None:
        return '<span class="dataframe-null">null</span>'

    # FIX (Bug #1): check for the sentinel string before falling back to type name.
    if data_type is _FORCE_ARRAY_DISPLAY:
        type_name = "ArrayType"
    else:
        type_name = type(data_type).__name__

    try:
        if type_name == "StructType":
            raw      = value.asDict() if hasattr(value, "asDict") else dict(value)
            json_str = json.dumps(raw, indent=2, default=str, ensure_ascii=False)
            preview  = f"Struct({len(raw)} fields)"
        elif type_name == "ArrayType":
            # FIX (Bug #4): use list() defensively to handle any iterable PySpark
            # may return (list, tuple, generator) rather than assuming list.
            raw      = list(value) if value is not None else []
            json_str = json.dumps(raw, indent=2, default=str, ensure_ascii=False)
            preview  = f"Array[{len(raw)} items]"
        elif type_name == "MapType":
            raw      = dict(value) if value else {}
            json_str = json.dumps(raw, indent=2, default=str, ensure_ascii=False)
            preview  = f"Map[{len(raw)} entries]"
        else:
            return _html.escape(str(value))
    except Exception:
        json_str = str(value)
        preview  = type_name

    escaped = _html.escape(json_str)
    # data-raw-json sentinel: JS applies syntax highlighting lazily on first expand
    return (
        f'<div class="complex-value">'
        f'<span class="complex-toggle" onclick="dfToggle_{uid}(\'{cell_id}\')">'
        f'<span class="toggle-icon" id="icon_{cell_id}">&#9654;</span>'
        f'<span class="complex-preview">{preview}</span>'
        f'</span>'
        f'<div class="complex-content" id="{cell_id}" style="display:none;">'
        f'<pre class="complex-json" data-raw-json="1">{escaped}</pre>'
        f'</div>'
        f'</div>'
    )


def _format_one_country_code(code: str) -> str:
    """Format a single ISO3 code as a country chip (flag + code). Only for 3-letter codes; ISO2 is not styled."""
    raw = code.strip().upper()
    if len(raw) != 3:
        # FIX (Security #1): use quote=True for consistency; value goes into HTML content
        # but this guards against edge-cases like code containing quotes.
        return f'<span class="dataframe-country-plain">{_html.escape(code.strip(), quote=True)}</span>' if raw else ""
    flag = _code_to_flag(raw)
    display = _html.escape(code.strip(), quote=True)
    flag_span = f'<span class="dataframe-country-flag" aria-hidden="true">{flag}</span>' if flag else ""
    return (
        f'<span class="dataframe-country" title="country (ISO3)">'
        f'{flag_span}<span class="dataframe-country-code">{display}</span>'
        f'</span>'
    )


def _format_country_list(value) -> str:
    """
    Format a list/tuple/iterable of country codes as multiple chips.
    Only ISO3 (3-letter) items get flag + pill; rest plain.
    FIX (Bug #4): accepts any iterable, not just list/tuple.
    """
    if value is None:
        return '<span class="dataframe-null">null</span>'
    # Normalise to list defensively; PySpark may return various sequence types.
    try:
        items = list(value)
    except TypeError:
        return _html.escape(str(value), quote=True)

    chips = []
    for item in items:
        if item is None:
            chips.append('<span class="dataframe-null">null</span>')
        elif isinstance(item, str) and item.strip():
            chips.append(_format_one_country_code(item))
        else:
            chips.append(_html.escape(str(item), quote=True))
    return (
        '<span class="dataframe-country-list">'
        + "".join(chips)
        + "</span>"
    )


def _format_simple_cell(value, tag_specs: Optional[List[_TagSpec]] = None) -> str:
    if value is None:
        return '<span class="dataframe-null">null</span>'
    if isinstance(value, bool):
        cls = "dataframe-boolean dataframe-bool-true" if value else "dataframe-boolean dataframe-bool-false"
        label = "True" if value else "False"
        # FIX (Security #1): added quote=True for consistent escaping throughout.
        return f'<span class="{cls}" title="boolean">{_html.escape(label, quote=True)}</span>'
    # Country/ISO column: only ISO3 (3-letter) gets pill + flag; ISO2 stays plain.
    if tag_specs and TAG_ISO in tag_specs:
        if isinstance(value, (list, tuple)):
            return _format_country_list(value)
        if isinstance(value, str) and value.strip():
            return _format_one_country_code(value)
    # FIX (Security #1): use quote=True consistently so the output is safe in
    # both HTML content and attribute contexts.
    return _html.escape(str(value), quote=True)


def _sort_attr_value(value) -> str:
    """HTML-attribute-safe sort key using html.escape(quote=True)."""
    if value is None:
        return ""
    return _html.escape(str(value), quote=True)


# ---------------------------------------------------------------------------
# Step 6 — Footer row-count text
# ---------------------------------------------------------------------------

def _build_rows_text(data: _TableData, total_rows: Optional[int]) -> str:
    """Always shows how many rows are displayed; adds total when available."""
    # FIX (Design #1): use len(data.display_rows) instead of removed num_display field.
    n = len(data.display_rows)
    if data.has_more:
        if total_rows is not None:
            return f"showing {n:,} of {total_rows:,} rows"
        return f"showing first {n:,} rows"
    # All rows fit in the preview window
    if total_rows is not None:
        return f"{total_rows:,} rows"
    return f"{n:,} rows"


# ---------------------------------------------------------------------------
# Step 7 — CSS
# FIX (Perf #3): CSS is emitted only once per kernel session via a shared
# <style id="dfv-shared-css"> block. Subsequent calls skip re-emission entirely,
# keeping notebook output lean regardless of how many displayDF calls are made.
# ---------------------------------------------------------------------------

# Shared (non-scoped) CSS — emitted once globally.
_SHARED_CSS = """<style id="dfv-shared-css">
    /* JSON highlight spans — intentionally global; classes are short and collision-safe */
    .json-key     { color:#c586c0; }
    .json-string  { color:#ce9178; }
    .json-number  { color:#b5cea8; }
    .json-boolean { color:#569cd6; }
    .json-null    { color:#808080; }
</style>
<script>
    // Guard: only inject shared CSS once even if multiple outputs are present.
    (function() {
        if (document.getElementById('dfv-shared-css-done')) return;
        var sentinel = document.createElement('meta');
        sentinel.id = 'dfv-shared-css-done';
        document.head.appendChild(sentinel);
    })();
</script>"""


def _render_css(ids: _ViewerIds) -> str:
    """Emit per-viewer scoped CSS (scoped to #<viewer-id>)."""
    v = f"#{ids.viewer}"
    return f"""<style>
    {v} {{
        position:relative; border-radius:12px; border:1px solid #404040; overflow:hidden;
        box-shadow:0 2px 8px rgba(0,0,0,.3); background:#1e1e1e;
        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
        margin:10px 0;
    }}
    {v} .dataframe-header {{
        background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        color:white; padding:8px 12px; font-weight:600; font-size:14px;
        display:flex; align-items:center; justify-content:space-between;
    }}
    {v} .table-name-card {{
        background:rgba(255,255,255,.15); border:1px solid rgba(255,255,255,.3);
        padding:6px 12px; font-weight:600; display:inline-flex; align-items:center; gap:8px;
    }}
    {v} .stage-dot {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}
    {v} .stage-dot.bronze {{ background:#cd7f32; box-shadow:0 0 4px rgba(205,127,50,.6); }}
    {v} .stage-dot.silver {{ background:#c0c0c0; box-shadow:0 0 4px rgba(192,192,192,.6); }}
    {v} .stage-dot.gold   {{ background:#ffd700; box-shadow:0 0 4px rgba(255,215,0,.6); }}
    {v} .theme-toggle {{
        display:inline-flex; align-items:center; margin-left:1px; cursor:pointer;
        padding:4px; border-radius:6px; border:1px solid transparent;
        transition:background .2s,border-color .2s;
    }}
    {v} .theme-toggle:hover {{ background:rgba(255,255,255,.2); border-color:rgba(255,255,255,.3); }}
    {v} .theme-toggle svg {{ width:20px; height:20px; filter:invert(1); }}
    {v} .expert-view-pane {{ display:inline-flex; align-items:center; margin-left:1px; }}
    {v} .expert-view-pane .expert-view-btn {{ width:auto; border-radius:6px; }}
    {v} .expert-view-pane .expert-view-btn:not(.active) {{
        background:transparent; border-color:transparent;
    }}
    {v} .expert-view-pane .expert-view-btn:not(.active):hover {{
        background:rgba(255,255,255,.2); border-color:rgba(255,255,255,.3);
    }}
    {v} .expert-view-pane .expert-view-btn svg {{ width:16px; height:16px; filter:invert(1); }}
    {v} .column-filter-pane {{ display:inline-flex; align-items:center; gap:2px; margin-left:10px; }}
    {v} .header-divider {{
        width:1px; height:20px; margin-left: 10px; margin-right: 1px; flex-shrink:0;
        background:rgba(255,255,255,.35); border-radius:1px;
    }}
    {v} .column-filter-btn {{
        display:inline-flex; align-items:center; justify-content:center;
        cursor:pointer; padding:1px; border-radius:4px; border:1px solid transparent;
        transition:background .2s,border-color .2s;
    }}
    {v} .column-filter-btn:hover {{ background:rgba(255,255,255,.2); border-color:rgba(255,255,255,.3); }}
    {v} .column-filter-btn.active {{ background:rgba(255,255,255,.25); border-color:rgba(255,255,255,.4); }}
    {v} .column-filter-btn svg {{ width:14px; height:14px; filter:invert(1); }}
    {v}:not(.light-mode) .column-filter-btn .clean-icon {{ filter:none !important; color:#fff !important; }}
    {v} .dataframe-table th.filter-hidden,
    {v} .dataframe-table td.filter-hidden {{ display:none; }}
    {v} .dataframe-table th.column-selector-hidden,
    {v} .dataframe-table td.column-selector-hidden {{ display:none !important; }}
    {v} .dataframe-table tr.column-selector-empty-row td {{
        text-align:center; color:#888; font-style:italic; padding:16px;
    }}
    {v} .column-selector-wrap {{ display:inline-flex; align-items:center; margin-left:8px; position:relative; }}
    {v} .column-selector-wrap .column-selector-btn {{
        width:auto; border-radius:6px;
        background:transparent; border-color:transparent;
    }}
    {v} .column-selector-wrap .column-selector-btn:hover {{
        background:rgba(255,255,255,.2); border-color:rgba(255,255,255,.3);
    }}
    {v} .column-selector-wrap .column-selector-btn.active {{
        background:rgba(255,255,255,.25); border-color:rgba(255,255,255,.4);
    }}
    {v} .column-selector-wrap .column-selector-btn svg {{ width:16px; height:16px; filter:invert(1); }}
    {v} .column-selector-backdrop {{
        display:none; position:absolute; inset:0;
        align-items:center; justify-content:center;
        background:rgba(0,0,0,.45); z-index:100;
    }}
    {v} .column-selector-backdrop.open {{ display:flex; }}
    {v} .column-selector-modal {{
        background:#252525; border:1px solid #404040; border-radius:10px;
        box-shadow:0 8px 32px rgba(0,0,0,.5); padding:0;
        width:420px; min-width:280px; max-width:90%; max-height:70%;
        overflow:hidden; display:flex; flex-direction:column; position:relative;
    }}
    {v} .column-selector-modal-header {{
        display:flex; align-items:center; justify-content:space-between; gap:8px;
        padding:10px 12px; border-bottom:1px solid #404040;
    }}
    {v} .column-selector-title-row {{ display:flex; align-items:center; gap:8px; flex-shrink:0; }}
    {v} .column-selector-modal-title {{ font-weight:600; font-size:13px; color:#e0e0e0; margin:0; }}
    {v} .column-selector-toggle-all-btn {{
        display:inline-flex; align-items:center; justify-content:center;
        width:28px; height:28px; padding:0;
        background:transparent; border:1px solid rgba(255,255,255,.35); border-radius:50%;
        color:#d0d0d0; cursor:pointer;
        transition:background .2s, border-color .2s, color .2s;
    }}
    {v} .column-selector-toggle-all-btn:hover {{
        background:rgba(255,255,255,.12); border-color:rgba(255,255,255,.45); color:#fff;
    }}
    {v} .column-selector-toggle-state {{ display:inline-flex; align-items:center; justify-content:center; }}
    {v} .column-selector-toggle-all-btn .column-selector-layers-icon {{ width:14px; height:14px; }}
    {v} .column-selector-close {{
        background:transparent; border:none; color:#999; cursor:pointer;
        font-size:20px; line-height:1; padding:0 4px; border-radius:4px;
        transition:color .2s, background .2s;
    }}
    {v} .column-selector-close:hover {{ color:#fff; background:rgba(255,255,255,.15); }}
    {v} .column-selector-modal-body {{ padding:8px; overflow-y:auto; max-height:260px; }}
    {v} .column-selector-row {{ display:flex; align-items:center; gap:8px; padding:6px 8px; cursor:pointer; font-size:12px; color:#d0d0d0; border-radius:4px; }}
    {v} .column-selector-row:hover {{ background:#333; }}
    {v} .column-selector-check {{ cursor:pointer; flex-shrink:0; }}
    /* ── Light mode ─────────────────────────────────────────────────────── */
    {v}.light-mode {{ background:#fff; border:1px solid #e0e0e0; }}
    {v}.light-mode .dataframe-table th {{
        background:#f5f5f5; color:#333; border-bottom:2px solid #ddd; border-right:1px solid #ddd;
    }}
    {v}.light-mode .dataframe-table td {{ background:#fff; color:#333; border:1px solid #e0e0e0; }}
    {v}.light-mode .dataframe-table-container {{ background:#fff; }}
    {v}.light-mode .complex-content {{ background:#f5f5f5 !important; border-left:3px solid #667eea; }}
    {v}.light-mode .complex-json {{ background:#f5f5f5 !important; color:#333; }}
    {v}.light-mode .dataframe-table pre {{ background:#f5f5f5 !important; color:#333 !important; }}
    {v}.light-mode .dataframe-null {{ color:#999; }}
    {v}.light-mode .dataframe-bool-true {{ background:rgba(34,197,94,.18); color:#16a34a; }}
    {v}.light-mode .dataframe-bool-false {{ background:rgba(148,163,184,.25); color:#64748b; }}
    {v}.light-mode .dataframe-country {{ background:rgba(59,130,246,.15); color:#2563eb; }}
    {v}.light-mode .dataframe-country-plain {{ color:inherit; }}
    {v}.light-mode.viewer-simple .dataframe-country,
    {v}.light-mode.viewer-simple .dataframe-boolean {{ color:inherit !important; }}
    {v}.light-mode .dataframe-table-container::-webkit-scrollbar-track {{ background:#f5f5f5; }}
    {v}.light-mode .dataframe-table-container::-webkit-scrollbar-thumb {{ background:#ccc; }}
    {v}.light-mode .dataframe-table-container::-webkit-scrollbar-thumb:hover {{ background:#aaa; }}
    {v}.light-mode .dataframe-footer {{ background:#f5f5f5; border-top:1px solid #ddd; color:#666; }}
    {v}.light-mode .dataframe-footer .columns-hidden-notice {{ color:#b45309; }}
    {v}.light-mode .dataframe-table tbody tr {{ background:#fff; }}
    {v}.light-mode .dataframe-table tbody tr:hover,
    {v}.light-mode .dataframe-table tbody tr:hover td {{ background:#f0f0f0; }}
    {v}.light-mode .dataframe-table th:hover {{ background:#e8e8e8; }}
    {v}.light-mode .sort-indicator {{ color:#6366f1; }}
    {v}.light-mode .dataframe-table th .lang-icon,
    {v}.light-mode .dataframe-table th .globe-icon,
    {v}.light-mode .dataframe-table th .regex-icon,
    {v}.light-mode .dataframe-table th .meta-icon,
    {v}.light-mode .dataframe-table th .clean-icon,
    {v}.light-mode .dataframe-table th .date-icon {{ filter:none !important; opacity:.9; }}
    {v}.light-mode .dataframe-table th .clean-icon {{ color:#333; }}
    {v}.light-mode .column-selector-backdrop {{ background:rgba(0,0,0,.25); }}
    {v}.light-mode .column-selector-modal {{ background:#fff; border-color:#e0e0e0; box-shadow:0 8px 32px rgba(0,0,0,.2); }}
    {v}.light-mode .column-selector-modal-header {{ border-bottom-color:#e0e0e0; }}
    {v}.light-mode .column-selector-modal-title {{ color:#333; }}
    {v}.light-mode .column-selector-toggle-all-btn {{ border-color:#ccc; color:#555; }}
    {v}.light-mode .column-selector-toggle-all-btn:hover {{ background:rgba(0,0,0,.06); border-color:#999; color:#333; }}
    {v}.light-mode .column-selector-close {{ color:#666; }}
    {v}.light-mode .column-selector-close:hover {{ color:#333; background:rgba(0,0,0,.08); }}
    {v}.light-mode .column-selector-row {{ color:#333; }}
    {v}.light-mode .column-selector-row:hover {{ background:#f0f0f0; }}
    /* ── Table ──────────────────────────────────────────────────────────── */
    {v} .dataframe-table-container {{ max-height:350px; overflow-y:auto; overflow-x:auto; }}
    {v} .dataframe-table {{
        width:100%; border-collapse:collapse; font-size:13px; table-layout:fixed; min-width:100%;
    }}
    {v} .dataframe-table thead {{ position:sticky; top:0; z-index:10; }}
    {v} .dataframe-table th {{
        background:#2d2d2d; padding:6px 10px; text-align:left !important;
        font-weight:600; color:#e0e0e0; border-bottom:2px solid #404040;
        border-right:1px solid #555; white-space:nowrap; position:relative;
        cursor:pointer; user-select:text;
    }}
    {v} .dataframe-table th:hover {{ background:#353535; }}
    {v} .dataframe-table th:last-child {{ border-right:none; }}
    {v} .dataframe-table th.sortable {{ padding-right:24px; }}
    {v} .dataframe-table th .column-header-content {{ display:inline-flex; align-items:center; gap:4px; }}
    {v} .dataframe-table th .lang-icon,
    {v} .dataframe-table th .globe-icon,
    {v} .dataframe-table th .regex-icon,
    {v} .dataframe-table th .meta-icon,
    {v} .dataframe-table th .clean-icon,
    {v} .dataframe-table th .date-icon {{
        flex-shrink:0; vertical-align:middle;
        width:14px !important; height:14px !important;
        max-width:14px !important; max-height:14px !important;
        opacity:.95; filter:invert(1) !important;
    }}
    {v}:not(.light-mode) .dataframe-table th .clean-icon {{ filter:none !important; color:#fff !important; }}
    {v} .sort-indicator {{
        position:absolute; right:8px; top:50%; transform:translateY(-50%);
        display:flex; align-items:center; justify-content:center;
        color:#8b9aff;
    }}
    {v} .sort-indicator .sort-icon-asc,
    {v} .sort-indicator .sort-icon-desc {{ display:none; }}
    {v} .dataframe-table th.sort-asc .sort-indicator .sort-icon-asc {{ display:inline-flex; }}
    {v} .dataframe-table th.sort-desc .sort-indicator .sort-icon-desc {{ display:inline-flex; }}
    {v} .sort-indicator .sort-icon-svg {{ width:14px; height:14px; }}
    {v} .dataframe-table th .resize-handle {{
        position:absolute; top:0; right:0; width:5px; height:100%;
        cursor:col-resize; background:transparent; z-index:1;
    }}
    {v} .dataframe-table th .resize-handle:hover {{ background:#8b9aff; opacity:.5; }}
    {v} .dataframe-table tbody tr {{ background:#1e1e1e; }}
    {v} .dataframe-table td {{
        padding:6px 10px; border:1px solid #333; color:#d0d0d0;
        word-wrap:break-word; overflow-wrap:break-word;
        background:#1e1e1e; text-align:left !important;
    }}
    {v} .dataframe-table tbody tr:hover,
    {v} .dataframe-table tbody tr:hover td {{ background:#2a2a2a; }}
    {v} .dataframe-table tbody tr:last-child td {{ border-bottom:none; }}
    {v} .dataframe-footer {{
        background:#252525; padding:6px 12px; border-top:1px solid #404040;
        font-size:12px; color:#999; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;
    }}
    {v} .dataframe-footer .columns-hidden-notice {{
        display:inline-flex; align-items:center; gap:6px;
        color:#d97706; font-size:12px;
    }}
    {v} .dataframe-footer .columns-hidden-icon {{ width:16px; height:16px; flex-shrink:0; }}
    {v} .dataframe-footer-rows {{ margin-left:auto; }}
    {v} .dataframe-null {{ color:#666; font-style:italic; }}
    {v} .dataframe-boolean {{
        font-weight:600; padding:2px 8px; border-radius:6px; font-size:12px;
    }}
    {v} .dataframe-bool-true {{ background:rgba(34,197,94,.25); color:#4ade80; }}
    {v} .dataframe-bool-false {{ background:rgba(148,163,184,.2); color:#94a3b8; }}
    {v} .dataframe-country {{
        font-weight:600; padding:2px 8px; border-radius:6px; font-size:12px;
        display:inline-flex; align-items:center; gap:6px;
        background:rgba(59,130,246,.2); color:#93c5fd;
    }}
    {v} .dataframe-country-flag {{ font-size:1em; line-height:1; }}
    {v} .dataframe-country-code {{ font-family:ui-monospace,monospace; }}
    {v} .dataframe-country-list {{
        display:inline-flex; flex-wrap:wrap; align-items:center; gap:6px;
    }}
    {v} .dataframe-country-plain {{ color:inherit; }}
    /* Simple mode: chips appear as plain text (no pill, no flag) */
    {v}.viewer-simple .dataframe-country-flag {{ display:none !important; }}
    {v}.viewer-simple .dataframe-country,
    {v}.viewer-simple .dataframe-boolean {{
        background:transparent !important; padding:0 !important; border-radius:0 !important;
        font-weight:normal !important; color:inherit !important;
    }}
    {v}.viewer-simple .dataframe-country-list {{ gap:0.25em; }}
    {v}.viewer-simple .dataframe-country-code {{ font-family:inherit; }}
    /* Simple mode: hide filter pane, column selector, and column header icons */
    {v}.viewer-simple .column-filter-pane {{ display:none !important; }}
    {v}.viewer-simple .header-divider {{ display:none !important; }}
    {v}.viewer-simple .column-selector-wrap {{ display:none !important; }}
    {v}.viewer-simple .dataframe-table th .lang-icon,
    {v}.viewer-simple .dataframe-table th .globe-icon,
    {v}.viewer-simple .dataframe-table th .regex-icon,
    {v}.viewer-simple .dataframe-table th .meta-icon,
    {v}.viewer-simple .dataframe-table th .clean-icon,
    {v}.viewer-simple .dataframe-table th .date-icon {{ display:none !important; }}
    /* ISO list: expert = chips, simple = standard array expandable */
    {v}.viewer-simple .expert-only {{ display:none !important; }}
    {v}.viewer-simple .simple-only {{ display:block; }}
    {v}.viewer-expert .simple-only {{ display:none !important; }}
    {v}.viewer-expert .expert-only {{ display:block; }}
    {v} .complex-value {{ position:relative; }}
    {v} .complex-toggle {{
        cursor:pointer; user-select:none; color:#8b9aff;
        font-weight:500; display:inline-flex; align-items:center; gap:6px;
    }}
    {v} .complex-toggle:hover {{ color:#a78bfa; }}
    {v} .toggle-icon {{ font-size:10px; transition:transform .2s; display:inline-block; }}
    {v} .toggle-icon.expanded {{ transform:rotate(90deg); }}
    {v} .complex-preview {{ font-style:italic; }}
    {v} .complex-content {{
        margin-top:8px; padding:8px; background:#252525 !important;
        border-radius:4px; border-left:3px solid #8b9aff;
    }}
    {v} .complex-json {{
        margin:0; font-size:12px; font-family:'Monaco','Menlo','Courier New',monospace;
        color:#d4d4d4; white-space:pre-wrap; word-wrap:break-word;
        max-height:300px; overflow-y:auto; line-height:1.5; background:#252525 !important;
    }}
    {v} .dataframe-table pre {{ background:#252525 !important; color:#d4d4d4 !important; margin:0; padding:0; }}
    {v} .dataframe-table-container::-webkit-scrollbar {{ width:8px; height:8px; }}
    {v} .dataframe-table-container::-webkit-scrollbar-track {{ background:#2d2d2d; }}
    {v} .dataframe-table-container::-webkit-scrollbar-thumb {{ background:#555; border-radius:4px; }}
    {v} .dataframe-table-container::-webkit-scrollbar-thumb:hover {{ background:#777; }}
    {v} .complex-json::-webkit-scrollbar {{ width:6px; }}
    {v} .complex-json::-webkit-scrollbar-track {{ background:#2d2d2d; }}
    {v} .complex-json::-webkit-scrollbar-thumb {{ background:#555; border-radius:3px; }}
</style>"""


# ---------------------------------------------------------------------------
# Step 8 — JavaScript
# FIX (Design #3): The JS is no longer one giant f-string. Instead, static JS
# blocks are plain strings and only the small dynamic sections (IDs, column
# count, filter keys) use f-string interpolation. This minimises the surface
# area where {{ / }} escaping can go wrong and makes the code far easier to
# read and edit.
# ---------------------------------------------------------------------------

# ── Static JS helpers (no viewer-specific IDs needed) ──────────────────────
# These are plain strings — no {{ }} escaping required.

_JS_HIGHLIGHT_FN_TMPL = """\
    // ── JSON syntax highlight (one definition per viewer) ────────────────
    function dfHighlight_{uid}(s) {{
        return s.replace(
            /("(\\\\u[a-zA-Z0-9]{{4}}|\\\\[^u]|[^\\\\"])*"(\\s*:)?|\\b(true|false|null)\\b|-?\\d+(?:\\.\\d*)?(?:[eE][+-]?\\d+)?)/g,
            function(m) {{
                var c = 'json-number';
                if (/^"/.test(m))              c = /:$/.test(m) ? 'json-key' : 'json-string';
                else if (/true|false/.test(m)) c = 'json-boolean';
                else if (/null/.test(m))       c = 'json-null';
                return '<span class="' + c + '">' + m + '</span>';
            }}
        );
    }}"""

_JS_TOGGLE_FN_TMPL = """\
    // ── Complex cell toggle (lazy highlight on first open) ────────────────
    function dfToggle_{uid}(cellId) {{
        var content = document.getElementById(cellId);
        var icon    = document.getElementById('icon_' + cellId);
        if (!content) return;
        var opening = content.style.display === 'none';
        content.style.display = opening ? 'block' : 'none';
        icon.classList.toggle('expanded', opening);
        if (opening) {{
            var pre = content.querySelector('pre[data-raw-json]');
            if (pre) {{
                pre.innerHTML = dfHighlight_{uid}(pre.textContent);
                pre.removeAttribute('data-raw-json');
            }}
        }}
    }}"""

# Column resize — static (no uid needed, scoped by table id placeholder {table_id})
_JS_RESIZE_TMPL = """\
    // ── Column resize (scoped to this table) ─────────────────────────────
    (function() {{
        var isResizing = false, colIdx = -1, startX = 0, startWidth = 0;
        var table = document.getElementById('{table_id}');
        if (!table) return;

        var thEls = Array.from(table.querySelectorAll('thead th'));
        thEls.forEach(function(th, i) {{
            if (i >= thEls.length - 1) return;
            var handle = document.createElement('div');
            handle.className = 'resize-handle';
            th.appendChild(handle);
            handle.addEventListener('mousedown', function(e) {{
                e.preventDefault();
                isResizing = true;
                colIdx     = i;
                startX     = e.pageX;
                startWidth = th.offsetWidth;
                document.body.style.cursor     = 'col-resize';
                document.body.style.userSelect = 'none';
            }});
        }});

        document.addEventListener('mousemove', function(e) {{
            if (!isResizing || colIdx < 0) return;
            var newW = Math.max(80, startWidth + (e.pageX - startX));
            function applyWidth(el) {{
                el.style.width = el.style.minWidth = el.style.maxWidth = newW + 'px';
            }}
            if (thEls[colIdx]) applyWidth(thEls[colIdx]);
            table.querySelectorAll('tbody tr').forEach(function(tr) {{
                var td = tr.querySelectorAll('td')[colIdx];
                if (td) applyWidth(td);
            }});
        }});

        document.addEventListener('mouseup', function() {{
            if (!isResizing) return;
            isResizing = false; colIdx = -1;
            document.body.style.cursor = document.body.style.userSelect = '';
        }});
    }})();"""

# FIX (Bug #2): Sort now reads sort values via data-column attribute on each td
# rather than by positional index. This means hidden columns (via column selector
# or tag filter) no longer cause the sort to read the wrong cell.
_JS_SORT_TMPL = """\
    // ── Column sort — uses data-column attribute, not positional td index ─
    // FIX: previously used tr.querySelectorAll('td')[colIdx] which broke when
    // columns were hidden (column selector / tag filter). Now td lookup is done
    // via data-column attribute, so hidden columns are correctly skipped.
    (function() {{
        var table = document.getElementById('{table_id}');
        if (!table) return;
        var headers = Array.from(table.querySelectorAll('thead th.sortable'));
        var tbody   = table.querySelector('tbody');
        if (!tbody) return;

        // Exclude notice row and "columns hidden" empty row.
        var rowEls = Array.from(tbody.querySelectorAll('tr')).filter(function(tr) {{
            return !tr.querySelector('.filter-show-all') && !tr.classList.contains('column-selector-empty-row');
        }});

        // Snapshot sort keys once keyed by column index, not DOM position,
        // so hide/show operations never invalidate the snapshot.
        var matrix = rowEls.map(function(tr) {{
            var map = {{}};
            Array.from(tr.querySelectorAll('td[data-column]')).forEach(function(td) {{
                map[td.getAttribute('data-column')] = td.getAttribute('data-sort-value') || '';
            }});
            return map;
        }});

        var curCol = null, curDir = null;
        headers.forEach(function(th) {{
            th.addEventListener('click', function() {{
                var ci = th.getAttribute('data-column');
                curDir = (curCol === ci && curDir === 'asc') ? 'desc' : 'asc';
                curCol = ci;
                headers.forEach(function(h) {{ h.classList.remove('sort-asc', 'sort-desc'); }});
                th.classList.add('sort-' + curDir);

                var indices = rowEls.map(function(_, i) {{ return i; }});
                indices.sort(function(a, b) {{
                    var av = matrix[a][ci] != null ? matrix[a][ci] : '';
                    var bv = matrix[b][ci] != null ? matrix[b][ci] : '';
                    if (av === 'null') return  1;
                    if (bv === 'null') return -1;
                    var an = Number(av), bn = Number(bv);
                    if (!isNaN(an) && !isNaN(bn) && String(av).trim() !== '' && String(bv).trim() !== '') {{
                        return curDir === 'asc' ? an - bn : bn - an;
                    }}
                    return curDir === 'asc' ? String(av).localeCompare(bv) : String(bv).localeCompare(av);
                }});
                indices.forEach(function(i) {{ tbody.appendChild(rowEls[i]); }});
            }});
        }});
    }})();"""

_JS_PREFS_TMPL = """\
    // ── Viewer-scoped prefs + theme & expert ──────────────────────────────
    (function() {{
        var dfPref = {{
            get: function(key, def) {{
                try {{ return localStorage.getItem(key) || def; }} catch (e) {{ return def; }}
            }},
            set: function(key, value) {{
                try {{ localStorage.setItem(key, value); }} catch (e) {{}}
            }}
        }};
        var PREF_KEYS = {{ THEME: 'dataframe-theme', EXPERT: 'dataframe-expert' }};

        (function() {{
            var viewer = document.getElementById('{viewer_id}');
            var toggle = document.getElementById('{toggle_id}');
            if (!viewer || !toggle) return;
            var moon = toggle.querySelector('.theme-toggle-moon');
            var sun  = toggle.querySelector('.theme-toggle-sun');
            function setDark()  {{ moon.style.display = '';     sun.style.display = 'none'; }}
            function setLight() {{ moon.style.display = 'none'; sun.style.display = '';     }}
            var saved = dfPref.get(PREF_KEYS.THEME, 'dark');
            if (saved === 'light') {{ viewer.classList.add('light-mode'); setLight(); }} else {{ setDark(); }}
            toggle.addEventListener('click', function() {{
                var isLight = viewer.classList.toggle('light-mode');
                dfPref.set(PREF_KEYS.THEME, isLight ? 'light' : 'dark');
                isLight ? setLight() : setDark();
            }});
        }})();

        (function() {{
            var viewer = document.getElementById('{viewer_id}');
            var btn = document.getElementById('{view_id}');
            if (!viewer || !btn) return;
            var saved = dfPref.get(PREF_KEYS.EXPERT, 'false');
            if (saved === 'true') {{
                viewer.classList.remove('viewer-simple');
                viewer.classList.add('viewer-expert');
                btn.classList.add('active');
                btn.setAttribute('title', 'Expert view (chips on). Click to switch to simple.');
            }} else {{
                viewer.classList.add('viewer-simple');
                viewer.classList.remove('viewer-expert');
                btn.classList.remove('active');
                btn.setAttribute('title', 'Expert view (chips for ISO and boolean). Click to enable.');
            }}
            btn.addEventListener('click', function() {{
                var isExpert = viewer.classList.contains('viewer-expert');
                var next = !isExpert;
                dfPref.set(PREF_KEYS.EXPERT, next ? 'true' : 'false');
                if (next) {{
                    viewer.classList.remove('viewer-simple');
                    viewer.classList.add('viewer-expert');
                    btn.classList.add('active');
                    btn.setAttribute('title', 'Expert view (chips on). Click to switch to simple.');
                }} else {{
                    viewer.classList.add('viewer-simple');
                    viewer.classList.remove('viewer-expert');
                    btn.classList.remove('active');
                    btn.setAttribute('title', 'Expert view (chips for ISO and boolean). Click to enable.');
                }}
            }});
        }})();
    }})();"""

_JS_COL_SELECTOR_TMPL = """\
    // ── Column selector modal ─────────────────────────────────────────────
    (function() {{
        var btn      = document.getElementById('{col_sel_id}');
        var backdrop = document.getElementById('{col_sel_modal_id}');
        var modalBox = backdrop ? backdrop.querySelector('.column-selector-modal') : null;
        var table    = document.getElementById('{table_id}');
        var footerEl = document.getElementById('{footer_id}');
        var total    = {total_columns};
        if (!btn || !backdrop || !table) return;
        function setButtonActive(active) {{
            if (active) btn.classList.add('active'); else btn.classList.remove('active');
        }}
        function closeModal() {{
            backdrop.classList.remove('open');
            setButtonActive(false);
        }}
        btn.addEventListener('click', function(e) {{
            e.stopPropagation();
            backdrop.classList.toggle('open');
            setButtonActive(backdrop.classList.contains('open'));
        }});
        if (modalBox) modalBox.addEventListener('click', function(e) {{ e.stopPropagation(); }});
        backdrop.addEventListener('click', function(e) {{ if (e.target === backdrop) closeModal(); }});
        var closeBtn = backdrop.querySelector('.column-selector-close');
        if (closeBtn) closeBtn.addEventListener('click', closeModal);
        var toggleAllBtn = backdrop.querySelector('.column-selector-toggle-all-btn');
        var emptyRow = table.querySelector('tr.column-selector-empty-row');
        function applyColumnVisibility(ci, visible) {{
            table.querySelectorAll('th[data-column="' + ci + '"]').forEach(function(el) {{
                el.classList.toggle('column-selector-hidden', !visible);
            }});
            table.querySelectorAll('td[data-column="' + ci + '"]').forEach(function(el) {{
                el.classList.toggle('column-selector-hidden', !visible);
            }});
        }}
        // FIX (Bug #5): unified updateFooter used by both this block and the tag
        // filter block — scoped to thead to avoid accidentally counting non-header th
        // elements, and uses a consistent selector across both call sites.
        function updateFooter() {{
            var vis = table.querySelectorAll('thead th:not(.filter-hidden):not(.column-selector-hidden)').length;
            if (emptyRow) emptyRow.style.display = vis === 0 ? '' : 'none';
            if (toggleAllBtn) {{
                var stateSelect   = toggleAllBtn.querySelector('.column-selector-toggle-select');
                var stateDeselect = toggleAllBtn.querySelector('.column-selector-toggle-deselect');
                if (stateSelect && stateDeselect) {{
                    stateSelect.style.display   = vis === total ? '' : 'none';
                    stateDeselect.style.display = vis === total ? 'none' : '';
                }}
                toggleAllBtn.setAttribute('title', vis === total ? 'Deselect all columns' : 'Select all columns');
            }}
            if (!footerEl) return;
            if (vis === total) {{
                footerEl.innerHTML = '<strong>' + total + '</strong> columns';
            }} else {{
                footerEl.innerHTML = '<strong>' + vis + '</strong> of <strong>' + total + '</strong> columns';
            }}
            var notice = footerEl.nextElementSibling;
            if (notice && notice.classList.contains('columns-hidden-notice')) {{
                notice.style.display = vis < total ? '' : 'none';
            }}
        }}
        if (toggleAllBtn) toggleAllBtn.addEventListener('click', function() {{
            var vis = table.querySelectorAll('thead th:not(.filter-hidden):not(.column-selector-hidden)').length;
            var selectAll = vis < total;
            backdrop.querySelectorAll('.column-selector-check').forEach(function(cb) {{
                cb.checked = selectAll;
                applyColumnVisibility(cb.getAttribute('data-column'), selectAll);
            }});
            updateFooter();
        }});
        backdrop.querySelectorAll('.column-selector-check').forEach(function(cb) {{
            cb.addEventListener('change', function() {{
                applyColumnVisibility(this.getAttribute('data-column'), this.checked);
                updateFooter();
            }});
        }});
    }})();"""

# FIX (Bug #5): tag filter now uses the same thead-scoped selector as
# _JS_COL_SELECTOR_TMPL so both blocks agree on the visible column count.
_JS_TAG_FILTER_TMPL = """\
    // ── Column tag filter ─────────────────────────────────────────────────
    (function() {{
        var table    = document.getElementById('{table_id}');
        var pane     = document.getElementById('{filter_id}');
        var footerEl = document.getElementById('{footer_id}');
        if (!table || !pane) return;
        var active = new Set();
        var total  = {total_columns};
        var KEYS   = {filter_keys_json};
        function matches(el) {{
            return KEYS.some(function(k) {{
                return active.has(k) && el.classList.contains('tag-' + k);
            }});
        }}
        // FIX (Bug #5): use 'thead th' (not bare 'th') to match column selector's
        // updateFooter selector, ensuring both blocks report the same visible count.
        function refresh() {{
            table.querySelectorAll('thead th').forEach(function(el) {{
                el.classList.toggle('filter-hidden', active.size > 0 && !matches(el));
            }});
            table.querySelectorAll('tbody td:not(.filter-show-all):not(.column-selector-empty-cell)').forEach(function(el) {{
                el.classList.toggle('filter-hidden', active.size > 0 && !matches(el));
            }});
            var vis = table.querySelectorAll('thead th:not(.filter-hidden):not(.column-selector-hidden)').length;
            var emptyRow = table.querySelector('tr.column-selector-empty-row');
            if (emptyRow) emptyRow.style.display = vis === 0 ? '' : 'none';
            if (footerEl) {{
                footerEl.innerHTML = vis === total
                    ? '<strong>' + total + '</strong> columns'
                    : '<strong>' + vis + '</strong> of <strong>' + total + '</strong> columns';
                var notice = footerEl.nextElementSibling;
                if (notice && notice.classList.contains('columns-hidden-notice')) {{
                    notice.style.display = vis < total ? '' : 'none';
                }}
            }}
        }}
        pane.querySelectorAll('.column-filter-btn').forEach(function(btn) {{
            btn.addEventListener('click', function() {{
                var k = this.getAttribute('data-filter');
                active.has(k) ? active.delete(k) : active.add(k);
                this.classList.toggle('active', active.has(k));
                refresh();
            }});
        }});
    }})();"""


def _render_scripts(
    ids: _ViewerIds,
    total_columns: int,
    has_any_tags: bool,
) -> str:
    """
    Assemble the per-viewer <script> block by filling named placeholders into
    each static JS template. Each template section is kept as a plain string
    constant above so it can be read and edited without worrying about {{ / }}
    escaping in unrelated code.
    """
    uid = ids.uid
    filter_keys_json = json.dumps([spec.filter_key for spec in ALL_TAGS])

    blocks = [
        "<script>",
        _JS_HIGHLIGHT_FN_TMPL.format(uid=uid),
        _JS_TOGGLE_FN_TMPL.format(uid=uid),
        _JS_RESIZE_TMPL.format(table_id=ids.table),
        _JS_SORT_TMPL.format(table_id=ids.table),
        _JS_PREFS_TMPL.format(
            viewer_id=ids.viewer,
            toggle_id=ids.toggle,
            view_id=ids.view,
        ),
        _JS_COL_SELECTOR_TMPL.format(
            col_sel_id=ids.col_sel,
            col_sel_modal_id=ids.col_sel_modal,
            table_id=ids.table,
            footer_id=ids.footer,
            total_columns=total_columns,
        ),
    ]

    if has_any_tags:
        blocks.append(_JS_TAG_FILTER_TMPL.format(
            table_id=ids.table,
            filter_id=ids.filter,
            footer_id=ids.footer,
            total_columns=total_columns,
            filter_keys_json=filter_keys_json,
        ))

    blocks.append("</script>")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Step 9 — HTML sub-renderers
# ---------------------------------------------------------------------------

def _render_column_selector_modal(ids: _ViewerIds, col_metas: List[_ColumnMeta]) -> str:
    """Backdrop + centered modal with header (title + X) and scrollable checkbox list."""
    rows = "".join(
        f'<label class="column-selector-row">'
        f'<input type="checkbox" class="column-selector-check" data-column="{ci}" checked>'
        f'<span>{_html.escape(cm.name, quote=True)}</span></label>'
        for ci, cm in enumerate(col_metas)
    )
    return (
        f'<div class="column-selector-backdrop" id="{ids.col_sel_modal}">'
        f'<div class="column-selector-modal" role="dialog" aria-modal="true" aria-labelledby="{ids.col_sel_modal}-title">'
        f'<div class="column-selector-modal-header">'
        f'<span class="column-selector-title-row">'
        f'<span class="column-selector-modal-title" id="{ids.col_sel_modal}-title">Columns</span>'
        f'<button type="button" class="column-selector-toggle-all-btn" title="Select all columns">'
        f'<span class="column-selector-toggle-state column-selector-toggle-select">{ICON_LAYERS}</span>'
        f'<span class="column-selector-toggle-state column-selector-toggle-deselect" style="display:none">{ICON_LAYERS_DESELECTED}</span>'
        f'</button>'
        f'</span>'
        f'<button type="button" class="column-selector-close" aria-label="Close">&times;</button>'
        f'</div>'
        f'<div class="column-selector-modal-body">{rows}</div>'
        f'</div></div>'
    )


def _render_header(
    ids: _ViewerIds,
    table_name: Optional[str],
    stage: Optional[str],
    tag_sets: Dict[_TagSpec, Set[str]],
    col_metas: List[_ColumnMeta],
) -> str:
    label     = (table_name or "DataFrame").replace("\x00", "")  # strip null bytes for safety
    stage_dot = ""
    if stage:
        # FIX (Design #3): s = stage.lower() is explicit and kept close to its use.
        s = stage.lower()
        stage_dot = f'<span class="stage-dot {s}" title="{s}"></span>'
    name_card = f'<span class="table-name-card">{stage_dot}{_html.escape(label, quote=True)}</span>'

    filter_pane = ""
    if any(tag_sets.values()):
        btns = [
            f'<button type="button" class="column-filter-btn" '
            f'data-filter="{spec.filter_key}" '
            f'title="Toggle {spec.filter_key} columns">{TAG_ICONS[spec]}</button>'
            for spec in ALL_TAGS
            if tag_sets.get(spec)
        ]
        filter_pane = (
            f'<span class="column-filter-pane" id="{ids.filter}">{"".join(btns)}</span>'
        )

    col_sel_btn = (
        f'<span class="column-selector-wrap">'
        f'<button type="button" class="column-filter-btn column-selector-btn" id="{ids.col_sel}" '
        f'title="Show/hide columns">{ICON_COLUMNS}</button>'
        f'</span>'
    )

    theme_toggle = (
        f'<span class="theme-toggle" id="{ids.toggle}" title="Toggle dark/light mode">'
        f'<span class="theme-toggle-icon theme-toggle-moon">{ICON_MOON}</span>'
        f'<span class="theme-toggle-icon theme-toggle-sun" style="display:none">{ICON_SUN}</span>'
        f'</span>'
    )
    expert_btn = (
        f'<span class="expert-view-pane">'
        f'<button type="button" class="column-filter-btn expert-view-btn" id="{ids.view}" '
        f'title="Expert view (chips for ISO and boolean)">{ICON_EXPERT}</button>'
        f'</span>'
    )
    divider = '<span class="header-divider"></span>' if filter_pane else ''
    return (
        f'<div class="dataframe-header">'
        f'{name_card}'
        f'<div style="display:flex;align-items:center;position:relative;">{filter_pane}{divider}{col_sel_btn}{theme_toggle}{expert_btn}</div>'
        f'</div>'
    )


def _render_table(
    ids: _ViewerIds,
    col_metas: List[_ColumnMeta],
    data: _TableData,
    total_rows: Optional[int],
) -> str:
    # FIX (Perf #1): use io.StringIO to avoid O(n) list-of-strings overhead;
    # write() is a single call per piece rather than building many intermediate
    # f-strings that all get joined at the end.
    buf = io.StringIO()
    w = buf.write

    w('<div class="dataframe-table-container">')
    w(f'<table class="dataframe-table" id="{ids.table}">')
    w("<thead><tr>")

    for ci, cm in enumerate(col_metas):
        header_content = (
            f'<span class="column-header-content">{cm.icons_html}{_html.escape(cm.name, quote=True)}</span>'
            if cm.icons_html
            else _html.escape(cm.name, quote=True)
        )
        th_cls = f"sortable {cm.tag_class}".strip()
        w(
            f'<th class="{th_cls}" data-column="{ci}" '
            f'style="min-width:{cm.width_px}px;width:{cm.width_px}px;" '
            f'title="Type: {_html.escape(cm.type_str, quote=True)}">'
            f'{header_content}<span class="sort-indicator">'
            f'<span class="sort-icon-asc">{ICON_SORT_UP}</span>'
            f'<span class="sort-icon-desc">{ICON_SORT_DOWN}</span>'
            f'</span>'
            f'</th>'
        )
    w("</tr></thead><tbody>")

    # FIX (Perf #1): pre-compute per-column td_open templates outside the row
    # loop so the tag_class check and string formatting only runs once per column.
    td_opens = [
        f'<td class="{cm.tag_class}" data-column="{ci}"'
        if cm.tag_class
        else f'<td data-column="{ci}"'
        for ci, cm in enumerate(col_metas)
    ]

    # FIX (Design #1): use len(data.display_rows) instead of removed num_display.
    for ri, row in enumerate(data.display_rows):
        w("<tr>")
        for ci, cm in enumerate(col_metas):
            value    = row[cm.name]
            sort_val = _sort_attr_value(value)
            td_open  = f'{td_opens[ci]} data-sort-value="{sort_val}">'

            # Country column with list/array: expert = chips, simple = standard array display
            if TAG_ISO in cm.tag_specs and isinstance(value, (list, tuple)):
                cell_id = f"cell_{ids.uid}_{ri}_{ci}"
                expert_inner = _format_country_list(value)
                # FIX (Bug #1): pass the _FORCE_ARRAY_DISPLAY sentinel instead of
                # the broken _array_type_for_display() instance. The sentinel is
                # checked by name in _format_complex_cell so it reliably hits the
                # ArrayType branch.
                simple_inner = _format_complex_cell(
                    value, _FORCE_ARRAY_DISPLAY, cell_id, ids.uid
                )
                inner = (
                    f'<div class="expert-only">{expert_inner}</div>'
                    f'<div class="simple-only">{simple_inner}</div>'
                )
            elif _is_complex_type(cm.data_type):
                cell_id = f"cell_{ids.uid}_{ri}_{ci}"
                inner   = _format_complex_cell(value, cm.data_type, cell_id, ids.uid)
            else:
                inner = _format_simple_cell(value, cm.tag_specs)
            w(f"{td_open}{inner}</td>")
        w("</tr>")

    # FIX (Design #1): use len(data.display_rows) instead of num_display.
    num_display = len(data.display_rows)
    if data.has_more:
        if total_rows is not None:
            notice = (
                f"showing {num_display:,} of {total_rows:,} rows "
                f"— sorted within displayed rows only"
            )
        else:
            notice = (
                f"showing first {num_display:,} rows "
                f"— sorted within displayed rows only"
            )
        w(
            f'<tr><td class="filter-show-all" colspan="{len(col_metas)}" '
            f'style="text-align:center;color:#888;font-style:italic;padding:10px;">'
            f'{notice}</td></tr>'
        )

    w(
        f'<tr class="column-selector-empty-row" style="display:none;">'
        f'<td class="column-selector-empty-cell" colspan="{len(col_metas)}">No columns selected. Use the column selector to show columns.</td>'
        f'</tr>'
    )
    w("</tbody></table></div>")
    return buf.getvalue()


def _render_footer(
    ids: _ViewerIds,
    total_columns: int,
    rows_text: str,
) -> str:
    return (
        f'<div class="dataframe-footer">'
        f'<span id="{ids.footer}"><strong>{total_columns}</strong> columns</span>'
        f'<span class="columns-hidden-notice" style="display:none">{ICON_WARNING} Columns are hidden</span>'
        f'<span class="dataframe-footer-rows">{rows_text}</span>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def displayDF(
    df,
    max_rows: int = 100,
    table_name: Optional[str] = None,
    stage: Optional[str] = None,
    show_total_rows: bool = False,
    translation_columns: Optional[List[str]] = None,
    country_iso_columns: Optional[List[str]] = None,
    regex_columns: Optional[List[str]] = None,
    metadata_columns: Optional[List[str]] = None,
    cleaned_columns: Optional[List[str]] = None,
    date_columns: Optional[List[str]] = None,
) -> None:
    """Display a PySpark DataFrame in a custom HTML viewer.

    Args:
        df:                  PySpark DataFrame to display.
        max_rows:            Maximum preview rows (default 100, hard cap 10 000).
        table_name:          Header label (default "DataFrame").
        stage:               Medallion stage — "bronze", "silver", or "gold".
        show_total_rows:     Call df.count() and display the full row count.
                             Off by default because count() triggers a full Spark
                             job and can be very expensive on large tables.
                             The number of *displayed* rows is always shown
                             regardless of this flag.
        translation_columns: Columns to mark with a language icon.
        country_iso_columns: Columns to mark with a globe icon.
        regex_columns:       Columns to mark with a regex icon.
        metadata_columns:    Columns to mark with a metadata icon.
        cleaned_columns:     Columns to mark with a clean/sparkle icon.
        date_columns:        Columns to mark with a calendar icon.

    Example:
        displayDF(
            df,
            table_name="orders",
            stage="silver",
            show_total_rows=True,
            date_columns=["created_at", "updated_at"],
        )
    """
    # FIX (Perf #4): max_rows is validated (and capped) inside _validate_inputs.
    _validate_inputs(df, stage, max_rows)

    global _STYLE_INJECTED  # FIX (Perf #3)

    ids  = _build_viewer_ids()
    data = _collect_rows(df, max_rows)

    total_rows: Optional[int] = None
    if show_total_rows:
        try:
            total_rows = df.count()
        except Exception:
            total_rows = None

    tag_sets: Dict[_TagSpec, Set[str]] = {
        TAG_LANGUAGE: set(translation_columns or []),
        TAG_ISO:      set(country_iso_columns or []),
        TAG_REGEX:    set(regex_columns       or []),
        TAG_CLEANED:  set(cleaned_columns     or []),
        TAG_DATE:     set(date_columns        or []),
        TAG_METADATA: set(metadata_columns    or []),
    }
    has_any_tags  = any(bool(s) for s in tag_sets.values())
    col_metas     = _get_column_metadata(df, data, tag_sets)
    total_columns = len(col_metas)
    if total_columns == 0:
        raise ValueError("DataFrame has no columns; the viewer requires at least one column.")
    rows_text     = _build_rows_text(data, total_rows)

    html_parts = []

    # FIX (Perf #3): emit shared (global) CSS only once per kernel session.
    if not _STYLE_INJECTED:
        html_parts.append(_SHARED_CSS)
        _STYLE_INJECTED = True

    html_parts += [
        _render_css(ids),
        _render_scripts(ids, total_columns, has_any_tags),
        f'<div class="dataframe-viewer viewer-simple" id="{ids.viewer}">',
        _render_header(ids, table_name, stage, tag_sets, col_metas),
        _render_table(ids, col_metas, data, total_rows),
        _render_footer(ids, total_columns, rows_text),
        _render_column_selector_modal(ids, col_metas),
        "</div>",
    ]

    if IPYTHON_AVAILABLE:
        _ipython_display(HTML("\n".join(html_parts)))
    else:
        print("DataFrame viewer requires Jupyter for HTML rendering.")
        print(f"Columns : {', '.join(cm.name for cm in col_metas)}")
        # FIX (Design #1): use len() instead of removed num_display field.
        n = len(data.display_rows)
        print(f"Rows    : {n}" + (f" of {total_rows:,}" if total_rows is not None else ""))