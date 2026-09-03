"""
Standalone VCI (Vietcap) financial data provider.
Bypasses vnstock's 4-period limit by NOT slicing the response.

API flow:
  1. Handshake: GET https://trading.vietcap.com.vn/priceboard  → cookies
  2. Metadata:  GET https://iq.vietcap.com.vn/.../financial-statement/metrics
  3. Data:      GET https://iq.vietcap.com.vn/.../financial-statement?section=...
                GET https://iq.vietcap.com.vn/.../statistics-financial  (ratios)

The VCI API returns ALL periods in one response.  vnstock slices to 4 via
``combined_df.head(4)``.  We skip that step entirely.
"""

import re
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VCIQ_BASE = "https://iq.vietcap.com.vn/api/iq-insight-service"
_PRICEBOARD_URL = "https://trading.vietcap.com.vn/priceboard"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,vi-VN;q=0.8,vi;q=0.7",
    "Referer": "https://trading.vietcap.com.vn/",
    "Origin": "https://trading.vietcap.com.vn",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "DNT": "1",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "sec-ch-ua-platform": '"Windows"',
    "sec-ch-ua-mobile": "?0",
}

_SECTION_MAP = {
    "income_statement": "INCOME_STATEMENT",
    "balance_sheet": "BALANCE_SHEET",
    "cash_flow": "CASH_FLOW",
}

_PERIOD_KEY = {"year": "years", "quarter": "quarters"}

_COMMON_META_COLS = {
    "organCode",
    "ticker",
    "year",
    "yearReport",
    "quarter",
    "lengthReport",
    "report_period",
    "createDate",
    "updateDate",
    "publicDate",
    "_period",
}


def _to_snake_case(name: str) -> str:
    """Convert an English name to a snake_case identifier."""
    if not name:
        return ""
    s = re.sub(r"[^\w\s]", "", name)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s.lower()


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class VCIFinance:
    """
    Fetch financial statements from Vietcap (VCI) API.

    The VCI API returns **all** available periods in a single response.
    vnstock artificially caps this at 4 via ``df.head(4)``.
    This provider keeps everything.

    Usage::

        vci = VCIFinance("PVS", period="quarter")
        df = vci.income_statement()
        print(f"Got {len(df.columns) - 3} periods!")
    """

    def __init__(self, symbol: str, period: str = "quarter"):
        """
        Parameters
        ----------
        symbol : str
            Stock ticker, e.g. ``"PVS"``.
        period : str
            ``"quarter"`` or ``"year"``.
        """
        self.symbol = symbol.upper()
        if period not in _PERIOD_KEY:
            raise ValueError(f"period must be 'year' or 'quarter', got '{period}'")
        self.period = period
        self._target_key = _PERIOD_KEY[period]

        # Persistent session — keeps cookies across requests
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

        # Field mapping cache (lazy-loaded)
        self._field_cache: Optional[Dict[str, Dict[str, str]]] = None

        # Handshake to obtain cookies
        self._handshake()

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    def _handshake(self) -> None:
        """GET the priceboard page to obtain session cookies."""
        try:
            self._session.get(_PRICEBOARD_URL, timeout=15)
        except requests.RequestException as exc:
            print(f"[VCI] Handshake warning: {exc}")

    # ------------------------------------------------------------------
    # Field metadata
    # ------------------------------------------------------------------

    def _load_field_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Fetch the metrics endpoint and build per-field vi/en name dicts.

        Returns a dict keyed by report section name.  Each value is itself
        a dict  ``{field_code: {"vi": ..., "en": ...}}``.
        """
        url = f"{_VCIQ_BASE}/v1/company/{self.symbol}/financial-statement/metrics"
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", payload)

        mapping: Dict[str, Dict[str, str]] = {}
        for section, fields in data.items():
            if not isinstance(fields, list):
                continue
            section_map: Dict[str, Dict[str, str]] = {}
            for f in fields:
                code = f.get("field", "")
                section_map[code] = {
                    "vi": f.get("titleVi", code),
                    "en": f.get("titleEn", code),
                }
            mapping[section] = section_map
        return mapping

    def _get_field_mapping(self) -> Dict[str, Dict[str, str]]:
        """Return cached field mapping, fetching once if needed."""
        if self._field_cache is None:
            self._field_cache = self._load_field_mapping()
        return self._field_cache

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_statement(self, section: str) -> pd.DataFrame:
        """
        Fetch a financial statement section and return ALL periods.

        This is where the bypass happens: we simply do **not** slice.
        """
        url = f"{_VCIQ_BASE}/v1/company/{self.symbol}/financial-statement"
        params = {"section": section}

        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", payload)

        period_rows = data.get(self._target_key, [])
        if not period_rows:
            return pd.DataFrame()

        df = pd.DataFrame(period_rows)
        return df

    def _fetch_ratios(self) -> pd.DataFrame:
        """Fetch financial ratios and filter by selected period."""
        url = f"{_VCIQ_BASE}/v1/company/{self.symbol}/statistics-financial"
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", payload)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Ratio endpoint có thể trả lẫn dữ liệu năm và quý.
        # Nếu period = quarter thì chỉ giữ dòng có quarter hợp lệ 1-4.
        if self.period == "quarter":
            if "quarter" in df.columns:
                df = df[df["quarter"].notna()].copy()
                df = df[df["quarter"].astype(str).isin(["1", "2", "3", "4"])].copy()
        # Nếu period = year thì chỉ giữ dòng không có quarter.
        elif self.period == "year":
            if "quarter" in df.columns:
                df = df[df["quarter"].isna()].copy()

        return df

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _build_period_label(row: pd.Series, period: str) -> str:
        """Create a human-readable period label from a data row."""
        year = row.get("year") or row.get("yearReport")
        quarter = row.get("quarter") or row.get("lengthReport")

        try:
            year = int(year)
        except (TypeError, ValueError):
            return "N/A"

        if period == "quarter":
            try:
                q = int(quarter)
                if 1 <= q <= 4:
                    return f"{year}-Q{q}"
            except (TypeError, ValueError):
                pass
        return str(year)

    def _format_statement(
        self,
        raw_df: pd.DataFrame,
        report_type: str,
        lang: str = "vi",
    ) -> pd.DataFrame:
        """
        Transform raw API rows into the standard output format:
        items as rows, periods as columns.
        """
        if raw_df.empty:
            return raw_df

        # Build period labels
        raw_df = raw_df.copy()
        raw_df["_period"] = raw_df.apply(
            lambda r: self._build_period_label(r, self.period), axis=1
        )

        # Get field mapping for labels
        mapping = self._get_field_mapping()

        # Find the right section mapping
        # The mapping keys match section names from the metrics endpoint
        section_mapping: Dict[str, Dict[str, str]] = {}
        for _sec_name, sec_map in mapping.items():
            section_mapping.update(sec_map)

        # Identify metric columns (everything except metadata columns)
        meta_cols = _COMMON_META_COLS
        metric_cols = [c for c in raw_df.columns if c not in meta_cols]

        # Build output rows
        period_labels = raw_df["_period"].tolist()
        rows = []
        for col in metric_cols:
            field_info = section_mapping.get(col, {})

            if not field_info:
                continue

            name_vi = field_info.get("vi", col)
            name_en = field_info.get("en", col)

            if name_vi == col and name_en == col:
                continue

            item_id = f"{_to_snake_case(name_en)}_{col}" if name_en != col else col

            row: Dict[str, Any] = {
                "item": name_vi if lang == "vi" else name_en,
                "item_en": name_en,
                "item_id": item_id,
            }
            for idx, plabel in enumerate(period_labels):
                row[plabel] = raw_df.iloc[idx].get(col)
            rows.append(row)

        result = pd.DataFrame(rows)

        # Reorder columns
        meta = [c for c in ["item", "item_en", "item_id"] if c in result.columns]
        periods = [c for c in result.columns if c not in meta]
        result = result[meta + periods]
        result.attrs["symbol"] = self.symbol
        result.attrs["periods"] = periods
        return result

    def _format_ratios(
        self,
        raw_df: pd.DataFrame,
        lang: str = "vi",
    ) -> pd.DataFrame:
        """Format ratios into the standard items × periods layout."""
        if raw_df.empty:
            return raw_df

        raw_df = raw_df.copy()
        raw_df["_period"] = raw_df.apply(
            lambda r: self._build_period_label(r, self.period), axis=1
        )

        meta_cols = _COMMON_META_COLS | {"ratioTTMId", "ratioType", "ratioYearId"}
        metric_cols = [c for c in raw_df.columns if c not in meta_cols]
        period_labels = raw_df["_period"].tolist()

        # For ratios we use a static name mapping (subset)
        _RATIO_NAMES: Dict[str, Dict[str, str]] = {
            "pe": {"vi": "P/E", "en": "P/E"},
            "pb": {"vi": "P/B", "en": "P/B"},
            "ps": {"vi": "P/S", "en": "P/S"},
            "roe": {"vi": "ROE (%)", "en": "ROE (%)"},
            "roa": {"vi": "ROA (%)", "en": "ROA (%)"},
            "grossMargin": {"vi": "Biên LN gộp (%)", "en": "Gross Margin (%)"},
            "ebitMargin": {"vi": "Biên EBIT (%)", "en": "EBIT Margin (%)"},
            "afterTaxProfitMargin": {"vi": "Biên LN sau thuế (%)", "en": "Net Margin (%)"},
            "currentRatio": {"vi": "Hệ số thanh toán hiện hành", "en": "Current Ratio"},
            "quickRatio": {"vi": "Hệ số thanh toán nhanh", "en": "Quick Ratio"},
            "cashRatio": {"vi": "Hệ số thanh toán tiền", "en": "Cash Ratio"},
            "debtToEquity": {"vi": "Nợ/Vốn chủ", "en": "Debt/Equity"},
            "dividendYield": {"vi": "Tỷ suất cổ tức (%)", "en": "Dividend Yield (%)"},
            "marketCap": {"vi": "Vốn hóa", "en": "Market Cap"},
            "ebit": {"vi": "EBIT", "en": "EBIT"},
            "ebitda": {"vi": "EBITDA", "en": "EBITDA"},
            "roic": {"vi": "ROIC", "en": "ROIC"},
            "npl": {"vi": "Nợ xấu (%)", "en": "NPL (%)"},
        }

        rows = []
        for col in metric_cols:
            info = _RATIO_NAMES.get(col, {})

            # Bỏ qua ratio không có tên rõ ràng trong mapping
            if not info:
                continue

            name_vi = info.get("vi", col)
            name_en = info.get("en", col)
            item_id = _to_snake_case(name_en) if name_en != col else _to_snake_case(col)

            row: Dict[str, Any] = {
                "item": name_vi if lang == "vi" else name_en,
                "item_en": name_en,
                "item_id": item_id,
            }
            for idx, plabel in enumerate(period_labels):
                row[plabel] = raw_df.iloc[idx].get(col)
            rows.append(row)

        result = pd.DataFrame(rows)
        meta = [c for c in ["item", "item_en", "item_id"] if c in result.columns]
        periods = [c for c in result.columns if c not in meta]
        result = result[meta + periods]
        result.attrs["symbol"] = self.symbol
        result.attrs["periods"] = periods
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def income_statement(self, lang: str = "vi") -> pd.DataFrame:
        """
        Fetch income statement — ALL periods, no limit.

        Parameters
        ----------
        lang : str
            ``"vi"`` for Vietnamese names, ``"en"`` for English.
        """
        raw = self._fetch_statement(_SECTION_MAP["income_statement"])
        return self._format_statement(raw, "income_statement", lang=lang)

    def balance_sheet(self, lang: str = "vi") -> pd.DataFrame:
        """Fetch balance sheet — ALL periods."""
        raw = self._fetch_statement(_SECTION_MAP["balance_sheet"])
        return self._format_statement(raw, "balance_sheet", lang=lang)

    def cash_flow(self, lang: str = "vi") -> pd.DataFrame:
        """Fetch cash flow — ALL periods."""
        raw = self._fetch_statement(_SECTION_MAP["cash_flow"])
        return self._format_statement(raw, "cash_flow", lang=lang)

    def ratio(self, lang: str = "vi") -> pd.DataFrame:
        """Fetch financial ratios — ALL periods."""
        raw = self._fetch_ratios()
        return self._format_ratios(raw, lang=lang)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("VCI Provider - NO period limit demo")
    print("=" * 70)

    vci = VCIFinance("PVS", period="quarter")
    df = vci.income_statement(lang="vi")

    if not df.empty:
        period_cols = [c for c in df.columns if c not in ("item", "item_en", "item_id")]
        print(f"\nSymbol: PVS")
        print(f"Periods retrieved: {len(period_cols)}")
        if period_cols:
            print(f"Period range: {period_cols[0]} -> {period_cols[-1]}")
        print(f"\nFirst 10 items (showing latest 4 periods):")
        show_cols = ["item"] + period_cols[-4:]
        print(df[show_cols].head(10).to_string(index=False))
    else:
        print("No data returned! (VCI may require different auth)")
