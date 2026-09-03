"""
Standalone KBS (KB Securities) financial data provider.

This provider mirrors the public KBS finance endpoint without vnstock. The
endpoint supports page/pageSize parameters, but live testing on 2026-05-21
shows it currently exposes only the latest four unique periods for PVS even
when pageSize is increased. Use VCI as the primary provider when you need a
true long financial statement history.

API: https://kbbuddywts.kbsec.com.vn/iis-server/investment/stock/finance-info/{symbol}
No authentication required. Supports pagination (page + pageSize).
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://kbbuddywts.kbsec.com.vn/iis-server/investment/stock/finance-info"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,vi-VN;q=0.8,vi;q=0.7",
    "Referer": "https://kbbuddywts.kbsec.com.vn/",
    "Origin": "https://kbbuddywts.kbsec.com.vn",
}

# Vietnamese keys in the API "Content" dict
_CONTENT_KEYS = {
    "KQKD": "K\u1ebft qu\u1ea3 kinh doanh",
    "CDKT": "C\u00e2n \u0111\u1ed1i k\u1ebf to\u00e1n",
    "CSTC": "Ch\u1ec9 s\u1ed1 t\u00e0i ch\u00ednh",
    "LCTT_indirect": "L\u01b0u chuy\u1ec3n ti\u1ec1n t\u1ec7 gi\u00e1n ti\u1ebfp",
    "LCTT_direct": "L\u01b0u chuy\u1ec3n ti\u1ec1n t\u1ec7 tr\u1ef1c ti\u1ebfp",
}

_PERIOD_TYPE = {"year": 1, "quarter": 2}


def _to_snake_case(name: str) -> str:
    """Convert an English field name to snake_case identifier."""
    if not name:
        return ""
    s = re.sub(r"[^\w\s]", "", name)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s.lower()


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class KBSFinance:
    """
    Fetch financial statements from KB Securities API.

    This provider pages through the API to collect as many periods as the
    public KBS endpoint exposes. In current live checks, KBS still returns
    only the latest four unique periods for some symbols, so VCI is the
    preferred source for long history.

    Usage::

        kbs = KBSFinance("PVS", period="quarter")
        df = kbs.income_statement(limit=20)   # 20 quarters = 5 years
        print(df)
    """

    def __init__(self, symbol: str, period: str = "quarter"):
        self.symbol = symbol.upper()
        if period not in _PERIOD_TYPE:
            raise ValueError(f"period must be 'year' or 'quarter', got '{period}'")
        self.period = period
        self._period_type = _PERIOD_TYPE[period]
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_page(
        self,
        report_type: str,
        page: int = 1,
        page_size: int = 4,
    ) -> Dict[str, Any]:
        """Fetch a single page from the KBS finance API."""
        url = f"{_BASE_URL}/{self.symbol}"
        params: Dict[str, Any] = {
            "page": page,
            "pageSize": page_size,
            "type": report_type,
            "unit": 1000,
            "termtype": self._period_type,
        }
        if report_type == "LCTT":
            params["code"] = self.symbol
            params["termType"] = self._period_type
        else:
            params["languageid"] = 1

        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_periods(head_list: List[Dict]) -> List[Tuple[str, int]]:
        """
        Build period labels from the Head array, sorted by ID.

        Returns list of (label, 1-based-value-index) tuples.
        Deduplicates by label -- the API sometimes returns the same
        quarter twice (audited + unaudited); we keep only the first.
        """
        sorted_heads = sorted(head_list, key=lambda h: h.get("ID", 0))
        seen: set = set()
        result: List[Tuple[str, int]] = []

        for idx, h in enumerate(sorted_heads, start=1):
            year = h.get("YearPeriod", "")
            term = str(h.get("TermName", ""))
            if "Qu" in term:
                q = term.replace("Qu\u00fd", "").strip()
                label = f"{year}-Q{q}"
            else:
                label = str(year)

            if label not in seen:
                seen.add(label)
                result.append((label, idx))
        return result

    def _parse_content(
        self,
        data: Dict[str, Any],
        content_key: str,
    ) -> pd.DataFrame:
        """Parse a single-page API response into a DataFrame."""
        head_list = data.get("Head", [])
        content = data.get("Content", {})
        records = content.get(content_key, [])

        if not records or not head_list:
            return pd.DataFrame()

        period_info = self._extract_periods(head_list)
        period_labels = [p[0] for p in period_info]

        rows = []
        seen_ids: Dict[str, int] = {}
        for rec in records:
            name_vi = rec.get("Name", "")
            name_en = rec.get("NameEn", "")
            item_id = _to_snake_case(name_en) if name_en else _to_snake_case(name_vi)

            if item_id in seen_ids:
                seen_ids[item_id] += 1
                item_id = f"{item_id}_{seen_ids[item_id]}"
            else:
                seen_ids[item_id] = 0

            row: Dict[str, Any] = {
                "item": name_vi,
                "item_en": name_en,
                "item_id": item_id,
            }

            for label, val_idx in period_info:
                val = rec.get(f"Value{val_idx}")
                if val is not None:
                    try:
                        val = float(val) * 1000
                    except (ValueError, TypeError):
                        pass
                row[label] = val

            rows.append(row)

        df = pd.DataFrame(rows)
        df.attrs["periods"] = period_labels
        return df

    def _detect_cash_flow_key(self) -> str:
        """Probe the API to find which cash-flow variant this company uses."""
        probe = self._fetch_page("LCTT", page=1, page_size=1)
        content = probe.get("Content", {})
        if _CONTENT_KEYS["LCTT_indirect"] in content:
            return _CONTENT_KEYS["LCTT_indirect"]
        if _CONTENT_KEYS["LCTT_direct"] in content:
            return _CONTENT_KEYS["LCTT_direct"]
        if content:
            return next(iter(content))
        return _CONTENT_KEYS["LCTT_indirect"]

    def _fetch_report(
        self,
        report_type: str,
        content_key: str,
        limit: int = 20,
        page_size: int = 4,
    ) -> pd.DataFrame:
        """
        Fetch multiple pages and merge into a single DataFrame.

        This attempts to go beyond one 4-period page by fetching
        ceil(limit / page_size) pages, when the endpoint exposes them.
        """
        all_dfs: List[pd.DataFrame] = []
        all_period_labels: List[str] = []
        page = 1
        max_pages = (limit // page_size) + 3

        while len(all_period_labels) < limit and page <= max_pages:
            data = self._fetch_page(report_type, page=page, page_size=page_size)
            df = self._parse_content(data, content_key)

            if df.empty:
                break

            new_periods = df.attrs.get("periods", [])
            truly_new = [p for p in new_periods if p not in all_period_labels]

            if not truly_new:
                break

            if len(truly_new) < len(new_periods):
                drop = [p for p in new_periods if p not in truly_new]
                df = df.drop(columns=drop, errors="ignore")
                df.attrs["periods"] = truly_new

            all_dfs.append(df)
            all_period_labels.extend(truly_new)
            page += 1

            if page <= max_pages and len(all_period_labels) < limit:
                time.sleep(0.5)

        if not all_dfs:
            return pd.DataFrame()

        merged = all_dfs[0]
        for extra_df in all_dfs[1:]:
            extra_periods = extra_df.attrs.get("periods", [])
            merge_cols = ["item_id"] + extra_periods
            merge_cols = [c for c in merge_cols if c in extra_df.columns]
            merged = merged.merge(
                extra_df[merge_cols], on="item_id", how="outer"
            )

        if len(all_period_labels) > limit:
            drop_periods = all_period_labels[limit:]
            merged = merged.drop(columns=drop_periods, errors="ignore")
            all_period_labels = all_period_labels[:limit]

        meta_cols = [c for c in ["item", "item_en", "item_id"] if c in merged.columns]
        period_cols = [c for c in all_period_labels if c in merged.columns]
        merged = merged[meta_cols + period_cols]
        merged.attrs["periods"] = period_cols
        merged.attrs["symbol"] = self.symbol
        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def income_statement(self, limit: int = 20) -> pd.DataFrame:
        """Fetch income statement. Default limit=20 periods."""
        return self._fetch_report("KQKD", _CONTENT_KEYS["KQKD"], limit=limit)

    def balance_sheet(self, limit: int = 20) -> pd.DataFrame:
        """Fetch balance sheet."""
        return self._fetch_report("CDKT", _CONTENT_KEYS["CDKT"], limit=limit)

    def cash_flow(self, limit: int = 20) -> pd.DataFrame:
        """Fetch cash flow (auto-detects direct vs indirect)."""
        key = self._detect_cash_flow_key()
        return self._fetch_report("LCTT", key, limit=limit)

    def ratio(self, limit: int = 20) -> pd.DataFrame:
        """Fetch financial ratios."""
        return self._fetch_report("CSTC", _CONTENT_KEYS["CSTC"], limit=limit)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("KBS Provider - Bypass 4-period limit demo")
    print("=" * 70)

    kbs = KBSFinance("PVS", period="quarter")
    df = kbs.income_statement(limit=20)

    if not df.empty:
        period_cols = [c for c in df.columns if c not in ("item", "item_en", "item_id")]
        print(f"\nSymbol: PVS")
        print(f"Periods retrieved: {len(period_cols)}")
        print(f"Period range: {period_cols[-1]} -> {period_cols[0]}")
        print(f"\nFirst 8 items:")
        print(df[["item_en"] + period_cols[:5]].head(8).to_string(index=False))
    else:
        print("No data returned!")
