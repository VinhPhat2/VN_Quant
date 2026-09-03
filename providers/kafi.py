"""
Standalone Kafi (FiinTrade) market data provider.

Fetches OHLCV and Investor Flow classification (Ca nhan, To chuc, Tu doanh, Nuoc ngoai)
optimized using PageSize=300 and persistent Session for maximum throughput.
"""

import time
import random
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

_KAFI_URL = "https://wlgw-technical.fiintrade.vn/PriceData/GetPriceData"

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "origin": "https://app-kafi.fiintrade.vn",
    "referer": "https://app-kafi.fiintrade.vn/",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": _USER_AGENTS[0],
}

TARGET_COLUMNS = [
    "NGÀY", "GIÁ", "GIÁ THÔ", "THAY ĐỔI", "%THAY ĐỔI", "MỞ", "CAO", "THẤP",
    "KL KHỚP", "GT KHỚP", "KL THỎA THUẬN", "GT THỎA THUẬN",
    "TỔNG KHỐI LƯỢNG", "TỔNG GIÁ TRỊ",
    "KL Cá Nhân Khớp Ròng", "KL Tổ chức Khớp Ròng", "KL Tự doanh Khớp Ròng", "KL Nước Ngoài Khớp Ròng",
    "GT Cá Nhân Khớp Ròng", "GT Tổ chức Khớp Ròng", "GT Tự doanh Khớp Ròng", "GT Nước Ngoài Khớp Ròng"
]


def _get_val(row: pd.Series, *keys: str, default=0) -> Any:
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default


class KafiFinance:
    """
    Fetch market price and investor flow data from Kafi/FiinTrade API.
    """

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def fetch_raw(
        self,
        symbol: str,
        total_bars: int = 1800,
        frequency: str = "Daily",
        delay_between_pages: float = 0.2,
    ) -> pd.DataFrame:
        """
        Fetch raw JSON records from Kafi API using max allowed PageSize=300.
        """
        page_size = 300
        max_pages = (total_bars + page_size - 1) // page_size
        all_records: List[Dict[str, Any]] = []

        for page in range(1, max_pages + 1):
            params = {
                "language": "vi",
                "Code": symbol.upper(),
                "Frequently": frequency,
                "Page": page,
                "PageSize": page_size,
            }

            try:
                self._session.headers["user-agent"] = random.choice(_USER_AGENTS)
                resp = self._session.get(_KAFI_URL, params=params, timeout=15)
                
                if resp.status_code == 429:
                    time.sleep(2.0)
                    resp = self._session.get(_KAFI_URL, params=params, timeout=15)

                if resp.status_code != 200:
                    break

                data = resp.json()
                items = data.get("items", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                if not items:
                    break

                all_records.extend(items)

                if len(items) < page_size:
                    break

                if page < max_pages and delay_between_pages > 0:
                    time.sleep(delay_between_pages)

            except Exception as e:
                print(f"[Kafi] Warning: error fetching {symbol} page {page}: {e}", flush=True)
                break

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df.insert(0, "ticker_symbol", symbol.upper())
        return df

    def transform_to_standard(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw Kafi API DataFrame to standard 22-column schema.
        """
        if raw_df is None or raw_df.empty:
            return pd.DataFrame(columns=TARGET_COLUMNS)

        date_cols = [c for c in ["tradingDate", "TradingDate", "date"] if c in raw_df.columns]
        if not date_cols:
            return pd.DataFrame(columns=TARGET_COLUMNS)

        date_col = date_cols[0]
        df = raw_df.copy()
        df["_dt"] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["_dt"]).sort_values(by="_dt", ascending=False)
        df = df.drop_duplicates(subset=["_dt"])

        rows = []
        for _, row in df.iterrows():
            kl_ca_nhan = _get_val(row, "retailNetVolume", "localIndividualNetVolume") or (
                _get_val(row, "localIndividualBuyMatchVolume") - _get_val(row, "localIndividualSellMatchVolume")
            )
            gt_ca_nhan = _get_val(row, "retailNetValue", "localIndividualNetValue") or (
                _get_val(row, "localIndividualBuyMatchValue") - _get_val(row, "localIndividualSellMatchValue")
            )

            kl_to_chuc = _get_val(row, "institutionNetVolume", "localInstitutionalNetVolume") or (
                _get_val(row, "localInstitutionalBuyMatchVolume") - _get_val(row, "localInstitutionalSellMatchVolume")
            )
            gt_to_chuc = _get_val(row, "institutionNetValue", "localInstitutionalNetValue") or (
                _get_val(row, "localInstitutionalBuyMatchValue") - _get_val(row, "localInstitutionalSellMatchValue")
            )

            kl_tu_doanh = _get_val(row, "proprietaryNetVolume", "netProprietaryMatchVolume") or (
                _get_val(row, "proprietaryTotalMatchBuyTradeVolume") - _get_val(row, "proprietaryTotalMatchSellTradeVolume")
            )
            gt_tu_doanh = _get_val(row, "proprietaryNetValue", "netProprietaryMatchValue") or (
                _get_val(row, "proprietaryTotalMatchBuyTradeValue") - _get_val(row, "proprietaryTotalMatchSellTradeValue")
            )

            kl_nuoc_ngoai = _get_val(row, "foreignNetVolume", "foreignNetVolumeMatched") or (
                _get_val(row, "foreignBuyVolumeMatched") - _get_val(row, "foreignSellVolumeMatched")
            )
            gt_nuoc_ngoai = _get_val(row, "foreignNetValue", "foreignNetValueMatched") or (
                _get_val(row, "foreignBuyValueMatched") - _get_val(row, "foreignSellValueMatched")
            )

            gia_dieu_chinh = _get_val(row, "closeValue", "closePrice", "ClosePrice")
            gia_tho = _get_val(row, "referencePrice", "close_raw", "rawClose", "unadjustedClose", default=gia_dieu_chinh)

            new_row = {
                "NGÀY": row["_dt"].strftime("%d/%m/%Y"),
                "GIÁ": gia_dieu_chinh,
                "GIÁ THÔ": gia_tho if gia_tho != 0 else gia_dieu_chinh,
                "THAY ĐỔI": _get_val(row, "valueChange", "priceChange"),
                "%THAY ĐỔI": _get_val(row, "percentValueChange", "pctChange"),
                "MỞ": _get_val(row, "openValue", "openPrice"),
                "CAO": _get_val(row, "highestValue", "highPrice"),
                "THẤP": _get_val(row, "lowestValue", "lowPrice"),
                "KL KHỚP": _get_val(row, "totalMatchVolume"),
                "GT KHỚP": _get_val(row, "totalMatchValue"),
                "KL THỎA THUẬN": _get_val(row, "totalDealVolume"),
                "GT THỎA THUẬN": _get_val(row, "totalDealValue"),
                "TỔNG KHỐI LƯỢNG": _get_val(row, "totalVolume"),
                "TỔNG GIÁ TRỊ": _get_val(row, "totalValue"),
                "KL Cá Nhân Khớp Ròng": kl_ca_nhan,
                "KL Tổ chức Khớp Ròng": kl_to_chuc,
                "KL Tự doanh Khớp Ròng": kl_tu_doanh,
                "KL Nước Ngoài Khớp Ròng": kl_nuoc_ngoai,
                "GT Cá Nhân Khớp Ròng": gt_ca_nhan,
                "GT Tổ chức Khớp Ròng": gt_to_chuc,
                "GT Tự doanh Khớp Ròng": gt_tu_doanh,
                "GT Nước Ngoài Khớp Ròng": gt_nuoc_ngoai,
            }
            rows.append(new_row)

        res_df = pd.DataFrame(rows, columns=TARGET_COLUMNS)
        return res_df

    def get_history(self, symbol: str, total_bars: int = 1800, frequency: str = "Daily") -> pd.DataFrame:
        """
        Fetch and format standard DataFrame in one call.
        """
        raw = self.fetch_raw(symbol, total_bars=total_bars, frequency=frequency)
        return self.transform_to_standard(raw)
