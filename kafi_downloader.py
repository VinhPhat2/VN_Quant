"""
KAFI / FIINTRADE MARKET DATA DOWNLOADER - FULL HOSE & INCREMENTAL UPDATER
------------------------------------------------------------------------
Tự động cập nhật dữ liệu hàng ngày cho toàn bộ sàn HOSE (405 mã) và các chỉ số thị trường.
Chuẩn hóa 22 cột (OHLCV + Dòng tiền 4 nhóm nhà đầu tư) lưu vào thư mục csv_data/.
"""

import os
import glob
import json
import time
import random
import sys
from typing import Any, Dict, List
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_KAFI_URL = "https://wlgw-technical.fiintrade.vn/PriceData/GetPriceData"
_ORGAN_URL = "https://wlgw-core.fiintrade.vn/Master/GetListOrganization"

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
}

TARGET_COLUMNS = [
    "NGÀY", "GIÁ", "GIÁ THÔ", "THAY ĐỔI", "%THAY ĐỔI", "MỞ", "CAO", "THẤP",
    "KL KHỚP", "GT KHỚP", "KL THỎA THUẬN", "GT THỎA THUẬN",
    "TỔNG KHỐI LƯỢNG", "TỔNG GIÁ TRỊ",
    "KL Cá Nhân Khớp Ròng", "KL Tổ chức Khớp Ròng", "KL Tự doanh Khớp Ròng", "KL Nước Ngoài Khớp Ròng",
    "GT Cá Nhân Khớp Ròng", "GT Tổ chức Khớp Ròng", "GT Tự doanh Khớp Ròng", "GT Nước Ngoài Khớp Ròng"
]

INDEX_LIST = ["VNINDEX", "VN30", "VN100", "VNMID", "VNSML"]


def _get_val(row: pd.Series, *keys: str, default=0) -> Any:
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default


class KafiDownloader:
    def __init__(self, map_path: str = "fiintrade_master_map.json"):
        import requests
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._map_path = map_path
        self._organ_map: Dict[str, str] = {}
        self._load_master_map()

    def _load_master_map(self):
        if os.path.exists(self._map_path):
            try:
                with open(self._map_path, "r", encoding="utf-8") as f:
                    self._organ_map = json.load(f)
                    if self._organ_map:
                        return
            except Exception:
                pass

        try:
            resp = self._session.get(_ORGAN_URL, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", []) if isinstance(data, dict) else []
                for it in items:
                    t = it.get("ticker")
                    o = it.get("organCode")
                    if t and o:
                        self._organ_map[t.upper()] = o
                if self._organ_map:
                    with open(self._map_path, "w", encoding="utf-8") as f:
                        json.dump(self._organ_map, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Không thể tải master map: {e}")

    def get_organ_code(self, symbol: str) -> str:
        return self._organ_map.get(symbol.upper().strip(), symbol.upper().strip())

    def fetch_raw(self, symbol: str, total_bars: int = 1800, frequency: str = "Daily") -> pd.DataFrame:
        page_size = 300
        max_pages = (total_bars + page_size - 1) // page_size
        all_records: List[Dict[str, Any]] = []

        code = self.get_organ_code(symbol)

        for page in range(1, max_pages + 1):
            params = {
                "language": "vi",
                "Code": code,
                "Frequently": frequency,
                "Page": page,
                "PageSize": page_size,
            }

            try:
                self._session.headers["user-agent"] = random.choice(_USER_AGENTS)
                resp = self._session.get(_KAFI_URL, params=params, timeout=15)

                if resp.status_code == 429:
                    time.sleep(5)
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

                if page < max_pages:
                    time.sleep(0.1)

            except Exception as e:
                print(f"   ❌ Lỗi tải {symbol} (Trang {page}): {e}", flush=True)
                break

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df.insert(0, "ticker_symbol", symbol.upper())
        return df

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
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

        return pd.DataFrame(rows, columns=TARGET_COLUMNS)

    def sync_to_csv(self, symbol: str, output_dir: str = "csv_data", is_index: bool = False, total_bars: int = 1800, force_refetch: bool = False):
        symbol = symbol.upper().strip()
        prefix = f"Kafi_Giá_{symbol}" if is_index else f"Kafi_Phân_Loại_Nhà_Đầu_Tư_{symbol}"
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, f"{prefix}.csv")

        existing_df = pd.DataFrame(columns=TARGET_COLUMNS)
        existing_dates = set()

        if os.path.exists(csv_path) and not force_refetch:
            try:
                existing_df = pd.read_csv(csv_path, encoding="utf-8-sig")
                if not existing_df.empty and "NGÀY" in existing_df.columns:
                    d_conv = pd.to_datetime(existing_df["NGÀY"], errors="coerce", dayfirst=True)
                    existing_dates = set(d_conv.dt.strftime("%Y-%m-%d").dropna())
            except Exception:
                pass

        bars_to_fetch = 300 if existing_dates else total_bars

        t0 = time.time()
        raw_df = self.fetch_raw(symbol, total_bars=bars_to_fetch)
        fetch_time = time.time() - t0

        if raw_df.empty:
            print(f"❌ [{symbol}] Không thu thập được dữ liệu.", flush=True)
            return

        std_df = self.transform(raw_df)
        
        if force_refetch or existing_df.empty:
            final_df = std_df
            print(f"🔥 [{symbol}] Tải mới toàn bộ {len(final_df)} phiên ({fetch_time:.2f}s) có đầy đủ dòng tiền.", flush=True)
        else:
            # =========================================================================
            # KIỂM TRA BẤT THƯỜNG DO CHIA CỔ TỨC / PHÁT HÀNH THÊM (CORPORATE ACTIONS)
            # =========================================================================
            has_dividend_split = False
            
            # 1. Kiểm tra cờ quyền split / benefit trong raw Kafi
            if not is_index:
                for col in ["split", "benefit"]:
                    if col in raw_df.columns:
                        non_empty = raw_df[col].dropna()
                        if not non_empty.empty and any(str(x).strip() not in ("", "None", "nan", "0") for x in non_empty.head(10)):
                            has_dividend_split = True
                            print(f"⚡ [{symbol}] Kafi báo cờ sự kiện quyền ({col}): {non_empty.iloc[0]}", flush=True)
                            break

            # 2. Kiểm tra lệch giá tham chiếu phiên mới so với giá đóng cửa phiên trước (reprice)
            if not has_dividend_split and not is_index and len(raw_df) >= 2:
                if "referencePrice" in raw_df.columns and "closePrice" in raw_df.columns:
                    ref_p = float(raw_df["referencePrice"].iloc[0] or 0)
                    prev_close = float(raw_df["closePrice"].iloc[1] or 0)
                    if ref_p > 0 and prev_close > 0 and abs(ref_p - prev_close) > 100:
                        has_dividend_split = True
                        print(f"⚡ [{symbol}] Phát hiện re-price bất thường: Tham chiếu={ref_p:,.0f} vs Đóng cửa cũ={prev_close:,.0f}", flush=True)

            # 3. Kiểm tra hồi tố giá đóng cửa cũ trong file CSV vs giá điều chỉnh mới từ Kafi
            if not has_dividend_split and not is_index and not existing_df.empty:
                common_dates = set(existing_df["NGÀY"]).intersection(set(std_df["NGÀY"]))
                if common_dates:
                    try:
                        latest_common = sorted(list(common_dates), key=lambda x: pd.to_datetime(x, format="%d/%m/%Y", errors="coerce"))[-1]
                        old_close = float(existing_df.loc[existing_df["NGÀY"] == latest_common, "GIÁ"].iloc[0] or 0)
                        new_close = float(std_df.loc[std_df["NGÀY"] == latest_common, "GIÁ"].iloc[0] or 0)
                        if old_close > 0 and new_close > 0 and abs(old_close - new_close) / old_close > 0.005:
                            has_dividend_split = True
                            print(f"⚡ [{symbol}] Phát hiện Kafi điều chỉnh hồi tố giá do chia cổ tức tại {latest_common}: Cũ={old_close:,.0f} -> Mới={new_close:,.0f}", flush=True)
                    except Exception:
                        pass

            # Nếu phát hiện chia cổ tức / chia tách: Tải lại toàn bộ 1800 phiên để cập nhật đồng bộ toàn bộ chuỗi giá điều chỉnh
            if has_dividend_split:
                print(f"🔄 [{symbol}] TỰ ĐỘNG ĐIỀU CHỈNH GIÁ: Tải lại toàn bộ lịch sử 1.800 phiên để đồng bộ giá điều chỉnh...", flush=True)
                raw_df_full = self.fetch_raw(symbol, total_bars=1800)
                if not raw_df_full.empty:
                    final_df = self.transform(raw_df_full)
                    print(f"✅ [{symbol}] Đã cập nhật lại toàn bộ chuỗi giá điều chỉnh cổ tức ({len(final_df)} phiên).", flush=True)
                else:
                    final_df = std_df
            else:
                std_df["_check_date"] = pd.to_datetime(std_df["NGÀY"], format="%d/%m/%Y", errors="coerce").dt.strftime("%Y-%m-%d")
                new_rows = std_df[~std_df["_check_date"].isin(existing_dates)].copy()
                new_rows.drop(columns=["_check_date"], inplace=True)
                std_df.drop(columns=["_check_date"], inplace=True)

                if new_rows.empty:
                    print(f"ℹ️ [{symbol}] Đã ở trạng thái mới nhất ({fetch_time:.2f}s).", flush=True)
                    return

                print(f"🔥 [{symbol}] Tải {len(std_df)} phiên ({fetch_time:.2f}s) -> Thêm mới {len(new_rows)} phiên.", flush=True)
                final_df = pd.concat([new_rows, existing_df], ignore_index=True)

        final_df = final_df.reindex(columns=TARGET_COLUMNS)

        if "NGÀY" in final_df.columns:
            final_df["_dt_sort"] = pd.to_datetime(final_df["NGÀY"], format="%d/%m/%Y", errors="coerce")
            final_df = final_df.dropna(subset=["_dt_sort"]).drop_duplicates(subset=["_dt_sort"])
            final_df = final_df.sort_values(by="_dt_sort", ascending=False)
            final_df.drop(columns=["_dt_sort"], inplace=True)

        try:
            final_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"🎉 [{symbol}] Đã lưu CSV: {os.path.basename(csv_path)} (Tổng {len(final_df)} phiên)\n", flush=True)
        except Exception as e:
            print(f"❌ [{symbol}] Ghi file CSV thất bại: {e}\n", flush=True)


if __name__ == "__main__":
    downloader = KafiDownloader()
    OUTPUT_DIRECTORY = "csv_data"

    # Lấy toàn bộ các mã cổ phiếu hiện có trong thư mục csv_data để cập nhật
    stock_files = glob.glob(f"{OUTPUT_DIRECTORY}/Kafi_Phân_Loại_Nhà_Đầu_Tư_*.csv")
    stock_list = sorted([os.path.basename(f).replace("Kafi_Phân_Loại_Nhà_Đầu_Tư_", "").replace(".csv", "") for f in stock_files])

    print("=" * 70, flush=True)
    print(f"🚀 BẮT ĐẦU CẬP NHẬT DỮ LIỆU KAFI CHO TOÀN BỘ SÀN HOSE ({len(stock_list)} MÃ CP & {len(INDEX_LIST)} CHỈ SỐ)", flush=True)
    print(f"📁 Thư mục lưu: {os.path.abspath(OUTPUT_DIRECTORY)}", flush=True)
    print("=" * 70, flush=True)

    # 1. Cập nhật chỉ số
    for idx_sym in INDEX_LIST:
        downloader.sync_to_csv(idx_sym, output_dir=OUTPUT_DIRECTORY, is_index=True)
        time.sleep(0.15)

    # 2. Cập nhật tất cả cổ phiếu HOSE
    for i, sym in enumerate(stock_list, 1):
        print(f"[{i:03d}/{len(stock_list):03d}] Cập nhật: {sym}", flush=True)
        downloader.sync_to_csv(sym, output_dir=OUTPUT_DIRECTORY, is_index=False)
        time.sleep(0.15)

    print("=" * 70, flush=True)
    print(f"🎉 HOÀN THÀNH CẬP NHẬT TOÀN BỘ {len(stock_list)} MÃ SÀN HOSE VÀ {len(INDEX_LIST)} CHỈ SỐ!", flush=True)
    print("=" * 70, flush=True)
