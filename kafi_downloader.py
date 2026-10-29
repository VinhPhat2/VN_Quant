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

        cutoff_date = pd.to_datetime("2026-08-28")
        rows = []
        for _, row in df.iterrows():
            is_after_cutoff = row["_dt"] > cutoff_date

            if is_after_cutoff:
                # Sau 28/08/2026: Kafi không còn phân loại cá nhân & tổ chức trong nước -> để trống (None)
                kl_ca_nhan = None
                gt_ca_nhan = None
                kl_to_chuc = None
                gt_to_chuc = None
            else:
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

            # Tự doanh: lấy từ Kafi (khớp ròng hoặc mua - bán)
            kl_tu_doanh = _get_val(row, "proprietaryNetVolume", "netProprietaryMatchVolume") or (
                _get_val(row, "proprietaryTotalMatchBuyTradeVolume") - _get_val(row, "proprietaryTotalMatchSellTradeVolume")
            )
            gt_tu_doanh = _get_val(row, "proprietaryNetValue", "netProprietaryMatchValue") or (
                _get_val(row, "proprietaryTotalMatchBuyTradeValue") - _get_val(row, "proprietaryTotalMatchSellTradeValue")
            )

            # Nước ngoài: lấy từ Kafi (khớp ròng hoặc mua - bán)
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
        cutoff_date = pd.to_datetime("2026-08-28")

        if os.path.exists(csv_path) and not force_refetch:
            try:
                existing_df = pd.read_csv(csv_path, encoding="utf-8-sig")
                if not existing_df.empty and "NGÀY" in existing_df.columns:
                    existing_df["_dt_chk"] = pd.to_datetime(existing_df["NGÀY"], errors="coerce", dayfirst=True)
                    if not is_index:
                        # Bảo toàn dữ liệu 4 nhóm nhà đầu tư chuẩn <= 28/08/2026,
                        # Xóa bỏ các dòng tự tính toán / tự trừ trong tháng 9
                        existing_df = existing_df[existing_df["_dt_chk"] <= cutoff_date].copy()
                    existing_df.drop(columns=["_dt_chk"], inplace=True, errors="ignore")
            except Exception:
                pass

        bars_to_fetch = 300
        t0 = time.time()
        raw_df = self.fetch_raw(symbol, total_bars=bars_to_fetch)
        fetch_time = time.time() - t0

        if raw_df.empty:
            print(f"❌ [{symbol}] Không thu thập được dữ liệu.", flush=True)
            return

        std_df = self.transform(raw_df)

        if force_refetch or existing_df.empty:
            final_df = std_df
            print(f"🔥 [{symbol}] Tải mới toàn bộ {len(final_df)} phiên ({fetch_time:.2f}s).", flush=True)
        else:
            has_dividend_split = False
            
            # 1. Kiểm tra cờ quyền split / benefit trong raw Kafi
            if not is_index:
                for col in ["split", "benefit"]:
                    if col in raw_df.columns:
                        non_empty = raw_df[col].dropna()
                        if not non_empty.empty and any(str(x).strip() not in ("", "None", "nan", "0") for x in non_empty.head(25)):
                            has_dividend_split = True
                            print(f"⚡ [{symbol}] Kafi báo cờ sự kiện quyền ({col}): {non_empty.iloc[0]}", flush=True)
                            break

            # 2. Kiểm tra lệch giá hồi tố do chia cổ tức trên TOÀN BỘ các phiên trùng lặp (trước và sau 28/08)
            if not has_dividend_split and not is_index and not existing_df.empty:
                try:
                    chk_old = existing_df[["NGÀY", "GIÁ"]].dropna().copy()
                    chk_new = std_df[["NGÀY", "GIÁ"]].dropna().copy()
                    merged_chk = pd.merge(chk_old, chk_new, on="NGÀY", suffixes=("_old", "_new"))
                    
                    if not merged_chk.empty:
                        # Tính độ lệch tương đối giữa giá điều chỉnh cũ trong file và giá mới từ Kafi
                        merged_chk["diff"] = (merged_chk["GIÁ_old"] - merged_chk["GIÁ_new"]).abs() / merged_chk["GIÁ_old"].replace(0, float("nan"))
                        mismatches = merged_chk[merged_chk["diff"] > 0.005]  # Lệch > 0.5%
                        
                        if not mismatches.empty:
                            has_dividend_split = True
                            sample = mismatches.iloc[0]
                            print(f"⚡ [{symbol}] Phát hiện chia cổ tức / chia tách giá lịch sử (lệch tại phiên {sample['NGÀY']} và {len(mismatches)}/{len(merged_chk)} phiên trước/sau 28/08): Cũ={sample['GIÁ_old']:,.0f} -> Mới={sample['GIÁ_new']:,.0f} (lệch {sample['diff']*100:.1f}%)", flush=True)
                except Exception:
                    pass

            if has_dividend_split:
                print(f"🔄 [{symbol}] TỰ ĐỘNG ĐIỀU CHỈNH GIÁ: Tải lại 1.800 phiên lịch sử để cập nhật chuỗi giá chia cổ tức...", flush=True)
                raw_df_full = self.fetch_raw(symbol, total_bars=1800)
                if not raw_df_full.empty:
                    std_full = self.transform(raw_df_full)
                    # Giữ nguyên dòng tiền 4 nhóm nhà đầu tư cũ <= 28/08/2026, cập nhật giá mới
                    flow_cols = [
                        "KL Cá Nhân Khớp Ròng", "GT Cá Nhân Khớp Ròng", "KL Tổ chức Khớp Ròng", "GT Tổ chức Khớp Ròng",
                        "KL Tự doanh Khớp Ròng", "GT Tự doanh Khớp Ròng", "KL Nước Ngoài Khớp Ròng", "GT Nước Ngoài Khớp Ròng"
                    ]
                    old_map = existing_df.set_index("NGÀY")
                    for col in flow_cols:
                        if col in old_map.columns:
                            s_old = std_full["NGÀY"].map(old_map[col])
                            mask_has_old = s_old.notna()
                            std_full.loc[mask_has_old, col] = s_old[mask_has_old]
                    final_df = std_full
                    print(f"✅ [{symbol}] Đã cập nhật chuỗi giá chia cổ tức, bảo toàn dòng tiền lịch sử ({len(final_df)} phiên).", flush=True)
                else:
                    final_df = std_df
            else:
                if is_index:
                    existing_dates = set(pd.to_datetime(existing_df["NGÀY"], format="%d/%m/%Y", errors="coerce").dt.strftime("%d/%m/%Y").dropna())
                    new_rows = std_df[~std_df["NGÀY"].isin(existing_dates)].copy()
                else:
                    std_df["_dt_chk"] = pd.to_datetime(std_df["NGÀY"], format="%d/%m/%Y", errors="coerce")
                    new_rows = std_df[std_df["_dt_chk"] > cutoff_date].copy()
                    new_rows.drop(columns=["_dt_chk"], inplace=True)

                if new_rows.empty and not existing_df.empty:
                    print(f"ℹ️ [{symbol}] Đã ở trạng thái mới nhất ({fetch_time:.2f}s).", flush=True)
                    return

                print(f"🔥 [{symbol}] Tải {len(std_df)} phiên ({fetch_time:.2f}s) -> Thêm mới {len(new_rows)} phiên chuẩn tới ngày hiện tại.", flush=True)
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

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 1. Cập nhật chỉ số
    for idx_sym in INDEX_LIST:
        downloader.sync_to_csv(idx_sym, output_dir=OUTPUT_DIRECTORY, is_index=True)

    # 2. Cập nhật tất cả cổ phiếu HOSE đa luồng (8 workers) siêu tốc
    print(f"⚡ Đang chạy song song 8 luồng cập nhật cho {len(stock_list)} mã cổ phiếu...", flush=True)
    completed_cnt = 0
    def _sync_worker(s):
        try:
            downloader.sync_to_csv(s, output_dir=OUTPUT_DIRECTORY, is_index=False)
        except Exception as err:
            print(f"❌ [{s}] Lỗi sync: {err}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_sync_worker, sym): sym for sym in stock_list}
        for future in as_completed(futures):
            completed_cnt += 1
            if completed_cnt % 50 == 0 or completed_cnt == len(stock_list):
                print(f"✅ Đã xử lý {completed_cnt}/{len(stock_list)} mã cổ phiếu...", flush=True)

    print("=" * 70, flush=True)
    print(f"🎉 HOÀN THÀNH CẬP NHẬT TOÀN BỘ {len(stock_list)} MÃ SÀN HOSE VÀ {len(INDEX_LIST)} CHỈ SỐ!", flush=True)
    print("=" * 70, flush=True)
