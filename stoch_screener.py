import os
import glob
import json
import time
import pandas as pd
import numpy as np
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def compute_stoch_signals(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "csv_data")
    
    csv_files = glob.glob(os.path.join(data_dir, "Kafi_Phân_Loại_Nhà_Đầu_Tư_*.csv"))
    if not csv_files:
        return {"stocks": [], "market_breadth": {}, "updated_at": ""}
        
    stocks = []
    advancers = decliners = unchanged = 0
    above_ma20 = 0
    valid_count = 0
    latest_market_date = ""

    from server import get_symbol_sector

    for p in csv_files:
        sym = os.path.basename(p).replace("Kafi_Phân_Loại_Nhà_Đầu_Tư_", "").replace(".csv", "")
        try:
            df = pd.read_csv(p, nrows=45, usecols=['NGÀY', 'GIÁ', 'MỞ', 'CAO', 'THẤP', 'TỔNG KHỐI LƯỢNG', 'TỔNG GIÁ TRỊ'])
            if len(df) < 15:
                continue
            df = df.iloc[::-1].reset_index(drop=True)
            
            cl = df['GIÁ'].astype(float)
            hi = df['CAO'].astype(float)
            lo = df['THẤP'].astype(float)
            val = df['TỔNG GIÁ TRỊ'].astype(float)
            
            # Stochastic Oscillator (20, 3, 3)
            low20 = lo.rolling(20).min()
            high20 = hi.rolling(20).max()
            fast_k = 100 * (cl - low20) / (high20 - low20 + 1e-8)
            stoch_k = fast_k.rolling(3).mean()
            stoch_d = stoch_k.rolling(3).mean()

            # Stochastic RSI (9, 9, 3, 3)
            delta = cl.diff()
            gain9 = delta.clip(lower=0)
            loss9 = -delta.clip(upper=0)
            avg_gain9 = gain9.ewm(alpha=1/9, min_periods=9, adjust=False).mean()
            avg_loss9 = loss9.ewm(alpha=1/9, min_periods=9, adjust=False).mean()
            rs9 = avg_gain9 / (avg_loss9 + 1e-8)
            rsi9 = 100 - (100 / (1 + rs9))

            rsi_low9 = rsi9.rolling(9).min()
            rsi_high9 = rsi9.rolling(9).max()
            stoch_rsi_raw = 100 * (rsi9 - rsi_low9) / (rsi_high9 - rsi_low9 + 1e-8)
            srsi_k_s = stoch_rsi_raw.rolling(3).mean()
            srsi_d_s = srsi_k_s.rolling(3).mean()
            srsi_k_val = round(float(srsi_k_s.iloc[-1]), 1) if pd.notna(srsi_k_s.iloc[-1]) else 50.0
            srsi_d_val = round(float(srsi_d_s.iloc[-1]), 1) if pd.notna(srsi_d_s.iloc[-1]) else 50.0
            
            # Giá trị phiên mới nhất và phiên trước đó
            k_now = float(stoch_k.iloc[-1])
            k_prev = float(stoch_k.iloc[-2]) if len(stoch_k) > 1 else k_now
            d_now = float(stoch_d.iloc[-1])
            d_prev = float(stoch_d.iloc[-2]) if len(stoch_d) > 1 else d_now
            
            c_now = float(cl.iloc[-1])
            c_prev = float(cl.iloc[-2]) if len(cl) > 1 else c_now
            change_pct = ((c_now - c_prev) / c_prev * 100) if c_prev else 0
            turnover_ty = val.iloc[-1] / 1e9 if pd.notna(val.iloc[-1]) else 0
            
            # MA20 để đo lường trạng thái thị trường
            ma20 = cl.rolling(20, min_periods=5).mean().iloc[-1]
            valid_count += 1
            if c_now > ma20:
                above_ma20 += 1
            if change_pct > 0:
                advancers += 1
            elif change_pct < 0:
                decliners += 1
            else:
                unchanged += 1
                
            cur_date = str(df['NGÀY'].iloc[-1])
            if not latest_market_date:
                latest_market_date = cur_date

            # ATR 14 để tham chiếu biến động
            tr = np.maximum(hi - lo, np.maximum((hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()))
            atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(c_now * 0.03)

            # =======================================================
            # ĐIỀU KIỆN TÍN HIỆU THUẦN TÚY STOCHASTIC (PURE STOCHASTIC)
            # =======================================================
            is_golden_cross = (k_prev <= d_prev) and (k_now > d_now)
            is_death_cross = (k_prev >= d_prev) and (k_now < d_now)
            k_diff = k_now - d_now  # Độ phân kỳ giữa K và D

            if is_golden_cross and (k_now <= 30):
                # Golden cross trong vùng quá bán -> Điểm mua bắt đáy mạnh nhất
                signal = "MUA (Golden Cross Quá Bán)"
                signal_type = "BUY_GOLDEN"
                status_desc = "K cắt lên D trong vùng quá bán (<=30)"
                # Điểm số xếp hạng: K càng thấp trong giao cắt thì điểm càng cao
                score = round(100 - k_now, 2)
            elif k_now <= 20:
                # Nằm sâu trong vùng quá bán
                if is_golden_cross:
                    signal = "MUA (Đảo Chiều Đáy Stoch)"
                    signal_type = "BUY_GOLDEN"
                    status_desc = "K cắt lên D tại vùng đáy (<=20)"
                    score = round(98 - k_now, 2)
                else:
                    signal = "QUÁ BÁN (Vùng Đáy ≤ 20)"
                    signal_type = "OVERSOLD"
                    status_desc = f"Quá bán sâu (K={k_now:.1f}), chờ điểm nổ"
                    score = round(90 - k_now, 2)
            elif is_golden_cross and (30 < k_now < 70):
                signal = "MOMENTUM TĂNG (K cắt lên D)"
                signal_type = "BULLISH"
                status_desc = "Giao cắt tăng ở vùng trung lập (30-70)"
                score = round(60 + (k_now / 2), 2)
            elif k_now >= 80:
                if is_death_cross:
                    signal = "BÁN (Death Cross Quá Mua)"
                    signal_type = "SELL_DEATH"
                    status_desc = "K cắt xuống D tại đỉnh quá mua (>=80)"
                    score = 10.0
                else:
                    signal = "QUÁ MUA (Vùng Đỉnh ≥ 80)"
                    signal_type = "OVERBOUGHT"
                    status_desc = f"Quá mua cao (K={k_now:.1f}), cảnh báo chốt lời"
                    score = 20.0
            elif is_death_cross and (k_now >= 65):
                signal = "CẢNH BÁO BÁN (K cắt xuống D)"
                signal_type = "SELL"
                status_desc = "Giao cắt giảm ở vùng giá cao"
                score = 15.0
            else:
                signal = "TRUNG TÍNH (Dao Động)"
                signal_type = "NEUTRAL"
                status_desc = "Dao động trong biên độ 20 - 80"
                score = 40.0

            target_price = c_now + 1.6 * atr if atr > 0 else c_now * 1.08
            stop_loss = max(c_now - 1.2 * atr, c_now * 0.85) if atr > 0 else c_now * 0.94

            stocks.append({
                "symbol": sym,
                "sector": get_symbol_sector(sym),
                "date": cur_date,
                "price": c_now,
                "change_pct": round(change_pct, 2),
                "turnover_ty": round(turnover_ty, 2),
                "stoch_k": round(k_now, 1),
                "stoch_d": round(d_now, 1),
                "stoch_rsi_k": srsi_k_val,
                "stoch_rsi_d": srsi_d_val,
                "stoch_prev_k": round(k_prev, 1),
                "stoch_prev_d": round(d_prev, 1),
                "k_diff": round(k_diff, 1),
                "signal": signal,
                "signal_type": signal_type,
                "status_desc": status_desc,
                "score": score,
                "target": round(target_price, 0),
                "stop_loss": round(stop_loss, 0)
            })
        except Exception:
            continue

    # =======================================================
    # XẾP HẠNG RANKING CHỈ BÁO STOCHASTIC
    # Ưu tiên:
    # 1. Các mã Mua Golden Cross & Quá Bán sâu nhất (%K thấp nhất) lên đầu bảng
    # 2. Xếp theo Score và Stochastic %K tăng dần (K thấp nhất = Quá bán nhất)
    # =======================================================
    stocks = sorted(stocks, key=lambda x: (-x["score"], x["stoch_k"], -x["turnover_ty"]))
    for idx, s in enumerate(stocks):
        s["rank"] = idx + 1

    pct_ma20 = round((above_ma20 / valid_count * 100), 1) if valid_count > 0 else 50.0
    if pct_ma20 < 20:
        regime = "Hoảng Loạn Tột Độ (Cơ Hội Lớn)"
        regime_badge = "hoang_loan"
    elif pct_ma20 < 40:
        regime = "Vùng Hấp Thụ Đáy (Thận Trọng Tích Lũy)"
        regime_badge = "tich_luy"
    elif pct_ma20 < 60:
        regime = "Đi Ngang Giằng Co (Phân Hóa Cao)"
        regime_badge = "di_ngang"
    elif pct_ma20 < 80:
        regime = "Sóng Tăng Lan Tỏa (Thuận Lợi)"
        regime_badge = "song_tang"
    else:
        regime = "Hưng Phấn / Quá Mua (Cẩn Trọng Rủi Ro)"
        regime_badge = "hung_phan"

    buy_count = len([s for s in stocks if s["signal_type"] == "BUY_GOLDEN"])
    oversold_count = len([s for s in stocks if s["stoch_k"] <= 20])
    overbought_count = len([s for s in stocks if s["stoch_k"] >= 80])

    market_breadth = {
        "date": latest_market_date,
        "total_scanned": len(stocks),
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": unchanged,
        "above_ma20_pct": pct_ma20,
        "regime": regime,
        "regime_badge": regime_badge,
        "buy_count": buy_count,
        "oversold_count": oversold_count,
        "overbought_count": overbought_count
    }

    result = {
        "market_breadth": market_breadth,
        "stocks": stocks,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    cache_file = os.path.join(os.path.dirname(__file__), "stoch_signals_cache.json")
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Error saving cache:", e)

    return result

if __name__ == "__main__":
    res = compute_stoch_signals()
    print("Done! Total stocks scanned:", res["market_breadth"]["total_scanned"])
    print("Market date:", res["market_breadth"]["date"])
    print("Buy (Golden Cross) count:", res["market_breadth"]["buy_count"])
    print("Oversold count:", res["market_breadth"]["oversold_count"])
    print("Overbought count:", res["market_breadth"]["overbought_count"])
    print("\nTop 5 Ranked Stocks:")
    for s in res["stocks"][:5]:
        print(f"Rank #{s['rank']} | {s['symbol']} | Stoch %K: {s['stoch_k']} | %D: {s['stoch_d']} | Signal: {s['signal']}")
