import os
import glob
import json
import time
import pandas as pd
import numpy as np

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
            op = df['MỞ'].astype(float)
            hi = df['CAO'].astype(float)
            lo = df['THẤP'].astype(float)
            val = df['TỔNG GIÁ TRỊ'].astype(float)
            
            # Stochastic %K, %D (14, 3, 3)
            low14 = lo.rolling(14).min()
            high14 = hi.rolling(14).max()
            fast_k = 100 * (cl - low14) / (high14 - low14 + 1e-8)
            stoch_k = fast_k.rolling(3).mean()
            stoch_d = stoch_k.rolling(3).mean()
            
            # RSI 14
            delta = cl.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            # Williams %R (10)
            high10 = hi.rolling(10).max()
            low10 = lo.rolling(10).min()
            wpr = (cl - high10) / (high10 - low10 + 1e-8) * 100
            
            # Latest & prev values
            k_now = stoch_k.iloc[-1]
            k_prev = stoch_k.iloc[-2] if len(stoch_k) > 1 else k_now
            d_now = stoch_d.iloc[-1]
            d_prev = stoch_d.iloc[-2] if len(stoch_d) > 1 else d_now
            rsi_now = rsi.iloc[-1]
            wpr_now = wpr.iloc[-1]
            c_now = cl.iloc[-1]
            c_prev = cl.iloc[-2] if len(cl) > 1 else c_now
            change_pct = ((c_now - c_prev) / c_prev * 100) if c_prev else 0
            turnover_ty = val.iloc[-1] / 1e9 if pd.notna(val.iloc[-1]) else 0
            
            # MA20
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

            # Price Action & Candlesticks
            o_now = op.iloc[-1]
            h_now = hi.iloc[-1]
            l_now = lo.iloc[-1]
            body = abs(c_now - o_now)
            lower_wick = min(o_now, c_now) - l_now
            
            # ATR 14
            tr = np.maximum(hi - lo, np.maximum((hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()))
            atr = tr.rolling(14).mean().iloc[-1]
            
            # Patterns
            is_pinbar = (lower_wick > body * 1.5) and (lower_wick > atr * 0.4)
            is_bullish = c_now > o_now
            o_prev = op.iloc[-2] if len(op) > 1 else o_now
            is_engulfing = (c_now > o_now) and (c_prev < o_prev) and (c_now > o_prev)
            is_bearish = (c_now < o_now) and (c_prev > o_prev) and (c_now < o_prev) and (o_now > c_prev)
            
            # Signals Logic
            is_golden_cross = (k_prev <= d_prev) and (k_now > d_now)
            is_death_cross = (k_prev >= d_prev) and (k_now < d_now)
            
            if is_golden_cross and (k_now <= 35) and (is_pinbar or is_bullish or is_engulfing):
                signal = "BẮT ĐÁY MẠNH (Golden Cross)"
                signal_type = "BUY_STRONG"
                score = 95
            elif ((k_now <= 25) or (wpr_now <= -75)) and (is_pinbar or is_bullish or is_engulfing):
                signal = "BẮT ĐÁY (Quá Bán + Đảo Chiều)"
                signal_type = "BUY"
                score = 85
            elif k_now <= 20:
                signal = "THEO DÕI (Vùng Quá Bán)"
                signal_type = "WATCH_OVERSOLD"
                score = 72
            elif is_golden_cross and (35 < k_now < 70):
                signal = "MOMENTUM TĂNG (Bứt Phá)"
                signal_type = "BULLISH"
                score = 65
            elif k_now >= 80:
                signal = "QUÁ MUA (Cảnh Báo Chốt Lời)"
                signal_type = "OVERBOUGHT"
                score = 25
            elif is_death_cross and (k_now >= 70):
                signal = "CẢNH BÁO BÁN (Death Cross)"
                signal_type = "SELL"
                score = 15
            elif is_bearish:
                signal = "CẮT KỸ THUẬT (Nến Giảm)"
                signal_type = "SELL"
                score = 20
            else:
                signal = "TRUNG TÍNH (Đi Ngang)"
                signal_type = "NEUTRAL"
                score = 50
                
            target_price = c_now + 1.6 * atr if atr > 0 else c_now * 1.08
            stop_loss = max(c_now - 1.2 * atr, c_now * 0.85) if atr > 0 else c_now * 0.94
            
            pattern_str = "Pinbar Rút Chân" if is_pinbar else ("Bullish Engulfing" if is_engulfing else ("Nến Xanh Tăng" if is_bullish else ("Nhấn Chìm Giảm" if is_bearish else "Nến Đỏ")))
            
            stocks.append({
                "symbol": sym,
                "sector": get_symbol_sector(sym),
                "date": cur_date,
                "price": c_now,
                "change_pct": round(change_pct, 2),
                "turnover_ty": round(turnover_ty, 2),
                "stoch_k": round(float(k_now), 1) if pd.notna(k_now) else 50.0,
                "stoch_d": round(float(d_now), 1) if pd.notna(d_now) else 50.0,
                "rsi": round(float(rsi_now), 1) if pd.notna(rsi_now) else 50.0,
                "wpr": round(float(wpr_now), 1) if pd.notna(wpr_now) else -50.0,
                "atr": round(float(atr), 1) if pd.notna(atr) else 0.0,
                "target": round(float(target_price), 0),
                "stop_loss": round(float(stop_loss), 0),
                "rr_ratio": round(1.6 / 1.2, 2),
                "signal": signal,
                "signal_type": signal_type,
                "score": score,
                "pattern": pattern_str
            })
        except Exception:
            continue

    # Sorting: Highest score first, then highest liquidity
    stocks = sorted(stocks, key=lambda x: (x["score"], x["turnover_ty"]), reverse=True)
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

    buy_count = len([s for s in stocks if s["signal_type"] in ["BUY_STRONG", "BUY"]])
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
    print("Done! Scanned:", res["market_breadth"]["total_scanned"], "stocks.")
    print("Market regime:", res["market_breadth"]["regime"])
    print("Buy count:", res["market_breadth"]["buy_count"])
