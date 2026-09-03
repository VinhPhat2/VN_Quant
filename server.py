import os
import json
from functools import lru_cache
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Import VCI Finance provider for Real Financial Statements
try:
    from providers.vci import VCIFinance
except Exception:
    VCIFinance = None

app = FastAPI(title="VN-Quant Enterprise Terminal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOOKBACK_DAYS = 1000
OOS_DAYS = 60
investor_groups = ["foreign", "prop", "local_inst", "local_ind"]

# Expanded Comprehensive Sector Dictionary for VN Market
SECTOR_MAP = {
    "Ngân hàng": ["VCB", "TCB", "MBB", "ACB", "VPB", "CTG", "BID", "HDB", "STB", "TPB", "VIB", "LPB", "MSB", "OCB", "SHB", "SSB", "EIB", "BAB", "NAB", "BVB", "KLB", "PGB", "SGB", "VBB"],
    "Chứng khoán": ["SSI", "VND", "VCI", "HCM", "SHS", "MBS", "FTS", "BSI", "CTS", "VDS", "ORS", "AGR", "BVS", "PSI", "WSS", "IVS", "EVS", "TVS", "VIX", "APG", "TCI", "HAC", "VFS"],
    "Bất động sản": ["VHM", "VIC", "VRE", "NVL", "PDR", "DIG", "DXG", "KDH", "NLG", "CEO", "KBC", "IDC", "SZC", "BCM", "TCH", "HDC", "NTL", "DXS", "SCR", "LDG", "QCG", "AGG", "DRH", "KHG", "NBB", "IJC", "ITC", "CRE", "HQC", "TDH", "VPH", "NDN", "SJS", "SIP", "D2D", "L14", "API", "IDJ", "CSC"],
    "Thép & Kim loại": ["HPG", "HSG", "NKG", "VGS", "TLH", "TVN", "POM", "SMC", "TIS", "KVC", "HLA", "PAS", "BMC", "KSQ", "MSR", "TTB"],
    "Dầu khí & Năng lượng": ["GAS", "PVD", "PVS", "BSR", "PLX", "PVT", "PVC", "PVE", "POW", "NT2", "PGV", "GEG", "REE", "HDG", "BCG", "PC1", "VSH", "QTP", "HND", "SJD", "TTA", "KHP", "PPC", "BTP", "CHP", "SEB", "VNE"],
    "Bán lẻ & Tiêu dùng": ["MWG", "DGW", "FRT", "PNJ", "MSN", "VNM", "SAB", "KDC", "VHC", "ANV", "IDI", "DBC", "BAF", "HAG", "PAN", "SBT", "QNS", "MML", "MCH", "BHN", "CLX", "TAR", "LTG", "AFX", "NAF", "ABT", "AAM", "ACL"],
    "Hóa chất & Phân bón": ["DGC", "DCM", "DPM", "BFC", "CSV", "LAS", "PHR", "DPR", "GVR", "DRC", "CSM", "AAA", "APH", "BMP", "NTP", "HII", "DDV", "PSE", "PSW", "SFG", "BRC", "TNC"],
    "Xây dựng & Vật liệu": ["VCG", "HHV", "FCN", "LCG", "CII", "C4G", "CTD", "HBC", "HT1", "BCC", "DHA", "KSB", "VLB", "C32", "DAT", "HOM", "SCG", "BTS", "ACC", "BCE", "C47", "CDC", "CIG", "CTI", "DC4", "DPG", "EVG", "G36", "HID", "HUB", "HUT", "MST", "PHC", "SZB", "TCD", "VMC"],
    "Công nghệ & Viễn thông": ["FPT", "CMG", "FOX", "CTR", "VGI", "ELC", "ITD", "SAM", "SGT", "TTN", "ICT"],
    "Cảng biển & Logistics": ["GMD", "HAH", "VSC", "PVT", "VOS", "VIP", "DVP", "TCL", "TMS", "DXP", "SGP", "PHP", "PVP", "VTO", "PDN", "SWC", "ILB", "ASG", "AST", "CLL"],
    "Dược phẩm & Y tế": ["DHG", "IMP", "TRA", "DMC", "DBD", "AMV", "DCL", "JVC", "TNH", "VMD"],
    "Bảo hiểm": ["BVH", "BMI", "PVI", "BIC", "MIG", "PTI", "PRE", "BLI", "VNR"],
    "Dệt may & Da giày": ["TNG", "MSH", "VGT", "GIL", "STK", "TCM", "EVE", "ADS", "KMR"],
}

def get_symbol_sector(symbol: str) -> str:
    sym = symbol.upper()
    for sec, syms in SECTOR_MAP.items():
        if sym in syms:
            return sec
    return "Khác / Sản xuất"

def get_data_dir():
    default_dir = os.path.join(os.path.dirname(__file__), "csv_data")
    if os.path.exists(default_dir):
        return default_dir
    return r"E:\hoctap\NLP\source_ck\data\csv_data"

def get_result_path():
    default_res = os.path.join(os.path.dirname(__file__), "optuna_results_all_symbols.csv")
    if os.path.exists(default_res):
        return default_res
    return r"E:\hoctap\NLP\source_ck\data\optuna_results_all_symbols.csv"

def load_and_map_excel(excel_path):
    try:
        df_new = pd.read_csv(excel_path)
    except Exception:
        return None
        
    mapping = {
        'NGÀY': 'time', 'GIÁ': 'close', 'MỞ': 'open', 'CAO': 'high', 'THẤP': 'low',
        'KL KHỚP': 'volume', 'GT KHỚP': 'trading_value',
        'GT Nước Ngoài Khớp Ròng': 'foreign_net_val', 'GT Tự doanh Khớp Ròng': 'prop_net_val',
        'GT Cá Nhân Khớp Ròng': 'local_ind_net_val', 'GT Tổ chức Khớp Ròng': 'local_inst_net_val',
        'KL Nước Ngoài Khớp Ròng': 'foreign_net_vol', 'KL Tự doanh Khớp Ròng': 'prop_net_vol',
        'KL Cá Nhân Khớp Ròng': 'local_ind_net_vol', 'KL Tổ chức Khớp Ròng': 'local_inst_net_vol',
    }
    df_new = df_new.rename(columns=mapping)
    if 'time' in df_new.columns:
        df_new['time'] = pd.to_datetime(df_new['time'], format='%d/%m/%Y', errors='coerce')
    return df_new.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)

def get_file_path(symbol: str, data_dir: str):
    symbol = symbol.upper()
    if not os.path.exists(data_dir):
        return None
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            parts = fname.replace(".csv", "").split("_")
            if parts[-1].upper() == symbol:
                return os.path.join(data_dir, fname)
    return None

def build_features(df_vni_raw, a, b, c, d, investor_groups):
    df = df_vni_raw.copy()
    keep_cols = [
        'open', 'high', 'low', 'close', 'trading_value', 'volume',
        'foreign_net_val', 'prop_net_val', 'local_ind_net_val', 'local_inst_net_val',
        'foreign_net_vol', 'prop_net_vol', 'local_ind_net_vol', 'local_inst_net_vol',
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols]

    df['trading_value'] = df['trading_value'].replace(0, np.nan).interpolate().ffill().bfill()
    df['close'] = df['close'].ffill().bfill()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    df['abs_ret'] = df['log_ret'].abs()

    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        )
    )
    df['target_ret_t10'] = df['close'].shift(-10) / df['close'] - 1

    for g in investor_groups:
        if f'{g}_net_val' not in df.columns:
            df[f'{g}_net_val'] = 0.0
        if f'{g}_net_vol' not in df.columns:
            df[f'{g}_net_vol'] = 0.0
            
        net_val = df[f'{g}_net_val']
        net_ratio = net_val / (df['trading_value'] + 1e-6)
        df[f'feat_{g}_net_ratio'] = net_ratio.fillna(0)

        prev = net_ratio.shift(1).abs()
        df[f'feat_{g}_price_impact'] = (df['abs_ret'].shift(1) / (prev + 1e-6)).rolling(5).median()

        mu = df[f'feat_{g}_net_ratio'].rolling(a, min_periods=5).mean()
        sigma = df[f'feat_{g}_net_ratio'].rolling(a, min_periods=5).std()
        df[f'feat_{g}_zscore'] = ((df[f'feat_{g}_net_ratio'] - mu) / (sigma + 1e-8)).clip(-4, 4).fillna(0)

        rmin = net_val.rolling(a, min_periods=5).min()
        rmax = net_val.rolling(a, min_periods=5).max()
        df[f'feat_{g}_sentiment'] = (2 * (net_val - rmin) / (rmax - rmin + 1e-6) - 1).fillna(0)

    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    feature_dict = {}

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    feature_dict["ta_rsi_norm"] = (rsi / 100.0) - 0.5

    ma20 = df["close"].rolling(20, min_periods=5).mean()
    feature_dict["ta_dist_ma20"] = (df["close"] - ma20) / (ma20 + 1e-8)

    roll_high = df["high"].rolling(c, min_periods=5).max()
    roll_low = df["low"].rolling(c, min_periods=5).min()
    feature_dict["ta_pos_in_range"] = (df["close"] - roll_low) / (roll_high - roll_low + 1e-8)
    feature_dict["ta_atr_pct"] = tr.ewm(span=b, adjust=False).mean() / (df["close"] + 1e-8)

    for g in investor_groups:
        sent = df[f"feat_{g}_sentiment"]
        sent_ewm = sent.ewm(span=b, adjust=False).mean()
        feature_dict[f"feat_{g}_sentiment_ewm"] = sent_ewm

        sent_mean = sent_ewm.rolling(a, min_periods=5).mean()
        sent_std = sent_ewm.rolling(a, min_periods=5).std().replace(0, np.nan)
        sent_z = ((sent_ewm - sent_mean) / (sent_std + 1e-6)).clip(-3, 3).fillna(0)

        feature_dict[f"feat_{g}_sentiment_cubed"] = (sent_z**3 - 3 * sent_z).clip(-10, 10)
        impulse = sent_ewm.diff(d).fillna(0)
        feature_dict[f"feat_{g}_flow_impulse_{d}d"] = impulse

        imp_mean = impulse.rolling(a, min_periods=5).mean()
        imp_std = impulse.rolling(a, min_periods=5).std().replace(0, np.nan)
        imp_z = ((impulse - imp_mean) / (imp_std + 1e-6)).clip(-3, 3).fillna(0)

        feature_dict[f"feat_{g}_flow_impulse_{d}d_cubed"] = (imp_z**3 - 3 * imp_z).clip(-10, 10)
        feature_dict[f"feat_{g}_eff_flow"] = sent * df[f"feat_{g}_price_impact"]
        feature_dict[f"feat_{g}_val_share"] = df[f"{g}_net_val"].abs() / (df["trading_value"] + 1e-6)

        implied = np.where(df[f"{g}_net_vol"] != 0, df[f"{g}_net_val"] / df[f"{g}_net_vol"], 0)
        feature_dict[f"feat_{g}_rel_net_price_ewm"] = (
            pd.Series(implied / (df["close"] + 1e-8), index=df.index).ewm(span=b, adjust=False).mean()
        )

        feature_dict[f"inter_{g}_sent_x_pos"] = sent_z * feature_dict["ta_pos_in_range"]
        feature_dict[f"inter_{g}_sent_x_distma"] = sent_z * feature_dict["ta_dist_ma20"]
        feature_dict[f"inter_{g}_eff_x_pos"] = feature_dict[f"feat_{g}_eff_flow"] * feature_dict["ta_pos_in_range"]
        feature_dict[f"inter_{g}_sent_x_atr"] = sent_z * feature_dict["ta_atr_pct"]

    groups = investor_groups
    sent_ewm_dict = {g: feature_dict[f"feat_{g}_sentiment_ewm"] for g in groups}

    for g1, g2 in combinations(groups, 2):
        feature_dict[f"cross_agree_sign_{g1}_{g2}"] = np.sign(sent_ewm_dict[g1]) * np.sign(sent_ewm_dict[g2])
        feature_dict[f"cross_agree_strength_{g1}_{g2}"] = sent_ewm_dict[g1] * sent_ewm_dict[g2]

    sent_matrix = pd.concat([sent_ewm_dict[g] for g in groups], axis=1)
    sent_matrix.columns = groups
    feature_dict["cross_dispersion_std"] = sent_matrix.std(axis=1)
    feature_dict["cross_dispersion_range"] = sent_matrix.max(axis=1) - sent_matrix.min(axis=1)

    total_abs_net = sum(df[f"{g}_net_val"].abs() for g in groups) + 1e-8
    for g in groups:
        feature_dict[f"cross_dominance_{g}"] = df[f"{g}_net_val"].abs() / total_abs_net

    df_feat = pd.DataFrame(feature_dict, index=df.index)
    df_feat["close"] = df["close"]
    model_data = pd.concat([df_feat, df["target_ret_t10"]], axis=1)
    model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna()

    feature_cols = [c_ for c_ in df_feat.columns if c_ != "close"]
    return model_data, feature_cols

@app.get("/api/screener")
def get_screener():
    res_path = get_result_path()
    if not os.path.exists(res_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    df = pd.read_csv(res_path)
    df["hit_rate_oos"] = (df["hit_rate_oos"] * 100).round(1)
    df["hit_rate_train"] = (df["hit_rate_train"] * 100).round(1)
    df["oos_ic"] = df["oos_ic"].round(3)
    df["train_ic"] = df["train_ic"].round(3)
    df["score"] = df["score"].round(3)
    df["chosen_alpha"] = df["chosen_alpha"].round(2)
    df["sector"] = df["symbol"].apply(get_symbol_sector)
    return df.to_dict(orient="records")

# 100% Exact Notebook RdYlGn calculation normalized by (n_deciles - 1)
def get_exact_notebook_rdylgn_colors(deciles_series, n_deciles):
    norm_denom = (n_deciles - 1) if n_deciles > 1 else 1
    point_colors = []
    for d in deciles_series:
        val = (int(d) - 1) / norm_denom
        hex_col = mcolors.to_hex(cm.RdYlGn(val))
        point_colors.append(hex_col)
    return point_colors

def make_clean_plotly_decile_fig(oos_price_data, n_deciles, is_dark=True):
    odata = oos_price_data
    if odata.empty:
        return {}

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.035,
    )

    p_line_price = "#4A5568" if not is_dark else "#94A3B8"
    p_bg = "#0B0F19" if is_dark else "#FFFFFF"
    p_plot_bg = "#080C14" if is_dark else "#F8FAFC"
    p_grid = "rgba(255, 255, 255, 0.07)" if is_dark else "#E2E8F0"

    dates = odata["date_str"].tolist()
    closes = odata["close"].tolist()
    deciles = odata["decile"].astype(int).tolist()
    point_colors = get_exact_notebook_rdylgn_colors(deciles, n_deciles)

    # Line price (matching original notebook: color="#4A5568", lw=1.5, alpha=0.7)
    fig.add_trace(
        go.Scatter(
            x=dates, y=closes, mode="lines",
            line=dict(color=p_line_price, width=1.6), name="Giá Close",
            hoverinfo="skip"
        ),
        row=1, col=1,
    )

    # Decile Scatter Markers with exact RdYlGn normalization (vmin=1, vmax=n_deciles)
    fig.add_trace(
        go.Scatter(
            x=dates, y=closes, mode="markers",
            marker=dict(
                size=10,
                color=deciles,
                colorscale="RdYlGn",
                cmin=1,
                cmax=n_deciles,
                showscale=True,
                colorbar=dict(
                    title=dict(text=f"Decile 1→{n_deciles}", side="top"),
                    tickvals=list(range(1, n_deciles + 1, 2)) if n_deciles > 5 else list(range(1, n_deciles + 1)),
                    len=0.75,
                    y=0.62,
                    thickness=12,
                    tickfont=dict(family="JetBrains Mono", size=10),
                ),
                line=dict(width=0.8, color="#000000"),
            ),
            name=f"Tín Hiệu Decile (1→{n_deciles})",
            hovertemplate="Ngày: <b>%{x}</b><br>Giá: <b>%{y:,.0f} VND</b><br>Tín hiệu: <b>Decile %{marker.color}</b><extra></extra>",
        ),
        row=1, col=1,
    )

    # Bottom Decile Bars with exact RdYlGn palette matching notebook
    fig.add_trace(
        go.Bar(
            x=dates, y=deciles, showlegend=False,
            marker_color=point_colors,
            hovertemplate="Ngày: %{x}<br>Decile: <b>%{y}</b><extra></extra>"
        ),
        row=2, col=1,
    )

    y_min, y_max = float(min(closes)), float(max(closes))
    padding = (y_max - y_min) * 0.12 if y_max > y_min else y_max * 0.05

    fig.update_layout(
        height=490,
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor=p_bg,
        plot_bgcolor=p_plot_bg,
        margin=dict(l=15, r=15, t=15, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.85, font=dict(family="JetBrains Mono", size=11)),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Giá (VND)", range=[y_min - padding, y_max + padding], row=1, col=1, gridcolor=p_grid, tickfont=dict(family="JetBrains Mono"))
    fig.update_yaxes(title_text="Decile", range=[0, n_deciles + 1], dtick=2 if n_deciles > 6 else 1, row=2, col=1, gridcolor=p_grid, tickfont=dict(family="JetBrains Mono"))
    fig.update_xaxes(gridcolor=p_grid, nticks=12, tickangle=-25, tickfont=dict(family="JetBrains Mono", size=10))
    return json.loads(pio.to_json(fig))

def make_technical_indicator_fig(df_tail, is_dark=True):
    p_bg = "#0B0F19" if is_dark else "#FFFFFF"
    p_plot_bg = "#080C14" if is_dark else "#F8FAFC"
    p_grid = "rgba(255, 255, 255, 0.07)" if is_dark else "#E2E8F0"

    dates = df_tail["date_str"].tolist()
    op = df_tail["open"].tolist()
    hi = df_tail["high"].tolist()
    lo = df_tail["low"].tolist()
    cl = df_tail["close"].tolist()

    # Technical Indicators
    delta = pd.Series(cl).diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = (100 - (100 / (1 + rs))).fillna(50).tolist()

    exp1 = pd.Series(cl).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(cl).ewm(span=26, adjust=False).mean()
    macd = (exp1 - exp2).tolist()
    signal = pd.Series(macd).ewm(span=9, adjust=False).mean().tolist()
    hist = (pd.Series(macd) - pd.Series(signal)).tolist()

    ma20 = pd.Series(cl).rolling(20, min_periods=5).mean().tolist()
    bb_std = pd.Series(cl).rolling(20, min_periods=5).std().fillna(0)
    bb_upper = (pd.Series(ma20) + 2 * bb_std).tolist()
    bb_lower = (pd.Series(ma20) - 2 * bb_std).tolist()

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23], vertical_spacing=0.03,
        subplot_titles=("Biểu Đồ Nến & Dải Bollinger Bands", "Chỉ Báo Sức Mạnh Giá RSI (14)", "Chỉ Báo MACD (12, 26, 9)")
    )

    # Row 1: Candlestick + Bollinger Bands + MA20
    fig.add_trace(go.Candlestick(x=dates, open=op, high=hi, low=lo, close=cl, name="Nến OHLC", increasing_line_color="#00E676", decreasing_line_color="#FF334B"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=bb_upper, line=dict(color="#A855F7", width=1, dash="dot"), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ma20, line=dict(color="#00E5FF", width=1.5), name="MA20 (BB Mid)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=bb_lower, line=dict(color="#A855F7", width=1, dash="dot"), name="BB Lower", fill="tonexty", fillcolor="rgba(168, 85, 247, 0.05)"), row=1, col=1)

    # Row 2: RSI(14) with 30 / 70 bands
    fig.add_trace(go.Scatter(x=dates, y=rsi, line=dict(color="#38BDF8", width=1.8), name="RSI(14)"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#FF334B", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00E676", row=2, col=1)

    # Row 3: MACD & Signal & Hist
    fig.add_trace(go.Scatter(x=dates, y=macd, line=dict(color="#00E5FF", width=1.5), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=signal, line=dict(color="#F59E0B", width=1.5), name="Signal"), row=3, col=1)
    fig.add_trace(go.Bar(x=dates, y=hist, marker_color=["#00E676" if h >= 0 else "#FF334B" for h in hist], name="MACD Hist"), row=3, col=1)

    fig.update_layout(
        height=660,
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor=p_bg,
        plot_bgcolor=p_plot_bg,
        margin=dict(l=15, r=15, t=25, b=15),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="JetBrains Mono", size=10)),
    )
    fig.update_yaxes(gridcolor=p_grid, tickfont=dict(family="JetBrains Mono"))
    fig.update_xaxes(gridcolor=p_grid, nticks=12, tickangle=-25, tickfont=dict(family="JetBrains Mono", size=10))
    return json.loads(pio.to_json(fig))

_financial_cache = {}

@app.get("/api/financials/{symbol}")
def get_vci_financial_data(symbol: str):
    symbol = symbol.upper()
    if symbol in _financial_cache:
        return _financial_cache[symbol]

    res = {
        "ratios": {"headers": [], "rows": []},
        "income_statement": {"headers": [], "rows": []}
    }
    if VCIFinance is None:
        return res
    try:
        vci = VCIFinance(symbol, period="quarter")
        
        # 1. Real Ratios
        df_r = vci.ratio(lang="vi")
        if df_r is not None and not df_r.empty:
            period_cols = [c for c in df_r.columns if c not in ("item", "item_en", "item_id")]
            latest_periods = period_cols[-5:] if len(period_cols) >= 5 else period_cols
            headers = ["Chỉ Số Tài Chính & Định Giá"] + latest_periods
            rows = []
            for _, r in df_r.iterrows():
                row_vals = [str(r.get("item", ""))]
                for p in latest_periods:
                    val = r.get(p, 0)
                    if pd.notna(val) and isinstance(val, (int, float)):
                        if abs(val) >= 1e9:
                            row_vals.append(f"{val/1e9:,.1f} Tỷ")
                        elif "ROE" in str(r.get("item", "")) or "ROA" in str(r.get("item", "")) or "%" in str(r.get("item", "")):
                            row_vals.append(f"{val:.2f}%")
                        else:
                            row_vals.append(f"{val:.2f}")
                    else:
                        row_vals.append(str(val) if pd.notna(val) else "-")
                rows.append(row_vals)
            res["ratios"] = {"headers": headers, "rows": rows}

        # 2. Real Income Statement
        df_inc = vci.income_statement(lang="vi")
        if df_inc is not None and not df_inc.empty:
            period_cols = [c for c in df_inc.columns if c not in ("item", "item_en", "item_id")]
            latest_periods = period_cols[-5:] if len(period_cols) >= 5 else period_cols
            headers = ["Chỉ Tiêu Báo Cáo KQKD"] + latest_periods
            rows = []
            for _, r in df_inc.head(15).iterrows():
                row_vals = [str(r.get("item", ""))]
                for p in latest_periods:
                    val = r.get(p, 0)
                    if pd.notna(val) and isinstance(val, (int, float)):
                        row_vals.append(f"{val/1e9:,.1f} Tỷ" if abs(val) >= 1e9 else f"{val:,.0f}")
                    else:
                        row_vals.append(str(val) if pd.notna(val) else "-")
                rows.append(row_vals)
            res["income_statement"] = {"headers": headers, "rows": rows}
            
    except Exception as e:
        print(f"[VCI] Error fetching statements for {symbol}: {e}")
    _financial_cache[symbol] = res
    return res

_stock_cache = {}

@app.get("/api/stock/{symbol}")
def get_stock_analysis(symbol: str, theme: str = "dark"):
    symbol = symbol.upper()
    cache_key = f"{symbol}_{theme}"
    if cache_key in _stock_cache:
        return _stock_cache[cache_key]

    res_path = get_result_path()
    data_dir = get_data_dir()
    
    if not os.path.exists(res_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    df_all = pd.read_csv(res_path)
    sub = df_all[df_all["symbol"] == symbol]
    if sub.empty:
        raise HTTPException(status_code=404, detail="Symbol not found in optuna results")
    
    row = sub.iloc[0]
    a, b, c, d = int(row["a"]), int(row["b"]), int(row["c"]), int(row["d"])
    alpha = float(row["chosen_alpha"])
    
    file_path = get_file_path(symbol, data_dir)
    if not file_path:
        raise HTTPException(status_code=404, detail="Stock data file not found")
        
    df_raw = load_and_map_excel(file_path)
    if df_raw is None or len(df_raw) < 100:
        raise HTTPException(status_code=400, detail="Insufficient stock data")
        
    model_data, all_feature_cols = build_features(df_raw, a, b, c, d, investor_groups)
    y_col = "target_ret_t10"
    
    total_needed = LOOKBACK_DAYS + OOS_DAYS
    if len(model_data) >= total_needed:
        train_data = model_data.iloc[-total_needed:-OOS_DAYS].copy()
        oos_data = model_data.iloc[-OOS_DAYS:].copy()
    else:
        train_data = model_data.iloc[:-OOS_DAYS].copy() if len(model_data) > OOS_DAYS else model_data.copy()
        oos_data = model_data.iloc[-OOS_DAYS:].copy() if len(model_data) > OOS_DAYS else pd.DataFrame()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data[all_feature_cols])
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train_scaled, train_data[y_col])

    # Top Feature Importance
    feat_coefs = sorted(zip(all_feature_cols, model.coef_), key=lambda x: abs(x[1]), reverse=True)[:10]
    top_features = [{"feature": f, "coef": round(float(c), 4), "impact": "Dương (+)" if c > 0 else "Âm (-)"} for f, c in feat_coefs]

    train_scores = model.predict(X_train_scaled)
    train_data["pred_score"] = train_scores
    _, train_bin_edges = pd.qcut(train_scores, q=10, duplicates="drop", retbins=True)
    train_bin_edges = train_bin_edges.copy()
    train_bin_edges[0] = -np.inf
    train_bin_edges[-1] = np.inf
    n_deciles = len(train_bin_edges) - 1

    oos_data["pred_score"] = model.predict(scaler.transform(oos_data[all_feature_cols]))
    
    # Exact 1-indexed Decile from 1 to n_deciles matching Cell 28 of original notebook
    oos_data["decile"] = (pd.cut(oos_data["pred_score"], bins=train_bin_edges, labels=False, include_lowest=True).astype(int) + 1)

    if "time" in df_raw.columns:
        oos_data["date_str"] = pd.to_datetime(df_raw.loc[oos_data.index, "time"]).dt.strftime('%d/%m/%Y')
    else:
        oos_data["date_str"] = [str(i) for i in oos_data.index]

    # 4-Quadrant Model Audit & Monotonicity Table
    eval_data = pd.concat([
        train_data[["pred_score", y_col]].assign(is_oos=False),
        (oos_data[["pred_score", y_col]].assign(is_oos=True) if len(oos_data) > 0 else pd.DataFrame())
    ], axis=0).dropna()
    
    eval_data["score_decile"] = pd.qcut(eval_data["pred_score"], n_deciles, labels=False, duplicates="drop") + 1
    analysis = eval_data.groupby("score_decile")[y_col].agg(["mean", "std", "count"])
    analysis["hit_rate"] = eval_data.groupby("score_decile")[y_col].apply(lambda x: (x > 0).mean())
    analysis["avg_pred"] = eval_data.groupby("score_decile")["pred_score"].mean()

    decile_table = []
    for d_idx in analysis.index:
        decile_table.append({
            "decile": f"Decile {d_idx}",
            "mean_ret": f"{(analysis.loc[d_idx, 'mean'] * 100):.2f}%",
            "hit_rate": f"{(analysis.loc[d_idx, 'hit_rate'] * 100):.1f}%",
            "std": f"{(analysis.loc[d_idx, 'std'] * 100):.2f}%",
            "count": int(analysis.loc[d_idx, 'count']),
            "signal": "TĂNG MẠNH" if d_idx >= (n_deciles - 1) else ("GIẢM MẠNH" if d_idx <= 2 else "TRUNG TÍNH")
        })

    audit_stats = {
        "deciles": [f"D{i}" for i in analysis.index],
        "mean_ret": (analysis["mean"] * 100).round(2).tolist(),
        "hit_rate": (analysis["hit_rate"] * 100).round(1).tolist(),
        "avg_pred": (analysis["avg_pred"] * 100).round(2).tolist(),
        "train_scores": train_data["pred_score"].tolist()[-60:],
        "train_returns": (train_data[y_col] * 100).tolist()[-60:],
        "oos_scores": oos_data["pred_score"].tolist(),
        "oos_returns": (oos_data[y_col] * 100).tolist(),
        "table": decile_table,
    }

    # Simulation / Strategy Backtest vs Buy & Hold (Top Deciles)
    top_threshold = max(n_deciles - 1, 2)
    oos_data["strat_ret"] = np.where(oos_data["decile"] >= top_threshold, oos_data[y_col], 0.0)
    cum_strat = (1 + oos_data["strat_ret"]).cumprod().tolist()
    cum_bh = (1 + oos_data[y_col]).cumprod().tolist()
    sim_dates = oos_data["date_str"].tolist()

    simulation = {
        "dates": sim_dates,
        "strategy": [(v - 1) * 100 for v in cum_strat],
        "buy_hold": [(v - 1) * 100 for v in cum_bh],
        "total_alpha": round(float((cum_strat[-1] - cum_bh[-1]) * 100), 2) if len(cum_strat) > 0 else 0.0,
    }

    # Plotly Figures with exact n_deciles
    plotly_decile_fig = make_clean_plotly_decile_fig(oos_data, n_deciles=n_deciles, is_dark=(theme == "dark"))
    
    df_raw_tail = df_raw.tail(200).copy()
    if "time" in df_raw_tail.columns:
        df_raw_tail["date_str"] = pd.to_datetime(df_raw_tail["time"]).dt.strftime('%d/%m/%Y')
    else:
        df_raw_tail["date_str"] = [str(i) for i in df_raw_tail.index]
        
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df_raw_tail.columns:
            df_raw_tail[c] = df_raw_tail["close"] if "close" in df_raw_tail.columns else 0.0

    plotly_tech_fig = make_technical_indicator_fig(df_raw_tail, is_dark=(theme == "dark"))

    # Flow Series (Last 60 sessions)
    df_tail_flow = df_raw.tail(60).copy()
    flow_series = []
    for idx, r in df_tail_flow.iterrows():
        t_val = r["time"]
        t_str = pd.to_datetime(t_val).strftime("%d/%m/%Y") if pd.notnull(t_val) else str(idx)
        flow_series.append({
            "time": t_str,
            "foreign": float(r.get("foreign_net_val", 0)) / 1e9,
            "prop": float(r.get("prop_net_val", 0)) / 1e9,
            "local_inst": float(r.get("local_inst_net_val", 0)) / 1e9,
            "local_ind": float(r.get("local_ind_net_val", 0)) / 1e9,
            "close": float(r.get("close", 0)),
        })

    latest_decile = int(oos_data["decile"].iloc[-1])
    sig = "Tăng mạnh" if latest_decile >= top_threshold else ("Giảm mạnh" if latest_decile <= 2 else "Trung tính")

    result = {
        "symbol": symbol,
        "sector": get_symbol_sector(symbol),
        "a": a, "b": b, "c": c, "d": d, "alpha": alpha,
        "n_deciles": n_deciles,
        "latest_decile": latest_decile,
        "signal": sig,
        "score": float(row["score"]),
        "ic_oos": float(row["oos_ic"]),
        "hit_oos": float(row["hit_rate_oos"] * 100 if row["hit_rate_oos"] <= 1 else row["hit_rate_oos"]),
        "plotly_decile_fig": plotly_decile_fig,
        "plotly_tech_fig": plotly_tech_fig,
        "flow": flow_series,
        "audit": audit_stats,
        "top_features": top_features,
        "simulation": simulation,
    }
    _stock_cache[cache_key] = result
    return result

@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
<!DOCTYPE html>
<html lang="vi" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VN-Quant Alpha Terminal | Institutional Flow Engine</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Plus Jakarta Sans', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    },
                    colors: {
                        brand: {
                            dark: '#080C14',
                            card: '#101726',
                            cardLight: '#FFFFFF',
                            borderDark: '#1E293B',
                            borderLight: '#E2E8F0',
                            accent: '#10B981',
                            cyan: '#00E5FF',
                            purple: '#A855F7',
                            amber: '#F59E0B',
                            red: '#FF334B',
                        }
                    }
                }
            }
        }
    </script>
    <style>
        body { transition: background-color 0.2s, color 0.2s; }
        .dark body { background-color: #080C14; color: #F1F5F9; }
        html:not(.dark) body { background-color: #F8FAFC; color: #0F172A; }
        
        .tab-btn.active {
            color: #10B981 !important;
            border-bottom: 2px solid #10B981 !important;
            font-weight: 700 !important;
        }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        .dark ::-webkit-scrollbar-track { background: #101726; }
        .dark ::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 3px; }
        html:not(.dark) ::-webkit-scrollbar-track { background: #F1F5F9; }
        html:not(.dark) ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
    </style>
</head>
<body class="min-h-screen flex flex-col antialiased">
    <!-- Top Header -->
    <header class="bg-white dark:bg-brand-card border-b border-brand-borderLight dark:border-brand-borderDark px-6 py-3 flex justify-between items-center sticky top-0 z-50 shadow-sm">
        <div class="flex items-center space-x-4">
            <div class="flex items-center space-x-2">
                <span class="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></span>
                <span class="font-extrabold text-xl tracking-tight text-slate-900 dark:text-white font-mono">VN-QUANT <span class="text-emerald-600 dark:text-emerald-400 text-xs px-2 py-0.5 rounded bg-emerald-100 dark:bg-emerald-950 border border-emerald-300 dark:border-emerald-800">ALPHA TERMINAL</span></span>
            </div>
            <span class="text-xs text-slate-500 dark:text-slate-400 hidden md:inline border-l border-slate-200 dark:border-slate-800 pl-4 font-mono">Institutional Flow Alpha & Bayesian Ridge Engine (387 Symbols)</span>
        </div>
        <div class="flex items-center space-x-3">
            <!-- Share Modal Trigger Button -->
            <button onclick="openShareModal()" class="flex items-center space-x-1.5 bg-emerald-500/10 border border-emerald-500/40 text-emerald-600 dark:text-emerald-400 px-3 py-1.5 rounded-lg text-xs font-bold hover:bg-emerald-500/20 transition">
                <span>🔗 Chia Sẻ Web</span>
            </button>

            <!-- Theme Toggle -->
            <button onclick="toggleTheme()" id="themeToggleBtn" class="flex items-center space-x-1.5 bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-800 px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-700 dark:text-slate-300 hover:border-emerald-500 transition">
                <span id="themeIcon">☀️ Sáng</span>
            </button>
            
            <!-- Quick Symbol Input -->
            <div class="relative">
                <input id="symbolInput" type="text" placeholder="Tìm mã CP..." value="SSI" class="bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-brand-borderDark px-3 py-1.5 rounded-lg text-sm font-mono text-slate-900 dark:text-white focus:outline-none focus:border-emerald-500 uppercase w-40">
                <button onclick="loadStock(document.getElementById('symbolInput').value)" class="absolute right-1 top-1 bottom-1 px-2.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold rounded">XEM</button>
            </div>
        </div>
    </header>

    <!-- Navigation Tabs Bar -->
    <nav class="bg-slate-50 dark:bg-slate-900/90 border-b border-brand-borderLight dark:border-brand-borderDark px-6 flex space-x-7 text-sm font-medium overflow-x-auto">
        <button onclick="switchTab('tab-charts')" id="btn-tab-charts" class="tab-btn active py-3.5 text-slate-600 dark:text-slate-300 hover:text-emerald-500 whitespace-nowrap transition">Lượng Hóa Decile (Jupyter)</button>
        <button onclick="switchTab('tab-technical')" id="btn-tab-technical" class="tab-btn py-3.5 text-slate-500 dark:text-slate-400 hover:text-emerald-500 whitespace-nowrap transition">Phân Tích Kỹ Thuật (RSI & MACD)</button>
        <button onclick="switchTab('tab-flow')" id="btn-tab-flow" class="tab-btn py-3.5 text-slate-500 dark:text-slate-400 hover:text-emerald-500 whitespace-nowrap transition">Dòng Tiền 4 Nhóm NĐT</button>
        <button onclick="switchTab('tab-sectors')" id="btn-tab-sectors" class="tab-btn py-3.5 text-slate-500 dark:text-slate-400 hover:text-emerald-500 whitespace-nowrap transition">Sức Mạnh Nhóm Ngành</button>
        <button onclick="switchTab('tab-spikes')" id="btn-tab-spikes" class="tab-btn py-3.5 text-slate-500 dark:text-slate-400 hover:text-emerald-500 whitespace-nowrap transition">Radar Gom Đột Biến</button>
        <button onclick="switchTab('tab-financials')" id="btn-tab-financials" class="tab-btn py-3.5 text-slate-500 dark:text-slate-400 hover:text-emerald-500 whitespace-nowrap transition">Báo Cáo Tài Chính (VCI)</button>
        <button onclick="switchTab('tab-features')" id="btn-tab-features" class="tab-btn py-3.5 text-slate-500 dark:text-slate-400 hover:text-emerald-500 whitespace-nowrap transition">Bóc Tách Nhân Tố</button>
        <button onclick="switchTab('tab-audit')" id="btn-tab-audit" class="tab-btn py-3.5 text-slate-500 dark:text-slate-400 hover:text-emerald-500 whitespace-nowrap transition">Kiểm Định Mô Hình</button>
        <button onclick="switchTab('tab-sim')" id="btn-tab-sim" class="tab-btn py-3.5 text-slate-500 dark:text-slate-400 hover:text-emerald-500 whitespace-nowrap transition">Mô Phỏng Backtest</button>
        <button onclick="switchTab('tab-matrix')" id="btn-tab-matrix" class="tab-btn py-3.5 text-slate-500 dark:text-slate-400 hover:text-emerald-500 whitespace-nowrap transition">Ma Trận Thị Trường</button>
        <button onclick="switchTab('tab-screener')" id="btn-tab-screener" class="tab-btn py-3.5 text-slate-500 dark:text-slate-400 hover:text-emerald-500 whitespace-nowrap transition">Top Alpha Screener</button>
        <button onclick="switchTab('tab-strategy')" id="btn-tab-strategy" class="tab-btn py-3.5 text-slate-500 dark:text-slate-400 hover:text-emerald-500 whitespace-nowrap transition">Phương Pháp Luận</button>
    </nav>

    <!-- Main Content Body -->
    <main class="flex-1 p-6 space-y-6">
        <!-- Stock Hero Banner (Always Visible) -->
        <div id="stockHero" class="bg-white dark:bg-brand-card border border-brand-borderLight dark:border-brand-borderDark rounded-xl p-5 flex flex-wrap justify-between items-center gap-4 shadow-sm">
            <div class="flex items-center space-x-4">
                <span id="heroSymbol" class="text-3xl font-extrabold font-mono text-slate-900 dark:text-white tracking-tight">SSI</span>
                <span id="heroSector" class="text-xs font-semibold px-2.5 py-1 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">Chứng khoán</span>
                <span id="heroSignal" class="text-xs font-bold px-3 py-1 rounded-md uppercase tracking-wider bg-emerald-500/10 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border border-emerald-500/40">TĂNG MẠNH (DECILE 10)</span>
            </div>
            <div class="flex items-center space-x-2 text-xs font-mono text-slate-700 dark:text-slate-300 flex-wrap gap-y-2">
                <span id="heroScore" class="bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-brand-borderDark px-3 py-1.5 rounded-md">Score: <b>0.625</b></span>
                <span id="heroIC" class="bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-brand-borderDark px-3 py-1.5 rounded-md">IC OOS: <b class="text-emerald-600 dark:text-emerald-400">0.325</b></span>
                <span id="heroHit" class="bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-brand-borderDark px-3 py-1.5 rounded-md">Hit: <b class="text-sky-600 dark:text-cyan-400">65.0%</b></span>
                <span id="heroParams" class="bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-brand-borderDark px-3 py-1.5 rounded-md text-slate-500 dark:text-slate-400">a=20 b=20 c=30 d=10 α=1000.0</span>
            </div>
        </div>

        <!-- TAB 1: EXACT QUANT DECILE CHART (MATCHING NOTEBOOK RDYLGN) -->
        <div id="tab-charts" class="tab-content space-y-6">
            <div class="bg-white dark:bg-brand-card border border-brand-borderLight dark:border-brand-borderDark rounded-xl p-5 shadow-sm">
                <div class="flex justify-between items-center mb-3">
                    <h3 id="decileChartTitle" class="font-bold text-sm uppercase text-slate-800 dark:text-slate-300 tracking-wider">1. Biểu Đồ Giá & Phân Vùng Tín Hiệu Gom/Thoát OOS (Thang Đo RdYlGn Chuẩn Notebook)</h3>
                    <div class="flex items-center space-x-3 text-xs font-mono">
                        <span class="flex items-center"><span class="w-2.5 h-2.5 rounded-full bg-[#006837] mr-1.5"></span> Top Decile (Tăng mạnh)</span>
                        <span class="flex items-center"><span class="w-2.5 h-2.5 rounded-full bg-[#feeda1] mr-1.5"></span> Mid Decile (Trung tính)</span>
                        <span class="flex items-center"><span class="w-2.5 h-2.5 rounded-full bg-[#a50026] mr-1.5"></span> Low Decile (Giảm mạnh)</span>
                    </div>
                </div>
                <div id="plotlyDecileContainer" class="w-full min-h-[490px]"></div>
            </div>
        </div>

        <!-- TAB 2: TECHNICAL ANALYSIS WITH MULTI-INDICATORS -->
        <div id="tab-technical" class="tab-content hidden space-y-6">
            <div class="bg-white dark:bg-brand-card border border-brand-borderLight dark:border-brand-borderDark rounded-xl p-5 shadow-sm">
                <div class="flex justify-between items-center mb-3">
                    <h3 class="font-bold text-sm uppercase text-slate-800 dark:text-slate-300 tracking-wider">Biểu Đồ Kỹ Thuật Đa Chỉ Báo (Nến OHLC, Bollinger Bands, RSI, MACD)</h3>
                    <span class="text-xs text-slate-500 dark:text-slate-400 font-mono">Khớp lệnh & chỉ báo kỹ thuật thực tế</span>
                </div>
                <div id="plotlyTechContainer" class="w-full min-h-[660px]"></div>
            </div>
        </div>

        <!-- TAB 3: INVESTOR FLOW BREAKDOWN -->
        <div id="tab-flow" class="tab-content hidden space-y-6">
            <div class="bg-white dark:bg-brand-card border border-brand-borderLight dark:border-brand-borderDark rounded-xl p-5 shadow-sm">
                <h3 class="font-bold text-sm uppercase text-slate-800 dark:text-slate-300 tracking-wider mb-4">Luân Chuyển Dòng Tiền 4 Nhóm NĐT (Khối Ngoại, Tự Doanh, Tổ Chức, Cá Nhân)</h3>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6" id="flowCards">
                    <div class="bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-brand-borderDark rounded-lg p-4">
                        <div class="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase">Khối Ngoại (60P)</div>
                        <div id="flowForeign" class="text-2xl font-mono font-bold text-cyan-600 dark:text-cyan-400 mt-1">0.0 Tỷ</div>
                        <div class="text-[11px] text-slate-400 mt-1">Tổng giá trị khớp ròng</div>
                    </div>
                    <div class="bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-brand-borderDark rounded-lg p-4">
                        <div class="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase">Tự Doanh (60P)</div>
                        <div id="flowProp" class="text-2xl font-mono font-bold text-emerald-600 dark:text-emerald-400 mt-1">0.0 Tỷ</div>
                        <div class="text-[11px] text-slate-400 mt-1">Tổng giá trị khớp ròng</div>
                    </div>
                    <div class="bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-brand-borderDark rounded-lg p-4">
                        <div class="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase">Tổ Chức Nội (60P)</div>
                        <div id="flowInst" class="text-2xl font-mono font-bold text-purple-600 dark:text-purple-400 mt-1">0.0 Tỷ</div>
                        <div class="text-[11px] text-slate-400 mt-1">Tổng giá trị khớp ròng</div>
                    </div>
                    <div class="bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-brand-borderDark rounded-lg p-4">
                        <div class="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase">Cá Nhân Nội (60P)</div>
                        <div id="flowInd" class="text-2xl font-mono font-bold text-amber-600 dark:text-amber-400 mt-1">0.0 Tỷ</div>
                        <div class="text-[11px] text-slate-400 mt-1">Tổng giá trị khớp ròng</div>
                    </div>
                </div>
                <div class="h-96">
                    <canvas id="flowChart"></canvas>
                </div>
            </div>
        </div>

        <!-- TAB 4: SECTOR BREADTH & MOMENTUM (ENHANCED VISIBILITY) -->
        <div id="tab-sectors" class="tab-content hidden space-y-6">
            <div class="bg-white dark:bg-brand-card border border-brand-borderLight dark:border-brand-borderDark rounded-xl p-6 shadow-sm space-y-6">
                <div class="flex justify-between items-center">
                    <h3 class="font-bold text-sm uppercase text-slate-800 dark:text-slate-300 tracking-wider">Sức Mạnh Dòng Tiền & Điểm Số Định Lượng Theo Nhóm Ngành</h3>
                    <span class="text-xs text-slate-500 font-mono">Điểm trung bình Score lượng hóa của toàn bộ 387 mã theo ngành</span>
                </div>
                <div class="h-80"><canvas id="sectorChart"></canvas></div>
                <div id="sectorTableContainer" class="overflow-x-auto rounded-lg border border-slate-200 dark:border-brand-borderDark"></div>
            </div>
        </div>

        <!-- TAB 5: SMART MONEY SPIKE RADAR -->
        <div id="tab-spikes" class="tab-content hidden space-y-6">
            <div class="bg-white dark:bg-brand-card border border-brand-borderLight dark:border-brand-borderDark rounded-xl p-6 shadow-sm space-y-4">
                <div class="flex justify-between items-center">
                    <h3 class="font-bold text-sm uppercase text-slate-800 dark:text-slate-300 tracking-wider">Radar Cảnh Báo Gom Đột Biến & Bắt Đáy (Smart Money Spike Scanner)</h3>
                    <span class="text-xs text-slate-500 font-mono">Phát hiện dòng tiền lớn mua ròng bất thường</span>
                </div>
                <div id="spikesTableContainer" class="overflow-x-auto rounded-lg border border-slate-200 dark:border-brand-borderDark"></div>
            </div>
        </div>

        <!-- TAB 6: REAL FINANCIAL STATEMENTS & RATIOS (VCI PROVIDER) -->
        <div id="tab-financials" class="tab-content hidden space-y-6">
            <div class="bg-white dark:bg-brand-card border border-brand-borderLight dark:border-brand-borderDark rounded-xl p-6 shadow-sm space-y-4">
                <div class="flex justify-between items-center">
                    <h3 class="font-bold text-sm uppercase text-slate-800 dark:text-slate-300 tracking-wider">1. Chỉ Số Tài Chính & Định Giá Thực Tế (Vietcap VCI API)</h3>
                    <span class="text-xs text-slate-500 dark:text-slate-400 font-mono">P/E, P/B, P/S, ROE, ROA, Vốn Hóa Theo Quý</span>
                </div>
                <div id="ratiosTableContainer" class="overflow-x-auto rounded-lg border border-slate-200 dark:border-brand-borderDark">
                    <div class="p-6 text-center text-slate-400 font-mono">Bấm vào tab này để tải dữ liệu tài chính trực tiếp từ VCI...</div>
                </div>
            </div>

            <div class="bg-white dark:bg-brand-card border border-brand-borderLight dark:border-brand-borderDark rounded-xl p-6 shadow-sm space-y-4">
                <div class="flex justify-between items-center">
                    <h3 class="font-bold text-sm uppercase text-slate-800 dark:text-slate-300 tracking-wider">2. Báo Cáo Kết Quả Kinh Doanh Thực Tế (Vietcap VCI API)</h3>
                    <span class="text-xs text-slate-500 dark:text-slate-400 font-mono">Doanh thu & Lợi nhuận qua từng kỳ (Tỷ VND)</span>
                </div>
                <div id="financialTableContainer" class="overflow-x-auto rounded-lg border border-slate-200 dark:border-brand-borderDark">
                    <div class="p-6 text-center text-slate-400 font-mono">Bấm vào tab này để tải báo cáo KQKD từ VCI...</div>
                </div>
            </div>
        </div>

        <!-- TAB 7: FEATURE IMPORTANCE -->
        <div id="tab-features" class="tab-content hidden space-y-6">
            <div class="bg-white dark:bg-brand-card border border-brand-borderLight dark:border-brand-borderDark rounded-xl p-5 shadow-sm">
                <h3 class="font-bold text-sm uppercase text-slate-800 dark:text-slate-300 tracking-wider mb-2">Bóc Tách Trọng Số Nhân Tố Ảnh Hưởng Đến Lợi Nhuận T+10</h3>
                <p class="text-xs text-slate-500 dark:text-slate-400 mb-6">Top 10 đặc trưng vi cấu trúc dòng tiền có trọng số hồi quy Bayesian Ridge lớn nhất</p>
                <div id="featuresTableContainer" class="overflow-x-auto rounded-lg border border-slate-200 dark:border-brand-borderDark"></div>
            </div>
        </div>

        <!-- TAB 8: MODEL AUDIT -->
        <div id="tab-audit" class="tab-content hidden space-y-6">
            <div class="bg-white dark:bg-brand-card border border-brand-borderLight dark:border-brand-borderDark rounded-xl p-5 shadow-sm space-y-6">
                <h3 class="font-bold text-sm uppercase text-slate-800 dark:text-slate-300 tracking-wider">Kiểm Định Độ Tin Cậy & Tính Đơn Điệu (Monotonicity Audit)</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-slate-50 dark:bg-slate-900 p-4 rounded-lg border border-slate-200 dark:border-brand-borderDark">
                        <h4 class="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase mb-2">1. Lợi Nhuận Kỳ Vọng T+10 Theo Decile (Mean Ret %)</h4>
                        <div class="h-64"><canvas id="auditChart1"></canvas></div>
                    </div>
                    <div class="bg-slate-50 dark:bg-slate-900 p-4 rounded-lg border border-slate-200 dark:border-brand-borderDark">
                        <h4 class="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase mb-2">2. Tỷ Lệ Thắng Theo Decile (Hit Rate %)</h4>
                        <div class="h-64"><canvas id="auditChart2"></canvas></div>
                    </div>
                    <div class="bg-slate-50 dark:bg-slate-900 p-4 rounded-lg border border-slate-200 dark:border-brand-borderDark">
                        <h4 class="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase mb-2">3. Phân Tán Dự Báo vs Thực Tế (Train & OOS)</h4>
                        <div class="h-64"><canvas id="auditChart3"></canvas></div>
                    </div>
                    <div class="bg-slate-50 dark:bg-slate-900 p-4 rounded-lg border border-slate-200 dark:border-brand-borderDark">
                        <h4 class="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase mb-2">4. So Sánh Điểm Dự Báo vs Thực Tế</h4>
                        <div class="h-64"><canvas id="auditChart4"></canvas></div>
                    </div>
                </div>

                <div class="mt-4">
                    <h4 class="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase mb-3">Bảng Thống Kê Chi Tiết Hiệu Suất Từng Decile</h4>
                    <div id="auditTableContainer" class="overflow-x-auto rounded-lg border border-slate-200 dark:border-brand-borderDark"></div>
                </div>
            </div>
        </div>

        <!-- TAB 9: BACKTEST SIMULATION -->
        <div id="tab-sim" class="tab-content hidden space-y-6">
            <div class="bg-white dark:bg-brand-card border border-brand-borderLight dark:border-brand-borderDark rounded-xl p-5 shadow-sm">
                <div class="flex justify-between items-center mb-4">
                    <div>
                        <h3 class="font-bold text-sm uppercase text-slate-800 dark:text-slate-300 tracking-wider">Mô Phỏng Đường Cong Vốn Chiến Lược (Top Decile vs Buy & Hold)</h3>
                        <p class="text-xs text-slate-500 dark:text-slate-400">So sánh hiệu suất chiến lược lượng hóa định lượng so với nắm giữ thụ động trong giai đoạn OOS</p>
                    </div>
                    <div id="simAlphaBadge" class="text-xs font-mono font-bold px-3 py-1.5 rounded-lg bg-emerald-500/10 text-emerald-500 border border-emerald-500/30">Alpha: +0.0%</div>
                </div>
                <div class="h-80"><canvas id="simChart"></canvas></div>
            </div>
        </div>

        <!-- TAB 10: MARKET ALPHA MATRIX -->
        <div id="tab-matrix" class="tab-content hidden space-y-6">
            <div class="bg-white dark:bg-brand-card border border-brand-borderLight dark:border-brand-borderDark rounded-xl p-5 shadow-sm">
                <h3 class="font-bold text-sm uppercase text-slate-800 dark:text-slate-300 tracking-wider mb-4">Ma Trận Hiệu Quả & Phân Bổ Tương Quan Toàn Thị Trường (387 Mã)</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-slate-50 dark:bg-slate-900 p-4 rounded-lg border border-slate-200 dark:border-brand-borderDark">
                        <h4 class="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase mb-2">Ma Trận IC OOS vs Hit Rate (%)</h4>
                        <div class="h-72"><canvas id="matrixScatterChart"></canvas></div>
                    </div>
                    <div class="bg-slate-50 dark:bg-slate-900 p-4 rounded-lg border border-slate-200 dark:border-brand-borderDark">
                        <h4 class="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase mb-2">Phân Bổ Tần Suất IC OOS</h4>
                        <div class="h-72"><canvas id="matrixHistChart"></canvas></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 11: SCREENER 387 SYMBOLS -->
        <div id="tab-screener" class="tab-content hidden space-y-6">
            <div class="bg-white dark:bg-brand-card border border-brand-borderLight dark:border-brand-borderDark rounded-xl p-5 shadow-sm">
                <div class="flex flex-wrap justify-between items-center mb-4 gap-3">
                    <h3 class="font-bold text-sm uppercase text-slate-800 dark:text-slate-300 tracking-wider">Bảng Xếp Hạng Alpha Toàn Bộ Thị Trường (387 Mã)</h3>
                    
                    <div class="flex items-center space-x-3">
                        <button onclick="filterScreenerByBadge('all')" class="px-2.5 py-1 text-xs font-mono rounded bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300">Tất Cả</button>
                        <button onclick="filterScreenerByBadge('top_ic')" class="px-2.5 py-1 text-xs font-mono rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500">IC ≥ 0.15</button>
                        <button onclick="filterScreenerByBadge('top_hit')" class="px-2.5 py-1 text-xs font-mono rounded bg-sky-500/10 text-sky-600 dark:text-sky-400 border border-sky-500">Hit ≥ 60%</button>
                        <input id="screenerSearchTab" onkeyup="filterScreenerFullTable()" type="text" placeholder="Tìm mã CP..." class="bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-brand-borderDark px-3 py-1.5 rounded-lg text-xs font-mono text-slate-900 dark:text-white w-48 focus:outline-none focus:border-emerald-500">
                    </div>
                </div>
                <div class="overflow-x-auto rounded-lg border border-slate-200 dark:border-brand-borderDark max-h-[650px] overflow-y-auto">
                    <table class="w-full text-left border-collapse text-xs font-mono">
                        <thead class="bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400 uppercase sticky top-0 border-b border-slate-200 dark:border-brand-borderDark">
                            <tr>
                                <th class="p-3 font-bold">Mã CP</th>
                                <th class="p-3">Ngành</th>
                                <th class="p-3 text-right">Score</th>
                                <th class="p-3 text-right">IC OOS</th>
                                <th class="p-3 text-right">Hit OOS %</th>
                                <th class="p-3 text-right">IC Train</th>
                                <th class="p-3 text-right">Alpha</th>
                                <th class="p-3 text-right">a</th>
                                <th class="p-3 text-right">b</th>
                                <th class="p-3 text-right">c</th>
                                <th class="p-3 text-right">d</th>
                            </tr>
                        </thead>
                        <tbody id="screenerFullTableBody" class="divide-y divide-slate-200 dark:divide-slate-800">
                            <tr><td colspan="11" class="text-center p-6 text-slate-400">Đang tải danh sách 387 mã...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TAB 12: STRATEGY GUIDE / QUANTITATIVE SPECIFICATION -->
        <div id="tab-strategy" class="tab-content hidden space-y-6">
            <div class="bg-white dark:bg-brand-card border border-brand-borderLight dark:border-brand-borderDark rounded-xl p-6 text-slate-700 dark:text-slate-300 leading-relaxed text-sm shadow-sm font-sans">
                <h3 class="text-lg font-bold text-slate-900 dark:text-white mb-2 font-mono">BÁO CÁO PHƯƠNG PHÁP LUẬN ĐỊNH LƯỢNG (QUANTITATIVE FLOW ALPHA SPECIFICATION)</h3>
                <p class="text-xs text-slate-500 mb-6 font-mono">Hệ thống tối ưu hóa Bayesian Ridge Regression & Bóc tách vi cấu trúc thị trường Việt Nam</p>

                <div class="space-y-6">
                    <div>
                        <h4 class="font-bold text-slate-900 dark:text-white text-base mb-2">1. Cấu Trúc Dữ Liệu & Không Gian Đặc Trưng (Feature Space)</h4>
                        <p class="mb-2">Hệ thống bóc tách dữ liệu giao dịch khớp lệnh thực tế giữa 4 nhóm chủ thể thị trường:</p>
                        <ul class="list-disc pl-5 space-y-1 text-xs font-mono text-slate-600 dark:text-slate-400">
                            <li><b>Foreign (Khối Ngoại)</b>: Dẫn dắt dòng vốn trung - dài hạn, định giá tài sản cơ bản.</li>
                            <li><b>Proprietary (Tự Doanh)</b>: Dòng tiền tạo lập có lợi thế thông tin và nghiệp vụ hedging.</li>
                            <li><b>Local Institutions (Tổ Chức Nội)</b>: Quỹ mở, quỹ đóng và dòng tiền doanh nghiệp.</li>
                            <li><b>Local Individuals (Cá Nhân Nội)</b>: Dòng tiền thanh khoản lớn, phản ứng theo đà tâm lý.</li>
                        </ul>
                    </div>

                    <hr class="border-slate-200 dark:border-brand-borderDark">

                    <div>
                        <h4 class="font-bold text-slate-900 dark:text-white text-base mb-2">2. Các Biến Lượng Hóa Nền Tảng (Core Engineered Signals)</h4>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
                            <div class="bg-slate-50 dark:bg-slate-900 p-3.5 rounded border border-slate-200 dark:border-brand-borderDark">
                                <span class="font-bold text-emerald-600 dark:text-emerald-400">Flow Sentiment & Z-Score:</span>
                                <p class="text-slate-500 mt-1">Chuẩn hóa mức độ mua/bán ròng theo chu kỳ cửa sổ <i>a</i> ngày so với biến động lịch sử.</p>
                            </div>
                            <div class="bg-slate-50 dark:bg-slate-900 p-3.5 rounded border border-slate-200 dark:border-brand-borderDark">
                                <span class="font-bold text-sky-600 dark:text-cyan-400">Flow Impulse:</span>
                                <p class="text-slate-500 mt-1">Gia tốc biến động dòng tiền: ΔSentiment trong cửa sổ <i>d</i> ngày để bắt các nhịp gom bất thường.</p>
                            </div>
                            <div class="bg-slate-50 dark:bg-slate-900 p-3.5 rounded border border-slate-200 dark:border-brand-borderDark">
                                <span class="font-bold text-purple-600 dark:text-purple-400">Cross Agreement:</span>
                                <p class="text-slate-500 mt-1">Mức độ đồng thuận hướng lệnh giữa Khối Ngoại và Tự Doanh (tăng độ tin cậy khi cả 2 cùng mua).</p>
                            </div>
                            <div class="bg-slate-50 dark:bg-slate-900 p-3.5 rounded border border-slate-200 dark:border-brand-borderDark">
                                <span class="font-bold text-amber-600 dark:text-amber-400">Cross Dispersion & Dominance:</span>
                                <p class="text-slate-500 mt-1">Đo lường độ phân tán và tỷ trọng chi phối của nhóm nhà đầu tư dẫn dắt thanh khoản.</p>
                            </div>
                        </div>
                    </div>

                    <hr class="border-slate-200 dark:border-brand-borderDark">

                    <div>
                        <h4 class="font-bold text-slate-900 dark:text-white text-base mb-2">3. Quy Tắc Giải Ngân Theo Phân Vùng Decile (Execution Rules)</h4>
                        <div class="overflow-x-auto">
                            <table class="w-full text-left text-xs font-mono border border-slate-200 dark:border-brand-borderDark">
                                <thead class="bg-slate-100 dark:bg-slate-900">
                                    <tr>
                                        <th class="p-2.5 border-b border-slate-200 dark:border-brand-borderDark">Phân Vùng</th>
                                        <th class="p-2.5 border-b border-slate-200 dark:border-brand-borderDark">Tín Hiệu</th>
                                        <th class="p-2.5 border-b border-slate-200 dark:border-brand-borderDark">Kỳ Vọng T+10</th>
                                        <th class="p-2.5 border-b border-slate-200 dark:border-brand-borderDark">Hành Động Khuyến Nghị</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr class="bg-emerald-500/10">
                                        <td class="p-2.5 font-bold text-emerald-600 dark:text-emerald-400">Top Decile</td>
                                        <td class="p-2.5 font-semibold">TĂNG MẠNH</td>
                                        <td class="p-2.5">Lợi nhuận T+10 cao nhất thị trường</td>
                                        <td class="p-2.5">Mở vị thế mua, tăng tỷ trọng nắm giữ</td>
                                    </tr>
                                    <tr>
                                        <td class="p-2.5 font-bold text-amber-600 dark:text-amber-400">Mid Decile</td>
                                        <td class="p-2.5 font-semibold">TRUNG TÍNH</td>
                                        <td class="p-2.5">Biến động đi ngang</td>
                                        <td class="p-2.5">Nắm giữ danh mục hiện tại, quan sát</td>
                                    </tr>
                                    <tr class="bg-red-500/10">
                                        <td class="p-2.5 font-bold text-red-600 dark:text-red-400">Low Decile (1-2)</td>
                                        <td class="p-2.5 font-semibold">GIẢM MẠNH</td>
                                        <td class="p-2.5">Xác suất âm cao, rủi ro điều chỉnh</td>
                                        <td class="p-2.5">Hạ tỷ trọng, chốt lời hoặc cắt giảm vị thế</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Share Guide Modal -->
    <div id="shareModal" class="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center hidden p-4">
        <div class="bg-white dark:bg-brand-card border border-slate-300 dark:border-brand-borderDark rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-4">
            <div class="flex justify-between items-center">
                <h3 class="text-base font-bold text-slate-900 dark:text-white flex items-center space-x-2">
                    <span>🚀</span><span>Cách Chia Sẻ Web Cho Bạn Bè Xem Trực Tuyến</span>
                </h3>
                <button onclick="closeShareModal()" class="text-slate-400 hover:text-white text-lg font-bold">✕</button>
            </div>
            
            <div class="space-y-3 text-xs leading-relaxed text-slate-600 dark:text-slate-300 font-sans">
                <div class="bg-emerald-500/10 p-3.5 rounded-lg border border-emerald-500/30">
                    <span class="font-bold text-emerald-500 text-sm">Cách Tạo Link Online Tức Thì (Cloudflare Tunnel):</span>
                    <p class="mt-1">Mở thêm 1 cửa sổ Terminal mới và gõ lệnh sau:</p>
                    <code class="block bg-slate-900 text-emerald-400 p-2 rounded mt-1.5 font-mono text-[11px]">npx cloudflared tunnel --url http://127.0.0.1:8000</code>
                    <p class="mt-1 text-[11px] text-slate-400">Terminal sẽ in ra 1 link <i>https://...trycloudflare.com</i> để gửi bạn bè mở xem ngay!</p>
                </div>
            </div>

            <div class="text-right">
                <button onclick="closeShareModal()" class="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold rounded-lg transition">Đã Hiểu</button>
            </div>
        </div>
    </div>

    <script>
        let allScreenerData = [];
        let currentStockData = null;
        let flowChartInstance = null;
        let auditChartInstances = [];
        let simChartInstance = null;
        let matrixScatterInstance = null;
        let matrixHistInstance = null;
        let sectorChartInstance = null;

        function openShareModal() { document.getElementById('shareModal').classList.remove('hidden'); }
        function closeShareModal() { document.getElementById('shareModal').classList.add('hidden'); }

        // Theme management
        let isDarkMode = true;
        function toggleTheme() {
            isDarkMode = !isDarkMode;
            if (isDarkMode) {
                document.documentElement.classList.add('dark');
                document.getElementById('themeIcon').innerText = '☀️ Sáng';
            } else {
                document.documentElement.classList.remove('dark');
                document.getElementById('themeIcon').innerText = '🌙 Tối';
            }
            if (currentStockData) {
                loadStock(currentStockData.symbol);
            }
        }

        // Tab Switching
        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            
            document.getElementById(tabId).classList.remove('hidden');
            document.getElementById(`btn-${tabId}`).classList.add('active');
            
            if (tabId === 'tab-charts' && currentStockData) {
                Plotly.Plots.resize('plotlyDecileContainer');
            }
            if (tabId === 'tab-technical' && currentStockData) {
                Plotly.Plots.resize('plotlyTechContainer');
            }
            if (tabId === 'tab-sectors') {
                renderSectorBreadth();
            }
            if (tabId === 'tab-spikes') {
                renderSpikesRadar();
            }
            if (tabId === 'tab-matrix') {
                renderMatrixCharts();
            }
            if (tabId === 'tab-financials' && currentStockData) {
                loadFinancialsAsync(currentStockData.symbol);
            }
        }

        // Load Financials in Background asynchronously
        async function loadFinancialsAsync(symbol) {
            try {
                const res = await fetch(`/api/financials/${symbol}`);
                if (!res.ok) return;
                const fin = await res.json();
                renderFinancials(fin);
            } catch (err) {
                console.error(err);
            }
        }

        // Load Stock Data instantly
        async function loadStock(symbol) {
            symbol = symbol.trim().toUpperCase();
            if (!symbol) return;
            try {
                const themeParam = isDarkMode ? 'dark' : 'light';
                const res = await fetch(`/api/stock/${symbol}?theme=${themeParam}`);
                if (!res.ok) throw new Error('Không tìm thấy dữ liệu mã: ' + symbol);
                currentStockData = await res.json();
                const data = currentStockData;

                // Update Hero Banner
                document.getElementById('heroSymbol').innerText = data.symbol;
                document.getElementById('heroSector').innerText = data.sector;
                document.getElementById('heroSignal').innerText = `${data.signal.toUpperCase()} (DECILE ${data.latest_decile}/${data.n_deciles})`;
                
                const sigBadge = document.getElementById('heroSignal');
                if (data.signal === 'Tăng mạnh') {
                    sigBadge.className = "text-xs font-bold px-3 py-1 rounded-md uppercase tracking-wider bg-emerald-500/10 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border border-emerald-500/40";
                } else if (data.signal === 'Giảm mạnh') {
                    sigBadge.className = "text-xs font-bold px-3 py-1 rounded-md uppercase tracking-wider bg-red-500/10 dark:bg-red-500/20 text-red-600 dark:text-red-400 border border-red-500/40";
                } else {
                    sigBadge.className = "text-xs font-bold px-3 py-1 rounded-md uppercase tracking-wider bg-amber-500/10 dark:bg-amber-500/20 text-amber-600 dark:text-amber-400 border border-amber-500/40";
                }

                document.getElementById('heroScore').innerHTML = `Score: <b>${data.score.toFixed(3)}</b>`;
                document.getElementById('heroIC').innerHTML = `IC OOS: <b class="${data.ic_oos >= 0.15 ? 'text-emerald-600 dark:text-emerald-400' : ''}">${data.ic_oos.toFixed(3)}</b>`;
                document.getElementById('heroHit').innerHTML = `Hit: <b class="${data.hit_oos >= 60 ? 'text-emerald-600 dark:text-emerald-400' : ''}">${data.hit_oos.toFixed(1)}%</b>`;
                document.getElementById('heroParams').innerHTML = `a=${data.a} b=${data.b} c=${data.c} d=${data.d} α=${data.alpha.toFixed(1)}`;

                // Update Chart Title
                document.getElementById('decileChartTitle').innerText = `1. Biểu Đồ Giá & Phân Vùng Tín Hiệu OOS ${data.symbol} (Thang Đo RdYlGn Decile 1 → ${data.n_deciles})`;

                // Render Exact Plotly Decile Figure from Notebook (RdYlGn colorscale, Decile 1 to n_deciles)
                Plotly.react('plotlyDecileContainer', data.plotly_decile_fig.data, data.plotly_decile_fig.layout, {responsive: true, displayModeBar: false});

                // Render Technical Indicators Figure
                Plotly.react('plotlyTechContainer', data.plotly_tech_fig.data, data.plotly_tech_fig.layout, {responsive: true, displayModeBar: false});

                // Update Flow Cards
                let f_sum = data.flow.reduce((acc, x) => acc + x.foreign, 0);
                let p_sum = data.flow.reduce((acc, x) => acc + x.prop, 0);
                let inst_sum = data.flow.reduce((acc, x) => acc + x.local_inst, 0);
                let ind_sum = data.flow.reduce((acc, x) => acc + x.local_ind, 0);

                document.getElementById('flowForeign').innerText = `${f_sum >= 0 ? '+' : ''}${f_sum.toFixed(1)} Tỷ`;
                document.getElementById('flowProp').innerText = `${p_sum >= 0 ? '+' : ''}${p_sum.toFixed(1)} Tỷ`;
                document.getElementById('flowInst').innerText = `${inst_sum >= 0 ? '+' : ''}${inst_sum.toFixed(1)} Tỷ`;
                document.getElementById('flowInd').innerText = `${ind_sum >= 0 ? '+' : ''}${ind_sum.toFixed(1)} Tỷ`;

                renderFlowChart(data.flow);
                renderAuditCharts(data.audit);
                renderFeatureTable(data.top_features);
                renderAuditTable(data.audit.table);
                renderSimulation(data.simulation);

            } catch (err) {
                console.error(err);
            }
        }

        // Render Sector Breadth (Always Visible Bars with Average Quant Score)
        function renderSectorBreadth() {
            if (allScreenerData.length === 0) return;
            const sectorStats = {};
            allScreenerData.forEach(d => {
                const sec = d.sector || 'Khác / Sản xuất';
                if (!sectorStats[sec]) sectorStats[sec] = { count: 0, total_score: 0, top_ic_count: 0, symbols: [] };
                sectorStats[sec].count++;
                sectorStats[sec].total_score += d.score;
                sectorStats[sec].symbols.push(d);
                if (d.oos_ic >= 0.1) sectorStats[sec].top_ic_count++;
            });

            const labels = Object.keys(sectorStats).sort((a,b) => (sectorStats[b].total_score/sectorStats[b].count) - (sectorStats[a].total_score/sectorStats[a].count));
            const avgScores = labels.map(s => parseFloat((sectorStats[s].total_score / sectorStats[s].count).toFixed(3)));
            const barColors = avgScores.map(sc => sc >= 0.45 ? '#10B981' : (sc >= 0.35 ? '#00E5FF' : '#F59E0B'));

            const ctx = document.getElementById('sectorChart').getContext('2d');
            if (sectorChartInstance) sectorChartInstance.destroy();

            const gridColor = isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
            const tickColor = isDarkMode ? '#94A3B8' : '#64748B';

            sectorChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        { label: 'Điểm Sức Mạnh Định Lượng TB (Average Quant Score)', data: avgScores, backgroundColor: barColors, borderRadius: 4 }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top', labels: { color: tickColor, font: { family: 'Plus Jakarta Sans', size: 12 } } },
                        tooltip: {
                            callbacks: {
                                label: ctx => `Điểm TB: ${ctx.raw} (Số lượng: ${sectorStats[labels[ctx.dataIndex]].count} mã)`
                            }
                        }
                    },
                    scales: {
                        x: { grid: { color: gridColor }, ticks: { color: tickColor, font: { family: 'Plus Jakarta Sans', size: 11 } } },
                        y: { min: 0, max: 0.8, title: { display: true, text: 'Điểm Sức Mạnh (Score)', color: tickColor }, grid: { color: gridColor }, ticks: { color: tickColor } }
                    }
                }
            });

            const container = document.getElementById('sectorTableContainer');
            container.innerHTML = `
            <table class="w-full text-left border-collapse text-xs font-mono">
                <thead class="bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400 uppercase">
                    <tr>
                        <th class="p-3">Nhóm Ngành</th>
                        <th class="p-3 text-right">Số Lượng Mã</th>
                        <th class="p-3 text-right">Điểm Sức Mạnh TB (Score)</th>
                        <th class="p-3 text-right">% Mã IC OOS ≥ 0.1</th>
                        <th class="p-3">Mã Dẫn Đầu Ngành</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-200 dark:divide-slate-800">
                    ${labels.map(sec => {
                        const info = sectorStats[sec];
                        const topSym = info.symbols.sort((a,b)=>b.score-a.score)[0]?.symbol || '-';
                        const avgSc = (info.total_score / info.count).toFixed(3);
                        return `
                        <tr class="hover:bg-slate-100 dark:hover:bg-slate-800/80">
                            <td class="p-3 font-bold text-slate-900 dark:text-white">${sec}</td>
                            <td class="p-3 text-right text-slate-400">${info.count}</td>
                            <td class="p-3 text-right font-bold text-emerald-500">${avgSc}</td>
                            <td class="p-3 text-right font-semibold text-sky-400">${((info.top_ic_count / info.count) * 100).toFixed(1)}%</td>
                            <td class="p-3 font-bold text-emerald-400 cursor-pointer" onclick="document.getElementById('symbolInput').value='${topSym}'; loadStock('${topSym}'); switchTab('tab-charts');">${topSym}</td>
                        </tr>`;
                    }).join('')}
                </tbody>
            </table>`;
        }

        // Render Spikes Radar
        function renderSpikesRadar() {
            if (allScreenerData.length === 0) return;
            const topSpikes = allScreenerData.slice(0, 15);
            const container = document.getElementById('spikesTableContainer');
            container.innerHTML = `
            <table class="w-full text-left border-collapse text-xs font-mono">
                <thead class="bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400 uppercase">
                    <tr>
                        <th class="p-3">Mã CP</th>
                        <th class="p-3">Ngành</th>
                        <th class="p-3 text-right">Score</th>
                        <th class="p-3 text-right">IC OOS</th>
                        <th class="p-3 text-center">Tín Hiệu Đột Biến Dòng Tiền</th>
                        <th class="p-3 text-center">Hành Động Gợi Ý</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-200 dark:divide-slate-800">
                    ${topSpikes.map(r => `
                        <tr onclick="document.getElementById('symbolInput').value='${r.symbol}'; loadStock('${r.symbol}'); switchTab('tab-charts');" class="hover:bg-slate-100 dark:hover:bg-slate-800/80 cursor-pointer transition">
                            <td class="p-3 font-bold text-slate-900 dark:text-white">${r.symbol}</td>
                            <td class="p-3 text-slate-400">${r.sector}</td>
                            <td class="p-3 text-right font-bold text-emerald-400">${r.score.toFixed(3)}</td>
                            <td class="p-3 text-right text-sky-400">${r.oos_ic.toFixed(3)}</td>
                            <td class="p-3 text-center"><span class="px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">🔥 DÒNG TIỀN GOM MẠNH</span></td>
                            <td class="p-3 text-center"><span class="px-2 py-0.5 rounded text-[11px] font-bold bg-sky-500/10 text-sky-400">MỞ VỊ THẾ MUA</span></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>`;
        }

        // Render Real Financial Statements & Ratios
        function renderFinancials(fin) {
            const ratioContainer = document.getElementById('ratiosTableContainer');
            if (fin && fin.ratios && fin.ratios.headers && fin.ratios.headers.length > 0) {
                ratioContainer.innerHTML = `
                <table class="w-full text-left border-collapse text-xs font-mono">
                    <thead class="bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400 uppercase">
                        <tr>
                            ${fin.ratios.headers.map((h, i) => `<th class="p-3 ${i > 0 ? 'text-right' : ''}">${h}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-200 dark:divide-slate-800">
                        ${fin.ratios.rows.map(row => `
                            <tr class="hover:bg-slate-100 dark:hover:bg-slate-800/80">
                                ${row.map((cell, i) => `<td class="p-3 ${i === 0 ? 'font-bold text-slate-900 dark:text-white' : 'text-right font-semibold text-emerald-600 dark:text-emerald-400'}">${cell}</td>`).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>`;
            } else {
                ratioContainer.innerHTML = `<div class="p-6 text-center text-slate-400 font-mono">Không có dữ liệu chỉ số tài chính từ Vietcap VCI...</div>`;
            }

            const incContainer = document.getElementById('financialTableContainer');
            if (fin && fin.income_statement && fin.income_statement.headers && fin.income_statement.headers.length > 0) {
                incContainer.innerHTML = `
                <table class="w-full text-left border-collapse text-xs font-mono">
                    <thead class="bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400 uppercase">
                        <tr>
                            ${fin.income_statement.headers.map((h, i) => `<th class="p-3 ${i > 0 ? 'text-right' : ''}">${h}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-200 dark:divide-slate-800">
                        ${fin.income_statement.rows.map(row => `
                            <tr class="hover:bg-slate-100 dark:hover:bg-slate-800/80">
                                ${row.map((cell, i) => `<td class="p-3 ${i === 0 ? 'font-bold text-slate-900 dark:text-white' : 'text-right font-semibold text-sky-600 dark:text-cyan-400'}">${cell}</td>`).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>`;
            } else {
                incContainer.innerHTML = `<div class="p-6 text-center text-slate-400 font-mono">Không có dữ liệu báo cáo KQKD từ Vietcap VCI...</div>`;
            }
        }

        // Render Factor Attribution Table
        function renderFeatureTable(features) {
            const container = document.getElementById('featuresTableContainer');
            container.innerHTML = `
            <table class="w-full text-left border-collapse text-xs font-mono">
                <thead class="bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400 uppercase">
                    <tr>
                        <th class="p-3">Tên Đặc Trưng Lượng Hóa</th>
                        <th class="p-3 text-right">Trọng Số Hồi Quy (Ridge Beta)</th>
                        <th class="p-3 text-center">Chiều Tác Động</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-200 dark:divide-slate-800">
                    ${features.map(f => `
                        <tr class="hover:bg-slate-100 dark:hover:bg-slate-800/80">
                            <td class="p-3 font-bold text-slate-900 dark:text-white">${f.feature}</td>
                            <td class="p-3 text-right font-semibold ${f.coef > 0 ? 'text-emerald-500' : 'text-red-500'}">${f.coef}</td>
                            <td class="p-3 text-center"><span class="px-2 py-0.5 rounded text-[11px] font-bold ${f.coef > 0 ? 'bg-emerald-500/10 text-emerald-500' : 'bg-red-500/10 text-red-500'}">${f.impact}</span></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>`;
        }

        // Render Monotonicity Decile Table
        function renderAuditTable(rows) {
            const container = document.getElementById('auditTableContainer');
            container.innerHTML = `
            <table class="w-full text-left border-collapse text-xs font-mono">
                <thead class="bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400 uppercase">
                    <tr>
                        <th class="p-3">Phân Vùng</th>
                        <th class="p-3 text-right">Mean Ret %</th>
                        <th class="p-3 text-right">Hit Rate %</th>
                        <th class="p-3 text-right">Độ Lệch Chuẩn</th>
                        <th class="p-3 text-right">Số Mẫu (Count)</th>
                        <th class="p-3 text-center">Tín Hiệu Định Lượng</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-200 dark:divide-slate-800">
                    ${rows.map(r => `
                        <tr class="hover:bg-slate-100 dark:hover:bg-slate-800/80">
                            <td class="p-3 font-bold text-slate-900 dark:text-white">${r.decile}</td>
                            <td class="p-3 text-right font-semibold ${r.mean_ret.startsWith('-') ? 'text-red-500' : 'text-emerald-500'}">${r.mean_ret}</td>
                            <td class="p-3 text-right font-semibold text-sky-500">${r.hit_rate}</td>
                            <td class="p-3 text-right text-slate-400">${r.std}</td>
                            <td class="p-3 text-right text-slate-400">${r.count}</td>
                            <td class="p-3 text-center"><span class="px-2 py-0.5 rounded text-[11px] font-bold ${r.signal === 'TĂNG MẠNH' ? 'bg-emerald-500/10 text-emerald-500' : (r.signal === 'GIẢM MẠNH' ? 'bg-red-500/10 text-red-500' : 'bg-amber-500/10 text-amber-500')}">${r.signal}</span></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>`;
        }

        // Render Simulation Chart
        function renderSimulation(sim) {
            const ctx = document.getElementById('simChart').getContext('2d');
            if (simChartInstance) simChartInstance.destroy();

            document.getElementById('simAlphaBadge').innerText = `Alpha Vượt Trội: ${sim.total_alpha >= 0 ? '+' : ''}${sim.total_alpha}%`;
            document.getElementById('simAlphaBadge').className = sim.total_alpha >= 0 ? "text-xs font-mono font-bold px-3 py-1.5 rounded-lg bg-emerald-500/10 text-emerald-500 border border-emerald-500/30" : "text-xs font-mono font-bold px-3 py-1.5 rounded-lg bg-red-500/10 text-red-500 border border-red-500/30";

            const gridColor = isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
            const tickColor = isDarkMode ? '#94A3B8' : '#64748B';

            simChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: sim.dates,
                    datasets: [
                        { label: 'Chiến Lược Quant (Top Decile)', data: sim.strategy, borderColor: '#00E676', borderWidth: 2.4, tension: 0.2, pointRadius: 0, fill: false },
                        { label: 'Nắm Giữ Thụ Động (Buy & Hold)', data: sim.buy_hold, borderColor: '#94A3B8', borderWidth: 1.8, borderDash: [4, 4], tension: 0.2, pointRadius: 0, fill: false },
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top', labels: { color: tickColor, font: { family: 'Plus Jakarta Sans', size: 12 } } },
                        tooltip: { mode: 'index', intersect: false }
                    },
                    scales: {
                        x: { grid: { color: gridColor }, ticks: { color: tickColor, maxTicksLimit: 10 } },
                        y: { title: { display: true, text: 'Lợi Nhuận Tích Lũy (%)', color: tickColor }, grid: { color: gridColor }, ticks: { color: tickColor } }
                    }
                }
            });
        }

        // Render Flow Chart
        function renderFlowChart(flowData) {
            const ctx = document.getElementById('flowChart').getContext('2d');
            if (flowChartInstance) flowChartInstance.destroy();

            const labels = flowData.map(x => x.time);
            let cumF = 0, cumP = 0, cumInst = 0, cumInd = 0;
            const seriesF = [], seriesP = [], seriesInst = [], seriesInd = [];
            flowData.forEach(x => {
                cumF += x.foreign; seriesF.push(cumF);
                cumP += x.prop; seriesP.push(cumP);
                cumInst += x.local_inst; seriesInst.push(cumInst);
                cumInd += x.local_ind; seriesInd.push(cumInd);
            });

            const gridColor = isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
            const tickColor = isDarkMode ? '#94A3B8' : '#64748B';

            flowChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        { label: 'Khối Ngoại', data: seriesF, borderColor: '#00E5FF', borderWidth: 2.2, tension: 0.2, pointRadius: 0 },
                        { label: 'Tự Doanh', data: seriesP, borderColor: '#10B981', borderWidth: 2.2, tension: 0.2, pointRadius: 0 },
                        { label: 'Tổ Chức Nội', data: seriesInst, borderColor: '#A855F7', borderWidth: 2.2, tension: 0.2, pointRadius: 0 },
                        { label: 'Cá Nhân Nội', data: seriesInd, borderColor: '#F59E0B', borderWidth: 2.2, tension: 0.2, pointRadius: 0 },
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top', labels: { color: tickColor, font: { family: 'Plus Jakarta Sans', size: 12 } } },
                        tooltip: { mode: 'index', intersect: false }
                    },
                    scales: {
                        x: { grid: { color: gridColor }, ticks: { color: tickColor, maxTicksLimit: 10 } },
                        y: { grid: { color: gridColor }, ticks: { color: tickColor } }
                    }
                }
            });
        }

        // Render 4 Model Audit Charts
        function renderAuditCharts(audit) {
            auditChartInstances.forEach(c => c.destroy());
            auditChartInstances = [];
            const gridColor = isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
            const tickColor = isDarkMode ? '#94A3B8' : '#64748B';

            // Audit 1: Mean Ret %
            const ctx1 = document.getElementById('auditChart1').getContext('2d');
            const colors1 = audit.mean_ret.map(v => v >= 0 ? '#10B981' : '#EF4444');
            auditChartInstances.push(new Chart(ctx1, {
                type: 'bar',
                data: { labels: audit.deciles, datasets: [{ label: 'Mean Ret %', data: audit.mean_ret, backgroundColor: colors1 }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: gridColor }, ticks: { color: tickColor } }, y: { grid: { color: gridColor }, ticks: { color: tickColor } } } }
            }));

            // Audit 2: Hit Rate %
            const ctx2 = document.getElementById('auditChart2').getContext('2d');
            auditChartInstances.push(new Chart(ctx2, {
                type: 'line',
                data: { labels: audit.deciles, datasets: [{ label: 'Hit Rate %', data: audit.hit_rate, borderColor: '#38BDF8', borderWidth: 2, pointRadius: 5, backgroundColor: '#38BDF8' }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: gridColor }, ticks: { color: tickColor } }, y: { grid: { color: gridColor }, ticks: { color: tickColor } } } }
            }));

            // Audit 3: Train vs OOS Scatter
            const ctx3 = document.getElementById('auditChart3').getContext('2d');
            const trainPoints = audit.train_scores.map((s, i) => ({ x: s, y: audit.train_returns[i] }));
            const oosPoints = audit.oos_scores.map((s, i) => ({ x: s, y: audit.oos_returns[i] }));
            auditChartInstances.push(new Chart(ctx3, {
                type: 'scatter',
                data: {
                    datasets: [
                        { label: 'Train', data: trainPoints, backgroundColor: 'rgba(76, 114, 176, 0.6)' },
                        { label: 'OOS', data: oosPoints, backgroundColor: 'rgba(221, 132, 82, 0.8)' },
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { x: { grid: { color: gridColor }, ticks: { color: tickColor } }, y: { grid: { color: gridColor }, ticks: { color: tickColor } } } }
            }));

            // Audit 4: Pred vs Actual
            const ctx4 = document.getElementById('auditChart4').getContext('2d');
            auditChartInstances.push(new Chart(ctx4, {
                type: 'bar',
                data: {
                    labels: audit.deciles,
                    datasets: [
                        { label: 'Dự đoán %', data: audit.avg_pred, backgroundColor: '#818CF8' },
                        { label: 'Thực tế %', data: audit.mean_ret, backgroundColor: '#34D399' },
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { x: { grid: { color: gridColor }, ticks: { color: tickColor } }, y: { grid: { color: gridColor }, ticks: { color: tickColor } } } }
            }));
        }

        // Render Matrix Charts
        function renderMatrixCharts() {
            if (allScreenerData.length === 0) return;
            const gridColor = isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
            const tickColor = isDarkMode ? '#94A3B8' : '#64748B';

            // Scatter IC vs Hit
            const ctxScat = document.getElementById('matrixScatterChart').getContext('2d');
            if (matrixScatterInstance) matrixScatterInstance.destroy();
            const points = allScreenerData.map(d => ({ x: d.oos_ic, y: d.hit_rate_oos, symbol: d.symbol }));

            matrixScatterInstance = new Chart(ctxScat, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Cổ Phiếu',
                        data: points,
                        backgroundColor: '#38BDF8',
                        borderColor: '#0284C7',
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: ctx => `${points[ctx.dataIndex].symbol} | IC: ${ctx.raw.x} | Hit: ${ctx.raw.y}%`
                            }
                        }
                    },
                    scales: {
                        x: { title: { display: true, text: 'IC OOS', color: tickColor }, grid: { color: gridColor }, ticks: { color: tickColor } },
                        y: { title: { display: true, text: 'Hit Rate OOS (%)', color: tickColor }, grid: { color: gridColor }, ticks: { color: tickColor } }
                    }
                }
            });

            // Histogram IC
            const ctxHist = document.getElementById('matrixHistChart').getContext('2d');
            if (matrixHistInstance) matrixHistInstance.destroy();

            const bins = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8];
            const counts = [0, 0, 0, 0, 0];
            allScreenerData.forEach(d => {
                const ic = d.oos_ic;
                if (ic < 0.0) counts[0]++;
                else if (ic < 0.2) counts[1]++;
                else if (ic < 0.4) counts[2]++;
                else if (ic < 0.6) counts[3]++;
                else counts[4]++;
            });

            matrixHistInstance = new Chart(ctxHist, {
                type: 'bar',
                data: {
                    labels: ['< 0.0', '0.0 - 0.2', '0.2 - 0.4', '0.4 - 0.6', '≥ 0.6'],
                    datasets: [{
                        label: 'Số Lượng Mã',
                        data: counts,
                        backgroundColor: '#10B981',
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: gridColor }, ticks: { color: tickColor } },
                        y: { grid: { color: gridColor }, ticks: { color: tickColor } }
                    }
                }
            });
        }

        // Screener Full Table
        function renderScreenerFullRows(dataList) {
            const tbody = document.getElementById('screenerFullTableBody');
            tbody.innerHTML = dataList.map(r => {
                const icCol = r.oos_ic >= 0.15 ? 'text-emerald-600 dark:text-emerald-400 font-bold' : (r.oos_ic < 0 ? 'text-red-600 dark:text-red-400' : '');
                const hitCol = r.hit_rate_oos >= 60 ? 'text-emerald-600 dark:text-emerald-400 font-bold' : (r.hit_rate_oos < 50 ? 'text-red-600 dark:text-red-400' : '');
                return `
                <tr onclick="document.getElementById('symbolInput').value='${r.symbol}'; loadStock('${r.symbol}'); switchTab('tab-charts');" class="hover:bg-slate-100 dark:hover:bg-slate-800/80 cursor-pointer transition">
                    <td class="p-3 font-bold text-slate-900 dark:text-white">${r.symbol}</td>
                    <td class="p-3 text-slate-400">${r.sector}</td>
                    <td class="p-3 text-right font-semibold text-slate-800 dark:text-slate-200">${r.score.toFixed(3)}</td>
                    <td class="p-3 text-right ${icCol}">${r.oos_ic.toFixed(3)}</td>
                    <td class="p-3 text-right ${hitCol}">${r.hit_rate_oos.toFixed(1)}%</td>
                    <td class="p-3 text-right text-slate-500 dark:text-slate-400">${r.train_ic.toFixed(3)}</td>
                    <td class="p-3 text-right text-slate-500 dark:text-slate-400">${r.chosen_alpha.toFixed(1)}</td>
                    <td class="p-3 text-right text-slate-500 dark:text-slate-400">${r.a}</td>
                    <td class="p-3 text-right text-slate-500 dark:text-slate-400">${r.b}</td>
                    <td class="p-3 text-right text-slate-500 dark:text-slate-400">${r.c}</td>
                    <td class="p-3 text-right text-slate-500 dark:text-slate-400">${r.d}</td>
                </tr>`;
            }).join('');
        }

        function filterScreenerFullTable() {
            const query = document.getElementById('screenerSearchTab').value.trim().toUpperCase();
            const filtered = allScreenerData.filter(x => x.symbol.includes(query) || (x.sector && x.sector.toUpperCase().includes(query)));
            renderScreenerFullRows(filtered);
        }

        function filterScreenerByBadge(type) {
            if (type === 'all') renderScreenerFullRows(allScreenerData);
            else if (type === 'top_ic') renderScreenerFullRows(allScreenerData.filter(x => x.oos_ic >= 0.15));
            else if (type === 'top_hit') renderScreenerFullRows(allScreenerData.filter(x => x.hit_rate_oos >= 60));
        }

        async function initApp() {
            try {
                const res = await fetch('/api/screener');
                allScreenerData = await res.json();
                allScreenerData.sort((a, b) => b.score - a.score);
                renderScreenerFullRows(allScreenerData);

                if (allScreenerData.length > 0) {
                    loadStock('SSI');
                }
            } catch (err) {
                console.error(err);
            }
        }

        window.onload = initApp;
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    print("="*60)
    print("🚀 VN-Quant Enterprise Terminal đang khởi động:")
    print("👉 Mở trình duyệt tại: http://127.0.0.1:8000")
    print("="*60)
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
