"""
=============================================================================
VN-QUANT ALPHA TERMINAL | QUANTITATIVE INVESTOR FLOW ENGINE
=============================================================================
Thiet ke chuan Quy dau tu dinh luong quoc te.
"""

import os
import glob
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="VN-Quant Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark Theme"

is_dark = "Dark" in st.session_state.theme_mode

if is_dark:
    c_bg = "#0B0F17"
    c_card = "#131B2A"
    c_card_hover = "#1B2538"
    c_border = "rgba(255, 255, 255, 0.12)"
    c_text_title = "#FFFFFF"
    c_text_body = "#F1F5F9"
    c_text_sub = "#CBD5E1"
    c_text_muted = "#94A3B8"
    c_table_header = "#1E293B"
    c_table_row = "#131B2A"
    c_table_row_alt = "#162032"
    plotly_theme = "plotly_dark"
    p_bg = "#131B2A"
    p_plot_bg = "#0B0F17"
    p_grid = "rgba(255, 255, 255, 0.08)"
    p_line_price = "#38BDF8"
else:
    c_bg = "#F8FAFC"
    c_card = "#FFFFFF"
    c_card_hover = "#F1F5F9"
    c_border = "#E2E8F0"
    c_text_title = "#0F172A"
    c_text_body = "#1E293B"
    c_text_sub = "#475569"
    c_text_muted = "#64748B"
    c_table_header = "#F1F5F9"
    c_table_row = "#FFFFFF"
    c_table_row_alt = "#F8FAFC"
    plotly_theme = "plotly_white"
    p_bg = "#FFFFFF"
    p_plot_bg = "#F8FAFC"
    p_grid = "#E2E8F0"
    p_line_price = "#0284C7"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {{
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        background-color: {c_bg} !important;
        color: {c_text_body} !important;
    }}
    
    section[data-testid="stSidebar"] {{
        display: none !important;
    }}
    header[data-testid="stHeader"] {{
        display: none !important;
    }}
    
    .block-container {{
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }}
    
    /* Stat Card */
    .stat-card-pro {{
        background: {c_card};
        border: 1px solid {c_border};
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        height: 100%;
    }}
    .stat-label-pro {{
        font-size: 0.78rem;
        font-weight: 700;
        color: {c_text_muted};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }}
    .stat-num-pro {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.7rem;
        font-weight: 700;
        color: {c_text_title};
    }}
    .stat-desc-pro {{
        font-size: 0.78rem;
        color: {c_text_sub};
        margin-top: 2px;
    }}
    
    /* Hero Ticker Card */
    .ticker-hero-pro {{
        background: {c_card};
        border: 1px solid {c_border};
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
    }}
    .ticker-title-pro {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 800;
        color: {c_text_title};
    }}
    .pill-tag-pro {{
        font-family: 'JetBrains Mono', monospace;
        background: {c_table_header};
        border: 1px solid {c_border};
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.85rem;
        color: {c_text_body};
        margin-right: 6px;
    }}
    
    /* Crystal-Clear Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px !important;
        border-bottom: 1px solid {c_border} !important;
        padding-bottom: 4px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 46px !important;
        color: {c_text_sub} !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0 16px !important;
        border-radius: 8px 8px 0 0 !important;
        background: transparent !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {c_text_title} !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: #10B981 !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #10B981 !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {c_text_title} !important;
        font-weight: 700 !important;
    }}
    
    /* Pure HTML Table */
    .fin-table {{
        width: 100%;
        border-collapse: collapse;
        border: 1px solid {c_border};
        border-radius: 8px;
        overflow: hidden;
        font-size: 0.88rem;
        margin-top: 10px;
    }}
    .fin-table th {{
        background-color: {c_table_header};
        color: {c_text_muted};
        font-weight: 700;
        padding: 12px 14px;
        text-align: right;
        border-bottom: 1px solid {c_border};
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }}
    .fin-table th:first-child, .fin-table td:first-child {{
        text-align: left;
    }}
    .fin-table td {{
        padding: 10px 14px;
        border-bottom: 1px solid {c_border};
        color: {c_text_body};
        font-family: 'JetBrains Mono', monospace;
        text-align: right;
        background-color: {c_table_row};
    }}
    .fin-table tr:nth-child(even) td {{
        background-color: {c_table_row_alt};
    }}
    .fin-table tr:hover td {{
        background-color: {c_card_hover};
    }}
</style>
""", unsafe_allow_html=True)

# 2. CONSTANTS
LOOKBACK_DAYS = 1000
OOS_DAYS = 60
investor_groups = ["foreign", "prop", "local_inst", "local_ind"]

DECILE_STRONG_UP = {8, 9}
DECILE_NEUTRAL = {2, 3, 4, 5, 6, 7}
DECILE_STRONG_DOWN = {0, 1}

def classify_decile(dec):
    if dec is None or (isinstance(dec, float) and np.isnan(dec)):
        return "Chua phan loai"
    dec = int(dec)
    if dec in DECILE_STRONG_UP:
        return "Tang manh"
    if dec in DECILE_STRONG_DOWN:
        return "Giam manh"
    return "Trung tinh"

# 3. DATA LOADERS
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
    path1 = os.path.join(data_dir, f"Kafi_Phân_Loại_Nhà_Đầu_Tư_{symbol}.csv")
    if os.path.exists(path1):
        return path1
    path2 = os.path.join(data_dir, f"Kafi_Giá_{symbol}.csv")
    if os.path.exists(path2):
        return path2
    return None

def load_symbol_data(symbol: str, data_dir: str):
    file_path = get_file_path(symbol, data_dir)
    if file_path is None:
        return None
    try:
        df = load_and_map_excel(file_path)
        if df is None or len(df) < 150:
            return None
        return df
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_raw_overview(result_path: str):
    if not os.path.exists(result_path):
        return pd.DataFrame()
    df = pd.read_csv(result_path)
    rename_map = {
        'symbol': 'Mã',
        'oos_ic': 'IC OOS',
        'hit_rate_oos': 'Hit OOS (%)',
        'train_ic': 'IC Train',
        'hit_rate_train': 'Hit Train (%)',
        'train_oos_gap': 'Gap Train-OOS',
        'n_signals': 'N Signals OOS',
        'chosen_alpha': 'Alpha',
        'score': 'Score',
    }
    df = df.rename(columns=rename_map)
    if 'Hit OOS (%)' in df.columns:
        df['Hit OOS (%)'] = (df['Hit OOS (%)'] * 100).round(1)
    if 'Hit Train (%)' in df.columns:
        df['Hit Train (%)'] = (df['Hit Train (%)'] * 100).round(1)
    if 'IC OOS' in df.columns:
        df['IC OOS'] = df['IC OOS'].round(3)
    if 'IC Train' in df.columns:
        df['IC Train'] = df['IC Train'].round(3)
    if 'Score' in df.columns:
        df['Score'] = df['Score'].round(3)
    if 'Alpha' in df.columns:
        df['Alpha'] = df['Alpha'].round(2)
    return df

# 4. FEATURE ENGINEERING & RIDGE ENGINE
def build_features(df_vni_raw, a, b, c, d, investor_groups):
    df = df_vni_raw.copy()
    keep_cols = [
        'open', 'high', 'low', 'close', 'trading_value',
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

@st.cache_data(show_spinner=False)
def analyze_single_symbol_cached(symbol, a, b, c, d, alpha, data_dir, lookback_days=LOOKBACK_DAYS, oos_days=OOS_DAYS):
    y_col = "target_ret_t10"
    df_raw = load_symbol_data(symbol, data_dir)
    if df_raw is None:
        return None

    try:
        model_data, all_feature_cols = build_features(df_raw, a, b, c, d, investor_groups)
    except Exception:
        return None
        
    X_cols = all_feature_cols
    total_needed = lookback_days + oos_days
    if len(model_data) >= total_needed:
        train_data = model_data.iloc[-total_needed:-oos_days].copy()
        oos_data = model_data.iloc[-oos_days:].copy()
    else:
        train_data = model_data.iloc[:-oos_days].copy() if len(model_data) > oos_days else model_data.copy()
        oos_data = model_data.iloc[-oos_days:].copy() if len(model_data) > oos_days else pd.DataFrame()

    if len(train_data) < 30 or len(X_cols) == 0:
        return None

    X_train = train_data[X_cols]
    y_train = train_data[y_col]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train_scaled, y_train)

    train_data = train_data.copy()
    train_data["pred_score"] = model.predict(X_train_scaled)
    train_data["actual_ret"] = y_train

    ic_train, _ = spearmanr(train_data["pred_score"], train_data["actual_ret"])
    hit_train = (train_data.loc[train_data["pred_score"] > 0, "actual_ret"] > 0).mean()

    ic_oos = hit_oos = n_sig_oos = np.nan
    if len(oos_data) > 0:
        X_oos = oos_data[X_cols]
        X_oos_scaled = scaler.transform(X_oos)
        oos_data = oos_data.copy()
        oos_data["pred_score"] = model.predict(X_oos_scaled)
        oos_data["actual_ret"] = oos_data[y_col]

        ic_oos, _ = spearmanr(oos_data["pred_score"], oos_data["actual_ret"])
        n_sig_oos = int((oos_data["pred_score"] > 0).sum())
        hit_oos = (oos_data.loc[oos_data["pred_score"] > 0, "actual_ret"] > 0).mean() if n_sig_oos > 0 else np.nan

    eval_data = pd.concat([
        train_data[["pred_score", y_col]].assign(is_oos=False),
        (oos_data[["pred_score", y_col]].assign(is_oos=True) if len(oos_data) > 0 else pd.DataFrame())
    ], axis=0).dropna()
    
    eval_data["score_decile"] = pd.qcut(eval_data["pred_score"], 10, labels=False, duplicates="drop")
    analysis = eval_data.groupby("score_decile")[y_col].agg(["mean", "std", "count"])
    analysis["hit_rate"] = eval_data.groupby("score_decile")[y_col].apply(lambda x: (x > 0).mean())
    analysis["avg_pred"] = eval_data.groupby("score_decile")["pred_score"].mean()
    rank_ic = eval_data["pred_score"].corr(eval_data[y_col], method="spearman")

    latest_signal, latest_decile = "Không xác định", np.nan
    oos_price_data = pd.DataFrame()
    if len(oos_data) > 0:
        train_scores = train_data["pred_score"]
        _, bin_edges = pd.qcut(train_scores, q=10, duplicates="drop", retbins=True)
        bin_edges = bin_edges.copy()
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        oos_price_data = oos_data.copy()
        oos_price_data["decile"] = pd.cut(
            oos_price_data["pred_score"], bins=bin_edges, labels=False, include_lowest=True
        )

        if "time" in df_raw.columns:
            oos_price_data["date"] = pd.to_datetime(df_raw.loc[oos_price_data.index, "time"])
        else:
            oos_price_data["date"] = pd.to_datetime(oos_price_data.index)

        latest_decile = oos_price_data["decile"].iloc[-1]
        latest_signal = classify_decile(latest_decile)

    return {
        "symbol": symbol, "a": a, "b": b, "c": c, "d": d, "alpha": alpha,
        "df_raw": df_raw,
        "train_data": train_data, "oos_data": oos_data, "eval_data": eval_data,
        "analysis": analysis, "rank_ic": rank_ic,
        "ic_train": ic_train, "hit_train": hit_train,
        "ic_oos": ic_oos, "hit_oos": hit_oos, "n_signals_oos": n_sig_oos,
        "oos_price_data": oos_price_data,
        "latest_decile": latest_decile, "latest_signal": latest_signal,
    }

# 5. CHARTS
def make_clean_line_decile_fig(res):
    odata = res["oos_price_data"]
    if odata.empty:
        fig = go.Figure()
        fig.update_layout(title="Không đủ dữ liệu OOS", template=plotly_theme)
        return fig

    n_deciles = int(odata["decile"].max()) + 1 if not np.isnan(odata["decile"].max()) else 10

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Scatter(
            x=odata["date"], y=odata["close"], mode="lines",
            line=dict(color=p_line_price, width=2.2), name="Giá đóng cửa (Close)",
            hoverinfo="skip"
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=odata["date"], y=odata["close"], mode="markers",
            marker=dict(
                size=9, color=odata["decile"], colorscale="RdYlGn",
                cmin=0, cmax=n_deciles - 1, showscale=True,
                colorbar=dict(
                    title=dict(text=f"Decile (0→{n_deciles-1})", font=dict(size=11, color=c_text_muted)),
                    len=0.75, y=0.62, thickness=14,
                    tickfont=dict(size=10, color=c_text_sub),
                ),
                line=dict(width=1, color="#0B0F17" if is_dark else "#FFFFFF"),
            ),
            name="Decile Signal",
            hovertemplate="Ngày: <b>%{x|%d/%m/%Y}</b><br>Giá: <b>%{y:,.0f} VND</b><br>Decile: <b>%{marker.color}</b><extra></extra>"
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=odata["date"], y=odata["decile"], showlegend=False,
            marker=dict(color=odata["decile"], colorscale="RdYlGn", cmin=0, cmax=n_deciles - 1),
            hovertemplate="Ngày: %{x|%d/%m/%Y}<br>Decile: <b>%{y}</b><extra></extra>"
        ),
        row=2, col=1,
    )

    y_min, y_max = float(odata["close"].min()), float(odata["close"].max())
    padding = (y_max - y_min) * 0.12 if y_max > y_min else y_max * 0.05

    fig.update_layout(
        height=500,
        template=plotly_theme,
        paper_bgcolor=p_bg,
        plot_bgcolor=p_plot_bg,
        margin=dict(l=15, r=15, t=15, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Giá (VND)", range=[y_min - padding, y_max + padding], row=1, col=1, gridcolor=p_grid)
    fig.update_yaxes(title_text="Decile", range=[-0.5, n_deciles - 0.5], row=2, col=1, gridcolor=p_grid)
    fig.update_xaxes(gridcolor=p_grid)
    return fig

def make_model_audit_fig(res):
    analysis = res["analysis"]
    eval_data = res["eval_data"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "1. Lợi Nhuận Kỳ Vọng T+10 Theo Decile", "2. Tỷ Lệ Thắng (Hit Rate %)",
            "3. Điểm Dự Báo vs Lợi Nhuận Thực Tế", "4. So Sánh Dự Báo vs Thực Tế",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.08
    )

    colors = ["#EF4444" if v < 0 else "#10B981" for v in analysis["mean"]]
    fig.add_trace(
        go.Bar(
            x=[f"D{i}" for i in analysis.index],
            y=analysis["mean"] * 100,
            marker_color=colors,
            name="Mean Ret %",
            hovertemplate="%{x}: <b>%{y:.2f}%</b><extra></extra>"
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color=c_text_muted, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=[f"D{i}" for i in analysis.index],
            y=analysis["hit_rate"] * 100,
            mode="lines+markers",
            marker=dict(size=8, color="#38BDF8"),
            line=dict(width=2.2, color="#38BDF8"),
            name="Hit Rate %",
            hovertemplate="%{x}: <b>%{y:.1f}%</b><extra></extra>"
        ),
        row=1, col=2,
    )
    fig.add_hline(y=50, line_dash="dot", line_color="#EF4444", row=1, col=2)

    for flag, name, color in [(False, "Train", "#4C72B0"), (True, "OOS", "#DD8452")]:
        sub = eval_data[eval_data["is_oos"] == flag]
        if len(sub) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sub["pred_score"], y=sub["target_ret_t10"] * 100,
                    mode="markers", name=name,
                    marker=dict(color=color, opacity=0.5, size=5),
                    hovertemplate="Score: %{x:.3f}<br>Return: %{y:.1f}%<extra></extra>"
                ),
                row=2, col=1,
            )

    fig.add_trace(
        go.Bar(x=[f"D{i}" for i in analysis.index], y=analysis["avg_pred"] * 100, name="Dự đoán %", marker_color="#818CF8"),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(x=[f"D{i}" for i in analysis.index], y=analysis["mean"] * 100, name="Thực tế %", marker_color="#34D399"),
        row=2, col=2
    )

    fig.update_layout(
        height=580,
        barmode="group",
        template=plotly_theme,
        paper_bgcolor=p_bg,
        plot_bgcolor=p_plot_bg,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        margin=dict(l=15, r=15, t=30, b=15)
    )
    fig.update_xaxes(gridcolor=p_grid)
    fig.update_yaxes(gridcolor=p_grid)
    return fig

def make_powerbi_flow_fig(df_raw, symbol, selected_group_bar="foreign"):
    if df_raw is None or len(df_raw) == 0:
        return go.Figure()
    
    df_sub = df_raw.tail(60).copy()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.60, 0.40], vertical_spacing=0.05,
    )
    
    groups_config = {
        "foreign_net_val": ("Khối Ngoại", "#00E5FF"),
        "prop_net_val": ("Tự Doanh", "#00E676"),
        "local_inst_net_val": ("Tổ Chức Nội", "#D500F9"),
        "local_ind_net_val": ("Cá Nhân Nội", "#FFD600"),
    }
    
    for col, (label, color) in groups_config.items():
        if col in df_sub.columns:
            cum_val = df_sub[col].cumsum() / 1e9 if df_sub[col].abs().max() > 1e6 else df_sub[col].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=df_sub["time"] if "time" in df_sub.columns else df_sub.index,
                    y=cum_val,
                    mode="lines+markers",
                    marker=dict(size=4),
                    line=dict(width=2.5, color=color),
                    name=f"Tích lũy: {label}",
                    hovertemplate=f"<b>{label}</b>: %{{y:,.1f}} Tỷ<extra></extra>"
                ),
                row=1, col=1
            )
            
    col_bar = f"{selected_group_bar}_net_val"
    group_name = groups_config.get(col_bar, ("Nhóm chọn", "#38BDF8"))[0]
    
    if col_bar in df_sub.columns:
        vals = df_sub[col_bar] / 1e9 if df_sub[col_bar].abs().max() > 1e6 else df_sub[col_bar]
        bar_colors = ["#10B981" if v >= 0 else "#EF4444" for v in vals]
        
        fig.add_trace(
            go.Bar(
                x=df_sub["time"] if "time" in df_sub.columns else df_sub.index,
                y=vals,
                marker_color=bar_colors,
                name=f"{group_name} Ròng Ngày (Xanh: Mua / Đỏ: Bán)",
                hovertemplate="Ngày: %{x|%d/%m/%Y}<br>Giá trị: <b>%{y:+,.2f} Tỷ</b><extra></extra>"
            ),
            row=2, col=1
        )
        fig.add_hline(y=0, line_color=c_text_muted, line_width=1, row=2, col=1)
        
    fig.update_layout(
        height=520,
        template=plotly_theme,
        paper_bgcolor=p_bg,
        plot_bgcolor=p_plot_bg,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=15, r=15, t=20, b=15),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Lũy kế (Tỷ VND)", row=1, col=1, gridcolor=p_grid)
    fig.update_yaxes(title_text=f"{group_name} (Tỷ)", row=2, col=1, gridcolor=p_grid)
    fig.update_xaxes(gridcolor=p_grid)
    return fig

# 6. MAIN APP INTERFACE
def main():
    data_dir = get_data_dir()
    result_path = get_result_path()
    
    df_raw = load_raw_overview(result_path)
    if df_raw.empty:
        st.error(f"Không tìm thấy file kết quả tại: {result_path}")
        return

    # Top Executive Bar
    c_head1, c_head2 = st.columns([3, 1])
    with c_head1:
        st.markdown(f"""
        <div style="margin-bottom: 12px;">
            <h2 style="margin: 0; font-size: 1.65rem; font-weight: 800; color: {c_text_title}; letter-spacing: -0.02em;">VN-Quant Alpha Terminal</h2>
            <p style="margin: 4px 0 0 0; color: {c_text_sub}; font-size: 0.85rem;">Hệ thống định lượng dòng tiền 4 nhóm nhà đầu tư & Mô hình Bayesian Ridge (387 Cổ Phiếu)</p>
        </div>
        """, unsafe_allow_html=True)
    with c_head2:
        theme_choice = st.selectbox(
            "Theme:",
            options=["Dark Theme", "Light Theme"],
            index=0 if is_dark else 1,
            label_visibility="collapsed"
        )
        if theme_choice != st.session_state.theme_mode:
            st.session_state.theme_mode = theme_choice
            st.rerun()

    # 5 KPI Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    total_syms = len(df_raw)
    high_ic = int((df_raw["IC OOS"] >= 0.15).sum())
    high_hit = int((df_raw["Hit OOS (%)"] >= 60).sum())
    avg_ic = float(df_raw["IC OOS"].dropna().mean()) if len(df_raw["IC OOS"].dropna()) > 0 else 0.0
    avg_hit = float(df_raw["Hit OOS (%)"].dropna().mean()) if len(df_raw["Hit OOS (%)"].dropna()) > 0 else 0.0

    with c1:
        st.markdown(f"""
        <div class="stat-card-pro">
            <div class="stat-label-pro">Tổng Cổ Phiếu</div>
            <div class="stat-num-pro">{total_syms}</div>
            <div class="stat-desc-pro">387 mã đã học tham số</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        pct_ic = high_ic / total_syms * 100 if total_syms > 0 else 0
        st.markdown(f"""
        <div class="stat-card-pro">
            <div class="stat-label-pro">IC OOS Đỉnh (≥0.15)</div>
            <div class="stat-num-pro" style="color: #10B981;">{high_ic} <span style="font-size: 0.85rem; font-weight: 500;">({pct_ic:.0f}%)</span></div>
            <div class="stat-desc-pro">Tương quan dự đoán cao</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        pct_hit = high_hit / total_syms * 100 if total_syms > 0 else 0
        st.markdown(f"""
        <div class="stat-card-pro">
            <div class="stat-label-pro">Hit Rate OOS (≥60%)</div>
            <div class="stat-num-pro" style="color: #38BDF8;">{high_hit} <span style="font-size: 0.85rem; font-weight: 500;">({pct_hit:.0f}%)</span></div>
            <div class="stat-desc-pro">Độ chuẩn xác chiều giá</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="stat-card-pro">
            <div class="stat-label-pro">Trung Bình IC OOS</div>
            <div class="stat-num-pro" style="color: #A855F7;">{avg_ic:.3f}</div>
            <div class="stat-desc-pro">Toàn bộ thị trường</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="stat-card-pro">
            <div class="stat-label-pro">Hit Rate OOS TB</div>
            <div class="stat-num-pro" style="color: #F59E0B;">{avg_hit:.1f}%</div>
            <div class="stat-desc-pro">Tỷ lệ thắng trung bình</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 5 Clean Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Top Alpha Screener", 
        "Phân Tích Chi Tiết Cổ Phiếu", 
        "Dòng Tiền 4 Nhóm Nhà Đầu Tư",
        "So Sánh Cổ Phiếu",
        "Phương Pháp & Chiến Lược"
    ])

    # ---------------- TAB 1: SCREENER ----------------
    with tab1:
        col_l, col_r = st.columns([1, 1.4])
        with col_l:
            st.markdown(f"<h5 style='color: {c_text_title}; font-weight: 700;'>Phân Bổ Chất Lượng IC OOS (387 Mã)</h5>", unsafe_allow_html=True)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_raw["IC OOS"], nbinsx=25,
                marker_color="#10B981", opacity=0.85,
            ))
            fig_hist.update_layout(
                template=plotly_theme, height=250,
                paper_bgcolor=p_bg, plot_bgcolor=p_plot_bg,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="IC OOS", yaxis_title="Số lượng mã",
            )
            fig_hist.update_xaxes(gridcolor=p_grid)
            fig_hist.update_yaxes(gridcolor=p_grid)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_r:
            st.markdown(f"<h5 style='color: {c_text_title}; font-weight: 700;'>Ma Trận Hiệu Quả (IC OOS vs Hit Rate)</h5>", unsafe_allow_html=True)
            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scatter(
                x=df_raw["IC OOS"], y=df_raw["Hit OOS (%)"],
                mode="markers",
                marker=dict(color="#38BDF8", size=7, opacity=0.8, line=dict(width=0.5, color=c_bg)),
                text=df_raw["Mã"],
                hovertemplate="<b>Mã: %{text}</b><br>IC OOS: %{x:.3f}<br>Hit OOS: %{y:.1f}%<extra></extra>",
            ))
            fig_scat.update_layout(
                template=plotly_theme, height=250,
                paper_bgcolor=p_bg, plot_bgcolor=p_plot_bg,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="IC OOS", yaxis_title="Hit Rate OOS (%)",
            )
            fig_scat.update_xaxes(gridcolor=p_grid)
            fig_scat.update_yaxes(gridcolor=p_grid)
            st.plotly_chart(fig_scat, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='color: {c_text_title}; font-weight: 700;'>Bảng Xếp Hạng Alpha Toàn Bộ Thị Trường ({len(df_raw)} mã)</h5>", unsafe_allow_html=True)
        
        c_srch, _ = st.columns([1.5, 3])
        with c_srch:
            search_ticker = st.text_input("Tìm kiếm mã cổ phiếu:", placeholder="Nhập mã (ví dụ: SSI, VNM, MWG)...").strip().upper()
            
        df_display = df_raw.sort_values(by="Score", ascending=False)
        if search_ticker:
            df_display = df_display[df_display["Mã"].str.contains(search_ticker, na=False)]
            
        t_rows = []
        for _, r in df_display.head(60).iterrows():
            ic_col = "#10B981" if r["IC OOS"] >= 0.15 else ("#EF4444" if r["IC OOS"] < 0 else c_text_body)
            hit_col = "#10B981" if r["Hit OOS (%)"] >= 60 else ("#EF4444" if r["Hit OOS (%)"] < 50 else c_text_body)
            t_rows.append(f"<tr><td><b>{r['Mã']}</b></td><td><b>{r['Score']:.3f}</b></td><td style='color:{ic_col}; font-weight:700;'>{r['IC OOS']:.3f}</td><td style='color:{hit_col}; font-weight:700;'>{r['Hit OOS (%)']:.1f}%</td><td>{r['IC Train']:.3f}</td><td>{r['Hit Train (%)']:.1f}%</td><td>{r['Alpha']:.1f}</td><td>{r['a']}</td><td>{r['b']}</td><td>{r['c']}</td><td>{r['d']}</td></tr>")
        
        table_html = f"""<table class="fin-table">
<thead><tr>
<th>Mã CP</th><th>Score</th><th>IC OOS</th><th>Hit OOS %</th><th>IC Train</th><th>Hit Train %</th><th>Alpha</th><th>a</th><th>b</th><th>c</th><th>d</th>
</tr></thead><tbody>
{"".join(t_rows)}
</tbody></table>"""
        st.markdown(table_html, unsafe_allow_html=True)
        if len(df_display) > 60:
            st.caption(f"Đang hiển thị 60 / {len(df_display)} mã (sắp xếp theo Score giảm dần).")

    # ---------------- TAB 2: STOCK DEEP DIVE ----------------
    with tab2:
        all_symbols = sorted(df_raw["Mã"].astype(str).unique().tolist())
        if not all_symbols:
            return

        c_sel1, c_sel2 = st.columns([1.2, 3])
        with c_sel1:
            selected_symbol = st.selectbox(
                "Chọn mã cổ phiếu muốn phân tích:",
                options=all_symbols,
                index=0
            )

        sym_row = df_raw[df_raw["Mã"] == selected_symbol].iloc[0]

        with st.spinner(f"Đang tính toán biểu đồ & phân vùng Decile cho mã {selected_symbol}..."):
            res = analyze_single_symbol_cached(
                symbol=selected_symbol,
                a=int(sym_row["a"]), b=int(sym_row["b"]), c=int(sym_row["c"]), d=int(sym_row["d"]),
                alpha=float(sym_row["Alpha"]),
                data_dir=data_dir
            )

        if res is not None:
            sig = res["latest_signal"]
            badge_bg = "rgba(16, 185, 129, 0.15)" if sig == "Tang manh" else ("rgba(239, 68, 68, 0.15)" if sig == "Giam manh" else "rgba(245, 158, 11, 0.15)")
            badge_color = "#10B981" if sig == "Tang manh" else ("#EF4444" if sig == "Giam manh" else "#F59E0B")
            dec_val = int(res["latest_decile"]) if not np.isnan(res["latest_decile"]) else "N/A"
            
            st.markdown(f"""
            <div class="ticker-hero-pro">
                <div>
                    <span class="ticker-title-pro">{res["symbol"]}</span>
                    <span style="margin-left: 15px; background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}; padding: 6px 16px; border-radius: 8px; font-weight: 700; font-size: 0.95rem;">
                        {sig.upper()} (DECILE {dec_val})
                    </span>
                </div>
                <div>
                    <span class="pill-tag-pro">a = {res["a"]}</span>
                    <span class="pill-tag-pro">b = {res["b"]}</span>
                    <span class="pill-tag-pro">c = {res["c"]}</span>
                    <span class="pill-tag-pro">d = {res["d"]}</span>
                    <span class="pill-tag-pro">Alpha = {res["alpha"]:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"<h5 style='color: {c_text_title}; font-weight: 700;'>1. Biểu Đồ Giá & Phân Vùng Tín Hiệu Gom/Thoát OOS ({len(res['oos_price_data'])} Phiên)</h5>", unsafe_allow_html=True)
            fig_price = make_clean_line_decile_fig(res)
            st.plotly_chart(fig_price, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='color: {c_text_title}; font-weight: 700;'>2. Kiểm Định Tính Đơn Điệu (Monotonicity) & Độ Tin Cậy Mô Hình (IC = {res['rank_ic']:.3f})</h5>", unsafe_allow_html=True)
            fig_audit = make_model_audit_fig(res)
            st.plotly_chart(fig_audit, use_container_width=True)
        else:
            st.error(f"Không thể phân tích dữ liệu cho mã {selected_symbol}.")

    # ---------------- TAB 3: FLOW BREAKDOWN ----------------
    with tab3:
        st.markdown(f"<h5 style='color: {c_text_title}; font-weight: 700;'>Theo Dõi Luân Chuyển Dòng Tiền 4 Nhóm Nhà Đầu Tư ({selected_symbol})</h5>", unsafe_allow_html=True)
        if res is not None and res.get("df_raw") is not None:
            df_s = res["df_raw"].tail(60)
            f_sum = df_s["foreign_net_val"].sum() / 1e9 if "foreign_net_val" in df_s.columns else 0
            p_sum = df_s["prop_net_val"].sum() / 1e9 if "prop_net_val" in df_s.columns else 0
            inst_sum = df_s["local_inst_net_val"].sum() / 1e9 if "local_inst_net_val" in df_s.columns else 0
            ind_sum = df_s["local_ind_net_val"].sum() / 1e9 if "local_ind_net_val" in df_s.columns else 0
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                color_f = "#00E5FF" if f_sum >= 0 else "#EF4444"
                st.markdown(f"""
                <div class="stat-card-pro">
                    <div class="stat-label-pro">Khối Ngoại (60 Phiên)</div>
                    <div class="stat-num-pro" style="color: {color_f};">{f_sum:+,.1f} Tỷ</div>
                    <div class="stat-desc-pro">Tổng giá trị khớp ròng</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                color_p = "#00E676" if p_sum >= 0 else "#EF4444"
                st.markdown(f"""
                <div class="stat-card-pro">
                    <div class="stat-label-pro">Tự Doanh (60 Phiên)</div>
                    <div class="stat-num-pro" style="color: {color_p};">{p_sum:+,.1f} Tỷ</div>
                    <div class="stat-desc-pro">Tổng giá trị khớp ròng</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                color_inst = "#D500F9" if inst_sum >= 0 else "#EF4444"
                st.markdown(f"""
                <div class="stat-card-pro">
                    <div class="stat-label-pro">Tổ Chức Nội (60 Phiên)</div>
                    <div class="stat-num-pro" style="color: {color_inst};">{inst_sum:+,.1f} Tỷ</div>
                    <div class="stat-desc-pro">Tổng giá trị khớp ròng</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                color_ind = "#FFD600" if ind_sum >= 0 else "#EF4444"
                st.markdown(f"""
                <div class="stat-card-pro">
                    <div class="stat-label-pro">Cá Nhân Nội (60 Phiên)</div>
                    <div class="stat-num-pro" style="color: {color_ind};">{ind_sum:+,.1f} Tỷ</div>
                    <div class="stat-desc-pro">Tổng giá trị khớp ròng</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            c_bar_opt, _ = st.columns([1.5, 3])
            with c_bar_opt:
                sel_bar_group = st.selectbox(
                    "Chọn nhóm NĐT để xem khớp ròng từng ngày (Xanh: Mua / Đỏ: Bán):",
                    options=["foreign", "prop", "local_inst", "local_ind"],
                    format_func=lambda x: {"foreign": "Khối Ngoại", "prop": "Tự Doanh", "local_inst": "Tổ Chức Nội", "local_ind": "Cá Nhân Nội"}[x],
                    index=0
                )
                
            fig_flow = make_powerbi_flow_fig(res["df_raw"], selected_symbol, selected_group_bar=sel_bar_group)
            st.plotly_chart(fig_flow, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu dòng tiền chi tiết.")

    # ---------------- TAB 4: PAIR COMPARISON ----------------
    with tab4:
        st.markdown(f"<h5 style='color: {c_text_title}; font-weight: 700;'>So Sánh Cặp Cổ Phiếu</h5>", unsafe_allow_html=True)
        cp1, cp2 = st.columns(2)
        with cp1:
            sym_a = st.selectbox("Cổ phiếu A:", options=all_symbols, index=0)
        with cp2:
            sym_b = st.selectbox("Cổ phiếu B:", options=all_symbols, index=min(1, len(all_symbols)-1))
            
        df_comp = df_raw[df_raw["Mã"].isin([sym_a, sym_b])][["Mã", "Score", "IC OOS", "Hit OOS (%)", "IC Train", "Alpha", "a", "b", "c", "d"]]
        st.dataframe(df_comp, use_container_width=True)

    # ---------------- TAB 5: STRATEGY GUIDE ----------------
    with tab5:
        st.markdown(f"""
        <div style="background: {c_card}; border: 1px solid {c_border}; border-radius: 12px; padding: 24px; color: {c_text_body};">
            <h4 style="margin-top: 0; color: {c_text_title};">Cẩm Nang Chiến Lược Định Lượng Dòng Tiền (Quant Flow Alpha)</h4>
            <br>
            <h5 style="color: {c_text_title};">1. Nguyên Lý Khai Thác Alpha Dòng Tiền</h5>
            <p style="color: {c_text_sub};">Thị trường chứng khoán Việt Nam vận động chủ yếu bởi sự luân chuyển giữa 4 nhóm nhà đầu tư:</p>
            <ul>
                <li><b>Khối Ngoại (foreign)</b>: Dẫn dắt xu hướng trung dài hạn.</li>
                <li><b>Tự Doanh (prop)</b>: Dòng tiền tạo lập có lợi thế thông tin lớn.</li>
                <li><b>Tổ Chức Nội (local_inst)</b>: Quỹ đầu tư nội và dòng tiền doanh nghiệp.</li>
                <li><b>Cá Nhân Nội (local_ind)</b>: Dòng tiền tâm lý, thường mua đỉnh/bán đáy.</li>
            </ul>
            <p style="color: {c_text_sub};">Mô hình chuẩn hóa hành vi khớp ròng bằng các hàm Z-score, đo gia tốc bứt phá (Flow Impulse), sự phân tán (Dispersion) và mức độ đồng thuận giữa Khối Ngoại & Tự Doanh.</p>
            <hr style="border: 0; border-top: 1px solid {c_border}; margin: 20px 0;">
            <h5 style="color: {c_text_title};">2. Chiến Lược Giải Ngân Theo Decile</h5>
            <ul>
                <li><b>Decile 8 - 9 (Tín Hiệu Mua Mạnh)</b>: Xác suất sinh lời T+10 cao nhất, dòng tiền tổ chức và tự doanh đang gom ròng quyết liệt.</li>
                <li><b>Decile 2 - 7 (Trung Tính)</b>: Thị trường đi ngang tích lũy, phù hợp nắm giữ hoặc quan sát.</li>
                <li><b>Decile 0 - 1 (Tín Hiệu Bán/Hạ Tỷ Trọng)</b>: Dòng tiền lớn thoát mạnh, rủi ro điều chỉnh cao.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
