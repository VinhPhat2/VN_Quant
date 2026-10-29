"""
Update latest_deciles_cache.csv for all stocks based on the latest market prices.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from server import get_stock_analysis, get_result_path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def process_symbol(sym):
    try:
        res = get_stock_analysis(sym)
        pred_ret = 0.0
        if res.get("audit", {}).get("oos_scores"):
            pred_ret = round(float(res["audit"]["oos_scores"][-1] * 100), 2)
        return {
            "symbol": sym,
            "latest_decile": res["latest_decile"],
            "latest_pred_ret": pred_ret,
            "n_deciles": res["n_deciles"],
        }
    except Exception:
        return None


def main():
    res_path = get_result_path()
    if not os.path.exists(res_path):
        print(f"File not found: {res_path}")
        return

    df_res = pd.read_csv(res_path)
    symbols = df_res["symbol"].dropna().unique().tolist()
    print(f"Computing latest deciles for {len(symbols)} symbols...")

    records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for item in executor.map(process_symbol, symbols):
            if item is not None:
                records.append(item)

    df_out = pd.DataFrame(records).sort_values("symbol").reset_index(drop=True)
    out_path = os.path.join(os.path.dirname(__file__), "latest_deciles_cache.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Successfully updated {len(df_out)} symbols in {out_path}!")


if __name__ == "__main__":
    main()
