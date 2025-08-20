import pandas as pd
import os

# Get absolute path of the current script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CONVERTED THE DATA FROM EXCEL TO PARQUET FOR FASTER LOADING
excel_path = os.path.join(BASE_DIR, "SPX_database_2010.xlsx")
stock_prices = pd.read_excel(excel_path, sheet_name="prices")
mkt_cap = pd.read_excel(excel_path, sheet_name="mkt_cap")
stock_prices.to_parquet("_stock_prices.parquet", index=False, engine="pyarrow")
mkt_cap.to_parquet("_mkt_cap.parquet", index=False, engine="pyarrow")

# Build paths relative to this script
stock_prices_path = os.path.join(BASE_DIR, "stock_prices.parquet")
mkt_cap_path = os.path.join(BASE_DIR, "mkt_cap.parquet")

print("BASE ORIGINAL CONVERTIDA PARA PARQUET")

print(stock_prices.head(5))
print(mkt_cap.head(5))

#%%--------------------------------------------------------------------------------------

def clean_dataframe(df):
    def shorten(col: str) -> str:
        s = str(col)
        first_token = s.split(" ", 1)[0]
        return first_token[:5] if first_token else s[:6]

    out = df.copy()

    # shorten and make unique
    new_cols = [shorten(c) for c in out.columns]
    seen, uniq_cols = {}, []
    for c in new_cols:
        if c not in seen:
            seen[c] = 0
            uniq_cols.append(c)
        else:
            seen[c] += 1
            uniq_cols.append(f"{c}_{seen[c]}")
    out.columns = uniq_cols

    # handle Dates
    if "Dates" in out.columns:
        out["Dates"] = pd.to_datetime(out["Dates"], errors="coerce")
        out = out.set_index("Dates")

    return out

clean_stock_prices = clean_dataframe(stock_prices)
clean_mkt_cap = clean_dataframe(mkt_cap)

# -------------------------------------------------------------------
# Keep only the 30 target S&P tickers
# -------------------------------------------------------------------
tickers = [
    "AAPL","MSFT","AMZN","GOOGL","NVDA","CAT","ABT","JPM","V","MA",
    "NFLX","UNH","MCK","PG","HD","XOM","CVX","WMT","KO","MMM",
    "BAC","C","GS","JCI","IBM","INTC","CSCO","BA","GE","MCD"
]

# Filter both datasets to intersection of requested tickers and available columns
common_cols = [t for t in tickers if t in clean_stock_prices.columns and t in clean_mkt_cap.columns]
clean_stock_prices = clean_stock_prices[common_cols]
clean_mkt_cap = clean_mkt_cap[common_cols]

# ------------------- NEW: cut to analysis window -------------------
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"
clean_stock_prices = clean_stock_prices.loc[START_DATE:END_DATE]
clean_mkt_cap      = clean_mkt_cap.loc[START_DATE:END_DATE]
# -------------------------------------------------------------------

print("BASES LIMPAS (FILTRADAS NOS 30 TICKERS): ")
print(clean_stock_prices.head())
print(clean_mkt_cap.head())

# ------------------- Save to clean_data folder -------------------
CLEAN_DIR = os.path.join(BASE_DIR, "clean_data")
os.makedirs(CLEAN_DIR, exist_ok=True)

clean_stock_prices.to_parquet(os.path.join(CLEAN_DIR, "clean_stock_prices.parquet"), engine="pyarrow")
clean_mkt_cap.to_parquet(os.path.join(CLEAN_DIR, "clean_mkt_cap.parquet"), engine="pyarrow")

#%% --------------------- QUICK SANITY CHECKS (minimal) ---------------------

def brief_checks(prices: pd.DataFrame, caps: pd.DataFrame, thr_extreme=0.30):
    problems = []

    # 1) Index basics
    for name, df in [("PRICES", prices), ("MKT_CAP", caps)]:
        if not isinstance(df.index, pd.DatetimeIndex):
            problems.append(f"{name}: index is not DatetimeIndex")
        if not df.index.is_monotonic_increasing:
            problems.append(f"{name}: index not sorted ascending")
        dups = int(df.index.duplicated().sum())
        if dups:
            problems.append(f"{name}: duplicated dates = {dups}")
        if df.empty or df.shape[1] == 0:
            problems.append(f"{name}: empty dataframe")

    # 2) Missingness (coarse)
    miss_prices = int(prices.isna().sum().sum())
    miss_caps   = int(caps.isna().sum().sum())
    if miss_prices: problems.append(f"PRICES: total missing cells = {miss_prices}")
    if miss_caps:   problems.append(f"MKT_CAP: total missing cells = {miss_caps}")

    # 3) Non-positive values
    np_prices_cols = int(((prices <= 0).sum() > 0).sum())
    np_caps_cols   = int(((caps   <= 0).sum() > 0).sum())
    if np_prices_cols:
        problems.append(f"PRICES: columns with non-positive values = {np_prices_cols}")
    if np_caps_cols:
        problems.append(f"MKT_CAP: columns with non-positive values = {np_caps_cols}")

    # 4) Extreme daily returns (|r| > thr_extreme) — quick outlier sniff
    if prices.shape[0] > 1:
        rets = prices.pct_change()
        n_extreme = int((rets.abs() > thr_extreme).sum().sum())
        if n_extreme:
            problems.append(f"PRICES: extreme daily moves |r|>{thr_extreme:.0%} = {n_extreme}")

    # 5) Prices × Caps alignment
    common_cols = prices.columns.intersection(caps.columns)
    if len(common_cols) == 0:
        problems.append("ALIGN: no common tickers between prices and mkt_cap")
    if len(common_cols) < min(len(prices.columns), len(caps.columns)):
        problems.append(f"ALIGN: common tickers = {len(common_cols)} (prices={len(prices.columns)}, caps={len(caps.columns)})")

    idx_inter = prices.index.intersection(caps.index)
    if len(idx_inter) == 0:
        problems.append("ALIGN: no overlapping dates between prices and mkt_cap")

    # 6) Sudden cap jumps (very coarse)
    if caps.shape[0] > 1:
        cap_rets = caps.pct_change().abs()
        big_caps = int((cap_rets > 1.0).sum().sum())  # >100% day-over-day
        if big_caps:
            problems.append(f"MKT_CAP: >100% day jumps = {big_caps}")

    # --- Output ---
    print("\n=== DATA QUICK CHECK ===")
    print(f"Prices shape:  {prices.shape},  span: {prices.index.min()} → {prices.index.max()}")
    print(f"MktCap shape:  {caps.shape},     span: {caps.index.min()} → {caps.index.max()}")
    if problems:
        print("⚠ Issues detected:")
        for p in problems:
            print(" -", p)
    else:
        print("✓ No obvious breakages found.")

# Run checks on the cleaned frames you just saved
brief_checks(clean_stock_prices, clean_mkt_cap)
