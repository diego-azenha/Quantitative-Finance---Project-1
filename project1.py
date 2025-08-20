# ======= minimal backtest (no plots) =======
# - EW vs VW with 1-day info lag, monthly rebalancing
# - daily returns for both
# - stats: ann. return, ann. stdev, Sharpe (vs 4.92%), Information Ratio vs S&P500
# - turnover (rebalance-only)
# - Part B diversification (variance vs n), 10 trials
# ===========================================

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import os

# make sure results directory exists
os.makedirs("results", exist_ok=True)


ANNUAL_RF = 0.0492
TRADING_DAYS = 252

# ---------- load data (from clean_data/) ----------
base = Path(__file__).resolve().parent
data_dir = base / "clean_data"

prices = pd.read_parquet(data_dir / "clean_stock_prices.parquet").sort_index()
caps   = pd.read_parquet(data_dir / "clean_mkt_cap.parquet").sort_index()

prices.index = pd.to_datetime(prices.index)
caps.index   = pd.to_datetime(caps.index)
caps = caps.reindex(prices.index).ffill()


# simple daily returns
rets = prices.pct_change().dropna(how="all")
rets = rets.dropna(axis=1, how="all")

# ---------- month boundaries: last TD of each month ----------
last_td = (
    pd.Series(1, index=rets.index)
      .groupby(rets.index.to_period("M"))
      .tail(1)
      .index
)

# ---------- simulate EW & VW (half-open window (start, end]) ----------
ew_daily, vw_daily = [], []
ew_w_hist, vw_w_hist = [], []
ew_turns, vw_turns = [], []

cols = rets.columns
N = len(cols)
prev_w_ew = None
prev_w_vw = None

for i in range(len(last_td) - 1):
    start = last_td[i]      # info date (last TD of month m)
    end   = last_td[i+1]    # end of m+1

    month_rets = rets.loc[start:end]
    month_rets = month_rets[month_rets.index > start]   # (start, end]
    if month_rets.empty:
        continue

    # targets at first TD of m+1 using data from 'start'
    ew_target = pd.Series(1.0/N, index=cols)

    mcaps = caps.loc[start].reindex(cols)
    if mcaps.notna().any():
        vw_target = mcaps / mcaps.sum()
    else:
        vw_target = pd.Series(1.0/N, index=cols)

    # turnover (rebalance only)
    if prev_w_ew is not None:
        ew_turns.append(float((ew_target - prev_w_ew).abs().sum()))
    if prev_w_vw is not None:
        vw_turns.append(float((vw_target - prev_w_vw).abs().sum()))

    # drift inside month
    w_ew = ew_target.copy()
    w_vw = vw_target.copy()
    for dt, r in month_rets.iterrows():
        g_ew = float(np.dot(w_ew.values, (1.0 + r.values)))
        g_vw = float(np.dot(w_vw.values, (1.0 + r.values)))
        ew_daily.append(pd.Series(g_ew - 1.0, index=[dt]))
        vw_daily.append(pd.Series(g_vw - 1.0, index=[dt]))

        w_ew = w_ew * (1.0 + r.values)
        s = w_ew.sum()
        w_ew = (w_ew / s) if s > 0 else ew_target.copy()
        ew_w_hist.append(pd.Series(w_ew, index=cols, name=dt))

        w_vw = w_vw * (1.0 + r.values)
        s2 = w_vw.sum()
        w_vw = (w_vw / s2) if s2 > 0 else vw_target.copy()
        vw_w_hist.append(pd.Series(w_vw, index=cols, name=dt))

    prev_w_ew = w_ew.copy()
    prev_w_vw = w_vw.copy()

# convert to Series/DataFrame with unique index
ew_ret = pd.concat(ew_daily).sort_index().squeeze()
vw_ret = pd.concat(vw_daily).sort_index().squeeze()
ew_weights = pd.DataFrame(ew_w_hist)
vw_weights = pd.DataFrame(vw_w_hist)
ew_turnover = float(np.mean(ew_turns)) if ew_turns else np.nan
vw_turnover = float(np.mean(vw_turns)) if vw_turns else np.nan

# ---------- benchmark ----------
spx = (
    yf.download("^GSPC",
                start=str(rets.index.min().date()),
                end=str(rets.index.max().date()),
                auto_adjust=True, progress=False)["Close"]
      .sort_index()
      .pct_change()
      .dropna()
)

# align all series to common dates
idx = ew_ret.index.intersection(vw_ret.index).intersection(spx.index)
ew_ret = ew_ret.reindex(idx)
vw_ret = vw_ret.reindex(idx)
spx    = spx.reindex(idx)
rf_daily = pd.Series(ANNUAL_RF/TRADING_DAYS, index=idx, name="RF")

# ---------- stats helpers (robust against Series/DataFrame) ----------

TRADING_DAYS = 252
ANNUAL_RF = 0.0492

def _ser(x):
    """Coerce to a 1-D Series. If a DataFrame slips in:
       - squeeze if 1 column; otherwise average across columns."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            x = x.mean(axis=1)
    return x

def ann_ret(x) -> float:
    x = _ser(x)
    return float((1 + x).prod() ** (TRADING_DAYS / len(x)) - 1) if len(x) else np.nan

def ann_vol(x) -> float:
    x = _ser(x)
    return float(x.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(x) else np.nan

def sharpe(x) -> float:
    x = _ser(x)
    r, v = ann_ret(x), ann_vol(x)
    return float((r - ANNUAL_RF) / v) if v and v > 0 else np.nan

def info_ratio(x, bench) -> float:
    x = _ser(x)
    bench = _ser(bench)
    # align on common dates
    idx = x.index.intersection(bench.index)
    x = x.reindex(idx)
    bench = bench.reindex(idx)
    ex = x - bench
    mu = float(ex.mean() * TRADING_DAYS)
    sd = float(ex.std(ddof=0) * np.sqrt(TRADING_DAYS))
    return float(mu / sd) if sd > 0 else np.nan


# ---------- build results table (pure floats) ----------
rows = [
    ("Equally Weighted", ann_ret(ew_ret), ann_vol(ew_ret), sharpe(ew_ret), info_ratio(ew_ret, spx), ew_turnover),
    ("Value Weighted",   ann_ret(vw_ret), ann_vol(vw_ret), sharpe(vw_ret), info_ratio(vw_ret, spx), vw_turnover),
    ("S&P 500",          ann_ret(spx),    ann_vol(spx),    sharpe(spx),    np.nan,                 np.nan)
]
results = pd.DataFrame(rows, columns=["Portfolio","Ann. Return","Ann. Volatility","Sharpe Ratio","Info Ratio","Turnover"]).set_index("Portfolio")

# pretty print
tbl = results.copy()
tbl["Ann. Return"]    = (100 * tbl["Ann. Return"]).map(lambda v: f"{v:.2f}%" if pd.notnull(v) else "–")
tbl["Ann. Volatility"]= (100 * tbl["Ann. Volatility"]).map(lambda v: f"{v:.2f}%" if pd.notnull(v) else "–")
for c in ["Sharpe Ratio","Info Ratio","Turnover"]:
    tbl[c] = tbl[c].map(lambda v: f"{v:.2f}" if pd.notnull(v) else "–")

print("\nPortfolio Statistics (2015–2024):")
print(tbl)

print("\nWhich portfolio has the highest turnover?")
if pd.notnull(ew_turnover) and pd.notnull(vw_turnover):
    print("Equally Weighted" if ew_turnover > vw_turnover else "Value Weighted" if vw_turnover > ew_turnover else "Tie")
else:
    print("Insufficient data to compare.")

# ---------- curves ready for plotting (in a separate file) ----------
ew_curve  = (1 + ew_ret).cumprod(); ew_curve  /= ew_curve.iloc[0]
vw_curve  = (1 + vw_ret).cumprod(); vw_curve  /= vw_curve.iloc[0]
spx_curve = (1 + spx).cumprod();   spx_curve /= spx_curve.iloc[0]
rf_curve  = (1 + rf_daily).cumprod(); rf_curve /= rf_curve.iloc[0]

# ---------- Part B: diversification ----------
rng = np.random.default_rng(42)
names = list(rets.columns)
sizes = range(1, len(names) + 1)
var_by_n = {n: [] for n in sizes}

for n in sizes:
    for _ in range(10):
        subset = list(rng.choice(names, size=n, replace=False))
        var_by_n[n].append(float(rets[subset].mean(axis=1).var()))

var_df = pd.DataFrame({n: pd.Series(v) for n, v in var_by_n.items()})
div_mean_var = var_df.mean()
div_std_var  = var_df.std()

# === Keep daily weights and shares for plotting ===

# Shares (NAV-like series, starting at 1.0)
ew_share = (1 + ew_ret).cumprod()
vw_share = (1 + vw_ret).cumprod()
ew_share /= ew_share.iloc[0]
vw_share /= vw_share.iloc[0]
ew_share.name = "EW_share"
vw_share.name = "VW_share"

# Make sure weights align with returns index
ew_weights = ew_weights.reindex(ew_share.index)
vw_weights = vw_weights.reindex(vw_share.index)


from pathlib import Path
import matplotlib.pyplot as plt
from plots import show_combined_dashboard, plot_diversification_dispersion

# Ensure results folder exists
Path("results").mkdir(exist_ok=True)

# 1) Combined dashboard: cumulative returns (left) + weights (right)
fig_dash = show_combined_dashboard(
    ew_share, vw_share,
    ew_weights, vw_weights,
    spx_share=spx_curve, rf_share=rf_curve
)
fig_dash.savefig("results/combined_dashboard.png", dpi=150)

# 2) Variance dispersion (with shaded ±1σ)
fig_var = plot_diversification_dispersion(div_mean_var, div_std_var)
fig_var.savefig("results/diversification_dispersion.png", dpi=150)

plt.show()



