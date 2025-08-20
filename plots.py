# plots.py — Economist-style plotting
# 1) Cumulative returns (EW, VW, SPX, RF)
# 2) Portfolio weights (EW & VW) stacked areas (analogous palettes + labels)
# 3) Diversification dispersion: mean ± 1σ (shaded band)
# 4) Combined dashboard (returns left; EW/VW weights right)

from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import numpy as np
import colorsys

# ------------------------- Aesthetics (Economist-inspired) -------------------------
ECON = {
    "EW":  "#006BA4",  # deep blue
    "VW":  "#D64937",  # economist red
    "SPX": "#4B9B9B",  # teal
    "RF":  "#6E6E6E",  # neutral gray
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.color": "#B8B8B8",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

def _style_axis(ax, *, ygrid=True):
    """Minimal Economist look: horizontal grid only, clean ticks."""
    ax.grid(axis="y", visible=ygrid)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", labelsize=10)
    ax.margins(x=0.01, y=0.05)

def _as_share(x: pd.Series | pd.DataFrame) -> pd.Series:
    """Convert returns or compounded series into normalized share (start=1)."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected Series or single-column DataFrame")
        x = x.iloc[:, 0]
    s = x.dropna()
    if s.iloc[0] >= 0.9 and (s > 0).all():
        return s / s.iloc[0]
    return (1 + s).cumprod() / (1 + s).cumprod().iloc[0]

# ------------------------- Color helpers (analogous palettes) -------------------------

def _analogous_palette(
    n: int,
    *,
    base_hue_deg: float,
    spread_deg: float = 16,
    l_range=(0.30, 0.84),
    s_range=(0.35, 0.80),
):
    """
    Generate n distinct shades around one hue family (elegant, non-rainbow).
    base_hue_deg ≈ 210 for cool blues (EW), ≈ 18 for warm oranges (VW).
    Distinction comes mainly from lightness/saturation, hue varies only slightly.
    """
    if n <= 0:
        return []
    # tiny hue fan around base hue (at most 7 different angles)
    k = np.linspace(-spread_deg, spread_deg, min(n, 7))
    hues = (base_hue_deg + np.pad(k, (0, max(0, n - len(k))), mode="wrap"))[:n] / 360.0
    # interleave light/dark to reduce adjacent blending
    ls = np.linspace(l_range[1], l_range[0], n)
    ss = np.linspace(s_range[0], s_range[1], n)
    order = np.ravel(np.column_stack((np.arange(0, n, 2), np.arange(1, n, 2)))).tolist()[:n]
    ls, ss, hues = ls[order], ss[order], hues[order]
    cols = [colorsys.hls_to_rgb(h, l, s) for h, l, s in zip(hues, ls, ss)]
    return cols

def _stack_with_labels(
    ax,
    df: pd.DataFrame,
    colors,
    *,
    top_k: int | None = None,
    highlight: str | None = None,
):
    """
    Draws a labeled stackplot:
    - optional top_k: keep top-K by average weight and sum the rest into 'Other'
    - largest weights anchored at the bottom, smallest at the top
    - darkest shades at the bottom, lighter at the top
    - right-edge labels with white stroke for readability
    """
    import matplotlib.patheffects as pe

    df = df.copy()
    if top_k is not None and 0 < top_k < df.shape[1]:
        means = df.mean().sort_values(ascending=False)
        keep = list(means.index[:top_k])
        other = df.drop(columns=keep).sum(axis=1)
        df = pd.concat([df[keep], other.rename("Other")], axis=1)

    # order columns by final weight (largest → smallest)
    order = df.iloc[-1].sort_values(ascending=False).index.tolist()
    df = df[order]

    # ensure enough colors
    if len(colors) < df.shape[1]:
        colors = (colors * (df.shape[1] // len(colors) + 1))[:df.shape[1]]

    # reverse colors so darkest aligns with bottom layers (largest weights)
    colors = list(reversed(colors))

    ax.stackplot(
        df.index,
        df.T.values,
        colors=colors,
        alpha=0.93,
        edgecolor="white",
        linewidth=0.4,
    )
    _style_axis(ax)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # right-edge labels
    x_last = df.index[-1]
    cum = df.cumsum(axis=1)
    mids = cum.subtract(df / 2.0, axis=0)
    for i, col in enumerate(df.columns):
        y = mids.iloc[-1, i]
        kw = dict(va="center", ha="left", fontsize=9, color=colors[i])
        if highlight and col == highlight:
            kw["fontweight"] = "bold"
        ax.annotate(
            col,
            (x_last, y),
            xytext=(5, 0),
            textcoords="offset points",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            **kw,
        )
    ax.margins(x=0.02)
    ax.grid(axis="x", visible=False)

# --------------------------------- Plots ---------------------------------

def plot_cumulative_returns(
    ew_share: pd.Series,
    vw_share: pd.Series,
    spx_share: Optional[pd.Series] = None,
    rf_share: Optional[pd.Series] = None,
    title: str = "Cumulative returns",
    subtitle: Optional[str] = None,
) -> plt.Figure:
    ew = _as_share(ew_share)
    vw = _as_share(vw_share)
    idx = ew.index.intersection(vw.index)
    ew, vw = ew.reindex(idx), vw.reindex(idx)
    spx = _as_share(spx_share).reindex(idx) if spx_share is not None else None
    rf  = _as_share(rf_share).reindex(idx)  if rf_share  is not None else None

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(ew.index, ew.values, color=ECON["EW"], lw=2.6, label="Equally weighted")
    ax.plot(vw.index, vw.values, color=ECON["VW"], lw=2.6, label="Value weighted")
    if spx is not None:
        ax.plot(spx.index, spx.values, color=ECON["SPX"], lw=2.0, ls="--", label="S&P 500")
    if rf is not None:
        ax.plot(rf.index,  rf.values,  color=ECON["RF"],  lw=1.8, ls=":",  label="Risk-free")

    ax.set_title(title, fontsize=15, weight="bold", loc="left")
    if subtitle:
        ax.set_title(subtitle, fontsize=11, color="#6E6E6E", loc="left", pad=20)
    ax.set_ylabel("Growth multiple", fontsize=12)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.1f}×"))
    _style_axis(ax)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    return fig

def plot_weights_subplots(
    ew_weights: pd.DataFrame,
    vw_weights: pd.DataFrame,
    title: str = "Portfolio weights over time",
    subtitle: Optional[str] = None,
    *,
    top_k: int | None = None,
    highlight: str | None = None,
) -> plt.Figure:
    # Align dates
    idx = ew_weights.index.intersection(vw_weights.index)
    ew = ew_weights.reindex(idx)
    vw = vw_weights.reindex(idx)

    # Analogous palettes (no rainbow)
    colors_ew = _analogous_palette(ew.shape[1], base_hue_deg=210)  # cool blues
    colors_vw = _analogous_palette(vw.shape[1], base_hue_deg=18)   # warm oranges

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1, ax2 = axes

    # Equal-weighted stack with labels
    _stack_with_labels(ax1, ew, colors_ew, top_k=top_k, highlight=highlight)
    ax1.set_title("Equal-weighted", fontsize=13, weight="bold", loc="left")
    ax1.set_ylabel("Weight", fontsize=12)

    # Value-weighted stack with labels
    _stack_with_labels(ax2, vw, colors_vw, top_k=top_k, highlight=highlight)
    ax2.set_title("Value-weighted", fontsize=13, weight="bold", loc="left")
    ax2.set_ylabel("Weight", fontsize=12)

    fig.suptitle(title, fontsize=15, weight="bold", x=0.01, ha="left", y=0.98)
    if subtitle:
        fig.text(0.01, 0.94, subtitle, fontsize=11, color="#6E6E6E", ha="left")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_diversification_dispersion(
    div_mean_var: pd.Series,
    div_std_var: pd.Series,
    title: str = "Diversification reduces variance",
    subtitle: Optional[str] = "Mean ± 1σ variance across random EW portfolios",
    y_label: str = "Daily return variance",
) -> plt.Figure:
    x = [int(n) for n in div_mean_var.index]
    mean = div_mean_var.values.astype(float)
    std  = div_std_var.values.astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(x, mean - std, mean + std, color=ECON["EW"], alpha=0.20, label="±1σ")
    ax.plot(x, mean, color=ECON["EW"], lw=2.8, marker="o", label="Mean")

    ax.set_title(title, fontsize=15, weight="bold", loc="left")
    if subtitle:
        ax.set_title(subtitle, fontsize=11, color="#6E6E6E", loc="left", pad=20)
    ax.set_xlabel("Number of assets (N)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    _style_axis(ax)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    return fig

def show_combined_dashboard(
    ew_share: pd.Series,
    vw_share: pd.Series,
    ew_weights: pd.DataFrame,
    vw_weights: pd.DataFrame,
    spx_share: Optional[pd.Series] = None,
    rf_share: Optional[pd.Series] = None,
    title_left: str = "Cumulative returns",
    title_right: str = "Portfolio weights",
    *,
    top_k: int | None = None,
    highlight: str | None = None,
) -> plt.Figure:
    """Left = returns; Right = EW (top) & VW (bottom) weights with analogous palettes + labels."""
    from matplotlib.gridspec import GridSpec

    # Prepare series
    ew = _as_share(ew_share)
    vw = _as_share(vw_share)
    idx = ew.index.intersection(vw.index)
    spx = _as_share(spx_share).reindex(idx) if spx_share is not None else None
    rf  = _as_share(rf_share).reindex(idx)  if rf_share  is not None else None
    ew, vw = ew.reindex(idx), vw.reindex(idx)
    ew_w = ew_weights.reindex(idx)
    vw_w = vw_weights.reindex(idx)

    # Colors for stacks (analogous, elegant)
    ew_cols = _analogous_palette(ew_w.shape[1], base_hue_deg=210)
    vw_cols = _analogous_palette(vw_w.shape[1], base_hue_deg=18)

    fig = plt.figure(figsize=( fifteen := 15, 7 ))
    gs = GridSpec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1], figure=fig)

    # Left: cumulative returns (full height)
    axL = fig.add_subplot(gs[:, 0])
    axL.plot(ew.index, ew.values, color=ECON["EW"], lw=2.8, label="Equally weighted")
    axL.plot(vw.index, vw.values, color=ECON["VW"], lw=2.8, label="Value weighted")
    if spx is not None:
        axL.plot(spx.index, spx.values, color=ECON["SPX"], lw=2.0, ls="--", label="S&P 500")
    if rf is not None:
        axL.plot(rf.index,  rf.values,  color=ECON["RF"],  lw=1.8, ls=":",  label="Risk-free")

    axL.set_title(title_left, fontsize=15, weight="bold", loc="left")
    axL.set_ylabel("Growth multiple", fontsize=12)
    axL.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.1f}×"))
    _style_axis(axL)
    axL.legend(loc="upper left", fontsize=10)

    # Right top: EW weights (labeled)
    axRT = fig.add_subplot(gs[0, 1])
    _stack_with_labels(axRT, ew_w, ew_cols, top_k=top_k, highlight=highlight)
    axRT.set_title("Equal-weighted", fontsize=13, weight="bold", loc="left")

    # Right bottom: VW weights (labeled)
    axRB = fig.add_subplot(gs[1, 1], sharex=axRT)
    _stack_with_labels(axRB, vw_w, vw_cols, top_k=top_k, highlight=highlight)
    axRB.set_title("Value-weighted", fontsize=13, weight="bold", loc="left")

    # layout: reserve bottom space for source note
    fig.tight_layout(rect=[0, 0.06, 1, 1])   # leave ~6% at the bottom

    # Data sources footnote (Economist-style), inside figure coords
    fig.text(
        0.01, 0.02,
        "Sources: Dataset, Yahoo Finance, Risk-free = 4.92% (30Y T-Bill)",
        fontsize=9, color="#555555", ha="left", va="bottom"
    )

    return fig
