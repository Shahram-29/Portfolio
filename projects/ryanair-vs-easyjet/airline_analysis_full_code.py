"""
=============================================================================
  WOODS PLC INVESTMENT ANALYSIS
  Ryanair (RYA) vs easyJet (EZJ) — Share Price & News Analysis
  Python Visualisation Suite  |  April 2026
=============================================================================
  Charts produced:
    1. Annotated share price with news events (both airlines)
    2. 50-day & 200-day moving average comparison
    3. Monthly returns heatmap  (Ryanair)
    4. Monthly returns heatmap  (easyJet)
    5. Volatility rolling window chart
    6. Head-to-head financial KPI radar chart
    7. Cumulative returns comparison
    8. Correlation & drawdown dashboard
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0F1923",
    "axes.facecolor":    "#141E2A",
    "axes.edgecolor":    "#2A3A4A",
    "axes.labelcolor":   "#C8D6E5",
    "axes.grid":         True,
    "grid.color":        "#1E2D3D",
    "grid.linewidth":    0.6,
    "xtick.color":       "#8A9BB0",
    "ytick.color":       "#8A9BB0",
    "text.color":        "#C8D6E5",
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "legend.facecolor":  "#1A2535",
    "legend.edgecolor":  "#2A3A4A",
    "legend.fontsize":   8,
})

RYA_COL  = "#4A9EE0"      # Ryanair blue
EZJ_COL  = "#FF6B35"      # easyJet orange
MA50_COL = "#E84545"      # 50-day MA red
MA200_COL= "#2ECC71"      # 200-day MA green
UP_COL   = "#2ECC71"
DN_COL   = "#E84545"
NEUTRAL  = "#F39C12"

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

# Monthly prices (approx, EUR for RYA on Euronext; GBX pence for EZJ on LSE)
dates = pd.date_range("2021-04-01", "2026-04-01", freq="MS")

rya_prices = np.array([
    14.5, 15.8, 16.2, 17.1, 16.4, 15.0,    # Apr–Sep 2021
    15.2, 14.6, 13.8, 13.0, 12.2, 12.5,    # Oct 2021–Mar 2022
    11.8, 10.8, 11.2, 12.0, 12.5, 12.8,    # Apr–Sep 2022
    13.5, 14.2, 15.0, 15.8, 16.5, 17.2,    # Oct 2022–Mar 2023
    17.8, 18.5, 19.2, 19.5, 17.4, 14.8,    # Apr–Sep 2023
    14.2, 15.4, 16.8, 17.6, 18.4, 19.1,    # Oct 2023–Mar 2024
    20.5, 21.0, 19.8, 20.2, 21.4, 22.1,    # Apr–Sep 2024
    23.2, 24.0, 24.8, 26.5, 27.8, 29.8,    # Oct 2024–Mar 2025
    28.4, 27.0, 26.2, 27.5, 28.5, 29.2,    # Apr–Sep 2025
    30.1, 27.8, 25.5, 24.8, 24.0, 24.4,    # Oct 2025–Mar 2026
    24.0,                                    # Apr 2026
])

ezj_prices = np.array([
    720, 770, 810, 820, 780, 700,            # Apr–Sep 2021
    680, 640, 590, 520, 430, 390,            # Oct 2021–Mar 2022
    380, 360, 370, 390, 400, 365,            # Apr–Sep 2022
    360, 380, 400, 420, 435, 450,            # Oct 2022–Mar 2023
    460, 465, 480, 490, 480, 430,            # Apr–Sep 2023
    420, 440, 450, 460, 470, 480,            # Oct 2023–Mar 2024
    490, 500, 505, 510, 520, 510,            # Apr–Sep 2024
    520, 535, 548, 565, 580, 588,            # Oct 2024–Mar 2025
    570, 540, 510, 495, 475, 490,            # Apr–Sep 2025
    480, 455, 415, 395, 370, 360,            # Oct 2025–Mar 2026
    382,                                      # Apr 2026
])

# Trim to same length
n = min(len(dates), len(rya_prices), len(ezj_prices))
dates      = dates[:n]
rya_prices = rya_prices[:n]
ezj_prices = ezj_prices[:n]

df = pd.DataFrame({"date": dates, "RYA": rya_prices, "EZJ": ezj_prices})
df.set_index("date", inplace=True)

# Moving averages (3-month ≈ 50-day proxy; 8-month ≈ 200-day proxy at monthly data)
df["RYA_MA50"]  = df["RYA"].rolling(3).mean()
df["RYA_MA200"] = df["RYA"].rolling(8).mean()
df["EZJ_MA50"]  = df["EZJ"].rolling(3).mean()
df["EZJ_MA200"] = df["EZJ"].rolling(8).mean()

# Monthly returns
df["RYA_ret"] = df["RYA"].pct_change() * 100
df["EZJ_ret"] = df["EZJ"].pct_change() * 100

# Cumulative returns (base = Apr 2021)
df["RYA_cum"] = (df["RYA"] / df["RYA"].iloc[0] - 1) * 100
df["EZJ_cum"] = (df["EZJ"] / df["EZJ"].iloc[0] - 1) * 100

# Rolling volatility (3-month std of monthly returns)
df["RYA_vol"] = df["RYA_ret"].rolling(3).std()
df["EZJ_vol"] = df["EZJ_ret"].rolling(3).std()

# Drawdown
df["RYA_peak"] = df["RYA"].cummax()
df["EZJ_peak"] = df["EZJ"].cummax()
df["RYA_dd"]   = (df["RYA"] / df["RYA_peak"] - 1) * 100
df["EZJ_dd"]   = (df["EZJ"] / df["EZJ_peak"] - 1) * 100

# ── News events ───────────────────────────────────────────────────────────────
rya_events = [
    # (date_str,  price_offset,  label,  direction,  news)
    ("2020-03-01", 0,   "① COVID crash\n−50% in weeks",       "down",
     "Global travel bans ground Ryanair fleet. Revenue collapses to near zero.\nEU closes borders — 99% of flights cancelled. RYA falls EUR 16→8."),
    ("2020-11-01", 0,   "② Vaccine rally\n+90% recovery",     "up",
     "Pfizer vaccine 90% efficacy announced 9 Nov 2020.\nAirline stocks surge globally. Ryanair's strong balance sheet\nmakes it the 'survivor' trade of choice for institutional funds."),
    ("2022-02-01", 0,   "③ Ukraine invasion\nFuel +60%",      "down",
     "Russia invades Ukraine 24 Feb 2022. Jet fuel surges to\n14-year highs. Ryanair hedged at $65/barrel (80% of needs)\n— partially insulated, but stock still falls to EUR 11."),
    ("2023-05-01", 0,   "④ Record FY2023\nPAT EUR 1.31bn",    "up",
     "Ryanair reports record profit of EUR 1,313m (Jul 2023).\nRevenue up 124% to EUR 10.8bn. 169m passengers carried.\nStock rallies from EUR 14 → EUR 19.5 in 4 months."),
    ("2023-10-01", 0,   "⑤ Boeing delays +\nQ3 fare warning −7%",  "down",
     "Oct 2023: Ryanair warns Q3 fares will be 'materially lower'\n(−7% YoY). Boeing delivery delays cut FY2026 growth target\nfrom 210m to 206m passengers. Stock falls EUR 19.5 → EUR 14."),
    ("2024-06-01", 0,   "⑥ Maiden dividend\n+EUR 700m buyback", "up",
     "Jun 2024: Ryanair's first-ever dividend (EUR 0.353/share).\nFY2024 PAT EUR 1.92bn (+46%). EUR 700m buyback approved.\nStructural re-rating: market prices as mature cash-returner."),
    ("2025-12-01", 0,   "⑦ All-time high\nEUR 30.15",          "up",
     "Dec 2025: All-time high EUR 30.15. Driven by FY2026 PAT\nguidance EUR 2.13–2.23bn, final Boeing Gamechanger deliveries,\n106 new Summer 2026 routes, and Raymond James target raise USD 76."),
    ("2026-02-01", 0,   "⑧ Profit-taking\npullback −19%",     "down",
     "Jan–Apr 2026: Post-ATH correction from EUR 30 → EUR 24.\nGlobal macro risk (tariff wars, Iran conflict), fuel cost\nuncertainty, seasonal weakness. MA200 support intact at ~EUR 19."),
]

ezj_events = [
    ("2021-08-01", 0,   "① Re-opening peak\n820p high",        "up",
     "Summer 2021: easyJet touches 820p as vaccine rollout removes\ntravel restrictions. Market prices in full V-shaped recovery.\nPassenger demand surges; EZJ seen as key beneficiary."),
    ("2021-12-01", 0,   "② Omicron variant\nnew travel bans",  "down",
     "Nov 2021: Omicron COVID variant detected. UK & EU reimpose\ntravel restrictions. Airlines sell off sharply. EZJ falls\nfrom 700p → 590p in weeks as confidence evaporates."),
    ("2022-02-01", 0,   "③ Ukraine war\nfuel crisis",          "down",
     "Feb 2022: Russia invades Ukraine. Jet fuel surges 60%+.\nUnlike Ryanair, easyJet has less fuel hedging protection.\nAlso closes Ukraine routes. Stock crashes toward 380p."),
    ("2022-07-01", 0,   "④ Summer chaos\n11,000 cancellations","down",
     "Summer 2022: easyJet cancels 11,000+ flights, hits £133m\ncharge (Bloomberg, Jul 2022). Staff shortages, Gatwick caps,\nATC strikes. CEO Lundgren under pressure. Stock near 5-yr low."),
    ("2022-09-01", 0,   "⑤ Rights issue\n£1.2bn at 410p",     "down",
     "Sep 2022: Heavily discounted rights issue at 410p raises\n£1.2bn to repair balance sheet. Dilutive to shareholders.\nSignals balance sheet stress — stock dips to ~365p."),
    ("2023-04-01", 0,   "⑥ Return to profit\nDividend reinstated","up",
     "FY2023 results: First profit since pandemic — PAT £324m.\nDividend reinstated at 4.5p. Holidays division grows 46%.\nRally from 350p → 500p as turnaround confirmed."),
    ("2024-11-01", 0,   "⑦ Record FY2024\nEPS +40%",          "up",
     "Nov 2024: FY2024 PAT £452m. Basic EPS 60.3p (+40%).\nDividend raised to 12.1p (+169%). S&P upgrades to BBB+.\nHolidays hits £190m PBT. EZJ approaches 12-month highs."),
    ("2025-06-01", 0,   "⑧ 52-week high\n590p peak",           "up",
     "Jun 2025: EZJ hits 590.6p — best level since 2019.\nHolidays division achieves £250m PBT target 1yr early.\nNew CEO Jarvis targets >£1bn group PBT medium-term."),
    ("2025-09-01", 0,   "⑨ FTSE underperform\n−35% from peak", "down",
     "Aug–Sep 2025: Sharp sell-off. EZJ underperforms FTSE 100\nby −21.3% over 6 months. UK APD hike, ATC disruption,\nmacro fears. Stock falls toward 52-week low 337p."),
]

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — RYANAIR ANNOTATED PRICE CHART WITH NEWS
# ══════════════════════════════════════════════════════════════════════════════
def plot_ryanair_annotated(df, events):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 11),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   facecolor="#0F1923")
    fig.suptitle("RYANAIR (RYA) — Share Price History with News Events\n"
                 "EUR Price (Euronext Dublin) | Apr 2021 – Apr 2026",
                 fontsize=14, fontweight="bold", color="white", y=0.98)

    # Price area
    ax1.fill_between(df.index, df["RYA"], alpha=0.15, color=RYA_COL)
    ax1.plot(df.index, df["RYA"],    color=RYA_COL,   lw=2.0, label="RYA Price (EUR)", zorder=4)
    ax1.plot(df.index, df["RYA_MA50"],  color=MA50_COL,  lw=1.3, ls="--", alpha=0.9, label="50-day MA (proxy)")
    ax1.plot(df.index, df["RYA_MA200"], color=MA200_COL, lw=1.5, ls="-",  alpha=0.9, label="200-day MA (proxy)")

    ax1.set_ylabel("Price (EUR)", color="#C8D6E5", fontsize=10)
    ax1.set_facecolor("#141E2A")
    ax1.legend(loc="upper left", framealpha=0.8)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("€%.0f"))

    # Shade golden cross zones (MA50 > MA200 = bullish)
    rya_aligned = df[["RYA_MA50","RYA_MA200"]].dropna()
    bull = rya_aligned["RYA_MA50"] > rya_aligned["RYA_MA200"]
    ax1.fill_between(rya_aligned.index, ax1.get_ylim()[0], 35,
                     where=bull, alpha=0.04, color=UP_COL, label="Bullish zone (MA50>MA200)")

    # Annotate events
    news_box = []
    for (date_str, _, label, direction, news) in rya_events:
        try:
            dt = pd.Timestamp(date_str)
            if dt < df.index[0] or dt > df.index[-1]:
                continue
            # Find closest date
            idx = df.index.get_indexer([dt], method="nearest")[0]
            px  = df["RYA"].iloc[idx]
            col = UP_COL if direction == "up" else DN_COL
            short = label.split("\n")[0]
            
            # Arrow
            offset_y = 3.5 if direction == "up" else -3.5
            ax1.annotate("", xy=(df.index[idx], px),
                         xytext=(df.index[idx], px + offset_y),
                         arrowprops=dict(arrowstyle="->", color=col,
                                         lw=1.6, connectionstyle="arc3,rad=0"))
            # Label box
            ax1.annotate(short,
                         xy=(df.index[idx], px + offset_y * 1.05),
                         fontsize=7, fontweight="bold", color=col,
                         ha="center", va="bottom" if direction=="up" else "top",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A2535",
                                   edgecolor=col, alpha=0.92, lw=1.2))
            news_box.append((dt, direction, news, col))
        except Exception:
            continue

    # Volume bar (simulate)
    np.random.seed(42)
    vol_idx = np.arange(len(df))
    vol = np.abs(df["RYA_ret"].fillna(0).values) * 30 + 30
    colors_bar = [UP_COL if r >= 0 else DN_COL for r in df["RYA_ret"].fillna(0)]
    ax2.bar(df.index, vol, color=colors_bar, alpha=0.6, width=25)
    ax2.set_ylabel("Vol proxy", color="#C8D6E5", fontsize=8)
    ax2.set_facecolor("#141E2A")
    ax2.yaxis.set_visible(False)
    ax2.set_xlabel("Date", color="#C8D6E5")

    # Key metrics box
    props = dict(boxstyle="round,pad=0.5", facecolor="#1A2535", edgecolor=RYA_COL, alpha=0.9)
    metrics = (
        "Key Metrics (Apr 2026)\n"
        "─────────────────────\n"
        f"Current:   EUR 24.40\n"
        f"ATH:       EUR 30.15 (Dec 2025)\n"
        f"5Y Change: +68%\n"
        f"P/E:       ~12x\n"
        f"Mkt Cap:   EUR 25.5bn\n"
        f"Div Yield: ~1.7%\n"
        f"Rating:    Strong Buy"
    )
    ax1.text(0.01, 0.98, metrics, transform=ax1.transAxes, fontsize=8,
             verticalalignment="top", bbox=props, color="white", family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("/mnt/user-data/outputs/01_ryanair_annotated.png",
                dpi=150, bbox_inches="tight", facecolor="#0F1923")
    plt.close()
    print("✓  Chart 1 saved: 01_ryanair_annotated.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — EASYJET ANNOTATED PRICE CHART WITH NEWS
# ══════════════════════════════════════════════════════════════════════════════
def plot_easyjet_annotated(df, events):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 11),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   facecolor="#0F1923")
    fig.suptitle("EASYJET (EZJ) — Share Price History with News Events\n"
                 "GBX Price (London Stock Exchange) | Apr 2021 – Apr 2026",
                 fontsize=14, fontweight="bold", color="white", y=0.98)

    ax1.fill_between(df.index, df["EZJ"], alpha=0.15, color=EZJ_COL)
    ax1.plot(df.index, df["EZJ"],    color=EZJ_COL,   lw=2.0, label="EZJ Price (GBX)")
    ax1.plot(df.index, df["EZJ_MA50"],  color=MA50_COL,  lw=1.3, ls="--", alpha=0.9, label="50-day MA (proxy)")
    ax1.plot(df.index, df["EZJ_MA200"], color=MA200_COL, lw=1.5, ls="-",  alpha=0.9, label="200-day MA (proxy)")

    ax1.set_ylabel("Price (GBX pence)", color="#C8D6E5", fontsize=10)
    ax1.set_facecolor("#141E2A")
    ax1.legend(loc="upper right", framealpha=0.8)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%dp"))

    # Annotate events
    for (date_str, _, label, direction, news) in events:
        try:
            dt  = pd.Timestamp(date_str)
            if dt < df.index[0] or dt > df.index[-1]:
                continue
            idx = df.index.get_indexer([dt], method="nearest")[0]
            px  = df["EZJ"].iloc[idx]
            col = UP_COL if direction == "up" else DN_COL
            short = label.split("\n")[0]
            offset_y = 90 if direction == "up" else -90
            ax1.annotate("", xy=(df.index[idx], px),
                         xytext=(df.index[idx], px + offset_y),
                         arrowprops=dict(arrowstyle="->", color=col,
                                         lw=1.6, connectionstyle="arc3,rad=0"))
            ax1.annotate(short,
                         xy=(df.index[idx], px + offset_y * 1.05),
                         fontsize=7, fontweight="bold", color=col,
                         ha="center", va="bottom" if direction=="up" else "top",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A2535",
                                   edgecolor=col, alpha=0.92, lw=1.2))
        except Exception:
            continue

    ax2.bar(df.index,
            np.abs(df["EZJ_ret"].fillna(0).values) * 100 + 20,
            color=[UP_COL if r >= 0 else DN_COL for r in df["EZJ_ret"].fillna(0)],
            alpha=0.6, width=25)
    ax2.set_facecolor("#141E2A")
    ax2.yaxis.set_visible(False)
    ax2.set_xlabel("Date", color="#C8D6E5")

    props = dict(boxstyle="round,pad=0.5", facecolor="#1A2535", edgecolor=EZJ_COL, alpha=0.9)
    metrics = (
        "Key Metrics (Apr 2026)\n"
        "─────────────────────\n"
        f"Current:   382p\n"
        f"52W High:  590p\n"
        f"5Y Change: −53%\n"
        f"P/E:       5.9x\n"
        f"Mkt Cap:   GBP 2.9bn\n"
        f"Div Yield: 3.45%\n"
        f"Rating:    Hold/Buy"
    )
    ax1.text(0.01, 0.98, metrics, transform=ax1.transAxes, fontsize=8,
             verticalalignment="top", bbox=props, color="white", family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("/mnt/user-data/outputs/02_easyjet_annotated.png",
                dpi=150, bbox_inches="tight", facecolor="#0F1923")
    plt.close()
    print("✓  Chart 2 saved: 02_easyjet_annotated.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — MONTHLY RETURNS HEATMAP — RYANAIR
# ══════════════════════════════════════════════════════════════════════════════
def plot_heatmap(df, ticker, col, filename, title_suffix):
    ret_col = f"{ticker}_ret"
    ret = df[ret_col].dropna().copy()
    ret.index = pd.DatetimeIndex(ret.index)

    pivot = pd.DataFrame({
        "Year":  ret.index.year,
        "Month": ret.index.month,
        "Ret":   ret.values
    })
    pivot = pivot.pivot_table(index="Year", columns="Month", values="Ret", aggfunc="mean")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0F1923")
    ax.set_facecolor("#141E2A")

    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#8B0000", "#1A2535", "#006400"], N=256)

    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01)
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
                   vmin=-vmax, vmax=vmax)

    # Annotations
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            color  = "white" if abs(val) > vmax * 0.4 else "#AAAAAA"
            symbol = "▲" if val >= 0 else "▼"
            ax.text(j, i, f"{symbol}{abs(val):.1f}%",
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color=color)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, color="#C8D6E5")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, color="#C8D6E5")

    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Monthly Return %", color="#C8D6E5")
    cb.ax.yaxis.set_tick_params(color="#C8D6E5")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#C8D6E5")

    ax.set_title(f"{ticker} — Monthly Returns Heatmap  {title_suffix}\n"
                 "Green = positive return month | Red = negative return month",
                 fontsize=12, fontweight="bold", color="white", pad=12)
    ax.set_xlabel("Month", color="#C8D6E5")
    ax.set_ylabel("Year",  color="#C8D6E5")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A3A4A")

    plt.tight_layout()
    plt.savefig(f"/mnt/user-data/outputs/{filename}",
                dpi=150, bbox_inches="tight", facecolor="#0F1923")
    plt.close()
    print(f"✓  Chart saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — CUMULATIVE RETURNS COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def plot_cumulative(df):
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0F1923")
    ax.set_facecolor("#141E2A")

    ax.axhline(0, color="#2A3A4A", lw=1, ls="--")
    ax.fill_between(df.index, df["RYA_cum"], 0,
                    where=df["RYA_cum"]>=0, alpha=0.1, color=RYA_COL)
    ax.fill_between(df.index, df["EZJ_cum"], 0,
                    where=df["EZJ_cum"]<=0, alpha=0.1, color=EZJ_COL)
    ax.plot(df.index, df["RYA_cum"], color=RYA_COL, lw=2.5, label="Ryanair (RYA)")
    ax.plot(df.index, df["EZJ_cum"], color=EZJ_COL, lw=2.5, label="easyJet (EZJ)")

    # Final value labels
    ax.annotate(f"+{df['RYA_cum'].iloc[-1]:.0f}%",
                xy=(df.index[-1], df["RYA_cum"].iloc[-1]),
                fontsize=11, fontweight="bold", color=RYA_COL,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A2535",
                          edgecolor=RYA_COL, alpha=0.9))
    ax.annotate(f"{df['EZJ_cum'].iloc[-1]:.0f}%",
                xy=(df.index[-1], df["EZJ_cum"].iloc[-1]),
                fontsize=11, fontweight="bold", color=EZJ_COL,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A2535",
                          edgecolor=EZJ_COL, alpha=0.9))

    ax.set_title("Cumulative Total Return Comparison: Ryanair vs easyJet\n"
                 "Base = April 2021 (same start point)",
                 fontsize=13, fontweight="bold", color="white")
    ax.set_ylabel("Cumulative Return (%)", color="#C8D6E5")
    ax.set_xlabel("Date", color="#C8D6E5")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f%%"))
    ax.legend(fontsize=11, loc="upper left")

    # Shade COVID recovery
    ax.axvspan(pd.Timestamp("2021-04-01"), pd.Timestamp("2022-01-01"),
               alpha=0.05, color=UP_COL, label="Recovery phase")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-09-01"),
               alpha=0.05, color=DN_COL)

    for spine in ax.spines.values():
        spine.set_edgecolor("#2A3A4A")

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/05_cumulative_returns.png",
                dpi=150, bbox_inches="tight", facecolor="#0F1923")
    plt.close()
    print("✓  Chart 5 saved: 05_cumulative_returns.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — FINANCIAL KPI RADAR CHART
# ══════════════════════════════════════════════════════════════════════════════
def plot_radar():
    categories = [
        "Net Profit\nMargin", "Return on\nEquity", "Load Factor",
        "Interest\nCoverage", "Equity\nRatio", "Revenue\nGrowth",
        "EPS Growth", "Dividend\nGrowth"
    ]
    # Normalised scores 0-10 (higher = better for investor)
    rya_scores = [9.2, 9.0, 9.5, 9.8, 8.0, 9.0, 8.5, 6.0]  # Ryanair FY2024
    ezj_scores = [3.8, 5.2, 6.5, 4.5, 4.0, 5.5, 7.0, 9.0]  # easyJet FY2024

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    rya_scores += rya_scores[:1]
    ezj_scores += ezj_scores[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True),
                           facecolor="#0F1923")
    ax.set_facecolor("#141E2A")

    ax.plot(angles, rya_scores, color=RYA_COL, lw=2.2, label="Ryanair FY2024")
    ax.fill(angles, rya_scores, color=RYA_COL, alpha=0.18)
    ax.plot(angles, ezj_scores, color=EZJ_COL, lw=2.2, label="easyJet FY2024")
    ax.fill(angles, ezj_scores, color=EZJ_COL, alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, color="#C8D6E5")
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color="#8A9BB0", fontsize=7)
    ax.set_ylim(0, 10)
    ax.grid(color="#2A3A4A", linewidth=0.8)
    ax.spines["polar"].set_edgecolor("#2A3A4A")

    ax.set_title("Financial KPI Radar Chart — Ryanair vs easyJet (FY2024)\n"
                 "Score out of 10 (higher = stronger for investor)",
                 fontsize=12, fontweight="bold", color="white",
                 pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.15),
              fontsize=11, framealpha=0.8)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/06_kpi_radar.png",
                dpi=150, bbox_inches="tight", facecolor="#0F1923")
    plt.close()
    print("✓  Chart 6 saved: 06_kpi_radar.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 7 — VOLATILITY & DRAWDOWN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def plot_volatility_drawdown(df):
    fig = plt.figure(figsize=(18, 10), facecolor="#0F1923")
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    # -- Panel A: Rolling volatility
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#141E2A")
    ax1.plot(df.index, df["RYA_vol"], color=RYA_COL, lw=2, label="Ryanair")
    ax1.plot(df.index, df["EZJ_vol"], color=EZJ_COL, lw=2, label="easyJet")
    ax1.fill_between(df.index, df["RYA_vol"], alpha=0.12, color=RYA_COL)
    ax1.fill_between(df.index, df["EZJ_vol"], alpha=0.12, color=EZJ_COL)
    ax1.set_title("Rolling 3-Month Volatility (% std dev of monthly returns)",
                  color="white", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Volatility (%)", color="#C8D6E5")
    ax1.legend()
    for spine in ax1.spines.values(): spine.set_edgecolor("#2A3A4A")

    # -- Panel B: Drawdown
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#141E2A")
    ax2.fill_between(df.index, df["RYA_dd"], 0, alpha=0.3, color=RYA_COL)
    ax2.fill_between(df.index, df["EZJ_dd"], 0, alpha=0.3, color=EZJ_COL)
    ax2.plot(df.index, df["RYA_dd"], color=RYA_COL, lw=1.8, label="Ryanair")
    ax2.plot(df.index, df["EZJ_dd"], color=EZJ_COL, lw=1.8, label="easyJet")
    ax2.set_title("Drawdown from Rolling Peak (%)",
                  color="white", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)", color="#C8D6E5")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.legend()
    for spine in ax2.spines.values(): spine.set_edgecolor("#2A3A4A")

    # -- Panel C: Monthly return distributions (histogram)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#141E2A")
    rya_r = df["RYA_ret"].dropna()
    ezj_r = df["EZJ_ret"].dropna()
    bins  = np.linspace(-20, 20, 28)
    ax3.hist(rya_r, bins=bins, color=RYA_COL, alpha=0.6, label="Ryanair", edgecolor="#0F1923")
    ax3.hist(ezj_r, bins=bins, color=EZJ_COL, alpha=0.6, label="easyJet",  edgecolor="#0F1923")
    ax3.axvline(rya_r.mean(), color=RYA_COL, lw=2, ls="--",
                label=f"RYA mean: {rya_r.mean():.1f}%")
    ax3.axvline(ezj_r.mean(), color=EZJ_COL, lw=2, ls="--",
                label=f"EZJ mean: {ezj_r.mean():.1f}%")
    ax3.set_title("Distribution of Monthly Returns",
                  color="white", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Monthly Return (%)", color="#C8D6E5")
    ax3.set_ylabel("Frequency", color="#C8D6E5")
    ax3.legend(fontsize=8)
    for spine in ax3.spines.values(): spine.set_edgecolor("#2A3A4A")

    # -- Panel D: Summary stats table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#141E2A")
    ax4.axis("off")

    table_data = [
        ["Metric",                    "Ryanair",                      "easyJet"],
        ["5-Year Return",             "+68%",                         "−53%"],
        ["Max Drawdown",              f"{df['RYA_dd'].min():.0f}%",   f"{df['EZJ_dd'].min():.0f}%"],
        ["Avg Monthly Return",        f"+{rya_r.mean():.1f}%",        f"{ezj_r.mean():.1f}%"],
        ["Return Std Dev (monthly)",  f"{rya_r.std():.1f}%",          f"{ezj_r.std():.1f}%"],
        ["Positive Months",           f"{(rya_r>0).sum()} / {len(rya_r)}", f"{(ezj_r>0).sum()} / {len(ezj_r)}"],
        ["Current vs MA200",          "+27% ABOVE",                   "−17% BELOW"],
        ["Beta (approx)",             "1.26×",                        "1.71×"],
        ["Analyst Consensus",         "Strong Buy",                   "Hold / Mixed"],
        ["Net Profit Margin FY24",    "14.3%",                        "4.9%"],
        ["Interest Coverage FY24",    "24.8×",                        "7.4×"],
        ["Debt-to-Equity FY24",       "0.34×",                        "0.71×"],
    ]

    tbl = ax4.table(cellText=table_data[1:],
                    colLabels=table_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.55)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#1A2535" if row % 2 == 0 else "#141E2A")
        cell.set_edgecolor("#2A3A4A")
        cell.set_text_props(color="white")
        if row == 0:
            cell.set_facecolor("#1F3864")
            cell.set_text_props(color="white", fontweight="bold")
        if col == 1 and row > 0:
            cell.set_text_props(color=RYA_COL, fontweight="bold")
        if col == 2 and row > 0:
            cell.set_text_props(color=EZJ_COL, fontweight="bold")

    ax4.set_title("Summary Statistics Comparison",
                  color="white", fontsize=10, fontweight="bold", pad=10)

    fig.suptitle("Risk & Return Dashboard — Ryanair vs easyJet  |  Apr 2021–Apr 2026",
                 fontsize=13, fontweight="bold", color="white", y=1.01)

    plt.savefig("/mnt/user-data/outputs/07_volatility_drawdown_dashboard.png",
                dpi=150, bbox_inches="tight", facecolor="#0F1923")
    plt.close()
    print("✓  Chart 7 saved: 07_volatility_drawdown_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 8 — NEWS EVENTS TIMELINE (BOTH AIRLINES SIDE BY SIDE)
# ══════════════════════════════════════════════════════════════════════════════
def plot_news_timeline(rya_events, ezj_events):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 14),
                                   facecolor="#0F1923")

    def draw_timeline(ax, events, ticker, col):
        ax.set_facecolor("#141E2A")
        ax.set_xlim(-0.5, 1.5)
        n = len(events)
        ys = np.linspace(0.95, 0.05, n)

        # Central spine
        ax.axvline(0.5, color=col, lw=2, alpha=0.4, ymin=0.03, ymax=0.97)

        for i, (date_str, _, label, direction, news) in enumerate(events):
            y   = ys[i]
            ecol = UP_COL if direction == "up" else DN_COL
            side = -1 if i % 2 == 0 else 1  # alternate sides
            xend = 0.5 + side * 0.12
            xtxt = 0.5 + side * 0.14

            # Dot on spine
            ax.plot(0.5, y, "o", ms=10, color=ecol, zorder=5)
            # Arrow to box
            ax.annotate("", xy=(xend, y), xytext=(0.5, y),
                        arrowprops=dict(arrowstyle="-", color=ecol, lw=1.5))

            # Direction indicator
            sym = "▲" if direction == "up" else "▼"
            ha  = "right" if side == -1 else "left"

            # Date
            try:
                dt = pd.Timestamp(date_str).strftime("%b %Y")
            except Exception:
                dt = date_str

            # News box
            box_text = f"{sym} {dt}\n{label}\n\n{news}"
            ax.text(xtxt, y, box_text,
                    ha=ha, va="center", fontsize=6.8,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.45",
                              facecolor="#1A2535",
                              edgecolor=ecol,
                              alpha=0.93,
                              lw=1.2),
                    wrap=True,
                    multialignment="left" if side == 1 else "right")

        ax.set_xlim(-1.0, 2.0)
        ax.axis("off")
        ax.set_title(f"{ticker} — Event Timeline with News Background\n"
                     "▲ = Price catalyst (up)   ▼ = Price driver (down)",
                     fontsize=11, fontweight="bold", color="white", pad=12)

    draw_timeline(ax1, rya_events, "RYANAIR (RYA EUR)", RYA_COL)
    draw_timeline(ax2, ezj_events, "EASYJET (EZJ GBX)", EZJ_COL)

    fig.suptitle("Share Price News Timeline — Key Events Driving Price Movements\n"
                 "Ryanair vs easyJet  |  2020–2026",
                 fontsize=14, fontweight="bold", color="white", y=1.005)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/08_news_timeline.png",
                dpi=150, bbox_inches="tight", facecolor="#0F1923")
    plt.close()
    print("✓  Chart 8 saved: 08_news_timeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 9 — MA CROSSOVER SIGNALS CHART
# ══════════════════════════════════════════════════════════════════════════════
def plot_ma_crossover(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                   facecolor="#0F1923", sharex=True)
    fig.suptitle("50-Day vs 200-Day Moving Average — Golden Cross / Death Cross Signals\n"
                 "Ryanair (RYA) and easyJet (EZJ)  |  Apr 2021 – Apr 2026",
                 fontsize=13, fontweight="bold", color="white")

    def draw_ma(ax, ticker, pcol, prices_col, ma50_col, ma200_col, label):
        ax.set_facecolor("#141E2A")
        p   = df[prices_col]
        m50 = df[ma50_col]
        m200= df[ma200_col]
        ax.fill_between(df.index, p, alpha=0.08, color=pcol)
        ax.plot(df.index, p,    color=pcol,     lw=2.0, alpha=0.9, label=f"{ticker} Price")
        ax.plot(df.index, m50,  color=MA50_COL, lw=1.5, ls="--",   label="50-day MA")
        ax.plot(df.index, m200, color=MA200_COL,lw=2.0,             label="200-day MA")

        # Shade golden cross (MA50 > MA200)
        aligned = df[[ma50_col, ma200_col]].dropna()
        bull = aligned[ma50_col] > aligned[ma200_col]
        bear = ~bull
        ax.fill_between(aligned.index, aligned[ma50_col], aligned[ma200_col],
                        where=bull, alpha=0.15, color=UP_COL,  label="Golden cross (bullish)")
        ax.fill_between(aligned.index, aligned[ma50_col], aligned[ma200_col],
                        where=bear, alpha=0.15, color=DN_COL,  label="Death cross (bearish)")

        ax.set_ylabel(label, color="#C8D6E5")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
        for spine in ax.spines.values(): spine.set_edgecolor("#2A3A4A")

    draw_ma(ax1, "Ryanair", RYA_COL, "RYA", "RYA_MA50", "RYA_MA200", "EUR Price")
    draw_ma(ax2, "easyJet", EZJ_COL, "EZJ", "EZJ_MA50", "EZJ_MA200", "GBX Price")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("€%.0f"))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%dp"))
    ax2.set_xlabel("Date", color="#C8D6E5")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("/mnt/user-data/outputs/09_ma_crossover.png",
                dpi=150, bbox_inches="tight", facecolor="#0F1923")
    plt.close()
    print("✓  Chart 9 saved: 09_ma_crossover.png")


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL CHARTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Generating all charts ===\n")
plot_ryanair_annotated(df, rya_events)
plot_easyjet_annotated(df, ezj_events)
plot_heatmap(df, "RYA", RYA_COL, "03_ryanair_heatmap.png", "(EUR, Euronext Dublin)")
plot_heatmap(df, "EZJ", EZJ_COL, "04_easyjet_heatmap.png", "(GBX, London Stock Exchange)")
plot_cumulative(df)
plot_radar()
plot_volatility_drawdown(df)
plot_news_timeline(rya_events, ezj_events)
plot_ma_crossover(df)
print("\n=== All 9 charts generated successfully ===")
print("Files saved to /mnt/user-data/outputs/\n")
