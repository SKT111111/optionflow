import io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm

import streamlit as st


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="GEX Profile Viewer",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# 日本語フォント設定
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
FONT_PATH = BASE_DIR / "fonts" / "NotoSansJP-Regular.ttf"


def setup_japanese_font():
    if FONT_PATH.exists():
        fm.fontManager.addfont(str(FONT_PATH))
        font_name = fm.FontProperties(fname=str(FONT_PATH)).get_name()
        matplotlib.rcParams["font.family"] = font_name
        matplotlib.rcParams["axes.unicode_minus"] = False
        return font_name, True

    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None, False


FONT_NAME, FONT_FOUND = setup_japanese_font()


# =========================================================
# 定数
# =========================================================
DEFAULT_RISK_FREE = 0.045
DEFAULT_DIV_YIELD = 0.0

PATTERNS = [
    ("当日 (0DTE)",      (0, 1)),    # Exp Date から計算するDTEは小数のため1日以内で判定
    ("1週間以内 (0-7)",  (0, 7)),
    ("1ヶ月以内 (0-30)", (0, 30)),
    ("全満期",           (0, None)),
]


# =========================================================
# 入力読み込み
# =========================================================
def _strip_barchart_footer(df):
    """
    Barchart CSV は末尾に
    'Downloaded from Barchart.com as of ...'
    という1列だけの行が入るので除去する。
    """
    if df.empty:
        return df

    first_col = df.columns[0]
    mask = df[first_col].astype(str).str.contains(
        "Downloaded from Barchart", case=False, na=False
    )
    if mask.any():
        df = df[~mask].copy()

    return df


def read_uploaded_table(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".csv") or name.endswith(".txt"):
        df = pd.read_csv(uploaded_file)
        return _strip_barchart_footer(df)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
        return _strip_barchart_footer(df)

    raise ValueError("未対応のファイル形式です。.csv / .xlsx / .xls を使ってください。")


def validate_required_columns(df):
    required = [
        "Price~",
        "Strike",
        "Open Int",
        "DTE",
        "IV",
        "Volume",
        "Side",
        "Type",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "必要な列が不足しています。\n"
            f"不足列: {missing}\n"
            f"実際の列: {list(df.columns)}"
        )


def load_data_from_upload(uploaded_file):
    df = read_uploaded_table(uploaded_file)
    df.columns = [str(c).strip() for c in df.columns]

    # DTE が無ければ Exp Date から計算して補完する
    if "DTE" not in df.columns and "Exp Date" in df.columns:
        exp = pd.to_datetime(df["Exp Date"], errors="coerce", utc=True)
        now = pd.Timestamp.now(tz="UTC")
        df["DTE"] = (exp - now).dt.total_seconds() / 86400.0
        df["DTE"] = df["DTE"].clip(lower=0)

    validate_required_columns(df)

    # 主要な数値列がすべて欠損している行（脚注等）を除去する保険
    key_cols = [c for c in ["Price~", "Strike", "Open Int", "DTE"] if c in df.columns]
    if key_cols:
        df = df.dropna(subset=key_cols, how="all").copy()

    if "Symbol" not in df.columns:
        df["Symbol"] = "UNKNOWN"

    df["Price~"] = pd.to_numeric(df["Price~"], errors="coerce")
    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
    df["Open Int"] = pd.to_numeric(df["Open Int"], errors="coerce")
    df["DTE"] = pd.to_numeric(df["DTE"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df["IV"] = (
        df["IV"]
        .astype(str)
        .str.rstrip("%")
        .replace("nan", np.nan)
    )
    df["IV"] = pd.to_numeric(df["IV"], errors="coerce") / 100.0

    for col in ["Side", "Code", "Type", "Symbol"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df = df.dropna(subset=["Strike", "Open Int", "IV", "Price~", "DTE"]).copy()
    df = df[df["IV"] > 0].copy()

    return df


# =========================================================
# 基本情報
# =========================================================
def get_symbol(df):
    if "Symbol" in df.columns and not df["Symbol"].dropna().empty:
        return str(df["Symbol"].dropna().iloc[0]).strip()
    return "UNKNOWN"


def nearest_strike(net, spot):
    if net.empty:
        return None
    idx = (net["Strike"] - spot).abs().idxmin()
    return net.loc[idx, "Strike"]


def filter_dte(df, dte_range):
    lo, hi = dte_range
    out = df[df["DTE"] >= lo]
    if hi is not None:
        out = out[out["DTE"] <= hi]
    return out.copy()


# =========================================================
# フロー集計
# =========================================================
def flow_detail(df_raw, strike, opt_type=None):
    sub = df_raw[df_raw["Strike"] == strike]

    if opt_type is not None:
        sub = sub[sub["Type"].str.lower() == opt_type.lower()]

    if sub.empty:
        return 0, 0, 0

    side = sub["Side"].astype(str).str.strip().str.lower()
    vol = pd.to_numeric(sub["Volume"], errors="coerce").fillna(0)

    ask_v = vol[side == "ask"].sum()
    bid_v = vol[side == "bid"].sum()
    mid_v = vol[side == "mid"].sum()

    return ask_v, bid_v, mid_v


def flow_short(ask, bid, mid):
    if ask > bid:
        tag = "買優勢"
    elif bid > ask:
        tag = "売優勢"
    else:
        tag = "拮抗"

    if mid > max(ask, bid):
        tag += "?"
    return tag


def flow_full(ask, bid, mid):
    if ask > bid:
        s = f"買い優勢(ask{ask:.0f} vs bid{bid:.0f})"
    elif bid > ask:
        s = f"売り優勢(bid{bid:.0f} vs ask{ask:.0f})"
    else:
        s = f"拮抗(ask{ask:.0f} bid{bid:.0f})"

    if mid > 0:
        s += f" mid{mid:.0f}"

    return s


# =========================================================
# 正規分布PDF（scipy不要）
# =========================================================
def norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


# =========================================================
# BS Gamma / GEX
# =========================================================
def bs_gamma(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = np.exp(-q * T) * norm_pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma


def compute_gex(df, S, risk_free, div_yield):
    rows = []

    for _, r in df.iterrows():
        T = max(r["DTE"], 0.0001) / 365.0
        gamma = bs_gamma(S, r["Strike"], T, risk_free, div_yield, r["IV"])
        sign = 1.0 if str(r["Type"]).lower() == "call" else -1.0
        gex = gamma * r["Open Int"] * 100 * (S ** 2) * 0.01 * sign

        rows.append({
            "Strike": r["Strike"],
            "GEX": gex
        })

    if not rows:
        return pd.DataFrame(columns=["Strike", "GEX"])

    g = pd.DataFrame(rows)
    net = g.groupby("Strike", as_index=False)["GEX"].sum()
    net = net.sort_values("Strike").reset_index(drop=True)
    return net


def find_top_walls(net, n=3):
    if net.empty:
        return [], []

    pos = net[net["GEX"] > 0].sort_values("GEX", ascending=False).head(n)
    neg = net[net["GEX"] < 0].sort_values("GEX", ascending=True).head(n)

    call_walls = list(zip(pos["Strike"], pos["GEX"]))
    put_walls = list(zip(neg["Strike"], neg["GEX"]))

    return call_walls, put_walls


def find_hvl(S, net):
    if net.empty:
        return None, 0.0

    net = net.copy()
    net["cum"] = net["GEX"].cumsum()
    crossings = []

    for i in range(1, len(net)):
        y0 = net["cum"].iloc[i - 1]
        y1 = net["cum"].iloc[i]
        x0 = net["Strike"].iloc[i - 1]
        x1 = net["Strike"].iloc[i]

        if (y0 >= 0 and y1 < 0) or (y0 < 0 and y1 >= 0):
            x_cross = x0 + (x1 - x0) * (y0 / (y0 - y1))
            crossings.append(x_cross)

    if crossings:
        hvl = min(crossings, key=lambda x: abs(x - S))
    else:
        hvl = net.loc[(net["Strike"] - S).abs().idxmin(), "Strike"]

    total = net["GEX"].sum()
    return hvl, total


# =========================================================
# 表示用補助
# =========================================================
def get_x_limits(net):
    if net.empty:
        return (-1, 1)

    gex_m = net["GEX"] / 1e6
    raw_xmax = max(float(gex_m.max()), 0.0)
    raw_xmin = min(float(gex_m.min()), 0.0)
    x_span = max(raw_xmax - raw_xmin, 1.0)

    left_pad = x_span * 0.08
    right_pad = x_span * 0.42
    return raw_xmin - left_pad, raw_xmax + right_pad


def get_zoom_ylim(S, call_walls, put_walls, hvl=None):
    focus_levels = [S]
    focus_levels += [k for k, _ in call_walls]
    focus_levels += [k for k, _ in put_walls]
    if hvl is not None:
        # HVLが近い時だけ見切れ防止
        if abs(hvl - S) <= max(S * 0.25, 80):
            focus_levels.append(hvl)

    lo = min(focus_levels)
    hi = max(focus_levels)
    span = max(hi - lo, 1.0)

    pad = max(span * 0.35, S * 0.08, 20.0)
    return max(0, lo - pad), hi + pad


def get_overview_ylim(net, S, call_walls, put_walls, hvl=None, threshold_ratio=0.01):
    """
    全体俯瞰のy軸範囲を、意味のあるデータ帯までに絞る。
    threshold_ratio:
        max(|GEX|) に対する比率。これ未満の極小ノイズstrikeは無視。
    """
    focus_levels = [S]
    focus_levels += [k for k, _ in call_walls]
    focus_levels += [k for k, _ in put_walls]
    if hvl is not None:
        focus_levels.append(hvl)

    if net.empty:
        lo = min(focus_levels)
        hi = max(focus_levels)
        pad = max((hi - lo) * 0.1, 10)
        return max(0, lo - pad), hi + pad

    abs_gex = net["GEX"].abs()
    max_abs = float(abs_gex.max()) if len(abs_gex) else 0.0

    if max_abs <= 0:
        meaningful = net.copy()
    else:
        cutoff = max_abs * threshold_ratio
        meaningful = net[abs_gex >= cutoff].copy()
        if meaningful.empty:
            meaningful = net.copy()

    levels = meaningful["Strike"].tolist() + focus_levels
    lo = min(levels)
    hi = max(levels)
    pad = max((hi - lo) * 0.10, 10.0)

    return max(0, lo - pad), hi + pad


def spread_label_positions(levels, y_min, y_max, min_gap):
    if not levels:
        return {}

    items = sorted(levels, key=lambda x: x[1])
    adjusted = []

    for key, y in items:
        if not adjusted:
            adjusted.append([key, y, y])
        else:
            prev_y = adjusted[-1][2]
            y_text = max(y, prev_y + min_gap)
            adjusted.append([key, y, y_text])

    overflow = adjusted[-1][2] - y_max
    if overflow > 0:
        for item in adjusted:
            item[2] -= overflow

    underflow = y_min - adjusted[0][2]
    if underflow > 0:
        for item in adjusted:
            item[2] += underflow

    return {key: y_text for key, _, y_text in adjusted}


def build_label_items(sub_raw, call_walls, put_walls, hvl, S, spot_flow):
    items = []

    for j, (k, g) in enumerate(call_walls, 1):
        a, b, m = flow_detail(sub_raw, k, "Call")
        fl = flow_short(a, b, m)
        items.append({
            "key": f"cw{j}",
            "y": k,
            "text": f"CW{j}: {k:.1f} [{fl}]",
            "color": "#006400" if j == 1 else "#2ca02c",
            "linecolor": "#006400" if j == 1 else "#2ca02c",
            "linewidth": 2.0 if j == 1 else 1.1,
            "linestyle": "-" if j == 1 else ":",
            "fontsize": 7.2 if j == 1 else 6.6,
        })

    for j, (k, g) in enumerate(put_walls, 1):
        a, b, m = flow_detail(sub_raw, k, "Put")
        fl = flow_short(a, b, m)
        items.append({
            "key": f"pw{j}",
            "y": k,
            "text": f"PW{j}: {k:.1f} [{fl}]",
            "color": "#8b0000" if j == 1 else "#d62728",
            "linecolor": "#8b0000" if j == 1 else "#d62728",
            "linewidth": 2.0 if j == 1 else 1.1,
            "linestyle": "-" if j == 1 else ":",
            "fontsize": 7.2 if j == 1 else 6.6,
        })

    if hvl is not None:
        items.append({
            "key": "hvl",
            "y": hvl,
            "text": f"HVL: {hvl:.1f}",
            "color": "#1f77b4",
            "linecolor": "#1f77b4",
            "linewidth": 2.0,
            "linestyle": "--",
            "fontsize": 8.0,
        })

    items.append({
        "key": "spot",
        "y": S,
        "text": f"Spot: {S:.2f} [ATM近傍: {spot_flow}]",
        "color": "black",
        "linecolor": "black",
        "linewidth": 1.8,
        "linestyle": ":",
        "fontsize": 7.2,
    })

    return items


def draw_bars(ax, net):
    if net.empty:
        return

    tmp = net.copy()
    tmp["GEX_M"] = tmp["GEX"] / 1e6
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in tmp["GEX_M"]]

    diff_med = tmp["Strike"].diff().median()
    if pd.isna(diff_med) or diff_med <= 0:
        bar_h = 0.8
    else:
        bar_h = diff_med * 0.8

    ax.barh(
        tmp["Strike"],
        tmp["GEX_M"],
        color=colors,
        height=bar_h,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.85,
        zorder=2
    )


def draw_reference_lines(ax, label_items, show_legend=False):
    legend_used = {"cw": False, "pw": False, "hvl": False, "spot": False}

    for item in label_items:
        key = item["key"]
        y = item["y"]
        legend_label = None

        if key.startswith("cw") and not legend_used["cw"]:
            legend_label = f"Call Wall  {y:.1f}"
            legend_used["cw"] = True
        elif key.startswith("pw") and not legend_used["pw"]:
            legend_label = f"Put Wall   {y:.1f}"
            legend_used["pw"] = True
        elif key == "hvl" and not legend_used["hvl"]:
            legend_label = f"HVL        {y:.1f}"
            legend_used["hvl"] = True
        elif key == "spot" and not legend_used["spot"]:
            legend_label = f"Spot       {y:.2f}"
            legend_used["spot"] = True

        ax.axhline(
            y,
            color=item["linecolor"],
            ls=item["linestyle"],
            lw=item["linewidth"],
            alpha=0.85,
            zorder=3,
            label=legend_label
        )

    ax.axvline(0, color="gray", lw=0.7, zorder=1)

    if show_legend:
        ax.legend(loc="lower right", fontsize=7, framealpha=0.95)


# =========================================================
# ズーム図
# =========================================================
def plot_zoom_panel(
    ax,
    S,
    net,
    sub_raw,
    call_walls,
    put_walls,
    hvl,
    total,
    label,
    spot_flow,
    xlim,
    ylim,
    show_legend=False
):
    if net.empty:
        ax.text(0.5, 0.5, "データなし", ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title(label)
        return

    draw_bars(ax, net)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if call_walls and put_walls:
        lo, hi = sorted([put_walls[0][0], call_walls[0][0]])
        ax.axhspan(lo, hi, color="#ffbf00", alpha=0.08, zorder=0)

    label_items = build_label_items(sub_raw, call_walls, put_walls, hvl, S, spot_flow)
    draw_reference_lines(ax, label_items, show_legend=show_legend)

    visible_items = [item for item in label_items if ylim[0] <= item["y"] <= ylim[1]]

    ymin, ymax = ax.get_ylim()
    min_gap = (ymax - ymin) * 0.05 if ("0DTE" in label or "当日" in label) else (ymax - ymin) * 0.035
    levels = [(item["key"], item["y"]) for item in visible_items]
    y_text_map = spread_label_positions(levels, ymin, ymax, min_gap)

    x_left, x_right = xlim
    x_span = x_right - x_left
    gex_m = net["GEX"] / 1e6
    raw_xmax = max(float(gex_m.max()), 0.0)

    x_anchor = raw_xmax + x_span * 0.01
    x_text = raw_xmax + x_span * 0.22

    for item in visible_items:
        key = item["key"]
        y_actual = item["y"]
        y_text = y_text_map[key]

        ax.annotate(
            item["text"],
            xy=(x_anchor, y_actual),
            xytext=(x_text, y_text),
            ha="right",
            va="center",
            fontsize=item["fontsize"],
            color=item["color"],
            arrowprops=dict(
                arrowstyle="-",
                color=item["color"],
                lw=0.8,
                shrinkA=0,
                shrinkB=0
            ),
            zorder=4,
            clip_on=False
        )

    if hvl is not None and not (ylim[0] <= hvl <= ylim[1]):
        ax.text(
            0.98, 0.02,
            f"HVL: {hvl:.1f} は全体俯瞰参照",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="#1f77b4"
        )

    regime = "+γ (安定)" if (hvl is not None and S > hvl) else "-γ (増幅)"
    ax.set_title(f"{label}\ntotalGEX {total / 1e6:.0f}M  レジーム {regime}", fontsize=10)
    ax.set_xlabel("NET GEX (M)  ← Put優勢 / Call優勢 →")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.grid(axis="x", ls=":", alpha=0.4, zorder=0)


def build_zoom_figure(panel_data, symbol, S, timestamp):
    fig, axes = plt.subplots(1, len(panel_data), figsize=(20, 6), sharey=False)

    for i, pdata in enumerate(panel_data):
        ax = axes[i]
        plot_zoom_panel(
            ax=ax,
            S=S,
            net=pdata["net"],
            sub_raw=pdata["sub"],
            call_walls=pdata["call_walls"],
            put_walls=pdata["put_walls"],
            hvl=pdata["hvl"],
            total=pdata["total"],
            label=pdata["label"],
            spot_flow=pdata["spot_flow"],
            xlim=pdata["xlim"],
            ylim=pdata["zoom_ylim"],
            show_legend=(i == 0)
        )

        if i == 0:
            ax.set_ylabel("Strike (zoom)")
        else:
            ax.tick_params(axis="y", labelleft=False)

    fig.suptitle(
        f"{symbol} GEX Profile 満期別  |  Spot {S:.2f}  |  {timestamp}\n"
        f"Spot近辺ズーム",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# =========================================================
# 全体俯瞰図
# =========================================================
def plot_overview_panel(
    ax,
    S,
    net,
    sub_raw,
    call_walls,
    put_walls,
    hvl,
    label,
    xlim,
    ylim,
    zoom_ylim,
    show_legend=False
):
    if net.empty:
        ax.text(0.5, 0.5, "データなし", ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title(label)
        return

    draw_bars(ax, net)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    label_items = build_label_items(sub_raw, call_walls, put_walls, hvl, S, "")
    draw_reference_lines(ax, label_items, show_legend=show_legend)

    # ズーム範囲を全体俯瞰に薄く表示
    ax.axhspan(zoom_ylim[0], zoom_ylim[1], color="#90caf9", alpha=0.14, zorder=0)

    ax.set_title(label, fontsize=10)
    ax.set_xlabel("NET GEX (M)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.grid(axis="x", ls=":", alpha=0.3, zorder=0)


def build_overview_figure(panel_data, symbol, S, timestamp):
    fig, axes = plt.subplots(1, len(panel_data), figsize=(20, 6), sharey=False)

    for i, pdata in enumerate(panel_data):
        ax = axes[i]
        plot_overview_panel(
            ax=ax,
            S=S,
            net=pdata["net"],
            sub_raw=pdata["sub"],
            call_walls=pdata["call_walls"],
            put_walls=pdata["put_walls"],
            hvl=pdata["hvl"],
            label=pdata["label"],
            xlim=pdata["xlim"],
            ylim=pdata["overview_ylim"],
            zoom_ylim=pdata["zoom_ylim"],
            show_legend=(i == 0)
        )

        if i == 0:
            ax.set_ylabel("Strike (overview)")
        else:
            ax.tick_params(axis="y", labelleft=False)

    fig.suptitle(
        f"{symbol} GEX Profile 満期別  |  Spot {S:.2f}  |  {timestamp}\n"
        f"全体俯瞰（意味のあるデータ帯まで表示）",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# =========================================================
# パネルデータまとめ
# =========================================================
def build_panel_data(df, top_n, risk_free, div_yield, overview_threshold_ratio):
    S = float(df["Price~"].iloc[0])
    symbol = get_symbol(df)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    panel_data = []
    summary_rows = []

    for (label, dte_range) in PATTERNS:
        sub = filter_dte(df, dte_range)
        net = compute_gex(sub, S, risk_free, div_yield)
        call_walls, put_walls = find_top_walls(net, n=top_n)
        hvl, total = find_hvl(S, net)

        atm = nearest_strike(net, S)
        if atm is not None:
            a, b, m = flow_detail(sub, atm)
            spot_flow = flow_short(a, b, m)
            spot_flow_full = flow_full(a, b, m)
        else:
            spot_flow = ""
            spot_flow_full = ""

        xlim = get_x_limits(net)
        zoom_ylim = get_zoom_ylim(S, call_walls, put_walls, hvl=hvl)
        overview_ylim = get_overview_ylim(
            net=net,
            S=S,
            call_walls=call_walls,
            put_walls=put_walls,
            hvl=hvl,
            threshold_ratio=overview_threshold_ratio
        )

        panel_data.append({
            "label": label,
            "sub": sub,
            "net": net,
            "call_walls": call_walls,
            "put_walls": put_walls,
            "hvl": hvl,
            "total": total,
            "spot_flow": spot_flow,
            "spot_flow_full": spot_flow_full,
            "xlim": xlim,
            "zoom_ylim": zoom_ylim,
            "overview_ylim": overview_ylim,
        })

        row = {
            "Pattern": label,
            "Spot": S,
            "HVL": hvl,
            "totalGEX_M": round(total / 1e6, 2),
            "Spot_flow": spot_flow_full,
            "Zoom_min": round(zoom_ylim[0], 2),
            "Zoom_max": round(zoom_ylim[1], 2),
            "Overview_min": round(overview_ylim[0], 2),
            "Overview_max": round(overview_ylim[1], 2),
        }

        for j, (k, g) in enumerate(call_walls, 1):
            a, b, m = flow_detail(sub, k, "Call")
            row[f"CW{j}_strike"] = k
            row[f"CW{j}_GEX_M"] = round(g / 1e6, 2)
            row[f"CW{j}_flow"] = flow_full(a, b, m)

        for j, (k, g) in enumerate(put_walls, 1):
            a, b, m = flow_detail(sub, k, "Put")
            row[f"PW{j}_strike"] = k
            row[f"PW{j}_GEX_M"] = round(g / 1e6, 2)
            row[f"PW{j}_flow"] = flow_full(a, b, m)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    return panel_data, summary_df, symbol, S, timestamp


# =========================================================
# ダウンロード用変換
# =========================================================
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def df_to_csv_bytes(df):
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# =========================================================
# UI
# =========================================================
st.title("📈 GEX Profile Viewer")
st.caption("CSV / Excel をドラッグ＆ドロップすると、『Spot近辺ズーム』と『全体俯瞰』を別々に表示します。")

with st.sidebar:
    st.subheader("設定")
    top_n = st.slider("表示する Wall の本数", min_value=1, max_value=5, value=3, step=1)
    risk_free = st.number_input("無リスク金利", value=DEFAULT_RISK_FREE, step=0.005, format="%.4f")
    div_yield = st.number_input("配当利回り", value=DEFAULT_DIV_YIELD, step=0.005, format="%.4f")

    st.divider()
    st.subheader("全体俯瞰の絞り込み")
    overview_threshold_ratio = st.slider(
        "俯瞰で無視する微小GEX比率",
        min_value=0.0,
        max_value=0.10,
        value=0.01,
        step=0.005,
        help="max(|GEX|) に対してこの比率未満の strike は俯瞰レンジ決定から除外"
    )

    st.divider()
    st.subheader("フォント状態")
    if FONT_FOUND:
        st.success(f"日本語フォント読込済み: {FONT_NAME}")
        st.caption(FONT_PATH.name)
    else:
        st.error("fonts/NotoSansJP-Regular.ttf が見つかりません")
        st.caption("画像内日本語が文字化けする可能性があります")

uploaded_file = st.file_uploader(
    "ここに CSV / Excel をドラッグ＆ドロップ",
    type=["csv", "txt", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("ファイルをアップロードすると、ここにグラフが表示されます。")
else:
    try:
        df = load_data_from_upload(uploaded_file)

        if df.empty:
            st.warning("有効な行がありません。IV > 0 のオプションデータを確認してください。")
        else:
            panel_data, summary_df, symbol, spot, timestamp = build_panel_data(
                df=df,
                top_n=top_n,
                risk_free=risk_free,
                div_yield=div_yield,
                overview_threshold_ratio=overview_threshold_ratio
            )

            zoom_fig = build_zoom_figure(panel_data, symbol, spot, timestamp)
            overview_fig = build_overview_figure(panel_data, symbol, spot, timestamp)

            zoom_png = fig_to_png_bytes(zoom_fig)
            overview_png = fig_to_png_bytes(overview_fig)
            csv_bytes = df_to_csv_bytes(summary_df)

            plt.close(zoom_fig)
            plt.close(overview_fig)

            st.session_state["zoom_png"] = zoom_png
            st.session_state["overview_png"] = overview_png
            st.session_state["csv_bytes"] = csv_bytes
            st.session_state["symbol"] = symbol
            st.session_state["timestamp"] = timestamp

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.metric("Symbol", symbol)
            with c2:
                st.metric("Spot", f"{spot:.2f}")
            with c3:
                st.write(f"生成時刻: {timestamp}")

            st.subheader("Spot近辺ズーム")
            st.image(
                st.session_state["zoom_png"],
                caption=f"{symbol} GEX Profile - Zoom"
            )

            z1, z2 = st.columns([1, 2])
            with z1:
                st.download_button(
                    label="ズーム画像をダウンロード (PNG)",
                    data=st.session_state["zoom_png"],
                    file_name=f"{st.session_state['symbol']}_gex_zoom_{st.session_state['timestamp']}.png",
                    mime="image/png",
                    on_click="ignore"
                )

            st.subheader("全体俯瞰")
            st.image(
                st.session_state["overview_png"],
                caption=f"{symbol} GEX Profile - Overview"
            )

            o1, o2 = st.columns([1, 2])
            with o1:
                st.download_button(
                    label="全体俯瞰画像をダウンロード (PNG)",
                    data=st.session_state["overview_png"],
                    file_name=f"{st.session_state['symbol']}_gex_overview_{st.session_state['timestamp']}.png",
                    mime="image/png",
                    on_click="ignore"
                )

            st.subheader("集計テーブル")
            st.dataframe(summary_df, width="stretch")

            st.download_button(
                label="集計をダウンロード (CSV)",
                data=st.session_state["csv_bytes"],
                file_name=f"{st.session_state['symbol']}_summary_{st.session_state['timestamp']}.csv",
                mime="text/csv",
                on_click="ignore"
            )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
