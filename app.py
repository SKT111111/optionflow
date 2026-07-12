import io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Streamlit用の非GUIバックエンド
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
FONT_DIR = BASE_DIR / "fonts"

FONT_CANDIDATES = [
    FONT_DIR / "NotoSansJP-Regular.ttf",
    FONT_DIR / "NotoSansJP-Medium.ttf",
    FONT_DIR / "NotoSansCJKjp-Regular.otf",
    FONT_DIR / "IPAexGothic.ttf",
    FONT_DIR / "ipag.ttf",
]


def setup_japanese_font():
    """
    同梱フォントをMatplotlibに登録して、日本語描画を安定させる。
    戻り値: (font_name, found_flag, font_path_or_none)
    """
    font_name = None
    found_font_path = None

    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            font_name = fm.FontProperties(fname=str(font_path)).get_name()
            found_font_path = font_path
            break

    if font_name is not None:
        matplotlib.rcParams["font.family"] = font_name
    else:
        # フォールバック。環境依存で日本語が出ない可能性あり。
        matplotlib.rcParams["font.family"] = [
            "Noto Sans CJK JP",
            "Noto Sans JP",
            "IPAexGothic",
            "IPAGothic",
            "Yu Gothic",
            "Meiryo",
            "MS Gothic",
            "DejaVu Sans",
        ]

    matplotlib.rcParams["axes.unicode_minus"] = False

    return font_name, found_font_path is not None, found_font_path


FONT_NAME, FONT_FOUND, FONT_PATH = setup_japanese_font()


# =========================================================
# 定数
# =========================================================
DEFAULT_RISK_FREE = 0.045
DEFAULT_DIV_YIELD = 0.0

PATTERNS = [
    ("当日 (0DTE)",      (0, 0)),
    ("1週間以内 (0-7)",  (0, 7)),
    ("1ヶ月以内 (0-30)", (0, 30)),
    ("全満期",           (0, None)),
]


# =========================================================
# 入力読み込み
# =========================================================
def read_uploaded_table(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".csv") or name.endswith(".txt"):
        return pd.read_csv(uploaded_file)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

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

    validate_required_columns(df)

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
# ラベル重なり回避
# =========================================================
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

        if j == 1:
            items.append({
                "key": f"cw{j}",
                "y": k,
                "text": f"CW {k:.1f} [{fl}]",
                "color": "#006400",
                "linecolor": "#006400",
                "linewidth": 2.0,
                "linestyle": "-",
                "fontsize": 7.2,
                "fontweight": "bold",
            })
        else:
            items.append({
                "key": f"cw{j}",
                "y": k,
                "text": f"CW{j} {k:.1f} [{fl}]",
                "color": "#2ca02c",
                "linecolor": "#2ca02c",
                "linewidth": 1.1,
                "linestyle": ":",
                "fontsize": 6.6,
                "fontweight": "normal",
            })

    for j, (k, g) in enumerate(put_walls, 1):
        a, b, m = flow_detail(sub_raw, k, "Put")
        fl = flow_short(a, b, m)

        if j == 1:
            items.append({
                "key": f"pw{j}",
                "y": k,
                "text": f"PW {k:.1f} [{fl}]",
                "color": "#8b0000",
                "linecolor": "#8b0000",
                "linewidth": 2.0,
                "linestyle": "-",
                "fontsize": 7.2,
                "fontweight": "bold",
            })
        else:
            items.append({
                "key": f"pw{j}",
                "y": k,
                "text": f"PW{j} {k:.1f} [{fl}]",
                "color": "#d62728",
                "linecolor": "#d62728",
                "linewidth": 1.1,
                "linestyle": ":",
                "fontsize": 6.6,
                "fontweight": "normal",
            })

    if hvl is not None:
        items.append({
            "key": "hvl",
            "y": hvl,
            "text": f"HVL {hvl:.1f}",
            "color": "#1f77b4",
            "linecolor": "#1f77b4",
            "linewidth": 2.0,
            "linestyle": "--",
            "fontsize": 8.0,
            "fontweight": "bold",
        })

    items.append({
        "key": "spot",
        "y": S,
        "text": f"Spot {S:.2f} [ATM近傍: {spot_flow}]",
        "color": "black",
        "linecolor": "black",
        "linewidth": 1.8,
        "linestyle": ":",
        "fontsize": 7.2,
        "fontweight": "bold",
    })

    return items


# =========================================================
# 1パネル描画
# =========================================================
def plot_panel(
    ax,
    S,
    net,
    sub_raw,
    call_walls,
    put_walls,
    hvl,
    total,
    label,
    spot_flow="",
    show_legend=False,
    global_ylim=None
):
    if net.empty:
        ax.text(
            0.5, 0.5, "データなし",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=12
        )
        ax.set_title(label)
        return

    net = net.copy()
    net["GEX_M"] = net["GEX"] / 1e6

    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in net["GEX_M"]]

    diff_med = net["Strike"].diff().median()
    if pd.isna(diff_med) or diff_med <= 0:
        bar_h = 0.8
    else:
        bar_h = diff_med * 0.8

    ax.barh(
        net["Strike"],
        net["GEX_M"],
        color=colors,
        height=bar_h,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.85,
        zorder=2
    )

    raw_xmax = max(float(net["GEX_M"].max()), 0.0)
    raw_xmin = min(float(net["GEX_M"].min()), 0.0)
    x_span = max(raw_xmax - raw_xmin, 1.0)

    right_pad = x_span * 0.42
    left_pad = x_span * 0.08

    x_left = raw_xmin - left_pad
    x_right = raw_xmax + right_pad
    ax.set_xlim(x_left, x_right)

    if global_ylim is not None:
        ax.set_ylim(global_ylim)

    ymin, ymax = ax.get_ylim()

    if call_walls and put_walls:
        lo, hi = sorted([put_walls[0][0], call_walls[0][0]])
        ax.axhspan(lo, hi, color="#ffbf00", alpha=0.08, zorder=0, label="Transition Zone")

    label_items = build_label_items(sub_raw, call_walls, put_walls, hvl, S, spot_flow)

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
            alpha=0.95 if item["fontweight"] == "bold" else 0.72,
            zorder=3,
            label=legend_label
        )

    ax.axvline(0, color="gray", lw=0.7, zorder=1)

    if "0DTE" in label or "当日" in label:
        min_gap = (ymax - ymin) * 0.045
    else:
        min_gap = (ymax - ymin) * 0.028

    levels = [(item["key"], item["y"]) for item in label_items]
    y_text_map = spread_label_positions(levels, ymin, ymax, min_gap)

    x_anchor = raw_xmax + x_span * 0.02
    x_text = raw_xmax + x_span * 0.36

    for item in label_items:
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
            fontweight=item["fontweight"],
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

    regime = "+γ (安定)" if (hvl is not None and S > hvl) else "-γ (増幅)"
    ax.set_title(f"{label}\ntotalGEX {total / 1e6:.0f}M  レジーム {regime}", fontsize=10)
    ax.set_xlabel("NET GEX (M)  ← Put優勢 / Call優勢 →")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.grid(axis="x", ls=":", alpha=0.4, zorder=0)

    if show_legend:
        ax.legend(loc="lower right", fontsize=7, framealpha=0.95)


# =========================================================
# 図と集計表の生成
# =========================================================
def build_figure_and_summary(df, top_n, risk_free, div_yield):
    S = float(df["Price~"].iloc[0])
    symbol = get_symbol(df)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    panel_data = []
    all_y_values = [S]

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

        if not net.empty:
            all_y_values.extend(net["Strike"].tolist())

        for k, _ in call_walls:
            all_y_values.append(k)
        for k, _ in put_walls:
            all_y_values.append(k)
        if hvl is not None:
            all_y_values.append(hvl)

        panel_data.append({
            "label": label,
            "sub": sub,
            "net": net,
            "call_walls": call_walls,
            "put_walls": put_walls,
            "hvl": hvl,
            "total": total,
            "atm": atm,
            "spot_flow": spot_flow,
            "spot_flow_full": spot_flow_full,
        })

    if all_y_values:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        y_pad = max((y_max - y_min) * 0.05, 1.0)
        global_ylim = (y_min - y_pad, y_max + y_pad)
    else:
        global_ylim = None

    fig, axes = plt.subplots(1, len(PATTERNS), figsize=(22, 10), sharey=True)
    axes[0].set_ylabel("Strike")

    summary_rows = []

    for i, (ax, pdata) in enumerate(zip(axes, panel_data)):
        label = pdata["label"]
        sub = pdata["sub"]
        net = pdata["net"]
        call_walls = pdata["call_walls"]
        put_walls = pdata["put_walls"]
        hvl = pdata["hvl"]
        total = pdata["total"]
        spot_flow = pdata["spot_flow"]
        spot_flow_full = pdata["spot_flow_full"]

        plot_panel(
            ax=ax,
            S=S,
            net=net,
            sub_raw=sub,
            call_walls=call_walls,
            put_walls=put_walls,
            hvl=hvl,
            total=total,
            label=label,
            spot_flow=spot_flow,
            show_legend=(i == 0),
            global_ylim=global_ylim
        )

        row = {
            "Pattern": label,
            "Spot": S,
            "HVL": hvl,
            "totalGEX_M": round(total / 1e6, 2),
            "Spot_flow": spot_flow_full
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

    fig.suptitle(
        f"{symbol} GEX Profile 満期別  |  Spot {S:.2f}  |  {timestamp}\n"
        f"（各Wall脇 [買優勢/売優勢/拮抗] = 当日フローの傾き・参考）",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    summary_df = pd.DataFrame(summary_rows)
    return fig, summary_df, symbol, S, timestamp


# =========================================================
# ダウンロード用変換
# =========================================================
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def df_to_csv_bytes(df):
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# =========================================================
# UI
# =========================================================
st.title("📈 GEX Profile Viewer")
st.caption("CSV / Excel をドラッグ＆ドロップすると、満期別の GEX 4パネル画像を表示します。")

with st.sidebar:
    st.subheader("設定")
    top_n = st.slider("表示する Wall の本数", min_value=1, max_value=5, value=3, step=1)
    risk_free = st.number_input("無リスク金利", value=DEFAULT_RISK_FREE, step=0.005, format="%.4f")
    div_yield = st.number_input("配当利回り", value=DEFAULT_DIV_YIELD, step=0.005, format="%.4f")

    st.divider()
    st.subheader("フォント状態")
    if FONT_FOUND:
        st.success(f"日本語フォント読込済み: {FONT_NAME}")
        st.caption(f"{FONT_PATH.name}")
    else:
        st.warning("同梱日本語フォントが見つかりません。画像内日本語が文字化けする可能性があります。")
        st.caption("fonts/NotoSansJP-Regular.ttf などを配置してください。")

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
            fig, summary_df, symbol, spot, timestamp = build_figure_and_summary(
                df=df,
                top_n=top_n,
                risk_free=risk_free,
                div_yield=div_yield
            )

            png_bytes = fig_to_png_bytes(fig)
            csv_bytes = df_to_csv_bytes(summary_df)

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.metric("Symbol", symbol)
            with c2:
                st.metric("Spot", f"{spot:.2f}")
            with c3:
                st.write(f"生成時刻: {timestamp}")

            st.pyplot(fig, use_container_width=True)

            st.subheader("集計テーブル")
            st.dataframe(summary_df, use_container_width=True)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    label="画像をダウンロード (PNG)",
                    data=png_bytes,
                    file_name=f"{symbol}_gex_by_expiry_{timestamp}.png",
                    mime="image/png"
                )
            with d2:
                st.download_button(
                    label="集計をダウンロード (CSV)",
                    data=csv_bytes,
                    file_name=f"{symbol}_summary_{timestamp}.csv",
                    mime="text/csv"
                )

            plt.close(fig)

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
