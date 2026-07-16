def build_panel_data(df, top_n, risk_free, div_yield, overview_threshold_ratio):
    S = float(df["Price~"].iloc[0])
    symbol = get_symbol(df)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 実行日を基準に直近SQ / 次のMSQ を決定
    today = datetime.now().date()
    exp_dates = df["_ExpDate"].dropna().unique().tolist() if "_ExpDate" in df.columns else []
    sq_date, msq_date = pick_sq_msq_dates(exp_dates, today)

    # 満期区分の定義（ラベル, 対象日, モード）
    patterns = [
        (f"直近SQ ({sq_date})" if sq_date else "直近SQ (該当なし)", sq_date, "on"),
        (f"MSQ ({msq_date})" if msq_date else "MSQ (該当なし)", msq_date, "on"),
        ("全満期", None, "all"),
    ]

    panel_data = []
    summary_rows = []

    for (label, target_date, mode) in patterns:
        sub = filter_by_expiry(df, target_date, mode)
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
            net=net, S=S, call_walls=call_walls, put_walls=put_walls,
            hvl=hvl, threshold_ratio=overview_threshold_ratio
        )

        panel_data.append({
            "label": label, "sub": sub, "net": net,
            "call_walls": call_walls, "put_walls": put_walls,
            "hvl": hvl, "total": total,
            "spot_flow": spot_flow, "spot_flow_full": spot_flow_full,
            "xlim": xlim, "zoom_ylim": zoom_ylim, "overview_ylim": overview_ylim,
        })

        row = {
            "Pattern": label, "Spot": S, "HVL": hvl,
            "totalGEX_M": round(total / 1e6, 2),
            "Spot_flow": spot_flow_full,
            "Zoom_min": round(zoom_ylim[0], 2), "Zoom_max": round(zoom_ylim[1], 2),
            "Overview_min": round(overview_ylim[0], 2), "Overview_max": round(overview_ylim[1], 2),
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
