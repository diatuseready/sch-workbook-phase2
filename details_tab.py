import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

DETAILS_RENAME = {
    "Open Inv": "Opening Inv",
    "Batch In (RECEIPTS_BBL)": "Batch In",
    "Batch Out (DELIVERIES_BBL)": "Batch Out",
    "Rack/Liftings": "Rack/Lifting",
}

DETAILS_COLS = [
    "source",
    "Product",
    "Close Inv",
    "Opening Inv",
    "Batch In",
    "Batch Out",
    "Rack/Lifting",
    "Pipeline In",
    "Pipeline Out",
    "Gain/Loss",
    "Transfers",
    "Notes",
]

FORECAST_FLOW_COLS = [
    "Batch In (RECEIPTS_BBL)",
    "Batch Out (DELIVERIES_BBL)",
    "Rack/Liftings",
    "Pipeline In",
    "Pipeline Out",
    "Production",
    "Adjustments",
    "Gain/Loss",
    "Transfers",
]

INFLOW_COLS = [
    "Batch In (RECEIPTS_BBL)",
    "Pipeline In",
    "Production",
]

OUTFLOW_COLS = [
    "Batch Out (DELIVERIES_BBL)",
    "Rack/Liftings",
    "Pipeline Out",
]

NET_COLS = [
    "Adjustments",
    "Gain/Loss",
    "Transfers",
]

SOURCE_BG = {
    "system": "#d9f2d9",
    "forecast": "#d9ecff",
}

LOCKED_BASE_COLS = [
    "Date",
    "{id_col}",
    "source",
    "Product",
    "Close Inv",
    "Opening Inv",
]


# We set an explicit height so the grid shows ~15 rows before scrolling.
DETAILS_EDITOR_VISIBLE_ROWS = 15
DETAILS_EDITOR_ROW_PX = 35  # approx row height incl. padding
DETAILS_EDITOR_HEADER_PX = 35
DETAILS_EDITOR_HEIGHT_PX = DETAILS_EDITOR_HEADER_PX + (DETAILS_EDITOR_VISIBLE_ROWS * DETAILS_EDITOR_ROW_PX)


def _style_source_cells(df: pd.DataFrame, cols_to_color: list[str]) -> "pd.io.formats.style.Styler":
    cols = list(df.columns)
    cols_set = set(cols_to_color)

    def _row_style(row: pd.Series) -> list[str]:
        bg = SOURCE_BG.get(str(row.get("source", "")).strip().lower(), "")
        style = f"background-color: {bg};" if bg else ""
        return [style if (c in cols_set and style) else "" for c in cols]

    return df.style.apply(_row_style, axis=1).hide(axis="index")


def _locked_cols(id_col: str, cols: list[str]) -> list[str]:
    wanted = [c.format(id_col=id_col) for c in LOCKED_BASE_COLS]
    return [c for c in wanted if c in cols]


def _column_config(df: pd.DataFrame, cols: list[str], id_col: str):
    locked = set(_locked_cols(id_col, cols))

    cfg: dict[str, object] = {
        "Date": st.column_config.DateColumn("Date", disabled=True, format="YYYY-MM-DD"),
        id_col: st.column_config.TextColumn(id_col, disabled=True),
        "source": st.column_config.SelectboxColumn(
            "Source",
            options=["system", "forecast", "manual"],
            required=True,
            disabled=True,
        ),
        "Product": st.column_config.TextColumn("Product", disabled=True),
        "updated": st.column_config.CheckboxColumn("updated", default=False),
        "Notes": st.column_config.TextColumn("Notes"),
    }

    for c in cols:
        if c in cfg or c == "Notes":
            continue
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            cfg[c] = st.column_config.NumberColumn(c, disabled=(c in locked), format="%.2f")

    for c in locked:
        if c in {"Date", id_col, "source", "Product"}:
            continue
        if c in cols and c not in cfg:
            cfg[c] = st.column_config.TextColumn(c, disabled=True)

    return {k: v for k, v in cfg.items() if k in cols}


def _aggregate_daily_details(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    group_cols = ["Date", id_col, "Product"]
    agg_map: dict[str, str] = {}

    if "Open Inv" in df.columns:
        agg_map["Open Inv"] = "first"
    if "Close Inv" in df.columns:
        agg_map["Close Inv"] = "last"

    for c in _available_flow_cols(df):
        agg_map[c] = "sum"

    if "source" in df.columns:
        agg_map["source"] = "first"
    if "updated" in df.columns:
        agg_map["updated"] = "max"
    if "Notes" in df.columns:
        agg_map["Notes"] = "last"

    return df.groupby(group_cols, as_index=False).agg(agg_map)


def _available_flow_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in FORECAST_FLOW_COLS if c in df.columns]


def _ensure_lineage_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "source" not in out.columns:
        out["source"] = "system"
    else:
        out["source"] = out["source"].fillna("system")

    if "updated" not in out.columns:
        out["updated"] = 0
    else:
        out["updated"] = pd.to_numeric(out["updated"]).fillna(0).astype(int)

    return out


def _weekday_weighted_means(
    hist: pd.DataFrame,
    flow_cols: list[str],
    max_weeks: int = 6,
    decay: float = 0.70,
) -> dict[tuple[int, str], float]:
    if hist.empty or not flow_cols:
        return {}

    h = hist.sort_values("Date").copy()
    h["__weekday"] = h["Date"].dt.weekday

    recent = h.tail(21)
    fallback = {c: float(recent[c].mean()) if c in recent.columns and len(recent) else 0.0 for c in flow_cols}

    out: dict[tuple[int, str], float] = {}
    for wd in range(7):
        subset = h[h["__weekday"] == wd].sort_values("Date", ascending=False).head(max_weeks)
        if subset.empty:
            for c in flow_cols:
                out[(wd, c)] = fallback.get(c, 0.0)
            continue

        weights = np.array([decay**i for i in range(len(subset))], dtype=float)
        wsum = float(weights.sum()) or 1.0

        for c in flow_cols:
            vals = subset[c].astype(float).to_numpy() if c in subset.columns else np.zeros(len(subset), dtype=float)
            out[(wd, c)] = float((vals * weights).sum() / wsum)

    return out


def estimate_forecast_flows(
    group: pd.DataFrame,
    flow_cols: list[str],
    d: pd.Timestamp,
) -> dict[str, float]:
    means = _weekday_weighted_means(group, flow_cols=flow_cols)
    wd = int(d.weekday())
    return {c: float(means.get((wd, c), 0.0)) for c in flow_cols}


def _roll_inventory(prev_close: float, flows: dict[str, float], flow_cols: list[str]) -> tuple[float, float]:
    opening = float(prev_close)
    inflow = sum(float(flows.get(c, 0.0) or 0.0) for c in INFLOW_COLS if c in flow_cols)
    outflow = sum(float(flows.get(c, 0.0) or 0.0) for c in OUTFLOW_COLS if c in flow_cols)
    net = sum(float(flows.get(c, 0.0) or 0.0) for c in NET_COLS if c in flow_cols)
    closing = opening + inflow - outflow + net
    return opening, closing


def _forecast_dates(last_date: pd.Timestamp, forecast_end: pd.Timestamp | None, default_days: int) -> pd.DatetimeIndex:
    start = last_date + timedelta(days=1)
    if forecast_end is not None:
        if start > forecast_end:
            return pd.DatetimeIndex([])
        return pd.date_range(start=start, end=forecast_end, freq="D")
    return pd.date_range(start=start, periods=int(default_days), freq="D")


def _last_close_inv(group: pd.DataFrame) -> float:
    if "Close Inv" not in group.columns:
        return 0.0

    last_date = group["Date"].max()
    last_rows = group[group["Date"] == last_date]
    if last_rows.empty:
        return 0.0

    val = last_rows["Close Inv"].iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def _extend_with_30d_forecast(
    df: pd.DataFrame,
    *,
    id_col: str,
    forecast_end: pd.Timestamp | None = None,
    default_days: int = 30,
) -> pd.DataFrame:
    if df.empty:
        return df

    daily = _aggregate_daily_details(df, id_col=id_col)
    if daily.empty:
        return daily

    daily = _ensure_lineage_cols(daily).sort_values("Date")
    flow_cols = _available_flow_cols(daily)

    if forecast_end is not None:
        forecast_end = pd.Timestamp(forecast_end)

    forecast_rows: list[dict] = []

    for (id_val, product), group in daily.groupby([id_col, "Product"], dropna=False):
        group = group.sort_values("Date")
        last_date = pd.Timestamp(group["Date"].max())
        prev_close = _last_close_inv(group)
        for d in _forecast_dates(last_date, forecast_end, default_days):
            flows = estimate_forecast_flows(group, flow_cols=flow_cols, d=d)
            opening, closing = _roll_inventory(prev_close, flows, flow_cols)
            prev_close = closing

            row = {
                "Date": d,
                id_col: id_val,
                "Product": product,
                "source": "forecast",
                "updated": 0,
                "Notes": "Forecast",
                "Open Inv": opening,
                "Close Inv": closing,
                **flows,
            }
            forecast_rows.append(row)

    if not forecast_rows:
        return daily

    combined = pd.concat([daily, pd.DataFrame(forecast_rows)], ignore_index=True)
    for c in ["Open Inv", "Close Inv"] + flow_cols:
        if c in combined.columns:
            combined[c] = pd.to_numeric(combined[c]).fillna(0.0)

    return combined


def build_details_view(df: pd.DataFrame, id_col: str):
    df = df.sort_values("Date").rename(columns=DETAILS_RENAME)

    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    cols = ["Date", id_col] + DETAILS_COLS
    cols = [c for c in cols if c in df.columns]

    for c in cols:
        if c in {"Date", id_col, "source", "Product", "Notes", "updated"}:
            continue
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(2)

    return df, cols


def display_midcon_details(df_filtered: pd.DataFrame, active_region: str, forecast_end: pd.Timestamp):
    st.subheader("🧾 Group Daily Details")

    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return

    df_all = _extend_with_30d_forecast(df_filtered, id_col="System", forecast_end=forecast_end)
    df_display, cols = build_details_view(df_all, id_col="System")

    locked_cols = _locked_cols("System", cols)
    column_config = _column_config(df_display, cols, "System")

    # Hide internal lineage columns from the UI.
    column_order = [c for c in cols if c != "source"]
    column_config = {k: v for k, v in column_config.items() if k in column_order}

    # Ensure we have a RangeIndex so `hide_index=True` works with `num_rows='dynamic'`.
    editor_df = df_display[cols].reset_index(drop=True)

    st.data_editor(
        _style_source_cells(editor_df, locked_cols),
        num_rows="dynamic",
        width="stretch",
        height=DETAILS_EDITOR_HEIGHT_PX,
        hide_index=True,
        column_order=column_order,
        key=f"{active_region}_edit",
        column_config=column_config,
    )

    st.markdown('<div class="save-btn-bottom">', unsafe_allow_html=True)
    if st.button("💾 Save Changes", key=f"save_{active_region}"):
        st.success("✅ Changes saved successfully!")
    st.markdown("</div>", unsafe_allow_html=True)


def display_location_details(df_filtered: pd.DataFrame, active_region: str, forecast_end: pd.Timestamp):
    st.subheader("🏭 Locations")

    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return

    region_locs = sorted(df_filtered["Location"].dropna().unique().tolist()) if "Location" in df_filtered.columns else []

    if not region_locs:
        st.write("*(No locations available in the current selection)*")
        return

    for i, loc in enumerate(st.tabs(region_locs)):
        with loc:
            st.markdown(f"### 📍 {region_locs[i]}")
            df_loc = df_filtered[df_filtered["Location"] == region_locs[i]]

            if df_loc.empty:
                st.write("*(No data for this location)*")
            else:
                df_all = _extend_with_30d_forecast(df_loc, id_col="Location", forecast_end=forecast_end)
                df_display, cols = build_details_view(df_all, id_col="Location")

                locked_cols = _locked_cols("Location", cols)
                column_config = _column_config(df_display, cols, "Location")

                # Hide internal lineage columns from the UI.
                column_order = [c for c in cols if c != "source"]
                column_config = {k: v for k, v in column_config.items() if k in column_order}

                # Ensure we have a RangeIndex so `hide_index=True` works with `num_rows='dynamic'`.
                editor_df = df_display[cols].reset_index(drop=True)

                st.data_editor(
                    _style_source_cells(editor_df, locked_cols),
                    num_rows="dynamic",
                    width="stretch",
                    height=DETAILS_EDITOR_HEIGHT_PX,
                    hide_index=True,
                    column_order=column_order,
                    key=f"{active_region}_{region_locs[i]}_edit",
                    column_config=column_config,
                )

            st.markdown('<div class="save-btn-bottom">', unsafe_allow_html=True)
            if st.button(f"💾 Save {region_locs[i]}", key=f"save_{active_region}_{region_locs[i]}"):
                st.success(f"✅ Changes for {region_locs[i]} saved successfully!")
            st.markdown("</div>", unsafe_allow_html=True)


def display_details_tab(df_filtered: pd.DataFrame, active_region: str, end_ts: pd.Timestamp):
    if active_region == "Midcon":
        display_midcon_details(df_filtered, active_region, forecast_end=end_ts)
    else:
        display_location_details(df_filtered, active_region, forecast_end=end_ts)
