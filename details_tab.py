import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
 
from admin_config import get_visible_columns, get_threshold_overrides
 
DETAILS_RENAME = {
    "Open Inv": "Opening Inv",
    "Batch In (RECEIPTS_BBL)": "Batch In",
    "Batch Out (DELIVERIES_BBL)": "Batch Out",
    "Rack/Liftings": "Rack/Lifting",
}
 
DETAILS_COLS = [
    "source",
    "Product",
    "Opening Inv",
    "Close Inv",
    "Batch In",
    "Batch Out",
    "Rack/Lifting",
    "Pipeline In",
    "Pipeline Out",
    "Adjustments",  # New addition for Magellan
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
 
 
# Flow-column names *after* `DETAILS_RENAME` has been applied.
DISPLAY_INFLOW_COLS = [
    "Batch In",
    "Pipeline In",
    "Production",
]
 
DISPLAY_OUTFLOW_COLS = [
    "Batch Out",
    "Rack/Lifting",
    "Pipeline Out",
]
 
DISPLAY_NET_COLS = [
    "Adjustments",
    "Gain/Loss",
    "Transfers",
]
 
 
def _style_source_cells(df: pd.DataFrame, cols_to_color: list[str]) -> "pd.io.formats.style.Styler":
    cols = list(df.columns)
    cols_set = set(cols_to_color)
 
    def _row_style(row: pd.Series) -> list[str]:
        bg = SOURCE_BG.get(str(row.get("source", "")).strip().lower(), "")
        style = f"background-color: {bg};" if bg else ""
        return [style if (c in cols_set and style) else "" for c in cols]
 
    return df.style.apply(_row_style, axis=1).hide(axis="index")
 
 
def _to_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return 0.0
        return float(x)
    except Exception:
        return 0.0
 
 
def _sum_row(row: pd.Series, cols: list[str]) -> float:
    return float(sum(_to_float(row.get(c, 0.0)) for c in cols if c in row.index))
 
 
# def _recalculate_open_close_inv(df: pd.DataFrame, *, id_col: str) -> pd.DataFrame:
#     """Recompute Opening/Close inventory based on editable flow columns.
 
#     Streamlit's `st.data_editor` triggers a rerun on every edit. By recomputing
#     the derived columns and then rerunning once more, users see Opening/Close
#     update “live” as they change flows like Rack/Lifting, Batch In, etc.
 
#     Rules:
#     - Compute sequentially per (id_col, Product) ordered by Date.
#     - First row keeps its existing Opening Inv (or 0.0 if missing).
#     - Subsequent rows: Opening Inv := previous row's computed Close Inv.
#     - Close Inv := Opening + inflow - outflow + net.
#     """
#     if df is None or df.empty:
#         return df
 
#     out = df.copy()
 
#     # Work with datetimes internally for stable sorting; convert back to date at end.
#     out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
 
#     numeric_candidates = [
#         "Opening Inv",
#         "Close Inv",
#         *DISPLAY_INFLOW_COLS,
#         *DISPLAY_OUTFLOW_COLS,
#         *DISPLAY_NET_COLS,
#     ]
#     for c in numeric_candidates:
#         if c in out.columns:
#             out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
 
#     group_cols = [id_col]
#     if "Product" in out.columns:
#         group_cols.append("Product")
 
#     # Stable sort so we don't get UI flicker when other columns tie.
#     sort_cols = ["Date"] + group_cols
#     out = out.sort_values(sort_cols, kind="mergesort")
 
#     def _apply(g: pd.DataFrame) -> pd.DataFrame:
#         g = g.sort_values("Date", kind="mergesort").copy()
#         prev_close = 0.0
 
#         # NEW: Extract system and product for Magellan detection
#         system_val = g[id_col].iloc[0] if id_col in g.columns else None
#         product_val = g["Product"].iloc[0] if "Product" in g.columns else None
#         is_magellan = (id_col == "System" and str(system_val) == "Magellan")
 
#         for i, idx in enumerate(g.index):
 
#             # **KEY FIX: Skip recalculation for "system" source rows**
#             current_source = str(g.at[idx, "source"]).strip().lower() if "source" in g.columns else ""
           
#             if current_source == "system":
#                 # Use existing values from database
#                 if "Opening Inv" in g.columns:
#                     prev_close = _to_float(g.at[idx, "Close Inv"]) if "Close Inv" in g.columns else 0.0
#                 continue  # Don't recalculate system rows
 
#             if i == 0:
#                 opening = _to_float(g.at[idx, "Opening Inv"]) if "Opening Inv" in g.columns else 0.0
#             else:
#                 opening = prev_close
 
#             # MAGELLAN-SPECIFIC LOGIC: Use special formula for closing inventory
#             if is_magellan:
#                 # For Magellan: Closing = Adjustments - Rack/Lifting + Opening
#                 adjustments = _to_float(g.at[idx, "Adjustments"]) if "Adjustments" in g.columns else 0.0
#                 rack_lifting = _to_float(g.at[idx, "Rack/Lifting"]) if "Rack/Lifting" in g.columns else 0.0
#                 close = float(adjustments - rack_lifting + opening)
#             else:
#                 # STANDARD LOGIC: For all other systems (unchanged)
#                 inflow = _sum_row(g.loc[idx], DISPLAY_INFLOW_COLS)
#                 outflow = _sum_row(g.loc[idx], DISPLAY_OUTFLOW_COLS)
#                 net = _sum_row(g.loc[idx], DISPLAY_NET_COLS)
#                 close = float(opening + inflow - outflow + net)
 
#             if "Opening Inv" in g.columns:
#                 g.at[idx, "Opening Inv"] = opening
#             if "Close Inv" in g.columns:
#                 g.at[idx, "Close Inv"] = close
 
#             prev_close = close
 
#         return g
 
#     out = out.groupby(group_cols, dropna=False, group_keys=False).apply(_apply)
 
#     # Make sure the UI sees date-only values.
#     out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date
 
#     # Keep numbers tidy for display.
#     for c in out.columns:
#         if c in {"updated"}:
#             continue
#         if pd.api.types.is_numeric_dtype(out[c]):
#             out[c] = out[c].round(2)
 
#     return out
 
def _recalculate_open_close_inv(df: pd.DataFrame, *, id_col: str) -> pd.DataFrame:
    """Recompute Opening/Close inventory based on editable flow columns.
 
    Rules:
    - PRESERVE system source rows - keep their database Opening/Close Inv values
    - Compute sequentially per (id_col, Product) ordered by Date.
    - For forecast/manual rows:
      - Opening Inv := previous row's Close Inv
      - Close Inv := calculated based on flows (Magellan uses special formula)
    """
    if df is None or df.empty:
        return df
 
    out = df.copy()
 
    # Work with datetimes internally for stable sorting; convert back to date at end.
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
 
    numeric_candidates = [
        "Opening Inv",
        "Close Inv",
        *DISPLAY_INFLOW_COLS,
        *DISPLAY_OUTFLOW_COLS,
        *DISPLAY_NET_COLS,
    ]
    for c in numeric_candidates:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
 
    group_cols = [id_col]
    if "Product" in out.columns:
        group_cols.append("Product")
 
    # Stable sort so we don't get UI flicker when other columns tie.
    sort_cols = ["Date"] + group_cols
    out = out.sort_values(sort_cols, kind="mergesort")
 
    def _apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date", kind="mergesort").copy()
        prev_close = 0.0
 
        # Extract system and product for Magellan detection
        system_val = g[id_col].iloc[0] if id_col in g.columns else None
        product_val = g["Product"].iloc[0] if "Product" in g.columns else None
        is_magellan = (id_col == "System" and str(system_val) == "Magellan")
 
        for i, idx in enumerate(g.index):
            current_source = str(g.at[idx, "source"]).strip().lower() if "source" in g.columns else ""
           
            # For system rows: preserve existing Opening/Close Inv from database
            if current_source == "system":
                # Keep the database values as-is
                existing_close = _to_float(g.at[idx, "Close Inv"]) if "Close Inv" in g.columns else 0.0
                # Update prev_close so next row (forecast) can use this as opening
                prev_close = existing_close
                continue  # Don't recalculate system rows
           
            # For forecast/manual rows: calculate inventory
            if i == 0:
                # First row in group: use its existing opening or 0
                opening = _to_float(g.at[idx, "Opening Inv"]) if "Opening Inv" in g.columns else 0.0
            else:
                # Subsequent rows: opening = previous row's closing
                opening = prev_close
 
            # Calculate Closing Inv based on system type
            if is_magellan:
                # MAGELLAN FORMULA: Closing = Adjustments - Rack/Lifting + Opening
                adjustments = _to_float(g.at[idx, "Adjustments"]) if "Adjustments" in g.columns else 0.0
                rack_lifting = _to_float(g.at[idx, "Rack/Lifting"]) if "Rack/Lifting" in g.columns else 0.0
                close = float(adjustments - rack_lifting + opening)
            else:
                # STANDARD FORMULA: Closing = Opening + Inflow - Outflow + Net
                inflow = _sum_row(g.loc[idx], DISPLAY_INFLOW_COLS)
                outflow = _sum_row(g.loc[idx], DISPLAY_OUTFLOW_COLS)
                net = _sum_row(g.loc[idx], DISPLAY_NET_COLS)
                close = float(opening + inflow - outflow + net)
 
            # Update the dataframe with calculated values
            if "Opening Inv" in g.columns:
                g.at[idx, "Opening Inv"] = opening
            if "Close Inv" in g.columns:
                g.at[idx, "Close Inv"] = close
 
            # Update prev_close for the next iteration
            prev_close = close
 
        return g
 
    out = out.groupby(group_cols, dropna=False, group_keys=False).apply(_apply)
 
    # Make sure the UI sees date-only values.
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date
 
    # Keep numbers tidy for display.
    for c in out.columns:
        if c in {"updated"}:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(2)
 
    return out
 
 
 
def _needs_inventory_rerun(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    """Return True if Opening/Close differ between two dfs (shape-safe)."""
    if before is None or after is None:
        return False
    if before.shape[0] != after.shape[0]:
        return True
 
    for c in ["Opening Inv", "Close Inv"]:
        if c not in before.columns or c not in after.columns:
            continue
        b = pd.to_numeric(before[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        a = pd.to_numeric(after[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=0, atol=1e-9):
            return True
    return False
 
 
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
 
 
# def _roll_inventory(prev_close: float, flows: dict[str, float], flow_cols: list[str]) -> tuple[float, float]:
#     opening = float(prev_close)
#     inflow = sum(float(flows.get(c, 0.0) or 0.0) for c in INFLOW_COLS if c in flow_cols)
#     outflow = sum(float(flows.get(c, 0.0) or 0.0) for c in OUTFLOW_COLS if c in flow_cols)
#     net = sum(float(flows.get(c, 0.0) or 0.0) for c in NET_COLS if c in flow_cols)
#     closing = opening + inflow - outflow + net
#     return opening, closing
 
 
def _roll_inventory(prev_close: float, flows: dict[str, float], flow_cols: list[str], system: str = None, product: str = None) -> tuple[float, float]:
    # Opening inventory is always the previous day's closing inventory
    opening = float(prev_close)
   
    # MAGELLAN-SPECIFIC LOGIC: Special calculation for Midcon Magellan
    if system == "Magellan" and product:
        # For Magellan: Closing Inv = Adjustments - Rack/Lifting + Previous Day Closing Inv
        adjustments = float(flows.get("Adjustments", 0.0) or 0.0)
        rack_lifting = float(flows.get("Rack/Liftings", 0.0) or 0.0)
        closing = adjustments - rack_lifting + opening
    else:
        # STANDARD LOGIC: For all other systems/regions (unchanged)
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
 
            # opening, closing = _roll_inventory(prev_close, flows, flow_cols)
 
            # UPDATED CALL: Pass system/product for Magellan detection
            opening, closing = _roll_inventory(
                prev_close,
                flows,
                flow_cols,
                system=str(id_val) if id_col == "System" else None,  # NEW: Pass system name
                product=str(product) if product else None  # NEW: Pass product name
            )
 
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
 
 
def _threshold_values(*, region: str, location: str | None) -> tuple[float | None, float | None]:
    ovr = get_threshold_overrides(region=region, location=location)
    bottom = ovr.get("BOTTOM")
    safefill = ovr.get("SAFEFILL")
    b = float(bottom) if bottom is not None and not pd.isna(bottom) else None
    s = float(safefill) if safefill is not None and not pd.isna(safefill) else None
    return b, s
 
 
def _show_thresholds(*, region_label: str, bottom: float | None, safefill: float | None):
    c0, c1, c2 = st.columns([5, 2, 2])
 
    with c0:
        st.markdown(f"### 📍 {region_label}")
 
    with c1:
        v = "—" if safefill is None else f"{safefill:,.0f}"
        st.markdown(
            f"""
            <div class="mini-card">
              <p class="label">SafeFill</p>
              <p class="value">{v}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
 
    with c2:
        v = "—" if bottom is None else f"{bottom:,.0f}"
        st.markdown(
            f"""
            <div class="mini-card">
              <p class="label">Bottom</p>
              <p class="value">{v}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
 
 
def _build_editor_df(df_display: pd.DataFrame, *, id_col: str, ui_cols: list[str]) -> pd.DataFrame:
    """Return the dataframe we should keep in editor state.
 
    We intentionally keep *extra* flow columns (like Production/Adjustments)
    even if they aren't shown in `DETAILS_COLS`, so inventory math remains
    consistent.
    """
    extra = [
        "Production",
        "Adjustments",
    ]
    # Always keep columns required for grouping + lineage.
    base = ["Date", id_col, "source", "Product", "updated", "Notes", "Opening Inv", "Close Inv"]
    desired = []
    for c in base + ui_cols + extra:
        if c in df_display.columns and c not in desired:
            desired.append(c)
    return df_display[desired].reset_index(drop=True)
 
 
# def display_midcon_details(df_filtered: pd.DataFrame, active_region: str, forecast_end: pd.Timestamp):
#     st.subheader("🧾 Group Daily Details")
 
#     if df_filtered.empty:
#         st.info("No data available for the selected filters.")
#         return
 
#     df_all = _extend_with_30d_forecast(df_filtered, id_col="System", forecast_end=forecast_end)
   
#     df_display, cols = build_details_view(df_all, id_col="System")
   
#     scope_sys = None
#     if df_filtered is not None and not df_filtered.empty and "System" in df_filtered.columns:
#         systems = sorted(df_filtered["System"].dropna().unique().tolist())
#         if len(systems) == 1:
#             scope_sys = systems[0]
 
#     bottom, safefill = _threshold_values(region=active_region, location=str(scope_sys) if scope_sys is not None else None)
#     _show_thresholds(region_label=active_region, bottom=bottom, safefill=safefill)
 
#     visible = get_visible_columns(region=active_region, location=str(scope_sys) if scope_sys is not None else None)
#     must_have = ["Date", "System", "Product", "Opening Inv", "Close Inv"]
#     column_order = []
#     for c in must_have + visible:
#         if c in cols and c not in column_order and c != "source":
#             column_order.append(c)
 
#     locked_cols = _locked_cols("System", cols)
#     column_config = _column_config(df_display, cols, "System")
 
#     column_config = {k: v for k, v in column_config.items() if k in column_order}
 
#     # Ensure we have a RangeIndex so `hide_index=True` works with `num_rows='dynamic'`.
#     base_key = f"{active_region}_edit"
#     df_key = f"{base_key}__df"
#     ver_key = f"{base_key}__ver"
#     widget_key = f"{base_key}__v{int(st.session_state.get(ver_key, 0))}"
 
#     editor_df = _build_editor_df(df_display, id_col="System", ui_cols=cols)
 
#     # Keep editor state across reruns so we can push computed column updates back in.
#     if df_key not in st.session_state or list(st.session_state[df_key].columns) != list(editor_df.columns):
#         st.session_state[df_key] = _recalculate_open_close_inv(editor_df, id_col="System")
 
#     styled = _style_source_cells(st.session_state[df_key], locked_cols)
 
#     # edited = st.data_editor(
#     #     styled,
#     #     num_rows="dynamic",
#     #     width="stretch",
#     #     height=DETAILS_EDITOR_HEIGHT_PX,
#     #     hide_index=True,
#     #     column_order=column_order,
#     #     key=widget_key,
#     #     column_config=column_config,
#     # )
#     edited = st.data_editor(
#         styled,
#         use_container_width=True,
#         height=400,
#         hide_index=True,
#         column_order=column_order,
#         key=widget_key,
#         column_config=column_config,
#     )
 
 
#     # Recompute derived inventory columns based on the just-edited flows.
#     recomputed = _recalculate_open_close_inv(edited, id_col="System")
#     st.session_state[df_key] = recomputed
 
#     # Force one extra rerun so the editor repaints with the new Opening/Close values.
#     if _needs_inventory_rerun(edited, recomputed):
#         st.session_state[ver_key] = int(st.session_state.get(ver_key, 0)) + 1
#         st.rerun()
 
#     st.markdown('<div class="save-btn-bottom">', unsafe_allow_html=True)
#     if st.button("💾 Save Changes", key=f"save_{active_region}"):
#         st.success("✅ Changes saved successfully!")
#     st.markdown("</div>", unsafe_allow_html=True)
def display_midcon_details(df_filtered: pd.DataFrame, active_region: str, forecast_end: pd.Timestamp):
    st.subheader("🧾 Group Daily Details")
 
    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return
 
    # # DEBUG: Show what we're getting
    # st.info(f"Input rows: {len(df_filtered)}, Date range: {df_filtered['Date'].min()} to {df_filtered['Date'].max()}")
    # st.info(f"Forecast end date: {forecast_end}")
    # st.info(f"Unique sources: {df_filtered['source'].unique() if 'source' in df_filtered.columns else 'No source column'}")
 
    # df_all = _extend_with_30d_forecast(df_filtered, id_col="System", forecast_end=forecast_end)
   
    # # DEBUG: Show what forecast generated
    # st.info(f"After forecast rows: {len(df_all)}, Date range: {df_all['Date'].min()} to {df_all['Date'].max()}")
    # if 'source' in df_all.columns:
    #     st.info(f"Source breakdown: {df_all['source'].value_counts().to_dict()}")
   
    # FIX: Extend forecast_end by 30 days to actually generate forecasts
    actual_forecast_end = forecast_end + pd.Timedelta(days=30)
   
    df_all = _extend_with_30d_forecast(df_filtered, id_col="System", forecast_end=actual_forecast_end)
   
    df_display, cols = build_details_view(df_all, id_col="System")
    df_display, cols = build_details_view(df_all, id_col="System")
 
    scope_sys = None
    if df_filtered is not None and not df_filtered.empty and "System" in df_filtered.columns:
        systems = sorted(df_filtered["System"].dropna().unique().tolist())
        if len(systems) == 1:
            scope_sys = systems[0]
 
    bottom, safefill = _threshold_values(region=active_region, location=str(scope_sys) if scope_sys is not None else None)
    _show_thresholds(region_label=active_region, bottom=bottom, safefill=safefill)
 
    visible = get_visible_columns(region=active_region, location=str(scope_sys) if scope_sys is not None else None)
    must_have = ["Date", "System", "Product", "Opening Inv", "Close Inv"]
    column_order = []
    for c in must_have + visible:
        if c in cols and c not in column_order and c != "source":
            column_order.append(c)
 
    locked_cols = _locked_cols("System", cols)
    column_config = _column_config(df_display, cols, "System")
    column_config = {k: v for k, v in column_config.items() if k in column_order}
 
    base_key = f"{active_region}_edit"
    df_key = f"{base_key}__df"
    ver_key = f"{base_key}__ver"
    selected_prods = st.session_state.get("selected_prods", [])
    prods_key = f"{base_key}__prods"
   
    if prods_key not in st.session_state or st.session_state[prods_key] != selected_prods:
        st.session_state[prods_key] = selected_prods
        if df_key in st.session_state:
            del st.session_state[df_key]  # Force rebuild with new filters
    widget_key = f"{base_key}__v{int(st.session_state.get(ver_key, 0))}"
 
    editor_df = _build_editor_df(df_display, id_col="System", ui_cols=cols)
 
    if df_key not in st.session_state or list(st.session_state[df_key].columns) != list(editor_df.columns):
        st.session_state[df_key] = _recalculate_open_close_inv(editor_df, id_col="System")
 
    styled = _style_source_cells(st.session_state[df_key], locked_cols)
 
    edited = st.data_editor(
        styled,
        use_container_width=True,
        height=400,
        hide_index=True,
        column_order=column_order,
        key=widget_key,
        column_config=column_config,
    )
 
    recomputed = _recalculate_open_close_inv(edited, id_col="System")
    st.session_state[df_key] = recomputed
 
    if _needs_inventory_rerun(edited, recomputed):
        st.session_state[ver_key] = int(st.session_state.get(ver_key, 0)) + 1
        st.rerun()
 
    st.markdown('<div class="save-btn-bottom">', unsafe_allow_html=True)
    if st.button("💾 Save Changes", key=f"save_{active_region}"):
        st.success("✅ Changes saved successfully!")
    st.markdown("</div>", unsafe_allow_html=True)
 
def display_location_details(df_filtered: pd.DataFrame, active_region: str, forecast_end: pd.Timestamp):
    st.subheader("🏭 Locations")
 
    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return
 
    # Prefer the user-selected locations from the sidebar (when available) so that
    # a location tab doesn't “disappear” simply because other filters (like Product)
    # result in no rows for that location.
    selected_locs = st.session_state.get("selected_locs")
    if isinstance(selected_locs, (list, tuple)) and len(selected_locs):
        region_locs = [str(x) for x in selected_locs]
    else:
        region_locs = sorted(df_filtered["Location"].dropna().unique().tolist()) if "Location" in df_filtered.columns else []
 
    if not region_locs:
        st.write("*(No locations available in the current selection)*")
        return
 
    for i, loc in enumerate(st.tabs(region_locs)):
        with loc:
            # st.markdown(f"### 📍 {region_locs[i]}")
            df_loc = df_filtered[df_filtered["Location"] == region_locs[i]]
 
            if df_loc.empty:
                st.write("*(No data for this location)*")
            else:
                actual_forecast_end = forecast_end + pd.Timedelta(days=30)
                df_all = _extend_with_30d_forecast(df_loc, id_col="Location", forecast_end=actual_forecast_end)
                df_display, cols = build_details_view(df_all, id_col="Location")
 
                bottom, safefill = _threshold_values(region=active_region, location=str(region_locs[i]))
                _show_thresholds(region_label=str(region_locs[i]), bottom=bottom, safefill=safefill)
 
                visible = get_visible_columns(region=active_region, location=str(region_locs[i]))
                must_have = ["Date", "Location", "Product", "Opening Inv", "Close Inv"]
                column_order = []
                for c in must_have + visible:
                    if c in cols and c not in column_order and c != "source":
                        column_order.append(c)
 
                locked_cols = _locked_cols("Location", cols)
                column_config = _column_config(df_display, cols, "Location")
 
                column_config = {k: v for k, v in column_config.items() if k in column_order}
 
                # Ensure we have a RangeIndex so `hide_index=True` works with `num_rows='dynamic'`.
                base_key = f"{active_region}_{region_locs[i]}_edit"
                df_key = f"{base_key}__df"
                ver_key = f"{base_key}__ver"
                selected_prods = st.session_state.get("selected_prods", [])
                prods_key = f"{base_key}__prods"
               
                if prods_key not in st.session_state or st.session_state[prods_key] != selected_prods:
                    st.session_state[prods_key] = selected_prods
                    if df_key in st.session_state:
                        del st.session_state[df_key]  # Force rebuild with new filters
                widget_key = f"{base_key}__v{int(st.session_state.get(ver_key, 0))}"
 
                editor_df = _build_editor_df(df_display, id_col="Location", ui_cols=cols)
 
                if df_key not in st.session_state or list(st.session_state[df_key].columns) != list(editor_df.columns):
                    st.session_state[df_key] = _recalculate_open_close_inv(editor_df, id_col="Location")
 
                styled = _style_source_cells(st.session_state[df_key], locked_cols)
 
                edited = st.data_editor(
                    styled,
                    num_rows="dynamic",
                    use_container_width=True,
                    height=DETAILS_EDITOR_HEIGHT_PX,
                    hide_index=True,
                    column_order=column_order,
                    key=widget_key,
                    column_config=column_config,
                )
 
                recomputed = _recalculate_open_close_inv(edited, id_col="Location")
                st.session_state[df_key] = recomputed
                if _needs_inventory_rerun(edited, recomputed):
                    st.session_state[ver_key] = int(st.session_state.get(ver_key, 0)) + 1
                    st.rerun()
 
            st.markdown('<div class="save-btn-bottom">', unsafe_allow_html=True)
            if st.button(f"💾 Save {region_locs[i]}", key=f"save_{active_region}_{region_locs[i]}"):
                st.success(f"✅ Changes for {region_locs[i]} saved successfully!")
            st.markdown("</div>", unsafe_allow_html=True)
 
 
def display_details_tab(df_filtered: pd.DataFrame, active_region: str, end_ts: pd.Timestamp):
    if active_region == "Midcon":
        display_midcon_details(df_filtered, active_region, forecast_end=end_ts)
    else:
        display_location_details(df_filtered, active_region, forecast_end=end_ts)
 