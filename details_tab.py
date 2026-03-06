"""Details tab: inventory data grid, editing, and save flow."""
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from admin_config import get_visible_columns, get_threshold_overrides, get_rack_lifting_forecast_method
from app_logging import logged_button, log_audit
from config import (
    COL_ACCOUNTING_INV,
    COL_ADJUSTMENTS,
    COL_ADJUSTMENTS_FACT,
    COL_ARGENTINE,
    COL_AVAILABLE,
    COL_AVAILABLE_FACT,
    COL_AVAILABLE_SPACE,
    COL_BATCH,
    COL_BATCH_BREAKDOWN,
    COL_BATCH_IN,
    COL_BATCH_IN_FACT,
    COL_BATCH_IN_FACT_RAW,
    COL_BATCH_IN_RAW,
    COL_BATCH_OUT,
    COL_BATCH_OUT_FACT,
    COL_BATCH_OUT_FACT_RAW,
    COL_BATCH_OUT_RAW,
    COL_CALCULATED_RECEIPT,
    COL_CLOSE_INV_FACT_RAW,
    COL_CLOSE_INV_RAW,
    COL_EL_DORADO,
    COL_FROM_327_RECEIPT,
    COL_GAIN_LOSS,
    COL_GAIN_LOSS_FACT,
    COL_INTRANSIT,
    COL_INTRANSIT_FACT,
    COL_LOADABLE,
    COL_MTD_AVG_RACK,
    COL_NOTES,
    COL_OPEN_INV_FACT_RAW,
    COL_OPEN_INV_RAW,
    COL_OPENING_INV,
    COL_OPENING_INV_FACT,
    COL_OTHER,
    COL_PIPELINE_IN,
    COL_PIPELINE_IN_FACT,
    COL_PIPELINE_OUT,
    COL_PIPELINE_OUT_FACT,
    COL_PRODUCT,
    COL_PRODUCTION,
    COL_PRODUCTION_FACT,
    COL_RACK_LIFTING,
    COL_RACK_LIFTING_FACT,
    COL_RACK_LIFTINGS_FACT_RAW,
    COL_RACK_LIFTINGS_RAW,
    COL_STORAGE,
    COL_TOTAL_CLOSING_INV,
    COL_TOTAL_INVENTORY,
    COL_TRANSFERS,
    COL_TRANSFERS_FACT,
    COL_TULSA,
    COL_VESSEL,
    COL_VESSEL_VOLUME,
    COL_7DAY_AVG_RACK,
    DATA_SOURCE,
    DETAILS_RENAME_MAP,
    ROLE_DISPLAY,
)
from data_loader import persist_details_rows, get_user_role, load_filtered_inventory_data, _load_inventory_data_filtered_cached
from ui_components import _render_blocking_overlay, _render_threshold_cards, _view_files_dialog
from utils import dynamic_input_data_editor, _to_float, _to_numeric_series, _sum_row


# ---------------------------------------------------------------------------
# Column lists
# ---------------------------------------------------------------------------

# Columns shown in the details editor (post-rename display names)
DETAILS_COLS = [
    COL_PRODUCT,
    COL_OPENING_INV,
    COL_AVAILABLE,
    COL_INTRANSIT,
    COL_CLOSE_INV_RAW,
    COL_TOTAL_CLOSING_INV,
    COL_AVAILABLE_SPACE,
    COL_LOADABLE,
    COL_TOTAL_INVENTORY,
    COL_ACCOUNTING_INV,
    COL_BATCH_IN,
    COL_BATCH_OUT,
    COL_RACK_LIFTING,
    COL_CALCULATED_RECEIPT,
    COL_7DAY_AVG_RACK,
    COL_MTD_AVG_RACK,
    COL_PIPELINE_IN,
    COL_PIPELINE_OUT,
    COL_GAIN_LOSS,
    COL_TRANSFERS,
    COL_PRODUCTION,
    COL_ADJUSTMENTS,
    COL_TULSA,
    COL_EL_DORADO,
    COL_OTHER,
    COL_ARGENTINE,
    COL_FROM_327_RECEIPT,
    COL_STORAGE,
    COL_VESSEL,
    COL_VESSEL_VOLUME,
    COL_BATCH,
    COL_BATCH_BREAKDOWN,
    COL_NOTES,
]

# UI-only action column (never persisted)
COL_VIEW_FILE = "View File"

# Lineage/tracking columns preserved through build_details_view
_TRACKING_COLS = {"updated", "SOURCE_TYPE", "FILE_LOCATION"}

# ---------------------------------------------------------------------------
# Flow column groups
# Pre-rename names — used in forecast / _roll_inventory (operate on raw data)
# ---------------------------------------------------------------------------

FORECAST_FLOW_COLS = [
    COL_BATCH_IN_RAW,
    COL_BATCH_OUT_RAW,
    COL_RACK_LIFTINGS_RAW,
    COL_PIPELINE_IN,
    COL_PIPELINE_OUT,
    COL_PRODUCTION,
    COL_ADJUSTMENTS,
    COL_GAIN_LOSS,
    COL_TRANSFERS,
]

INFLOW_COLS = [COL_BATCH_IN_RAW, COL_PIPELINE_IN, COL_PRODUCTION]
OUTFLOW_COLS = [COL_BATCH_OUT_RAW, COL_RACK_LIFTINGS_RAW, COL_PIPELINE_OUT]
NET_COLS = [COL_ADJUSTMENTS, COL_GAIN_LOSS, COL_TRANSFERS]

# Post-rename display names — used in _recalculate_open_close_inv
DISPLAY_INFLOW_COLS = [COL_BATCH_IN, COL_PIPELINE_IN, COL_PRODUCTION,
                       COL_TULSA, COL_EL_DORADO, COL_OTHER, COL_FROM_327_RECEIPT]
DISPLAY_OUTFLOW_COLS = [COL_BATCH_OUT, COL_RACK_LIFTING, COL_PIPELINE_OUT, COL_ARGENTINE, COL_VESSEL_VOLUME]
DISPLAY_NET_COLS = [COL_ADJUSTMENTS, COL_GAIN_LOSS, COL_TRANSFERS]

# ---------------------------------------------------------------------------
# Fact column mappings  (base display name → fact display name)
# ---------------------------------------------------------------------------

FACT_COL_MAP: dict[str, str] = {
    COL_OPENING_INV: COL_OPENING_INV_FACT,
    COL_AVAILABLE: COL_AVAILABLE_FACT,
    COL_INTRANSIT: COL_INTRANSIT_FACT,
    COL_CLOSE_INV_RAW: COL_CLOSE_INV_FACT_RAW,
    COL_BATCH_IN: COL_BATCH_IN_FACT,
    COL_BATCH_OUT: COL_BATCH_OUT_FACT,
    COL_RACK_LIFTING: COL_RACK_LIFTING_FACT,
    COL_PIPELINE_IN: COL_PIPELINE_IN_FACT,
    COL_PIPELINE_OUT: COL_PIPELINE_OUT_FACT,
    COL_ADJUSTMENTS: COL_ADJUSTMENTS_FACT,
    COL_GAIN_LOSS: COL_GAIN_LOSS_FACT,
    COL_TRANSFERS: COL_TRANSFERS_FACT,
    COL_PRODUCTION: COL_PRODUCTION_FACT,
}

FACT_COL_MAP_RAW: dict[str, str] = {
    COL_OPEN_INV_RAW: COL_OPEN_INV_FACT_RAW,
    COL_CLOSE_INV_RAW: COL_CLOSE_INV_FACT_RAW,
    COL_BATCH_IN_RAW: COL_BATCH_IN_FACT_RAW,
    COL_BATCH_OUT_RAW: COL_BATCH_OUT_FACT_RAW,
    COL_RACK_LIFTINGS_RAW: COL_RACK_LIFTINGS_FACT_RAW,
    COL_PIPELINE_IN: COL_PIPELINE_IN_FACT,
    COL_PIPELINE_OUT: COL_PIPELINE_OUT_FACT,
    COL_ADJUSTMENTS: COL_ADJUSTMENTS_FACT,
    COL_GAIN_LOSS: COL_GAIN_LOSS_FACT,
    COL_TRANSFERS: COL_TRANSFERS_FACT,
    COL_PRODUCTION: COL_PRODUCTION_FACT,
}

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------

SYSTEM_DISCREPANCY_THRESHOLD_BBL = 5.0

# Row / cell background colors
TODAY_BG = "#cce5ff"
MATCH_BG = "#d9f2d9"
MISMATCH_BG = "#fff2cc"
OLDER_DAY_BG = "#e8e8e8"
FACT_BG = "#eeeeee"
CLOSE_INV_ABOVE_SAFEFILL_BG = "#ffb3b3"
CLOSE_INV_BELOW_BOTTOM_BG = "#ffb3b3"
YESTERDAY_HIGHLIGHT_COLS = {"Date", "Opening Inv", "Close Inv"}
OLDER_DAY_HIGHLIGHT_COLS = {"Date", "Opening Inv", "Close Inv"}

DETAILS_EDITOR_VISIBLE_ROWS = 15
DETAILS_EDITOR_ROW_PX = 35
DETAILS_EDITOR_HEADER_PX = 35
DETAILS_EDITOR_HEIGHT_PX = DETAILS_EDITOR_HEADER_PX + (DETAILS_EDITOR_VISIBLE_ROWS * DETAILS_EDITOR_ROW_PX)

# Columns that are always read-only  ({id_col} is substituted at runtime)
LOCKED_BASE_COLS = [
    "Date", "{id_col}", "Product", "Close Inv",
    COL_TOTAL_CLOSING_INV, COL_AVAILABLE_SPACE, COL_LOADABLE,
    COL_TOTAL_INVENTORY, COL_ACCOUNTING_INV, COL_7DAY_AVG_RACK, COL_MTD_AVG_RACK,
    COL_CALCULATED_RECEIPT,
    "Opening Inv",
]


# ---------------------------------------------------------------------------
# Inventory calculations
# ---------------------------------------------------------------------------

def _recalculate_open_close_inv(df: pd.DataFrame, *, id_col: str) -> pd.DataFrame:
    """Roll Opening Inv and Close Inv forward row-by-row using flow columns."""
    if df is None or df.empty:
        return df

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    # Coerce all flow and inventory columns to numeric up front
    for c in ["Opening Inv", "Opening Inv Fact", COL_AVAILABLE, COL_AVAILABLE_FACT,
              COL_INTRANSIT, COL_INTRANSIT_FACT, "Close Inv", "Close Inv Fact",
              *DISPLAY_INFLOW_COLS, *DISPLAY_OUTFLOW_COLS, *DISPLAY_NET_COLS]:
        if c in out.columns:
            out[c] = _to_numeric_series(out[c]).fillna(0.0)

    group_cols = [id_col] + (["Product"] if "Product" in out.columns else [])
    sort_cols = ["Date"] + group_cols
    out = out.sort_values(sort_cols, kind="mergesort")

    def _apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date", kind="mergesort").copy()
        prev_close = 0.0
        for i, idx in enumerate(g.index):
            opening = _to_float(g.at[idx, "Opening Inv"]) if i == 0 and "Opening Inv" in g.columns else prev_close
            inflow = _sum_row(g.loc[idx], DISPLAY_INFLOW_COLS)
            outflow = _sum_row(g.loc[idx], DISPLAY_OUTFLOW_COLS)
            net = _sum_row(g.loc[idx], DISPLAY_NET_COLS)
            close = opening + inflow - outflow + net
            if "Opening Inv" in g.columns:
                g.at[idx, "Opening Inv"] = opening
            if "Close Inv" in g.columns:
                g.at[idx, "Close Inv"] = close
            prev_close = close
        return g

    parts = [_apply(g) for _, g in out.groupby(group_cols, dropna=False, sort=False)]
    out = pd.concat(parts, axis=0) if parts else out
    out = out.sort_values(sort_cols, kind="mergesort")
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date

    for c in out.columns:
        if c != "updated" and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(2)
    return out


def _recalculate_total_closing_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Total Closing Inv = Available + Intransit."""
    if df is None or df.empty:
        return df
    out = df.copy()
    available = _to_numeric_series(out[COL_AVAILABLE]).fillna(0.0) if COL_AVAILABLE in out.columns else pd.Series(0.0, index=out.index)
    intransit = _to_numeric_series(out[COL_INTRANSIT]).fillna(0.0) if COL_INTRANSIT in out.columns else pd.Series(0.0, index=out.index)
    out[COL_TOTAL_CLOSING_INV] = (available.astype(float) + intransit.astype(float)).round(2)
    return out


def _recalculate_available_space(df: pd.DataFrame, *, safefill: float | None) -> pd.DataFrame:
    """Available Space = SafeFill − Close Inv."""
    if df is None or df.empty or "Close Inv" not in df.columns:
        return df
    out = df.copy()
    close = _to_numeric_series(out["Close Inv"]).fillna(0.0)
    out[COL_AVAILABLE_SPACE] = (float(safefill) - close.astype(float)).round(2) if safefill is not None else np.nan
    return out


def _recalculate_loadable(df: pd.DataFrame, *, bottom: float | None) -> pd.DataFrame:
    """Loadable = Close Inv − Bottom."""
    if df is None or df.empty or "Close Inv" not in df.columns:
        return df
    out = df.copy()
    close = _to_numeric_series(out["Close Inv"]).fillna(0.0)
    out[COL_LOADABLE] = (close.astype(float) - float(bottom)).round(2) if bottom is not None else np.nan
    return out


def _recalculate_total_inventory(df: pd.DataFrame, *, bottom: float | None) -> pd.DataFrame:
    """Total Inventory = Close Inv + Bottom."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "Close Inv" not in out.columns or bottom is None:
        out[COL_TOTAL_INVENTORY] = np.nan
        return out
    close = _to_numeric_series(out["Close Inv"]).fillna(0.0)
    out[COL_TOTAL_INVENTORY] = (close.astype(float) + float(bottom)).round(2)
    return out


def _recalculate_accounting_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Accounting Inventory = Close Inv − Storage."""
    if df is None or df.empty or "Close Inv" not in df.columns:
        return df
    out = df.copy()
    if COL_STORAGE not in out.columns:
        out[COL_STORAGE] = np.nan
    close = _to_numeric_series(out["Close Inv"]).fillna(0.0)
    storage = _to_numeric_series(out[COL_STORAGE]).fillna(0.0)
    out[COL_ACCOUNTING_INV] = (close.astype(float) - storage.astype(float)).round(2)
    return out


def _recalculate_calculated_receipt(
    df: pd.DataFrame,
    *,
    id_col: str,
    df_hist: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculated Receipt = Today's Available \u2212 Yesterday's Available + Today's Rack/Lifting.

    Behaviour:
    - Past rows and today: always computed.  When the date filter starts mid-period
      (so the first visible row has no in-grid predecessor), the function looks back
      into the full historical data (df_hist) to find the prior-day Available \u2014 so
      the value is never NaN purely because of the filter window.
    - Future rows (date > today, i.e. forecast rows): always NaN \u2014 no meaningful
      receipt figure can be derived from projected Available values.
    - Groups are partitioned by (id_col, Product) so each location/product is
      independent.
    """
    if df is None or df.empty:
        return df
    if COL_AVAILABLE not in df.columns:
        df = df.copy()
        df[COL_CALCULATED_RECEIPT] = np.nan
        return df

    out = df.copy()
    today = pd.Timestamp.today().normalize()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    # Coerce inputs
    out[COL_AVAILABLE] = _to_numeric_series(out[COL_AVAILABLE]).fillna(0.0)
    if COL_RACK_LIFTING in out.columns:
        out[COL_RACK_LIFTING] = _to_numeric_series(out[COL_RACK_LIFTING]).fillna(0.0)
    else:
        out[COL_RACK_LIFTING] = 0.0

    # Build a prior-day Available lookup from full history so the first visible row
    # of a filtered window never suffers a spurious NaN.
    # Key: (id_val, prod_val, date_normalized) -> available
    hist_avail: dict[tuple, float] = {}
    if df_hist is not None and not df_hist.empty and COL_AVAILABLE in df_hist.columns:
        h = df_hist.copy()
        h["Date"] = pd.to_datetime(h["Date"], errors="coerce")
        h = h[h["Date"].notna()]
        # Exclude forecast rows from the reference data
        if "SOURCE_TYPE" in h.columns:
            h = h[h["SOURCE_TYPE"].astype(str).str.lower() != "forecast"]
        h[COL_AVAILABLE] = pd.to_numeric(h[COL_AVAILABLE], errors="coerce").fillna(0.0)
        id_col_h = id_col if id_col in h.columns else None
        prod_col_h = "Product" if "Product" in h.columns else None
        for _, row in h.iterrows():
            id_val = str(row[id_col_h]) if id_col_h else "*"
            prod_val = str(row[prod_col_h]) if prod_col_h else "*"
            hist_avail[(id_val, prod_val, row["Date"].normalize())] = float(row[COL_AVAILABLE])

    group_cols = [id_col] + (["Product"] if "Product" in out.columns else [])
    sort_cols = ["Date"] + group_cols
    out = out.sort_values(sort_cols, kind="mergesort")

    def _apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date", kind="mergesort").copy()
        avail_s = _to_numeric_series(g[COL_AVAILABLE]).fillna(0.0)
        rack_s = _to_numeric_series(g[COL_RACK_LIFTING]).fillna(0.0)
        id_val = str(g[id_col].iloc[0]) if id_col in g.columns else "*"
        prod_val = str(g["Product"].iloc[0]) if "Product" in g.columns else "*"
        results: list = []
        for i, idx in enumerate(g.index):
            row_date = g.at[idx, "Date"]
            # Forecast rows (date > today) -> NaN
            if pd.isna(row_date) or row_date.normalize() > today:
                results.append(np.nan)
                continue
            today_avail = float(avail_s.iloc[i])
            today_rack = float(rack_s.iloc[i])
            if i > 0:
                # Prior row is within the grid -- use it directly
                prev_avail: float | None = float(avail_s.iloc[i - 1])
            else:
                # First visible row: resolve prior-day Available from full history
                yesterday = row_date.normalize() - pd.Timedelta(days=1)
                prev_avail = hist_avail.get((id_val, prod_val, yesterday))
            if prev_avail is None or (isinstance(prev_avail, float) and np.isnan(prev_avail)):
                results.append(np.nan)
            else:
                results.append(round(today_avail - prev_avail + today_rack, 2))
        g[COL_CALCULATED_RECEIPT] = results
        return g

    parts = [_apply(g) for _, g in out.groupby(group_cols, dropna=False, sort=False)]
    out = pd.concat(parts, axis=0) if parts else out
    out = out.sort_values(sort_cols, kind="mergesort")
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date
    return out
def _fill_rack_averages_per_row(df: pd.DataFrame, df_hist: pd.DataFrame) -> pd.DataFrame:
    """Compute per-row 7-day and MTD rack averages from historical data."""
    if df is None or df.empty:
        return df

    out = df.copy()
    rack_col = next((c for c in (COL_RACK_LIFTINGS_RAW, COL_RACK_LIFTING) if c in df_hist.columns), None)
    if rack_col is None:
        out[COL_7DAY_AVG_RACK] = np.nan
        out[COL_MTD_AVG_RACK] = np.nan
        return out

    # Build clean historical base: no forecast rows, up to today, daily aggregated
    hist = df_hist.copy()
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    if "SOURCE_TYPE" in hist.columns:
        hist = hist[hist["SOURCE_TYPE"].astype(str).str.lower() != "forecast"]
    hist = hist.dropna(subset=["Date"])
    hist[rack_col] = pd.to_numeric(hist[rack_col], errors="coerce")
    today = pd.Timestamp.today().normalize()
    hist = hist[hist["Date"] <= today]
    hist = hist.groupby("Date", as_index=False)[rack_col].sum().sort_values("Date").reset_index(drop=True)

    dates = pd.to_datetime(out["Date"], errors="coerce")
    avg_7day_list, avg_mtd_list = [], []
    for row_date in dates:
        if pd.isna(row_date) or row_date > today:
            avg_7day_list.append(np.nan)
            avg_mtd_list.append(np.nan)
            continue
        h_upto = hist[hist["Date"] <= row_date]
        v7 = h_upto.tail(7)[rack_col].mean()
        avg_7day_list.append(round(float(v7), 2) if pd.notna(v7) else np.nan)
        month_start = row_date.replace(day=1)
        vm = h_upto[h_upto["Date"] >= month_start][rack_col].mean()
        avg_mtd_list.append(round(float(vm), 2) if pd.notna(vm) else np.nan)

    out[COL_7DAY_AVG_RACK] = avg_7day_list
    out[COL_MTD_AVG_RACK] = avg_mtd_list
    return out


def _overlay_rack_edits(df_prod_raw: pd.DataFrame, edited_df: pd.DataFrame) -> pd.DataFrame:
    """Merge live rack edits from the editor back into the raw historical df for accurate averages."""
    if df_prod_raw is None or df_prod_raw.empty:
        return df_prod_raw
    if edited_df is None or edited_df.empty or COL_RACK_LIFTING not in edited_df.columns:
        return df_prod_raw
    if COL_RACK_LIFTINGS_RAW not in df_prod_raw.columns:
        return df_prod_raw

    hist = df_prod_raw.copy()
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    ev = edited_df[["Date", COL_RACK_LIFTING]].copy()
    ev["Date"] = pd.to_datetime(ev["Date"], errors="coerce")
    ev = ev.dropna(subset=["Date"])
    ev[COL_RACK_LIFTING] = pd.to_numeric(ev[COL_RACK_LIFTING], errors="coerce")
    ev_grouped = ev.groupby("Date")[COL_RACK_LIFTING].last().reset_index()
    ev_grouped.columns = ["Date", "__rack_edit"]
    hist = hist.merge(ev_grouped, on="Date", how="left")
    mask = hist["__rack_edit"].notna()
    hist.loc[mask, COL_RACK_LIFTINGS_RAW] = hist.loc[mask, "__rack_edit"]
    return hist.drop(columns=["__rack_edit"])


def _recalculate_inventory_metrics(
    df: pd.DataFrame,
    *,
    id_col: str,
    safefill: float | None,
    bottom: float | None = None,
    df_hist: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run all derived column calculations in sequence."""
    out = _recalculate_open_close_inv(df, id_col=id_col)
    out = _recalculate_total_closing_inv(out)
    out = _recalculate_available_space(out, safefill=safefill)
    out = _recalculate_loadable(out, bottom=bottom)
    out = _recalculate_total_inventory(out, bottom=bottom)
    out = _recalculate_accounting_inv(out)
    out = _recalculate_calculated_receipt(out, id_col=id_col, df_hist=df_hist)
    if df_hist is not None:
        out = _fill_rack_averages_per_row(out, df_hist)
    return out


def _needs_inventory_rerun(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    """Return True if calculated inventory columns have changed enough to warrant a re-render."""
    if before is None or after is None or before.shape[0] != after.shape[0]:
        return False
    for c in ["Opening Inv", "Close Inv", COL_ACCOUNTING_INV, COL_CALCULATED_RECEIPT]:
        if c not in before.columns or c not in after.columns:
            continue
        b = _to_numeric_series(before[c]).fillna(0.0).to_numpy(dtype=float)
        a = _to_numeric_series(after[c]).fillna(0.0).to_numpy(dtype=float)
        # 0.005 bbl tolerance absorbs float noise without masking real user changes
        if not np.allclose(a, b, rtol=0, atol=0.005):
            return True
    return False


# ---------------------------------------------------------------------------
# Forecast helpers
# ---------------------------------------------------------------------------

def _available_flow_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in FORECAST_FLOW_COLS if c in df.columns]


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
    fallback = {c: float(recent[c].mean()) if c in recent.columns else 0.0 for c in flow_cols}
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
            vals = subset[c].astype(float).to_numpy() if c in subset.columns else np.zeros(len(subset))
            out[(wd, c)] = float((vals * weights).sum() / wsum)
    return out


def _constant_means_excluding_zeros(
    hist: pd.DataFrame,
    flow_cols: list[str],
    *,
    tail_n: int | None,
) -> dict[str, float]:
    if hist is None or hist.empty or not flow_cols:
        return {c: 0.0 for c in (flow_cols or [])}
    h = hist.sort_values("Date").copy()
    if tail_n is not None:
        h = h.tail(int(tail_n))
    return {
        c: float(pd.to_numeric(h[c], errors="coerce").mean()) if c in h.columns and pd.to_numeric(h[c], errors="coerce").notna().any() else 0.0
        for c in flow_cols
    }


def _make_forecast_flow_estimator(hist: pd.DataFrame, *, flow_cols: list[str], method: str):
    """Return a callable(date) -> dict[col, value] for the requested forecast method."""
    m = str(method or "").strip() or "weekday_weighted"

    if m == "7_day_avg":
        const = _constant_means_excluding_zeros(hist, flow_cols, tail_n=7)
        return lambda d: dict(const)

    if m == "mtd_avg":
        _today = pd.Timestamp.today().normalize()
        _hist_mtd = hist.copy()
        _hist_mtd["Date"] = pd.to_datetime(_hist_mtd["Date"], errors="coerce")
        _hist_mtd = _hist_mtd[(_hist_mtd["Date"] >= _today.replace(day=1)) & (_hist_mtd["Date"] <= _today)]
        if _hist_mtd.empty:
            _hist_mtd = hist
        const = _constant_means_excluding_zeros(_hist_mtd, flow_cols, tail_n=None)
        return lambda d: dict(const)

    means = _weekday_weighted_means(hist, flow_cols=flow_cols)
    return lambda d: {c: float(means.get((int(d.weekday()), c), 0.0)) for c in flow_cols}


def _roll_inventory(prev_close: float, flows: dict[str, float], flow_cols: list[str]) -> tuple[float, float]:
    opening = float(prev_close)
    inflow = sum(float(flows.get(c, 0.0) or 0.0) for c in INFLOW_COLS if c in flow_cols)
    outflow = sum(float(flows.get(c, 0.0) or 0.0) for c in OUTFLOW_COLS if c in flow_cols)
    net = sum(float(flows.get(c, 0.0) or 0.0) for c in NET_COLS if c in flow_cols)
    return opening, opening + inflow - outflow + net


def _forecast_dates(
    last_date: pd.Timestamp, forecast_end: pd.Timestamp | None, default_days: int
) -> pd.DatetimeIndex:
    start = last_date + timedelta(days=1)
    if forecast_end is not None:
        return pd.date_range(start=start, end=forecast_end, freq="D") if start <= forecast_end else pd.DatetimeIndex([])
    return pd.date_range(start=start, periods=int(default_days), freq="D")


def _last_close_inv(group: pd.DataFrame) -> float:
    if "Close Inv" not in group.columns:
        return 0.0
    last_rows = group[group["Date"] == group["Date"].max()]
    val = last_rows["Close Inv"].iloc[-1] if not last_rows.empty else 0.0
    return float(val) if pd.notna(val) else 0.0


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _aggregate_daily_details(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Aggregate intra-day duplicate rows to one row per date/location/product."""
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
    if COL_AVAILABLE in df.columns:
        agg_map[COL_AVAILABLE] = "last"
    if COL_INTRANSIT in df.columns:
        agg_map[COL_INTRANSIT] = "last"
    for c in _available_flow_cols(df):
        agg_map[c] = "sum"
    for base_col, fact_col in {**FACT_COL_MAP_RAW, **FACT_COL_MAP}.items():
        if fact_col not in df.columns:
            continue
        base_s = str(base_col)
        if base_s in {"Opening Inv", "Open Inv"}:
            agg_map[fact_col] = "first"
        elif base_s in {"Close Inv", COL_AVAILABLE, COL_INTRANSIT}:
            agg_map[fact_col] = "last"
        else:
            agg_map[fact_col] = "sum"
    if "updated" in df.columns:
        agg_map["updated"] = "max"
    if "Batch" in df.columns:
        agg_map["Batch"] = "last"
    if COL_BATCH_BREAKDOWN in df.columns:
        agg_map[COL_BATCH_BREAKDOWN] = "last"
    if "Notes" in df.columns:
        agg_map["Notes"] = "last"
    if "SOURCE_TYPE" in df.columns:
        agg_map["SOURCE_TYPE"] = "first"
    for sub in [COL_TULSA, COL_EL_DORADO, COL_OTHER, COL_ARGENTINE, COL_FROM_327_RECEIPT]:
        if sub in df.columns:
            agg_map[sub] = "sum"
    if COL_STORAGE in df.columns:
        agg_map[COL_STORAGE] = "last"
    if COL_VESSEL in df.columns:
        agg_map[COL_VESSEL] = "last"
    if COL_VESSEL_VOLUME in df.columns:
        agg_map[COL_VESSEL_VOLUME] = "last"
    if "FILE_LOCATION" in df.columns:
        agg_map["FILE_LOCATION"] = "last"

    return df.groupby(group_cols, as_index=False).agg(agg_map)


def _ensure_lineage_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "updated" not in out.columns:
        out["updated"] = 0
    else:
        out["updated"] = pd.to_numeric(out["updated"]).fillna(0).astype(int)
    return out


def _fill_missing_internal_dates(
    daily: pd.DataFrame,
    *,
    id_col: str,
    start_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Ensure a continuous day-by-day grid within the observed date range."""
    if daily is None or daily.empty or "Date" not in daily.columns:
        return daily

    out = daily.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    group_cols = [id_col] + (["Product"] if "Product" in out.columns else [])
    flow_cols = _available_flow_cols(out)
    zero_fill_cols = [
        "Open Inv", "Close Inv", COL_AVAILABLE, COL_INTRANSIT, *flow_cols,
        *[c for c in out.columns if str(c).endswith(" Fact")],
    ]
    zero_fill_cols = [c for c in zero_fill_cols if c in out.columns]

    filled_parts: list[pd.DataFrame] = []
    for keys, g in out.groupby(group_cols, dropna=False, sort=False):
        g = g.sort_values("Date", kind="mergesort").copy()
        if g.empty:
            filled_parts.append(g)
            continue
        min_d = pd.to_datetime(g["Date"].min(), errors="coerce")
        max_d = pd.to_datetime(g["Date"].max(), errors="coerce")
        if pd.isna(min_d) or pd.isna(max_d):
            filled_parts.append(g)
            continue

        start_d = min_d
        if start_date is not None:
            sd = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(sd) and sd < start_d:
                start_d = sd

        g2 = g.set_index("Date").reindex(pd.date_range(start=start_d, end=max_d, freq="D"))

        if start_date is not None and "Open Inv" in g2.columns:
            try:
                anchor_open = float(pd.to_numeric(g.loc[g["Date"] == min_d, "Open Inv"], errors="coerce").fillna(0.0).iloc[0])
            except Exception:
                anchor_open = 0.0
            fill_cols = ["Open Inv", "Close Inv"] if "Close Inv" in g2.columns else ["Open Inv"]
            g2.loc[g2.index < min_d, fill_cols] = anchor_open

        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, val in zip(group_cols, keys):
            g2[col] = val

        if "updated" in g2.columns:
            g2["updated"] = pd.to_numeric(g2["updated"], errors="coerce").fillna(0).astype(int)
        if "SOURCE_TYPE" in g2.columns:
            g2["SOURCE_TYPE"] = g2["SOURCE_TYPE"].fillna("")
        if "Batch" in g2.columns:
            g2["Batch"] = g2["Batch"].fillna("")
        if COL_BATCH_BREAKDOWN in g2.columns:
            g2[COL_BATCH_BREAKDOWN] = g2[COL_BATCH_BREAKDOWN].fillna("")
        if "Notes" in g2.columns:
            g2["Notes"] = g2["Notes"].fillna("")
        if COL_VESSEL in g2.columns:
            g2[COL_VESSEL] = g2[COL_VESSEL].fillna("")
        if "FILE_LOCATION" in g2.columns:
            g2["FILE_LOCATION"] = g2["FILE_LOCATION"].apply(
                lambda v: [] if (v is None or (isinstance(v, float) and pd.isna(v))) else v
            )
        for c in zero_fill_cols:
            g2[c] = pd.to_numeric(g2[c], errors="coerce").fillna(0.0)

        filled_parts.append(g2.reset_index().rename(columns={"index": "Date"}))

    return pd.concat(filled_parts, ignore_index=True) if filled_parts else out


def _extend_with_30d_forecast(
    df: pd.DataFrame,
    *,
    id_col: str,
    region: str | None,
    location: str | None,
    history_start: pd.Timestamp | None = None,
    forecast_end: pd.Timestamp | None = None,
    default_days: int = 30,
) -> pd.DataFrame:
    """Append app-generated forecast rows to historical data."""
    if df.empty:
        return df

    daily = _aggregate_daily_details(df, id_col=id_col)
    if daily.empty:
        return daily

    daily = _ensure_lineage_cols(daily).sort_values("Date")
    daily = _fill_missing_internal_dates(daily, id_col=id_col, start_date=history_start).sort_values("Date")
    flow_cols = _available_flow_cols(daily)

    today = pd.Timestamp.today().normalize()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
    hist_daily = daily[daily["Date"] < today].copy()

    forecast_flow_cols = [c for c in [COL_RACK_LIFTINGS_RAW] if c in flow_cols]
    if forecast_end is not None:
        forecast_end = pd.Timestamp(forecast_end)

    forecast_method = get_rack_lifting_forecast_method(region=str(region or "Unknown"), location=location)
    forecast_rows: list[dict] = []

    for (id_val, product), group in hist_daily.groupby([id_col, "Product"], dropna=False):
        group = group.sort_values("Date")
        last_date = pd.Timestamp(group["Date"].max())
        estimate = _make_forecast_flow_estimator(group, flow_cols=forecast_flow_cols, method=forecast_method)
        prev_close = _last_close_inv(group)

        for d in _forecast_dates(last_date, forecast_end, default_days):
            flows = {c: 0.0 for c in flow_cols}
            if forecast_flow_cols:
                flows.update(estimate(d))
            opening, closing = _roll_inventory(prev_close, flows, flow_cols)
            prev_close = closing
            forecast_rows.append({
                "Date": d, id_col: id_val, "Product": product,
                "SOURCE_TYPE": "forecast", "updated": 0, "Batch": "", "Notes": "",
                "FILE_LOCATION": [], "Open Inv": opening, "Close Inv": closing, **flows,
            })

    if not forecast_rows:
        return hist_daily

    combined = pd.concat([hist_daily, pd.DataFrame(forecast_rows)], ignore_index=True)
    for c in ["Open Inv", "Close Inv"] + flow_cols:
        if c in combined.columns:
            combined[c] = pd.to_numeric(combined[c]).fillna(0.0)
    return combined


def build_details_view(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Rename raw columns for display, filter to relevant columns, and sort by date."""
    df = df.sort_values("Date").rename(columns=DETAILS_RENAME_MAP)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    # Keep: display columns + fact columns + lineage/tracking columns
    display_set = {"Date", id_col} | set(DETAILS_COLS)
    fact_set = {c for c in df.columns if str(c).endswith(" Fact")}
    keep = [c for c in df.columns if c in display_set or c in fact_set or c in _TRACKING_COLS]
    df = df[keep].copy()

    no_round = {"Date", id_col, "Product", "Notes", "Batch", COL_BATCH_BREAKDOWN, COL_VESSEL, "updated", "FILE_LOCATION", "SOURCE_TYPE"}
    for c in df.columns:
        if c not in no_round and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(2)
    return df


def _build_editor_df(df_display: pd.DataFrame) -> pd.DataFrame:
    """Add tracking/action columns with safe defaults for the editor."""
    out = df_display.copy()
    if "FILE_LOCATION" not in out.columns:
        out["FILE_LOCATION"] = [[] for _ in range(len(out))]
    if COL_VIEW_FILE not in out.columns:
        out[COL_VIEW_FILE] = False
    else:
        out[COL_VIEW_FILE] = out[COL_VIEW_FILE].fillna(False).astype(bool)
    if COL_STORAGE not in out.columns:
        out[COL_STORAGE] = np.nan
    if COL_VESSEL_VOLUME not in out.columns:
        out[COL_VESSEL_VOLUME] = np.nan
    if COL_VESSEL not in out.columns:
        out[COL_VESSEL] = ""
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Column config and styling
# ---------------------------------------------------------------------------

def _locked_cols(id_col: str, cols: list[str]) -> list[str]:
    """Return column names that must be disabled (read-only) in the editor."""
    wanted = [c.format(id_col=id_col) for c in LOCKED_BASE_COLS]
    return [c for c in wanted if c in cols]


def _column_config(df: pd.DataFrame, cols: list[str], id_col: str) -> dict:
    """Build Streamlit column config for the details editor."""
    locked = set(_locked_cols(id_col, cols))
    locked.update({c for c in cols if str(c).endswith(" Fact")})
    locked.add("SOURCE_TYPE")
    NUM_FMT = "accounting"

    cfg: dict = {
        "Date": st.column_config.DateColumn("Date", disabled=True, format="YYYY-MM-DD"),
        id_col: st.column_config.TextColumn(id_col, disabled=True),
        "Product": st.column_config.TextColumn("Product", disabled=True),
        "updated": st.column_config.CheckboxColumn("updated", default=False),
        "Batch": st.column_config.TextColumn("Batch"),
        COL_BATCH_BREAKDOWN: st.column_config.TextColumn(COL_BATCH_BREAKDOWN),
        "Notes": st.column_config.TextColumn("Notes"),
        COL_STORAGE: st.column_config.NumberColumn(
            COL_STORAGE, disabled=False, format=NUM_FMT,
            help="Volume held in storage. Used to compute Accounting Inventory (Close Inv − Storage).",
        ),
        COL_VESSEL: st.column_config.TextColumn(
            COL_VESSEL,
            help="Vessel name or identifier for this row.",
        ),
        COL_VESSEL_VOLUME: st.column_config.NumberColumn(
            COL_VESSEL_VOLUME, disabled=False, format=NUM_FMT,
            help="Volume associated with the vessel (BBL).",
        ),
        "SOURCE_TYPE": st.column_config.TextColumn("SOURCE_TYPE", disabled=True),
        COL_VIEW_FILE: st.column_config.CheckboxColumn(
            COL_VIEW_FILE, default=False, disabled=(DATA_SOURCE != "snowflake"),
            help="Check to open a popup with downloadable system files for this row.",
        ),
    }

    # Auto-configure any remaining column not already defined
    for c in cols:
        if c in cfg:
            continue
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            cfg[c] = st.column_config.NumberColumn(c, disabled=(c in locked), format=NUM_FMT)
        elif c in locked:
            cfg[c] = st.column_config.TextColumn(c, disabled=True)

    return {k: v for k, v in cfg.items() if k in cols}


def _style_source_cells(
    df: pd.DataFrame,
    *,
    safefill: float | None = None,
    bottom: float | None = None,
) -> "pd.io.formats.style.Styler":
    """Apply row/cell background colors based on date context and inventory thresholds."""
    today = date.today()
    yesterday = today - timedelta(days=1)
    cols = list(df.columns)
    fact_cols = {c for c in cols if str(c).endswith(" Fact")}

    def _to_date(v):
        if v is None:
            return None
        if isinstance(v, date):
            return v
        try:
            return pd.Timestamp(v).date()
        except Exception:
            return None

    def _close_inv_matches(row: pd.Series) -> bool:
        base_val = _to_float(row.get("Close Inv", 0.0))
        fact_val = row.get("Close Inv Fact") if "Close Inv Fact" in row.index else None
        if fact_val is None:
            return True
        return abs(base_val - _to_float(fact_val)) <= SYSTEM_DISCREPANCY_THRESHOLD_BBL

    def _row_style(row: pd.Series) -> list[str]:
        row_date = _to_date(row.get("Date") if "Date" in row.index else None)
        yesterday_bg = (MATCH_BG if _close_inv_matches(row) else MISMATCH_BG) if row_date == yesterday else ""

        styles = []
        for c in cols:
            if c in fact_cols:
                styles.append(f"background-color: {FACT_BG};")
            elif row_date == today:
                styles.append(f"background-color: {TODAY_BG};" if c == "Date" else "")
            elif row_date == yesterday:
                styles.append(f"background-color: {yesterday_bg};" if c in YESTERDAY_HIGHLIGHT_COLS else "")
            elif row_date is not None and row_date < yesterday:
                styles.append(f"background-color: {OLDER_DAY_BG};" if c in OLDER_DAY_HIGHLIGHT_COLS else "")
            else:
                styles.append("")

        # Threshold color for future rows
        if row_date is not None and row_date > today:
            for target_col in {"Close Inv", "Total Closing Inv", "Loadable"}:
                if target_col in cols:
                    raw_val = row.get(target_col) if target_col in row.index else None
                    if raw_val is not None and not (isinstance(raw_val, float) and pd.isna(raw_val)):
                        val = _to_float(raw_val)
                        idx = cols.index(target_col)
                        if safefill is not None and val > safefill:
                            styles[idx] = f"background-color: {CLOSE_INV_ABOVE_SAFEFILL_BG};"
                        elif bottom is not None and val < bottom:
                            styles[idx] = f"background-color: {CLOSE_INV_BELOW_BOTTOM_BG};"
        return styles

    return df.style.apply(_row_style, axis=1).hide(axis="index")


# ---------------------------------------------------------------------------
# Column order building
# ---------------------------------------------------------------------------

def _ensure_cols_after(
    column_order: list[str],
    *,
    required: list[str],
    after: str,
    before: str | None = None,
) -> list[str]:
    """Re-insert required columns immediately after the `after` column."""
    out = [c for c in column_order if c not in required]
    if after in out:
        pos = out.index(after) + 1
    elif before is not None and before in out:
        pos = out.index(before)
    else:
        pos = len(out)
    for i, c in enumerate(required):
        out.insert(pos + i, c)
    return out


def _insert_fact_columns(column_order: list[str], *, df_cols: list[str], show_fact: bool) -> list[str]:
    """Insert '<col> Fact' columns immediately after their base column."""
    if not show_fact:
        return column_order
    out, seen, df_set = [], set(), set(df_cols)
    for c in column_order:
        if c not in seen:
            out.append(c)
            seen.add(c)
        fact = FACT_COL_MAP.get(c)
        if fact and fact in df_set and fact not in seen:
            out.append(fact)
            seen.add(fact)
    return out


def _build_column_order(df: pd.DataFrame, *, visible: list[str], show_fact: bool) -> list[str]:
    """Build the display column order for the details editor.

    Respects the admin-configured visible list exactly. The only
    post-processing is:
      1. Date is always first.
      2. Fact columns (when toggled on) are inserted right after their
         paired base column.
    """
    df_cols = set(df.columns)

    # Use the admin-configured order exactly, filtering to columns that exist
    order = [c for c in visible if c in df_cols and c != "Date"]
    if "Date" in df_cols:
        order = ["Date"] + order

    # Fact columns after their base columns (only when Terminal Feed is on)
    if show_fact:
        order = _insert_fact_columns(order, df_cols=list(df_cols), show_fact=True)

    return order


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def _threshold_values(
    *, region: str, location: str | None, product: str | None = None,
) -> tuple[float | None, float | None, str | None]:
    """Return (bottom, safefill, note) thresholds for the given scope."""
    ovr = get_threshold_overrides(region=region, location=location, product=product)
    bottom = ovr.get("BOTTOM")
    safefill = ovr.get("SAFEFILL")
    note = ovr.get("NOTE")
    b = float(bottom) if bottom is not None and not pd.isna(bottom) else None
    s = float(safefill) if safefill is not None and not pd.isna(safefill) else None
    n = None
    if note is not None and not (isinstance(note, float) and pd.isna(note)):
        n = str(note).strip() or None
    return b, s, n


# ---------------------------------------------------------------------------
# Save dialogs (details-tab-specific)
# ---------------------------------------------------------------------------

def _confirm_save_dialog(*, payload: dict) -> None:
    """
    Start save immediately (no popup).
    Function name kept for compatibility.
    """
    st.session_state["details_save_stage"] = "pre_save"
    st.session_state["details_save_payload"] = payload
    st.session_state["details_save_overlay"] = {
        "on": True,
        "df_key": payload.get("df_key"),
    }


# @st.dialog("Save Result")
def _save_result_dialog(*, result: dict) -> None:
    # Hide the dialog close button to force the auto-countdown
    st.markdown(
        """<style>
        [data-testid="stDialog"] button[aria-label="Close"],
        [data-testid="stDialog"] button[title="Close"] { display: none !important; }
        </style>""",
        unsafe_allow_html=True,
    )
    ok = bool(result.get("ok"))
    n = int(result.get("n") or 0)
    err = str(result.get("error") or "")

    if ok:
        # st.toast(f"Saved successfully ({n} rows).")
        time.sleep(0.2)
        st.session_state["details_save_stage"] = None
        st.session_state["details_save_result"] = None
        st.session_state["details_save_overlay_removal_pending"] = result.get("df_key")
        st.rerun()
    else:
        st.error("Save failed! Please try again, or reach out to an administrator.")
        if err:
            st.code(err)


# ---------------------------------------------------------------------------
# Main display functions
# ---------------------------------------------------------------------------

def display_location_details(
    df_filtered: pd.DataFrame,
    active_region: str,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    selected_loc: str | None,
) -> None:
    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return
    if not selected_loc:
        st.info("Select a Location/System above and press Submit.")
        return

    df_loc = (
        df_filtered[df_filtered["Location"] == str(selected_loc)]
        if "Location" in df_filtered.columns
        else pd.DataFrame()
    )
    if df_loc.empty:
        st.info("No data available for the selected location + date range.")
        return

    products = (
        sorted(df_loc["Product"].dropna().astype(str).unique().tolist())
        if "Product" in df_loc.columns
        else []
    )
    if not products:
        st.info("No products available for the selected location.")
        return

    # These are location-level values — compute once outside the product loop
    forecast_method = get_rack_lifting_forecast_method(
        region=str(active_region or "Unknown"), location=selected_loc
    )
    visible = get_visible_columns(region=active_region, location=str(selected_loc))

    # Terminal Feed toggle + location label + Reset / Formulas icons
    c_toggle, c_loc, c_reset_loc, c_formulas_loc = st.columns([4.5, 4.5, 0.5, 0.5])
    with c_toggle:
        show_fact = st.toggle(
            "Show Terminal Feed",
            value=False,
            key=f"details_show_fact|{active_region}|{selected_loc}",
            help="Show upstream system values next to the editable columns.",
        )
    with c_loc:
        st.markdown(
            f"<h1 style='color: green; font-weight: 700; font-size: 1.2rem'>{selected_loc}</h1>",
            unsafe_allow_html=True,
        )
    with c_reset_loc:
        st.markdown('<div class="transparent-icon"></div>', unsafe_allow_html=True)
        reset_clicked = st.button(
            "↺", key=f"reset_{active_region}_{selected_loc}",
            help="Discard unsaved changes and reload data from the database.",
        )
    with c_formulas_loc:
        st.markdown('<div class="transparent-icon"></div>', unsafe_allow_html=True)
        with st.popover("ℹ️"):
            st.markdown(
                "**Calculated Column Formulas**\n\n"
                "| Column | Formula |\n"
                "|---|---|\n"
                "| **Opening Inv** | Previous day's Close Inv |\n"
                "| **Close Inv** | Opening Inv + Receipts + Pipeline In + Production "
                "− Deliveries − Rack/Lifting − Pipeline Out "
                "+ Adjustments + Gain/Loss + Transfers "
                "+ Tulsa + El Dorado + Other − Argentine + From 327 Receipt |\n"
                "| **Total Closing Inv** | Available + Intransit |\n"
                "| **Available Space** | SafeFill − Close Inv |\n"
                "| **Loadable** | Close Inv − Bottom |\n"
                "| **Total Inventory** | Close Inv + Bottom |\n"
                "| **Accounting Inv** | Close Inv − Storage |\n"
                "| **7 Day Avg** | 7-day rolling average of Rack/Lifting |\n"
                "| **MTD Avg** | Month-to-date average of Rack/Lifting |\n"
                "| **Calculated Receipt** | Today's Available − Yesterday's Available + Today's Rack/Lifting |",
            )

    # Handle reset at location level (clears all product tabs)
    if reset_clicked:
        _load_inventory_data_filtered_cached.clear()
        details_cache_key = f"df_details|{active_region}"
        fresh_filters = {
            "active_region": active_region,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "selected_loc": selected_loc,
            "loc_col": "Location",
        }
        st.session_state[details_cache_key] = load_filtered_inventory_data(fresh_filters)
        loc_prefix = f"{active_region}_{selected_loc}_"
        for k in list(st.session_state.keys()):
            if str(k).startswith(loc_prefix):
                st.session_state.pop(k, None)

        st.session_state.pop("details_save_stage", None)
        st.session_state.pop("details_save_payload", None)
        st.session_state.pop("details_save_result", None)        
        st.rerun()

    for i, tab in enumerate(st.tabs(products)):
        prod_name = products[i]
        with tab:
            # ── State keys ──────────────────────────────────────────────────
            # show_fact is intentionally excluded from state_key so the editor
            # widget key stays the same when the Terminal Feed toggle changes.
            # Toggling only updates column_order, preserving unsaved edits.
            state_key = (
                f"{active_region}_{selected_loc}_{prod_name}"
                f"|{pd.Timestamp(start_ts).date()}|{pd.Timestamp(end_ts).date()}"
                f"|m={forecast_method}|edit"
            )
            df_key = f"{state_key}__df"           # canonical DataFrame in session state
            ver_key = f"{state_key}__ver"          # version counter; increment forces editor remount
            ver = int(st.session_state.get(ver_key, 0))
            base_key = f"{state_key}__base_v{ver}"  # stable snapshot passed to editor this version
            widget_key = f"{state_key}__editor_v{ver}"

            # ── View File dialog ─────────────────────────────────────────────
            vf_payload = st.session_state.get("details_view_file_payload")
            if isinstance(vf_payload, dict) and vf_payload.get("df_key") == df_key:
                _view_files_dialog(
                    file_locations=vf_payload.get("file_locations"),
                    context={
                        "date": vf_payload.get("date"),
                        "location": vf_payload.get("location"),
                        "product": vf_payload.get("product"),
                    },
                )
                st.session_state["details_view_file_payload"] = None

            # ── Save overlay ─────────────────────────────────────────────────
            overlay = st.session_state.get("details_save_overlay") or {}
            if overlay.get("on") and overlay.get("df_key") == df_key:
                _render_blocking_overlay(True, message="Saving…")
            if (
                st.session_state.get("details_save_stage") is None and
                st.session_state.get("details_save_overlay_removal_pending") == df_key and
                overlay.get("on")
            ):
                st.session_state["details_save_overlay"] = {"on": False, "df_key": None}
                st.session_state["details_save_overlay_removal_pending"] = None
                st.rerun()

            # ── Thresholds ───────────────────────────────────────────────────
            bottom, safefill, note = _threshold_values(
                region=active_region, location=str(selected_loc), product=str(prod_name),
            )

            # ── Header row: threshold cards + Enable Save + Save button ──
            c_sf, c_bt, c_note, c_info, c_enable, c_save = st.columns([1.75, 1.75, 2.7, 1.2, 1.3, 2.3])
            _render_threshold_cards(
                bottom=bottom, safefill=safefill, note=note,
                c_safefill=c_sf, c_bottom=c_bt, c_note=c_note, c_info=c_info,
                display_forecast_method=forecast_method,
            )

            enable_key = f"details_enable_save|{active_region}|{selected_loc}|{prod_name}"
            enable_ver_key = f"{enable_key}__ver"
            enable_widget_key = f"{enable_key}__v{int(st.session_state.get(enable_ver_key, 0))}"

            with c_enable:
                enable_save = st.toggle(
                    "Enable Save", value=False, key=enable_widget_key,
                    disabled=(get_user_role() == ROLE_DISPLAY),
                    help="Toggle ON before saving to ensure the last edited cell commits.",
                )
            with c_save:
                save_clicked = logged_button(
                    f"Save {prod_name} Data",
                    key=f"save_{active_region}_{selected_loc}_{prod_name}",
                    type="primary",
                    disabled=not bool(enable_save),
                    event="details_save_clicked",
                    metadata={"region": active_region, "scope": "location",
                              "location": selected_loc, "product": prod_name},
                )
            # ── Load and prepare data (only on first render for this state_key) ──
            df_prod = df_loc[df_loc["Product"].astype(str) == str(prod_name)]

            if df_key not in st.session_state:
                df_all = _extend_with_30d_forecast(
                    df_prod, id_col="Location", region=active_region,
                    location=str(selected_loc), history_start=start_ts, forecast_end=end_ts,
                )
                df_display = build_details_view(df_all, id_col="Location")
                df_display = df_display[
                    df_display["Date"] >= pd.Timestamp(start_ts).normalize().date()
                ].reset_index(drop=True)

                # Add sub-breakdown columns if configured but absent from source data
                for sub in [COL_TULSA, COL_EL_DORADO, COL_OTHER, COL_ARGENTINE, COL_FROM_327_RECEIPT]:
                    if sub in visible and sub not in df_display.columns:
                        df_display[sub] = 0.0

                editor_df = _build_editor_df(df_display)
                st.session_state[df_key] = _recalculate_inventory_metrics(
                    editor_df, id_col="Location", safefill=safefill, bottom=bottom, df_hist=df_prod,
                )

            # ── Stable snapshot for this editor version ──────────────────────
            # The snapshot is created once per version and passed as the editor's
            # baseline on every render. Changing column_order (e.g., show_fact)
            # does NOT create a new snapshot — the editor preserves its edits.
            if base_key not in st.session_state:
                st.session_state.pop(f"{state_key}__base_v{ver - 1}", None)  # clean up previous
                st.session_state[base_key] = st.session_state[df_key].copy().reset_index(drop=True)

            base_df = st.session_state[base_key]

            # ── Column order: changes with show_fact but NOT the editor key ──
            column_order = _build_column_order(base_df, visible=visible, show_fact=show_fact)
            column_config = _column_config(base_df, column_order, "Location")

            # ── Render editor ────────────────────────────────────────────────
            styled = _style_source_cells(base_df, safefill=safefill, bottom=bottom)
            edited = dynamic_input_data_editor(
                styled,
                num_rows="fixed",
                width="stretch",
                height=DETAILS_EDITOR_HEIGHT_PX,
                hide_index=True,
                column_order=column_order,
                key=widget_key,
                column_config=column_config,
            )

            _pre_sync_text: dict = {}
            for _snap_col in [COL_NOTES, COL_BATCH, COL_BATCH_BREAKDOWN, COL_VESSEL]:
                if base_key in st.session_state and _snap_col in st.session_state[base_key].columns:
                    _pre_sync_text[_snap_col] = (
                        st.session_state[base_key][_snap_col].fillna("").values.copy()
                    )
            _TEXT_COLS = [COL_NOTES, COL_BATCH, COL_BATCH_BREAKDOWN, COL_VESSEL]
            if edited is not None and not edited.empty:
                for col in _TEXT_COLS:
                    if col in edited.columns:
                        vals = edited[col].fillna("").values

                        if df_key in st.session_state and col in st.session_state[df_key].columns:
                            st.session_state[df_key][col] = vals

                        if base_key in st.session_state and col in st.session_state[base_key].columns:
                            st.session_state[base_key][col] = vals

            # ── Recalculate derived columns from the current editor state ────
            df_hist_for_avg = _overlay_rack_edits(df_prod, edited)
            recomputed = _recalculate_inventory_metrics(
                edited, id_col="Location", safefill=safefill, bottom=bottom,
                df_hist=df_hist_for_avg,
            ).reset_index(drop=True)

            # ── Merge recomputed values back into canonical df ───────────────
            canonical = st.session_state[df_key].copy().reset_index(drop=True)
            if canonical.shape[0] == recomputed.shape[0]:
                for c in recomputed.columns:
                    canonical[c] = recomputed[c].values
                st.session_state[df_key] = canonical

            if edited is not None and not edited.empty and base_key in st.session_state:
                for _col in _TEXT_COLS:
                    if _col in edited.columns and _col in st.session_state[base_key].columns:
                        st.session_state[base_key][_col] = edited[_col].values 
            _text_cols_changed = edited is not None and not edited.empty and any(
                col in edited.columns
                and col in _pre_sync_text
                and not np.array_equal(
                    _pre_sync_text[col], edited[col].fillna("").values
                )
                for col in [COL_NOTES, COL_BATCH, COL_BATCH_BREAKDOWN, COL_VESSEL]
            )
            if _text_cols_changed:
                st.rerun()            

            # ── Handle "View File" checkbox action ───────────────────────────
            if COL_VIEW_FILE in recomputed.columns:
                view_mask = recomputed[COL_VIEW_FILE].fillna(False).astype(bool)
                if view_mask.any():
                    idx = int(view_mask[view_mask].index[0])
                    file_locs = recomputed.at[idx, "FILE_LOCATION"] if "FILE_LOCATION" in recomputed.columns else []
                    st.session_state[df_key].at[idx, COL_VIEW_FILE] = False
                    st.session_state["details_view_file_payload"] = {
                        "df_key": df_key, "row": idx,
                        "date": str(recomputed.at[idx, "Date"]) if "Date" in recomputed.columns else None,
                        "location": str(selected_loc),
                        "product": str(prod_name),
                        "file_locations": file_locs,
                    }
                    st.session_state[ver_key] = ver + 1
                    st.rerun()

            # ── Force a re-render if calculated columns changed ──────────────
            if _needs_inventory_rerun(edited, recomputed):
                st.session_state[ver_key] = ver + 1
                st.rerun()

            # ── Save flow ────────────────────────────────────────────────────
            if save_clicked:
                log_audit(
                    event="details_save_dialog_opened",
                    metadata={"region": active_region, "scope": "location",
                              "location": selected_loc, "product": prod_name},
                )
                _confirm_save_dialog(payload={
                    "df_key": df_key,
                    "region": active_region,
                    "location": selected_loc,
                    "system": None,
                    "product": prod_name,
                    "scope_label": f"{selected_loc} / {prod_name}",
                })

            # Execute save after confirm dialog triggers a rerun with overlay visible
            payload = st.session_state.get("details_save_payload") or {}
            if st.session_state.get("details_save_stage") == "pre_save" and payload.get("df_key") == df_key:
                try:
                    n = persist_details_rows(
                        st.session_state[df_key],
                        region=str(payload.get("region") or active_region),
                        location=payload.get("location"),
                        system=payload.get("system"),
                        product=payload.get("product"),
                    )
                    log_audit(
                        event="details_save_success",
                        metadata={"region": str(payload.get("region") or active_region),
                                  "location": payload.get("location"),
                                  "product": payload.get("product"),
                                  "rows_saved": int(n)},
                    )
                    st.session_state["details_save_result"] = {"ok": True, "n": int(n), "df_key": df_key}
                except Exception as e:
                    log_audit(
                        event="details_save_failed",
                        metadata={"region": str(payload.get("region") or active_region),
                                  "location": payload.get("location"),
                                  "product": payload.get("product"),
                                  "error": str(e)},
                    )
                    st.session_state["details_save_result"] = {"ok": False, "error": str(e), "df_key": df_key}

                st.session_state[enable_ver_key] = int(st.session_state.get(enable_ver_key, 0)) + 1
                st.session_state.pop(enable_widget_key, None)
                st.session_state["details_save_stage"] = "result"
                st.rerun()

            # Show save result dialog
            result = st.session_state.get("details_save_result")
            if (
                st.session_state.get("details_save_stage") == "result" and
                isinstance(result, dict) and
                result.get("df_key") == df_key
            ):
                _save_result_dialog(result=result)


def display_details_tab(
    df_filtered: pd.DataFrame,
    active_region: str,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    selected_loc: str | None = None,
) -> None:
    display_location_details(
        df_filtered, active_region,
        start_ts=start_ts, end_ts=end_ts, selected_loc=selected_loc,
    )


def render_details_filters(*, regions: list[str], active_region: str | None) -> dict:
    from data_loader import load_region_filter_metadata
    from admin_config import get_default_date_window

    def _un_collapse_expandables() -> None:
        st.session_state["collapse_expandables"] = False

    if not active_region:
        return {
            "active_region": None,
            "start_ts": pd.Timestamp.today().normalize(),
            "end_ts": pd.Timestamp.today().normalize(),
            "selected_loc": None,
            "loc_col": "Location",
            "locations": [],
        }

    loc_col = "Location"
    region_norm = str(active_region).strip().lower()
    filter_label = "System" if region_norm == "midcon" else "Location"

    meta = load_region_filter_metadata(region=active_region, loc_col=loc_col)
    locations = meta.get("locations", [])

    c1, c2, c3 = st.columns([2.2, 2.8, 1.2])
    with c1:
        if not locations:
            st.warning("No locations available")
            selected_loc = None
        else:
            key_loc = f"details_selected_loc|{active_region}"
            _persist_key = f"_details_loc_persist|{active_region}"
            current = st.session_state.get(key_loc) or st.session_state.get(_persist_key)
            index = locations.index(current) if current in locations else 0
            selected_loc = st.selectbox(
                filter_label, options=locations, index=index,
                key=key_loc, on_change=_un_collapse_expandables,
            )
            st.session_state[_persist_key] = selected_loc

    today = pd.Timestamp.today().date()
    start_off, end_off = get_default_date_window(
        region=active_region, location=(str(selected_loc) if selected_loc else None)
    )
    default_start = today + timedelta(days=int(start_off))
    default_end = today + timedelta(days=int(end_off))

    with c2:
        df_min = meta.get("min_date", pd.NaT)
        df_max = meta.get("max_date", pd.NaT)
        df_min_d = pd.to_datetime(df_min, errors="coerce").date() if pd.notna(df_min) else default_start
        df_max_d = pd.to_datetime(df_max, errors="coerce").date() if pd.notna(df_max) else default_end
        min_value = min(df_min_d, default_start)
        max_value = max(df_max_d, default_end)

        key_dates = f"details_date|{active_region}|{str(selected_loc) if selected_loc else 'all'}"
        date_range = st.date_input(
            "Date Range",
            value=(default_start, default_end),
            min_value=min_value,
            max_value=max_value,
            key=key_dates,
            on_change=_un_collapse_expandables,
        )

        if isinstance(date_range, (list, tuple)):
            start_date, end_date = (date_range[0], date_range[-1]) if len(date_range) >= 2 else (date_range[0], date_range[0])
        else:
            start_date = end_date = date_range

    with c3:
        st.markdown('<div class="btn-spacer"></div>', unsafe_allow_html=True)
        submitted = logged_button(
            "Submit", type="primary",
            event="details_filters_submit",
            metadata={"region": active_region, "selected_loc": selected_loc},
        )

    if bool(submitted):
        # Reset Terminal Feed toggles when filters change
        for k in list(st.session_state.keys()):
            if str(k).startswith("details_show_fact|"):
                st.session_state[k] = False

    return {
        "active_region": active_region,
        "start_ts": pd.to_datetime(start_date),
        "end_ts": pd.to_datetime(end_date),
        "selected_loc": selected_loc,
        "loc_col": loc_col,
        "locations": locations,
        "submitted": bool(submitted),
    }