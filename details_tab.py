import streamlit as st
import pandas as pd
import numpy as np
import html
import time
from datetime import date, timedelta

from admin_config import get_visible_columns, get_threshold_overrides, get_rack_lifting_forecast_method
from utils import dynamic_input_data_editor
from data_loader import persist_details_rows
from data_loader import generate_snowflake_signed_urls
from app_logging import logged_button, log_audit, log_error
from config import (
    COL_ADJUSTMENTS,
    COL_ADJUSTMENTS_FACT,
    COL_AVAILABLE,
    COL_AVAILABLE_FACT,
    COL_BATCH_IN,
    COL_BATCH_IN_RAW,
    COL_BATCH_IN_FACT_RAW,
    COL_BATCH_IN_FACT,
    COL_BATCH_OUT,
    COL_BATCH_OUT_RAW,
    COL_BATCH_OUT_FACT_RAW,
    COL_BATCH_OUT_FACT,
    COL_CLOSE_INV_RAW,
    COL_CLOSE_INV_FACT_RAW,
    COL_INTRANSIT,
    COL_INTRANSIT_FACT,
    COL_OPEN_INV_RAW,
    COL_OPEN_INV_FACT_RAW,
    COL_OPENING_INV,
    COL_OPENING_INV_FACT,
    COL_PIPELINE_IN,
    COL_PIPELINE_IN_FACT,
    COL_PIPELINE_OUT,
    COL_PIPELINE_OUT_FACT,
    COL_PRODUCT,
    COL_PRODUCTION,
    COL_PRODUCTION_FACT,
    COL_RACK_LIFTING,
    COL_RACK_LIFTINGS_RAW,
    COL_RACK_LIFTINGS_FACT_RAW,
    COL_RACK_LIFTING_FACT,
    COL_TRANSFERS,
    COL_TRANSFERS_FACT,
    COL_GAIN_LOSS,
    COL_GAIN_LOSS_FACT,
    COL_BATCH,
    COL_NOTES,
    COL_AVAILABLE_SPACE,
    COL_TOTAL_CLOSING_INV,
    COL_LOADABLE,
    # Phase-2 new UI-only display columns
    COL_TOTAL_INVENTORY,
    COL_ACCOUNTING_INV,
    COL_7DAY_AVG_RACK,
    COL_MTD_AVG_RACK,
    # Phase-2 user-editable persisted columns
    COL_STORAGE,
    COL_TULSA,
    COL_EL_DORADO,
    COL_OTHER,
    COL_ARGENTINE,
    COL_FROM_327_RECEIPT,
    DETAILS_RENAME_MAP,
    DATA_SOURCE,
)

DETAILS_RENAME = DETAILS_RENAME_MAP

DETAILS_COLS = [
    COL_PRODUCT,
    COL_OPENING_INV,
    COL_AVAILABLE,
    COL_INTRANSIT,
    COL_CLOSE_INV_RAW,
    # UI-only calculated column: Close Inv + Intransit
    COL_TOTAL_CLOSING_INV,
    # UI-only calculated column: SafeFill - Close Inv
    COL_AVAILABLE_SPACE,
    # UI-only calculated column: Close Inv - Bottoms
    COL_LOADABLE,
    # UI-only calculated column: Close Inv + Bottoms (threshold)
    COL_TOTAL_INVENTORY,
    # UI-only calculated column: Close Inv - Storage (Storage is now persisted; see sub-breakdown group below)
    COL_ACCOUNTING_INV,
    COL_BATCH_IN,
    COL_BATCH_OUT,
    COL_RACK_LIFTING,
    # UI-only: 7-day rolling average of Rack/Lifting (historical, excl. zeros)
    COL_7DAY_AVG_RACK,
    # UI-only: month-to-date average of Rack/Lifting (current calendar month, excl. zeros)
    COL_MTD_AVG_RACK,
    COL_PIPELINE_IN,
    COL_PIPELINE_OUT,
    COL_GAIN_LOSS,
    COL_TRANSFERS,
    COL_PRODUCTION,
    COL_ADJUSTMENTS,
    # Sub-breakdown columns: user-editable and persisted to DB
    COL_TULSA,
    COL_EL_DORADO,
    COL_OTHER,
    COL_ARGENTINE,
    COL_FROM_327_RECEIPT,
    COL_STORAGE,
    COL_BATCH,
    COL_NOTES,
]

# UI-only column (not persisted)
COL_VIEW_FILE = "View File"


@st.dialog("System Files")
def _view_files_dialog(*, file_locations: list[str] | None, context: dict | None = None) -> None:
    """Popup showing signed download links for system files for a given row/day."""

    ctx = context or {}
    date_label = ctx.get("date")
    loc_label = ctx.get("location")
    prod_label = ctx.get("product")

    title_bits = [b for b in [date_label, loc_label, prod_label] if b]
    if title_bits:
        st.caption(" / ".join(str(b) for b in title_bits))

    paths = file_locations or []
    paths = [str(p).strip() for p in paths if p is not None and str(p).strip()]

    if DATA_SOURCE != "snowflake":
        st.info("File downloads are only available in Snowflake mode.")
        return

    if not paths:
        st.info("No system files found for this row.")
        return

    with st.spinner("Generating signed URLs…"):
        signed = generate_snowflake_signed_urls(paths, expiry_seconds=3600)

    if not signed:
        st.warning("No downloadable links could be generated.")
        return

    st.write("Click a file to download:")

    def _short_label(name: str, *, max_len: int = 55) -> str:
        s = str(name or "")
        if len(s) <= max_len:
            return s
        # Keep start + end so users can still distinguish files.
        head = max(10, (max_len - 1) // 2)
        tail = max(10, max_len - 1 - head)
        return s[:head].rstrip() + "…" + s[-tail:].lstrip()

    for item in signed:
        p = str(item.get("path") or "")
        url = str(item.get("url") or "")
        label = p.split("/")[-1] if "/" in p else p
        if url:
            st.link_button(label=_short_label(label), url=url)
            # Full path (useful when truncated)
            st.caption(p)


FORECAST_FLOW_COLS = [
    COL_BATCH_IN_RAW,
    COL_BATCH_OUT_RAW,
    COL_RACK_LIFTINGS_RAW,
    COL_PIPELINE_IN,
    COL_PIPELINE_OUT,
    "Production",
    COL_ADJUSTMENTS,
    COL_GAIN_LOSS,
    COL_TRANSFERS,
]

INFLOW_COLS = [
    COL_BATCH_IN_RAW,
    COL_PIPELINE_IN,
    "Production",
]

OUTFLOW_COLS = [
    COL_BATCH_OUT_RAW,
    COL_RACK_LIFTINGS_RAW,
    COL_PIPELINE_OUT,
]

NET_COLS = [
    COL_ADJUSTMENTS,
    COL_GAIN_LOSS,
    COL_TRANSFERS,
]

SOURCE_BG = {
    "system": "#d9f2d9",
    # "forecast": "#d9ecff",
}

SYSTEM_DISCREPANCY_BG = "#fff2cc"

SYSTEM_DISCREPANCY_THRESHOLD_BBL = 5.0

# Row coloring by date
TODAY_BG = "#cce5ff"        # blue highlight for today's row  – today's Date cell only
MATCH_BG = "#d9f2d9"        # green – Close Inv matches Close Inv Fact – (retained but no longer used for past rows)
MISMATCH_BG = "#fff2cc"     # yellow – Close Inv does NOT match Close Inv Fact  (retained but no longer used for past rows)

# Yesterday: green/yellow applied to Date / Opening Inv / Close Inv cells only
# (green = Close Inv matches Fact, yellow = mismatch)
# Columns that receive the yesterday highlight (exact display-column names)
YESTERDAY_HIGHLIGHT_COLS = {"Date", "Opening Inv", "Close Inv"}

# Visual cue for read-only fact columns
FACT_BG = "#eeeeee"

# Closing Inv threshold coloring (cell-level overrides)
CLOSE_INV_ABOVE_SAFEFILL_BG = "#ffb3b3"   # red   – Close Inv > SafeFill (overfill risk)
CLOSE_INV_BELOW_BOTTOM_BG   = "#ffe0b2"   # orange – Close Inv < Bottom  (below minimum)

LOCKED_BASE_COLS = [
    "Date",
    "{id_col}",
    "Product",
    "Close Inv",
    COL_TOTAL_CLOSING_INV,
    COL_AVAILABLE_SPACE,
    COL_LOADABLE,
    # Phase-2 calculated columns — always read-only (Storage is excluded: it is editable)
    COL_TOTAL_INVENTORY,
    COL_ACCOUNTING_INV,
    COL_7DAY_AVG_RACK,
    COL_MTD_AVG_RACK,
    "Opening Inv",
]


def _recalculate_available_space(df: pd.DataFrame, *, safefill: float | None) -> pd.DataFrame:
    """UI-only metric: Available Space = SafeFill - Close Inv.

    If SafeFill is not configured for the scope, show NaN (blank-ish) rather than
    an arbitrary number.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    if "Close Inv" not in out.columns:
        return out

    close = _to_numeric_series(out["Close Inv"]).fillna(0.0)
    if safefill is None or (isinstance(safefill, float) and pd.isna(safefill)):
        out[COL_AVAILABLE_SPACE] = np.nan
    else:
        out[COL_AVAILABLE_SPACE] = float(safefill) - close.astype(float)

    # Keep tidy for display.
    if COL_AVAILABLE_SPACE in out.columns and pd.api.types.is_numeric_dtype(out[COL_AVAILABLE_SPACE]):
        out[COL_AVAILABLE_SPACE] = out[COL_AVAILABLE_SPACE].round(2)

    return out


def _recalculate_total_closing_inv(df: pd.DataFrame) -> pd.DataFrame:
    """UI-only metric: Total Closing Inv = Close Inv + Intransit."""
    if df is None or df.empty:
        return df

    out = df.copy()
    if "Close Inv" not in out.columns or COL_INTRANSIT not in out.columns:
        return out

    close = _to_numeric_series(out["Close Inv"]).fillna(0.0)
    intransit = _to_numeric_series(out[COL_INTRANSIT]).fillna(0.0)
    out[COL_TOTAL_CLOSING_INV] = (close.astype(float) + intransit.astype(float)).round(2)
    return out


def _recalculate_loadable(df: pd.DataFrame, *, bottom: float | None) -> pd.DataFrame:
    """UI-only metric: Loadable = Close Inv - Bottoms.

    If Bottoms is not configured for the scope, show NaN (blank-ish) rather than
    an arbitrary number.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    if "Close Inv" not in out.columns:
        return out

    close = _to_numeric_series(out["Close Inv"]).fillna(0.0)
    if bottom is None or (isinstance(bottom, float) and pd.isna(bottom)):
        out[COL_LOADABLE] = np.nan
    else:
        out[COL_LOADABLE] = close.astype(float) - float(bottom)

    if COL_LOADABLE in out.columns and pd.api.types.is_numeric_dtype(out[COL_LOADABLE]):
        out[COL_LOADABLE] = out[COL_LOADABLE].round(2)

    return out


def _recalculate_total_inventory(df: pd.DataFrame, *, bottom: float | None) -> pd.DataFrame:
    """UI-only metric: Total Inventory = Close Inv + Bottoms (threshold).

    Represents the total physical inventory including the unusable "dead stock"
    (bottom / minimum inventory level).  Mirrors the symmetry with Loadable
    (Close Inv - Bottoms) but adds the bottom back in.

    Why read-only: the formula is purely derived; changing it would be
    meaningless without changing the underlying Close Inv or Bottom threshold.

    Shows NaN when no Bottom is configured (consistent with Loadable / Available
    Space behaviour – we never fabricate a number from NaN thresholds).
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    if "Close Inv" not in out.columns:
        return out

    close = _to_numeric_series(out["Close Inv"]).fillna(0.0)
    if bottom is None or (isinstance(bottom, float) and pd.isna(bottom)):
        out[COL_TOTAL_INVENTORY] = np.nan
    else:
        out[COL_TOTAL_INVENTORY] = (close.astype(float) + float(bottom)).round(2)

    return out


def _recalculate_accounting_inv(df: pd.DataFrame) -> pd.DataFrame:
    """UI-only metric: Accounting Inventory = Close Inv - Storage.

    Storage is a manual, session-only numeric field the user fills in per row.
    It represents inventory that is physically present but "tied up" (e.g. held
    in storage, nominated, or otherwise unavailable for dispatch) and thus
    excluded from the operationally-available closing figure.

    Why this column is computed (not editable): its value is *always* derived
    from two other columns; making it directly editable would create an
    inconsistency with the formula.

    Why Storage does NOT feed into Close Inv: Storage is a deduction for
    *accounting* purposes only and does not represent a physical movement.
    The inventory balance chain (Open Inv + Inflows - Outflows = Close Inv)
    is physical; accounting adjustments are tracked separately here.

    Initialises Storage to 0 if not yet present so that Accounting Inventory
    equals Close Inv until the user starts entering Storage values.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    if "Close Inv" not in out.columns:
        return out

    # Initialise Storage if this is the first time we see the df (e.g. freshly
    # loaded from DB where the column does not exist yet).
    if COL_STORAGE not in out.columns:
        out[COL_STORAGE] = np.nan

    close = _to_numeric_series(out["Close Inv"]).fillna(0.0)
    storage = _to_numeric_series(out[COL_STORAGE]).fillna(0.0)
    out[COL_ACCOUNTING_INV] = (close.astype(float) - storage.astype(float)).round(2)
    return out


def _fill_rack_averages(
    df: pd.DataFrame,
    *,
    avg_7day: float | None,
    avg_mtd: float | None,
) -> pd.DataFrame:
    """Scalar fallback: broadcast pre-computed rack averages to every row.

    Kept for compatibility; prefer _fill_rack_averages_per_row for display
    so each row reflects its own date-contextual average.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out[COL_7DAY_AVG_RACK] = round(float(avg_7day), 2) if avg_7day is not None else np.nan
    out[COL_MTD_AVG_RACK] = round(float(avg_mtd), 2) if avg_mtd is not None else np.nan
    return out


def _fill_rack_averages_per_row(
    df: pd.DataFrame,
    df_hist: pd.DataFrame,
) -> pd.DataFrame:
    """UI-only metrics: per-row 7-day and MTD rack averages keyed on each row's date.

    For each row in ``df`` at date D:

    • 7 Day Avg  – non-zero mean of the 7 most recent *historical* rows
      (SOURCE_TYPE != 'forecast') whose date is ≤ D.  Gives a rolling
      current-week reference rate that advances as dates move forward.

    • MTD Avg    – non-zero mean of *historical* rows whose date falls within
      [1st of D's month, D].  Each row reflects its own month-to-date period
      rather than a single today-anchored constant.

    Both averages exclude zeros and NaNs.  Forecast rows in ``df_hist`` are
    stripped so projected values never contaminate the statistics.
    Shows NaN when there is insufficient history for a given date.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Resolve rack column (raw or renamed form).
    rack_col: str | None = None
    for candidate in (COL_RACK_LIFTINGS_RAW, COL_RACK_LIFTING):
        if candidate in df_hist.columns:
            rack_col = candidate
            break

    if rack_col is None:
        out[COL_7DAY_AVG_RACK] = np.nan
        out[COL_MTD_AVG_RACK] = np.nan
        return out

    # Build clean historical base (no forecast rows, sorted by date).
    hist = df_hist.copy()
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    if "SOURCE_TYPE" in hist.columns:
        hist = hist[hist["SOURCE_TYPE"].astype(str).str.lower() != "forecast"]
    hist = hist.dropna(subset=["Date"])
    hist[rack_col] = pd.to_numeric(hist[rack_col], errors="coerce")
    # Aggregate sub-entries: multiple DB rows for the same date (e.g. 284 + 142 on Feb 10)
    # are summed into a single daily total (426) so that each calendar day counts as
    # exactly ONE data point in both the 7-day window and the MTD window.
    # Without this, a date with 2 sub-entries occupies 2 slots in tail(7), causing
    # heavily-split days to be over-represented and producing a lower/distorted average.
    hist = hist.groupby("Date", as_index=False)[rack_col].sum()
    hist = hist.sort_values("Date").reset_index(drop=True)

    dates = pd.to_datetime(out["Date"], errors="coerce")
    today = pd.Timestamp.today().normalize()

    avg_7day_list: list = []
    avg_mtd_list: list = []

    for row_date in dates:
        if pd.isna(row_date):
            avg_7day_list.append(np.nan)
            avg_mtd_list.append(np.nan)
            continue

        # Only compute averages for rows up to and including today.
        # Future (forecast) rows show None for both metrics.
        if row_date > today:
            avg_7day_list.append(np.nan)
            avg_mtd_list.append(np.nan)
            continue

        h_upto = hist[hist["Date"] <= row_date]

        # 7-day: last 7 historical rows up to this date, non-zero mean.
        last7 = h_upto.tail(7)
        v7 = _nonzero_mean(last7[rack_col]) if not last7.empty else None
        avg_7day_list.append(round(float(v7), 2) if v7 else np.nan)

        # MTD: rows in [1st of row_date's month, row_date], non-zero mean.
        month_start = row_date.replace(day=1)
        mtd = h_upto[h_upto["Date"] >= month_start]
        vm = _nonzero_mean(mtd[rack_col]) if not mtd.empty else None
        avg_mtd_list.append(round(float(vm), 2) if vm else np.nan)

    out[COL_7DAY_AVG_RACK] = avg_7day_list
    out[COL_MTD_AVG_RACK] = avg_mtd_list
    return out


def _recalculate_inventory_metrics(
    df: pd.DataFrame,
    *,
    id_col: str,
    safefill: float | None,
    bottom: float | None = None,
    ensure_fact_cols: bool = True,
    rack_7day_avg: float | None = None,
    rack_mtd_avg: float | None = None,
    df_hist: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = _recalculate_open_close_inv(df, id_col=id_col, ensure_fact_cols=ensure_fact_cols)
    out = _recalculate_total_closing_inv(out)
    out = _recalculate_available_space(out, safefill=safefill)
    out = _recalculate_loadable(out, bottom=bottom)
    # Phase-2 derived columns
    out = _recalculate_total_inventory(out, bottom=bottom)
    out = _recalculate_accounting_inv(out)
    if df_hist is not None:
        out = _fill_rack_averages_per_row(out, df_hist)
    else:
        out = _fill_rack_averages(out, avg_7day=rack_7day_avg, avg_mtd=rack_mtd_avg)
    return out


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


def _insert_fact_columns(column_order: list[str], *, df_cols: list[str], show_fact: bool) -> list[str]:
    """Insert "<col> Fact" columns immediately after their base column."""
    if not show_fact:
        return column_order

    out: list[str] = []
    seen = set()
    df_set = set(df_cols)

    for c in column_order:
        if c not in seen:
            out.append(c)
            seen.add(c)
        fact = FACT_COL_MAP.get(c)
        if fact and fact in df_set and fact not in seen:
            out.append(fact)
            seen.add(fact)

    return out


def _ensure_cols_after(
    column_order: list[str],
    *,
    required: list[str],
    after: str,
    before: str | None = None,
) -> list[str]:

    out = list(column_order)
    required = [c for c in required if c]

    # Remove existing occurrences (we'll re-insert in the desired spot).
    out = [c for c in out if c not in required]

    # Preferred insertion point: right after `after`.
    if after in out:
        pos = out.index(after) + 1
    # If `after` missing, insert before `before` if present, else append.
    elif before is not None and before in out:
        pos = out.index(before)
    else:
        pos = len(out)

    for i, c in enumerate(required):
        out.insert(pos + i, c)
    return out


DETAILS_EDITOR_VISIBLE_ROWS = 15
DETAILS_EDITOR_ROW_PX = 35  # approx row height incl. padding
DETAILS_EDITOR_HEADER_PX = 35
DETAILS_EDITOR_HEIGHT_PX = DETAILS_EDITOR_HEADER_PX + (DETAILS_EDITOR_VISIBLE_ROWS * DETAILS_EDITOR_ROW_PX)


def _render_blocking_overlay(show: bool, *, message: str = "Saving…") -> None:
    """Render a full-screen overlay to block clicks during long operations."""
    if not show:
        return

    # High z-index to sit above the whole app.
    msg = html.escape(str(message or "Saving…"))
    st.markdown(
        f"""
        <style>
        #details-save-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.35);
            /* Keep below Streamlit dialogs/modals, but above the app */
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            pointer-events: all;
        }}
        #details-save-overlay .card {{
            background: white;
            border-radius: 12px;
            padding: 18px 22px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.25);
            font-weight: 700;
            color: #2D3748;
        }}
        </style>
        <div id="details-save-overlay">
          <div class="card">{msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Confirm Save")
def _confirm_save_dialog(*, payload: dict) -> None:
    """Confirmation dialog for saving edited details rows."""

    scope_label = payload.get("scope_label") or "this selection"
    st.write(f"Are you sure you want to save changes for **{scope_label}**?")

    # Right-aligned primary action (no cancel; user can close with X)
    _, c_yes = st.columns([8, 2])
    with c_yes:
        if logged_button(
            "Save",
            type="primary",
            event="details_confirm_save",
            metadata={
                "region": payload.get("region"),
                "location": payload.get("location"),
                "system": payload.get("system"),
                "product": payload.get("product"),
                "scope_label": payload.get("scope_label"),
            },
        ):
            st.session_state["details_save_stage"] = "pre_save"
            st.session_state["details_save_payload"] = payload
            st.session_state["details_save_overlay"] = {"on": True, "df_key": payload.get("df_key")}
            st.rerun()


@st.dialog("Save Result")
def _save_result_dialog(*, result: dict) -> None:
    st.markdown(
        """
        <style>
        [data-testid="stDialog"] button[aria-label="Close"],
        [data-testid="stDialog"] button[title="Close"],
        [data-baseweb="modal"] button[aria-label="Close"],
        [data-baseweb="modal"] button[title="Close"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    ok = bool(result.get("ok"))
    n = int(result.get("n") or 0)
    err = str(result.get("error") or "")

    if ok:
        st.success(f"Saved successfully ({n} rows).")
        ph = st.empty()
        for i in range(5, 0, -1):
            ph.caption(f"Closing in {i}s…")
            time.sleep(1)
        # Auto-close: close popup first, then remove overlay on the next rerun.
        st.session_state["details_save_stage"] = None
        st.session_state["details_save_result"] = None
        st.session_state["details_save_overlay_removal_pending"] = result.get("df_key")
        st.rerun()
    else:
        st.error("Save failed! Please try again, or reach out to an administrator.")
        if err:
            st.code(err)


# Flow-column names *after* `DETAILS_RENAME` has been applied.
DISPLAY_INFLOW_COLS = [
    "Receipts",
    "Pipeline In",
    "Production",
]

DISPLAY_OUTFLOW_COLS = [
    "Deliveries",
    "Rack/Lifting",
    "Pipeline Out",
]

DISPLAY_NET_COLS = [
    "Adjustments",
    "Gain/Loss",
    "Transfers",
]


def _style_source_cells(
    df: pd.DataFrame,
    cols_to_color: list[str],
    *,
    fact_reference: pd.DataFrame | None = None,
    safefill: float | None = None,
    bottom: float | None = None,
) -> "pd.io.formats.style.Styler":
    """Apply cell-level background styles.

    Today (row_date == today):
      • Only the "Date" cell is highlighted blue (TODAY_BG).
      • All other cells in the row are uncolored.

    Yesterday (row_date == today − 1):
      • "Date", "Opening Inv", "Close Inv" cells → lighter yellow (YESTERDAY_HIGHLIGHT_BG).
      • All other cells (Total Closing Inv, Available Space, Loadable, View File, …) → no color.

    Older rows (row_date < today − 1):
      • No row-level color at all.

    Fact columns always get the FACT_BG grey regardless of date.

    Close Inv threshold override (applied last, wins over all row colors):
      • Red   (CLOSE_INV_ABOVE_SAFEFILL_BG) when Close Inv > SafeFill
      • Orange (CLOSE_INV_BELOW_BOTTOM_BG)  when Close Inv < Bottom
    """

    today = date.today()
    yesterday = today - timedelta(days=1)
    cols = list(df.columns)
    fact_cols = {c for c in cols if str(c).endswith(" Fact")}
    ref = fact_reference if isinstance(fact_reference, pd.DataFrame) else None

    def _get_fact_value(row: pd.Series, fact_col: str):
        if fact_col in row.index:
            return row.get(fact_col)
        if ref is not None and fact_col in ref.columns and row.name in ref.index:
            return ref.at[row.name, fact_col]
        return None

    def _close_inv_matches(row: pd.Series) -> bool:
        """Return True when Close Inv and Close Inv Fact are within threshold."""
        base_val = _to_float(row.get("Close Inv")) if "Close Inv" in row.index else 0.0
        fact_val = _get_fact_value(row, "Close Inv Fact")
        if fact_val is None:
            return True  # no fact to compare → treat as match
        return abs(base_val - _to_float(fact_val)) <= SYSTEM_DISCREPANCY_THRESHOLD_BBL

    def _to_date(v):
        """Normalise whatever the Date cell holds to a datetime.date."""
        if v is None:
            return None
        if isinstance(v, date):
            return v
        try:
            return pd.Timestamp(v).date()
        except Exception:
            return None

    def _row_style(row: pd.Series) -> list[str]:
        row_date = _to_date(row.get("Date") if "Date" in row.index else None)

        styles: list[str] = []

        # Determine yesterday's highlight colour once per row (avoids recalculating per cell).
        if row_date == yesterday:
            yesterday_bg = MATCH_BG if _close_inv_matches(row) else MISMATCH_BG
        else:
            yesterday_bg = ""

        for c in cols:
            if c in fact_cols:
                # Fact columns always get grey
                styles.append(f"background-color: {FACT_BG};")
            elif row_date == today:
                # Today: only the Date cell gets blue
                styles.append(f"background-color: {TODAY_BG};" if c == "Date" else "")
            elif row_date == yesterday:
                # Yesterday: Date / Opening Inv / Close Inv get green (match) or yellow (mismatch).
                # All other cells (Total Closing Inv, Available Space, Loadable, etc.) → no color.
                styles.append(
                    f"background-color: {yesterday_bg};"
                    if c in YESTERDAY_HIGHLIGHT_COLS
                    else ""
                )
            else:
                # All other rows (past or future): no row-level color
                styles.append("")

        # --- Cell-level override: Close Inv threshold coloring ---
        # Applies to any row with a future date (strictly after today),
        # regardless of SOURCE_TYPE — this covers both system-feed records
        # with future dates AND app-generated forecast rows.
        # Runs last so it wins over every row-level color.
        is_future = row_date is not None and row_date > today
        if is_future and "Close Inv" in cols:
            ci_idx = cols.index("Close Inv")
            raw_val = row.get("Close Inv") if "Close Inv" in row.index else None
            if raw_val is not None and not (isinstance(raw_val, float) and pd.isna(raw_val)):
                close_val = _to_float(raw_val)
                if safefill is not None and close_val > safefill:
                    styles[ci_idx] = f"background-color: {CLOSE_INV_ABOVE_SAFEFILL_BG};"
                elif bottom is not None and close_val < bottom:
                    styles[ci_idx] = f"background-color: {CLOSE_INV_BELOW_BOTTOM_BG};"
        return styles

    return df.style.apply(_row_style, axis=1).hide(axis="index")


def _to_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return 0.0

        if isinstance(x, str):
            s = x.strip()
            if s in {"", "—", "-"}:
                return 0.0
            # remove thousands separators
            s = s.replace(",", "")
            return float(s)
        return float(x)
    except Exception:
        return 0.0


def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce a Series to numeric, tolerating formatted strings like '1,234.00'."""
    if s is None:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    # Remove commas in strings, then coerce.
    s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    # Treat blanks / em dashes as NaN
    s2 = s2.replace({"": np.nan, "—": np.nan, "-": np.nan})
    return pd.to_numeric(s2, errors="coerce")


def _sum_row(row: pd.Series, cols: list[str]) -> float:
    return float(sum(_to_float(row.get(c, 0.0)) for c in cols if c in row.index))


def _recalculate_open_close_inv(
    df: pd.DataFrame,
    *,
    id_col: str,
    ensure_fact_cols: bool = True,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()

    # Work with datetimes internally for stable sorting; convert back to date at end.
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    numeric_candidates = [
        "Opening Inv",
        "Opening Inv Fact",
        COL_AVAILABLE,
        COL_AVAILABLE_FACT,
        COL_INTRANSIT,
        COL_INTRANSIT_FACT,
        "Close Inv",
        "Close Inv Fact",
        *DISPLAY_INFLOW_COLS,
        *DISPLAY_OUTFLOW_COLS,
        *DISPLAY_NET_COLS,
    ]
    for c in numeric_candidates:
        if c in out.columns:
            out[c] = _to_numeric_series(out[c]).fillna(0.0)

    group_cols = [id_col]
    if "Product" in out.columns:
        group_cols.append("Product")

    # Stable sort so we don't get UI flicker when other columns tie.
    sort_cols = ["Date"] + group_cols
    out = out.sort_values(sort_cols, kind="mergesort")

    def _apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date", kind="mergesort").copy()
        prev_close = 0.0

        for i, idx in enumerate(g.index):
            if i == 0:
                opening = _to_float(g.at[idx, "Opening Inv"]) if "Opening Inv" in g.columns else 0.0
            else:
                opening = prev_close

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

    parts: list[pd.DataFrame] = []
    for _, g in out.groupby(group_cols, dropna=False, sort=False):
        parts.append(_apply(g))

    out = pd.concat(parts, axis=0) if parts else out
    # Preserve stable UI ordering.
    out = out.sort_values(sort_cols, kind="mergesort")

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
        # The editor may return strings with commas; normalize them before compare.
        b = _to_numeric_series(before[c]).fillna(0.0).to_numpy(dtype=float)
        a = _to_numeric_series(after[c]).fillna(0.0).to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=0, atol=1e-9):
            return True
    return False


def _locked_cols(id_col: str, cols: list[str]) -> list[str]:
    wanted = [c.format(id_col=id_col) for c in LOCKED_BASE_COLS]
    return [c for c in wanted if c in cols]


def _column_config(df: pd.DataFrame, cols: list[str], id_col: str):
    locked = set(_locked_cols(id_col, cols))
    # Fact columns should always be read-only.
    locked.update({c for c in cols if str(c).endswith(" Fact")})
    locked.add("SOURCE_TYPE")
    NUM_FMT = "accounting"

    cfg: dict[str, object] = {
        "Date": st.column_config.DateColumn("Date", disabled=True, format="YYYY-MM-DD"),
        id_col: st.column_config.TextColumn(id_col, disabled=True),
        "Product": st.column_config.TextColumn("Product", disabled=True),
        "updated": st.column_config.CheckboxColumn("updated", default=False),
        "Batch": st.column_config.TextColumn("Batch"),
        "Notes": st.column_config.TextColumn("Notes"),
        # Storage: user-editable numeric field persisted to DB as STORAGE_BBL.
        # Intentionally NOT disabled so operators can type in a value.
        # Does NOT flow into Close Inv; it drives the Accounting Inventory derived column only.
        COL_STORAGE: st.column_config.NumberColumn(
            COL_STORAGE,
            disabled=False,
            format=NUM_FMT,
            help=(
                "Manual entry only. Enter the volume held in storage. "
                "This value does NOT affect Closing Inventory. "
                "It is used to compute Accounting Inventory (Close Inv − Storage)."
            ),
        ),
        "SOURCE_TYPE": st.column_config.TextColumn("SOURCE_TYPE", disabled=True),
        COL_VIEW_FILE: st.column_config.CheckboxColumn(
            COL_VIEW_FILE,
            default=False,
            disabled=(DATA_SOURCE != "snowflake"),
            help="Check to open a popup with downloadable system files for this row.",
        ),
    }

    for c in cols:
        if c in cfg or c == "Notes":
            continue
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            cfg[c] = st.column_config.NumberColumn(c, disabled=(c in locked), format=NUM_FMT)

    for c in locked:
        if c in {"Date", id_col, "Product"}:
            continue
        if c in cols and c not in cfg:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                cfg[c] = st.column_config.NumberColumn(c, disabled=True, format=NUM_FMT)
            else:
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

    # Additional inventory metrics (not flows)
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
    if "Notes" in df.columns:
        agg_map["Notes"] = "last"
    if "SOURCE_TYPE" in df.columns:
        agg_map["SOURCE_TYPE"] = "first"

    # Sub-breakdown columns (phase-2): sum like flow columns.
    for _sub_col in [COL_TULSA, COL_EL_DORADO, COL_OTHER, COL_ARGENTINE, COL_FROM_327_RECEIPT]:
        if _sub_col in df.columns:
            agg_map[_sub_col] = "sum"
    # Storage is inventory-level: keep the last value of the day.
    if COL_STORAGE in df.columns:
        agg_map[COL_STORAGE] = "last"

    # Keep file locations for the day (Snowflake-only; list column).
    if "FILE_LOCATION" in df.columns:
        agg_map["FILE_LOCATION"] = "last"

    return df.groupby(group_cols, as_index=False).agg(agg_map)


def _available_flow_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in FORECAST_FLOW_COLS if c in df.columns]


def _ensure_lineage_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

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


def _nonzero_mean(s: pd.Series) -> float:
    """Mean excluding zeros (and NaNs). Returns 0.0 if nothing remains."""
    if s is None:
        return 0.0
    vals = pd.to_numeric(s, errors="coerce").fillna(0.0)
    vals = vals[vals != 0]
    return float(vals.mean()) if len(vals) else 0.0


def _compute_rack_averages(df_prod: pd.DataFrame) -> tuple[float | None, float | None]:
    """Compute 7-day and MTD averages for Rack/Liftings from historical rows.

    Parameters
    ----------
    df_prod : pd.DataFrame
        The per-product slice *before* forecast rows are appended.  Columns
        are still in their raw/pre-rename form (e.g. "Rack/Liftings", not
        "Rack/Lifting").

    Returns
    -------
    (avg_7day, avg_mtd) : tuple[float | None, float | None]
        Both values exclude zeros and NaNs (via _nonzero_mean).
        Returns None for an average when there is insufficient data.

    Design notes
    ------------
    • We consume the RAW column name (COL_RACK_LIFTINGS_RAW = "Rack/Liftings")
      because this function is called before build_details_view applies the
      DETAILS_RENAME_MAP that turns "Rack/Liftings" → "Rack/Lifting".

    • Forecast rows are excluded: only SOURCE_TYPE != 'forecast' contributes,
      so projected future values never inflate these averages.

    • 7 Day Avg  : last 7 calendar rows (sorted by date), zeros excluded.
      Gives a current-week reference for rack throughput.

    • MTD Avg    : rows from the 1st of the *current calendar month* up to
      today (UTC date), zeros excluded.  Aligns with standard MTD reporting
      periods (i.e. the current billing/settlement month).
    """
    if df_prod is None or df_prod.empty:
        return None, None

    # Accept both raw ("Rack/Liftings") and renamed ("Rack/Lifting") columns.
    rack_col: str | None = None
    for candidate in (COL_RACK_LIFTINGS_RAW, COL_RACK_LIFTING):
        if candidate in df_prod.columns:
            rack_col = candidate
            break

    if rack_col is None:
        return None, None

    hist = df_prod.copy()
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")

    # Strip forecast-only rows so future projections don't distort the averages.
    if "SOURCE_TYPE" in hist.columns:
        hist = hist[hist["SOURCE_TYPE"].astype(str).str.lower() != "forecast"]

    hist = hist.dropna(subset=["Date"]).sort_values("Date")
    if hist.empty:
        return None, None

    # --- 7 Day Avg: last 7 historical rows, zeros excluded ---
    last7 = hist.tail(7)
    avg_7day_val = _nonzero_mean(last7[rack_col])
    avg_7day = avg_7day_val if avg_7day_val != 0.0 else None

    # --- MTD Avg: current calendar month to today, zeros excluded ---
    today_ts = pd.Timestamp.today().normalize()
    month_start = today_ts.replace(day=1)
    mtd = hist[(hist["Date"] >= month_start) & (hist["Date"] <= today_ts)]
    if mtd.empty:
        avg_mtd = None
    else:
        avg_mtd_val = _nonzero_mean(mtd[rack_col])
        avg_mtd = avg_mtd_val if avg_mtd_val != 0.0 else None

    return avg_7day, avg_mtd


def _constant_means_excluding_zeros(
    hist: pd.DataFrame,
    flow_cols: list[str],
    *,
    tail_n: int | None,
) -> dict[str, float]:
    """Compute constant means for flow columns.

    Rules:
    - Optionally restrict to last N rows (by Date)
    - Exclude 0-valued days from the mean
    """
    if hist is None or hist.empty or not flow_cols:
        return {c: 0.0 for c in (flow_cols or [])}

    h = hist.sort_values("Date").copy()
    if tail_n is not None:
        h = h.tail(int(tail_n))

    out: dict[str, float] = {}
    for c in flow_cols:
        if c not in h.columns:
            out[c] = 0.0
            continue
        out[c] = _nonzero_mean(h[c])
    return out


def _make_forecast_flow_estimator(
    hist: pd.DataFrame,
    *,
    flow_cols: list[str],
    method: str,
):
    """Return a function(d) -> {flow_col: value} for a given method."""
    m = str(method or "").strip() or "weekday_weighted"

    if m == "7_day_avg":
        const = _constant_means_excluding_zeros(hist, flow_cols, tail_n=7)
        return lambda d: dict(const)

    if m == "mtd_avg":
        const = _constant_means_excluding_zeros(hist, flow_cols, tail_n=None)
        return lambda d: dict(const)

    means = _weekday_weighted_means(hist, flow_cols=flow_cols)
    return lambda d: {c: float(means.get((int(d.weekday()), c), 0.0)) for c in flow_cols}


def _roll_inventory(prev_close: float, flows: dict[str, float], flow_cols: list[str], system: str = None, product: str = None) -> tuple[float, float]:
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


def _fill_missing_internal_dates(
    daily: pd.DataFrame,
    *,
    id_col: str,
    start_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    By default we fill only the internal gaps between the first and last existing
    date for each group.

    Note: prepending normally changes the rolling inventory anchor (the first row
    drives the whole series). To keep the series stable, we seed the prepended
    rows' Open/Close Inv to the first observed Open Inv (so inventory stays flat
    across the prepended window unless the user edits flows).
    """

    if daily is None or daily.empty or "Date" not in daily.columns:
        return daily

    out = daily.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    group_cols = [id_col]
    if "Product" in out.columns:
        group_cols.append("Product")

    # Identify which numeric columns should be zero-filled for inserted days.
    flow_cols = _available_flow_cols(out)
    zero_fill_cols = [
        "Open Inv",
        "Close Inv",
        COL_AVAILABLE,
        COL_INTRANSIT,
        *flow_cols,
    ]
    # Also zero-fill any fact columns if present.
    zero_fill_cols += [c for c in out.columns if str(c).endswith(" Fact")]
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

        full_idx = pd.date_range(start=start_d, end=max_d, freq="D")
        g2 = g.set_index("Date").reindex(full_idx)

        if start_date is not None and "Open Inv" in g2.columns:
            try:
                anchor_open = float(pd.to_numeric(g.loc[g["Date"] == min_d, "Open Inv"], errors="coerce").fillna(0.0).iloc[0])
            except Exception:
                anchor_open = 0.0
            if "Close Inv" in g2.columns:
                g2.loc[g2.index < min_d, ["Open Inv", "Close Inv"]] = anchor_open
            else:
                g2.loc[g2.index < min_d, ["Open Inv"]] = anchor_open

        # Restore non-date group key columns.
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, val in zip(group_cols, keys):
            g2[col] = val

        # Defaults for inserted rows.
        if "updated" in g2.columns:
            g2["updated"] = pd.to_numeric(g2["updated"], errors="coerce").fillna(0).astype(int)
        if "SOURCE_TYPE" in g2.columns:
            g2["SOURCE_TYPE"] = g2["SOURCE_TYPE"].fillna("")

        # Strings / misc columns.
        if "Batch" in g2.columns:
            g2["Batch"] = g2["Batch"].fillna("")
        if "Notes" in g2.columns:
            g2["Notes"] = g2["Notes"].fillna("")
        if "FILE_LOCATION" in g2.columns:
            # Only fill NaNs; keep existing lists.
            g2["FILE_LOCATION"] = g2["FILE_LOCATION"].apply(
                lambda v: ([] if (v is None or (isinstance(v, float) and pd.isna(v))) else v)
            )

        # Numeric columns: 0 for inserted days.
        for c in zero_fill_cols:
            g2[c] = pd.to_numeric(g2[c], errors="coerce").fillna(0.0)

        g2 = g2.reset_index().rename(columns={"index": "Date"})
        filled_parts.append(g2)

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
    if df.empty:
        return df

    daily = _aggregate_daily_details(df, id_col=id_col)
    if daily.empty:
        return daily

    daily = _ensure_lineage_cols(daily).sort_values("Date")
    # Ensure the UI has a continuous day-by-day grid within the observed history
    daily = _fill_missing_internal_dates(daily, id_col=id_col, start_date=history_start).sort_values("Date")
    flow_cols = _available_flow_cols(daily)

    forecast_flow_cols = [c for c in [COL_RACK_LIFTINGS_RAW] if c in flow_cols]

    if forecast_end is not None:
        forecast_end = pd.Timestamp(forecast_end)

    forecast_rows: list[dict] = []

    forecast_method = get_rack_lifting_forecast_method(region=str(region or "Unknown"), location=location)

    for (id_val, product), group in daily.groupby([id_col, "Product"], dropna=False):
        group = group.sort_values("Date")
        last_date = pd.Timestamp(group["Date"].max())

        estimate = _make_forecast_flow_estimator(group, flow_cols=forecast_flow_cols, method=forecast_method)

        prev_close = _last_close_inv(group)
        for d in _forecast_dates(last_date, forecast_end, default_days):
            # Only estimate Rack/Liftings; keep everything else 0.
            flows = {c: 0.0 for c in flow_cols}
            if forecast_flow_cols:
                flows.update(estimate(d))

            opening, closing = _roll_inventory(
                prev_close,
                flows,
                flow_cols,
                system=str(id_val) if id_col == "System" else None,
                product=str(product) if product else None
            )

            prev_close = closing

            row = {
                "Date": d,
                id_col: id_val,
                "Product": product,
                "SOURCE_TYPE": "forecast",
                "updated": 0,
                "Batch": "",
                "Notes": "",
                "FILE_LOCATION": [],
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
        if c in {"Date", id_col, "Product", "Notes", "updated"}:
            continue
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(2)

    return df, cols


def _threshold_values(
    *,
    region: str,
    location: str | None,
    product: str | None = None,
) -> tuple[float | None, float | None, str | None]:
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


def _render_threshold_cards(
    *,
    bottom: float | None,
    safefill: float | None,
    note: str | None = None,
    c_safefill,
    c_bottom,
    c_note,
    c_info,
) -> None:

    with c_safefill:
        v = "—" if safefill is None else f"{safefill:,.0f}"
        st.markdown(
            f"""
            <div class="mini-card" style="margin-bottom:1rem
            ;">
              <p class="label">SafeFill</p>
              <p class="value">{v}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_bottom:
        v = "—" if bottom is None else f"{bottom:,.0f}"
        st.markdown(
            f"""
            <div class="mini-card" style="margin-bottom:1rem;">
              <p class="label">Bottom</p>
              <p class="value">{v}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_note:
        v = "—" if note in (None, "") else str(note)
        st.markdown(
            f"""
            <div class="mini-card" style="margin-bottom:1rem;">
              <p class="label">Note</p>
              <p class="value" style="font-size:0.95rem; font-weight:700;">{v}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_info:
        st.markdown(
            """
            <div style="font-size:0.72rem; line-height:1.55; margin-top:0.25rem;">
              <span style="
                display:inline-block; width:10px; height:10px;
                background:#ffb3b3; border:1px solid #ccc;
                border-radius:2px; margin-right:4px;
              "></span>Above SafeFill<br>
              <span style="
                display:inline-block; width:10px; height:10px;
                background:#ffe0b2; border:1px solid #ccc;
                border-radius:2px; margin-right:4px;
              "></span>Below Bottom
            </div>
            """,
            unsafe_allow_html=True,
        )


def _build_editor_df(df_display: pd.DataFrame, *, id_col: str, ui_cols: list[str]) -> pd.DataFrame:
    # Keep any flow columns that might exist, even if not currently visible.
    extra = [
        "Production",
        "Adjustments",
        "Receipts",
        "Deliveries",
        "Pipeline In",
        "Pipeline Out",
        "Gain/Loss",
        "Transfers",
        "Rack/Lifting",
        "Batch",
    ]

    base = [
        "Date",
        id_col,
        "Product",
        "SOURCE_TYPE",
        "updated",
        "Batch",
        "Notes",
        # Storage: user-editable field persisted to DB as STORAGE_BBL; initialised to 0 if not present
        COL_STORAGE,
        "FILE_LOCATION",
        COL_VIEW_FILE,
        "Opening Inv",
        "Close Inv",
        "Opening Inv Fact",
        "Close Inv Fact",
    ]

    fact_cols = [c for c in df_display.columns if str(c).endswith(" Fact")]
    desired = []
    always_include = {"FILE_LOCATION", COL_VIEW_FILE}
    for c in base + fact_cols + ui_cols + extra:
        if c in desired:
            continue
        if c in always_include or c in df_display.columns:
            desired.append(c)
        # COL_STORAGE may not exist in df_display yet; always include it
        elif c == COL_STORAGE:
            desired.append(c)

    out = df_display.copy()
    # Ensure list column exists even in SQLite/forecast rows.
    if "FILE_LOCATION" not in out.columns:
        out["FILE_LOCATION"] = [[] for _ in range(len(out))]
    # UI action column: checkbox that behaves like a button.
    if COL_VIEW_FILE not in out.columns:
        out[COL_VIEW_FILE] = False
    else:
        out[COL_VIEW_FILE] = out[COL_VIEW_FILE].fillna(False).astype(bool)
    # Storage: initialise to NaN for rows where the user has not entered a value.
    if COL_STORAGE not in out.columns:
        out[COL_STORAGE] = np.nan

    return out[desired].reset_index(drop=True)


def display_location_details(
    df_filtered: pd.DataFrame,
    active_region: str,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    selected_loc: str | None,
):

    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return

    if selected_loc in (None, ""):
        st.info("Select a Location/System above and press Submit.")
        return

    df_loc = df_filtered[df_filtered["Location"] == str(selected_loc)] if "Location" in df_filtered.columns else pd.DataFrame()
    if df_loc.empty:
        st.info("No data available for the selected location + date range.")
        return

    # Tabs by Product within this Location
    products = sorted(df_loc["Product"].dropna().astype(str).unique().tolist()) if "Product" in df_loc.columns else []
    if not products:
        st.info("No products available for the selected location.")
        return

    c_toggle, c_loc, _ = st.columns([5, 5, 1])
    with c_toggle:
        show_fact = st.toggle(
            "Show Terminal Feed",
            value=bool(st.session_state.get(f"details_show_fact|{active_region}|{selected_loc}|location", False)),
            key=f"details_show_fact|{active_region}|{selected_loc}|location",
            help="Show upstream system values next to the editable columns.",
        )
    with c_loc:
        st.markdown(
            f"<h1 style='color: green; font-weight: 700; font-size: 1.2rem'>{selected_loc}</h1>",
            unsafe_allow_html=True
        )

    for i, tab in enumerate(st.tabs(products)):
        prod_name = products[i]
        with tab:
            # Compute keys early so we can render overlay/result properly.
            state_key = (
                f"{active_region}_{selected_loc}_{prod_name}"
                f"|{pd.Timestamp(start_ts).date()}|{pd.Timestamp(end_ts).date()}"
                f"|edit"
            )

            # Canonical df key MUST NOT depend on the fact-toggle; otherwise
            # we'd end up with two independent session_state dfs and edits won't
            # persist when toggling FACT columns on/off.
            df_key = f"{state_key}__df"

            # Widget state *can* depend on fact-toggle because the visible columns differ.
            widget_scope_key = f"{state_key}|fact={int(bool(show_fact))}"
            ver_key = f"{widget_scope_key}__ver"
            widget_key = f"{widget_scope_key}__editor_v{int(st.session_state.get(ver_key, 0))}"

            # If a prior rerun requested opening the View File dialog, do it now.
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

            bottom, safefill, note = _threshold_values(
                region=active_region,
                location=str(selected_loc),
                product=str(prod_name),
            )

            # Threshold cards + Enable-Save toggle + Save button on the same row.
            c_sf, c_bt, c_note, c_info, c_enable, c_save = st.columns([2, 2, 3, 0.7, 1.3, 2])
            _render_threshold_cards(
                bottom=bottom,
                safefill=safefill,
                note=note,
                c_safefill=c_sf,
                c_bottom=c_bt,
                c_note=c_note,
                c_info=c_info,
            )

            enable_key = f"details_enable_save|{active_region}|{selected_loc}|{prod_name}"
            enable_ver_key = f"{enable_key}__ver"
            enable_widget_key = f"{enable_key}__v{int(st.session_state.get(enable_ver_key, 0))}"

            with c_enable:
                enable_save = st.toggle(
                    "Enable Save",
                    value=bool(st.session_state.get(enable_widget_key, False)),
                    key=enable_widget_key,
                    help="Toggle ON before saving to ensure the last edited cell commits to the table.",
                )

            with c_save:
                save_clicked = logged_button(
                    f"Save {prod_name} Data",
                    key=f"save_{active_region}_{selected_loc}_{prod_name}",
                    type="primary",
                    disabled=(not bool(enable_save)),
                    help="Save all rows shown in the grid for this product.",
                    event="details_save_clicked",
                    metadata={
                        "region": active_region,
                        "scope": "location",
                        "location": selected_loc,
                        "product": prod_name,
                    },
                )

            df_prod = df_loc[df_loc["Product"].astype(str) == str(prod_name)]

            # Compute rack averages from *historical* rows before forecast rows are
            # appended.  This prevents future projected values from contaminating
            # the 7-day and MTD statistics shown in the grid.
            rack_7day_avg, rack_mtd_avg = _compute_rack_averages(df_prod)

            # Forecast should be bounded by the user-selected date range.
            df_all = _extend_with_30d_forecast(
                df_prod,
                id_col="Location",
                region=active_region,
                location=str(selected_loc),
                history_start=start_ts,
                forecast_end=end_ts,
            )
            df_display, cols = build_details_view(df_all, id_col="Location")

            df_display = _recalculate_total_closing_inv(df_display)
            df_display = _recalculate_available_space(df_display, safefill=safefill)
            df_display = _recalculate_loadable(df_display, bottom=bottom)
            # Phase-2 derived columns: computed every render pass so they always
            # reflect the current Bottom threshold, Storage entries, and historical
            # rack data without requiring a separate save/reload cycle.
            df_display = _recalculate_total_inventory(df_display, bottom=bottom)
            df_display = _recalculate_accounting_inv(df_display)
            df_display = _fill_rack_averages_per_row(df_display, df_prod)
            cols = [c for c in (["Date", "Location"] + DETAILS_COLS) if c in df_display.columns]

            visible = get_visible_columns(region=active_region, location=str(selected_loc))
            column_order: list[str] = []
            for c in visible:
                if c == COL_VIEW_FILE:
                    if c not in column_order:
                        column_order.append(c)
                    continue

                if c in cols and c not in column_order:
                    column_order.append(c)

            if "Close Inv" in column_order:
                # Always show the UI-only calculated metrics immediately after Close Inv.
                column_order = _ensure_cols_after(
                    column_order,
                    required=[COL_TOTAL_CLOSING_INV, COL_AVAILABLE_SPACE],
                    after="Close Inv",
                    before=None,
                )

                column_order = _ensure_cols_after(
                    column_order,
                    required=[COL_AVAILABLE_SPACE],
                    after=COL_TOTAL_CLOSING_INV if COL_TOTAL_CLOSING_INV in column_order else "Close Inv",
                    before=None,
                )

                column_order = _ensure_cols_after(
                    column_order,
                    required=[COL_LOADABLE],
                    after=COL_AVAILABLE_SPACE if COL_AVAILABLE_SPACE in column_order else "Close Inv",
                    before=None,
                )

                # Phase-2: Total Inventory, Storage, and Accounting Inventory
                # are grouped immediately after Loadable so the operator sees all
                # inventory-level metrics in one contiguous block before the flow
                # columns begin.
                #
                # Ordering rationale:
                #   Loadable  →  Total Inventory  →  Storage  →  Accounting Inv
                #
                # Total Inventory (Close + Bottoms) sits next to Loadable
                # (Close - Bottoms) for easy comparison of usable vs. total stock.
                # Storage is editable and placed before Accounting Inv so users
                # can see the immediate impact of their entry on Accounting Inv.
                anchor_after_loadable = (
                    COL_LOADABLE if COL_LOADABLE in column_order else
                    COL_AVAILABLE_SPACE if COL_AVAILABLE_SPACE in column_order else
                    "Close Inv"
                )
                column_order = _ensure_cols_after(
                    column_order,
                    required=[COL_TOTAL_INVENTORY],
                    after=anchor_after_loadable,
                    before=None,
                )
                column_order = _ensure_cols_after(
                    column_order,
                    required=[COL_STORAGE],
                    after=COL_TOTAL_INVENTORY if COL_TOTAL_INVENTORY in column_order else anchor_after_loadable,
                    before=None,
                )
                column_order = _ensure_cols_after(
                    column_order,
                    required=[COL_ACCOUNTING_INV],
                    after=COL_STORAGE if COL_STORAGE in column_order else anchor_after_loadable,
                    before=None,
                )

                if COL_VIEW_FILE in column_order:
                    # Keep View File after the last Close-Inv-group column.
                    anchor = (
                        COL_ACCOUNTING_INV if COL_ACCOUNTING_INV in column_order else
                        COL_LOADABLE if COL_LOADABLE in column_order else
                        COL_AVAILABLE_SPACE if COL_AVAILABLE_SPACE in column_order else
                        "Close Inv"
                    )
                    column_order = _ensure_cols_after(
                        column_order,
                        required=[COL_VIEW_FILE],
                        after=anchor,
                        before=None,
                    )

                if "Batch" in column_order:
                    anchor = COL_VIEW_FILE if COL_VIEW_FILE in column_order else "Close Inv"
                    column_order = _ensure_cols_after(
                        column_order,
                        required=["Batch"],
                        after=anchor,
                        before=None,
                    )

            # Phase-2: 7 Day Avg and MTD Avg are placed immediately after
            # Rack/Lifting so operators can compare the actual lifting against
            # the recent average in a single glance.
            if COL_RACK_LIFTING in column_order:
                column_order = _ensure_cols_after(
                    column_order,
                    required=[COL_7DAY_AVG_RACK, COL_MTD_AVG_RACK],
                    after=COL_RACK_LIFTING,
                    before=None,
                )

            # Phase-2: Sub-breakdown columns placed after Adjustments.
            # Force-inserted so they appear even when loading an old stored
            # column config that pre-dates these columns being added.
            _sub_breakdown_cols = [
                COL_TULSA, COL_EL_DORADO, COL_OTHER,
                COL_ARGENTINE, COL_FROM_327_RECEIPT,
            ]
            _sub_anchor = (
                COL_ADJUSTMENTS if COL_ADJUSTMENTS in column_order else
                "Adjustments" if "Adjustments" in column_order else
                COL_PRODUCTION if COL_PRODUCTION in column_order else
                "Production"
            )
            _sub_available = [c for c in _sub_breakdown_cols if c in cols]
            if _sub_available and _sub_anchor in column_order:
                column_order = _ensure_cols_after(
                    column_order,
                    required=_sub_available,
                    after=_sub_anchor,
                    before=None,
                )

            column_order = _insert_fact_columns(column_order, df_cols=list(df_display.columns), show_fact=show_fact)

            locked_cols = _locked_cols("Location", cols)
            if show_fact:
                for base in list(locked_cols):
                    fact = FACT_COL_MAP.get(base)
                    if fact and fact in df_display.columns and fact not in locked_cols:
                        locked_cols.append(fact)

            column_config = _column_config(df_display, column_order, "Location")
            column_config = {k: v for k, v in column_config.items() if k in column_order}

            editor_df = _build_editor_df(df_display, id_col="Location", ui_cols=column_order)

            # Initialize/refresh canonical df (schema should be stable across FACT toggle).
            if df_key not in st.session_state or list(st.session_state[df_key].columns) != list(editor_df.columns):
                st.session_state[df_key] = _recalculate_inventory_metrics(
                    editor_df,
                    id_col="Location",
                    safefill=safefill,
                    bottom=bottom,
                    ensure_fact_cols=True,
                    rack_7day_avg=rack_7day_avg,
                    rack_mtd_avg=rack_mtd_avg,
                    df_hist=df_prod,
                )

            st.session_state[df_key] = st.session_state[df_key].reset_index(drop=True)

            # Build the view df for the current FACT toggle (hide fact cols if toggle is off).
            view_cols = [
                c for c in st.session_state[df_key].columns
                if (bool(show_fact) or (not str(c).endswith(" Fact")))
            ]

            view_df = st.session_state[df_key].loc[:, view_cols].copy()

            styled = _style_source_cells(
                view_df,
                locked_cols,
                fact_reference=(st.session_state[df_key] if not bool(show_fact) else None),
                safefill=safefill,
                bottom=bottom,
            )

            editor_column_order = [c for c in column_order if c in view_cols]
            editor_column_config = {k: v for k, v in column_config.items() if k in editor_column_order}

            edited = dynamic_input_data_editor(
                styled,
                num_rows="fixed",
                width="stretch",
                height=DETAILS_EDITOR_HEIGHT_PX,
                hide_index=True,
                column_order=editor_column_order,
                key=widget_key,
                column_config=editor_column_config,
            )

            # Recompute inventory on the view df only (do not re-add hidden FACT columns).
            # Pass the pre-computed rack averages so they stay in sync even after edits.
            recomputed_view = _recalculate_inventory_metrics(
                edited,
                id_col="Location",
                safefill=safefill,
                bottom=bottom,
                ensure_fact_cols=False,
                rack_7day_avg=rack_7day_avg,
                rack_mtd_avg=rack_mtd_avg,
                df_hist=df_prod,
            ).reset_index(drop=True)

            # Merge the edited view back into the canonical df so edits persist across toggle.
            canonical = st.session_state[df_key].reset_index(drop=True)
            if canonical.shape[0] != recomputed_view.shape[0]:
                # Safety fallback: rebuild canonical if something changed the rowset.
                canonical = _recalculate_inventory_metrics(
                    editor_df,
                    id_col="Location",
                    safefill=safefill,
                    bottom=bottom,
                    ensure_fact_cols=True,
                    rack_7day_avg=rack_7day_avg,
                    rack_mtd_avg=rack_mtd_avg,
                    df_hist=df_prod,
                ).reset_index(drop=True)
            else:
                for c in recomputed_view.columns:
                    if c in canonical.columns:
                        canonical[c] = recomputed_view[c].values
                    else:
                        canonical[c] = recomputed_view[c].values

            st.session_state[df_key] = canonical

            # Use canonical df for downstream actions.
            recomputed = st.session_state[df_key]

            if COL_VIEW_FILE in recomputed.columns:
                view_mask = recomputed[COL_VIEW_FILE].fillna(False).astype(bool)
                if bool(view_mask.any()):
                    idx = int(view_mask[view_mask].index[0])
                    file_locations = recomputed.at[idx, "FILE_LOCATION"] if "FILE_LOCATION" in recomputed.columns else []

                    # Clear the action cell so it behaves like a button.
                    st.session_state[df_key].at[idx, COL_VIEW_FILE] = False

                    st.session_state["details_view_file_payload"] = {
                        "df_key": df_key,
                        "row": idx,
                        "date": str(recomputed.at[idx, "Date"]) if "Date" in recomputed.columns else None,
                        "location": str(selected_loc) if selected_loc is not None else None,
                        "product": str(prod_name) if prod_name is not None else None,
                        "file_locations": file_locations,
                    }
                    # Force the editor to refresh so the selectbox resets.
                    st.session_state[ver_key] = int(st.session_state.get(ver_key, 0)) + 1
                    st.rerun()

            if _needs_inventory_rerun(edited, recomputed_view):
                st.session_state[ver_key] = int(st.session_state.get(ver_key, 0)) + 1
                st.rerun()

            # Save flow (after editor so we persist the latest recomputed values).
            if save_clicked:
                log_audit(
                    event="details_save_dialog_opened",
                    metadata={"region": active_region, "scope": "location", "location": selected_loc, "product": prod_name},
                )
                _confirm_save_dialog(
                    payload={
                        "df_key": df_key,
                        "region": active_region,
                        "location": selected_loc,
                        "system": None,
                        "product": prod_name,
                        "scope_label": f"{selected_loc} / {prod_name}",
                    }
                )

            # Execute save (after confirm dialog triggers a rerun and overlay is visible).
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
                        metadata={
                            "region": str(payload.get("region") or active_region),
                            "location": payload.get("location"),
                            "system": payload.get("system"),
                            "product": payload.get("product"),
                            "rows_saved": int(n),
                        },
                    )
                    st.session_state["details_save_result"] = {"ok": True, "n": int(n), "df_key": df_key}
                except Exception as e:
                    log_error(
                        error_code="DETAILS_SAVE_FAILED",
                        error_message=str(e),
                        stack_trace=__import__("traceback").format_exc(),
                        service_module="UI",
                    )
                    log_audit(
                        event="details_save_failed",
                        metadata={
                            "region": str(payload.get("region") or active_region),
                            "location": payload.get("location"),
                            "system": payload.get("system"),
                            "product": payload.get("product"),
                            "error": str(e),
                        },
                    )
                    st.session_state["details_save_result"] = {"ok": False, "error": str(e), "df_key": df_key}

                st.session_state[enable_ver_key] = int(st.session_state.get(enable_ver_key, 0)) + 1
                st.session_state.pop(enable_widget_key, None)

                st.session_state["details_save_stage"] = "result"
                st.rerun()

            result = st.session_state.get("details_save_result")
            if st.session_state.get("details_save_stage") == "result" and isinstance(result, dict) and result.get("df_key") == df_key:
                _save_result_dialog(result=result)


def display_details_tab(
    df_filtered: pd.DataFrame,
    active_region: str,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    selected_loc: str | None = None,
):
    display_location_details(
        df_filtered,
        active_region,
        start_ts=start_ts,
        end_ts=end_ts,
        selected_loc=selected_loc,
    )


def render_details_filters(*, regions: list[str], active_region: str | None) -> dict:
    from data_loader import load_region_filter_metadata
    from admin_config import get_default_date_window

    def _un_collapse_expandables() -> None:
        st.session_state["collapse_expandables"] = False

    region = active_region
    if not region:
        return {
            "active_region": None,
            "start_ts": pd.Timestamp.today().normalize(),
            "end_ts": pd.Timestamp.today().normalize(),
            "selected_loc": None,
            "loc_col": "Location",
            "locations": [],
        }

    loc_col = "Location"
    region_norm = str(region or "").strip().lower()
    filter_label = "System" if region_norm == "midcon" else "Location"

    meta = load_region_filter_metadata(region=region, loc_col=loc_col)
    locations = meta.get("locations", [])

    c1, c2, c3 = st.columns([2.2, 2.8, 1.2])
    with c1:
        if not locations:
            st.warning("No locations available")
            selected_loc = None
        else:
            # Persist selection per region
            key_loc = f"details_selected_loc|{region}"
            current = st.session_state.get(key_loc)
            index = locations.index(current) if current in locations else 0
            selected_loc = st.selectbox(
                filter_label,
                options=locations,
                index=index,
                key=key_loc,
                on_change=_un_collapse_expandables,
            )

    today = pd.Timestamp.today().date()
    start_off, end_off = get_default_date_window(region=region, location=(str(selected_loc) if selected_loc else None))
    default_start = today + timedelta(days=int(start_off))
    default_end = today + timedelta(days=int(end_off))

    with c2:
        # Allow selection wider than defaults, within dataset bounds.
        df_min = meta.get("min_date", pd.NaT)
        df_max = meta.get("max_date", pd.NaT)
        df_min_d = pd.to_datetime(df_min, errors="coerce").date() if pd.notna(df_min) else default_start
        df_max_d = pd.to_datetime(df_max, errors="coerce").date() if pd.notna(df_max) else default_end
        min_value = min(df_min_d, default_start)
        max_value = max(df_max_d, default_end)

        key_dates = f"details_date|{region}|{str(selected_loc) if selected_loc else 'all'}"
        date_range = st.date_input(
            "Date Range",
            value=(default_start, default_end),
            min_value=min_value,
            max_value=max_value,
            key=key_dates,
            on_change=_un_collapse_expandables,
        )

        if isinstance(date_range, (list, tuple)):
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = end_date = date_range[0] if date_range else today
        else:
            start_date = end_date = date_range

    with c3:
        st.markdown('<div class="btn-spacer"></div>', unsafe_allow_html=True)
        submitted = logged_button(
            "Submit",
            type="primary",
            event="details_filters_submit",
            metadata={"region": region, "selected_loc": selected_loc},
        )

    if bool(submitted):
        # Reset FACT toggle(s)
        for k in list(st.session_state.keys()):
            if str(k).startswith("details_show_fact|"):
                st.session_state[k] = False

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    return {
        "active_region": region,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "selected_loc": selected_loc,
        "loc_col": loc_col,
        "locations": locations,
        "submitted": bool(submitted),
    }
