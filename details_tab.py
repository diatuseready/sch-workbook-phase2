import streamlit as st
import pandas as pd
import numpy as np
import html
import time
from datetime import timedelta

from admin_config import get_visible_columns, get_threshold_overrides, get_rack_lifting_forecast_method
from utils import dynamic_input_data_editor
from data_loader import persist_details_rows
from data_loader import generate_snowflake_signed_urls
from app_logging import logged_button, log_audit, log_error
from config import (
    COL_ADJUSTMENTS,
    COL_ADJUSTMENTS_FACT,
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
    COL_SOURCE,
    COL_TRANSFERS,
    COL_TRANSFERS_FACT,
    COL_GAIN_LOSS,
    COL_GAIN_LOSS_FACT,
    COL_BATCH,
    COL_NOTES,
    DETAILS_RENAME_MAP,
    DATA_SOURCE,
)

DETAILS_RENAME = DETAILS_RENAME_MAP

DETAILS_COLS = [
    COL_SOURCE,
    COL_PRODUCT,
    COL_OPENING_INV,
    COL_CLOSE_INV_RAW,
    COL_BATCH_IN,
    COL_BATCH_OUT,
    COL_RACK_LIFTING,
    COL_PIPELINE_IN,
    COL_PIPELINE_OUT,
    COL_GAIN_LOSS,
    COL_TRANSFERS,
    COL_PRODUCTION,
    COL_ADJUSTMENTS,
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

# Visual cue for read-only fact columns
FACT_BG = "#eeeeee"

LOCKED_BASE_COLS = [
    "Date",
    "{id_col}",
    "source",
    "Product",
    "Close Inv",
    "Opening Inv",
]

FACT_COL_MAP: dict[str, str] = {
    COL_OPENING_INV: COL_OPENING_INV_FACT,
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
) -> "pd.io.formats.style.Styler":

    cols = list(df.columns)
    cols_set = set(cols_to_color)
    fact_cols = {c for c in cols if str(c).endswith(" Fact")}

    ref = fact_reference if isinstance(fact_reference, pd.DataFrame) else None

    def _is_system_inv_discrepancy(row: pd.Series) -> bool:
        src = str(row.get("source", "")).strip().lower()
        if src != "system":
            return False

        def _get_fact(idx, fact_col: str):
            if fact_col in row.index:
                return row.get(fact_col)
            if ref is not None and fact_col in ref.columns and idx in ref.index:
                return ref.at[idx, fact_col]
            return None

        # We only consider Close Inv for discrepancy highlighting.
        base = "Close Inv"
        fact = "Close Inv Fact"
        if base in row.index:
            fact_val = _get_fact(row.name, fact)
            if fact_val is not None:
                if abs(_to_float(row.get(base)) - _to_float(fact_val)) > SYSTEM_DISCREPANCY_THRESHOLD_BBL:
                    return True
        return False

    def _row_style(row: pd.Series) -> list[str]:
        src = str(row.get("source", "")).strip().lower()

        bg = SOURCE_BG.get(src, "")
        if src == "system" and _is_system_inv_discrepancy(row):
            bg = SYSTEM_DISCREPANCY_BG
        base_style = f"background-color: {bg};" if bg else ""

        styles: list[str] = []
        for c in cols:
            # Fact columns: always grey (read-only indicator), regardless of source.
            if c in fact_cols:
                styles.append(f"background-color: {FACT_BG};")
                continue

            # Regular source-based coloring.
            styles.append(base_style if (c in cols_set and base_style) else "")
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

    # Optional: ensure fact columns exist so we can compare system vs fact values.
    # When rendering a UI *view* df (where fact cols might be intentionally omitted),
    # set ensure_fact_cols=False to avoid re-adding hidden columns.
    if ensure_fact_cols and "source" in out.columns:
        src = out["source"].astype(str).str.strip().str.lower()
        if "Opening Inv" in out.columns and "Opening Inv Fact" not in out.columns:
            out["Opening Inv Fact"] = np.where(src.eq("system"), out["Opening Inv"], np.nan)
        if "Close Inv" in out.columns and "Close Inv Fact" not in out.columns:
            out["Close Inv Fact"] = np.where(src.eq("system"), out["Close Inv"], np.nan)

    # Work with datetimes internally for stable sorting; convert back to date at end.
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    numeric_candidates = [
        "Opening Inv",
        "Opening Inv Fact",
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
    NUM_FMT = "accounting"

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
        "Batch": st.column_config.TextColumn("Batch"),
        "Notes": st.column_config.TextColumn("Notes"),
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
        if c in {"Date", id_col, "source", "Product"}:
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

    for c in _available_flow_cols(df):
        agg_map[c] = "sum"

    for base_col, fact_col in {**FACT_COL_MAP_RAW, **FACT_COL_MAP}.items():
        if fact_col not in df.columns:
            continue

        base_s = str(base_col)
        if base_s in {"Opening Inv", "Open Inv"}:
            agg_map[fact_col] = "first"
        elif base_s == "Close Inv":
            agg_map[fact_col] = "last"
        else:
            agg_map[fact_col] = "sum"

    if "source" in df.columns:
        agg_map["source"] = "first"
    if "updated" in df.columns:
        agg_map["updated"] = "max"
    if "Batch" in df.columns:
        agg_map["Batch"] = "last"
    if "Notes" in df.columns:
        agg_map["Notes"] = "last"

    # Keep file locations for the day (Snowflake-only; list column).
    if "FILE_LOCATION" in df.columns:
        agg_map["FILE_LOCATION"] = "last"

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


def _nonzero_mean(s: pd.Series) -> float:
    """Mean excluding zeros (and NaNs). Returns 0.0 if nothing remains."""
    if s is None:
        return 0.0
    vals = pd.to_numeric(s, errors="coerce").fillna(0.0)
    vals = vals[vals != 0]
    return float(vals.mean()) if len(vals) else 0.0


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


def _extend_with_30d_forecast(
    df: pd.DataFrame,
    *,
    id_col: str,
    region: str | None,
    location: str | None,
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
                "source": "forecast",
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
        if c in {"Date", id_col, "source", "Product", "Notes", "updated"}:
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
        "source",
        "Product",
        "updated",
        "Batch",
        "Notes",
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

    out = df_display.copy()
    # Ensure list column exists even in SQLite/forecast rows.
    if "FILE_LOCATION" not in out.columns:
        out["FILE_LOCATION"] = [[] for _ in range(len(out))]
    # UI action column: checkbox that behaves like a button.
    if COL_VIEW_FILE not in out.columns:
        out[COL_VIEW_FILE] = False
    else:
        out[COL_VIEW_FILE] = out[COL_VIEW_FILE].fillna(False).astype(bool)

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

    c_toggle, c_loc, _ = st.columns([3.5, 4.5, 2])
    with c_toggle:
        show_fact = st.toggle(
            "Show Terminal Feed",
            value=bool(st.session_state.get(f"details_show_fact|{active_region}|{selected_loc}|location", False)),
            key=f"details_show_fact|{active_region}|{selected_loc}|location",
            help="Show upstream system values next to the editable columns.",
        )
    with c_loc:
        st.markdown(f"**{str(selected_loc)}**")

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
            reset_enable_key = f"{enable_key}__reset_pending"

            if st.session_state.get(reset_enable_key):
                st.session_state.pop(enable_key, None)
                st.session_state[reset_enable_key] = False

            with c_enable:
                enable_save = st.toggle(
                    "Enable Save",
                    value=bool(st.session_state.get(enable_key, False)),
                    key=enable_key,
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

            # Forecast should be bounded by the user-selected date range.
            df_all = _extend_with_30d_forecast(
                df_prod,
                id_col="Location",
                region=active_region,
                location=str(selected_loc),
                forecast_end=end_ts,
            )
            df_display, cols = build_details_view(df_all, id_col="Location")

            visible = get_visible_columns(region=active_region, location=str(selected_loc))

            # Location + Product are already known from the current selection and the tab,
            # so we don't show them in the grid.
            must_have = ["Date", "Opening Inv", "Close Inv"]
            column_order = []
            for c in must_have + visible:
                if c in {"Location", "Product"}:
                    continue
                if c in cols and c not in column_order and c != "source":
                    column_order.append(c)

            column_order = _ensure_cols_after(
                column_order,
                required=["Production", "Adjustments"],
                after="Transfers",
                before="Notes",
            )

            # UI action column (Snowflake-only).
            column_order = _ensure_cols_after(
                column_order,
                required=[COL_VIEW_FILE],
                after="Close Inv",
                before=None,
            )

            # Keep Batch immediately after View File.
            column_order = _ensure_cols_after(
                column_order,
                required=["Batch"],
                after=COL_VIEW_FILE,
                before="Notes",
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
                st.session_state[df_key] = _recalculate_open_close_inv(editor_df, id_col="Location", ensure_fact_cols=True)

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
            recomputed_view = _recalculate_open_close_inv(
                edited,
                id_col="Location",
                ensure_fact_cols=False,
            ).reset_index(drop=True)

            # Merge the edited view back into the canonical df so edits persist across toggle.
            canonical = st.session_state[df_key].reset_index(drop=True)
            if canonical.shape[0] != recomputed_view.shape[0]:
                # Safety fallback: rebuild canonical if something changed the rowset.
                canonical = _recalculate_open_close_inv(editor_df, id_col="Location", ensure_fact_cols=True).reset_index(drop=True)
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
                # Reset the enable toggle on the next rerun (can't touch widget state in this run).
                st.session_state[reset_enable_key] = True
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
