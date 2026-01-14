import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

from admin_config import get_visible_columns, get_threshold_overrides
from utils import dynamic_input_data_editor
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
    COL_NOTES,
    DETAILS_RENAME_MAP,
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
    COL_ADJUSTMENTS,  # New addition for Magellan
    COL_GAIN_LOSS,
    COL_TRANSFERS,
    COL_NOTES,
]

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
    "Production": COL_PRODUCTION_FACT,
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
    "Production": COL_PRODUCTION_FACT,
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
    fact_cols = {c for c in cols if str(c).endswith(" Fact")}

    def _row_style(row: pd.Series) -> list[str]:
        bg = SOURCE_BG.get(str(row.get("source", "")).strip().lower(), "")
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


def _recalculate_open_close_inv(df: pd.DataFrame, *, id_col: str) -> pd.DataFrame:
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

        # Extract system for Magellan detection
        system_val = g[id_col].iloc[0] if id_col in g.columns else None
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
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                cfg[c] = st.column_config.NumberColumn(c, disabled=True, format="%.2f")
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

    forecast_flow_cols = [c for c in [COL_RACK_LIFTINGS_RAW] if c in flow_cols]

    if forecast_end is not None:
        forecast_end = pd.Timestamp(forecast_end)

    forecast_rows: list[dict] = []

    for (id_val, product), group in daily.groupby([id_col, "Product"], dropna=False):
        group = group.sort_values("Date")
        last_date = pd.Timestamp(group["Date"].max())

        prev_close = _last_close_inv(group)
        for d in _forecast_dates(last_date, forecast_end, default_days):
            # Only estimate Rack/Liftings; keep everything else 0.
            flows = {c: 0.0 for c in flow_cols}
            if forecast_flow_cols:
                flows.update(estimate_forecast_flows(group, flow_cols=forecast_flow_cols, d=d))

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


def _show_thresholds(*, region_label: str, bottom: float | None, safefill: float | None, note: str | None = None):
    c0, c1, c2, c3 = st.columns([5, 2, 2, 3])

    with c0:
        st.markdown("")
        pass

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

    with c3:
        v = "—" if note in (None, "") else str(note)
        st.markdown(
            f"""
            <div class="mini-card">
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
        "Batch In",
        "Batch Out",
        "Pipeline In",
        "Pipeline Out",
        "Gain/Loss",
        "Transfers",
        "Rack/Lifting",
    ]
    # Always keep columns required for grouping + lineage.
    base = ["Date", id_col, "source", "Product", "updated", "Notes", "Opening Inv", "Close Inv"]
    desired = []
    for c in base + ui_cols + extra:
        if c in df_display.columns and c not in desired:
            desired.append(c)
    return df_display[desired].reset_index(drop=True)


def display_midcon_details(
    df_filtered: pd.DataFrame,
    active_region: str,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
):
    st.subheader("🧾 Group Daily Details")

    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return

    df_all = _extend_with_30d_forecast(df_filtered, id_col="System", forecast_end=end_ts)

    df_display, cols = build_details_view(df_all, id_col="System")

    scope_sys = None
    if df_filtered is not None and not df_filtered.empty and "System" in df_filtered.columns:
        systems = sorted(df_filtered["System"].dropna().unique().tolist())
        if len(systems) == 1:
            scope_sys = systems[0]

    # In Midcon view there isn't a single Product in scope (grid spans products),
    # so we show location-level thresholds.
    bottom, safefill, note = _threshold_values(
        region=active_region,
        location=str(scope_sys) if scope_sys is not None else None,
        product=None,
    )
    _show_thresholds(region_label=active_region, bottom=bottom, safefill=safefill, note=note)

    show_fact = st.toggle(
        "Show Fact Columns",
        value=bool(st.session_state.get(f"details_show_fact|{active_region}|{scope_sys or ''}|midcon", False)),
        key=f"details_show_fact|{active_region}|{scope_sys or ''}|midcon",
        help="Show upstream FACT_* values next to the editable columns.",
    )

    visible = get_visible_columns(region=active_region, location=str(scope_sys) if scope_sys is not None else None)
    must_have = ["Date", "System", "Product", "Opening Inv", "Close Inv"]
    column_order = []
    for c in must_have + visible:
        if c in cols and c not in column_order and c != "source":
            column_order.append(c)

    column_order = _insert_fact_columns(column_order, df_cols=list(df_display.columns), show_fact=show_fact)

    # Include fact columns for coloring if the base column is colored.
    locked_cols = _locked_cols("System", cols)
    if show_fact:
        for base in list(locked_cols):
            fact = FACT_COL_MAP.get(base)
            if fact and fact in df_display.columns and fact not in locked_cols:
                locked_cols.append(fact)

    column_config = _column_config(df_display, column_order, "System")
    column_config = {k: v for k, v in column_config.items() if k in column_order}

    base_key = (
        f"{active_region}|{scope_sys or ''}|{pd.Timestamp(start_ts).date()}|{pd.Timestamp(end_ts).date()}"
        f"|fact={int(bool(show_fact))}_edit"
    )
    df_key = f"{base_key}__df"
    ver_key = f"{base_key}__ver"
    widget_key = f"{base_key}__editor_v{int(st.session_state.get(ver_key, 0))}"

    editor_df = _build_editor_df(df_display, id_col="System", ui_cols=column_order)

    if df_key not in st.session_state or list(st.session_state[df_key].columns) != list(editor_df.columns):
        st.session_state[df_key] = _recalculate_open_close_inv(editor_df, id_col="System")

    st.session_state[df_key] = st.session_state[df_key].reset_index(drop=True)

    styled = _style_source_cells(st.session_state[df_key], locked_cols)

    edited = dynamic_input_data_editor(
        styled,
        width="stretch",
        height=400,
        hide_index=True,
        column_order=column_order,
        key=widget_key,
        column_config=column_config,
    )

    recomputed = _recalculate_open_close_inv(edited, id_col="System").reset_index(drop=True)
    st.session_state[df_key] = recomputed
    if _needs_inventory_rerun(edited, recomputed):
        st.session_state[ver_key] = int(st.session_state.get(ver_key, 0)) + 1
        st.rerun()

    st.markdown('<div class="save-btn-bottom">', unsafe_allow_html=True)
    if st.button("💾 Save Changes", key=f"save_{active_region}"):
        st.success("✅ Changes saved successfully!")
    st.markdown("</div>", unsafe_allow_html=True)


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

    # st.caption(f"Location: {selected_loc}")

    show_fact = st.toggle(
        "Show Fact Columns",
        value=bool(st.session_state.get(f"details_show_fact|{active_region}|{selected_loc}|location", False)),
        key=f"details_show_fact|{active_region}|{selected_loc}|location",
        help="Show upstream FACT_* values next to the editable columns.",
    )

    for i, tab in enumerate(st.tabs(products)):
        prod_name = products[i]
        with tab:
            df_prod = df_loc[df_loc["Product"].astype(str) == str(prod_name)]

            # Forecast should be bounded by the user-selected date range.
            df_all = _extend_with_30d_forecast(df_prod, id_col="Location", forecast_end=end_ts)
            df_display, cols = build_details_view(df_all, id_col="Location")

            bottom, safefill, note = _threshold_values(
                region=active_region,
                location=str(selected_loc),
                product=str(prod_name),
            )
            _show_thresholds(region_label=str(selected_loc), bottom=bottom, safefill=safefill, note=note)

            visible = get_visible_columns(region=active_region, location=str(selected_loc))
            must_have = ["Date", "Location", "Product", "Opening Inv", "Close Inv"]
            column_order = []
            for c in must_have + visible:
                if c in cols and c not in column_order and c != "source":
                    column_order.append(c)

            column_order = _insert_fact_columns(column_order, df_cols=list(df_display.columns), show_fact=show_fact)

            locked_cols = _locked_cols("Location", cols)
            if show_fact:
                for base in list(locked_cols):
                    fact = FACT_COL_MAP.get(base)
                    if fact and fact in df_display.columns and fact not in locked_cols:
                        locked_cols.append(fact)

            column_config = _column_config(df_display, column_order, "Location")
            column_config = {k: v for k, v in column_config.items() if k in column_order}

            # Include filters in the key so changing sidebar date range refreshes the editor.
            base_key = (
                f"{active_region}_{selected_loc}_{prod_name}"
                f"|{pd.Timestamp(start_ts).date()}|{pd.Timestamp(end_ts).date()}"
                f"|fact={int(bool(show_fact))}_edit"
            )
            df_key = f"{base_key}__df"
            ver_key = f"{base_key}__ver"
            widget_key = f"{base_key}__editor_v{int(st.session_state.get(ver_key, 0))}"

            editor_df = _build_editor_df(df_display, id_col="Location", ui_cols=column_order)
            if df_key not in st.session_state or list(st.session_state[df_key].columns) != list(editor_df.columns):
                st.session_state[df_key] = _recalculate_open_close_inv(editor_df, id_col="Location")

            st.session_state[df_key] = st.session_state[df_key].reset_index(drop=True)

            styled = _style_source_cells(st.session_state[df_key], locked_cols)

            edited = dynamic_input_data_editor(
                styled,
                num_rows="dynamic",
                width="stretch",
                height=DETAILS_EDITOR_HEIGHT_PX,
                hide_index=True,
                column_order=column_order,
                key=widget_key,
                column_config=column_config,
            )

            recomputed = _recalculate_open_close_inv(edited, id_col="Location").reset_index(drop=True)
            st.session_state[df_key] = recomputed
            if _needs_inventory_rerun(edited, recomputed):
                st.session_state[ver_key] = int(st.session_state.get(ver_key, 0)) + 1
                st.rerun()

            st.markdown('<div class="save-btn-bottom">', unsafe_allow_html=True)
            if st.button(f"💾 Save {selected_loc} / {prod_name}", key=f"save_{active_region}_{selected_loc}_{prod_name}"):
                st.success(f"✅ Changes for {selected_loc} / {prod_name} saved successfully!")
            st.markdown("</div>", unsafe_allow_html=True)


def display_details_tab(
    df_filtered: pd.DataFrame,
    active_region: str,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    selected_loc: str | None = None,
):
    if active_region == "Midcon":
        display_midcon_details(df_filtered, active_region, start_ts=start_ts, end_ts=end_ts)
    else:
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

    loc_col = "System" if region == "Midcon" else "Location"
    filter_label = "🏭 System" if loc_col == "System" else "Location"

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
            selected_loc = st.selectbox(filter_label, options=locations, index=index, key=key_loc)

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
        submitted = st.button("Submit", type="primary")

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
