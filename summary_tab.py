import streamlit as st
import pandas as pd
import numpy as np
import datetime
from typing import Any, cast

from admin_config import get_default_date_window, get_threshold_overrides
from config import (
    REQUIRED_MAX_DEFAULTS,
    REQUIRED_MIN_DEFAULTS,
    INTRANSIT_DEFAULTS,
    GLOBAL_REQUIRED_MAX_FALLBACK,
    GLOBAL_REQUIRED_MIN_FALLBACK,
    GLOBAL_INTRANSIT_FALLBACK,
)

from config import (
    COL_AVAILABLE,
    COL_AVAILABLE_SPACE,
    COL_ADJUSTMENTS,
    COL_AURORA_PIPELINE_IN,
    COL_BATCH_IN_RAW,
    COL_BATCH_OUT_RAW,
    COL_CLOSE_INV_RAW,
    COL_DATE,
    COL_DUPONT_PIPELINE_IN,
    COL_EL_DORADO,
    COL_FROM_327_RECEIPT,
    COL_GAIN_LOSS,
    COL_INTRANSIT,
    COL_MED_BOW_PIPELINE_IN,
    COL_MEDICINE_PIPELINE_OUT,
    COL_OPEN_INV_RAW,
    COL_OFFLINE,
    COL_OTHER,
    COL_PIPELINE_IN,
    COL_PIPELINE_OUT,
    COL_PIONEER_PIPELINE_OUT,
    COL_PTO,
    COL_PRODUCT,
    COL_PRODUCTION,
    COL_RACK_LIFTINGS_RAW,
    COL_RECON_FROM_191,
    COL_RECON_TO_182,
    COL_RMPL_PIPELINE_OUT,
    COL_SAFE_FILL_LIMIT,
    COL_SEMINOE_PIPELINE_OUT,
    COL_SYSTEM,
    COL_LOCATION,
    COL_TANK_CAPACITY,
    COL_TRANSFER_IN,
    COL_TRANSFER_OUT,
    COL_TRANSFERS,
    COL_TULSA,
    COL_VESSEL_VOLUME,
    COL_REGION,
)


def _normalize_region(active_region: str) -> str:
    return "Midcon" if active_region == "Group Supply Report (Midcon)" else active_region


def _is_midcon(active_region: str) -> bool:
    return _normalize_region(active_region) == "Midcon"


def calculate_required_max(row, group_cols, df_filtered):
    region = str(row.get(COL_REGION) or "Unknown")
    loc_or_sys = row.get(group_cols[0])
    prod = str(row.get(COL_PRODUCT) or "")
    overrides = get_threshold_overrides(
        region=_normalize_region(region),
        location=str(loc_or_sys) if pd.notna(loc_or_sys) else None,
        product=prod or None,
    )

    safefill = overrides.get("SAFEFILL")
    if safefill is not None and not pd.isna(safefill):
        return float(safefill)

    key = f"{loc_or_sys}|{prod}"
    if key in REQUIRED_MAX_DEFAULTS:
        return float(REQUIRED_MAX_DEFAULTS[key])
    if prod in REQUIRED_MAX_DEFAULTS:
        return float(REQUIRED_MAX_DEFAULTS[prod])

    if COL_SAFE_FILL_LIMIT in df_filtered.columns and group_cols[0] in df_filtered.columns:
        safe_fill = df_filtered[
            (df_filtered[group_cols[0]] == row[group_cols[0]]) &
            (df_filtered[COL_PRODUCT] == row[COL_PRODUCT])
        ][COL_SAFE_FILL_LIMIT].max()
        if pd.notna(safe_fill) and safe_fill > 0:
            return float(safe_fill)

    return float(GLOBAL_REQUIRED_MAX_FALLBACK)


def calculate_intransit(row, group_cols, df_filtered):
    """Calculate intransit based on pipeline data or defaults."""
    if group_cols[0] == COL_SYSTEM:
        key = f"{row[COL_SYSTEM]}|{row[COL_PRODUCT]}"
        prod_key = row[COL_PRODUCT]
    else:
        key = f"{row[COL_LOCATION]}|{row[COL_PRODUCT]}"
        prod_key = row[COL_PRODUCT]

    pipeline_val = row.get(COL_PIPELINE_IN, 0)
    if pd.notna(pipeline_val) and pipeline_val > 0:
        return float(pipeline_val)

    if key in INTRANSIT_DEFAULTS:
        return INTRANSIT_DEFAULTS[key]
    if prod_key in INTRANSIT_DEFAULTS:
        return INTRANSIT_DEFAULTS[prod_key]

    return GLOBAL_INTRANSIT_FALLBACK


def calculate_required_min(row, group_cols, df_filtered):
    region = str(row.get(COL_REGION) or "Unknown")
    loc_or_sys = row.get(group_cols[0])
    prod = str(row.get(COL_PRODUCT) or "")
    overrides = get_threshold_overrides(
        region=_normalize_region(region),
        location=str(loc_or_sys) if pd.notna(loc_or_sys) else None,
        product=prod or None,
    )

    bottom = overrides.get("BOTTOM")
    if bottom is not None and not pd.isna(bottom):
        return float(bottom)

    key = f"{loc_or_sys}|{prod}"
    if key in REQUIRED_MIN_DEFAULTS:
        return float(REQUIRED_MIN_DEFAULTS[key])
    if prod in REQUIRED_MIN_DEFAULTS:
        return float(REQUIRED_MIN_DEFAULTS[prod])

    return float(GLOBAL_REQUIRED_MIN_FALLBACK)


def _us_number_column_config(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    """Return Streamlit column_config for numeric columns with US thousands separators."""
    cfg: dict[str, Any] = {}
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            # "accounting" yields comma-separated values like 1,226,275.00
            cfg[c] = st.column_config.NumberColumn(c, format="accounting")
    return cfg


def _as_float(value: object) -> float:
    try:
        out = float(cast(Any, value))
        return out if not pd.isna(out) else 0.0
    except (TypeError, ValueError):
        return 0.0


def region_details_basis_cache_key(
    active_region: str,
    *,
    as_of_ts: pd.Timestamp | None = None,
) -> str:
    as_of = (as_of_ts or pd.Timestamp.today().normalize()).strftime("%Y-%m-%d")
    return f"region_details_basis|{active_region}|{as_of}"


def build_region_details_basis(
    df_filtered: pd.DataFrame,
    active_region: str,
) -> pd.DataFrame:
    """Build the exact Details-tab displayed Close Inv rows for a region.

    This does not derive a separate Summary-only basis. It runs the same
    location/product pipeline that Details uses to build its final displayed
    dataframe, then keeps the Date/Location/Product/Close Inv rows needed by
    Summary and Midcon consumers.
    """
    if df_filtered is None or df_filtered.empty:
        return pd.DataFrame(columns=[COL_DATE, COL_LOCATION, COL_PRODUCT, COL_CLOSE_INV_RAW])

    from details_tab import (
        _build_editor_df,
        _extend_with_30d_forecast,
        _recalculate_inventory_metrics,
        build_details_view,
    )

    out = df_filtered.copy()
    if not all(col in out.columns for col in [COL_LOCATION, COL_PRODUCT, COL_DATE]):
        return pd.DataFrame(columns=[COL_DATE, COL_LOCATION, COL_PRODUCT, COL_CLOSE_INV_RAW])

    out[COL_DATE] = pd.to_datetime(out[COL_DATE], errors="coerce")
    out = out.dropna(subset=[COL_DATE])
    if out.empty:
        return pd.DataFrame(columns=[COL_DATE, COL_LOCATION, COL_PRODUCT, COL_CLOSE_INV_RAW])

    today_ts = pd.Timestamp.today().normalize()
    result_parts: list[pd.DataFrame] = []

    locations = sorted(
        out[COL_LOCATION].dropna().astype(str).str.strip().unique().tolist()
    )
    for location in locations:
        loc_df = out[out[COL_LOCATION].astype(str).str.strip() == location].copy()
        if loc_df.empty:
            continue

        start_off, end_off = get_default_date_window(
            region=str(active_region or "Unknown"),
            location=location,
        )
        start_ts = today_ts + pd.Timedelta(days=int(start_off))
        end_ts = today_ts + pd.Timedelta(days=int(end_off))

        products = sorted(
            loc_df[COL_PRODUCT].dropna().astype(str).str.strip().unique().tolist()
        )
        for product in products:
            df_prod = loc_df[
                loc_df[COL_PRODUCT].astype(str).str.strip() == product
            ].copy()
            if df_prod.empty:
                continue

            df_all = _extend_with_30d_forecast(
                df_prod,
                id_col=COL_LOCATION,
                region=active_region,
                location=location,
                history_start=start_ts,
                forecast_end=end_ts,
            )
            df_display = build_details_view(df_all, id_col=COL_LOCATION)
            df_display = df_display[
                df_display[COL_DATE] >= pd.Timestamp(start_ts).normalize().date()
            ].reset_index(drop=True)
            if df_display.empty:
                continue

            overrides = get_threshold_overrides(
                region=str(active_region or "Unknown"),
                location=location,
                product=product,
            )
            bottom = overrides.get("BOTTOM")
            safefill = overrides.get("SAFEFILL")
            bottom_val = float(bottom) if bottom is not None and not pd.isna(bottom) else None
            safefill_val = float(safefill) if safefill is not None and not pd.isna(safefill) else None

            final_df = _recalculate_inventory_metrics(
                _build_editor_df(df_display),
                id_col=COL_LOCATION,
                safefill=safefill_val,
                bottom=bottom_val,
                df_hist=df_prod,
            )
            keep_cols = [
                c for c in [COL_DATE, COL_LOCATION, COL_PRODUCT, COL_CLOSE_INV_RAW]
                if c in final_df.columns
            ]
            if keep_cols:
                result_parts.append(final_df[keep_cols].copy())

    if not result_parts:
        return pd.DataFrame(columns=[COL_DATE, COL_LOCATION, COL_PRODUCT, COL_CLOSE_INV_RAW])

    basis = pd.concat(result_parts, ignore_index=True)
    basis[COL_DATE] = pd.to_datetime(basis[COL_DATE], errors="coerce")
    return basis


def build_close_lookup_from_basis(
    daily_basis: pd.DataFrame,
    *,
    target_ts: pd.Timestamp,
) -> dict[tuple[str, str], float]:
    """Return (Location, Product) -> Close Inv on target date or latest prior date."""
    if daily_basis is None or daily_basis.empty:
        return {}

    basis = daily_basis.copy()
    basis[COL_DATE] = pd.to_datetime(basis[COL_DATE], errors="coerce")
    basis = basis.dropna(subset=[COL_DATE])
    if basis.empty:
        return {}

    group_cols = [COL_LOCATION, COL_PRODUCT]
    exact = basis[basis[COL_DATE].dt.normalize() == target_ts]
    if exact.empty:
        exact = (
            basis[basis[COL_DATE] <= target_ts]
            .sort_values(COL_DATE, kind="mergesort")
            .groupby(group_cols, as_index=False, dropna=False)
            .last()
        )

    return {
        (str(row[COL_LOCATION]), str(row[COL_PRODUCT])): _as_float(row.get(COL_CLOSE_INV_RAW))
        for _, row in exact.iterrows()
    }


def build_close_lookup_from_open_details_session(
    active_region: str,
    *,
    target_ts: pd.Timestamp,
) -> dict[tuple[str, str], float]:
    """Read Close Inv directly from live Details-tab session-state dataframes.

    When the user has already opened a product in Details, the canonical df in
    session state is the exact dataframe being displayed, including unsaved live
    edits. Summary should prefer that source when available.
    """
    lookup: dict[tuple[str, str], float] = {}
    prefix = f"{active_region}_"

    for key, value in st.session_state.items():
        if not isinstance(key, str) or not key.endswith("|edit__df"):
            continue
        if not key.startswith(prefix):
            continue
        if not isinstance(value, pd.DataFrame):
            continue
        if not all(col in value.columns for col in [COL_DATE, COL_LOCATION, COL_PRODUCT, COL_CLOSE_INV_RAW]):
            continue

        df_live = value.copy()
        df_live[COL_DATE] = pd.to_datetime(df_live[COL_DATE], errors="coerce")
        df_live = df_live.dropna(subset=[COL_DATE])
        if df_live.empty:
            continue

        exact = df_live[df_live[COL_DATE].dt.normalize() == target_ts]
        chosen = exact
        if chosen.empty:
            chosen = df_live[df_live[COL_DATE] <= target_ts].sort_values(COL_DATE).tail(1)
        if chosen.empty:
            continue

        row = chosen.iloc[-1]
        lookup[(str(row[COL_LOCATION]), str(row[COL_PRODUCT]))] = _as_float(row.get(COL_CLOSE_INV_RAW))

    return lookup


def _dynamic_height(n_rows: int, *, row_px: int = 35, header_px: int = 38, min_px: int = 80, max_px: int = 600) -> int:
    """Return a pixel height that fits *n_rows* without excess blank space."""
    return max(min_px, min(max_px, header_px + n_rows * row_px))


def display_regional_summary(df_filtered, active_region):
    """Display the regional summary section."""
    st.subheader("Summary")

    if df_filtered.empty:
        st.info("No data available for the selected region and filters.")
        return

    sales_cols = [c for c in [COL_RACK_LIFTINGS_RAW, COL_BATCH_OUT_RAW] if c in df_filtered.columns]
    region_name = _normalize_region(active_region)

    # Always group by Location + Product
    group_cols = [COL_LOCATION, COL_PRODUCT]

    if not all(col in df_filtered.columns for col in group_cols):
        st.warning("Required columns not found in the data.")
        return

    # ── Exclude forecast rows so only real (system/user) data feeds into the
    # current-state Summary columns (Gross Inventory, etc.).  Forecast rows
    # are projections and must not be treated as actual inventory.
    if "SOURCE_TYPE" in df_filtered.columns:
        df_filtered = df_filtered[
            ~df_filtered["SOURCE_TYPE"].astype(str).str.strip().str.lower().isin(
                {"forecast", "forecast_user"}
            )
        ].copy()

    today_ts = pd.Timestamp.today().normalize()
    # Prior day = today − 1: matches the "yesterday" reference used in Details tab
    prior_day_ts = today_ts - pd.Timedelta(days=1)
    prior_day_date = prior_day_ts.date()

    # ── Cache the expensive Details-basis computation (runs once per region/day;
    #    subsequent reruns triggered by filter changes skip it entirely).
    _basis_cache_key = region_details_basis_cache_key(active_region, as_of_ts=today_ts)
    if _basis_cache_key not in st.session_state:
        with st.spinner("Computing inventory basis\u2026"):
            details_basis = build_region_details_basis(df_filtered, active_region)
        st.session_state[_basis_cache_key] = details_basis
    else:
        details_basis = st.session_state[_basis_cache_key]
    _gi_lookup = build_close_lookup_from_basis(details_basis, target_ts=prior_day_ts)
    _gi_lookup.update(
        build_close_lookup_from_open_details_session(
            active_region,
            target_ts=prior_day_ts,
        )
    )

    # Build aggregation map; include Intransit and Available when present
    _agg: dict = {
        COL_CLOSE_INV_RAW: "last",
        COL_OPEN_INV_RAW: "first",
        COL_BATCH_IN_RAW: "sum",
        COL_BATCH_OUT_RAW: "sum",
        COL_RACK_LIFTINGS_RAW: "sum",
        COL_PRODUCTION: "sum",
        COL_PIPELINE_IN: "sum",
        COL_PIPELINE_OUT: "sum",
        COL_TANK_CAPACITY: "max",
        COL_SAFE_FILL_LIMIT: "max",
        COL_AVAILABLE_SPACE: "mean",
    }
    if COL_INTRANSIT in df_filtered.columns:
        _agg[COL_INTRANSIT] = "last"
    if COL_AVAILABLE in df_filtered.columns:
        _agg[COL_AVAILABLE] = "last"

    daily = (
        df_filtered
        .groupby(group_cols + [COL_DATE], as_index=False)
        .agg(_agg)
    )
    daily[COL_DATE] = pd.to_datetime(daily[COL_DATE], errors="coerce")

    if daily.empty:
        st.info("No data available for the selected filters.")
        return

    # Sales column = Rack/Lifting + Deliveries (for Prior Day Sales)
    daily["Sales"] = daily[sales_cols].sum(axis=1) if sales_cols else 0

    # ── Cap to actual rows only (≤ today − 1) so forecast rows don't corrupt
    # the current inventory snapshot.  Details tab uses the prior day as its
    # reference date for Total Inventory / Close Inv display.
    daily_actual = daily[daily[COL_DATE] <= prior_day_ts].copy()
    if daily_actual.empty:
        # No actual data yet — fall back to all rows so something is shown
        daily_actual = daily.copy()

    # Latest ACTUAL date per pair (= the most recent day with real data)
    latest_actual_date = daily_actual[COL_DATE].max()

    # ── 7 Day Average: Rack/Lifting only (same metric as Details tab "7 Day Avg")
    # Details tab _fill_rack_averages_per_row uses Rack/Liftings, not total Sales.
    _rack_col = COL_RACK_LIFTINGS_RAW if COL_RACK_LIFTINGS_RAW in daily_actual.columns else None

    def compute_7day_avg(g):
        g = g.sort_values(COL_DATE)
        # Use only actual (≤ today-1) rows and take the trailing 7 days
        g_actual = g[g[COL_DATE] <= prior_day_ts]
        if _rack_col and not g_actual.empty:
            window = g_actual.tail(7)
            val = window[_rack_col].mean() if not window.empty else 0.0
        else:
            val = 0.0
        return pd.Series({"Seven_Day_Avg_Sales": val})

    seven_day = (
        daily.groupby(group_cols)
        .apply(compute_7day_avg, include_groups=False)
        .reset_index()
    )

    # ── "Current" inventory = most recent ACTUAL row per pair (≤ today − 1)
    # This matches Details tab: Close Inv shown for yesterday, never a future row.
    latest = (
        daily_actual
        .sort_values([COL_DATE])
        .groupby(group_cols, as_index=False)
        .last()
    )

    # ── Prior Day Sales = yesterday's (today − 1) Rack + Deliveries
    pds_rows = daily_actual[daily_actual[COL_DATE].dt.date == prior_day_date]
    if not pds_rows.empty:
        pds = (
            pds_rows.groupby(group_cols, as_index=False)["Sales"].sum()
            .rename(columns={"Sales": "Prior_Day_Sales"})
        )
    else:
        # Fallback to most recent available day
        if not daily_actual.empty:
            last_avail = daily_actual[COL_DATE].max()
            pds = (
                daily_actual[daily_actual[COL_DATE] == last_avail]
                .groupby(group_cols, as_index=False)["Sales"].sum()
                .rename(columns={"Sales": "Prior_Day_Sales"})
            )
        else:
            pds = latest[group_cols].copy()
            pds["Prior_Day_Sales"] = 0

    # Build summary DataFrame — include Intransit if present
    _merge_cols = group_cols + [COL_CLOSE_INV_RAW, COL_PIPELINE_IN]
    if COL_INTRANSIT in latest.columns:
        _merge_cols.append(COL_INTRANSIT)

    summary_df = (
        latest[_merge_cols]
        .merge(pds, on=group_cols, how="left")
        .merge(seven_day, on=group_cols, how="left")
    )

    summary_df["Region"] = region_name

    summary_df["SafeFill"] = summary_df.apply(
        lambda row: calculate_required_max(row, group_cols, df_filtered),
        axis=1,
    ).astype(float)

    summary_df["Bottom"] = summary_df.apply(
        lambda row: calculate_required_min(row, group_cols, df_filtered),
        axis=1,
    ).astype(float)

    # In-Transit: prefer the real INTRANSIT_BBL column from the DB;
    # fall back to Pipeline In proxy (same fallback as calculate_intransit)
    def _in_transit(row) -> float:
        if COL_INTRANSIT in row.index:
            v = row.get(COL_INTRANSIT)
            if v is not None and pd.notna(v) and float(v) != 0.0:
                return float(v)
        return calculate_intransit(row, group_cols, df_filtered)

    summary_df["In-Transit"] = summary_df.apply(_in_transit, axis=1).astype(float)

    # Gross Inventory = Details-aligned Close Inv for today − 1.
    # This uses the same daily aggregation + gap-fill + roll-forward logic as
    # the Details tab instead of relying on the raw region-level DB row order.
    summary_df["Gross Inventory"] = summary_df.apply(
        lambda r: _gi_lookup.get(
            (str(r.get("Location", "")), str(r.get("Product", ""))),
            float(r.get(COL_CLOSE_INV_RAW, 0) or 0),
        ),
        axis=1,
    ).astype(float)

    # Total Balance = Gross Inventory + In-Transit = Close Inv + Intransit
    # Matches Details tab column "Total Balance" exactly.
    # (Note: Details tab "Total Inventory" = Close Inv + Bottom — that is different.)
    summary_df["Total Balance"] = (
        summary_df["Gross Inventory"] + summary_df["In-Transit"].fillna(0)
    ).astype(float)

    # Available Net Inventory = Gross Inventory − Bottom
    # (Same as Details tab "Loadable" = Close Inv − Bottom)
    summary_df["Available Net Inventory"] = (
        summary_df["Gross Inventory"] - summary_df["Bottom"].fillna(0)
    ).astype(float)

    # Number days' Supply = Available Net Inventory / 7 Day Average
    sda = summary_df["Seven_Day_Avg_Sales"].replace({0: np.nan})
    summary_df["Number days' Supply"] = (
        summary_df["Available Net Inventory"] / sda
    ).replace([np.inf, -np.inf], np.nan)

    display_df = summary_df.rename(
        columns={
            "Prior_Day_Sales": "Prior Day Sales",
            "Seven_Day_Avg_Sales": "7 Day Average",
        }
    )

    desired_order = [
        "Location",
        "Product",
        "Available Net Inventory",
        "Prior Day Sales",
        "7 Day Average",
        "Number days' Supply",
        "Bottom",
        "SafeFill",
        "In-Transit",
        "Gross Inventory",
        "Total Balance",
    ]

    final_cols = [c for c in desired_order if c in display_df.columns]
    df_out = display_df[final_cols]

    # ── Location + Product filters ────────────────────────────────────────────
    _all_locs = sorted(df_out["Location"].dropna().astype(str).unique().tolist())
    _all_prods = sorted(df_out["Product"].dropna().astype(str).unique().tolist())

    _sf1, _sf2, _sf_info = st.columns([3, 3, 0.5])
    with _sf1:
        _sum_loc_filter: list[str] = st.multiselect(
            "Filter by Location:",
            options=_all_locs,
            default=[],
            placeholder="All locations",
            key="summary_loc_filter",
        )
    with _sf2:
        if _sum_loc_filter:
            _avail_prods = sorted(
                df_out.loc[df_out["Location"].isin(_sum_loc_filter), "Product"]
                .dropna().astype(str).unique().tolist()
            )
        else:
            _avail_prods = _all_prods
        _sum_prod_filter: list[str] = st.multiselect(
            "Filter by Product:",
            options=_avail_prods,
            default=[],
            placeholder="All products",
            key="summary_prod_filter",
        )
    with _sf_info:
        st.markdown('<div class="transparent-icon"></div>', unsafe_allow_html=True)
        _prior_day_label = prior_day_ts.strftime("%b %d, %Y")
        with st.popover("ℹ️"):
            st.markdown("### Summary — Column Formulas")
            st.caption(
                "All values use actual data only (≤ yesterday). "
                f"Current snapshot date: **{_prior_day_label}**"
            )
            st.markdown(
                "| Column | Source / Formula |\n"
                "|---|---|\n"
                f"| **Gross Inventory** | `Close Inv` from Details tab — most recent actual date (≤ {_prior_day_label}) |\n"
                "| **Bottom** | Required Min / Heel threshold from Admin Config → matches Details tab *Loadable* baseline |\n"
                "| **SafeFill** | Safe Fill capacity ceiling from Admin Config |\n"
                "| **In-Transit** | `INTRANSIT_BBL` (Details tab *Intransit* col); falls back to *Pipeline In* if zero |\n"
                "| **Total Balance** | `Gross Inventory + In-Transit` = `Close Inv + Intransit` — same as Details tab *Total Balance* column |\n"
                "| **Available Net Inventory** | `Gross Inventory − Bottom` — same as Details tab *Loadable* |\n"
                "| **Prior Day Sales** | `Rack/Lifting + Deliveries` on yesterday |\n"
                "| **7 Day Average** | 7-day trailing average of `Rack/Lifting` only — same metric as Details tab *7 Day Avg* |\n"
                "| **Number days' Supply** | `Available Net Inventory ÷ 7 Day Average` |"
            )

    # Store filter selections so Forecast and Midcon tables can read them
    st.session_state["_summary_loc_filter"] = _sum_loc_filter
    st.session_state["_summary_prod_filter"] = _sum_prod_filter

    if _sum_loc_filter:
        df_out = df_out[df_out["Location"].isin(_sum_loc_filter)]
    if _sum_prod_filter:
        df_out = df_out[df_out["Product"].isin(_sum_prod_filter)]

    column_config = _us_number_column_config(df_out, final_cols)

    _prior_day_label = prior_day_ts.strftime("%b %d, %Y")
    st.caption(
        f"Gross Inventory: **{_prior_day_label}** (yesterday)  |  "
        f"7 Day Average: trailing 7 days of Rack/Lifting"
    )

    st.dataframe(
        df_out,
        width="stretch",
        height=_dynamic_height(len(df_out)),
        column_config=column_config,
    )


def display_forecast_table(df_filtered, active_region):
    """Display the forecast table section."""
    st.markdown("### Forecast")

    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return

    group_cols = [COL_LOCATION, COL_PRODUCT]

    if not all(col in df_filtered.columns for col in group_cols):
        st.info("No forecast data available for the selected filters.")
        return

    today = datetime.date.today()
    today_ts = pd.Timestamp.today().normalize()
    last_month_end = today.replace(day=1) - datetime.timedelta(days=1)

    if today.month == 12:
        next_month_start = today.replace(year=today.year + 1, month=1, day=1)
    else:
        next_month_start = today.replace(month=today.month + 1, day=1)
    curr_month_end = next_month_start - datetime.timedelta(days=1)
    curr_month_end_ts = pd.Timestamp(curr_month_end)

    # ── Strip DB forecast rows BEFORE generating our own forecast extension.
    # DB forecast rows are separate projections and conflict with the app's
    # own _extend_with_30d_forecast logic, causing double-counting.
    if "SOURCE_TYPE" in df_filtered.columns:
        df_filtered = df_filtered[
            ~df_filtered["SOURCE_TYPE"].astype(str).str.strip().str.lower().isin(
                {"forecast", "forecast_user"}
            )
        ].copy()

    # ── Extend df_filtered with app-generated forecast rows so Projected EOM
    # has data for the month-end date even when the DB has no rows that far
    # ahead.  The extension is cached in session state per active_region so
    # it only runs once per load cycle (same pattern as HFS/ONEOK tables).
    _FC_EXT_KEY = f"summary_fc_ext|{active_region}"
    if _FC_EXT_KEY not in st.session_state:
        from details_tab import _extend_with_30d_forecast
        _unique_locs = (
            df_filtered["Location"].dropna().astype(str).str.strip().unique().tolist()
            if "Location" in df_filtered.columns else []
        )
        _ext_parts: list[pd.DataFrame] = []
        for _loc in _unique_locs:
            _loc_df = df_filtered[
                df_filtered["Location"].astype(str).str.strip() == _loc
            ]
            if _loc_df.empty:
                continue
            try:
                _ext_parts.append(
                    _extend_with_30d_forecast(
                        _loc_df,
                        id_col="Location",
                        region=active_region,
                        location=_loc,
                        forecast_end=curr_month_end_ts,
                        default_days=30,
                    )
                )
            except Exception:
                _ext_parts.append(_loc_df)
        st.session_state[_FC_EXT_KEY] = (
            pd.concat(_ext_parts, ignore_index=True) if _ext_parts else df_filtered
        )

    df_extended: pd.DataFrame = st.session_state[_FC_EXT_KEY].copy()
    df_extended[COL_DATE] = pd.to_datetime(df_extended[COL_DATE], errors="coerce")

    # ── Helper: Close Inv on target_date or most recent date before it
    # Must use the SAME non-zero-preferring logic as HFS's _close_on()
    # in midcon_report._build_hfs_display_df so that matching date points
    # produce identical numbers (e.g. Forecast Beginning = HFS Prev EOM).
    def _close_on_or_before(loc_prod_data: pd.DataFrame, target: datetime.date) -> float:
        # 1. Exact match — accept only non-zero
        exact = loc_prod_data[loc_prod_data[COL_DATE].dt.date == target]
        if not exact.empty:
            v = pd.to_numeric(exact[COL_CLOSE_INV_RAW].iloc[-1], errors="coerce")
            if pd.notna(v) and float(v) != 0.0:
                return float(v)
        # 2. Most recent non-zero ≤ target
        before = loc_prod_data[
            loc_prod_data[COL_DATE].dt.date <= target
        ].sort_values(COL_DATE)
        if not before.empty:
            vals = pd.to_numeric(before[COL_CLOSE_INV_RAW], errors="coerce")
            good = before[vals.notna() & (vals != 0.0)]
            if not good.empty:
                return float(pd.to_numeric(good[COL_CLOSE_INV_RAW].iloc[-1], errors="coerce"))
        return 0.0

    unique_combos = df_extended.groupby(group_cols, dropna=False).size().reset_index()[group_cols]
    forecast_data = []

    for _, row in unique_combos.iterrows():
        loc_prod_data = df_extended[
            (df_extended[COL_LOCATION] == row[COL_LOCATION]) &
            (df_extended[COL_PRODUCT] == row[COL_PRODUCT])
        ]

        # Bottom from Admin Config — same lookup as HFS _build_hfs_display_df.
        # Both Beginning inventory and Projected EOM use Total Inventory
        # (Close Inv + Bottom) so they match HFS "Prev EOM End Inv" / "EOM Projections".
        _ovr = get_threshold_overrides(
            region=_normalize_region(active_region),
            location=str(row[COL_LOCATION]),
            product=str(row[COL_PRODUCT]),
        )
        _bottom = float(_ovr.get("BOTTOM") or 0.0)

        # Beginning inventory = Total Inventory on last day of previous month
        # = Close Inv + Bottom  →  matches HFS table "Prev EOM End Inv" exactly
        beginning_inv = _close_on_or_before(loc_prod_data, last_month_end) + _bottom

        # Projected EOM = Total Inventory on last day of current month
        # = Close Inv + Bottom  →  matches HFS table "EOM Projections" exactly
        projected_eom = _close_on_or_before(loc_prod_data, curr_month_end) + _bottom

        build_draw = projected_eom - beginning_inv

        forecast_data.append({
            COL_LOCATION: row[COL_LOCATION],
            COL_PRODUCT: row[COL_PRODUCT],
            "Beginning inventory": round(beginning_inv, 0),
            "Projected EOM": round(projected_eom, 0),
            "Build/Draw": round(build_draw, 0),
        })

    if forecast_data:
        forecast_df = pd.DataFrame(forecast_data)
        forecast_cols = [COL_LOCATION, COL_PRODUCT, "Beginning inventory", "Projected EOM", "Build/Draw"]
        forecast_cols = [c for c in forecast_cols if c in forecast_df.columns]
        df_out = forecast_df[forecast_cols]

        # ── Read shared filters + formula dialog ─────────────────────────────
        _fc_loc_filter: list[str] = st.session_state.get("_summary_loc_filter", [])
        _fc_prod_filter: list[str] = st.session_state.get("_summary_prod_filter", [])

        _ff_spacer, _ff_info = st.columns([6, 0.5])
        with _ff_info:
            st.markdown('<div class="transparent-icon"></div>', unsafe_allow_html=True)
            with st.popover("ℹ️"):
                st.markdown("### Forecast — Column Formulas")
                _lme = last_month_end.strftime("%b %d, %Y")
                _cme = curr_month_end.strftime("%b %d, %Y")
                st.caption(
                    f"Beginning inventory: actual data (≤ {_lme}).  "
                    f"Projected EOM: forecast-extended data (≤ {_cme})."
                )
                st.markdown(
                    "| Column | Source / Formula |\n"
                    "|---|---|\n"
                    f"| **Beginning inventory** | `Close Inv + Bottom` on {_lme} (last day of previous month) = Details tab *Total Inventory* for that date ≡ HFS *Prev EOM End Inv* |\n"
                    f"| **Projected EOM** | `Close Inv + Bottom` on {_cme} (last day of current month) from forecast extension = Details tab *Total Inventory* for that date ≡ HFS *EOM Projections* |\n"
                    "| **Build/Draw** | `Projected EOM − Beginning inventory` |"
                )

        if _fc_loc_filter:
            df_out = df_out[df_out["Location"].isin(_fc_loc_filter)]
        if _fc_prod_filter:
            df_out = df_out[df_out["Product"].isin(_fc_prod_filter)]

        column_config = _us_number_column_config(df_out, forecast_cols)

        _lme_label = last_month_end.strftime("%b %d, %Y")
        _cme_label = curr_month_end.strftime("%b %d, %Y")
        st.caption(
            f"Beginning inventory: **{_lme_label}** (prev month end)  |  "
            f"Projected EOM: **{_cme_label}** (current month end)"
        )

        st.dataframe(
            df_out,
            width="stretch",
            height=_dynamic_height(len(df_out)),
            column_config=column_config,
        )
    else:
        st.info("No forecast data available for the selected filters.")
