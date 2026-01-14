import streamlit as st
import pandas as pd
import numpy as np
import datetime

from admin_config import get_threshold_overrides
from config import (
    REQUIRED_MAX_DEFAULTS,
    REQUIRED_MIN_DEFAULTS,
    INTRANSIT_DEFAULTS,
    GLOBAL_REQUIRED_MAX_FALLBACK,
    GLOBAL_REQUIRED_MIN_FALLBACK,
    GLOBAL_INTRANSIT_FALLBACK,
)

from config import (
    COL_AVAILABLE_SPACE,
    COL_BATCH_IN_RAW,
    COL_BATCH_OUT_RAW,
    COL_CLOSE_INV_RAW,
    COL_DATE,
    COL_OPEN_INV_RAW,
    COL_PIPELINE_IN,
    COL_PIPELINE_OUT,
    COL_PRODUCT,
    COL_PRODUCTION,
    COL_RACK_LIFTINGS_RAW,
    COL_SAFE_FILL_LIMIT,
    COL_SYSTEM,
    COL_LOCATION,
    COL_TANK_CAPACITY,
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


def _us_number_column_config(df: pd.DataFrame, cols: list[str]) -> dict[str, object]:
    """Return Streamlit column_config for numeric columns with US thousands separators."""
    cfg: dict[str, object] = {}
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            # "accounting" yields comma-separated values like 1,226,275.00
            cfg[c] = st.column_config.NumberColumn(c, format="accounting")
    return cfg


def display_regional_summary(df_filtered, active_region):
    """Display the regional summary section."""
    st.subheader("Summary")

    if df_filtered.empty:
        st.info("No data available for the selected region and filters.")
        return

    # Determine sales column
    sales_cols = [c for c in [COL_RACK_LIFTINGS_RAW, COL_BATCH_OUT_RAW] if c in df_filtered.columns]
    region_name = _normalize_region(active_region)

    # Group by Location/System and Product
    if _is_midcon(active_region):
        group_cols = [COL_SYSTEM, COL_PRODUCT]
    else:
        group_cols = [COL_LOCATION, COL_PRODUCT]

    if not all(col in df_filtered.columns for col in group_cols):
        st.warning("Required columns not found in the data.")
        return

    # Daily aggregation
    daily = (
        df_filtered
        .groupby(group_cols + [COL_DATE], as_index=False)
        .agg({
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
        })
    )

    daily["Sales"] = daily[sales_cols].sum(axis=1) if sales_cols else 0

    # Get latest date metrics
    if daily.empty:
        st.info("No data available for the selected filters.")
        return

    latest_date = daily[COL_DATE].max()
    prior_mask = daily[COL_DATE] < latest_date
    prior_day = daily.loc[prior_mask, COL_DATE].max() if prior_mask.any() else pd.NaT

    # Calculate 7-day average
    def compute_7day_avg(g):
        g = g.sort_values(COL_DATE)
        window = g[g[COL_DATE] <= latest_date].tail(7)
        return pd.Series({"Seven_Day_Avg_Sales": window["Sales"].mean() if not window.empty else 0})

    seven_day = (
        daily.groupby(group_cols)
        .apply(compute_7day_avg, include_groups=False)
        .reset_index()
    )

    # Latest inventory
    latest = (
        daily
        .sort_values([COL_DATE])
        .groupby(group_cols, as_index=False)
        .last()
    )

    # Prior day sales
    if pd.notna(prior_day):
        pds = (
            daily[daily[COL_DATE] == prior_day]
            .groupby(group_cols, as_index=False)["Sales"].sum()
            .rename(columns={"Sales": "Prior_Day_Sales"})
        )
    else:
        pds = latest[group_cols].copy()
        pds["Prior_Day_Sales"] = 0

    # Build summary DataFrame
    summary_df = (
        latest[group_cols + [COL_CLOSE_INV_RAW, COL_PIPELINE_IN]]
        .merge(pds, on=group_cols, how="left")
        .merge(seven_day, on=group_cols, how="left")
    )

    summary_df["Region"] = region_name

    # Calculate thresholds and in-transit
    summary_df["SafeFill"] = summary_df.apply(
        lambda row: calculate_required_max(row, group_cols, df_filtered),
        axis=1,
    ).astype(float)

    summary_df["Bottom"] = summary_df.apply(
        lambda row: calculate_required_min(row, group_cols, df_filtered),
        axis=1,
    ).astype(float)

    summary_df["In-Transit"] = summary_df.apply(
        lambda row: calculate_intransit(row, group_cols, df_filtered),
        axis=1,
    ).astype(float)

    # Inventory metrics
    summary_df["Gross Inventory"] = summary_df[COL_CLOSE_INV_RAW].fillna(0).astype(float)
    summary_df["Total Inventory"] = (
        summary_df["Gross Inventory"] + summary_df["In-Transit"].fillna(0)
    ).astype(float)

    # Net Inventory = Gross - Required Min (Heels)
    summary_df["Available Net Inventory"] = (summary_df["Gross Inventory"] - summary_df["Bottom"].fillna(0)).astype(float)

    # Days supply calculation: Net Inventory / 7 Day Avg
    sda = summary_df["Seven_Day_Avg_Sales"].replace({0: np.nan})
    summary_df["Number days' Supply"] = (
        summary_df["Available Net Inventory"] / sda
    ).replace([np.inf, -np.inf], np.nan)

    # Display formatting
    if _is_midcon(active_region):
        display_df = summary_df.copy()
        display_df["Location"] = display_df["System"]
    else:
        display_df = summary_df.copy()

    display_df = display_df.rename(
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
        "Total Inventory",
    ]

    final_cols = [c for c in desired_order if c in display_df.columns]

    df_out = display_df[final_cols]
    column_config = _us_number_column_config(df_out, final_cols)

    st.dataframe(
        df_out,
        width="stretch",
        height=320,
        column_config=column_config,
    )


def display_forecast_table(df_filtered, active_region):
    """Display the forecast table section."""
    st.markdown("### Forecast")

    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return

    # Determine group columns based on region
    if _is_midcon(active_region):
        group_cols = [COL_SYSTEM, COL_PRODUCT]
    else:
        group_cols = [COL_LOCATION, COL_PRODUCT]

    if not all(col in df_filtered.columns for col in group_cols):
        st.info("No forecast data available for the selected filters.")
        return

    today = datetime.date.today()
    last_month_end = today.replace(day=1) - datetime.timedelta(days=1)

    if today.month == 12:
        next_month_start = today.replace(year=today.year + 1, month=1, day=1)
    else:
        next_month_start = today.replace(month=today.month + 1, day=1)
    curr_month_end = next_month_start - datetime.timedelta(days=1)

    # Generate forecast data
    forecast_data = []

    # Get unique combinations
    unique_combos = df_filtered.groupby(group_cols, dropna=False).size().reset_index()[group_cols]

    for _, row in unique_combos.iterrows():
        loc_prod_data = df_filtered[
            (df_filtered[group_cols[0]] == row[group_cols[0]]) &
            (df_filtered[COL_PRODUCT] == row[COL_PRODUCT])
        ]

        # Beginning inventory (last day of previous month)
        beg_inv_row = loc_prod_data[loc_prod_data[COL_DATE].dt.date == last_month_end]
        beginning_inv = beg_inv_row[COL_CLOSE_INV_RAW].iloc[0] if not beg_inv_row.empty else 0

        # Projected EOM (last day of current month)
        proj_inv_row = loc_prod_data[loc_prod_data[COL_DATE].dt.date == curr_month_end]
        projected_eom = proj_inv_row[COL_CLOSE_INV_RAW].iloc[0] if not proj_inv_row.empty else 0

        build_draw = projected_eom - beginning_inv

        forecast_row = {
            group_cols[0]: row[group_cols[0]],
            COL_PRODUCT: row[COL_PRODUCT],
            "Beginning inventory": round(beginning_inv, 0),
            "Projected EOM": round(projected_eom, 0),
            "Build/Draw": round(build_draw, 0),
        }
        forecast_data.append(forecast_row)

    if forecast_data:
        forecast_df = pd.DataFrame(forecast_data)
        if group_cols[0] == COL_SYSTEM:
            forecast_df = forecast_df.rename(columns={COL_SYSTEM: COL_LOCATION})

        forecast_cols = [COL_LOCATION, COL_PRODUCT, "Beginning inventory", "Projected EOM", "Build/Draw"]
        forecast_cols = [c for c in forecast_cols if c in forecast_df.columns]

        df_out = forecast_df[forecast_cols]
        column_config = _us_number_column_config(df_out, forecast_cols)

        st.dataframe(
            df_out,
            width="stretch",
            height=320,
            column_config=column_config,
        )

    else:
        st.info("No forecast data available for the selected filters.")
