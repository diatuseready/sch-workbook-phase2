"""
Summary Tab module for HF Sinclair Scheduler Dashboard
Contains all logic and display components for the Regional Summary tab
"""

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


def _normalize_region(active_region: str) -> str:
    return "Midcon" if active_region == "Group Supply Report (Midcon)" else active_region


def _is_midcon(active_region: str) -> bool:
    return _normalize_region(active_region) == "Midcon"


def calculate_required_max(row, group_cols, df_filtered):
    region = str(row.get("Region") or "Unknown")
    loc_or_sys = row.get(group_cols[0])
    overrides = get_threshold_overrides(region=_normalize_region(region), location=str(loc_or_sys) if pd.notna(loc_or_sys) else None)

    safefill = overrides.get("SAFEFILL")
    if safefill is not None and not pd.isna(safefill):
        return float(safefill)

    prod = str(row.get("Product") or "")
    key = f"{loc_or_sys}|{prod}"
    if key in REQUIRED_MAX_DEFAULTS:
        return float(REQUIRED_MAX_DEFAULTS[key])
    if prod in REQUIRED_MAX_DEFAULTS:
        return float(REQUIRED_MAX_DEFAULTS[prod])

    if "Safe Fill Limit" in df_filtered.columns and group_cols[0] in df_filtered.columns:
        safe_fill = df_filtered[
            (df_filtered[group_cols[0]] == row[group_cols[0]]) &
            (df_filtered["Product"] == row["Product"])
        ]["Safe Fill Limit"].max()
        if pd.notna(safe_fill) and safe_fill > 0:
            return float(safe_fill)

    return float(GLOBAL_REQUIRED_MAX_FALLBACK)


def calculate_intransit(row, group_cols, df_filtered):
    """Calculate intransit based on pipeline data or defaults."""
    if group_cols[0] == "System":
        key = f"{row['System']}|{row['Product']}"
        prod_key = row["Product"]
    else:
        key = f"{row['Location']}|{row['Product']}"
        prod_key = row["Product"]

    pipeline_val = row.get("Pipeline In", 0)
    if pd.notna(pipeline_val) and pipeline_val > 0:
        return float(pipeline_val)

    if key in INTRANSIT_DEFAULTS:
        return INTRANSIT_DEFAULTS[key]
    if prod_key in INTRANSIT_DEFAULTS:
        return INTRANSIT_DEFAULTS[prod_key]

    return GLOBAL_INTRANSIT_FALLBACK


def calculate_required_min(row, group_cols, df_filtered):
    region = str(row.get("Region") or "Unknown")
    loc_or_sys = row.get(group_cols[0])
    overrides = get_threshold_overrides(region=_normalize_region(region), location=str(loc_or_sys) if pd.notna(loc_or_sys) else None)

    bottom = overrides.get("BOTTOM")
    if bottom is not None and not pd.isna(bottom):
        return float(bottom)

    prod = str(row.get("Product") or "")
    key = f"{loc_or_sys}|{prod}"
    if key in REQUIRED_MIN_DEFAULTS:
        return float(REQUIRED_MIN_DEFAULTS[key])
    if prod in REQUIRED_MIN_DEFAULTS:
        return float(REQUIRED_MIN_DEFAULTS[prod])

    return float(GLOBAL_REQUIRED_MIN_FALLBACK)


def display_regional_summary(df_filtered, active_region):
    """Display the regional summary section."""
    st.subheader("📊 Regional Summary")

    if df_filtered.empty:
        st.info("No data available for the selected region and filters.")
        return

    # Determine sales column
    sales_cols = [c for c in ["Rack/Liftings", "Batch Out (DELIVERIES_BBL)"] if c in df_filtered.columns]
    region_name = _normalize_region(active_region)

    # Group by Location/System and Product
    if _is_midcon(active_region):
        group_cols = ["System", "Product"]
    else:
        group_cols = ["Location", "Product"]

    if not all(col in df_filtered.columns for col in group_cols):
        st.warning("Required columns not found in the data.")
        return

    # Daily aggregation
    daily = (
        df_filtered
        .groupby(group_cols + ["Date"], as_index=False)
        .agg({
            "Close Inv": "last",
            "Open Inv": "first",
            "Batch In (RECEIPTS_BBL)": "sum",
            "Batch Out (DELIVERIES_BBL)": "sum",
            "Rack/Liftings": "sum",
            "Production": "sum",
            "Pipeline In": "sum",
            "Pipeline Out": "sum",
            "Tank Capacity": "max",
            "Safe Fill Limit": "max",
            "Available Space": "mean"
        })
    )

    daily["Sales"] = daily[sales_cols].sum(axis=1) if sales_cols else 0

    # Get latest date metrics
    if daily.empty:
        st.info("No data available for the selected filters.")
        return

    latest_date = daily["Date"].max()
    prior_mask = daily["Date"] < latest_date
    prior_day = daily.loc[prior_mask, "Date"].max() if prior_mask.any() else pd.NaT

    # Calculate 7-day average
    def compute_7day_avg(g):
        g = g.sort_values("Date")
        window = g[g["Date"] <= latest_date].tail(7)
        return pd.Series({"Seven_Day_Avg_Sales": window["Sales"].mean() if not window.empty else 0})

    seven_day = daily.groupby(group_cols).apply(compute_7day_avg).reset_index()

    # Latest inventory
    latest = (
        daily[daily["Date"] == latest_date]
        .sort_values(["Date"])
        .groupby(group_cols, as_index=False)
        .last()
    )

    # Prior day sales
    if pd.notna(prior_day):
        pds = (
            daily[daily["Date"] == prior_day]
            .groupby(group_cols, as_index=False)["Sales"].sum()
            .rename(columns={"Sales": "Prior_Day_Sales"})
        )
    else:
        pds = latest[group_cols].copy()
        pds["Prior_Day_Sales"] = 0

    # Build summary DataFrame
    summary_df = (
        latest[group_cols + ["Close Inv", "Pipeline In"]]
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
    summary_df["Gross Inventory"] = summary_df["Close Inv"].fillna(0).astype(float)
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
        # st.dataframe(
        #     display_df[final_cols],
        #     width="stretch",
        #     height=320
        # )

    st.dataframe(
    display_df[final_cols],
    use_container_width=True,
    height=320
    )



def display_forecast_table(df_filtered, active_region):
    """Display the forecast table section."""
    st.markdown("### 📈 Forecast Table")

    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return

    # Determine group columns based on region
    if _is_midcon(active_region):
        group_cols = ["System", "Product"]
    else:
        group_cols = ["Location", "Product"]

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
    unique_combos = df_filtered.groupby(group_cols).size().reset_index()[group_cols]
    unique_combos = unique_combos.head(6)  # Limit to first 6 for display

    for _, row in unique_combos.iterrows():
        loc_prod_data = df_filtered[
            (df_filtered[group_cols[0]] == row[group_cols[0]]) &
            (df_filtered["Product"] == row["Product"])
        ]

        beg_inv_row = loc_prod_data[loc_prod_data["Date"].dt.date == last_month_end]
        if not beg_inv_row.empty:
            beginning_inv = beg_inv_row["Close Inv"].iloc[0]
        else:
            beginning_inv = 0

        proj_inv_row = loc_prod_data[loc_prod_data["Date"].dt.date == curr_month_end]
        if not proj_inv_row.empty:
            projected_eom = proj_inv_row["Close Inv"].iloc[0]
        else:
            projected_eom = 0

        build_draw = projected_eom - beginning_inv

        forecast_row = {
            group_cols[0]: row[group_cols[0]],
            "Product": row["Product"],
            "Beginning inventory": round(beginning_inv, 0),
            "Projected EOM": round(projected_eom, 0),
            "Build/Draw": round(build_draw, 0),
        }
        forecast_data.append(forecast_row)

    if forecast_data:
        forecast_df = pd.DataFrame(forecast_data)
        if group_cols[0] == "System":
            forecast_df = forecast_df.rename(columns={"System": "Location"})

        forecast_cols = ["Location", "Product", "Beginning inventory", "Projected EOM", "Build/Draw"]
        forecast_cols = [c for c in forecast_cols if c in forecast_df.columns]
        # st.dataframe(forecast_df[forecast_cols], width="stretch", height=320)
        st.dataframe(
            forecast_df[forecast_cols],
            use_container_width=True,
            height=320
        )

    else:
        st.info("No forecast data available for the selected filters.")
