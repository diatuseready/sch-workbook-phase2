"""
Summary Tab module for HF Sinclair Scheduler Dashboard
Contains all logic and display components for the Regional Summary tab
"""

import streamlit as st
import pandas as pd
import numpy as np
from config import (
    REQUIRED_MAX_DEFAULTS,
    INTRANSIT_DEFAULTS,
    GLOBAL_REQUIRED_MAX_FALLBACK,
    GLOBAL_INTRANSIT_FALLBACK
)


def calculate_required_max(row, group_cols, df_filtered):
    """Calculate required max based on historical data or defaults."""
    # Use Tank Capacity * 0.85 as required max, or fall back to default
    if group_cols[0] == "System":
        key = f"{row['System']}|{row['Product']}"
        prod_key = row['Product']
    else:
        key = f"{row['Location']}|{row['Product']}"
        prod_key = row['Product']

    if key in REQUIRED_MAX_DEFAULTS:
        return REQUIRED_MAX_DEFAULTS[key]
    elif prod_key in REQUIRED_MAX_DEFAULTS:
        return REQUIRED_MAX_DEFAULTS[prod_key]
    else:
        # Calculate based on tank capacity if available
        tank_cap_data = df_filtered[
            (df_filtered[group_cols[0]] == row[group_cols[0]]) &
            (df_filtered["Product"] == row["Product"])
        ]["Tank Capacity"].max()

        if pd.notna(tank_cap_data) and tank_cap_data > 0:
            return tank_cap_data * 0.85
        else:
            return GLOBAL_REQUIRED_MAX_FALLBACK


def calculate_intransit(row, group_cols, df_filtered):
    """Calculate intransit based on pipeline data or defaults."""
    # Use Pipeline In data or defaults
    if group_cols[0] == "System":
        key = f"{row['System']}|{row['Product']}"
        prod_key = row['Product']
    else:
        key = f"{row['Location']}|{row['Product']}"
        prod_key = row['Product']

    if key in INTRANSIT_DEFAULTS:
        return INTRANSIT_DEFAULTS[key]
    elif prod_key in INTRANSIT_DEFAULTS:
        return INTRANSIT_DEFAULTS[prod_key]
    else:
        # Use average pipeline in as intransit estimate
        pipeline_data = df_filtered[
            (df_filtered[group_cols[0]] == row[group_cols[0]]) &
            (df_filtered["Product"] == row["Product"])
        ]["Pipeline In"].mean()

        if pd.notna(pipeline_data) and pipeline_data > 0:
            return pipeline_data
        else:
            return GLOBAL_INTRANSIT_FALLBACK


def display_regional_summary(df_filtered, active_region):
    """Display the regional summary section."""
    st.subheader("📊 Regional Summary")

    if df_filtered.empty:
        st.info("No data available for the selected region and filters.")
        return

    # Determine sales column
    sales_cols = [c for c in ["Rack/Liftings", "Batch Out (DELIVERIES_BBL)"] if c in df_filtered.columns]
    sales_col = sales_cols[0] if sales_cols else None

    # Group by Location/System and Product
    if active_region == "Group Supply Report (Midcon)":
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

    daily["Sales"] = daily[sales_col] if sales_col else 0

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

    # Total aggregates
    totals = (
        df_filtered
        .groupby(group_cols, as_index=False)
        .agg({
            "Batch In (RECEIPTS_BBL)": "sum",
            "Batch Out (DELIVERIES_BBL)": "sum",
            "Rack/Liftings": "sum"
        })
        .rename(columns={
            "Batch In (RECEIPTS_BBL)": "Total In",
            "Batch Out (DELIVERIES_BBL)": "Total Out",
            "Rack/Liftings": "Total Rack"
        })
    )

    # Build summary DataFrame
    summary_df = (
        latest[group_cols + ["Close Inv"]]
        .merge(totals, on=group_cols, how="left")
        .merge(pds, on=group_cols, how="left")
        .merge(seven_day, on=group_cols, how="left")
    )

    # Calculate Required Max and Intransit
    summary_df["Required Maximums"] = summary_df.apply(
        lambda row: calculate_required_max(row, group_cols, df_filtered),
        axis=1
    ).astype(float)

    summary_df["Intransit Bbls"] = summary_df.apply(
        lambda row: calculate_intransit(row, group_cols, df_filtered),
        axis=1
    ).astype(float)

    # Calculate inventory metrics
    summary_df["Gross Inventory"] = (
        summary_df["Close Inv"].fillna(0) +
        summary_df["Intransit Bbls"].fillna(0)
    ).astype(float)

    summary_df["Avail. (NET) Inventory"] = (
        summary_df["Gross Inventory"] -
        summary_df["Required Maximums"]
    ).astype(float)

    # Days supply calculation
    sda = summary_df["Seven_Day_Avg_Sales"].replace({0: np.nan})
    summary_df["# Days Supply"] = (
        summary_df["Close Inv"] / sda
    ).replace([np.inf, -np.inf], np.nan)

    # Display formatting
    if active_region == "Group Supply Report (Midcon)":
        display_df = summary_df.rename(columns={
            "System": "System",
            "Product": "Product",
            "Close Inv": "Closing Inv",
            "Prior_Day_Sales": "Prior Day Sales",
            "Seven_Day_Avg_Sales": "7 Day Average"
        })
        desired_order = [
            "System", "Product", "Closing Inv", "Intransit Bbls",
            "Gross Inventory", "Required Maximums", "Avail. (NET) Inventory",
            "Prior Day Sales", "7 Day Average", "# Days Supply",
            "Total In", "Total Out", "Total Rack"
        ]
    else:
        display_df = summary_df.rename(columns={
            "Location": "Location",
            "Product": "Product",
            "Close Inv": "Closing Inv",
            "Prior_Day_Sales": "Prior Day Sales",
            "Seven_Day_Avg_Sales": "7 Day Average"
        })
        desired_order = [
            "Location", "Product", "Closing Inv", "Intransit Bbls",
            "Gross Inventory", "Required Maximums", "Avail. (NET) Inventory",
            "Prior Day Sales", "7 Day Average", "# Days Supply",
            "Total In", "Total Out", "Total Rack"
        ]

    final_cols = [c for c in desired_order if c in display_df.columns]
    st.dataframe(
        display_df[final_cols],
        width="stretch",
        height=320
    )


def display_forecast_table(df_filtered, active_region):
    """Display the forecast table section."""
    st.markdown("### 📈 Forecast Table")

    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return

    # Determine group columns based on region
    if active_region == "Group Supply Report (Midcon)":
        group_cols = ["System", "Product"]
    else:
        group_cols = ["Location", "Product"]

    if not all(col in df_filtered.columns for col in group_cols):
        st.info("No forecast data available for the selected filters.")
        return

    # Generate forecast data based on actual data
    forecast_data = []

    # Get unique combinations
    unique_combos = df_filtered.groupby(group_cols).size().reset_index()[group_cols]
    unique_combos = unique_combos.head(6)  # Limit to first 6 for display

    for _, row in unique_combos.iterrows():
        # Calculate forecasts based on historical data
        loc_prod_data = df_filtered[
            (df_filtered[group_cols[0]] == row[group_cols[0]]) &
            (df_filtered["Product"] == row["Product"])
        ]

        current_inv = loc_prod_data["Close Inv"].iloc[-1] if not loc_prod_data.empty else 0
        avg_daily_change = loc_prod_data["Close Inv"].diff().mean() if len(loc_prod_data) > 1 else 100

        forecast_row = {
            group_cols[0]: row[group_cols[0]],
            "Product": row["Product"],
            "Current Inventory": round(current_inv, 0),
            "EOM Projections": round(current_inv + (avg_daily_change * 30), 0),
            "LIFO Target": round(current_inv * 1.1, 0),
            "OPS Target": round(current_inv * 1.05, 0),
            "EOM vs OPS": round((current_inv + (avg_daily_change * 30)) - (current_inv * 1.05), 0),
            "LIFO vs OPS": round((current_inv * 1.1) - (current_inv * 1.05), 0),
            "Build and Draw": round(avg_daily_change * 30, 0)
        }
        forecast_data.append(forecast_row)

    if forecast_data:
        forecast_df = pd.DataFrame(forecast_data)
        forecast_cols = [
            group_cols[0], "Product", "Current Inventory", "EOM Projections",
            "LIFO Target", "OPS Target", "EOM vs OPS", "LIFO vs OPS", "Build and Draw"
        ]
        st.dataframe(forecast_df[forecast_cols], width="stretch", height=320)
    else:
        st.info("No forecast data available for the selected filters.")
