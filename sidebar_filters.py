"""
Sidebar Filters module for HF Sinclair Scheduler Dashboard
Handles all sidebar filter components and logic
"""

import streamlit as st
import pandas as pd
from datetime import timedelta, date

from admin_config import get_default_date_window


def create_sidebar_filters(regions, df_region):
    """Create and manage sidebar filters."""

    st.sidebar.header("🔍 Filters")

    # Region selector
    if regions:
        active_region = st.sidebar.selectbox("Select Region", regions, key="active_region")
    else:
        active_region = None
        st.sidebar.warning("No regions available")

    loc_col = "System" if active_region == "Midcon" else "Location"
    filter_label = "🏭 System" if active_region == "Midcon" else "📍 Location"
    locations = sorted(df_region[loc_col].dropna().unique().tolist()) if loc_col in df_region.columns and not df_region.empty else []

    # Default to *all* locations. Limiting defaults can make certain location tabs
    # appear/disappear in confusing ways when other filters (like Product) are also
    # constrained.
    selected_locs = st.sidebar.multiselect(
        filter_label,
        options=locations,
        default=locations,
        key="selected_locs",
    )

    # Date range selector
    today = date.today()
    scope_location = selected_locs[0] if len(selected_locs) == 1 else None
    start_off, end_off = get_default_date_window(region=active_region or "Unknown", location=scope_location)
    
    # st.sidebar.info(f"start_off and end_off {start_off} to {end_off}")
    default_start = today + timedelta(days=int(start_off))
    default_end = today + timedelta(days=int(end_off))

    if not df_region.empty and "Date" in df_region.columns:
        df_min = df_region["Date"].min()
        df_max = df_region["Date"].max()
        df_min_d = df_min.date() if pd.notna(df_min) else default_start
        df_max_d = df_max.date() if pd.notna(df_max) else default_end
    else:
        df_min_d = default_start
        df_max_d = default_end

    min_value = min(df_min_d, default_start)
    max_value = max(df_max_d, default_end) 

    # st.sidebar.info(f"min_value {min_value} max_value {max_value}")

    #  FIX: Default to showing database date range if it exists
    # If database has older dates than default range, start from database min
    actual_start = df_min_d if df_min_d < default_start else default_start
    actual_end = max_value  # Always extend to max available or forecast end
    

    
    # date_range = st.sidebar.date_input(
    #     "Date Range",
    #     value=(default_start, default_end),
    #     min_value=min_value,
    #     max_value=max_value,
    #     key=f"date_{active_region}_{scope_location or 'all'}"
    # )

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(actual_start, actual_end),
        min_value=min_value,
        max_value=max_value,
        key=f"date_{active_region}_{scope_location or 'all'}"
    )

    if isinstance(date_range, (list, tuple)):
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range[0] if date_range else date.today()
    else:
        start_date = end_date = date_range

    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)

    # Product filter
    subset = df_region[df_region[loc_col].isin(selected_locs)] if selected_locs else df_region
    products = sorted(subset["Product"].dropna().unique().tolist()) if "Product" in subset.columns and not subset.empty else []
    # Default to *all* products. If we only preselect the first N products, locations
    # whose data exists only for non-selected products will disappear from the Details
    # tabs (because df_filtered becomes empty for those locations).
    selected_prods = st.sidebar.multiselect(
        "🧪 Product",
        options=products,
        default=products,
        key="selected_prods",
    )

    return {
        "active_region": active_region,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "selected_locs": selected_locs,
        "selected_prods": selected_prods,
        "loc_col": loc_col,
        "locations": locations,
        "products": products
    }


def apply_filters(df_region, filters):
    """Apply the selected filters to the dataframe."""
    df_filtered = df_region.copy()

    if df_filtered.empty:
        return df_filtered

    # Apply date filter
    df_filtered = df_filtered[
        (df_filtered["Date"] >= filters["start_ts"]) &
        (df_filtered["Date"] <= filters["end_ts"])
    ]

    # Apply location/system filter
    loc_col = filters.get("loc_col", "Location")
    if filters["selected_locs"] and loc_col in df_filtered.columns:
        if len(filters["selected_locs"]) < len(filters["locations"]):
            df_filtered = df_filtered[df_filtered[loc_col].isin(filters["selected_locs"])]

    # Apply product filter
    if filters["selected_prods"] and "Product" in df_filtered.columns:
        if len(filters["selected_prods"]) < len(filters["products"]):
            df_filtered = df_filtered[df_filtered["Product"].isin(filters["selected_prods"])]

    return df_filtered
