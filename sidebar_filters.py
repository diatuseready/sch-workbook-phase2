"""
Sidebar Filters module for HF Sinclair Scheduler Dashboard
Handles all sidebar filter components and logic
"""

import streamlit as st
import pandas as pd
from datetime import timedelta, date

def create_sidebar_filters(regions, df_region):
    """Create and manage sidebar filters."""
    
    st.sidebar.header("🔍 Filters")
    
    # Region selector
    if regions:
        active_region = st.sidebar.selectbox("Select Region", regions, key="active_region")
    else:
        active_region = None
        st.sidebar.warning("No regions available")
    
    # Date range selector
    if not df_region.empty:
        min_date = df_region["Date"].min()
        max_date = df_region["Date"].max()
    else:
        min_date = pd.Timestamp.today() - timedelta(days=30)
        max_date = pd.Timestamp.today()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date.date() if pd.notna(min_date) else date.today(),
               max_date.date() if pd.notna(max_date) else date.today()),
        key=f"date_{active_region}"
    )
    
    # Handle date input format
    if isinstance(date_range, (list, tuple)):
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range[0] if date_range else date.today()
    else:
        start_date = end_date = date_range
    
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    
    # Location/System filter
    locations = sorted(df_region["Location"].dropna().unique().tolist()) if "Location" in df_region.columns and not df_region.empty else []
    
    # Change filter label based on region
    if active_region == "Group Supply Report (Midcon)":
        filter_label = "🏭 System"
    else:
        filter_label = "📍 Location"
    
    selected_locs = st.sidebar.multiselect(
        filter_label, 
        options=locations, 
        default=locations[:5] if len(locations) > 5 else locations
    )
    
    # Product filter
    subset = df_region[df_region["Location"].isin(selected_locs)] if selected_locs else df_region
    products = sorted(subset["Product"].dropna().unique().tolist()) if "Product" in subset.columns and not subset.empty else []
    selected_prods = st.sidebar.multiselect(
        "🧪 Product", 
        options=products, 
        default=products[:5] if len(products) > 5 else products
    )
    
    return {
        "active_region": active_region,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "selected_locs": selected_locs,
        "selected_prods": selected_prods,
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
    
    # Apply location filter
    if filters["selected_locs"] and "Location" in df_filtered.columns:
        if len(filters["selected_locs"]) < len(filters["locations"]):
            df_filtered = df_filtered[df_filtered["Location"].isin(filters["selected_locs"])]
    
    # Apply product filter
    if filters["selected_prods"] and "Product" in df_filtered.columns:
        if len(filters["selected_prods"]) < len(filters["products"]):
            df_filtered = df_filtered[df_filtered["Product"].isin(filters["selected_prods"])]
    
    return df_filtered
