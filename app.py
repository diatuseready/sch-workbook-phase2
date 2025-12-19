"""
HF Sinclair Scheduler Dashboard - Main Application
Modularized version for better maintainability
"""

import streamlit as st
import pandas as pd

# Import modules
from ui_components import setup_page, apply_custom_css, display_header, display_data_freshness_cards
from data_loader import initialize_data, ensure_numeric_columns
from sidebar_filters import create_sidebar_filters, apply_filters
from summary_tab import display_regional_summary, display_forecast_table
from details_tab import display_details_tab


def main():
    """Main application function."""

    # Page setup
    setup_page()
    apply_custom_css()
    display_header()

    # Initialize data
    regions = initialize_data()

    # Ensure we have regions loaded
    if "regions" not in st.session_state:
        st.warning("Loading data...")
        st.stop()

    regions = st.session_state.regions

    # Get active region from sidebar (or first region if none selected)
    if regions:
        # Get region data
        active_region = st.session_state.get("active_region", regions[0])
        df_region = st.session_state.data.get(active_region, pd.DataFrame())
    else:
        active_region = None
        df_region = pd.DataFrame()

    # Create sidebar filters
    filters = create_sidebar_filters(regions, df_region)
    active_region = filters["active_region"]

    # Update region data based on selected region
    if active_region:
        df_region = st.session_state.data.get(active_region, pd.DataFrame())

    # Apply filters
    df_filtered = apply_filters(df_region, filters)

    # Ensure numeric columns
    df_filtered = ensure_numeric_columns(df_filtered)

    # Display data freshness cards
    if active_region:
        display_data_freshness_cards(active_region)

    # Create main tabs
    summary_tab, details_tab = st.tabs(["📊 Regional Summary", "🧾 Details"])

    # Summary Tab
    with summary_tab:
        display_regional_summary(df_filtered, active_region)
        display_forecast_table(df_filtered, active_region)

    # Details Tab
    with details_tab:
        display_details_tab(df_filtered, active_region)


if __name__ == "__main__":
    main()
