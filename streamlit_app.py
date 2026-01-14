"""
HF Sinclair Scheduler Dashboard - Main Application
Modularized version for better maintainability
"""

import streamlit as st
import pandas as pd

# Import modules
from ui_components import setup_page, apply_custom_css, display_header, display_data_freshness_cards
from data_loader import (
    initialize_data,
    ensure_numeric_columns,
    create_sidebar_filters,
    load_filtered_inventory_data,
    require_selected_location,
    apply_filters,
)
from summary_tab import display_regional_summary, display_forecast_table
from details_tab import display_details_tab
from admin_config import display_super_admin_panel


def main():
    """Main application function."""
    setup_page()
    apply_custom_css()
    display_header()

    # Initialize lightweight metadata (regions + source status)
    regions = initialize_data()

    if "admin_view" not in st.session_state:
        st.session_state.admin_view = False

    if "regions" not in st.session_state:
        st.warning("Loading data...")
        st.stop()

    regions = st.session_state.regions

    active_region = st.session_state.get("active_region", regions[0] if regions else None)
    df_region = pd.DataFrame()

    if st.session_state.admin_view:
        st.sidebar.button("⬅️ Back", key="admin_back", on_click=lambda: st.session_state.update({"admin_view": False}))
        display_super_admin_panel(
            regions=regions,
            active_region=active_region,
            all_data=None,
        )
        return

    st.sidebar.header("🔍 Filters")
    active_region = st.sidebar.selectbox("Select Region", regions, key="active_region") if regions else None

    with st.sidebar.form("filters_form", clear_on_submit=False):
        filters = create_sidebar_filters(regions, df_region)
        submitted = st.form_submit_button("✅ Submit Filters")

    active_region = filters.get("active_region")

    # Load data ONLY when submitted, or on first load if we don't have data yet.
    if submitted or ("df_filtered" not in st.session_state and active_region is not None):
        with st.spinner("Fetching filtered inventory data..."):
            require_selected_location(filters)
            df_loaded = load_filtered_inventory_data(filters)
            # Safety-net in-memory filtering (should typically be no-op)
            df_loaded = apply_filters(df_loaded, filters)
            df_loaded = ensure_numeric_columns(df_loaded)

        st.session_state.df_filtered = df_loaded
        st.session_state.active_filters = filters

    st.sidebar.markdown("---")
    if st.sidebar.button("🛠️ Super Admin Config", key="admin_open"):
        st.session_state.admin_view = True
        st.rerun()

    df_filtered = st.session_state.get("df_filtered", pd.DataFrame())
    active_filters = st.session_state.get("active_filters", filters)

    # Display data freshness cards (scoped to selected Location/System)
    source_status = st.session_state.get("source_status", pd.DataFrame())
    display_data_freshness_cards(
        active_region=active_region,
        selected_loc=active_filters.get("selected_loc"),
        loc_col=str(active_filters.get("loc_col") or "Location"),
        source_status=source_status,
    )

    main_tabs = ["📊 Regional Summary", "🧾 Details"]
    if "main_tab" not in st.session_state:
        st.session_state.main_tab = main_tabs[0]

    st.radio(
        "",
        options=main_tabs,
        key="main_tab",
        horizontal=True,
        label_visibility="collapsed",
    )

    if st.session_state.main_tab == "📊 Regional Summary":
        display_regional_summary(df_filtered, active_region)
        display_forecast_table(df_filtered, active_region)
    else:
        display_details_tab(
            df_filtered,
            active_region,
            start_ts=active_filters["start_ts"],
            end_ts=active_filters["end_ts"],
        )


if __name__ == "__main__":
    main()
