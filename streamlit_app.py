"""HF Sinclair Scheduler Dashboard - Main Application.

Redesigned UX (top-down, no sidebar):
- Region selector under the header
- Summary/Details switch appears only after Region is selected
- Summary is region-scoped and independent of Details filters
- Details has in-page Location + Date filters + Submit above the table
"""

import streamlit as st
import pandas as pd

from ui_components import setup_page, apply_custom_css, display_header, display_data_freshness_cards
from data_loader import (
    initialize_data,
    load_filtered_inventory_data,
    load_region_inventory_data,
    require_selected_location,
)
from summary_tab import display_regional_summary, display_forecast_table
from details_tab import display_details_tab, render_details_filters
from admin_config import display_super_admin_panel


def _normalize_region_label(active_region: str | None) -> str | None:
    """Match the same Midcon normalization used elsewhere in the app."""
    if active_region is None:
        return None
    return "Midcon" if active_region == "Group Supply Report (Midcon)" else active_region


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

    # ---------------------------
    # Top controls (collapsible)
    # ---------------------------

    if not regions:
        st.warning("No regions available")
        st.stop()

    with st.expander("Controls", expanded=True):
        header_c1, header_c2 = st.columns([4, 1])
        with header_c1:
            st.selectbox("Region", regions, key="active_region")
        with header_c2:
            # Align button with the selectbox (which renders its label above).
            st.markdown('<div class="btn-spacer"></div>', unsafe_allow_html=True)
            right = st.columns([1, 1])[1]
            with right:
                if st.button("Admin Config", key="admin_open"):
                    st.session_state.admin_view = True
                    st.rerun()

        main_tabs = ["📊 Regional Summary", "🧾 Details"]
        if "main_tab" not in st.session_state:
            st.session_state.main_tab = main_tabs[0]

        st.radio(
            "View",
            options=main_tabs,
            key="main_tab",
            horizontal=True,
            label_visibility="collapsed",
        )

    active_region_norm = _normalize_region_label(st.session_state.get("active_region"))

    # Admin view (full-screen)
    if st.session_state.admin_view:
        if st.button("⬅️ Back", key="admin_back"):
            st.session_state.admin_view = False
            st.rerun()
        display_super_admin_panel(regions=regions, active_region=active_region_norm, all_data=None)
        return

    if not active_region_norm:
        st.info("Select a Region to continue.")
        return

    # ----------------
    # Regional Summary
    # ----------------
    if st.session_state.main_tab == "📊 Regional Summary":
        cache_key = f"df_region_summary|{active_region_norm}"
        if cache_key not in st.session_state:
            with st.spinner("Loading regional summary data..."):
                st.session_state[cache_key] = load_region_inventory_data(region=active_region_norm)

        df_region = st.session_state.get(cache_key, pd.DataFrame())
        display_regional_summary(df_region, active_region_norm)
        display_forecast_table(df_region, active_region_norm)
        return

    # -------
    # Details
    # -------

    with st.expander("Filters", expanded=True):
        details_filters = render_details_filters(regions=regions, active_region=active_region_norm)

    # Load details data only on submit (but keep previous results displayed).
    details_cache_key = f"df_details|{active_region_norm}"
    filters_cache_key = f"details_filters|{active_region_norm}"

    if details_filters.get("submitted"):
        try:
            require_selected_location(details_filters)
        except Exception:
            # require_selected_location already surfaces a warning + st.stop()
            return

        with st.spinner("Fetching details data..."):
            st.session_state[details_cache_key] = load_filtered_inventory_data(details_filters)
            st.session_state[filters_cache_key] = details_filters

    df_details = st.session_state.get(details_cache_key, pd.DataFrame())
    effective_filters = st.session_state.get(filters_cache_key, details_filters)

    # Optional: show freshness only when a location/system is in-scope.
    with st.expander("Data Freshness", expanded=False):
        source_status = st.session_state.get("source_status", pd.DataFrame())
        display_data_freshness_cards(
            active_region=active_region_norm,
            selected_loc=effective_filters.get("selected_loc"),
            loc_col=str(effective_filters.get("loc_col") or "Location"),
            source_status=source_status,
        )

    if df_details is None or df_details.empty:
        st.info("Select filters and press Submit to load details.")
        return

    display_details_tab(
        df_details,
        active_region_norm,
        start_ts=effective_filters["start_ts"],
        end_ts=effective_filters["end_ts"],
        selected_loc=(None if effective_filters.get("selected_loc") in (None, "") else str(effective_filters.get("selected_loc"))),
    )


if __name__ == "__main__":
    main()
