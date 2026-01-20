import streamlit as st
import pandas as pd

from app_logging import logged_button, logged_callback

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


def main():
    """Main application function."""
    setup_page()
    apply_custom_css()
    display_header()

    collapse_now = bool(st.session_state.get("collapse_expandables", False))

    def _un_collapse_expandables() -> None:
        """Allow expanders to render expanded again (used by control/filter changes)."""
        st.session_state["collapse_expandables"] = False

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

    with st.expander("Controls", expanded=(False if collapse_now else True)):
        header_c1, header_c2 = st.columns([4, 1])
        with header_c1:
            # If the user changes the Region, we should stop force-collapsing expanders.
            st.selectbox("Region", regions, key="active_region", on_change=_un_collapse_expandables)
        with header_c2:
            # Align button with the selectbox (which renders its label above).
            st.markdown('<div class="btn-spacer"></div>', unsafe_allow_html=True)
            right = st.columns([1, 1])[1]
            with right:
                def _open_admin_config():
                    # Collapse all expanders before entering admin view.
                    st.session_state["collapse_expandables"] = True
                    st.session_state.admin_view = True

                st.button(
                    "Admin Config",
                    key="admin_open",
                    on_click=logged_callback(
                        _open_admin_config,
                        event="nav_admin_open",
                        metadata={"from": "main"},
                        service_module="UI",
                    ),
                )

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

    # Region label is used as-is (no Midcon special-casing).
    active_region_norm = st.session_state.get("active_region")

    # Admin view (full-screen)
    if st.session_state.admin_view:
        if logged_button("⬅️ Back", key="admin_back", event="nav_admin_back"):
            st.session_state.admin_view = False
            # Returning from admin view should not force everything collapsed.
            st.session_state["collapse_expandables"] = False
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

    with st.expander("Filters", expanded=(False if collapse_now else True)):
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

        # Collapse expanders after Submit and keep them collapsed across reruns until
        # the user changes a control/filter.
        st.session_state["collapse_expandables"] = True
        st.rerun()

    df_details = st.session_state.get(details_cache_key, pd.DataFrame())
    effective_filters = st.session_state.get(filters_cache_key, details_filters)

    # Optional: show freshness only when a location/system is in-scope.
    with st.expander("Data Freshness", expanded=(False if collapse_now else False)):
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
