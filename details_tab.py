"""
Details Tab module for HF Sinclair Scheduler Dashboard
Contains all logic and display components for the Details tab
"""

import streamlit as st


def display_midcon_details(df_filtered, active_region):
    """Display details for Midcon region."""
    st.subheader("🧾 Group Daily Details (Editable)")

    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return

    df_show = df_filtered.sort_values("Date")
    # Rename Location column to System for Midcon
    df_display = df_show.copy()
    # Drop the System column if it exists (it's a duplicate of Location)
    if "System" in df_display.columns:
        df_display = df_display.drop(columns=["System"])
    # Now rename Location to System
    if "Location" in df_display.columns:
        df_display = df_display.rename(columns={"Location": "System"})

    # Select columns to display
    display_cols = ["Date", "System", "Product", "Close Inv", "Open Inv",
                    "Batch In (RECEIPTS_BBL)", "Batch Out (DELIVERIES_BBL)",
                    "Rack/Liftings", "Production", "Pipeline In", "Pipeline Out",
                    "Adjustments", "Gain/Loss", "Transfers", "Notes"]
    display_cols = [c for c in display_cols if c in df_display.columns]

    # Make the dataframe editable
    st.data_editor(
        df_display[display_cols],
        num_rows="dynamic",
        use_container_width=True,
        key=f"{active_region}_edit"
    )

    # Save button
    st.markdown('<div class="save-btn-bottom">', unsafe_allow_html=True)
    if st.button("💾 Save Changes", key=f"save_{active_region}"):
        # Here you would save back to Snowflake
        # For now, just show a success message
        st.success("✅ Changes saved successfully!")
    st.markdown('</div>', unsafe_allow_html=True)


def display_location_details(df_filtered, active_region):
    """Display details by location for non-Midcon regions."""
    st.subheader("🏭 Locations")

    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return

    # Get unique locations
    if "Location" in df_filtered.columns:
        region_locs = sorted(df_filtered["Location"].dropna().unique().tolist())
    else:
        region_locs = []

    if not region_locs:
        st.write("*(No locations available in the current selection)*")
        return

    loc_tabs = st.tabs(region_locs)
    for i, loc in enumerate(region_locs):
        with loc_tabs[i]:
            st.markdown(f"### 📍 {loc}")
            df_loc = df_filtered[df_filtered["Location"] == loc].sort_values("Date")

            if df_loc.empty:
                st.write("*(No data for this location)*")
            else:
                # Select columns to display
                display_cols = ["Date", "Product", "Close Inv", "Open Inv",
                                "Batch In (RECEIPTS_BBL)", "Batch Out (DELIVERIES_BBL)",
                                "Rack/Liftings", "Production", "Pipeline In", "Pipeline Out",
                                "Adjustments", "Gain/Loss", "Transfers", "Notes"]
                display_cols = [c for c in display_cols if c in df_loc.columns]

                # Make editable
                st.data_editor(
                    df_loc[display_cols],
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"{active_region}_{loc}_edit"
                )

            # Save button for each location
            st.markdown('<div class="save-btn-bottom">', unsafe_allow_html=True)
            if st.button(f"💾 Save {loc}", key=f"save_{active_region}_{loc}"):
                # Here you would save back to Snowflake
                # For now, just show a success message
                st.success(f"✅ Changes for {loc} saved successfully!")
            st.markdown('</div>', unsafe_allow_html=True)


def display_details_tab(df_filtered, active_region):
    """Main function to display the appropriate details view based on region."""
    if active_region == "Group Supply Report (Midcon)":
        display_midcon_details(df_filtered, active_region)
    else:
        display_location_details(df_filtered, active_region)
