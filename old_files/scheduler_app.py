import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------
# Professional Theme Colors
PRIMARY_BLUE = "#008000"      # Changed to green
ACCENT_GREEN = "#38A169"      # Changed to red (secondary)
WARNING_ORANGE = "#ED8936"
ERROR_RED = "#E53E3E"
BG_LIGHT = "#F5F6FA"
TEXT_DARK = "#2D3748"
CARD_BG = "#FFFFFF"

# ----------------------------
# Hardcoded defaults (used when metric cannot be computed from data)
# You can edit these per Location|Group (Product/System) or per Group.
REQUIRED_MAX_DEFAULTS = {
    # "Houston|ULSD": 18000,
    # "ULSD": 15000,
}
INTRANSIT_DEFAULTS = {
    # "Houston|ULSD": 2500,
    # "ULSD": 2000,
}
GLOBAL_REQUIRED_MAX_FALLBACK = 10000  # used if no specific default found
GLOBAL_INTRANSIT_FALLBACK = 0         # used if no specific default found

# ----------------------------
# Page Setup & CSS
st.set_page_config(
    page_title="HF Sinclair Scheduler Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# st.image("HF-Sinclair.jpg", width=190)


# ...existing code...
st.markdown(f"""
<style>
body {{
    background-color: {BG_LIGHT};
    color: {TEXT_DARK};
    font-family: 'Inter', sans-serif;
}}
.main-header {{
    background-color: {PRIMARY_BLUE};
    color: white;
    text-align: center;
    font-size: 1.7rem;
    font-weight: 600;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1.8rem;
}}
.stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}  /* Increased tab gap */
.stTabs [data-baseweb="tab"] {{
    background-color: {CARD_BG};
    border-radius: 8px 8px 0 0;
    color: {PRIMARY_BLUE};
    font-weight: 600;
    border: 1px solid #E2E8F0;
    padding: 0.1rem 0.8rem;
}}
.stTabs [aria-selected="true"] {{
    background-color: {PRIMARY_BLUE} !important;
    color: white !important;
    border-bottom: 3px solid {ACCENT_GREEN} !important;
}}
div.stButton > button {{
    background: {PRIMARY_BLUE} !important;
    color: white;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    border: none;
    transition: 0.3s;
}}
div.stButton > button:hover {{ opacity: 0.9; }}
.card {{
    background-color: {CARD_BG};
    padding: 0.9rem;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    margin-bottom: 0.6rem;
}}
</style>
""", unsafe_allow_html=True)
# ...existing code...

st.markdown('<div class="main-header">HF Sinclair Scheduler Dashboard</div>', unsafe_allow_html=True)


# ----------------------------
# Region Config
region_files = {
    "Front Range": "Front_Range.csv",
    "Group Supply Report (Midcon)": "Group_Supply_Report_Midcon.csv",
    "Navajo Product System": "Navajo_Product_System.csv",
    "PSR Stock One Drive": "PSR_Stock_One_Drive.csv",
    "WX-Sinclair Supply": "WX_Sinclair_Supply.csv"
}

# ----------------------------
# Load Data
if "data_loaded" not in st.session_state:
    st.session_state.data = {}
    for region, file in region_files.items():
        try:
            df = pd.read_csv(file, parse_dates=["Date"])
        except FileNotFoundError:
            df = pd.DataFrame(columns=[
                "Date", "Location", "Product",
                "Batch In (RECEIPTS_BBL)",
                "Batch Out (DELIVERIES_BBL)",
                "Rack/Liftings", "Close Inv", "Notes"
            ])
        if "Notes" in df.columns:
            df["Notes"] = df["Notes"].fillna("")

        # For Midcon, set System = Location
        if region == "Group Supply Report (Midcon)":
            df["System"] = df["Location"]

        st.session_state.data[region] = df
    st.session_state.data_loaded = True

# ----------------------------
# Sidebar Filters
st.sidebar.header("ðŸ” Filters")
active_region = st.sidebar.selectbox("Select Region", list(region_files.keys()), key="active_region")
df_region = st.session_state.data.get(active_region)

min_date = df_region["Date"].min() if not df_region.empty else pd.Timestamp.today()
max_date = df_region["Date"].max() if not df_region.empty else pd.Timestamp.today()

start_date, end_date = st.sidebar.date_input(
    "Date Range",
    value=(min_date.date(), max_date.date()),
    key=f"date_{active_region}"
)
if isinstance(start_date, (list, tuple)):
    start_date, end_date = start_date
start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)

locations = sorted(df_region["Location"].dropna().unique().tolist()) if "Location" in df_region.columns else []
# Change filter label based on region
if active_region == "Group Supply Report (Midcon)":
    filter_label = "ðŸ“ System"
else:
    filter_label = "ðŸ“ Location"
selected_locs = st.sidebar.multiselect(filter_label, options=locations, default=locations)

subset = df_region[df_region["Location"].isin(selected_locs)] if selected_locs else df_region
products = sorted(subset["Product"].dropna().unique().tolist()) if "Product" in subset.columns else []
selected_prods = st.sidebar.multiselect("ðŸ§ª Product", options=products, default=products)

# ----------------------------
# Mock Source Freshness Data
mock_sources = {
    "PSR Stock One Drive": [
        {"source": "Anacortes Rack", "last_update": "2025-10-29 06:40 AM", "status": "Up to Date"},
        {"source": "PSR Refinery", "last_update": "2025-10-29 07:05 AM", "status": "Up to Date"},
        {"source": "Shell Portland Terminal", "last_update": "2025-10-28 10:50 PM", "status": "Delayed"},
        {"source": "Seaport Tacoma Terminal", "last_update": "2025-10-27 08:00 PM", "status": "Error"}
    ],
    "Navajo Product System": [
        {"source": "Navajo Refinery", "last_update": "2025-10-29 06:00 AM", "status": "Up to Date"},
        {"source": "Magellan El Paso Terminal", "last_update": "2025-10-28 09:30 PM", "status": "Delayed"},
        {"source": "Marathon Albuquerque Terminal", "last_update": "2025-10-27 08:50 PM", "status": "Error"}
    ],
    "Front Range": [
        {"source": "Aurora Terminal", "last_update": "2025-10-29 07:20 AM", "status": "Up to Date"},
        {"source": "Casper Terminal", "last_update": "2025-10-28 09:45 PM", "status": "Delayed"}
    ],
    "WX-Sinclair Supply": [
        {"source": "Woods Cross Refinery", "last_update": "2025-10-29 06:35 AM", "status": "Up to Date"},
        {"source": "Las Vegas Terminal", "last_update": "2025-10-28 10:50 PM", "status": "Delayed"},
        {"source": "Spokane Terminal", "last_update": "2025-10-29 07:10 AM", "status": "Up to Date"}
    ],
    "Group Supply Report (Midcon)": [
        {"source": "Kansas City-Argentine", "last_update": "2025-10-29 07:15 AM", "status": "Up to Date"},
        {"source": "Magellan", "last_update": "2025-10-28 11:10 PM", "status": "Delayed"},
        {"source": "Nustar (East)", "last_update": "2025-10-29 06:45 AM", "status": "Up to Date"}
    ]
}
status_colors = {"Up to Date": ACCENT_GREEN, "Delayed": WARNING_ORANGE, "Error": ERROR_RED}

# ----------------------------
# Data Freshness Cards (shown above the new tabs)
st.subheader("ðŸ“ˆ Data Freshness & Source Status")
region_sources = mock_sources.get(active_region, [])
if region_sources:
    cols = st.columns(len(region_sources))
    for i, src in enumerate(region_sources):
        with cols[i]:
            color = status_colors.get(src["status"], "#A0AEC0")
            st.markdown(f"""
            <div class="card">
                <h4 style="color:{PRIMARY_BLUE}; margin-bottom:0.2rem;">{src['source']}</h4>
                <p style="margin:0; font-size:0.9rem; color:{TEXT_DARK};">
                    Last Updated: <b>{src['last_update']}</b><br>
                    Status: <span style="color:{color}; font-weight:700;">{src['status']}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------
# Apply filters
df_filtered = df_region[(df_region["Date"] >= start_ts) & (df_region["Date"] <= end_ts)].copy() if not df_region.empty else df_region.copy()
if selected_locs and "Location" in df_filtered.columns and len(selected_locs) < len(locations):
    df_filtered = df_filtered[df_filtered["Location"].isin(selected_locs)]
if selected_prods and "Product" in df_filtered.columns and len(selected_prods) < len(products):
    df_filtered = df_filtered[df_filtered["Product"].isin(selected_prods)]

# Ensure numeric columns are numeric
for c in ["Close Inv", "Batch In (RECEIPTS_BBL)", "Batch Out (DELIVERIES_BBL)", "Rack/Liftings"]:
    if c in df_filtered.columns:
        df_filtered[c] = pd.to_numeric(df_filtered[c], errors="coerce")

# ----------------------------
# TOP-LEVEL TABS: Summary | Details
summary_tab, details_tab = st.tabs(["ðŸ“Š Regional Summary", "ðŸ§¾ Details"])

with summary_tab:
    st.subheader("ðŸ“Š Regional Summary")

    sales_cols = [c for c in ["Rack/Liftings", "Batch Out (DELIVERIES_BBL)"] if c in df_filtered.columns]
    sales_col = sales_cols[0] if sales_cols else None

    # For Midcon, group by System and Product; for others, Location and Product
    if active_region == "Group Supply Report (Midcon)":
        group_cols = ["System", "Product"]
    else:
        group_cols = ["Location", "Product"]

    if all(col in df_filtered.columns for col in group_cols) and not df_filtered.empty:
        daily = (
            df_filtered
            .groupby(group_cols + ["Date"], as_index=False)
            .agg({
                "Close Inv": "last",
                "Batch In (RECEIPTS_BBL)": "sum",
                "Batch Out (DELIVERIES_BBL)": "sum",
                "Rack/Liftings": "sum"
            })
        )

        daily["Sales"] = daily[sales_col] if sales_col else np.nan

        latest_date = daily["Date"].max()
        prior_mask = daily["Date"] < latest_date
        prior_day = daily.loc[prior_mask, "Date"].max() if prior_mask.any() else pd.NaT

        def compute_7day_avg(g):
            g = g.sort_values("Date")
            window = g[g["Date"] <= latest_date].tail(7)
            return pd.Series({"Seven_Day_Avg_Sales": window["Sales"].mean(skipna=True)})

        seven_day = daily.groupby(group_cols).apply(compute_7day_avg).reset_index()

        latest = (
            daily[daily["Date"] == latest_date]
            .sort_values(["Date"])
            .groupby(group_cols, as_index=False)
            .last()
        )

        if pd.notna(prior_day):
            pds = (
                daily[daily["Date"] == prior_day]
                .groupby(group_cols, as_index=False)["Sales"].sum()
                .rename(columns={"Sales": "Prior_Day_Sales"})
            )
        else:
            pds = latest[group_cols].copy()
            pds["Prior_Day_Sales"] = np.nan

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

        summary_df = (
            latest[group_cols + ["Close Inv"]]
            .merge(totals, on=group_cols, how="left")
            .merge(pds, on=group_cols, how="left")
            .merge(seven_day, on=group_cols, how="left")
        )

        for c in ["Close Inv", "Total In", "Total Out", "Total Rack", "Prior_Day_Sales", "Seven_Day_Avg_Sales"]:
            if c in summary_df.columns:
                summary_df[c] = pd.to_numeric(summary_df[c], errors="coerce")

        def required_max_for(loc: str, prod: str) -> float:
            key_combo = f"{loc}|{prod}"
            if key_combo in REQUIRED_MAX_DEFAULTS:
                return REQUIRED_MAX_DEFAULTS[key_combo]
            if prod in REQUIRED_MAX_DEFAULTS:
                return REQUIRED_MAX_DEFAULTS[prod]
            return GLOBAL_REQUIRED_MAX_FALLBACK

        def intransit_for(loc: str, prod: str) -> float:
            key_combo = f"{loc}|{prod}"
            if key_combo in INTRANSIT_DEFAULTS:
                return INTRANSIT_DEFAULTS[key_combo]
            if prod in INTRANSIT_DEFAULTS:
                return INTRANSIT_DEFAULTS[prod]
            return GLOBAL_INTRANSIT_FALLBACK

        if active_region == "Group Supply Report (Midcon)":
            summary_df["Required Maximums"] = summary_df.apply(
                lambda r: required_max_for(r["System"], r["Product"]), axis=1
            ).astype(float)
            summary_df["Intransit Bbls"] = summary_df.apply(
                lambda r: intransit_for(r["System"], r["Product"]), axis=1
            ).astype(float)
        else:
            summary_df["Required Maximums"] = summary_df.apply(
                lambda r: required_max_for(r["Location"], r["Product"]), axis=1
            ).astype(float)
            summary_df["Intransit Bbls"] = summary_df.apply(
                lambda r: intransit_for(r["Location"], r["Product"]), axis=1
            ).astype(float)

        summary_df["Gross Inventory"] = (summary_df["Close Inv"].fillna(0) + summary_df["Intransit Bbls"].fillna(0)).astype(float)
        summary_df["Avail. (NET) Inventory"] = (summary_df["Gross Inventory"] - summary_df["Required Maximums"]).astype(float)

        sda = summary_df["Seven_Day_Avg_Sales"].replace({0: np.nan})
        summary_df["# Days Supply"] = (summary_df["Close Inv"] / sda).replace([np.inf, -np.inf], np.nan)

        if active_region == "Group Supply Report (Midcon)":
            display_df = summary_df.rename(columns={
                "System": "System",
                "Product": "Product",
                "Close Inv": "Closing Inv",
                "Prior_Day_Sales": "Prior Day Sales",
                "Seven_Day_Avg_Sales": "7 Day Average"
            })
            desired_order = [
                "System",
                "Product",
                "Closing Inv",
                "Intransit Bbls",
                "Gross Inventory",
                "Required Maximums",
                "Avail. (NET) Inventory",
                "Prior Day Sales",
                "7 Day Average",
                "# Days Supply",
                "Total In",
                "Total Out",
                "Total Rack",
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
                "Location",
                "Product",
                "Closing Inv",
                "Intransit Bbls",
                "Gross Inventory",
                "Required Maximums",
                "Avail. (NET) Inventory",
                "Prior Day Sales",
                "7 Day Average",
                "# Days Supply",
                "Total In",
                "Total Out",
                "Total Rack",
            ]
        final_cols = [c for c in desired_order if c in display_df.columns]
        st.dataframe(display_df[final_cols], use_container_width=True, height=320)
    else:
        empty_cols = [
            "System", "Product", "Closing Inv", "Intransit Bbls", "Gross Inventory", "Required Maximums",
            "Avail. (NET) Inventory", "Prior Day Sales", "7 Day Average", "# Days Supply",
            "Total In", "Total Out", "Total Rack"
        ] if active_region == "Group Supply Report (Midcon)" else [
            "Location", "Product", "Closing Inv", "Intransit Bbls", "Gross Inventory", "Required Maximums",
            "Avail. (NET) Inventory", "Prior Day Sales", "7 Day Average", "# Days Supply",
            "Total In", "Total Out", "Total Rack"
        ]
        st.dataframe(pd.DataFrame(columns=empty_cols), use_container_width=True, height=150)

    # ----------- Forecast Table Below Summary Table -----------
    st.markdown("### ðŸ“ˆ Forecast Table")

    if active_region == "Group Supply Report (Midcon)":
        forecast_data = [
            {
                "System": "Kansas City-Argentine", "Product": "ULSD",
                "Current Inventory": 12000, "EOM Projections": 14000,
                "LIFO Target": 13500, "OPS Target": 13000,
                "EOM vs OPS": 1000, "LIFO vs OPS": 500, "Build and Draw": 2000
            },
            {
                "System": "Kansas City-Argentine", "Product": "Premium",
                "Current Inventory": 9000, "EOM Projections": 9500,
                "LIFO Target": 9200, "OPS Target": 9100,
                "EOM vs OPS": 400, "LIFO vs OPS": 100, "Build and Draw": 500
            },
            {
                "System": "Magellan", "Product": "ULSD",
                "Current Inventory": 8000, "EOM Projections": 8200,
                "LIFO Target": 8100, "OPS Target": 8000,
                "EOM vs OPS": 200, "LIFO vs OPS": 100, "Build and Draw": 300
            },
            {
                "System": "Magellan", "Product": "Premium",
                "Current Inventory": 7000, "EOM Projections": 7300,
                "LIFO Target": 7200, "OPS Target": 7100,
                "EOM vs OPS": 200, "LIFO vs OPS": 100, "Build and Draw": 200
            },
            {
                "System": "Nustar (East)", "Product": "Regular",
                "Current Inventory": 9500, "EOM Projections": 10000,
                "LIFO Target": 9800, "OPS Target": 9700,
                "EOM vs OPS": 300, "LIFO vs OPS": 100, "Build and Draw": 400
            },
            {
                "System": "Nustar (East)", "Product": "Premium",
                "Current Inventory": 8500, "EOM Projections": 8700,
                "LIFO Target": 8600, "OPS Target": 8500,
                "EOM vs OPS": 200, "LIFO vs OPS": 100, "Build and Draw": 200
            }
        ]
        forecast_df = pd.DataFrame(forecast_data)
        forecast_cols = [
            "System", "Product", "Current Inventory", "EOM Projections",
            "LIFO Target", "OPS Target", "EOM vs OPS", "LIFO vs OPS", "Build and Draw"
        ]
    else:
        forecast_data = [
            {
                "Location": "Aurora Terminal", "Product": "ULSD",
                "Current Inventory": 9000, "EOM Projections": 9500,
                "LIFO Target": 9200, "OPS Target": 9100,
                "EOM vs OPS": 400, "LIFO vs OPS": 100, "Build and Draw": 500
            },
            {
                "Location": "Aurora Terminal", "Product": "Premium",
                "Current Inventory": 7000, "EOM Projections": 7300,
                "LIFO Target": 7200, "OPS Target": 7100,
                "EOM vs OPS": 200, "LIFO vs OPS": 100, "Build and Draw": 300
            },
            {
                "Location": "Casper Terminal", "Product": "ULSD",
                "Current Inventory": 8000, "EOM Projections": 8200,
                "LIFO Target": 8100, "OPS Target": 8000,
                "EOM vs OPS": 200, "LIFO vs OPS": 100, "Build and Draw": 200
            },
            {
                "Location": "Casper Terminal", "Product": "Premium",
                "Current Inventory": 6500, "EOM Projections": 6700,
                "LIFO Target": 6600, "OPS Target": 6500,
                "EOM vs OPS": 200, "LIFO vs OPS": 100, "Build and Draw": 200
            },
            {
                "Location": "Aurora Terminal", "Product": "Regular",
                "Current Inventory": 8500, "EOM Projections": 8700,
                "LIFO Target": 8600, "OPS Target": 8500,
                "EOM vs OPS": 200, "LIFO vs OPS": 100, "Build and Draw": 200
            },
            {
                "Location": "Casper Terminal", "Product": "Regular",
                "Current Inventory": 7500, "EOM Projections": 7700,
                "LIFO Target": 7600, "OPS Target": 7500,
                "EOM vs OPS": 200, "LIFO vs OPS": 100, "Build and Draw": 200
            }
        ]
        forecast_df = pd.DataFrame(forecast_data)
        forecast_cols = [
            "Location", "Product", "Current Inventory", "EOM Projections",
            "LIFO Target", "OPS Target", "EOM vs OPS", "LIFO vs OPS", "Build and Draw"
        ]

    st.dataframe(forecast_df[forecast_cols], use_container_width=True, height=320)

with details_tab:
    if active_region == "Group Supply Report (Midcon)":
        st.subheader("ðŸ§¾ Group Daily Details (Editable)")
        df_show = df_filtered.sort_values("Date")
        # Rename Location column to System for Midcon
        df_display = df_show.copy()
        # Drop the System column if it exists (it's a duplicate of Location)
        if "System" in df_display.columns:
            df_display = df_display.drop(columns=["System"])
        # Now rename Location to System
        if "Location" in df_display.columns:
            df_display = df_display.rename(columns={"Location": "System"})
        st.data_editor(df_display, num_rows="dynamic", key=f"{active_region}_edit")
        st.markdown('<div class="save-btn-bottom">', unsafe_allow_html=True)
        if st.button("ðŸ’¾ Save", key=f"save_{active_region}"):
            save_df = st.session_state.data[active_region]
            if "System" in save_df.columns:
                save_df = save_df.drop(columns=["System"])
            safe_name = active_region.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            save_df.to_csv(f"{safe_name}.csv", index=False)
            st.success(f"âœ… Saved changes to {safe_name}.csv")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.subheader("ðŸ­ Locations")
        if "Location" in df_filtered.columns:
            region_locs = sorted(df_filtered["Location"].dropna().unique().tolist())
        else:
            region_locs = []
        if region_locs:
            loc_tabs = st.tabs(region_locs)
            for i, loc in enumerate(region_locs):
                with loc_tabs[i]:
                    st.markdown(f"### ðŸ“ {loc}")
                    df_loc = df_filtered[df_filtered["Location"] == loc].sort_values("Date")
                    if df_loc.empty:
                        st.write("*(No data for this location)*")
                    else:
                        st.data_editor(df_loc, num_rows="dynamic", key=f"{active_region}_{loc}_edit")
                    st.markdown('<div class="save-btn-bottom">', unsafe_allow_html=True)
                    if st.button(f"ðŸ’¾ Save {loc}", key=f"save_{active_region}_{loc}"):
                        save_df = st.session_state.data[active_region]
                        safe_name = active_region.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                        save_df.to_csv(f"{safe_name}.csv", index=False)
                        st.success(f"âœ… Saved changes to {safe_name}.csv")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.write("*(No locations available in the current selection)*")
