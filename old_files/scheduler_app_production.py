import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from snowflake.snowpark.context import get_active_session
from functools import lru_cache

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

# Custom CSS styling
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
.stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
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

st.markdown('<div class="main-header">HF Sinclair Scheduler Dashboard</div>', unsafe_allow_html=True)

# ----------------------------
# Snowflake Connection
@st.cache_resource(show_spinner=False)
def get_snowflake_session():
    """Get the active Snowflake session."""
    return get_active_session()

# Get Snowflake session
session = get_snowflake_session()

# Set warehouse
warehouse_sql = "USE WAREHOUSE HFS_ADHOC_WH"
session.sql(warehouse_sql).collect()

# ----------------------------
# Table definitions
raw_inventory_table = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.DAILY_INVENTORY_FACT"


# ----------------------------
# Load Data from Snowflake
@st.cache_data(ttl=300, show_spinner=False)
def load_inventory_data():
    """Load inventory data from Snowflake."""
    query = f"""
    SELECT 
        DATA_DATE,
        REGION_CODE,
        LOCATION_CODE,
        PRODUCT_DESCRIPTION,
        SOURCE_OPERATOR,
        CAST(COALESCE(RECEIPTS_BBL, 0) AS FLOAT) as RECEIPTS_BBL,
        CAST(COALESCE(DELIVERIES_BBL, 0) AS FLOAT) as DELIVERIES_BBL,
        CAST(COALESCE(RACK_LIFTINGS_BBL, 0) AS FLOAT) as RACK_LIFTINGS_BBL,
        CAST(COALESCE(CLOSING_INVENTORY_BBL, 0) AS FLOAT) as CLOSING_INVENTORY_BBL,
        CAST(COALESCE(OPENING_INVENTORY_BBL, 0) AS FLOAT) as OPENING_INVENTORY_BBL,
        CAST(COALESCE(PRODUCTION_BBL, 0) AS FLOAT) as PRODUCTION_BBL,
        CAST(COALESCE(PIPELINE_IN_BBL, 0) AS FLOAT) as PIPELINE_IN_BBL,
        CAST(COALESCE(PIPELINE_OUT_BBL, 0) AS FLOAT) as PIPELINE_OUT_BBL,
        -- Handle VARCHAR columns for ADJUSTMENTS_BBL, GAIN_LOSS_BBL, TRANSFERS_BBL
        CAST(COALESCE(TRY_TO_DOUBLE(ADJUSTMENTS_BBL), 0) AS FLOAT) as ADJUSTMENTS_BBL,
        CAST(COALESCE(TRY_TO_DOUBLE(GAIN_LOSS_BBL), 0) AS FLOAT) as GAIN_LOSS_BBL,
        CAST(COALESCE(TRY_TO_DOUBLE(TRANSFERS_BBL), 0) AS FLOAT) as TRANSFERS_BBL,
        CAST(COALESCE(TANK_CAPACITY_BBL, 0) AS FLOAT) as TANK_CAPACITY_BBL,
        CAST(COALESCE(SAFE_FILL_LIMIT_BBL, 0) AS FLOAT) as SAFE_FILL_LIMIT_BBL,
        CAST(COALESCE(AVAILABLE_SPACE_BBL, 0) AS FLOAT) as AVAILABLE_SPACE_BBL,
        INVENTORY_KEY,
        SOURCE_FILE_ID,
        CREATED_AT
    FROM {raw_inventory_table}
    WHERE DATA_DATE IS NOT NULL
    ORDER BY DATA_DATE DESC, LOCATION_CODE, PRODUCT_DESCRIPTION
    """
    
    df = session.sql(query).to_pandas()
    
    # Debug: Print column names to see what Snowflake returns
    st.write("Debug - Raw columns from Snowflake:", df.columns.tolist())
    
    # Create a clean dataframe with renamed columns
    # Map Snowflake columns to app columns  
    df_clean = pd.DataFrame()
    
    # Convert date
    df_clean["Date"] = pd.to_datetime(df["DATA_DATE"], errors='coerce')
    
    # Map region
    df_clean["Region"] = df["REGION_CODE"].fillna("Unknown")
    
    # Map other columns
    df_clean["Location"] = df["LOCATION_CODE"]
    df_clean["Product"] = df["PRODUCT_DESCRIPTION"]
    df_clean["System"] = df["SOURCE_OPERATOR"]
    df_clean["Batch In (RECEIPTS_BBL)"] = df["RECEIPTS_BBL"]
    df_clean["Batch Out (DELIVERIES_BBL)"] = df["DELIVERIES_BBL"]
    df_clean["Rack/Liftings"] = df["RACK_LIFTINGS_BBL"]
    df_clean["Close Inv"] = df["CLOSING_INVENTORY_BBL"]
    df_clean["Open Inv"] = df["OPENING_INVENTORY_BBL"]
    df_clean["Production"] = df["PRODUCTION_BBL"]
    df_clean["Pipeline In"] = df["PIPELINE_IN_BBL"]
    df_clean["Pipeline Out"] = df["PIPELINE_OUT_BBL"]
    df_clean["Adjustments"] = df["ADJUSTMENTS_BBL"]
    df_clean["Gain/Loss"] = df["GAIN_LOSS_BBL"]
    df_clean["Transfers"] = df["TRANSFERS_BBL"]
    df_clean["Tank Capacity"] = df["TANK_CAPACITY_BBL"]
    df_clean["Safe Fill Limit"] = df["SAFE_FILL_LIMIT_BBL"]
    df_clean["Available Space"] = df["AVAILABLE_SPACE_BBL"]
    df_clean["INVENTORY_KEY"] = df["INVENTORY_KEY"]
    df_clean["SOURCE_FILE_ID"] = df["SOURCE_FILE_ID"]
    df_clean["CREATED_AT"] = df["CREATED_AT"]
    df_clean["Notes"] = ""
    
    # For Midcon, set System = Location if System is null
    midcon_mask = df_clean["Region"] == "Group Supply Report (Midcon)"
    df_clean.loc[midcon_mask & df_clean["System"].isna(), "System"] = df_clean.loc[midcon_mask & df_clean["System"].isna(), "Location"]
    
    df = df_clean
    
    return df


# Load data
if "data_loaded" not in st.session_state:
    with st.spinner("Loading inventory data from Snowflake..."):
        all_data = load_inventory_data()
        
        # Get unique regions dynamically from the data
        regions = sorted(all_data["Region"].dropna().unique().tolist())
        
        # Store regions in session state
        st.session_state.regions = regions
        
        # Split data by region
        st.session_state.data = {}
        for region in regions:
            st.session_state.data[region] = all_data[all_data["Region"] == region].copy()
        
        st.session_state.data_loaded = True
        st.session_state.all_data = all_data
else:
    # Use stored regions from session state
    regions = st.session_state.get("regions", [])

# ----------------------------
# Sidebar Filters
st.sidebar.header("🔍 Filters")

# Ensure we have regions loaded
if "regions" not in st.session_state:
    st.sidebar.warning("Loading regions...")
    st.stop()

regions = st.session_state.regions
active_region = st.sidebar.selectbox("Select Region", regions, key="active_region") if regions else None

if active_region:
    df_region = st.session_state.data.get(active_region, pd.DataFrame())
else:
    df_region = pd.DataFrame()

if not df_region.empty:
    min_date = df_region["Date"].min()
    max_date = df_region["Date"].max()
else:
    min_date = pd.Timestamp.today() - timedelta(days=30)
    max_date = pd.Timestamp.today()

start_date, end_date = st.sidebar.date_input(
    "Date Range",
    value=(min_date.date() if pd.notna(min_date) else date.today(),
           max_date.date() if pd.notna(max_date) else date.today()),
    key=f"date_{active_region}"
)

if isinstance(start_date, (list, tuple)):
    start_date, end_date = start_date
start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)

# Location/System filter
locations = sorted(df_region["Location"].dropna().unique().tolist()) if "Location" in df_region.columns and not df_region.empty else []

# Change filter label based on region
if active_region == "Group Supply Report (Midcon)":
    filter_label = "🏭 System"
else:
    filter_label = "📍 Location"
    
selected_locs = st.sidebar.multiselect(filter_label, options=locations, default=locations[:5] if len(locations) > 5 else locations)

# Product filter
subset = df_region[df_region["Location"].isin(selected_locs)] if selected_locs else df_region
products = sorted(subset["Product"].dropna().unique().tolist()) if "Product" in subset.columns and not subset.empty else []
selected_prods = st.sidebar.multiselect("🧪 Product", options=products, default=products[:5] if len(products) > 5 else products)

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
# Data Freshness Cards
st.subheader("📈 Data Freshness & Source Status")
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
df_filtered = df_region.copy()
if not df_filtered.empty:
    df_filtered = df_filtered[(df_filtered["Date"] >= start_ts) & (df_filtered["Date"] <= end_ts)]
    
    if selected_locs and "Location" in df_filtered.columns and len(selected_locs) < len(locations):
        df_filtered = df_filtered[df_filtered["Location"].isin(selected_locs)]
    
    if selected_prods and "Product" in df_filtered.columns and len(selected_prods) < len(products):
        df_filtered = df_filtered[df_filtered["Product"].isin(selected_prods)]

# Ensure numeric columns are numeric
numeric_cols = ["Close Inv", "Open Inv", "Batch In (RECEIPTS_BBL)", 
                "Batch Out (DELIVERIES_BBL)", "Rack/Liftings", 
                "Production", "Pipeline In", "Pipeline Out",
                "Adjustments", "Gain/Loss", "Transfers",
                "Tank Capacity", "Safe Fill Limit", "Available Space"]

for c in numeric_cols:
    if c in df_filtered.columns:
        df_filtered[c] = pd.to_numeric(df_filtered[c], errors="coerce").fillna(0)

# ----------------------------
# TOP-LEVEL TABS: Summary | Details
summary_tab, details_tab = st.tabs(["📊 Regional Summary", "🧾 Details"])

with summary_tab:
    st.subheader("📊 Regional Summary")
    
    if not df_filtered.empty:
        # Determine sales column
        sales_cols = [c for c in ["Rack/Liftings", "Batch Out (DELIVERIES_BBL)"] if c in df_filtered.columns]
        sales_col = sales_cols[0] if sales_cols else None

        # Group by Location/System and Product
        if active_region == "Group Supply Report (Midcon)":
            group_cols = ["System", "Product"]
        else:
            group_cols = ["Location", "Product"]

        if all(col in df_filtered.columns for col in group_cols):
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
            if not daily.empty:
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

                # Calculate Required Max and Intransit based on historical data or defaults
                def calculate_required_max(row):
                    # Use Tank Capacity * 0.85 as required max, or fall back to default
                    if active_region == "Group Supply Report (Midcon)":
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

                def calculate_intransit(row):
                    # Use Pipeline In data or defaults
                    if active_region == "Group Supply Report (Midcon)":
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

                summary_df["Required Maximums"] = summary_df.apply(calculate_required_max, axis=1).astype(float)
                summary_df["Intransit Bbls"] = summary_df.apply(calculate_intransit, axis=1).astype(float)

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
                    use_container_width=True,
                    height=320
                )
            else:
                st.info("No data available for the selected filters.")
        else:
            st.warning("Required columns not found in the data.")
    else:
        st.info("No data available for the selected region and filters.")

    # Forecast Table
    st.markdown("### 📈 Forecast Table")
    
    if not df_filtered.empty and all(col in df_filtered.columns for col in group_cols):
        # Generate forecast data based on actual data
        forecast_data = []
        
        # Get unique location/system and product combinations
        if active_region == "Group Supply Report (Midcon)":
            unique_combos = df_filtered.groupby(["System", "Product"]).size().reset_index()[["System", "Product"]]
            unique_combos = unique_combos.head(6)  # Limit to first 6 for display
            
            for _, row in unique_combos.iterrows():
                # Calculate forecasts based on historical data
                loc_prod_data = df_filtered[
                    (df_filtered["System"] == row["System"]) &
                    (df_filtered["Product"] == row["Product"])
                ]
                
                current_inv = loc_prod_data["Close Inv"].iloc[-1] if not loc_prod_data.empty else 0
                avg_daily_change = loc_prod_data["Close Inv"].diff().mean() if len(loc_prod_data) > 1 else 100
                
                forecast_data.append({
                    "System": row["System"],
                    "Product": row["Product"],
                    "Current Inventory": round(current_inv, 0),
                    "EOM Projections": round(current_inv + (avg_daily_change * 30), 0),
                    "LIFO Target": round(current_inv * 1.1, 0),
                    "OPS Target": round(current_inv * 1.05, 0),
                    "EOM vs OPS": round((current_inv + (avg_daily_change * 30)) - (current_inv * 1.05), 0),
                    "LIFO vs OPS": round((current_inv * 1.1) - (current_inv * 1.05), 0),
                    "Build and Draw": round(avg_daily_change * 30, 0)
                })
            
            if forecast_data:
                forecast_df = pd.DataFrame(forecast_data)
                forecast_cols = [
                    "System", "Product", "Current Inventory", "EOM Projections",
                    "LIFO Target", "OPS Target", "EOM vs OPS", "LIFO vs OPS", "Build and Draw"
                ]
            else:
                forecast_df = pd.DataFrame()
                forecast_cols = []
        else:
            unique_combos = df_filtered.groupby(["Location", "Product"]).size().reset_index()[["Location", "Product"]]
            unique_combos = unique_combos.head(6)  # Limit to first 6 for display
            
            for _, row in unique_combos.iterrows():
                # Calculate forecasts based on historical data
                loc_prod_data = df_filtered[
                    (df_filtered["Location"] == row["Location"]) &
                    (df_filtered["Product"] == row["Product"])
                ]
                
                current_inv = loc_prod_data["Close Inv"].iloc[-1] if not loc_prod_data.empty else 0
                avg_daily_change = loc_prod_data["Close Inv"].diff().mean() if len(loc_prod_data) > 1 else 100
                
                forecast_data.append({
                    "Location": row["Location"],
                    "Product": row["Product"],
                    "Current Inventory": round(current_inv, 0),
                    "EOM Projections": round(current_inv + (avg_daily_change * 30), 0),
                    "LIFO Target": round(current_inv * 1.1, 0),
                    "OPS Target": round(current_inv * 1.05, 0),
                    "EOM vs OPS": round((current_inv + (avg_daily_change * 30)) - (current_inv * 1.05), 0),
                    "LIFO vs OPS": round((current_inv * 1.1) - (current_inv * 1.05), 0),
                    "Build and Draw": round(avg_daily_change * 30, 0)
                })
            
            if forecast_data:
                forecast_df = pd.DataFrame(forecast_data)
                forecast_cols = [
                    "Location", "Product", "Current Inventory", "EOM Projections",
                    "LIFO Target", "OPS Target", "EOM vs OPS", "LIFO vs OPS", "Build and Draw"
                ]
            else:
                forecast_df = pd.DataFrame()
                forecast_cols = []
        
        if not forecast_df.empty and forecast_cols:
            st.dataframe(forecast_df[forecast_cols], use_container_width=True, height=320)
        else:
            st.info("No forecast data available for the selected filters.")

with details_tab:
    if active_region == "Group Supply Report (Midcon)":
        st.subheader("🧾 Group Daily Details (Editable)")
        
        if not df_filtered.empty:
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
            edited_df = st.data_editor(
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
        else:
            st.info("No data available for the selected filters.")
    else:
        st.subheader("🏭 Locations")
        
        if not df_filtered.empty:
            # Get unique locations
            if "Location" in df_filtered.columns:
                region_locs = sorted(df_filtered["Location"].dropna().unique().tolist())
            else:
                region_locs = []
            
            if region_locs:
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
                            edited_df = st.data_editor(
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
            else:
                st.write("*(No locations available in the current selection)*")
        else:
            st.info("No data available for the selected filters.")
