"""
Data Loader module for HF Sinclair Scheduler Dashboard
Handles all Snowflake connections and data loading operations
"""

import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session
from config import RAW_INVENTORY_TABLE

@st.cache_resource(show_spinner=False)
def get_snowflake_session():
    """Get the active Snowflake session."""
    return get_active_session()

@st.cache_data(ttl=300, show_spinner=False)
def load_inventory_data():
    """Load inventory data from Snowflake."""
    session = get_snowflake_session()
    
    # Set warehouse
    warehouse_sql = "USE WAREHOUSE HFS_ADHOC_WH"
    session.sql(warehouse_sql).collect()
    
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
    FROM {RAW_INVENTORY_TABLE}
    WHERE DATA_DATE IS NOT NULL
    ORDER BY DATA_DATE DESC, LOCATION_CODE, PRODUCT_CODE
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

def initialize_data():
    """Initialize data loading and store in session state."""
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
    
    return st.session_state.get("regions", [])

def ensure_numeric_columns(df_filtered):
    """Ensure numeric columns are properly typed."""
    numeric_cols = ["Close Inv", "Open Inv", "Batch In (RECEIPTS_BBL)", 
                    "Batch Out (DELIVERIES_BBL)", "Rack/Liftings", 
                    "Production", "Pipeline In", "Pipeline Out",
                    "Adjustments", "Gain/Loss", "Transfers",
                    "Tank Capacity", "Safe Fill Limit", "Available Space"]
    
    for c in numeric_cols:
        if c in df_filtered.columns:
            df_filtered[c] = pd.to_numeric(df_filtered[c], errors="coerce").fillna(0)
    
    return df_filtered
