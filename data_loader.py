from __future__ import annotations

import pandas as pd
import streamlit as st

from config import RAW_INVENTORY_TABLE

DATA_SOURCE = "sqlite"  # "snowflake"
SQLITE_DB_PATH = "inventory.db"
SQLITE_TABLE = "DAILY_INVENTORY_FACT"
SNOWFLAKE_WAREHOUSE = "HFS_ADHOC_WH"

NUMERIC_COLUMN_MAP = {
    "Batch In (RECEIPTS_BBL)": "RECEIPTS_BBL",
    "Batch Out (DELIVERIES_BBL)": "DELIVERIES_BBL",
    "Rack/Liftings": "RACK_LIFTINGS_BBL",
    "Close Inv": "CLOSING_INVENTORY_BBL",
    "Open Inv": "OPENING_INVENTORY_BBL",
    "Production": "PRODUCTION_BBL",
    "Pipeline In": "PIPELINE_IN_BBL",
    "Pipeline Out": "PIPELINE_OUT_BBL",
    "Adjustments": "ADJUSTMENTS_BBL",
    "Gain/Loss": "GAIN_LOSS_BBL",
    "Transfers": "TRANSFERS_BBL",
    "Tank Capacity": "TANK_CAPACITY_BBL",
    "Safe Fill Limit": "SAFE_FILL_LIMIT_BBL",
    "Available Space": "AVAILABLE_SPACE_BBL",
}


def _col(raw_df: pd.DataFrame, name: str, default=None) -> pd.Series:
    if name in raw_df.columns:
        return raw_df[name]
    return pd.Series([default] * len(raw_df), index=raw_df.index)


@st.cache_resource(show_spinner=False)
def get_snowflake_session():
    from snowflake.snowpark.context import get_active_session  # type: ignore

    return get_active_session()


def _normalize_inventory_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=raw_df.index)

    df["Date"] = pd.to_datetime(_col(raw_df, "DATA_DATE"), errors="coerce")
    df["Region"] = _col(raw_df, "REGION_CODE", "Unknown").fillna("Unknown")
    df["Location"] = _col(raw_df, "LOCATION_CODE")
    df["Product"] = _col(raw_df, "PRODUCT_DESCRIPTION")

    system = _col(raw_df, "SOURCE_OPERATOR")
    if system.isna().all():
        system = _col(raw_df, "SOURCE_SYSTEM")
    df["System"] = system

    for out_col, raw_col in NUMERIC_COLUMN_MAP.items():
        df[out_col] = pd.to_numeric(_col(raw_df, raw_col, 0), errors="coerce").fillna(0)

    df["INVENTORY_KEY"] = _col(raw_df, "INVENTORY_KEY")
    df["SOURCE_FILE_ID"] = _col(raw_df, "SOURCE_FILE_ID")
    df["CREATED_AT"] = pd.to_datetime(_col(raw_df, "CREATED_AT"), errors="coerce")

    df["Notes"] = ""

    midcon_mask = df["Region"] == "Group Supply Report (Midcon)"
    needs_system = midcon_mask & df["System"].isna()
    df.loc[needs_system, "System"] = df.loc[needs_system, "Location"]

    return df


@st.cache_data(ttl=300, show_spinner=False)
def _load_inventory_data_cached(source: str, sqlite_db_path: str, sqlite_table: str) -> pd.DataFrame:
    if source == "sqlite":
        import sqlite3

        conn = sqlite3.connect(sqlite_db_path)
        raw_df = pd.read_sql_query(
            f"SELECT * FROM {sqlite_table} WHERE DATA_DATE IS NOT NULL ORDER BY DATA_DATE DESC, LOCATION_CODE",
            conn,
        )
        conn.close()
        return _normalize_inventory_df(raw_df)

    if source != "snowflake":
        raise ValueError("DATA_SOURCE must be 'snowflake' or 'sqlite'")

    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()

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

    raw_df = session.sql(query).to_pandas()
    return _normalize_inventory_df(raw_df)


def load_inventory_data() -> pd.DataFrame:
    return _load_inventory_data_cached(DATA_SOURCE, SQLITE_DB_PATH, SQLITE_TABLE)


def initialize_data():
    if "data_loaded" not in st.session_state:
        label = "Snowflake" if DATA_SOURCE == "snowflake" else "SQLite"

        with st.spinner(f"Loading inventory data from {label}..."):
            all_data = load_inventory_data()
            regions = sorted(all_data["Region"].dropna().unique().tolist())

            st.session_state.regions = regions
            st.session_state.data = {}
            for region in regions:
                st.session_state.data[region] = all_data[all_data["Region"] == region].copy()

            st.session_state.data_loaded = True
            st.session_state.all_data = all_data

    return st.session_state.get("regions", [])


def ensure_numeric_columns(df_filtered: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMN_MAP.keys():
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce").fillna(0)
    return df_filtered
