from __future__ import annotations

import pandas as pd
import streamlit as st

from config import RAW_INVENTORY_TABLE

DATA_SOURCE = "sqlite"  # "snowflake"
SQLITE_DB_PATH = "inventory.db"
# Local dev SQLite table name
SQLITE_TABLE = "APP_INVENTORY"
SQLITE_SOURCE_STATUS_TABLE = "APP_SOURCE_STATUS"
SNOWFLAKE_WAREHOUSE = "HFS_ADHOC_WH"
SNOWFLAKE_SOURCE_STATUS_TABLE = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.APP_SOURCE_STATUS"

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

    # Row lineage tracking (for SQLite we persist these columns; for Snowflake they may not exist)
    df["source"] = _col(raw_df, "source", "system").fillna("system")
    df["updated"] = pd.to_numeric(_col(raw_df, "updated", 0), errors="coerce").fillna(0).astype(int)

    # Midcon rows often use SOURCE_SYSTEM/OPERATOR differently; if System is missing, fall back to Location.
    midcon_mask = df["Region"] == "Midcon"
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


def _normalize_source_status_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize source status records for UI display."""
    df = raw_df.copy()

    # Standardize column names to match CSV/table.
    # (SQLite table uses these exact names; Snowflake equivalent should match.)
    for c in ["RECEIVED_TIMESTAMP", "PROCESSED_AT"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Prefer PROCESSED_AT when present, otherwise RECEIVED_TIMESTAMP
    if "PROCESSED_AT" in df.columns and "RECEIVED_TIMESTAMP" in df.columns:
        df["LAST_UPDATED_AT"] = df["PROCESSED_AT"].fillna(df["RECEIVED_TIMESTAMP"])
    elif "PROCESSED_AT" in df.columns:
        df["LAST_UPDATED_AT"] = df["PROCESSED_AT"]
    elif "RECEIVED_TIMESTAMP" in df.columns:
        df["LAST_UPDATED_AT"] = df["RECEIVED_TIMESTAMP"]
    else:
        df["LAST_UPDATED_AT"] = pd.NaT

    # Human friendly name for cards
    if "LOCATION" in df.columns:
        df["DISPLAY_NAME"] = df["LOCATION"].fillna("")
    else:
        df["DISPLAY_NAME"] = ""

    if "SOURCE_OPERATOR" in df.columns:
        op = df["SOURCE_OPERATOR"].fillna("")
    else:
        op = ""
    if "SOURCE_SYSTEM" in df.columns:
        sys = df["SOURCE_SYSTEM"].fillna("")
    else:
        sys = ""
    # Pick best available label
    df["SOURCE_LABEL"] = op
    if isinstance(sys, pd.Series):
        df.loc[df["SOURCE_LABEL"].astype(str).str.strip().eq(""), "SOURCE_LABEL"] = sys

    # Ensure REGION exists
    if "REGION" not in df.columns:
        df["REGION"] = "Unknown"
    df["REGION"] = df["REGION"].fillna("Unknown")

    # Standardize processing status
    if "PROCESSING_STATUS" in df.columns:
        df["PROCESSING_STATUS"] = df["PROCESSING_STATUS"].fillna("")
    else:
        df["PROCESSING_STATUS"] = ""

    return df


@st.cache_data(ttl=300, show_spinner=False)
def _load_source_status_cached(source: str, sqlite_db_path: str) -> pd.DataFrame:
    if source == "sqlite":
        import sqlite3

        conn = sqlite3.connect(sqlite_db_path)
        raw_df = pd.read_sql_query(
            f"SELECT * FROM {SQLITE_SOURCE_STATUS_TABLE}",
            conn,
        )
        conn.close()
        return _normalize_source_status_df(raw_df)

    if source != "snowflake":
        raise ValueError("DATA_SOURCE must be 'snowflake' or 'sqlite'")

    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()

    query = f"""
    SELECT
        CLASS,
        LOCATION,
        REGION,
        SOURCE_OPERATOR,
        SOURCE_SYSTEM,
        SOURCE_TYPE,
        FILE_ID,
        INTEGRATION_JOB_ID,
        FILE_NAME,
        SOURCE_PATH,
        PROCESSING_STATUS,
        ERROR_MESSAGE,
        WARNING_COLUMNS,
        RECORD_COUNT,
        RECEIVED_TIMESTAMP,
        PROCESSED_AT
    FROM {SNOWFLAKE_SOURCE_STATUS_TABLE}
    """

    raw_df = session.sql(query).to_pandas()
    return _normalize_source_status_df(raw_df)


def load_source_status() -> pd.DataFrame:
    return _load_source_status_cached(DATA_SOURCE, SQLITE_DB_PATH)


def initialize_data():
    if "data_loaded" not in st.session_state:
        label = "Snowflake" if DATA_SOURCE == "snowflake" else "SQLite"

        with st.spinner(f"Loading inventory data from {label}..."):
            all_data = load_inventory_data()
            regions = sorted(all_data["Region"].dropna().unique().tolist())

            # Load and cache source freshness/status
            try:
                st.session_state.source_status = load_source_status()
            except Exception:
                # Don't block the app if status table isn't available
                st.session_state.source_status = pd.DataFrame()

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
