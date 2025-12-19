"""data_loader.py

Data Loader module for HF Sinclair Scheduler Dashboard.

This module was originally built to load inventory data from Snowflake.
It now supports switching between:

- Snowflake (default)
- Local CSV (e.g. `testing_data.csv`)

Switching data sources
----------------------
Set one of the following environment variables before running Streamlit:

- `HFS_DATA_SOURCE`:
    - "snowflake" (default)
    - "csv"

- `HFS_INVENTORY_CSV_PATH` (optional): path to the CSV file
    - default: "testing_data.csv"

Example:
    HFS_DATA_SOURCE=csv HFS_INVENTORY_CSV_PATH=testing_data.csv streamlit run app.py

The dataframe returned by `load_inventory_data()` is normalized to the same
schema the rest of the app expects.
"""

from __future__ import annotations

import os
from typing import Final

import pandas as pd
import streamlit as st

from config import RAW_INVENTORY_TABLE

DEFAULT_DATA_SOURCE: Final[str] = "snowflake"
DEFAULT_CSV_PATH: Final[str] = "testing_data.csv"


def _get_data_source() -> str:
    source = (os.getenv("HFS_DATA_SOURCE", "csv") or DEFAULT_DATA_SOURCE).strip().lower()
    if source not in {"snowflake", "csv"}:
        raise ValueError(
            "Invalid HFS_DATA_SOURCE. Expected 'snowflake' or 'csv'. "
            f"Got: {source!r}"
        )
    return source


def _get_csv_path() -> str:
    return (os.getenv("HFS_INVENTORY_CSV_PATH") or DEFAULT_CSV_PATH).strip()


@st.cache_resource(show_spinner=False)
def get_snowflake_session():
    """Get the active Snowflake session.

    Note: Snowflake imports are intentionally *lazy* so that local CSV runs
    do not require the Snowflake runtime.
    """

    from snowflake.snowpark.context import get_active_session  # type: ignore

    return get_active_session()


def _normalize_inventory_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Snowflake/CSV inventory extracts into the app's expected schema."""

    def _col(name: str, default=None) -> pd.Series:
        if name in raw_df.columns:
            return raw_df[name]
        # Ensure we return a series aligned to the dataframe index
        return pd.Series([default] * len(raw_df), index=raw_df.index)

    df_clean = pd.DataFrame(index=raw_df.index)

    # Convert date
    df_clean["Date"] = pd.to_datetime(_col("DATA_DATE"), errors="coerce")

    # Region (Snowflake uses REGION_CODE; csv sample uses REGION_CODE too)
    df_clean["Region"] = _col("REGION_CODE", "Unknown").fillna("Unknown")

    # Map other columns
    df_clean["Location"] = _col("LOCATION_CODE")
    df_clean["Product"] = _col("PRODUCT_DESCRIPTION")

    # CSV has both SOURCE_OPERATOR and SOURCE_SYSTEM/SOURCE_OPERATOR; prefer SOURCE_OPERATOR
    df_clean["System"] = _col("SOURCE_OPERATOR")

    # Numeric columns
    df_clean["Batch In (RECEIPTS_BBL)"] = pd.to_numeric(_col("RECEIPTS_BBL"), errors="coerce").fillna(0)
    df_clean["Batch Out (DELIVERIES_BBL)"] = pd.to_numeric(_col("DELIVERIES_BBL"), errors="coerce").fillna(0)
    df_clean["Rack/Liftings"] = pd.to_numeric(_col("RACK_LIFTINGS_BBL"), errors="coerce").fillna(0)

    df_clean["Close Inv"] = pd.to_numeric(_col("CLOSING_INVENTORY_BBL"), errors="coerce").fillna(0)
    df_clean["Open Inv"] = pd.to_numeric(_col("OPENING_INVENTORY_BBL"), errors="coerce").fillna(0)

    df_clean["Production"] = pd.to_numeric(_col("PRODUCTION_BBL"), errors="coerce").fillna(0)
    df_clean["Pipeline In"] = pd.to_numeric(_col("PIPELINE_IN_BBL"), errors="coerce").fillna(0)
    df_clean["Pipeline Out"] = pd.to_numeric(_col("PIPELINE_OUT_BBL"), errors="coerce").fillna(0)

    # Some sources may store these as strings (Snowflake query used TRY_TO_DOUBLE)
    df_clean["Adjustments"] = pd.to_numeric(_col("ADJUSTMENTS_BBL"), errors="coerce").fillna(0)
    df_clean["Gain/Loss"] = pd.to_numeric(_col("GAIN_LOSS_BBL"), errors="coerce").fillna(0)
    df_clean["Transfers"] = pd.to_numeric(_col("TRANSFERS_BBL"), errors="coerce").fillna(0)

    df_clean["Tank Capacity"] = pd.to_numeric(_col("TANK_CAPACITY_BBL"), errors="coerce").fillna(0)
    df_clean["Safe Fill Limit"] = pd.to_numeric(_col("SAFE_FILL_LIMIT_BBL"), errors="coerce").fillna(0)
    df_clean["Available Space"] = pd.to_numeric(_col("AVAILABLE_SPACE_BBL"), errors="coerce").fillna(0)

    # Identifiers/audit
    df_clean["INVENTORY_KEY"] = _col("INVENTORY_KEY")
    df_clean["SOURCE_FILE_ID"] = _col("SOURCE_FILE_ID")
    df_clean["CREATED_AT"] = pd.to_datetime(_col("CREATED_AT"), errors="coerce")

    # App-specific
    df_clean["Notes"] = ""

    # For Midcon, set System = Location if System is null
    midcon_mask = df_clean["Region"] == "Group Supply Report (Midcon)"
    needs_system = midcon_mask & df_clean["System"].isna()
    df_clean.loc[needs_system, "System"] = df_clean.loc[needs_system, "Location"]

    return df_clean


@st.cache_data(ttl=300, show_spinner=False)
def _load_inventory_data_cached(source: str, csv_path: str) -> pd.DataFrame:
    """Internal cached loader (cache key includes source + csv_path)."""

    if source == "csv":
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV file not found: {csv_path!r}. "
                "Set HFS_INVENTORY_CSV_PATH or place the file in the project root."
            )

        raw_df = pd.read_csv(csv_path)
        # Keep ordering similar to Snowflake query: latest dates first
        if "DATA_DATE" in raw_df.columns:
            raw_df["DATA_DATE"] = pd.to_datetime(raw_df["DATA_DATE"], errors="coerce")
            raw_df = raw_df.sort_values(
                by=["DATA_DATE", "LOCATION_CODE"],
                ascending=[False, True],
                kind="mergesort",
            )

        return _normalize_inventory_df(raw_df)

    # Snowflake
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

    raw_df = session.sql(query).to_pandas()
    return _normalize_inventory_df(raw_df)


def load_inventory_data() -> pd.DataFrame:
    """Load inventory data from the configured source (Snowflake or CSV)."""

    source = _get_data_source()
    csv_path = _get_csv_path()
    return _load_inventory_data_cached(source, csv_path)


def initialize_data():
    """Initialize data loading and store in session state."""

    if "data_loaded" not in st.session_state:
        source = _get_data_source()
        label = "Snowflake" if source == "snowflake" else "CSV"

        with st.spinner(f"Loading inventory data from {label}..."):
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


def ensure_numeric_columns(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns are properly typed."""

    numeric_cols = [
        "Close Inv",
        "Open Inv",
        "Batch In (RECEIPTS_BBL)",
        "Batch Out (DELIVERIES_BBL)",
        "Rack/Liftings",
        "Production",
        "Pipeline In",
        "Pipeline Out",
        "Adjustments",
        "Gain/Loss",
        "Transfers",
        "Tank Capacity",
        "Safe Fill Limit",
        "Available Space",
    ]

    for c in numeric_cols:
        if c in df_filtered.columns:
            df_filtered[c] = pd.to_numeric(df_filtered[c], errors="coerce").fillna(0)

    return df_filtered
