from __future__ import annotations
import pandas as pd
import streamlit as st

from datetime import date, timedelta
from datetime import datetime
from uuid import uuid4

from config import (
    # Data source configuration
    DATA_SOURCE,
    RAW_INVENTORY_TABLE,
    SNOWFLAKE_SOURCE_STATUS_TABLE,
    SNOWFLAKE_WAREHOUSE,
    SQLITE_DB_PATH,
    SQLITE_SOURCE_STATUS_TABLE,
    SQLITE_TABLE,

    # Base columns
    COL_OPEN_INV_RAW,
    COL_CLOSE_INV_RAW,

    # Flow columns
    COL_BATCH_IN_RAW,
    COL_BATCH_OUT_RAW,
    COL_RACK_LIFTINGS_RAW,
    COL_PIPELINE_IN,
    COL_PIPELINE_OUT,
    COL_PRODUCTION,
    COL_ADJUSTMENTS,
    COL_GAIN_LOSS,
    COL_TRANSFERS,

    # Capacity/threshold columns
    COL_TANK_CAPACITY,
    COL_SAFE_FILL_LIMIT,
    COL_AVAILABLE_SPACE,

    # Fact columns (optional UI display)
    COL_OPEN_INV_FACT_RAW,
    COL_CLOSE_INV_FACT_RAW,
    COL_BATCH_IN_FACT_RAW,
    COL_BATCH_OUT_FACT_RAW,
    COL_RACK_LIFTINGS_FACT_RAW,
    COL_PIPELINE_IN_FACT,
    COL_PIPELINE_OUT_FACT,
    COL_PRODUCTION_FACT,
    COL_ADJUSTMENTS_FACT,
    COL_GAIN_LOSS_FACT,
    COL_TRANSFERS_FACT,
)

NUMERIC_COLUMN_MAP = {
    COL_BATCH_IN_RAW: "RECEIPTS_BBL",
    COL_BATCH_OUT_RAW: "DELIVERIES_BBL",
    COL_RACK_LIFTINGS_RAW: "RACK_LIFTINGS_BBL",
    COL_CLOSE_INV_RAW: "CLOSING_INVENTORY_BBL",
    COL_OPEN_INV_RAW: "OPENING_INVENTORY_BBL",
    COL_PRODUCTION: "PRODUCTION_BBL",
    COL_PIPELINE_IN: "PIPELINE_IN_BBL",
    COL_PIPELINE_OUT: "PIPELINE_OUT_BBL",
    COL_ADJUSTMENTS: "ADJUSTMENTS_BBL",
    COL_GAIN_LOSS: "GAIN_LOSS_BBL",
    COL_TRANSFERS: "TRANSFERS_BBL",
    COL_TANK_CAPACITY: "TANK_CAPACITY_BBL",
    COL_SAFE_FILL_LIMIT: "SAFE_FILL_LIMIT_BBL",
    COL_AVAILABLE_SPACE: "AVAILABLE_SPACE_BBL",

    # Fact columns (optional UI display)
    COL_OPEN_INV_FACT_RAW: "FACT_OPENING_INVENTORY_BBL",
    COL_CLOSE_INV_FACT_RAW: "FACT_CLOSING_INVENTORY_BBL",
    COL_BATCH_IN_FACT_RAW: "FACT_RECEIPTS_BBL",
    COL_BATCH_OUT_FACT_RAW: "FACT_DELIVERIES_BBL",
    COL_RACK_LIFTINGS_FACT_RAW: "FACT_RACK_LIFTINGS_BBL",
    COL_PIPELINE_IN_FACT: "FACT_PIPELINE_IN_BBL",
    COL_PIPELINE_OUT_FACT: "FACT_PIPELINE_OUT_BBL",
    COL_PRODUCTION_FACT: "FACT_PRODUCTION_BBL",
    COL_ADJUSTMENTS_FACT: "FACT_ADJUSTMENTS_BBL",
    COL_GAIN_LOSS_FACT: "FACT_GAIN_LOSS_BBL",
    COL_TRANSFERS_FACT: "FACT_TRANSFERS_BBL",
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

    # Notes: best-effort mapping for manual override reason.
    # (SQLite schema includes MANUAL_OVERRIDE_REASON; Snowflake may as well.)
    df["Notes"] = _col(raw_df, "MANUAL_OVERRIDE_REASON", "").fillna("")

    # Row lineage tracking:
    # - If explicit columns exist (e.g. in Snowflake), use them.
    # - Otherwise infer from DATA_SOURCE (we set DATA_SOURCE='manual' for manual inserts).
    if "source" in raw_df.columns:
        df["source"] = _col(raw_df, "source", "system").fillna("system")
    else:
        ds = _col(raw_df, "DATA_SOURCE", "").fillna("").astype(str).str.strip().str.lower()
        df["source"] = ds.map(lambda v: "manual" if v == "manual" else "system")

    if "updated" in raw_df.columns:
        df["updated"] = pd.to_numeric(_col(raw_df, "updated", 0), errors="coerce").fillna(0).astype(int)
    else:
        df["updated"] = (df["source"].astype(str).str.strip().str.lower().eq("manual")).astype(int)

    # Midcon rows often use SOURCE_SYSTEM/OPERATOR differently; if System is missing, fall back to Location.
    midcon_mask = df["Region"] == "Midcon"
    needs_system = midcon_mask & df["System"].isna()
    df.loc[needs_system, "System"] = df.loc[needs_system, "Location"]

    return df


def insert_manual_product_today(
    *,
    region: str,
    location: str,
    product: str,
    opening_inventory_bbl: float,
    closing_inventory_bbl: float,
    note: str,
) -> None:

    region_s = str(region).strip() or "Unknown"
    location_s = str(location).strip()
    product_s = str(product).strip()
    if not location_s:
        raise ValueError("Location is required")
    if not product_s:
        raise ValueError("Product name is required")

    today = date.today().strftime("%Y-%m-%d")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    inv_key = str(uuid4())
    prod_code = product_s.upper().replace(" ", "_")[:50]

    if DATA_SOURCE == "sqlite":
        import sqlite3

        conn = sqlite3.connect(SQLITE_DB_PATH)
        try:
            # Guard against duplicates for today + location + product.
            exists = conn.execute(
                f"""
                SELECT 1
                FROM {SQLITE_TABLE}
                WHERE DATA_DATE = ? AND REGION_CODE = ? AND LOCATION_CODE = ? AND PRODUCT_DESCRIPTION = ?
                LIMIT 1
                """,
                (today, region_s, location_s, product_s),
            ).fetchone()
            if exists:
                raise ValueError("A row for this Region/Location/Product already exists for today")

            conn.execute(
                f"""
                INSERT INTO {SQLITE_TABLE} (
                    INVENTORY_KEY,
                    DATA_DATE,
                    REGION_CODE,
                    LOCATION_CODE,
                    PRODUCT_CODE,
                    PRODUCT_DESCRIPTION,
                    DATA_SOURCE,
                    OPENING_INVENTORY_BBL,
                    CLOSING_INVENTORY_BBL,
                    BATCH,
                    RECEIPTS_BBL,
                    DELIVERIES_BBL,
                    PRODUCTION_BBL,
                    RACK_LIFTINGS_BBL,
                    PIPELINE_IN_BBL,
                    PIPELINE_OUT_BBL,
                    ADJUSTMENTS_BBL,
                    GAIN_LOSS_BBL,
                    TRANSFERS_BBL,
                    REBRANDS_BBL,
                    TANK_CAPACITY_BBL,
                    SAFE_FILL_LIMIT_BBL,
                    BOTTOM_HEEL_BBL,
                    AVAILABLE_SPACE_BBL,
                    LIFTABLE_VOLUME_BBL,
                    DATA_QUALITY_SCORE,
                    VALIDATION_STATUS,
                    MANUAL_OVERRIDE_FLAG,
                    MANUAL_OVERRIDE_USER,
                    MANUAL_OVERRIDE_REASON,
                    CREATED_AT,
                    UPDATED_AT
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?
                )
                """,
                (
                    inv_key,
                    today,
                    region_s,
                    location_s,
                    prod_code,
                    product_s,
                    "manual",
                    float(opening_inventory_bbl or 0.0),
                    float(closing_inventory_bbl or 0.0),
                    0.0,  # BATCH
                    0.0,  # RECEIPTS
                    0.0,  # DELIVERIES
                    0.0,  # PRODUCTION
                    0.0,  # RACK
                    0.0,  # PIPELINE_IN
                    0.0,  # PIPELINE_OUT
                    0.0,  # ADJUSTMENTS
                    0.0,  # GAIN_LOSS
                    0.0,  # TRANSFERS
                    0.0,  # REBRANDS
                    0.0,  # TANK_CAPACITY
                    0.0,  # SAFE_FILL
                    0.0,  # BOTTOM_HEEL
                    0.0,  # AVAILABLE_SPACE
                    0.0,  # LIFTABLE_VOLUME
                    0.0,  # DATA_QUALITY_SCORE
                    0.0,  # VALIDATION_STATUS
                    1.0,  # MANUAL_OVERRIDE_FLAG
                    "super_admin",
                    str(note),
                    now,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    else:
        session = get_snowflake_session()
        session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()

        def _sql_str(v) -> str:
            return "NULL" if v is None else "'" + str(v).replace("'", "''") + "'"

        def _sql_num(v) -> str:
            if v is None:
                return "0"
            return str(float(v))

        # Duplicate check
        dup_q = (
            f"SELECT 1 FROM {RAW_INVENTORY_TABLE} "
            f"WHERE DATA_DATE = '{today}' "
            f"AND REGION_CODE = {_sql_str(region_s)} "
            f"AND LOCATION_CODE = {_sql_str(location_s)} "
            f"AND PRODUCT_DESCRIPTION = {_sql_str(product_s)} "
            f"LIMIT 1"
        )
        if session.sql(dup_q).collect():
            raise ValueError("A row for this Region/Location/Product already exists for today")

        insert_sql = f"""
            INSERT INTO {RAW_INVENTORY_TABLE} (
                INVENTORY_KEY,
                DATA_DATE,
                REGION_CODE,
                LOCATION_CODE,
                PRODUCT_CODE,
                PRODUCT_DESCRIPTION,
                DATA_SOURCE,
                OPENING_INVENTORY_BBL,
                CLOSING_INVENTORY_BBL,
                BATCH,
                RECEIPTS_BBL,
                DELIVERIES_BBL,
                PRODUCTION_BBL,
                RACK_LIFTINGS_BBL,
                PIPELINE_IN_BBL,
                PIPELINE_OUT_BBL,
                ADJUSTMENTS_BBL,
                GAIN_LOSS_BBL,
                TRANSFERS_BBL,
                REBRANDS_BBL,
                TANK_CAPACITY_BBL,
                SAFE_FILL_LIMIT_BBL,
                BOTTOM_HEEL_BBL,
                AVAILABLE_SPACE_BBL,
                MANUAL_OVERRIDE_FLAG,
                MANUAL_OVERRIDE_USER,
                MANUAL_OVERRIDE_REASON,
                CREATED_AT,
                UPDATED_AT
            ) VALUES (
                {_sql_str(inv_key)},
                {_sql_str(today)},
                {_sql_str(region_s)},
                {_sql_str(location_s)},
                {_sql_str(prod_code)},
                {_sql_str(product_s)},
                {_sql_str('manual')},
                {_sql_num(opening_inventory_bbl)},
                {_sql_num(closing_inventory_bbl)},
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                1,
                {_sql_str('super_admin')},
                {_sql_str(note)},
                CURRENT_TIMESTAMP(),
                CURRENT_TIMESTAMP()
            )
        """
        session.sql(insert_sql).collect()

    # Invalidate relevant caches so the new row becomes visible immediately.
    _load_inventory_data_cached.clear()
    _load_inventory_data_filtered_cached.clear()
    load_region_filter_metadata.clear()
    load_products_for_admin_scope.clear()
    load_region_location_pairs.clear()
    load_regions.clear()


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
        SOURCE_SYSTEM,
        DATA_SOURCE,
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
        CAST(COALESCE(FACT_OPENING_INVENTORY_BBL, 0) AS FLOAT) as FACT_OPENING_INVENTORY_BBL,
        CAST(COALESCE(FACT_CLOSING_INVENTORY_BBL, 0) AS FLOAT) as FACT_CLOSING_INVENTORY_BBL,
        CAST(COALESCE(FACT_RECEIPTS_BBL, 0) AS FLOAT) as FACT_RECEIPTS_BBL,
        CAST(COALESCE(FACT_DELIVERIES_BBL, 0) AS FLOAT) as FACT_DELIVERIES_BBL,
        CAST(COALESCE(FACT_PRODUCTION_BBL, 0) AS FLOAT) as FACT_PRODUCTION_BBL,
        CAST(COALESCE(FACT_RACK_LIFTINGS_BBL, 0) AS FLOAT) as FACT_RACK_LIFTINGS_BBL,
        CAST(COALESCE(FACT_PIPELINE_IN_BBL, 0) AS FLOAT) as FACT_PIPELINE_IN_BBL,
        CAST(COALESCE(FACT_PIPELINE_OUT_BBL, 0) AS FLOAT) as FACT_PIPELINE_OUT_BBL,
        CAST(COALESCE(TRY_TO_DOUBLE(FACT_ADJUSTMENTS_BBL), 0) AS FLOAT) as FACT_ADJUSTMENTS_BBL,
        CAST(COALESCE(TRY_TO_DOUBLE(FACT_GAIN_LOSS_BBL), 0) AS FLOAT) as FACT_GAIN_LOSS_BBL,
        CAST(COALESCE(TRY_TO_DOUBLE(FACT_TRANSFERS_BBL), 0) AS FLOAT) as FACT_TRANSFERS_BBL,
        CAST(COALESCE(TANK_CAPACITY_BBL, 0) AS FLOAT) as TANK_CAPACITY_BBL,
        CAST(COALESCE(SAFE_FILL_LIMIT_BBL, 0) AS FLOAT) as SAFE_FILL_LIMIT_BBL,
        CAST(COALESCE(AVAILABLE_SPACE_BBL, 0) AS FLOAT) as AVAILABLE_SPACE_BBL,
        INVENTORY_KEY,
        SOURCE_FILE_ID,
        CREATED_AT,
        MANUAL_OVERRIDE_REASON
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
        # Load and cache source freshness/status
        try:
            st.session_state.source_status = load_source_status()
        except Exception:
            # Don't block the app if status table isn't available
            st.session_state.source_status = pd.DataFrame()

        st.session_state.regions = load_regions()
        st.session_state.data_loaded = True

    return st.session_state.get("regions", [])


@st.cache_data(ttl=300, show_spinner=False)
def load_regions() -> list[str]:
    """Return all regions available in the source (distinct list)."""

    if DATA_SOURCE == "sqlite":
        import sqlite3

        conn = sqlite3.connect(SQLITE_DB_PATH)
        try:
            df = pd.read_sql_query(
                f"SELECT DISTINCT REGION_CODE AS Region FROM {SQLITE_TABLE} WHERE REGION_CODE IS NOT NULL ORDER BY REGION_CODE",
                conn,
            )
        finally:
            conn.close()
        return sorted(df["Region"].dropna().astype(str).unique().tolist())

    # Snowflake
    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()
    # NOTE: Snowflake uppercases unquoted identifiers/aliases, so `AS Region` often
    # comes back as a column named `REGION`. We both (a) quote the alias to
    # preserve case and (b) read the result case-insensitively for robustness.
    df = session.sql(
        f'SELECT DISTINCT REGION_CODE AS "Region" FROM {RAW_INVENTORY_TABLE} '
        f"WHERE REGION_CODE IS NOT NULL ORDER BY REGION_CODE"
    ).to_pandas()

    region_col = None
    if "Region" in df.columns:
        region_col = "Region"
    elif "REGION" in df.columns:
        region_col = "REGION"
    else:
        # Last resort: case-insensitive match (handles connector-specific quirks)
        for c in df.columns:
            if str(c).strip('"').upper() == "REGION":
                region_col = c
                break

    if region_col is None:
        # Avoid crashing the app during initialization; return empty list.
        return []

    return sorted(df[region_col].dropna().astype(str).unique().tolist())


@st.cache_data(ttl=300, show_spinner=False)
def load_region_filter_metadata(*, region: str | None, loc_col: str) -> dict:
    """Return lightweight metadata for sidebar filters: locations/systems + date bounds."""

    region_norm = _normalize_region_label(region) if region else None

    if DATA_SOURCE == "sqlite":
        import sqlite3

        conn = sqlite3.connect(SQLITE_DB_PATH)
        try:
            if loc_col == "System":
                sys_sql = f"""
                    SELECT DISTINCT
                        CASE
                            WHEN SOURCE_OPERATOR IS NOT NULL AND TRIM(SOURCE_OPERATOR) != '' THEN SOURCE_OPERATOR
                            WHEN SOURCE_SYSTEM IS NOT NULL AND TRIM(SOURCE_SYSTEM) != '' THEN SOURCE_SYSTEM
                            ELSE LOCATION_CODE
                        END AS System
                    FROM {SQLITE_TABLE}
                    WHERE DATA_DATE IS NOT NULL
                      AND (? IS NULL OR REGION_CODE = ?)
                    ORDER BY System
                """
                df_locs = pd.read_sql_query(sys_sql, conn, params=[region_norm, region_norm])
                locations = sorted(df_locs["System"].dropna().astype(str).unique().tolist())
            else:
                loc_sql = f"""
                    SELECT DISTINCT LOCATION_CODE AS Location
                    FROM {SQLITE_TABLE}
                    WHERE DATA_DATE IS NOT NULL
                      AND (? IS NULL OR REGION_CODE = ?)
                      AND LOCATION_CODE IS NOT NULL
                    ORDER BY Location
                """
                df_locs = pd.read_sql_query(loc_sql, conn, params=[region_norm, region_norm])
                locations = sorted(df_locs["Location"].dropna().astype(str).unique().tolist())

            dates_sql = f"""
                SELECT MIN(DATA_DATE) AS min_date, MAX(DATA_DATE) AS max_date
                FROM {SQLITE_TABLE}
                WHERE DATA_DATE IS NOT NULL
                  AND (? IS NULL OR REGION_CODE = ?)
            """
            df_dates = pd.read_sql_query(dates_sql, conn, params=[region_norm, region_norm])
        finally:
            conn.close()

        min_date = pd.to_datetime(df_dates.iloc[0]["min_date"], errors="coerce") if not df_dates.empty else pd.NaT
        max_date = pd.to_datetime(df_dates.iloc[0]["max_date"], errors="coerce") if not df_dates.empty else pd.NaT
        return {
            "locations": locations,
            "min_date": min_date,
            "max_date": max_date,
        }

    # Snowflake
    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()
    region_escaped = str(region_norm).replace("'", "''") if region_norm else ""
    region_filter = "" if not region_norm else f" AND REGION_CODE = '{region_escaped}'"

    if loc_col == "System":
        loc_query = f"""
            SELECT DISTINCT
                COALESCE(NULLIF(SOURCE_OPERATOR, ''), NULLIF(SOURCE_SYSTEM, ''), LOCATION_CODE) AS System
            FROM {RAW_INVENTORY_TABLE}
            WHERE DATA_DATE IS NOT NULL {region_filter}
            ORDER BY System
        """
        df_locs = session.sql(loc_query).to_pandas()
        locations = sorted(df_locs["SYSTEM"].dropna().astype(str).unique().tolist()) if "SYSTEM" in df_locs.columns else []
    else:
        loc_query = f"""
            SELECT DISTINCT LOCATION_CODE AS Location
            FROM {RAW_INVENTORY_TABLE}
            WHERE DATA_DATE IS NOT NULL {region_filter}
              AND LOCATION_CODE IS NOT NULL
            ORDER BY Location
        """
        df_locs = session.sql(loc_query).to_pandas()
        locations = sorted(df_locs["LOCATION"].dropna().astype(str).unique().tolist()) if "LOCATION" in df_locs.columns else []

    date_query = f"""
        SELECT MIN(DATA_DATE) AS min_date, MAX(DATA_DATE) AS max_date
        FROM {RAW_INVENTORY_TABLE}
        WHERE DATA_DATE IS NOT NULL {region_filter}
    """
    df_dates = session.sql(date_query).to_pandas()
    min_date = pd.to_datetime(df_dates.iloc[0]["MIN_DATE"], errors="coerce") if not df_dates.empty else pd.NaT
    max_date = pd.to_datetime(df_dates.iloc[0]["MAX_DATE"], errors="coerce") if not df_dates.empty else pd.NaT
    return {"locations": locations, "min_date": min_date, "max_date": max_date}


def ensure_numeric_columns(df_filtered: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMN_MAP.keys():
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce").fillna(0)
    return df_filtered


def _normalize_region_label(active_region: str | None) -> str | None:
    """Normalize UI region labels to match data Region values."""
    if active_region is None:
        return None
    return "Midcon" if active_region == "Group Supply Report (Midcon)" else active_region


def create_sidebar_filters(regions: list[str], df_region: pd.DataFrame) -> dict:
    active_region = st.session_state.get("active_region")

    # Location/System selector (options depend on active_region)
    loc_col = "System" if _normalize_region_label(active_region) == "Midcon" else "Location"
    filter_label = "🏭 System" if loc_col == "System" else "📍 Location"

    if df_region is not None and not df_region.empty and loc_col in df_region.columns:
        locations = sorted(df_region[loc_col].dropna().unique().tolist())
        df_min = df_region["Date"].min() if "Date" in df_region.columns else pd.NaT
        df_max = df_region["Date"].max() if "Date" in df_region.columns else pd.NaT
    else:
        meta = load_region_filter_metadata(region=active_region, loc_col=loc_col)
        locations = meta.get("locations", [])
        df_min = meta.get("min_date", pd.NaT)
        df_max = meta.get("max_date", pd.NaT)

    prev_loc = st.session_state.get("selected_loc")
    if prev_loc is not None and prev_loc not in locations:
        st.session_state.selected_loc = None

    if not locations:
        st.warning("No locations available")
        selected_loc = None
    else:

        current = st.session_state.get("selected_loc")
        index = locations.index(current) if current in locations else 0
        selected_loc = st.selectbox(filter_label, options=locations, index=index, key="selected_loc")
    today = date.today()
    scope_location = None if selected_loc is None else str(selected_loc)
    from admin_config import get_default_date_window

    start_off, end_off = get_default_date_window(
        region=_normalize_region_label(active_region or "Unknown") or "Unknown",
        location=scope_location,
    )

    default_start = today + timedelta(days=int(start_off))
    default_end = today + timedelta(days=int(end_off))

    df_min_d = pd.to_datetime(df_min, errors="coerce").date() if pd.notna(df_min) else default_start
    df_max_d = pd.to_datetime(df_max, errors="coerce").date() if pd.notna(df_max) else default_end

    min_value = min(df_min_d, default_start)
    max_value = max(df_max_d, default_end)

    actual_start = default_start
    actual_end = default_end

    date_range = st.date_input(
        "Date Range",
        value=(actual_start, actual_end),
        min_value=min_value,
        max_value=max_value,
        key=f"date_{active_region}_{scope_location or 'all'}",
    )

    if isinstance(date_range, (list, tuple)):
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range[0] if date_range else date.today()
    else:
        start_date = end_date = date_range

    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)

    return {
        "active_region": active_region,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "selected_loc": selected_loc,
        "loc_col": loc_col,
        "locations": locations,
    }


def apply_filters(df_region: pd.DataFrame, filters: dict) -> pd.DataFrame:
    df_filtered = df_region.copy()

    if df_filtered.empty:
        return df_filtered

    # Apply date filter
    if "Date" in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered["Date"] >= filters["start_ts"]) &
            (df_filtered["Date"] <= filters["end_ts"])
        ]

    # Apply location/system filter
    loc_col = filters.get("loc_col", "Location")
    selected_loc = filters.get("selected_loc")
    if selected_loc and loc_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[loc_col].isin([selected_loc])]

    return df_filtered


@st.cache_data(ttl=300, show_spinner=False)
def _load_inventory_data_filtered_cached(
    source: str,
    sqlite_db_path: str,
    sqlite_table: str,
    *,
    region: str | None,
    loc_col: str,
    selected_loc: str | None,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:

    region_norm = _normalize_region_label(region) if region else None

    if source == "sqlite":
        import sqlite3

        start_s = pd.Timestamp(start_ts).strftime("%Y-%m-%d")
        end_s = pd.Timestamp(end_ts).strftime("%Y-%m-%d")

        where = ["DATA_DATE IS NOT NULL", "DATA_DATE >= ?", "DATA_DATE <= ?"]
        params: list[object] = [start_s, end_s]

        if region_norm:
            where.append("REGION_CODE = ?")
            params.append(region_norm)

        if selected_loc and loc_col == "Location":
            where.append("LOCATION_CODE = ?")
            params.append(str(selected_loc))

        sql = f"""
            SELECT *
            FROM {sqlite_table}
            WHERE {' AND '.join(where)}
            ORDER BY DATA_DATE DESC, LOCATION_CODE
        """

        conn = sqlite3.connect(sqlite_db_path)
        raw_df = pd.read_sql_query(sql, conn, params=params)
        conn.close()

        df = _normalize_inventory_df(raw_df)
        if selected_loc and loc_col == "System":
            df = df[df["System"].astype(str) == str(selected_loc)]
        return df

    if source != "snowflake":
        raise ValueError("DATA_SOURCE must be 'snowflake' or 'sqlite'")

    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()

    # Snowflake filter pushdown
    conditions = ["DATA_DATE IS NOT NULL", "DATA_DATE >= %(start)s", "DATA_DATE <= %(end)s"]
    binds: dict[str, object] = {
        "start": pd.Timestamp(start_ts).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(end_ts).strftime("%Y-%m-%d"),
    }
    if region_norm:
        conditions.append("REGION_CODE = %(region)s")
        binds["region"] = region_norm
    if selected_loc and loc_col == "Location":
        conditions.append("LOCATION_CODE = %(loc)s")
        binds["loc"] = str(selected_loc)

    where_sql = " AND ".join(conditions)
    query = f"""
    SELECT
        DATA_DATE,
        REGION_CODE,
        LOCATION_CODE,
        PRODUCT_DESCRIPTION,
        SOURCE_OPERATOR,
        SOURCE_SYSTEM,
        DATA_SOURCE,
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
        CAST(COALESCE(FACT_OPENING_INVENTORY_BBL, 0) AS FLOAT) as FACT_OPENING_INVENTORY_BBL,
        CAST(COALESCE(FACT_CLOSING_INVENTORY_BBL, 0) AS FLOAT) as FACT_CLOSING_INVENTORY_BBL,
        CAST(COALESCE(FACT_RECEIPTS_BBL, 0) AS FLOAT) as FACT_RECEIPTS_BBL,
        CAST(COALESCE(FACT_DELIVERIES_BBL, 0) AS FLOAT) as FACT_DELIVERIES_BBL,
        CAST(COALESCE(FACT_PRODUCTION_BBL, 0) AS FLOAT) as FACT_PRODUCTION_BBL,
        CAST(COALESCE(FACT_RACK_LIFTINGS_BBL, 0) AS FLOAT) as FACT_RACK_LIFTINGS_BBL,
        CAST(COALESCE(FACT_PIPELINE_IN_BBL, 0) AS FLOAT) as FACT_PIPELINE_IN_BBL,
        CAST(COALESCE(FACT_PIPELINE_OUT_BBL, 0) AS FLOAT) as FACT_PIPELINE_OUT_BBL,
        CAST(COALESCE(TRY_TO_DOUBLE(FACT_ADJUSTMENTS_BBL), 0) AS FLOAT) as FACT_ADJUSTMENTS_BBL,
        CAST(COALESCE(TRY_TO_DOUBLE(FACT_GAIN_LOSS_BBL), 0) AS FLOAT) as FACT_GAIN_LOSS_BBL,
        CAST(COALESCE(TRY_TO_DOUBLE(FACT_TRANSFERS_BBL), 0) AS FLOAT) as FACT_TRANSFERS_BBL,
        CAST(COALESCE(TANK_CAPACITY_BBL, 0) AS FLOAT) as TANK_CAPACITY_BBL,
        CAST(COALESCE(SAFE_FILL_LIMIT_BBL, 0) AS FLOAT) as SAFE_FILL_LIMIT_BBL,
        CAST(COALESCE(AVAILABLE_SPACE_BBL, 0) AS FLOAT) as AVAILABLE_SPACE_BBL,
        INVENTORY_KEY,
        SOURCE_FILE_ID,
        CREATED_AT,
        MANUAL_OVERRIDE_REASON
    FROM {RAW_INVENTORY_TABLE}
    WHERE {where_sql}
    ORDER BY DATA_DATE DESC, LOCATION_CODE, PRODUCT_CODE
    """

    # Bind substitution (safe basic string quoting)
    for k, v in binds.items():
        query = query.replace(f"%({k})s", "'" + str(v).replace("'", "''") + "'")

    raw_df = session.sql(query).to_pandas()
    df = _normalize_inventory_df(raw_df)
    if selected_loc and loc_col == "System":
        df = df[df["System"].astype(str) == str(selected_loc)]
    return df


def load_filtered_inventory_data(filters: dict) -> pd.DataFrame:
    """Load inventory data using filter pushdown."""
    return _load_inventory_data_filtered_cached(
        DATA_SOURCE,
        SQLITE_DB_PATH,
        SQLITE_TABLE,
        region=filters.get("active_region"),
        loc_col=str(filters.get("loc_col") or "Location"),
        selected_loc=(None if filters.get("selected_loc") in (None, "") else str(filters.get("selected_loc"))),
        start_ts=pd.Timestamp(filters.get("start_ts")),
        end_ts=pd.Timestamp(filters.get("end_ts")),
    )


def load_region_inventory_data(*, region: str) -> pd.DataFrame:
    loc_col = "System" if _normalize_region_label(region) == "Midcon" else "Location"
    meta = load_region_filter_metadata(region=region, loc_col=loc_col)
    max_date = meta.get("max_date", pd.NaT)

    # The summary calculations only need a recent window (latest date, prior
    # day, and 7-day average). To keep queries light, we load a bounded slice
    # ending at the region's max date.
    window_days = 90

    if pd.isna(max_date):
        end_ts = pd.Timestamp.today().normalize()
    else:
        end_ts = pd.to_datetime(max_date)

    start_ts = end_ts - pd.Timedelta(days=window_days)

    df = _load_inventory_data_filtered_cached(
        DATA_SOURCE,
        SQLITE_DB_PATH,
        SQLITE_TABLE,
        region=region,
        loc_col=loc_col,
        selected_loc=None,
        start_ts=pd.Timestamp(start_ts),
        end_ts=pd.Timestamp(end_ts),
    )
    return ensure_numeric_columns(df)


def require_selected_location(filters: dict) -> None:
    """Enforce that a location/system must be selected before loading data."""
    if filters.get("selected_loc") in (None, ""):
        st.warning("Please select a Location/System before submitting filters.")
        st.stop()


@st.cache_data(ttl=300, show_spinner=False)
def load_region_location_pairs() -> pd.DataFrame:
    """Small helper for admin UI: distinct Region/Location pairs."""

    if DATA_SOURCE == "sqlite":
        import sqlite3

        conn = sqlite3.connect(SQLITE_DB_PATH)
        try:
            df = pd.read_sql_query(
                f"""
                SELECT DISTINCT
                    REGION_CODE AS Region,
                    LOCATION_CODE AS Location
                FROM {SQLITE_TABLE}
                WHERE REGION_CODE IS NOT NULL
                  AND LOCATION_CODE IS NOT NULL
                """,
                conn,
            )
        finally:
            conn.close()
        return df

    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()
    query = f"""
        SELECT DISTINCT
            REGION_CODE AS "Region",
            LOCATION_CODE AS "Location"
        FROM {RAW_INVENTORY_TABLE}
        WHERE REGION_CODE IS NOT NULL
          AND LOCATION_CODE IS NOT NULL
    """
    df = session.sql(query).to_pandas()

    # Same Snowflake alias-casing issue: be defensive and normalize columns.
    if "Region" not in df.columns and "REGION" in df.columns:
        df = df.rename(columns={"REGION": "Region"})
    if "Location" not in df.columns and "LOCATION" in df.columns:
        df = df.rename(columns={"LOCATION": "Location"})

    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_products_for_admin_scope(*, region: str, location: str | None) -> list[str]:
    """Small helper for admin UI: distinct products for Region (+ optional Location)."""

    region_norm = _normalize_region_label(region) if region else None

    if DATA_SOURCE == "sqlite":
        import sqlite3

        conn = sqlite3.connect(SQLITE_DB_PATH)
        try:
            where = ["REGION_CODE IS NOT NULL", "PRODUCT_DESCRIPTION IS NOT NULL"]
            params: list[object] = []
            if region_norm:
                where.append("REGION_CODE = ?")
                params.append(region_norm)
            if location is not None and str(location).strip() != "":
                where.append("LOCATION_CODE = ?")
                params.append(str(location))

            sql = f"""
                SELECT DISTINCT PRODUCT_DESCRIPTION AS Product
                FROM {SQLITE_TABLE}
                WHERE {' AND '.join(where)}
                ORDER BY Product
            """
            df = pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()
        return sorted(df["Product"].dropna().astype(str).unique().tolist()) if not df.empty else []

    # Snowflake
    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()
    region_escaped = str(region_norm).replace("'", "''") if region_norm else ""
    region_filter = "" if not region_norm else f" AND REGION_CODE = '{region_escaped}'"
    loc_escaped = str(location).replace("'", "''") if location not in (None, "") else ""
    loc_filter = "" if location in (None, "") else f" AND LOCATION_CODE = '{loc_escaped}'"
    query = f"""
        SELECT DISTINCT PRODUCT_DESCRIPTION AS Product
        FROM {RAW_INVENTORY_TABLE}
        WHERE PRODUCT_DESCRIPTION IS NOT NULL {region_filter} {loc_filter}
        ORDER BY Product
    """
    df = session.sql(query).to_pandas()
    # Snowflake column case may be upper.
    col = "PRODUCT" if "PRODUCT" in df.columns else "Product"
    return sorted(df[col].dropna().astype(str).unique().tolist()) if not df.empty and col in df.columns else []
