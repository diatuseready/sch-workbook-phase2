from __future__ import annotations
import json
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
    SNOWFLAKE_LOCATION_MAPPING_TABLE,
    SNOWFLAKE_WAREHOUSE,
    SNOWFLAKE_WORKBOOK_STAGE,
    SQLITE_DB_PATH,
    SQLITE_SOURCE_STATUS_TABLE,
    SQLITE_TABLE,

    # Base columns
    COL_OPEN_INV_RAW,
    COL_CLOSE_INV_RAW,

    # Free-text column
    COL_BATCH,

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


# ----------------------------------------------------------------------------
# System file download support (Snowflake-only)
# ----------------------------------------------------------------------------

def generate_snowflake_signed_urls(file_paths: list[str], *, expiry_seconds: int = 3600) -> list[dict[str, str]]:
    """Generate Snowflake signed URLs for a list of stage file paths.

    This uses Snowflake's GET_PRESIGNED_URL against the configured stage.
    Only works when DATA_SOURCE == "snowflake".

    Returns a list of {"path": <original>, "url": <signed_url>}.
    """

    if not file_paths:
        return []

    if DATA_SOURCE != "snowflake":
        # Feature explicitly not supported for SQLite/local.
        return []

    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()

    out: list[dict[str, str]] = []
    for p in file_paths:
        path = str(p or "").strip()
        if not path:
            continue

        # Quote path + stage safely.
        stage_sql = "'" + str(SNOWFLAKE_WORKBOOK_STAGE).replace("'", "''") + "'"
        path_sql = "'" + path.replace("'", "''") + "'"
        exp_sql = str(int(expiry_seconds))

        q = f"SELECT GET_PRESIGNED_URL({stage_sql}, {path_sql}, {exp_sql}) AS URL"
        rows = session.sql(q).collect()
        if rows:
            # Snowpark Row behaves like dict/attr; be defensive.
            url = None
            try:
                url = rows[0]["URL"]
            except Exception:
                try:
                    url = rows[0].URL
                except Exception:
                    url = None

            if url:
                out.append({"path": path, "url": str(url)})

    return out


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

    def _parse_file_locations(v) -> list[str]:
        """Parse Snowflake VARIANT array (or its string form) into list[str]."""
        try:
            if v is None:
                return []
            if isinstance(v, float) and pd.isna(v):
                return []

            if isinstance(v, (list, tuple)):
                return [str(x) for x in v if x is not None and str(x).strip()]

            if isinstance(v, str):
                s = v.strip()
                if not s or s.lower() == "null":
                    return []
                if s.startswith("["):
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, list):
                            return [str(x) for x in parsed if x is not None and str(x).strip()]
                    except Exception:
                        # fall through
                        pass
                # Treat as single path.
                return [s]

            # Some connectors may return a dict-like; best-effort.
            if isinstance(v, dict) and "data" in v and isinstance(v.get("data"), list):
                return [str(x) for x in v.get("data") if x is not None and str(x).strip()]

            return [str(v)]
        except Exception:
            return []

    date_raw = _col(raw_df, "OPERATIONAL_DATE")
    if date_raw.isna().all():
        date_raw = _col(raw_df, "DATA_DATE")

    if "DATA_DATE" in raw_df.columns:
        date_raw = date_raw.fillna(_col(raw_df, "DATA_DATE"))
    df["Date"] = pd.to_datetime(date_raw, errors="coerce")
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
    df["SOURCE_TYPE"] = _col(raw_df, "SOURCE_TYPE", "").fillna("")

    # VARIANT array of file paths used for Details -> "View File".
    file_loc_raw = _col(raw_df, "FILE_LOCATION")
    if file_loc_raw.isna().all():
        file_loc_raw = _col(raw_df, "file_location")
    df["FILE_LOCATION"] = file_loc_raw.map(_parse_file_locations)

    df["Notes"] = _col(raw_df, "MANUAL_OVERRIDE_REASON", "").fillna("")

    df[COL_BATCH] = _col(raw_df, "BATCH", "").fillna("").astype(str)

    if "source" in raw_df.columns:
        df["source"] = _col(raw_df, "source", "system").fillna("system")
    else:
        ds = _col(raw_df, "DATA_SOURCE", "").fillna("").astype(str).str.strip().str.lower()

        def _map_source(v: str) -> str:
            if v in {"manual", "forecast", "system"}:
                return v
            return "system"

        df["source"] = ds.map(_map_source)

    if "updated" in raw_df.columns:
        df["updated"] = pd.to_numeric(_col(raw_df, "updated", 0), errors="coerce").fillna(0).astype(int)
    elif "MANUAL_OVERRIDE_FLAG" in raw_df.columns:
        df["updated"] = pd.to_numeric(_col(raw_df, "MANUAL_OVERRIDE_FLAG", 0), errors="coerce").fillna(0).astype(int)
    else:
        df["updated"] = (df["source"].astype(str).str.strip().str.lower().eq("manual")).astype(int)

    # No region-specific overrides.

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
                WHERE OPERATIONAL_DATE = ? AND REGION_CODE = ? AND LOCATION_CODE = ? AND PRODUCT_DESCRIPTION = ?
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
                    OPERATIONAL_DATE,
                    DATA_DATE,
                    REGION_CODE,
                    LOCATION_CODE,
                    PRODUCT_CODE,
                    PRODUCT_DESCRIPTION,
                    DATA_SOURCE,
                    SOURCE_TYPE,
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
                    ?, ?, ?, ?, ?, ?, ?,
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
                    today,
                    region_s,
                    location_s,
                    prod_code,
                    product_s,
                    "manual",
                    "user",
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
            f"WHERE OPERATIONAL_DATE = '{today}' "
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
                OPERATIONAL_DATE,
                DATA_DATE,
                REGION_CODE,
                LOCATION_CODE,
                PRODUCT_CODE,
                PRODUCT_DESCRIPTION,
                DATA_SOURCE,
                SOURCE_TYPE,
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
                {_sql_str(today)},
                {_sql_str(region_s)},
                {_sql_str(location_s)},
                {_sql_str(prod_code)},
                {_sql_str(product_s)},
                {_sql_str('manual')},
                {_sql_str('user')},
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


def _product_code(product_description: str) -> str:
    s = str(product_description or "").strip()
    if not s:
        return "UNKNOWN"
    return s.upper().replace(" ", "_")[:50]


def persist_details_rows(
    df_details: pd.DataFrame,
    *,
    region: str,
    location: str | None,
    system: str | None = None,
    product: str | None = None,
) -> int:
    """Persist the Details-grid rows back to the underlying inventory table.
    """

    if df_details is None or df_details.empty:
        return 0

    region_s = str(region).strip() or "Unknown"
    location_s = None if location in (None, "") else str(location).strip()
    system_s = None if system in (None, "") else str(system).strip()
    product_s_filter = None if product in (None, "") else str(product).strip()

    df = df_details.copy()

    # Optional filter: in Location view we save only the active product tab.
    if product_s_filter and "Product" in df.columns:
        df = df[df["Product"].astype(str) == product_s_filter]
        if df.empty:
            return 0

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Resolve location.
    if location_s is None and "Location" in df.columns:
        locs = df["Location"].dropna().astype(str).unique().tolist()
        if len(locs) == 1:
            location_s = str(locs[0]).strip()

    if not location_s:
        raise ValueError("Location scope is required to save details")

    # Map UI/display columns back to storage columns.
    NUM_MAP = {
        "Opening Inv": "OPENING_INVENTORY_BBL",
        "Close Inv": "CLOSING_INVENTORY_BBL",
        "Receipts": "RECEIPTS_BBL",
        "Deliveries": "DELIVERIES_BBL",
        "Production": "PRODUCTION_BBL",
        "Rack/Lifting": "RACK_LIFTINGS_BBL",
        "Pipeline In": "PIPELINE_IN_BBL",
        "Pipeline Out": "PIPELINE_OUT_BBL",
        "Transfers": "TRANSFERS_BBL",
        "Adjustments": "ADJUSTMENTS_BBL",
        "Gain/Loss": "GAIN_LOSS_BBL",
    }

    FACT_MAP = {
        "Opening Inv Fact": "FACT_OPENING_INVENTORY_BBL",
        "Close Inv Fact": "FACT_CLOSING_INVENTORY_BBL",
        "Receipts Fact": "FACT_RECEIPTS_BBL",
        "Deliveries Fact": "FACT_DELIVERIES_BBL",
        "Production Fact": "FACT_PRODUCTION_BBL",
        "Rack/Lifting Fact": "FACT_RACK_LIFTINGS_BBL",
        "Pipeline In Fact": "FACT_PIPELINE_IN_BBL",
        "Pipeline Out Fact": "FACT_PIPELINE_OUT_BBL",
        "Transfers Fact": "FACT_TRANSFERS_BBL",
        "Adjustments Fact": "FACT_ADJUSTMENTS_BBL",
        "Gain/Loss Fact": "FACT_GAIN_LOSS_BBL",
    }

    def _num(v) -> float:
        try:
            if v is None:
                return 0.0
            return float(pd.to_numeric(pd.Series([v]).astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0.0).iloc[0])
        except Exception:
            return 0.0

    # Build a staging dataframe for upsert.
    rows: list[dict] = []
    for _, r in df.iterrows():
        prod_desc = product_s_filter or (str(r.get("Product") or "").strip() or None)
        if not prod_desc:
            # Details UI always has Product, but don't hard-crash if missing.
            prod_desc = "Unknown"

        src = str(r.get("source") or "system").strip().lower() or "system"
        if src not in {"system", "manual", "forecast"}:
            src = "system"

        date_val = r.get("Date")
        date_s = None if pd.isna(date_val) else str(date_val)

        d = {
            "INVENTORY_KEY": str(uuid4()),
            "OPERATIONAL_DATE": date_s,
            "DATA_DATE": date_s,
            "REGION_CODE": region_s,
            "LOCATION_CODE": location_s,
            "PRODUCT_CODE": _product_code(prod_desc),
            "PRODUCT_DESCRIPTION": prod_desc,
            "DATA_SOURCE": src,
            # When a user saves from the Details grid, they "own" the whole
            # grid for that scope (we persist the full dataframe), so stamp all
            # rows with SOURCE_TYPE='user'.
            "SOURCE_TYPE": "user",
            "MANUAL_OVERRIDE_FLAG": int(_num(r.get("updated", 0)) or 0),
            "MANUAL_OVERRIDE_REASON": str(r.get("Notes") or ""),
            "MANUAL_OVERRIDE_USER": "streamlit_app",
        }

        if COL_BATCH in df.columns:
            d["BATCH"] = str(r.get(COL_BATCH) or "")

        if system_s:
            d["SOURCE_OPERATOR"] = system_s
            d["SOURCE_SYSTEM"] = system_s

        for ui_col, db_col in {**NUM_MAP, **FACT_MAP}.items():
            if ui_col in df.columns:
                d[db_col] = _num(r.get(ui_col))

        rows.append(d)

    if not rows:
        return 0

    # Persist
    if DATA_SOURCE == "sqlite":
        import sqlite3

        conn = sqlite3.connect(SQLITE_DB_PATH)
        try:
            cur = conn.cursor()
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            def _exists_key(op_date: str, prod_desc: str) -> str | None:
                res = cur.execute(
                    f"""
                    SELECT INVENTORY_KEY
                    FROM {SQLITE_TABLE}
                    WHERE COALESCE(OPERATIONAL_DATE, DATA_DATE) = ?
                      AND REGION_CODE = ?
                      AND LOCATION_CODE = ?
                      AND PRODUCT_DESCRIPTION = ?
                    LIMIT 1
                    """,
                    (op_date, region_s, location_s, prod_desc),
                ).fetchone()
                return str(res[0]) if res else None

            for row in rows:
                op_date = row["OPERATIONAL_DATE"]
                prod_desc = row["PRODUCT_DESCRIPTION"]
                existing_key = _exists_key(op_date, prod_desc)

                # Columns we write (common subset across SQLite/Snowflake).
                write_cols = [
                    "OPERATIONAL_DATE",
                    "DATA_DATE",
                    "REGION_CODE",
                    "LOCATION_CODE",
                    "PRODUCT_CODE",
                    "PRODUCT_DESCRIPTION",
                    "DATA_SOURCE",
                    "SOURCE_TYPE",
                    "OPENING_INVENTORY_BBL",
                    "CLOSING_INVENTORY_BBL",
                    "RECEIPTS_BBL",
                    "DELIVERIES_BBL",
                    "PRODUCTION_BBL",
                    "RACK_LIFTINGS_BBL",
                    "PIPELINE_IN_BBL",
                    "PIPELINE_OUT_BBL",
                    "TRANSFERS_BBL",
                    "ADJUSTMENTS_BBL",
                    "GAIN_LOSS_BBL",
                    "FACT_OPENING_INVENTORY_BBL",
                    "FACT_CLOSING_INVENTORY_BBL",
                    "FACT_RECEIPTS_BBL",
                    "FACT_DELIVERIES_BBL",
                    "FACT_PRODUCTION_BBL",
                    "FACT_RACK_LIFTINGS_BBL",
                    "FACT_PIPELINE_IN_BBL",
                    "FACT_PIPELINE_OUT_BBL",
                    "FACT_TRANSFERS_BBL",
                    "FACT_ADJUSTMENTS_BBL",
                    "FACT_GAIN_LOSS_BBL",
                    "MANUAL_OVERRIDE_FLAG",
                    "MANUAL_OVERRIDE_REASON",
                    "MANUAL_OVERRIDE_USER",
                    "BATCH",
                ]

                if system_s:
                    write_cols.insert(4, "SOURCE_OPERATOR")
                    write_cols.insert(5, "SOURCE_SYSTEM")

                for c in write_cols:
                    row.setdefault(c, 0.0 if c.endswith("_BBL") else None)

                if existing_key:
                    set_sql = ", ".join([f"{c}=?" for c in write_cols] + ["UPDATED_AT=?"])
                    params = [row.get(c) for c in write_cols] + [now, existing_key]
                    cur.execute(
                        f"UPDATE {SQLITE_TABLE} SET {set_sql} WHERE INVENTORY_KEY = ?",
                        params,
                    )
                else:
                    insert_cols = ["INVENTORY_KEY"] + write_cols + ["CREATED_AT", "UPDATED_AT"]
                    placeholders = ",".join(["?"] * len(insert_cols))
                    params = [row.get(c) for c in insert_cols]
                    # Fill created/updated at
                    params[-2] = now
                    params[-1] = now
                    cur.execute(
                        f"INSERT INTO {SQLITE_TABLE} ({', '.join(insert_cols)}) VALUES ({placeholders})",
                        params,
                    )

            conn.commit()
        finally:
            conn.close()

    else:
        session = get_snowflake_session()
        session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()

        stage = pd.DataFrame(rows)

        # Build MERGE (Snowflake identifiers are uppercased by default).
        cols = [c for c in stage.columns]
        update_cols = [c for c in cols if c != "INVENTORY_KEY"]

        def _is_nullish(v) -> bool:
            return v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, pd.Timestamp) and pd.isna(v))

        def _sql_literal(v, *, col: str) -> str:
            if _is_nullish(v):
                return "NULL"

            # Dates are passed as strings like YYYY-MM-DD.
            if col in {"OPERATIONAL_DATE", "DATA_DATE"}:
                return "'" + str(v).replace("'", "''") + "'"

            # Most numeric columns in this table are *_BBL plus the override flag.
            if col == "MANUAL_OVERRIDE_FLAG":
                try:
                    return str(int(float(v)))
                except Exception:
                    return "0"

            if col.endswith("_BBL"):
                try:
                    return str(float(v))
                except Exception:
                    return "0"

            # Strings
            return "'" + str(v).replace("'", "''") + "'"

        def _cast_expr(c: str) -> str:
            if c in {"OPERATIONAL_DATE", "DATA_DATE"}:
                return f"TO_DATE({c}) AS {c}"
            if c == "MANUAL_OVERRIDE_FLAG":
                return f"CAST({c} AS NUMBER(1,0)) AS {c}"
            if c.endswith("_BBL"):
                return f"CAST({c} AS DOUBLE) AS {c}"
            return f"CAST({c} AS STRING) AS {c}"

        update_set = ",\n            ".join([f"{c} = s.{c}" for c in update_cols if c not in {"CREATED_AT"}])
        insert_cols = ", ".join(cols + ["CREATED_AT", "UPDATED_AT"])
        insert_vals = ", ".join([f"s.{c}" for c in cols] + ["CURRENT_TIMESTAMP()", "CURRENT_TIMESTAMP()"])

        chunk_size = 500
        total_written = 0
        for start in range(0, len(stage), chunk_size):
            chunk = stage.iloc[start:start + chunk_size]

            values_rows: list[str] = []
            for _, r in chunk.iterrows():
                values_rows.append(
                    "(" + ", ".join(_sql_literal(r.get(c), col=c) for c in cols) + ")"
                )
            values_sql = ",\n                ".join(values_rows)

            cte_cols = ", ".join(cols)
            cast_select = ",\n                    ".join(_cast_expr(c) for c in cols)

            merge_sql = f"""
            MERGE INTO {RAW_INVENTORY_TABLE} t
            USING (
                SELECT
                    {cast_select}
                FROM VALUES
                {values_sql}
                AS s_raw({cte_cols})
            ) s
            ON COALESCE(t.OPERATIONAL_DATE, t.DATA_DATE) = s.OPERATIONAL_DATE
               AND t.REGION_CODE = s.REGION_CODE
               AND t.LOCATION_CODE = s.LOCATION_CODE
               AND t.PRODUCT_DESCRIPTION = s.PRODUCT_DESCRIPTION
            WHEN MATCHED THEN UPDATE SET
                {update_set},
                UPDATED_AT = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT ({insert_cols})
            VALUES ({insert_vals})
            """

            session.sql(merge_sql).collect()
            total_written += int(len(chunk))

        rows_written = total_written

    # Invalidate caches so the updated rows can be fetched.
    _load_inventory_data_cached.clear()
    _load_inventory_data_filtered_cached.clear()
    load_region_filter_metadata.clear()
    load_products_for_admin_scope.clear()
    load_region_location_pairs.clear()
    load_regions.clear()

    return int(locals().get("rows_written", len(rows)))


@st.cache_data(ttl=300, show_spinner=False)
def _load_inventory_data_cached(source: str, sqlite_db_path: str, sqlite_table: str) -> pd.DataFrame:
    if source == "sqlite":
        import sqlite3

        conn = sqlite3.connect(sqlite_db_path)
        # Prefer OPERATIONAL_DATE; fall back to DATA_DATE for older DBs.
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info('{sqlite_table}')").fetchall()}
        # Use COALESCE so we don't drop rows where OPERATIONAL_DATE is NULL.
        date_expr = "COALESCE(OPERATIONAL_DATE, DATA_DATE)" if "OPERATIONAL_DATE" in cols else "DATA_DATE"
        raw_df = pd.read_sql_query(
            f"SELECT * FROM {sqlite_table} WHERE {date_expr} IS NOT NULL ORDER BY {date_expr} DESC, LOCATION_CODE",
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
        OPERATIONAL_DATE,
        DATA_DATE,
        REGION_CODE,
        LOCATION_CODE,
        PRODUCT_DESCRIPTION,
        SOURCE_OPERATOR,
        SOURCE_SYSTEM,
        DATA_SOURCE,
        CAST(COALESCE(BATCH, '') AS STRING) as BATCH,
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
        SOURCE_TYPE,
        TO_JSON(FILE_LOCATION) AS FILE_LOCATION,
        CREATED_AT,
        MANUAL_OVERRIDE_REASON
    FROM {RAW_INVENTORY_TABLE}
    WHERE COALESCE(OPERATIONAL_DATE, DATA_DATE) IS NOT NULL
    ORDER BY COALESCE(OPERATIONAL_DATE, DATA_DATE) DESC, LOCATION_CODE, PRODUCT_CODE
    """

    raw_df = session.sql(query).to_pandas()
    return _normalize_inventory_df(raw_df)


def load_inventory_data() -> pd.DataFrame:
    return _load_inventory_data_cached(DATA_SOURCE, SQLITE_DB_PATH, SQLITE_TABLE)


def _normalize_source_status_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize source status records for UI display."""
    df = raw_df.copy()

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

    if "APP_LOCATION_DESC" in df.columns:
        df["DISPLAY_NAME"] = df["APP_LOCATION_DESC"].fillna("")
    elif "LOCATION" in df.columns:
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
    SELECT DISTINCT
        m.APP_LOCATION_DESC,
        s.CLASS,
        s.LOCATION,
        s.REGION,
        s.SOURCE_OPERATOR,
        s.SOURCE_SYSTEM,
        s.SOURCE_TYPE,
        s.FILE_ID,
        s.INTEGRATION_JOB_ID,
        s.FILE_NAME,
        s.SOURCE_PATH,
        s.PROCESSING_STATUS,
        s.ERROR_MESSAGE,
        s.WARNING_COLUMNS,
        s.RECORD_COUNT,
        s.RECEIVED_TIMESTAMP,
        s.PROCESSED_AT
    FROM {SNOWFLAKE_LOCATION_MAPPING_TABLE} m
    LEFT JOIN {SNOWFLAKE_SOURCE_STATUS_TABLE} s
        ON m.CLASS_NAME = s.CLASS
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

    region_norm = region

    if DATA_SOURCE == "sqlite":
        import sqlite3

        conn = sqlite3.connect(SQLITE_DB_PATH)
        try:
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info('{SQLITE_TABLE}')").fetchall()}
            date_expr = "COALESCE(OPERATIONAL_DATE, DATA_DATE)" if "OPERATIONAL_DATE" in cols else "DATA_DATE"
            loc_sql = f"""
                SELECT DISTINCT LOCATION_CODE AS Location
                FROM {SQLITE_TABLE}
                WHERE {date_expr} IS NOT NULL
                  AND (? IS NULL OR REGION_CODE = ?)
                  AND LOCATION_CODE IS NOT NULL
                ORDER BY Location
            """
            df_locs = pd.read_sql_query(loc_sql, conn, params=[region_norm, region_norm])
            locations = sorted(df_locs["Location"].dropna().astype(str).unique().tolist())

            dates_sql = f"""
                SELECT MIN({date_expr}) AS min_date, MAX({date_expr}) AS max_date
                FROM {SQLITE_TABLE}
                WHERE {date_expr} IS NOT NULL
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

    loc_query = f"""
        SELECT DISTINCT LOCATION_CODE AS Location
        FROM {RAW_INVENTORY_TABLE}
        WHERE COALESCE(OPERATIONAL_DATE, DATA_DATE) IS NOT NULL {region_filter}
          AND LOCATION_CODE IS NOT NULL
        ORDER BY Location
    """
    df_locs = session.sql(loc_query).to_pandas()
    locations = sorted(df_locs["LOCATION"].dropna().astype(str).unique().tolist()) if "LOCATION" in df_locs.columns else []

    date_query = f"""
        SELECT MIN(COALESCE(OPERATIONAL_DATE, DATA_DATE)) AS min_date,
               MAX(COALESCE(OPERATIONAL_DATE, DATA_DATE)) AS max_date
        FROM {RAW_INVENTORY_TABLE}
        WHERE COALESCE(OPERATIONAL_DATE, DATA_DATE) IS NOT NULL {region_filter}
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
    # Backward compatible alias; no normalization.
    return active_region


def create_sidebar_filters(regions: list[str], df_region: pd.DataFrame) -> dict:
    active_region = st.session_state.get("active_region")

    # Location selector
    loc_col = "Location"
    filter_label = "📍 Location"

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
        region=str(active_region or "Unknown"),
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

    region_norm = region

    if source == "sqlite":
        import sqlite3

        start_s = pd.Timestamp(start_ts).strftime("%Y-%m-%d")
        end_s = pd.Timestamp(end_ts).strftime("%Y-%m-%d")

        conn = sqlite3.connect(sqlite_db_path)
        try:
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info('{sqlite_table}')").fetchall()}
        finally:
            conn.close()

        date_expr = "COALESCE(OPERATIONAL_DATE, DATA_DATE)" if "OPERATIONAL_DATE" in cols else "DATA_DATE"

        where = [f"{date_expr} IS NOT NULL", f"{date_expr} >= ?", f"{date_expr} <= ?"]
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
            ORDER BY {date_expr} DESC, LOCATION_CODE
        """

        conn = sqlite3.connect(sqlite_db_path)
        raw_df = pd.read_sql_query(sql, conn, params=params)
        conn.close()

        df = _normalize_inventory_df(raw_df)
        return df

    if source != "snowflake":
        raise ValueError("DATA_SOURCE must be 'snowflake' or 'sqlite'")

    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()

    # Snowflake filter pushdown
    conditions = [
        "COALESCE(OPERATIONAL_DATE, DATA_DATE) IS NOT NULL",
        "COALESCE(OPERATIONAL_DATE, DATA_DATE) >= %(start)s",
        "COALESCE(OPERATIONAL_DATE, DATA_DATE) <= %(end)s",
    ]
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
        OPERATIONAL_DATE,
        DATA_DATE,
        REGION_CODE,
        LOCATION_CODE,
        PRODUCT_DESCRIPTION,
        SOURCE_OPERATOR,
        SOURCE_SYSTEM,
        DATA_SOURCE,
        CAST(COALESCE(BATCH, '') AS STRING) as BATCH,
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
        SOURCE_TYPE,
        TO_JSON(FILE_LOCATION) AS FILE_LOCATION,
        CREATED_AT,
        MANUAL_OVERRIDE_REASON
    FROM {RAW_INVENTORY_TABLE}
    WHERE {where_sql}
    ORDER BY COALESCE(OPERATIONAL_DATE, DATA_DATE) DESC, LOCATION_CODE, PRODUCT_CODE
    """

    # Bind substitution (safe basic string quoting)
    for k, v in binds.items():
        query = query.replace(f"%({k})s", "'" + str(v).replace("'", "''") + "'")

    raw_df = session.sql(query).to_pandas()
    df = _normalize_inventory_df(raw_df)
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
    loc_col = "Location"
    meta = load_region_filter_metadata(region=region, loc_col=loc_col)
    max_date = meta.get("max_date", pd.NaT)

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
        st.warning("Please select a Location before submitting filters.")
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

    region_norm = region

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
