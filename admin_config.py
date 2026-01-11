from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from config import (
    SQLITE_ADMIN_CONFIG_TABLE,
    SNOWFLAKE_ADMIN_CONFIG_TABLE,
    DATA_SOURCE,
    SQLITE_DB_PATH,
    SNOWFLAKE_WAREHOUSE,
)
from data_loader import get_snowflake_session, load_region_location_pairs


def _to_float_or_none(x):
    """Best-effort parse to float; return None if blank/invalid."""
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return None if pd.isna(v) else float(v)


def _to_int_or(x, fallback: int):
    """Best-effort parse to int; return fallback if blank/invalid."""
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return fallback if pd.isna(v) else int(v)


@dataclass(frozen=True)
class Scope:
    region: str
    location: str | None
    product: str | None


DEFAULT_VISIBLE_COLUMNS = [
    "Date",
    "Location",
    "System",
    "Product",
    "Opening Inv",
    "Close Inv",
    "Batch In",
    "Batch Out",
    "Rack/Lifting",
    "Pipeline In",
    "Pipeline Out",
    "Gain/Loss",
    "Transfers",
    "Notes",
]


def _location_key(loc: str | None) -> str:
    return "*" if loc is None or str(loc).strip() == "" else str(loc).strip()


def _product_key(prod: str | None) -> str:
    return "*" if prod is None or str(prod).strip() == "" else str(prod).strip()


def _new_row(*, region: str, location: str | None, product: str | None) -> dict:
    return {
        "REGION": str(region).strip() or "Unknown",
        "LOCATION": _location_key(location),
        "PRODUCT": _product_key(product),
        "VISIBLE_COLUMNS_JSON": json.dumps(DEFAULT_VISIBLE_COLUMNS),
        "BOTTOM": None,
        "SAFEFILL": None,
        "NOTE": None,
        "DEFAULT_START_DAYS": -10,
        "DEFAULT_END_DAYS": 30,
    }


def ensure_admin_config_table_sqlite():
    import sqlite3

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()

    cur.execute(f"PRAGMA table_info({SQLITE_ADMIN_CONFIG_TABLE})")
    existing_info = cur.fetchall()
    existing_cols = {r[1] for r in existing_info}
    pk_cols = [r[1] for r in existing_info if int(r[5] or 0) > 0]  # (cid,name,type,notnull,dflt,pk)

    needs_migration = False
    if existing_info:
        # Old schema did not have PRODUCT and PK did not include it.
        if "PRODUCT" not in existing_cols:
            needs_migration = True
        elif set(pk_cols) == {"REGION", "LOCATION"}:
            needs_migration = True

    if needs_migration:
        old = f"{SQLITE_ADMIN_CONFIG_TABLE}__OLD"
        cur.execute(f"ALTER TABLE {SQLITE_ADMIN_CONFIG_TABLE} RENAME TO {old}")

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SQLITE_ADMIN_CONFIG_TABLE} (
                REGION TEXT NOT NULL,
                LOCATION TEXT NOT NULL,
                PRODUCT TEXT NOT NULL,
                VISIBLE_COLUMNS_JSON TEXT,
                BOTTOM REAL,
                SAFEFILL REAL,
                NOTE TEXT,
                DEFAULT_START_DAYS INTEGER,
                DEFAULT_END_DAYS INTEGER,
                UPDATED_AT TEXT,
                PRIMARY KEY (REGION, LOCATION, PRODUCT)
            )
            """
        )

        cur.execute(f"PRAGMA table_info({old})")
        old_cols = {r[1] for r in cur.fetchall()}
        has_updated = "UPDATED_AT" in old_cols
        if has_updated:
            cur.execute(
                f"""
                INSERT INTO {SQLITE_ADMIN_CONFIG_TABLE}
                (REGION, LOCATION, PRODUCT, VISIBLE_COLUMNS_JSON, BOTTOM, SAFEFILL, NOTE, DEFAULT_START_DAYS, DEFAULT_END_DAYS, UPDATED_AT)
                SELECT
                    REGION,
                    LOCATION,
                    '*',
                    VISIBLE_COLUMNS_JSON,
                    BOTTOM,
                    SAFEFILL,
                    NULL,
                    DEFAULT_START_DAYS,
                    DEFAULT_END_DAYS,
                    UPDATED_AT
                FROM {old}
                """
            )
        else:
            cur.execute(
                f"""
                INSERT INTO {SQLITE_ADMIN_CONFIG_TABLE}
                (REGION, LOCATION, PRODUCT, VISIBLE_COLUMNS_JSON, BOTTOM, SAFEFILL, NOTE, DEFAULT_START_DAYS, DEFAULT_END_DAYS, UPDATED_AT)
                SELECT
                    REGION,
                    LOCATION,
                    '*',
                    VISIBLE_COLUMNS_JSON,
                    BOTTOM,
                    SAFEFILL,
                    NULL,
                    DEFAULT_START_DAYS,
                    DEFAULT_END_DAYS,
                    datetime('now')
                FROM {old}
                """
            )

        cur.execute(f"DROP TABLE {old}")
    else:
        # Fresh install or already migrated.
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SQLITE_ADMIN_CONFIG_TABLE} (
                REGION TEXT NOT NULL,
                LOCATION TEXT NOT NULL,
                PRODUCT TEXT NOT NULL,
                VISIBLE_COLUMNS_JSON TEXT,
                BOTTOM REAL,
                SAFEFILL REAL,
                NOTE TEXT,
                DEFAULT_START_DAYS INTEGER,
                DEFAULT_END_DAYS INTEGER,
                UPDATED_AT TEXT,
                PRIMARY KEY (REGION, LOCATION, PRODUCT)
            )
            """
        )

    cur.execute(f"PRAGMA table_info({SQLITE_ADMIN_CONFIG_TABLE})")
    existing = {r[1] for r in cur.fetchall()}
    desired = {
        "PRODUCT": "TEXT",
        "VISIBLE_COLUMNS_JSON": "TEXT",
        "BOTTOM": "REAL",
        "SAFEFILL": "REAL",
        "NOTE": "TEXT",
        "DEFAULT_START_DAYS": "INTEGER",
        "DEFAULT_END_DAYS": "INTEGER",
        "UPDATED_AT": "TEXT",
    }
    for col, typ in desired.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE {SQLITE_ADMIN_CONFIG_TABLE} ADD COLUMN {col} {typ}")

    conn.commit()
    conn.close()


@st.cache_data(ttl=60, show_spinner=False)
def load_admin_config_df() -> pd.DataFrame:
    cols = [
        "REGION",
        "LOCATION",
        "PRODUCT",
        "VISIBLE_COLUMNS_JSON",
        "BOTTOM",
        "SAFEFILL",
        "NOTE",
        "DEFAULT_START_DAYS",
        "DEFAULT_END_DAYS",
    ]

    if DATA_SOURCE == "sqlite":
        import sqlite3

        ensure_admin_config_table_sqlite()
        conn = sqlite3.connect(SQLITE_DB_PATH)
        df = pd.read_sql_query(f"SELECT {', '.join(cols)} FROM {SQLITE_ADMIN_CONFIG_TABLE}", conn)
        conn.close()
        return df

    # Snowflake: ensure PRODUCT exists and is backfilled to '*'
    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()

    # Best-effort schema evolution.
    try:
        session.sql(f"ALTER TABLE {SNOWFLAKE_ADMIN_CONFIG_TABLE} ADD COLUMN IF NOT EXISTS PRODUCT STRING").collect()
        session.sql(f"ALTER TABLE {SNOWFLAKE_ADMIN_CONFIG_TABLE} ADD COLUMN IF NOT EXISTS NOTE STRING").collect()
        session.sql(f"UPDATE {SNOWFLAKE_ADMIN_CONFIG_TABLE} SET PRODUCT='*' WHERE PRODUCT IS NULL").collect()
    except Exception:
        # Don't block UI if the executing role doesn't have DDL rights.
        pass

    # Coalesce PRODUCT to wildcard for older rows.
    query = (
        f"SELECT REGION, LOCATION, COALESCE(PRODUCT, '*') AS PRODUCT, "
        f"VISIBLE_COLUMNS_JSON, BOTTOM, SAFEFILL, NOTE, DEFAULT_START_DAYS, DEFAULT_END_DAYS "
        f"FROM {SNOWFLAKE_ADMIN_CONFIG_TABLE}"
    )
    return session.sql(query).to_pandas()


def _persist_sqlite(row: dict):
    import sqlite3

    ensure_admin_config_table_sqlite()
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        f"""
        INSERT INTO {SQLITE_ADMIN_CONFIG_TABLE}
        (REGION, LOCATION, PRODUCT, VISIBLE_COLUMNS_JSON, BOTTOM, SAFEFILL, NOTE, DEFAULT_START_DAYS, DEFAULT_END_DAYS, UPDATED_AT)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(REGION, LOCATION, PRODUCT) DO UPDATE SET
            VISIBLE_COLUMNS_JSON=excluded.VISIBLE_COLUMNS_JSON,
            BOTTOM=excluded.BOTTOM,
            SAFEFILL=excluded.SAFEFILL,
            NOTE=excluded.NOTE,
            DEFAULT_START_DAYS=excluded.DEFAULT_START_DAYS,
            DEFAULT_END_DAYS=excluded.DEFAULT_END_DAYS,
            UPDATED_AT=datetime('now')
        """,
        (
            row["REGION"],
            row["LOCATION"],
            row["PRODUCT"],
            row.get("VISIBLE_COLUMNS_JSON"),
            row.get("BOTTOM"),
            row.get("SAFEFILL"),
            row.get("NOTE"),
            row.get("DEFAULT_START_DAYS"),
            row.get("DEFAULT_END_DAYS"),
        ),
    )
    conn.commit()
    conn.close()


def _persist_snowflake(row: dict):
    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()

    # Best-effort schema evolution.
    try:
        session.sql(f"ALTER TABLE {SNOWFLAKE_ADMIN_CONFIG_TABLE} ADD COLUMN IF NOT EXISTS PRODUCT STRING").collect()
        session.sql(f"ALTER TABLE {SNOWFLAKE_ADMIN_CONFIG_TABLE} ADD COLUMN IF NOT EXISTS NOTE STRING").collect()
        session.sql(f"UPDATE {SNOWFLAKE_ADMIN_CONFIG_TABLE} SET PRODUCT='*' WHERE PRODUCT IS NULL").collect()
    except Exception:
        pass

    def _sql_str(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "NULL"
        return "'" + str(v).replace("'", "''") + "'"

    def _sql_num(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "NULL"
        return str(float(v))

    def _sql_int(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "NULL"
        return str(int(v))

    region = _sql_str(row["REGION"])
    location = _sql_str(row["LOCATION"])
    product = _sql_str(row.get("PRODUCT") or "*")
    vis = _sql_str(row.get("VISIBLE_COLUMNS_JSON") or "[]")
    bottom = _sql_num(row.get("BOTTOM"))
    safefill = _sql_num(row.get("SAFEFILL"))
    note = _sql_str(row.get("NOTE"))
    start_days = _sql_int(row.get("DEFAULT_START_DAYS"))
    end_days = _sql_int(row.get("DEFAULT_END_DAYS"))

    sql = f"""
    MERGE INTO {SNOWFLAKE_ADMIN_CONFIG_TABLE} t
    USING (
        SELECT
            {region}::STRING AS REGION,
            {location}::STRING AS LOCATION,
            {product}::STRING AS PRODUCT,
            {vis}::STRING AS VISIBLE_COLUMNS_JSON,
            {bottom}::FLOAT AS BOTTOM,
            {safefill}::FLOAT AS SAFEFILL,
            {note}::STRING AS NOTE,
            {start_days}::INTEGER AS DEFAULT_START_DAYS,
            {end_days}::INTEGER AS DEFAULT_END_DAYS
    ) s
    ON t.REGION = s.REGION AND t.LOCATION = s.LOCATION AND COALESCE(t.PRODUCT, '*') = s.PRODUCT
    WHEN MATCHED THEN UPDATE SET
        VISIBLE_COLUMNS_JSON = s.VISIBLE_COLUMNS_JSON,
        BOTTOM = s.BOTTOM,
        SAFEFILL = s.SAFEFILL,
        NOTE = s.NOTE,
        DEFAULT_START_DAYS = s.DEFAULT_START_DAYS,
        DEFAULT_END_DAYS = s.DEFAULT_END_DAYS,
        UPDATED_AT = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN INSERT (
        REGION, LOCATION, PRODUCT, VISIBLE_COLUMNS_JSON, BOTTOM, SAFEFILL, NOTE, DEFAULT_START_DAYS, DEFAULT_END_DAYS, UPDATED_AT
    ) VALUES (
        s.REGION, s.LOCATION, s.PRODUCT, s.VISIBLE_COLUMNS_JSON, s.BOTTOM, s.SAFEFILL, s.NOTE, s.DEFAULT_START_DAYS, s.DEFAULT_END_DAYS, CURRENT_TIMESTAMP()
    )
    """
    session.sql(sql).collect()


def save_admin_config(*, region: str, location: str | None, product: str | None, updates: dict):
    row = _new_row(region=region, location=location, product=product)
    row.update(updates or {})

    if DATA_SOURCE == "sqlite":
        _persist_sqlite(row)
    else:
        _persist_snowflake(row)

    load_admin_config_df.clear()


def _rows_for_scope(df: pd.DataFrame, scope: Scope) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    region = str(scope.region).strip() or "Unknown"
    loc = _location_key(scope.location)
    prod = _product_key(scope.product)
    return df[(df["REGION"] == region) & (df["LOCATION"] == loc) & (df["PRODUCT"] == prod)].copy()


def get_effective_config(*, region: str, location: str | None, product: str | None = None) -> dict:
    df = load_admin_config_df()

    # Precedence (last wins):
    #   region default            (LOCATION='*', PRODUCT='*')
    #   region + product-only     (LOCATION='*', PRODUCT=prod)
    #   region + location-only    (LOCATION=loc, PRODUCT='*')
    #   region + location+product (LOCATION=loc, PRODUCT=prod)
    # This gives location overrides precedence over product-only overrides when
    # both are present.

    region_default = _rows_for_scope(df, Scope(region=region, location=None, product=None))
    product_only = _rows_for_scope(df, Scope(region=region, location=None, product=product))
    location_only = _rows_for_scope(df, Scope(region=region, location=location, product=None))
    location_product = _rows_for_scope(df, Scope(region=region, location=location, product=product))

    base = _new_row(region=region, location=location, product=product)
    for rows in [region_default, product_only, location_only, location_product]:
        if not rows.empty:
            base.update({k: rows.iloc[-1].get(k) for k in base.keys() if k in rows.columns})

    return base


def get_visible_columns(*, region: str, location: str | None) -> list[str]:
    # Column visibility remains Region/Location scoped (not Product scoped).
    cfg = get_effective_config(region=region, location=location, product=None)
    raw = cfg.get("VISIBLE_COLUMNS_JSON")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return list(DEFAULT_VISIBLE_COLUMNS)

    cols = json.loads(str(raw) or "[]")
    if isinstance(cols, list) and cols:
        return [str(c) for c in cols]
    return list(DEFAULT_VISIBLE_COLUMNS)


def get_default_date_window(*, region: str, location: str | None) -> tuple[int, int]:
    # Date-window defaults remain Region/Location scoped (not Product scoped).
    cfg = get_effective_config(region=region, location=location, product=None)
    s = pd.to_numeric(pd.Series([cfg.get("DEFAULT_START_DAYS")]), errors="coerce").iloc[0]
    e = pd.to_numeric(pd.Series([cfg.get("DEFAULT_END_DAYS")]), errors="coerce").iloc[0]
    start = int(s) if pd.notna(s) else -10
    end = int(e) if pd.notna(e) else 30
    return start, end


def get_threshold_overrides(*, region: str, location: str | None, product: str | None = None) -> dict:
    cfg = get_effective_config(region=region, location=location, product=product)
    return {
        "BOTTOM": cfg.get("BOTTOM"),
        "SAFEFILL": cfg.get("SAFEFILL"),
        "NOTE": cfg.get("NOTE"),
    }


def display_super_admin_panel(*, regions: list[str], active_region: str | None, all_data: pd.DataFrame | None = None):
    """Streamlit UI for super-admin configuration.

    Note: This used to live in `admin_panel.py` but was moved here so the admin
    config storage + UI are in one place.
    """

    st.subheader("🛠️ Super Admin Configuration")

    region = st.selectbox(
        "Region",
        options=regions or ["Unknown"],
        index=(regions.index(active_region) if active_region in (regions or []) else 0),
    )

    # We no longer preload the full inventory table. Use a small distinct query.
    locs: list[str] = []
    try:
        pairs = load_region_location_pairs()
        if pairs is not None and not pairs.empty and "Region" in pairs.columns and "Location" in pairs.columns:
            locs = sorted(pairs[pairs["Region"] == region]["Location"].dropna().astype(str).unique().tolist())
    except Exception:
        # Admin UI should still load even if query fails.
        locs = []

    scope_opts = ["(Region default)"] + locs
    scope = st.selectbox("Location (optional)", options=scope_opts)
    location = None if scope == "(Region default)" else scope

    # Product list depends on Region + optional Location.
    products: list[str] = []
    try:
        from data_loader import load_products_for_admin_scope

        products = load_products_for_admin_scope(region=region, location=location)
    except Exception:
        products = []

    prod_opts = ["(All products)"] + (products or [])
    prod_sel = st.selectbox("Product (optional)", options=prod_opts)
    product = None if prod_sel == "(All products)" else prod_sel

    cfg = get_effective_config(region=region, location=location, product=product)

    if product is not None:
        st.info("Product-scoped rules currently affect thresholds only (Bottom / SafeFill).")
        selected_cols = None
        start_days = _to_int_or(cfg.get("DEFAULT_START_DAYS"), -10)
        end_days = _to_int_or(cfg.get("DEFAULT_END_DAYS"), 30)
    else:
        st.markdown("#### Column Visibility")

        all_cols = sorted(set(DEFAULT_VISIBLE_COLUMNS))
        current_cols = json.loads(str(cfg.get("VISIBLE_COLUMNS_JSON") or "[]"))
        if not isinstance(current_cols, list):
            current_cols = list(DEFAULT_VISIBLE_COLUMNS)

        selected_cols = st.multiselect(
            "Visible columns",
            options=all_cols,
            default=[c for c in current_cols if c in all_cols],
        )

    st.markdown("#### Thresholds")
    c1, c2 = st.columns(2)
    with c1:
        bottom = st.text_input(
            "Bottom",
            value="" if cfg.get("BOTTOM") is None or pd.isna(cfg.get("BOTTOM")) else str(cfg.get("BOTTOM")),
            placeholder="Leave blank for no override",
        )
    with c2:
        safefill = st.text_input(
            "SafeFill",
            value="" if cfg.get("SAFEFILL") is None or pd.isna(cfg.get("SAFEFILL")) else str(cfg.get("SAFEFILL")),
            placeholder="Leave blank for no override",
        )

    note = st.text_input(
        "Note",
        value="" if cfg.get("NOTE") is None or (isinstance(cfg.get("NOTE"), float) and pd.isna(cfg.get("NOTE"))) else str(cfg.get("NOTE")),
        placeholder="Optional note for this scope",
    )

    if product is None:
        st.markdown("#### Default Date Range Selection")
        d1, d2 = st.columns(2)
        with d1:
            start_days = st.number_input(
                "Start offset (days from today)",
                value=_to_int_or(cfg.get("DEFAULT_START_DAYS"), -10),
                step=1,
            )
        with d2:
            end_days = st.number_input(
                "End offset (days from today)",
                value=_to_int_or(cfg.get("DEFAULT_END_DAYS"), 30),
                step=1,
            )

    if st.button("💾 Save configuration"):
        updates = {
            # Only persist these fields at Region/Location scope.
            "VISIBLE_COLUMNS_JSON": (
                json.dumps(selected_cols or DEFAULT_VISIBLE_COLUMNS)
                if product is None
                else cfg.get("VISIBLE_COLUMNS_JSON")
            ),
            "BOTTOM": _to_float_or_none(bottom),
            "SAFEFILL": _to_float_or_none(safefill),
            "NOTE": (str(note).strip() or None),
            "DEFAULT_START_DAYS": int(start_days),
            "DEFAULT_END_DAYS": int(end_days),
        }
        save_admin_config(region=region, location=location, product=product, updates=updates)
        st.success("Saved")

    st.markdown("#### Current stored rows")
    df = load_admin_config_df()
    if df is None or df.empty:
        st.write("(No config rows yet)")
    else:
        sort_cols = [c for c in ["REGION", "LOCATION", "PRODUCT"] if c in df.columns]
        st.dataframe(df.sort_values(sort_cols, kind="mergesort"), width="stretch", height=260)
