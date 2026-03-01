from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from app_logging import logged_button, log_audit, log_error

from config import (
    SQLITE_ADMIN_CONFIG_TABLE,
    SNOWFLAKE_ADMIN_CONFIG_TABLE,
    DATA_SOURCE,
    SQLITE_DB_PATH,
    SNOWFLAKE_WAREHOUSE,
    RACK_LIFTING_FORECAST_METHOD_DEFAULT,
    RACK_LIFTING_FORECAST_METHODS,
    INPUT_INCOMING_COLS,
    INPUT_OUTGOING_COLS,
    INPUT_ADJUSTMENT_COLS,
    CALCULATED_COLS,
    MISC_COLS,
)
from data_loader import get_snowflake_session, load_region_location_pairs


def _to_float_or_none(x):
    """Best-effort parse to float; return None if blank/invalid."""
    if isinstance(x, str):
        x = x.replace(",", "").strip()
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


# ---------------------------------------------------------------------------
# Column group mapping for the admin column-order editor
# ---------------------------------------------------------------------------

_ADMIN_GROUP_MAP: dict[str, str] = {}
for _col in INPUT_INCOMING_COLS:
    _ADMIN_GROUP_MAP[str(_col)] = "Input – Incoming"
for _col in INPUT_OUTGOING_COLS:
    _ADMIN_GROUP_MAP[str(_col)] = "Input – Outgoing"
for _col in INPUT_ADJUSTMENT_COLS:
    _ADMIN_GROUP_MAP[str(_col)] = "Input – Adjustment"
for _col in CALCULATED_COLS:
    _ADMIN_GROUP_MAP[str(_col)] = "Calculated"
for _col in MISC_COLS:
    _ADMIN_GROUP_MAP[str(_col)] = "Misc"
_ADMIN_GROUP_MAP["View File"] = "Misc"

# All columns the admin can configure (Date is always forced first — excluded here)
_ALL_CONFIGURABLE_COLS: list[str] = [
    # Calculated
    "Opening Inv", "Close Inv", "Total Closing Inv", "Available Space",
    "Loadable", "Total Inventory", "Accounting Inventory", "7 Day Avg", "MTD Avg",
    # Input – Incoming
    "Receipts", "Pipeline In", "Production",
    # Input – Outgoing
    "Deliveries", "Rack/Lifting", "Pipeline Out",
    # Input – Adjustment
    "Adjustments", "Gain/Loss", "Transfers",
    # Misc
    "Available", "Intransit", "Storage",
    "View File", "Batch", "Batch Breakdown", "Notes",
    "Tulsa", "El Dorado", "Other", "Argentine", "From 327 Receipt",
]

# ---------------------------------------------------------------------------
# Default visible columns (Date is always first; admins can customize the rest)
# ---------------------------------------------------------------------------

DEFAULT_VISIBLE_COLUMNS = [
    "Date",
    "Opening Inv",
    "Available",
    "Intransit",
    "Close Inv",
    "Total Closing Inv",
    "Available Space",
    "Loadable",
    "Total Inventory",
    "Storage",
    "Accounting Inventory",
    "View File",
    "Receipts",
    "Deliveries",
    "Rack/Lifting",
    "7 Day Avg",
    "MTD Avg",
    "Pipeline In",
    "Pipeline Out",
    "Gain/Loss",
    "Transfers",
    "Production",
    "Adjustments",
    "Tulsa",
    "El Dorado",
    "Other",
    "Argentine",
    "From 327 Receipt",
    "Batch",
    "Batch Breakdown",
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
        "RACK_LIFTING_FORECAST_METHOD": RACK_LIFTING_FORECAST_METHOD_DEFAULT,
    }


def ensure_admin_config_table_sqlite():
    import sqlite3

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()

    cur.execute(f"PRAGMA table_info({SQLITE_ADMIN_CONFIG_TABLE})")
    existing_info = cur.fetchall()
    existing_cols = {r[1] for r in existing_info}
    pk_cols = [r[1] for r in existing_info if int(r[5] or 0) > 0]

    needs_migration = False
    if existing_info:
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
                RACK_LIFTING_FORECAST_METHOD TEXT,
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
                (REGION, LOCATION, PRODUCT, VISIBLE_COLUMNS_JSON, BOTTOM, SAFEFILL, NOTE,
                 DEFAULT_START_DAYS, DEFAULT_END_DAYS, RACK_LIFTING_FORECAST_METHOD, UPDATED_AT)
                SELECT REGION, LOCATION, '*', VISIBLE_COLUMNS_JSON, BOTTOM, SAFEFILL, NULL,
                       DEFAULT_START_DAYS, DEFAULT_END_DAYS,
                       '{RACK_LIFTING_FORECAST_METHOD_DEFAULT}', UPDATED_AT
                FROM {old}
                """
            )
        else:
            cur.execute(
                f"""
                INSERT INTO {SQLITE_ADMIN_CONFIG_TABLE}
                (REGION, LOCATION, PRODUCT, VISIBLE_COLUMNS_JSON, BOTTOM, SAFEFILL, NOTE,
                 DEFAULT_START_DAYS, DEFAULT_END_DAYS, RACK_LIFTING_FORECAST_METHOD, UPDATED_AT)
                SELECT REGION, LOCATION, '*', VISIBLE_COLUMNS_JSON, BOTTOM, SAFEFILL, NULL,
                       DEFAULT_START_DAYS, DEFAULT_END_DAYS,
                       '{RACK_LIFTING_FORECAST_METHOD_DEFAULT}', datetime('now')
                FROM {old}
                """
            )

        cur.execute(f"DROP TABLE {old}")
    else:
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
                RACK_LIFTING_FORECAST_METHOD TEXT,
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
        "RACK_LIFTING_FORECAST_METHOD": "TEXT",
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
        "REGION", "LOCATION", "PRODUCT", "VISIBLE_COLUMNS_JSON",
        "BOTTOM", "SAFEFILL", "NOTE", "DEFAULT_START_DAYS",
        "DEFAULT_END_DAYS", "RACK_LIFTING_FORECAST_METHOD",
    ]

    if DATA_SOURCE == "sqlite":
        import sqlite3

        ensure_admin_config_table_sqlite()
        conn = sqlite3.connect(SQLITE_DB_PATH)
        df = pd.read_sql_query(f"SELECT {', '.join(cols)} FROM {SQLITE_ADMIN_CONFIG_TABLE}", conn)
        conn.close()
        return df

    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()

    try:
        session.sql(f"ALTER TABLE {SNOWFLAKE_ADMIN_CONFIG_TABLE} ADD COLUMN IF NOT EXISTS PRODUCT STRING").collect()
        session.sql(f"ALTER TABLE {SNOWFLAKE_ADMIN_CONFIG_TABLE} ADD COLUMN IF NOT EXISTS NOTE STRING").collect()
        session.sql(
            f"ALTER TABLE {SNOWFLAKE_ADMIN_CONFIG_TABLE} ADD COLUMN IF NOT EXISTS RACK_LIFTING_FORECAST_METHOD STRING"
        ).collect()
        session.sql(f"UPDATE {SNOWFLAKE_ADMIN_CONFIG_TABLE} SET PRODUCT='*' WHERE PRODUCT IS NULL").collect()
        session.sql(
            f"UPDATE {SNOWFLAKE_ADMIN_CONFIG_TABLE} "
            f"SET RACK_LIFTING_FORECAST_METHOD='{RACK_LIFTING_FORECAST_METHOD_DEFAULT}' "
            f"WHERE RACK_LIFTING_FORECAST_METHOD IS NULL"
        ).collect()
    except Exception:
        pass

    query = (
        f"SELECT REGION, LOCATION, COALESCE(PRODUCT, '*') AS PRODUCT, "
        f"VISIBLE_COLUMNS_JSON, BOTTOM, SAFEFILL, NOTE, DEFAULT_START_DAYS, DEFAULT_END_DAYS, "
        f"COALESCE(RACK_LIFTING_FORECAST_METHOD, '{RACK_LIFTING_FORECAST_METHOD_DEFAULT}') "
        f"AS RACK_LIFTING_FORECAST_METHOD "
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
        (REGION, LOCATION, PRODUCT, VISIBLE_COLUMNS_JSON, BOTTOM, SAFEFILL, NOTE,
         DEFAULT_START_DAYS, DEFAULT_END_DAYS, RACK_LIFTING_FORECAST_METHOD, UPDATED_AT)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(REGION, LOCATION, PRODUCT) DO UPDATE SET
            VISIBLE_COLUMNS_JSON=excluded.VISIBLE_COLUMNS_JSON,
            BOTTOM=excluded.BOTTOM,
            SAFEFILL=excluded.SAFEFILL,
            NOTE=excluded.NOTE,
            DEFAULT_START_DAYS=excluded.DEFAULT_START_DAYS,
            DEFAULT_END_DAYS=excluded.DEFAULT_END_DAYS,
            RACK_LIFTING_FORECAST_METHOD=excluded.RACK_LIFTING_FORECAST_METHOD,
            UPDATED_AT=datetime('now')
        """,
        (
            row["REGION"], row["LOCATION"], row["PRODUCT"],
            row.get("VISIBLE_COLUMNS_JSON"), row.get("BOTTOM"), row.get("SAFEFILL"),
            row.get("NOTE"), row.get("DEFAULT_START_DAYS"), row.get("DEFAULT_END_DAYS"),
            row.get("RACK_LIFTING_FORECAST_METHOD"),
        ),
    )
    conn.commit()
    conn.close()


def _persist_snowflake(row: dict):
    session = get_snowflake_session()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()

    try:
        session.sql(f"ALTER TABLE {SNOWFLAKE_ADMIN_CONFIG_TABLE} ADD COLUMN IF NOT EXISTS PRODUCT STRING").collect()
        session.sql(f"ALTER TABLE {SNOWFLAKE_ADMIN_CONFIG_TABLE} ADD COLUMN IF NOT EXISTS NOTE STRING").collect()
        session.sql(
            f"ALTER TABLE {SNOWFLAKE_ADMIN_CONFIG_TABLE} ADD COLUMN IF NOT EXISTS RACK_LIFTING_FORECAST_METHOD STRING"
        ).collect()
        session.sql(f"UPDATE {SNOWFLAKE_ADMIN_CONFIG_TABLE} SET PRODUCT='*' WHERE PRODUCT IS NULL").collect()
        session.sql(
            f"UPDATE {SNOWFLAKE_ADMIN_CONFIG_TABLE} "
            f"SET RACK_LIFTING_FORECAST_METHOD='{RACK_LIFTING_FORECAST_METHOD_DEFAULT}' "
            f"WHERE RACK_LIFTING_FORECAST_METHOD IS NULL"
        ).collect()
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
    method = _sql_str(row.get("RACK_LIFTING_FORECAST_METHOD") or RACK_LIFTING_FORECAST_METHOD_DEFAULT)

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
            {end_days}::INTEGER AS DEFAULT_END_DAYS,
            {method}::STRING AS RACK_LIFTING_FORECAST_METHOD
    ) s
    ON t.REGION = s.REGION AND t.LOCATION = s.LOCATION AND COALESCE(t.PRODUCT, '*') = s.PRODUCT
    WHEN MATCHED THEN UPDATE SET
        VISIBLE_COLUMNS_JSON = s.VISIBLE_COLUMNS_JSON,
        BOTTOM = s.BOTTOM,
        SAFEFILL = s.SAFEFILL,
        NOTE = s.NOTE,
        DEFAULT_START_DAYS = s.DEFAULT_START_DAYS,
        DEFAULT_END_DAYS = s.DEFAULT_END_DAYS,
        RACK_LIFTING_FORECAST_METHOD = s.RACK_LIFTING_FORECAST_METHOD,
        UPDATED_AT = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN INSERT (
        REGION, LOCATION, PRODUCT, VISIBLE_COLUMNS_JSON, BOTTOM, SAFEFILL, NOTE,
        DEFAULT_START_DAYS, DEFAULT_END_DAYS, RACK_LIFTING_FORECAST_METHOD, UPDATED_AT
    ) VALUES (
        s.REGION, s.LOCATION, s.PRODUCT, s.VISIBLE_COLUMNS_JSON, s.BOTTOM, s.SAFEFILL, s.NOTE,
        s.DEFAULT_START_DAYS, s.DEFAULT_END_DAYS, s.RACK_LIFTING_FORECAST_METHOD, CURRENT_TIMESTAMP()
    )
    """
    session.sql(sql).collect()


def get_rack_lifting_forecast_method(*, region: str, location: str | None) -> str:
    cfg = get_effective_config(region=region, location=location, product=None)
    method = str(cfg.get("RACK_LIFTING_FORECAST_METHOD") or "").strip() or RACK_LIFTING_FORECAST_METHOD_DEFAULT
    if method not in set(RACK_LIFTING_FORECAST_METHODS):
        return RACK_LIFTING_FORECAST_METHOD_DEFAULT
    return method


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
    #   region default         (LOCATION='*', PRODUCT='*')
    #   region + product-only  (LOCATION='*', PRODUCT=prod)
    #   region + location-only (LOCATION=loc, PRODUCT='*')
    #   region + loc + product (LOCATION=loc, PRODUCT=prod)
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
    """Return the ordered list of visible columns for the given scope.

    Date is always first. The rest come from the stored config (or defaults).
    """
    cfg = get_effective_config(region=region, location=location, product=None)
    raw = cfg.get("VISIBLE_COLUMNS_JSON")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return list(DEFAULT_VISIBLE_COLUMNS)

    cols = json.loads(str(raw) or "[]")
    if not isinstance(cols, list) or not cols:
        return list(DEFAULT_VISIBLE_COLUMNS)

    # Backward-compatible column renames
    rename = {
        "Batch In": "Receipts",
        "Batch Out": "Deliveries",
        "Batch In Fact": "Receipts Fact",
        "Batch Out Fact": "Deliveries Fact",
    }
    out = [rename.get(str(c), str(c)) for c in cols]

    # Ensure Date is always the first column
    out = [c for c in out if c != "Date"]
    out = ["Date"] + out

    return out


def get_default_date_window(*, region: str, location: str | None) -> tuple[int, int]:
    cfg = get_effective_config(region=region, location=location, product=None)
    s = pd.to_numeric(pd.Series([cfg.get("DEFAULT_START_DAYS")]), errors="coerce").iloc[0]
    e = pd.to_numeric(pd.Series([cfg.get("DEFAULT_END_DAYS")]), errors="coerce").iloc[0]
    start = int(s) if pd.notna(s) else -10
    end = int(e) if pd.notna(e) else 30
    return start, end


def get_threshold_overrides(*, region: str, location: str | None, product: str | None = None) -> dict:
    if product is None or str(product).strip() == "":
        return {"BOTTOM": None, "SAFEFILL": None, "NOTE": None}

    df = load_admin_config_df()
    if df is None or df.empty:
        return {"BOTTOM": None, "SAFEFILL": None, "NOTE": None}

    region = str(region).strip() or "Unknown"
    loc = _location_key(location)
    prod = _product_key(product)

    product_only = _rows_for_scope(df, Scope(region=region, location=None, product=prod))
    location_product = _rows_for_scope(df, Scope(region=region, location=loc, product=prod))

    out = {"BOTTOM": None, "SAFEFILL": None, "NOTE": None}
    for rows in [product_only, location_product]:
        if not rows.empty:
            r = rows.iloc[-1]
            for k in out.keys():
                if k in rows.columns:
                    out[k] = r.get(k)

    return out


def display_super_admin_panel(*, regions: list[str], active_region: str | None, all_data: pd.DataFrame | None = None):
    st.subheader("Super Admin Configuration")

    @st.dialog("Add New Product")
    def _add_new_product_dialog(*, default_region: str, region_options: list[str]):
        from datetime import date
        from data_loader import insert_manual_product_today, load_region_location_pairs

        st.caption(f"Creates a new row for today ({date.today().strftime('%Y-%m-%d')}) with all flows set to 0.")
        _k = "admin_add_product"

        region_in = st.selectbox(
            "Region",
            options=region_options or ["Unknown"],
            index=(region_options.index(default_region) if default_region in (region_options or []) else 0),
            key=f"{_k}_region",
        )

        locs: list[str] = []
        try:
            pairs = load_region_location_pairs()
            if pairs is not None and not pairs.empty and "Region" in pairs.columns and "Location" in pairs.columns:
                locs = sorted(pairs[pairs["Region"] == region_in]["Location"].dropna().astype(str).unique().tolist())
        except Exception:
            locs = []

        mode = st.radio(
            "Location entry",
            options=["Select Existing", "Add New Location"],
            horizontal=True,
            key=f"{_k}_loc_mode",
        )

        if mode == "Select Existing":
            location_in = st.selectbox(
                "Location",
                options=(locs or ["(No locations found)"]),
                key=f"{_k}_loc_select",
            )
        else:
            location_in = st.text_input(
                "New Location",
                placeholder="Please be careful with spaces and capitalization",
                key=f"{_k}_loc_new",
            )

        product_in = st.text_input("Product name", key=f"{_k}_product")
        c1, c2 = st.columns(2)
        with c1:
            opening_in = st.number_input("Opening Inventory (today)", value=0.0, step=1.0, format="%.2f", key=f"{_k}_opening")
        with c2:
            closing_in = st.number_input("Closing Inventory (today)", value=0.0, step=1.0, format="%.2f", key=f"{_k}_closing")

        note = "This Product was added manually today"
        st.text_input("Note", value=note, disabled=True)

        b1, b2 = st.columns(2)
        with b1:
            if logged_button("💾 Save", type="primary", event="admin_add_product_save",
                             metadata={"region": region_in, "location": location_in, "product": product_in}):
                try:
                    if mode == "Select Existing" and not locs:
                        raise ValueError("No locations available for selected region")
                    insert_manual_product_today(
                        region=region_in, location=location_in, product=product_in,
                        opening_inventory_bbl=float(opening_in),
                        closing_inventory_bbl=float(closing_in),
                        note=note,
                    )
                    st.success("Added")
                    st.rerun()
                except Exception as e:
                    log_error(
                        error_code="ADMIN_ADD_PRODUCT_FAILED",
                        error_message=str(e),
                        stack_trace=__import__("traceback").format_exc(),
                        service_module="UI",
                    )
                    st.error(str(e))
        with b2:
            if logged_button("Cancel", event="admin_add_product_cancel", metadata={"region": region_in}):
                st.rerun()

    region = st.selectbox(
        "Region",
        options=regions or ["Unknown"],
        index=(regions.index(active_region) if active_region in (regions or []) else 0),
    )

    locs: list[str] = []
    try:
        pairs = load_region_location_pairs()
        if pairs is not None and not pairs.empty and "Region" in pairs.columns and "Location" in pairs.columns:
            locs = sorted(pairs[pairs["Region"] == region]["Location"].dropna().astype(str).unique().tolist())
    except Exception:
        locs = []

    scope_opts = ["(Region default)"] + locs
    scope = st.selectbox("Location (optional)", options=scope_opts)
    location = None if scope == "(Region default)" else scope

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
        rl_method = str(cfg.get("RACK_LIFTING_FORECAST_METHOD") or RACK_LIFTING_FORECAST_METHOD_DEFAULT)
    else:
        # ── Column Visibility & Order ──────────────────────────────────────
        st.markdown("#### Column Visibility & Order")
        st.caption(
            "**Date** is always shown first and cannot be removed. "
            "Toggle columns on/off and set their **Position** (lower number = displayed earlier)."
        )

        current_cols = json.loads(str(cfg.get("VISIBLE_COLUMNS_JSON") or "[]"))
        if not isinstance(current_cols, list) or not current_cols:
            current_cols = list(DEFAULT_VISIBLE_COLUMNS)
        # Remove Date from the configurable list (always forced first)
        current_non_date = [c for c in current_cols if c != "Date"]
        current_set = set(current_non_date)

        # Build position map: column → 1-based position from current config
        pos_map = {col: i + 1 for i, col in enumerate(current_non_date)}
        max_pos = len(current_non_date)

        col_rows = []
        for col in _ALL_CONFIGURABLE_COLS:
            if col in current_set:
                pos = pos_map[col]
            else:
                max_pos += 1
                pos = max_pos
            col_rows.append({
                "Column": col,
                "Position": pos,
                "Show": col in current_set,
            })

        col_df = pd.DataFrame(col_rows)
        col_df = col_df.sort_values("Position").reset_index(drop=True)

        edited_col_df = st.data_editor(
            col_df,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Column": st.column_config.TextColumn("Column", disabled=True),
                "Position": st.column_config.NumberColumn(
                    "Position", min_value=1, step=1, format="%d",
                    help="Lower = shown first. Only applies to visible columns.",
                ),
                "Show": st.column_config.CheckboxColumn("Show", default=False),
            },
            width="stretch",
            height=550,
            key=f"col_order_editor|{region}|{scope}",
        )

        # Build ordered list: visible columns sorted by Position, Date always first
        visible_rows = edited_col_df[edited_col_df["Show"]].sort_values("Position")
        selected_cols = ["Date"] + visible_rows["Column"].tolist()

        # ── Forecast ──────────────────────────────────────────────────────
        st.markdown("#### Forecast (Rack/Liftings)")
        rl_method = st.selectbox(
            "Rack/Liftings forecast method",
            options=list(RACK_LIFTING_FORECAST_METHODS),
            index=(
                list(RACK_LIFTING_FORECAST_METHODS).index(str(cfg.get("RACK_LIFTING_FORECAST_METHOD")))
                if str(cfg.get("RACK_LIFTING_FORECAST_METHOD")) in set(RACK_LIFTING_FORECAST_METHODS)
                else list(RACK_LIFTING_FORECAST_METHODS).index(RACK_LIFTING_FORECAST_METHOD_DEFAULT)
            ),
            help="Controls how Details forecast rows estimate Rack/Liftings.",
        )

    st.markdown("#### Thresholds")
    bottom = ""
    safefill = ""
    note = ""

    if product is None:
        st.info("Select a Product to edit Bottom / SafeFill thresholds.")
    else:
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
            value=(
                "" if cfg.get("NOTE") is None or (isinstance(cfg.get("NOTE"), float) and pd.isna(cfg.get("NOTE")))
                else str(cfg.get("NOTE"))
            ),
            placeholder="Optional note for this product scope",
        )

    if product is None:
        st.markdown("#### Default Date Range Selection")
        d1, d2 = st.columns(2)
        with d1:
            start_days = st.number_input(
                "Start offset (days from today; negative = past)",
                value=_to_int_or(cfg.get("DEFAULT_START_DAYS"), -10),
                step=1,
            )
        with d2:
            end_days = st.number_input(
                "End offset (days from today; positive = future)",
                value=_to_int_or(cfg.get("DEFAULT_END_DAYS"), 30),
                step=1,
            )

    a1, a2 = st.columns([1, 1])
    with a1:
        save_clicked = logged_button(
            "💾 Save Configuration",
            event="admin_config_save_clicked",
            metadata={"region": region, "location": location, "product": product},
        )
    with a2:
        if logged_button("Add New Product", event="admin_add_product_open", metadata={"region": region}):
            _add_new_product_dialog(default_region=region, region_options=regions)

    if save_clicked:
        if product is not None:
            bottom_val = _to_float_or_none(bottom)
            safefill_val = _to_float_or_none(safefill)
            bottom_invalid = bottom.strip() != "" and bottom_val is None
            safefill_invalid = safefill.strip() != "" and safefill_val is None
            if bottom_invalid or safefill_invalid:
                bad = [f for f, flag in [("Bottom", bottom_invalid), ("SafeFill", safefill_invalid)] if flag]
                st.error(f"Invalid value(s) for: {', '.join(bad)}. Please enter numbers only.")
                st.stop()

        updates = {
            "VISIBLE_COLUMNS_JSON": (
                json.dumps(selected_cols or DEFAULT_VISIBLE_COLUMNS)
                if product is None
                else cfg.get("VISIBLE_COLUMNS_JSON")
            ),
            "BOTTOM": (_to_float_or_none(bottom) if product is not None else None),
            "SAFEFILL": (_to_float_or_none(safefill) if product is not None else None),
            "NOTE": ((str(note).strip() or None) if product is not None else None),
            "DEFAULT_START_DAYS": int(start_days),
            "DEFAULT_END_DAYS": int(end_days),
            "RACK_LIFTING_FORECAST_METHOD": (
                str(rl_method).strip() if product is None else cfg.get("RACK_LIFTING_FORECAST_METHOD")
            ),
        }
        try:
            save_admin_config(region=region, location=location, product=product, updates=updates)
            log_audit(
                event="admin_config_save_success",
                metadata={"region": region, "location": location, "product": product, "updates": updates},
            )
            st.success("Saved")
        except Exception as e:
            log_error(
                error_code="ADMIN_CONFIG_SAVE_FAILED",
                error_message=str(e),
                stack_trace=__import__("traceback").format_exc(),
                service_module="UI",
            )
            log_audit(
                event="admin_config_save_failed",
                metadata={"region": region, "location": location, "product": product, "error": str(e)},
            )
            raise

    st.markdown("#### Current stored rows")
    df = load_admin_config_df()
    if df is None or df.empty:
        st.write("(No config rows yet)")
    else:
        sort_cols = [c for c in ["REGION", "LOCATION", "PRODUCT"] if c in df.columns]
        st.dataframe(df.sort_values(sort_cols, kind="mergesort"), width="stretch", height=260)
