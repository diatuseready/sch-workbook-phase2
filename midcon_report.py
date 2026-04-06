"""
Midcon-only inventory tables rendered inside the   Regional Summary  tab.

Two tables are added when the active region is Midcon:

  1.  HFS SYSTEM INVENTORY
      Columns:  Product | Gross Inventory | LIFO Target | OPS Target |
                EOM Vs Ops | Current Vs Ops | EOM Projections |
                Prev EOM End Inv | Build (Draw) M-O-M

  2.  ONEOK SYSTEM INVENTORY
      Columns:  Product | Max | Current | % of Tank Capacity |
                EOM Y-O-Y | Yesterday MPL Rack Loadings |
                % of Magellan Liftings | Nustar System Inventory


"""

from __future__ import annotations

import datetime
import pandas as pd
import streamlit as st
from uuid import uuid4

from config import DATA_SOURCE, SNOWFLAKE_WAREHOUSE, SQLITE_DB_PATH, RAW_INVENTORY_TABLE, SQLITE_TABLE

# ─────────────────────────────────────────────────────────────────────────────
# Region detection helper
# ─────────────────────────────────────────────────────────────────────────────

def is_midcon(active_region: str) -> bool:
    """Return True when the active region resolves to Midcon."""
    norm = "Midcon" if active_region == "Group Supply Report (Midcon)" else active_region
    return norm == "Midcon"


# ─────────────────────────────────────────────────────────────────────────────
# Table & column constants
# ─────────────────────────────────────────────────────────────────────────────

SNOWFLAKE_HFS_TABLE   = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.APP_MIDCON_HFS_SYSTEM_INVENTORY"
SNOWFLAKE_ONEOK_TABLE = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.APP_MIDCON_ONEOK_SYSTEM_INVENTORY"
SQLITE_HFS_TABLE      = "MIDCON_HFS_SYSTEM_INVENTORY"
SQLITE_ONEOK_TABLE    = "MIDCON_ONEOK_SYSTEM_INVENTORY"

# ── HFS columns ───────────────────────────────────────────────────────────────
# Display order shown in the editor
HFS_DISPLAY_COLS: list[str] = [
    "Gross Inventory",      # CALCULATED — latest Close Inv per product
    "LIFO Target",          # INPUT       — pure user entry
    "OPS Target",           # INPUT       — pure user entry
    "EOM Vs Ops",           # CALCULATED — EOM Projections − OPS Target
    "Current Vs Ops",       # CALCULATED — Gross Inventory − OPS Target
    "EOM Projections",      # CALCULATED — Close Inv on last day of current month
    "Prev EOM End Inv",     # INPUT       — auto-filled default, user-overrideable
    "Build (Draw) M-O-M",   # CALCULATED — EOM Projections − Prev EOM End Inv
]

# Columns computed at render time — read-only in the editor, never stored in DB
HFS_CALC_COLS: frozenset[str] = frozenset({
    "Gross Inventory",
    "EOM Vs Ops",
    "Current Vs Ops",
    "EOM Projections",
    "Build (Draw) M-O-M",
})

# Columns stored in DB: UI display name → DB column name
_HFS_SAVE_MAP: dict[str, str] = {
    "LIFO Target":      "LIFO_TARGET",
    "OPS Target":       "OPS_TARGET",
    "Prev EOM End Inv": "PREV_EOM_END_INV",
}

# ── ONEOK columns ─────────────────────────────────────────────────────────────
ONEOK_DISPLAY_COLS: list[str] = [
    "Max",                          # INPUT       — auto-filled from SafeFill, overrideable
    "Current",                      # INPUT       — auto-filled from latest Close Inv, overrideable
    "% of Tank Capacity",           # CALCULATED — (Current / Max) × 100
    "EOM Y-O-Y",                    # INPUT       — pure user entry
    "Yesterday MPL Rack Loadings",  # INPUT       — auto-filled from Magellan prior-day rack
    "% of Magellan Liftings",       # CALCULATED — (Yesterday MPL / total Midcon rack) × 100
    "Nustar System Inventory",      # INPUT       — pure user entry
]

ONEOK_CALC_COLS: frozenset[str] = frozenset({
    "% of Tank Capacity",
    "% of Magellan Liftings",
})

_ONEOK_SAVE_MAP: dict[str, str] = {
    "Max":                          "MAX_CAPACITY",
    "Current":                      "CURRENT_INV",
    "EOM Y-O-Y":                    "EOM_YOY",
    "Yesterday MPL Rack Loadings":  "YESTERDAY_MPL_RACK_LOADINGS",
    "Nustar System Inventory":      "NUSTAR_SYSTEM_INVENTORY",
}


# ─────────────────────────────────────────────────────────────────────────────
# SQLite – auto-create backing tables (input columns only)
# ─────────────────────────────────────────────────────────────────────────────

def _sqlite_has_product_only_unique(cur, table: str) -> bool:
    """Return True if `table` has a UNIQUE index on (PRODUCT) alone (old schema)."""
    try:
        indexes = cur.execute(f"PRAGMA index_list({table})").fetchall()
        for idx in indexes:
            if idx[2]:  # unique flag
                cols = [c[2] for c in cur.execute(f"PRAGMA index_info({idx[1]})").fetchall()]
                if cols == ["PRODUCT"]:
                    return True
    except Exception:
        pass
    return False


def _rebuild_hfs_table(cur) -> None:
    """Rebuild HFS table to replace UNIQUE(PRODUCT) with UNIQUE(LOCATION, PRODUCT)."""
    cur.execute(f"ALTER TABLE {SQLITE_HFS_TABLE} RENAME TO {SQLITE_HFS_TABLE}_migration_bak")
    cur.execute(f"""
        CREATE TABLE {SQLITE_HFS_TABLE} (
            RECORD_KEY       TEXT PRIMARY KEY,
            LOCATION         TEXT NOT NULL DEFAULT '',
            PRODUCT          TEXT NOT NULL,
            LIFO_TARGET      REAL DEFAULT 0,
            OPS_TARGET       REAL DEFAULT 0,
            PREV_EOM_END_INV REAL DEFAULT 0,
            UPDATED_BY       TEXT DEFAULT 'streamlit_app',
            UPDATED_AT       TEXT,
            CREATED_AT       TEXT,
            UNIQUE (LOCATION, PRODUCT)
        )
    """)
    cur.execute(f"""
        INSERT INTO {SQLITE_HFS_TABLE}
            (RECORD_KEY, LOCATION, PRODUCT, LIFO_TARGET, OPS_TARGET,
             PREV_EOM_END_INV, UPDATED_BY, UPDATED_AT, CREATED_AT)
        SELECT
            RECORD_KEY,
            COALESCE(LOCATION, '') AS LOCATION,
            PRODUCT,
            COALESCE(LIFO_TARGET, 0),
            COALESCE(OPS_TARGET, 0),
            COALESCE(PREV_EOM_END_INV, 0),
            COALESCE(UPDATED_BY, 'streamlit_app'),
            UPDATED_AT,
            CREATED_AT
        FROM {SQLITE_HFS_TABLE}_migration_bak
    """)
    cur.execute(f"DROP TABLE {SQLITE_HFS_TABLE}_migration_bak")


def _rebuild_oneok_table(cur) -> None:
    """Rebuild ONEOK table to replace UNIQUE(PRODUCT) with UNIQUE(LOCATION, PRODUCT)."""
    cur.execute(f"ALTER TABLE {SQLITE_ONEOK_TABLE} RENAME TO {SQLITE_ONEOK_TABLE}_migration_bak")
    cur.execute(f"""
        CREATE TABLE {SQLITE_ONEOK_TABLE} (
            RECORD_KEY                  TEXT PRIMARY KEY,
            LOCATION                    TEXT NOT NULL DEFAULT '',
            PRODUCT                     TEXT NOT NULL,
            MAX_CAPACITY                REAL DEFAULT 0,
            CURRENT_INV                 REAL DEFAULT 0,
            EOM_YOY                     REAL DEFAULT 0,
            YESTERDAY_MPL_RACK_LOADINGS REAL DEFAULT 0,
            NUSTAR_SYSTEM_INVENTORY     REAL DEFAULT 0,
            UPDATED_BY                  TEXT DEFAULT 'streamlit_app',
            UPDATED_AT                  TEXT,
            CREATED_AT                  TEXT,
            UNIQUE (LOCATION, PRODUCT)
        )
    """)
    cur.execute(f"""
        INSERT INTO {SQLITE_ONEOK_TABLE}
            (RECORD_KEY, LOCATION, PRODUCT, MAX_CAPACITY, CURRENT_INV, EOM_YOY,
             YESTERDAY_MPL_RACK_LOADINGS, NUSTAR_SYSTEM_INVENTORY,
             UPDATED_BY, UPDATED_AT, CREATED_AT)
        SELECT
            RECORD_KEY,
            COALESCE(LOCATION, '') AS LOCATION,
            PRODUCT,
            COALESCE(MAX_CAPACITY, 0),
            COALESCE(CURRENT_INV, 0),
            COALESCE(EOM_YOY, 0),
            COALESCE(YESTERDAY_MPL_RACK_LOADINGS, 0),
            COALESCE(NUSTAR_SYSTEM_INVENTORY, 0),
            COALESCE(UPDATED_BY, 'streamlit_app'),
            UPDATED_AT,
            CREATED_AT
        FROM {SQLITE_ONEOK_TABLE}_migration_bak
    """)
    cur.execute(f"DROP TABLE {SQLITE_ONEOK_TABLE}_migration_bak")


def _ensure_sqlite_tables() -> None:
    """Create Midcon SQLite tables if they do not already exist.

    Only INPUT columns are persisted.  Calculated columns are derived at
    render time and are never written to the database.
    """
    import sqlite3

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()

    # HFS — only the 3 user-editable input columns; keyed on (LOCATION, PRODUCT)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SQLITE_HFS_TABLE} (
            RECORD_KEY       TEXT PRIMARY KEY,
            LOCATION         TEXT NOT NULL DEFAULT '',
            PRODUCT          TEXT NOT NULL,
            LIFO_TARGET      REAL DEFAULT 0,
            OPS_TARGET       REAL DEFAULT 0,
            PREV_EOM_END_INV REAL DEFAULT 0,
            UPDATED_BY       TEXT DEFAULT 'streamlit_app',
            UPDATED_AT       TEXT,
            CREATED_AT       TEXT,
            UNIQUE (LOCATION, PRODUCT)
        )
    """)
    # Migrate existing tables that pre-date the LOCATION column
    try:
        cur.execute(f"ALTER TABLE {SQLITE_HFS_TABLE} ADD COLUMN LOCATION TEXT NOT NULL DEFAULT ''")
    except Exception:
        pass
    # Migrate old schema that had UNIQUE(PRODUCT) instead of UNIQUE(LOCATION, PRODUCT)
    if _sqlite_has_product_only_unique(cur, SQLITE_HFS_TABLE):
        _rebuild_hfs_table(cur)

    # ONEOK — only the 5 user-editable input columns; keyed on (LOCATION, PRODUCT)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SQLITE_ONEOK_TABLE} (
            RECORD_KEY                  TEXT PRIMARY KEY,
            LOCATION                    TEXT NOT NULL DEFAULT '',
            PRODUCT                     TEXT NOT NULL,
            MAX_CAPACITY                REAL DEFAULT 0,
            CURRENT_INV                 REAL DEFAULT 0,
            EOM_YOY                     REAL DEFAULT 0,
            YESTERDAY_MPL_RACK_LOADINGS REAL DEFAULT 0,
            NUSTAR_SYSTEM_INVENTORY     REAL DEFAULT 0,
            UPDATED_BY                  TEXT DEFAULT 'streamlit_app',
            UPDATED_AT                  TEXT,
            CREATED_AT                  TEXT,
            UNIQUE (LOCATION, PRODUCT)
        )
    """)
    # Migrate existing tables that pre-date the LOCATION column
    try:
        cur.execute(f"ALTER TABLE {SQLITE_ONEOK_TABLE} ADD COLUMN LOCATION TEXT NOT NULL DEFAULT ''")
    except Exception:
        pass
    # Migrate old schema that had UNIQUE(PRODUCT) instead of UNIQUE(LOCATION, PRODUCT)
    if _sqlite_has_product_only_unique(cur, SQLITE_ONEOK_TABLE):
        _rebuild_oneok_table(cur)

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Magellan rack total  (denominator for % of Magellan Liftings)
# ─────────────────────────────────────────────────────────────────────────────

def _get_magellan_rack_total(source: str, sqlite_db_path: str) -> tuple[float, str]:
    """
    Return (total_rack_bbl, date_used) where total_rack_bbl is the SUM of
    RACK_LIFTINGS_BBL for all Midcon + Magellan rows on the prior day (today − 1).

    Why a direct DB query instead of reading df_region?
    ────────────────────────────────────────────────────
    data_loader.py maps SOURCE_OPERATOR → df["System"] (not SOURCE_SYSTEM).
    For Midcon Magellan rows: SOURCE_OPERATOR='OneOk', SOURCE_SYSTEM='Magellan'.
    So df_region["System"] == 'OneOk', never 'Magellan'.  There is no column in
    df_region that carries SOURCE_SYSTEM, making it impossible to isolate Magellan
    rows from df_region alone.

    Date logic:
    ───────────
    Always uses prior_day = today − 1.  If no Magellan data exists for that date
    (e.g. late data load, weekend gap) the SUM returns 0 — the denominator becomes
    NaN and % of Magellan Liftings shows 0 for every row.  This is intentional:
    the column should reflect reality for the specific prior date, not silently
    fall back to a stale MAX date.

    Query (aggregates ALL products + ALL locations for SOURCE_SYSTEM='Magellan'):
        SELECT SUM(RACK_LIFTINGS_BBL)
        FROM APP_INVENTORY
        WHERE REGION_CODE  = 'Midcon'
          AND SOURCE_SYSTEM = 'Magellan'
          AND DATA_DATE    = <today − 1>
    """
    prior_day     = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).date()
    prior_day_str = str(prior_day)   # e.g. "2026-04-05"

    if source == "sqlite":
        import sqlite3
        conn = sqlite3.connect(sqlite_db_path)
        row = conn.execute(
            f"SELECT COALESCE(SUM(RACK_LIFTINGS_BBL), 0) FROM {SQLITE_TABLE} "
            f"WHERE REGION_CODE='Midcon' AND SOURCE_SYSTEM='Magellan' AND DATA_DATE=?",
            (prior_day_str,),
        ).fetchone()
        conn.close()
        total = float(row[0] or 0.0) if row else 0.0
    else:
        from data_loader import get_snowflake_session
        session = get_snowflake_session()
        session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()
        row_df = session.sql(
            f"SELECT COALESCE(SUM(RACK_LIFTINGS_BBL), 0) AS TOTAL_RACK "
            f"FROM {RAW_INVENTORY_TABLE} "
            f"WHERE REGION_CODE='Midcon' AND SOURCE_SYSTEM='Magellan' "
            f"AND DATA_DATE='{prior_day_str}'"
        ).to_pandas()
        total = float(row_df.iloc[0, 0] or 0.0) if not row_df.empty else 0.0

    return total, prior_day_str


# ─────────────────────────────────────────────────────────────────────────────
# Raw DB loaders  (only input columns — no calculated columns)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _load_hfs_db_cached(source: str, sqlite_db_path: str) -> pd.DataFrame:
    """Return the stored HFS input rows from DB (Location + Product + 3 input cols)."""
    if source == "sqlite":
        import sqlite3
        _ensure_sqlite_tables()
        conn = sqlite3.connect(sqlite_db_path)
        raw = pd.read_sql_query(
            f"SELECT LOCATION, PRODUCT, LIFO_TARGET, OPS_TARGET, PREV_EOM_END_INV "
            f"FROM {SQLITE_HFS_TABLE} ORDER BY LOCATION, PRODUCT",
            conn,
        )
        conn.close()
    else:
        from data_loader import get_snowflake_session
        session = get_snowflake_session()
        session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()
        raw = session.sql(f"""
            SELECT
                COALESCE(LOCATION, '') AS LOCATION,
                PRODUCT,
                CAST(COALESCE(LIFO_TARGET,      0) AS DOUBLE) AS LIFO_TARGET,
                CAST(COALESCE(OPS_TARGET,       0) AS DOUBLE) AS OPS_TARGET,
                CAST(COALESCE(PREV_EOM_END_INV, 0) AS DOUBLE) AS PREV_EOM_END_INV
            FROM {SNOWFLAKE_HFS_TABLE}
            ORDER BY LOCATION, PRODUCT
        """).to_pandas()

    # Rename DB → UI
    raw = raw.rename(columns={
        "LOCATION":         "Location",
        "PRODUCT":          "Product",
        "LIFO_TARGET":      "LIFO Target",
        "OPS_TARGET":       "OPS Target",
        "PREV_EOM_END_INV": "Prev EOM End Inv",
    })
    if "Location" not in raw.columns:
        raw["Location"] = ""
    raw["Location"] = raw["Location"].fillna("").astype(str)
    for c in ["LIFO Target", "OPS Target", "Prev EOM End Inv"]:
        if c not in raw.columns:
            raw[c] = 0.0
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)
    return raw


@st.cache_data(ttl=300, show_spinner=False)
def _load_oneok_db_cached(source: str, sqlite_db_path: str) -> pd.DataFrame:
    """Return the stored ONEOK input rows from DB (Location + Product + 5 input cols)."""
    if source == "sqlite":
        import sqlite3
        _ensure_sqlite_tables()
        conn = sqlite3.connect(sqlite_db_path)
        raw = pd.read_sql_query(
            f"SELECT LOCATION, PRODUCT, MAX_CAPACITY, CURRENT_INV, EOM_YOY, "
            f"YESTERDAY_MPL_RACK_LOADINGS, NUSTAR_SYSTEM_INVENTORY "
            f"FROM {SQLITE_ONEOK_TABLE} ORDER BY LOCATION, PRODUCT",
            conn,
        )
        conn.close()
    else:
        from data_loader import get_snowflake_session
        session = get_snowflake_session()
        session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()
        raw = session.sql(f"""
            SELECT
                COALESCE(LOCATION, '') AS LOCATION,
                PRODUCT,
                CAST(COALESCE(MAX_CAPACITY,                0) AS DOUBLE) AS MAX_CAPACITY,
                CAST(COALESCE(CURRENT_INV,                 0) AS DOUBLE) AS CURRENT_INV,
                CAST(COALESCE(EOM_YOY,                     0) AS DOUBLE) AS EOM_YOY,
                CAST(COALESCE(YESTERDAY_MPL_RACK_LOADINGS, 0) AS DOUBLE) AS YESTERDAY_MPL_RACK_LOADINGS,
                CAST(COALESCE(NUSTAR_SYSTEM_INVENTORY,     0) AS DOUBLE) AS NUSTAR_SYSTEM_INVENTORY
            FROM {SNOWFLAKE_ONEOK_TABLE}
            ORDER BY LOCATION, PRODUCT
        """).to_pandas()

    raw = raw.rename(columns={
        "LOCATION":                    "Location",
        "PRODUCT":                     "Product",
        "MAX_CAPACITY":                "Max",
        "CURRENT_INV":                 "Current",
        "EOM_YOY":                     "EOM Y-O-Y",
        "YESTERDAY_MPL_RACK_LOADINGS": "Yesterday MPL Rack Loadings",
        "NUSTAR_SYSTEM_INVENTORY":     "Nustar System Inventory",
    })
    if "Location" not in raw.columns:
        raw["Location"] = ""
    raw["Location"] = raw["Location"].fillna("").astype(str)
    for c in ["Max", "Current", "EOM Y-O-Y", "Yesterday MPL Rack Loadings", "Nustar System Inventory"]:
        if c not in raw.columns:
            raw[c] = 0.0
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Live-calculation builders
# (merge DB inputs with values computed fresh from df_region)
# ─────────────────────────────────────────────────────────────────────────────

def _add_total_inventory_col(df_r: pd.DataFrame, active_region: str) -> pd.DataFrame:
    """
    Add a 'Total Inventory' column to df_r.

    Formula: Total Inventory = Close Inv + Bottom (Required Mins/Heel)

    This is identical to the Details tab formula
    (_recalculate_total_inventory in details_tab.py, line ~373):
        out[COL_TOTAL_INVENTORY] = close + bottom

    where 'Bottom' is the per-(location, product) threshold read from
    admin_config via get_threshold_overrides().

    The computation is done once vectorised across all rows so the caller
    can simply read df_r["Total Inventory"] without any formula knowledge.
    df_region does NOT carry Total Inventory natively — it is a UI-only
    derived column — so we compute it here exactly as the Details tab does.
    """
    df_r = df_r.copy()
    if df_r.empty or "Close Inv" not in df_r.columns:
        df_r["Total Inventory"] = 0.0
        return df_r

    from admin_config import get_threshold_overrides

    # Build (location, product) → bottom once for all unique pairs
    loc_col  = df_r["Location"].astype(str).str.strip() if "Location" in df_r.columns else pd.Series([""] * len(df_r), index=df_r.index)
    prod_col = df_r["Product"].astype(str).str.strip()  if "Product"  in df_r.columns else pd.Series([""] * len(df_r), index=df_r.index)

    unique_pairs = set(zip(loc_col, prod_col)) - {("", "")}
    bottom_lookup: dict = {}  # (loc, prod) → float if configured, None if absent
    for loc, prod in unique_pairs:
        if not prod:
            continue
        ovr    = get_threshold_overrides(region=active_region, location=loc, product=prod)
        bottom = ovr.get("BOTTOM")
        # Align with Details tab _recalculate_total_inventory:
        #   when bottom is None  → Total Inventory = 0  (no threshold configured)
        #   when bottom is 0     → Total Inventory = Close Inv + 0  (threshold IS configured)
        #   when bottom > 0      → Total Inventory = Close Inv + Bottom
        b: float | None = None
        if bottom is not None:
            try:
                b_val = float(bottom)
                if not pd.isna(b_val):
                    b = b_val
            except (TypeError, ValueError):
                pass
        bottom_lookup[(loc, prod)] = b  # None  →  no threshold configured

    close_inv = pd.to_numeric(df_r["Close Inv"], errors="coerce").fillna(0.0)
    # Build a parallel boolean mask: True = bottom is configured (including 0)
    has_bottom = pd.Series(
        [bottom_lookup.get((l, p)) is not None for l, p in zip(loc_col, prod_col)],
        index=df_r.index,
    )
    bottom_vals = pd.Series(
        [bottom_lookup.get((l, p)) or 0.0 for l, p in zip(loc_col, prod_col)],
        index=df_r.index,
    )
    # When no bottom: show 0; when bottom configured: Close Inv + Bottom
    df_r["Total Inventory"] = (
        (close_inv + bottom_vals)
        .where(has_bottom, other=0.0)
        .round(2)
    )
    return df_r


def _build_hfs_display_df(
    df_region: pd.DataFrame,
    db_df: pd.DataFrame,
    active_region: str = "Midcon",
) -> pd.DataFrame:
    """
    Build the full HFS display dataframe.

    Calculated columns are always derived live from df_region so they are
    always up-to-date.  Input columns use the stored DB value when present,
    or auto-filled defaults where the row is new (stored value = 0).

    Formula derivations
    ───────────────────
    Total Inventory  =  Close Inv  +  Bottom (Required Mins/Heel)
                        — mirrors the Details tab formula exactly.
                        Close Inv comes from df_region (CLOSING_INVENTORY_BBL).
                        Bottom is the per-product/location threshold from
                        admin_config (get_threshold_overrides).

    Gross Inventory      = Total Inventory on the most recent row ≤ today
    EOM Projections      = Total Inventory on the last day of the current month
                           (falls back to the most recent row ≤ EOM date)
    Prev EOM End Inv     = Total Inventory on the last day of the previous month
                           (falls back to the most recent row ≤ that date);
                           auto-fills for new rows, user-overrideable

    User arithmetic:
        EOM Vs Ops       =  EOM Projections  −  OPS Target
        Current Vs Ops   =  Gross Inventory  −  OPS Target
        Build (Draw) MOM =  EOM Projections  −  Prev EOM End Inv
    """
    today = pd.Timestamp.today().normalize()
    today_date = today.date()

    # ── Date boundaries ──────────────────────────────────────────────────────
    last_month_end = (today.replace(day=1) - pd.Timedelta(days=1)).date()
    if today.month == 12:
        next_month_start = today.replace(year=today.year + 1, month=1, day=1)
    else:
        next_month_start = today.replace(month=today.month + 1, day=1)
    curr_month_end = (next_month_start - pd.Timedelta(days=1)).date()

    # ── Normalise df_region ──────────────────────────────────────────────────
    df_r = df_region.copy() if df_region is not None else pd.DataFrame()
    if not df_r.empty and "Date" in df_r.columns:
        df_r["Date"] = pd.to_datetime(df_r["Date"], errors="coerce")

    # ── (Location, Product) pairs — one row per pair ─────────────────────────
    loc_prod_pairs: list[tuple[str, str]] = []
    if not df_r.empty and "Location" in df_r.columns and "Product" in df_r.columns:
        loc_prod_pairs = sorted(
            {
                (str(r["Location"]).strip(), str(r["Product"]).strip())
                for _, r in df_r[["Location", "Product"]].dropna().iterrows()
                if str(r["Location"]).strip() and str(r["Product"]).strip()
            }
        )

    # ── Add Total Inventory column to df_r — same formula as the Details tab ─
    # Total Inventory = Close Inv + Bottom (Required Mins/Heel from admin_config)
    # We do this once upfront so the per-pair loop just reads the column.
    df_r = _add_total_inventory_col(df_r, active_region)

    # ── Per-(Location, Product) live values ──────────────────────────────────
    live_rows: list[dict] = []
    for location, product in loc_prod_pairs:
        pair_df = df_r[
            (df_r["Location"].astype(str).str.strip() == location) &
            (df_r["Product"].astype(str).str.strip() == product)
        ].copy()

        # Only rows on or before today (df_region may include future forecasts)
        hist_df = pair_df[pair_df["Date"].dt.date <= today_date].sort_values("Date")

        # Gross Inventory = Total Inventory on the single most-recent row ≤ today
        gross_inv = float(hist_df["Total Inventory"].iloc[-1]) if not hist_df.empty else 0.0

        # EOM Projections = Total Inventory on the last day of the current month
        eom_rows = pair_df[pair_df["Date"].dt.date == curr_month_end]
        if not eom_rows.empty:
            eom_proj = float(eom_rows["Total Inventory"].iloc[0])
        else:
            pre_eom = pair_df[pair_df["Date"].dt.date <= curr_month_end].sort_values("Date")
            eom_proj = float(pre_eom["Total Inventory"].iloc[-1]) if not pre_eom.empty else 0.0

        # Prev EOM End Inv default = Total Inventory on the last day of the previous month
        prev_rows = pair_df[pair_df["Date"].dt.date == last_month_end]
        if not prev_rows.empty:
            prev_eom_default = float(prev_rows["Total Inventory"].iloc[0])
        else:
            pre_prev = pair_df[pair_df["Date"].dt.date <= last_month_end].sort_values("Date")
            prev_eom_default = float(pre_prev["Total Inventory"].iloc[-1]) if not pre_prev.empty else 0.0

        live_rows.append({
            "Location":        location,
            "Product":         product,
            "_gross":          gross_inv,
            "_eom_proj":       eom_proj,
            "_prev_eom_dflt":  prev_eom_default,
        })

    if not live_rows:
        return pd.DataFrame(columns=["Location", "Product"] + HFS_DISPLAY_COLS)

    live_df = pd.DataFrame(live_rows)

    # ── Merge with DB-stored input values (keyed on Location + Product) ──────
    if not db_df.empty:
        db_cols_needed = [c for c in ["Location", "Product", "LIFO Target", "OPS Target", "Prev EOM End Inv"] if c in db_df.columns]
        merge = live_df.merge(db_df[db_cols_needed], on=["Location", "Product"], how="left")
    else:
        merge = live_df.copy()
        merge["LIFO Target"]      = 0.0
        merge["OPS Target"]       = 0.0
        merge["Prev EOM End Inv"] = 0.0

    for c in ["LIFO Target", "OPS Target", "Prev EOM End Inv"]:
        merge[c] = pd.to_numeric(merge.get(c), errors="coerce").fillna(0.0)

    # Auto-fill Prev EOM End Inv with live default when no DB row exists yet
    mask_new = merge["Prev EOM End Inv"] == 0.0
    merge.loc[mask_new, "Prev EOM End Inv"] = merge.loc[mask_new, "_prev_eom_dflt"]

    # ── Apply calculated columns ─────────────────────────────────────────────
    merge["Gross Inventory"]    = merge["_gross"].round(0)
    merge["EOM Projections"]    = merge["_eom_proj"].round(0)
    ops = pd.to_numeric(merge["OPS Target"], errors="coerce").fillna(0.0)
    merge["EOM Vs Ops"]         = (merge["EOM Projections"] - ops).round(0)
    merge["Current Vs Ops"]     = (merge["Gross Inventory"]  - ops).round(0)
    merge["Build (Draw) M-O-M"] = (merge["EOM Projections"] - merge["Prev EOM End Inv"]).round(0)

    return merge[["Location", "Product"] + HFS_DISPLAY_COLS].copy()


def _build_oneok_display_df(df_region: pd.DataFrame, db_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full ONEOK display dataframe.

    Formula derivations
    ───────────────────
    SOURCE: summary_tab.py  display_regional_summary()  line 238
        Current (default)  =  CLOSING_INVENTORY_BBL  (latest Close Inv per product)
        — same as Gross Inventory in HFS

    SOURCE: summary_tab.py  calculate_required_max()  lines 53-73
        Max (default)  =  SAFE_FILL_LIMIT_BBL  (operational capacity ceiling)
        Falls back to TANK_CAPACITY_BBL if SafeFill is 0.

    SOURCE: summary_tab.py  display_regional_summary()  lines 201-206
        Yesterday MPL Rack Loadings (default)  =
            prior-day RACK_LIFTINGS_BBL where System == 'Magellan'
        Same logic as Prior_Day_Sales but filtered to SOURCE_OPERATOR = Magellan.

    User-defined arithmetic:
        % of Tank Capacity   =  (Current / Max) × 100
        % of Magellan Liftings =  (Yesterday MPL / total Midcon rack yesterday) × 100
    """
    today     = pd.Timestamp.today().normalize()
    prior_day = (today - pd.Timedelta(days=1)).date()

    df_r = df_region.copy() if df_region is not None else pd.DataFrame()
    if not df_r.empty and "Date" in df_r.columns:
        df_r["Date"] = pd.to_datetime(df_r["Date"], errors="coerce")

    # ── (Location, Product) pairs ─────────────────────────────────────────────
    loc_prod_pairs: list[tuple[str, str]] = []
    if not df_r.empty and "Location" in df_r.columns and "Product" in df_r.columns:
        loc_prod_pairs = sorted(
            {
                (str(r["Location"]).strip(), str(r["Product"]).strip())
                for _, r in df_r[["Location", "Product"]].dropna().iterrows()
                if str(r["Location"]).strip() and str(r["Product"]).strip()
            }
        )

    # Total Magellan rack liftings for Midcon on the most recent available date
    # — denominator for % of Magellan Liftings.
    # Queried directly from APP_INVENTORY (SOURCE_SYSTEM = 'Magellan') so that
    # only Magellan-system rows are summed, not all Midcon operators.
    total_prior_rack, _magellan_date = _get_magellan_rack_total(DATA_SOURCE, SQLITE_DB_PATH)

    live_rows: list[dict] = []
    for location, product in loc_prod_pairs:
        pair_df = df_r[
            (df_r["Location"].astype(str).str.strip() == location) &
            (df_r["Product"].astype(str).str.strip() == product)
        ].copy().sort_values("Date").dropna(subset=["Date"])

        # Yesterday MPL Rack Loadings = prior-day rack where System == Magellan
        mpl_dflt = 0.0
        rack_col = next(
            (c for c in ("Rack/Liftings", "Rack/Lifting") if c in pair_df.columns), None
        )
        if rack_col and "System" in pair_df.columns:
            prior_mpl = pair_df[
                (pair_df["Date"].dt.date == prior_day) &
                (pair_df["System"].astype(str).str.lower().str.contains("magellan", na=False))
            ]
            if not prior_mpl.empty:
                mpl_dflt = float(
                    pd.to_numeric(prior_mpl[rack_col], errors="coerce").fillna(0).sum()
                )

        live_rows.append({
            "Location":       location,
            "Product":        product,
            "_total_rack":    total_prior_rack,
        })

    if not live_rows:
        return pd.DataFrame(columns=["Location", "Product"] + ONEOK_DISPLAY_COLS), 0.0, ""

    live_df = pd.DataFrame(live_rows)

    # ── Merge with DB-stored input values (keyed on Location + Product) ──────
    db_input_cols = ["Location", "Product", "Max", "Current", "EOM Y-O-Y",
                     "Yesterday MPL Rack Loadings", "Nustar System Inventory"]
    if not db_df.empty:
        existing = [c for c in db_input_cols if c in db_df.columns]
        merge = live_df.merge(db_df[existing], on=["Location", "Product"], how="left")
    else:
        merge = live_df.copy()
        for c in db_input_cols[2:]:
            merge[c] = 0.0

    for c in ["Current", "EOM Y-O-Y", "Nustar System Inventory", "Max", "Yesterday MPL Rack Loadings"]:
        if c not in merge.columns:
            merge[c] = 0.0
        merge[c] = pd.to_numeric(merge[c], errors="coerce").fillna(0.0)

    # ── Calculated columns ───────────────────────────────────────────────────
    max_safe = pd.to_numeric(merge["Max"], errors="coerce").replace(0.0, float("nan"))
    cur_num  = pd.to_numeric(merge["Current"], errors="coerce").fillna(0.0)
    merge["% of Tank Capacity"] = ((cur_num / max_safe) * 100).round(2).fillna(0.0)

    denom = total_prior_rack if total_prior_rack > 0 else float("nan")
    mpl_num = pd.to_numeric(merge["Yesterday MPL Rack Loadings"], errors="coerce").fillna(0.0)
    # Leave as None (blank cell) when denominator = 0 (no Magellan data for prior day)
    merge["% of Magellan Liftings"] = ((mpl_num / denom) * 100).round(2)

    return merge[["Location", "Product"] + ONEOK_DISPLAY_COLS].copy(), total_prior_rack, _magellan_date


# ─────────────────────────────────────────────────────────────────────────────
# Public loaders (used to prime session-state caches)
# ─────────────────────────────────────────────────────────────────────────────
# These return only the raw DB input rows — the display builder overlays
# the calculated columns on top when the table is rendered.

def load_hfs_db() -> pd.DataFrame:
    return _load_hfs_db_cached(DATA_SOURCE, SQLITE_DB_PATH)


def load_oneok_db() -> pd.DataFrame:
    return _load_oneok_db_cached(DATA_SOURCE, SQLITE_DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime recalculation helpers
# Mirror the _recalculate_* pattern from details_tab.py:
#   - After data_editor returns `edited`, call these to recompute calculated
#     cols from the current input values.
#   - The result is written back to session_state so the NEXT render (triggered
#     by the Streamlit rerun that already follows every committed cell edit)
#     shows updated calculated values without requiring a manual refresh.
# ─────────────────────────────────────────────────────────────────────────────

def _recalculate_hfs_calcs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute the HFS calculated columns from the editor's current state.

    Only the three columns that depend on user-editable inputs are updated here;
    Gross Inventory and EOM Projections come from df_region (source data) and
    are NOT changed by this function.
    """
    df = df.copy()
    ops      = pd.to_numeric(df.get("OPS Target"),      errors="coerce").fillna(0.0)
    eom_proj = pd.to_numeric(df.get("EOM Projections"), errors="coerce").fillna(0.0)
    gross    = pd.to_numeric(df.get("Gross Inventory"), errors="coerce").fillna(0.0)
    prev_eom = pd.to_numeric(df.get("Prev EOM End Inv"), errors="coerce").fillna(0.0)
    df["EOM Vs Ops"]         = (eom_proj - ops).round(0)
    df["Current Vs Ops"]     = (gross    - ops).round(0)
    df["Build (Draw) M-O-M"] = (eom_proj - prev_eom).round(0)
    return df


def _recalculate_oneok_calcs(df: pd.DataFrame, total_prior_rack: float) -> pd.DataFrame:
    """
    Recompute the ONEOK calculated columns from the editor's current state.
    """
    df = df.copy()
    current   = pd.to_numeric(df.get("Current"), errors="coerce").fillna(0.0)
    max_cap   = pd.to_numeric(df.get("Max"),     errors="coerce").replace(0.0, float("nan"))
    mpl       = pd.to_numeric(df.get("Yesterday MPL Rack Loadings"), errors="coerce").fillna(0.0)

    df["% of Tank Capacity"]    = ((current / max_cap) * 100).round(2).fillna(0.0)
    denom = total_prior_rack if total_prior_rack > 0 else float("nan")
    # Leave as None (blank cell) when denominator = 0 (no Magellan data for prior day)
    df["% of Magellan Liftings"] = ((mpl / denom) * 100).round(2)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# UPSERT  (saves only input columns — calculated columns are never persisted)
# ─────────────────────────────────────────────────────────────────────────────

def _save_inputs(
    df: pd.DataFrame,
    *,
    save_map: dict[str, str],          # UI col → DB col  (input cols only)
    sqlite_table: str,
    snowflake_table: str,
) -> int:
    """
    Upsert all rows in `df` to the target table — input columns only.

    The merge key is (LOCATION, PRODUCT).  Rows with an empty Product are
    skipped.  Duplicate (Location, Product) pairs within `df` are
    deduplicated (last row wins).  Calculated columns present in df are
    intentionally ignored here.

    Returns number of rows written.
    """
    if df is None or df.empty:
        return 0

    df = df.copy()
    if "Product" in df.columns:
        df = df[df["Product"].astype(str).str.strip().ne("")]
    if df.empty:
        return 0
    if "Location" not in df.columns:
        df["Location"] = ""
    df["Location"] = df["Location"].fillna("").astype(str).str.strip()

    df = df.drop_duplicates(subset=["Location", "Product"], keep="last")
    now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    db_cols: list[str] = list(save_map.values())

    def _f(v) -> float:
        try:
            if v is None:
                return 0.0
            f = float(str(v).replace(",", ""))
            return f if not pd.isna(f) else 0.0
        except (ValueError, TypeError):
            return 0.0

    rows: list[dict] = []
    for _, r in df.iterrows():
        product  = str(r.get("Product")  or "").strip()
        location = str(r.get("Location") or "").strip()
        if not product:
            continue
        d: dict = {"RECORD_KEY": str(uuid4()), "LOCATION": location, "PRODUCT": product, "UPDATED_BY": "streamlit_app"}
        for ui_col, db_col in save_map.items():
            d[db_col] = _f(r.get(ui_col, 0))
        rows.append(d)

    if not rows:
        return 0

    # ── SQLite ────────────────────────────────────────────────────────────────
    if DATA_SOURCE == "sqlite":
        import sqlite3

        _ensure_sqlite_tables()
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cur = conn.cursor()

        for row in rows:
            existing = cur.execute(
                f"SELECT RECORD_KEY FROM {sqlite_table} WHERE LOCATION = ? AND PRODUCT = ? LIMIT 1",
                (row["LOCATION"], row["PRODUCT"]),
            ).fetchone()

            if existing:
                set_clause = (
                    ", ".join(f"{c} = ?" for c in db_cols)
                    + ", UPDATED_BY = ?, UPDATED_AT = ?"
                )
                params = [row[c] for c in db_cols] + ["streamlit_app", now_str, existing[0]]
                cur.execute(
                    f"UPDATE {sqlite_table} SET {set_clause} WHERE RECORD_KEY = ?",
                    params,
                )
            else:
                all_cols = ["RECORD_KEY", "LOCATION", "PRODUCT"] + db_cols + ["UPDATED_BY", "UPDATED_AT", "CREATED_AT"]
                params = (
                    [row["RECORD_KEY"], row["LOCATION"], row["PRODUCT"]]
                    + [row[c] for c in db_cols]
                    + ["streamlit_app", now_str, now_str]
                )
                cur.execute(
                    f"INSERT INTO {sqlite_table} ({', '.join(all_cols)}) "
                    f"VALUES ({', '.join('?' * len(all_cols))})",
                    params,
                )

        conn.commit()
        conn.close()
        rows_written = len(rows)

    # ── Snowflake ─────────────────────────────────────────────────────────────
    else:
        from data_loader import get_snowflake_session

        session = get_snowflake_session()
        session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()

        cte_cols = ["RECORD_KEY", "LOCATION", "PRODUCT"] + db_cols + ["UPDATED_BY"]

        def _q(v) -> str:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return "NULL"
            return "'" + str(v).replace("'", "''") + "'"

        def _cast(col: str) -> str:
            if col in db_cols:
                return f"CAST({col} AS DOUBLE) AS {col}"
            if col == "RECORD_KEY":
                return f"CAST({col} AS VARCHAR(36)) AS {col}"
            return f"CAST({col} AS VARCHAR) AS {col}"

        update_set = ",\n                ".join(
            [f"{c} = s.{c}" for c in db_cols]
            + ["UPDATED_BY = s.UPDATED_BY", "UPDATED_AT = CURRENT_TIMESTAMP()"]
        )
        insert_cols_sql = ", ".join(cte_cols + ["CREATED_AT", "UPDATED_AT"])
        insert_vals_sql = ", ".join(
            [f"s.{c}" for c in cte_cols] + ["CURRENT_TIMESTAMP()", "CURRENT_TIMESTAMP()"]
        )
        cast_select = ",\n                    ".join(_cast(c) for c in cte_cols)

        rows_written = 0
        for start in range(0, len(rows), 200):
            chunk = rows[start : start + 200]
            values_rows = []
            for row in chunk:
                parts = []
                for col in cte_cols:
                    v = row.get(col)
                    if col in db_cols:
                        try:
                            parts.append(str(float(v)))
                        except Exception:
                            parts.append("0")
                    else:
                        parts.append(_q(v))
                values_rows.append("(" + ", ".join(parts) + ")")

            merge_sql = f"""
            MERGE INTO {snowflake_table} t
            USING (
                SELECT {cast_select}
                FROM VALUES {", ".join(values_rows)}
                AS s_raw({", ".join(cte_cols)})
            ) s
            ON t.LOCATION = s.LOCATION AND t.PRODUCT = s.PRODUCT
            WHEN MATCHED THEN UPDATE SET
                {update_set}
            WHEN NOT MATCHED THEN INSERT ({insert_cols_sql})
            VALUES ({insert_vals_sql})
            """
            session.sql(merge_sql).collect()
            rows_written += len(chunk)

    # Bust caches so next load is fresh
    _load_hfs_db_cached.clear()
    _load_oneok_db_cached.clear()

    return rows_written


def save_hfs_inventory(df: pd.DataFrame) -> int:
    return _save_inputs(
        df,
        save_map=_HFS_SAVE_MAP,
        sqlite_table=SQLITE_HFS_TABLE,
        snowflake_table=SNOWFLAKE_HFS_TABLE,
    )


def save_oneok_inventory(df: pd.DataFrame) -> int:
    return _save_inputs(
        df,
        save_map=_ONEOK_SAVE_MAP,
        sqlite_table=SQLITE_ONEOK_TABLE,
        snowflake_table=SNOWFLAKE_ONEOK_TABLE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Column config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _col_cfg(
    all_cols: list[str],
    calc_cols: frozenset[str],
) -> dict:
    """
    Build st.column_config for a Midcon inventory editor.

    Location + Product are both read-only TextColumns — they are populated
    automatically from df_region and cannot be changed by the user in the
    editor.  A location filter selectbox above each table lets users drill
    into a specific location (effectively a location-first cascade).

    - Location / Product  →  TextColumn(disabled=True)  read-only
    - Calculated columns  →  NumberColumn(disabled=True)  greyed out
    - Input columns       →  NumberColumn(editable)
    """
    cfg: dict = {}
    cfg["Location"] = st.column_config.TextColumn(
        "Location",
        disabled=True,
        width="medium",
    )
    cfg["Product"] = st.column_config.TextColumn(
        "Product",
        disabled=True,
        width="medium",
    )
    for c in all_cols:
        if c in calc_cols:
            cfg[c] = st.column_config.NumberColumn(
                c,
                format="accounting",
                disabled=True,
                help="Calculated automatically — cannot be edited directly.",
            )
        else:
            cfg[c] = st.column_config.NumberColumn(c, format="accounting")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Main public entry point
# ─────────────────────────────────────────────────────────────────────────────

def display_midcon_inventory_tables(
    df_region: pd.DataFrame,
    active_region: str,
) -> None:
    """
    Render both Midcon-specific inventory tables in the  📊 Regional Summary  tab.

    Call only when  is_midcon(active_region)  is True.
    `df_region` is the full Midcon regional inventory DataFrame already loaded
    by the app — it drives all calculated columns and product dropdown options.

    UI flow (mirrors details_tab.py  Enable Save → Save pattern)
    ─────────────────────────────────────────────────────────────
    1. Both tables are rendered with their current data.
    2. A single  "Enable Save"  toggle sits above the tables.
       When toggled ON, Streamlit reruns and commits the last in-progress cell
       edit (the same reason details_tab.py uses this pattern).
    3. A single  "💾 Save Both Tables"  button becomes active only when the
       toggle is ON.  Clicking it upserts input-column values for both tables
       to the DB in one operation, then reruns to show fresh data.
    """
    # ── Session-state keys ───────────────────────────────────────────────────
    HFS_DB_KEY            = "midcon_hfs_db"
    ONEOK_DB_KEY          = "midcon_oneok_db"
    HFS_DISPLAY_KEY       = "midcon_hfs_display"       # live display df (with recalculated cols)
    ONEOK_DISPLAY_KEY     = "midcon_oneok_display"
    ONEOK_RACK_KEY        = "midcon_oneok_total_rack"  # denominator for % of Magellan Liftings
    ONEOK_MAGELLAN_DATE_KEY = "midcon_oneok_magellan_date"  # date used for Magellan denominator
    REGION_EXT_KEY        = "midcon_df_region_ext"    # cached forecast-extended df_region
    SAVE_MSG_KEY      = "midcon_save_msg"
    ENABLE_KEY        = "midcon_enable_save"

    st.markdown("---")

    # ── Save result banner (from previous render cycle) ───────────────────────
    if saved_msg := st.session_state.pop(SAVE_MSG_KEY, None):
        if saved_msg.startswith("✅"):
            st.success(saved_msg)
        else:
            st.error(saved_msg)

    # ── Load raw DB rows (once per session; cleared after save) ───────────────
    if HFS_DB_KEY not in st.session_state:
        with st.spinner("Loading HFS data…"):
            st.session_state[HFS_DB_KEY] = load_hfs_db()
    if ONEOK_DB_KEY not in st.session_state:
        with st.spinner("Loading ONEOK data…"):
            st.session_state[ONEOK_DB_KEY] = load_oneok_db()

    hfs_db: pd.DataFrame   = st.session_state[HFS_DB_KEY]
    oneok_db: pd.DataFrame = st.session_state[ONEOK_DB_KEY]

    # ── Extend df_region with forecast rows (same as Details tab) ────────────
    # The Details tab calls _extend_with_30d_forecast() which rolls Close Inv
    # forward day-by-day from the last actual data date using rack lifting
    # averages.  Without this extension, pairs whose last actual row is
    # 2026-02-15 show stale Close Inv values instead of today's projection.
    from details_tab import _extend_with_30d_forecast

    _today = pd.Timestamp.today().normalize()
    if _today.month == 12:
        _next_month_first = _today.replace(year=_today.year + 1, month=1, day=1)
    else:
        _next_month_first = _today.replace(month=_today.month + 1, day=1)
    _curr_month_end_ts  = _next_month_first - pd.Timedelta(days=1)
    _last_month_end_ts  = _today.replace(day=1) - pd.Timedelta(days=1)
    _prior_day_ts       = _today - pd.Timedelta(days=1)

    _today_str          = _today.strftime("%b %d, %Y")
    _curr_month_end_str = _curr_month_end_ts.strftime("%b %d, %Y")
    _last_month_end_str = _last_month_end_ts.strftime("%b %d, %Y")
    _prior_day_str      = _prior_day_ts.strftime("%b %d, %Y")

    # Cache the extended dataframe so the expensive forecast extension only
    # runs once per load cycle, not on every render / rerun.
    # IMPORTANT: extend PER-LOCATION so each location uses its own configured
    # forecast method (via get_rack_lifting_forecast_method).  Passing
    # location=None would use only the region-level default, which can differ
    # from the location-specific method the Details tab uses — causing Gross
    # Inventory and EOM Projections to diverge from the Details tab values.
    if REGION_EXT_KEY not in st.session_state:
        if df_region is not None and not df_region.empty:
            _unique_locs = (
                df_region["Location"]
                .dropna().astype(str).str.strip()
                .unique()
                .tolist()
            )
            _ext_parts: list[pd.DataFrame] = []
            for _loc in _unique_locs:
                _loc_df = df_region[
                    df_region["Location"].astype(str).str.strip() == _loc
                ]
                if _loc_df.empty:
                    continue
                _ext_parts.append(
                    _extend_with_30d_forecast(
                        _loc_df,
                        id_col="Location",
                        region=active_region,
                        location=str(_loc),
                        forecast_end=_curr_month_end_ts,
                        default_days=30,
                    )
                )
            st.session_state[REGION_EXT_KEY] = (
                pd.concat(_ext_parts, ignore_index=True)
                if _ext_parts
                else df_region
            )
        else:
            st.session_state[REGION_EXT_KEY] = df_region
    df_region_ext = st.session_state[REGION_EXT_KEY]

    # ── Build display dataframes (live calculations overlaid on DB inputs) ────
    # Initialise the live display state once per load cycle.
    # After each edit the display state is updated in-place (see recalc below)
    # so calculated columns reflect the most recently committed input values.
    if HFS_DISPLAY_KEY not in st.session_state:
        st.session_state[HFS_DISPLAY_KEY] = _build_hfs_display_df(df_region_ext, hfs_db, active_region)

    # ── Always refresh Magellan denominator — never cache across renders ──────
    # Bug fix: previously total_prior_rack was stored in session state at init
    # and never refreshed.  If the user added Magellan data to APP_INVENTORY
    # after the session started (or if MPL was entered before data arrived),
    # _recalculate_oneok_calcs would use a stale 0 denominator → blank column.
    # Fix: query _get_magellan_rack_total() on EVERY render so the denominator
    # is always current, and re-apply the % calculation to the display df so
    # the column reflects both the current denominator and any saved MPL values.
    total_prior_rack, _magellan_date_display = _get_magellan_rack_total(DATA_SOURCE, SQLITE_DB_PATH)
    st.session_state[ONEOK_RACK_KEY]          = total_prior_rack
    st.session_state[ONEOK_MAGELLAN_DATE_KEY] = _magellan_date_display

    if ONEOK_DISPLAY_KEY not in st.session_state:
        oneok_df, _tr, _md = _build_oneok_display_df(df_region_ext, oneok_db)
        st.session_state[ONEOK_DISPLAY_KEY] = oneok_df

    hfs_display: pd.DataFrame   = st.session_state[HFS_DISPLAY_KEY]
    oneok_display: pd.DataFrame = st.session_state[ONEOK_DISPLAY_KEY]

    # Re-apply % of Magellan Liftings with the fresh denominator.
    # This handles two scenarios:
    #   (a) Denominator data was added to DB after session was first loaded
    #   (b) MPL values were saved/entered before denominator data existed
    oneok_display = _recalculate_oneok_calcs(oneok_display, total_prior_rack)
    st.session_state[ONEOK_DISPLAY_KEY] = oneok_display

    # ── Enable Save toggle + Save button + Formula info icon ────────────────
    # Mirrors details_tab.py column layout: toggle | save | spacer | ℹ️
    c_enable, c_save, _spacer, c_info = st.columns([1.4, 2.2, 7.5, 0.5])
    with c_enable:
        enable_save = st.toggle(
            "Enable Save",
            key=ENABLE_KEY,
            value=False,
            help=(
                "Toggle ON to commit the last edited cell, then click Save. "
            ),
        )
    with c_save:
        save_clicked = st.button(
            "💾 Save Tables",
            key="midcon_save_both_btn",
            type="primary",
            disabled=not bool(enable_save),
        )
    with c_info:
        st.markdown('<div class="transparent-icon"></div>', unsafe_allow_html=True)
        with st.popover("ℹ️"):
            st.markdown("### Calculated Column Formulas")
            st.caption(
                "Greyed-out columns are computed automatically and update "
                "immediately when you edit a related input column."
            )

            st.markdown("---")
            st.markdown("#### HFS System Inventory")
            st.markdown(
                "| Column | Formula |\n"
                "|---|---|\n"
                "| **Gross Inventory** | `Close Inv + Bottom` — most recent row on or before today |"
                " (matches *Total Inventory* in the Details tab) |\n"
                "| **EOM Projections** | `Close Inv + Bottom` — last day of current month |\n"
                "| **Prev EOM End Inv** | `Close Inv + Bottom` — last day of previous month |\n"
                "| **EOM Vs Ops** | `EOM Projections − OPS Target` |\n"
                "| **Current Vs Ops** | `Gross Inventory − OPS Target` |\n"
                "| **Build (Draw) M-O-M** | `EOM Projections − Prev EOM End Inv` |"
            )
            st.caption(
                "**Bottom** = Required Mins / Heel threshold from Admin Config — "
                "same value used in the Details tab to compute *Total Inventory*."
            )

            st.markdown("---")
            st.markdown("#### ONEOK System Inventory")
            st.markdown(
                "| Column | Formula |\n"
                "|---|---|\n"
                "| **% of Tank Capacity** | `(Current ÷ Max) × 100` |\n"
                "| **% of Magellan Liftings** | `(Yesterday MPL Rack Loadings ÷ SUM Magellan RACK_LIFTINGS_BBL for Midcon) × 100` |"
            )
            st.caption(
                "**% of Magellan Liftings:** numerator is the user-entered "
                "*Yesterday MPL Rack Loadings*. "
                "Denominator is prior days `SUM(RACK_LIFTINGS_BBL)` "
                "WHERE `REGION_CODE='Midcon'` AND `SOURCE_SYSTEM='Magellan'` AND `DATA_DATE = today −1`. "
                
            )

    # ── HFS System Inventory table ────────────────────────────────────────────
    st.markdown("### HFS System Inventory")
    st.caption(
        f"Gross Inventory & Current Vs Ops as of **{_today_str}**  |  "
        f"EOM Projections: **{_curr_month_end_str}**  |  "
        f"Prev EOM: **{_last_month_end_str}**"
    )

    hfs_col_cfg = _col_cfg(HFS_DISPLAY_COLS, HFS_CALC_COLS)
    hfs_edited: pd.DataFrame = st.data_editor(
        hfs_display,
        key="midcon_hfs_editor",
        column_config=hfs_col_cfg,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
    )
    # ── Live recalculation: update session_state and clear δ-state when any
    # input column changes so the next render shows clean base data.
    if hfs_edited is not None and not hfs_edited.empty:
        hfs_recalculated = _recalculate_hfs_calcs(hfs_edited)
        st.session_state[HFS_DISPLAY_KEY] = hfs_recalculated
        _hfs_changed = False
        for _col in ("LIFO Target", "OPS Target", "Prev EOM End Inv"):
            if _col in hfs_edited.columns and _col in hfs_display.columns:
                _old = pd.to_numeric(hfs_display[_col], errors="coerce").fillna(0).round(2)
                _new = pd.to_numeric(hfs_edited[_col], errors="coerce").fillna(0).round(2)
                if not _old.reset_index(drop=True).equals(_new.reset_index(drop=True)):
                    _hfs_changed = True
                    break
        if _hfs_changed:
            st.session_state.pop("midcon_hfs_editor", None)
            st.rerun()

    st.markdown("---")

    # ── ONEOK System Inventory table ──────────────────────────────────────────
    st.markdown("### ONEOK System Inventory")
    _mgl_date_label = (
        f"  |  Magellan total for: **{_magellan_date_display}** (prior day)"
        if _magellan_date_display else ""
    )
    st.caption(
        f"Current as of **{_today_str}**  |  "
        f"Yesterday MPL Rack Loadings: **{_prior_day_str}**"
        f"{_mgl_date_label}"
    )

    oneok_col_cfg = _col_cfg(ONEOK_DISPLAY_COLS, ONEOK_CALC_COLS)
    oneok_edited: pd.DataFrame = st.data_editor(
        oneok_display,
        key="midcon_oneok_editor",
        column_config=oneok_col_cfg,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
    )
    # ── Live recalculation: update session_state and clear δ-state when any
    # input column changes so the next render shows clean base data.
    if oneok_edited is not None and not oneok_edited.empty:
        oneok_recalculated = _recalculate_oneok_calcs(oneok_edited, total_prior_rack)
        st.session_state[ONEOK_DISPLAY_KEY] = oneok_recalculated
        _oneok_changed = False
        for _col in ("Max", "Current", "EOM Y-O-Y", "Yesterday MPL Rack Loadings", "Nustar System Inventory"):
            if _col in oneok_edited.columns and _col in oneok_display.columns:
                _old = pd.to_numeric(oneok_display[_col], errors="coerce").fillna(0).round(2)
                _new = pd.to_numeric(oneok_edited[_col], errors="coerce").fillna(0).round(2)
                if not _old.reset_index(drop=True).equals(_new.reset_index(drop=True)):
                    _oneok_changed = True
                    break
        if _oneok_changed:
            st.session_state.pop("midcon_oneok_editor", None)
            st.rerun()

    # ── Handle save ───────────────────────────────────────────────────────────
    if save_clicked:
        try:
            # Save ALL rows from the full session-state df (not just the filtered view)
            # so edits across all locations are persisted in one click.
            _full_hfs_to_save   = st.session_state.get(HFS_DISPLAY_KEY,   hfs_edited)
            _full_oneok_to_save = st.session_state.get(ONEOK_DISPLAY_KEY, oneok_edited)
            n_hfs   = save_hfs_inventory(_full_hfs_to_save)
            n_oneok = save_oneok_inventory(_full_oneok_to_save)

            # Clear cached DB rows, live display state, extended forecast
            # cache, and editor delta state so next render starts completely
            # fresh from the database.
            for k in (HFS_DB_KEY, ONEOK_DB_KEY, HFS_DISPLAY_KEY,
                      ONEOK_DISPLAY_KEY, ONEOK_RACK_KEY, ONEOK_MAGELLAN_DATE_KEY,
                      REGION_EXT_KEY, "midcon_hfs_editor", "midcon_oneok_editor"):
                st.session_state.pop(k, None)

            # Reset Enable Save toggle after a successful save — same pattern as
            # details_tab, which uses a version-key to reset the toggle widget.
            st.session_state.pop(ENABLE_KEY, None)

            st.session_state[SAVE_MSG_KEY] = (
                f"✅ Saved {n_hfs} HFS row(s) and {n_oneok} ONEOK row(s) to database."
            )
        except Exception as exc:
            st.session_state[SAVE_MSG_KEY] = f"❌ Save failed: {exc}"
        st.rerun()


