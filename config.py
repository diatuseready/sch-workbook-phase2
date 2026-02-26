PRIMARY_GREEN = "#008000"
ACCENT_GREEN = "#38A169"
BG_LIGHT = "#F5F6FA"
TEXT_DARK = "#2D3748"
CARD_BG = "#FFFFFF"

REQUIRED_MAX_DEFAULTS: dict[str, float] = {}
REQUIRED_MIN_DEFAULTS: dict[str, float] = {}
INTRANSIT_DEFAULTS: dict[str, float] = {}

GLOBAL_REQUIRED_MAX_FALLBACK = 10000
GLOBAL_REQUIRED_MIN_FALLBACK = 0
GLOBAL_INTRANSIT_FALLBACK = 0

RAW_INVENTORY_TABLE = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.APP_INVENTORY"
SNOWFLAKE_ADMIN_CONFIG_TABLE = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.APP_SUPERADMIN_CONFIG"
SQLITE_ADMIN_CONFIG_TABLE = "APP_SUPERADMIN_CONFIG"


# -----------------------------------------------------------------------------
# Data source / storage configuration
# -----------------------------------------------------------------------------

# Choose data source: "sqlite" for local/dev or "snowflake" for prod
DATA_SOURCE = "sqlite"  # "snowflake"

# SQLite configuration
SQLITE_DB_PATH = "inventory.db"
SQLITE_TABLE = "APP_INVENTORY"
SQLITE_SOURCE_STATUS_TABLE = "APP_SOURCE_STATUS"

# Snowflake configuration
SNOWFLAKE_WAREHOUSE = "HFS_ADHOC_WH"
SNOWFLAKE_SOURCE_STATUS_TABLE = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.APP_SOURCE_STATUS"
SNOWFLAKE_WORKBOOK_STAGE = "@CONSUMPTION.HFS_COMMERCIAL_INVENTORY.SCHEDULERWORKBOOKS_INTERNAL_COPY_STG"
SNOWFLAKE_LOCATION_MAPPING_TABLE = "CONFORMED.HFS_COMMERCIAL_INVENTORY.MASTER_LOCATION_CODE_MAPPING"


# -----------------------------------------------------------------------------
# App audit + error logging
# -----------------------------------------------------------------------------

# SQLite tables (local/dev)
SQLITE_AUDIT_LOG_TABLE = "APP_AUDIT_LOG"
SQLITE_ERROR_LOG_TABLE = "APP_ERROR_LOG"

# Snowflake tables (prod)
SNOWFLAKE_AUDIT_LOG_TABLE = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.APP_AUDIT_LOG"
SNOWFLAKE_ERROR_LOG_TABLE = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.APP_ERROR_LOG"


# -----------------------------------------------------------------------------
# Canonical column names used across the dashboard
# -----------------------------------------------------------------------------

# Base columns
COL_DATE = "Date"
COL_REGION = "Region"
COL_LOCATION = "Location"
COL_SYSTEM = "System"
COL_PRODUCT = "Product"
COL_UPDATED = "updated"
COL_NOTES = "Notes"
COL_BATCH = "Batch"

# Inventory columns
COL_OPEN_INV_RAW = "Open Inv"
COL_CLOSE_INV_RAW = "Close Inv"
COL_OPENING_INV = "Opening Inv"  # renamed for UI/editor

# Additional inventory metrics
COL_AVAILABLE = "Available"
COL_INTRANSIT = "Intransit"

COL_OPEN_INV_FACT_RAW = "Open Inv Fact"
COL_OPENING_INV_FACT = "Opening Inv Fact"  # renamed for UI/editor

COL_CLOSE_INV_FACT_RAW = "Close Inv Fact"  # display name is the same

COL_BATCH_IN_FACT_RAW = "Batch In Fact (FACT_RECEIPTS_BBL)"
# Display name in the UI/details editor
COL_BATCH_IN_FACT = "Receipts Fact"

COL_BATCH_OUT_FACT_RAW = "Batch Out Fact (FACT_DELIVERIES_BBL)"
# Display name in the UI/details editor
COL_BATCH_OUT_FACT = "Deliveries Fact"

COL_RACK_LIFTINGS_FACT_RAW = "Rack/Liftings Fact"
COL_RACK_LIFTING_FACT = "Rack/Lifting Fact"

COL_PIPELINE_IN_FACT = "Pipeline In Fact"
COL_PIPELINE_OUT_FACT = "Pipeline Out Fact"
COL_PRODUCTION_FACT = "Production Fact"
COL_ADJUSTMENTS_FACT = "Adjustments Fact"
COL_GAIN_LOSS_FACT = "Gain/Loss Fact"
COL_TRANSFERS_FACT = "Transfers Fact"

# Fact inventory metrics
COL_AVAILABLE_FACT = "Available Fact"
COL_INTRANSIT_FACT = "Intransit Fact"

# Flows
COL_BATCH_IN_RAW = "Batch In (RECEIPTS_BBL)"
COL_BATCH_OUT_RAW = "Batch Out (DELIVERIES_BBL)"
# Display names in the UI/details editor
COL_BATCH_IN = "Receipts"
COL_BATCH_OUT = "Deliveries"

COL_RACK_LIFTINGS_RAW = "Rack/Liftings"
COL_RACK_LIFTING = "Rack/Lifting"  # renamed for UI/editor

COL_PIPELINE_IN = "Pipeline In"
COL_PIPELINE_OUT = "Pipeline Out"
COL_PRODUCTION = "Production"
COL_ADJUSTMENTS = "Adjustments"
COL_GAIN_LOSS = "Gain/Loss"
COL_TRANSFERS = "Transfers"

# Capacities/thresholds
COL_TANK_CAPACITY = "Tank Capacity"
COL_SAFE_FILL_LIMIT = "Safe Fill Limit"
COL_AVAILABLE_SPACE = "Available Space"

# UI-only / derived inventory metrics (not persisted)
COL_TOTAL_CLOSING_INV = "Total Closing Inv"
COL_LOADABLE = "Loadable"  # Close Inv - Bottoms

# UI-only calculated columns (not persisted to DB)
COL_TOTAL_INVENTORY = "Total Inventory"     # Close Inv + Bottoms (threshold)
COL_ACCOUNTING_INV = "Accounting Inventory"  # Close Inv - Storage
COL_7DAY_AVG_RACK = "7 Day Avg"             # 7-day rolling average of Rack/Lifting (historical, excl. zeros)
COL_MTD_AVG_RACK = "MTD Avg"               # Month-to-date average of Rack/Lifting (current calendar month, excl. zeros)

# User-editable columns that ARE persisted to DB (→ STORAGE_BBL etc.)
COL_STORAGE = "Storage"                          # Manual entry; persisted as STORAGE_BBL; drives Accounting Inventory
COL_TULSA = "Tulsa"                              # Location sub-breakdown; persisted as TULSA_BBL
COL_EL_DORADO = "El Dorado"                      # Location sub-breakdown; persisted as EL_DORADO_BBL
COL_OTHER = "Other"                              # Residual sub-breakdown; persisted as OTHER_BBL
COL_ARGENTINE = "Argentine"                      # Sub-location breakdown; persisted as ARGENTINE_BBL
COL_FROM_327_RECEIPT = "From 327 Receipt"        # Receipt sub-breakdown; persisted as FROM_327_RECEIPT_BBL


# Convenience groups
SUMMARY_AGG_COLS = (
    COL_CLOSE_INV_RAW,
    COL_OPEN_INV_RAW,
    COL_BATCH_IN_RAW,
    COL_BATCH_OUT_RAW,
    COL_RACK_LIFTINGS_RAW,
    COL_PRODUCTION,
    COL_PIPELINE_IN,
    COL_PIPELINE_OUT,
    COL_AVAILABLE,
    COL_INTRANSIT,
    COL_TANK_CAPACITY,
    COL_SAFE_FILL_LIMIT,
    COL_AVAILABLE_SPACE,
)

DETAILS_RENAME_MAP = {
    COL_OPEN_INV_RAW: COL_OPENING_INV,
    COL_OPEN_INV_FACT_RAW: COL_OPENING_INV_FACT,
    # Free-text column stored in APP_INVENTORY.BATCH
    "BATCH": COL_BATCH,
    COL_BATCH_IN_RAW: COL_BATCH_IN,
    COL_BATCH_IN_FACT_RAW: COL_BATCH_IN_FACT,
    COL_BATCH_OUT_RAW: COL_BATCH_OUT,
    COL_BATCH_OUT_FACT_RAW: COL_BATCH_OUT_FACT,
    COL_RACK_LIFTINGS_RAW: COL_RACK_LIFTING,
    COL_RACK_LIFTINGS_FACT_RAW: COL_RACK_LIFTING_FACT,
}

RACK_LIFTING_FORECAST_METHOD_DEFAULT = "7_day_avg"
RACK_LIFTING_FORECAST_METHODS: tuple[str, ...] = (
    "weekday_weighted",
    "7_day_avg",
    "mtd_avg",
)

# Snowflake role names that control feature access
ROLE_POWER = "SCHEDULER_WORKBOOK_POWER_FR"    # full access
ROLE_CHANGE = "SCHEDULER_WORKBOOK_CHANGE_FR"  # Admin Config disabled
ROLE_DISPLAY = "SCHEDULER_WORKBOOK_DISPLAY_FR"  # Admin Config + Enable Save disabled
