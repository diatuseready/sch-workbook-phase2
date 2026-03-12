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
SQLITE_COLUMN_LINKS_TABLE = "APP_COLUMN_LINKS"
SNOWFLAKE_COLUMN_LINKS_TABLE = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.APP_COLUMN_LINKS"

DATA_SOURCE = "sqlite"  # "snowflake"

SQLITE_DB_PATH = "inventory.db"
SQLITE_TABLE = "APP_INVENTORY"
SQLITE_SOURCE_STATUS_TABLE = "APP_SOURCE_STATUS"

SNOWFLAKE_WAREHOUSE = "HFS_SCHEDULER_WORKBOOK_STREAMLIT_WH"
SNOWFLAKE_SOURCE_STATUS_TABLE = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.APP_SOURCE_STATUS"
SNOWFLAKE_WORKBOOK_STAGE = "@CONSUMPTION.HFS_COMMERCIAL_INVENTORY.SCHEDULERWORKBOOKS_INTERNAL_COPY_STG"
SNOWFLAKE_LOCATION_MAPPING_TABLE = "CONFORMED.HFS_COMMERCIAL_INVENTORY.MASTER_LOCATION_CODE_MAPPING"

SQLITE_AUDIT_LOG_TABLE = "APP_AUDIT_LOG"
SQLITE_ERROR_LOG_TABLE = "APP_ERROR_LOG"

SNOWFLAKE_AUDIT_LOG_TABLE = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.APP_AUDIT_LOG"
SNOWFLAKE_ERROR_LOG_TABLE = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.APP_ERROR_LOG"


# =============================================================================
# Canonical column names
# =============================================================================

# --- Identity / metadata ---
COL_DATE = "Date"
COL_REGION = "Region"
COL_LOCATION = "Location"
COL_SYSTEM = "System"
COL_PRODUCT = "Product"
COL_UPDATED = "updated"

# --- Input columns — Incoming  (positive contribution to Closing Inventory) ---
# Close Inv = Opening Inv + Incoming − Outgoing + Adjustments
COL_BATCH_IN_RAW = "Batch In (RECEIPTS_BBL)"   # raw DB column name
COL_BATCH_IN = "Receipts"                        # display name in UI/editor
COL_PIPELINE_IN = "Pipeline In"
COL_PRODUCTION = "Production"

# --- Input columns — Outgoing  (reduce Closing Inventory) ---
COL_BATCH_OUT_RAW = "Batch Out (DELIVERIES_BBL)"   # raw DB column name
COL_BATCH_OUT = "Deliveries"                         # display name in UI/editor
COL_RACK_LIFTINGS_RAW = "Rack/Liftings"             # raw DB column name
COL_RACK_LIFTING = "Rack/Lifting"                    # display name in UI/editor
COL_PIPELINE_OUT = "Pipeline Out"

# --- Input columns — Adjustments  (net effect on Closing Inventory) ---
COL_ADJUSTMENTS = "Adjustments"
COL_GAIN_LOSS = "Gain/Loss"
COL_TRANSFERS = "Transfers"

# --- Calculated columns  (auto-derived, always read-only in the editor) ---
COL_OPEN_INV_RAW = "Open Inv"                  # raw DB name; renamed to Opening Inv in UI
COL_OPENING_INV = "Opening Inv"                # = previous day's Close Inv
COL_CLOSE_INV_RAW = "Close Inv"               # = Opening + Incoming − Outgoing + Adjustments
COL_AVAILABLE = "Available"                    # system-supplied; used in Total/Accounting Inv
COL_INTRANSIT = "Intransit"                    # system-supplied; used in Total Closing Inv
COL_TOTAL_CLOSING_INV = "Total Closing Inv"   # = Available + Intransit
COL_AVAILABLE_SPACE = "Available Space"        # = SafeFill − Close Inv
COL_LOADABLE = "Loadable"                      # = Close Inv − Bottom
COL_TOTAL_INVENTORY = "Total Inventory"        # = Close Inv + Bottom
COL_ACCOUNTING_INV = "Accounting Inventory"    # = Close Inv − Storage
COL_7DAY_AVG_RACK = "7 Day Avg"               # 7-day rolling average of Rack/Lifting
COL_MTD_AVG_RACK = "MTD Avg"                  # month-to-date average of Rack/Lifting
COL_CALCULATED_RECEIPT = "Calculated Receipt"  # = Today Available − Yesterday Available + Today Rack/Lifting

# --- Misc columns  (user-editable; no direct impact on Closing Inventory) ---
COL_STORAGE = "Storage"            # user-entered; drives Accounting Inventory
COL_VESSEL = "Vessel"              # user-entered; free-text vessel name
COL_VESSEL_VOLUME = "Vessel Volume"  # user-entered; vessel volume (BBL)
COL_BATCH = "Batch"                # free-text batch label
COL_NOTES = "Notes"                # free-text notes
COL_TULSA = "Tulsa"                # receipts sub-breakdown
COL_EL_DORADO = "El Dorado"        # receipts sub-breakdown
COL_OTHER = "Other"                # receipts sub-breakdown
COL_OFFLINE = "Offline"            # renamed from Argentine; maps to OFFLINE_BBL in DB
COL_ARGENTINE = COL_OFFLINE         # backward-compatible alias
COL_FROM_327_RECEIPT = "From 327 Receipt"  # receipts sub-breakdown
COL_BATCH_BREAKDOWN = "Batch Breakdown"    # free-text batch breakdown
COL_RMPL_PIPELINE_OUT = "RMPL Pipeline Out"
COL_SEMINOE_PIPELINE_OUT = "Seminoe Pipeline Out"
COL_MEDICINE_PIPELINE_OUT = "Medicine Pipeline Out"
COL_PIONEER_PIPELINE_OUT = "Pioneer Pipeline Out"
COL_PTO = "PTO"
COL_RECON_FROM_191 = "Recon From 191"   # outgoing: deducted from Close Inv
COL_RECON_TO_182 = "Recon To 182"       # outgoing: deducted from Close Inv
COL_RMPL_BATCH_ID = "RMPL Batch ID"
COL_SEMINOE_BATCH_ID = "Seminoe Batch ID"
COL_MEDICINE_BATCH_ID = "Medicine Batch ID"
COL_PIONEER_BATCH_ID = "Pioneer Batch ID"

# --- Fact / Terminal Feed columns  (paired read-only columns; toggled via UI) ---
COL_OPEN_INV_FACT_RAW = "Open Inv Fact"
COL_OPENING_INV_FACT = "Opening Inv Fact"       # renamed for UI/editor

COL_CLOSE_INV_FACT_RAW = "Close Inv Fact"

COL_BATCH_IN_FACT_RAW = "Batch In Fact (FACT_RECEIPTS_BBL)"
COL_BATCH_IN_FACT = "Receipts Fact"

COL_BATCH_OUT_FACT_RAW = "Batch Out Fact (FACT_DELIVERIES_BBL)"
COL_BATCH_OUT_FACT = "Deliveries Fact"

COL_RACK_LIFTINGS_FACT_RAW = "Rack/Liftings Fact"
COL_RACK_LIFTING_FACT = "Rack/Lifting Fact"

COL_PIPELINE_IN_FACT = "Pipeline In Fact"
COL_PIPELINE_OUT_FACT = "Pipeline Out Fact"
COL_PRODUCTION_FACT = "Production Fact"
COL_ADJUSTMENTS_FACT = "Adjustments Fact"
COL_GAIN_LOSS_FACT = "Gain/Loss Fact"
COL_TRANSFERS_FACT = "Transfers Fact"

COL_AVAILABLE_FACT = "Available Fact"
COL_INTRANSIT_FACT = "Intransit Fact"

# --- Capacity / threshold columns (reference only, not editable) ---
COL_TANK_CAPACITY = "Tank Capacity"
COL_SAFE_FILL_LIMIT = "Safe Fill Limit"


# =============================================================================
# Column group tuples
# Used for admin config UI (group labels, ordering) and editor locking logic.
# =============================================================================

_EDITABLE_COMPARE_COLS = [
    "Receipts", "Deliveries", "Rack/Lifting",
    "Pipeline In", "Pipeline Out",
    "RMPL Pipeline Out", "Seminoe Pipeline Out", "Medicine Pipeline Out", "Pioneer Pipeline Out",
    "PTO", "Recon From 191", "Recon To 182",
    "Production", "Adjustments", "Gain/Loss", "Transfers",
    "Available", "Intransit", "Storage", "Vessel Volume",
    "Tulsa", "El Dorado", "Other", "Offline", "From 327 Receipt",
    "Notes", "Batch", "Batch Breakdown", "Vessel",
    "RMPL Batch ID", "Seminoe Batch ID", "Medicine Batch ID", "Pioneer Batch ID",
]

# Input — Incoming: added to Closing Inventory
INPUT_INCOMING_COLS: tuple[str, ...] = (
    COL_BATCH_IN, COL_PIPELINE_IN, COL_PRODUCTION,
    COL_TULSA, COL_EL_DORADO, COL_OTHER, COL_FROM_327_RECEIPT,
)

# Input — Outgoing: subtracted from Closing Inventory
INPUT_OUTGOING_COLS: tuple[str, ...] = (
    COL_BATCH_OUT,
    COL_RACK_LIFTING,
    COL_PIPELINE_OUT,
    COL_RMPL_PIPELINE_OUT,
    COL_SEMINOE_PIPELINE_OUT,
    COL_MEDICINE_PIPELINE_OUT,
    COL_PIONEER_PIPELINE_OUT,
    COL_PTO,
    COL_OFFLINE,
    COL_RECON_FROM_191,
    COL_RECON_TO_182,
)

# Input — Adjustments: net effect on Closing Inventory
INPUT_ADJUSTMENT_COLS: tuple[str, ...] = (COL_ADJUSTMENTS, COL_GAIN_LOSS, COL_TRANSFERS)

# All input columns combined
INPUT_COLS: tuple[str, ...] = INPUT_INCOMING_COLS + INPUT_OUTGOING_COLS + INPUT_ADJUSTMENT_COLS

# Calculated: auto-derived, always read-only in the editor
CALCULATED_COLS: tuple[str, ...] = (
    COL_OPENING_INV,
    COL_CLOSE_INV_RAW,
    COL_TOTAL_CLOSING_INV,
    COL_AVAILABLE_SPACE,
    COL_LOADABLE,
    COL_TOTAL_INVENTORY,
    COL_ACCOUNTING_INV,
    COL_7DAY_AVG_RACK,
    COL_MTD_AVG_RACK,
    COL_CALCULATED_RECEIPT,
)

# Misc: editable, no direct impact on Closing Inventory
# (Storage drives Accounting Inventory; Available/Intransit come from the system feed)
MISC_COLS: tuple[str, ...] = (
    COL_AVAILABLE,
    COL_INTRANSIT,
    COL_STORAGE,
    COL_VESSEL,
    COL_VESSEL_VOLUME,
    COL_BATCH,
    COL_NOTES,
    COL_BATCH_BREAKDOWN,
    COL_RMPL_BATCH_ID,
    COL_SEMINOE_BATCH_ID,
    COL_MEDICINE_BATCH_ID,
    COL_PIONEER_BATCH_ID,
)


# =============================================================================
# Rename map  (raw DB column name → UI display name)
# =============================================================================

DETAILS_RENAME_MAP = {
    COL_OPEN_INV_RAW: COL_OPENING_INV,
    COL_OPEN_INV_FACT_RAW: COL_OPENING_INV_FACT,
    "BATCH": COL_BATCH,
    COL_BATCH_IN_RAW: COL_BATCH_IN,
    COL_BATCH_IN_FACT_RAW: COL_BATCH_IN_FACT,
    COL_BATCH_OUT_RAW: COL_BATCH_OUT,
    COL_BATCH_OUT_FACT_RAW: COL_BATCH_OUT_FACT,
    COL_RACK_LIFTINGS_RAW: COL_RACK_LIFTING,
    COL_RACK_LIFTINGS_FACT_RAW: COL_RACK_LIFTING_FACT,
    "BATCH_BREAKDOWN": COL_BATCH_BREAKDOWN,
}

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

RACK_LIFTING_FORECAST_METHOD_DEFAULT = "7_day_avg"
RACK_LIFTING_FORECAST_METHODS: tuple[str, ...] = (
    "weekday_weighted",
    "7_day_avg",
    "mtd_avg",
)

# Snowflake role names that control feature access
ROLE_POWER = "SCHEDULER_WORKBOOK_POWER_FR"
ROLE_CHANGE = "SCHEDULER_WORKBOOK_CHANGE_FR"
ROLE_DISPLAY = "SCHEDULER_WORKBOOK_DISPLAY_FR"
