"""
Configuration module for HF Sinclair Scheduler Dashboard
Contains theme colors, default values, and table definitions
"""

# Professional Theme Colors
PRIMARY_BLUE = "#008000"      # Changed to green
ACCENT_GREEN = "#38A169"      # Changed to red (secondary)
WARNING_ORANGE = "#ED8936"
ERROR_RED = "#E53E3E"
BG_LIGHT = "#F5F6FA"
TEXT_DARK = "#2D3748"
CARD_BG = "#FFFFFF"

# Hardcoded defaults (used when metric cannot be computed from data)
REQUIRED_MAX_DEFAULTS = {
    # "Houston|ULSD": 18000,
    # "ULSD": 15000,
}

INTRANSIT_DEFAULTS = {
    # "Houston|ULSD": 2500,
    # "ULSD": 2000,
}

GLOBAL_REQUIRED_MAX_FALLBACK = 10000  # used if no specific default found
GLOBAL_INTRANSIT_FALLBACK = 0         # used if no specific default found

# Required Minimum defaults/fallbacks
# If a specific "Location|Product" (or "System|Product") key is not present,
# we will fall back to product-level default, otherwise to this global fallback.
REQUIRED_MIN_DEFAULTS = {
    # "Houston|ULSD": 3000,
    # "ULSD": 2500,
}

GLOBAL_REQUIRED_MIN_FALLBACK = 0

# Table definitions
# NOTE: The app now reads from APP_INVENTORY (both locally in SQLite and in Snowflake).
# Ensure this table/view exists in Snowflake with the expected columns used by `data_loader.py`.
RAW_INVENTORY_TABLE = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.APP_INVENTORY"

# NOTE: Data freshness is now sourced from APP_SOURCE_STATUS.
