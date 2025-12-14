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

# Table definitions
RAW_INVENTORY_TABLE = "CONSUMPTION.HFS_COMMERCIAL_INVENTORY.DAILY_INVENTORY_FACT"

# Mock Source Freshness Data
MOCK_SOURCES = {
    "PSR Stock One Drive": [
        {"source": "Anacortes Rack", "last_update": "2025-10-29 06:40 AM", "status": "Up to Date"},
        {"source": "PSR Refinery", "last_update": "2025-10-29 07:05 AM", "status": "Up to Date"},
        {"source": "Shell Portland Terminal", "last_update": "2025-10-28 10:50 PM", "status": "Delayed"},
        {"source": "Seaport Tacoma Terminal", "last_update": "2025-10-27 08:00 PM", "status": "Error"}
    ],
    "Navajo Product System": [
        {"source": "Navajo Refinery", "last_update": "2025-10-29 06:00 AM", "status": "Up to Date"},
        {"source": "Magellan El Paso Terminal", "last_update": "2025-10-28 09:30 PM", "status": "Delayed"},
        {"source": "Marathon Albuquerque Terminal", "last_update": "2025-10-27 08:50 PM", "status": "Error"}
    ],
    "Front Range": [
        {"source": "Aurora Terminal", "last_update": "2025-10-29 07:20 AM", "status": "Up to Date"},
        {"source": "Casper Terminal", "last_update": "2025-10-28 09:45 PM", "status": "Delayed"}
    ],
    "WX-Sinclair Supply": [
        {"source": "Woods Cross Refinery", "last_update": "2025-10-29 06:35 AM", "status": "Up to Date"},
        {"source": "Las Vegas Terminal", "last_update": "2025-10-28 10:50 PM", "status": "Delayed"},
        {"source": "Spokane Terminal", "last_update": "2025-10-29 07:10 AM", "status": "Up to Date"}
    ],
    "Group Supply Report (Midcon)": [
        {"source": "Kansas City-Argentine", "last_update": "2025-10-29 07:15 AM", "status": "Up to Date"},
        {"source": "Magellan", "last_update": "2025-10-28 11:10 PM", "status": "Delayed"},
        {"source": "Nustar (East)", "last_update": "2025-10-29 06:45 AM", "status": "Up to Date"}
    ]
}

STATUS_COLORS = {"Up to Date": ACCENT_GREEN, "Delayed": WARNING_ORANGE, "Error": ERROR_RED}
