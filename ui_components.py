"""
UI Components module for HF Sinclair Scheduler Dashboard
Contains styling, CSS, and UI helper functions
"""

import streamlit as st
import pandas as pd

from config import BG_LIGHT, TEXT_DARK, PRIMARY_BLUE, CARD_BG, ACCENT_GREEN


def setup_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="HF Sinclair Scheduler Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def apply_custom_css():
    """Apply custom CSS styling to the app."""
    css_style = f"""
    <style>
    body {{
        background-color: {BG_LIGHT};
        color: {TEXT_DARK};
        font-family: 'Inter', sans-serif;
    }}
    .main-header {{
        background-color: {PRIMARY_BLUE};
        color: white;
        text-align: center;
        font-size: 1.7rem;
        font-weight: 600;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1.8rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {CARD_BG};
        border-radius: 8px 8px 0 0;
        color: {PRIMARY_BLUE};
        font-weight: 600;
        border: 1px solid #E2E8F0;
        padding: 0.1rem 0.8rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY_BLUE} !important;
        color: white !important;
        border-bottom: 3px solid {ACCENT_GREEN} !important;
    }}
    div.stButton > button {{
        background: {PRIMARY_BLUE} !important;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        border: none;
        transition: 0.3s;
    }}
    div.stButton > button:hover {{ opacity: 0.9; }}
    .card {{
        background-color: {CARD_BG};
        padding: 0.9rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        margin-bottom: 0.6rem;
    }}

    .mini-card {{
        background-color: {CARD_BG};
        padding: 0.5rem 0.75rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-left: 4px solid {ACCENT_GREEN};
        line-height: 1.1;
    }}
    .mini-card .label {{
        font-size: 0.8rem;
        font-weight: 700;
        color: {TEXT_DARK};
        opacity: 0.8;
        margin: 0;
    }}
    .mini-card .value {{
        font-size: 1.1rem;
        font-weight: 800;
        color: {PRIMARY_BLUE};
        margin: 0.15rem 0 0 0;
    }}
    </style>
    """
    st.markdown(css_style, unsafe_allow_html=True)


def display_header():
    """Display the main header of the application."""
    st.markdown('<div class="main-header">HF Sinclair Scheduler Dashboard</div>', unsafe_allow_html=True)


def _pipeline_down(processing_status: str) -> bool:
    """Return True when the pipeline is considered down for a location."""
    s = str(processing_status or "").strip().upper()
    return (
        s in {"FAILED", "ERROR", "DOWN"} or
        "FAIL" in s or
        "ERROR" in s or
        "DOWN" in s
    )


def _freshness_badge(processing_status: str, last_updated_at) -> tuple[str, str]:
    """Return (label, color) per business rules.

    Rules:
    - Red: pipeline down
    - Green: refreshed < 24 hours
    - Yellow: >= 24 hours (or unknown timestamp)
    """
    if _pipeline_down(processing_status):
        return "DOWN", "#E53E3E"  # red

    ts = pd.to_datetime(last_updated_at, errors="coerce")
    if pd.isna(ts):
        return "STALE", "#D69E2E"  # yellow

    # Compare using naive timestamps to avoid tz issues.
    now = pd.Timestamp.utcnow().tz_localize(None)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)

    age_hours = (now - ts).total_seconds() / 3600.0
    if age_hours < 24:
        return "CURRENT", "#38A169"  # green
    return "STALE", "#D69E2E"  # yellow


def _format_ts(ts) -> str:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return "—"
    try:
        t = pd.to_datetime(ts)
        if pd.isna(t):
            return "—"
        return t.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def display_data_freshness_cards(
    *,
    active_region: str | None,
    selected_loc: str | None,
    loc_col: str,
    source_status: "pd.DataFrame",
):
    """Display data freshness cards for the selected location/system.

    Previously, this section filtered only by Region, which could show cards for
    many locations at once. The desired UX is to show freshness cards for the
    currently selected Location/System.
    """
    st.subheader("📈 Data Freshness")

    if source_status is None or source_status.empty:
        st.info("No source status data available.")
        return

    # Require a location/system selection (Submit enforces this for inventory,
    # but Data Freshness should also behave sensibly before Submit).
    if selected_loc in (None, ""):
        st.info("Select a Location/System to view data freshness.")
        return

    df = source_status.copy()

    # Optional: keep region context (helps avoid cross-region collisions when
    # locations share codes). We still show *only* the selected location.
    if active_region and "REGION" in df.columns:
        df = df[df["REGION"].fillna("Unknown") == active_region]

    selected_loc_s = str(selected_loc)

    # Filter to the selected Location/System.
    if str(loc_col) == "Location":
        if "LOCATION" in df.columns:
            df = df[df["LOCATION"].astype(str) == selected_loc_s]
    else:
        # "System" selection (Midcon): normalize similarly to inventory logic.
        # System = COALESCE(NULLIF(SOURCE_OPERATOR,''), NULLIF(SOURCE_SYSTEM,''), LOCATION)
        op = df["SOURCE_OPERATOR"] if "SOURCE_OPERATOR" in df.columns else ""
        sys = df["SOURCE_SYSTEM"] if "SOURCE_SYSTEM" in df.columns else ""
        loc = df["LOCATION"] if "LOCATION" in df.columns else ""

        system_series = op
        if isinstance(system_series, pd.Series):
            system_series = system_series.fillna("")

        if "SOURCE_SYSTEM" in df.columns:
            sys_s = sys.fillna("") if isinstance(sys, pd.Series) else ""
            system_series = system_series.where(system_series.astype(str).str.strip().ne(""), sys_s)
        if "LOCATION" in df.columns:
            loc_s = loc.fillna("") if isinstance(loc, pd.Series) else ""
            system_series = system_series.where(system_series.astype(str).str.strip().ne(""), loc_s)

        df = df[system_series.astype(str) == selected_loc_s]

    if df.empty:
        st.info(f"No source status rows found for {loc_col}: '{selected_loc_s}'")
        return

    # Pick most recent row per CLASS (reduces duplicates)
    if "LAST_UPDATED_AT" in df.columns:
        df = df.sort_values("LAST_UPDATED_AT", ascending=False)
        if "CLASS" in df.columns:
            df = df.drop_duplicates(subset=["CLASS"], keep="first")

    # Limit cards to avoid overly wide layout
    max_cards = 8
    df = df.head(max_cards)

    cols = st.columns(min(len(df), max_cards))
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i]:
            # Keep display minimal: Location name, Last Updated, Status
            name = str(row.get("DISPLAY_NAME") or row.get("LOCATION") or row.get("CLASS") or "Source")
            source_system = str(
                row.get("SOURCE_SYSTEM") or
                row.get("SOURCE_OPERATOR") or
                row.get("SOURCE_LABEL") or
                ""
            ).strip() or "—"
            raw_status = str(row.get("PROCESSING_STATUS") or "").strip() or "UNKNOWN"
            status_label, color = _freshness_badge(raw_status, row.get("LAST_UPDATED_AT"))
            last_upd = _format_ts(row.get("LAST_UPDATED_AT"))

            st.markdown(
                f"""
                <div class="card">
                    <h4 style="color:{PRIMARY_BLUE}; margin-bottom:0.2rem;">{name}</h4>
                    <p style="margin:0; font-size:0.9rem; color:{TEXT_DARK};">
                        Last Updated: <b>{last_upd}</b><br>
                        Source System: <b>{source_system}</b><br>
                        Status: <span style="color:{color}; font-weight:700;">{status_label}</span>
                        <span style="color:#A0AEC0; font-weight:600;">({raw_status})</span><br>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
