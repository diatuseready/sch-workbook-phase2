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


def display_data_freshness_cards(active_region: str, source_status: "pd.DataFrame"):
    """Display data freshness cards for the active region using APP_SOURCE_STATUS."""
    st.subheader("📈 Data Freshness")

    if source_status is None or source_status.empty:
        st.info("No source status data available.")
        return

    # Filter to selected region (matches inventory Region values)
    df = source_status.copy()
    if "REGION" in df.columns:
        df = df[df["REGION"].fillna("Unknown") == active_region]

    if df.empty:
        st.info(f"No source status rows found for region: '{active_region}'")
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
