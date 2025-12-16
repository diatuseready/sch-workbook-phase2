"""
UI Components module for HF Sinclair Scheduler Dashboard
Contains styling, CSS, and UI helper functions
"""

import streamlit as st
from config import *

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

def display_data_freshness_cards(active_region):
    """Display data freshness cards for the active region."""
    st.subheader("📈 Data Freshness & Source Status")
    
    # Debug: Show what region we're looking for
    st.info(f"Debug: Looking for region: '{active_region}'")
    st.info(f"Debug: Available regions in MOCK_SOURCES: {list(MOCK_SOURCES.keys())}")
    
    region_sources = MOCK_SOURCES.get(active_region, [])
    
    if region_sources:
        cols = st.columns(len(region_sources))
        for i, src in enumerate(region_sources):
            with cols[i]:
                color = STATUS_COLORS.get(src["status"], "#A0AEC0")
                st.markdown(f"""
                <div class="card">
                    <h4 style="color:{PRIMARY_BLUE}; margin-bottom:0.2rem;">{src['source']}</h4>
                    <p style="margin:0; font-size:0.9rem; color:{TEXT_DARK};">
                        Last Updated: <b>{src['last_update']}</b><br>
                        Status: <span style="color:{color}; font-weight:700;">{src['status']}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning(f"No data freshness sources configured for region: '{active_region}'")
