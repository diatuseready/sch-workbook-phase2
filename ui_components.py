import base64
import html
from pathlib import Path

import streamlit as st
import pandas as pd

from config import BG_LIGHT, TEXT_DARK, PRIMARY_GREEN, CARD_BG, ACCENT_GREEN, DATA_SOURCE


def setup_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="HF Sinclair Scheduler Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed"
    )


def apply_custom_css():
    """Apply custom CSS styling to the app."""
    css_style = f"""
    <style>
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {{
        display: none !important;
    }}

    .block-container {{
        padding-top: 1rem;
        padding-bottom: 1.5rem;
    }}

    [data-testid="stSidebar"],
    [data-testid="stSidebarNav"],
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}

    section[data-testid="stMain"] {{
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }}

    body {{
        background-color: {BG_LIGHT};
        color: {TEXT_DARK};
        font-family: 'Inter', sans-serif;
    }}
    .main-header {{
        background-color: {PRIMARY_GREEN};
        color: white;
        font-size: 1.7rem;
        font-weight: 600;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
    }}

    .main-header .header-left,
    .main-header .header-right {{
        width: 10%;
        min-width: 56px;
        display: flex;
        align-items: center;
        justify-content: flex-start;
    }}
    .main-header .header-right {{
        justify-content: flex-end;
    }}
    .main-header .header-title {{
        flex: 1;
        text-align: center;
        line-height: 1.1;
    }}
    .main-header .header-logo {{
        height: 40px;
        width: auto;
        display: block;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {CARD_BG};
        border-radius: 8px 8px 0 0;
        color: {PRIMARY_GREEN};
        font-weight: 600;
        border: 1px solid #E2E8F0;
        padding: 0.1rem 0.8rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY_GREEN} !important;
        color: white !important;
        border-bottom: 3px solid {ACCENT_GREEN} !important;
    }}

    div.stButton > button,
    div[data-testid="stButton"] > button,
    button[kind="primary"],
    button[kind="secondary"] {{
        background: {PRIMARY_GREEN} !important;
        color: white !important;
        border-radius: 6px;
        font-weight: 600;
        border: none !important;
        transition: 0.3s;
    }}

    .btn-spacer {{
        height: 1.6rem;
    }}

    div.stButton > button:hover,
    div[data-testid="stButton"] > button:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover {{ opacity: 0.9; }}

    [data-testid="stDataEditor"] [data-testid="stElementToolbar"],
    [data-testid="stDataEditor"] [data-testid="stToolbar"] {{
        display: none !important;
    }}
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
        margin-bottom: 0.5rem;
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
        color: {PRIMARY_GREEN};
        margin: 0.15rem 0 0 0;
    }}

    /* Transparent icon buttons (Reset ↺ and Formulas ℹ️) —
       targets any stColumn whose subtree contains a .transparent-icon marker */
    [data-testid="stColumn"]:has(.transparent-icon) button {{
        background: transparent !important;
        border: 1px solid #000 !important;
        box-shadow: none !important;
        color: {TEXT_DARK} !important;
        padding: 6px 14px !important;
        font-size: 14px !important;
    }}
    [data-testid="stColumn"]:has(.transparent-icon) button:hover {{
        opacity: 0.6 !important;
        background: transparent !important;
    }}
    </style>
    """
    st.markdown(css_style, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _get_logo_data_uri(filename: str = "hfs_dino_logo.png") -> str | None:
    """Return a base64 data URI for the header logo, or None if not found."""
    p = Path(__file__).resolve().with_name(filename)
    if not p.exists():
        p = Path(filename)
    if not p.exists():
        return None
    data = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def display_header():
    """Display the main header of the application."""
    logo_uri = _get_logo_data_uri()
    if logo_uri:
        st.markdown(
            f"""
            <div class="main-header">
                <div class="header-left">
                    <img class="header-logo" src="{logo_uri}" alt="HF Sinclair" />
                </div>
                <div class="header-title">Scheduler FlowSight</div>
                <div class="header-right"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="main-header"><div class="header-title">HF Sinclair Scheduler FlowSight</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Details-tab UI helpers
# ---------------------------------------------------------------------------

@st.dialog("System Files")
def _view_files_dialog(*, file_locations: list[str] | None, context: dict | None = None) -> None:
    """Popup showing signed download links for system files for a given row/day."""
    from data_loader import generate_snowflake_signed_urls

    ctx = context or {}
    title_bits = [b for b in [ctx.get("date"), ctx.get("location"), ctx.get("product")] if b]
    if title_bits:
        st.caption(" / ".join(str(b) for b in title_bits))

    paths = [str(p).strip() for p in (file_locations or []) if p is not None and str(p).strip()]

    if DATA_SOURCE != "snowflake":
        st.info("File downloads are only available in Snowflake mode.")
        return

    if not paths:
        st.info("No system files found for this row.")
        return

    with st.spinner("Generating signed URLs…"):
        signed = generate_snowflake_signed_urls(paths, expiry_seconds=3600)

    if not signed:
        st.warning("No downloadable links could be generated.")
        return

    st.write("Click a file to download:")

    def _short_label(name: str, *, max_len: int = 55) -> str:
        s = str(name or "")
        if len(s) <= max_len:
            return s
        head = max(10, (max_len - 1) // 2)
        tail = max(10, max_len - 1 - head)
        return s[:head].rstrip() + "…" + s[-tail:].lstrip()

    for item in signed:
        p = str(item.get("path") or "")
        url = str(item.get("url") or "")
        label = p.split("/")[-1] if "/" in p else p
        if url:
            st.link_button(label=_short_label(label), url=url)
            st.caption(p)


def _render_blocking_overlay(show: bool, *, message: str = "Saving…") -> None:
    """Render a full-screen overlay to block clicks during long operations."""
    if not show:
        return
    msg = html.escape(str(message or "Saving…"))
    st.markdown(
        f"""
        <style>
        #details-save-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.35);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            pointer-events: all;
        }}
        #details-save-overlay .card {{
            background: white;
            border-radius: 12px;
            padding: 18px 22px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.25);
            font-weight: 700;
            color: #2D3748;
        }}
        </style>
        <div id="details-save-overlay">
          <div class="card">{msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_threshold_cards(
    *,
    bottom: float | None,
    safefill: float | None,
    note: str | None = None,
    c_safefill,
    c_bottom,
    c_note,
    c_info,
    display_forecast_method: str | None = None,
) -> None:
    """Render SafeFill, Bottom, Note, and Forecast Method mini-cards."""
    with c_safefill:
        v = "—" if safefill is None else f"{safefill:,.0f}"
        st.markdown(
            f'<div class="mini-card" style="margin-bottom:1rem;">'
            f'<p class="label">SafeFill</p><p class="value">{v}</p></div>',
            unsafe_allow_html=True,
        )

    with c_bottom:
        v = "—" if bottom is None else f"{bottom:,.0f}"
        st.markdown(
            f'<div class="mini-card" style="margin-bottom:1rem;">'
            f'<p class="label">Bottom</p><p class="value">{v}</p></div>',
            unsafe_allow_html=True,
        )

    with c_note:
        v = "—" if note in (None, "") else str(note)
        st.markdown(
            f'<div class="mini-card" style="margin-bottom:1rem;">'
            f'<p class="label">Note</p>'
            f'<p class="value" style="font-size:0.95rem; font-weight:700;">{v}</p></div>',
            unsafe_allow_html=True,
        )

    with c_info:
        method = "—" if not display_forecast_method else display_forecast_method
        st.markdown(
            f'<div class="mini-card" style="margin-bottom:1rem;">'
            f'<p class="label">Forecast Method</p>'
            f'<p class="value" style="font-size:0.95rem; font-weight:700;">{method}</p></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Source status / data freshness
# ---------------------------------------------------------------------------

def _pipeline_down(processing_status: str) -> bool:
    s = str(processing_status or "").strip().upper()
    return s in {"FAILED", "ERROR", "DOWN"} or "FAIL" in s or "ERROR" in s or "DOWN" in s


def _freshness_badge(processing_status: str, last_updated_at) -> tuple[str, str]:
    """Return (label, color) based on pipeline status and data age."""
    if _pipeline_down(processing_status):
        return "DOWN", "#E53E3E"

    ts = pd.to_datetime(last_updated_at, errors="coerce")
    if pd.isna(ts):
        return "STALE", "#D69E2E"

    now = pd.Timestamp.utcnow().tz_localize(None)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)

    if (now - ts).total_seconds() / 3600.0 < 24:
        return "CURRENT", "#38A169"
    return "STALE", "#D69E2E"


def _format_ts(ts) -> str:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return "—"
    try:
        t = pd.to_datetime(ts)
        return "—" if pd.isna(t) else t.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def display_data_freshness_cards(
    *,
    active_region: str | None,
    selected_loc: str | None,
    loc_col: str,
    source_status: "pd.DataFrame",
):
    if source_status is None or source_status.empty:
        st.info("No source status data available.")
        return

    if selected_loc in (None, ""):
        st.info("Select a Location/System to view data freshness.")
        return

    df = source_status.copy()

    if active_region and "REGION_CODE" in df.columns:
        df = df[df["REGION_CODE"].fillna("Unknown") == active_region]

    selected_loc_s = str(selected_loc)

    if str(loc_col) == "Location":
        if "APP_LOCATION_DESC" in df.columns:
            df = df[df["APP_LOCATION_DESC"].astype(str) == selected_loc_s]
        elif "LOCATION" in df.columns:
            df = df[df["LOCATION"].astype(str) == selected_loc_s]
    else:
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

    if "LAST_UPDATED_AT" in df.columns:
        df = df.sort_values("LAST_UPDATED_AT", ascending=False)
        if "CLASS" in df.columns:
            df = df.drop_duplicates(subset=["CLASS"], keep="first")

    df = df.head(8)
    cols = st.columns(min(len(df), 8))
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i]:
            name = str(row.get("DISPLAY_NAME") or row.get("LOCATION") or row.get("CLASS") or "Source")
            source_system = str(row.get("SOURCE_SYSTEM") or row.get("SOURCE_OPERATOR") or "").strip() or "—"
            raw_status = str(row.get("PROCESSING_STATUS") or "").strip() or "UNKNOWN"
            status_label, color = _freshness_badge(raw_status, row.get("LAST_UPDATED_AT"))
            last_upd = _format_ts(row.get("LAST_UPDATED_AT"))

            if last_upd == "—":
                color = "#718096"
                # Tells the user that this could be a manual location with no automated status, in a different style
                st.markdown(
                    f"""
                    <div class="card" style="border:1px dashed {color};">
                        <h4 style="color:{color}; margin-bottom:0.2rem;">{name}</h4>
                        <p style="margin:0; font-size:0.9rem; color:{TEXT_DARK};">
                            This may be a manual location with no automated status.<br>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.markdown(
                    f"""
                    <div class="card">
                        <h4 style="color:{PRIMARY_GREEN}; margin-bottom:0.2rem;">{name}</h4>
                        <p style="margin:0; font-size:0.9rem; color:{TEXT_DARK};">
                            Last Updated (CST): <b>{last_upd}</b><br>
                            Source System: <b>{source_system}</b><br>
                            Status: <span style="color:{color}; font-weight:700;">{status_label}</span>
                            <span style="color:#A0AEC0; font-weight:600;">({raw_status})</span><br>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
