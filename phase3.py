import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="HF Sinclair - Phase 3 Marketing Dashboard Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Header bar */
    .header-container {
        background-color: #1b7d3f;
        color: white;
        padding: 12px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #15642f;
        margin-bottom: 20px;
    }
    .header-left {
        display: flex;
        gap: 10px;
        align-items: center;
    }
    .header-logo {
        background-color: white;
        color: #1b7d3f;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
    }
    .header-title {
        font-size: 18px;
        font-weight: 600;
    }
    .header-right {
        font-size: 12px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    .sidebar-label {
        font-size: 12px;
        font-weight: 600;
        color: #333;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Main content */
    .main-title {
        font-size: 24px;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 4px;
    }
    .main-subtitle {
        font-size: 13px;
        color: #666;
        margin-bottom: 16px;
    }

    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 10px;
    }

    /* Granularity controls */
    .granularity-label {
        font-size: 12px;
        font-weight: 600;
        color: #333;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }

    /* Grid header row */
    .grid-header-cell {
        font-weight: 600;
        font-size: 12px;
        color: #333;
        padding: 8px 4px;
        border-bottom: 2px solid #dddddd;
    }

    .location-cell {
        font-weight: 500;
        font-size: 13px;
        color: #333;
        padding-top: 10px;
    }

    .hint-text {
        font-size: 12px;
        color: #888;
    }

    .suggestion-label {
        font-size: 12px;
        color: #999;
    }

    .suggestion-value {
        font-size: 12px;
        color: #999;
        text-align: center;
        margin-top: 6px;
    }

    .last-saved {
        color: #999;
        font-size: 11px;
        text-align: center;
    }
    
    .metric-highlight {
        background-color: #f0f9f6;
        border-left: 5px solid #1b7d3f;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .status-good { color: #27ae60; font-weight: bold; }
    .status-warning { color: #e74c3c; font-weight: bold; }
    
    /* KPI Card Styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SAMPLE DATA - Based on Marketing Dashboard
# ============================================================================


@st.cache_data
def load_marketing_data():
    """
    Create structured mock data for the demo.
    This uses realistic patterns inspired by Marketing-Dashboard-12.1.2025.xlsx.
    """
    # Use 14 days of data for demo
    dates = pd.date_range(start="2025-12-01", periods=14, freq="D")

    # Region -> Location mapping
    location_config = {
        "HEP": {
            "locations": {
                "Burley": {
                    "products": ["2D15 (XUD)", "84T", "87T", "88.5"],
                    "base_targets": [26100, 15000, 12000, 5000],
                },
                "El Paso": {
                    "products": ["2D15 (XUD)", "84T", "87T", "88.5"],
                    "base_targets": [28000, 16000, 13000, 5500],
                },
            }
        },
        "PSR": {
            "locations": {
                "Orla": {
                    "products": ["2D15 (XUD)", "84T", "87T", "88.5"],
                    "base_targets": [22000, 13000, 11000, 4500],
                },
            }
        },
        "Rockies": {
            "locations": {
                "Boise": {
                    "products": ["2D15 (XUD)", "84T", "87T", "88.5"],
                    "base_targets": [24000, 14000, 10000, 4800],
                },
                "Casper": {
                    "products": ["2D15 (XUD)", "84T", "87T", "88.5"],
                    "base_targets": [22000, 13000, 9500, 4500],
                },
                "Las Vegas": {
                    "products": ["2D15 (XUD)", "84T", "87T", "88.5"],
                    "base_targets": [26000, 15000, 11000, 5200],
                },
            }
        },
    }

    data = []
    np.random.seed(42)

    for region, region_data in location_config.items():
        for location, location_data in region_data["locations"].items():
            for date in dates:
                for product_idx, product in enumerate(location_data["products"]):
                    base_target = location_data["base_targets"][product_idx]

                    # Vary targets by +/- 15%
                    target_total = base_target * np.random.uniform(0.85, 1.15)

                    # Sales vary vs. target: -10% to +15%
                    sales_total = target_total * np.random.uniform(0.90, 1.15)

                    for consignee in ["Branded", "Unbranded", "Contracts"]:
                        # Allocate volume across consignees
                        if consignee == "Branded":
                            split_factor = np.random.uniform(0.3, 0.5)
                        else:
                            split_factor = np.random.uniform(0.15, 0.35)

                        data.append(
                            {
                                "Region": region,
                                "Location": location,
                                "Date": date,
                                "Product": product,
                                "Consignee": consignee,
                                "Target": target_total * split_factor,
                                "Actuals": sales_total * split_factor,
                            }
                        )

    return pd.DataFrame(data)


if "demo_data" not in st.session_state:
    st.session_state.demo_data = load_marketing_data()

demo_df = st.session_state.demo_data

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def format_bpd(value: float) -> str:
    """Format as barrels per day with no decimals."""
    return f"{value:,.0f}"


# ============================================================================
# SCREEN 1: DATA ENTRY DEMO
# ============================================================================


def screen_data_entry(selected_region, selected_location, selected_date, selected_products, scale_percent=0):
    # Main title section
    st.markdown(f'<div class="main-title">{selected_region} - {selected_location} Plan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">'
        "Manage rack sale plans and view daily performance vs targets."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Plan Entry Section Header
    st.markdown('<div class="section-header">Rack Sale Plan Entry</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">'
        f"Input marketing projections for {selected_location} ({selected_region}). Data entered here updates the Scheduler App."
        "</div>",
        unsafe_allow_html=True,
    )

    # Granularity controls and Scale Adjustment
    gran_col, scale_col = st.columns([1, 1])

    with gran_col:
        st.markdown('<div class="granularity-label">Input Granularity</div>', unsafe_allow_html=True)
        granularity = st.radio(
            "Input Granularity",
            ["Daily", "Weekly", "Monthly"],
            horizontal=True,
            label_visibility="collapsed",
            key="granularity_radio"
        )

    with scale_col:
        st.markdown('<div class="granularity-label">📊 Scale Adjustment</div>', unsafe_allow_html=True)
        scale_percent = st.slider(
            "Adjust all target values by percentage",
            min_value=-50,
            max_value=50,
            value=0,
            step=5,
            key="scale_percent",
            label_visibility="collapsed",
            help="Increase or decrease all target values by the selected percentage"
        )

    # st.markdown(
    #     '<div class="hint-text">📅 Showing projections for selected date range</div>',
    #     unsafe_allow_html=True,
    # )
    if scale_percent != 0:
        st.info(f"All targets adjusted by **{scale_percent:+d}%**")
    st.write("")

    # Get date range based on granularity
    if granularity == "Daily":
        dates = pd.date_range(start=selected_date, periods=7, freq="D")
        date_headers = [d.strftime("%b %-d") if hasattr(d, "strftime") else d.strftime("%b %d") for d in dates]
    elif granularity == "Weekly":
        dates = pd.date_range(start=selected_date, periods=4, freq="W")
        date_headers = [f"Week {i + 1}" for i in range(len(dates))]
    else:  # Monthly
        dates = pd.date_range(start=selected_date, periods=3, freq="MS")
        date_headers = [d.strftime("%b %Y") for d in dates]

    # Products for this location (filtered by selection)
    region_products = (
        demo_df[
            (demo_df["Region"] == selected_region) &
            (demo_df["Location"] == selected_location)
        ]["Product"].unique()
    )

    # Further filter by selected products if not "All Products"
    if "All Products" not in selected_products:
        region_products = [p for p in region_products if p in selected_products]

    if len(region_products) == 0:
        st.warning("No products found for this region and selection.")
        return

    # Create data grid - Product-based with expandable consignee details
    np.random.seed(42)

    # Header row
    header_cols = st.columns([2] + [1] * len(date_headers))
    with header_cols[0]:
        st.markdown('<div class="grid-header-cell">PRODUCT / CONSIGNEE</div>', unsafe_allow_html=True)
    for i, dh in enumerate(date_headers):
        with header_cols[i + 1]:
            st.markdown(
                f'<div class="grid-header-cell" style="text-align:center;">{dh}</div>',
                unsafe_allow_html=True,
            )

    # Loop through each product
    for product in region_products:
        # First, get consignee data for all dates to calculate product totals
        consignees = ["Branded", "Unbranded", "Contracts"]

        # Product-level Target row (sum of all consignees)
        row_cols = st.columns([2] + [1] * len(date_headers))
        with row_cols[0]:
            st.markdown(f'<div class="location-cell"><b>{product} (Target)</b></div>', unsafe_allow_html=True)

        product_targets = []
        for i, date_val in enumerate(dates):
            with row_cols[i + 1]:
                # Calculate as sum of three consignees
                total_target = 0
                for consignee in consignees:
                    cons_data = demo_df[
                        (demo_df["Region"] == selected_region)
                        & (demo_df["Location"] == selected_location)
                        & (demo_df["Product"] == product)
                        & (demo_df["Consignee"] == consignee)
                        & (demo_df["Date"].dt.date == date_val.date())
                    ]
                    if len(cons_data) > 0:
                        total_target += int(cons_data["Target"].sum())

                # Fallback if no data
                if total_target == 0:
                    total_target = int(np.random.randint(3000, 8000))

                # Apply scale adjustment
                scaled_target = int(total_target * (1 + scale_percent / 100))
                product_targets.append(scaled_target)

                st.number_input(
                    f"{product}_{i}_target",
                    value=scaled_target,
                    label_visibility="collapsed",
                    key=f"target_{product}_total_{i}_{granularity}",
                    step=100,
                    help=f"Sum of Branded, Unbranded, and Contracts targets"
                )

        # Product-level Actuals row (sum of all consignees)
        actuals_cols = st.columns([2] + [1] * len(date_headers))
        with actuals_cols[0]:
            st.markdown(f'<div class="location-cell"><b>{product} (Actuals)</b></div>', unsafe_allow_html=True)

        product_actuals = []
        for i, date_val in enumerate(dates):
            with actuals_cols[i + 1]:
                # Calculate as sum of three consignees
                total_actuals = 0
                for consignee in consignees:
                    cons_data = demo_df[
                        (demo_df["Region"] == selected_region)
                        & (demo_df["Location"] == selected_location)
                        & (demo_df["Product"] == product)
                        & (demo_df["Consignee"] == consignee)
                        & (demo_df["Date"].dt.date == date_val.date())
                    ]
                    if len(cons_data) > 0:
                        total_actuals += int(cons_data["Actuals"].sum())

                # Fallback if no data
                if total_actuals == 0:
                    total_actuals = int(np.random.randint(2500, 7500))

                product_actuals.append(total_actuals)

                st.number_input(
                    f"{product}_{i}_actuals",
                    value=total_actuals,
                    label_visibility="collapsed",
                    key=f"actuals_{product}_total_{i}_{granularity}",
                    step=100,
                    help=f"Sum of Branded, Unbranded, and Contracts actuals"
                )

        # Expandable consignee breakdown
        with st.expander(f"View {product} by Consignee", expanded=False):
            consignees = ["Branded", "Unbranded", "Contracts"]

            for consignee in consignees:
                # Consignee Target row
                cons_target_cols = st.columns([2] + [1] * len(date_headers))
                with cons_target_cols[0]:
                    st.markdown(
                        f'<div class="location-cell" style="padding-left: 20px;">↳ {consignee} (Target)</div>',
                        unsafe_allow_html=True
                    )

                for i, date_val in enumerate(dates):
                    with cons_target_cols[i + 1]:
                        cons_data = demo_df[
                            (demo_df["Region"] == selected_region)
                            & (demo_df["Location"] == selected_location)
                            & (demo_df["Product"] == product)
                            & (demo_df["Consignee"] == consignee)
                            & (demo_df["Date"].dt.date == date_val.date())
                        ]

                        if len(cons_data) > 0:
                            value = int(cons_data["Target"].sum())
                        else:
                            value = int(np.random.randint(1000, 3000))

                        # Apply scale adjustment
                        scaled_value = int(value * (1 + scale_percent / 100))

                        st.number_input(
                            f"{product}_{consignee}_{i}_target",
                            value=scaled_value,
                            label_visibility="collapsed",
                            key=f"target_{product}_{consignee}_{i}_{granularity}",
                            step=50,
                        )

                # Consignee Actuals row
                cons_actuals_cols = st.columns([2] + [1] * len(date_headers))
                with cons_actuals_cols[0]:
                    st.markdown(
                        f'<div class="location-cell" style="padding-left: 20px;">↳ {consignee} (Actuals)</div>',
                        unsafe_allow_html=True
                    )

                for i, date_val in enumerate(dates):
                    with cons_actuals_cols[i + 1]:
                        cons_data = demo_df[
                            (demo_df["Region"] == selected_region)
                            & (demo_df["Location"] == selected_location)
                            & (demo_df["Product"] == product)
                            & (demo_df["Consignee"] == consignee)
                            & (demo_df["Date"].dt.date == date_val.date())
                        ]

                        if len(cons_data) > 0:
                            value = int(cons_data["Actuals"].sum())
                        else:
                            value = int(np.random.randint(800, 2800))

                        st.number_input(
                            f"{product}_{consignee}_{i}_actuals",
                            value=value,
                            label_visibility="collapsed",
                            key=f"actuals_{product}_{consignee}_{i}_{granularity}",
                            step=50,
                        )

                st.write("")

        st.write("")

    # Action buttons
    st.divider()
    b_reset, b_save, _, _ = st.columns([1, 1, 1, 2])

    with b_reset:
        if st.button("🔄 Reset", use_container_width=True, key="reset_btn"):
            st.success("Form reset (mock)")

    with b_save:
        if st.button("💾 Save Changes", use_container_width=True, type="primary", key="save_btn"):
            st.success("✅ Changes saved (mock)")

    st.write("")
    current_time = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    st.markdown(
        f'<div class="last-saved">Last saved: {current_time} IST (demo)</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# SCREEN 2: DASHBOARD DEMO
# ============================================================================


def screen_dashboard(selected_region, selected_location, selected_products, selected_date, end_date, scale_percent=0):
    st.markdown(f'<div class="main-title">Actuals & Target Dashboard - {selected_region}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">'
        "Demo: Marketing Performance Overview - High-level analytics and trends."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Filter data
    filtered_df = demo_df.copy()

    # Filter by region and location
    filtered_df = filtered_df[
        (filtered_df["Region"] == selected_region) &
        (filtered_df["Location"] == selected_location)
    ]

    # Filter by date range from sidebar
    filtered_df = filtered_df[
        (filtered_df["Date"].dt.date >= selected_date) &
        (filtered_df["Date"].dt.date <= end_date)
    ]

    # Filter by selected products if not "All Products"
    if "All Products" not in selected_products:
        filtered_df = filtered_df[filtered_df["Product"].isin(selected_products)]

    # Apply scale adjustment to targets
    if scale_percent != 0:
        filtered_df["Target"] = filtered_df["Target"] * (1 + scale_percent / 100)

    # --- KPIs ---
    st.subheader("📈 Key Performance Indicators")

    k1, k2, k3 = st.columns(3)

    total_target = filtered_df["Target"].sum()
    total_actuals = filtered_df["Actuals"].sum()
    total_variance = total_actuals - total_target
    variance_pct = (total_variance / total_target * 100) if total_target > 0 else 0

    with k1:
        st.metric("Total Target", format_bpd(total_target))

    with k2:
        st.metric("Total Actuals", format_bpd(total_actuals))

    with k3:
        if variance_pct >= 0:
            st.metric(
                "Variance",
                format_bpd(total_variance),
                f"{variance_pct:+.1f}%",
                delta_color="normal",
            )
        else:
            st.metric(
                "Variance",
                format_bpd(total_variance),
                f"{variance_pct:+.1f}%",
                delta_color="inverse",
            )

    st.divider()

    # --- Product Tabs ---
    # Get unique products for tabs
    available_tab_products = filtered_df["Product"].unique().tolist()

    # Create tabs for "All Products" + each individual product
    tab_names = ["All Products"] + available_tab_products
    product_tabs = st.tabs(tab_names)

    # Iterate through each tab
    for idx, tab in enumerate(product_tabs):
        with tab:
            # Filter data for this product tab
            if idx == 0:  # "All Products" tab
                tab_filtered_df = filtered_df.copy()
            else:
                current_product = available_tab_products[idx - 1]
                tab_filtered_df = filtered_df[filtered_df["Product"] == current_product]

            # --- Daily Summary for this product ---
            st.subheader("📊 Daily Summary")

            daily_summary = []
            recent_dates = sorted(tab_filtered_df["Date"].dt.date.unique(), reverse=True)

            for date in recent_dates:
                day_data = tab_filtered_df[tab_filtered_df["Date"].dt.date == date]

                # Calculate totals for variance
                day_target_total = day_data["Target"].sum()
                day_actuals_total = day_data["Actuals"].sum()
                day_variance = day_actuals_total - day_target_total
                day_variance_pct = (day_variance / day_target_total * 100) if day_target_total > 0 else 0

                # Calculate by consignee
                branded_data = day_data[day_data["Consignee"] == "Branded"]
                unbranded_data = day_data[day_data["Consignee"] == "Unbranded"]
                contracts_data = day_data[day_data["Consignee"] == "Contracts"]

                target_branded = branded_data["Target"].sum()
                target_unbranded = unbranded_data["Target"].sum()
                target_contracts = contracts_data["Target"].sum()

                actuals_branded = branded_data["Actuals"].sum()
                actuals_unbranded = unbranded_data["Actuals"].sum()
                actuals_contracts = contracts_data["Actuals"].sum()

                daily_summary.append(
                    {
                        "📅 Date": str(date),
                        "Target Branded": format_bpd(target_branded),
                        "Target Unbranded": format_bpd(target_unbranded),
                        "Target Contracts": format_bpd(target_contracts),
                        "Actuals Branded": format_bpd(actuals_branded),
                        "Actuals Unbranded": format_bpd(actuals_unbranded),
                        "Actuals Contracts": format_bpd(actuals_contracts),
                        "Variance %": f"{day_variance_pct:+.1f}%",
                    }
                )

            if daily_summary:
                summary_df = pd.DataFrame(daily_summary)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            else:
                st.info("No data available for the selected date range and product.")

    st.divider()

    # --- Product & Consignee Breakdown ---
    c1, c2 = st.columns(2)

    # By Product
    with c1:
        st.subheader("📦 Performance by Product")

        product_summary = []
        for product in filtered_df["Product"].unique():
            prod_data = filtered_df[filtered_df["Product"] == product]
            prod_target = prod_data["Target"].sum()
            prod_actuals = prod_data["Actuals"].sum()
            prod_variance = prod_actuals - prod_target
            prod_variance_pct = (
                prod_variance / prod_target * 100 if prod_target > 0 else 0
            )

            product_summary.append(
                {
                    "Product": product,
                    "Target": format_bpd(prod_target),
                    "Actuals": format_bpd(prod_actuals),
                    "Variance": f"{prod_variance:+,.0f}",
                    "Variance %": f"{prod_variance_pct:+.1f}%",
                }
            )

        product_df = pd.DataFrame(product_summary)
        st.dataframe(product_df, use_container_width=True, hide_index=True)

    # By Consignee
    with c2:
        st.subheader("👥 Performance by Consignee")

        consignee_summary = []
        for consignee in ["Branded", "Unbranded", "Contracts"]:
            cons_data = filtered_df[filtered_df["Consignee"] == consignee]
            cons_target = cons_data["Target"].sum()
            cons_actuals = cons_data["Actuals"].sum()
            cons_variance = cons_actuals - cons_target
            cons_variance_pct = (
                cons_variance / cons_target * 100 if cons_target > 0 else 0
            )

            consignee_summary.append(
                {
                    "Consignee": consignee,
                    "Target": format_bpd(cons_target),
                    "Actuals": format_bpd(cons_actuals),
                    "Variance": f"{cons_variance:+,.0f}",
                    "Variance %": f"{cons_variance_pct:+.1f}%",
                }
            )

        consignee_df = pd.DataFrame(consignee_summary)
        st.dataframe(consignee_df, use_container_width=True, hide_index=True)


# ============================================================================
# MAIN APP
# ============================================================================


def main():
    # HEADER
    st.markdown(
        """
    <div class="header-container">
        <div class="header-left">
            <div class="header-logo">HF</div>
            <div class="header-title">HFS Marketing Dashboard</div>
        </div>
        <div class="header-right">
            User: Marketing_User_1
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # SIDEBAR - Persistent Filters
    with st.sidebar:
        st.markdown('<div class="sidebar-label">🔍 Filters</div>', unsafe_allow_html=True)
        st.divider()

        # Region Filter
        st.markdown('<div class="sidebar-label">Region</div>', unsafe_allow_html=True)
        selected_region = st.selectbox(
            "Select Region",
            ["HEP", "PSR", "Rockies"],
            index=2,  # Default to Rockies
            key="global_region",
            label_visibility="collapsed",
        )

        # Location Filter - based on selected region
        st.markdown('<div class="sidebar-label">Location</div>', unsafe_allow_html=True)
        available_locations = demo_df[demo_df["Region"] == selected_region]["Location"].unique().tolist()
        selected_location = st.selectbox(
            "Select Location",
            available_locations,
            key="global_location",
            label_visibility="collapsed",
        )

        # Get available products for the selected region/location
        available_products = demo_df[
            (demo_df["Region"] == selected_region) &
            (demo_df["Location"] == selected_location)
        ]["Product"].unique().tolist()

        # Product Filter
        st.markdown('<div class="sidebar-label">Products</div>', unsafe_allow_html=True)
        selected_products = st.multiselect(
            "Products",
            options=available_products,
            default=available_products,
            key="global_products",
            label_visibility="collapsed",
        )
        # If nothing selected, show all
        if not selected_products:
            selected_products = ["All Products"]

        # Date Filter - Single date for Data Entry
        st.markdown('<div class="sidebar-label">Date Range</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            selected_date = st.date_input(
                "Start Date",
                value=datetime(2025, 12, 1),
                min_value=datetime(2025, 12, 1),
                max_value=datetime(2025, 12, 14),
                key="global_date",
                label_visibility="collapsed",
            )
        with c2:
            end_date = st.date_input(
                "End Date",
                value=datetime(2025, 12, 7),
                min_value=datetime(2025, 12, 1),
                max_value=datetime(2025, 12, 14),
                key="global_end_date",
                label_visibility="collapsed",
            )

    # Get scale_percent from session state if it exists (set in data entry screen)
    scale_percent = st.session_state.get("scale_percent", 0)

    # MAIN AREA - Tabs for Data Entry and Dashboard
    tab1, tab2 = st.tabs(["📋 Data Entry", "📊 Dashboard"])

    with tab1:
        screen_data_entry(selected_region, selected_location, selected_date, selected_products, scale_percent)

    with tab2:
        screen_dashboard(selected_region, selected_location, selected_products, selected_date, end_date, scale_percent)


if __name__ == "__main__":
    main()
