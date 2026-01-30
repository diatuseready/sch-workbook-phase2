import pandas as pd
import streamlit as st


FORECAST_VISIBLE_COLS: tuple[str, ...] = ("Rack/Lifting", "Opening Inv", "Close Inv")

# Columns we display in the details editor that may need formatting/hiding.
DISPLAY_NUMERIC_COLS: tuple[str, ...] = (
    "Opening Inv",
    "Opening Inv Fact",
    "Close Inv",
    "Close Inv Fact",
    "Receipts",
    "Receipts Fact",
    "Deliveries",
    "Deliveries Fact",
    "Rack/Lifting",
    "Rack/Lifting Fact",
    "Pipeline In",
    "Pipeline In Fact",
    "Pipeline Out",
    "Pipeline Out Fact",
    "Adjustments",
    "Adjustments Fact",
    "Gain/Loss",
    "Gain/Loss Fact",
    "Transfers",
    "Transfers Fact",
    "Production",
    "Production Fact",
)


def _format_forecast_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df_display = df.copy()
    for col in DISPLAY_NUMERIC_COLS:
        if col not in df_display.columns:
            continue

        # Coerce to numeric for formatting; non-numeric values become NaN.
        s_num = pd.to_numeric(df_display[col], errors="coerce")

        # Format as strings for TextColumn rendering.
        df_display[col] = s_num.fillna(0.0).map(lambda v: f"{float(v):,.2f}")

    return df_display


def dynamic_input_data_editor(data, key, **_kwargs):
    changed_key = f"{key}__changed"

    user_on_change = _kwargs.get("on_change")
    user_args = _kwargs.get("args", ())
    user_kwargs = _kwargs.get("kwargs", {})

    def on_data_editor_changed():
        # Preserve any caller-provided callback.
        if callable(user_on_change):
            user_on_change(*user_args, **user_kwargs)
        st.session_state[changed_key] = True

    __kwargs = _kwargs.copy()
    __kwargs.update({"data": data, "key": key, "on_change": on_data_editor_changed})
    return st.data_editor(**__kwargs)
