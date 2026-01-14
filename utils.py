import pandas as pd
import streamlit as st


FORECAST_VISIBLE_COLS: tuple[str, ...] = ("Rack/Lifting", "Opening Inv", "Close Inv")

# Columns we display in the details editor that may need formatting/hiding.
DISPLAY_NUMERIC_COLS: tuple[str, ...] = (
    "Opening Inv",
    "Opening Inv Fact",
    "Close Inv",
    "Close Inv Fact",
    "Batch In",
    "Batch In Fact",
    "Batch Out",
    "Batch Out Fact",
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
    """Return a display-friendly dataframe for the details editor.

    - Formats numeric columns with thousand separators and 2 decimals.
    - For rows where ``source == 'forecast'``, hides values for flow columns
      (everything except :data:`FORECAST_VISIBLE_COLS`) by displaying ``0.00``.

    Note: This function intentionally returns *strings* for the formatted columns.
    """

    if df is None or df.empty:
        return df

    df_display = df.copy()
    is_forecast = (
        df_display.get("source", "")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("forecast")
    )

    for col in DISPLAY_NUMERIC_COLS:
        if col not in df_display.columns:
            continue

        # Coerce to numeric for formatting; non-numeric values become NaN.
        s_num = pd.to_numeric(df_display[col], errors="coerce")

        # Hide forecast flows (keep Opening/Close + Rack/Lifting visible).
        if col not in FORECAST_VISIBLE_COLS:
            s_num = s_num.mask(is_forecast, 0.0)

        # Format as strings for TextColumn rendering.
        df_display[col] = s_num.fillna(0.0).map(lambda v: f"{float(v):,.2f}")

    return df_display


def dynamic_input_data_editor(data, key, **_kwargs):
    changed_key = f'{key}_khkhkkhkkhkhkihsdhsaskskhhfgiolwmxkahs'
    initial_data_key = f'{key}_khkhkkhkkhkhkihsdhsaskskhhfgiolwmxkahs__initial_data'

    def on_data_editor_changed():
        if 'on_change' in _kwargs:
            args = _kwargs['args'] if 'args' in _kwargs else ()
            kwargs = _kwargs['kwargs'] if 'kwargs' in _kwargs else {}
            _kwargs['on_change'](*args, **kwargs)
        st.session_state[changed_key] = True

    if changed_key in st.session_state and st.session_state[changed_key]:
        data = st.session_state[initial_data_key]
        st.session_state[changed_key] = False
    else:
        st.session_state[initial_data_key] = data

    __kwargs = _kwargs.copy()
    __kwargs.update({'data': data, 'key': key, 'on_change': on_data_editor_changed})
    return st.data_editor(**__kwargs)
