import numpy as np
import pandas as pd
import streamlit as st


FORECAST_VISIBLE_COLS: tuple[str, ...] = ("Rack/Lifting", "Opening Inv", "Close Inv")

DISPLAY_NUMERIC_COLS: tuple[str, ...] = (
    "Opening Inv", "Opening Inv Fact",
    "Close Inv", "Close Inv Fact",
    "Receipts", "Receipts Fact",
    "Deliveries", "Deliveries Fact",
    "Rack/Lifting", "Rack/Lifting Fact",
    "Pipeline In", "Pipeline In Fact",
    "Pipeline Out", "Pipeline Out Fact",
    "Adjustments", "Adjustments Fact",
    "Gain/Loss", "Gain/Loss Fact",
    "Transfers", "Transfers Fact",
    "Production", "Production Fact",
)


def _to_float(x) -> float:
    """Safely convert any value to float; returns 0.0 on failure."""
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return 0.0
        if isinstance(x, str):
            s = x.strip()
            if s in {"", "—", "-"}:
                return 0.0
            return float(s.replace(",", ""))
        return float(x)
    except Exception:
        return 0.0

def _to_float_or_none(x):
    """Best-effort parse to float; return None if blank/invalid."""
    if isinstance(x, str):
        x = x.replace(",", "").strip()
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return None if pd.isna(v) else float(v)

def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce a Series to numeric, tolerating formatted strings like '1,234.00'."""
    if s is None:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    s2 = s2.replace({"": np.nan, "—": np.nan, "-": np.nan})
    return pd.to_numeric(s2, errors="coerce")


def _sum_row(row: pd.Series, cols: list[str]) -> float:
    """Sum values of given columns in a Series row; treats missing/NaN as 0."""
    return float(sum(_to_float(row.get(c, 0.0)) for c in cols if c in row.index))


def _format_forecast_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df_display = df.copy()
    for col in DISPLAY_NUMERIC_COLS:
        if col not in df_display.columns:
            continue
        s_num = pd.to_numeric(df_display[col], errors="coerce")
        df_display[col] = s_num.fillna(0.0).map(lambda v: f"{float(v):,.2f}")
    return df_display


def dynamic_input_data_editor(data, key, **_kwargs):
    changed_key = f"{key}__changed"

    user_on_change = _kwargs.get("on_change")
    user_args = _kwargs.get("args", ())
    user_kwargs = _kwargs.get("kwargs", {})

    def on_data_editor_changed():
        if callable(user_on_change):
            user_on_change(*user_args, **user_kwargs)
        st.session_state[changed_key] = True

    __kwargs = _kwargs.copy()
    __kwargs.update({"data": data, "key": key, "on_change": on_data_editor_changed})
    return st.data_editor(**__kwargs)
