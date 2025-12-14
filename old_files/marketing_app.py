# Import python packages

import streamlit as st
import pandas as pd
import uuid
import time
import json
from decimal import Decimal
from datetime import datetime, timedelta, date
from functools import lru_cache
from snowflake.snowpark.context import get_active_session


st.set_page_config(page_title="Price Forecasting Input", layout="wide")
st.markdown(
    """
    <style>
      /* enable smooth scroll on the whole page */
      html {
        scroll-behavior: smooth;
      }
    </style>
    <!-- anchor target -->
    <div id="top"></div>
    """,
    unsafe_allow_html=True
)


@st.cache_resource(show_spinner=False)
def get_snowflake_session():
    return get_active_session()


# Get the current credentials
session = get_snowflake_session()
warehouse_sql = f"USE WAREHOUSE PRICE_FORECASTING_STREAMLIT_CUST_WH"
session.sql(warehouse_sql).collect()

col1, col2, col3 = st.columns([0.2, 0.7, 0.1])

calls_table_name = "CONSUMPTION.PU_HFS_COMMERCIAL.PRICE_FORECASTING_PRICE_CALLS"
forecast_updates_table = "CONSUMPTION.PU_HFS_COMMERCIAL.PRICE_FORECASTING_PRICE_CALL_FORECAST"
call_metrics_summary_table = "CONSUMPTION.PU_HFS_COMMERCIAL.PRICE_FORECASTING_CALL_SUMMARY_METRICS"
event_log_table = "CONSUMPTION.PU_HFS_COMMERCIAL.PRICE_FORECASTING_TRADER_APP_CALL_LOGS"
locks_table_name = "CONSUMPTION.PU_HFS_COMMERCIAL.PRICE_FORECASTING_PRICE_CALLS_LOCK"
call_lock_config_table_name = "CONSUMPTION.PU_HFS_COMMERCIAL.PRICE_FORECASTING_CALL_LOCK_CONFIG"


# pull all the CALL_IDs where LOCK_FLAG = 'Y'
@st.cache_data(ttl=3600, show_spinner=False)
def get_toggle_call_ids() -> set[int]:
    df = session.sql(f"""
        SELECT CALL_ID
        FROM {call_lock_config_table_name}
        WHERE LOCK_FLAG = 'Y'
    """).collect()
    # Snowpark returns integers already, but just in case:
    return {int(row["CALL_ID"]) for row in df}


TOGGLE_CALL_IDS = get_toggle_call_ids()

if "processing" not in st.session_state:
    st.session_state.processing = False

if "freeze_editors" not in st.session_state:
    st.session_state.freeze_editors = False

if "edited_calls" not in st.session_state:
    st.session_state["edited_calls"] = {}

# Keep track of calls the user has "Reviewed – Keep As Is" (no value changes).
# We'll add call_id keys here whenever the user ticks that checkbox.
if "reviewed_keys" not in st.session_state:
    st.session_state.reviewed_keys = set()


@st.dialog("Success", width="large", dismissible=False, on_dismiss="rerun")
def pop_up(count):
    if count == 0:
        msg = "No changes found—there's nothing to save"
        st.warning(f"{msg}")
    else:
        noun = "call" if count == 1 else "calls"
        verb = "has" if count == 1 else "have"
        message_text = f"{count} {noun} {verb} been successfully updated"
        st.success(f"{message_text}")
    st.session_state.processing = False
    st.session_state.reviewed_keys.clear()

    def process():
        for key, info in st.session_state["edited_calls"].items():
            new_df = info["edited_data"]

            # Apply display rename for CALL_ID 4
            if info["call_df"]["CALL_ID"].iloc[0] == 4:
                new_df.loc[
                    new_df["ENTRY"] == "DIFFERENTIAL - INPUT", "ENTRY"
                ] = "RESULTING CRACK"

            # overwrite the values that Apply Defaults and Reviewed rely on
            st.session_state[f"orig_data_{key}"] = new_df.copy()
            st.session_state[f"initial_orig_data_{key}"] = new_df.copy()

        st.session_state["edited_calls"].clear()

    if st.button("Ok", type="primary", on_click=process):
        st.rerun()


def block_ui(
    total_items: int,
    message: str = "Save in progress – feel free to minimize the browser, "
    "but don't close it."
):
    """
    Draws a semi-transparent blocker + banner + spinner + a sleek pill-shaped 
    progress bar, then returns (overlay_ph, progress_bar, percent_ph).
    """
    overlay_ph = st.empty()
    overlay_ph.markdown(
        f"""
        <style>
          /* full-screen translucent blocker */
          #ui-blocker {{
            position: fixed; top: 0; left: 0;
            width: 100vw; height: 100vh;
            background: rgba(0,0,0,0.4);
            z-index: 9998;
          }}

          /* banner message */
          #ui-blocker-message {{
            position: fixed; top: 80px; left: 50%;
            transform: translateX(-50%);
            background: #fff9c4;
            color: #333;
            padding: 10px 20px;
            font-size: 1.2rem; font-weight: 600;
            border-radius: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            z-index: 9999;
          }}

          /* spinner */
          #ui-spinner {{
            position: fixed; top: 163px; left: 50%;
            transform: translateX(-50%);
            width: 50px; height: 50px;
            border: 6px solid rgba(255,255,255,0.8);
            border-top: 6px solid #76c7c0;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            box-shadow: 0 0 8px rgba(0,0,0,0.2);
            z-index: 9999;
          }}
          @keyframes spin {{
            to {{ transform: translateX(-50%) rotate(360deg); }}
          }}

          /* progress bar container */
          .stProgress {{
            position: fixed !important;
            bottom: 200px !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            width: 60% !important;
            z-index: 10000 !important;
          }}
          /* track: remove box-shadow lines, keep pill shape */
          .stProgress > div {{
            background: #eceff1 !important;
          }}

          /* fill: use default color by not overriding */

          /* percentage text */
          .progress-percentage {{
            position: fixed !important;
            bottom: 230px !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            font-size: 2.0rem;  /* increased font size */
            font-weight: 700;
            color: #333; /* default text color */
            z-index: 10000 !important;
          }}
        </style>

        <div id="ui-blocker"></div>
        <div id="ui-blocker-message">
          {message}
        </div>
        <div id="ui-spinner"></div>
        """,
        unsafe_allow_html=True,
    )

    progress_bar = st.progress(0)
    percent_ph = st.empty()
    percent_ph.markdown(
        f"<div class='progress-percentage'>0/{total_items} call(s) completed</div>",
        unsafe_allow_html=True
    )

    return overlay_ph, progress_bar, percent_ph


def get_sw_lock_flag(_session) -> bool:
    df = _session.sql("""
        SELECT COALESCE(MAX(IFF(IS_LOCKED, 1, 0)), 0) AS L
        FROM CONSUMPTION.PU_HFS_COMMERCIAL.PRICE_FORECASTING_SUMMER_WINTER_PRICES
    """).to_pandas()
    return (not df.empty) and bool(df["L"].iloc[0])


is_locked = get_sw_lock_flag(session)
summer_winter_disabled = ["Summer", "Winter"] if is_locked else []


@lru_cache(maxsize=None)
def _normalize_unit(u: str) -> str:
    """Match exactly whatever your SQL UDF does under the covers."""
    return u.strip().upper().replace(" ", "")


@st.cache_data(ttl=3600, show_spinner=False)
def load_conversion_map(_session):
    """Fetches and builds the in-memory unit conversion dict once per hour."""
    df = (
        _session
          .sql("""
            SELECT *
            FROM CONSUMPTION.PU_HFS_COMMERCIAL.PRICE_FORECASTING_UNIT_OF_CONVERSION
          """)
          .to_pandas()
    )

    return {
        (_normalize_unit(r.INPUT_UNIT), _normalize_unit(r.TARGET_UNIT)): r.CONVERSION_FACTOR
        for r in df.itertuples()
    }


CONVERSION_MAP = load_conversion_map(session)


def convert_value(value: float, from_u: str, to_u: str) -> float:
    """
    • normalizes units
    • looks up factor (0 or missing → 1.0)
    • casts Decimal → float
    """
    key = (_normalize_unit(from_u), _normalize_unit(to_u))
    raw = CONVERSION_MAP.get(key, 0)
    factor = float(raw) if isinstance(raw, Decimal) else (raw or 1.0)

    return value * factor


# Function to load data from Snowflake
def load_data(table_name):
    try:
        input_calls_df = session.table(table_name)
        return input_calls_df.toPandas()
    except Exception as e:
        st.error(f"Error fetching data from snowflake: {e}")
        return pd.DataFrame()


def load_single_call(calls_table_name, call_id, prompt_wk):
    """
    Fetch every column for the one call & week from Snowflake.
    """
    df = session.sql(f"""
        SELECT *
        FROM {calls_table_name}
        WHERE CALL_ID = '{call_id}'
        AND PROMPT_WEEK_START_DATE = '{prompt_wk:%Y-%m-%d}'
    """)
    return df.toPandas()


def log_event(call_id, event_type, details_df):
    """
    Logs user actions into Snowflake for tracking and debugging.
    Ensures that date and timestamp columns are retained in proper format.
    """
    try:
        created_timestamp = datetime.now()

        # Ensure details_df is converted to JSON while retaining date formats
        if isinstance(details_df, pd.DataFrame):
            details_df = details_df.copy()
            # Convert date columns to string format
            for col in ["PROMPT_WEEK_START_DATE", "LAST_UPDATED_TIMESTAMP"]:
                if col in details_df.columns:
                    details_df[col] = details_df[col].apply(
                        lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None
                    )

            details_json = json.dumps(details_df.to_dict(orient="records"), indent=2)
            # Ensures date retention

        else:
            details_json = json.dumps(details_df)  # For non-DataFrame cases

        # Escape single quotes for safe insertion into Snowflake
        details_json = details_json.replace("'", "''")

        session.sql(f"""
            INSERT INTO {event_log_table}
            (CALL_ID, EVENT_TYPE, CREATED_TIMESTAMP, DETAILS)
            VALUES ('{call_id}', '{event_type}', '{created_timestamp}', '{details_json}')
        """).collect_nowait()

    except Exception as e:
        st.error(f"Error logging event: {e}")
        raise


def calculate_flatprice_and_crack_for_entered_diff(input_df, call_df, diff_df):
    df_dict = {}  # Dictionary to store intermediate DataFrames

    if (not input_df.empty) & (not call_df.empty) & (not diff_df.empty):
        call_id = diff_df["CALL_ID"].iloc[0]  # Track call ID for logging
        # Log entry for Differential - Input
        log_event(call_id, "DIFFERENTIAL_INPUT", 
                 f"User entered differentials: {diff_df.to_json()}")

        # if not call_df[call_df["ENTRY"] == "BASELINE"].empty:
        #     baseline_row = call_df[call_df["ENTRY"] == "BASELINE"].iloc[0]
        #     flatprice_row = call_df[call_df["ENTRY"] == "FLAT PRICE"].iloc[0]
        # Safely pull out baseline & flat-price
        baseline_rows = call_df[call_df["ENTRY"] == "BASELINE"]
        flatprice_rows = call_df[call_df["ENTRY"] == "FLAT PRICE"]
        # Only calculate FLAT PRICE if both exist
        if not baseline_rows.empty and not flatprice_rows.empty:
            baseline_row = baseline_rows.iloc[0]
            flatprice_row = flatprice_rows.iloc[0]

            # Calculate "Flat Price" by adding differentials to the baseline
            flat_price_df = diff_df[diff_df["FIELD_ORDER"] == 1].copy()
            flat_price_df["ENTRY"] = "FLAT PRICE"
            flat_price_df["FIELD_ORDER"] = 3

            for col in ["SW_PLUS1_AVG", "SW_PLUS2_AVG", "SW_PLUS3_AVG", "SW_PLUS4_AVG",
                        "SM_PLUS1_AVG", "SM_PLUS2_AVG", "SM_PLUS3_AVG", "SM_PLUS4_AVG",
                        "SUMMER_PRICE", "WINTER_PRICE"]:
                diff_unit = diff_df["UNITS"].iloc[0]
                baseline_unit = baseline_row["UNITS"]
                target_unit = flatprice_row["UNITS"]

                if diff_unit.startswith("%OF"):
                    percent_factor = diff_df[col] / 100
                    flat_price_val = percent_factor * baseline_row[col]
                    flat_price_df[col] = convert_value(flat_price_val, baseline_unit, target_unit)
                else:
                    # convert both pieces in one go via our in-memory map
                    converted_diff = convert_value(diff_df[col], diff_unit, target_unit)
                    converted_baseline = convert_value(baseline_row[col], baseline_unit, target_unit)
                    flat_price_df[col] = converted_diff + converted_baseline

            flat_price_df["UNITS"] = target_unit

            # Log entry for Flat Price
            log_event(call_id, "FLAT_PRICE_CALCULATED", 
                     f"Flat Price calculated: {flat_price_df.to_json()}")

            # Store in dictionary
            df_dict["flat_price"] = flat_price_df

        # if not call_df[call_df["ENTRY"] == "BASELINE CRACK"].empty:
        # baseline_crack_row = call_df[call_df["ENTRY"] == "BASELINE CRACK"].iloc[0]
        baseline_crack_rows = call_df[call_df["ENTRY"] == "BASELINE CRACK"]
        if not baseline_crack_rows.empty and "flat_price" in df_dict:
            baseline_crack_row = baseline_crack_rows.iloc[0]

            # Calculate "Resulting Crack" by subtracting the baseline crack from the flat price
            if "flat_price" in df_dict:  # Ensure "flat_price_df" exists
                resulting_crack_df = diff_df[diff_df["FIELD_ORDER"] == 1].copy()
                resulting_crack_df["ENTRY"] = "RESULTING CRACK"
                resulting_crack_df["FIELD_ORDER"] = 5

                flat_price_df = df_dict["flat_price"]

                for col in ["SW_PLUS1_AVG", "SW_PLUS2_AVG", "SW_PLUS3_AVG", "SW_PLUS4_AVG",
                            "SM_PLUS1_AVG", "SM_PLUS2_AVG", "SM_PLUS3_AVG", "SM_PLUS4_AVG",
                            "SUMMER_PRICE", "WINTER_PRICE"]:
                    source_unit = flat_price_df["UNITS"].iloc[0]
                    target_unit = baseline_crack_row["UNITS"]
                    # one line local conversion + subtraction
                    converted = convert_value(flat_price_df[col], source_unit, target_unit)
                    resulting_crack_df[col] = converted - baseline_crack_row[col]

                resulting_crack_df["UNITS"] = target_unit

                # --- For CALL_ID 4: keep a copy of Differential-Input row for display ---
                if call_id == 4:
                    diff_copy = diff_df[diff_df["FIELD_ORDER"] == 1].copy()
                    diff_copy["ENTRY"] = "RESULTING CRACK"
                    diff_copy["FIELD_ORDER"] = 5
                    # Merge user-entered values with calculated values
                    resulting_crack_df.update(diff_copy)

                # Log entry for Resulting Crack
                log_event(call_id, "RESULTING_CRACK_CALCULATED",
                          f"Resulting Crack calculated: {resulting_crack_df.to_json()}")

                # Store in dictionary
                df_dict["resulting_crack"] = resulting_crack_df

        # Process Baseline
        baseline_df = call_df[call_df["ENTRY"] == "BASELINE"]
        baseline_df = baseline_df[["CALL_ID", "ENTRY"]]
        baseline_df = pd.merge(
            baseline_df,
            input_df[["PROMPT_WEEK_START_DATE", "USER_ID", "CALL_ID", "ENTRY", "CALL_LABEL",
                      "CALL_ORDER", "FIELD_ORDER", "SW_PLUS1_AVG", "SW_PLUS2_AVG", "SW_PLUS3_AVG",
                      "SW_PLUS4_AVG", "SM_PLUS1_AVG", "SM_PLUS2_AVG", "SM_PLUS3_AVG",
                      "SM_PLUS4_AVG", "SUMMER_PRICE", "WINTER_PRICE"]],
            on=["CALL_ID", "ENTRY"],
            how="inner"
        )
        # Log entry for Baseline
        log_event(call_id, "BASELINE_LOADED", f"Baseline data loaded: {baseline_df.to_json()}")

        # Store in dictionary
        if not baseline_df.empty:
            df_dict["baseline"] = baseline_df

        # Store diff_df
        if not diff_df.empty:
            df_dict["diff"] = diff_df

        # Combine all DataFrames in the dictionary
        if df_dict:
            update_df = pd.concat(df_dict.values(), ignore_index=True)
            # copy metadata from the diff_df (or call_df) so no row ever has NaN here
            for meta_col in ["PROMPT_WEEK_START_DATE", "USER_ID", "CALL_LABEL", "CALL_ORDER"]:
                update_df[meta_col] = diff_df[meta_col].iloc[0]
            update_df["LAST_UPDATED_TIMESTAMP"] = pd.Timestamp.now()
            update_df["IS_LATEST"] = True

            # Log final DataFrame before saving
            log_event(call_id, "FINAL_COMBINED_DF", 
                     f"Final combined DataFrame before saving: {update_df.to_json()}")
        else:
            update_df = pd.DataFrame()

    return update_df


# Function to update a Snowflake table with new data
def update_snowflake_table(
    input_df, call_df, edit_df, table_name, market_condition_comments, 
    call_comments, copy_to_unchanged=False
):
    """
    Writes the edited call (single row) to the specified Snowflake table.
    Ensures that previous entries have IS_LATEST set to False before adding new entries.
    """
    try:
        # Prepare the DataFrame for saving
        df = calculate_flatprice_and_crack_for_entered_diff(input_df, call_df, edit_df)

        # Add user comments to the DataFrame
        df["MARKET_CONDITIONS_COMMENTS"] = market_condition_comments
        df["CALL_COMMENTS"] = call_comments

        # Define the order of columns for Snowflake compatibility
        column_order = [
            "PROMPT_WEEK_START_DATE",
            "USER_ID",
            "CALL_ID",
            "CALL_LABEL",
            "CALL_ORDER",
            "FIELD_ORDER",
            "SW_PLUS1_AVG",
            "SW_PLUS2_AVG",
            "SW_PLUS3_AVG",
            "SW_PLUS4_AVG",
            "SM_PLUS1_AVG",
            "SM_PLUS2_AVG",
            "SM_PLUS3_AVG",
            "SM_PLUS4_AVG",
            "LAST_UPDATED_TIMESTAMP",
            "IS_LATEST",
            "MARKET_CONDITIONS_COMMENTS",
            "CALL_COMMENTS",
            "UNITS",
            "SUMMER_PRICE",
            "WINTER_PRICE"
        ]
        df = df[column_order]

        # Extract unique identifiers for the edited call
        call_id = df["CALL_ID"].iloc[0]
        prompt_week_start_date = df["PROMPT_WEEK_START_DATE"].iloc[0]
        # If somehow the date is missing, bail out early
        if pd.isna(prompt_week_start_date):
            st.warning(f"Skipping save for CALL_ID {call_id}: missing prompt week start date")
            return

        # Otherwise format it cleanly
        date_str = prompt_week_start_date.strftime("%Y-%m-%d")

        # Mark previous entries as not the latest
        session.sql(f"""
            UPDATE {table_name}
            SET IS_LATEST = False
            WHERE CALL_ID = '{call_id}'
              AND PROMPT_WEEK_START_DATE = '{prompt_week_start_date}'
              AND IS_LATEST = TRUE
              AND FIELD_ORDER != 4
        """).collect()

        # Create a Snowpark DataFrame from the Pandas DataFrame
        snowflake_df = session.create_dataframe(
            df,
            schema={
                "PROMPT_WEEK_START_DATE": "DATE",
                "USER_ID": "NUMBER(38,0)",
                "CALL_ID": "INTEGER",
                "CALL_LABEL": "VARCHAR(16777216)",
                "CALL_ORDER": "NUMBER(38,0)",
                "FIELD_ORDER": "NUMBER(38,0)",
                "SW_PLUS1_AVG": "NUMBER(38,2)",
                "SW_PLUS2_AVG": "NUMBER(38,2)",
                "SW_PLUS3_AVG": "NUMBER(38,2)",
                "SW_PLUS4_AVG": "NUMBER(38,2)",
                "SM_PLUS1_AVG": "NUMBER(38,2)",
                "SM_PLUS2_AVG": "NUMBER(38,2)",
                "SM_PLUS3_AVG": "NUMBER(38,2)",
                "SM_PLUS4_AVG": "NUMBER(38,2)",
                "LAST_UPDATED_TIMESTAMP": "TIMESTAMP_LTZ(9)",
                "IS_LATEST": "BOOLEAN",
                "MARKET_CONDITIONS_COMMENTS": "VARCHAR(16777216)",
                "CALL_COMMENTS": "VARCHAR(16777216)",
                "UNITS": "VARCHAR(16777216)",
                "SUMMER_PRICE": "NUMBER(38,2)",
                "WINTER_PRICE": "NUMBER(38,2)",
            }
        )

        # Write data back to the Snowflake table
        snowflake_df.write.save_as_table(
            table_name,
            mode="append",  # Use "append" mode to add the data
            column_order="name"
        )

        # Provide feedback to the user in the UI
        # placeholder = st.empty()
        # placeholder.success("Updated!")
        # time.sleep(2)
        # placeholder.empty()

        if copy_to_unchanged:
            unchanged_val_table_name = (
                "CONSUMPTION.PU_HFS_COMMERCIAL."
                "PRICE_FORECASTING_PRICE_CALLS_CHECKED_UNCHANGED"
            )
            # Mark previous entries as not the latest
            session.sql(f"""
                UPDATE {unchanged_val_table_name}
                SET IS_LATEST = False
                WHERE CALL_ID = '{call_id}'
                AND PROMPT_WEEK_START_DATE = '{prompt_week_start_date}'
                AND IS_LATEST = TRUE
                AND FIELD_ORDER != 4
            """).collect()

            # Write data back to the Snowflake table for unchanged records 
            # and checked reviewed
            snowflake_df.write.save_as_table(
                unchanged_val_table_name,
                mode="append",  # Use "append" mode to add the data
                column_order="name"
            )

    except Exception as e:
        st.error(f"Error writing back to Snowflake table: {e}")
        raise


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


def apply_defaults(call_df, unique_key):
    # ADD SESSION STATE MANAGEMENT HERE SIMILAR TO data_editor
    default_rows = call_df[call_df["FIELD_ORDER"].isin([10, 11, 14])]

    if default_rows.empty:
        # st.warning("No defaults available to apply.")
        return None

    # Select the first default row
    default_row = default_rows.sort_values(by="FIELD_ORDER").iloc[0]

    edit_df = (
        st.session_state["edited_calls"].get(unique_key, {}).get("edited_data", call)
    )
    display_to_original_columns = {
        week_minus2_date.strftime(DISPLAY_DATE_FORMAT): "SW_MINUS2_AVG",
        week_minus1_date.strftime(DISPLAY_DATE_FORMAT): "SW_MINUS1_AVG",
        week1_date.strftime(DISPLAY_DATE_FORMAT): "SW_PLUS1_AVG",
        week2_date.strftime(DISPLAY_DATE_FORMAT): "SW_PLUS2_AVG",
        week3_date.strftime(DISPLAY_DATE_FORMAT): "SW_PLUS3_AVG",
        week4_date.strftime(DISPLAY_DATE_FORMAT): "SW_PLUS4_AVG",
        month1_date: "SM_PLUS1_AVG",
        month2_date: "SM_PLUS2_AVG",
        month3_date: "SM_PLUS3_AVG",
        month4_date: "SM_PLUS4_AVG",
        "Summer": "SUMMER_PRICE",
        "Winter": "WINTER_PRICE",
    }
    # Revert column names in edited_data to original names
    edit_df.rename(columns=display_to_original_columns, inplace=True)

    updated_edit_df = edit_df[edit_df["ENTRY"] == 'DIFFERENTIAL - INPUT'].copy()

    for col in ["SW_PLUS1_AVG", "SW_PLUS2_AVG", "SW_PLUS3_AVG", "SW_PLUS4_AVG",
                "SM_PLUS1_AVG", "SM_PLUS2_AVG", "SM_PLUS3_AVG", "SM_PLUS4_AVG"]:
        if pd.notna(default_row[col]):  # Only apply default if it's a valid number
            updated_edit_df[col] = default_row[col]  # Overwrite existing values

    edit_vol_df = edit_df[edit_df["ENTRY"] != 'DIFFERENTIAL - INPUT'].copy()
    ad_df = pd.concat([updated_edit_df, edit_vol_df])

    ad_df.rename(
        columns={
            "HISTORICAL_MARKER": "SYMBOL",
            "SW_MINUS2_AVG": week_minus2_date.strftime(DISPLAY_DATE_FORMAT),
            "SW_MINUS1_AVG": week_minus1_date.strftime(DISPLAY_DATE_FORMAT),
            "SW_PLUS1_AVG": week1_date.strftime(DISPLAY_DATE_FORMAT),
            "SW_PLUS2_AVG": week2_date.strftime(DISPLAY_DATE_FORMAT),
            "SW_PLUS3_AVG": week3_date.strftime(DISPLAY_DATE_FORMAT),
            "SW_PLUS4_AVG": week4_date.strftime(DISPLAY_DATE_FORMAT),
            "SM_PLUS1_AVG": month1_date,
            "SM_PLUS2_AVG": month2_date,
            "SM_PLUS3_AVG": month3_date,
            "SM_PLUS4_AVG": month4_date,
            "SUMMER_PRICE": "Summer",
            "WINTER_PRICE": "Winter",
        },
        inplace=True,
    )
    return ad_df


def call_cascade_insert_sp(call_id):
    try:
        cascade_sp_call = (
            f"CALL CONSUMPTION.PU_HFS_COMMERCIAL."
            f"RUN_PRICE_FORECASTING_PRICE_CALL_CASCADE_UPDATES({call_id})"
        )
        cascade_result = session.sql(cascade_sp_call).collect()
        cascade_sp_return_val = (
            cascade_result[0][0] 
            if cascade_result and len(cascade_result) > 0 
            else "No return message"
        )

        # default_values_sp_call = (
        #     f"CALL CONSUMPTION.PU_HFS_COMMERCIAL."
        #     f"UPDATE_PRICE_FORECASTING_PRICE_CALL_DEFAULT_VALUES_INSERT_SP({call_id})"
        # )
        # default_values_result = session.sql(default_values_sp_call).collect()
        # default_values_sp_return_val = (
        #     default_values_result[0][0] 
        #     if default_values_result and len(default_values_result) > 0 
        #     else "No return message"
        # )

        # cascade_ans_crack_sp_call = (
        #     f"CALL CONSUMPTION.PU_HFS_COMMERCIAL."
        #     f"UPDATE_PRICE_FORECASTING_PRICE_CALL_FORECASTS_ANS_CRACK_SP({call_id})"
        # )
        # cascade_ans_crack_result = session.sql(cascade_ans_crack_sp_call).collect()
        # cascade_ans_crack_sp_return_val = (
        #     cascade_ans_crack_result[0][0] 
        #     if cascade_ans_crack_result and len(cascade_ans_crack_result) > 0 
        #     else "No return message"
        # )
        # st.success(f"Updated!")
    except Exception as e:
        st.error(f"Error updating: {e}")
        raise


def get_last_updated_timestamp(timestamp):
    if not timestamp:  # If no timestamp exists
        return ""

    # Convert timestamp to datetime if it's a Pandas Timestamp
    if isinstance(timestamp, pd.Timestamp):
        updated_time = timestamp.to_pydatetime()
    elif isinstance(timestamp, datetime):
        updated_time = timestamp
    else:
        updated_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

    # Ensure both now and updated_time have the same timezone awareness
    if updated_time.tzinfo is not None:  # If updated_time is timezone-aware
        now = datetime.now(updated_time.tzinfo)  # Make now timezone-aware
    else:
        now = datetime.now()  # Keep now naive

    delta = now - updated_time

    # Determine the appropriate time unit
    if delta < timedelta(seconds=60):
        return f"{int(delta.total_seconds())} seconds ago"
    elif delta < timedelta(minutes=60):
        return f"{int(delta.total_seconds() // 60)} minutes ago"
    elif delta < timedelta(hours=24):
        return f"{int(delta.total_seconds() // 3600)} hours ago"
    else:
        return f"{int(delta.total_seconds() // 86400)} days ago"


def persist_lock_row(call_id, prompt_wk, selected_lock_until, comment_text):
    # 1) demote previous latest
    session.sql(f"""
        UPDATE {locks_table_name}
           SET IS_LATEST = FALSE
         WHERE CALL_ID = {call_id}
           AND IS_LATEST = TRUE;
    """).collect()

    prompt_week_str = pd.to_datetime(prompt_wk).strftime("%Y-%m-%d")

    if isinstance(selected_lock_until, date):
        # LOCK: start today, end at chosen date
        comment_sql = (comment_text or "").replace("'", "''")
        session.sql(f"""
            INSERT INTO {locks_table_name}
            (CALL_ID, IS_LOCKED, PROMPT_WEEK_START_DATE, LAST_UPDATED_TIMESTAMP, 
             IS_LATEST, LOCK_START_DATE, LOCK_END_DATE, COMMENTS)
            VALUES ({call_id}, TRUE, '{prompt_week_str}', CURRENT_TIMESTAMP(), TRUE,
                    CURRENT_DATE, DATE('{selected_lock_until.strftime('%Y-%m-%d')}'), 
                    '{comment_sql}');
        """).collect()
    else:
        # UNLOCK: new row with NULL dates; keep whatever comment_text is passed
        comment_sql = (comment_text or "").replace("'", "''")
        session.sql(f"""
            INSERT INTO {locks_table_name}
            (CALL_ID, IS_LOCKED, PROMPT_WEEK_START_DATE, LAST_UPDATED_TIMESTAMP, 
             IS_LATEST, LOCK_START_DATE, LOCK_END_DATE, COMMENTS)
            VALUES ({call_id}, FALSE, '{prompt_week_str}', CURRENT_TIMESTAMP(), TRUE,
                    NULL, NULL, '{comment_sql}');
        """).collect()


def _snapshot_all_lock_comments():
    """
    Copy every visible lock comment's current textarea value into its _staged twin.
    This ensures keystrokes aren't lost when *another* widget triggers a rerun.
    """
    for key in list(st.session_state.keys()):
        if key.endswith("_lock_comment"):
            base = key[: -len("_lock_comment")]
            cur = st.session_state.get(key, "") or ""
            st.session_state[f"{base}_lock_comment_staged"] = cur


input_df = load_data(calls_table_name)


# Column datattype conversion for UI compatibility
string_columns = [
    "DESCRIPTION",
    "CALL_LABEL",
    "HIERARCHY_1",
    "HIERARCHY_2",
    "HIERARCHY_3",
    "ENTRY",
    "UNITS",
    "HISTORICAL_MARKER",
    "USER_NAME",
    "TOPIC",
]
for col in string_columns:
    if col in input_df.columns:
        input_df[col] = input_df[col].fillna("").astype(str)

# Columns to be displayed in the editor dataframe i.e. call data
editor_display_columns = [
    "FIELD_ORDER",
    "ENTRY",
    "HISTORICAL_MARKER",
    "UNITS",
    "CALL_ID",
    "SW_MINUS2_AVG",
    "SW_MINUS1_AVG",
    "SW_PLUS1_AVG",
    "SW_PLUS2_AVG",
    "SW_PLUS3_AVG",
    "SW_PLUS4_AVG",
    "SM_PLUS1_AVG",
    "SM_PLUS2_AVG",
    "SM_PLUS3_AVG",
    "SM_PLUS4_AVG",
    "SUMMER_PRICE",
    "WINTER_PRICE",
]
# Call data that can be edited
editable_columns = [
    "SW_PLUS1_AVG",
    "SW_PLUS2_AVG",
    "SW_PLUS3_AVG",
    "SW_PLUS4_AVG",
    "SM_PLUS1_AVG",
    "SM_PLUS2_AVG",
    "SM_PLUS3_AVG",
    "SM_PLUS4_AVG",
    "SUMMER_PRICE",
    "WINTER_PRICE",
]

non_editable_columns = [
    col for col in editor_display_columns if col not in editable_columns
]

# Market context columns to be displayed on the UI
context_display_columns = [
    "ENTRY",
    "HISTORICAL_MARKER",
    "UNITS",
    "SW_PLUS1_AVG",
    "SW_PLUS2_AVG",
    "SW_PLUS3_AVG",
    "SW_PLUS4_AVG",
    "SM_PLUS1_AVG",
    "SM_PLUS2_AVG",
    "SM_PLUS3_AVG",
    "SM_PLUS4_AVG",
    "SUMMER_PRICE",
    "WINTER_PRICE",
]

# Results of the calculations for the differentials entered by the user
results_display_columns = [
    "ENTRY",
    "HISTORICAL_MARKER",
    "UNITS",
    "SW_MINUS2_AVG",
    "SW_MINUS1_AVG",
    "SW_PLUS1_AVG",
    "SW_PLUS2_AVG",
    "SW_PLUS3_AVG",
    "SW_PLUS4_AVG",
    "SM_PLUS1_AVG",
    "SM_PLUS2_AVG",
    "SM_PLUS3_AVG",
    "SM_PLUS4_AVG",
    "SUMMER_PRICE",
    "WINTER_PRICE",
]

# Authenticate via SSO
user_email = st.user.email
force_login_email = "l.a.pandya@hfsinclair.com"

if user_email.lower() == force_login_email.lower():
    user_first_name, user_last_name = "Force", "Login"
    user_info = "Force Login User"
    force_login = True
else:
    user_first_name, user_last_name = user_email.split("@")[0].split(".")
    user_info = f"{user_first_name} {user_last_name}"
    force_login = False

# Initialize session state for the selected user
if "selected_user" not in st.session_state:
    st.session_state.selected_user = ""

# TODO : ADD EMAIL TO THE DATASET
# Set up dropdown options
if input_df.empty or "USER_NAME" not in input_df.columns:
    dropdown_options = [""]
else:
    dropdown_options = [""] + input_df['USER_NAME'].sort_values().unique().tolist()

# Handle pre-authenticated user
if user_info:
    default_user = user_info if not force_login else force_login_email
    # Set the session state to default user if it's not set yet
    if not st.session_state.selected_user:
        st.session_state.selected_user = default_user

# Define a 2-column layout
col1, col2 = st.columns([4, 1])  # 4 parts for the first column, 1 part for the second (right column)

# st.info("Placeholder for displaying any user instruction if required")
with col1:
    st.title("Price Forecasting Input")

with col2:
    # Dropdown for user selection, now in the second (right) column
    selected_user = st.selectbox(
        "Select a user:",
        dropdown_options,
        index=dropdown_options.index(
            st.session_state.selected_user) if st.session_state.selected_user in dropdown_options else 0,
        key="user_dropdown",
    )

# Display containers if a user is selected
# Update session state with the dropdown value and rerun if changed
if selected_user != st.session_state.selected_user:
    st.session_state.selected_user = selected_user

    st.session_state.freeze_editors = False
    st.session_state.reviewed_keys.clear()
    st.session_state["edited_calls"].clear()
    st.rerun()

# If the user is authenticated (either directly or via dropdown)
if st.session_state.selected_user:
    # Filter DataFrame based on selected user
    input_df = input_df[input_df['USER_NAME'] == st.session_state.selected_user.upper()]

    # Update the `user` variable dynamically
    user = st.session_state.selected_user

topic = input_df['TOPIC'].unique()[0]
prompt_week = input_df['PROMPT_WEEK_START_DATE'].unique()[0]

if isinstance(prompt_week, str):
    prompt_week = pd.to_datetime(prompt_week)

# Calculate dynamic column headers i.e. convert the text column names to actual dates based on the prompt week
week_minus2_date = prompt_week - timedelta(weeks=2)
week_minus1_date = prompt_week - timedelta(weeks=1)
week1_date = prompt_week + timedelta(weeks=1)
week2_date = prompt_week + timedelta(weeks=2)
week3_date = prompt_week + timedelta(weeks=3)
week4_date = prompt_week + timedelta(weeks=4)
month1_date = (week1_date).strftime("%b")
month2_date = (week1_date + pd.offsets.MonthBegin(1)).strftime("%b")
month3_date = (week1_date + pd.offsets.MonthBegin(2)).strftime("%b")
month4_date = (week1_date + pd.offsets.MonthBegin(3)).strftime("%b")
DISPLAY_DATE_FORMAT = '%m/%d/%y'
# for lock prizes
month4_start = week1_date + pd.offsets.MonthBegin(3)          # first day of month4 (e.g., 2025-12-01)
month4_end = (month4_start + pd.offsets.MonthEnd(1)).date()  # last day of month4 (e.g., 2025-12-31)
min_lock_date = month4_start  # + timedelta(days=1)
max_lock_date = min_lock_date + timedelta(days=365*2)

# --- Month4-aware DB sync ---


def expire_locked_calls_on_pricing_date():
    try:
        # First day of month4 (e.g., 2025-12-01)
        m4_month_first = pd.to_datetime(month4_start).strftime("%Y-%m-01")
        pw_str = pd.to_datetime(prompt_week).strftime("%Y-%m-%d")

        # (A) Expire month4-violating locks in-place
        session.sql(f"""
            UPDATE {locks_table_name}
               SET IS_LATEST = FALSE,
                   IS_LOCKED = FALSE
             WHERE IS_LATEST = TRUE
               AND IS_LOCKED = TRUE
               AND LOCK_END_DATE IS NOT NULL
               -- expire when lock_end's month is strictly before this view's month4 month
               AND DATE_TRUNC('MONTH', LOCK_END_DATE) < DATE('{m4_month_first}');
        """).collect()

    except Exception as e:
        pass
        # st.warning(f"Month4 DB lock sync failed: {e}")


def is_lock_active_for_view_month4(lock_end_date, prompt_wk) -> bool:
    # month4 for this view
    m4_start = prompt_wk + timedelta(weeks=1) + pd.offsets.MonthBegin(3)
    m4_ym = (int(m4_start.year), int(m4_start.month))
    lock_end_dt = pd.to_datetime(lock_end_date).date()
    lock_ym = (lock_end_dt.year, lock_end_dt.month)
    # active while month4 (YYYY,MM) <= lock_end (YYYY,MM)
    return m4_ym <= lock_ym


# Load Dataframe for locked calls
locks_df = session.sql(
    f"""
    SELECT CALL_ID, LOCK_START_DATE, LOCK_END_DATE,COMMENTS
      FROM {locks_table_name}
     WHERE IS_LOCKED = TRUE
       AND IS_LATEST = TRUE
       AND LOCK_START_DATE IS NOT NULL
       AND LOCK_END_DATE   IS NOT NULL
    """
).collect()

# Build the effective lock map for this view
locked_call_info = {
    row['CALL_ID']: (row['LOCK_START_DATE'], row['LOCK_END_DATE'], row['COMMENTS'])
    for row in locks_df
    if is_lock_active_for_view_month4(row['LOCK_END_DATE'], prompt_week)
}

locked_call_ids = set(locked_call_info.keys())


def get_latest_lock_comment(call_id) -> str:
    """
    Return the newest comment for this call_id
    """
    df = session.sql(f"""
        SELECT COMMENTS
          FROM {locks_table_name}
         WHERE CALL_ID = {call_id}
         ORDER BY LAST_UPDATED_TIMESTAMP DESC
         LIMIT 1
    """).to_pandas()
    return (df["COMMENTS"].iloc[0] if not df.empty and df["COMMENTS"].iloc[0] else "").strip()


# Calculate call metrics
call_metrics_df = load_data(call_metrics_summary_table)
call_metrics_summary = call_metrics_df[call_metrics_df['USER_NAME'] == st.session_state.selected_user.upper()]

total_calls = call_metrics_summary["TOTAL_CALLS"].unique()[0]
updated_calls = call_metrics_summary["UPDATED_CALLS"].unique()[0]
pending_calls = call_metrics_summary["PENDING_CALLS"].unique()[0]
unchanged_calls = call_metrics_summary["UNCHANGED_CALLS"].unique()[0]

# st.info("Placeholder for any user instructions")

# 2) re-render your fixed bar
st.markdown(
    f"""
<style>
    .sticky-metrics {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #f0f0f0;
        border-bottom: 2px solid #ddd;
        padding: 12px 24px;
        font-size: 16px;
        z-index: 9999;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: Arial, sans-serif;
        flex-wrap: wrap;
    }}
    .sticky-left {{
        display: flex;
        flex-wrap: wrap;
        gap: 24px;
        flex: 1;
        min-width: 0;
    }}
    .metric {{
        white-space: nowrap;
    }}
    .metric-value {{
        color: green;
        margin-left: 4px;
    }}
    .sticky-right {{
        font-size: 16px;
        color: black;
        font-weight: bold;
        white-space: normal;
        text-align: right;
        min-width: 250px;
        max-width: 300px;
        overflow-wrap: anywhere;
        flex-shrink: 0;
        margin-left: auto;
    }}
    .stApp {{
        padding-top: 80px;
    }}
</style>

<div class="sticky-metrics">
    <div class="sticky-left">
        <span class="metric"><strong>User:</strong><span class="metric-value"><i>{user}</i></span></span>
        <span class="metric"><strong>Topic:</strong><span class="metric-value"><i>{topic}</i></span></span>
        <span class="metric"><strong>Total Calls:</strong><span class="metric-value">{total_calls}</span></span>
        <span class="metric"><strong>Updated Calls:</strong><span class="metric-value">{updated_calls}</span></span>
        <span class="metric"><strong>Pending Calls:</strong><span class="metric-value">{pending_calls}</span></span>
        <span class="metric"><strong>Unchanged Calls:</strong><span class="metric-value">{unchanged_calls}</span></span>
        <span class="metric"><strong>Prompt Week:</strong><span class="metric-value">{prompt_week.strftime("%Y-%m-%d")}</span></span>
    </div>
    <div class="sticky-right">
        Page will expire if inactive for 15 mins
    </div>
</div>
""",
    unsafe_allow_html=True
)

input_df = input_df.sort_values(by=["CALL_ORDER", "CALL_ID", "FIELD_ORDER"]).reset_index(drop=True)

container = st.container()

with container:
    col_main, col_cb, col_button = st.columns([7, 1.6, 1])
    with col_cb:
        freeze_state = st.checkbox(
            "**Confirm Changes**",
            key ="freeze_editors",
            help = "Select the “Confirm Changes” checkbox to enable the Save button"
        )

    with col_button:
        save_button_placeholder = col_button.empty()
        # spinner_ph = col_button.empty()   # where the spinner will go
        # status_ph  = col_button.empty()   # where the "Saved!" message will go

    subcategories = input_df["HIERARCHY_1"].dropna().unique()

    for subcategory in subcategories:
        with st.expander(
                f"**:green[{subcategory}]**", expanded=False):
            st.markdown("<hr style='border: 2px solid #d3d3d3;'>", unsafe_allow_html=True, )

            subcategory_data = input_df[input_df["HIERARCHY_1"] == subcategory]

            calls = []
            current_call = []
            previous_CALL_LABEL = None

            for _, row in subcategory_data.iterrows():
                current_CALL_LABEL = row["CALL_LABEL"]

                if current_CALL_LABEL != previous_CALL_LABEL and current_call:
                    calls.append(pd.DataFrame(current_call))
                    current_call = []

                current_call.append(row)
                previous_CALL_LABEL = current_CALL_LABEL

            if current_call:
                calls.append(pd.DataFrame(current_call))

            for i, call in enumerate(calls):
                call_identifier = call["CALL_LABEL"].iloc[0]
                call_info = call["DESCRIPTION"].iloc[0]
                unique_key = f"call_{i}_{call_identifier}"
                call_id = call["CALL_ID"].iloc[0]
                k = f"call_{call_id}"

                if call_id in locked_call_ids:
                    expire_locked_calls_on_pricing_date()

                # NEW: pre-seed comment & date for this call, once, preferring staged text
                ck = f"{k}_lock_comment"
                if ck not in st.session_state:
                    st.session_state[ck] = (
                        st.session_state.get(f"{k}_lock_comment_staged")          # user-typed (if any)
                        or get_latest_lock_comment(call_id)                        # DB last-known
                        or ""
                    )
                row_index = call.index[0]
                uuid_key = f"{unique_key}_editor_id"
                if uuid_key not in st.session_state:
                    st.session_state[uuid_key] = uuid.uuid4().hex

                comment_key_1 = f"comment_1_{unique_key}"
                comment_key_2 = f"comment_2_{unique_key}"

                with st.form(key=f"save_form_{unique_key}"):
                    col1_1, col2_1, col3_1 = st.columns([0.8, 0.18, 0.18])
                    with col1_1:
                        # st.markdown(
                        #     f"""<span style='font-weight:bold;'>{call_identifier}:</span> <span>{call_info}</span>""",
                        #     unsafe_allow_html=True,
                        # )
                        st.markdown(
                            f"""<span style='font-weight:bold;'>{call_identifier}""",
                            unsafe_allow_html=True,
                        )

                    with col2_1:
                        call_id = call["CALL_ID"].iloc[0]
                        apply_defaults_button = st.form_submit_button(
                            "Apply Defaults", disabled=call_id in locked_call_ids, type="primary"
                        )
                        status_placeholder = st.empty()

                    # with col3_1:
                    #     pass
                    #     # save_button = st.form_submit_button("Save Call")

                    with col3_1:
                        # Display when the call was last updated
                        # Get the current call's ID
                        call_id = call["CALL_ID"].iloc[0]

                        # Check if there is a valid last_updated_timestamp for ENTRY = "DIFFERENTIAL - INPUT"
                        differential_row = call[
                            (call["ENTRY"] == "DIFFERENTIAL - INPUT") & (call["CALL_ID"] == call_id)
                            ]
                        if call_id in locked_call_ids:
                            # Locked → always "Updated", but show last-updated time if available
                            status = "Updated"
                            color = "green"
                            if not differential_row.empty and pd.notna(differential_row["LAST_UPDATED_TIMESTAMP"].iloc[0]):
                                relative_time = get_last_updated_timestamp(
                                    differential_row["LAST_UPDATED_TIMESTAMP"].iloc[0]
                                )
                            else:
                                relative_time = ""
                        else:
                            if not differential_row.empty and pd.notna(differential_row["LAST_UPDATED_TIMESTAMP"].iloc[0]):
                                last_updated_timestamp = differential_row["LAST_UPDATED_TIMESTAMP"].iloc[0]
                                prompt_week_start_date = differential_row["PROMPT_WEEK_START_DATE"].iloc[0]

                                last_updated_timestamp = pd.to_datetime(last_updated_timestamp).tz_localize(None)
                                prompt_week_start_date_dt = pd.to_datetime(prompt_week_start_date).tz_localize(None)
                                prompt_week_start_date_dt_minus_1day = prompt_week_start_date_dt - pd.Timedelta(days=1)

                                if last_updated_timestamp >= prompt_week_start_date_dt_minus_1day:
                                    # Call is updated
                                    status = "Updated"
                                    color = "green"
                                    relative_time = get_last_updated_timestamp(
                                        differential_row["LAST_UPDATED_TIMESTAMP"].iloc[0])
                                else:
                                    status = "Pending"
                                    color = "gray"
                                    relative_time = ""
                            else:
                                # Call is pending
                                status = "Pending"
                                color = "gray"
                                relative_time = ""

                        # Display Status and Relative Time
                        st.markdown(
                            f"<span style='font-size:18px; color:{color}; font-weight:bold;'>{status} {relative_time}</span>",
                            # f"<span style='font-size:14px; color:{color}; font-weight:bold;'>{relative_time}</span>",
                            unsafe_allow_html=True,
                        )

                # Apply custom logic only for CALL_ID == 4
                call_id = call["CALL_ID"].iloc[0]

                # 1. Get the "current" DataFrame (from session_state if exists, else backend)
                display_call = st.session_state.get("edited_calls", {}).get(unique_key, {}).get("call_df", call.copy())

                if call_id == 4:
                    # Rename 'DIFFERENTIAL - INPUT' to 'RESULTING CRACK' for display
                    display_call.loc[display_call["ENTRY"] == "DIFFERENTIAL - INPUT", "ENTRY"] = "RESULTING CRACK"

                # Update display column names
                dynamic_columns_config = {
                    "ENTRY": st.column_config.TextColumn(width="medium"),
                    "SYMBOL": st.column_config.TextColumn(width='small'),
                    "HISTORICAL_MARKER": st.column_config.TextColumn(width="small"),
                    "UNITS": st.column_config.TextColumn(width="small"),
                    "SW_MINUS2_AVG": st.column_config.NumberColumn(
                        label=week_minus2_date.strftime("%Y-%m-%d"),
                        width="medium"
                    ),
                    "SW_MINUS1_AVG": st.column_config.NumberColumn(
                        label=week_minus1_date.strftime("%Y-%m-%d"),
                        width="medium"
                    ),
                    "SW_PLUS1_AVG": st.column_config.NumberColumn(
                        label=week1_date.strftime("%Y-%m-%d"),
                        width="medium"
                    ),
                    "SW_PLUS2_AVG": st.column_config.NumberColumn(
                        label=week2_date.strftime("%Y-%m-%d"),
                        width="medium"
                    ),
                    "SW_PLUS3_AVG": st.column_config.NumberColumn(
                        label=week3_date.strftime("%Y-%m-%d"),
                        width="medium"
                    ),
                    "SW_PLUS4_AVG": st.column_config.NumberColumn(
                        label=week4_date.strftime("%Y-%m-%d"),
                        width="medium"
                    ),
                    "SM_PLUS1_AVG": st.column_config.NumberColumn(label=month1_date, width="medium"),
                    "SM_PLUS2_AVG": st.column_config.NumberColumn(label=month2_date, width="medium"),
                    "SM_PLUS3_AVG": st.column_config.NumberColumn(label=month3_date, width="medium"),
                    "SM_PLUS4_AVG": st.column_config.NumberColumn(label=month4_date, width="medium"),
                    "SUMMER_PRICE": st.column_config.NumberColumn(width="medium"),
                    "WINTER_PRICE": st.column_config.NumberColumn(width="medium"),
                }
                persisted_edited_data = st.session_state.get("edited_calls", {}).get(unique_key, {}).get("edited_data")
                if persisted_edited_data is not None:
                    # If state exists in session_state, use it as the starting point for the editor
                    data_for_editor = persisted_edited_data.copy()  # Use a copy
                else:
                    # Editable Table
                    call_data = display_call[editor_display_columns].reset_index(drop=True)
                    # edit_original_df = call_data.loc[call_data["FIELD_ORDER"].isin([1, 15, 16])].copy()

                    edit_df = call_data.loc[call_data["FIELD_ORDER"].isin([1, 15, 16])].copy()
                    edit_df = edit_df.drop(columns=["FIELD_ORDER", "CALL_ID"])
                    edit_df.rename(
                        columns={
                            "HISTORICAL_MARKER": "SYMBOL",
                            "SW_MINUS2_AVG": week_minus2_date.strftime(DISPLAY_DATE_FORMAT),
                            "SW_MINUS1_AVG": week_minus1_date.strftime(DISPLAY_DATE_FORMAT),
                            "SW_PLUS1_AVG": week1_date.strftime(DISPLAY_DATE_FORMAT),
                            "SW_PLUS2_AVG": week2_date.strftime(DISPLAY_DATE_FORMAT),
                            "SW_PLUS3_AVG": week3_date.strftime(DISPLAY_DATE_FORMAT),
                            "SW_PLUS4_AVG": week4_date.strftime(DISPLAY_DATE_FORMAT),
                            "SM_PLUS1_AVG": month1_date,
                            "SM_PLUS2_AVG": month2_date,
                            "SM_PLUS3_AVG": month3_date,
                            "SM_PLUS4_AVG": month4_date,
                            "SUMMER_PRICE": "Summer",
                            "WINTER_PRICE": "Winter",
                        },
                        inplace=True,
                    )
                    data_for_editor = edit_df

                non_editable_columns_mapped = [
                    col.replace("SW_MINUS2_AVG", week_minus2_date.strftime(DISPLAY_DATE_FORMAT))
                    .replace("SW_MINUS1_AVG", week_minus1_date.strftime(DISPLAY_DATE_FORMAT))
                    .replace("SW_PLUS2_AVG", week2_date.strftime(DISPLAY_DATE_FORMAT))
                    .replace("SW_PLUS3_AVG", week3_date.strftime(DISPLAY_DATE_FORMAT))
                    .replace("SW_PLUS4_AVG", week4_date.strftime(DISPLAY_DATE_FORMAT))
                    .replace("SM_PLUS1_AVG", month1_date)
                    .replace("SM_PLUS2_AVG", month2_date)
                    .replace("SM_PLUS3_AVG", month3_date)
                    .replace("SM_PLUS4_AVG", month4_date)
                    for col in non_editable_columns
                ]
                # 1a) forever save the very first data_for_editor you ever saw (col names + values)
                initial_orig_key = f"initial_orig_data_{unique_key}"
                if initial_orig_key not in st.session_state:
                    st.session_state[initial_orig_key] = data_for_editor.copy(deep=True)

                # 1b) this is the "current baseline" that Apply Defaults will overwrite
                orig_key = f"orig_data_{unique_key}"
                if orig_key not in st.session_state:
                    st.session_state[orig_key] = st.session_state[initial_orig_key].copy(deep=True)

                # 1c) actually hand the editor a working copy of that baseline
                working_df = st.session_state[orig_key].copy()

                if apply_defaults_button:
                    applied_defaults_df = apply_defaults(call, unique_key)
                    if applied_defaults_df is not None:
                        data_for_editor = applied_defaults_df  # Use the DataFrame with defaults
                        orig_key = f"orig_data_{unique_key}"
                        st.session_state[orig_key] = data_for_editor.copy(deep=True)
                        st.session_state["edited_calls"][unique_key] = {
                                    "edited_data": data_for_editor,
                                    "call_df": call.copy()
                                }
                        st.success("Defaults applied!")  # Optional feedback
                        st.session_state[uuid_key] = uuid.uuid4().hex
                        _snapshot_all_lock_comments()
                        st.rerun()
                    else:
                        st.warning("No defaults available.")

                # col1, col2= st.columns([9, 2])
                # with col2:
                col_toggle, col_review = st.columns([70, 16])
                with col_review:
                    # "Reviewed (No Changes)" checkbox, placed right after Apply Defaults:
                    keep_key = f"keep_as_is_{unique_key}"
                    is_reviewed = unique_key in st.session_state.reviewed_keys

                    keep_checked = st.checkbox(
                        "**Reviewed(No Changes)**",
                        key=keep_key,
                        value=is_reviewed,
                        disabled=call_id in locked_call_ids,
                        help="Mark this call as reviewed if you don't want to change any values"
                        )

                with col_toggle:
                    show_toggle = (call_id in TOGGLE_CALL_IDS)
                    if show_toggle:
                        # col_label, col_date, col_unlock = st.columns([0.6, 0.9, 0.7])
                        # col_label, col_date, col_unlock = st.columns([0.4, 0.6, 0.7])
                        col_label, col_date, col_unlock, col_comment = st.columns([5.4, 9.5, 6.5, 48])

                        with col_label:
                            st.markdown("**Lock until**")

                        with col_date:
                            locked = (call_id in locked_call_ids)
                            lock_end_value = (
                                locked_call_info.get(call_id, (None, None, None))[1] 
                                if locked else None
                            )
                            # --- Clear the date widget on the run *after* unlock 
                            # (must happen BEFORE rendering the widget) ---
                            _clear_flag = f"{k}_clear_lock"
                            date_key = f"{k}_lock_until"
                            if st.session_state.get(_clear_flag):
                                st.session_state.pop(date_key, None)
                                st.session_state[_clear_flag] = False

                            # date selected ⇒ LOCK; empty ⇒ UNLOCK (handled via Unlock button)
                            date_kwargs = dict(
                                label="",
                                min_value=min_lock_date,
                                max_value=max_lock_date,
                                key=date_key,
                                label_visibility="collapsed",
                            )

                            # Only pass 'value' when Session State doesn't already hold this widget's value
                            # if f"{k}_lock_until" not in st.session_state:
                                # If currently locked, show DB end-date; otherwise omit value 
                                # (Streamlit will pick a valid default ≥ min_value)
                            if locked:
                                date_kwargs["value"] = lock_end_value  # from DB when locked
                            else:
                                date_kwargs["value"] = None

                            lock_until = st.date_input(**date_kwargs, disabled=locked)

                            # # date selected ⇒ LOCK; empty ⇒ UNLOCK
                            # lock_until = st.date_input(
                            #     label="",
                            #     value=lock_end_value,
                            #     min_value=min_lock_date,
                            #     key=f"{k}_lock_until",
                            #     label_visibility="collapsed"
                            # )
                        with col_unlock:
                            # Always show "Unlock"; enable only when the call is currently locked
                            unlock_disabled = call_id not in locked_call_ids
                            if st.button("Unlock", key=f"{k}_unlock_btn", disabled=unlock_disabled, type='primary'):
                                overlay_ph, progress, pct_ph = block_ui(
                                    total_items=1, message="Unlocking..."
                                )
                                pct_ph.markdown("<div class='progress-percentage'>Unlocking…</div>", unsafe_allow_html=True)

                                _snapshot_all_lock_comments()  # NEW: capture *all* lock comment textareas

                                # still snapshot this call explicitly (harmless redundancy)
                                current_comment = st.session_state.get(f"{k}_lock_comment", "")
                                st.session_state[f"{k}_lock_comment_staged"] = current_comment

                                persist_lock_row(call_id, prompt_week, None, current_comment)

                                st.session_state[f"{k}_clear_lock"] = True
                                overlay_ph.empty()
                                progress.empty()
                                pct_ph.empty()
                                # st.success(f"Call {call_id} has been unlocked.")
                                st.rerun()
                        with col_comment:
                            # 2a) seed once so reruns don't pull from DB again
                            comment_key = f"{k}_lock_comment"
                            if comment_key not in st.session_state:
                                st.session_state[comment_key] = get_latest_lock_comment(call_id) or ""

                            # 2b) stage latest keystroke reliably
                            def _stage_comment():
                                raw = st.session_state.get(f"{k}_lock_comment", "")
                                st.session_state[f"{k}_lock_comment_staged"] = raw
                                st.session_state[f"{k}_lock_comment_dirty"] = True

                            st.text_area(
                                "Lock comment",
                                key=comment_key,
                                on_change=_stage_comment,              # <-- important
                                placeholder="Enter your comments.",
                                label_visibility="collapsed",
                                height=5,
                                disabled=locked
                                        )
                        effective_locked = (call_id in locked_call_ids)
                        disabled_param = (
                            True if effective_locked 
                            else non_editable_columns_mapped + summer_winter_disabled
                        )
                    else:
                        disabled_param = non_editable_columns_mapped + summer_winter_disabled

                if keep_checked and unique_key not in st.session_state.reviewed_keys:
                    st.session_state.reviewed_keys.add(unique_key)
                    _snapshot_all_lock_comments()
                    # st.rerun()
                elif not keep_checked and unique_key in st.session_state.reviewed_keys:
                    st.session_state.reviewed_keys.discard(unique_key)
                    _snapshot_all_lock_comments()
                    st.rerun()

                if keep_checked:
                    no_edit_df = st.session_state[initial_orig_key]
                    no_edit_df.rename(
                        columns={
                            "HISTORICAL_MARKER": "SYMBOL",
                            "SW_MINUS2_AVG": week_minus2_date.strftime(DISPLAY_DATE_FORMAT),
                            "SW_MINUS1_AVG": week_minus1_date.strftime(DISPLAY_DATE_FORMAT),
                            "SW_PLUS1_AVG": week1_date.strftime(DISPLAY_DATE_FORMAT),
                            "SW_PLUS2_AVG": week2_date.strftime(DISPLAY_DATE_FORMAT),
                            "SW_PLUS3_AVG": week3_date.strftime(DISPLAY_DATE_FORMAT),
                            "SW_PLUS4_AVG": week4_date.strftime(DISPLAY_DATE_FORMAT),
                            "SM_PLUS1_AVG": month1_date,
                            "SM_PLUS2_AVG": month2_date,
                            "SM_PLUS3_AVG": month3_date,
                            "SM_PLUS4_AVG": month4_date,
                            "SUMMER_PRICE": "Summer",
                            "WINTER_PRICE": "Winter",
                        },
                        inplace=True,
                    )

                    # Reviewed mode: show the pristine copy
                    st.dataframe(
                        no_edit_df,
                        column_config=dynamic_columns_config,
                        hide_index=True,

                        use_container_width=True
                    )
                    edited_data = no_edit_df

                else:
                    editor_key = f"data_editor_{unique_key}_{st.session_state[uuid_key]}"
                    edited_data = dynamic_input_data_editor(
                        working_df,
                        column_config=dynamic_columns_config,
                        hide_index=True,
                        use_container_width=True,
                        disabled=disabled_param,
                        key=editor_key
                    )
                    # ── Persist these edits so they survive the toggle's rerun ──
                    st.session_state[f"orig_data_{unique_key}"] = edited_data.copy(deep=True)

                # ---------------------------
                # Map 'RESULTING CRACK' back to 'DIFFERENTIAL - INPUT' for CALL_ID == 4
                if call_id == 4:
                    edited_data = edited_data.copy()
                    edited_data.loc[edited_data["ENTRY"] == "RESULTING CRACK", "ENTRY"] = "DIFFERENTIAL - INPUT"
                # ---------------------------

                # Context
                call_data = call[editor_display_columns].reset_index(drop=True)
                context_df = call_data.sort_values(by="FIELD_ORDER").copy()
                context_df = context_df.loc[
                    ~context_df["FIELD_ORDER"].isin([1, 2, 3, 4, 5, 15, 16]), context_display_columns
                ]
                context_df["SW_MINUS2_AVG"] = pd.NA
                context_df["SW_MINUS1_AVG"] = pd.NA
                context_column_order = [
                    "ENTRY", "HISTORICAL_MARKER", "UNITS", "SW_MINUS2_AVG", "SW_MINUS1_AVG",
                    "SW_PLUS1_AVG", "SW_PLUS2_AVG", "SW_PLUS3_AVG", "SW_PLUS4_AVG", "SM_PLUS1_AVG",
                    "SM_PLUS2_AVG", "SM_PLUS3_AVG", "SM_PLUS4_AVG", "SUMMER_PRICE", "WINTER_PRICE"
                ]
                context_df = context_df[context_column_order]

                context_df.rename(
                    columns={
                        "HISTORICAL_MARKER": "SYMBOL",
                        "SW_MINUS2_AVG": week_minus2_date.strftime(DISPLAY_DATE_FORMAT),
                        "SW_MINUS1_AVG": week_minus1_date.strftime(DISPLAY_DATE_FORMAT),
                        "SW_PLUS1_AVG": '    ' + week1_date.strftime(DISPLAY_DATE_FORMAT) + '   ',
                        "SW_PLUS2_AVG": '    ' + week2_date.strftime(DISPLAY_DATE_FORMAT) + '   ',
                        "SW_PLUS3_AVG": '    ' + week3_date.strftime(DISPLAY_DATE_FORMAT) + '   ',
                        "SW_PLUS4_AVG": '    ' + week4_date.strftime(DISPLAY_DATE_FORMAT) + '   ',
                        "SM_PLUS1_AVG": '    ' + month1_date + '   ',
                        "SM_PLUS2_AVG": '    ' + month2_date + '   ',
                        "SM_PLUS3_AVG": '    ' + month3_date + '   ',
                        "SM_PLUS4_AVG": '    ' + month4_date + '   ',
                        "SUMMER_PRICE": '    ' + "Summer" + '   ',
                        "WINTER_PRICE": '    ' + "Winter" + '   ',
                    },
                    inplace=True,
                )

                st.dataframe(
                    context_df,
                    column_config=dynamic_columns_config,
                    hide_index=True,
                    use_container_width=True
                )

                # Results
                results_df = call_data.sort_values(by="FIELD_ORDER").copy()
                if call_id == 4:
                    results_df.loc[results_df["ENTRY"] == "BASELINE", "ENTRY"] = "CRUDE"
                    results_df = results_df.loc[call_data["FIELD_ORDER"].isin([2, 3]), results_display_columns]
                else:
                    results_df = results_df.loc[call_data["FIELD_ORDER"].isin([2, 3, 5]), results_display_columns]

                results_df.rename(
                    columns={
                        "HISTORICAL_MARKER": "SYMBOL",
                        "SW_MINUS2_AVG": week_minus2_date.strftime(DISPLAY_DATE_FORMAT),
                        "SW_MINUS1_AVG": week_minus1_date.strftime(DISPLAY_DATE_FORMAT),
                        "SW_PLUS1_AVG": '    ' + week1_date.strftime(DISPLAY_DATE_FORMAT) + '   ',
                        "SW_PLUS2_AVG": '    ' + week2_date.strftime(DISPLAY_DATE_FORMAT) + '   ',
                        "SW_PLUS3_AVG": '    ' + week3_date.strftime(DISPLAY_DATE_FORMAT) + '   ',
                        "SW_PLUS4_AVG": '    ' + week4_date.strftime(DISPLAY_DATE_FORMAT) + '   ',
                        "SM_PLUS1_AVG": '    ' + month1_date + '   ',
                        "SM_PLUS2_AVG": '    ' + month2_date + '   ',
                        "SM_PLUS3_AVG": '    ' + month3_date + '   ',
                        "SM_PLUS4_AVG": '    ' + month4_date + '   ',
                        "SUMMER_PRICE": '    ' + "Summer" + '   ',
                        "WINTER_PRICE": '    ' + "Winter" + '   ',
                    },
                    inplace=True,
                )

                st.dataframe(
                    results_df,
                    column_config=dynamic_columns_config,
                    hide_index=True,
                    use_container_width=True
                )

                if (
                        "MARKET_CONDITIONS_COMMENTS" in call.columns
                        and "CALL_COMMENTS" in call.columns
                ):
                    diff_input_rows = call[call["ENTRY"] == "DIFFERENTIAL - INPUT"]
                    if not diff_input_rows.empty:
                        existing_comment_1 = diff_input_rows["MARKET_CONDITIONS_COMMENTS"].iloc[0]
                        existing_comment_2 = diff_input_rows["CALL_COMMENTS"].iloc[0]
                    else:
                        existing_comment_1 = ""
                        existing_comment_2 = ""
                else:
                    existing_comment_1 = ""
                    existing_comment_2 = ""

                comment_col1, comment_col2 = st.columns(2)
                comment_key_1 = f"comment_1_{unique_key}"  # Unique key for the first comment box
                with comment_col1:
                    comment_1 = st.text_area(
                        "Comment 1",
                        key=comment_key_1,
                        value=existing_comment_1,
                        placeholder="Enter your comments on current market conditions here...",
                        label_visibility="hidden"
                    )

                # Second comment box
                comment_key_2 = f"comment_2_{unique_key}"  # Unique key for the second comment box
                with comment_col2:
                    comment_2 = st.text_area(
                        "Comment 2",
                        value=existing_comment_2,
                        key=comment_key_2,
                        placeholder="Enter your comments on the basis for the forward looking call here...",
                        label_visibility="hidden"
                    )

                st.session_state["edited_calls"][unique_key] = {
                            "edited_data": edited_data,
                            "call_df": call.copy()         # store the entire original call DataFrame
                        }

                if i < len(calls) - 1:
                    st.markdown(
                        "<hr style='border: 2px solid #d3d3d3;'>",
                        unsafe_allow_html=True,
                    )

    # … end of our container/loops …
    changed_keys = []
    for unique_key, info in st.session_state["edited_calls"].items():
        # original call DataFrame as it was rendered
        original = info["call_df"]
        # the DataFrame coming out of the data_editor
        edited = info["edited_data"]

        # Reconstruct the "before" grid exactly as you passed to data_editor
        before = (
            original[editor_display_columns]
            .loc[original["FIELD_ORDER"].isin([1, 15, 16])]
            .drop(columns=["FIELD_ORDER", "CALL_ID"])
            .rename(columns={
            "HISTORICAL_MARKER": "SYMBOL",
            "SW_MINUS2_AVG": week_minus2_date.strftime(DISPLAY_DATE_FORMAT),
            "SW_MINUS1_AVG": week_minus1_date.strftime(DISPLAY_DATE_FORMAT),
            "SW_PLUS1_AVG": week1_date.strftime(DISPLAY_DATE_FORMAT),
            "SW_PLUS2_AVG": week2_date.strftime(DISPLAY_DATE_FORMAT),
            "SW_PLUS3_AVG": week3_date.strftime(DISPLAY_DATE_FORMAT),
            "SW_PLUS4_AVG": week4_date.strftime(DISPLAY_DATE_FORMAT),
            "SM_PLUS1_AVG": month1_date,
            "SM_PLUS2_AVG": month2_date,
            "SM_PLUS3_AVG": month3_date,
            "SM_PLUS4_AVG": month4_date,
            "SUMMER_PRICE": "Summer",
            "WINTER_PRICE": "Winter",
            })
            .reset_index(drop=True)
        )

        # # If anything differs, include that key
        # if not before.equals(edited.reset_index(drop=True)):
        #     changed_keys.append(unique_key)
        # only keys with actual value changes get flagged
        if not before.equals(edited.reset_index(drop=True)):
            changed_keys.append(unique_key)

        # # If it's been edited OR explicitly marked "Reviewed", include it exactly once
        # if unique_key in st.session_state.reviewed_keys or not before.equals(edited.reset_index(drop=True)):
        #     changed_keys.append(unique_key)

    # NEW: detect lock-date changes OR comment-only changes
    lock_keys = []
    for unique_key, info in st.session_state["edited_calls"].items():
        original_df = info["call_df"]
        call_id = original_df["CALL_ID"].iloc[0]
        k = f"call_{call_id}"

        # Only consider calls that support locking
        if call_id not in TOGGLE_CALL_IDS:
            continue

        ui_selected = st.session_state.get(f"{k}_lock_until", None)
        ui_comment = (
            st.session_state.get(f"{k}_lock_comment_staged",
                                st.session_state.get(f"{k}_lock_comment", "")) or ""
        ).strip()

        db_end = (
            locked_call_info.get(call_id, (None, None, None))[1] 
            if call_id in locked_call_ids else None
        )
        db_comment = (get_latest_lock_comment(call_id) or "").strip()

        # What changed?
        date_changed = (isinstance(ui_selected, date) and ui_selected != db_end) or (
            ui_selected is None and db_end is not None
        )
        comment_changed = (ui_comment != db_comment)

        if date_changed or comment_changed:
            lock_keys.append(unique_key)

    # ───────────────────────────────────────────────────────
    def reset():
        _snapshot_all_lock_comments()
        st.session_state.processing = True
        st.session_state.freeze_editors = False

    freeze_active = st.session_state.get("freeze_editors", False)
    save_disabled = not (freeze_active)
    save_all = save_button_placeholder.button(
        "Save All", disabled=save_disabled or st.session_state.processing,
        on_click=reset, type="primary"
    )

    if st.session_state.processing:
        keys_to_save = list(dict.fromkeys(list(st.session_state.reviewed_keys) + changed_keys + lock_keys))
        overlay_ph, progress, pct_ph = block_ui(len(keys_to_save))

        # for unique_key, info in st.session_state["edited_calls"].items():
        # for unique_key in changed_keys:
        for i, unique_key in enumerate(keys_to_save):
            try:

                # (1) pull out the edited DataFrame and its original slice
                info = st.session_state["edited_calls"][unique_key]
                edited_df = info["edited_data"]
                original_df = info["call_df"].copy()
                call_id = original_df["CALL_ID"].iloc[0]
                prompt_wk = original_df["PROMPT_WEEK_START_DATE"].iloc[0]
                fresh_call_df = load_single_call(calls_table_name, call_id, prompt_wk)
                # NEW: persist lock state derived from the date_input (for lockable calls)
                # Persist lock state ONLY if this call supports locking and the lock date changed
                k = f"call_{call_id}"
                if call_id in TOGGLE_CALL_IDS and unique_key in lock_keys:
                    persist_lock_row(
                        call_id,
                        prompt_wk,
                        st.session_state.get(f"{k}_lock_until", None),
                        st.session_state.get(
                            f"{k}_lock_comment_staged",
                            st.session_state.get(f"{k}_lock_comment", "")
                        )
                    )

                # full_calls = load_data(calls_table_name)
                # fresh_call_df = full_calls[
                #     (full_calls["CALL_ID"] == call_id) &
                #     (full_calls["PROMPT_WEEK_START_DATE"] == prompt_wk)
                # ]

                # (2) rename UI columns back to Snowflake names
                display_to_original = {
                    week_minus2_date.strftime(DISPLAY_DATE_FORMAT): "SW_MINUS2_AVG",
                    week_minus1_date.strftime(DISPLAY_DATE_FORMAT): "SW_MINUS1_AVG",
                    week1_date.strftime(DISPLAY_DATE_FORMAT): "SW_PLUS1_AVG",
                    week2_date.strftime(DISPLAY_DATE_FORMAT): "SW_PLUS2_AVG",
                    week3_date.strftime(DISPLAY_DATE_FORMAT): "SW_PLUS3_AVG",
                    week4_date.strftime(DISPLAY_DATE_FORMAT): "SW_PLUS4_AVG",
                    month1_date: "SM_PLUS1_AVG",
                    month2_date: "SM_PLUS2_AVG",
                    month3_date: "SM_PLUS3_AVG",
                    month4_date: "SM_PLUS4_AVG",
                    "Summer": "SUMMER_PRICE",
                    "Winter": "WINTER_PRICE",
                }
                save_df = edited_df.rename(columns=display_to_original)
                save_df = save_df.merge(
                    fresh_call_df[["CALL_ID", "ENTRY"]],
                    on="ENTRY", how="left"
                )

                # (3) stitch in metadata & timestamp
                merged = save_df.merge(
                    input_df[[
                        "PROMPT_WEEK_START_DATE", "USER_ID", "CALL_ID", "ENTRY",
                        "CALL_LABEL", "CALL_ORDER", "FIELD_ORDER"
                    ]],
                    on=["CALL_ID", "ENTRY"], how="left"
                )
                merged["PROMPT_WEEK_START_DATE"] = pd.to_datetime(
                    merged["PROMPT_WEEK_START_DATE"]
                ).dt.date
                merged["LAST_UPDATED_TIMESTAMP"] = pd.Timestamp.now()
                merged["IS_LATEST"] = 1

                # (4) enforce your column order
                final_cols = [
                    "PROMPT_WEEK_START_DATE",
                    "USER_ID",
                    "CALL_ID",
                    "CALL_LABEL",
                    "CALL_ORDER",
                    "FIELD_ORDER",
                    "UNITS",
                    "SW_PLUS1_AVG",
                    "SW_PLUS2_AVG",
                    "SW_PLUS3_AVG",
                    "SW_PLUS4_AVG",
                    "SM_PLUS1_AVG",
                    "SM_PLUS2_AVG",
                    "SM_PLUS3_AVG",
                    "SM_PLUS4_AVG",
                    "SUMMER_PRICE",
                    "WINTER_PRICE",
                    "LAST_UPDATED_TIMESTAMP",
                    "IS_LATEST"
                ]
                merged = merged[final_cols]

                # (5) persist each call
                comments_1 = st.session_state.get(f"comment_1_{unique_key}", "")
                comments_2 = st.session_state.get(f"comment_2_{unique_key}", "")
                input_df = load_data(calls_table_name)

                if unique_key in st.session_state.reviewed_keys:
                    update_snowflake_table(input_df, fresh_call_df, merged,
                                        forecast_updates_table,
                                        comments_1, comments_2, copy_to_unchanged=True)

                elif unique_key in changed_keys:
                    update_snowflake_table(
                        input_df, fresh_call_df, merged,
                        forecast_updates_table,
                        comments_1, comments_2
                    )
                    # input_df = load_data(calls_table_name)
                    st.session_state["edited_calls"][unique_key]["call_df"] = fresh_call_df
                    call_cascade_insert_sp(call_id)

                st.session_state["edited_calls"][unique_key]["call_df"] = fresh_call_df
                frac = (i + 1) / len(keys_to_save)
                progress.progress(frac)
                pct_ph.markdown(
                    f"<div class='progress-percentage'>{i+1}/{len(keys_to_save)} call(s) completed</div>", 
                    unsafe_allow_html=True
                )
                # time.sleep(4)

            except Exception as e:
                st.error(f"Error saving {unique_key}: {e}")

        overlay_ph.empty()
        progress.empty()
        pct_ph.empty()

        # spinner_ph.empty()
        # status_ph.success(f"Saved {len(changed_keys)} calls!")

        pop_up(count=len(keys_to_save))

        # finally:
        #     st.session_state.processing = False
        #     st.rerun()

st.markdown(
    """
    <style>
      /* container fixed in bottom-right corner */
      #back-to-top {
        position: fixed;
        bottom: 20px;
        right: 5px;
        z-index: 999;
      }
      /* style the link as a button */
      #back-to-top a {
        display: inline-block;
        padding: 4px 1px;
        font-size: 0.9rem;
        background: #f0f0f0;
        border-radius: 3px;
        text-decoration: none;
        color: inherit;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }
      /* hover effect */
      #back-to-top a:hover {
        background: #e0e0e0;
      }
    </style>

    <div id="back-to-top">
      <a href="#top">🔝 Go to top</a>
    </div>
    """,
    unsafe_allow_html=True
)
