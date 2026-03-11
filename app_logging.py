from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import streamlit as st

from config import (
    DATA_SOURCE,
    SQLITE_DB_PATH,
    SQLITE_AUDIT_LOG_TABLE,
    SQLITE_ERROR_LOG_TABLE,
    SNOWFLAKE_AUDIT_LOG_TABLE,
    SNOWFLAKE_ERROR_LOG_TABLE,
    SNOWFLAKE_WAREHOUSE,
)
from data_loader import get_snowflake_session


DEFAULT_SQLITE_USER_ID = "local_user"


def _best_effort_use_warehouse(session) -> None:
    """Try to set warehouse, but never fail logging if the runtime disallows it."""
    try:
        session.sql(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}").collect()
    except Exception:
        return


def _remember_logging_error(exc: Exception) -> None:
    """Store last logging error for debugging (non-breaking)."""
    try:
        st.session_state["_last_logging_error"] = str(exc)
    except Exception:
        return


def get_session_id() -> str:
    """Return a stable per-Streamlit-session id."""
    if "app_session_id" not in st.session_state:
        st.session_state["app_session_id"] = uuid4().hex
    return str(st.session_state["app_session_id"])


def get_user_id() -> str | None:
    """Return Snowflake user id in Snowflake mode, else a default local id."""
    if DATA_SOURCE == "sqlite":
        return DEFAULT_SQLITE_USER_ID

    try:
        session = get_snowflake_session()
        _best_effort_use_warehouse(session)
        # CURRENT_USER() returns the active Snowflake user
        df = session.sql("SELECT CURRENT_USER() AS USER_ID").to_pandas()
        if df is not None and not df.empty:
            v = df.iloc[0].get("USER_ID")
            return None if v is None else str(v)
    except Exception:
        # Never crash the app due to logging
        return None

    return None


def _json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return json.dumps({"unserializable": True, "repr": repr(obj)})


def ensure_log_tables() -> None:
    """Best-effort create log tables if they don't exist (SQLite always, Snowflake if permitted)."""
    if DATA_SOURCE == "sqlite":
        import sqlite3

        conn = sqlite3.connect(SQLITE_DB_PATH)
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {SQLITE_AUDIT_LOG_TABLE} (
                    AUDIT_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    USER_ID TEXT,
                    TIMESTAMP TEXT NOT NULL DEFAULT (datetime('now')),
                    APP_ID INTEGER,
                    SESSION_ID TEXT,
                    METADATA_JSON TEXT
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {SQLITE_ERROR_LOG_TABLE} (
                    ERROR_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    ERROR_CODE TEXT,
                    ERROR_MESSAGE TEXT,
                    STACK_TRACE TEXT,
                    SERVICE_MODULE TEXT,
                    USER_ID TEXT,
                    SESSION_ID TEXT,
                    TIMESTAMP TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.commit()
        finally:
            conn.close()
        return

    # Snowflake
    try:
        # Avoid repeated DDL attempts in the same session.
        if st.session_state.get("_log_tables_checked"):
            return

        session = get_snowflake_session()
        _best_effort_use_warehouse(session)

        session.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {SNOWFLAKE_AUDIT_LOG_TABLE} (
              AUDIT_ID INTEGER AUTOINCREMENT START 1 INCREMENT 1,
              USER_ID VARCHAR(50),
              TIMESTAMP TIMESTAMP_NTZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
              APP_ID INTEGER,
              SESSION_ID VARCHAR(100),
              METADATA_JSON VARIANT,
              PRIMARY KEY (AUDIT_ID)
            )
            """
        ).collect()

        session.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {SNOWFLAKE_ERROR_LOG_TABLE} (
              ERROR_ID INTEGER AUTOINCREMENT START 1 INCREMENT 1,
              ERROR_CODE VARCHAR(50),
              ERROR_MESSAGE VARCHAR(1000),
              STACK_TRACE VARCHAR(5000),
              SERVICE_MODULE VARCHAR(100),
              USER_ID VARCHAR(50),
              SESSION_ID VARCHAR(100),
              TIMESTAMP TIMESTAMP_NTZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
              PRIMARY KEY (ERROR_ID)
            )
            """
        ).collect()

        st.session_state["_log_tables_checked"] = True
    except Exception:
        # Don't block UI if role doesn't have DDL rights.
        st.session_state["_log_tables_checked"] = True
        return


def log_audit(
    event: str,
    *,
    metadata: dict[str, Any] | None = None,
    app_id: int | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Write an audit log row.

    Notes:
    - `event` is included into METADATA_JSON.
    - `app_id` is optional; we don't have a stable numeric FK in this app yet.
    """
    try:
        ensure_log_tables()

        uid = user_id if user_id is not None else get_user_id()
        sid = session_id if session_id is not None else get_session_id()
        payload = {"event": event, "metadata": (metadata or {})}

        if DATA_SOURCE == "sqlite":
            import sqlite3

            conn = sqlite3.connect(SQLITE_DB_PATH)
            try:
                conn.execute(
                    f"""
                    INSERT INTO {SQLITE_AUDIT_LOG_TABLE} (USER_ID, APP_ID, SESSION_ID, METADATA_JSON)
                    VALUES (?, ?, ?, ?)
                    """,
                    (uid, app_id, sid, _json_dumps_safe(payload)),
                )
                conn.commit()
            finally:
                conn.close()
            return

        session = get_snowflake_session()
        _best_effort_use_warehouse(session)

        meta_sql = _json_dumps_safe(payload).replace("'", "''")
        uid_sql = "NULL" if uid is None else "'" + str(uid).replace("'", "''") + "'"
        sid_sql = "NULL" if sid is None else "'" + str(sid).replace("'", "''") + "'"
        app_sql = "NULL" if app_id is None else str(int(app_id))
        session.sql(
            f"""
            INSERT INTO {SNOWFLAKE_AUDIT_LOG_TABLE} (USER_ID, APP_ID, SESSION_ID, METADATA_JSON)
            SELECT {uid_sql}, {app_sql}, {sid_sql}, PARSE_JSON('{meta_sql}')
            """
        ).collect()
    except Exception:
        # Logging must never break the app
        _remember_logging_error(Exception(traceback.format_exc()))
        return


def log_error(
    *,
    error_code: str,
    error_message: str,
    stack_trace: str | None = None,
    service_module: str = "UI",
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Write an error log row (best-effort)."""
    try:
        ensure_log_tables()

        uid = user_id if user_id is not None else get_user_id()
        sid = session_id if session_id is not None else get_session_id()
        msg = (error_message or "")[:1000]
        stx = (stack_trace or "")[:5000]

        if DATA_SOURCE == "sqlite":
            import sqlite3

            conn = sqlite3.connect(SQLITE_DB_PATH)
            try:
                conn.execute(
                    f"""
                    INSERT INTO {SQLITE_ERROR_LOG_TABLE} (
                        ERROR_CODE, ERROR_MESSAGE, STACK_TRACE, SERVICE_MODULE, USER_ID, SESSION_ID
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (str(error_code), msg, stx, str(service_module), uid, sid),
                )
                conn.commit()
            finally:
                conn.close()
            return

        session = get_snowflake_session()
        _best_effort_use_warehouse(session)

        def _sql_str(v: str | None) -> str:
            if v is None:
                return "NULL"
            return "'" + str(v).replace("'", "''") + "'"

        session.sql(
            f"""
            INSERT INTO {SNOWFLAKE_ERROR_LOG_TABLE} (
                ERROR_CODE, ERROR_MESSAGE, STACK_TRACE, SERVICE_MODULE, USER_ID, SESSION_ID
            ) VALUES (
                {_sql_str(error_code)},
                {_sql_str(msg)},
                {_sql_str(stx)},
                {_sql_str(service_module)},
                {_sql_str(uid)},
                {_sql_str(sid)}
            )
            """
        ).collect()
    except Exception:
        _remember_logging_error(Exception(traceback.format_exc()))
        return


@dataclass(frozen=True)
class LoggedButtonResult:
    clicked: bool


def logged_button(
    label: str,
    *,
    key: str | None = None,
    event: str | None = None,
    metadata: dict[str, Any] | None = None,
    app_id: int | None = None,
    service_module: str = "UI",
    **button_kwargs,
) -> bool:
    """Drop-in replacement for st.button that logs every click."""
    clicked = st.button(label, key=key, **button_kwargs)
    if clicked:
        log_audit(
            event=(event or f"button_click:{label}"),
            metadata={
                "label": label,
                "key": key,
                **(metadata or {}),
            },
            app_id=app_id,
        )
    return clicked


def logged_callback(
    fn,
    *,
    event: str,
    metadata: dict[str, Any] | None = None,
    app_id: int | None = None,
    service_module: str = "UI",
):
    """Wrap an on_click callback so it logs audit + errors."""

    def _wrapped(*args, **kwargs):
        log_audit(event=event, metadata=metadata or {}, app_id=app_id)
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            log_error(
                error_code="UI_CALLBACK_ERROR",
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                service_module=service_module,
            )
            raise

    return _wrapped


def log_exception(*, error_code: str, exc: BaseException, service_module: str = "UI") -> None:
    """Convenience helper to log an exception with traceback."""
    log_error(
        error_code=error_code,
        error_message=str(exc),
        stack_trace=traceback.format_exc(),
        service_module=service_module,
    )
