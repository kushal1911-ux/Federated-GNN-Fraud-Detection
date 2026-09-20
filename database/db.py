"""
RFGN DATABASE CONNECTION MODULE

Provides the central DuckDB connection used by the
real-time fraud detection pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import duckdb


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# ============================================================
# DATABASE PATH
# ============================================================

DATABASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "database"
)

DATABASE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATABASE_PATH = (
    DATABASE_DIR
    / "rfgndb.duckdb"
)


# ============================================================
# CONNECTION
# ============================================================

_connection: Optional[
    duckdb.DuckDBPyConnection
] = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Return the shared DuckDB connection.

    A single connection is reused by the application.
    """

    global _connection

    if _connection is None:

        _connection = duckdb.connect(
            str(DATABASE_PATH)
        )

    return _connection


# ============================================================
# CLOSE CONNECTION
# ============================================================

def close_connection() -> None:
    """
    Close the active DuckDB connection.
    """

    global _connection

    if _connection is not None:

        _connection.close()

        _connection = None


# ============================================================
# EXECUTE SQL
# ============================================================

def execute(
    sql: str,
    parameters=None,
):
    """
    Execute SQL using the shared connection.
    """

    connection = get_connection()

    if parameters is None:

        return connection.execute(sql)

    return connection.execute(
        sql,
        parameters
    )


# ============================================================
# DATABASE HEALTH CHECK
# ============================================================

def health_check() -> dict:

    connection = get_connection()

    result = connection.execute(
        "SELECT 1"
    ).fetchone()

    if result != (1,):

        raise RuntimeError(
            "DuckDB health check failed."
        )

    return {
        "database": "DuckDB",
        "version": duckdb.__version__,
        "path": str(DATABASE_PATH),
        "status": "healthy",
    }


# ============================================================
# SELF TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 70)
    print("RFGN DATABASE CONNECTION TEST")
    print("=" * 70)

    connection = get_connection()

    result = connection.execute(
        "SELECT 1"
    ).fetchone()

    if result != (1,):

        raise RuntimeError(
            "Database connection test failed."
        )

    print()
    print(
        "DuckDB version :",
        duckdb.__version__
    )

    print(
        "Database path  :",
        DATABASE_PATH
    )

    print(
        "Connection     : PASS"
    )

    print(
        "SQL execution  : PASS"
    )

    health = health_check()

    print(
        "Health check   :",
        health["status"].upper()
    )

    print()
    print(
        "Database connection module: PASS"
    )

    print("=" * 70)

    close_connection()

    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_self_test()