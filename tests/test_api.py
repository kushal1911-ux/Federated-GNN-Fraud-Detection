"""
RFGN API Automated Tests

Tests the running FastAPI service through HTTP.

Coverage:
    1. Health endpoint
    2. System status endpoint
    3. Valid prediction
    4. Negative amount validation
    5. isFraud injection protection
    6. Unknown field protection
    7. Missing required field validation
    8. NaN validation
    9. Infinity validation
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "http://127.0.0.1:8000"

REQUEST_TIMEOUT = 60


# ============================================================
# HTTP REQUEST HELPER
# ============================================================

def request_api(
    method: str,
    path: str,
    payload: dict | None = None,
):
    """
    Send an HTTP request to the running RFGN API.

    Returns:
        tuple:
            status_code
            response_body
    """

    url = f"{BASE_URL}{path}"

    headers = {
        "Content-Type": "application/json",
    }

    data = None

    if payload is not None:

        data = json.dumps(
            payload
        ).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        data=data,
        headers=headers,
        method=method,
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT,
        ) as response:

            body = response.read().decode(
                "utf-8"
            )

            try:

                parsed = json.loads(
                    body
                )

            except json.JSONDecodeError:

                parsed = body

            return response.status, parsed

    except urllib.error.HTTPError as exc:

        body = exc.read().decode(
            "utf-8"
        )

        try:

            parsed = json.loads(
                body
            )

        except json.JSONDecodeError:

            parsed = body

        return exc.code, parsed

    except urllib.error.URLError as exc:

        print()
        print(
            "ERROR: Could not connect to "
            "the RFGN API."
        )

        print(
            f"Reason: {exc.reason}"
        )

        print()
        print(
            "Make sure Uvicorn is running:"
        )

        print(
            "uvicorn api.main:app --reload"
        )

        print()

        sys.exit(1)

    except TimeoutError:

        print()
        print(
            "ERROR: RFGN API request timed out."
        )

        print(
            f"Timeout: {REQUEST_TIMEOUT} seconds"
        )

        print()

        sys.exit(1)


# ============================================================
# TEST HELPER
# ============================================================

def run_test(
    name: str,
    expected_status: int,
    method: str,
    path: str,
    payload: dict | None = None,
) -> bool:
    """
    Execute one API test and report the result.
    """

    status_code, response = request_api(
        method=method,
        path=path,
        payload=payload,
    )

    passed = (
        status_code == expected_status
    )

    if passed:

        print(
            f"{name:<45} : PASS"
        )

    else:

        print(
            f"{name:<45} : FAIL"
        )

        print(
            f"  Expected HTTP {expected_status}"
        )

        print(
            f"  Received HTTP {status_code}"
        )

        print(
            f"  Response: {response}"
        )

    return passed


# ============================================================
# VALID TRANSACTION
# ============================================================

VALID_TRANSACTION = {

    "TransactionDT": 1500000,

    "TransactionAmt": 149.50,

    "ProductCD": "W",

    "card1": 13926,

    "card2": 555,

    "card3": 150,

    "card5": 226,

    "addr1": 315,

    "addr2": 87,

    "P_emaildomain": "gmail.com",

    "R_emaildomain": "gmail.com",

    "DeviceInfo": "Windows",
}


# ============================================================
# TEST SUITE
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RFGN API AUTOMATED TESTS")
    print("=" * 70)
    print()

    results = []


    # ========================================================
    # TEST 1 — HEALTH
    # ========================================================

    results.append(
        run_test(
            name="GET /health",
            expected_status=200,
            method="GET",
            path="/health",
        )
    )


    # ========================================================
    # TEST 2 — SYSTEM STATUS
    # ========================================================

    results.append(
        run_test(
            name="GET /system/status",
            expected_status=200,
            method="GET",
            path="/system/status",
        )
    )


    # ========================================================
    # TEST 3 — VALID PREDICTION
    # ========================================================

    results.append(
        run_test(
            name="POST /predict - valid transaction",
            expected_status=200,
            method="POST",
            path="/predict",
            payload=VALID_TRANSACTION,
        )
    )


    # ========================================================
    # TEST 4 — NEGATIVE AMOUNT
    # ========================================================

    negative_amount = (
        VALID_TRANSACTION.copy()
    )

    negative_amount[
        "TransactionAmt"
    ] = -100

    results.append(
        run_test(
            name="POST /predict - negative amount",
            expected_status=422,
            method="POST",
            path="/predict",
            payload=negative_amount,
        )
    )


    # ========================================================
    # TEST 5 — FRAUD LABEL INJECTION
    # ========================================================

    fraud_label = (
        VALID_TRANSACTION.copy()
    )

    fraud_label[
        "isFraud"
    ] = 1

    results.append(
        run_test(
            name="POST /predict - isFraud injection",
            expected_status=422,
            method="POST",
            path="/predict",
            payload=fraud_label,
        )
    )


    # ========================================================
    # TEST 6 — UNKNOWN FIELD
    # ========================================================

    unknown_field = (
        VALID_TRANSACTION.copy()
    )

    unknown_field[
        "RandomField"
    ] = "test"

    results.append(
        run_test(
            name="POST /predict - unknown field",
            expected_status=422,
            method="POST",
            path="/predict",
            payload=unknown_field,
        )
    )


    # ========================================================
    # TEST 7 — MISSING REQUIRED FIELD
    # ========================================================

    missing_amount = (
        VALID_TRANSACTION.copy()
    )

    del missing_amount[
        "TransactionAmt"
    ]

    results.append(
        run_test(
            name="POST /predict - missing amount",
            expected_status=422,
            method="POST",
            path="/predict",
            payload=missing_amount,
        )
    )


    # ========================================================
    # TEST 8 — NaN
    # ========================================================
    #
    # IMPORTANT:
    #
    # JSON does not officially support NaN.
    #
    # Therefore we send "NaN" as a string.
    # Pydantic should reject it because TransactionAmt
    # is defined as a float.
    #
    # ========================================================

    nan_transaction = (
        VALID_TRANSACTION.copy()
    )

    nan_transaction[
        "TransactionAmt"
    ] = "NaN"

    results.append(
        run_test(
            name="POST /predict - NaN amount",
            expected_status=422,
            method="POST",
            path="/predict",
            payload=nan_transaction,
        )
    )


    # ========================================================
    # TEST 9 — INFINITY
    # ========================================================
    #
    # JSON does not officially support Infinity.
    #
    # Therefore we send "Infinity" as a string.
    # Pydantic should reject it because TransactionAmt
    # is defined as a float.
    #
    # ========================================================

    infinity_transaction = (
        VALID_TRANSACTION.copy()
    )

    infinity_transaction[
        "TransactionAmt"
    ] = "Infinity"

    results.append(
        run_test(
            name="POST /predict - Infinity amount",
            expected_status=422,
            method="POST",
            path="/predict",
            payload=infinity_transaction,
        )
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    passed = sum(
        1
        for result in results
        if result
    )

    total = len(results)


    print()
    print("=" * 70)


    if passed == total:

        print(
            "RFGN API TEST SUITE: PASS"
        )

    else:

        print(
            "RFGN API TEST SUITE: FAIL"
        )


    print(
        f"Tests passed: {passed}/{total}"
    )

    print("=" * 70)
    print()


    # ========================================================
    # EXIT CODE
    # ========================================================

    if passed != total:

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()