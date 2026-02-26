"""
Unit tests for compute_dte_dse_features() in fetchBulkData.py.

Run with:
    python -m pytest test_dte_dse.py -v
or:
    python test_dte_dse.py
"""

import pandas as pd
import numpy as np
import pytest
from fetchBulkData import compute_dte_dse_features


def make_trading_dates(start, end):
    """Generate business days between start and end inclusive."""
    return pd.bdate_range(start=start, end=end)


def make_df(trading_dates):
    """Minimal DataFrame with a 'date' column (string) sorted ascending."""
    return pd.DataFrame({"date": trading_dates.strftime("%Y-%m-%d")})


# ── helpers ───────────────────────────────────────────────────────────────────

def dte_on(result_df, date_str):
    return result_df.loc[result_df["date"] == date_str, "dte"].iloc[0]

def dse_on(result_df, date_str):
    return result_df.loc[result_df["date"] == date_str, "dse"].iloc[0]


# ── Test 1: basic DTE and DSE ─────────────────────────────────────────────────

def test_basic_dte_dse():
    """
    Trading dates: 2024-01-02 to 2024-03-29
    Past earnings effective sessions: 2024-01-25, 2024-04-25 (quarterly)

    On 2024-01-22 (3 trading days before Jan 25): dte = 3
    On 2024-01-25 (earnings day):                 dte = 0
    On 2024-01-26 (day after earnings):           dte points to Apr 25
    On 2024-01-26:                                dse = 1
    On 2024-01-25:                                dse = 0
    """
    td = make_trading_dates("2024-01-02", "2024-03-29")
    df = make_df(td)

    eff = [pd.Timestamp("2024-01-25"), pd.Timestamp("2024-04-25")]
    result = compute_dte_dse_features(df, eff, "TEST")

    # DTE on earnings day itself = 0
    assert dte_on(result, "2024-01-25") == 0, "dte should be 0 on earnings day"

    # DTE three trading days before: Jan 22, 23, 24, 25 → distance = 3
    assert dte_on(result, "2024-01-22") == 3, "dte should be 3 on Jan 22"

    # DSE on earnings day = 0
    assert dse_on(result, "2024-01-25") == 0, "dse should be 0 on earnings day"

    # DSE one trading day after = 1
    assert dse_on(result, "2024-01-26") == 1, "dse should be 1 the day after earnings"


# ── Test 2: earn_in gate flags ────────────────────────────────────────────────

def test_earn_in_gates():
    """
    earn_in_5  = 1 iff dte in [0, 5]
    earn_in_10 = 1 iff dte in [0, 10]
    earn_in_20 = 1 iff dte in [0, 20]
    """
    td = make_trading_dates("2024-01-02", "2024-03-29")
    df = make_df(td)
    eff = [pd.Timestamp("2024-01-25")]
    result = compute_dte_dse_features(df, eff, "TEST")

    # 3 days before earnings → in all three windows
    row_3 = result[result["date"] == "2024-01-22"].iloc[0]
    assert row_3["earn_in_5"]  == 1
    assert row_3["earn_in_10"] == 1
    assert row_3["earn_in_20"] == 1

    # 7 trading days before (approx Jan 16): dte=7 → out of earn_in_5, in 10 and 20
    row_7 = result[result["date"] == "2024-01-16"].iloc[0]
    assert row_7["dte"] == 7, f"expected dte=7, got {row_7['dte']}"
    assert row_7["earn_in_5"]  == 0
    assert row_7["earn_in_10"] == 1
    assert row_7["earn_in_20"] == 1

    # 15 trading days before (approx Jan 4): dte=15 → out of 5 and 10, in 20
    row_15 = result[result["date"] == "2024-01-04"].iloc[0]
    assert row_15["dte"] == 15, f"expected dte=15, got {row_15['dte']}"
    assert row_15["earn_in_5"]  == 0
    assert row_15["earn_in_10"] == 0
    assert row_15["earn_in_20"] == 1


# ── Test 3: sentinel 999 only when eff_sessions is empty ──────────────────────

def test_sentinel_empty_eff_sessions(capsys):
    """
    dte=999 should only appear when eff_sessions is completely empty (no historical
    earnings data to extrapolate from).  A non-empty eff_sessions — even all past
    dates — should extrapolate via +3-month increments, not return 999.
    """
    td = make_trading_dates("2024-03-01", "2024-03-29")
    df = make_df(td)

    # Empty eff_sessions → no way to extrapolate → 999
    result_empty = compute_dte_dse_features(df, [], "XTEST")
    assert (result_empty["dte"] == 999).all(), \
        "All rows should have dte=999 when eff_sessions is empty"

    captured = capsys.readouterr()
    assert "XTEST" in captured.out and "dte=999" in captured.out, \
        "Warning message should mention symbol and dte=999"


# ── Test 3b: past-only eff_sessions extrapolates via +3 months ───────────────

def test_past_only_eff_sessions_extrapolates():
    """
    When all effective sessions are in the past, dte should be extrapolated as
    last_known + 3-month increments (not 999).
    """
    td = make_trading_dates("2024-03-01", "2024-03-29")
    df = make_df(td)

    # Only a past earnings date — all trading dates are after it
    eff = [pd.Timestamp("2024-01-25")]
    result = compute_dte_dse_features(df, eff, "XTEST")

    # Should NOT be 999 — extrapolated next = 2024-01-25 + 3 months = 2024-04-25
    assert (result["dte"] != 999).all(), \
        "dte should not be 999 when eff_sessions has at least one historical date"
    # Should be positive and reasonable (< 100 trading days / ~5 months out)
    assert (result["dte"] > 0).all(), "dte should be positive"
    assert (result["dte"] < 100).all(), "dte should be a reasonable approximation"


# ── Test 4: earnings beyond price history uses calendar approximation ─────────

def test_future_earnings_beyond_price_history():
    """
    When next_report_date is beyond the last trading day in df, dte should be
    a positive approximation (not 999, not 0).
    """
    td = make_trading_dates("2024-01-02", "2024-01-31")
    df = make_df(td)

    # Earnings 30 calendar days after last trading day
    future_date = pd.Timestamp("2024-03-01")   # well beyond Jan 31
    eff = [pd.Timestamp("2024-01-10"), future_date]   # one past, one future
    result = compute_dte_dse_features(df, eff, "TEST")

    last_dte = result["dte"].iloc[-1]
    assert last_dte != 999,  "dte should not be 999 when next_report_date is provided"
    assert last_dte > 0,     "dte should be positive when earnings are in the future"
    assert last_dte < 999,   "dte should be a reasonable approximation"


# ── Test 5: DSE is NaN before first known earnings ────────────────────────────

def test_dse_nan_before_first_earnings():
    """
    Rows before the first effective session should have dse = NaN.
    """
    td = make_trading_dates("2024-01-02", "2024-03-29")
    df = make_df(td)
    eff = [pd.Timestamp("2024-02-01")]
    result = compute_dte_dse_features(df, eff, "TEST")

    before = result[result["date"] < "2024-02-01"]
    assert before["dse"].isna().all(), "dse should be NaN before first earnings"

    after = result[result["date"] >= "2024-02-01"]
    assert after["dse"].notna().all(), "dse should not be NaN on or after first earnings"


# ── Test 6: columns added and original df not mutated ────────────────────────

def test_columns_added_and_no_mutation():
    td = make_trading_dates("2024-01-02", "2024-03-29")
    df = make_df(td)
    original_cols = set(df.columns)

    eff = [pd.Timestamp("2024-02-15")]
    result = compute_dte_dse_features(df, eff, "TEST")

    expected_new = {"dte", "dse", "earn_in_5", "earn_in_10", "earn_in_20"}
    assert expected_new.issubset(set(result.columns)), "All 5 new columns should be present"
    assert set(df.columns) == original_cols, "Original df should not be mutated"


# ── Test 7: auto-estimate next report as last known + 3 months ───────────────

def test_auto_estimate_next_report_date():
    """
    When next_report_date is not in param, fetch_all_data estimates it as
    last known effective session + 3 months.  Simulate that here and confirm:
      - dte is positive (not 999) for rows after the last known earnings
      - dte is a reasonable approximation (< 100 trading days / ~5 months)
      - the sentinel 999 never fires
    """
    td = make_trading_dates("2024-01-02", "2024-03-29")
    df = make_df(td)

    # Last known earnings was 2024-01-25; no next_report_date in param
    last_known = pd.Timestamp("2024-01-25")
    estimated_next = last_known + pd.DateOffset(months=3)   # ≈ 2024-04-25

    eff = [last_known, estimated_next]
    result = compute_dte_dse_features(df, eff, "TEST")

    # No sentinel 999 should appear
    assert (result["dte"] != 999).all(), "dte=999 should not appear with auto-estimated date"

    # After the last known earnings, dte should be positive and < 100
    after = result[result["date"] > "2024-01-25"]
    assert (after["dte"] > 0).all(),   "dte should be positive after last earnings"
    assert (after["dte"] < 100).all(), "dte should be a reasonable approximation (< 100)"


# ── run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
