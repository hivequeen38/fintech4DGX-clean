"""
Unit tests for fetch_next_report_date() in fetchBulkData.py.

Covers:
  - AV CSV parsing (mocked — no network)
  - Live AV fetch for NVDA (verifies 2026-02-25)
  - AMC shift: effective session is the next trading day after reportDate
  - Graceful fallback when AV returns empty CSV

Run with:
    python -m pytest test_fetch_earnings.py -v
or:
    python test_fetch_earnings.py
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from fetchBulkData import fetch_next_report_date, compute_dte_dse_features
from trendConfig import config


# ── helpers ───────────────────────────────────────────────────────────────────

def make_trading_dates(start, end):
    return pd.bdate_range(start=start, end=end)

def make_df(trading_dates):
    return pd.DataFrame({"date": trading_dates.strftime("%Y-%m-%d")})

def dte_on(result_df, date_str):
    return result_df.loc[result_df["date"] == date_str, "dte"].iloc[0]


# ── Test 8: AV CSV parsing (mocked) ──────────────────────────────────────────

def test_av_csv_parsing_mocked():
    """
    Verify fetch_next_report_date correctly parses the AV EARNINGS_CALENDAR
    CSV response without making a real network call.
    """
    fake_csv = (
        "symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\n"
        "NVDA,NVIDIA CORP,2026-02-25,2026-01-31,1.45,USD,post-market\n"
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = fake_csv.encode()
    mock_resp.text = fake_csv

    with patch("fetchBulkData.requests.get", return_value=mock_resp):
        nrd = fetch_next_report_date("NVDA", "FAKE_KEY")

    assert nrd is not None, "Should parse a date from the mock CSV"
    assert nrd == pd.Timestamp("2026-02-25"), f"Expected 2026-02-25, got {nrd}"


# ── Test 9: AV live fetch for NVDA ───────────────────────────────────────────

def test_fetch_next_report_date_live_nvda():
    """
    Live call to Alpha Vantage EARNINGS_CALENDAR for NVDA.
    Expected: 2026-02-25 (Q4 FY2026, post-market).
    Update this assertion after earnings pass.
    """
    api_key = config["alpha_vantage"]["key"]
    nrd = fetch_next_report_date("NVDA", api_key)

    assert nrd is not None, "AV should return a date for NVDA"
    assert isinstance(nrd, pd.Timestamp)
    assert nrd == pd.Timestamp("2026-02-25"), (
        f"Expected NVDA next earnings = 2026-02-25 (Q4 FY2026), got {nrd.date()}"
    )


# ── Test 10: AMC shift — effective session is the day AFTER the report ────────

def test_amc_shift_effective_session():
    """
    AMC assumption: earnings after market close means dte=0 falls on the
    NEXT trading day, not the report date itself.

    NVDA reports 2026-02-25 AMC → AV returns reportDate = 2026-02-25.
    The shift logic in fetch_all_data:
        candidates = df_dates[df_dates > nrd]   # first date after Feb 25
        eff_sessions.append(candidates.min())   # → 2026-02-26

    So dte=0 on 2026-02-26, and dte=1 on 2026-02-25 (the report day).
    """
    td = make_trading_dates("2026-02-02", "2026-02-28")
    df = make_df(td)

    nrd = pd.Timestamp("2026-02-25")          # raw AV report date
    df_dates = pd.to_datetime(df["date"])

    # Replicate the AMC shift from fetch_all_data
    candidates = df_dates[df_dates > nrd]
    assert not candidates.empty, "df should contain trading days after Feb 25"
    effective_session = candidates.min()

    assert str(effective_session.date()) == "2026-02-26", (
        f"Effective session should be 2026-02-26 (next trading day after AMC), "
        f"got {effective_session.date()}"
    )

    # Verify dte values using the shifted effective session
    result = compute_dte_dse_features(df, [effective_session], "NVDA")

    assert dte_on(result, "2026-02-26") == 0, "dte=0 on the effective session (Feb 26)"
    assert dte_on(result, "2026-02-25") == 1, "dte=1 on the report day itself (AMC → next day)"
    assert dte_on(result, "2026-02-24") == 2, "dte=2 two trading days before the effective session"


# ── Test 11: AV returns empty CSV — falls through gracefully ──────────────────

def test_av_empty_response_falls_through():
    """
    When AV returns a header-only CSV (no rows), and yfinance also fails,
    fetch_next_report_date should return None without raising.
    """
    fake_csv = "symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\n"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = fake_csv.encode()
    mock_resp.text = fake_csv

    mock_ticker = MagicMock()
    mock_ticker.calendar = None

    with patch("fetchBulkData.requests.get", return_value=mock_resp), \
         patch("yfinance.Ticker", return_value=mock_ticker):
        result = fetch_next_report_date("XTEST", "FAKE_KEY")

    assert result is None, "Should return None when both AV and yfinance yield nothing"


# ── run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
