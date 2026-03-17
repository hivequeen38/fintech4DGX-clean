#!/usr/bin/env python3
"""
cron_market_check.py — market-aware nightly pipeline actions

Called by market_scheduler.py (daemon) for each timed action.
Can also be invoked manually for debugging:

    python3 cron_market_check.py --action=morning
    python3 cron_market_check.py --action=retry
    python3 cron_market_check.py --action=fallback
    python3 cron_market_check.py --action=confirm
    python3 cron_market_check.py --action=trigger [--date YYYY-MM-DD]

Flag file: /workspace/market_open_today.flag
    Values: "OPEN YYYY-MM-DD" | "CLOSED YYYY-MM-DD" | "SKIP YYYY-MM-DD"
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import pytz
import requests

# ── Constants ────────────────────────────────────────────────────────────────
FLAG_PATH   = "/workspace/market_open_today.flag"
NIGHTLY_PY  = "/workspace/nightly_run.py"
PYTHON_BIN  = "/usr/bin/python3"
LOG_PREFIX  = "[market_check]"
eastern     = pytz.timezone("US/Eastern")
AV_TIMEOUT  = 15  # seconds


# ── AV key ───────────────────────────────────────────────────────────────────
def _get_av_key() -> str:
    from trendConfig import config as cfg
    return cfg["alpha_vantage"]["key"]


# ── Market status checks ──────────────────────────────────────────────────────
def check_av_market_status(api_key: str) -> str:
    """
    Query AV MARKET_STATUS endpoint.
    Returns 'open', 'closed', or 'error'.
    """
    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "MARKET_STATUS", "apikey": api_key},
            timeout=AV_TIMEOUT,
        )
        resp.raise_for_status()
        for m in resp.json().get("markets", []):
            if (m.get("region") == "United States"
                    and m.get("market_type") == "Equity"):
                return m.get("current_status", "error").lower()
    except Exception as e:
        print(f"{LOG_PREFIX} AV MARKET_STATUS error: {e}")
    return "error"


def check_spy_latest_date(api_key: str, today_str: str) -> bool:
    """
    Belt-and-suspenders: query AV GLOBAL_QUOTE for SPY.
    Returns True if the latest trading day reported == today_str.
    Reliable on holidays because AV returns the *prior* trading day's date.
    """
    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": "SPY", "apikey": api_key},
            timeout=AV_TIMEOUT,
        )
        resp.raise_for_status()
        latest = resp.json().get("Global Quote", {}).get("07. latest trading day", "")
        match = (latest == today_str)
        print(f"{LOG_PREFIX} SPY latest_trading_day={latest!r}, today={today_str!r} → {'MATCH' if match else 'NO MATCH'}")
        return match
    except Exception as e:
        print(f"{LOG_PREFIX} AV GLOBAL_QUOTE/SPY error: {e}")
        return False


def is_market_open(api_key: str, today_str: str) -> bool:
    """
    Primary check: AV MARKET_STATUS == 'open'.

    At 10 AM ET on a trading day AV correctly returns 'open'.
    On holidays/weekends it returns 'closed'.

    NOTE: The SPY GLOBAL_QUOTE latest_trading_day check is NOT used here
    because AV does not update the daily quote to today's date until after
    market close (~5-6 PM ET). Requiring that check at 10 AM would always
    see yesterday's date and incorrectly flag every trading day as closed.
    The SPY check is retained only as a last-resort fallback when AV
    MARKET_STATUS itself is unreachable (network/API error).
    """
    av_status = check_av_market_status(api_key)
    print(f"{LOG_PREFIX} AV_MARKET_STATUS={av_status!r}")

    if av_status == "open":
        return True
    if av_status == "closed":
        return False

    # av_status == "error" — AV completely unreachable
    # Fall back to SPY date check with a caveat: this is only reliable
    # after market close, so a False result here at 10 AM is ambiguous.
    print(f"{LOG_PREFIX} WARNING: AV MARKET_STATUS unavailable — falling back to "
          f"SPY date check. Result unreliable before market close.")
    spy_today = check_spy_latest_date(api_key, today_str)
    print(f"{LOG_PREFIX} SPY_date_match={spy_today}")
    return spy_today


# ── Flag file helpers ─────────────────────────────────────────────────────────
def write_flag(value: str, today_str: str) -> None:
    with open(FLAG_PATH, "w") as f:
        f.write(f"{value} {today_str}\n")
    print(f"{LOG_PREFIX} Flag written: {value} {today_str}")


def read_flag(today_str: str) -> str:
    """Returns 'OPEN', 'CLOSED', 'SKIP', or 'MISSING'."""
    try:
        parts = open(FLAG_PATH).read().strip().split()
        if len(parts) == 2 and parts[1] == today_str:
            return parts[0]
        return "MISSING"   # stale from a prior day
    except FileNotFoundError:
        return "MISSING"


# ── Actions ───────────────────────────────────────────────────────────────────
def action_morning(today_str: str, api_key: str) -> None:
    """10:00 AM ET — initial market-open check."""
    print(f"{LOG_PREFIX} [morning] checking market status for {today_str}")
    if is_market_open(api_key, today_str):
        write_flag("OPEN", today_str)
        print(f"{LOG_PREFIX} [morning] Market OPEN — nightly run scheduled for 6:30 PM.")
    else:
        # Write CLOSED tentatively; retry at 10:30 handles delayed opens
        write_flag("CLOSED", today_str)
        print(f"{LOG_PREFIX} [morning] Market appears CLOSED — "
              f"retry at 10:30 AM to confirm (handles delayed opens).")


def action_retry(today_str: str, api_key: str) -> None:
    """
    10:30 AM ET — retry if morning check wrote CLOSED.
    Handles delayed-open scenarios (rare technical issues at NYSE/NASDAQ).
    If flag is already OPEN, skip — nothing to do.
    """
    current = read_flag(today_str)
    if current == "OPEN":
        print(f"{LOG_PREFIX} [retry] Flag already OPEN — skipping retry.")
        return
    print(f"{LOG_PREFIX} [retry] Re-checking market status for {today_str}")
    if is_market_open(api_key, today_str):
        write_flag("OPEN", today_str)
        print(f"{LOG_PREFIX} [retry] Market now OPEN (delayed open confirmed) — "
              f"nightly run scheduled.")
    else:
        # Remains CLOSED — legitimate holiday or closure
        write_flag("CLOSED", today_str)
        print(f"{LOG_PREFIX} [retry] Market still CLOSED at 10:30 AM — "
              f"treating as non-trading day.")


def action_fallback(today_str: str, api_key: str) -> None:
    """
    11:00 AM ET — last-resort fallback if morning check AND retry both failed
    (e.g. container was down from 10-10:30 AM, flag is MISSING).
    If flag is already set (any value), skip.
    """
    current = read_flag(today_str)
    if current != "MISSING":
        print(f"{LOG_PREFIX} [fallback] Flag already set ({current}) — skipping.")
        return
    print(f"{LOG_PREFIX} [fallback] Flag MISSING — running emergency market check.")
    if is_market_open(api_key, today_str):
        write_flag("OPEN", today_str)
        print(f"{LOG_PREFIX} [fallback] Market OPEN — nightly run will proceed at 6:30 PM.")
    else:
        write_flag("CLOSED", today_str)
        print(f"{LOG_PREFIX} [fallback] Market CLOSED — nightly run will be skipped.")


def action_confirm(today_str: str, api_key: str) -> None:
    """
    4:30 PM ET — confirm market has closed.
    Only relevant if flag is currently OPEN. Covers early-close days.
    If market is still showing open at 4:30 PM (unexpected), write SKIP.
    """
    current = read_flag(today_str)
    if current != "OPEN":
        print(f"{LOG_PREFIX} [confirm] Flag is {current!r} — nothing to confirm.")
        return
    av_status = check_av_market_status(api_key)
    if av_status == "open":
        # Should not happen — either emergency closure / extended hours bleed
        print(f"{LOG_PREFIX} [confirm] WARNING: AV still shows market OPEN at 4:30 PM. "
              f"Writing SKIP to prevent inference against live/stale data.")
        write_flag("SKIP", today_str)
    else:
        print(f"{LOG_PREFIX} [confirm] Market confirmed CLOSED. "
              f"Nightly run still on for 6:30 PM.")


def action_trigger(today_str: str) -> None:
    """
    6:30 PM ET — launch nightly_run.py if flag is OPEN.
    """
    flag = read_flag(today_str)
    print(f"{LOG_PREFIX} [trigger] Flag={flag!r} for {today_str}")

    if flag == "MISSING":
        print(f"{LOG_PREFIX} [trigger] ERROR: Flag is MISSING — all three checks "
              f"(10 AM, 10:30 AM, 11 AM) failed. Skipping nightly run. "
              f"Investigate network/AV availability.")
        return

    if flag != "OPEN":
        print(f"{LOG_PREFIX} [trigger] Nightly run SKIPPED (flag={flag!r}).")
        return

    log_path = (f"/workspace/nightly_"
                f"{datetime.now(eastern).strftime('%Y%m%dT%H%M')}.log")
    print(f"{LOG_PREFIX} [trigger] Launching nightly_run.py "
          f"--date {today_str} → {log_path}")
    subprocess.Popen(
        [PYTHON_BIN, NIGHTLY_PY, "--date", today_str, "--log-path", log_path],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    print(f"{LOG_PREFIX} [trigger] nightly_run.py launched (background).")


# ── CLI entry point (manual invocation / debugging) ───────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market-aware pipeline scheduler actions")
    parser.add_argument(
        "--action",
        choices=["morning", "retry", "fallback", "confirm", "trigger"],
        required=True,
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override today's date (YYYY-MM-DD); defaults to current ET date",
    )
    args = parser.parse_args()

    today_str = args.date or datetime.now(eastern).strftime("%Y-%m-%d")
    api_key   = _get_av_key()

    print(f"{LOG_PREFIX} Manual invocation: action={args.action} date={today_str} "
          f"{datetime.now(eastern).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    if args.action == "morning":   action_morning(today_str, api_key)
    elif args.action == "retry":   action_retry(today_str, api_key)
    elif args.action == "fallback": action_fallback(today_str, api_key)
    elif args.action == "confirm": action_confirm(today_str, api_key)
    elif args.action == "trigger": action_trigger(today_str)
