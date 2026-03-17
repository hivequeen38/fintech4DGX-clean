#!/usr/bin/env python3
"""
market_scheduler.py — single persistent daemon for market-aware nightly pipeline

Replaces cron entirely. Handles DST automatically via pytz.
Fires timed actions at correct US/Eastern wall-clock times regardless of
container system timezone.

Usage:
    python3 /workspace/market_scheduler.py            # normal (logs to stdout)
    python3 /workspace/market_scheduler.py --dry-run  # print schedule, fire nothing

Startup:
    Called by /workspace/start_services.sh on container start.
    Logs to /workspace/market_scheduler.log

Action schedule (all US/Eastern):
    10:00 AM  Mon-Fri  morning   — initial market-open check
    10:30 AM  Mon-Fri  retry     — retry if morning saw CLOSED (handles delayed opens)
    11:00 AM  Mon-Fri  fallback  — last-resort if flag still MISSING
     4:30 PM  Mon-Fri  confirm   — verify market has closed (catches early-close days)
     6:30 PM  Mon-Fri  trigger   — launch nightly_run.py if flag==OPEN

State: /workspace/market_open_today.flag
    See cron_market_check.py for flag values and action logic.
"""

import argparse
import sys
import time
from datetime import datetime, date

import pytz

import cron_market_check as chk

# ── Constants ─────────────────────────────────────────────────────────────────
EASTERN        = pytz.timezone("US/Eastern")
POLL_INTERVAL  = 30          # seconds between clock checks
LOG_PREFIX     = "[scheduler]"

# Action schedule: (hour, minute, weekdays, action_name)
# weekdays: set of isoweekday() values — 1=Mon … 5=Fri
_WEEKDAYS = {1, 2, 3, 4, 5}

SCHEDULE = [
    #  hh   mm  days      name
    (10,   0, _WEEKDAYS, "morning"),
    (10,  30, _WEEKDAYS, "retry"),
    (11,   0, _WEEKDAYS, "fallback"),
    (16,  30, _WEEKDAYS, "confirm"),
    (18,  30, _WEEKDAYS, "trigger"),
]

# How wide a window (minutes) around the target time to fire the action.
# If the daemon is restarted mid-day, actions that already fired today are
# skipped via the flag file + fired_today set.
FIRE_WINDOW_MINUTES = 4


# ── State ─────────────────────────────────────────────────────────────────────
# Tracks which (date, action_name) pairs have already been fired this process
# lifetime. On restart, we rely on read_flag() to skip redundant morning/retry/
# fallback (flag already set). confirm + trigger also check the flag before acting.
_fired: set = set()
_last_reset_date: date = None


def _reset_if_new_day(today: date) -> None:
    global _fired, _last_reset_date
    if today != _last_reset_date:
        _fired = set()
        _last_reset_date = today
        print(f"{LOG_PREFIX} New day {today} — fired-set reset.", flush=True)


def _already_fired(today: date, action: str) -> bool:
    return (today, action) in _fired


def _mark_fired(today: date, action: str) -> None:
    _fired.add((today, action))


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(dry_run: bool = False) -> None:
    api_key = chk._get_av_key()
    print(f"{LOG_PREFIX} Daemon started. dry_run={dry_run}  "
          f"poll_interval={POLL_INTERVAL}s", flush=True)
    print(f"{LOG_PREFIX} Schedule (US/Eastern):", flush=True)
    for hh, mm, _, name in SCHEDULE:
        print(f"  {hh:02d}:{mm:02d}  {name}", flush=True)

    while True:
        now_et    = datetime.now(EASTERN)
        today_et  = now_et.date()
        today_str = today_et.isoformat()

        _reset_if_new_day(today_et)

        for hh, mm, weekdays, action_name in SCHEDULE:
            # Weekday gate
            if now_et.isoweekday() not in weekdays:
                continue

            # Time window gate: fire if within [target − FIRE_WINDOW, target + FIRE_WINDOW)
            target_minutes = hh * 60 + mm
            now_minutes    = now_et.hour * 60 + now_et.minute
            in_window      = abs(now_minutes - target_minutes) < FIRE_WINDOW_MINUTES

            if not in_window:
                continue

            # Already fired today?
            if _already_fired(today_et, action_name):
                continue

            # Fire
            _mark_fired(today_et, action_name)
            ts = now_et.strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"\n{LOG_PREFIX} ── {action_name.upper()} ── {ts}", flush=True)

            if dry_run:
                print(f"{LOG_PREFIX} [DRY RUN] would call action_{action_name}()", flush=True)
                continue

            try:
                if action_name == "morning":
                    chk.action_morning(today_str, api_key)
                elif action_name == "retry":
                    chk.action_retry(today_str, api_key)
                elif action_name == "fallback":
                    chk.action_fallback(today_str, api_key)
                elif action_name == "confirm":
                    chk.action_confirm(today_str, api_key)
                elif action_name == "trigger":
                    chk.action_trigger(today_str)
            except Exception as e:
                print(f"{LOG_PREFIX} ERROR in {action_name}: {e}", flush=True)

        time.sleep(POLL_INTERVAL)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market-aware nightly pipeline daemon")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print when actions would fire without executing them",
    )
    args = parser.parse_args()

    try:
        run(dry_run=args.dry_run)
    except KeyboardInterrupt:
        print(f"\n{LOG_PREFIX} Interrupted — exiting.", flush=True)
        sys.exit(0)
