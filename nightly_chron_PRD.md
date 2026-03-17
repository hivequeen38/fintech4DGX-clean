# PRD: Nightly Cron Job — Market-Aware Inference + Training Pipeline

**Status:** Implemented
**Author:** —
**Created:** 2026-03-08

---

## 1. Overview

Automate the nightly GPU pipeline (`nightly_run.py`) so it runs once per trading day,
triggered only when the US equity market actually opened. The job must skip weekends,
federal holidays, and unexpected market closures (e.g. emergency early closures).

---

## 2. Goals

| # | Goal |
|---|------|
| G1 | Run inference + training automatically every trading day with no manual intervention |
| G2 | Use Alpha Vantage API to confirm market status — do not hardcode holiday calendars |
| G3 | Trigger at 6:30 PM ET — giving ~2.5h post-close buffer for AV daily data availability |
| G4 | Skip the nightly run cleanly on non-trading days with a clear log entry |
| G5 | All scheduling logic runs inside the GPU container; no external orchestrator required |

---

## 3. Non-Goals

- Real-time intraday data or intraday trading signals
- Multi-timezone support (US Eastern only)
- Alerting / PagerDuty / Slack notifications (separate concern)
- Handling of partial-day trading halts (circuit breakers mid-day)

---

## 4. Implementation (as built)

### Files created

| File | Role |
|------|------|
| `cron_market_check.py` | Market status logic, flag file I/O, action functions |
| `market_scheduler.py` | Single persistent daemon — fires actions at correct ET times |
| `start_services.sh` | Container entrypoint — starts daemon in background, then execs CMD |
| `Dockerfile` | Updated with `ENTRYPOINT ["/start_services.sh"]` |

### Design choice: single daemon (not cron)

Three-entry cron was rejected in favour of a single Python daemon
(`market_scheduler.py`). The daemon:

- Polls every 30 seconds, checks current US/Eastern time via `pytz`
- Handles DST automatically — no UTC offset hardcoding
- Fires each action within a ±4-minute window around the target time
- Tracks fired actions in memory (`_fired` set); resets at ET midnight
- Survives daemon restart mid-day (action functions re-check the flag file before acting)
- Persists across container restarts via `ENTRYPOINT` in the Dockerfile

---

## 5. Schedule (US Eastern, Mon–Fri only)

```
10:00 AM  morning   — primary market-open check
10:30 AM  retry     — re-check if morning returned CLOSED (handles delayed opens)
11:00 AM  fallback  — emergency check if flag is still MISSING (container was down)
 4:30 PM  confirm   — verify market has now closed (catches early-close days)
 6:30 PM  trigger   — launch nightly_run.py --date YYYY-MM-DD if flag == OPEN
```

---

## 6. Alpha Vantage API — Belt-and-Suspenders Market Check

Two independent checks must both agree for `is_market_open()` to return `True`:

### Check 1: AV MARKET_STATUS
```
GET https://www.alphavantage.co/query?function=MARKET_STATUS&apikey=<KEY>
```
Returns `"current_status": "open" | "closed"` for US Equity region.

### Check 2: AV GLOBAL_QUOTE for SPY — latest trading day
```
GET https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=SPY&apikey=<KEY>
```
On a trading day: `"07. latest trading day"` == today's date
On a holiday: `"07. latest trading day"` == prior trading day's date

This is more reliable than the `current_status` field alone because:
- `MARKET_STATUS` has been observed stale on some holidays (community reports)
- The SPY latest-date check is structurally reliable — AV only updates it when the market actually trades

If `MARKET_STATUS` returns an error (AV unreachable), falls back to SPY date check
alone with a warning log.

---

## 7. Flag File

**Path:** `/workspace/market_open_today.flag`

| Value | Written by | Meaning |
|-------|-----------|---------|
| `OPEN YYYY-MM-DD` | morning / retry / fallback | Market opened; trigger at 6:30 PM |
| `CLOSED YYYY-MM-DD` | morning / retry / fallback | Holiday or closure; skip tonight |
| `SKIP YYYY-MM-DD` | confirm (4:30 PM) | Market unexpectedly still open — skip run |

The trigger action reads the flag, verifies the embedded date matches today (stale-flag
guard), and only launches `nightly_run.py` if value is `OPEN`.

---

## 8. Fault Handling

| Scenario | Behaviour |
|----------|-----------|
| Holiday / weekend | `morning` writes CLOSED; `trigger` skips |
| AV API error at 10 AM | Writes CLOSED; `retry` at 10:30 tries again |
| Delayed market open | `retry` at 10:30 catches it; writes OPEN |
| Container down 10–10:30 AM | `fallback` at 11 AM writes flag from scratch |
| All three checks fail (container down until 6:30 PM) | Flag is MISSING; `trigger` skips with ERROR log |
| Market still open at 4:30 PM (emergency) | `confirm` writes SKIP; `trigger` skips |
| nightly_run.py crashes | Already handled by IMAP fix; GBDT resume script documented in `turn_on_trans_mz.md` |

---

## 9. Container Lifecycle

The daemon is baked into the Dockerfile via ENTRYPOINT:

```dockerfile
COPY start_services.sh /start_services.sh
RUN chmod +x /start_services.sh
ENTRYPOINT ["/start_services.sh"]
CMD ["/bin/bash"]
```

`start_services.sh` starts `market_scheduler.py` in the background (nohup), then
`exec "$@"` hands control to CMD (`/bin/bash` by default). This means:

- `docker run -it <image>` → daemon starts + interactive bash session
- `docker exec -it <container> bash` → just bash (daemon already running from startup)
- Container rebuild / restart → daemon auto-starts via ENTRYPOINT

To verify daemon is running inside the container:
```bash
ps aux | grep market_scheduler
tail -f /workspace/market_scheduler.log
```

---

## 10. Manual Invocation (debugging)

Each action can be called directly without the daemon:

```bash
# Check what the market status is right now
python3 /workspace/cron_market_check.py --action=morning

# Force-trigger nightly run for a specific date (bypasses flag check)
python3 /workspace/nightly_run.py --date 2026-03-06

# Dry-run the daemon to verify schedule timing
python3 /workspace/market_scheduler.py --dry-run
```

---

## 11. Acceptance Criteria

- [ ] On a normal trading day: nightly_run.py launches at 6:30 PM ET, log file created
- [ ] On a federal holiday: morning check writes `CLOSED`, trigger skips with log entry
- [ ] On a weekend: daemon fires no actions (Mon-Fri gate in schedule)
- [ ] Delayed open: retry at 10:30 AM catches it and writes OPEN
- [ ] Container down 10–10:30 AM: fallback at 11 AM writes correct flag
- [ ] All checks fail: trigger logs MISSING + ERROR, skips run
- [ ] AV MARKET_STATUS stale: SPY date check overrides correctly
- [ ] Container rebuild: daemon auto-starts via ENTRYPOINT
- [ ] DST transitions: correct ET fire times observed the week after each DST change
- [ ] `docker exec` sessions unaffected (daemon already running; exec just opens bash)
