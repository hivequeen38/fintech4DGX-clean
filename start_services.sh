#!/bin/bash
# start_services.sh — container entrypoint
#
# Starts the market scheduler daemon in the background, then hands off
# to whatever CMD was specified (default: /bin/bash for interactive use).
#
# The daemon logs to /workspace/market_scheduler.log.
# To check daemon status:   ps aux | grep market_scheduler
# To tail daemon log:        tail -f /workspace/market_scheduler.log

set -e

SCHEDULER="/workspace/market_scheduler.py"
LOG="/workspace/market_scheduler.log"
PYTHON="/usr/bin/python3"

if [ -f "$SCHEDULER" ]; then
    echo "[entrypoint] Starting market_scheduler.py ..."
    nohup "$PYTHON" "$SCHEDULER" >> "$LOG" 2>&1 &
    SCHED_PID=$!
    echo "[entrypoint] market_scheduler.py started (PID $SCHED_PID)"
else
    echo "[entrypoint] WARNING: $SCHEDULER not found — scheduler not started."
fi

# Hand off to CMD (e.g. /bin/bash for interactive, or a specific script)
exec "$@"
