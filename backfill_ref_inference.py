"""
Backfill daily inference for NVDA reference (shuffle=True) model.

Runs inference 10 times — once per trading day over the last 10 days in
TMP.csv — and appends each result to NVDA_15d_from_today_predictions.csv.
Uploads all results to GCS after the loop completes.

Usage:
    /usr/bin/python3 /workspace/backfill_ref_inference.py

Notes:
  - load_cache=False  → uses existing NVDA_TMP.csv (no API calls)
  - Model loaded: model/model_NVDA_ref_fixed_noTimesplit_{1..15}.pth
  - Dedup is automatic: processDeltaFromTodayResults skips rows with same
    date + comment prefix, so re-running this script is idempotent.
  - Upload happens once at the end (not per date) to minimise GCS round-trips.
"""

import sys
import time
from datetime import datetime

import pandas as pd
import pytz

sys.path.insert(0, '/workspace')

import PLTR_param
import mainDeltafromToday
import get_historical_html

# ── Config ────────────────────────────────────────────────────────────────────

PARAM       = PLTR_param.reference          # model_name='ref', shuffle=True
N_DAYS      = 10                            # how many trailing trading days to fill
UPLOAD      = True                          # set False to dry-run without GCS upload

# ── Derive last N trading days from TMP.csv ───────────────────────────────────

df_tmp = pd.read_csv('/workspace/PLTR_TMP.csv')
df_tmp['date'] = pd.to_datetime(df_tmp['date'], format='mixed').dt.normalize()
df_tmp = df_tmp.sort_values('date').reset_index(drop=True)

trading_days = df_tmp['date'].iloc[-N_DAYS:].dt.strftime('%Y-%m-%d').tolist()

eastern  = pytz.timezone('US/Eastern')
run_date = datetime.now(eastern).strftime('%Y-%m-%d %H:%M')

print(f"NVDA reference backfill — {N_DAYS} dates")
print(f"Run date : {run_date}")
print(f"Dates    : {trading_days[0]}  →  {trading_days[-1]}")
print(f"Upload   : {'ON' if UPLOAD else 'OFF (dry run)'}")
print("=" * 60)

# ── Inference loop ────────────────────────────────────────────────────────────

for i, date_str in enumerate(trading_days, 1):
    print(f"\n[{i}/{N_DAYS}]  {date_str}")
    t0 = time.time()
    try:
        mainDeltafromToday.inference(
            PARAM,
            end_date      = date_str,
            run_date      = run_date,
            input_comment = f'(inf)(ref)',
            load_cache    = False,      # use existing TMP.csv — no API calls
        )
        print(f"  done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  ERROR on {date_str}: {e}")
        # continue to next date rather than abort the whole loop
        continue

# ── Upload once at the end ────────────────────────────────────────────────────

last_date = trading_days[-1]
print(f"\n{'='*60}")
print(f"Uploading all results (date={last_date}, upload={UPLOAD}) ...")
get_historical_html.upload_all_results(last_date, upload_to_cloud=UPLOAD)
print("Upload complete.")
print(f"\nBackfill done.  {N_DAYS} dates processed → PLTR_15d_from_today_predictions.csv")
