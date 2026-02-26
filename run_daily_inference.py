"""
Daily inference script — use yesterday's trained models to generate signals.

Runs inference (no training) for all 5 active stocks × 3 param sets each.
Fetches fresh data (same as training pipeline), loads existing .pth files,
and produces the same result format + cloud upload as the training run.

Usage:
    python run_daily_inference.py

Typical runtime: ~10 min (vs ~3 hours for full training run).
Run this after market close for a same-day signal before overnight training completes.
"""

from datetime import datetime
import pytz
import get_historical_html
import mainDeltafromToday
import NVDA_param
import PLTR_param
import APP_param
import INOD_param
import CRDO_param

today_date_str = datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d")

response = input("Upload results to Google Cloud? (Y/N): ").strip().upper()
upload_to_cloud = response == 'Y'
print(f"Running daily inference for {today_date_str}")
print(f"Cloud upload: {'ON' if upload_to_cloud else 'OFF - results saved locally only'}")

# ── reference runs ────────────────────────────────────────────────────────────
mainDeltafromToday.inference(CRDO_param.reference, end_date=today_date_str, input_comment='(ref) inference')
mainDeltafromToday.inference(NVDA_param.reference, end_date=today_date_str, input_comment='(ref) inference')
get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

mainDeltafromToday.inference(PLTR_param.reference, end_date=today_date_str, input_comment='(ref) inference')
mainDeltafromToday.inference(APP_param.reference,  end_date=today_date_str, input_comment='(ref) inference')
mainDeltafromToday.inference(INOD_param.reference, end_date=today_date_str, input_comment='(ref) inference')

# ── AAII_option_vol_ratio runs ────────────────────────────────────────────────
mainDeltafromToday.inference(CRDO_param.AAII_option_vol_ratio, end_date=today_date_str, input_comment='(AAII) inference')
mainDeltafromToday.inference(NVDA_param.AAII_option_vol_ratio, end_date=today_date_str, input_comment='(AAII) inference')
get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

mainDeltafromToday.inference(PLTR_param.AAII_option_vol_ratio, end_date=today_date_str, input_comment='(AAII) inference')
mainDeltafromToday.inference(APP_param.AAII_option_vol_ratio,  end_date=today_date_str, input_comment='(AAII) inference')
mainDeltafromToday.inference(INOD_param.AAII_option_vol_ratio, end_date=today_date_str, input_comment='(AAII) inference')

# ── reference_no_shuffle runs ─────────────────────────────────────────────────
mainDeltafromToday.inference(CRDO_param.reference_no_shuffle, end_date=today_date_str, input_comment='(ref_noshuf) inference')
mainDeltafromToday.inference(NVDA_param.reference_no_shuffle, end_date=today_date_str, input_comment='(ref_noshuf) inference')
mainDeltafromToday.inference(PLTR_param.reference_no_shuffle, end_date=today_date_str, input_comment='(ref_noshuf) inference')
mainDeltafromToday.inference(APP_param.reference_no_shuffle,  end_date=today_date_str, input_comment='(ref_noshuf) inference')
mainDeltafromToday.inference(INOD_param.reference_no_shuffle, end_date=today_date_str, input_comment='(ref_noshuf) inference')

get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

print(f"\nDaily inference complete for {today_date_str}.")
