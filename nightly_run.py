#!/usr/bin/env /usr/bin/python3
"""
nightly_run.py  —  GPU container nightly pipeline

Phase 1 — INFERENCE:  Generate today's p1-p15 predictions for all active
                       tickers / model param sets trained yesterday.

Phase 2 — TRAINING:   Full retrain identical to manual_daily_train.py
                       (ref, AAII_option_vol_ratio, ref_noshuf for all 5 stocks).

Phase 3 — BACKFILL:   cp_ratio_backfill for NVDA and PLTR only
                       (limited to 2 tickers — the full 5-stock run takes too long).

Usage:
    /usr/bin/python3 /workspace/nightly_run.py            # full run, upload ON
    /usr/bin/python3 /workspace/nightly_run.py --no-upload
    /usr/bin/python3 /workspace/nightly_run.py --skip-inference
    /usr/bin/python3 /workspace/nightly_run.py --skip-train
    /usr/bin/python3 /workspace/nightly_run.py --skip-backfill
"""

import argparse
import sys
import time
from datetime import datetime

import pytz

# ── parse flags before any heavy imports ────────────────────────────────────
parser = argparse.ArgumentParser(description='Nightly GPU pipeline')
parser.add_argument('--no-upload',       action='store_true', help='Skip GCS upload')
parser.add_argument('--skip-inference',  action='store_true', help='Skip Phase 1 (inference)')
parser.add_argument('--skip-train',      action='store_true', help='Skip Phase 2 (training)')
parser.add_argument('--skip-backfill',   action='store_true', help='Skip Phase 3 (backfill)')
args = parser.parse_args()

upload_to_cloud = not args.no_upload

# ── imports (after flag parse so --help is fast) ────────────────────────────
import mainDeltafromToday
import get_historical_html
import NVDA_param
import PLTR_param
import APP_param
import CRDO_param
import INOD_param
from trendConfig import config as trendConfig_cfg
from cp_ratio_backfill import backfill_symbol

# ── date ─────────────────────────────────────────────────────────────────────
eastern       = pytz.timezone('US/Eastern')
today_date_str = datetime.now(eastern).strftime('%Y-%m-%d')

# ── param sets per ticker ────────────────────────────────────────────────────
# Each entry: (param_dict, model_type)
# model_type="transformer" = 15 single-horizon models (default)
TICKER_PARAMS = [
    # CRDO
    (CRDO_param.reference,              'transformer'),
    (CRDO_param.AAII_option_vol_ratio,  'transformer'),
    (CRDO_param.reference_no_shuffle,   'transformer'),
    # NVDA
    (NVDA_param.reference,              'transformer'),
    (NVDA_param.AAII_option_vol_ratio,  'transformer'),
    (NVDA_param.reference_no_shuffle,   'transformer'),
    (NVDA_param.mz_reference,           'trans_mz'),   # Phase 1 skipped (B-MH5)
    # PLTR
    (PLTR_param.reference,              'transformer'),
    (PLTR_param.AAII_option_vol_ratio,  'transformer'),
    (PLTR_param.reference_no_shuffle,   'transformer'),
    (PLTR_param.mz_reference,           'trans_mz'),   # Phase 1 skipped (B-MH5)
    # APP
    (APP_param.reference,               'transformer'),
    (APP_param.AAII_option_vol_ratio,   'transformer'),
    (APP_param.reference_no_shuffle,    'transformer'),
    # INOD
    (INOD_param.reference,              'transformer'),
    (INOD_param.AAII_option_vol_ratio,  'transformer'),
    (INOD_param.reference_no_shuffle,   'transformer'),
]

BACKFILL_SYMBOLS = ['NVDA', 'PLTR']

# ── helpers ──────────────────────────────────────────────────────────────────
def phase_banner(n, title):
    print(f'\n{"="*70}')
    print(f'  PHASE {n}: {title}')
    print(f'  {datetime.now(eastern).strftime("%Y-%m-%d %H:%M:%S %Z")}')
    print(f'{"="*70}\n')

def step_banner(label):
    print(f'\n{"─"*60}')
    print(f'  {label}')
    print(f'{"─"*60}')

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
if not args.skip_inference:
    phase_banner(1, 'INFERENCE — generating today\'s predictions')
    t0 = time.time()

    for param, mtype in TICKER_PARAMS:
        sym  = param['symbol']
        name = param.get('model_name', mtype)
        step_banner(f'inference  {sym} / {name}')
        # inference() does not yet support model_type="trans_mz" (backlog B-MH5)
        # skip any non-transformer param sets gracefully
        if mtype != 'transformer':
            print(f'[SKIP] {sym}/{name}: inference not yet implemented for model_type={mtype}')
            continue
        mainDeltafromToday.inference(param, end_date=today_date_str,
                                     input_comment=f'(inf)({name})')

    get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)
    print(f'\n[Phase 1 done in {time.time()-t0:.0f}s]')
else:
    print('[Phase 1 skipped]')

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
if not args.skip_train:
    phase_banner(2, 'TRAINING — full retrain')
    t0 = time.time()

    # ── ref runs ───────────────────────────────────────────────────────────────
    step_banner('Training: ref (CRDO + NVDA)')
    mainDeltafromToday.main(CRDO_param.reference, end_date=today_date_str,
                            input_comment='(ref)')
    mainDeltafromToday.main(NVDA_param.reference, end_date=today_date_str,
                            input_comment='(ref)')
    get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

    step_banner('Training: ref (PLTR + APP + INOD)')
    mainDeltafromToday.main(PLTR_param.reference, end_date=today_date_str,
                            input_comment='(ref)')
    mainDeltafromToday.main(APP_param.reference,  end_date=today_date_str,
                            input_comment='(ref)')
    mainDeltafromToday.main(INOD_param.reference, end_date=today_date_str,
                            input_comment='(ref)')

    # ── AAII_option_vol_ratio runs ─────────────────────────────────────────────
    step_banner('Training: AAII_option_vol_ratio (CRDO + NVDA)')
    mainDeltafromToday.main(CRDO_param.AAII_option_vol_ratio, end_date=today_date_str,
                            input_comment='(AAII_option_vol_ratio)')
    mainDeltafromToday.main(NVDA_param.AAII_option_vol_ratio, end_date=today_date_str,
                            input_comment='(AAII_option_vol_ratio)')
    get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

    step_banner('Training: AAII_option_vol_ratio (PLTR + APP + INOD)')
    mainDeltafromToday.main(PLTR_param.AAII_option_vol_ratio, end_date=today_date_str,
                            input_comment='(AAII_option_vol_ratio)')
    mainDeltafromToday.main(APP_param.AAII_option_vol_ratio,  end_date=today_date_str,
                            input_comment='(AAII_option_vol_ratio)')
    mainDeltafromToday.main(INOD_param.AAII_option_vol_ratio, end_date=today_date_str,
                            input_comment='(AAII_option_vol_ratio)')
    get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

    # ── ref_noshuf runs ────────────────────────────────────────────────────────
    step_banner('Training: ref_noshuf (all 5 tickers)')
    mainDeltafromToday.main(CRDO_param.reference_no_shuffle, end_date=today_date_str,
                            input_comment='(ref_noshuf)')
    mainDeltafromToday.main(NVDA_param.reference_no_shuffle, end_date=today_date_str,
                            input_comment='(ref_noshuf)')
    mainDeltafromToday.main(PLTR_param.reference_no_shuffle, end_date=today_date_str,
                            input_comment='(ref_noshuf)')
    mainDeltafromToday.main(APP_param.reference_no_shuffle,  end_date=today_date_str,
                            input_comment='(ref_noshuf)')
    mainDeltafromToday.main(INOD_param.reference_no_shuffle, end_date=today_date_str,
                            input_comment='(ref_noshuf)')
    get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

    # ── MZ (multi-horizon) runs — NVDA + PLTR only ────────────────────────────
    step_banner('Training: trans_mz (NVDA + PLTR)')
    mainDeltafromToday.main(NVDA_param.mz_reference, end_date=today_date_str,
                            model_type='trans_mz', input_comment='(mh_mz)')
    mainDeltafromToday.main(PLTR_param.mz_reference, end_date=today_date_str,
                            model_type='trans_mz', input_comment='(mh_mz)')
    get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

    print(f'\n[Phase 2 done in {time.time()-t0:.0f}s]')
else:
    print('[Phase 2 skipped]')

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — BACKFILL  (NVDA + PLTR only — full 5-stock run takes too long)
# ═══════════════════════════════════════════════════════════════════════════════
if not args.skip_backfill:
    phase_banner(3, f'BACKFILL — {BACKFILL_SYMBOLS}')
    t0 = time.time()

    api_key = trendConfig_cfg['alpha_vantage']['key']
    for symbol in BACKFILL_SYMBOLS:
        step_banner(f'Backfill: {symbol}')
        backfill_symbol(symbol, api_key)

    print(f'\n[Phase 3 done in {time.time()-t0:.0f}s]')
else:
    print('[Phase 3 skipped]')

# ── Final summary ─────────────────────────────────────────────────────────────
print(f'\n{"="*70}')
print(f'  nightly_run.py COMPLETE')
print(f'  {datetime.now(eastern).strftime("%Y-%m-%d %H:%M:%S %Z")}')
print(f'  Cloud upload: {"ON" if upload_to_cloud else "OFF"}')
print(f'{"="*70}\n')
