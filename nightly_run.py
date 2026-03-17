#!/usr/bin/env /usr/bin/python3
"""
nightly_run.py  —  GPU container nightly pipeline

Phase 1 — INFERENCE:  Generate today's p1-p15 predictions for all active
                       tickers / model param sets trained yesterday.
                       Includes Transformer (ref, AAII, ref_noshuf, MZ) and
                       GBDT (lgbm_reference) inference for all 5 stocks.

Phase 2 — TRAINING:   Full retrain for all model types:
                       - Transformer: ref, AAII_option_vol_ratio, ref_noshuf (all 5)
                       - Transformer MZ: NVDA + PLTR only
                       - GBDT lgbm_reference: all 5 stocks

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
import os
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
parser.add_argument('--date',            default=None,        help='Override end date (YYYY-MM-DD); defaults to today')
parser.add_argument('--log-path',        default=None,        help='Path of this run\'s log file (set by scheduler; enables F1 summary)')
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
try:
    import NVDA_gbdt_param
    import PLTR_gbdt_param
    import APP_gbdt_param
    import CRDO_gbdt_param
    import INOD_gbdt_param
    import gbdt_pipeline
    GBDT_AVAILABLE = True
except ImportError as _e:
    print(f'[WARN] GBDT unavailable ({_e}) — skipping GBDT phases. '
          f'Fix: /usr/bin/python3 -m pip install lightgbm')
    GBDT_AVAILABLE = False
from trendConfig import config as trendConfig_cfg
from cp_ratio_backfill import backfill_symbol

# ── date ─────────────────────────────────────────────────────────────────────
eastern        = pytz.timezone('US/Eastern')
today_date_str = args.date if args.date else datetime.now(eastern).strftime('%Y-%m-%d')

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
    (NVDA_param.mz_reference,           'trans_mz'),
    # PLTR
    (PLTR_param.reference,              'transformer'),
    (PLTR_param.AAII_option_vol_ratio,  'transformer'),
    (PLTR_param.reference_no_shuffle,   'transformer'),
    (PLTR_param.mz_reference,           'trans_mz'),
    # TODO(tuning): swap above line to mz_reference_v2 when activating v2 training
    # APP
    (APP_param.reference,               'transformer'),
    (APP_param.AAII_option_vol_ratio,   'transformer'),
    (APP_param.reference_no_shuffle,    'transformer'),
    (APP_param.mz_reference,            'trans_mz'),
    # CRDO
    (CRDO_param.mz_reference,           'trans_mz'),
    # INOD
    (INOD_param.reference,              'transformer'),
    (INOD_param.AAII_option_vol_ratio,  'transformer'),
    (INOD_param.reference_no_shuffle,   'transformer'),
    (INOD_param.mz_reference,           'trans_mz'),
]

BACKFILL_SYMBOLS = ['NVDA', 'PLTR', 'APP', 'CRDO', 'INOD']

# ── GBDT param sets (one lgbm_reference per ticker) ──────────────────────────
# lgbm_reference = Tier1 + sector + options + IV (where backfill complete)
# lgbm_reference_base = fallback without IV (not trained by default)
GBDT_PARAMS = [
    NVDA_gbdt_param.lgbm_reference,
    PLTR_gbdt_param.lgbm_reference,
    APP_gbdt_param.lgbm_reference,
    CRDO_gbdt_param.lgbm_reference,
    INOD_gbdt_param.lgbm_reference,
] if GBDT_AVAILABLE else []

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

def _train(param, **kwargs):
    sym  = param['symbol']
    name = param.get('model_name', '')
    try:
        mainDeltafromToday.main(param, **kwargs)
    except Exception as e:
        print(f'  [ERROR] train {sym}/{name}: {e}')

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — BACKFILL  (must run before training so AAII models get today's cp_ratio)
# ═══════════════════════════════════════════════════════════════════════════════
if not args.skip_backfill:
    phase_banner(1, f'BACKFILL — {BACKFILL_SYMBOLS}')
    t0 = time.time()

    api_key = trendConfig_cfg['alpha_vantage']['key']
    for symbol in BACKFILL_SYMBOLS:
        step_banner(f'Backfill: {symbol}')
        backfill_symbol(symbol, api_key)

    print(f'\n[Phase 1 done in {time.time()-t0:.0f}s]')
else:
    print('[Phase 1 skipped]')

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
if not args.skip_inference:
    phase_banner(2, 'INFERENCE — generating today\'s predictions')
    t0 = time.time()

    for param, mtype in TICKER_PARAMS:
        sym  = param['symbol']
        name = param.get('model_name', mtype)
        step_banner(f'inference  {sym} / {name}')
        if mtype not in ('transformer', 'trans_mz'):
            print(f'[SKIP] {sym}/{name}: inference not implemented for model_type={mtype}')
            continue
        try:
            mainDeltafromToday.inference(param, end_date=today_date_str,
                                         model_type=mtype,
                                         input_comment=f'(inf)({name})')
        except Exception as e:
            print(f'  [ERROR] {sym}/{name}: {e}')

    # ── GBDT inference ─────────────────────────────────────────────────────────
    step_banner('Inference: GBDT lgbm_reference (all 5 tickers)')
    for param in GBDT_PARAMS:
        sym  = param['symbol']
        name = param['model_name']
        step_banner(f'GBDT inference  {sym} / {name}')
        # Only run if models exist (skip gracefully if not yet trained)
        h1_path = f'/workspace/model/gbdt_{sym}_{name}_short.txt'
        if not os.path.exists(h1_path):
            print(f'  [SKIP] {sym}/{name}: no trained model found at {h1_path}')
            continue
        try:
            gbdt_pipeline.infer(param)
        except Exception as e:
            print(f'  [ERROR] {sym}/{name}: {e}')

    get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)
    print(f'\n[Phase 2 done in {time.time()-t0:.0f}s]')
else:
    print('[Phase 2 skipped]')

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
if not args.skip_train:
    phase_banner(3, 'TRAINING — full retrain')
    t0 = time.time()

    # ── ref runs ───────────────────────────────────────────────────────────────
    step_banner('Training: ref (CRDO + NVDA)')
    _train(CRDO_param.reference, end_date=today_date_str, input_comment='(ref)+eps_rev+cp_prop')
    _train(NVDA_param.reference, end_date=today_date_str, input_comment='(ref)+eps_rev+cp_prop')
    get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

    step_banner('Training: ref (PLTR + APP + INOD)')
    _train(PLTR_param.reference, end_date=today_date_str, input_comment='(ref)+eps_rev+cp_prop')
    _train(APP_param.reference,  end_date=today_date_str, input_comment='(ref)+eps_rev+cp_prop')
    _train(INOD_param.reference, end_date=today_date_str, input_comment='(ref)+eps_rev+cp_prop')

    # ── AAII_option_vol_ratio runs ─────────────────────────────────────────────
    step_banner('Training: AAII_option_vol_ratio (CRDO + NVDA)')
    _train(CRDO_param.AAII_option_vol_ratio, end_date=today_date_str, input_comment='(AAII_option_vol_ratio)+eps_rev+cp_prop')
    _train(NVDA_param.AAII_option_vol_ratio, end_date=today_date_str, input_comment='(AAII_option_vol_ratio)+eps_rev+cp_prop')
    get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

    step_banner('Training: AAII_option_vol_ratio (PLTR + APP + INOD)')
    _train(PLTR_param.AAII_option_vol_ratio, end_date=today_date_str, input_comment='(AAII_option_vol_ratio)+eps_rev+cp_prop')
    _train(APP_param.AAII_option_vol_ratio,  end_date=today_date_str, input_comment='(AAII_option_vol_ratio)+eps_rev+cp_prop')
    _train(INOD_param.AAII_option_vol_ratio, end_date=today_date_str, input_comment='(AAII_option_vol_ratio)+eps_rev+cp_prop')
    get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

    # ── ref_noshuf runs ────────────────────────────────────────────────────────
    step_banner('Training: ref_noshuf (all 5 tickers)')
    _train(CRDO_param.reference_no_shuffle, end_date=today_date_str, input_comment='(ref_noshuf)+eps_rev+cp_prop')
    _train(NVDA_param.reference_no_shuffle, end_date=today_date_str, input_comment='(ref_noshuf)+eps_rev+cp_prop')
    _train(PLTR_param.reference_no_shuffle, end_date=today_date_str, input_comment='(ref_noshuf)+eps_rev+cp_prop')
    _train(APP_param.reference_no_shuffle,  end_date=today_date_str, input_comment='(ref_noshuf)+eps_rev+cp_prop')
    _train(INOD_param.reference_no_shuffle, end_date=today_date_str, input_comment='(ref_noshuf)+eps_rev+cp_prop')
    get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

    # ── MZ (multi-horizon) runs — all 5 tickers ───────────────────────────────
    step_banner('Training: trans_mz (all 5 tickers)')
    _train(NVDA_param.mz_reference, end_date=today_date_str, model_type='trans_mz', input_comment='(mh_mz)+eps_rev+cp_prop')
    # TODO(tuning): change mz_reference → mz_reference_v2 below to activate PLTR MZ v2
    _train(PLTR_param.mz_reference, end_date=today_date_str, model_type='trans_mz', input_comment='(mh_mz)+eps_rev+cp_prop')
    # mainDeltafromToday.main(PLTR_param.mz_reference_v2, end_date=today_date_str,
    #                         model_type='trans_mz', input_comment='(mh_mz_v2)')
    _train(APP_param.mz_reference,  end_date=today_date_str, model_type='trans_mz', input_comment='(mh_mz)+eps_rev+cp_prop')
    _train(CRDO_param.mz_reference, end_date=today_date_str, model_type='trans_mz', input_comment='(mh_mz)+eps_rev+cp_prop')
    _train(INOD_param.mz_reference, end_date=today_date_str, model_type='trans_mz', input_comment='(mh_mz)+eps_rev+cp_prop')
    get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

    # ── GBDT lgbm_reference — all 5 tickers ───────────────────────────────────
    step_banner('Training: GBDT lgbm_reference (all 5 tickers)')
    for param in GBDT_PARAMS:
        sym  = param['symbol']
        name = param['model_name']
        step_banner(f'GBDT train  {sym} / {name}')
        try:
            accepted = gbdt_pipeline.train(param)
            status = 'ACCEPTED' if accepted else 'REJECTED'
            print(f'  [{status}] {sym}/{name}')
        except Exception as e:
            print(f'  [ERROR] {sym}/{name}: {e}')
    get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

    print(f'\n[Phase 3 done in {time.time()-t0:.0f}s]')
else:
    print('[Phase 3 skipped]')

# ── Cache cleanup ─────────────────────────────────────────────────────────────
try:
    import fetch_cache
    fetch_cache.purge_old_cache()
except Exception as _e:
    print(f'[WARN] cache purge failed: {_e}')

# ── Final summary ─────────────────────────────────────────────────────────────
print(f'\n{"="*70}')
print(f'  nightly_run.py COMPLETE')
print(f'  {datetime.now(eastern).strftime("%Y-%m-%d %H:%M:%S %Z")}')
print(f'  Cloud upload: {"ON" if upload_to_cloud else "OFF"}')
print(f'{"="*70}\n')

# ── F1 summary vs personal best ───────────────────────────────────────────────
if args.log_path:
    try:
        import sys as _sys
        _sys.stdout.flush()
        import f1_summary
        f1_summary.print_summary(args.log_path, today_date_str)
    except Exception as _e:
        print(f'[WARN] F1 summary failed: {_e}')
