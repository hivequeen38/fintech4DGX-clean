"""
nvda_mz_compare.py

Train MultiHorizonTransformer with NVDA mz_reference param set (start_date 2021-03-01)
and compare per-class F1 (flat / UP / DOWN) for each of the 15 horizons against
last night's production run from NVDA_trend.jsonl:
  - ref             (shuffled — look-ahead inflated)
  - ref_noshuf      (honest chronological split)
  - AAII_option_vol_ratio

ISOLATION: all model + scaler artifacts go to a temp directory (auto-cleaned).
Production models in /workspace/model/ are never touched.

Usage:
    python3 /workspace/nvda_mz_compare.py
"""

import json
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, '/workspace')

# ────────────────────────────────────────────────────────────────────────────
# 1. Param setup
# ────────────────────────────────────────────────────────────────────────────

from NVDA_param import mz_reference
from trendAnalysisFromTodayNew import (
    analyze_trend_multi_horizon,
    build_multi_horizon_labels,
)

NUM_HORIZONS = 15
JSONL_PATH   = '/workspace/NVDA_trend.jsonl'
CACHE_PATH   = '/workspace/NVDA_TMP.csv'

mh_param = {
    **mz_reference,
    'model_type':     'multi_horizon_transformer',
    'model_name':     'mh_mz',
    'shuffle_splits': False,
    'num_horizons':   NUM_HORIZONS,
    'num_epochs':     200,
}

assert not mh_param.get('shuffle_splits', False), \
    "shuffle_splits must be False for multi-horizon"

assert os.path.exists(CACHE_PATH), \
    f"Cached NVDA data not found: {CACHE_PATH}. Run a training session first."

# ────────────────────────────────────────────────────────────────────────────
# 2. Train MZ model (artifacts go to temp dir — never touches /workspace/model/)
# ────────────────────────────────────────────────────────────────────────────

save_dir = tempfile.mkdtemp(prefix='nvda_mz_compare_')
prod_model_dir = os.path.abspath('/workspace/model/')
assert not os.path.abspath(save_dir).startswith(prod_model_dir), \
    "temp dir must not be inside production model dir"

print(f"\n{'='*70}")
print("NVDA MZ (MultiHorizonTransformer) — Full Class F1 Comparison")
print(f"  Feature set:  mz_reference  ({len([c for c in mz_reference['selected_columns'] if c != 'label'])} features requested)")
print(f"  Start date:   {mh_param['start_date']}")
print(f"  save_dir:     {save_dir}")
print(f"  num_epochs:   {mh_param['num_epochs']}")
print(f"{'='*70}\n")

model, test_preds, test_labels, model_path, scaler_path = analyze_trend_multi_horizon(
    config={},
    param=mh_param,
    current_day_offset='mz_compare',
    incr_df=pd.DataFrame(),
    turn_random_on=False,
    use_cached_data=True,
    save_dir=save_dir,
)

# Confirm isolation
assert os.path.exists(model_path), f"Model not saved: {model_path}"
assert not os.path.abspath(model_path).startswith(prod_model_dir), \
    f"ISOLATION BREACH: model written to production dir: {model_path}"
print(f"\n[OK] Isolation confirmed: model at {model_path}")

# ────────────────────────────────────────────────────────────────────────────
# 3. Compute per-horizon per-class F1 for the MH model
# ────────────────────────────────────────────────────────────────────────────

assert test_preds.shape == test_labels.shape, \
    f"Shape mismatch: preds={test_preds.shape} labels={test_labels.shape}"
assert not np.isnan(test_preds).any(), "test_preds contain NaN"

unique_preds = np.unique(test_preds)
if len(unique_preds) == 1:
    warnings.warn(f"MZ model collapsed: only predicts class {unique_preds}")

mz_per_horizon = []
for h in range(NUM_HORIZONS):
    _, _, f1_h, _ = precision_recall_fscore_support(
        test_labels[:, h], test_preds[:, h],
        average=None, labels=[0, 1, 2], zero_division=0
    )
    mz_per_horizon.append({
        'horizon': h + 1,
        'f1_flat': float(f1_h[0]),
        'f1_up':   float(f1_h[1]),
        'f1_down': float(f1_h[2]),
        'macro':   float(np.mean(f1_h)),
    })

# ────────────────────────────────────────────────────────────────────────────
# 4. Load per-horizon per-class F1 from NVDA_trend.jsonl (read-only)
# ────────────────────────────────────────────────────────────────────────────

def load_latest_per_horizon_f1(jsonl_path, model_name):
    """Return {h: {f1_flat, f1_up, f1_down, macro}} for the most recent run entries."""
    if not os.path.exists(jsonl_path):
        return {}
    horizon_f1 = {}
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                _, params, results = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if params.get('model_name') != model_name:
                continue
            h = params.get('target_size')
            if h is None:
                continue
            f0 = results.get('Test F1 for class 0: ')
            f1 = results.get('Test F1 for class 1: ')
            f2 = results.get('Test F1 for class 2: ')
            if f0 is None or f1 is None or f2 is None:
                continue
            horizon_f1[h] = {
                'f1_flat': f0,
                'f1_up':   f1,
                'f1_down': f2,
                'macro':   np.mean([f0, f1, f2]),
            }
    return horizon_f1


baseline_profiles = ['ref', 'ref_noshuf', 'AAII_option_vol_ratio']
baselines = {}
for profile in baseline_profiles:
    data = load_latest_per_horizon_f1(JSONL_PATH, profile)
    if not data:
        warnings.warn(f"No JSONL data found for '{profile}' — baseline column will be blank")
    baselines[profile] = data

# ────────────────────────────────────────────────────────────────────────────
# 5. Print full comparison table
# ────────────────────────────────────────────────────────────────────────────

COL_W = 32   # width of each model block

header_models = ['MH/MZ', 'ref (shuffled)', 'ref_noshuf', 'AAII_CP_VOL']
labels_map    = {'ref': 'ref (shuffled)', 'ref_noshuf': 'ref_noshuf',
                 'AAII_option_vol_ratio': 'AAII_CP_VOL'}

sep  = '─' * (5 + COL_W * 4)
sep2 = '═' * (5 + COL_W * 4)

def fmt_row(h_label, mz, bl_ref, bl_ns, bl_aaii):
    """Format one data row: each block is flat/UP/DN/mac."""
    def blk(d):
        if d is None:
            return f"{'n/a':>6} {'n/a':>6} {'n/a':>6} {'n/a':>6}"
        return (f"{d['f1_flat']:6.3f} {d['f1_up']:6.3f} "
                f"{d['f1_down']:6.3f} {d['macro']:6.3f}")
    return f"  {h_label:<4}  {blk(mz)}  {blk(bl_ref)}  {blk(bl_ns)}  {blk(bl_aaii)}"

def hdr_blk(label):
    return f"{'flat':>6} {'UP':>6} {'DN':>6} {'mac':>6}"

print(f"\n{sep2}")
print("NVDA — Multi-Zone (MH) vs Single-Horizon Baselines: Per-Class F1 per Horizon")
print(sep2)
print(f"  {'':4}  {'MH / MZ (mz_reference)':^{COL_W-2}}"
      f"  {'ref (shuffled—inflated)':^{COL_W-2}}"
      f"  {'ref_noshuf (honest)':^{COL_W-2}}"
      f"  {'AAII_CP_VOL':^{COL_W-2}}")
print(f"  {'h':4}  {hdr_blk('mz'):<{COL_W-2}}"
      f"  {hdr_blk('ref'):<{COL_W-2}}"
      f"  {hdr_blk('ns'):<{COL_W-2}}"
      f"  {hdr_blk('aaii'):<{COL_W-2}}")
print(sep)

rows_out = []
for row in mz_per_horizon:
    h    = row['horizon']
    thr  = '±3%' if h <= 5 else '±5%'
    tag  = f"h={h:2d}({thr})"

    mz_blk   = row
    ref_blk  = baselines['ref'].get(h)
    ns_blk   = baselines['ref_noshuf'].get(h)
    aaii_blk = baselines['AAII_option_vol_ratio'].get(h)

    print(fmt_row(tag, mz_blk, ref_blk, ns_blk, aaii_blk))
    rows_out.append({
        'horizon': h, 'threshold': thr,
        'mz_flat': row['f1_flat'], 'mz_up': row['f1_up'],
        'mz_dn':   row['f1_down'], 'mz_mac': row['macro'],
        'ref_flat':  ref_blk['f1_flat']  if ref_blk  else None,
        'ref_up':    ref_blk['f1_up']    if ref_blk  else None,
        'ref_dn':    ref_blk['f1_down']  if ref_blk  else None,
        'ref_mac':   ref_blk['macro']    if ref_blk  else None,
        'ns_flat':   ns_blk['f1_flat']   if ns_blk   else None,
        'ns_up':     ns_blk['f1_up']     if ns_blk   else None,
        'ns_dn':     ns_blk['f1_down']   if ns_blk   else None,
        'ns_mac':    ns_blk['macro']     if ns_blk   else None,
        'aaii_flat': aaii_blk['f1_flat'] if aaii_blk else None,
        'aaii_up':   aaii_blk['f1_up']   if aaii_blk else None,
        'aaii_dn':   aaii_blk['f1_down'] if aaii_blk else None,
        'aaii_mac':  aaii_blk['macro']   if aaii_blk else None,
    })

print(sep)

# Bucket summaries
for bname, h_list in [('h=1-5  (±3%)', range(1, 6)), ('h=6-15 (±5%)', range(6, 16))]:
    def bucket_means(col_key):
        vals = [rows_out[h-1][col_key] for h in h_list if rows_out[h-1][col_key] is not None]
        return np.mean(vals) if vals else float('nan')

    mz_mac = bucket_means('mz_mac')
    ns_mac = bucket_means('ns_mac')
    flag   = '✓' if mz_mac >= ns_mac else '↓'
    print(f"  {bname}  MZ macro={mz_mac:.3f}  ref_noshuf={ns_mac:.3f}  {flag}")

print(sep2)
print("Note: 'ref' uses shuffled train/test split → inflated F1 (look-ahead bias).")
print("      MH/MZ thresholds: ±3% for h=1-5, ±5% for h=6-15.")
print(f"      MZ feature set: mz_reference ({test_preds.shape[1]} horizons, 82 features).")
print(f"      4 SOX trend/vol features absent from cache (dropped silently).")
print(sep2)

# ────────────────────────────────────────────────────────────────────────────
# 6. Save comparison CSV
# ────────────────────────────────────────────────────────────────────────────

csv_path = '/workspace/nvda_mz_f1_comparison.csv'
pd.DataFrame(rows_out).to_csv(csv_path, index=False)
print(f"\n[OK] Full comparison saved to {csv_path}")
print("[DONE]")
