"""
Retrospective daily inference backtest — NVDA Transformer models.

For each of the last 100 trading days (prediction dates), loads the already-trained
Transformer models for ref_noshuf and AAII_option_vol_ratio, runs a forward pass
using the same window logic as make_prediciton_test(), and compares against actual
next-h-day returns from TMP.csv.

Transformer class encoding: 0 = flat (__), 1 = UP, 2 = DN
Label threshold: param['threshold'] = 0.05 (flat band ± 5%)

Output: per-class F1 table for each param set, averaged over 100 prediction dates.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from joblib import load
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace')
import NVDA_param
from gbdt_pipeline import _merge_cp_ratios   # reuse cp-ratio derivation

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_PRED_DATES = 100
THRESHOLD    = 0.05            # ± 5% flat band (matches both param['threshold'])
BATCH_SIZE   = 128
TARGET_SIZE  = 15
SEQ_LEN      = BATCH_SIZE + TARGET_SIZE   # 143

PARAM_SETS = {
    'ref_noshuf':          NVDA_param.reference_no_shuffle,
    'AAII_option_vol_ratio': NVDA_param.AAII_option_vol_ratio,
}

# ---------------------------------------------------------------------------
# Load and prepare TMP data
# ---------------------------------------------------------------------------

print("Loading NVDA_TMP.csv ...")
df_base = pd.read_csv('/workspace/NVDA_TMP.csv')
df_base['date'] = pd.to_datetime(df_base['date'], format='mixed').dt.normalize()
df_base = df_base.sort_values('date').reset_index(drop=True)
print(f"  {len(df_base)} rows  {df_base['date'].iloc[0].date()} → {df_base['date'].iloc[-1].date()}")

# Merge CP ratios (needed for AAII; harmless for ref_noshuf)
print("Merging CP ratio features ...")
df_cp = _merge_cp_ratios(df_base.copy().set_index('date'), 'NVDA').reset_index()
# _merge_cp_ratios expects index=date, returns with index=date → we reset it
# Ensure date column is clean
df_cp['date'] = pd.to_datetime(df_cp['date']).dt.normalize()
df_cp = df_cp.sort_values('date').reset_index(drop=True)
print(f"  cp_sentiment_ratio non-zero: {(df_cp['cp_sentiment_ratio'] != 0).sum()}/{len(df_cp)}")

# ---------------------------------------------------------------------------
# Prediction date window
# ---------------------------------------------------------------------------

all_dates = df_base['date'].tolist()
# Leave the last 15 rows as "future actuals" for h=15 predictions
# Prediction dates: positions [-(N+15) .. -15]
pred_dates = all_dates[-(N_PRED_DATES + TARGET_SIZE):-TARGET_SIZE]
print(f"\nPrediction window: {pred_dates[0].date()} → {pred_dates[-1].date()}  "
      f"({len(pred_dates)} dates)")

# ---------------------------------------------------------------------------
# Helper: compute actual zone from return
# ---------------------------------------------------------------------------

def actual_zone(price_t0: float, price_th: float) -> int:
    """0=flat, 1=UP, 2=DN  (matches Transformer class encoding)."""
    ret = (price_th - price_t0) / price_t0
    if ret > THRESHOLD:
        return 1
    if ret < -THRESHOLD:
        return 2
    return 0

# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

results = {}   # {ps_name: {h: {'preds': [...], 'actuals': [...]}}}

for ps_name, param in PARAM_SETS.items():
    model_name_str = param['model_name']
    print(f"\n{'='*70}")
    print(f"  Backtesting: {ps_name}  (model_name={model_name_str})")
    print(f"{'='*70}")

    # Select the right data frame (AAII needs CP columns)
    selected_cols = [c for c in param['selected_columns'] if c != 'label']
    needs_cp = any(c in selected_cols for c in
                   ('cp_sentiment_ratio', 'options_volume_ratio',
                    'iv_30d', 'iv_skew_30d', 'iv_term_ratio'))
    df_use = df_cp if needs_cp else df_base

    # Check all selected columns exist in df_use
    missing = [c for c in selected_cols if c not in df_use.columns]
    if missing:
        print(f"  [SKIP] Missing columns in data: {missing}")
        continue

    # Load scaler
    scaler_path = f'/workspace/NVDA_{model_name_str}_scaler.joblib'
    if not os.path.exists(scaler_path):
        print(f"  [SKIP] Scaler not found: {scaler_path}")
        continue
    scaler = load(scaler_path)
    print(f"  Scaler loaded from {scaler_path}")

    # Load all 15 horizon models once
    print("  Loading 15 models ...", end=' ', flush=True)
    models = {}
    all_ok = True
    for h in range(1, TARGET_SIZE + 1):
        mp = f'/workspace/model/model_NVDA_{model_name_str}_fixed_noTimesplit_{h}.pth'
        if not os.path.exists(mp):
            print(f"\n  [SKIP] Model missing: {mp}")
            all_ok = False
            break
        models[h] = torch.load(mp, weights_only=False)
        models[h].eval()
    if not all_ok:
        continue
    print("done")

    # Per-horizon storage
    store = {h: {'preds': [], 'actuals': []} for h in range(1, TARGET_SIZE + 1)}

    for pred_date in pred_dates:
        # Find index of pred_date in df_use
        idx_matches = df_use.index[df_use['date'] == pred_date].tolist()
        if not idx_matches:
            continue
        pred_idx = idx_matches[0]

        if pred_idx < SEQ_LEN - 1:
            continue   # not enough history

        # Extract feature window: last SEQ_LEN rows up to and including pred_date
        window = df_use.iloc[pred_idx - SEQ_LEN + 1 : pred_idx + 1][selected_cols].copy()
        window = window.ffill().bfill()

        if len(window) != SEQ_LEN:
            continue

        try:
            features = scaler.transform(window.values)
        except Exception as e:
            print(f"  Scaler error on {pred_date.date()}: {e}")
            continue

        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, SEQ_LEN, n_feat]

        # Find pred_date index in df_base for actual price lookup
        base_idx_matches = df_base.index[df_base['date'] == pred_date].tolist()
        if not base_idx_matches:
            continue
        base_idx = base_idx_matches[0]
        price_t0 = df_base['adjusted close'].iloc[base_idx]

        for h in range(1, TARGET_SIZE + 1):
            # Actual price at t0 + h business days
            target_base_idx = base_idx + h
            if target_base_idx >= len(df_base):
                continue
            price_th = df_base['adjusted close'].iloc[target_base_idx]

            act = actual_zone(price_t0, price_th)

            with torch.no_grad():
                out    = models[h](input_tensor)          # [1, SEQ_LEN, 3]
                probs  = torch.softmax(out, dim=2)
                pred_c = torch.argmax(probs, dim=2).squeeze(0)  # [SEQ_LEN]
                # Last of the last target_size positions = prediction for pred_date
                pred   = pred_c[-1].item()               # 0=flat, 1=UP, 2=DN

            store[h]['preds'].append(pred)
            store[h]['actuals'].append(act)

    results[ps_name] = store
    n_eval = len(store[1]['preds'])
    print(f"  Evaluated {n_eval} prediction dates across 15 horizons")

# ---------------------------------------------------------------------------
# Print comparison tables
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: 'flat', 1: 'UP', 2: 'DN'}

def agg(store):
    """Average per-class F1 across horizons."""
    rows = []
    for h in range(1, 16):
        preds   = np.array(store[h]['preds'])
        actuals = np.array(store[h]['actuals'])
        if len(preds) == 0:
            rows.append({'h': h, 'macro': 0, 'flat': 0, 'up': 0, 'dn': 0})
            continue
        macro = f1_score(actuals, preds, average='macro', zero_division=0)
        per   = f1_score(actuals, preds, labels=[0, 1, 2], average=None, zero_division=0)
        rows.append({'h': h, 'macro': macro, 'flat': per[0], 'up': per[1], 'dn': per[2]})
    return rows

print(f"\n\n{'═'*80}")
print(f"  NVDA Transformer Retrospective Backtest  —  last {N_PRED_DATES} trading days")
print(f"  Threshold: ±{THRESHOLD*100:.0f}%  |  Classes: 0=flat, 1=UP, 2=DN")
print(f"{'═'*80}")

# Summary table
print(f"\n  {'MODEL':<30} {'macro_F1':>9} {'up_F1':>8} {'dn_F1':>8} {'flat_F1':>9}")
print(f"  {'─'*30} {'─'*9} {'─'*8} {'─'*8} {'─'*9}")
for ps_name in PARAM_SETS:
    if ps_name not in results:
        continue
    rows = agg(results[ps_name])
    avg_macro = np.mean([r['macro'] for r in rows])
    avg_up    = np.mean([r['up']    for r in rows])
    avg_dn    = np.mean([r['dn']    for r in rows])
    avg_flat  = np.mean([r['flat']  for r in rows])
    print(f"  {ps_name:<30} {avg_macro:>9.3f} {avg_up:>8.3f} {avg_dn:>8.3f} {avg_flat:>9.3f}")

# Short/Long split
print(f"\n  {'SHORT (h=1..5)':<30}  macro    up    dn  flat  ║  {'LONG (h=6..15)':<30}  macro    up    dn  flat")
print(f"  {'─'*30}  {'─'*5} {'─'*5} {'─'*5} {'─'*5}  ║  {'─'*30}  {'─'*5} {'─'*5} {'─'*5} {'─'*5}")
for ps_name in PARAM_SETS:
    if ps_name not in results:
        continue
    rows = agg(results[ps_name])
    s_rows = rows[:5];  l_rows = rows[5:]
    sm = np.mean([r['macro'] for r in s_rows]); su = np.mean([r['up'] for r in s_rows])
    sd = np.mean([r['dn']    for r in s_rows]); sf = np.mean([r['flat'] for r in s_rows])
    lm = np.mean([r['macro'] for r in l_rows]); lu = np.mean([r['up'] for r in l_rows])
    ld = np.mean([r['dn']    for r in l_rows]); lf = np.mean([r['flat'] for r in l_rows])
    print(f"  {ps_name:<30}  {sm:.3f} {su:.3f} {sd:.3f} {sf:.3f}  ║  {'':30}  {lm:.3f} {lu:.3f} {ld:.3f} {lf:.3f}")

# Per-horizon detail for each param set
for ps_name in PARAM_SETS:
    if ps_name not in results:
        continue
    rows = agg(results[ps_name])
    n_eval = len(results[ps_name][1]['preds'])
    print(f"\n\n  {ps_name} — Per-Horizon  (N={n_eval} prediction dates)")
    print(f"  {'h':>3}  {'macro_F1':>9}  {'up_F1':>7}  {'dn_F1':>7}  {'flat_F1':>8}")
    print(f"  {'─'*3}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*8}")
    for r in rows:
        print(f"  {r['h']:>3}  {r['macro']:>9.3f}  {r['up']:>7.3f}  {r['dn']:>7.3f}  {r['flat']:>8.3f}")
    avg_macro = np.mean([r['macro'] for r in rows])
    avg_up    = np.mean([r['up']    for r in rows])
    avg_dn    = np.mean([r['dn']    for r in rows])
    avg_flat  = np.mean([r['flat']  for r in rows])
    print(f"  {'─'*3}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*8}")
    print(f"  {'AVG':>3}  {avg_macro:>9.3f}  {avg_up:>7.3f}  {avg_dn:>7.3f}  {avg_flat:>8.3f}")

print(f"\n{'═'*80}")
print("  NOTE: Both models are evaluated on the same 100 dates (2025-09-16 → 2026-02-06).")
print("  These dates fall within the TRAINING period for shuffled models but are")
print("  the CHRONOLOGICAL TEST period for no-shuffle models (ref_noshuf, AAII).")
print("  Interpret ref_noshuf as approximately out-of-sample; AAII may see some")
print("  overlap with its training data depending on the split used at training time.")
print(f"{'═'*80}\n")
