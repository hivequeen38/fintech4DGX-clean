#!/usr/bin/env /usr/bin/python3
"""
nvda_model_comparison.py

1. Trains GBDT (NVDA_gbdt_param.lgbm_reference) on the full data range
   (2021-03-01 → last avail in TMP.csv).
2. Loads manifest → per-horizon and per-class test metrics (2024-10-01 to present).
3. Evaluates Transformer param sets (reference, AAII_option_vol_ratio, mz_reference)
   on the SAME test period using NVDA_15d_from_today_predictions.csv + actual
   returns from NVDA_TMP.csv.
4. Prints a side-by-side comparison table.

Usage:
    /usr/bin/python3 /workspace/nvda_model_comparison.py
"""
import os, sys
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
GBDT_TEST_START = pd.Timestamp('2024-10-01')
GBDT_TEST_END   = pd.Timestamp('2026-03-02')   # last TMP date

THR_SHORT = 0.03   # h <= 5
THR_LONG  = 0.05   # h >= 6

LABEL_ENC = {'DN': 0, '__': 1, 'UP': 2}
LABEL_DEC = {0: 'DN', 1: '__', 2: 'UP'}

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Train GBDT
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Step 1: Training NVDA GBDT (lgbm_reference) ...")
print("=" * 70)

import NVDA_gbdt_param
import gbdt_pipeline

passed = gbdt_pipeline.train(NVDA_gbdt_param.lgbm_reference)
print(f"\n[GBDT] acceptance_passed = {passed}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load GBDT manifest metrics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 2: Loading GBDT manifest ...")
print("=" * 70)

manifest  = gbdt_pipeline.load_manifest(NVDA_gbdt_param.lgbm_reference)
eval_m    = manifest['eval_metrics']       # keyed by str(h)
trading   = manifest['trading_metrics']
baselines = {}   # not stored in manifest separately; we'll use eval always-up

# Build per-horizon GBDT rows
gbdt_metrics = []
for h in range(1, 16):
    m = eval_m[str(h)]
    gbdt_metrics.append({
        'h':        h,
        'n_test':   m.get('n_test', 0),
        'macro_f1': m['macro_f1'],
        'up_f1':    m['up_f1'],
        'down_f1':  m['down_f1'],
        'flat_f1':  m['flat_f1'],
        'dn_prec':  m.get('down_precision', np.nan),
        'brier_red': m.get('brier_reduction_pct', 0.0),
        'T_star':   m.get('T_star', np.nan),
        'theta_up': m.get('theta_up', np.nan),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Evaluate Transformer param sets on the SAME test period
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 3: Evaluating Transformer models on prediction history ...")
print("=" * 70)

# --- Load price series for actual return computation ---
tmp_df = pd.read_csv('/workspace/NVDA_TMP.csv')
tmp_df['date'] = pd.to_datetime(tmp_df['date'], format='mixed').dt.normalize()
tmp_df = tmp_df.set_index('date').sort_index()
price  = tmp_df['adjusted close']

def get_actual_label(date, h):
    """h-day actual label from date. Returns int (0/1/2) or None."""
    thr = THR_SHORT if h <= 5 else THR_LONG
    if date not in price.index:
        return None
    loc = price.index.get_loc(date)
    target_loc = loc + h
    if target_loc >= len(price):
        return None
    ret = price.iloc[target_loc] / price.iloc[loc] - 1
    if ret > thr:
        return 2   # UP
    elif ret < -thr:
        return 0   # DN
    else:
        return 1   # flat

# --- Load Transformer prediction CSV ---
pred_csv = pd.read_csv('/workspace/NVDA_15d_from_today_predictions.csv')
pred_csv['date'] = pd.to_datetime(pred_csv['date'], format='mixed').dt.normalize()

def classify_comment(c):
    c = str(c)
    if 'AAII_option_vol_ratio' in c:
        return 'AAII_option_vol_ratio'
    if 'ref_noshuf' in c:
        return 'ref_noshuf'
    if 'mz' in c.lower():
        return 'mz_reference'
    if '(tra)' in c or '(inf)' in c or 'reference' in c.lower() or 'Fixed' in c:
        return 'reference'
    return 'other'

pred_csv['param_set'] = pred_csv['comment'].apply(classify_comment)

# Filter to the GBDT test period (only evaluate on dates with sufficient lookforward)
# Leave 15 business days before end to ensure full lookforward
from pandas.tseries.offsets import BDay
cutoff = GBDT_TEST_END - 15 * BDay()

test_pred = pred_csv[
    (pred_csv['date'] >= GBDT_TEST_START) &
    (pred_csv['date'] <= cutoff)
].copy()

print(f"Transformer eval period: {GBDT_TEST_START.date()} → {cutoff.date()}")
print("Rows per param set in that window:")
print(test_pred.groupby('param_set').size().to_string())

# --- Compute per-class metrics for each Transformer param set ---
def compute_trans_metrics(rows_df, h):
    """Returns dict with n, macro_f1, up_f1, down_f1, flat_f1, dn_prec."""
    ph_col = f'p{h}'
    if ph_col not in rows_df.columns:
        return None

    y_true, y_pred = [], []
    for _, row in rows_df.iterrows():
        actual = get_actual_label(row['date'], h)
        if actual is None:
            continue
        pred_str = str(row[ph_col]) if pd.notna(row[ph_col]) else '__'
        if pred_str not in LABEL_ENC:
            pred_str = '__'
        y_true.append(actual)
        y_pred.append(LABEL_ENC[pred_str])

    if len(y_true) < 5:
        return None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels = [0, 1, 2]
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
    per_f1   = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    # Down precision: P(actual=DN | predicted=DN)
    dn_mask = y_pred == 0
    dn_prec = float((y_true[dn_mask] == 0).mean()) if dn_mask.sum() > 0 else np.nan

    return {
        'n':        len(y_true),
        'macro_f1': macro_f1,
        'up_f1':    float(per_f1[2]),
        'down_f1':  float(per_f1[0]),
        'flat_f1':  float(per_f1[1]),
        'dn_prec':  dn_prec,
    }

PARAM_SETS = ['reference', 'AAII_option_vol_ratio', 'ref_noshuf', 'mz_reference']
trans_metrics = {ps: [] for ps in PARAM_SETS}

for ps in PARAM_SETS:
    rows = test_pred[test_pred['param_set'] == ps]
    if len(rows) == 0:
        print(f"  [SKIP] {ps}: no rows in test window")
        continue
    for h in range(1, 16):
        m = compute_trans_metrics(rows, h)
        if m is None:
            m = {'h': h, 'n': 0, 'macro_f1': np.nan, 'up_f1': np.nan,
                 'down_f1': np.nan, 'flat_f1': np.nan, 'dn_prec': np.nan}
        m['h'] = h
        trans_metrics[ps].append(m)
    n_total = sum(m['n'] for m in trans_metrics[ps] if m.get('n', 0) > 0)
    print(f"  {ps}: evaluated, total prediction-date evals = {len(rows)}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Print comparison tables
# ─────────────────────────────────────────────────────────────────────────────

def fmt(val, pct=False, bold=False):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "  n/a "
    if pct:
        return f"{val*100:6.1f}%"
    return f"{val:6.3f}"

def agg_metrics(rows_list):
    """Average over h=1..15."""
    vals = {k: [] for k in ['macro_f1','up_f1','down_f1','flat_f1','dn_prec']}
    for r in rows_list:
        for k in vals:
            v = r.get(k, np.nan)
            if not np.isnan(float(v if v is not None else np.nan)):
                vals[k].append(float(v))
    return {k: float(np.mean(v)) if v else np.nan for k, v in vals.items()}

def agg_for_h_range(rows_list, h_start, h_end):
    """Average over specific horizon range."""
    vals = {k: [] for k in ['macro_f1','up_f1','down_f1','flat_f1','dn_prec']}
    for r in rows_list:
        if h_start <= r['h'] <= h_end:
            for k in vals:
                v = r.get(k, np.nan)
                try:
                    fv = float(v) if v is not None else np.nan
                    if not np.isnan(fv):
                        vals[k].append(fv)
                except:
                    pass
    return {k: float(np.mean(v)) if v else np.nan for k, v in vals.items()}


SEP   = "─" * 110
SEP2  = "═" * 110
HSEP  = "─" * 110

print("\n\n" + SEP2)
print("  NVDA MODEL COMPARISON  —  Test period: 2024-10-01 → 2026-02-09")
print("  GBDT: calendar-split test set (~350 days)  |  Transformer: from daily inference history")
print(SEP2)

# ─── Table 1: Summary across ALL horizons ────────────────────────────────────
models_summary = [
    ('GBDT lgbm_ref',          agg_metrics(gbdt_metrics)),
    ('Transformer ref',        agg_metrics(trans_metrics.get('reference', []))),
    ('Transformer AAII',       agg_metrics(trans_metrics.get('AAII_option_vol_ratio', []))),
    ('Transformer ref_noshuf', agg_metrics(trans_metrics.get('ref_noshuf', []))),
    ('Transformer MZ',         agg_metrics(trans_metrics.get('mz_reference', []))),
]

print(f"\n{'':30s} {'macro_F1':>9} {'up_F1':>8} {'dn_F1':>8} {'flat_F1':>9} {'dn_prec':>9}")
print(f"{'MODEL (avg h=1..15)':30s} {'─'*9} {'─'*8} {'─'*8} {'─'*9} {'─'*9}")
for name, m in models_summary:
    if not m or all(np.isnan(v) for v in m.values() if isinstance(v, float)):
        continue
    print(
        f"{name:30s} "
        f"{fmt(m.get('macro_f1')):>9} "
        f"{fmt(m.get('up_f1')):>8} "
        f"{fmt(m.get('down_f1')):>8} "
        f"{fmt(m.get('flat_f1')):>9} "
        f"{fmt(m.get('dn_prec')):>9}"
    )

# ─── Table 2: Short (h=1..5) vs Long (h=6..15) split ────────────────────────
print(f"\n\n  SHORT-HORIZON (h=1..5)                 LONG-HORIZON (h=6..15)")
print(f"  {'MODEL':28s}  {'macro':>6}  {'up':>6}  {'dn':>6}  {'flat':>6}  ║  {'macro':>6}  {'up':>6}  {'dn':>6}  {'flat':>6}")
print(f"  {'─'*28}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  ║  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")

models_list = [
    ('GBDT lgbm_ref',          gbdt_metrics),
    ('Transformer ref',        trans_metrics.get('reference', [])),
    ('Transformer AAII',       trans_metrics.get('AAII_option_vol_ratio', [])),
    ('Transformer ref_noshuf', trans_metrics.get('ref_noshuf', [])),
    ('Transformer MZ',         trans_metrics.get('mz_reference', [])),
]

for name, rows in models_list:
    if not rows:
        continue
    s = agg_for_h_range(rows, 1, 5)
    l = agg_for_h_range(rows, 6, 15)
    def f(m, k):
        v = m.get(k, np.nan)
        return f"{v:6.3f}" if not np.isnan(float(v)) else "  n/a"
    print(
        f"  {name:28s}  {f(s,'macro_f1'):>6}  {f(s,'up_f1'):>6}  {f(s,'down_f1'):>6}  {f(s,'flat_f1'):>6}  ║  "
        f"{f(l,'macro_f1'):>6}  {f(l,'up_f1'):>6}  {f(l,'down_f1'):>6}  {f(l,'flat_f1'):>6}"
    )

# ─── Table 3: Per-horizon detail — GBDT ──────────────────────────────────────
print(f"\n\n  GBDT lgbm_reference — Per-Horizon Test Metrics")
print(f"  {'h':>3}  {'N_test':>7}  {'macro_F1':>9}  {'up_F1':>7}  {'dn_F1':>7}  {'flat_F1':>8}  {'dn_prec':>8}  {'brier_red%':>11}  {'T*':>6}  {'θ_up':>6}")
print(f"  {'─'*3}  {'─'*7}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*11}  {'─'*6}  {'─'*6}")
for r in gbdt_metrics:
    dp  = f"{r['dn_prec']:.3f}"  if not np.isnan(r['dn_prec'])   else "   n/a"
    T   = f"{r['T_star']:.3f}"   if not np.isnan(r['T_star'])     else "  n/a"
    th  = f"{r['theta_up']:.3f}" if not np.isnan(r['theta_up'])   else "  n/a"
    br  = f"{r['brier_red']:+.1f}%"
    print(
        f"  {r['h']:>3}  {r['n_test']:>7}  {r['macro_f1']:>9.3f}  {r['up_f1']:>7.3f}  "
        f"{r['down_f1']:>7.3f}  {r['flat_f1']:>8.3f}  {dp:>8}  {br:>11}  {T:>6}  {th:>6}"
    )
avg_m = np.mean([r['macro_f1'] for r in gbdt_metrics])
avg_u = np.mean([r['up_f1']    for r in gbdt_metrics])
avg_d = np.mean([r['down_f1']  for r in gbdt_metrics])
avg_f = np.mean([r['flat_f1']  for r in gbdt_metrics])
print(f"  {'─'*3}  {'─'*7}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*11}")
print(f"  {'AVG':>3}  {'':>7}  {avg_m:>9.3f}  {avg_u:>7.3f}  {avg_d:>7.3f}  {avg_f:>8.3f}")

# ─── Table 4: Per-horizon detail — each Transformer param set ────────────────
for pset_name, pset_rows in [
    ('reference',          trans_metrics.get('reference', [])),
    ('AAII_option_vol_ratio', trans_metrics.get('AAII_option_vol_ratio', [])),
    ('mz_reference',       trans_metrics.get('mz_reference', [])),
]:
    if not pset_rows:
        continue
    n_pred_rows = len(test_pred[test_pred['param_set'] == pset_name])
    print(f"\n\n  Transformer {pset_name} — Per-Horizon  (N prediction-dates in window = {n_pred_rows})")
    print(f"  {'h':>3}  {'N_evals':>7}  {'macro_F1':>9}  {'up_F1':>7}  {'dn_F1':>7}  {'flat_F1':>8}  {'dn_prec':>8}")
    print(f"  {'─'*3}  {'─'*7}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}")
    avgs = {'macro_f1':[], 'up_f1':[], 'down_f1':[], 'flat_f1':[]}
    for r in pset_rows:
        if r.get('n', 0) == 0 or np.isnan(r.get('macro_f1', np.nan)):
            print(f"  {r['h']:>3}  {'n/a':>7}")
            continue
        dp = f"{r['dn_prec']:.3f}" if not np.isnan(r.get('dn_prec', np.nan)) else "   n/a"
        print(
            f"  {r['h']:>3}  {r['n']:>7}  {r['macro_f1']:>9.3f}  {r['up_f1']:>7.3f}  "
            f"{r['down_f1']:>7.3f}  {r['flat_f1']:>8.3f}  {dp:>8}"
        )
        for k in avgs:
            v = r.get(k, np.nan)
            if not np.isnan(float(v)):
                avgs[k].append(float(v))
    print(f"  {'─'*3}  {'─'*7}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*8}")
    am = np.mean(avgs['macro_f1']) if avgs['macro_f1'] else np.nan
    au = np.mean(avgs['up_f1'])    if avgs['up_f1']    else np.nan
    ad = np.mean(avgs['down_f1'])  if avgs['down_f1']  else np.nan
    af = np.mean(avgs['flat_f1'])  if avgs['flat_f1']  else np.nan
    print(f"  {'AVG':>3}  {'':>7}  {am:>9.3f}  {au:>7.3f}  {ad:>7.3f}  {af:>8.3f}")

# ─── Table 5: Trading metrics ─────────────────────────────────────────────────
print(f"\n\n  GBDT Trading Metrics (test set, h=10 focus)")
print(f"  {'─'*60}")
for k, v in trading.items():
    if v is not None and not (isinstance(v, float) and np.isnan(v)):
        print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")

print(f"\n  Acceptance: {'PASSED ✓' if passed else 'FAILED ✗'}  (path={manifest.get('acceptance_path','?')})")
print(f"\n  always_up_p1 (baseline P@1): {manifest.get('always_up_p1', 'n/a')}")
print("\n" + SEP2)
print("  NOTE: GBDT test period is a PROPER hold-out (calendar split 2024-10-01 → present).")
print("  Transformer metrics are computed from daily inference history; no shuffle/time-split")
print("  adjustment has been applied. Metric comparability is approximate.")
print(SEP2 + "\n")
