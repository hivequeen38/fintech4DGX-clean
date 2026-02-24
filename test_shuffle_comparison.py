#!/usr/bin/env /usr/bin/python3.10
"""
One-time integration test: shuffle_splits=False (new fix) vs shuffle_splits=True (old)
for NVDA reference config, date=2026-02-23, all 15 prediction horizons.

Captures per-horizon validation and test F1, then prints a side-by-side comparison.

Usage:
    /usr/bin/python3.10 test_shuffle_comparison.py
"""

import sys
import io
import re
import copy
import contextlib
import numpy as np
import pandas as pd

import trendConfig
import trendAnalysisFromTodayNew
import NVDA_param

# ── constants ────────────────────────────────────────────────────────────────

TEST_DATE = '2026-02-23'
HORIZONS  = list(range(1, 16))
INPUT_COLS = ['date', 'close',
              'p1','p2','p3','p4','p5','p6','p7','p8','p9',
              'p10','p11','p12','p13','p14','p15','comment']

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_metrics(output: str) -> dict:
    """Extract val/test F1 per class and average F1 from captured stdout."""
    m = {}

    # --- validation section (appears before "Start test") ---
    val_section = output.split('Start test')[0] if 'Start test' in output else output
    for cls in [0, 1, 2]:
        hit = re.search(rf'F1 for class {cls}: ([0-9.nan]+)', val_section)
        if hit:
            try:
                m[f'val_f1_cls{cls}'] = float(hit.group(1))
            except ValueError:
                m[f'val_f1_cls{cls}'] = float('nan')

    hit = re.search(r'==>Avergae F1: ([0-9.nan]+)', val_section)
    if hit:
        try:
            m['val_avg_f1'] = float(hit.group(1))
        except ValueError:
            m['val_avg_f1'] = float('nan')

    # --- test section ---
    test_section = output.split('Start test')[-1] if 'Start test' in output else ''
    for cls in [0, 1, 2]:
        hit = re.search(rf'F1 for class {cls}: ([0-9.nan]+)', test_section)
        if hit:
            try:
                m[f'test_f1_cls{cls}'] = float(hit.group(1))
            except ValueError:
                m[f'test_f1_cls{cls}'] = float('nan')

    return m


def run_all_horizons(param: dict, label: str) -> list:
    """Run analyze_trend for all 15 horizons; return list of per-horizon metric dicts."""
    incr_df = pd.DataFrame(columns=INPUT_COLS).reset_index(drop=True)
    results = []

    for h in HORIZONS:
        p = copy.deepcopy(param)
        p['target_size'] = h
        p['threshold']   = 0.03 if h <= 5 else 0.05
        p['comment']     = f'TEST({label}) h={h}'

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            trendAnalysisFromTodayNew.analyze_trend(
                trendConfig.config,
                p,
                h,          # current_day_offset
                incr_df,    # accumulator DataFrame
                False,      # turn_random_on
                True,       # use_cached_data
                p.get('use_time_split', False)
            )
        output = buf.getvalue()
        metrics = parse_metrics(output)
        metrics['horizon'] = h
        results.append(metrics)

        vf = metrics.get('val_avg_f1', float('nan'))
        tf = [metrics.get(f'test_f1_cls{c}', float('nan')) for c in range(3)]
        print(f"  h={h:2d}  val_avg_f1={vf:.3f}  "
              f"test_f1=[DN:{tf[0]:.2f} NEU:{tf[1]:.2f} UP:{tf[2]:.2f}]")

    return results


# ── main ─────────────────────────────────────────────────────────────────────

print(f"\n{'='*72}")
print(f"  SHUFFLE COMPARISON — NVDA reference — date={TEST_DATE}")
print(f"{'='*72}\n")

# Load data into cache once (shared by both runs)
base_param = copy.deepcopy(NVDA_param.reference)
base_param['end_date'] = TEST_DATE

print("Loading NVDA data into cache (once)...")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    trendAnalysisFromTodayNew.load_data_to_cache(trendConfig.config, base_param)
print("Done.\n")

# ── RUN 1: shuffle=False (new fix) ───────────────────────────────────────────
print("─── RUN 1: shuffle_splits=False (new/fixed) ───")
p_new = copy.deepcopy(base_param)
p_new['shuffle_splits'] = False
res_new = run_all_horizons(p_new, 'noShuffle')

# ── RUN 2: shuffle=True (old behaviour) ─────────────────────────────────────
print("\n─── RUN 2: shuffle_splits=True (old behaviour) ───")
p_old = copy.deepcopy(base_param)
p_old['shuffle_splits'] = True
res_old = run_all_horizons(p_old, 'shuffle')

# ── COMPARISON TABLE ─────────────────────────────────────────────────────────
df_new = pd.DataFrame(res_new).set_index('horizon')
df_old = pd.DataFrame(res_old).set_index('horizon')

print(f"\n{'='*72}")
print("  SIDE-BY-SIDE: Validation avg F1  |  Test macro F1  (DN=0, NEU=1, UP=2)")
print(f"{'='*72}")
hdr = (f"{'H':>3} | "
       f"{'noShuffle val':>13} {'shuffle val':>11} {'Δval':>6} | "
       f"{'noShuffle test':>14} {'shuffle test':>12} {'Δtest':>6}")
print(hdr)
print('-' * len(hdr))

val_deltas, test_deltas = [], []
for h in HORIZONS:
    n = df_new.loc[h]
    o = df_old.loc[h]

    nv = n.get('val_avg_f1', float('nan'))
    ov = o.get('val_avg_f1', float('nan'))
    dv = nv - ov

    nt = np.nanmean([n.get(f'test_f1_cls{c}', float('nan')) for c in range(3)])
    ot = np.nanmean([o.get(f'test_f1_cls{c}', float('nan')) for c in range(3)])
    dt = nt - ot

    val_deltas.append(dv)
    test_deltas.append(dt)

    av = '↑' if dv > 0.005 else ('↓' if dv < -0.005 else '~')
    at = '↑' if dt > 0.005 else ('↓' if dt < -0.005 else '~')

    print(f"{h:>3} | {nv:>13.3f} {ov:>11.3f} {dv:>+5.3f}{av} | "
          f"{nt:>14.3f} {ot:>12.3f} {dt:>+5.3f}{at}")

avg_dv = float(np.nanmean(val_deltas))
avg_dt = float(np.nanmean(test_deltas))
wins_v = sum(1 for d in val_deltas if d > 0.005)
wins_t = sum(1 for d in test_deltas if d > 0.005)

print('-' * len(hdr))
print(f"{'AVG':>3} | {'':>13} {'':>11} {avg_dv:>+5.3f}  | {'':>14} {'':>12} {avg_dt:>+5.3f}")
print(f"\n  noShuffle wins (val):  {wins_v}/15 horizons (Δ > 0.005)")
print(f"  noShuffle wins (test): {wins_t}/15 horizons (Δ > 0.005)")

# ── PREDICTION DIRECTION DIFF ────────────────────────────────────────────────
print(f"\n{'='*72}")
print("  PREDICTION DIRECTION for 2026-02-23 (p1..p15)")
print(f"{'='*72}")

pred_df = pd.read_csv('/workspace/NVDA_15d_from_today_predictions.csv')
pred_df_test = pred_df[pred_df['comment'].str.contains('TEST\\(', na=False, regex=True)]
pcols = [f'p{i}' for i in range(1, 16)]

for lbl, tag in [('noShuffle', 'noShuffle'), ('shuffle', 'shuffle')]:
    rows = pred_df_test[pred_df_test['comment'].str.contains(tag, na=False)]
    if rows.empty:
        print(f"  {lbl}: no prediction rows captured")
        continue
    # aggregate vote across horizons: take the last row added per horizon
    preds = []
    for h in HORIZONS:
        h_rows = rows[rows['comment'].str.contains(f'h={h}\\b', na=False, regex=True)]
        preds.append(h_rows.iloc[-1][f'p{h}'] if not h_rows.empty else '?')
    print(f"  {lbl:12s}: {' '.join(f'{p:>3}' for p in preds)}")

print(f"  {'horizon':12s}: {' '.join(f'{i:>3}' for i in range(1, 16))}")

print(f"\n  Positive Δ = noShuffle (new fix) is better.\n")
