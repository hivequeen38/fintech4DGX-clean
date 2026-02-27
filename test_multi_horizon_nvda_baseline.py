"""
test_multi_horizon_nvda_baseline.py

Slow integration test: trains MultiHorizonTransformer on NVDA, evaluates per-class
F1 per horizon bucket, and compares against existing baseline profiles.

Tagged @pytest.mark.slow — excluded from the fast unit-test suite.
Run separately:
    pytest /workspace/test_multi_horizon_nvda_baseline.py -v -m slow -s

ISOLATION GUARANTEE
-------------------
All model + scaler artifacts are written to pytest's tmp_path (auto-cleaned).
Production models in /workspace/model/ are never touched.
Production scalers are loaded READ-ONLY for baseline evaluation.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import precision_recall_fscore_support

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_latest_jsonl_f1(jsonl_path: str, model_name: str):
    """Return the most recent per-horizon test F1 dict for a given model_name.

    Returns dict keyed by target_size (1..15) with sub-dict:
        {'f1_flat': float, 'f1_up': float, 'f1_down': float}
    Returns {} if jsonl_path not found or no matching entries.
    """
    if not os.path.exists(jsonl_path):
        return {}
    horizon_f1 = {}
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                run_time, params, results = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if params.get('model_name') != model_name:
                continue
            h = params.get('target_size')
            if h is None:
                continue
            horizon_f1[h] = {
                'f1_flat': results.get('Test F1 for class 0: '),
                'f1_up':   results.get('Test F1 for class 1: '),
                'f1_down': results.get('Test F1 for class 2: '),
            }
    return horizon_f1


def _bucket_macro_f1(horizon_f1: dict, h_range: range) -> float:
    """Compute mean macro-F1 across a range of horizons."""
    macros = []
    for h in h_range:
        row = horizon_f1.get(h)
        if row is None:
            continue
        vals = [v for v in (row['f1_flat'], row['f1_up'], row['f1_down']) if v is not None]
        if vals:
            macros.append(np.mean(vals))
    return float(np.mean(macros)) if macros else float('nan')


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_nvda_multi_horizon_vs_baseline(tmp_path):
    """
    Train a MultiHorizonTransformer on NVDA (cached data) and compare per-class
    test F1 against existing single-horizon baselines (ref, ref_noshuf, AAII_option_vol_ratio).

    Isolation: all model + scaler files go to tmp_path — /workspace/model/ is never written.
    """
    import sys
    sys.path.insert(0, '/workspace')

    from trendAnalysisFromTodayNew import (
        analyze_trend_multi_horizon,
        build_multi_horizon_labels,
    )
    from NVDA_param import AAII_option_vol_ratio as _aaii_param

    JSONL_PATH  = '/workspace/NVDA_trend.jsonl'
    CACHE_PATH  = '/workspace/NVDA_TMP.csv'
    START_DATE  = '2021-03-01'
    NUM_HORIZONS = 15

    # ------------------------------------------------------------------
    # 1. Build the multi-horizon param dict (derived from AAII_option_vol_ratio)
    # ------------------------------------------------------------------
    mh_param = {
        **_aaii_param,
        'model_type':    'multi_horizon_transformer',
        'model_name':    'NVDA_mh_test',
        'shuffle_splits': False,
        'num_horizons':   NUM_HORIZONS,
        # Reduce epochs for test speed; production would use 200
        'num_epochs':    50,
    }

    assert not mh_param.get('shuffle_splits', False), \
        "shuffle_splits must be False for multi-horizon test"

    # ------------------------------------------------------------------
    # 2. Confirm /workspace/model/ will NOT be written
    # ------------------------------------------------------------------
    prod_model_dir = os.path.abspath('/workspace/model/')
    test_save_dir  = str(tmp_path)
    assert not os.path.abspath(test_save_dir).startswith(prod_model_dir), \
        "tmp_path must not be inside production model dir"

    # ------------------------------------------------------------------
    # 3. Verify cached data exists
    # ------------------------------------------------------------------
    assert os.path.exists(CACHE_PATH), \
        f"Cached NVDA data not found: {CACHE_PATH}. Run a training session first."

    # ------------------------------------------------------------------
    # 4. Train multi-horizon model (all artifacts go to tmp_path)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Training MultiHorizonTransformer on NVDA (test run)...")
    print(f"  save_dir = {test_save_dir}")
    print(f"  num_epochs = {mh_param['num_epochs']}  (reduced for test speed)")
    print(f"{'='*60}")

    model, test_preds, test_labels, model_path, scaler_path = analyze_trend_multi_horizon(
        config={},
        param=mh_param,
        current_day_offset='mh_test',
        incr_df=pd.DataFrame(),
        turn_random_on=False,
        use_cached_data=True,
        save_dir=test_save_dir,
    )

    # ------------------------------------------------------------------
    # 5. Verify isolation: model was saved to tmp_path, not production dir
    # ------------------------------------------------------------------
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    assert not os.path.abspath(model_path).startswith(prod_model_dir), \
        f"Model was written to production dir: {model_path}"
    assert not os.path.abspath(scaler_path).startswith(prod_model_dir), \
        f"Scaler was written to production dir: {scaler_path}"
    print(f"\n[OK] Model isolation confirmed: {model_path}")

    # ------------------------------------------------------------------
    # 6. Compute multi-horizon per-class F1 per horizon bucket
    # ------------------------------------------------------------------
    assert test_preds.shape == test_labels.shape, \
        f"Shape mismatch: preds={test_preds.shape} labels={test_labels.shape}"
    assert not np.isnan(test_preds).any(), "test_preds contain NaN"

    # Check model doesn't collapse to all-flat (class 0)
    unique_preds = np.unique(test_preds)
    assert len(unique_preds) > 1, \
        f"Model collapsed: only predicts class {unique_preds}. Check class weights."

    mh_bucket_results = {}   # {bucket_name: {f1_flat, f1_up, f1_down, macro}}
    mh_horizon_results = []  # one dict per horizon

    for h in range(NUM_HORIZONS):
        p, r, f1_h, _ = precision_recall_fscore_support(
            test_labels[:, h], test_preds[:, h],
            average=None, labels=[0, 1, 2], zero_division=0
        )
        mh_horizon_results.append({
            'horizon': h + 1,
            'f1_flat': float(f1_h[0]),
            'f1_up':   float(f1_h[1]),
            'f1_down': float(f1_h[2]),
            'macro':   float(np.mean(f1_h)),
        })

    for bucket_name, h_range in [('h1_5', range(5)), ('h6_15', range(5, 15))]:
        rows = [mh_horizon_results[h] for h in h_range]
        mh_bucket_results[bucket_name] = {
            'f1_flat': np.mean([r['f1_flat'] for r in rows]),
            'f1_up':   np.mean([r['f1_up']   for r in rows]),
            'f1_down': np.mean([r['f1_down']  for r in rows]),
            'macro':   np.mean([r['macro']    for r in rows]),
        }

    # ------------------------------------------------------------------
    # 7. Load baseline F1 from NVDA_trend.jsonl (read-only)
    # ------------------------------------------------------------------
    baseline_profiles = ['ref', 'ref_noshuf', 'AAII_option_vol_ratio']
    baseline_bucket   = {}   # {profile: {bucket: macro_f1}}

    for profile in baseline_profiles:
        horizon_f1 = _load_latest_jsonl_f1(JSONL_PATH, profile)
        if not horizon_f1:
            warnings.warn(f"No JSONL entries found for profile '{profile}' — skipping baseline comparison")
            continue
        baseline_bucket[profile] = {
            'h1_5':  _bucket_macro_f1(horizon_f1, range(1, 6)),
            'h6_15': _bucket_macro_f1(horizon_f1, range(6, 16)),
        }

    # ------------------------------------------------------------------
    # 8. Print comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("NVDA Multi-Horizon vs Baseline Comparison")
    print(f"{'='*70}")
    print(f"{'Model':<28} {'Bucket':<10} {'F1_flat':>8} {'F1_up':>8} {'F1_down':>8} {'macro':>8}")
    print('-' * 70)

    rows_out = []

    for bucket_name in ('h1_5', 'h6_15'):
        res = mh_bucket_results[bucket_name]
        label = 'h=1-5 (±3%)' if bucket_name == 'h1_5' else 'h=6-15 (±5%)'
        print(f"  {'multi_horizon':<26} {label:<10} {res['f1_flat']:>8.3f} "
              f"{res['f1_up']:>8.3f} {res['f1_down']:>8.3f} {res['macro']:>8.3f}")
        rows_out.append({'model': 'multi_horizon', 'bucket': label,
                         'F1_flat': res['f1_flat'], 'F1_up': res['f1_up'],
                         'F1_down': res['f1_down'], 'macro_F1': res['macro']})

    print()
    for profile, buckets in baseline_bucket.items():
        for bucket_name in ('h1_5', 'h6_15'):
            macro = buckets.get(bucket_name, float('nan'))
            label = 'h=1-5 (±3%)' if bucket_name == 'h1_5' else 'h=6-15 (±5%)'
            suffix = ' (shuffled—inflated)' if profile == 'ref' else ''
            print(f"  {profile + suffix:<26} {label:<10} {'n/a':>8} {'n/a':>8} {'n/a':>8} {macro:>8.3f}")
            rows_out.append({'model': profile + suffix, 'bucket': label,
                             'F1_flat': None, 'F1_up': None, 'F1_down': None,
                             'macro_F1': macro})

    print(f"{'='*70}")
    print("Note: baseline macro_F1 is averaged across all horizons in the bucket")
    print("      from the most recent logged training run per model.")
    print("      ref (shuffled) macro_F1 is inflated due to look-ahead bias.")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # 9. Save comparison CSV (written to CWD, not tmp_path — kept as artifact)
    # ------------------------------------------------------------------
    csv_path = '/workspace/nvda_mh_baseline_comparison.csv'
    pd.DataFrame(rows_out).to_csv(csv_path, index=False)
    print(f"[OK] Comparison saved to {csv_path}")

    # ------------------------------------------------------------------
    # 10. Assertions
    # ------------------------------------------------------------------
    # Hard: model must not collapse (already checked above)
    # Soft: compare against ref_noshuf if available (warn, not fail, for first run)
    if 'ref_noshuf' in baseline_bucket:
        ref_noshuf_h1_5  = baseline_bucket['ref_noshuf']['h1_5']
        ref_noshuf_h6_15 = baseline_bucket['ref_noshuf']['h6_15']
        mh_h1_5  = mh_bucket_results['h1_5']['macro']
        mh_h6_15 = mh_bucket_results['h6_15']['macro']

        if not np.isnan(ref_noshuf_h1_5):
            if mh_h1_5 < ref_noshuf_h1_5:
                warnings.warn(
                    f"Multi-horizon macro-F1 h=1-5 ({mh_h1_5:.3f}) < ref_noshuf ({ref_noshuf_h1_5:.3f}). "
                    "See §13.5 diagnostic steps in multi_horizon_model_implementation_plan.md"
                )

        if not np.isnan(ref_noshuf_h6_15):
            if mh_h6_15 < ref_noshuf_h6_15:
                warnings.warn(
                    f"Multi-horizon macro-F1 h=6-15 ({mh_h6_15:.3f}) < ref_noshuf ({ref_noshuf_h6_15:.3f}). "
                    "See §13.5 diagnostic steps in multi_horizon_model_implementation_plan.md"
                )
    else:
        print("[INFO] ref_noshuf not found in JSONL — skipping F1 threshold comparison")

    print("[PASS] test_nvda_multi_horizon_vs_baseline completed successfully.")
