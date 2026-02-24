"""
Integration smoke test: run NVDA reference training for target_size=1.

Verifies the full pipeline works end-to-end after bug fixes.
This test trains a real model (~30s) and checks output files and metrics.

Run with:
    python -m pytest test_smoke_nvda.py -v -s

Skip during fast unit runs:
    python -m pytest test_dte_dse.py test_fetch_earnings.py -v
"""

import os
import json
import pandas as pd
import pytest
import NVDA_param
import trendConfig
import trendAnalysisFromTodayNew

END_DATE = '2026-02-23'
SYMBOL = 'NVDA'
TARGET_SIZE = 1      # Only run 1-day model to keep the smoke test fast
MODEL_NAME = NVDA_param.reference['model_name']
SCALER_FILE = f"{SYMBOL}_{MODEL_NAME}_scaler.joblib"
MODEL_FILE = f"model/model_{SYMBOL}_{MODEL_NAME}_fixed_noTimesplit_{TARGET_SIZE}.pth"
TREND_JSON = f"{SYMBOL}_trend.json"


@pytest.fixture(scope='module')
def run_nvda_target1():
    """
    Load data and run analyze_trend for NVDA reference, target_size=1.
    Runs once for the whole module; all tests in this file share the result.
    """
    param = dict(NVDA_param.reference)
    param['end_date'] = END_DATE
    param['target_size'] = TARGET_SIZE
    param['threshold'] = 0.03  # first-5-day threshold

    trendAnalysisFromTodayNew.load_data_to_cache(trendConfig.config, param)

    incr_df = pd.DataFrame(columns=[
        'date', 'close',
        'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8',
        'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'comment'
    ])

    trendAnalysisFromTodayNew.analyze_trend(
        trendConfig.config, param, TARGET_SIZE, incr_df,
        False,   # turn_random_on
        True,    # use_cached_data
        False,   # use_timesplit
    )
    return param


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_pipeline_completes_without_exception(run_nvda_target1):
    """analyze_trend must finish without raising."""
    # If the fixture ran without raising, this passes.
    assert run_nvda_target1 is not None


def test_scaler_file_created(run_nvda_target1):
    """Scaler .joblib file must exist after training."""
    assert os.path.exists(SCALER_FILE), \
        f"Scaler file not found: {SCALER_FILE} — training may have crashed before saving"


def test_model_file_created(run_nvda_target1):
    """Model .pth file must exist after training."""
    assert os.path.exists(MODEL_FILE), \
        f"Model file not found: {MODEL_FILE} — training may have crashed before saving"


def test_trend_json_has_valid_metrics(run_nvda_target1):
    """
    NVDA_trend.json must contain a non-NaN entry for this run.
    Checks that the NaN-guard fix (#15) saved a real result.
    """
    assert os.path.exists(TREND_JSON), f"{TREND_JSON} not found"
    with open(TREND_JSON) as f:
        data = json.load(f)

    # Find the most recent entry matching this run's model_name and target_size
    matches = [
        r for r in data
        if isinstance(r, list) and len(r) >= 3
        and isinstance(r[1], dict)
        and r[1].get('model_name') == MODEL_NAME
        and r[1].get('target_size') == TARGET_SIZE
    ]
    assert matches, \
        f"No entry for model_name={MODEL_NAME}, target_size={TARGET_SIZE} in {TREND_JSON}"

    result = matches[-1][2]
    for key in ['F1 for Class 0 [no change]', 'F1 for Class 1 [Up]', 'F1 for Class 2 [Down]']:
        val = float(result[key])
        assert val == val, f"{key} is NaN — NaN guard (#15) may not have fired correctly"
        assert 0.0 <= val <= 1.0, f"{key}={val} is outside valid [0, 1] range"


def test_tmp_csv_updated(run_nvda_target1):
    """NVDA_TMP.csv must exist and contain the end_date row."""
    tmp_file = f"{SYMBOL}_TMP.csv"
    assert os.path.exists(tmp_file), f"{tmp_file} not found"
    df = pd.read_csv(tmp_file)
    assert END_DATE in df['date'].values, \
        f"end_date {END_DATE} not found in {tmp_file} — data fetch may have failed"


def test_dropna_removed_expected_rows(run_nvda_target1, capsys):
    """
    Smoke check: the dropna warning/OK line must have been printed.
    (Relies on capsys capturing stdout from the fixture run — see note.)

    NOTE: Since the fixture runs before capsys is active, this test just
    checks the TMP CSV row count is reasonable (>500 rows after dropping 1).
    """
    tmp_file = f"{SYMBOL}_TMP.csv"
    df = pd.read_csv(tmp_file)
    assert len(df) > 500, \
        f"Only {len(df)} rows in TMP CSV — unexpected mass dropna may have occurred"
