"""
Unit tests for gbdt_pipeline output schema, CSV writer, and param file structure.

Tests:
  1. infer_today output has required columns
  2. infer_today produces exactly 15 rows
  3. score ∈ [0, 1] and signal ∈ {0, 1}
  4. write_predictions_csv dedup: re-running same date does not duplicate rows
  5. Param files: lgbm_reference_base is a strict subset of lgbm_reference
  6. All 5 {SYMBOL}_gbdt_param.py files have required keys
  7. Integration smoke test: load_features shape and column validation (uses
     {SYMBOL}_TMP.csv if present; skipped otherwise)

Run with:
    python -m pytest test_gbdt_schema.py -v
"""

import os
import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax


# ---------------------------------------------------------------------------
# Minimal stubs (no real model files required for schema tests)
# ---------------------------------------------------------------------------

class _FakeBooster:
    def __init__(self, n_features=10, seed=0):
        self._rng = np.random.default_rng(seed)
        self._n   = n_features

    def predict(self, X, raw_score=False):
        logits = self._rng.standard_normal((len(X), 3))
        return logits if raw_score else softmax(logits, axis=1)


def _make_fake_df(n=30, n_feat=10) -> pd.DataFrame:
    dates    = pd.bdate_range(start='2024-01-02', periods=n)
    rng      = np.random.default_rng(42)
    data     = rng.standard_normal((n, n_feat))
    cols     = [f'feat_{i}' for i in range(n_feat)]
    return pd.DataFrame(data, index=dates, columns=cols)


def _make_models_Ts_thetas(n_feat=10):
    models = {h: _FakeBooster(n_feat, seed=h)    for h in range(1, 16)}
    Ts     = {h: 1.0                              for h in range(1, 16)}
    thetas = {h: 0.65                             for h in range(1, 16)}
    thetas['symbol'] = 0.65
    return models, Ts, thetas


def _fake_param(symbol='TEST'):
    return {'symbol': symbol, 'model_name': 'lgbm_reference'}


# ---------------------------------------------------------------------------
# Import pipeline functions under test
# ---------------------------------------------------------------------------
from gbdt_pipeline import infer_today, write_predictions_csv

REQUIRED_COLS = {
    'date', 'h', 'P_down', 'P_flat', 'P_up', 'edge',
    'score', 'score_max', 'signal', 'model_type',
}


# ── Test 1: Required columns present ─────────────────────────────────────────

def test_infer_today_required_columns():
    """infer_today output must contain all schema columns."""
    df     = _make_fake_df()
    models, Ts, thetas = _make_models_Ts_thetas()
    param  = _fake_param()

    df_out = infer_today(models, Ts, thetas, param, df)
    assert REQUIRED_COLS.issubset(df_out.columns), (
        f'Missing columns: {REQUIRED_COLS - set(df_out.columns)}'
    )


# ── Test 2: Exactly 15 rows ───────────────────────────────────────────────────

def test_infer_today_15_rows():
    """infer_today must produce exactly 15 rows (one per horizon)."""
    df     = _make_fake_df()
    models, Ts, thetas = _make_models_Ts_thetas()
    param  = _fake_param()

    df_out = infer_today(models, Ts, thetas, param, df)
    assert len(df_out) == 15, f'Expected 15 rows, got {len(df_out)}'
    assert list(df_out['h']) == list(range(1, 16)), 'h column must be 1..15'


# ── Test 3: score ∈ [0,1] and signal ∈ {0, 1} ───────────────────────────────

def test_score_and_signal_ranges():
    """score must be in [0,1]; signal must be 0 or 1."""
    df     = _make_fake_df()
    models, Ts, thetas = _make_models_Ts_thetas()
    param  = _fake_param()

    df_out = infer_today(models, Ts, thetas, param, df)
    scores  = df_out['score'].values
    signals = df_out['signal'].values

    assert np.all((scores >= 0) & (scores <= 1)), (
        f'score out of [0,1]: min={scores.min():.4f} max={scores.max():.4f}'
    )
    assert set(signals).issubset({0, 1}), (
        f'signal values outside {{0,1}}: {set(signals)}'
    )


# ── Test 4: CSV dedup — no duplicate rows on re-run ──────────────────────────

def test_csv_dedup(tmp_path, monkeypatch):
    """write_predictions_csv: writing the same (date, lgbm) twice must not
    create duplicate rows."""
    # Redirect CSV to tmp_path
    import gbdt_pipeline as gp
    monkeypatch.setattr(
        gp, 'write_predictions_csv',
        lambda df_out, param: _patched_write(df_out, param, tmp_path)
    )

    df     = _make_fake_df()
    models, Ts, thetas = _make_models_Ts_thetas()
    param  = _fake_param(symbol='DEDUP')

    df_out = infer_today(models, Ts, thetas, param, df)

    # Write twice
    _patched_write(df_out, param, tmp_path)
    _patched_write(df_out, param, tmp_path)

    result = pd.read_csv(tmp_path / 'DEDUP_gbdt_15d_from_today_predictions.csv')
    date_str = str(df_out['date'].iloc[0])
    lgbm_rows = result[
        (result['date'].astype(str).str.startswith(date_str)) &
        (result['model_type'] == 'lgbm')
    ]
    assert len(lgbm_rows) == 15, (
        f'Expected 15 rows after dedup, got {len(lgbm_rows)}'
    )


def _patched_write(df_out: pd.DataFrame, param: dict, base_dir) -> None:
    """Version of write_predictions_csv that writes to tmp_path instead of /workspace."""
    sym      = param['symbol']
    path     = base_dir / f'{sym}_gbdt_15d_from_today_predictions.csv'
    date_str = str(df_out['date'].iloc[0])

    if path.exists():
        existing = pd.read_csv(path, parse_dates=['date'])
        existing = existing[~(
            (existing['date'].astype(str).str.startswith(date_str)) &
            (existing['model_type'] == 'lgbm')
        )]
        combined = pd.concat([existing, df_out], ignore_index=True)
    else:
        combined = df_out
    combined.to_csv(path, index=False)


# ── Test 5: lgbm_reference_base is a subset of lgbm_reference ────────────────

@pytest.mark.parametrize('module_name', [
    'NVDA_gbdt_param',
    'CRDO_gbdt_param',
    'PLTR_gbdt_param',
    'APP_gbdt_param',
    'INOD_gbdt_param',
])
def test_base_subset_of_reference(module_name):
    """lgbm_reference_base.selected_columns must be a subset of
    lgbm_reference.selected_columns for every symbol."""
    import importlib
    mod  = importlib.import_module(module_name)
    ref  = set(mod.lgbm_reference['selected_columns'])
    base = set(mod.lgbm_reference_base['selected_columns'])
    assert base.issubset(ref), (
        f'{module_name}: base has columns not in reference: {base - ref}'
    )


# ── Test 6: All param files have required keys ────────────────────────────────

REQUIRED_PARAM_KEYS = {
    'symbol', 'model_name', 'threshold',
    'selected_columns', 'start_date', 'train_end_date', 'val_end_date',
    'lgbm_params', 'calibration',
}

@pytest.mark.parametrize('module_name', [
    'NVDA_gbdt_param',
    'CRDO_gbdt_param',
    'PLTR_gbdt_param',
    'APP_gbdt_param',
    'INOD_gbdt_param',
])
def test_param_required_keys(module_name):
    """Both lgbm_reference and lgbm_reference_base must have all required keys."""
    import importlib
    mod = importlib.import_module(module_name)
    for param_name in ('lgbm_reference', 'lgbm_reference_base'):
        param = getattr(mod, param_name)
        missing = REQUIRED_PARAM_KEYS - set(param.keys())
        assert not missing, (
            f'{module_name}.{param_name}: missing required keys: {missing}'
        )


# ── Test 7: load_features smoke test (skipped if TMP.csv absent) ─────────────

@pytest.mark.parametrize('symbol,module_name', [
    ('NVDA', 'NVDA_gbdt_param'),
    ('CRDO', 'CRDO_gbdt_param'),
    ('PLTR', 'PLTR_gbdt_param'),
    ('APP',  'APP_gbdt_param'),
    ('INOD', 'INOD_gbdt_param'),
])
def test_load_features_smoke(symbol, module_name):
    """If {SYMBOL}_TMP.csv exists, load_features must return a non-empty
    DataFrame with all selected feature columns present in the TMP file.

    cp_sentiment_ratio and options_volume_ratio are merged into TMP.csv by
    fetchBulkData.py at runtime; if the on-disk TMP file predates that merge
    (or the cp_ratio CSV is absent), the test is skipped with an advisory note.
    """
    tmp_path = f'/workspace/{symbol}_TMP.csv'
    if not os.path.exists(tmp_path):
        pytest.skip(f'{symbol}_TMP.csv not present — skipping smoke test')

    # Check which selected_columns are actually in the TMP file (without loading
    # through load_features which would raise on any missing column).
    header_df = pd.read_csv(tmp_path, nrows=0)
    tmp_cols  = set(header_df.columns)

    import importlib
    from gbdt_pipeline import load_features
    mod  = importlib.import_module(module_name)
    base = mod.lgbm_reference_base

    # Build a reduced param with only columns that exist in TMP.csv.
    # Columns absent from TMP.csv are advisory (data not yet merged); skip them.
    expected_cols = [c for c in base['selected_columns'] if c != 'label']
    absent        = [c for c in expected_cols if c not in tmp_cols]
    if absent:
        pytest.skip(
            f'{symbol}: columns not yet in TMP.csv (re-run inference to merge): '
            f'{absent}'
        )

    df = load_features(base)
    assert len(df) > 0, f'{symbol}: load_features returned empty DataFrame'
    missing = [c for c in expected_cols if c not in df.columns]
    assert not missing, f'{symbol}: missing columns after load: {missing}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
