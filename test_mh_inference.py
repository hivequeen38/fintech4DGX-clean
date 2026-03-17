"""
test_mh_inference.py

Unit tests for make_inference_multi_horizon() and the trans_mz branch of
mainDeltafromToday.inference() / processDeltaFromTodayResults().

All tests are fully in-memory and isolated to tmp_path — no production model
files or scalers are touched, no network calls are made.
"""
import os

import numpy as np
import pandas as pd
import pytest
import torch
from joblib import dump
from sklearn.preprocessing import MinMaxScaler

from trendAnalysisFromTodayNew import (
    MultiHorizonTransformer,
    build_model,
    make_inference_multi_horizon,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

_FEATURE_COLS = ['f1', 'f2', 'f3', 'f4', 'f5', 'label']
_INPUT_DIM    = 5   # features excluding 'label'
_NUM_HORIZONS = 15
_SYMBOL       = 'TEST'
_MODEL_NAME   = 'mh_test'

_BASE_PARAM = {
    'symbol':         _SYMBOL,
    'model_name':     _MODEL_NAME,
    'model_type':     'multi_horizon_transformer',
    'num_horizons':   _NUM_HORIZONS,
    'headcount':      4,
    'num_layers':     2,
    'dropout_rate':   0.1,
    'embedded_dim':   16,
    'selected_columns': _FEATURE_COLS,
}


def _make_tmp_csv(tmp_path: str, n_rows: int = 50) -> str:
    """Write a minimal TMP.csv with random feature data."""
    np.random.seed(0)
    df = pd.DataFrame(np.random.randn(n_rows, _INPUT_DIM), columns=_FEATURE_COLS[:-1])
    df['label'] = 0
    df['date'] = pd.date_range('2024-01-01', periods=n_rows).strftime('%Y-%m-%d')
    df['adjusted close'] = 100.0
    path = os.path.join(tmp_path, f'{_SYMBOL}_TMP.csv')
    df.to_csv(path, index=False)
    return path


def _make_checkpoint(tmp_path: str) -> tuple[str, str]:
    """
    Train a tiny MultiHorizonTransformer for 1 epoch and save checkpoint + scaler.
    Returns (model_path, scaler_path).
    """
    np.random.seed(42)
    torch.manual_seed(42)

    n = 40
    features = np.random.randn(n, _INPUT_DIM).astype(np.float32)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    param = {**_BASE_PARAM}
    model = build_model(param, input_dim=_INPUT_DIM, num_classes=3)

    # 1-epoch forward pass just to populate state_dict
    x = torch.FloatTensor(features_scaled)
    with torch.no_grad():
        _ = model(x)   # (n, 15, 3)

    # Save scaler
    scaler_filename = f'{_SYMBOL}_{_MODEL_NAME}_mh_scaler.joblib'
    scaler_path = os.path.join(tmp_path, scaler_filename)
    dump(scaler, scaler_path)

    # Save checkpoint
    model_filename = f'model_{_SYMBOL}_{_MODEL_NAME}_mh_fixed_noTimesplit.pth'
    model_path = os.path.join(tmp_path, model_filename)
    torch.save({
        'state_dict': model.state_dict(),
        'config':     param,
        'calib_temp': 1.0,
        'train_date': '2026-03-09 18:00:00',
    }, model_path)

    return model_path, scaler_path


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMakeInferenceMultiHorizon:
    """Unit tests for make_inference_multi_horizon()."""

    def test_returns_15_labels(self, tmp_path):
        """Function returns exactly 15 label strings."""
        _make_tmp_csv(str(tmp_path))
        _make_checkpoint(str(tmp_path))

        param = {**_BASE_PARAM}
        incr_df = pd.DataFrame(columns=['p' + str(i) for i in range(1, 16)])

        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            labels = make_inference_multi_horizon(
                config={}, param=param, incr_df=incr_df,
                turn_random_on=False, use_cached_data=True,
                save_dir=str(tmp_path),
            )
        finally:
            os.chdir(old_cwd)

        assert len(labels) == _NUM_HORIZONS
        assert all(l in ('__', 'UP', 'DN') for l in labels), \
            f"Unexpected label values: {labels}"

    def test_incr_df_populated(self, tmp_path):
        """incr_df columns p1..p15 are filled with valid labels."""
        _make_tmp_csv(str(tmp_path))
        _make_checkpoint(str(tmp_path))

        param = {**_BASE_PARAM}
        incr_df = pd.DataFrame(columns=['p' + str(i) for i in range(1, 16)])

        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            make_inference_multi_horizon(
                config={}, param=param, incr_df=incr_df,
                turn_random_on=False, use_cached_data=True,
                save_dir=str(tmp_path),
            )
        finally:
            os.chdir(old_cwd)

        for i in range(1, 16):
            col = f'p{i}'
            assert col in incr_df.columns, f"Missing column {col}"
            assert incr_df.iloc[0][col] in ('__', 'UP', 'DN'), \
                f"Invalid value in {col}: {incr_df.iloc[0][col]}"

    def test_deterministic_with_fixed_seed(self, tmp_path):
        """Two runs with turn_random_on=False produce identical predictions."""
        _make_tmp_csv(str(tmp_path))
        _make_checkpoint(str(tmp_path))

        param = {**_BASE_PARAM}

        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            incr1 = pd.DataFrame(columns=['p' + str(i) for i in range(1, 16)])
            labels1 = make_inference_multi_horizon(
                config={}, param=param, incr_df=incr1,
                turn_random_on=False, use_cached_data=True,
                save_dir=str(tmp_path),
            )
            incr2 = pd.DataFrame(columns=['p' + str(i) for i in range(1, 16)])
            labels2 = make_inference_multi_horizon(
                config={}, param=param, incr_df=incr2,
                turn_random_on=False, use_cached_data=True,
                save_dir=str(tmp_path),
            )
        finally:
            os.chdir(old_cwd)

        assert labels1 == labels2, "Fixed-seed runs produced different predictions"

    def test_missing_scaler_raises(self, tmp_path):
        """FileNotFoundError raised when scaler is absent."""
        _make_tmp_csv(str(tmp_path))
        model_path, scaler_path = _make_checkpoint(str(tmp_path))
        os.remove(scaler_path)

        param = {**_BASE_PARAM}
        incr_df = pd.DataFrame()

        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            with pytest.raises(FileNotFoundError, match="MH scaler not found"):
                make_inference_multi_horizon(
                    config={}, param=param, incr_df=incr_df,
                    turn_random_on=False, use_cached_data=True,
                    save_dir=str(tmp_path),
                )
        finally:
            os.chdir(old_cwd)

    def test_missing_model_raises(self, tmp_path):
        """FileNotFoundError raised when model checkpoint is absent."""
        _make_tmp_csv(str(tmp_path))
        model_path, _ = _make_checkpoint(str(tmp_path))
        os.remove(model_path)

        param = {**_BASE_PARAM}
        incr_df = pd.DataFrame()

        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            with pytest.raises(FileNotFoundError, match="MH model checkpoint not found"):
                make_inference_multi_horizon(
                    config={}, param=param, incr_df=incr_df,
                    turn_random_on=False, use_cached_data=True,
                    save_dir=str(tmp_path),
                )
        finally:
            os.chdir(old_cwd)

    def test_missing_feature_raises(self, tmp_path):
        """ValueError raised when selected_columns references a column absent from TMP."""
        _make_tmp_csv(str(tmp_path))
        _make_checkpoint(str(tmp_path))

        bad_param = {**_BASE_PARAM, 'selected_columns': ['nonexistent_col', 'label']}
        incr_df = pd.DataFrame()

        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            with pytest.raises(ValueError, match="features missing from TMP"):
                make_inference_multi_horizon(
                    config={}, param=bad_param, incr_df=incr_df,
                    turn_random_on=False, use_cached_data=True,
                    save_dir=str(tmp_path),
                )
        finally:
            os.chdir(old_cwd)


class TestProcessDeltaFromTodayResultsModelType:
    """Verify processDeltaFromTodayResults writes the correct model_type column."""

    def test_default_model_type_is_transformer(self, tmp_path):
        """Default model_type='transformer' written to CSV."""
        from mainDeltafromToday import processDeltaFromTodayResults

        sym = 'TTEST'
        out_csv = os.path.join(str(tmp_path), f'{sym}_15d_from_today_predictions.csv')

        incr_df = pd.DataFrame({f'p{i}': ['__'] for i in range(1, 16)})

        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            processDeltaFromTodayResults(
                sym, incr_df, '2026-03-09', 100.0, 'test comment', 0.0,
                {'symbol': sym, 'selected_columns': [], 'model_name': 'ref'},
            )
        finally:
            os.chdir(old_cwd)

        df = pd.read_csv(out_csv)
        assert df.iloc[-1]['model_type'] == 'transformer'

    def test_trans_mz_model_type_written(self, tmp_path):
        """model_type='trans_mz' is correctly written when passed explicitly."""
        from mainDeltafromToday import processDeltaFromTodayResults

        sym = 'TTEST'
        out_csv = os.path.join(str(tmp_path), f'{sym}_15d_from_today_predictions.csv')

        incr_df = pd.DataFrame({f'p{i}': ['UP'] for i in range(1, 16)})

        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            processDeltaFromTodayResults(
                sym, incr_df, '2026-03-09', 200.0, 'mz comment', 0.0,
                {'symbol': sym, 'selected_columns': [], 'model_name': 'mh_mz'},
                model_type='trans_mz',
            )
        finally:
            os.chdir(old_cwd)

        df = pd.read_csv(out_csv)
        assert df.iloc[-1]['model_type'] == 'trans_mz'
        assert df.iloc[-1]['profile'] == 'mh_mz'
