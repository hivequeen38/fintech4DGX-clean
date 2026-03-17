"""
Unit tests for gbdt_pipeline.make_splits()

Tests:
  1. No date overlap between train / val / test splits
  2. Label NaN rows excluded from all splits
  3. Calendar boundaries respected (train_end_date, val_end_date)
  4. Consistent row counts between X and y arrays

Run with:
    python -m pytest test_gbdt_splits.py -v
"""

import numpy as np
import pandas as pd
import pytest

from gbdt_pipeline import generate_labels, make_splits


def _make_param(train_end='2023-06-30', val_end='2023-09-30'):
    return {
        'symbol':         'TEST',
        'model_name':     'test',
        'threshold':      0.05,
        'train_end_date': train_end,
        'val_end_date':   val_end,
    }


def _make_df(n=400) -> pd.DataFrame:
    """400 business days of price data starting 2022-01-03."""
    dates  = pd.bdate_range(start='2022-01-03', periods=n)
    prices = np.cumprod(1 + np.random.default_rng(42).normal(0, 0.01, n)) * 100
    return pd.DataFrame({'adjusted close': prices}, index=dates)


# ── Test 1: No date overlap ───────────────────────────────────────────────────

def test_no_date_overlap():
    """Train, val, and test date sets must be mutually exclusive for every h."""
    df     = _make_df()
    labels = generate_labels(df, _make_param())
    splits = make_splits(df, labels, _make_param())

    for h in range(1, 16):
        train_dates = set(splits[h]['train']['dates'])
        val_dates   = set(splits[h]['val']['dates'])
        test_dates  = set(splits[h]['test']['dates'])

        assert len(train_dates & val_dates)  == 0, f'h={h}: train/val overlap'
        assert len(val_dates  & test_dates)  == 0, f'h={h}: val/test overlap'
        assert len(train_dates & test_dates) == 0, f'h={h}: train/test overlap'


# ── Test 2: NaN labels excluded ──────────────────────────────────────────────

def test_nan_labels_excluded():
    """The last h rows (NaN label) must not appear in any split."""
    df     = _make_df()
    labels = generate_labels(df, _make_param())
    splits = make_splits(df, labels, _make_param())

    for h in range(1, 16):
        for split_name in ('train', 'val', 'test'):
            y = splits[h][split_name]['y']
            assert not np.any(np.isnan(y.astype(float))), (
                f'h={h} {split_name}: NaN found in y array'
            )


# ── Test 3: Calendar boundaries ──────────────────────────────────────────────

def test_calendar_boundaries():
    """All train dates must be <= train_end_date; all val dates must be in
    (train_end_date, val_end_date]; all test dates must be > val_end_date."""
    param  = _make_param(train_end='2023-06-30', val_end='2023-09-30')
    df     = _make_df()
    labels = generate_labels(df, param)
    splits = make_splits(df, labels, param)

    train_end = pd.Timestamp('2023-06-30')
    val_end   = pd.Timestamp('2023-09-30')

    # Use h=10 as representative
    h = 10
    train_dates = pd.DatetimeIndex(splits[h]['train']['dates'])
    val_dates   = pd.DatetimeIndex(splits[h]['val']['dates'])
    test_dates  = pd.DatetimeIndex(splits[h]['test']['dates'])

    if len(train_dates) > 0:
        assert train_dates.max() <= train_end, 'train spills past train_end_date'
    if len(val_dates) > 0:
        assert val_dates.min()   >  train_end, 'val starts before train_end_date'
        assert val_dates.max()   <= val_end,   'val spills past val_end_date'
    if len(test_dates) > 0:
        assert test_dates.min()  >  val_end,   'test starts before val_end_date'


# ── Test 4: Consistent X / y row counts ──────────────────────────────────────

def test_consistent_row_counts():
    """X.shape[0] must equal len(y) and len(dates) for every split and horizon."""
    df     = _make_df()
    labels = generate_labels(df, _make_param())
    splits = make_splits(df, labels, _make_param())

    for h in range(1, 16):
        for split_name in ('train', 'val', 'test'):
            s = splits[h][split_name]
            n_X     = s['X'].shape[0]
            n_y     = len(s['y'])
            n_dates = len(s['dates'])
            assert n_X == n_y, (
                f'h={h} {split_name}: X rows={n_X} != y rows={n_y}'
            )
            assert n_X == n_dates, (
                f'h={h} {split_name}: X rows={n_X} != dates rows={n_dates}'
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
