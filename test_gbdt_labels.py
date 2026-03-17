"""
Unit tests for gbdt_pipeline.generate_labels()

Tests:
  1. Up label when forward return > threshold
  2. Down label when forward return < -threshold
  3. Flat label when |forward return| < threshold
  4. Tail rows (no future close) produce NaN labels
  5. Threshold boundary: h<=5 uses THR_SHORT=0.03, h>=6 uses THR_LONG=0.05

Run with:
    python -m pytest test_gbdt_labels.py -v
"""

import numpy as np
import pandas as pd
import pytest

from gbdt_pipeline import generate_labels, THR_SHORT, THR_LONG, LABEL_MAP


def _make_df(prices: list) -> pd.DataFrame:
    """Minimal DataFrame with 'adjusted close' and a DatetimeIndex."""
    dates = pd.bdate_range(start='2023-01-02', periods=len(prices))
    return pd.DataFrame({'adjusted close': prices}, index=dates)


def _param(thr=0.05):
    return {'symbol': 'TEST', 'model_name': 'test', 'threshold': thr}


# ── Test 1: Up label ──────────────────────────────────────────────────────────

def test_label_up():
    """A return of +10% (> 5%) at h=10 should produce label=2 (up)."""
    # Create 30 rows; row 0 close=100, row 10 close=111 (+11%)
    prices = [100.0] + [100.0] * 9 + [111.0] + [111.0] * 19
    df = _make_df(prices)
    labels = generate_labels(df, _param())
    assert int(labels[10].iloc[0]) == LABEL_MAP['up'], (
        f"Expected up=2, got {labels[10].iloc[0]}"
    )


# ── Test 2: Down label ────────────────────────────────────────────────────────

def test_label_down():
    """A return of -10% at h=10 should produce label=0 (down)."""
    prices = [100.0] + [100.0] * 9 + [90.0] + [90.0] * 19
    df = _make_df(prices)
    labels = generate_labels(df, _param())
    assert int(labels[10].iloc[0]) == LABEL_MAP['down'], (
        f"Expected down=0, got {labels[10].iloc[0]}"
    )


# ── Test 3: Flat label ────────────────────────────────────────────────────────

def test_label_flat():
    """A return of +1% at h=10 (below 5% threshold) should produce label=1 (flat)."""
    prices = [100.0] + [100.0] * 9 + [101.0] + [101.0] * 19
    df = _make_df(prices)
    labels = generate_labels(df, _param())
    assert int(labels[10].iloc[0]) == LABEL_MAP['flat'], (
        f"Expected flat=1, got {labels[10].iloc[0]}"
    )


# ── Test 4: Tail rows produce NaN ────────────────────────────────────────────

def test_tail_rows_nan():
    """The last h rows must have NaN label (no future close available)."""
    prices = list(range(1, 31))  # 30 rows
    df = _make_df(prices)
    labels = generate_labels(df, _param())
    for h in range(1, 16):
        tail_labels = labels[h].iloc[-h:]
        assert tail_labels.isna().all(), (
            f"h={h}: expected NaN in last {h} rows, got {tail_labels.values}"
        )


# ── Test 5: Threshold boundary (h<=5 vs h>=6) ────────────────────────────────

def test_threshold_boundary():
    """h=5 uses THR_SHORT=0.03; h=6 uses THR_LONG=0.05.

    A return of +4% should be 'up' at h=5 but 'flat' at h=6.
    """
    # 30 rows: row 0 = 100, row 5 = 104 (+4%), row 6 = 100 (flat relative to 100)
    prices = [100.0] * 30
    prices[5]  = 104.0   # h=5 forward return from row 0 = +4%
    prices[6]  = 100.0   # h=6 forward return from row 0 = 0%  → flat

    # Reset so row 0 close = 100, row 5 future close = 104
    df = _make_df(prices)
    labels = generate_labels(df, _param())

    # h=5: 4% > THR_SHORT(3%) → up
    assert int(labels[5].iloc[0]) == LABEL_MAP['up'], (
        f"h=5 with +4% return should be 'up' (THR_SHORT={THR_SHORT})"
    )
    # h=6: 0% < THR_LONG(5%) → flat
    assert int(labels[6].iloc[0]) == LABEL_MAP['flat'], (
        f"h=6 with 0% return should be 'flat' (THR_LONG={THR_LONG})"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
