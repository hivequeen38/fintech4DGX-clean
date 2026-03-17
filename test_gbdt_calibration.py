"""
Unit tests for gbdt_pipeline temperature scaling and threshold calibration.

Tests:
  1. fit_temperature returns T in [0.1, 10.0]
  2. Temperature-scaled probabilities sum to 1 (per row)
  3. calibrate_threshold returns θ in [0.40, 0.90]
  4. calibrate_threshold fallback (θ=0.65) when no θ achieves precision≥0.50

Run with:
    python -m pytest test_gbdt_calibration.py -v
"""

import numpy as np
import pytest
from scipy.special import softmax


# ---------------------------------------------------------------------------
# Lightweight stubs — avoid importing lightgbm (may be slow to initialise)
# ---------------------------------------------------------------------------

class _FakeBooster:
    """Minimal lgb.Booster stub that returns fixed raw logits."""

    def __init__(self, raw_logits: np.ndarray):
        self._logits = raw_logits   # shape (N, 3)

    def predict(self, X, raw_score=False):
        n = len(X)
        if raw_score:
            return self._logits[:n]
        return softmax(self._logits[:n], axis=1)


def _make_val_data(n=100, seed=0):
    """Return X_val (N×3 dummy) and y_val (random {0,1,2})."""
    rng   = np.random.default_rng(seed)
    X_val = rng.standard_normal((n, 3)).astype(np.float32)
    y_val = rng.integers(0, 3, size=n)
    return X_val, y_val


def _make_up_heavy_model(n=200, seed=7):
    """Booster that strongly predicts Up for all samples."""
    rng     = np.random.default_rng(seed)
    logits  = rng.standard_normal((n, 3))
    logits[:, 2] += 5   # large Up score
    return _FakeBooster(logits)


# ---------------------------------------------------------------------------
# Import only what we can without requiring a trained model file
# ---------------------------------------------------------------------------
from gbdt_pipeline import fit_temperature, calibrate_threshold


# ── Test 1: T_star in valid range ────────────────────────────────────────────

def test_temperature_in_valid_range():
    """fit_temperature must return T_star ∈ [0.1, 10.0]."""
    rng    = np.random.default_rng(1)
    n      = 80
    X_val  = rng.standard_normal((n, 3)).astype(np.float32)
    y_val  = rng.integers(0, 3, size=n)
    logits = rng.standard_normal((n, 3))
    model  = _FakeBooster(logits)

    T = fit_temperature(model, X_val, y_val)
    assert 0.1 <= T <= 10.0, f'T_star={T} outside [0.1, 10.0]'


# ── Test 2: Scaled probabilities sum to 1 ────────────────────────────────────

def test_scaled_probs_sum_to_one():
    """After temperature scaling, row probabilities must sum to 1 (within 1e-5)."""
    rng    = np.random.default_rng(2)
    n      = 50
    X_val  = rng.standard_normal((n, 3)).astype(np.float32)
    y_val  = rng.integers(0, 3, size=n)
    logits = rng.standard_normal((n, 3))
    model  = _FakeBooster(logits)

    T     = fit_temperature(model, X_val, y_val)
    probs = softmax(logits / T, axis=1)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), (
        f'Row probabilities do not sum to 1 (max deviation: {np.abs(row_sums-1).max():.2e})'
    )


# ── Test 3: θ in valid range ─────────────────────────────────────────────────

def test_threshold_in_valid_range():
    """calibrate_threshold must always return θ ∈ [0.40, 0.90]."""
    rng    = np.random.default_rng(3)
    n      = 100
    X_val  = rng.standard_normal((n, 3)).astype(np.float32)
    y_val  = rng.integers(0, 3, size=n)
    logits = rng.standard_normal((n, 3))
    model  = _FakeBooster(logits)
    T      = 1.0

    theta = calibrate_threshold(model, T, X_val, y_val, h=10, sym='TEST')
    assert 0.40 <= theta <= 0.90, f'theta={theta} outside [0.40, 0.90]'


# ── Test 4: Fallback θ=0.65 when precision target impossible ─────────────────

def test_threshold_fallback():
    """When no θ in [0.40, 0.90] achieves precision≥0.50 with coverage≥0.05,
    calibrate_threshold must return the fallback value 0.65."""
    # All samples are truly Down (label=0) but model is biased Up
    n      = 200
    y_val  = np.zeros(n, dtype=int)   # all ground-truth Down
    X_val  = np.zeros((n, 3), dtype=np.float32)
    logits = np.zeros((n, 3))
    logits[:, 2] = 5.0   # model always scores Up very high → precision for Up = 0%
    model = _FakeBooster(logits)
    T     = 1.0

    theta = calibrate_threshold(model, T, X_val, y_val, h=5, sym='TEST')
    assert theta == pytest.approx(0.65, abs=0.01), (
        f'Expected fallback θ≈0.65, got θ={theta:.4f}'
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
