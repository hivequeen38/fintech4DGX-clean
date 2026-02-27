"""
test_model_factory.py

Unit tests for the build_model() factory in trendAnalysisFromTodayNew.py.

All tests are fully in-memory — no disk I/O, no .pth files, no scalers.
Safe to run at any time without affecting trained models or inference runs.
"""
import pytest
import torch
from trendAnalysisFromTodayNew import build_model, TransformerModel


_BASE_PARAM = {
    'headcount': 8,
    'num_layers': 2,
    'dropout_rate': 0.1,
    'embedded_dim': 128,
}


def test_build_model_transformer_default():
    """Factory returns TransformerModel when model_type is absent (backward compat)."""
    model = build_model(_BASE_PARAM, input_dim=64, num_classes=3)
    assert isinstance(model, TransformerModel)


def test_build_model_transformer_explicit():
    """Factory returns TransformerModel when model_type='transformer'."""
    param = {**_BASE_PARAM, 'model_type': 'transformer'}
    model = build_model(param, input_dim=64, num_classes=3)
    assert isinstance(model, TransformerModel)


def test_build_model_unknown_raises():
    """Factory raises ValueError for an unrecognised model_type."""
    param = {'model_type': 'bogus'}
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_model(param, input_dim=64)


def test_factory_output_shape_matches_direct():
    """Factory-built model output shape matches directly instantiated TransformerModel."""
    torch.manual_seed(42)
    model_direct = TransformerModel(
        input_dim=64, num_classes=3, num_heads=8,
        num_layers=2, dropout_rate=0.1, embedded_dim=128,
    )
    model_factory = build_model(_BASE_PARAM, input_dim=64, num_classes=3)
    x = torch.randn(4, 64)
    assert model_direct(x).shape == model_factory(x).shape
