import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: slow integration tests that require model training (~5-30 min)"
    )
