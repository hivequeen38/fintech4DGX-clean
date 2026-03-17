"""
test_fetch_cache.py

Unit tests for fetch_cache.py — disk-backed Parquet/pickle cache layer.
All tests are isolated to a tmp_path directory; /workspace/cache/ is never touched.
No network calls.
"""

import os
import pickle
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

import fetch_cache


# ── helpers ───────────────────────────────────────────────────────────────────

def _sample_df() -> pd.DataFrame:
    idx = pd.date_range('2024-01-01', periods=3, name='DATE')
    return pd.DataFrame({'value': [1.0, 2.0, 3.0]}, index=idx)


def _sample_obj() -> dict:
    return {'2024-01-01': {'4. close': '100.0'}, '2024-01-02': {'4. close': '101.0'}}


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Redirect CACHE_DIR to tmp_path so tests never touch /workspace/cache/."""
    monkeypatch.setattr(fetch_cache, 'CACHE_DIR', str(tmp_path))
    yield tmp_path


@pytest.fixture
def today_str():
    import pytz
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern).strftime('%Y-%m-%d')


# ── save / load round-trip ────────────────────────────────────────────────────

def test_save_and_load_dataframe(today_str):
    df = _sample_df()
    fetch_cache.save('TEST_KEY', 'macro', df)
    result = fetch_cache.load('TEST_KEY', 'macro')
    assert result is not None
    pd.testing.assert_frame_equal(result, df, check_freq=False)


def test_save_and_load_dict(today_str):
    obj = _sample_obj()
    fetch_cache.save('TEST_DICT', 'macro', obj)
    result = fetch_cache.load('TEST_DICT', 'macro')
    assert result == obj


def test_load_miss_returns_none(today_str):
    result = fetch_cache.load('NONEXISTENT_KEY', 'macro')
    assert result is None


def test_dataframe_saved_as_parquet(tmp_path, today_str):
    df = _sample_df()
    fetch_cache.save('DF_KEY', 'macro', df)
    files = os.listdir(os.path.join(str(tmp_path), 'macro'))
    assert any(f.endswith('.parquet') and 'DF_KEY' in f for f in files)


def test_dict_saved_as_pickle(tmp_path, today_str):
    obj = _sample_obj()
    fetch_cache.save('DICT_KEY', 'macro', obj)
    files = os.listdir(os.path.join(str(tmp_path), 'macro'))
    assert any(f.endswith('.pkl') and 'DICT_KEY' in f for f in files)


# ── stale fallback ────────────────────────────────────────────────────────────

def test_load_stale_finds_yesterday(tmp_path):
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    tier_dir = os.path.join(str(tmp_path), 'macro')
    os.makedirs(tier_dir)
    df = _sample_df()
    path = os.path.join(tier_dir, f'{yesterday}_STALE_KEY.parquet')
    df.to_parquet(path, index=True)

    result = fetch_cache.load_stale('STALE_KEY', 'macro')
    assert result is not None
    pd.testing.assert_frame_equal(result, df, check_freq=False)


def test_load_stale_rejects_too_old(tmp_path):
    old_date = (datetime.now() - timedelta(days=fetch_cache.MAX_STALE_DAYS + 2)).strftime('%Y-%m-%d')
    tier_dir = os.path.join(str(tmp_path), 'macro')
    os.makedirs(tier_dir)
    df = _sample_df()
    path = os.path.join(tier_dir, f'{old_date}_OLD_KEY.parquet')
    df.to_parquet(path, index=True)

    result = fetch_cache.load_stale('OLD_KEY', 'macro')
    assert result is None


def test_load_stale_returns_none_when_dir_missing():
    result = fetch_cache.load_stale('ANY_KEY', 'macro')
    assert result is None


# ── purge_old_cache ───────────────────────────────────────────────────────────

def test_purge_removes_old_files(tmp_path):
    tier_dir = os.path.join(str(tmp_path), 'macro')
    os.makedirs(tier_dir)

    old_date = (datetime.now() - timedelta(days=fetch_cache.MAX_STALE_DAYS + 2)).strftime('%Y-%m-%d')
    new_date = datetime.now().strftime('%Y-%m-%d')

    old_file = os.path.join(tier_dir, f'{old_date}_OLD.parquet')
    new_file = os.path.join(tier_dir, f'{new_date}_NEW.parquet')
    _sample_df().to_parquet(old_file)
    _sample_df().to_parquet(new_file)

    removed = fetch_cache.purge_old_cache(max_age_days=fetch_cache.MAX_STALE_DAYS + 1)
    assert removed == 1
    assert not os.path.exists(old_file)
    assert os.path.exists(new_file)


def test_purge_empty_cache_dir_is_safe():
    removed = fetch_cache.purge_old_cache()
    assert removed == 0


# ── freshness threshold (Phase 2 parity check) ────────────────────────────────

def test_today_cache_is_fresh(today_str):
    df = _sample_df()
    fetch_cache.save('FRESH_KEY', 'symbol', df)
    result = fetch_cache.load('FRESH_KEY', 'symbol')
    assert result is not None


def test_yesterday_cache_is_stale_miss(tmp_path):
    """load() returns None for yesterday's file (only today is fresh)."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    tier_dir = os.path.join(str(tmp_path), 'symbol')
    os.makedirs(tier_dir)
    df = _sample_df()
    path = os.path.join(tier_dir, f'{yesterday}_YEST_KEY.parquet')
    df.to_parquet(path, index=True)

    result = fetch_cache.load('YEST_KEY', 'symbol')
    assert result is None   # load() only returns today's file


# ── key naming ────────────────────────────────────────────────────────────────

def test_macro_and_symbol_tiers_are_separate(tmp_path):
    df = _sample_df()
    fetch_cache.save('SHARED_KEY', 'macro', df)
    result_macro  = fetch_cache.load('SHARED_KEY', 'macro')
    result_symbol = fetch_cache.load('SHARED_KEY', 'symbol')
    assert result_macro  is not None
    assert result_symbol is None   # different directory


# ── get_latest_stale (Phase 4 foundation) ─────────────────────────────────────

def test_get_latest_stale_returns_newest(tmp_path):
    """get_latest_stale returns the most recent file regardless of age."""
    tier_dir = os.path.join(str(tmp_path), 'macro')
    os.makedirs(tier_dir)
    df_old = pd.DataFrame({'v': [1.0]}, index=pd.Index([pd.Timestamp('2024-01-01')], name='DATE'))
    df_new = pd.DataFrame({'v': [2.0]}, index=pd.Index([pd.Timestamp('2024-01-02')], name='DATE'))
    df_old.to_parquet(os.path.join(tier_dir, '2024-01-01_INC_KEY.parquet'))
    df_new.to_parquet(os.path.join(tier_dir, '2024-01-02_INC_KEY.parquet'))

    result, date_str = fetch_cache.get_latest_stale('INC_KEY', 'macro')
    assert result is not None
    assert date_str == '2024-01-02'   # newest first
    assert float(result['v'].iloc[0]) == 2.0


def test_get_latest_stale_missing_tier(tmp_path):
    result, date_str = fetch_cache.get_latest_stale('MISSING_KEY', 'macro')
    assert result is None
    assert date_str is None


def test_get_latest_stale_missing_key(tmp_path):
    tier_dir = os.path.join(str(tmp_path), 'macro')
    os.makedirs(tier_dir)
    result, date_str = fetch_cache.get_latest_stale('NO_SUCH_KEY', 'macro')
    assert result is None
    assert date_str is None


def test_get_latest_stale_returns_dict(tmp_path):
    """get_latest_stale works for pickled non-DataFrame objects (e.g. AV dicts)."""
    tier_dir = os.path.join(str(tmp_path), 'macro')
    os.makedirs(tier_dir)
    obj = {'2024-01-01': {'4. close': '100'}, '2024-01-02': {'4. close': '101'}}
    path = os.path.join(tier_dir, '2024-01-02_DICT_KEY.pkl')
    import pickle
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)

    result, date_str = fetch_cache.get_latest_stale('DICT_KEY', 'macro')
    assert result == obj
    assert date_str == '2024-01-02'


# ── incremental fetch logic (integration-level, no network) ───────────────────

def test_incremental_threshold_constant_exists():
    """_INCREMENTAL_THRESHOLD must be defined in fetchBulkData."""
    import fetchBulkData as fb
    assert hasattr(fb, '_INCREMENTAL_THRESHOLD')
    assert isinstance(fb._INCREMENTAL_THRESHOLD, int)
    assert fb._INCREMENTAL_THRESHOLD > 0
