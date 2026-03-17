"""
fetch_cache.py  —  Disk-backed cache layer for external API responses.

Cache lives under /workspace/cache/{tier}/ as Parquet files (DataFrames) or
pickle files (dicts / raw Alpha Vantage responses).

File naming:  {YYYY-MM-DD}_{key}.{parquet|pkl}
Freshness:    A file is "fresh" if its date prefix == today (US/Eastern).
Stale files:  Kept up to MAX_STALE_DAYS for failure recovery.

Typical usage (in fetchBulkData.py wrappers):
    import fetch_cache as _fc
    disk = _fc.load('FRED_DFF_2021-01-01', 'macro')
    if disk is not None:
        return disk
    result = <network fetch>
    _fc.save('FRED_DFF_2021-01-01', 'macro', result)
    return result
"""

import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
import pytz

# ── constants ─────────────────────────────────────────────────────────────────
CACHE_DIR      = '/workspace/cache'
MAX_STALE_DAYS = 3   # serve data up to 3 days old on fetch failure

_EASTERN = pytz.timezone('US/Eastern')


# ── internal helpers ──────────────────────────────────────────────────────────

def _today() -> str:
    return datetime.now(_EASTERN).strftime('%Y-%m-%d')


def _cache_path(tier: str, key: str, date: str, obj) -> str:
    ext = 'parquet' if isinstance(obj, pd.DataFrame) else 'pkl'
    return os.path.join(CACHE_DIR, tier, f'{date}_{key}.{ext}')


def _fresh_path(tier: str, key: str, obj) -> str:
    return _cache_path(tier, key, _today(), obj)


def _read_file(path: str):
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    with open(path, 'rb') as fh:
        return pickle.load(fh)


# ── public API ────────────────────────────────────────────────────────────────

def load(key: str, tier: str):
    """Return today's cached object for key, or None on miss.

    Tries both .parquet and .pkl variants — whichever exists.
    """
    today = _today()
    tier_dir = os.path.join(CACHE_DIR, tier)
    for ext in ('parquet', 'pkl'):
        path = os.path.join(tier_dir, f'{today}_{key}.{ext}')
        if os.path.exists(path):
            try:
                return _read_file(path)
            except Exception as e:
                print(f'[CACHE] WARN: failed to read {path}: {e}')
    return None


def save(key: str, tier: str, obj) -> None:
    """Write obj to today's disk cache for key.

    DataFrames → Parquet (with index).  Everything else → pickle.
    """
    tier_dir = os.path.join(CACHE_DIR, tier)
    os.makedirs(tier_dir, exist_ok=True)
    today = _today()
    if isinstance(obj, pd.DataFrame):
        path = os.path.join(tier_dir, f'{today}_{key}.parquet')
        obj.to_parquet(path, index=True)
    else:
        path = os.path.join(tier_dir, f'{today}_{key}.pkl')
        with open(path, 'wb') as fh:
            pickle.dump(obj, fh)
    print(f'[CACHE] Saved {path}')


def load_stale(key: str, tier: str):
    """Return the most recent cached file for key within MAX_STALE_DAYS.

    Used as a fallback when today's file is missing and a network fetch fails.
    Returns None if no stale file is found.
    """
    tier_dir = os.path.join(CACHE_DIR, tier)
    if not os.path.isdir(tier_dir):
        return None
    # Collect files whose name contains _{key}. (both .parquet and .pkl)
    candidates = sorted(
        [f for f in os.listdir(tier_dir) if f'_{key}.' in f],
        reverse=True   # newest first (YYYY-MM-DD prefix sorts lexicographically)
    )
    cutoff = (datetime.now(_EASTERN) - timedelta(days=MAX_STALE_DAYS)).strftime('%Y-%m-%d')
    for fname in candidates:
        date_prefix = fname[:10]
        if date_prefix < cutoff:
            break   # too old
        path = os.path.join(tier_dir, fname)
        try:
            result = _read_file(path)
            print(f'[CACHE] STALE: serving {path}')
            return result
        except Exception as e:
            print(f'[CACHE] WARN: stale read failed for {path}: {e}')
    return None


def get_latest_stale(key: str, tier: str):
    """Return (obj, date_str) of the most recent cached file for key.

    Unlike load(), this does NOT require today's date — it finds the newest
    file regardless of age.  Used by the incremental fetch logic to locate
    a base DataFrame to append a delta to.

    Returns (None, None) if no cached file is found.
    """
    tier_dir = os.path.join(CACHE_DIR, tier)
    if not os.path.isdir(tier_dir):
        return None, None
    candidates = sorted(
        [f for f in os.listdir(tier_dir) if f'_{key}.' in f],
        reverse=True   # newest first
    )
    for fname in candidates:
        date_prefix = fname[:10]   # 'YYYY-MM-DD'
        path = os.path.join(tier_dir, fname)
        try:
            obj = _read_file(path)
            return obj, date_prefix
        except Exception as e:
            print(f'[CACHE] WARN: get_latest_stale failed for {path}: {e}')
    return None, None


def purge_old_cache(max_age_days: int = MAX_STALE_DAYS + 1) -> int:
    """Delete cache files older than max_age_days. Returns number of files removed."""
    cutoff = (datetime.now(_EASTERN) - timedelta(days=max_age_days)).strftime('%Y-%m-%d')
    removed = 0
    for tier in ('macro', 'symbol'):
        tier_dir = os.path.join(CACHE_DIR, tier)
        if not os.path.isdir(tier_dir):
            continue
        for fname in os.listdir(tier_dir):
            if len(fname) >= 10 and fname[:10] < cutoff:
                try:
                    os.remove(os.path.join(tier_dir, fname))
                    removed += 1
                except OSError as e:
                    print(f'[CACHE] WARN: could not remove {fname}: {e}')
    if removed:
        print(f'[CACHE] Purged {removed} stale file(s) older than {max_age_days} days.')
    return removed
