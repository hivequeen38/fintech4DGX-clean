#!/usr/bin/env python3
"""
cp_ratio_backfill.py — Standalone script to bulk-fetch / backfill options CP ratio
and IV features for active symbols.

Reads trading dates from {SYMBOL}_TMP.csv and fetches AV HISTORICAL_OPTIONS for
each date, computing CP ratio metrics AND IV features (iv_7d, iv_30d, iv_90d,
iv_skew_30d, iv_term_ratio) in a single API call per date.

Writes results to {SYMBOL}-cp_ratios_sentiment_w_volume.csv — the same file the
daily training pipeline reads. Safe to re-run: already-processed dates are skipped.
Rows present in the CSV but missing iv_30d (pre-IV-support data) are automatically
re-fetched (IV backfill).

Usage:
    python cp_ratio_backfill.py                   # all 5 active stocks
    python cp_ratio_backfill.py NVDA              # single stock
    python cp_ratio_backfill.py NVDA CRDO PLTR    # explicit list

Rate: 1 API call per date, 12-second sleep (AV premium plan).
      ~1,200 historical dates × 12s ≈ 4 hours per stock.
      Run overnight or in a screen/tmux session.
"""

import sys
import os
import time

import numpy as np
import pandas as pd
import requests

from trendConfig import config

ACTIVE_SYMBOLS = ['NVDA', 'CRDO', 'PLTR', 'APP', 'INOD']


# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_cp_ratio_file(symbol):
    """Remove duplicate rows from the existing CSV."""
    file_path = f"{symbol}-cp_ratios_sentiment_w_volume.csv"
    if not os.path.exists(file_path):
        return
    try:
        df = pd.read_csv(file_path)
        original_count = len(df)
        if 'date' not in df.columns and df.shape[1] == 10:
            df.columns = [
                'date', 'call_volume', 'put_volume', 'call_oi', 'put_oi',
                'cp_volume_ratio', 'cp_oi_ratio', 'daily_sentiment',
                'bullish_volume', 'bearish_volume'
            ]
        df = df.drop_duplicates(subset=['date'])
        df.to_csv(file_path, index=False)
        removed = original_count - len(df)
        if removed:
            print(f"  Cleaned {file_path}: removed {removed} duplicate rows")
    except Exception as e:
        print(f"  Warning: could not clean {file_path}: {e}")


def fetch_cp_and_iv(symbol, dates_df, api_key):
    """
    Fetch CP ratio metrics and IV features for the given dates.
    Exact copy of get_historical_cp_ratios_with_sentiments_new() from fetchBulkData.py.
    Inlined here because fetchBulkData.py cannot be imported standalone
    (depends on pandas_datareader which is not installed in the venv).
    """
    file_path = f"{symbol}-cp_ratios_sentiment_w_volume.csv"
    IV_COLS = ['iv_7d', 'iv_30d', 'iv_90d', 'iv_skew_30d', 'iv_term_ratio']

    if os.path.exists(file_path):
        print(f"  Cleaning existing file for duplicates...")
        clean_cp_ratio_file(symbol)

    results_df = pd.DataFrame(columns=[
        'date', 'call_volume', 'put_volume', 'call_oi', 'put_oi',
        'cp_volume_ratio', 'cp_oi_ratio', 'daily_sentiment',
        'bullish_volume', 'bearish_volume',
        'iv_7d', 'iv_30d', 'iv_90d', 'iv_skew_30d', 'iv_term_ratio'
    ])
    results_df = results_df.astype({
        'date': 'object',
        'call_volume': 'float64', 'put_volume': 'float64',
        'call_oi': 'float64', 'put_oi': 'float64',
        'cp_volume_ratio': 'float64', 'cp_oi_ratio': 'float64',
        'daily_sentiment': 'object',
        'bullish_volume': 'float64', 'bearish_volume': 'float64',
        'iv_7d': 'float64', 'iv_30d': 'float64', 'iv_90d': 'float64',
        'iv_skew_30d': 'float64', 'iv_term_ratio': 'float64'
    })

    new_dates_set = set()
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        for c in IV_COLS:
            if c not in existing_df.columns:
                existing_df[c] = np.nan

        processed_dates = set(existing_df['date'])
        all_dates = set(dates_df['date'])
        remaining_dates = sorted(list(all_dates - processed_dates))
        new_dates_set = set(remaining_dates)

        iv_backfill_dates = existing_df[existing_df['iv_30d'].isna()]['date'].tolist()
        iv_backfill_dates = [d for d in iv_backfill_dates if d not in new_dates_set]

        print(f"  Existing: {len(existing_df)} rows  |  New dates: {len(remaining_dates)}  |  IV backfill: {len(iv_backfill_dates)}")

        results_df = existing_df.copy()
        all_to_process = sorted(list(set(remaining_dates + iv_backfill_dates)))
        dates_to_process_df = pd.DataFrame({'date': all_to_process})
    else:
        print(f"  Starting fresh — {len(dates_df)} dates to fetch")
        dates_to_process_df = dates_df.copy()
        new_dates_set = set(dates_df['date'])

    def _oi_weighted_iv(subset):
        w = subset['open_interest'].fillna(0)
        if w.sum() == 0:
            return float(subset['implied_volatility'].mean())
        return float((subset['implied_volatility'] * w).sum() / w.sum())

    for date_str in dates_to_process_df['date']:
        try:
            is_new_date = date_str in new_dates_set
            label = '(new)' if is_new_date else '(IV backfill)'
            print(f"  {date_str} {label}")

            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "HISTORICAL_OPTIONS",
                    "symbol": symbol,
                    "date": date_str,
                    "apikey": api_key
                }
            )
            data = response.json()

            row = {
                'date': date_str,
                'call_volume': 0.0, 'put_volume': 0.0,
                'call_oi': 0.0, 'put_oi': 0.0,
                'cp_volume_ratio': 0.0, 'cp_oi_ratio': 0.0,
                'daily_sentiment': 'no_trades',
                'bullish_volume': 0.0, 'bearish_volume': 0.0,
                'iv_7d': np.nan, 'iv_30d': np.nan, 'iv_90d': np.nan,
                'iv_skew_30d': np.nan, 'iv_term_ratio': np.nan
            }

            if "data" in data and data["data"]:
                df = pd.DataFrame(data["data"])

                if not df.empty:
                    for col in ['volume', 'open_interest', 'bid', 'ask', 'last']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                    df['mid'] = ((df['bid'] + df['ask']) / 2).fillna(df['last'])

                    df['sentiment'] = df.apply(lambda x:
                        'no_trade' if x['volume'] == 0 or pd.isna(x['last']) else
                        'bullish' if (x['last'] >= x['mid'] and x['type'] == 'call') or
                                    (x['last'] < x['mid'] and x['type'] == 'put') else
                        'bearish', axis=1)

                    type_totals = df.groupby('type').agg({
                        'volume': 'sum', 'open_interest': 'sum'
                    }).fillna(0)

                    row['call_volume'] = float(type_totals.loc['call', 'volume']) if 'call' in type_totals.index else 0.0
                    row['put_volume']  = float(type_totals.loc['put',  'volume']) if 'put'  in type_totals.index else 0.0
                    row['call_oi']     = float(type_totals.loc['call', 'open_interest']) if 'call' in type_totals.index else 0.0
                    row['put_oi']      = float(type_totals.loc['put',  'open_interest']) if 'put'  in type_totals.index else 0.0

                    if row['put_volume'] > 0:
                        row['cp_volume_ratio'] = row['call_volume'] / row['put_volume']
                    if row['put_oi'] > 0:
                        row['cp_oi_ratio'] = row['call_oi'] / row['put_oi']

                    row['bullish_volume'] = float(df[df['sentiment'] == 'bullish']['volume'].sum())
                    row['bearish_volume'] = float(df[df['sentiment'] == 'bearish']['volume'].sum())

                    if row['bullish_volume'] == 0 and row['bearish_volume'] == 0:
                        row['daily_sentiment'] = 'no_trades'
                    elif row['bullish_volume'] > row['bearish_volume']:
                        row['daily_sentiment'] = 'bullish'
                    elif row['bearish_volume'] > row['bullish_volume']:
                        row['daily_sentiment'] = 'bearish'
                    else:
                        row['daily_sentiment'] = 'neutral'

                    # ── IV features ──────────────────────────────────────────
                    if 'implied_volatility' in df.columns and 'expiration' in df.columns:
                        for col in ['implied_volatility', 'strike']:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')

                        df['dte'] = (pd.to_datetime(df['expiration']) - pd.to_datetime(date_str)).dt.days
                        df_iv = df[(df['implied_volatility'] > 0) & (df['dte'] > 0)].copy()

                        m7 = (df_iv['dte'] >= 1) & (df_iv['dte'] <= 10)
                        if m7.sum() > 0:
                            row['iv_7d'] = _oi_weighted_iv(df_iv[m7])

                        m30 = (df_iv['dte'] >= 20) & (df_iv['dte'] <= 45)
                        if m30.sum() > 0:
                            row['iv_30d'] = _oi_weighted_iv(df_iv[m30])
                            df30 = df_iv[m30]
                            calls30 = df30[df30['type'] == 'call']
                            puts30  = df30[df30['type'] == 'put']
                            call_iv = _oi_weighted_iv(calls30) if len(calls30) > 0 else np.nan
                            put_iv  = _oi_weighted_iv(puts30)  if len(puts30)  > 0 else np.nan
                            if pd.notna(call_iv) and pd.notna(put_iv):
                                row['iv_skew_30d'] = put_iv - call_iv

                        m90 = (df_iv['dte'] >= 60) & (df_iv['dte'] <= 120)
                        if m90.sum() > 0:
                            row['iv_90d'] = _oi_weighted_iv(df_iv[m90])

                        if pd.notna(row['iv_7d']) and pd.notna(row['iv_30d']) and row['iv_30d'] > 0:
                            row['iv_term_ratio'] = row['iv_7d'] / row['iv_30d']

            if is_new_date:
                results_df.loc[len(results_df)] = row
            else:
                mask = results_df['date'] == date_str
                for c in IV_COLS:
                    results_df.loc[mask, c] = row[c]

            results_df = results_df.drop_duplicates(subset=['date'])
            results_df.to_csv(file_path, index=False)
            time.sleep(12)

        except Exception as e:
            print(f"  Error processing {date_str}: {e}")
            continue

    cp_float_cols = ['call_volume', 'put_volume', 'call_oi', 'put_oi',
                     'cp_volume_ratio', 'cp_oi_ratio', 'bullish_volume', 'bearish_volume']
    results_df[cp_float_cols] = results_df[cp_float_cols].fillna(0.0)
    results_df['daily_sentiment'] = results_df['daily_sentiment'].fillna('no_trades')

    if not results_df.empty:
        results_df.sort_values(by='date').to_csv(file_path, index=False)
        results_df = results_df.sort_values(by='date').reset_index(drop=True)

    results_df = results_df.fillna({
        'call_volume': 0.0, 'put_volume': 0.0,
        'call_oi': 0.0, 'put_oi': 0.0,
        'cp_volume_ratio': 0.0, 'cp_oi_ratio': 0.0,
        'daily_sentiment': 'no_trades',
        'bullish_volume': 0.0, 'bearish_volume': 0.0
    })

    all_cols = ['date', 'call_volume', 'put_volume', 'cp_volume_ratio', 'cp_oi_ratio',
                'bullish_volume', 'bearish_volume', 'iv_7d', 'iv_30d', 'iv_90d',
                'iv_skew_30d', 'iv_term_ratio']
    return results_df[[c for c in all_cols if c in results_df.columns]]


# ── Main ─────────────────────────────────────────────────────────────────────

def backfill_symbol(symbol: str, api_key: str) -> None:
    tmp_path = f"{symbol}_TMP.csv"
    if not os.path.exists(tmp_path):
        print(f"[{symbol}] WARNING: {tmp_path} not found — skipping.")
        print(f"[{symbol}]   Run the training pipeline once to generate it.")
        return

    master_df = pd.read_csv(tmp_path)
    if 'date' not in master_df.columns:
        print(f"[{symbol}] ERROR: {tmp_path} has no 'date' column — skipping.")
        return

    dates_df = master_df[['date']].copy()
    dates_df['date'] = pd.to_datetime(dates_df['date']).dt.strftime('%Y-%m-%d')
    dates_df = dates_df.drop_duplicates().sort_values('date').reset_index(drop=True)

    print(f"\n[{symbol}] {len(dates_df)} total trading dates in TMP.csv")
    result = fetch_cp_and_iv(symbol, dates_df, api_key)
    csv_path = f"{symbol}-cp_ratios_sentiment_w_volume.csv"
    iv_coverage = result['iv_30d'].notna().sum()
    print(f"[{symbol}] Done — {len(result)} rows, {iv_coverage} with iv_30d coverage")


if __name__ == '__main__':
    api_key = config['alpha_vantage']['key']
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ACTIVE_SYMBOLS

    print("CP Ratio / IV Backfill")
    print(f"Symbols  : {symbols}")
    print(f"Rate     : 12s/date  (~4h per stock for full historical backfill)")
    print(f"Resume   : yes — already-complete dates are skipped automatically\n")

    for symbol in symbols:
        backfill_symbol(symbol, api_key)

    print("\nAll done.")
