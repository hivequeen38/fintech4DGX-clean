"""
merge_trends.py

Merges historical *_trend.json and *_trend.csv files from a Mac machine
into the DGX machine's existing files.

Usage:
    1. Place Mac's trend files in ./mac_trends/ directory
    2. Run: python3 merge_trends.py
    3. Merged files are written back to /workspace/*_trend.json and *_trend.csv
    4. Then re-run upload to regenerate HTMLs with full history

The merge deduplicates by run_time timestamp, keeping all unique entries
from both machines, sorted chronologically.
"""

import json
import os
import pandas as pd

STOCKS = ['NVDA', 'PLTR', 'CRDO', 'INOD', 'APP']
MAC_DIR = './mac_trends'
DGX_DIR = '.'


def merge_json(stock: str):
    dgx_path = os.path.join(DGX_DIR, f'{stock}_trend.json')
    mac_path = os.path.join(MAC_DIR, f'{stock}_trend.json')

    if not os.path.isfile(mac_path):
        print(f'  [{stock}] No Mac JSON found at {mac_path}, skipping.')
        return

    print(f'  [{stock}] Loading DGX JSON...', end=' ')
    with open(dgx_path) as f:
        dgx_data = json.load(f)
    print(f'{len(dgx_data)} records')

    print(f'  [{stock}] Loading Mac JSON...', end=' ')
    with open(mac_path) as f:
        mac_data = json.load(f)
    print(f'{len(mac_data)} records')

    # Deduplicate by run_time (first element of each record)
    seen = {}
    for record in mac_data + dgx_data:
        run_time = record[0]
        if run_time not in seen:
            seen[run_time] = record

    merged = sorted(seen.values(), key=lambda r: r[0])
    print(f'  [{stock}] Merged: {len(merged)} unique records '
          f'(+{len(merged) - len(dgx_data)} new from Mac)')

    with open(dgx_path, 'w') as f:
        json.dump(merged, f)
    print(f'  [{stock}] Written to {dgx_path}')
    return merged


def rebuild_csv_from_json(stock: str):
    """Regenerate the *_trend.csv from the merged *_trend.json."""
    json_path = os.path.join(DGX_DIR, f'{stock}_trend.json')
    csv_path = os.path.join(DGX_DIR, f'{stock}_trend.csv')

    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for record in data:
        run_time = record[0]
        params = record[1] if len(record) > 1 else {}
        metrics = record[2] if len(record) > 2 else {}
        row = {'run_time': run_time}
        row.update(params)
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=True)
    print(f'  [{stock}] CSV rebuilt: {len(df)} rows -> {csv_path}')


def main():
    if not os.path.isdir(MAC_DIR):
        print(f'ERROR: Mac trends directory not found: {MAC_DIR}')
        print('Create it and place the Mac *_trend.json files inside.')
        return

    print(f'=== Merging trend files from {MAC_DIR} into {DGX_DIR} ===\n')

    for stock in STOCKS:
        print(f'--- {stock} ---')
        merged = merge_json(stock)
        if merged:
            rebuild_csv_from_json(stock)
        print()

    print('=== Done! ===')
    print('Next step: regenerate and upload HTML:')
    print("  python3 -c \"import get_historical_html; get_historical_html.upload_all_results('2026-02-17', upload_to_cloud=True)\"")


if __name__ == '__main__':
    main()
