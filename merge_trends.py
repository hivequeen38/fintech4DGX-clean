"""
merge_trends.py

Merges historical *_trend.jsonl and *_trend.csv files from a Mac machine
into the DGX machine's existing files.

Usage:
    1. Place Mac's trend files in ./mac_trends/ directory
    2. Run: python3 merge_trends.py
    3. Merged files are written back to /workspace/*_trend.jsonl and *_trend.csv
    4. Then re-run upload to regenerate HTMLs with full history

The merge deduplicates by run_time timestamp, keeping all unique entries
from both machines, sorted chronologically.

NOTE: Mac machine must also be migrated to .jsonl format before merging.
Run migrate_json_to_jsonl.py on the Mac side first.
"""

import json
import os
import pandas as pd

STOCKS = ['NVDA', 'PLTR', 'CRDO', 'INOD', 'APP']
MAC_DIR = './mac_trends'
DGX_DIR = '.'


def _load_jsonl(path: str) -> list:
    """Load all records from a .jsonl file (one JSON object per line)."""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


def merge_jsonl(stock: str):
    dgx_path = os.path.join(DGX_DIR, f'{stock}_trend.jsonl')
    mac_path = os.path.join(MAC_DIR, f'{stock}_trend.jsonl')

    if not os.path.isfile(mac_path):
        print(f'  [{stock}] No Mac JSONL found at {mac_path}, skipping.')
        return

    print(f'  [{stock}] Loading DGX JSONL...', end=' ')
    dgx_data = _load_jsonl(dgx_path)
    print(f'{len(dgx_data)} records')

    print(f'  [{stock}] Loading Mac JSONL...', end=' ')
    mac_data = _load_jsonl(mac_path)
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
        for record in merged:
            f.write(json.dumps(record) + '\n')
    print(f'  [{stock}] Written to {dgx_path}')
    return merged


def rebuild_csv_from_jsonl(stock: str):
    """Regenerate the *_trend.csv from the merged *_trend.jsonl."""
    jsonl_path = os.path.join(DGX_DIR, f'{stock}_trend.jsonl')
    csv_path = os.path.join(DGX_DIR, f'{stock}_trend.csv')

    data = _load_jsonl(jsonl_path)

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
        print('Create it and place the Mac *_trend.jsonl files inside.')
        return

    print(f'=== Merging trend files from {MAC_DIR} into {DGX_DIR} ===\n')

    for stock in STOCKS:
        print(f'--- {stock} ---')
        merged = merge_jsonl(stock)
        if merged:
            rebuild_csv_from_jsonl(stock)
        print()

    print('=== Done! ===')
    print('Next step: regenerate and upload HTML:')
    print("  python3 -c \"import get_historical_html; get_historical_html.upload_all_results('2026-02-17', upload_to_cloud=True)\"")


if __name__ == '__main__':
    main()
