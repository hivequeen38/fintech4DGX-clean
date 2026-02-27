"""
migrate_json_to_jsonl.py

One-time migration: convert all *_trend.json files to *_trend.jsonl format.

- Reads each *_trend.json (array of records)
- Writes one compact JSON record per line to *_trend.jsonl
- Original .json files are left untouched (already backed up as .json.bak)

Run once:
    python3 migrate_json_to_jsonl.py
"""

import json
import os
import glob

def migrate(json_path: str):
    jsonl_path = json_path.replace('.json', '.jsonl')

    with open(json_path, 'r', encoding='latin-1') as f:
        raw = f.read().replace('\xa0', ' ')  # non-breaking space → regular space
    data = json.loads(raw)

    with open(jsonl_path, 'w') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')

    print(f'  {os.path.basename(json_path)} → {os.path.basename(jsonl_path)}'
          f'  ({len(data)} records, '
          f'{os.path.getsize(json_path)//1024}KB → {os.path.getsize(jsonl_path)//1024}KB)')


if __name__ == '__main__':
    json_files = sorted(glob.glob('*_trend.json'))
    if not json_files:
        print('No *_trend.json files found.')
    else:
        print(f'Migrating {len(json_files)} files...')
        for path in json_files:
            migrate(path)
        print('Done.')
