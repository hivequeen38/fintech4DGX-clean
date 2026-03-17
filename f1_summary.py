"""
f1_summary.py — F1 comparison report appended to each nightly log.

Parses F1 scores from the current nightly log, compares each
(ticker, profile) against the best F1 ever recorded in
f1_best_history.json, then prints a formatted table to stdout
(which is already redirected into the nightly log file).

Supports two log formats:
  - Single-horizon Transformer: "Model saved as model/model_SYM_PROFILE_..."
    followed by "====> Test set performance <====" block with
    "F1 for class 0/1/2: X.XXX" lines.
  - Multi-horizon MZ: "h=N: F1[flat=X  UP=Y  DN=Z]  macro=W" blocks
    within a known (sym, profile) context.

F1 per (sym, profile) is the average across all 15 horizon models.

History file: /workspace/f1_best_history.json
  {
    "NVDA": {
      "ref": {"flat": 0.715, "up": 0.852, "dn": 0.825, "macro": 0.797,
              "date": "2026-03-07"},
      ...
    },
    ...
  }
"""

import json
import os
import re
import sys

HISTORY_PATH = "/workspace/f1_best_history.json"


# ── Log parser ────────────────────────────────────────────────────────────────

def parse_f1_from_log(log_path: str) -> dict:
    """
    Parse the nightly log and return:
      {(sym, profile): {"flat": float, "up": float, "dn": float,
                        "macro": float, "n": int}}

    F1 values are averaged across all horizon models within each profile.
    """
    accumulated: dict = {}   # (sym, profile) -> list of (flat, up, dn)
    current_sym     = None
    current_profile = None
    in_test         = False
    f0 = f1 = f2    = None

    try:
        lines = open(log_path).readlines()
    except FileNotFoundError:
        return {}

    for line in lines:
        # ── Context: single-horizon model save ───────────────────────────────
        # "Model saved as model/model_NVDA_ref_fixed_noTimesplit_1.pth"
        # "Model saved as model/model_PLTR_ref_noshuf_fixed_noTimesplit_5.pth"
        m = re.search(
            r'Model saved as model/model_([A-Z]+)'
            r'_(ref|ref_noshuf|AAII_option_vol_ratio)_fixed',
            line
        )
        if m:
            current_sym     = m.group(1)
            current_profile = m.group(2)
            in_test = False
            f0 = f1 = f2 = None
            continue

        # ── Context: MZ model save ────────────────────────────────────────────
        # "Model saved as model/model_NVDA_mh_mz_mh_fixed_noTimesplit.pth"
        # "Model saved as model/model_PLTR_mz_reference_mh_fixed_noTimesplit.pth"
        m = re.search(
            r'Model saved as model/model_([A-Z]+)_(?:mh_mz_mh|mz_reference_mh)_fixed',
            line
        )
        if m:
            current_sym     = m.group(1)
            current_profile = 'mz_reference'
            in_test = False
            f0 = f1 = f2 = None
            continue

        # ── Single-horizon test block ─────────────────────────────────────────
        if 'Test set performance' in line:
            in_test = True
            f0 = f1 = f2 = None
            continue

        if in_test:
            m0 = re.search(r'F1 for class 0:\s*([\d.]+)', line)
            m1 = re.search(r'F1 for class 1:\s*([\d.]+)', line)
            m2 = re.search(r'F1 for class 2:\s*([\d.]+)', line)
            if m0: f0 = float(m0.group(1))
            if m1: f1 = float(m1.group(1))
            if m2: f2 = float(m2.group(1))
            if f0 is not None and f1 is not None and f2 is not None \
                    and current_sym and current_profile:
                key = (current_sym, current_profile)
                accumulated.setdefault(key, []).append((f0, f1, f2))
                in_test = False
                f0 = f1 = f2 = None

        # ── MZ per-horizon line ───────────────────────────────────────────────
        # "  h= 1: F1[flat=0.000  UP=0.251  DN=0.193]  macro=0.148"
        m = re.search(
            r'h=\s*\d+:\s*F1\[flat=([\d.]+)\s+UP=([\d.]+)\s+DN=([\d.]+)\]',
            line
        )
        if m and current_sym and current_profile == 'mz_reference':
            key = (current_sym, current_profile)
            accumulated.setdefault(key, []).append(
                (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            )

    # Average across horizons
    result = {}
    for key, vals in accumulated.items():
        n      = len(vals)
        flat_m = round(sum(v[0] for v in vals) / n, 3)
        up_m   = round(sum(v[1] for v in vals) / n, 3)
        dn_m   = round(sum(v[2] for v in vals) / n, 3)
        macro  = round((flat_m + up_m + dn_m) / 3, 3)
        result[key] = {'flat': flat_m, 'up': up_m, 'dn': dn_m,
                       'macro': macro, 'n': n}
    return result


# ── History I/O ───────────────────────────────────────────────────────────────

def load_history() -> dict:
    try:
        return json.load(open(HISTORY_PATH))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_history(history: dict) -> None:
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)


def get_best(history: dict, sym: str, profile: str) -> dict | None:
    return history.get(sym, {}).get(profile)


def update_best(history: dict, sym: str, profile: str,
                current: dict, run_date: str) -> bool:
    """
    Update history if any metric in current exceeds the stored best.
    Returns True if any value was updated.
    """
    prev = get_best(history, sym, profile)
    if prev is None:
        history.setdefault(sym, {})[profile] = {**current, 'date': run_date}
        del history[sym][profile]['n']
        return True

    updated = False
    entry = history[sym][profile]
    for metric in ('flat', 'up', 'dn', 'macro'):
        if current[metric] > entry.get(metric, 0.0):
            entry[metric] = current[metric]
            entry['date'] = run_date
            updated = True
    return updated


# ── Efficacy commentary ───────────────────────────────────────────────────────

def _efficacy_notes(current: dict, all_keys: list) -> str:
    """
    For each ticker (grouped), rank profiles by macro and emit one commentary
    line per profile describing its signal strengths and weaknesses.

    Example output:
      NVDA
        ref            macro=0.764  best overall | UP=0.882 DN=0.825 — strong across all classes
        ref_noshuf     macro=0.205  UP=0.544 best UP of non-ref | no DN signal
        mz_reference   macro=0.292  only non-ref DN signal (0.313) | UP=0.480 trails ref_noshuf
    """
    DN_THRESHOLD = 0.05   # below this → "no DN signal"
    UP_THRESHOLD = 0.05   # below this → "no UP signal"

    # Group by ticker preserving order
    from collections import defaultdict
    by_ticker: dict = defaultdict(list)
    for key in all_keys:
        by_ticker[key[0]].append(key)

    lines = []
    for sym, keys in by_ticker.items():
        profiles = {k[1]: current[k] for k in keys}

        best_macro   = max(v['macro'] for v in profiles.values())
        best_up      = max(v['up']    for v in profiles.values())
        best_dn      = max(v['dn']    for v in profiles.values())
        best_flat    = max(v['flat']  for v in profiles.values())

        lines.append(f'  {sym}')
        for key in keys:
            sym_, profile = key
            v = profiles[profile]

            parts = []

            # Is it the overall macro leader?
            if v['macro'] == best_macro and len(profiles) > 1:
                parts.append('best overall')

            # DN signal analysis
            has_dn = v['dn'] >= DN_THRESHOLD
            if has_dn:
                dn_leader = v['dn'] == best_dn
                other_dn  = any(
                    p != profile and profiles[p]['dn'] >= DN_THRESHOLD
                    for p in profiles
                )
                if dn_leader and not other_dn and len(profiles) > 1:
                    parts.append(f'only DN signal (DN={v["dn"]:.3f})')
                elif dn_leader and len(profiles) > 1:
                    parts.append(f'best DN={v["dn"]:.3f}')
                else:
                    parts.append(f'DN={v["dn"]:.3f}')
            else:
                parts.append('no DN signal')

            # UP signal analysis
            has_up = v['up'] >= UP_THRESHOLD
            if has_up:
                up_leader = v['up'] == best_up
                if up_leader and len(profiles) > 1:
                    parts.append(f'best UP={v["up"]:.3f}')
                else:
                    # note gap vs best
                    gap = best_up - v['up']
                    if gap > 0.01:
                        parts.append(f'UP={v["up"]:.3f} (−{gap:.3f} vs best)')
                    else:
                        parts.append(f'UP={v["up"]:.3f}')
            else:
                parts.append('no UP signal')

            # flat signal
            if v['flat'] == best_flat and len(profiles) > 1:
                parts.append(f'best flat={v["flat"]:.3f}')
            else:
                parts.append(f'flat={v["flat"]:.3f}')

            commentary = ' | '.join(parts)
            lines.append(
                f'    {profile:<28}  macro={v["macro"]:.3f}  {commentary}'
            )

    return '\n'.join(lines)


# ── Report printer ────────────────────────────────────────────────────────────

def _fmt(val, prev_best, width=6) -> str:
    """Format value with ▲ if it beats personal best."""
    s = f'{val:.3f}'
    marker = '▲' if val > prev_best + 0.001 else ' '
    return f'{marker}{s:>{width}}'


def print_summary(log_path: str, run_date: str) -> None:
    """
    Parse log_path, compare F1 to best history, print table to stdout,
    update history file.  Called at end of nightly_run.py.
    """
    current  = parse_f1_from_log(log_path)
    history  = load_history()

    if not current:
        print('\n[F1 SUMMARY] No F1 data found in log — skipping summary.')
        return

    PROFILE_ORDER = ['ref', 'AAII_option_vol_ratio', 'ref_noshuf', 'mz_reference']
    TICKER_ORDER  = ['NVDA', 'PLTR', 'APP', 'CRDO', 'INOD']

    all_keys = sorted(
        current.keys(),
        key=lambda k: (
            TICKER_ORDER.index(k[0]) if k[0] in TICKER_ORDER else 99,
            PROFILE_ORDER.index(k[1]) if k[1] in PROFILE_ORDER else 99,
        )
    )

    W = 94
    print(f'\n{"="*W}')
    print(f'  F1 SUMMARY vs PERSONAL BEST — {run_date}')
    print(f'{"="*W}')
    hdr = (f'  {"Ticker":<6} {"Profile":<28}'
           f'  {"flat":>7} {"UP":>7} {"DN":>7} {"macro":>7}'
           f'  {"bestFlat":>8} {"bestUP":>8} {"bestDN":>8} {"bestMacro":>9}'
           f'  {"New?":>5}')
    print(hdr)
    print(f'  {"─"*6} {"─"*28}  {"─"*7} {"─"*7} {"─"*7} {"─"*7}'
          f'  {"─"*8} {"─"*8} {"─"*8} {"─"*9}  {"─"*5}')

    any_new_best = False
    for key in all_keys:
        sym, profile = key
        cur  = current[key]
        prev = get_best(history, sym, profile) or {}

        pb_flat  = prev.get('flat',  0.0)
        pb_up    = prev.get('up',    0.0)
        pb_dn    = prev.get('dn',    0.0)
        pb_macro = prev.get('macro', 0.0)

        is_new = update_best(history, sym, profile, cur, run_date)
        any_new_best = any_new_best or is_new
        new_marker = '  ★  ' if is_new else '     '

        print(
            f'  {sym:<6} {profile:<28}'
            f'  {_fmt(cur["flat"],  pb_flat)}'
            f'  {_fmt(cur["up"],    pb_up)}'
            f'  {_fmt(cur["dn"],    pb_dn)}'
            f'  {_fmt(cur["macro"], pb_macro)}'
            f'  {pb_flat:>8.3f} {pb_up:>8.3f} {pb_dn:>8.3f} {pb_macro:>9.3f}'
            f'  {new_marker}'
        )

    print(f'  {"─"*W}')
    print(f'  ▲ = beats personal best for that metric   ★ = any metric improved')
    print(f'  F1 averaged across {current[all_keys[0]]["n"]} horizon models per profile.')
    print(f'  History saved to: {HISTORY_PATH}')
    print(f'{"="*W}')

    # ── Per-ticker efficacy notes ─────────────────────────────────────────────
    print(f'\n  PROFILE EFFICACY — signal strengths per ticker')
    print(f'  {"─"*W}')
    print(_efficacy_notes(current, all_keys))
    print(f'  {"─"*W}')
    print(f'  DN threshold={0.05:.2f}  UP threshold={0.05:.2f}  "best" = highest among profiles trained tonight')
    print(f'{"="*W}\n')

    save_history(history)


# ── CLI (standalone use) ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Print F1 summary for a nightly log')
    parser.add_argument('log_path', help='Path to nightly log file')
    parser.add_argument('--date', default=None,
                        help='Run date label (YYYY-MM-DD); defaults to log filename date')
    args = parser.parse_args()

    run_date = args.date
    if not run_date:
        m = re.search(r'(\d{8})', os.path.basename(args.log_path))
        run_date = (f'{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}'
                    if m else 'unknown')

    print_summary(args.log_path, run_date)
