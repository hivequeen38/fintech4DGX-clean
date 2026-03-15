"""
EPS analyst-estimate features from Alpha Vantage EARNINGS_ESTIMATES endpoint.

Provides forward-looking (and historical snapshot) analyst consensus EPS
estimates with revision history and breadth, forward-filled to daily rows.

Called from fetchBulkData.py after the core EPS/financial data merge.

Public API
----------
fetch_av_earnings_estimates(symbol, api_key) -> pd.DataFrame
    Raw quarterly estimates from AV.

build_daily_estimate_features(symbol, api_key, daily_df, fiscal_to_report_map=None)
    -> (augmented daily_df, raw quarterly df)
    Main entry point: fetch, align, compute Tier-1 features, forward-fill.

_get_fiscal_to_report_map(symbol, api_key) -> dict
    Build {fiscal_date_str -> report_date_str} by fetching AV EARNINGS.
    Used internally if caller doesn't supply the mapping.
"""

import numpy as np
import pandas as pd
import requests


# ── Column rename map (AV name → internal short name) ────────────────────────

_COL_MAP = {
    "date": "fiscal_date",
    "horizon": "horizon",
    "eps_estimate_average": "eps_est_avg",
    "eps_estimate_high": "eps_est_high",
    "eps_estimate_low": "eps_est_low",
    "eps_estimate_analyst_count": "eps_est_analyst_count",
    "eps_estimate_average_7_days_ago": "eps_est_avg_7d",
    "eps_estimate_average_30_days_ago": "eps_est_avg_30d",
    "eps_estimate_average_60_days_ago": "eps_est_avg_60d",
    "eps_estimate_average_90_days_ago": "eps_est_avg_90d",
    "eps_estimate_revision_up_trailing_7_days": "eps_rev_up_7",
    "eps_estimate_revision_down_trailing_7_days": "eps_rev_down_7",
    "eps_estimate_revision_up_trailing_30_days": "eps_rev_up_30",
    "eps_estimate_revision_down_trailing_30_days": "eps_rev_down_30",
    "revenue_estimate_average": "rev_est_avg",
    "revenue_estimate_high": "rev_est_high",
    "revenue_estimate_low": "rev_est_low",
    "revenue_estimate_analyst_count": "rev_est_analyst_count",
}

# Feature columns that get forward-filled into daily_df
DAILY_FEATURE_COLS = [
    "eps_est_avg",
    "eps_est_analyst_count",
    "eps_rev_7_pct",
    "eps_rev_30_pct",
    "eps_rev_accel",
    "eps_breadth_ratio_7",
    "eps_breadth_ratio_30",
    "eps_dispersion",
    "log_analyst_count",
]

# Cols knowable when Q_i is REPORTED (anchor = report_date[i]).
# Forward-looking estimates: valid to activate when the prior quarter closes.
_ESTIMATE_COLS = ["eps_est_avg", "eps_est_analyst_count", "log_analyst_count"]

# Cols only knowable when Q_{i+1} is REPORTED (anchor = report_date[i+1]).
# Revision/breadth stats are within-quarter snapshots; anchoring them at report_date[i]
# would broadcast end-of-Q_{i+1} revision activity back to the start of Q_{i+1} — look-ahead bias.
_REVISION_COLS = [c for c in DAILY_FEATURE_COLS if c not in _ESTIMATE_COLS]


# ── Fetching ──────────────────────────────────────────────────────────────────

def fetch_av_earnings_estimates(symbol: str, api_key: str) -> pd.DataFrame:
    """
    Fetch quarterly earnings estimates from AV EARNINGS_ESTIMATES endpoint.

    Returns a DataFrame with one row per quarterly period (historical + current
    forward estimate). Columns are renamed to short internal names (see _COL_MAP).
    Numeric columns are cast to float; nulls become NaN.
    """
    import fetch_cache as _fc
    _cached = _fc.load(f'AV_EPS_EST_{symbol}', 'symbol')
    if _cached is not None:
        print(f'[CACHE HIT disk] EPS_EST {symbol}')
        return _cached

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "EARNINGS_ESTIMATES",
        "symbol": symbol,
        "apikey": api_key,
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "estimates" not in data:
        raise ValueError(
            f"[{symbol}] EARNINGS_ESTIMATES: unexpected response keys: {list(data.keys())}"
        )

    df = pd.DataFrame(data["estimates"]).rename(columns=_COL_MAP)
    df["fiscal_date"] = pd.to_datetime(df["fiscal_date"])

    numeric_cols = [c for c in df.columns if c not in ("fiscal_date", "horizon")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    result = df.sort_values("fiscal_date").reset_index(drop=True)
    _fc.save(f'AV_EPS_EST_{symbol}', 'symbol', result)
    return result


def _get_fiscal_to_report_map(symbol: str, api_key: str) -> dict:
    """
    Fetch AV EARNINGS to build {fiscal_date_str -> report_date_str} mapping.

    Keys and values are ISO date strings ('YYYY-MM-DD').
    Used when the caller does not supply the mapping from an already-fetched
    EARNINGS response.
    """
    url = "https://www.alphavantage.co/query"
    r = requests.get(
        url,
        params={"function": "EARNINGS", "symbol": symbol, "apikey": api_key},
        timeout=100,
    )
    r.raise_for_status()
    data = r.json()

    mapping = {}
    for q in data.get("quarterlyEarnings", []):
        fde = q.get("fiscalDateEnding", "").strip()
        rd = q.get("reportedDate", "").strip()
        if fde and rd:
            mapping[fde] = rd

    return mapping


# ── Feature computation (quarterly grain) ────────────────────────────────────

def _compute_tier1_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Tier-1 derived features at the quarterly (row) level.

    All operations are element-wise; df is modified in-place and returned.
    NaN inputs propagate correctly — breadth nulls are treated as 0.

    Features added
    --------------
    eps_rev_7_pct       pct change in consensus over trailing 7 days
    eps_rev_30_pct      pct change in consensus over trailing 30 days
    eps_breadth_ratio_7   net analyst conviction score (-1..+1), 7-day window
    eps_breadth_ratio_30  net analyst conviction score (-1..+1), 30-day window
    eps_dispersion      (high - low) / |avg|  — analyst disagreement
    log_analyst_count   log1p(analyst_count)
    """
    df = df.copy()
    eps = df["eps_est_avg"]

    # Revision magnitude
    df["eps_rev_7_pct"] = (eps - df["eps_est_avg_7d"]) / (df["eps_est_avg_7d"].abs() + 1e-9)
    df["eps_rev_30_pct"] = (eps - df["eps_est_avg_30d"]) / (df["eps_est_avg_30d"].abs() + 1e-9)

    # Breadth — treat null revision counts as 0
    up7  = df["eps_rev_up_7"].fillna(0)
    dn7  = df["eps_rev_down_7"].fillna(0)
    up30 = df["eps_rev_up_30"].fillna(0)
    dn30 = df["eps_rev_down_30"].fillna(0)

    df["eps_breadth_ratio_7"]  = (up7  - dn7)  / (up7  + dn7  + 1)
    df["eps_breadth_ratio_30"] = (up30 - dn30) / (up30 + dn30 + 1)

    # Dispersion — analyst disagreement normalised by |average|
    df["eps_dispersion"] = (df["eps_est_high"] - df["eps_est_low"]) / (eps.abs() + 1e-9)

    # Coverage (log scale to dampen outliers)
    df["log_analyst_count"] = np.log1p(df["eps_est_analyst_count"].fillna(0))

    # Revision acceleration: is revision momentum speeding up?
    df["eps_rev_accel"] = df["eps_rev_7_pct"] - df["eps_rev_30_pct"]

    return df


# ── Alignment & forward-fill ──────────────────────────────────────────────────

def _build_anchor_series(
    quarterly: pd.DataFrame,
    fiscal_to_report_map: dict,
    next_q_row: pd.Series | None,
) -> tuple:
    """
    Build two anchor DataFrames: one for estimate cols, one for revision cols.

    Estimate cols (anchor = report_date[i]):
      When Q_i is reported, Q_{i+1}'s forward estimates become "active".

    Revision cols (anchor = report_date[i+1]):
      Revision/breadth stats for Q_{i+1} are only knowable once Q_{i+1} is
      reported — anchoring earlier would broadcast end-of-quarter activity
      back to the start of the quarter (look-ahead bias).
      Exception: next_q_row (current forward estimate) uses report_date[i]
      since its revision stats reflect today's actual state.

    Returns
    -------
    (est_anchors, rev_anchors) : two DataFrames sorted by anchor_date.
    """
    def _report_date(row):
        fiscal_str = row["fiscal_date"].strftime("%Y-%m-%d")
        rd = fiscal_to_report_map.get(fiscal_str)
        return pd.to_datetime(rd) if rd else row["fiscal_date"] + pd.Timedelta(days=45)

    est_rows = []
    rev_rows = []
    n = len(quarterly)

    for i in range(n):
        est_anchor = _report_date(quarterly.iloc[i])

        if i + 1 < n:
            feature_src = quarterly.iloc[i + 1]
            rev_anchor = _report_date(feature_src)
        elif next_q_row is not None:
            feature_src = next_q_row
            rev_anchor = est_anchor  # current snapshot — no look-ahead
        else:
            continue

        est_rec = {"anchor_date": est_anchor}
        for col in _ESTIMATE_COLS:
            est_rec[col] = feature_src.get(col, np.nan)
        est_rows.append(est_rec)

        rev_rec = {"anchor_date": rev_anchor}
        for col in _REVISION_COLS:
            rev_rec[col] = feature_src.get(col, np.nan)
        rev_rows.append(rev_rec)

    _empty_est = pd.DataFrame(columns=["anchor_date"] + _ESTIMATE_COLS)
    _empty_rev = pd.DataFrame(columns=["anchor_date"] + _REVISION_COLS)

    est_df = pd.DataFrame(est_rows).sort_values("anchor_date").reset_index(drop=True) if est_rows else _empty_est
    rev_df = pd.DataFrame(rev_rows).sort_values("anchor_date").reset_index(drop=True) if rev_rows else _empty_rev

    return est_df, rev_df


def _forward_fill_anchors(
    daily_df: pd.DataFrame,
    anchors: pd.DataFrame,
    cols: list,
) -> pd.DataFrame:
    """
    Write feature values from each anchor into daily_df rows.

    Rows from anchor_date[i] (inclusive) up to anchor_date[i+1] (exclusive)
    receive anchor[i]'s values.  Rows before the first anchor stay NaN.

    Only the columns listed in `cols` are written; others are left untouched.
    """
    daily = daily_df.copy()
    daily["_date"] = pd.to_datetime(daily["date"])

    for col in cols:
        if col not in daily.columns:
            daily[col] = np.nan

    for i, anchor in anchors.iterrows():
        lo = anchor["anchor_date"]
        mask = daily["_date"] >= lo
        if i + 1 < len(anchors):
            hi = anchors.iloc[i + 1]["anchor_date"]
            mask &= daily["_date"] < hi

        for col in cols:
            daily.loc[mask, col] = anchor[col]

    daily.drop(columns=["_date"], inplace=True)
    return daily


# ── Main entry point ──────────────────────────────────────────────────────────

def build_daily_estimate_features(
    symbol: str,
    api_key: str,
    daily_df: pd.DataFrame,
    fiscal_to_report_map: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch AV earnings estimates, align to trading days, compute Tier-1
    features, forward-fill, and merge into daily_df.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. 'NVDA').
    api_key : str
        Alpha Vantage API key.
    daily_df : pd.DataFrame
        Daily price/feature DataFrame with a 'date' column (str 'YYYY-MM-DD').
        Returned augmented with new feature columns.
    fiscal_to_report_map : dict or None
        Pre-built {fiscal_date_str -> report_date_str} mapping.
        If None, fetched from AV EARNINGS endpoint (one extra API call).

    Returns
    -------
    (augmented_daily_df, raw_quarterly_df)
        augmented_daily_df : daily_df with DAILY_FEATURE_COLS added.
        raw_quarterly_df   : quarterly estimates DataFrame (with Tier-1 features).
    """
    print(f">  [EPS estimates] Fetching EARNINGS_ESTIMATES for {symbol}...")
    raw = fetch_av_earnings_estimates(symbol, api_key)

    # Separate quarterly vs forward annual estimates
    quarterly_mask = raw["horizon"].isin(["historical fiscal quarter", "next fiscal quarter"])
    quarterly_hist = raw[quarterly_mask & (raw["horizon"] == "historical fiscal quarter")].copy()
    next_q_rows    = raw[raw["horizon"] == "next fiscal quarter"]

    # Compute Tier-1 features on all quarterly rows
    quarterly_hist = _compute_tier1_features(quarterly_hist)
    quarterly_hist = quarterly_hist.sort_values("fiscal_date").reset_index(drop=True)

    next_q_row = None
    if not next_q_rows.empty:
        next_q_row = _compute_tier1_features(next_q_rows.head(1)).iloc[0]

    # Build fiscal → report date mapping if not supplied
    if fiscal_to_report_map is None:
        print(f">  [EPS estimates] Fetching EARNINGS to build fiscal→report mapping...")
        fiscal_to_report_map = _get_fiscal_to_report_map(symbol, api_key)

    # Build anchor series and forward-fill (two passes: estimates then revisions)
    est_anchors, rev_anchors = _build_anchor_series(quarterly_hist, fiscal_to_report_map, next_q_row)
    augmented = _forward_fill_anchors(daily_df,  est_anchors, _ESTIMATE_COLS)
    augmented = _forward_fill_anchors(augmented, rev_anchors, _REVISION_COLS)

    covered = augmented["eps_est_avg"].notna().sum()
    print(
        f">  [EPS estimates] Added {len(DAILY_FEATURE_COLS)} features for {symbol}. "
        f"Coverage: {covered}/{len(augmented)} rows ({100*covered/max(len(augmented),1):.1f}%)."
    )

    return augmented, quarterly_hist
