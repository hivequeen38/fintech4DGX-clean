"""
Integration test: EPS analyst-estimate features for NVDA.

Fetches live data from Alpha Vantage EARNINGS_ESTIMATES, aligns quarterly
estimates to daily rows (forward-fill), writes enriched columns into
NVDA_TMP.csv alongside the existing hand-calculated estEPS, then reports
the comparison between old and new series.

Prerequisites:
    NVDA_TMP.csv must exist (produced by a prior training run).

Run with:
    pytest test_eps_estimates_nvda.py -v -s

Steps performed:
    3a.  Fetch & massage quarterly estimates so forward-fill is coherent.
    3b.  Add new columns to NVDA_TMP.csv (estEPS left untouched).
    3c.  Report old estEPS vs new eps_est_avg.

Split-adjustment note
---------------------
AV EARNINGS_ESTIMATES returns EPS in the historical unit at the time — i.e.,
pre-split values for dates before the split.  NVDA had two splits:
    2021-07-19  4:1 (cumulative factor 40x relative to today)
    2024-06-10  10:1 (cumulative factor 10x relative to today)

The existing estEPS column in NVDA_TMP.csv is FULLY split-adjusted (divided
by 40 for 2021 data, by 10 for 2022-2024 data).  eps_est_avg carries the
raw historical unit.  A split-adjusted version (eps_est_avg_adj) is computed
and included for apples-to-apples comparison.
"""

import os
import pytest
import numpy as np
import pandas as pd

import trendConfig
import fetch_eps_estimates

# ── Constants ─────────────────────────────────────────────────────────────────

SYMBOL = "NVDA"
TMP_CSV = f"{SYMBOL}_TMP.csv"
START_DATE = "2021-03-01"          # matches NVDA_param reference start_date

# NVDA split dates and CUMULATIVE inverse factors relative to today
# (to convert historical units → current split-adjusted units, divide by factor)
NVDA_SPLIT_SCHEDULE = [
    # (split_date, cumulative_factor_on_that_date)
    # Before 2021-07-19: price was 40x current  → EPS was 40x current unit
    # 2021-07-19 to 2024-06-10: factor 10x
    # After 2024-06-10: factor 1x
    ("2021-07-19", 40),
    ("2024-06-10", 10),
]


def _apply_split_adjustment(series: pd.Series, date_series: pd.Series) -> pd.Series:
    """
    Return split-adjusted EPS: divide raw value by the cumulative factor
    that was in effect on that date.

    Parameters
    ----------
    series : pd.Series of float — raw eps_est_avg (historical units)
    date_series : pd.Series of datetime64 — corresponding daily dates
    """
    adjusted = series.copy()

    split_2021 = pd.Timestamp("2021-07-19")
    split_2024 = pd.Timestamp("2024-06-10")

    pre_2021  = date_series < split_2021
    mid_split = (date_series >= split_2021) & (date_series < split_2024)
    # post_2024: no adjustment needed

    adjusted[pre_2021]  = adjusted[pre_2021]  / 40
    adjusted[mid_split] = adjusted[mid_split] / 10

    return adjusted


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def api_key():
    return trendConfig.config["alpha_vantage"]["key"]


@pytest.fixture(scope="module")
def tmp_df():
    """Load existing NVDA_TMP.csv (must exist from a prior run)."""
    assert os.path.exists(TMP_CSV), (
        f"{TMP_CSV} not found. Run a full NVDA training pass first."
    )
    df = pd.read_csv(TMP_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def raw_estimates(api_key):
    """Fetch raw quarterly EARNINGS_ESTIMATES from AV."""
    return fetch_eps_estimates.fetch_av_earnings_estimates(SYMBOL, api_key)


@pytest.fixture(scope="module")
def fiscal_map(api_key):
    """Build fiscal_date → report_date mapping from AV EARNINGS."""
    return fetch_eps_estimates._get_fiscal_to_report_map(SYMBOL, api_key)


@pytest.fixture(scope="module")
def enriched(tmp_df, api_key, fiscal_map):
    """
    3a: fetch + align.  3b: add new columns alongside existing estEPS.
    Returns augmented DataFrame (not yet written to CSV).
    """
    daily_str_dates = tmp_df.copy()
    # build_daily_estimate_features expects 'date' as str YYYY-MM-DD
    daily_str_dates["date"] = daily_str_dates["date"].dt.strftime("%Y-%m-%d")

    augmented, _ = fetch_eps_estimates.build_daily_estimate_features(
        symbol=SYMBOL,
        api_key=api_key,
        daily_df=daily_str_dates,
        fiscal_to_report_map=fiscal_map,
    )

    # Convert date back to datetime for downstream use
    augmented["date"] = pd.to_datetime(augmented["date"])

    # 3a: add split-adjusted eps_est_avg for comparison
    augmented["eps_est_avg_adj"] = _apply_split_adjustment(
        augmented["eps_est_avg"], augmented["date"]
    )

    return augmented


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_raw_estimates_fetched(raw_estimates):
    """EARNINGS_ESTIMATES must return rows for NVDA."""
    hist = raw_estimates[raw_estimates["horizon"] == "historical fiscal quarter"]
    assert len(hist) >= 10, f"Expected ≥10 historical quarters, got {len(hist)}"
    assert "eps_est_avg" in hist.columns
    assert hist["eps_est_avg"].notna().sum() >= 5


def test_raw_estimates_cover_start_date(raw_estimates):
    """Historical estimates should pre-date our training window."""
    hist = raw_estimates[raw_estimates["horizon"] == "historical fiscal quarter"]
    earliest = hist["fiscal_date"].min()
    assert earliest <= pd.Timestamp(START_DATE), (
        f"Earliest estimate {earliest.date()} is after start_date {START_DATE}. "
        "Forward-fill will leave the early training rows NaN."
    )


def test_fiscal_map_populated(fiscal_map):
    """Fiscal→report mapping must cover NVDA quarterly history."""
    assert len(fiscal_map) >= 10, (
        f"Only {len(fiscal_map)} fiscal→report date entries; expected ≥10."
    )


def test_forward_fill_coverage(enriched):
    """
    eps_est_avg must be non-NaN for the majority of daily rows.
    (Some NaN at the very start, before the first anchor date, is expected.)
    """
    coverage = enriched["eps_est_avg"].notna().mean()
    assert coverage >= 0.85, (
        f"eps_est_avg only covers {100*coverage:.1f}% of rows. "
        "Forward-fill may be misaligned."
    )


def test_tier1_features_present(enriched):
    """All Tier-1 derived feature columns must exist and be mostly non-NaN."""
    for col in fetch_eps_estimates.DAILY_FEATURE_COLS:
        assert col in enriched.columns, f"Missing column: {col}"
        nn = enriched[col].notna().mean()
        assert nn >= 0.85, f"{col} is only {100*nn:.1f}% populated."


def test_dispersion_non_negative(enriched):
    """eps_dispersion = (high - low)/|avg| must be ≥ 0."""
    valid = enriched["eps_dispersion"].dropna()
    assert (valid >= 0).all(), "eps_dispersion has negative values — check formula."


def test_breadth_ratios_bounded(enriched):
    """Breadth ratios must lie in [-1, 1]."""
    for col in ("eps_breadth_ratio_7", "eps_breadth_ratio_30"):
        valid = enriched[col].dropna()
        assert valid.between(-1, 1).all(), f"{col} has values outside [-1, 1]."


def test_write_enriched_to_tmp_csv(enriched, tmp_df):
    """
    3b: Write enriched DataFrame back to NVDA_TMP.csv.
    New columns are added; existing estEPS is preserved unchanged.
    """
    # Safety check: estEPS must be unchanged
    original_est = tmp_df["estEPS"].values
    enriched_est = enriched["estEPS"].values
    assert np.allclose(
        np.nan_to_num(original_est, nan=-999),
        np.nan_to_num(enriched_est, nan=-999),
    ), "estEPS was modified — should not happen."

    # Write back using original date strings
    out = enriched.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(TMP_CSV, index=False)
    print(f"\n[3b] Wrote {len(out)} rows with {len(out.columns)} columns to {TMP_CSV}.")
    assert os.path.exists(TMP_CSV)


def test_comparison_report(enriched, capsys):
    """
    3c: Report old estEPS vs new eps_est_avg_adj (split-adjusted).

    Comparison windows:
      Full period: 2021-03-01 onward (different scale pre/post splits).
      Post-2024-06-10: both on identical 10:1-adjusted basis.
    """
    df = enriched.copy()
    both = df[df["estEPS"].notna() & df["eps_est_avg_adj"].notna()].copy()
    post_split = both[both["date"] >= pd.Timestamp("2024-06-10")]

    print("\n" + "=" * 72)
    print("  EPS ESTIMATE COMPARISON REPORT: estEPS vs eps_est_avg (AV)")
    print("=" * 72)
    print(f"  Symbol         : {SYMBOL}")
    print(f"  Daily rows     : {len(df)}")
    print(f"  Overlapping    : {len(both)} rows (both estEPS & eps_est_avg non-NaN)")
    print(f"  Post-2024-06   : {len(post_split)} rows (same split-adjusted scale)")
    print()

    # ── Full-period stats ──────────────────────────────────────────────────
    if len(both) > 10:
        corr_full = both["estEPS"].corr(both["eps_est_avg_adj"])
        mae_full  = (both["estEPS"] - both["eps_est_avg_adj"]).abs().mean()
        print(f"  Full period")
        print(f"    Pearson correlation : {corr_full:.4f}")
        print(f"    Mean abs error      : {mae_full:.6f}")

    print()

    # ── Post-June-2024 stats (same scale) ─────────────────────────────────
    if len(post_split) > 10:
        corr_post = post_split["estEPS"].corr(post_split["eps_est_avg_adj"])
        mae_post  = (post_split["estEPS"] - post_split["eps_est_avg_adj"]).abs().mean()
        print(f"  Post-2024-06 (identical split basis)")
        print(f"    Pearson correlation : {corr_post:.4f}")
        print(f"    Mean abs error      : {mae_post:.6f}")
        print()

        # Sample table
        sample = post_split[["date", "estEPS", "eps_est_avg_adj", "eps_rev_30_pct",
                              "eps_breadth_ratio_30", "eps_dispersion"]].iloc[::max(1, len(post_split)//10)]
        sample = sample.copy()
        sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
        sample.columns = ["date", "estEPS(old)", "est_avg(new)", "rev30%", "breadth30", "disp"]
        print("  Sample rows (post-2024-06):")
        print("  " + sample.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print()
    print("  NOTE: eps_est_avg_adj applies split factors (÷40 pre-2021-07,")
    print("        ÷10 pre-2024-06) to bring raw AV values to current-share basis.")
    print("=" * 72 + "\n")

    # This test always passes — its job is to produce the report
    assert len(both) > 0, "No overlapping rows found — check alignment."
