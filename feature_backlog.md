# Feature Backlog

Features considered but not yet added to selected_columns.
Prioritized by expected signal value.

---

## EPS Analyst Estimate features (AV EARNINGS_ESTIMATES)

**Already implemented** (in `fetch_eps_estimates.py`, all available in TMP CSVs):
All columns listed here are already computed and forward-filled into the daily
DataFrame by `fetchBulkData.py`. Adding them to `selected_columns` is all that
is needed to activate them for training.

**Active in all NVDA profiles** (as of 2026-02-24):
- `eps_est_avg` — raw AV consensus average EPS estimate
- `eps_rev_30_pct` — % revision in consensus over 30 days
- `eps_rev_7_pct` — % revision in consensus over 7 days
- `eps_breadth_ratio_30` — net analyst conviction, 30-day window
- `eps_dispersion` — (high−low)/|avg|, analyst disagreement

### Tier 2 — Add next if Tier-1 shows improvement

| Feature | Formula / source | Notes |
|---|---|---|
| `eps_rev_accel` | `eps_rev_7_pct − eps_rev_30_pct` | Is revision momentum accelerating? Compute in fetch_eps_estimates._compute_tier1_features |
| `eps_breadth_ratio_7` | `(up_7 − down_7) / (up_7 + down_7 + 1)` | Already computed, not yet in selected_columns |
| `log_analyst_count` | `log1p(analyst_count)` | Coverage proxy (liquidity / attention). Already computed. |
| `rev_rev_30_pct` | Same formula for revenue estimate avg | Needs `rev_est_avg_30d` field — AV doesn't return revenue revision history, only current avg. Skip for now. |

### Tier 3 — Cross-features (after Tier 2 validated)

| Feature | Formula | Notes |
|---|---|---|
| `eps_revision_signal` | `eps_rev_30_pct / (eps_dispersion + 0.01)` | Revision scaled by uncertainty — high signal when both large & confident |
| `eps_rev_breadth_cross` | `eps_rev_30_pct × eps_breadth_ratio_30` | Magnitude × analyst conviction — needs compute step |

---

## Other feature ideas (non-EPS-estimates)

| Feature | Source | Notes |
|---|---|---|
| `insider_buy_sell_ratio` | SEC Form 4 / OpenInsider | Strong signal but complex to fetch reliably |
| `short_interest_ratio` | FINRA / yfinance | Days-to-cover; monthly cadence |
| `put_call_skew` | Options chain | Relative OTM put vs call IV — proxy for tail risk expectation |
| `sector_rotation_signal` | XLK vs SPY relative strength | Semiconductor sector flow |
| `news_sentiment_score` | AV NEWS_SENTIMENT (already fetched) | Currently not in selected_columns for most profiles |

---

## Notes on activation

To activate any backlog feature:
1. Verify it exists in `{SYMBOL}_TMP.csv` (it may already be computed)
2. Add the column name to `selected_columns` in the relevant `{SYMBOL}_param.py`
3. Run the integration test or smoke test to confirm no KeyError
4. Re-run training and compare F1 against baseline

**Baseline** (NVDA reference, target_size=1, end_date=2026-02-23, post-bug-fix):
- F1 class 0 (neutral): 0.716
- F1 class 1 (UP): 0.214
- F1 class 2 (DN): 0.212
