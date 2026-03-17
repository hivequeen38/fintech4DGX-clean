# Turn On Trans-MZ Inference (Backlog B-MH5)

## Context

Trans-MZ (multi-horizon multi-zone) models are trained nightly for NVDA and PLTR but
inference is currently **skipped** in Phase 2 — no prediction rows are written to
`{STOCK}_15d_from_today_predictions.csv` and nothing appears in the HTML report.

Backlog item: **B-MH5** — implement `inference()` support for `model_type='trans_mz'`
in `mainDeltafromToday.py`.

---

## How to Resume GBDT Training + HTML Upload (if nightly crashes mid-run)

If the nightly run dies after trans_mz training but before GBDT training (as happened
2026-03-07), run this from inside the GPU container (`docker exec -it c4c4bd838fcf bash`):

```bash
nohup python3 -c "
import NVDA_gbdt_param, PLTR_gbdt_param, APP_gbdt_param, CRDO_gbdt_param, INOD_gbdt_param
import gbdt_pipeline, get_historical_html

GBDT_PARAMS = [
    NVDA_gbdt_param.lgbm_reference,
    PLTR_gbdt_param.lgbm_reference,
    APP_gbdt_param.lgbm_reference,
    CRDO_gbdt_param.lgbm_reference,
    INOD_gbdt_param.lgbm_reference,
]
for param in GBDT_PARAMS:
    sym, name = param['symbol'], param['model_name']
    print(f'--- GBDT train {sym}/{name} ---', flush=True)
    try:
        accepted = gbdt_pipeline.train(param)
        print(f'[{\"ACCEPTED\" if accepted else \"REJECTED\"}] {sym}/{name}', flush=True)
    except Exception as e:
        print(f'[ERROR] {sym}/{name}: {e}', flush=True)

print('--- upload_all_results ---', flush=True)
get_historical_html.upload_all_results('YYYY-MM-DD', upload_to_cloud=True)
print('Done.', flush=True)
" > /workspace/gbdt_resume_$(date +%Y%m%d).log 2>&1 &
echo "PID: $!"
```

> **Replace `YYYY-MM-DD`** with the actual run date (e.g. `2026-03-06`).

Monitor with:
```bash
tail -f /workspace/gbdt_resume_$(date +%Y%m%d).log
```

---

## What Needs to Be Implemented (B-MH5)

To make trans_mz predictions visible in the report:

1. **`mainDeltafromToday.inference()`** — add a branch for `model_type='trans_mz'`
   that calls the MZ inference path (similar to how `multiZoneAnalysisNew.py` runs
   during training) and writes p1..p15 rows to `{STOCK}_15d_from_today_predictions.csv`.

2. **`nightly_run.py` TICKER_PARAMS** — remove the `# Phase 1 skipped (B-MH5)` skip
   comments once inference is implemented:
   ```python
   (NVDA_param.mz_reference, 'trans_mz'),  # currently skipped
   (PLTR_param.mz_reference, 'trans_mz'),  # currently skipped
   ```

3. **`get_historical_html.py`** — trans_mz rows will automatically get the `model_type`
   column once written; row styling will fall through to the `reference` style unless a
   dedicated CSS class is added (optional).
