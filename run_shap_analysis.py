"""
One-time Permutation Feature Importance Analysis
Uses the already-saved TMP.csv + scaler.joblib from training.
Loads all 15 saved models, averages their feature importance,
and outputs a ranked CSV + bar chart.

Usage:
    python run_shap_analysis.py                        # NVDA reference
    python run_shap_analysis.py NVDA reference
    python run_shap_analysis.py NVDA AAII_option_vol_ratio
    python run_shap_analysis.py CRDO reference
"""

import sys, os, importlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from joblib import load
from datetime import datetime
from tqdm import tqdm

# ── config ─────────────────────────────────────────────────────────────────────
SYMBOL    = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
PARAM_SET = sys.argv[2] if len(sys.argv) > 2 else "reference"
MODEL_DIR = "/workspace/model"
OUT_DIR   = "/workspace"
N_MODELS  = 15
# ──────────────────────────────────────────────────────────────────────────────

param_module = importlib.import_module(f"{SYMBOL}_param")
param        = getattr(param_module, PARAM_SET)
model_name   = param["model_name"]
selected_cols = param["selected_columns"]          # first is 'label'
features      = [c for c in selected_cols if c != "label"]

print(f"\n=== Permutation Feature Importance: {SYMBOL} / {PARAM_SET} ===")
print(f"Features  : {len(features)}")

# ── Load pre-built TMP.csv ─────────────────────────────────────────────────────
tmp_path = os.path.join(OUT_DIR, f"{SYMBOL}_TMP.csv")
if not os.path.exists(tmp_path):
    sys.exit(f"ERROR: {tmp_path} not found — run training first.")

df = pd.read_csv(tmp_path)
print(f"TMP rows  : {len(df)}")

# Apply date filters from param
df = df[df["date"] >= param["start_date"]]
if param.get("end_date"):
    df = df[df["date"] <= param["end_date"]]

# Keep only the selected columns that exist in TMP
cols_available = [c for c in selected_cols if c in df.columns]
missing = set(selected_cols) - set(cols_available)
if missing:
    print(f"⚠️  Columns missing from TMP (will be skipped): {missing}")
df = df[cols_available].dropna()
print(f"Rows after filter/dropna: {len(df)}")

# ── Load saved scaler ──────────────────────────────────────────────────────────
scaler_path = os.path.join(OUT_DIR, f"{SYMBOL}_{model_name}_scaler.joblib")
if not os.path.exists(scaler_path):
    sys.exit(f"ERROR: scaler not found at {scaler_path}")

scaler = load(scaler_path)
print(f"Scaler    : {scaler_path}")

feature_data = df[[c for c in cols_available if c != "label"]]
label_data   = df["label"].astype(int).values

# Re-align features list to only those present
features = feature_data.columns.tolist()
print(f"Features (post-filter): {len(features)}")

X_scaled = scaler.transform(feature_data.values).astype(np.float32)
y        = label_data

# Simple 80/20 split (same order as data — no shuffle for time-series fairness)
split    = int(len(X_scaled) * 0.8)
X_test   = torch.FloatTensor(X_scaled[split:])
y_test   = torch.LongTensor(y[split:])
print(f"Test set  : {len(X_test)} samples")

# ── Load model architecture ────────────────────────────────────────────────────
from trendAnalysisFromTodayNew import TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device    : {device}")

def load_model(path):
    # Models are saved as full objects (torch.save(model, path))
    m = torch.load(path, map_location=device)
    m.to(device)
    m.eval()
    return m

# ── Permutation importance ─────────────────────────────────────────────────────
def predict(model, X_d):
    """Run model and return class predictions, handling [B,1,C] or [B,C] output."""
    out = model(X_d)
    if out.dim() == 3:
        out = out.squeeze(1)   # [B, 1, C] → [B, C]
    return out.argmax(1)

def permutation_importance(model, X, y):
    """Drop in accuracy when each feature is shuffled — higher = more important."""
    X_d, y_d = X.to(device), y.to(device)
    with torch.no_grad():
        base_acc = (predict(model, X_d) == y_d).float().mean().item()

    X_np = X.numpy()
    importances = np.zeros(X_np.shape[1])
    for i in tqdm(range(X_np.shape[1]), desc="  features", leave=False):
        X_perm       = X_np.copy()
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        with torch.no_grad():
            perm_acc = (predict(model, torch.FloatTensor(X_perm).to(device)) == y_d).float().mean().item()
        importances[i] = base_acc - perm_acc   # positive = feature helps

    total = np.sum(np.abs(importances))
    return importances / total if total > 0 else importances

# ── Iterate over all 15 models ─────────────────────────────────────────────────
all_imps = []

for idx in range(1, N_MODELS + 1):
    path = os.path.join(MODEL_DIR, f"model_{SYMBOL}_{model_name}_fixed_noTimesplit_{idx}.pth")
    if not os.path.exists(path):
        print(f"  [skip] model {idx} not found")
        continue
    print(f"\n  Model {idx}/{N_MODELS}")
    model = load_model(path)
    imp   = permutation_importance(model, X_test, y_test)
    all_imps.append(imp)

if not all_imps:
    sys.exit("ERROR: No models found. Check MODEL_DIR and model filenames.")

avg_imp = np.mean(all_imps, axis=0)
std_imp = np.std(all_imps,  axis=0)

# ── Build ranked output ────────────────────────────────────────────────────────
result_df = pd.DataFrame({
    "feature":    features,
    "importance": avg_imp,
    "std":        std_imp,
}).sort_values("importance", ascending=False).reset_index(drop=True)
result_df.index += 1
result_df.index.name = "rank"

timestamp = datetime.now().strftime("%Y%m%dT%H%M")
csv_path  = os.path.join(OUT_DIR, f"{SYMBOL}_{model_name}_feature_importance_{timestamp}.csv")
result_df.to_csv(csv_path)
print(f"\nCSV → {csv_path}")

# ── Print ranked table ─────────────────────────────────────────────────────────
print(f"\n{'Rank':>4}  {'Feature':<35}  {'Importance':>10}  {'Std':>8}")
print("─" * 65)
for rank, row in result_df.iterrows():
    flag = " ◄ (negative — may hurt)" if row["importance"] < -0.001 else ""
    print(f"{rank:>4}  {row['feature']:<35}  {row['importance']:>10.6f}  {row['std']:>8.6f}{flag}")

# ── Bar chart ──────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_n   = min(30, len(result_df))
    plot_df = result_df.head(top_n).sort_values("importance")
    colors  = ["#d62728" if v < 0 else "#1f77b4" for v in plot_df["importance"]]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.32)))
    ax.barh(plot_df["feature"], plot_df["importance"],
            color=colors, xerr=plot_df["std"], capsize=3)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(
        f"{SYMBOL} {model_name} — Top {top_n} Feature Importance\n"
        f"(avg over {len(all_imps)} models, permutation method)",
        fontsize=12
    )
    ax.set_xlabel("Accuracy drop when feature is shuffled (normalised)")
    plt.tight_layout()

    png_path = os.path.join(OUT_DIR, f"{SYMBOL}_{model_name}_feature_importance_{timestamp}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart → {png_path}")
except Exception as e:
    print(f"Chart skipped: {e}")

print(f"\nDone. Analysed {len(all_imps)}/{N_MODELS} models.")
