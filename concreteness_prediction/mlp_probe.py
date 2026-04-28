"""
mlp_probe_correlation.py
============================
Evaluates how well per-layer contextual embeddings encode lexical concreteness
by training a multi-layer perceptron (MLP) probe on each transformer layer and
measuring its correlation with ground-truth concreteness scores.

Pipeline overview
-----------------
1. For each transformer layer, load the pre-extracted embeddings from the
   corresponding `layer_N.jsonl` file (produced by the extraction scripts).
2. Standardise the embeddings with a StandardScaler.
3. Run 10-fold cross-validated MLP regression to predict concreteness scores.
4. Evaluate each layer using Pearson r, Spearman ρ, and MSE.
5. Save layer-wise results to a CSV and plot the correlation profile.

Concreteness scores
-------------------
Ground-truth concreteness ratings come from:

    Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014).
    Concreteness ratings for 40 thousand generally known English word lemmas.
    Behavior Research Methods, 46(3), 904–911.
    https://doi.org/10.3758/s13428-013-0403-5

    ► Download the norms from the paper's supplementary materials or from:
      https://link.springer.com/article/10.3758/s13428-013-0403-5
      These scores are embedded in the `.jsonl` files under the `conc` field
      (written there during the extraction step).

Input files required
--------------------
- A folder of per-layer `.jsonl` files produced by one of the extraction
  scripts (e.g. `extract_llama_embeddings.py`).
  Each line must be a JSON record:
      {"word": ..., "sentence": ..., "conc": ..., "embedding": [...]}

Output
------
- `mlp_probe_results.csv`       : layer-wise Pearson, Spearman, and MSE scores
- `layer_wise_correlation.png`  : plot of Pearson r and Spearman ρ across layers
"""

# ── Standard library ────────────────────────────────────────────────────────
import os
import json

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# 1.  CONFIGURATION
# ============================================================================
# ► Set FOLDER to the embedding directory for the model you want to evaluate:
#       LLaMA-3.1-8B  →  "llama_emb"   (N_LAYERS = 32)
#       Qwen3-8B      →  "qwen_emb"    (N_LAYERS = 36)
#       Gemma-2-9B    →  "gemma_emb"   (N_LAYERS = 42)
#       GPT-OSS-20B   →  "gpt_emb"     (N_LAYERS = 24)

FOLDER   = "llama_emb"   # ← change to switch model
N_LAYERS = 32            # ← set to match the model above
N_SPLITS = 10            # number of cross-validation folds
SAVE_PLOT = True         # set to False to skip saving the figure

# MLP training hyperparameters
BATCH_SIZE = 15
EPOCHS     = 50
LR         = 1e-5
WEIGHT_DECAY = 1e-4

# ============================================================================
# 2.  DEVICE SETUP
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"\n🔍 Running MLP Probe across {N_LAYERS} layers ({N_SPLITS}-fold CV) "
      f"on embeddings from: {FOLDER}/\n")

# ============================================================================
# 3.  LAYER-WISE PROBE LOOP
# ============================================================================

pear_list  = []
spear_list = []
mse_list   = []
scaler     = StandardScaler()

for layer in tqdm(range(N_LAYERS), desc="Layers"):

    file_path = os.path.join(FOLDER, f"layer_{layer}.jsonl")

    # ── 3a. Guard: skip missing layer files ─────────────────────────────────
    if not os.path.exists(file_path):
        print(f"  ⚠️  Missing file: {file_path} — skipping")
        pear_list.append(np.nan)
        spear_list.append(np.nan)
        mse_list.append(np.nan)
        continue

    # ── 3b. Load embeddings and concreteness scores ──────────────────────────
    X, y = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            X.append(item["embedding"])
            y.append(item["conc"])

    X = scaler.fit_transform(np.array(X, dtype=np.float32))
    y = np.array(y, dtype=np.float32)

    # ── 3c. Guard: skip layers with too few samples ──────────────────────────
    if len(X) < 50:
        print(f"  ⚠️  Layer {layer}: too few samples ({len(X)}) — skipping")
        pear_list.append(np.nan)
        spear_list.append(np.nan)
        mse_list.append(np.nan)
        continue

    # ── 3d. 10-fold cross-validation ─────────────────────────────────────────
    preds = np.zeros_like(y)
    kf    = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train         = y[train_idx]

        # Convert to GPU tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
        X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device)

        in_dim = X_train.shape[1]

        # ── 3e. Define MLP probe ─────────────────────────────────────────────
        # Architecture: 4-layer MLP with ReLU activations and dropout
        mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        ).to(device)

        optimizer = torch.optim.AdamW(mlp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn   = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # ── 3f. Train the probe ──────────────────────────────────────────────
        mlp.train()
        for _ in range(EPOCHS):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(mlp(xb), yb)
                loss.backward()
                optimizer.step()

        # ── 3g. Predict on held-out fold ─────────────────────────────────────
        mlp.eval()
        with torch.no_grad():
            preds[test_idx] = mlp(X_test_t).cpu().numpy().flatten()

    # ── 3h. Compute layer-level metrics after full CV ────────────────────────
    mse         = mean_squared_error(y, preds)
    pear, _     = pearsonr(y, preds)
    spear, _    = spearmanr(y, preds)

    pear_list.append(pear)
    spear_list.append(spear)
    mse_list.append(mse)

    print(f"  Layer {layer:02d} → Pearson={pear:.3f} | Spearman={spear:.3f} | MSE={mse:.3f}")

# ============================================================================
# 4.  SAVE RESULTS
# ============================================================================

results_df = pd.DataFrame({
    "layer":    list(range(N_LAYERS)),
    "pearson":  pear_list,
    "spearman": spear_list,
    "mse":      mse_list
})

OUTPUT_CSV = "mlp_probe_results.csv"
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n💾 Saved results → {OUTPUT_CSV}")

# ============================================================================
# 5.  PLOT
# ============================================================================

if SAVE_PLOT:
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["layer"], results_df["pearson"],  marker="o", label="Pearson r")
    plt.plot(results_df["layer"], results_df["spearman"], marker="o", label="Spearman ρ")
    plt.xlabel("Layer")
    plt.ylabel("Correlation")
    plt.title(f"Layer-wise Concreteness Correlation — MLP Probe ({FOLDER})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("layer_wise_correlation.png", dpi=300)
    print("📊 Saved plot → layer_wise_correlation.png")