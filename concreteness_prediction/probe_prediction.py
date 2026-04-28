"""
mlp_probe_prediction_diff.py
==============================
Trains an MLP regression probe on per-layer contextual embeddings and
measures the difference between predicted and ground-truth concreteness scores
on a held-out test set.

Pipeline overview
-----------------
1. For each transformer layer, load pre-extracted embeddings from a training
   folder (e.g. embeddings from Wikipedia sentences).
2. Train an MLP regression probe to predict concreteness scores from the
   embeddings.
3. Load a separate test set of embeddings (e.g. synthetic high/low concreteness
   sentences as used in the paper) and predict their concreteness scores.
4. Compute Δ = (predicted − true) for each test sample and record the mean
   and standard deviation of Δ per layer.
5. Save layer-wise results to a CSV and plot the mean prediction error profile.

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
- A TRAIN folder of per-layer `.jsonl` files (e.g. `llama_emb/`), produced by
  one of the extraction scripts (e.g. `extract_llama_embeddings.py`).
- A TEST folder of per-layer `.jsonl` files for the sentences you want to
  predict on (e.g. `llama_emb_test/`). These must be extracted separately
  using the same extraction script on your test sentences.
  In our paper, the test sentences are synthetically generated high-concreteness
  and low-concreteness sentences.

Each `.jsonl` line must be a JSON record:
    {"word": ..., "sentence": ..., "conc": ..., "embedding": [...]}

Output
------
- `mlp_probe_diff.csv`                   : layer-wise mean Δ and std Δ
- `layerwise_mean_concreteness_diff.png` : plot of mean (predicted − true) across layers
"""

# ── Standard library ────────────────────────────────────────────────────────
import os
import json

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# 1.  CONFIGURATION
# ============================================================================
# ► Set TRAIN_FOLDER and TEST_FOLDER to the embedding directories for the
#   model you want to evaluate. Both folders must exist and contain matching
#   layer_N.jsonl files extracted by the same model's extraction script.
#
#   Model reference:
#       LLaMA-3.1-8B  →  N_LAYERS = 32
#       Qwen3-8B      →  N_LAYERS = 36
#       Gemma-2-9B    →  N_LAYERS = 42
#       GPT-OSS-20B   →  N_LAYERS = 24

TRAIN_FOLDER = "llama_emb"        # ← embeddings for training (e.g. Wikipedia sentences)
TEST_FOLDER  = "llama_emb_test"   # ← embeddings for test (e.g. synthetic sentences)
N_LAYERS     = 32                 # ← set to match the model above
SAVE_PLOT    = True               # set to False to skip saving the figure

# MLP training hyperparameters
BATCH_SIZE   = 15
EPOCHS       = 50
LR           = 1e-5
WEIGHT_DECAY = 1e-4

# ============================================================================
# 2.  DEVICE SETUP
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"\n🔍 Training on: {TRAIN_FOLDER}/  |  Testing on: {TEST_FOLDER}/\n")

# ============================================================================
# 3.  LAYER-WISE PROBE LOOP
# ============================================================================

layer_means = []
layer_stds  = []

for layer in range(N_LAYERS):

    train_path = os.path.join(TRAIN_FOLDER, f"layer_{layer}.jsonl")
    test_path  = os.path.join(TEST_FOLDER,  f"layer_{layer}.jsonl")

    # ── 3a. Guard: skip if either file is missing ────────────────────────────
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"  ⚠️  Missing data for layer {layer} — skipping")
        layer_means.append(np.nan)
        layer_stds.append(np.nan)
        continue

    # ── 3b. Load training embeddings and concreteness scores ─────────────────
    X_train, y_train = [], []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            X_train.append(item["embedding"])
            y_train.append(item["conc"])

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    # ── 3c. Guard: skip layers with too few training samples ──────────────────
    if len(X_train) < 20:
        print(f"  ⚠️  Layer {layer}: too few train samples ({len(X_train)}) — skipping")
        layer_means.append(np.nan)
        layer_stds.append(np.nan)
        continue

    # ── 3d. Load test embeddings and concreteness scores ─────────────────────
    X_test, y_test = [], []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            X_test.append(item["embedding"])
            y_test.append(item["conc"])

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # ── 3e. Convert to GPU tensors ───────────────────────────────────────────
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device)

    # ── 3f. Define MLP probe ──────────────────────────────────────────────────
    # Architecture: 4-layer MLP with ReLU activations and dropout
    # (identical architecture to mlp_probe_correlation.py)
    in_dim = X_train.shape[1]
    mlp = nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, 1)
    ).to(device)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn   = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ── 3g. Train the probe ───────────────────────────────────────────────────
    mlp.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(mlp(xb), yb)
            loss.backward()
            optimizer.step()

    # ── 3h. Predict on test set and compute Δ = (predicted − true) ───────────
    mlp.eval()
    with torch.no_grad():
        preds = mlp(X_test_t).cpu().numpy().flatten()

    diffs = preds - y_test
    layer_means.append(np.mean(diffs))
    layer_stds.append(np.std(diffs))

    print(f"  Layer {layer:02d} → Mean Δ = {np.mean(diffs):.4f} | Std Δ = {np.std(diffs):.4f}")

# ============================================================================
# 4.  SAVE RESULTS
# ============================================================================

results_df = pd.DataFrame({
    "layer":     list(range(N_LAYERS)),
    "mean_diff": layer_means,
    "std_diff":  layer_stds
})

OUTPUT_CSV = "mlp_probe_diff.csv"
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n💾 Saved results → {OUTPUT_CSV}")

# ============================================================================
# 5.  PLOT
# ============================================================================

if SAVE_PLOT:
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["layer"], results_df["mean_diff"], marker="o")
    plt.axhline(0, color="black", linestyle="--", alpha=0.6)  # zero reference line
    plt.xlabel("Layer")
    plt.ylabel("Mean (Predicted − True)")
    plt.title(f"Mean Concreteness Prediction Error vs Layer — {TRAIN_FOLDER}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("layerwise_mean_concreteness_diff.png", dpi=300)
    print("📊 Saved plot → layerwise_mean_concreteness_diff.png")