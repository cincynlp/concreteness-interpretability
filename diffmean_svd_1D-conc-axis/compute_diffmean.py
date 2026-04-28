"""
compute_diffmean.py
====================
Computes and saves the *diff-mean* direction — a vector pointing from the
mean embedding of low-concreteness words to the mean embedding of
high-concreteness words — at every transformer layer.

The diff-mean is computed over multiple random subsamples (repeats) to
assess its stability across different sample draws. Pairwise cosine
similarity between repeat vectors is recorded as a stability metric.

Pipeline overview
-----------------
1. For each transformer layer, load pre-extracted embeddings from the
   corresponding `layer_N.jsonl` file.
2. Split embeddings into a high-concreteness set (conc ≥ THRESH_HIGH) and a
   low-concreteness set (conc ≤ THRESH_LOW) based on Brysbaert norms.
3. Repeat REPEATS times: randomly subsample SAMPLES_PER_SET vectors from
   each set and compute the unit-normalised diff-mean vector:
       diffmean = mean(X_high) − mean(X_low)   [then L2-normalised]
4. Compute pairwise cosine similarity across repeats as a stability score.
5. Save all diff-mean vectors and the per-layer stability scores.

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
- A folder of per-layer `.jsonl` files (e.g. `llama_emb/`), produced by one
  of the extraction scripts (e.g. `extract_llama_embeddings.py`).

Each `.jsonl` line must be a JSON record:
    {"word": ..., "sentence": ..., "conc": ..., "embedding": [...]}

Output
------
- `diffmeans.npy`            : dict {layer → array of shape [REPEATS, hidden_dim]}
                               containing the unit-normalised diff-mean vectors
- `diffmean_stability.csv`   : layer-wise mean pairwise cosine similarity across repeats
"""

# ── Standard library ────────────────────────────────────────────────────────
import os
import json

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# 1.  CONFIGURATION
# ============================================================================
# ► Set TRAIN_FOLDER to the embedding directory for the model you want to use.
#
#   Model reference:
#       LLaMA-3.1-8B  →  N_LAYERS = 32
#       Qwen3-8B      →  N_LAYERS = 36
#       Gemma-2-9B    →  N_LAYERS = 42
#       GPT-OSS-20B   →  N_LAYERS = 24

TRAIN_FOLDER    = "llama_emb"   # ← change to switch model
N_LAYERS        = 32            # ← set to match the model above

THRESH_HIGH     = 4.0           # concreteness threshold for high-concreteness set (conc ≥ value)
THRESH_LOW      = 2.0           # concreteness threshold for low-concreteness set  (conc ≤ value)
SAMPLES_PER_SET = 250           # number of vectors to subsample from each set per repeat
REPEATS         = 10            # number of random subsampling repeats for stability estimation
SEED            = 42

rng = np.random.default_rng(SEED)

# ============================================================================
# 2.  HELPER FUNCTIONS
# ============================================================================

def load_by_threshold(path: str, hi: float, lo: float):
    """
    Load embeddings from a `.jsonl` file and split them into high- and
    low-concreteness sets based on the provided thresholds.

    Parameters
    ----------
    path : str   – Path to the layer's `.jsonl` file.
    hi   : float – Minimum concreteness score for the high set (conc ≥ hi).
    lo   : float – Maximum concreteness score for the low set  (conc ≤ lo).

    Returns
    -------
    Tuple (X_high, X_low) of np.ndarray, or (None, None) if either set is empty.
    """
    X_high, X_low = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            x = np.asarray(item["embedding"], dtype=np.float32)
            c = float(item["conc"])
            if c >= hi:
                X_high.append(x)
            elif c <= lo:
                X_low.append(x)

    if not X_high or not X_low:
        return None, None
    return np.stack(X_high), np.stack(X_low)


def safe_unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return the L2-normalised version of vector v (safe against zero-norm)."""
    return v / (np.linalg.norm(v) + eps)

# ============================================================================
# 3.  MAIN: COMPUTE DIFF-MEAN VECTORS & STABILITY
# ============================================================================

layer_diffmeans = {}   # layer → array of shape [REPEATS, hidden_dim]
stability_rows  = []   # records of (layer, mean_pairwise_cosine)

print(f"🛠️  Computing diff-mean vectors & cosine stability across {N_LAYERS} layers...\n")

for layer in tqdm(range(N_LAYERS), desc="Layers"):

    path = os.path.join(TRAIN_FOLDER, f"layer_{layer}.jsonl")

    # ── 3a. Guard: skip missing layer files ─────────────────────────────────
    if not os.path.exists(path):
        print(f"  ⚠️  Missing file: {path} — skipping")
        continue

    # ── 3b. Load and split by concreteness threshold ─────────────────────────
    X_high, X_low = load_by_threshold(path, THRESH_HIGH, THRESH_LOW)

    if X_high is None or X_low is None:
        print(f"  ⚠️  Layer {layer}: no embeddings above/below thresholds — skipping")
        continue

    # ── 3c. Guard: ensure enough samples in each set ─────────────────────────
    if len(X_high) < SAMPLES_PER_SET or len(X_low) < SAMPLES_PER_SET:
        print(f"  ⚠️  Layer {layer}: need ≥{SAMPLES_PER_SET} per set; "
              f"have high={len(X_high)}, low={len(X_low)} — skipping")
        continue

    # ── 3d. Compute diff-mean over REPEATS random subsamples ─────────────────
    diffmeans = []
    for _ in range(REPEATS):
        idx_h = rng.choice(len(X_high), SAMPLES_PER_SET, replace=False)
        idx_l = rng.choice(len(X_low),  SAMPLES_PER_SET, replace=False)
        dm = X_high[idx_h].mean(axis=0) - X_low[idx_l].mean(axis=0)
        diffmeans.append(safe_unit(dm))   # L2-normalise for cosine stability

    diffmeans = np.stack(diffmeans, axis=0)   # [REPEATS, hidden_dim]
    layer_diffmeans[layer] = diffmeans

    # ── 3e. Pairwise cosine similarity across repeats (stability metric) ──────
    cos  = cosine_similarity(diffmeans)
    mask = ~np.eye(REPEATS, dtype=bool)       # exclude self-similarity on diagonal
    mean_cosine = cos[mask].mean()
    stability_rows.append((layer, mean_cosine))

    print(f"  Layer {layer:02d} → mean pairwise cosine: {mean_cosine:.4f}")

# ============================================================================
# 4.  SAVE RESULTS
# ============================================================================

# Save all diff-mean vectors as a numpy dict
np.save("diffmeans.npy", layer_diffmeans)
print("\n💾 Saved diff-mean vectors → diffmeans.npy")

# Save per-layer stability scores
stability_df = pd.DataFrame(stability_rows, columns=["layer", "mean_pairwise_cosine"])
stability_df.to_csv("diffmean_stability.csv", index=False)
print("📊 Saved stability scores → diffmean_stability.csv")
print("\n✅ Done!")