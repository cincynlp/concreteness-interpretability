"""
project_and_evaluate.py
=========================
Projects contextual embeddings onto the 1-D concreteness subspace (the axis
computed by `compute_diffmean.py` and `compute_svd_subspace.py`) and evaluates
how well that single direction separates two classes of figurative language
embeddings using AUROC, F1, and accuracy.

Pipeline overview
-----------------
1. Load the global 1-D basis vector (concreteness axis) from the diff-mean
   or SVD output file.
2. For each transformer layer, load embeddings and binary labels from a
   figurative language embedding folder.
3. Project each embedding onto the 1-D axis to obtain a scalar concreteness
   score per sample.
4. Split into 80% train / 20% test (stratified).
5. Learn the optimal classification threshold on the TRAIN split by maximising
   F1 score.
6. Evaluate on the TEST split: AUROC (threshold-free), F1, and accuracy.

Note on AUROC
-------------
AUROC is computed directly on the raw projection scores (no threshold needed)
and therefore reflects the intrinsic separability of the two classes along the
concreteness axis, independent of any threshold choice.

Input files required
--------------------
- A basis file (e.g. `diffmeans.npy` or a per-layer basis from `subspace_1/`)
  produced by `compute_diffmean.py` or `compute_svd_subspace.py`.
- A folder of per-layer `.jsonl` embedding files for the figurative language
  dataset you want to evaluate on (e.g. `figurative_emb/`). These must be
  extracted separately using one of the extraction scripts.

Each `.jsonl` line must be a JSON record:
    {"embedding": [...], "label": 0 or 1, ...}

  If embeddings are concatenated [sentence ‖ noun] vectors of dimension 2H,
  the script automatically selects the noun half (last H dimensions).

Output
------
Per-layer results are printed to stdout in the format:
    Layer NN → AUROC(test)=..., F1(train@t*)=..., F1(test@t*)=...,
               Acc(test@t*)=..., t*=...
"""

# ── Standard library ────────────────────────────────────────────────────────
import os
import json

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# ============================================================================
# 1.  CONFIGURATION
# ============================================================================
# ► Set BASE_FOLDER to the figurative language embedding directory you want
#   to evaluate on.
# ► Set BASIS_PATH to the basis file produced by compute_diffmean.py or
#   compute_svd_subspace.py.
#
#   Model reference:
#       LLaMA-3.1-8B  →  N_LAYERS = 32
#       Qwen3-8B      →  N_LAYERS = 36
#       Gemma-2-9B    →  N_LAYERS = 42
#       GPT-OSS-20B   →  N_LAYERS = 24

BASE_FOLDER = "figurative_emb"   # ← folder of figurative language embeddings
BASIS_PATH  = "diffmeans.npy"    # ← concreteness axis (from compute_diffmean.py
                                 #   or compute_svd_subspace.py)
N_LAYERS    = 32                 # ← set to match the model used for extraction
TRAIN_RATIO = 0.80               # fraction of data used to learn the threshold
EPS         = 1e-8               # small constant for safe L2 normalisation

# ============================================================================
# 2.  HELPER FUNCTIONS
# ============================================================================

def load_global_basis(path: str) -> np.ndarray:
    """
    Load and normalise the 1-D concreteness axis from a `.npy` file.

    Handles multiple storage shapes:
        - [H]      : already a flat vector
        - [1, H]   : single-row matrix
        - [H, 1]   : single-column matrix
        - [H, K]   : multi-column basis — first column is taken

    Parameters
    ----------
    path : str – Path to the basis `.npy` file.

    Returns
    -------
    u : np.ndarray of shape [H], L2-normalised.
    """
    B = np.load(path, allow_pickle=True)

    if B.ndim == 1:
        u = B
    elif B.shape[0] == 1 and B.shape[1] > 1:
        u = B[0]                  # [1, H] → [H]
    elif B.shape[1] == 1 and B.shape[0] > 1:
        u = B[:, 0]               # [H, 1] → [H]
    else:
        u = B[:, 0]               # [H, K] → take first column (primary axis)

    u = u.astype(np.float32)
    u = u / (np.linalg.norm(u) + EPS)
    print(f"Loaded basis from '{path}'  |  hidden_dim = {u.shape[0]}")
    return u


def load_layer_projection(layer: int, u: np.ndarray):
    """
    Load embeddings and binary labels for one layer and project onto axis u.

    If embeddings are concatenated [sentence ‖ noun] vectors of dimension 2H,
    the noun half (last H dimensions) is used for projection.

    Parameters
    ----------
    layer : int         – Transformer layer index.
    u     : np.ndarray  – Unit-normalised 1-D basis vector of shape [H].

    Returns
    -------
    scores : np.ndarray of shape [N] – scalar projection scores
    labels : np.ndarray of shape [N] – binary class labels (0 or 1)
    Returns (None, None) if the layer file is missing.
    """
    path = os.path.join(BASE_FOLDER, f"layer_{layer}.jsonl")
    if not os.path.exists(path):
        print(f"  ⚠️  Missing layer file: {path}")
        return None, None

    H = u.shape[0]
    scores, labels = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item  = json.loads(line)
            emb   = np.asarray(item["embedding"], dtype=np.float32)
            label = int(item["label"])
            D     = emb.shape[0]

            # Handle concatenated [sentence ‖ noun] embeddings
            if D == 2 * H:
                emb = emb[H:]          # use the noun half
            elif D == H:
                pass                   # standard single embedding
            else:
                raise ValueError(
                    f"Unexpected embedding dim {D} at layer {layer}; "
                    f"expected {H} or {2 * H}."
                )

            scores.append(float(emb @ u))
            labels.append(label)

    return np.array(scores, dtype=np.float32), np.array(labels, dtype=int)


def best_threshold_by_f1(scores_train: np.ndarray, y_train: np.ndarray):
    """
    Find the threshold t* that maximises F1 on the training split.

    Iterates over all unique score values as candidate thresholds and
    classifies samples as positive if score ≥ t.

    Parameters
    ----------
    scores_train : np.ndarray – Projection scores for training samples.
    y_train      : np.ndarray – Ground-truth binary labels.

    Returns
    -------
    t_star   : float – Optimal threshold.
    best_f1  : float – F1 score achieved at t_star on the training split.
    """
    best_t, best_f1 = None, -1.0

    for t in np.unique(scores_train):
        y_pred = (scores_train >= t).astype(int)
        f1 = f1_score(y_train, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = float(t)

    return best_t, best_f1


def evaluate_layer(layer: int, u: np.ndarray):
    """
    Full evaluation pipeline for a single layer:
        1. Project embeddings onto u.
        2. Stratified 80/20 train-test split.
        3. Learn threshold on train (maximise F1).
        4. Evaluate AUROC, F1, and accuracy on test.

    Returns
    -------
    Tuple (auc_test, f1_train, f1_test, acc_test, t_star), or all-NaN on failure.
    """
    scores, labels = load_layer_projection(layer, u)
    if scores is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Stratified 80 / 20 split
    s_train, s_test, y_train, y_test = train_test_split(
        scores, labels,
        test_size=1 - TRAIN_RATIO,
        random_state=42,
        stratify=labels
    )

    # AUROC on test — threshold-free, measures raw separability along the axis
    auc_test = roc_auc_score(y_test, s_test)

    # Learn threshold on train, then apply to test
    t_star, f1_train = best_threshold_by_f1(s_train, y_train)
    y_pred_test      = (s_test >= t_star).astype(int)
    f1_test          = f1_score(y_test, y_pred_test, zero_division=0)
    acc_test         = accuracy_score(y_test, y_pred_test)

    return auc_test, f1_train, f1_test, acc_test, t_star

# ============================================================================
# 3.  MAIN: EVALUATE ACROSS ALL LAYERS
# ============================================================================

if __name__ == "__main__":

    u = load_global_basis(BASIS_PATH)

    print(f"\n🔍 Evaluating 1-D projection across {N_LAYERS} layers")
    print(f"   Embeddings : {BASE_FOLDER}/")
    print(f"   Basis      : {BASIS_PATH}")
    print(f"   Split      : {int(TRAIN_RATIO*100)}% train (threshold selection) / "
          f"{int((1-TRAIN_RATIO)*100)}% test (AUROC + F1 + Acc)\n")

    for layer in range(N_LAYERS):
        auc, f1_tr, f1_te, acc, t = evaluate_layer(layer, u)
        print(
            f"  Layer {layer:02d} → "
            f"AUROC(test)={auc:.3f} | "
            f"F1(train@t*)={f1_tr:.3f} | "
            f"F1(test@t*)={f1_te:.3f} | "
            f"Acc(test@t*)={acc:.3f} | "
            f"t*={t:.4f}"
        )