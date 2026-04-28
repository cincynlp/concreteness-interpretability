"""
compute_svd_subspace.py
========================
Computes a low-dimensional concreteness subspace at every transformer layer
by applying SVD to the stacked diff-mean vectors produced by `compute_diffmean.py`.

The resulting per-layer basis matrices can be used downstream to project
contextual embeddings onto the concreteness subspace.

Pipeline overview
-----------------
1. Load the diff-mean vectors saved by `compute_diffmean.py` (`diffmeans.npy`).
2. For each transformer layer, centre the diff-mean matrix and apply SVD.
3. Retain the top-K right singular vectors as the subspace basis B
   of shape [hidden_dim, K].
4. Save B for each layer as a `.npy` file in a dedicated output folder.

What K controls
---------------
K is the dimensionality of the concreteness subspace:
    K = 1  →  a single direction (axis) that best explains the variance
               between high- and low-concreteness embeddings.
    K > 1  →  a K-dimensional subspace capturing additional concreteness
               variance beyond the primary axis.
In our paper we use K = 1.

Input files required
--------------------
- `diffmeans.npy` : output of `compute_diffmean.py`.
  A numpy dict {layer (int) → array of shape [REPEATS, hidden_dim]}
  containing the unit-normalised diff-mean vectors per layer.

Output
------
A folder `subspace_K/` (e.g. `subspace_1/`) containing one `.npy` file
per layer:
    layer_0_concreteness_basis_k{K}.npy
    layer_1_concreteness_basis_k{K}.npy
    …
Each file contains a basis matrix B of shape [hidden_dim, K].
"""

# ── Standard library ────────────────────────────────────────────────────────
import os

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np

# ============================================================================
# 1.  CONFIGURATION
# ============================================================================

DIFFMEAN_FILE = "diffmeans.npy"   # ← output of compute_diffmean.py
SUBSPACE_K    = 1                 # ← dimensionality of the concreteness subspace
                                  #   (K=1: one-directional axis; K>1: richer subspace)

# ============================================================================
# 2.  SETUP
# ============================================================================

out_dir = f"subspace_{SUBSPACE_K}"
os.makedirs(out_dir, exist_ok=True)
print(f"📁 Saving per-layer subspace bases to: {out_dir}/")

# ============================================================================
# 3.  LOAD DIFF-MEAN VECTORS
# ============================================================================

# layer_diffmeans: {layer (int) → np.ndarray of shape [REPEATS, hidden_dim]}
layer_diffmeans = np.load(DIFFMEAN_FILE, allow_pickle=True).item()
print(f"Loaded diff-mean vectors for {len(layer_diffmeans)} layers.\n")

# ============================================================================
# 4.  PER-LAYER SVD & SAVE
# ============================================================================

for layer, dms in layer_diffmeans.items():

    # ── 4a. Centre the diff-mean matrix before SVD ───────────────────────────
    D = dms - dms.mean(axis=0, keepdims=True)   # [REPEATS, hidden_dim]

    # ── 4b. SVD: rows of Vt are the right singular vectors ───────────────────
    # U: [REPEATS, REPEATS], S: [REPEATS], Vt: [REPEATS, hidden_dim]
    _, S, Vt = np.linalg.svd(D, full_matrices=False)

    # ── 4c. Retain top-K right singular vectors as the subspace basis ─────────
    # B: [hidden_dim, K] — each column is one basis direction
    B = Vt[:SUBSPACE_K].T

    # ── 4d. Save basis matrix ─────────────────────────────────────────────────
    save_path = os.path.join(out_dir, f"layer_{layer}_concreteness_basis_k{SUBSPACE_K}.npy")
    np.save(save_path, B)

    print(f"  Layer {layer:02d} → basis shape {B.shape} | "
          f"top singular value: {S[0]:.4f} | saved to {save_path}")

# ============================================================================
# 5.  SUMMARY
# ============================================================================

print(f"\n✅ Per-layer SVD complete.")
print(f"   Output folder : {out_dir}/")
print(f"   Subspace dim  : K = {SUBSPACE_K}")
print(f"   Layers saved  : {len(layer_diffmeans)}")