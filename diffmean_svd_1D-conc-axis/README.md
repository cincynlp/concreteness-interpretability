# Concreteness Subspace Analysis

This folder contains scripts to identify a *concreteness axis* in the hidden
representation space of large language models and evaluate how well that axis
separates classes of figurative language. The analysis proceeds in three steps:
(1) computing a diff-mean direction from concreteness-labelled embeddings,
(2) refining it into a low-dimensional subspace via SVD, and (3) projecting
figurative language embeddings onto the axis and measuring separability.

---

## Scripts

### `compute_diffmean.py`
Computes a **diff-mean direction** at every transformer layer — a vector
pointing from the mean embedding of low-concreteness words to the mean
embedding of high-concreteness words.

To assess the stability of this direction, the computation is repeated
`REPEATS` times over random subsamples drawn from each concreteness set.
The pairwise cosine similarity across repeats is recorded as a per-layer
stability score.

**Key parameters (set in the configuration section):**

| Parameter | Description |
|---|---|
| `TRAIN_FOLDER` | Embedding folder to compute diff-means from |
| `THRESH_HIGH` | Minimum concreteness score for the high-concreteness set (conc ≥ value) |
| `THRESH_LOW` | Maximum concreteness score for the low-concreteness set (conc ≤ value) |
| `SAMPLES_PER_SET` | Number of embeddings to subsample per repeat |
| `REPEATS` | Number of random subsampling repeats |

**Output:**
- `diffmeans.npy` — dict `{layer → array [REPEATS, hidden_dim]}` of unit-normalised diff-mean vectors
- `diffmean_stability.csv` — layer-wise mean pairwise cosine similarity across repeats

---

### `compute_svd_subspace.py`
Takes the stacked diff-mean vectors from `compute_diffmean.py` and applies
**SVD** at each layer to extract a low-dimensional concreteness subspace.

The top-K right singular vectors are retained as the subspace basis B of
shape `[hidden_dim, K]`. K controls the dimensionality of the subspace:
- `K = 1` yields a single concreteness axis (used in our paper)
- `K > 1` yields a richer subspace capturing additional variance

**Key parameters (set in the configuration section):**

| Parameter | Description |
|---|---|
| `DIFFMEAN_FILE` | Path to `diffmeans.npy` from `compute_diffmean.py` |
| `SUBSPACE_K` | Dimensionality of the concreteness subspace |

**Output:**
- A folder `subspace_K/` (e.g. `subspace_1/`) containing one `.npy` file per layer:
  `layer_{N}_concreteness_basis_k{K}.npy`, each of shape `[hidden_dim, K]`

---

### `project_and_evaluate.py`
Projects figurative language embeddings onto the 1-D concreteness axis and
evaluates how well that single direction separates two classes (e.g. literal
vs. figurative, or high vs. low concreteness usage) using **AUROC, F1, and
accuracy**.

For each transformer layer:
1. Loads embeddings and binary labels from a figurative language embedding folder.
2. Projects each embedding onto the concreteness axis to obtain a scalar score.
3. Splits data into 80% train / 20% test (stratified).
4. Learns the optimal classification threshold on the train split by maximising F1.
5. Reports AUROC (threshold-free), F1, and accuracy on the test split.

> **Note on AUROC:** AUROC is computed directly on the raw projection scores
> and reflects the intrinsic separability of the two classes along the
> concreteness axis, independent of any threshold choice.

**Key parameters (set in the configuration section):**

| Parameter | Description |
|---|---|
| `BASE_FOLDER` | Folder of figurative language embeddings to evaluate on |
| `BASIS_PATH` | Concreteness axis file (from `compute_diffmean.py` or `compute_svd_subspace.py`) |
| `N_LAYERS` | Number of transformer layers (must match the extraction model) |
| `TRAIN_RATIO` | Fraction of data used to learn the threshold (default: 0.80) |

**Output:** Per-layer results printed to stdout:
```
Layer NN → AUROC(test)=... | F1(train@t*)=... | F1(test@t*)=... | Acc(test@t*)=... | t*=...
```

---

## Workflow

The three scripts are designed to be run in order:

```
compute_diffmean.py  →  compute_svd_subspace.py  →  project_and_evaluate.py
```

```bash
# Step 1: Compute diff-mean vectors and stability scores
python compute_diffmean.py

# Step 2: Extract the concreteness subspace via SVD
python compute_svd_subspace.py

# Step 3: Project figurative embeddings and evaluate
python project_and_evaluate.py
```

---

## Required Input Files

All scripts read from per-layer `.jsonl` embedding files produced by the
extraction scripts in the `embedding_extraction/` folder. Each line must be
a JSON record:

```json
{
  "word":      "apple",
  "sentence":  "She picked a ripe apple from the tree.",
  "conc":      4.85,
  "embedding": [0.012, -0.134, ..., 0.056]
}
```

For `project_and_evaluate.py`, each record must also contain a binary label:

```json
{
  "embedding": [0.012, -0.134, ..., 0.056],
  "label": 1
}
```

---

## Concreteness Scores

Ground-truth concreteness ratings come from:

> Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas. *Behavior Research Methods, 46*(3), 904–911. https://doi.org/10.3758/s13428-013-0403-5

**► Download the norms from the paper's supplementary materials:**
https://link.springer.com/article/10.3758/s13428-013-0403-5

---

## Model Reference

| Model | Embedding Folder | `N_LAYERS` |
|---|---|---|
| LLaMA-3.1-8B | `llama_emb/` | 32 |
| Qwen3-8B     | `qwen_emb/`  | 36 |
| Gemma-2-9B   | `gemma_emb/` | 42 |
| GPT-OSS-20B  | `gpt_emb/`   | 24 |

---

## Dependencies

```bash
pip install numpy pandas scikit-learn tqdm
```
