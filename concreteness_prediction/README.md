# Probing

This folder contains scripts to probe per-layer contextual embeddings for lexical concreteness using a non-linear MLP regressor. Two complementary analyses are provided: one evaluates how well each layer encodes concreteness in general, and the other measures how well the probe predicts concreteness on a held-out test set and quantifies the prediction error relative to human norms.

Both scripts use an identical MLP architecture and share the same embedding format produced by the extraction scripts in the `embedding_extraction/` folder.

---

## Scripts

### `mlp_probe_correlation.py`
Evaluates the relationship between per-layer contextual embeddings and ground-truth concreteness scores using **10-fold cross-validated MLP regression**.

For each transformer layer:
1. Loads the pre-extracted embeddings and concreteness scores from the layer's `.jsonl` file.
2. Standardises the embeddings with a `StandardScaler`.
3. Trains and evaluates an MLP probe using 10-fold cross-validation.
4. Reports Pearson *r*, Spearman *ρ*, and MSE for each layer.

**Output:**
- `mlp_probe_results.csv` — layer-wise Pearson, Spearman, and MSE scores
- `layer_wise_correlation.png` — correlation profile across all layers

---

### `mlp_probe_prediction_diff.py`
Trains an MLP probe on one set of embeddings (e.g. Wikipedia sentences) and tests it on a **separate held-out set** (e.g. synthetic high/low concreteness sentences), then measures the layer-wise prediction error relative to human concreteness norms.

For each transformer layer:
1. Trains an MLP probe on the training embeddings.
2. Predicts concreteness scores on the test embeddings.
3. Computes Δ = (predicted − true) for each test sample and records the mean and standard deviation of Δ per layer.

In our paper, the test sentences are **synthetically generated** high-concreteness and low-concreteness sentences. The test embeddings must be extracted separately using the same extraction script before running this script.

**Output:**
- `mlp_probe_diff.csv` — layer-wise mean Δ and std Δ
- `layerwise_mean_concreteness_diff.png` — mean prediction error profile across all layers

---

## MLP Architecture

Both scripts use the same 4-layer MLP regressor:

```
Linear(hidden_dim → 512) → ReLU → Dropout(0.2)
Linear(512 → 256)         → ReLU → Dropout(0.2)
Linear(256 → 128)         → ReLU → Dropout(0.2)
Linear(128 → 1)
```

Trained with AdamW (`lr=1e-5`, `weight_decay=1e-4`), MSE loss, batch size 15, for 50 epochs.

---

## Concreteness Scores

Ground-truth concreteness ratings come from:

> Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas. *Behavior Research Methods, 46*(3), 904–911. https://doi.org/10.3758/s13428-013-0403-5

**► Download the norms from the paper's supplementary materials:**
https://link.springer.com/article/10.3758/s13428-013-0403-5

These scores are embedded in the `.jsonl` input files under the `conc` field, written there during the embedding extraction step.

---

## Required Input Files

Each script reads from a folder of per-layer `.jsonl` files produced by the extraction scripts. Each line must be a JSON record in the following format:

```json
{
  "word":      "apple",
  "sentence":  "She picked a ripe apple from the tree.",
  "conc":      4.85,
  "embedding": [0.012, -0.134, ..., 0.056]
}
```

Set `FOLDER` (for `mlp_probe_correlation.py`) or `TRAIN_FOLDER` / `TEST_FOLDER` (for `mlp_probe_prediction_diff.py`) in the configuration section at the top of each script.

### Model reference

| Model | Embedding Folder | `N_LAYERS` |
|---|---|---|
| LLaMA-3.1-8B | `llama_emb/` | 32 |
| Qwen3-8B     | `qwen_emb/`  | 36 |
| Gemma-2-9B   | `gemma_emb/` | 42 |
| GPT-OSS-20B  | `gpt_emb/`   | 24 |

---

## Dependencies

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib
```

A CUDA-capable GPU is strongly recommended.

---

## Usage

```bash
# Evaluate layer-wise concreteness encoding (cross-validated)
python mlp_probe_correlation.py

# Measure prediction error on a held-out test set
python mlp_probe_prediction_diff.py
```

Make sure the embedding folders referenced in the configuration section of each script exist and are populated before running.
