# concreteness-interpretability

This repository contains the code for the paper:

> **Exploring Concreteness Through a Figurative Lens**
> Saptarshi Ghosh, Tianyu Jiang
> arXiv:2604.18296 · [https://arxiv.org/abs/2604.18296](https://arxiv.org/abs/2604.18296)

---

## Overview

Static concreteness ratings are widely used in NLP, yet a word's concreteness
can shift with context — especially in figurative language such as metaphor,
where common concrete nouns can take abstract interpretations. This paper
investigates how LLMs internally represent concreteness across layers and
model families.

We find that LLMs separate literal and figurative usage in early layers, and
that mid-to-late layers compress concreteness into a **one-dimensional
direction** that is consistent across models. We show that this geometric
structure is practically useful: a single concreteness axis supports efficient
figurative-language classification and enables **training-free steering** of
generation towards more literal or more figurative outputs.

---

## Repository Structure

```
├── extract_embeddings/         # Extract per-layer hidden-state embeddings
├── concreteness_prediction/    # MLP probe: correlation and prediction error
├── diffmean_svd_1D-conc-axis/  # Diff-mean, SVD subspace, and projection/AUROC
└── 1D_axis_steering/           # Activation steering along the concreteness axis
```

Each folder contains its own `README.md` with full usage instructions.

---

## Data & Downloads

### 1. Concreteness Norms (required for all experiments)
Ground-truth concreteness ratings from:

> Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings
> for 40 thousand generally known English word lemmas. *Behavior Research
> Methods, 46*(3), 904–911. https://doi.org/10.3758/s13428-013-0403-5

**► Download from the paper's supplementary materials:**
https://link.springer.com/article/10.3758/s13428-013-0403-5

Save the file as `concreteness.csv` (columns: `Word`, `Conc.M`) in the same
directory as whichever script you are running.

### 2. Word–Sentence Dataset
A CSV file (`sentences.csv`) containing target words and carrier sentences
(e.g. sourced from Wikipedia or another corpus).

| Column | Description |
|---|---|
| `target word` | The word whose concreteness is being probed |
| `sentence` | A sentence containing the target word |

### 3. Model Checkpoints
The four models used in the paper are all available on HuggingFace:

| Model | HuggingFace ID |
|---|---|
| LLaMA-3.1-8B-Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Qwen3-8B | `Qwen/Qwen3-8B` |
| Gemma-2-9B | `google/gemma-2-9b` |
| GPT-OSS-20B | `openai/gpt-oss-20b` |

> **Note:** Some checkpoints require accepting a licence agreement on
> HuggingFace before downloading. Make sure you are logged in via
> `huggingface-cli login` before running any extraction script.

---

## Quickstart

### Step 1 — Extract embeddings
```bash
cd extract_embeddings/
python extract_llama_embeddings.py   # or qwen / gemma / gpt
```

### Step 2 — Run the MLP probe
```bash
cd concreteness_prediction/
python mlp_probe_correlation.py
```

### Step 3 — Compute the concreteness subspace
```bash
cd diffmean_svd_1D-conc-axis/
python compute_diffmean.py
python 1D_axis_via_svd.py
```

### Step 4 — Evaluate on figurative language
```bash
python 1D_axis_projection.py
```

### Step 5 — Steer generation
```bash
cd 1D_axis_steering/
python steer.py
```

---

## Dependencies

```bash
pip install torch transformers numpy pandas scikit-learn scipy matplotlib tqdm
```

A CUDA-capable GPU is strongly recommended for all steps. All scripts use
`device_map="auto"` and `fp16` precision to fit large models on a single GPU
where possible.

---

## Citation

If you use this code or the findings of this paper, please cite:

```bibtex
@misc{ghosh2026exploringconcretenessfigurativelens,
      title={Exploring Concreteness Through a Figurative Lens}, 
      author={Saptarshi Ghosh and Tianyu Jiang},
      year={2026},
      eprint={2604.18296},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.18296}, 
}
```
