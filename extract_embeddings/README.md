# Embedding Extraction

This folder contains scripts to extract per-layer hidden-state embeddings from four large language models for a **word-in-context concreteness probing** task. Each script follows an identical pipeline and output format — only the model checkpoint differs.

---

## Models Covered

| Script | Model |
|---|---|
| `extract_llama_embeddings.py` | Meta LLaMA-3.1-8B-Instruct |
| `extract_qwen_embeddings.py`  | Qwen3-8B |
| `extract_gemma_embeddings.py` | Google Gemma-2-9B |
| `extract_gpt_embeddings.py`   | GPT-OSS-20B |

---

## Pipeline Overview

Each script runs the same four-step pipeline:

1. For each *(word, sentence)* pair in the dataset, a prompt is constructed that asks the model to rate the **concreteness** of the target word in context on a scale of 1–5.
2. The model generates exactly **one new token** (the predicted rating token).
3. The full sequence (prompt + generated token) is re-fed to the model with `output_hidden_states=True`.
4. The hidden state of the **generated token** is extracted at every transformer layer and written to a per-layer `.jsonl` file.

---

## Required Input Files

Before running any script, make sure the following two files are present in the **same directory** as the script:

### 1. `sentences.csv`
A CSV file containing the target words and their carrier sentences.

| Column | Description |
|---|---|
| `target word` | The word whose concreteness is being probed |
| `sentence` | A sentence containing the target word (e.g., sourced from Wikipedia) |

### 2. `concreteness.csv`
A CSV file containing ground-truth concreteness ratings from:

> Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas. *Behavior Research Methods, 46*(3), 904–911. https://doi.org/10.3758/s13428-013-0403-5

**► Download the norms from the paper's supplementary materials:**
https://link.springer.com/article/10.3758/s13428-013-0403-5

| Column | Description |
|---|---|
| `Word` | English word lemma |
| `Conc.M` | Mean concreteness rating (1–5 scale, 5 = most concrete) |

> **Note:** Words in `sentences.csv` that are not found in the Brysbaert norms are automatically skipped and counted in the final summary.

---

## Output

Each script creates a dedicated output folder and writes one `.jsonl` file per transformer layer:

| Script | Output Folder | No. of Layer Files |
|---|---|---|
| `extract_llama_embeddings.py` | `llama_emb/` | 32 |
| `extract_qwen_embeddings.py`  | `qwen_emb/`  | 36 |
| `extract_gemma_embeddings.py` | `gemma_emb/` | 42 |
| `extract_gpt_embeddings.py`   | `gpt_emb/`   | depends on checkpoint |

Each file is named `layer_0.jsonl`, `layer_1.jsonl`, … and each line is a JSON record:

```json
{
  "word":      "apple",
  "sentence":  "She picked a ripe apple from the tree.",
  "conc":      4.85,
  "embedding": [0.012, -0.134, ..., 0.056]
}
```

---

## Dependencies

```bash
pip install torch transformers pandas tqdm
```

A CUDA-capable GPU is strongly recommended. All scripts use `device_map="auto"` and `fp16` precision to fit large models on a single GPU where possible.

---

## Usage

```bash
# Example: extract embeddings using LLaMA-3.1-8B
python extract_llama_embeddings.py

# Example: extract embeddings using Qwen3-8B
python extract_qwen_embeddings.py
```

Make sure `sentences.csv` and `concreteness.csv` are in the same directory before running.
