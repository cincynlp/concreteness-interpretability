# Concreteness Steering

This folder contains a script to steer a large language model's generations
towards higher or lower lexical concreteness by injecting the 1-D concreteness
axis directly into the model's residual stream at inference time.

This is an implementation of **activation steering** (also known as
representation engineering): rather than fine-tuning the model, a scaled
direction vector is added to the hidden states at a single transformer layer,
nudging the model's internal representation towards the target concept during
generation. No training is required — the only input needed is the
concreteness axis produced by the subspace analysis scripts.

---

## Script

### `steer_concreteness.py`
Loads a model and a pre-computed 1-D concreteness axis, registers a forward
hook on a chosen transformer layer, and generates a rewrite of an input
sentence with the steering active.

The hook adds `α × u` to the hidden states at the target layer on every
forward pass, where `u` is the unit-normalised concreteness axis and `α`
controls the direction and magnitude of the intervention:

| α | Effect |
|---|---|
| Positive | Steers towards higher concreteness (more literal language) |
| Negative | Steers towards lower concreteness (more abstract / figurative language) |
| Larger \|α\| | Stronger effect (very large values may degrade fluency) |

The hook is registered immediately before generation and removed immediately
after, so the model is unaffected by the intervention outside of that call.

---

## Steering Parameters

Set these in the configuration section at the top of the script:

| Parameter | Description |
|---|---|
| `MODEL_NAME` | HuggingFace model checkpoint to steer |
| `BASIS_PATH` | Path to the concreteness axis `.npy` file |
| `TARGET_LAYER` | Transformer layer at which to inject the steering vector |
| `ALPHA` | Steering strength (positive = more concrete, negative = more abstract) |

**On choosing `TARGET_LAYER`:** Earlier layers affect broad semantic and
syntactic properties; later layers have a stronger influence on surface-level
word choice. In our paper, layer 20 is where figurativity is most salient for
LLaMA-3.1-8B. We recommend identifying the optimal layer using the AUROC
profiles produced by `project_and_evaluate.py` in the `subspace_analysis/`
folder — the layer with the highest AUROC is the most informative injection
point.

---

## Required Input Files

| File | Description | Produced by |
|---|---|---|
| `global_concreteness_basis_k1.npy` | 1-D concreteness axis vector | `compute_diffmean.py` or `compute_svd_subspace.py` |

The basis file must correspond to the **same model** used for steering. Axes
computed from LLaMA embeddings should not be applied to Qwen or Gemma, as the
hidden dimensions and representation spaces differ.

---

## Concreteness Scores

The concreteness axis is derived from ground-truth concreteness ratings from:

> Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas. *Behavior Research Methods, 46*(3), 904–911. https://doi.org/10.3758/s13428-013-0403-5

**► Download the norms from the paper's supplementary materials:**
https://link.springer.com/article/10.3758/s13428-013-0403-5

---

## Model Reference

| Model | `TARGET_LAYER` (paper) | Basis file hidden dim |
|---|---|---|
| LLaMA-3.1-8B | 20 | 4096 |
| Qwen3-8B     | —  | 4096 |
| Gemma-2-9B   | —  | 3584 |
| GPT-OSS-20B  | —  | varies |

---

## Dependencies

```bash
pip install torch transformers numpy
```

A CUDA-capable GPU is strongly recommended. The script uses `device_map="auto"`
and `fp16` precision to fit large models on a single GPU where possible.

---

## Usage

```bash
python steer_concreteness.py
```

To steer a different sentence, update `test_sentence` in the demo section at
the bottom of the script. To batch-process multiple sentences, call `rewrite()`
in a loop — the hook is safely scoped to each individual call.

**Example output:**
```
INPUT  : There is a bridge of trust that has developed between us.
α      : +20.0
OUTPUT : A strong sense of trust has grown between us over time.
```
