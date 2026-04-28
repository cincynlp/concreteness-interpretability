"""
extract_qwen_embeddings.py
============================
Extracts per-layer hidden-state embeddings from the Qwen3-8B model.

Pipeline overview
-----------------
1. For each (word, sentence) pair in the dataset, a prompt is constructed
   that asks the model to rate the *concreteness* of the target word in
   context on a scale of 1–5.
2. The model generates exactly one new token (the predicted rating token).
3. The full sequence (prompt + generated token) is re-fed to the model with
   `output_hidden_states=True`.
4. The hidden state of the *generated token* is extracted at every
   transformer layer and written to a per-layer `.jsonl` file.

Concreteness scores
-------------------
Ground-truth concreteness ratings come from:

    Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014).
    Concreteness ratings for 40 thousand generally known English word lemmas.
    Behavior Research Methods, 46(3), 904–911.
    https://doi.org/10.3758/s13428-013-0403-5

    ► Download the norms from the paper's supplementary materials or from:
      https://link.springer.com/article/10.3758/s13428-013-0403-5
      The relevant columns are: Word, Conc.M (mean concreteness, 1–5 scale).
      Save the file as `concreteness.csv` in the same directory as this script.

Input files required
--------------------
- sentences.csv    : columns  →  target word | sentence  (extracted from Wikipedia or any other source)
- concreteness.csv : columns  →  Word | Conc.M           (Brysbaert et al.)

Output
------
A folder `qwen_emb/` containing NUM_LAYERS files:
    layer_0.jsonl … layer_{NUM_LAYERS-1}.jsonl
Each line is a JSON record:
    {"word": ..., "sentence": ..., "conc": ..., "embedding": [...]}
"""

# ── Standard library ────────────────────────────────────────────────────────
import os
import json

# ── Third-party ─────────────────────────────────────────────────────────────
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# 1.  DEVICE SETUP
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================================
# 2.  LOAD QWEN3 MODEL & TOKENIZER
# ============================================================================
# Change `MODEL_NAME` to whichever Qwen3 checkpoint you have access to,
# e.g. "Qwen/Qwen3-8B" or a local path.

MODEL_NAME = "Qwen/Qwen3-8B"

print(f"Loading tokenizer from: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading model from: {MODEL_NAME}  (this may take a while …)")
qwen_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,   # use fp16 to fit on a single GPU
    device_map="auto",           # automatically distributes across available GPUs
    output_hidden_states=True    # needed so hidden states are always returned
)
qwen_model.eval()

# Total number of transformer layers (excluding the embedding layer)
NUM_LAYERS = qwen_model.config.num_hidden_layers
print(f"Model loaded. Number of transformer layers: {NUM_LAYERS}")

# ============================================================================
# 3.  EMBEDDING EXTRACTION FUNCTION
# ============================================================================

@torch.no_grad()
def get_all_layer_embeddings_from_generated(sentence: str, target_word: str):
    """
    Extract the hidden-state vector of a single *generated* token at every
    transformer layer.

    Steps
    -----
    a) Build a concreteness-rating prompt for (sentence, target_word).
    b) Generate exactly 1 new token (greedy / deterministic).
    c) Re-run the model on [prompt + generated token] with hidden states.
    d) Return the hidden state of the generated token at each layer.

    Parameters
    ----------
    sentence    : str  – The carrier sentence containing the target word.
    target_word : str  – The word whose concreteness is being probed.

    Returns
    -------
    list[list[float]] of shape [NUM_LAYERS, hidden_dim], or None on failure.
    """

    # ── 3a. Build prompt ────────────────────────────────────────────────────
    prompt = (
        f'sentence: "{sentence}"\n\n'
        f"On a scale of 1 to 5 (5 being the highest), "
        f"in the context of the sentence above, what is the concreteness "
        f"of the word '{target_word}'? [MASK]"
    )

    # ── 3b. Tokenise & generate one new token ───────────────────────────────
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(qwen_model.device) for k, v in enc.items()}

    generated_ids = qwen_model.generate(
        **enc,
        max_new_tokens=1,
        do_sample=False   # deterministic; set to True + temperature for stochastic
    )
    # generated_ids shape: [1, prompt_len + 1]

    # ── 3c. Re-run full sequence to obtain hidden states ────────────────────
    attention_mask = torch.ones_like(generated_ids, device=qwen_model.device)

    outputs = qwen_model(
        input_ids=generated_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )

    # hidden_states: tuple of (embedding_layer, layer_1, …, layer_N)
    # We skip index 0 (the raw token embedding) and keep transformer layers only.
    transformer_layers = outputs.hidden_states[1:]   # length == NUM_LAYERS

    # ── 3d. Extract the hidden state at the position of the generated token ─
    last_pos = generated_ids.shape[1] - 1  # index of the newly generated token

    layer_vectors = []
    for layer_state in transformer_layers:
        # layer_state: [batch=1, seq_len, hidden_dim]
        h = layer_state[0, last_pos, :]          # [hidden_dim]
        layer_vectors.append(h.detach().cpu().tolist())

    return layer_vectors   # [NUM_LAYERS, hidden_dim]


# ============================================================================
# 4.  LOAD INPUT DATA
# ============================================================================

# ── 4a. Word–sentence dataset ───────────────────────────────────────────────
# Expected columns: `target word` | `sentence` (Wikipedia/other sources sentence)
df = pd.read_csv("sentence.csv")
print(f"Loaded dataset: {len(df)} rows")

# ── 4b. Concreteness norms (Brysbaert et al., 2014) ─────────────────────────
# Download from the paper linked in the module docstring above.
# Required columns: Word | Conc.M
# Save the file as `concreteness.csv` in this directory before running.
conc_df = pd.read_csv("concreteness.csv")
conc_df["Word"] = conc_df["Word"].astype(str).str.lower()

# Build a fast word → score lookup dictionary
conc_dict = dict(zip(conc_df["Word"], conc_df["Conc.M"]))
print(f"Loaded concreteness norms: {len(conc_dict)} entries")

# ============================================================================
# 5.  PREPARE OUTPUT DIRECTORY & PER-LAYER FILE WRITERS
# ============================================================================

OUTPUT_DIR = "qwen_emb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# One .jsonl writer per transformer layer
writers = [
    open(os.path.join(OUTPUT_DIR, f"layer_{i}.jsonl"), "w", encoding="utf-8")
    for i in range(NUM_LAYERS)
]

# ============================================================================
# 6.  MAIN EXTRACTION LOOP
# ============================================================================

skipped = 0

for i in tqdm(range(len(df)), desc="Extracting embeddings"):
    word = str(df.loc[i, "target word"]).lower()
    sent = str(df.loc[i, "sentence"])

    # Look up concreteness score; skip words not in the Brysbaert norms
    score = conc_dict.get(word, None)
    if score is None:
        skipped += 1
        continue

    # Extract hidden states across all layers
    layer_vecs = get_all_layer_embeddings_from_generated(sent, word)
    if layer_vecs is None:
        skipped += 1
        continue

    # Write one JSON record per layer
    for layer_idx, vec in enumerate(layer_vecs):
        record = {
            "word":      word,
            "sentence":  sent,
            "conc":      score,
            "embedding": vec
        }
        writers[layer_idx].write(json.dumps(record) + "\n")

# ── Close all file handles ──────────────────────────────────────────────────
for w in writers:
    w.close()

# ============================================================================
# 7.  SUMMARY
# ============================================================================

print(f"\n✅ Extraction complete.")
print(f"   Output folder : {OUTPUT_DIR}/")
print(f"   Files created : layer_0.jsonl … layer_{NUM_LAYERS - 1}.jsonl")
print(f"   Rows skipped  : {skipped}  (word not found in concreteness norms)")