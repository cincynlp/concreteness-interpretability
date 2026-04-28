"""
steer_concreteness.py
======================
Steers a large language model's generations towards higher or lower
concreteness by injecting the 1-D concreteness axis directly into the
model's residual stream at a chosen transformer layer during inference.

This is an implementation of *activation steering* (also known as
representation engineering): instead of fine-tuning the model, we add a
scaled direction vector to the hidden states at a single layer, nudging
the model's internal representation towards the desired concept at
generation time.

Pipeline overview
-----------------
1. Load the model and tokenizer.
2. Load the 1-D concreteness axis (produced by `compute_diffmean.py` or
   `compute_svd_subspace.py`) and normalise it.
3. Register a forward hook on the chosen transformer layer that adds
   α × u to the hidden states at every forward pass.
4. Generate a rewrite of the input sentence with the hook active.
5. Remove the hook after generation.

Steering parameters
-------------------
Two parameters control the steering behaviour:

    TARGET_LAYER : int
        The transformer layer at which to inject the steering vector.
        Earlier layers affect syntax and semantics broadly; later layers
        tend to have stronger effects on surface-level word choice.
        In our paper, layer 20 is where figurativity is most salient for
        LLaMA-3.1-8B.

    ALPHA : float
        Scaling factor for the steering vector.
        Positive α  →  steers towards higher concreteness (more literal)
        Negative α  →  steers towards lower concreteness (more figurative)
        Larger |α|  →  stronger effect (may degrade fluency if too large)

Input files required
--------------------
- A concreteness basis file (e.g. `global_concreteness_basis_k1.npy`)
  produced by `compute_diffmean.py` or `compute_svd_subspace.py`.

Output
------
The rewritten sentence is printed to stdout.
"""

# ── Standard library ────────────────────────────────────────────────────────
# (none needed beyond third-party imports)

# ── Third-party ─────────────────────────────────────────────────────────────
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# 1.  CONFIGURATION
# ============================================================================
# ► Change MODEL_NAME to whichever checkpoint you want to steer.
# ► Change BASIS_PATH to the concreteness axis file for that model,
#   produced by compute_diffmean.py or compute_svd_subspace.py.
#
#   Steering parameters:
#       TARGET_LAYER : layer at which to inject the steering vector
#                      (in our paper, layer 20 for LLaMA-3.1-8B)
#       ALPHA        : steering strength
#                      positive → more concrete / literal
#                      negative → more abstract / figurative

MODEL_NAME   = "meta-llama/Meta-Llama-3.1-8B-Instruct"   # ← change to switch model
BASIS_PATH   = "global_concreteness_basis_k1.npy"         # ← concreteness axis file
TARGET_LAYER = 20                                          # ← injection layer
ALPHA        = 20.0                                        # ← steering strength

# ============================================================================
# 2.  LOAD MODEL & TOKENIZER
# ============================================================================

print(f"Loading tokenizer from: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading model from: {MODEL_NAME}  (this may take a while …)")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,   # fp16 to fit on a single GPU
    device_map="auto",           # automatically distributes across available GPUs
    output_hidden_states=True
)
model.eval()

# ============================================================================
# 3.  LOAD & PREPARE CONCRETENESS AXIS
# ============================================================================

u = np.load(BASIS_PATH).astype(np.float16)
u = u / (np.linalg.norm(u) + 1e-8)          # L2-normalise
u_tensor = torch.tensor(u, device=model.device)

print(f"Loaded concreteness axis from '{BASIS_PATH}'  |  dim = {u_tensor.shape[0]}")
print(f"Steering config: TARGET_LAYER={TARGET_LAYER}, ALPHA={ALPHA:+.1f}  "
      f"({'→ more concrete/literal' if ALPHA > 0 else '→ more abstract/figurative'})\n")

# ============================================================================
# 4.  STEERING HOOK
# ============================================================================

def steering_hook(module, inputs, output):
    """
    Forward hook that injects the concreteness axis into the residual stream.

    Added to a single transformer layer; called automatically on every forward
    pass while the hook is registered. The hook adds α × u to the hidden
    states, then returns the modified output while preserving all other
    elements of the layer's output tuple unchanged.
    """
    hidden_states = output[0]

    # Unwrap if hidden states are nested in a tuple
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]

    # Match device and dtype of the current hidden states
    steer_vec = u_tensor.to(hidden_states.device).to(hidden_states.dtype)

    # Inject: add α × u to every token position in the batch
    hidden_states = hidden_states + ALPHA * steer_vec

    # Return updated hidden states; preserve all other layer outputs unchanged
    return (hidden_states,) + output[1:]

# ============================================================================
# 5.  GENERATION FUNCTION
# ============================================================================

def rewrite(sentence: str) -> str:
    """
    Rewrite a sentence with the concreteness steering hook active.

    The hook is registered before generation and removed immediately after,
    so subsequent calls to the model outside this function are unaffected.

    Parameters
    ----------
    sentence : str – The input sentence to rewrite.

    Returns
    -------
    str – The model's rewritten output.
    """
    prompt = (
        "Rewrite the following sentence clearly and naturally:\n"
        f'"{sentence}"\nRewrite:'
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Register the hook on the chosen transformer layer
    target_module = model.model.layers[TARGET_LAYER]
    handle = target_module.register_forward_hook(steering_hook)

    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    finally:
        # Always remove the hook after generation, even if an error occurs
        handle.remove()

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text.split("Rewrite:")[-1].strip()

# ============================================================================
# 6.  DEMO
# ============================================================================

if __name__ == "__main__":
    test_sentence = "There is a bridge of trust that has developed between us."

    print(f"INPUT  : {test_sentence}")
    print(f"α      : {ALPHA:+.1f}")

    result = rewrite(test_sentence)
    print(f"OUTPUT : {result}")