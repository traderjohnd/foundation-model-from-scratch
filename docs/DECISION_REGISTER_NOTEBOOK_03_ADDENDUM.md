# Decision Register Addendum — Notebook 03 Final Architecture Evidence

This addendum records the implementation evidence that closes Notebook 03. It supplements `docs/DECISION_REGISTER.md` without changing the earlier decision history.

## D-059 — Exact implemented model family

**Decision**  
Finalize the previously targeted ~7M / ~17M / ~34M scaling family after full PyTorch implementation.

**Selected choice**
- Model A: 4 layers, `d_model=256`, 4 heads, head dim 64, `d_ff=704`
- Model B: 6 layers, `d_model=384`, 6 heads, head dim 64, `d_ff=1024`
- Model C: 8 layers, `d_model=512`, 8 heads, head dim 64, `d_ff=1360`

Exact implemented parameter counts:
- Model A: **7,407,872**
- Model B: **16,913,280**
- Model C: **33,497,600**

**Rationale**  
These dimensions preserve compound depth/width scaling while keeping a fixed 64-dimensional head and landing very close to the intended 7M / 17M / 34M family.

**Status / evidence**  
Implemented and verified in the final executed `notebooks/03_model_architecture.ipynb`. Analytical parameter formulas and `sum(p.numel())` agree exactly for all three models.

---

## D-060 — Exact modern decoder-only block implementation

**Decision**  
Close the design-level architecture choices as implementation-verified.

**Selected choice**
- learned token embeddings
- causal standard multi-head self-attention
- RoPE on Q and K only
- RMSNorm
- SwiGLU
- pre-norm residual structure
- final RMSNorm
- tied token-embedding / LM-head matrix
- no learned positional embedding table
- no linear biases
- dropout = 0.10
- context length = 512

**Rationale**  
This preserves the modernized decoder-only family selected earlier while keeping the implementation explicit enough for teaching and auditing.

**Status / evidence**  
Implemented and verified. One-block exact parameter counts are:
- Model A: 803,328
- Model B: 1,770,240
- Model C: 3,138,560

Attention parameters per block:
- A: 262,144
- B: 589,824
- C: 1,048,576

SwiGLU parameters per block:
- A: 540,672
- B: 1,179,648
- C: 2,088,960

---

## D-061 — Weight tying implementation evidence

**Decision**  
Verify that input embeddings and output vocabulary projection share one learned parameter tensor rather than merely having equal values.

**Selected choice**  
`lm_head.weight` is the same `Parameter` object and the same underlying storage as `token_embedding.weight`.

**Rationale**  
True sharing is required for the intended parameter efficiency and for the analytical parameter counts to remain valid.

**Status / evidence**  
Verified in Notebook 03. For Model A, tying avoids a second 4,194,304-parameter vocabulary matrix.

---

## D-062 — Depth-aware initialization implementation

**Decision**  
Finalize the explicit random initialization policy rather than relying on PyTorch defaults.

**Selected choice**
- token embeddings and ordinary linear weights: `Normal(mean=0, std=0.02)`
- RMSNorm scale vectors: 1.0
- attention output projections and SwiGLU down projections: `0.02 / sqrt(2L)`
- model seed: 42

Residual-output standard deviations:
- Model A (`L=4`): **0.007071**
- Model B (`L=6`): **0.005774**
- Model C (`L=8`): **0.005000**

**Rationale**  
The residual-output scaling reduces the initial contribution of each residual branch as depth increases while preserving a common base initialization philosophy across the family.

**Status / evidence**  
Implemented and empirically validated. Observed standard deviations matched targets within tolerance; RMSNorm weights initialized exactly to one; same-seed initialization reproduced identical checked weights; weight tying and exact parameter counts remained intact.

---

## D-063 — Final architecture audit and causality evidence

**Decision**  
Require a complete architecture audit before moving to the training pipeline.

**Selected choice**  
Notebook 03 must assert and verify:
- exact model dimensions
- exact parameter counts
- correct number of blocks and RMSNorms
- RoPE parameter-free behavior
- no learned positional embedding table
- all linear biases disabled
- dropout = 0.10
- tied embedding/output weights
- context-length enforcement
- finite forward logits
- causal information flow

**Rationale**  
The training experiment is only interpretable if the implementation is demonstrably consistent with the design specification.

**Status / evidence**  
Final executed Notebook 03 audit: **PASS** for Models A/B/C.

Causality was verified at multiple levels. Changing a future token produced no change in earlier attention outputs, no change in earlier full-block outputs, and no change in earlier complete-model logits.

---

## D-064 — Backward/autograd architecture sanity check

**Decision**  
Verify that a synthetic next-token objective can propagate gradients through all major learned components before beginning the real training loop.

**Selected choice**  
Use a tiny synthetic token batch, shift inputs and labels by one position, compute cross-entropy, and run `backward()` on untrained Model A.

**Rationale**  
A valid forward pass alone does not prove the training graph is usable. The architecture should demonstrate finite, nonzero gradients before Notebook 04 adds optimizer and data-pipeline complexity.

**Status / evidence**  
PASS. Synthetic next-token cross-entropy loss: **9.693123**. Finite, nonzero gradients were verified for:
- tied embedding / LM head
- attention Q projection
- attention output projection
- SwiGLU gate projection
- SwiGLU down projection
- final RMSNorm

This value is not a model-quality result; it is an autograd sanity check on an untrained model.

---

# Notebook 03 phase closure

Notebook 03 — **Model Architecture** is complete and canonical on `main` as:
- `notebooks/03_model_architecture.ipynb`

The architecture phase is closed. The next phase is **Notebook 04 — Training Pipeline**.

Notebook 04 must begin in a **new context window** and load the canonical GitHub context/decision files before implementation.

The first substantive Notebook 04 task is to resolve D-058: define the deterministic mapping from the exact 20,000,000-token corpus to 512-token causal inputs and shifted next-token targets, including the treatment of the 256-token arithmetic remainder.
