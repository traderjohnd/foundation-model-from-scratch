# Decision Register — 03 Model Architecture

Notebook 03 defines and verifies the modern decoder-only Transformer family. Early design decisions are preserved with their original IDs; later implementation evidence refines them without rewriting history.

---

## D-012 — Context length
**Selected choice:** **512 tokens**.  
**Why:** Practical balance among information available per prediction, memory, compute, and Colab-scale feasibility.  
**Alternatives considered:** 256, 1,024, longer contexts.  
**Presentation relevance:** context length affects available information and attention cost, not the mathematical definition of next-token cross-entropy.

## D-013 — Model architecture family
**Selected choice:** **modernized decoder-only Transformer** with causal MHA, RoPE, RMSNorm, SwiGLU, residual connections, pre-norm, and tied input/output weights.  
**Why:** The project should teach how current decoder-only architectures evolved beyond classic Transformer/GPT/BERT-era conventions.  
**Alternatives considered:** classic GPT-style LayerNorm/GELU/absolute positions; encoder-decoder; encoder-only.  
**Presentation relevance:** central architecture-evolution section.

## D-014 — RoPE
**Selected choice:** Rotary Position Embeddings on attention Q/K.  
**Why:** Representative modern positional method that avoids a learned absolute position table.  
**Alternatives considered:** sinusoidal; learned absolute embeddings.  
**Presentation relevance:** compare early vs modern positional approaches.

## D-015 — RMSNorm
**Selected choice:** RMSNorm.  
**Why:** Simpler modern normalization widely used in decoder-only LLMs.  
**Alternative:** LayerNorm.  
**Presentation relevance:** modern architecture evolution.

## D-016 — SwiGLU
**Selected choice:** SwiGLU feed-forward block.  
**Why:** Modern gated FFN design with different parameter economics than classic ReLU/GELU FFNs.  
**Alternatives considered:** ReLU, GELU.  
**Presentation relevance:** architecture and parameter-efficiency comparison.

## D-017 — SwiGLU hidden dimensions
**Initial selected choice:** A ~704; B ~1,024; C ~1,360.  
**Why:** Keep the three models near the target 7M/17M/34M sizes while using SwiGLU rather than blindly copying a classic `4*d_model` FFN ratio.  
**Historical outcome:** exact values confirmed in D-059.

## D-018 — Weight tying
**Selected choice:** tie the token-embedding and output LM-head weight matrix.  
**Why:** A separate output matrix would consume a disproportionate share of Model A's parameter budget.  
**Alternative:** untied matrices.  
**Historical outcome:** actual shared Parameter/storage verified in D-061.  
**Presentation relevance:** concrete parameter-efficiency mechanism.

## D-030 — Dropout
**Selected choice:** 0.10 dropout across all three models.  
**Why:** The fixed 20M-token corpus and up to three passes create more overfitting risk than frontier-scale one-pass regimes.  
**Alternatives considered:** 0; larger dropout.  
**Presentation relevance:** distinguish dropout from AdamW weight decay.

## D-037 — Weight initialization
**Initial selected choice:** base weights `Normal(0,0.02)` with depth-aware scaling on residual-output projections.  
**Why:** Small random weights break symmetry; residual scaling supports stability as depth increases.  
**Alternatives considered:** Xavier, He/Kaiming, zero initialization, other depth-aware schemes.  
**Historical outcome:** exact implementation and empirical verification recorded in D-062.  
**Presentation relevance:** “from scratch” literally begins with random learned parameters.

---

## D-059 — Exact implemented model family
**Selected choice:**
- Model A: 4 layers, `d_model=256`, 4 heads, head dim 64, `d_ff=704` → **7,407,872 params**
- Model B: 6 layers, `d_model=384`, 6 heads, head dim 64, `d_ff=1024` → **16,913,280 params**
- Model C: 8 layers, `d_model=512`, 8 heads, head dim 64, `d_ff=1360` → **33,497,600 params**

**Why:** Preserves compound depth/width scaling, fixed head dimension, and lands close to the intended target ladder.  
**Supersedes/refines:** D-004, D-005, D-017 provisional values.  
**Evidence:** analytical formulas and `sum(p.numel())` agree exactly for all three models.  
**Presentation relevance:** exact scaling table.

## D-060 — Exact modern decoder-only block implementation
**Selected choice:** learned token embeddings; causal standard MHA; RoPE on Q/K only; RMSNorm; SwiGLU; pre-norm residual structure; final RMSNorm; tied token embedding/LM head; no learned position table; no linear biases; dropout .10; context 512.  
**Why:** Closes the earlier architecture choices as one explicit, auditable implementation rather than a collection of design intentions.  
**Supersedes/refines:** D-013 through D-018 and D-030 as implementation evidence.  
**Evidence:** one-block exact parameter counts A 803,328; B 1,770,240; C 3,138,560. Attention and SwiGLU component counts were also analytically verified.  
**Presentation relevance:** direct comparison between design choices and realized parameter economics.

## D-061 — Weight tying implementation evidence
**Selected choice/evidence:** `lm_head.weight` is the same `Parameter` object and same underlying storage as `token_embedding.weight`.  
**Why:** Equal values would not prove true sharing; identity/storage evidence is required for parameter-count claims.  
**Evidence:** for Model A, tying avoids a second 4,194,304-parameter vocabulary matrix.  
**Presentation relevance:** strong visual example of weight reuse.

## D-062 — Depth-aware initialization implementation
**Selected choice:** token embeddings and ordinary linears `Normal(0,.02)`; RMSNorm scales 1.0; attention-output and SwiGLU-down residual projections `0.02/sqrt(2L)`; seed 42. Residual std targets: A .007071, B .005774, C .005000.  
**Why:** Reduces each residual branch's initial contribution as depth rises while preserving a common initialization philosophy.  
**Supersedes/refines:** D-037.  
**Evidence:** observed standard deviations matched targets; RMS scales exactly one; same seed reproduced checked weights; tying/counts remained intact.

## D-063 — Final architecture audit and causality evidence
**Selected choice:** require assertions for exact dimensions/counts, blocks/norms, RoPE parameter-free behavior, no learned position table, bias-free linears, dropout .10, weight tying, context enforcement, finite logits, and causal information flow.  
**Why:** The later scaling experiment is interpretable only if model implementations are demonstrably consistent with the intended family.  
**Evidence:** PASS for A/B/C. Changing a future token caused no change in earlier attention outputs, full-block outputs, or complete-model logits.  
**Presentation relevance:** evidence that the model is truly causal rather than merely described as causal.

## D-064 — Backward/autograd sanity check
**Selected choice:** synthetic shifted next-token batch on untrained Model A, cross-entropy, then `backward()`.  
**Why:** A valid forward pass does not prove the computational graph is trainable.  
**Evidence:** PASS; loss **9.693123** (sanity-check value, not model quality). Finite nonzero gradients verified for tied embedding/LM head, Q projection, attention output, SwiGLU gate/down, final RMSNorm.  
**Presentation relevance:** closes the architecture phase before adding optimizer/data-pipeline complexity.

## Notebook 03 closure
Canonical implementation lives in `src/model.py`. Notebook 03 ended with architecture, causality, initialization, parameter-count, weight-tying, forward, and backward audits passing before Notebook 04 began.
