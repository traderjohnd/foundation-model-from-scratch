# Building a Foundation Model from Scratch — Decision Register

## Purpose
This register is the detailed source of truth for project decisions. It is intended to serve both as an engineering record and as a possible appendix to the final presentation.

For each decision, preserve:
- the decision
- selected choice
- rationale
- alternatives considered
- presentation relevance
- status/evidence

---

## D-001 — Project scope

**Decision**  
Separate from-scratch pretraining from later fine-tuning work.

**Selected choice**  
This project covers training small decoder-only language models from random initialization. Fine-tuning will be a separate later project using a stronger existing open-weight pretrained model.

**Rationale**  
Fine-tuning the small from-scratch model would not be the strongest demonstration of adaptation capability. Separating the projects makes each competency clearer.

**Alternatives considered**
- Pretrain and fine-tune the same small model
- Combine pretraining and fine-tuning into one portfolio artifact

**Presentation relevance**  
Clearly distinguish:
- pretraining from scratch
- fine-tuning an existing foundation model

**Status / evidence**  
Locked before implementation.

---

## D-002 — Experimental objective

**Decision**  
Define the main project objective.

**Selected choice**  
> Build a series of small decoder-only Transformer language models from scratch and systematically scale them to observe how model capacity affects learning, compute cost, and generated language.

**Rationale**  
A controlled scaling experiment is more technically meaningful than simply training one toy language model.

**Alternatives considered**
- Train one ~20M model only
- Build a toy model solely to demonstrate the training loop

**Presentation relevance**  
This is the primary objective slide.

**Status / evidence**  
Locked.

---

## D-003 — Experimental question

**Decision**  
Define what the scaling experiment should answer.

**Selected choice**  
> How does increasing Transformer capacity affect language-model performance when the dataset and training methodology are controlled, and what tradeoffs emerge between capability and computational efficiency?

Higher-level question:
> At what point does increasing model capacity produce diminishing returns when training data and compute are constrained?

**Rationale**  
"Are bigger models better?" is too elementary. The chosen question focuses on real engineering tradeoffs among capacity, data, compute, and performance.

**Presentation relevance**  
Research question and conclusion framework.

**Status / evidence**  
Locked. Final evidence will come from training results.

---

## D-004 — Number of models and target scales

**Decision**  
Use three progressively scaled models.

**Selected choice**
- Model A: ~7M parameters
- Model B: ~17M parameters
- Model C: ~34M parameters

**Rationale**  
Creates a visible scaling progression while remaining practical for inexpensive single-GPU experimentation.

**Alternatives considered**
- One model only
- Different size ladders
- Scale only depth or only width

**Presentation relevance**  
Supports controlled scaling charts and resource/performance comparisons.

**Status / evidence**  
Locked. Exact parameter counts will be computed after implementation.

---

## D-005 — Scaling strategy

**Decision**  
How to increase model capacity.

**Selected choice**  
Compound scaling: increase both depth and width while keeping architectural ratios coherent.

Provisional family:
- A: 4 layers, width 256, 4 heads
- B: 6 layers, width 384, 6 heads
- C: 8 layers, width 512, 8 heads
- 64 dimensions per head

**Rationale**  
Represents progressively scaled members of the same model family rather than arbitrarily adding layers.

**Alternatives considered**
- Hold width constant and add layers only
- Hold depth constant and increase width only
- Independently optimize each model architecture

**Presentation relevance**  
Explain width vs depth and why width can be expensive because many parameter terms scale approximately with `d_model^2`.

**Status / evidence**  
Locked at target-design level.

---

## D-006 — Source dataset

**Decision**  
Choose a meaningful, general-purpose text corpus.

**Selected choice**  
**WikiText-103**

**Rationale**  
Real Wikipedia prose is more linguistically complex and presentation-relevant than simplified synthetic datasets such as TinyStories.

**Alternatives considered**
- TinyStories
- other general corpora
- domain-specific corpora

**Presentation relevance**  
Explain why realistic text makes the scaling experiment harder but more meaningful.

**Status / evidence**  
Locked.

---

## D-007 — Controlled training-token budget

**Decision**  
How much WikiText-103 training data to use.

**Selected choice**  
Exactly **20 million tokenizer-produced training tokens**.

**Rationale**  
Keeps compute affordable while intentionally creating a regime where the larger model may become constrained by data/compute, which supports the diminishing-returns research question.

**Alternatives considered**
- Use all ~103M training tokens
- 10M tokens
- 50M tokens

**Presentation relevance**  
Supports discussion of model size vs token budget vs compute-optimal training.

**Status / evidence**  
Locked.

---

## D-008 — Hold dataset constant across models

**Decision**  
Whether each model should receive the same data.

**Selected choice**  
Yes. Use the identical controlled 20M-token corpus for all three models.

**Rationale**  
If both data and model size changed, causal interpretation of performance differences would weaken.

**Presentation relevance**  
Core experimental-design principle: control variables not being tested.

**Status / evidence**  
Locked.

---

## D-009 — Tokenizer ownership

**Decision**  
Whether to reuse an existing tokenizer.

**Selected choice**  
Train the tokenizer from scratch.

**Rationale**  
The project is intended to reproduce the complete pretraining pipeline rather than inherit a pretrained model's linguistic interface.

**Alternatives considered**
- Reuse GPT tokenizer
- Reuse BERT tokenizer
- Reuse another open-model tokenizer

**Presentation relevance**  
Makes the pipeline genuinely start from raw text.

**Status / evidence**  
Locked.

---

## D-010 — Tokenizer algorithm

**Decision**  
Choose tokenizer family.

**Selected choice**  
**Byte-level BPE**

**Rationale**  
Provides efficient subword modeling while retaining robust byte-level coverage of arbitrary text. Representative of GPT-style tokenization.

**Alternatives considered**
- word-level
- character-level
- WordPiece
- Unigram / SentencePiece
- conventional BPE without byte-level fallback

**Presentation relevance**  
Explicitly compare BERT/WordPiece—the user's original NLP reference point—with newer GPT-style byte-aware tokenization.

**Status / evidence**  
Locked.

---

## D-011 — Vocabulary size

**Decision**  
Choose tokenizer vocabulary size.

**Selected choice**  
**16,384 tokens**

**Rationale**  
Balances:
- embedding/output parameter cost
- sequence length
- compute

A 32K vocabulary would consume a much larger share of the parameter budget in the smallest model.

**Alternatives considered**
- 8K
- 32K
- 50K+
- 100K+ production-style vocabularies

**Presentation relevance**  
Show that vocabulary size affects both parameter count and sequence efficiency:
- larger vocabulary → larger embedding/output matrices, potentially shorter sequences
- smaller vocabulary → smaller matrices, potentially longer sequences and more attention compute

**Status / evidence**  
Locked.

---

## D-012 — Context length

**Decision**  
Choose training context length.

**Selected choice**  
**512 tokens**

**Rationale**  
Reasonable balance for small-model Colab training.

**Alternatives considered**
- 256
- 1,024
- longer contexts

**Presentation relevance**  
Explain later that context length does not change the next-token objective or cross-entropy formula, but changes information available to each prediction, memory/compute requirements, and long-range dependency learning.

**Status / evidence**  
Locked. Detailed teaching deferred.

---

## D-013 — Model architecture family

**Decision**  
Classic GPT-style block vs modernized decoder-only Transformer.

**Selected choice**  
**Modernized decoder-only Transformer**

Core choices:
- causal multi-head attention
- RoPE
- RMSNorm
- SwiGLU
- residual connections
- pre-normalization-style modern design
- tied embeddings/output head

**Rationale**  
The user originally learned classic Transformer/GPT/BERT-era components; the project should demonstrate how modern decoder-only architectures evolved.

**Alternatives considered**
- classic GPT-style learned/sinusoidal positions + LayerNorm + GELU
- encoder-decoder Transformer
- encoder-only model

**Presentation relevance**  
Major teaching section comparing the 2017 paper with modern LLM architecture.

**Status / evidence**  
Locked.

---

## D-014 — RoPE

**Decision**  
Positional encoding approach.

**Selected choice**  
**Rotary Position Embeddings (RoPE)**

**Rationale**  
Representative of many modern decoder-only LLMs and useful for teaching architectural evolution beyond early sinusoidal/learned positional embeddings.

**Alternatives considered**
- sinusoidal positional encoding
- learned absolute positional embeddings

**Presentation relevance**  
Explain later; do not over-teach before implementation.

**Status / evidence**  
Locked.

---

## D-015 — Normalization

**Decision**  
Normalization method.

**Selected choice**  
**RMSNorm**

**Rationale**  
Representative modern decoder-only choice; simpler than LayerNorm and commonly used in newer LLM architectures.

**Alternatives considered**
- LayerNorm

**Presentation relevance**  
Compare to classic Transformer/BERT/GPT conventions.

**Status / evidence**  
Locked.

---

## D-016 — Feed-forward activation/block

**Decision**  
Feed-forward sublayer design.

**Selected choice**  
**SwiGLU**

**Rationale**  
Modern gated feed-forward design used in many contemporary LLM architectures.

**Alternatives considered**
- ReLU FFN
- GELU FFN

**Presentation relevance**  
Explain architecture evolution and parameter-budget implications.

**Status / evidence**  
Locked.

---

## D-017 — SwiGLU hidden dimensions

**Decision**  
Provisional feed-forward hidden sizes.

**Selected choice**
- A: ~704
- B: ~1,024
- C: ~1,360

**Rationale**  
Preserve sensible parameter economics for SwiGLU while keeping the overall models near the agreed 7M/17M/34M targets.

**Alternatives considered**
- blindly reuse classic `4 × d_model` FFN dimensions

**Presentation relevance**  
Shows that architectural changes alter parameter economics.

**Status / evidence**  
Provisional; verify after exact implementation.

---

## D-018 — Weight tying

**Decision**  
Separate or shared input embeddings and output projection.

**Selected choice**  
**Tie input embedding and output projection weights.**

**Rationale**  
The 16K vocabulary would otherwise consume a disproportionate share of the smallest model's parameters.

**Alternatives considered**
- untied input/output matrices

**Presentation relevance**  
Excellent example of how architectural choices change parameter efficiency.

**Status / evidence**  
Locked.

---

## D-019 — Training objective

**Decision**  
Choose pretraining objective.

**Selected choice**  
**Autoregressive causal language modeling / next-token prediction**

**Rationale**  
Standard objective for GPT-style decoder-only language models and directly supports controlled model-size comparison.

**Alternatives considered**
- masked language modeling
- encoder-decoder denoising objectives
- supervised task objectives

**Presentation relevance**  
Explain next-token prediction, causal masking, autoregression, and self-supervised learning.

**Status / evidence**  
Locked.

---

## D-020 — Loss function

**Decision**  
Choose training loss.

**Selected choice**  
**Cross-entropy loss**

**Rationale**  
Directly measures the penalty for assigning too little probability to the true next token.

**Alternatives considered**
- not meaningfully competitive for this standard objective

**Presentation relevance**  
Explain probability distribution over vocabulary and how loss drives backpropagation.

**Status / evidence**  
Locked.

---

## D-021 — Perplexity

**Decision**  
Primary language-model evaluation metric.

**Selected choice**  
Track **validation perplexity for all three models**, with final test perplexity at end.

**Rationale**  
Perplexity is a human-interpretable transformation of cross-entropy and is appropriate for comparing models when tokenizer and evaluation data are identical.

**Alternatives considered**
- loss only
- generation quality only

**Presentation relevance**  
Explain `perplexity = exp(cross-entropy)` and why lower is better.

**Status / evidence**  
Locked.

---

## D-022 — Optimizer

**Decision**  
Choose optimizer.

**Selected choice**  
**AdamW**

**Rationale**  
Strong standard choice for Transformer training; adaptive optimization plus decoupled weight decay.

**Alternatives considered**
- SGD
- Adam
- Adafactor
- Lion
- Muon / newer optimizers

**Presentation relevance**  
Explain AdamW and alternatives. Distinguish weight decay from dropout.

**Status / evidence**  
Locked.

---

## D-023 — Learning-rate experiment

**Decision**  
Do not treat learning rate as a copied default.

**Selected choice**  
Brief Model A experiment with candidate peak LRs:
- `1e-4`
- `3e-4`
- `1e-3`

Provisional expected production choice:
- **~3e-4**

**Rationale**  
Shows empirically what under-aggressive, appropriate, and potentially unstable learning rates look like.

**Alternatives considered**
- choose one LR with no test
- exhaustive hyperparameter search

**Presentation relevance**  
Use learning curves to explain convergence speed vs stability.

**Status / evidence**  
Locked as an experiment; final LR decided after probe run.

---

## D-024 — Learning-rate schedule

**Decision**  
Whether learning rate remains constant.

**Selected choice**  
**Warmup followed by cosine decay**

**Rationale**  
Avoids aggressive early updates from random initialization while allowing productive larger steps followed by careful refinement.

**Alternatives considered**
- constant LR
- linear decay
- other schedules

**Presentation relevance**  
Potential visual showing LR curve alongside training loss.

**Status / evidence**  
Locked.

---

## D-025 — Effective batch size

**Decision**  
Choose batch size in a hardware-independent way.

**Selected choice**  
**16,384 tokens per optimizer update**

With 512-token sequences, roughly 32 sequences contribute to each update.

**Rationale**  
Keeps optimization behavior consistent even if the largest model requires smaller physical microbatches.

**Alternatives considered**
- same physical batch size regardless of GPU memory
- different effective batch sizes for each model

**Presentation relevance**  
Explain microbatch, effective batch, gradient accumulation, memory, gradient noise, and learning-rate interactions.

**Status / evidence**  
Locked.

---

## D-026 — Gradient accumulation

**Decision**  
How to maintain effective batch size across model sizes.

**Selected choice**  
Use gradient accumulation as needed.

**Rationale**  
Allows the larger model to use smaller microbatches without changing the effective token batch per optimizer update.

**Presentation relevance**  
Practical training-engineering concept.

**Status / evidence**  
Locked.

---

## D-027 — Training duration

**Decision**  
Number of corpus passes.

**Selected choice**  
Maximum **3 epochs**.

**Rationale**  
One pass may be insufficient for useful learning curves; many passes increase overfitting/memorization risk.

**Alternatives considered**
- 1 epoch
- many repeated epochs
- fixed steps independent of corpus passes

**Presentation relevance**  
Supports overfitting/generalization discussion.

**Status / evidence**  
Locked.

---

## D-028 — Best-checkpoint policy

**Decision**  
Whether final epoch is assumed best.

**Selected choice**  
No. Save and retain the checkpoint with the best validation performance.

**Rationale**  
Training loss can continue falling after validation performance stops improving.

**Presentation relevance**  
Visual explanation of overfitting.

**Status / evidence**  
Locked.

---

## D-029 — Data split strategy

**Decision**  
Training/validation/test organization.

**Selected choice**
- 20M training tokens drawn only from WikiText-103 official training split
- official validation split for development/tuning/checkpoint selection
- official test split untouched until final evaluation

**Rationale**  
Preserve benchmark-defined held-out sets rather than force an arbitrary percentage split.

**Alternatives considered**
- 80/10/10
- 95/5 train/validation only
- custom repartition

**Presentation relevance**  
Explicitly teach all three roles and explain why 80/10/10 is a rule of thumb, not a universal law.

**Status / evidence**  
Locked.

---

## D-030 — Dropout

**Decision**  
Regularization through stochastic activation dropping.

**Selected choice**  
**0.10 dropout**

**Rationale**  
The project has only 20M unique training tokens and up to 3 epochs, creating more overfitting risk than trillion-token frontier pretraining.

**Alternatives considered**
- 0.0
- larger dropout

**Presentation relevance**  
Compare training regimes and distinguish dropout from AdamW weight decay.

**Status / evidence**  
Locked.

---

## D-031 — Training precision

**Decision**  
Numerical precision during training.

**Selected choice**
- preferred: **BF16 mixed precision**
- fallback: **FP16**
- framework may use FP32 for sensitive operations

**Rationale**  
Reduces memory and increases training speed while maintaining practical stability.

**Alternatives considered**
- full FP32

**Presentation relevance**  
Explain FP32 vs FP16 vs BF16. Briefly distinguish mixed precision from quantization.

**Status / evidence**  
Locked, subject to Colab GPU capability.

---

## D-032 — Quantization scope

**Decision**  
How deeply to cover quantization in this project.

**Selected choice**  
Introduce briefly only.

**Rationale**  
The user has a separate open-weight quantization project that will document quantization in depth.

**Presentation relevance**  
Mention as related efficiency technique without derailing the foundation-model training narrative.

**Status / evidence**  
Locked.

---

## D-033 — Validation cadence

**Decision**  
How often to evaluate during training.

**Selected choice**
- every **200 optimizer steps**
- plus end of each epoch

**Rationale**  
Frequent enough to generate useful learning curves and detect overfitting without excessive evaluation overhead.

**Alternatives considered**
- epoch-only validation
- much more frequent validation

**Presentation relevance**  
Produces evidence for training vs validation curves.

**Status / evidence**  
Locked.

---

## D-034 — Checkpoint-save rule

**Decision**  
When to save model checkpoints.

**Selected choice**  
Save whenever validation loss reaches a new best.

**Rationale**  
Ensures the strongest generalizing state is preserved.

**Presentation relevance**  
Connect model selection to validation rather than test data.

**Status / evidence**  
Locked.

---

## D-035 — Random seed

**Decision**  
Reproducibility policy.

**Selected choice**  
Primary seed: **42**

**Rationale**  
Controls stochasticity from initialization, shuffling, dropout, and related operations.

**Alternatives considered**
- no fixed seed
- multiple full runs per model

**Presentation relevance**  
Explain reproducibility vs statistical significance.

**Status / evidence**  
Locked.

---

## D-036 — Multi-seed testing

**Decision**  
Whether to run every model multiple times.

**Selected choice**  
Not initially. Optional reruns if results are suspicious or models are very close.

**Rationale**  
Multiple seeds improve statistical rigor but multiply compute cost.

**Presentation relevance**  
Honest statement of experimental limitations.

**Status / evidence**  
Locked as optional.

---

## D-037 — Weight initialization

**Decision**  
Initial random-weight distribution.

**Selected choice**
- base: `Normal(mean=0, std=0.02)`
- depth-aware scaling for residual-output projections

**Rationale**  
Small random weights are conventional and allow symmetry breaking; depth-aware scaling supports stable residual accumulation.

**Alternatives considered**
- Xavier/Glorot
- He/Kaiming
- zero initialization
- other depth-aware schemes

**Presentation relevance**  
Explain what "from scratch" literally means: architecture exists, but learned knowledge does not.

**Status / evidence**  
Locked.

---

## D-038 — Gradient clipping

**Decision**  
Whether to constrain unusually large gradients.

**Selected choice**  
Global gradient norm clipping at **1.0**

**Rationale**  
Provides a safety rail against destabilizing updates, especially early in mixed-precision training.

**Presentation relevance**  
Brief implementation concept, not a major slide.

**Status / evidence**  
Locked.

---

## D-039 — Qualitative generation evaluation

**Decision**  
How to assess generated language beyond perplexity.

**Selected choice**  
Standardized generation probe with fixed prompts and decoding settings.

Development:
- fixed prompts from validation data

Final:
- separate fixed prompts from untouched test data after model selection

Provisional decoding:
- temperature `0.8`
- top-p `0.9`

Rubric:
1. fluency / grammar
2. local coherence
3. topical continuity
4. repetition / degeneration

**Rationale**  
Perplexity is quantitative but does not show what improvement looks like to a human.

**Alternatives considered**
- random ad hoc prompts
- factual-accuracy scoring

**Presentation relevance**  
Side-by-side output from 7M vs 17M vs 34M may be one of the strongest visuals.

**Status / evidence**  
Locked; prompt set to be created later.

---

## D-040 — Factual accuracy metric

**Decision**  
Whether factual accuracy should be a primary evaluation.

**Selected choice**  
No.

**Rationale**  
These are small models trained on only 20M WikiText tokens. The experiment is about language modeling and scaling, not building a reliable factual knowledge system.

**Presentation relevance**  
Shows appropriate metric selection.

**Status / evidence**  
Locked.

---

## D-041 — Resource metrics

**Decision**  
What compute/efficiency evidence to collect.

**Selected choice**
- GPU type
- peak GPU memory
- wall-clock training time
- GPU-hours
- tokens/sec
- optimizer-step throughput
- total token exposures
- estimated cost

After training:
- inference latency
- inference throughput

**Rationale**  
Compute cost is part of the experimental objective.

**Alternatives considered**
- training time only
- mandatory energy/FLOPs accounting

**Presentation relevance**  
Supports capability-vs-cost comparison table.

**Status / evidence**  
Locked.

---

## D-042 — Energy/FLOPs measurement

**Decision**  
Whether energy consumption and theoretical FLOPs are mandatory.

**Selected choice**  
No. Add only if they can be measured or estimated cleanly later.

**Rationale**  
Avoid measurement overhead becoming the project.

**Presentation relevance**  
Potential optional appendix metric.

**Status / evidence**  
Deferred.

---

## D-043 — Implementation approach

**Decision**  
High-level Hugging Face Trainer vs explicit PyTorch implementation.

**Selected choice**  
**Option B: explicit PyTorch model and training loop**

Build directly:
- model architecture
- RMSNorm
- RoPE
- causal attention
- SwiGLU
- residuals
- forward pass
- loss setup
- training loop
- gradient accumulation
- validation loop
- checkpointing
- generation loop

**Rationale**  
The project is a relearning exercise intended to expose the mechanics.

**Alternatives considered**
- Hugging Face Trainer
- high-level AutoModel configuration

**Presentation relevance**  
Major learning-value decision.

**Status / evidence**  
Locked.

---

## D-044 — Appropriate library use

**Decision**  
Whether "from scratch" means avoiding mature libraries.

**Selected choice**  
Use mature infrastructure for commodity functionality:
- PyTorch
- Hugging Face Datasets
- Hugging Face Tokenizers
- Google Colab

Do not use:
- pretrained model weights
- pretrained tokenizer
- Hugging Face Trainer for the main loop

**Rationale**  
Good engineering should expose the mechanism being learned without unnecessarily reimplementing tensor libraries or dataset infrastructure.

**Presentation relevance**  
Key statement:
> From scratch does not mean without libraries; it means the learned model, tokenizer, architecture, and training process are not inherited from a pretrained model.

**Status / evidence**  
Locked.

---

## D-045 — Hugging Face Trainer teaching section

**Decision**  
Whether to explain what HF Trainer would automate.

**Selected choice**  
Yes. Include a dedicated batched explanation covering:
- batching/shuffling/data loading
- device placement
- forward/backward loop
- gradient accumulation
- optimizer integration
- LR scheduling/warmup
- mixed precision
- gradient clipping
- evaluation cadence and metrics
- logging
- checkpoint save/load
- best-model selection
- resume-from-checkpoint
- distributed/multi-GPU conveniences

**Rationale**  
Shows both underlying knowledge and awareness of production abstractions.

**Presentation relevance**  
Dedicated comparison slide/section.

**Status / evidence**  
Locked.

---

## D-046 — Systems-level presentation thesis

**Decision**  
How to address the claim that AI is "just next-token generation."

**Selected choice**  
Teach a layered systems view:

> Autoregressive next-token prediction is the base training/generation mechanism, but it is not a complete description of modern AI-system behavior.

Behavior can be shaped by:
- training data
- architecture
- pretraining objective
- post-training/alignment
- prompting/context
- retrieval/tools/orchestration
- guardrails
- governance
- final output/action restrictions

**Rationale**  
The model's probability mechanism is only one layer of the deployed system.

**Presentation relevance**  
Potential major section:
**"Next-Token Prediction Is the Mechanism, Not the Whole System."**

Potential analogy:
> Saying an AI system is "just next-token prediction" is like saying a computer is "just transistors switching on and off."

Potential stack:
**Data → Architecture → Pretraining → Post-training → Context → Tools → Guardrails → Governance → Output**

**Status / evidence**  
Locked as presentation thesis.

---

## D-047 — Historical architecture framing

**Decision**  
How to connect the project to the user's original NLP learning.

**Selected choice**  
Explicitly teach progression from:
- 2017 Transformer encoder-decoder architecture
- BERT / WordPiece / encoder-centric NLP
- early GPT-style decoder-only models
- modern decoder-only LLMs

**Rationale**  
The user learned classic GPT/BERT-era concepts and wants to understand how current architecture differs.

**Presentation relevance**  
Important historical/technical bridge.

**Status / evidence**  
Locked.

---

## D-048 — Repository structure

**Decision**  
How to organize implementation, experiments, results, and presentation evidence.

**Selected choice**
- `README.md`
- `docs/`
- `notebooks/`
- `src/`
- `configs/`
- `results/`
- `figures/`
- `checkpoints/`

Four notebooks:
1. data and tokenizer
2. model architecture
3. training pipeline
4. evaluation and scaling

**Rationale**  
Keeps learning chunks coherent and prevents one giant notebook from mixing everything together.

**Presentation relevance**  
The project structure itself demonstrates engineering discipline.

**Status / evidence**  
Locked.

---

## D-049 — Canonical documentation

**Decision**  
How to preserve context across chats and later build the presentation appendix.

**Selected choice**  
Maintain:
1. `PROJECT_CONTEXT.md`
2. `DECISION_REGISTER.md`

**Rationale**  
Project memory helps continuity, but canonical files provide precision and prevent reconstruction of the full conversation every time.

**Presentation relevance**  
`DECISION_REGISTER.md` can become a presentation appendix.

**Status / evidence**  
Locked.

---

## D-050 — Documentation update cadence

**Decision**  
When to update canonical docs.

**Selected choice**  
Update after major phases:
- planning complete
- tokenizer complete
- architecture complete
- training complete
- evaluation complete

**Rationale**  
Keeps new chats efficient and preserves the rationale while it is fresh.

**Presentation relevance**  
Ensures the final story is built continuously rather than reconstructed.

**Status / evidence**  
Locked.

---


## D-051 — WikiText normalization policy

**Decision**  
How to normalize WikiText-103 artifacts before tokenizer training and evaluation.

**Selected choice**  
Use one explicit, unit-tested `normalize_wikitext_text` function for train, validation, and later test. It restores `@-@`, `@,@`, and `@.@`; repairs common punctuation and bracket spacing; joins common apostrophe suffixes with Unicode-aware `\w+` matching; repairs currency and clock-time spacing; and pairs spaced double quotation marks deterministically within each row. It preserves case, Unicode, row order, headings, and article structure.

Spaced single quotation marks and bare plural possessives remain documented residue. Their surface forms are contextually ambiguous—for example, `Nameless ' unit` versus `players ' hopes`—and a context-free regex can silently corrupt one while repairing the other.

**Rationale**  
The tokenizer should learn natural punctuation instead of WikiText placeholders, and validation perplexity must be measured on the same transformation used for training. Conservatively preserving ambiguous single-apostrophe forms is preferable to changing their meaning.

**Alternatives considered**
- keep WikiText-103-raw-v1 unchanged
- restore placeholders only
- apply a broad single-quote/plural-possessive regex
- use different cleanup for each split

**Presentation relevance**  
Makes the preprocessing policy auditable and explains both the repaired artifacts and the intentionally retained residue.

**Status / evidence**  
Locked. Thirteen unit-test strings cover placeholders, brackets, contractions, Unicode apostrophes, currency, times, double quotes, ambiguous single quotes, plural possessives, and unchanged abbreviations.

---

## D-052 — Tokenizer training corpus

**Decision**  
What text the byte-level BPE tokenizer sees during training.

**Selected choice**  
Train the 16,384-token byte-level BPE tokenizer on the entire normalized official WikiText-103 training split. Do not use validation or test text. The language models still train on the fixed 20,000,000-token sampled subset.

**Rationale**  
The tokenizer must exist before exact tokenizer-produced article counts can be computed, so training it on the 20M subset would be circular. Full-train exposure also improves vocabulary coverage without leaking validation or test data.

**Alternatives considered**
- train the tokenizer on the eventual 20M model-training subset
- include validation text
- use a pretrained tokenizer

**Presentation relevance**  
Separates tokenizer vocabulary learning from the controlled model-training compute budget.

**Status / evidence**  
Locked; implementation is the next notebook chunk.

---

## D-053 — Document-boundary token accounting

**Decision**  
Whether article-boundary special tokens count toward the exact 20M-token corpus budget.

**Selected choice**  
Append one document-boundary token after each complete article and count it inside the exact 20,000,000-token budget. If the final selected article is truncated before its boundary token, that boundary token is not included.

**Rationale**  
“Exactly 20M tokenizer-produced tokens” then describes the actual sequence consumed by every model, including structural tokens.

**Alternatives considered**
- count 20M lexical tokens and add boundary tokens outside the budget
- exclude all special tokens from corpus accounting

**Presentation relevance**  
Makes the training-token claim precise and reproducible.

**Status / evidence**  
Locked; the corpus manifest will record boundary-token inclusion per article.

---

## D-054 — Article-level sampling and reproducibility manifest

**Decision**  
How to select the fixed 20M-token model-training corpus.

**Selected choice**  
Reconstruct WikiText articles from level-1 heading rows matching `= Title =`. Immediately before sampling, create `np.random.default_rng(42)`, generate one deterministic permutation of training article indices, encode whole articles, and append a boundary token. Add article sequences in that order until the running count crosses 20M, then truncate only the final selected sequence to exactly 20,000,000 tokens.

Save a manifest with the dataset fingerprint, seed, ordered article IDs, full and included token counts, boundary-token inclusion, final truncation length, tokenizer checksum, and total token count.

**Rationale**  
Article-level sampling preserves local coherence, avoids reliance on mutable global RNG state, and makes the exact corpus independently reconstructable.

**Alternatives considered**
- shuffle dataset rows or paragraphs
- sample individual tokens
- rely on RNG state initialized at notebook start
- omit the ordered selection manifest

**Presentation relevance**  
Provides a clear provenance story for the controlled training corpus.

**Status / evidence**  
Locked as a specification; article reconstruction has synthetic tests, and corpus construction follows tokenizer training.

---

# Current project state

**Notebook 01, through the preprocessing and sampling-policy checkpoint, is published on `main` via PR #1.**

The data audit, normalization policy, article reconstruction, and exact sampling specification are published on `main`. Tokenizer training and 20M-token corpus construction have not started.

## Immediate next step

Begin the next reviewed notebook chunk:
1. choose and document the document-boundary special token
2. train the 16,384-token byte-level BPE tokenizer on the full normalized training split
3. validate tokenizer behavior and save its checksum
4. construct the fixed 20M-token article-sampled corpus and manifest
5. preserve official validation/test splits for their intended roles

## Rule for future work
Continue in small, coherent chunks. Each major technical decision should be:
- understood
- implemented
- measured where possible
- recorded here
- mapped to presentation value
