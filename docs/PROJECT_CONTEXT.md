# Building a Foundation Model from Scratch — Project Context

## Purpose
This is the concise source of truth for resuming the project without reconstructing prior chats.

GitHub repository: `traderjohnd/foundation-model-from-scratch`

## Objective
> **Build a series of small decoder-only Transformer language models from scratch and systematically scale them to observe how model capacity affects learning, compute cost, and generated language.**

Experimental question:
> **How does increasing Transformer capacity affect language-model performance when the dataset and training methodology are controlled, and what tradeoffs emerge between capability and computational efficiency?**

Higher-level question:
> **At what point does increasing model capacity produce diminishing returns when training data and compute are constrained?**

This project covers pretraining from scratch. Fine-tuning an existing stronger open-weight model is a separate later project.

## Canonical notebook sequence
- **Notebook 01 — Data Preparation & Corpus Audit** — complete
- **Notebook 02 — Tokenizer Training & Corpus Construction** — complete
- **Notebook 03 — Model Architecture** — complete
- **Notebook 04 — Training Pipeline** — complete
- **Notebook 05 — Evaluation & Scaling** — complete
- **Notebook 06A — Model C Extended-Training Probe** — next exploratory notebook; separate from the frozen A/B/C scaling comparison
- **Notebook 06B** — reserved only if a separate evaluation/scope notebook is later created for the 06A extension

## Canonical decision/documentation system
The project uses an append-only, globally numbered decision record:

- `docs/PROJECT_CONTEXT.md` — concise current-state/resume document
- `docs/DECISION_INDEX.md` — project-wide chronological decision index
- `docs/decisions/00_project_definition.md`
- `docs/decisions/01_data_preparation_and_corpus_audit.md`
- `docs/decisions/02_tokenizer_training_and_corpus_construction.md`
- `docs/decisions/03_model_architecture.md`
- `docs/decisions/04_training_pipeline.md`
- `docs/decisions/05_evaluation_and_scaling.md`

Decision IDs are globally unique and never restart at notebook boundaries. Historical decisions are not rewritten to make them look current; later decisions refine/correct/supersede earlier decisions with a new ID and explicit link. The project index intentionally has no mutable “current status” column. The **why/rationale** is preserved because it is required for the final report/presentation.

Notebook 05 decisions run through **D-094**. The next globally unique decision is **D-095**, which will begin Notebook 06A.

## Data contract
Dataset: `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`.

Immutable Hub revision: `b08601e04326c79dfdd32d625aee71d232d685c3`

Official rows:
- train: 1,801,350
- validation: 3,760
- test: 4,358

Verified reconstructed articles:
- train: 28,472
- validation: 60

Training corpus:
- exactly **20,000,000 tokenizer-produced tokens**
- official train split only
- identical corpus for Models A/B/C
- 19,995,397 ordinary text tokens
- 4,603 `<|endoftext|>` boundaries
- 4,604 selected article records
- final selected article truncated after 1,312 of 4,410 text tokens; boundary omitted

Canonical hashes:
- tokenizer SHA-256: `6ec601a267cec7c843df47927f53c4dd108c85a1d059318aeec4442c7274604f`
- article permutation SHA-256: `d4e368c0c22c1ea044133f7648466201450e66dc170da8ba67235fc1cd3b836c`
- corpus token-stream SHA-256: `4101d5b18c38558a58110f54a161763186ab5318111366486ebbfa0a3fe584fa`
- corpus manifest SHA-256: `4a00196b39311a6c2e2790780e8fc43316f24a014d3d3649028b10a671f8d3fe`

## Tokenizer contract
- byte-level BPE trained from scratch on the full normalized official training split only
- `tokenizers==0.23.1`
- vocabulary: 16,384
- learned merges: 16,127
- sole special token: `<|endoftext|>` at ID 0
- no PAD / BOS / UNK
- ByteLevel pre-tokenizer: `add_prefix_space=False`, `use_regex=True`

Validation tokenizer count without boundaries: 256,579 tokens.

## Architecture contract
Canonical reusable module: `src/model.py`.

| Setting | Model A | Model B | Model C |
|---|---:|---:|---:|
| Layers | 4 | 6 | 8 |
| `d_model` | 256 | 384 | 512 |
| Heads | 4 | 6 | 8 |
| Head dimension | 64 | 64 | 64 |
| SwiGLU `d_ff` | 704 | 1,024 | 1,360 |
| Exact parameters | **7,407,872** | **16,913,280** | **33,497,600** |

Architecture: learned token embeddings, causal MHA, RoPE on Q/K, RMSNorm, SwiGLU, pre-norm residual blocks, final RMSNorm, tied embedding/LM-head weights, bias-free projections, dropout 0.10, context 512.

Initialization: `Normal(0,0.02)` base weights with depth-aware residual scaling `0.02/sqrt(2L)`; seed 42.

## Notebook 04 — frozen training protocol
Causal packing:
- context 512, stride 512
- complete fixed-length examples only
- no padding, wraparound, duplicate tail, or variable final example
- 39,062 causal examples/epoch
- 19,999,744 scored targets/epoch
- 1,221 optimizer updates/epoch
- 3 epochs = 3,663 updates/model
- 59,999,232 scored target exposures/model

Logical effective batch:
- 16,384 targets/update = 32 full 512-token sequences
- final update each epoch flushes 22 sequences = 11,264 targets

Physical micro-batch on controlled Tesla T4:
- A: 32 sequences
- B: 32 sequences
- C: 32 sequences

Optimizer/numerics:
- AdamW; betas `(0.9,0.95)`; epsilon `1e-8`
- weight decay 0.10 for `ndim>=2`; no decay for RMSNorm scales
- global grad clip 1.0
- Tesla T4: FP16 autocast + GradScaler; FP32 master parameters

LR schedule:
- peak LR **`2e-3`**, selected by controlled Model A LR probe
- 5% warmup = 183 updates
- cosine decay for 3,480 updates
- minimum LR = 10% of peak = **`2e-4`** at update 3,663
- scheduler clock = optimizer updates

Validation:
- official validation split only
- 60 reconstructed articles
- 256,639-token validation stream
- 501 causal examples
- 256,512 scored targets
- deterministic, no shuffle
- validation every 200 optimizer updates plus epoch end

## Notebook 04 — completed production results
All three models were trained under the same controlled protocol on a Tesla T4.

| Metric | Model A | Model B | Model C |
|---|---:|---:|---:|
| Parameters | 7,407,872 | 16,913,280 | 33,497,600 |
| Updates | 3,663 | 3,663 | 3,663 |
| Epochs | 3 | 3 | 3 |
| Target exposures | 59,999,232 | 59,999,232 | 59,999,232 |
| Best validation loss | **3.972054** | **3.776427** | **3.684501** |
| Best validation perplexity | **53.09** | **43.66** | **39.83** |
| Best validation update | 3,663 | 3,663 | 3,663 |
| Wall time | 11.78 min | 23.55 min | 42.22 min |
| Peak GPU memory | 4.94 GiB | 7.29 GiB | 10.62 GiB |

Production checkpoints and histories for A/B/C were persisted in Google Drive and verified. Large `.pt` checkpoints are intentionally not committed to GitHub.

## Notebook 05 — completed evaluation and scaling results
Notebook 05 executed to completion with no cell errors and emitted `Notebook 05 status: COMPLETE`.

### Predictive quality
Validation:
- A: loss **3.972054**, PPL **53.09**
- B: loss **3.776427**, PPL **43.66**
- C: loss **3.684501**, PPL **39.83**

Final untouched test over **293,376 scored targets/model**:
- A: loss **3.955290**, PPL **52.210814**
- B: loss **3.772079**, PPL **43.470355**
- C: loss **3.680554**, PPL **39.668379**

The monotonic A→B→C predictive-quality ranking survived intact on the official test split. Validation-to-test gaps were small and slightly negative for all three models.

### Compute/resource scaling
- wall time: **11.78 → 23.55 → 42.22 min**
- peak GPU memory: **4.94 → 7.29 → 10.62 GiB**
- effective end-to-end target exposures/sec: approximately **84,889 → 42,462 → 23,685**
- A→C: parameters **4.52×**, wall time **3.58×**, peak memory **2.15×**, throughput reduction about **72%**

### Diminishing returns
The first clear diminishing-return signal appears in **B→C**. Relative to A→B, the B→C step retained only about:
- loss/perplexity efficiency per added parameter: **26.9% / 23.3%**
- per added training minute: **29.6% / 25.6%**
- per added GiB peak GPU memory: **33.2% / 28.7%**

Model C remained best in absolute predictive quality; the result is declining marginal efficiency, not negative return.

### Qualitative generation
Three validation-derived prompts and three final test-derived prompts were generated for each A/B/C checkpoint with fixed temperature 0.8, top-p 0.9, 96 new tokens, and fixed per-prompt seeds. The samples showed recognizable WikiText-like prose but **did not** produce a stable monotonic A→B→C human-visible ranking. Repetition, invented entities, topic drift, and long-range incoherence remained common. This complements rather than contradicts the aggregate perplexity result.

### Key limitations
- three model sizes only
- one seed/model
- fixed 20M-token corpus
- fixed three-epoch training budget
- all three models were still best at the final update, so convergence/saturation was not established
- one Tesla T4 production environment
- six qualitative prompts total
- factual accuracy not a primary metric
- no mandatory FLOPs/energy accounting
- no defensible simple dollar-cost estimate from the Colab execution model

Canonical Notebook 05 decisions are **D-084 through D-094** in `docs/decisions/05_evaluation_and_scaling.md`.

## Immediate next step — Notebook 06A
Start a **new chat/context window** and begin **Notebook 06A — Model C Extended-Training Probe**.

This is a **separate exploratory experiment**, not an extension of the controlled A/B/C comparison. The frozen Model C result for the original experiment remains epoch 3 / update 3,663 / validation loss 3.684501 / PPL 39.83.

Notebook 06A should investigate whether Model C was training-duration constrained by continuing from its exact epoch-3 checkpoint and observing validation saturation/overfitting.

Initial design direction to formalize under **D-095** before execution:
1. resume the exact Model C update-3,663 checkpoint, including optimizer/scaler/RNG state where appropriate
2. preserve the same 20M-token training corpus, tokenizer, architecture, packing, batch semantics, validation split/procedure, dropout, optimizer parameter groups, and seed conventions
3. explicitly decide the extension learning-rate policy before training; the original cosine schedule ended at **`2e-4`**, and a constant terminal LR of `2e-4` is the leading candidate because the question is whether useful learning remains, not a retuning search
4. validate every 200 optimizer updates plus epoch end
5. record both training loss and validation loss across the extension
6. use validation-loss early stopping with **patience**, not “stop on first increase”; leading design candidate is roughly one epoch of patience (about six validation observations) with a small `min_delta` such as 0.001
7. retain every new best checkpoint and stop when patience is exhausted or a generous safety ceiling is reached (for example up to 10 additional epochs)
8. if validation loss continues improving through the ceiling, conclude that no saturation was observed within the tested range rather than forcing an overfitting result
9. do not substitute any extended-training checkpoint/result into the frozen Notebook 05 A/B/C scaling comparison
10. if a separate evaluation/scope notebook is later warranted for this extension, designate it **Notebook 06B**

The next new decision ID is **D-095**.

## Implementation philosophy
Use explicit PyTorch model and training code. Do not use a pretrained model or Hugging Face `Trainer` for the main implementation.

> **From scratch does not mean without libraries; it means the learned model, tokenizer, architecture, and training process are not inherited from a pretrained model.**

Systems-level thesis:
> **Autoregressive next-token prediction is the base training/generation mechanism, but it is not a complete description of modern AI-system behavior.**

Potential stack:
**Data → Architecture → Pretraining → Post-training → Context → Tools → Guardrails → Governance → Output**
