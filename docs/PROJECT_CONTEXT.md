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
- **Notebook 05 — Evaluation & Scaling** — next

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

The historical documentation files that previously mixed one master register, a Notebook 03 addendum, and a Notebook 04 continuation were migrated into this structure. Git history preserves the original files and legacy IDs. New Notebook 05 decisions begin at **D-084**.

## Canonical Notebook 04 implementation
Notebook 04 is packaged as three model-specific notebooks backed by one shared implementation so there is no duplicated training engine that can drift:

- `notebooks/04_training_pipeline.ipynb` — frozen training contract + Model A production runner (**D-081**)
- `notebooks/04_training_pipeline_model_b.ipynb` — standalone Model B production runner (**D-082**)
- `notebooks/04_training_pipeline_model_c.ipynb` — standalone Model C production runner (**D-083**)
- `src/training_pipeline.py` — canonical reusable training/corpus/checkpoint/resume implementation used by all three notebooks

Each model notebook can start from a fresh Colab T4 runtime. It clones/pulls the repository, mounts the persistent Google Drive production directory, imports `src/training_pipeline.py`, and runs/resumes/verifies only its assigned model. Completed persistent runs are verified and reused rather than retrained.

The detailed training decisions/evidence are in `docs/decisions/04_training_pipeline.md`; the compact cross-model result is in `results/training/production_scaling_summary.json`.

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
- minimum LR = 10% of peak
- scheduler clock = optimizer updates

Validation:
- official validation split only
- 60 reconstructed articles
- 256,639-token validation stream
- 501 causal examples
- 256,512 scored targets
- deterministic, no shuffle
- validation every 200 optimizer updates plus epoch end

Official test content remains untouched and reserved for Notebook 05 final evaluation.

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

Preliminary observations to test formally in Notebook 05:
- validation performance improved monotonically with capacity
- every model reached its best validation result at update 3,663
- no model showed validation deterioration within the fixed budget
- validation-loss gains shrank with scale while time and memory increased, motivating explicit diminishing-return analysis

Production checkpoints and histories for A/B/C were persisted in Google Drive and verified. Large `.pt` checkpoints are intentionally not committed to GitHub.

## Notebook 04 canonical decision closure
- D-065 deterministic causal packing through D-080 production wrapper/persistence: implemented and evidenced in the Notebook 04 decision register
- D-081 Model A production run: complete
- D-082 Model B production run: complete
- D-083 Model C production run / Notebook 04 freeze: complete

## Immediate next step — Notebook 05
Start a **new chat/context window** and begin **Notebook 05 — Evaluation & Scaling**.

Notebook 05 should:
1. ingest saved A/B/C run summaries and validation histories
2. compare validation learning curves under the controlled token budget
3. quantify parameter growth vs loss/perplexity improvement
4. quantify training-time, throughput, GPU-memory, and efficiency tradeoffs
5. analyze diminishing returns explicitly
6. run controlled qualitative generation using identical prompts/decoding across A/B/C
7. evaluate the untouched official test split only at the final evaluation stage
8. save final figures/tables/results for the presentation

Do **not** retune A/B/C in Notebook 05. The training protocol is frozen.

The next new decision ID is **D-084**.

## Implementation philosophy
Use explicit PyTorch model and training code. Do not use a pretrained model or Hugging Face `Trainer` for the main implementation.

> **From scratch does not mean without libraries; it means the learned model, tokenizer, architecture, and training process are not inherited from a pretrained model.**

Systems-level thesis:
> **Autoregressive next-token prediction is the base training/generation mechanism, but it is not a complete description of modern AI-system behavior.**

Potential stack:
**Data → Architecture → Pretraining → Post-training → Context → Tools → Guardrails → Governance → Output**
