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
- **Notebook 04 — Training Pipeline** — active; pipeline and LR selection complete, production accelerator preflight next
- **Notebook 05 — Evaluation & Scaling** — later

## Canonical documentation
- `docs/PROJECT_CONTEXT.md` — concise current state
- `docs/DECISION_REGISTER.md` — original detailed decision register through the pre-Notebook-04 transition
- `docs/NOTEBOOK04_DECISIONS.md` — Notebook 04 decision continuation, D-058 onward

The two decision files together form the current detailed project record until they are consolidated after the training phase.

## Data contract
Dataset: `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`.

Immutable Hub revision:
`b08601e04326c79dfdd32d625aee71d232d685c3`

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
- 4,603 explicit `<|endoftext|>` boundaries
- 4,604 selected article records
- final selected article truncated after 1,312 of 4,410 text tokens; its boundary omitted

Canonical hashes:
- tokenizer SHA-256: `6ec601a267cec7c843df47927f53c4dd108c85a1d059318aeec4442c7274604f`
- article permutation SHA-256: `d4e368c0c22c1ea044133f7648466201450e66dc170da8ba67235fc1cd3b836c`
- corpus token-stream SHA-256: `4101d5b18c38558a58110f54a161763186ab5318111366486ebbfa0a3fe584fa`
- corpus manifest SHA-256: `4a00196b39311a6c2e2790780e8fc43316f24a014d3d3649028b10a671f8d3fe`

Canonical corpus artifacts:
- `results/corpus/corpus_summary.json`
- `results/corpus/corpus_manifest.jsonl`

The 40 MB little-endian `uint16` raw token binary is intentionally not committed because the committed provenance artifacts reproduce it exactly.

## Tokenizer contract
- byte-level BPE trained from scratch on the full normalized official training split only
- `tokenizers==0.23.1`
- vocabulary: exactly 16,384 total tokens
- `min_frequency=2`
- ByteLevel pre-tokenizer: `add_prefix_space=False`, `use_regex=True`
- ByteLevel decoder
- all 256 byte symbols present
- learned merges: 16,127
- sole registered special token: `<|endoftext|>` at ID 0
- no PAD / BOS / UNK
- ordinary encoding does not auto-insert the boundary token

Validation tokenizer count without appended boundaries: 256,579 tokens.

## Architecture contract
Canonical reusable module: `src/model.py`.

| Setting | Model A | Model B | Model C |
|---|---:|---:|---:|
| Layers | 4 | 6 | 8 |
| `d_model` | 256 | 384 | 512 |
| Attention heads | 4 | 6 | 8 |
| Head dimension | 64 | 64 | 64 |
| SwiGLU `d_ff` | 704 | 1,024 | 1,360 |
| Exact parameters | **7,407,872** | **16,913,280** | **33,497,600** |

Implemented architecture:
- learned token embeddings
- causal standard multi-head self-attention
- RoPE on Q/K only
- RMSNorm
- SwiGLU
- pre-norm residual blocks
- final RMSNorm
- tied token-embedding / LM-head weights
- bias-free linear projections
- dropout 0.10
- context length 512
- no learned positional embedding table

Initialization:
- base embedding/linear weights: `Normal(0, 0.02)`
- RMSNorm scales: 1.0
- attention-output and SwiGLU-down residual projections: `0.02 / sqrt(2L)`
- seed: 42

Notebook 03 causality, initialization, parameter-count, tying, forward, and backward audits all passed.

## Notebook 04 — locked training-data behavior
Causal packing:
- context length 512
- stride 512
- `x=tokens[start:start+512]`
- `y=tokens[start+1:start+513]`
- complete fixed-length examples only
- no padding, wraparound, duplicate tail, or variable final example

Exact training arithmetic:
- 39,062 causal examples/epoch
- 19,999,744 scored targets/epoch
- 255 final target positions intentionally omitted by the fixed-length packing rule

Logical effective batch:
- 16,384 prediction targets/update
- 32 full 512-token sequences/update
- physical micro-batch is hardware-dependent
- final epoch update flushes the remaining 22 sequences = 11,264 targets
- 1,221 optimizer updates/epoch

## Notebook 04 — optimizer and numerical policy
Optimizer:
- AdamW
- betas `(0.9, 0.95)`
- epsilon `1e-8`
- weight decay 0.10 for `ndim >= 2`
- no decay for RMSNorm scale parameters
- tied parameters deduplicated safely

Gradient clipping:
- global norm 1.0

Precision:
- CUDA + native BF16 → BF16 autocast, no GradScaler
- CUDA without native BF16 → FP16 autocast + GradScaler
- CPU → FP32 smoke fallback only
- model/master parameters remain FP32

Tesla T4 evidence:
- compute capability 7.5
- native BF16: false
- resolved runtime: FP16 + GradScaler
- Model A physical micro-batch 32 sequences
- Model A real optimizer smoke peak allocation approximately 4.94 GiB

## Learning-rate schedule
Production schedule:
- maximum epochs: 3
- scheduled optimizer updates: 3,663
- warmup fraction: 5%
- warmup updates: 183
- cosine-decay updates: 3,480
- final/minimum LR: 10% of peak
- scheduler clock: optimizer updates, never micro-batches

## Learning-rate experiment — completed
Original Model A candidates:
- `1e-4`
- `3e-4`
- `1e-3`

Because `1e-3` won at the upper boundary, the probe was refined with:
- `2e-3`
- `3e-3`

All runs used the same initialization, data order, RNG restart, optimizer settings, precision policy, logical batch, clipping, 400 optimizer updates, and validation at updates 200 and 400.

Observed results:

| Peak LR | Val loss @200 | Val loss @400 | PPL @400 | Clip fraction |
|---:|---:|---:|---:|---:|
| `1e-4` | 7.067461 | 6.799811 | 897.68 | 28.7% |
| `3e-4` | 6.394534 | 6.163024 | 474.86 | 10.5% |
| `1e-3` | 5.988447 | 5.659233 | 286.93 | 6.5% |
| `2e-3` | 5.884712 | **5.488027** | **241.78** | 5.0% |
| `3e-3` | 6.159744 | 5.741478 | 311.52 | 4.5% |

**Locked production peak LR: `2e-3`.**

Rationale: `2e-3` produced the lowest validation loss at update 400 and lies below the highest tested value (`3e-3`), so the useful region is bracketed on the upper side.

Official test content was not used for LR selection.

## Validation contract
- official validation split only
- same pinned preprocessing and tokenizer
- original article order
- one boundary after every complete validation article
- validation articles: 60
- validation stream: 256,639 tokens
- causal examples: 501
- scored validation targets: 256,512
- no shuffle
- full summed cross-entropy divided by exact target count

Official test content remains reserved for final evaluation.

## Training duration and checkpoint policy
- maximum 3 epochs/model
- validation every 200 optimizer updates plus end-of-epoch
- retain the best validation checkpoint
- identical corpus, objective, optimizer family, LR schedule, logical effective batch, and seed across Models A/B/C

Resource metrics to capture:
- GPU type
- peak GPU memory
- wall-clock time
- GPU-hours
- tokens/sec
- optimizer steps/sec
- total token exposures
- estimated cost

## Current verified Notebook 04 checkpoints
- D-058 causal packing: PASS
- D-059 final partial effective batch: PASS
- D-060 micro-batch gradient equivalence: PASS
- D-061 native-BF16 / FP16 policy: PASS
- D-062 AdamW grouping: PASS
- D-063 scheduler: PASS
- D-064 LR-probe protocol: PASS
- D-065 validation pipeline: PASS
- D-066 single-run training engine: PASS
- D-067 `src/model.py` integration: PASS
- D-068 canonical corpus reconstruction/checksum gate: PASS
- D-069 real Model A optimizer smoke: PASS
- D-070 initial LR probe: PASS
- D-071 boundary-refined LR selection: PASS; production peak LR locked at `2e-3`

## Immediate next step
Run **D-072 — Production accelerator micro-batch preflight** for Models A/B/C on the active T4 GPU.

The logical effective batch must remain fixed at 32 sequences / 16,384 targets. D-072 only determines the largest physical micro-batch that fits each model. Do not launch the three full production runs until the D-072 result is reviewed.

After D-072, implement the production-run wrapper with:
- peak LR `2e-3`
- maximum 3 epochs
- 3,663 scheduled updates
- validation every 200 updates plus epoch end
- best-validation checkpoint retention
- per-model measured physical micro-batch
- throughput/memory/time accounting

## Implementation philosophy
Use explicit PyTorch model and training code. Do not use a pretrained model or Hugging Face `Trainer` for the main implementation.

Key statement:
> **From scratch does not mean without libraries; it means the learned model, tokenizer, architecture, and training process are not inherited from a pretrained model.**

## Presentation framing to preserve
Compare:
- 2017 Transformer encoder-decoder architecture
- BERT / encoder-centric NLP and WordPiece
- early GPT-style decoder-only conventions
- modern decoder-only architecture

Teach:
- RoPE vs sinusoidal/learned positional approaches
- RMSNorm vs LayerNorm
- SwiGLU vs ReLU/GELU
- pre-norm residual blocks
- weight tying and parameter efficiency
- vocabulary size vs embedding cost / sequence compression
- learning-rate probing as empirical optimization rather than copied defaults

Systems-level thesis:
> **Autoregressive next-token prediction is the base training/generation mechanism, but it is not a complete description of modern AI-system behavior.**

Potential stack:
**Data → Architecture → Pretraining → Post-training → Context → Tools → Guardrails → Governance → Output**
