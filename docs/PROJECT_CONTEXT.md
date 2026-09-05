# Building a Foundation Model from Scratch — Project Context

## Purpose
This is the concise source of truth for the project so a new chat can resume without reconstructing prior work.

GitHub repository: `traderjohnd/foundation-model-from-scratch`

## Objective
> **Build a series of small decoder-only Transformer language models from scratch and systematically scale them to observe how model capacity affects learning, compute cost, and generated language.**

Experimental question:
> **How does increasing Transformer capacity affect language-model performance when the dataset and training methodology are controlled, and what tradeoffs emerge between capability and computational efficiency?**

Higher-level question:
> **At what point does increasing model capacity produce diminishing returns when training data and compute are constrained?**

This project covers **pretraining from scratch**. Fine-tuning an existing stronger open-weight model is a separate later project.

## Canonical notebook sequence
- **Notebook 01 — Data Preparation & Corpus Audit** — complete
- **Notebook 02 — Tokenizer Training & Corpus Construction** — complete
- **Notebook 03 — Model Architecture** — complete
- **Notebook 04 — Training Pipeline** — next
- **Notebook 05 — Evaluation & Scaling** — later

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
- dataset Hub revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
- tokenizer SHA-256: `6ec601a267cec7c843df47927f53c4dd108c85a1d059318aeec4442c7274604f`
- article permutation SHA-256: `d4e368c0c22c1ea044133f7648466201450e66dc170da8ba67235fc1cd3b836c`
- corpus token-stream SHA-256: `4101d5b18c38558a58110f54a161763186ab5318111366486ebbfa0a3fe584fa`
- corpus manifest SHA-256: `4a00196b39311a6c2e2790780e8fc43316f24a014d3d3649028b10a671f8d3fe`

Canonical corpus artifacts:
- `results/corpus/corpus_summary.json`
- `results/corpus/corpus_manifest.jsonl`

The 40 MB little-endian `uint16` raw token binary is intentionally not committed because the committed provenance artifacts reproduce it exactly.

Official validation is development-visible. Official test remains untouched until final evaluation.

## Tokenizer contract
Tokenizer was trained from scratch on the entire normalized official training split only.

Configuration:
- byte-level BPE
- `tokenizers==0.23.1`
- vocabulary: exactly 16,384 total tokens
- `min_frequency=2`
- ByteLevel pre-tokenizer: `add_prefix_space=False`, `use_regex=True`
- ByteLevel decoder
- all 256 byte symbols present
- learned merges: 16,127

Special-token contract:
- sole registered special token: `<|endoftext|>`
- token ID: 0
- no PAD
- no BOS
- no UNK
- ordinary encoding does not auto-insert the boundary

Canonical tokenizer artifacts:
- `results/tokenizer/tokenizer.json`
- `results/tokenizer/tokenizer_metadata.json`

Tokenizer efficiency on all 60 development-visible validation articles, without appended boundaries:
- 256,579 tokens
- ~4.298 characters/token
- ~4.306 UTF-8 bytes/token

## Notebook 03 — final implemented architecture evidence
Canonical notebook:
- `notebooks/03_model_architecture.ipynb`

Architecture family is now **implemented and empirically validated**, not provisional.

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
- RoPE applied to Q and K only
- RMSNorm
- SwiGLU
- pre-norm residual blocks
- final RMSNorm
- tied token-embedding / LM-head weights
- bias-free linear projections
- dropout = 0.10
- no learned positional embedding table
- context length = 512

Exact per-block parameter evidence:
- Model A block: 803,328
- Model B block: 1,770,240
- Model C block: 3,138,560

Attention parameters per block:
- A: 262,144
- B: 589,824
- C: 1,048,576

SwiGLU parameters per block:
- A: 540,672
- B: 1,179,648
- C: 2,088,960

Weight tying is verified as the same `Parameter` object and same underlying storage. For Model A it avoids duplicating a 4,194,304-parameter vocabulary matrix.

### RoPE validation
- zero learned parameters
- position-0 rotation is identity
- norm preservation validated numerically
- equal relative displacement produces matching rotated Q/K dot products within floating-point tolerance
- all models use head dimension 64 and context 512

### Causality validation
Causal attention masking was verified directly:
- future attention probability mass = 0
- changing a future token leaves all earlier attention outputs unchanged
- the same property holds through a full Transformer block and the complete language model

### Initialization policy — implemented and validated
Base initialization:
- embeddings and ordinary linear weights: `Normal(0, 0.02)`
- RMSNorm scales: 1.0

Residual-output projections:
`0.02 / sqrt(2L)`

Resulting targets:
- Model A: 0.007071
- Model B: 0.005774
- Model C: 0.005000

The residual-scaled matrices are:
- attention output projection
- SwiGLU down projection

Same-seed initialization with seed 42 reproduced identical checked weights. Weight tying and parameter counts remained intact after initialization.

### Final architecture audit
The executed Notebook 03 final audit passed all assertions for Models A/B/C:
- exact dimensions and layer counts
- exact parameter counts
- RoPE parameter-free
- no learned positional embedding table
- RMSNorm count/placement
- all linear biases disabled
- dropout fixed at 0.10
- tied embedding/output weights
- initialization policy
- finite forward logits
- context-length enforcement
- causal behavior
- backward/autograd flow

Synthetic next-token backward sanity check on untrained Model A:
- cross-entropy loss: **9.693123**
- finite, nonzero gradients verified through:
  - tied embedding/LM head
  - attention Q projection
  - attention output projection
  - SwiGLU gate projection
  - SwiGLU down projection
  - final RMSNorm

This backward test is an architecture/autograd sanity check, not a training result.

## Training controls already locked for Notebook 04
Objective:
- autoregressive causal language modeling / next-token prediction

Loss:
- cross-entropy
- perplexity = `exp(loss)`

Optimizer:
- AdamW

Learning-rate probe on Model A:
- `1e-4`
- `3e-4`
- `1e-3`
- provisional expected production peak ~`3e-4`

Schedule:
- warmup + cosine decay

Effective batch:
- 16,384 tokens per optimizer update
- approximately 32 × 512-token sequences
- gradient accumulation as needed to hold effective batch constant

Training duration:
- maximum 3 epochs
- retain best validation checkpoint

Validation cadence:
- every 200 optimizer steps
- plus end of each epoch

Precision:
- BF16 preferred
- FP16 fallback
- FP32 retained where numerically appropriate

Gradient clipping:
- global norm 1.0

Seed:
- 42

Resource metrics to capture:
- GPU type
- peak GPU memory
- wall-clock time
- GPU-hours
- tokens/sec
- optimizer steps/sec
- total token exposures
- estimated cost

## Open Notebook 04 decision: packing
The exact corpus length is 20,000,000 tokens and:

`20,000,000 mod 512 = 256`

Notebook 02 preserves all 20M tokens. Do not silently discard the final 256-token remainder.

Notebook 04 must explicitly resolve D-058 by defining:
- input/target shift
- fixed-length packing rule
- stride/block convention
- treatment of any final incomplete example
- exact number of model inputs and prediction targets

The 256-token arithmetic remainder alone does not imply that exactly 256 prediction targets must be dropped.

## Implementation philosophy
Use explicit PyTorch model and training code. Do not use a pretrained Transformer model or Hugging Face `Trainer` for the main implementation.

Mature infrastructure is appropriate for commodity functions:
- PyTorch
- Hugging Face Datasets
- Hugging Face Tokenizers
- Google Colab

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

Systems-level thesis:
> **Autoregressive next-token prediction is the base training/generation mechanism, but it is not a complete description of modern AI-system behavior.**

Potential stack:
**Data → Architecture → Pretraining → Post-training → Context → Tools → Guardrails → Governance → Output**

## Repository structure
```text
foundation-model-from-scratch/
├── README.md
├── docs/
│   ├── PROJECT_CONTEXT.md
│   ├── DECISION_REGISTER.md
│   └── presentation_notes.md
├── notebooks/
│   ├── 01_data_preparation_and_corpus_audit.ipynb
│   ├── 02_tokenizer_training_and_corpus_construction.ipynb
│   ├── 03_model_architecture.ipynb
│   ├── 04_training_pipeline.ipynb
│   └── 05_evaluation_and_scaling.ipynb
├── src/
├── configs/
├── results/
├── figures/
└── checkpoints/
```

## Current status
**Notebooks 01, 02, and 03 are complete and verified.**

Notebook 03 final canonical evidence:
- Model A: 7,407,872 parameters
- Model B: 16,913,280 parameters
- Model C: 33,497,600 parameters
- full architecture audit: PASS
- synthetic next-token backward/autograd audit: PASS

## Immediate next step
**Start Notebook 04 — Training Pipeline in a NEW context window.**

The new chat should begin from the canonical `docs/PROJECT_CONTEXT.md` and `docs/DECISION_REGISTER.md` in GitHub and proceed in small, reviewable chunks.

Notebook 04 begins by resolving D-058 and implementing deterministic causal input/target packing before building batching, optimizer/scheduler, mixed precision, gradient accumulation, validation, checkpointing, and the explicit PyTorch training loop.

Do not continue Notebook 04 in this context window.
