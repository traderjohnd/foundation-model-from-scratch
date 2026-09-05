# Building a Foundation Model from Scratch — Project Context

## Purpose of this document
This is the concise source of truth for the project. It is designed so a new chat within the ChatGPT Project can resume work without reconstructing the full conversation history.

## Project title
**Building a Foundation Model from Scratch**

GitHub repository: `traderjohnd/foundation-model-from-scratch`

## Project objective
> **Build a series of small decoder-only Transformer language models from scratch and systematically scale them to observe how model capacity affects learning, compute cost, and generated language.**

## Experimental question
> **How does increasing Transformer capacity affect language-model performance when the dataset and training methodology are controlled, and what tradeoffs emerge between capability and computational efficiency?**

## Higher-level question
> **At what point does increasing model capacity produce diminishing returns when training data and compute are constrained?**

## Final value statement
> **I experimentally explored the relationship between model capacity, data, compute, and language-model performance by training a controlled series of Transformers from random initialization.**

## Important project boundary
This project covers **pretraining from scratch**.

A later, separate project will cover **fine-tuning an existing, stronger open-weight pretrained model**, likely with LoRA/QLoRA or a similar technique. The small model created here is **not** intended to be the primary fine-tuning demonstration.

## Core experimental design
Three progressively larger members of the same model family will be trained under controlled conditions:

- **Model A:** ~7M parameters
- **Model B:** ~17M parameters
- **Model C:** ~34M parameters

Primary variable:
- **Model capacity**

Held constant as much as practical:
- source corpus
- exact 20M-token training corpus
- tokenizer
- 16,384-token vocabulary
- 512-token context length
- training objective
- loss function
- optimizer family
- effective batch size
- training/evaluation methodology
- generation probes
- random-seed policy
- evaluation data

The exact parameter counts will be computed after implementation.

## Data
- Source corpus: **WikiText-103**
- Hugging Face dataset: `Salesforce/wikitext`
- Configuration: `wikitext-103-raw-v1`
- Immutable Hub revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
- Official rows: 1,801,350 train; 3,760 validation; 4,358 test
- Verified reconstructed articles: 28,472 train; 60 validation
- Training budget: **exactly 20,000,000 tokenizer-produced tokens**
- The same controlled 20M-token training corpus will be used for all three models.
- The 20M corpus is drawn only from the official training split.
- Official validation remains development-visible for tuning/checkpoint selection.
- Official test remains untouched until final evaluation.
- Do **not** force an arbitrary 80/10/10 split.

### Exact 20M corpus evidence
Notebook 02 deterministically constructed the training corpus with `np.random.default_rng(42)` created immediately before the article permutation.

Observed evidence:
- available training articles: **28,472**
- selected article records: **4,604**
- included ordinary text tokens: **19,995,397**
- included `<|endoftext|>` boundary tokens: **4,603**
- total: **20,000,000 tokens exactly**
- final selected article: `train:article-02505:row-156360`
- final article full text tokens: 4,410
- final article included text tokens: 1,312
- final article boundary omitted because the budget ended before it
- canonical raw token representation: little-endian `uint16`
- raw token bytes: **40,000,000**
- article-permutation SHA-256: `d4e368c0c22c1ea044133f7648466201450e66dc170da8ba67235fc1cd3b836c`
- corpus token-stream SHA-256: `4101d5b18c38558a58110f54a161763186ab5318111366486ebbfa0a3fe584fa`
- ordered manifest SHA-256: `4a00196b39311a6c2e2790780e8fc43316f24a014d3d3649028b10a671f8d3fe`

The committed manifest was independently replayed by re-encoding the recorded articles with the persisted tokenizer. Reconstruction produced exactly 20,000,000 tokens and the identical corpus SHA-256.

The 40 MB raw token binary is intentionally not committed to Git because the pinned data/preprocessing, persisted tokenizer, seed, and ordered manifest reproduce it exactly. Canonical corpus artifacts are:
- `results/corpus/corpus_summary.json`
- `results/corpus/corpus_manifest.jsonl`

## Tokenizer
- Trained **from scratch** on the entire normalized official training split only
- Type: **Byte-level BPE**
- Hugging Face `tokenizers==0.23.1`
- Vocabulary size: **16,384 total tokens**
- `min_frequency=2`
- Same tokenizer for all models
- ByteLevel pre-tokenizer: `add_prefix_space=False`, `use_regex=True`
- ByteLevel decoder
- Initial byte alphabet: all **256** byte-mapped symbols
- Learned BPE merges: **16,127**
- `min_frequency=2` did not bind before the requested vocabulary was filled

### Special-token contract
Sole registered special token:
- `<|endoftext|>` — document-boundary / EOS marker
- token ID: **0**

No registered:
- PAD
- BOS
- UNK

Ordinary encoding does **not** automatically insert `<|endoftext|>`; corpus construction appends it explicitly after each complete article. Boundary tokens count inside the exact 20M-token budget.

Development-visible collision audits found:
- literal `<|endoftext|>` occurrences: **0** in reconstructed train/validation articles
- literal `<unk>` occurrences: **0** in reconstructed train/validation articles

Structural validation showed:
- 256/256 ByteLevel alphabet symbols present
- no tokenizer padding
- no tokenizer truncation
- no post-processor
- exact encode/decode round trips on fixed byte/Unicode probes
- explicit `<|endoftext|>` remains one atomic special token
- persisted/reloaded tokenizer preserves IDs, tokens, and decoding behavior

### Canonical tokenizer artifact
Repository artifacts:
- `results/tokenizer/tokenizer.json`
- `results/tokenizer/tokenizer_metadata.json`

Canonical `tokenizer.json` SHA-256:
`6ec601a267cec7c843df47927f53c4dd108c85a1d059318aeec4442c7274604f`

Tokenizer file size:
- **486,750 bytes**

A repeated production training run in the same pinned environment produced byte-identical compact serialization. Treat that as empirical same-environment reproducibility evidence, not a claim of universal cross-platform determinism.

### Held-out tokenizer-efficiency diagnostic
On all 60 development-visible validation articles, without appended boundaries:
- tokens: **256,579**
- characters/token: **~4.298**
- UTF-8 bytes/token: **~4.306**

This is a tokenizer-efficiency diagnostic, not language-model quality.

Presentation must compare alternatives:
- word-level
- character-level
- WordPiece (especially BERT, which the user originally learned)
- Unigram / SentencePiece
- BPE
- byte-level BPE

Presentation must explicitly explain how vocabulary size affects:
- embedding/output parameter count
- sequence length
- compute

Key tokenizer mental models:
> **Byte-level coverage and BPE compression are different things.** The 256-byte alphabet guarantees representability; BPE merges common byte sequences so they require fewer tokens.

> **Vocabulary size controls how much compression capacity we allow. `min_frequency` controls how much evidence a pattern needs before it earns some of that capacity.**

## Model architecture
Use a **modernized decoder-only Transformer**, not a classic GPT-style implementation.

Core design:
- token embeddings
- causal multi-head self-attention
- RoPE positional encoding
- RMSNorm
- SwiGLU feed-forward network
- residual connections
- final normalization
- output logits
- tied input embedding/output projection weights

Target scaling family:

| Setting | Model A | Model B | Model C |
|---|---:|---:|---:|
| Target parameters | ~7M | ~17M | ~34M |
| Layers | 4 | 6 | 8 |
| Hidden width (`d_model`) | 256 | 384 | 512 |
| Attention heads | 4 | 6 | 8 |
| Head dimension | 64 | 64 | 64 |
| Provisional SwiGLU hidden dim | ~704 | ~1,024 | ~1,360 |

The SwiGLU hidden dimensions are provisional until the exact implementation and parameter count are verified.

### Historical/presentation framing
The presentation should explicitly compare:
- **2017 "Attention Is All You Need"** encoder-decoder architecture
- BERT / encoder-centric NLP era
- early GPT-style decoder-only conventions
- modern decoder-only LLM conventions

Modern components to teach later:
- RoPE vs sinusoidal/learned positional approaches
- RMSNorm vs LayerNorm
- SwiGLU vs older ReLU/GELU feed-forward blocks
- pre-normalization and modern residual conventions

Important precision:
Encoders have **not disappeared from AI**. They remain useful for embeddings, retrieval, classification, multimodal systems, and other architectures. Decoder-only became dominant for large generative text models.

## Context length
- **512 tokens**

Presentation should preserve the distinction:
- context length does **not** change the next-token objective
- context length changes how much prior information is available to each prediction
- it affects memory, compute, long-range dependency learning, and observed training behavior

### Packing issue deliberately unresolved
The exact corpus has 20,000,000 tokens and therefore:

`20,000,000 mod 512 = 256`

Notebook 02 preserves all 20,000,000 corpus tokens. It does **not** silently discard the 256-token remainder.

Do not yet conclude that exactly 256 prediction targets must be dropped. The number of usable inputs/targets depends on the later packing convention—for example, whether fixed-length next-token examples use overlapping source positions to create 512-token inputs and shifted labels. The input/target packing rule must be made explicit during the training-data/training-pipeline phase.

## Training objective
- **Autoregressive causal language modeling**
- **Next-token prediction**
- Self-supervised: the next token in the text provides the label

Terminology to explain:
- next-token prediction
- causal language modeling
- autoregressive language modeling

## Loss
- **Cross-entropy loss**

Key teaching point:
> Cross-entropy penalizes the model when it assigns too little probability to the true next token.

Loss is averaged across prediction positions/batches as appropriate.

## Perplexity
- Track **validation perplexity for all three models**
- Use test perplexity only at final evaluation
- Perplexity = `exp(cross-entropy loss)`

Key interpretation:
- lower is better
- useful for comparison only when tokenizer and evaluation data are controlled
- not an absolute "intelligence score"

## Optimizer
- **AdamW**

Presentation should explain:
- SGD
- Adam
- AdamW
- brief awareness of alternatives such as Adafactor, Lion, Muon, etc.

Key distinction:
- AdamW weight decay regularizes weights
- dropout regularizes activations

## Learning rate
Use a brief learning-rate experiment on Model A before the full runs.

Candidate peak learning rates:
- `1e-4`
- `3e-4`
- `1e-3`

Provisional expected choice:
- **peak LR ~3e-4**

Schedule:
- warmup
- then cosine decay

Presentation should show what:
- too-low LR
- appropriate LR
- too-high LR
look like in training behavior.

## Batch strategy
- Effective batch size: **16,384 tokens per optimizer update**
- With 512-token sequences, this is equivalent to roughly 32 sequences contributing to each update.
- Microbatch size may vary by model/GPU memory.
- Use **gradient accumulation** to keep effective batch size constant.

Presentation should explain:
- microbatch vs effective batch
- gradient noise
- learning-rate interaction
- GPU memory
- training efficiency

## Epochs
- Maximum: **3 epochs**
- Equivalent to up to ~60M corpus-token exposures before any later packing-edge convention is accounted for
- Validate during training
- Do not assume the final epoch is best
- Keep the checkpoint with best validation performance

Use this to discuss:
- generalization
- overfitting
- whether larger models overfit earlier or learn faster

## Dropout
- **0.10**
- Same across all models

Reason:
- 20M unique training tokens is small relative to modern LLM pretraining
- up to 3 passes through the corpus introduces overfitting risk

Presentation nuance:
Large frontier-scale models may use little or no dropout because of enormous data scale; our small-data experiment has different regularization needs.

## Precision
Preferred:
- **BF16 mixed precision**

Fallback:
- **FP16** if BF16 is not well supported by the available Colab GPU

Framework may retain FP32 for numerically sensitive operations.

Presentation should explain:
- FP32
- FP16
- BF16
- why BF16 is attractive for Transformer training
- memory/speed/stability tradeoffs

Quantization:
- introduce briefly only
- do **not** turn this project into a quantization project
- the user has a separate open-weight quantization project

## Checkpointing and validation cadence
- Validate every **200 optimizer steps**
- Also validate at the end of each epoch
- Save checkpoint whenever validation loss reaches a new best
- Final test evaluation uses the best validation checkpoint, not automatically the last checkpoint

## Reproducibility
Primary seed:
- **42**

Use fixed seed to control:
- initialization
- dropout randomness
- data shuffling
- other stochastic operations where practical

Pinned preprocessing source used by Notebook 02:
- `7d300f14c812d9a1caf36aa9ec0568bee5b0f275`

Canonical data/tokenizer fingerprints:
- dataset Hub revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
- tokenizer SHA-256: `6ec601a267cec7c843df47927f53c4dd108c85a1d059318aeec4442c7274604f`
- article-permutation SHA-256: `d4e368c0c22c1ea044133f7648466201450e66dc170da8ba67235fc1cd3b836c`
- corpus token-stream SHA-256: `4101d5b18c38558a58110f54a161763186ab5318111366486ebbfa0a3fe584fa`
- corpus manifest SHA-256: `4a00196b39311a6c2e2790780e8fc43316f24a014d3d3649028b10a671f8d3fe`

Guarded Hugging Face `_fingerprint` values are only local cache-state diagnostics; they are not upstream dataset version identifiers.

Presentation nuance:
A fixed seed improves reproducibility but a single run does not prove statistical significance. Multi-seed reruns are optional if results are suspicious or very close.

## Weight initialization
Base policy:
- `Normal(mean=0, std=0.02)` for embeddings and linear weights

Residual paths:
- use depth-aware scaling for residual-output projections

Same initialization policy for all models.

## Gradient clipping
- Global gradient norm: **1.0**

Purpose:
- protect against occasional unusually large gradients destabilizing training

## Generated-language evaluation
Use standardized generation probes in addition to loss/perplexity.

Control across models:
- same prompts
- same number of generated tokens
- same decoding settings
- same sampling seed

Development prompts:
- fixed small set from validation data

Final prompts:
- separate fixed set from untouched test data after model selection

Provisional decoding:
- temperature: **0.8**
- top-p: **0.9**

Qualitative rubric:
1. fluency / grammar
2. local coherence
3. topical continuity
4. repetition / degeneration

Do **not** use factual accuracy as a primary metric for these small WikiText-trained models.

## Resource and efficiency measurements
Track for all three models:
- GPU type
- peak GPU memory
- wall-clock training time
- GPU-hours
- tokens processed per second
- optimizer-step throughput
- total token exposures
- estimated training cost

After training, separately measure:
- inference latency
- inference throughput

Do not make energy use or theoretical FLOPs mandatory unless they can be measured or estimated cleanly later.

## Implementation philosophy
Choose **Option B: build the model and training loop explicitly in PyTorch**.

Implement directly:
- Transformer architecture
- RMSNorm
- RoPE integration
- causal multi-head self-attention
- SwiGLU
- residual connections
- forward pass
- next-token loss setup
- training loop
- gradient accumulation
- validation loop
- checkpointing
- generation loop

Use mature infrastructure where appropriate:
- PyTorch for tensors, autograd, GPU execution, optimizer
- Hugging Face Datasets for WikiText-103 access
- Hugging Face Tokenizers for training our tokenizer from scratch
- Google Colab for GPU compute

Do **not**:
- load a pretrained Transformer model
- use Hugging Face `Trainer` for the main implementation

Key teaching statement:
> **From scratch does not mean without libraries; it means the learned model, tokenizer, architecture, and training process are not inherited from a pretrained model.**

## Hugging Face Trainer presentation section
Presentation must explicitly explain that Hugging Face Trainer can automate/coordinate:
- batching, shuffling, and data loading
- device placement
- forward/backward loop
- gradient accumulation
- optimizer integration
- learning-rate warmup/scheduling
- mixed precision
- gradient clipping
- validation/evaluation cadence
- metric logging
- checkpoint save/load
- best-model selection
- resume-from-checkpoint
- distributed/multi-GPU conveniences

Explain why this project implements these directly for learning, while production engineering may reasonably use Trainer or other training frameworks.

## Presentation thesis: "next-token prediction" is not the whole AI system
Preserve this systems-level teaching point:

> **Autoregressive next-token prediction is the base training/generation mechanism, but it is not a complete description of modern AI-system behavior.**

Behavior can be shaped by:
1. training data
2. architecture
3. pretraining objective
4. post-training / alignment
5. prompting and runtime context
6. retrieval, tools, and orchestration
7. guardrails
8. governance
9. final output/action constraints

Strong explanatory analogy:
> Saying an AI system is "just next-token prediction" is like saying a computer is "just transistors switching on and off." It is true at one level of abstraction but insufficient to explain system-level behavior.

Potential presentation spine:
**Data → Architecture → Pretraining → Post-training → Context → Tools → Guardrails → Governance → Output**

## Repository structure
Current structure / plan:

```text
foundation-model-from-scratch/
│
├── README.md
├── docs/
│   ├── PROJECT_CONTEXT.md
│   ├── DECISION_REGISTER.md
│   └── presentation_notes.md
│
├── notebooks/
│   ├── 01_data_preparation_and_corpus_audit.ipynb
│   ├── 02_tokenizer_training_and_corpus_construction.ipynb
│   ├── 03_model_architecture.ipynb
│   ├── 04_training_pipeline.ipynb
│   └── 05_evaluation_and_scaling.ipynb
│
├── src/
│   ├── data.py
│   ├── tokenizer.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
│
├── configs/
│   ├── model_7m.yaml
│   ├── model_17m.yaml
│   └── model_34m.yaml
│
├── results/
│   ├── tokenizer/
│   │   ├── tokenizer.json
│   │   └── tokenizer_metadata.json
│   ├── corpus/
│   │   ├── corpus_summary.json
│   │   └── corpus_manifest.jsonl
│   ├── metrics/
│   ├── generations/
│   └── training_logs/
│
├── figures/
│   └── presentation_charts/
│
└── checkpoints/
```

Notebook boundaries are intentional:
- **Notebook 01 — Data Preparation & Corpus Audit:** raw WikiText → verified normalized/reconstructed articles
- **Notebook 02 — Tokenizer Training & Corpus Construction:** verified articles → persisted byte-level BPE tokenizer → exact 20M-token corpus
- **Notebook 03 — Model Architecture:** tokenizer/corpus contract → decoder-only Transformer implementation
- **Notebook 04 — Training Pipeline:** explicit PyTorch data packing, training/validation/checkpoint loop
- **Notebook 05 — Evaluation & Scaling:** controlled model comparison and final evidence

`src/data.py` is the canonical reusable implementation of the locked dataset loading, normalization, and article-reconstruction pipeline. Notebook 01 remains the completed audit artifact; Notebook 02 independently imports the shared implementation rather than relying on Notebook 01 kernel state.

## Documentation workflow
Maintain two canonical documents:
1. **PROJECT_CONTEXT.md** — concise project state and handoff context
2. **DECISION_REGISTER.md** — detailed project decision appendix

Update them after major phases:
- planning complete
- tokenizer/corpus complete
- architecture complete
- training complete
- evaluation complete

## Current status
**Notebook 01 — Data Preparation & Corpus Audit and Notebook 02 — Tokenizer Training & Corpus Construction are complete and verified.**

Notebook 01 established the immutable dataset, normalization, and article-reconstruction contract. Notebook 02 independently reproduced that contract through pinned `src/data.py`, trained and persisted the project tokenizer, and deterministically constructed the exact 20M-token model-training corpus.

Completed evidence:
1. immutable WikiText-103 Hub revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
2. 17/17 normalization regression tests pass
3. verified reconstructed articles: 28,472 train / 60 validation
4. tokenizer trained on normalized official train only
5. tokenizer vocabulary: 16,384 with 256/256 byte symbols and 16,127 learned merges
6. sole special token `<|endoftext|>` has ID 0; no PAD/BOS/UNK; no automatic EOS insertion
7. canonical tokenizer SHA-256: `6ec601a267cec7c843df47927f53c4dd108c85a1d059318aeec4442c7274604f`
8. exact corpus: 20,000,000 tokens from official train only, seed 42
9. corpus composition: 19,995,397 text tokens + 4,603 explicit article-boundary tokens
10. corpus token-stream SHA-256: `4101d5b18c38558a58110f54a161763186ab5318111366486ebbfa0a3fe584fa`
11. ordered manifest SHA-256: `4a00196b39311a6c2e2790780e8fc43316f24a014d3d3649028b10a671f8d3fe`
12. manifest-driven reconstruction reproduced exactly 20,000,000 tokens and the same corpus checksum
13. raw 40 MB token binary intentionally omitted from Git because it is exactly reproducible from committed provenance artifacts
14. official test text remains uninspected/unencoded

Resolved audit findings from Notebook 01 remain part of the record:
- ordinary prose normalization damaged 11 training title frames and one validation title frame
- shape-only heading matching introduced 969 false training boundaries
- all five one-sided candidates were references, table fragments, or citation metadata
- the original 28,475 summary differs by three from the reproducible 28,472-document released corpus
- no heuristic boundary exceptions are used

Open implementation decision carried forward:
- the exact corpus leaves a remainder of 256 tokens relative to 512-token context blocks; the later input/target packing convention must explicitly determine how the full stream maps to training examples without silently changing the corpus definition

## Next implementation step
**Start Notebook 03 — Model Architecture in a new chat/context window.**

The next notebook begins from the now-frozen tokenizer/corpus contract and should implement the modernized decoder-only Transformer in small, reviewable chunks:
1. freeze exact model configuration objects and parameter-count targets
2. implement/test RMSNorm
3. implement/test RoPE
4. implement causal multi-head self-attention
5. implement/test SwiGLU
6. assemble the pre-norm decoder block and full language model
7. tie input/output embeddings
8. verify causal behavior, tensor shapes, initialization, and exact parameter counts for Models A/B/C

Do not begin the training loop in Notebook 03. Data packing and optimization remain Notebook 04 responsibilities.

Continue in small, coherent, presentation-ready chunks.