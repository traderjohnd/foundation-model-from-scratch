# Building a Foundation Model from Scratch — Project Context

## Purpose of this document
This is the concise source of truth for the project. It is designed so a new chat within the ChatGPT Project can resume work without reconstructing the full conversation history.

## Project title
**Building a Foundation Model from Scratch**

Suggested GitHub repository name: `foundation-model-from-scratch`

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
- 20M-token training budget
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
- Training budget: **20 million tokenizer-produced tokens**
- The same controlled 20M-token training corpus is used for all three models.
- Use WikiText-103's official split structure:
  - training corpus drawn only from the official training split
  - official validation split used for development, tuning, and checkpoint selection
  - official test split remains untouched until final evaluation
- Do **not** force an arbitrary 80/10/10 split.

## Tokenizer
- Train tokenizer **from scratch**
- Type: **Byte-level BPE**
- Vocabulary size: **16,384**
- Same tokenizer for all models

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

Detailed teaching of context length is deferred until the training-objective/loss discussion.

Presentation should preserve the distinction:
- context length does **not** change the next-token objective
- context length changes how much prior information is available to each prediction
- it affects memory, compute, long-range dependency learning, and observed training behavior

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
- Equivalent to up to ~60M token exposures for the 20M-token training corpus
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
Current planned structure:

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
- **Notebook 02 — Tokenizer Training & Corpus Construction:** verified articles → tokenizer → exact 20M-token corpus
- **Notebook 03 — Model Architecture:** tokenizer/corpus contract → decoder-only Transformer implementation
- **Notebook 04 — Training Pipeline:** explicit PyTorch training/validation/checkpoint loop
- **Notebook 05 — Evaluation & Scaling:** controlled model comparison and final evidence

`src/data.py` is the canonical reusable implementation of the locked dataset loading, normalization, and article-reconstruction pipeline. Notebook 01 remains the completed audit artifact; Notebook 02 independently imports the shared implementation rather than relying on Notebook 01 kernel state.

## Documentation workflow
Maintain two canonical documents:
1. **PROJECT_CONTEXT.md** — concise project state and handoff context
2. **DECISION_REGISTER.md** — detailed project decision appendix

Update them after major phases:
- planning complete
- tokenizer complete
- architecture complete
- training complete
- evaluation complete

## Current status
**Notebook 01 data preparation and corpus audit are complete and verified. Notebook 02 Chunk 1 independently reproduces that audited corpus through the shared `src/data.py` implementation.**

Locked/reproduced evidence:
1. dataset: `Salesforce/wikitext`, `wikitext-103-raw-v1`
2. immutable Hub revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
3. official rows: 1,801,350 train; 3,760 validation; 4,358 test
4. structure-aware normalization: 17 tests and zero heading-level changes in the Notebook 01 audit
5. raw blank-padded reconstruction: 28,472 training documents and 60 validation documents
6. hard article-count assertions tied to the pinned revision
7. guarded `_fingerprint` values retained only as local diagnostics
8. shared preprocessing source pinned in Notebook 02 to commit `7d300f14c812d9a1caf36aa9ec0568bee5b0f275`
9. fresh Notebook 02 execution passed 17/17 normalization regression tests and independently reproduced 28,472 train / 60 validation articles

Resolved audit findings:
- ordinary prose normalization damaged 11 training title frames and one validation title frame
- shape-only heading matching introduced 969 false training boundaries
- all five one-sided candidates were references, table fragments, or citation metadata
- the original 28,475 summary differs by three from the reproducible 28,472-document released corpus
- no heuristic boundary exceptions are used

The test split remains uninspected. Tokenizer training and 20M-token corpus construction have not started.

## Next implementation step
Begin the next reviewable Notebook 02 chunk:
1. lock D-055: choose and document the document-boundary/EOS special token
2. verify literal special-token collision behavior and count literal `<unk>` occurrences in development-visible articles
3. configure the 16,384-token byte-level BPE tokenizer
4. train it on the full normalized pinned training split only
5. validate encode/decode behavior and special-token handling
6. record tokenizer files and checksum
7. then construct the exact 20M-token sampled corpus and manifest

Do not jump ahead. Continue in small, coherent, presentation-ready chunks.
