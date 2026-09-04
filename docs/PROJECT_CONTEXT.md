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
- context length changes how much prior information is available for each prediction
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
- With 512-token sequences, this is equivalent to roughly 32 sequences contributing to each optimizer update.
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
Recommended structure:

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
│   ├── 01_data_and_tokenizer.ipynb
│   ├── 02_model_architecture.ipynb
│   ├── 03_training_pipeline.ipynb
│   └── 04_evaluation_and_scaling.ipynb
│
├── src/
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
**Notebook 01 preprocessing is published on `main`; the final three unpadded training articles are being diagnosed in draft PR #3.**

Verified:
1. WikiText-103 official row counts and split fingerprints
2. tokenization-artifact audit without inspecting test text
3. the old normalizer destroyed 11 training title frames and one validation title frame
4. shape-only raw detection finds 29,444 training candidates and 60 validation candidates
5. the shortest extra training segments are internal table/equation rows, not articles
6. the synthetic reconstruction uses the real `= = Internal Section = =` form

Current decisions:
- D-051 is relocked: 17 tests pass and the full run reported zero heading-level changes
- raw blank-padded detection reconstructs all 60 validation articles and 28,472 credible training articles
- D-054 remains provisional while the three unpadded training articles are isolated
- tokenizer training and 20M-token corpus construction remain paused

The tokenizer will still train on the entire normalized official training split only. The deterministic sampling and manifest design remains provisional until the article counts are proven.

## Next implementation step
Run the v7 notebook in Colab. Review the blank-neighbor summary and the likely-real unpadded candidates. Use that evidence to define one deterministic fallback, confirm exactly 28,475/60, add hard assertions, and relock D-054.

Do not jump ahead. Continue in small, coherent, presentation-ready chunks.
