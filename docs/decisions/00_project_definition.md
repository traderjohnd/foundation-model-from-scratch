# Decision Register — 00 Project Definition

This register contains project-wide decisions that are not owned by a single implementation notebook.

## Record format
Each decision preserves the question, selected choice, **why**, alternatives where material, evidence/context, and presentation relevance. Historical decisions are append-only; later changes use a new decision ID and an explicit supersession link.

---

## D-001 — Project scope
**Decision:** Separate from-scratch pretraining from later fine-tuning work.  
**Selected choice:** This project trains small decoder-only language models from random initialization. Fine-tuning is a separate later project using a stronger existing open-weight model.  
**Why:** Combining the two would blur two different competencies. The small pretraining model is useful for learning the full pipeline, while a stronger pretrained model is a better vehicle for demonstrating adaptation.  
**Alternatives considered:** pretrain and fine-tune the same small model; combine both into one portfolio artifact.  
**Presentation relevance:** clean distinction between pretraining from scratch and fine-tuning.

## D-002 — Experimental objective
**Selected choice:** Build a series of small decoder-only Transformers from scratch and systematically scale them to observe how capacity affects learning, compute cost, and generated language.  
**Why:** A controlled scaling experiment is more technically meaningful than simply training one toy model.  
**Alternatives considered:** one ~20M model; a toy loop-only demonstration.  
**Presentation relevance:** primary objective slide.

## D-003 — Experimental question
**Selected choice:** Ask how increasing Transformer capacity affects language-model performance when dataset and training methodology are controlled, and what tradeoffs emerge between capability and computational efficiency. Higher-level question: at what point do capacity increases yield diminishing returns under constrained data/compute?  
**Why:** “Are bigger models better?” is too elementary; the useful question is about controlled tradeoffs.  
**Presentation relevance:** research-question and conclusion framework.

## D-004 — Number of models and target scales
**Selected choice:** three progressively scaled models: ~7M, ~17M, ~34M parameters.  
**Why:** Gives a visible capacity progression while remaining practical on inexpensive single-GPU hardware.  
**Alternatives considered:** one model; other size ladders; scale only width or only depth.  
**Later evidence:** exact implemented counts were finalized in D-059.  
**Presentation relevance:** controlled scaling charts.

## D-005 — Scaling strategy
**Selected choice:** compound scaling of depth and width, maintaining 64 dimensions/head. Initial family: A 4×256×4 heads; B 6×384×6; C 8×512×8.  
**Why:** Produces coherent members of one model family instead of arbitrary architecture changes.  
**Alternatives considered:** depth-only, width-only, independently optimize each model.  
**Later evidence:** exact implemented family finalized in D-059.  
**Presentation relevance:** depth/width scaling and approximately quadratic width costs.

## D-029 — Data split strategy
**Selected choice:** 20M training tokens from official WikiText-103 train only; official validation for development/checkpointing; official test untouched until final evaluation.  
**Why:** Preserves benchmark-defined held-out roles and prevents final-test leakage.  
**Alternatives considered:** 80/10/10; custom repartitions.  
**Presentation relevance:** explain train/validation/test roles and why percentage rules are not universal.

## D-032 — Quantization scope
**Selected choice:** introduce quantization briefly only.  
**Why:** A separate open-weight quantization project covers it in depth; expanding it here would distract from pretraining/scaling.  
**Presentation relevance:** related efficiency technique, not a core project axis.

## D-036 — Multi-seed testing
**Selected choice:** do not initially repeat every full run across multiple seeds; rerun only if results are suspicious or unusually close.  
**Why:** Multiple seeds improve statistical rigor but multiply compute cost.  
**Presentation relevance:** explicit experimental limitation.

## D-043 — Implementation approach
**Selected choice:** explicit PyTorch architecture and training loop rather than Hugging Face Trainer.  
**Why:** The project is a relearning exercise; implementing mechanics directly provides substantially more educational and evidentiary value.  
**Alternatives considered:** HF Trainer; high-level AutoModel configuration.  
**Presentation relevance:** major learning-value decision.

## D-044 — Appropriate library use
**Selected choice:** use mature commodity infrastructure (PyTorch, Hugging Face Datasets/Tokenizers, Colab) but no pretrained model weights, pretrained tokenizer, or HF Trainer for the main loop.  
**Why:** “From scratch” should mean the learned artifacts and training process are not inherited—not that tensor libraries must be reimplemented.  
**Presentation relevance:** preserve the statement: *From scratch does not mean without libraries; it means the learned model, tokenizer, architecture, and training process are not inherited from a pretrained model.*

## D-045 — Hugging Face Trainer teaching section
**Selected choice:** explain what Trainer would automate: batching, device placement, forward/backward, accumulation, optimizer/scheduler, mixed precision, clipping, evaluation, logging, checkpointing, best-model selection, resume, distributed conveniences.  
**Why:** Demonstrates understanding of both low-level mechanics and production abstractions.  
**Presentation relevance:** dedicated comparison slide/section.

## D-046 — Systems-level presentation thesis
**Selected choice:** teach that autoregressive next-token prediction is the base mechanism, but not a complete description of deployed AI-system behavior.  
**Why:** Architecture, post-training, context, tools, guardrails, governance, and output restrictions materially shape system behavior.  
**Presentation relevance:** potential stack: **Data → Architecture → Pretraining → Post-training → Context → Tools → Guardrails → Governance → Output**.

## D-047 — Historical architecture framing
**Selected choice:** explicitly connect 2017 Transformer → BERT/WordPiece/encoder-centric NLP → early GPT-style decoder-only → modern decoder-only LLMs.  
**Why:** Bridges the user's earlier NLP learning to the modern architecture implemented here.  
**Presentation relevance:** historical/technical bridge.

## D-048 — Repository and notebook structure
**Selected choice:** five notebooks with separate `docs/`, `notebooks/`, `src/`, `configs/`, `results/`, `figures/`, `checkpoints/`; reusable deterministic preprocessing in `src/data.py`.  
**Why:** The original four-notebook plan became too broad once the corpus audit was substantial. Splitting data audit from tokenizer/corpus construction creates cleaner contracts and eliminates hidden Colab-state dependencies.  
**Alternatives considered:** original four-notebook layout; duplicated preprocessing; runtime dependence between notebooks.  
**Supersedes:** original four-notebook organization.  
**Presentation relevance:** reproducibility and engineering discipline.

## D-049 — Canonical documentation
**Original selected choice:** maintain `PROJECT_CONTEXT.md` and a project decision register.  
**Why:** Cross-chat memory is useful but canonical repository files provide precision.  
**Historical note:** the original single-register implementation later became difficult to navigate. The documentation architecture was normalized after Notebook 04 into `DECISION_INDEX.md` plus one register per phase/notebook. The historical purpose of D-049 remains unchanged; the storage structure is refined by the post-NB04 documentation normalization recorded in the index rules.  
**Presentation relevance:** decision history can serve as report/presentation appendix material.

## D-050 — Documentation update cadence
**Selected choice:** update canonical documentation after major phases: planning, tokenizer/corpus, architecture, training, evaluation.  
**Why:** Preserves rationale while fresh and makes new context windows efficient.  
**Presentation relevance:** final narrative is accumulated rather than reconstructed.
