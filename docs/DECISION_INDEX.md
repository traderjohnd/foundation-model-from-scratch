# Foundation Model from Scratch — Decision Index

## Purpose

This is the project-wide, chronological index of engineering and experimental decisions.

The detailed rationale, alternatives, evidence, and presentation relevance live in one phase-specific register per notebook under `docs/decisions/`.

## Rules

1. **Decision IDs are globally unique and monotonically increasing across the project.** Notebook boundaries never restart numbering.
2. **Decision history is append-only.** Once an ID is assigned, its historical meaning is not rewritten to reflect later outcomes.
3. A later decision may **confirm, refine, correct, or supersede** an earlier decision, but it does so with a new ID and an explicit `Supersedes` link.
4. The index contains only durable historical facts. It intentionally has **no “current status” column** that would require retroactive maintenance.
5. Every substantive decision must preserve the **why**: the reasoning that made the choice defensible at the time.
6. Notebook 06A decisions begin at **D-095** and remain explicitly separate from the frozen Notebook 05 A/B/C comparison.

## Canonical phase registers

- [`decisions/00_project_definition.md`](decisions/00_project_definition.md)
- [`decisions/01_data_preparation_and_corpus_audit.md`](decisions/01_data_preparation_and_corpus_audit.md)
- [`decisions/02_tokenizer_training_and_corpus_construction.md`](decisions/02_tokenizer_training_and_corpus_construction.md)
- [`decisions/03_model_architecture.md`](decisions/03_model_architecture.md)
- [`decisions/04_training_pipeline.md`](decisions/04_training_pipeline.md)
- [`decisions/05_evaluation_and_scaling.md`](decisions/05_evaluation_and_scaling.md)
- [`decisions/06a_model_c_extended_training_probe.md`](decisions/06a_model_c_extended_training_probe.md)

## Chronological decision index

| ID | Phase | Kind | Decision | Selected choice | Why | Supersedes | Detail |
|---|---|---|---|---|---|---|---|
| D-001 | Project | Scope | Project scope | From-scratch pretraining here; fine-tuning later | Keeps pretraining and adaptation competencies distinct | — | 00 |
| D-002 | Project | Research design | Experimental objective | Controlled scaling of small decoder-only models | More meaningful than training one toy model | — | 00 |
| D-003 | Project | Research design | Experimental question | Capacity vs performance/compute; test diminishing returns | Focuses on real engineering tradeoffs | — | 00 |
| D-004 | Project | Research design | Number of models | Three sizes: ~7M, ~17M, ~34M | Creates a visible scaling ladder at practical cost | — | 00 |
| D-005 | Project | Research design | Scaling strategy | Compound depth/width scaling, 64-dim heads | Keeps models in one coherent family | — | 00 |
| D-006 | NB01 | Data | Source dataset | Pinned WikiText-103 raw revision | Realistic prose plus immutable provenance | — | 01 |
| D-007 | NB02 | Corpus | Training-token budget | Exactly 20M tokenizer-produced tokens | Affordable fixed budget that can expose diminishing returns | — | 02 |
| D-008 | NB02 | Experimental control | Hold data constant | Same 20M-token corpus for A/B/C | Isolates model-capacity effects | — | 02 |
| D-009 | NB02 | Tokenizer | Tokenizer ownership | Train tokenizer from scratch | Avoids inheriting a pretrained linguistic interface | — | 02 |
| D-010 | NB02 | Tokenizer | Tokenizer algorithm | Byte-level BPE | Efficient subwords with complete byte coverage | — | 02 |
| D-011 | NB02 | Tokenizer | Vocabulary size | 16,384 | Balances sequence compression against embedding cost | — | 02 |
| D-012 | NB03 | Architecture | Context length | 512 tokens | Practical memory/compute balance for Colab-scale training | — | 03 |
| D-013 | NB03 | Architecture | Model family | Modernized decoder-only Transformer | Teaches evolution beyond early GPT/BERT conventions | — | 03 |
| D-014 | NB03 | Architecture | Position encoding | RoPE | Modern positional method without learned position table | — | 03 |
| D-015 | NB03 | Architecture | Normalization | RMSNorm | Simpler modern decoder-only normalization choice | — | 03 |
| D-016 | NB03 | Architecture | FFN design | SwiGLU | Modern gated feed-forward block | — | 03 |
| D-017 | NB03 | Architecture | SwiGLU hidden sizes | A 704; B 1024; C 1360 | Preserves parameter economics near target model sizes | — | 03 |
| D-018 | NB03 | Architecture | Weight tying | Tie token embedding and LM-head weights | Saves a large duplicate vocabulary matrix | — | 03 |
| D-019 | NB04 | Objective | Training objective | Autoregressive next-token prediction | Standard causal-LM objective for decoder-only models | — | 04 |
| D-020 | NB04 | Objective | Loss | Cross-entropy | Directly penalizes low probability on the true next token | — | 04 |
| D-021 | NB05 | Metric design | Perplexity | Validation PPL for all; final test PPL | Interpretable transform of CE under fixed tokenization | — | 05 |
| D-022 | NB04 | Optimization | Optimizer | AdamW | Strong Transformer default with decoupled weight decay | — | 04 |
| D-023 | NB04 | Experiment design | LR experiment | Probe 1e-4, 3e-4, 1e-3; provisional ~3e-4 expectation | Choose LR empirically rather than copy a default | — | 04 |
| D-024 | NB04 | Optimization | LR schedule | Warmup then cosine decay | Stabilizes early training and refines later updates | — | 04 |
| D-025 | NB04 | Optimization | Effective batch | 16,384 prediction targets/update | Hardware-independent optimization control | — | 04 |
| D-026 | NB04 | Optimization | Gradient accumulation | Use as needed to preserve effective batch | Lets larger models use smaller physical batches if required | — | 04 |
| D-027 | NB04 | Training budget | Duration | Maximum 3 epochs | Enough learning signal without uncontrolled repeated passes | — | 04 |
| D-028 | NB04 | Checkpointing | Best-checkpoint policy | Keep best validation checkpoint | Final epoch is not automatically best | — | 04 |
| D-029 | Project | Data governance | Split strategy | Official train/validation/test roles | Preserves a genuinely untouched final test set | — | 00 |
| D-030 | NB03 | Architecture | Dropout | 0.10 | 20M unique tokens and repeated epochs create overfit risk | — | 03 |
| D-031 | NB04 | Numerics | Training precision | Prefer BF16; FP16 fallback | Improves speed/memory while preserving stable masters | — | 04 |
| D-032 | Project | Scope | Quantization depth | Brief treatment only | Separate quantization project covers it in depth | — | 00 |
| D-033 | NB04 | Validation | Validation cadence | Every 200 updates + epoch end | Enough curve resolution without excessive overhead | — | 04 |
| D-034 | NB04 | Checkpointing | Save rule | Save on new best validation loss | Selects generalizing state without test leakage | — | 04 |
| D-035 | NB04 | Reproducibility | Random seed | 42 | Controls initialization, shuffling, dropout, sampling | — | 04 |
| D-036 | Project | Experimental limits | Multi-seed testing | Not initially; optional if results suspicious | Preserves compute while acknowledging variance limitation | — | 00 |
| D-037 | NB03 | Architecture | Weight initialization | N(0,0.02) + depth-aware residual scaling | Conventional symmetry breaking with residual stability | — | 03 |
| D-038 | NB04 | Optimization | Gradient clipping | Global norm 1.0 | Protects against destabilizing updates | — | 04 |
| D-039 | NB05 | Evaluation | Qualitative generation | Fixed prompts + fixed decoding + rubric | Shows human-visible effects beyond perplexity | — | 05 |
| D-040 | NB05 | Evaluation | Factual accuracy | Not a primary metric | Small 20M-token models are not factual QA systems | — | 05 |
| D-041 | NB05 | Efficiency | Resource metrics | Memory, time, GPU-hours, throughput, exposures, cost | Compute efficiency is part of the research question | — | 05 |
| D-042 | NB05 | Efficiency | Energy/FLOPs | Optional, not mandatory | Avoids turning measurement overhead into the project | — | 05 |
| D-043 | Project | Implementation | Training implementation | Explicit PyTorch model/training loop | Maximizes learning value and mechanical transparency | — | 00 |
| D-044 | Project | Implementation | Library policy | Use PyTorch/Datasets/Tokenizers; no pretrained model/tokenizer/Trainer | “From scratch” concerns learned artifacts, not tensor primitives | — | 00 |
| D-045 | Project | Presentation | HF Trainer teaching section | Explain what Trainer would automate | Demonstrates understanding of both mechanics and abstractions | — | 00 |
| D-046 | Project | Presentation | Systems thesis | Next-token prediction is mechanism, not whole deployed system | Separates model probability mechanics from system behavior | — | 00 |
| D-047 | Project | Presentation | Historical framing | 2017 Transformer → BERT → early GPT → modern decoder-only | Connects prior learning to current architecture | — | 00 |
| D-048 | Project | Repository design | Repository/notebook structure | Five notebooks + reusable source/results/docs | Creates reproducible phase boundaries and avoids hidden state | Earlier four-notebook plan | 00 |
| D-049 | Project | Documentation | Canonical documentation | Project context + decision record | Prevents cross-chat reconstruction errors | — | 00 |
| D-050 | Project | Documentation | Update cadence | Update at major phase boundaries | Preserves rationale while fresh | — | 00 |
| D-051 | NB01 | Data | WikiText normalization | One structure-aware tested normalization function | Cleans artifacts without destroying heading structure | — | 01 |
| D-052 | NB02 | Tokenizer | Tokenizer training corpus | Full normalized official train only | Avoids circular 20M-subset selection and val/test leakage | — | 02 |
| D-053 | NB02 | Corpus | Boundary accounting | `<|endoftext|>` counts inside 20M budget | Makes token-budget claim exact for consumed sequence | — | 02 |
| D-054 | NB02 | Corpus | Article reconstruction/sampling manifest | Strict raw-heading rule + seed-42 article permutation + manifest | Makes boundaries and 20M sampling reproducible/auditable | — | 02 |
| D-055 | NB02 | Tokenizer | Special-token contract | Only `<|endoftext|>` at ID 0; no PAD/BOS/UNK | One explicit boundary signal with full byte coverage | — | 02 |
| D-056 | NB02 | Tokenizer | BPE min frequency | 2 | Requires repeated evidence before a pair earns vocabulary capacity | — | 02 |
| D-057 | NB02 | Reproducibility | Corpus persistence | Commit manifest/summary, not 40MB binary | Preserve exact reproducibility without repository bloat | — | 02 |
| D-058 | NB02→NB04 | Handoff | 512-token packing edge policy | Defer exact packing; preserve all 20M corpus tokens | Separates corpus membership from training-example construction | — | 02 |
| D-059 | NB03 | Implementation evidence | Exact model family | 7,407,872 / 16,913,280 / 33,497,600 params | Confirms target family after real implementation | D-004, D-005, D-017 refinements | 03 |
| D-060 | NB03 | Implementation evidence | Exact decoder block | RoPE/RMSNorm/SwiGLU/pre-norm/tied/bias-free/dropout .10 | Closes design choices as code-verified architecture | D-013–D-018 refinements | 03 |
| D-061 | NB03 | Verification | Weight tying evidence | Same Parameter/storage for embedding and LM head | Proves actual sharing rather than equal values | — | 03 |
| D-062 | NB03 | Verification | Initialization evidence | Explicit N(0,.02), RMS=1, residual 0.02/sqrt(2L) | Makes random initialization reproducible and depth-aware | D-037 refinement | 03 |
| D-063 | NB03 | Verification | Architecture audit | Shapes/counts/causality/ties/RoPE/dropout all asserted | Training is interpretable only if architecture is verified first | — | 03 |
| D-064 | NB03 | Verification | Backward sanity check | Synthetic next-token CE + backward | A valid forward graph does not prove trainability | — | 03 |
| D-065 | NB04 | Data pipeline | Deterministic causal packing | Stride-512 full 512/512 shifted examples; 255 targets unused | Fixed-shape, padding-free, duplication-free supervision | D-058 | 04 |
| D-066 | NB04 | Optimization | Partial final effective batch | Flush 22-sequence tail | Uses all admitted targets exactly once per epoch | — | 04 |
| D-067 | NB04 | Optimization | Micro-batch accumulation | Normalize by actual logical target count | Preserves gradients independent of physical batch boundaries | D-025, D-026 refinements | 04 |
| D-068 | NB04 | Numerics | Mixed-precision runtime policy | Native BF16 else FP16+GradScaler; T4→FP16 | Prevents BF16 emulation from being mistaken for native support | D-031 refinement | 04 |
| D-069 | NB04 | Optimization | AdamW parameter groups | ndim>=2 decay .10; norm scales no decay; betas .9/.95 | Makes optimizer behavior explicit and tie-safe | D-022 refinement | 04 |
| D-070 | NB04 | Optimization | Exact LR schedule | 183 warmup + 3480 cosine; min 0.1×peak | Converts broad schedule choice into exact update-clock behavior | D-024 refinement | 04 |
| D-071 | NB04 | Experiment | Initial LR probe protocol | 400 updates/candidate, val @200/@400 | Makes LR comparison controlled and reproducible | D-023 refinement | 04 |
| D-072 | NB04 | Validation | Validation protocol | Fixed official val reconstruction, 256,512 targets, no shuffle | Ensures model comparison uses identical held-out scoring | — | 04 |
| D-073 | NB04 | Implementation | Single-run training engine | Explicit engine with logical-update boundaries | Centralizes the frozen optimizer/validation semantics | — | 04 |
| D-074 | NB04 | Implementation | Canonical model module | `src/model.py` | Reuses verified architecture without notebook-state dependence | — | 04 |
| D-075 | NB04 | Reproducibility | Corpus reconstruction gate | Rebuild and hash-gate train/val streams | Refuses to train if canonical provenance drifts | — | 04 |
| D-076 | NB04 | Verification | Real Model A optimizer smoke | 3 updates + full validation | Verifies optimizer, scaler, grads, tying on real model/data | — | 04 |
| D-077 | NB04 | Experiment evidence | Original LR probe execution | 1e-3 wins original upper boundary | Shows original grid did not yet bracket optimum | — | 04 |
| D-078 | NB04 | Evidence-based selection | Production peak LR | 2e-3 after testing 2e-3 and 3e-3 | Lowest controlled val loss with upper side bracketed | D-023 provisional expectation, D-071 selection pending | 04 |
| D-079 | NB04 | Hardware evidence | T4 micro-batch preflight | Physical batch 32 for A/B/C | All models fit full logical batch on controlled hardware | D-026 conditional need | 04 |
| D-080 | NB04 | Checkpointing | Production wrapper/persistence | 21 validation events, persistent best/latest + JSON; v2 resumable for B/C | Protects experiment from Colab resets and preserves exact evidence | D-028, D-034 implementation | 04 |
| D-081 | NB04 | Production evidence | Model A training | 3.972054 val loss; PPL 53.09 | Establishes smallest-model controlled baseline | — | 04 |
| D-082 | NB04 | Production evidence | Model B training | 3.776427 val loss; PPL 43.66 | Measures capacity gain at same data/training budget | — | 04 |
| D-083 | NB04 | Production evidence | Model C training / freeze NB04 | 3.684501 val loss; PPL 39.83 | Completes controlled scaling runs before separate evaluation | — | 04 |
| D-084 | NB05 | Evaluation design | Evaluation scope and evidence/analysis boundary | Frozen Notebook 04 inputs; raw evidence in appendix; analysis in presentation-referenceable main sections; test sealed until final section | Preserves experimental integrity while making the notebook auditable and readable as a standalone analytical report | — | 05 |
| D-085 | NB05 | Artifact design | Evaluation output artifact contract | Stable machine-readable outputs under `results/evaluation/` and figures under `figures/evaluation/` | Keeps results reproducible and presentation-ready | — | 05 |
| D-086 | NB05 | Analysis design | Evaluation metric and figure contract | Precommit quality, scale, efficiency, and figure conventions before full-history inspection | Reduces hindsight-driven metric selection | — | 05 |
| D-087 | NB05 | Evidence audit | Validation-history ingestion and audit contract | Hard-gate complete A/B/C histories against frozen Notebook 04 protocol before comparison | Ensures all scaling claims use complete mutually comparable evidence | — | 05 |
| D-088 | NB05 | Quality analysis | Quality-scaling analysis | Compare all 21 shared validation checkpoints and endpoints for A/B/C | Isolates capacity's quality effect before combining with cost | — | 05 |
| D-089 | NB05 | Compute analysis | Compute-cost scaling analysis | Compare measured wall time/memory and derived end-to-end throughput under the identical workload | Makes the resource burden of scale explicit | — | 05 |
| D-090 | NB05 | Scaling analysis | Marginal return analysis | Evaluate A→B and B→C marginal quality per added parameters, time, and memory; do not fit a universal scaling law | Distinguishes absolute improvement from declining marginal efficiency | — | 05 |
| D-091 | NB05 | Qualitative evaluation | Controlled generation execution contract | Three fixed validation-derived prompts; temperature .8, top-p .9, 96 new tokens, fixed seeds | Provides reproducible human-visible evidence complementary to perplexity | D-039 refinement | 05 |
| D-092 | NB05 | Final evaluation | Untouched-test evaluation contract | Open official test only after all prior decisions freeze; score best A/B/C checkpoints once | Provides leakage-free final generalization evidence | — | 05 |
| D-093 | NB05 | Qualitative evaluation | Final test-prompt generation contract | Three fixed test-derived prompts after quantitative test results freeze; same decoding as D-091 | Completes the planned final qualitative probe without affecting selection | D-039 refinement | 05 |
| D-094 | NB05 | Closure | Notebook 05 final synthesis and closure | Freeze final A/B/C conclusions and keep future Model C extended training separate | Preserves the controlled experiment while enabling a clean exploratory follow-on | — | 05 |
| D-095 | NB06A | Experimental design | Model C extended-training contract | Resume exact update-3,663 state; constant 2e-4 LR; val-loss early stopping with min_delta .001 and patience 6; max 10 additional epochs; preserve full train/val history separately | Tests training-duration constraint without retuning or contaminating the frozen A/B/C comparison | — | 06A |
| D-096 | NB06A | Resume/provenance | Hard continuation gate and downstream evaluation policy | Require canonical NB05 artifacts, behavioral checkpoint identity, non-reset AdamW moments, exact data-order continuity, and precommit one-time exploratory test/fixed generation | Prevents a superficially valid but experimentally discontinuous continuation | — | 06A |
| D-097 | NB06A | Artifact recovery | Deterministic Notebook 05 artifact recovery and content-hash manifest | Prefer exact originals; if unavailable, re-execute Notebook 05 from frozen inputs, assert frozen validation/test metrics before commit, record test re-measurement, and SHA-256 every canonical artifact | Restores repository-level evidence without synthesizing or silently changing the frozen experiment | D-096 Gate 1 refinement | 06A |

## Next ID

**D-098** is the next globally unique decision ID.