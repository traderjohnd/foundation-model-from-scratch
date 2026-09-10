# Building a Foundation Model from Scratch - Project Context

## Purpose
This is the concise source of truth for resuming the project without reconstructing prior chats.

GitHub repository: `traderjohnd/foundation-model-from-scratch`

## Current project status
The **experimental portion of the project is complete**.

Canonical notebook sequence:
- **Notebook 01 - Data Preparation & Corpus Audit** - complete
- **Notebook 02 - Tokenizer Training & Corpus Construction** - complete
- **Notebook 03 - Model Architecture** - complete
- **Notebook 04 - Training Pipeline** - complete
- **Notebook 05 - Evaluation & Scaling** - complete and frozen controlled A/B/C experiment
- **Notebook 06A - Model C Extended-Training Probe** - complete exploratory follow-on; separate from Notebook 05

No additional model training, LR search, checkpoint selection, official-test scoring, or scope expansion is required for the core project. Remaining work is presentation/report assembly from frozen evidence. Fine-tuning, quantization, additional scaling, or architecture research are separate future projects/follow-ons.

Publication and presentation work started under **D-101**. The [`publication/CLAIM_EVIDENCE_MAP.md`](publication/CLAIM_EVIDENCE_MAP.md) file links public claims to evidence, decisions, figures, and limits.

The first [`publication/EXECUTIVE_NARRATIVE.md`](publication/EXECUTIVE_NARRATIVE.md) draft now applies STE-style rules. Review of this draft is the next publication step.

The next globally unique decision ID is **D-102**.

## Objective
> **Build a series of small decoder-only Transformer language models from scratch and systematically scale them to observe how model capacity affects learning, compute cost, and generated language.**

Experimental question:
> **How does increasing Transformer capacity affect language-model performance when the dataset and training methodology are controlled, and what tradeoffs emerge between capability and computational efficiency?**

Higher-level question:
> **At what point does increasing model capacity produce diminishing returns when training data and compute are constrained?**

## Canonical documentation
- `docs/README.md` - documentation map and reading order
- `docs/PROJECT_CONTEXT.md` - current-state/resume document
- `docs/DECISION_INDEX.md` - append-only project-wide chronological index
- `docs/decisions/` - one canonical decision register per project phase/notebook
- `docs/provenance/` - audit/recovery records that matter to reproducibility but are not decision registers
- `docs/publication/CLAIM_EVIDENCE_MAP.md` - publication/presentation claim wording, pinned supporting evidence, and interpretation limits
- `docs/publication/EXECUTIVE_NARRATIVE.md` - concise STE-style narrative for adaptation into the README, presentation, article, and monograph

Notebook 06A decisions **D-095 through D-100** are consolidated in the single canonical register `docs/decisions/06a_model_c_extended_training_probe.md`. The Notebook 05 artifact-retention incident and recovery are retained separately at `docs/provenance/05_evaluation_artifact_recovery.md`.

Decision history is append-only; consolidation changes only document organization and does not rewrite the historical meaning of any decision.

## Data contract
Dataset: `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`.

Pinned Hub revision: `b08601e04326c79dfdd32d625aee71d232d685c3`

Official rows:
- train: 1,801,350
- validation: 3,760
- test: 4,358

Verified reconstructed articles:
- train: 28,472
- validation: 60
- test: 60

Training corpus:
- exactly **20,000,000 tokenizer-produced tokens**
- official train split only
- identical frozen corpus for Models A/B/C
- 19,995,397 ordinary text tokens
- 4,603 `<|endoftext|>` boundaries
- 4,604 selected article records

Canonical hashes:
- tokenizer SHA-256: `6ec601a267cec7c843df47927f53c4dd108c85a1d059318aeec4442c7274604f`
- article permutation SHA-256: `d4e368c0c22c1ea044133f7648466201450e66dc170da8ba67235fc1cd3b836c`
- corpus token-stream SHA-256: `4101d5b18c38558a58110f54a161763186ab5318111366486ebbfa0a3fe584fa`
- corpus manifest SHA-256: `4a00196b39311a6c2e2790780e8fc43316f24a014d3d3649028b10a671f8d3fe`

## Tokenizer contract
- byte-level BPE trained from scratch on normalized official train only
- `tokenizers==0.23.1`
- vocabulary: 16,384
- learned merges: 16,127
- sole special token: `<|endoftext|>` at ID 0
- no PAD / BOS / UNK

## Architecture contract
Canonical module: `src/model.py`.

| Setting | Model A | Model B | Model C |
|---|---:|---:|---:|
| Layers | 4 | 6 | 8 |
| `d_model` | 256 | 384 | 512 |
| Heads | 4 | 6 | 8 |
| Head dimension | 64 | 64 | 64 |
| SwiGLU `d_ff` | 704 | 1,024 | 1,360 |
| Exact parameters | **7,407,872** | **16,913,280** | **33,497,600** |

Architecture: learned token embeddings, causal MHA, RoPE on Q/K, RMSNorm, SwiGLU, pre-norm residual blocks, final RMSNorm, tied embedding/LM-head weights, bias-free projections, dropout 0.10, context 512.

Initialization: `Normal(0,0.02)` base weights with residual scaling `0.02/sqrt(2L)`; seed 42.

## Frozen Notebook 04 training protocol
Causal packing:
- context 512, stride 512
- fixed complete examples only; no padding/wraparound/duplicate tail
- 39,062 examples/epoch
- 19,999,744 scored targets/epoch
- 1,221 optimizer updates/epoch
- 3 epochs = 3,663 updates/model
- 59,999,232 target exposures/model

Effective batch:
- 16,384 targets/update = 32 full sequences
- final update each epoch = 22 sequences / 11,264 targets

Optimizer/numerics:
- AdamW betas `(0.9,0.95)`, epsilon `1e-8`
- weight decay 0.10 for `ndim>=2`; no decay RMSNorm scales
- global grad clip 1.0
- canonical Tesla T4 path: FP16 autocast + GradScaler

LR schedule:
- peak `2e-3`
- 183-update warmup
- 3,480-update cosine decay
- terminal/min LR `2e-4` at update 3,663

Validation:
- fixed official validation stream
- 501 causal examples / 256,512 scored targets
- deterministic, no shuffle
- every 200 updates plus epoch end

## Frozen Notebook 04 production results
| Metric | Model A | Model B | Model C |
|---|---:|---:|---:|
| Parameters | 7,407,872 | 16,913,280 | 33,497,600 |
| Updates | 3,663 | 3,663 | 3,663 |
| Target exposures | 59,999,232 | 59,999,232 | 59,999,232 |
| Best validation loss | **3.972054** | **3.776427** | **3.684501** |
| Best validation PPL | **53.09** | **43.66** | **39.83** |
| Wall time | 11.78 min | 23.55 min | 42.22 min |
| Peak memory | 4.94 GiB | 7.29 GiB | 10.62 GiB |

## Frozen Notebook 05 evaluation and scaling results
Official test over **293,376 scored targets/model**:
- A: loss **3.955290**, PPL **52.210814**
- B: loss **3.772079**, PPL **43.470355**
- C: loss **3.680554**, PPL **39.668379**

The A→B→C predictive-quality ranking remained monotonic on held-out test.

Compute/resource scaling:
- wall time: **11.78 → 23.55 → 42.22 min**
- peak memory: **4.94 → 7.29 → 10.62 GiB**
- end-to-end target exposures/sec: about **84,889 → 42,462 → 23,685**
- A→C: **4.52×** parameters, **3.58×** wall time, **2.15×** peak memory, throughput down about **72%**

Diminishing returns first became clear in B→C. Relative to A→B, B→C retained roughly:
- loss/PPL efficiency per added parameter: **26.9% / 23.3%**
- per added minute: **29.6% / 25.6%**
- per added GiB: **33.2% / 28.7%**

Model C remained best in absolute likelihood. Diminishing returns means weaker marginal efficiency, not negative return.

Qualitative generation under fixed prompts/decoding did **not** produce a stable monotonic A→B→C human-visible ranking. Repetition, topic drift, invented entities, and long-range incoherence remained common; Model B often appeared more topically stable than Model C despite C's better likelihood.

Notebook 05 decisions are D-084 through D-094. The frozen Notebook 05 A/B/C comparison must never be replaced by 06A results.

## Notebook 06A - completed Model C extended-training probe
Research question: Was Model C materially training-duration constrained at the frozen three-epoch boundary?

### Resume/provenance gate
D-096 passed before extension training:
- behavioral resume identity reproduced frozen validation loss 3.684501 to five decimals
- full precision observed: **3.6845006885642775**
- AdamW moments/step continuity verified
- Model C parameter count: 33,497,600
- scaler/RNG/config/hashes verified
- epoch-4 deterministic shuffle seed: 46
- first ten epoch-4 example indices: `[23330, 14720, 12892, 24465, 36182, 35545, 35235, 2313, 29923, 36592]`

### Extension policy
- exact resume from global update 3,663
- constant LR `2e-4`
- frozen data/model/optimizer/batch/packing/validation controls
- `min_delta=0.001`, patience=6 validation events
- max 10 additional epochs / global ceiling 15,873
- official test sealed during training

### Completed extension outcome
- final global update: **13,263**
- extension updates: **9,600**
- extension target exposures: **157,250,560**
- validation events: **55**
- best checkpoint: **global update 12,210**
- best validation loss: **3.599946362767629**
- stop reason: `early_stopping_patience_exhausted`
- classification: **continued improvement followed by saturation / noisy validation plateau**
- FP16 overflow retries: **3**
- final FP16 loss scale: **524,288**

The runtime overflows were handled by deterministic same-logical-update replay with loss-scale backoff only; failed attempts did not consume optimizer updates or data position. Model/optimizer diagnostics showed no non-finite persisted tensors.

### Precommitted one-time exploratory official test
Only the validation-selected update-12,210 checkpoint was scored, exactly once:
- test loss: **3.606927575449253**
- test perplexity: **36.85265170399543**
- test targets: **293,376**
- frozen three-epoch Model C test loss: **3.6805543749744354**
- improvement in test loss: **0.07362679952518247**
- test stream SHA-256: `9578e1403a94bf085eb55372e76d4dd74e02c89f7085368c75a1d75f5537d188`
- official 06A scoring count: **1**
- no post-test checkpoint reselection or retuning permitted

The validation improvement generalized to the held-out test, confirming that frozen three-epoch Model C was materially **training-duration constrained**.

### Fixed D-091 generation follow-up
The same three validation-derived prompts, seeds 43/44/45, temperature 0.8, top-p 0.9, and 96 new tokens were reused.

Interpretation:
- prompts 1 and 2 showed better topical continuity / less repetition than frozen Model C, especially the geology prompt;
- prompt 3 remained strongly hallucinatory and invented biographical/achievement details;
- conclusion: **undertraining contributed to some original qualitative drift, but more training did not make the model reliably factual.**

### Final evidence
Repository namespace:
`results/extended_training/model_c/final_evidence/`

Committed small evidence:
- `extension_summary.json`
- `extension_progress.json`
- `one_time_exploratory_test.json`
- `fixed_d091_generation.json`
- `artifact_manifest_sha256.json`
- `EVIDENCE_PACKAGE_NOTE.md`

Complete extension history:
- records: **9,600**
- bytes: **3,151,019**
- SHA-256: `cdc0cb63dfcdab65f1718347a4e352b2764a9d11cd6900144cdb9e6d279ed380`
- exact bytes remain in persistent Drive and the independently verified six-file evidence package because the connected GitHub closure interface did not expose a direct multi-megabyte local-file upload path; no rounded/reconstructed substitute was committed.

External checkpoints (not committed by design):
- best SHA-256: `d4ead0686e01ea6457f74d780b2dc3859bd3fe6d2f2d164d257bda8f87ef4086`
- latest SHA-256: `effe670ebfbdb8f738635939ac4426570f36b4481cd3964d343633e4fec6335b`

## Final project conclusions
1. **Capacity helps within the tested range:** A→B→C improved validation and test likelihood under the equal-budget protocol. These three sizes do not support numerical extrapolation to larger models.
2. **Marginal efficiency declines:** validation-loss gain per approximate parameter doubling decreased from **0.164 nats** for A→B to **0.093 nats** for B→C.
3. **Budget can confound capacity with duration:** 06A showed the largest model was still undertrained at the original three-epoch cutoff.
4. **More training eventually reached saturation:** extended C improved materially, then reached the precommitted noisy-plateau stopping criterion.
5. **Likelihood is not the same as sample quality or factuality:** extra training improved some topical stability but did not eliminate hallucination.
6. **Governance/provenance mattered:** fail-closed gates exposed an actual evidence-retention gap, exact resume was behaviorally verified, the test split stayed sealed until selection froze, and the 06A test was scored once only.

## Presentation handoff

Writing preference: do not use em dashes in project documents or responses. Use ordinary hyphens or other suitable punctuation.
The presentation and report must now use frozen evidence, not more experiments. The central narrative is:

**Controlled scale improved predictive quality, but with declining marginal efficiency. The largest model was also duration constrained under the equal-budget experiment, and a separately governed continuation showed additional held-out gains before saturation. Better likelihood did not automatically produce factual or uniformly better generated language.**

Systems-level thesis retained for presentation:
> **Autoregressive next-token prediction is the base training/generation mechanism, but it is not a complete description of modern AI-system behavior.**
