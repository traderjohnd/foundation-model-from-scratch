# Decision Register — 05 Evaluation & Scaling

Notebook 05 begins from a **frozen training protocol**. It consumes saved A/B/C histories/checkpoints and is responsible for comparative analysis, controlled generation, efficiency tradeoffs, diminishing-return analysis, and final untouched-test evaluation.

Some evaluation decisions were made during project planning and therefore retain earlier globally unique IDs. New Notebook 05 decisions start at **D-084** and append to this file; existing IDs are never rewritten.

---

## D-021 — Perplexity
**Selected choice:** track validation perplexity for all three models and final test perplexity at the end.  
**Why:** Perplexity is a human-interpretable transform of cross-entropy and is appropriate for comparing models when tokenizer/evaluation data are identical.  
**Alternatives considered:** loss only; generation quality only.  
**Presentation relevance:** explain `perplexity = exp(cross-entropy)` and why lower is better.

## D-039 — Qualitative generation evaluation
**Selected choice:** standardized generation probe with fixed prompts and decoding settings. Development prompts come from validation; final prompts come from untouched test only after model selection. Decoding: temperature 0.8, top-p 0.9, 96 new tokens. Rubric: fluency/grammar, local coherence, topical continuity, repetition/degeneration.  
**Why:** Perplexity is quantitative but does not show what improvement looks like to a human. Fixed prompts/settings make outputs comparable across A/B/C.  
**Alternatives considered:** random ad-hoc prompts; factual-accuracy scoring as primary qualitative measure.  
**Presentation relevance:** side-by-side outputs illustrate why aggregate likelihood and individual stochastic generations are related but not identical.

## D-040 — Factual accuracy metric
**Selected choice:** factual accuracy is **not** a primary evaluation metric.  
**Why:** These small models see only 20M WikiText tokens; the experiment is about language modeling and scaling, not reliable factual knowledge.  
**Presentation relevance:** demonstrates appropriate metric selection and avoids overclaiming capability.

## D-041 — Resource metrics
**Selected choice:** collect GPU type, peak GPU memory, wall-clock training time, GPU-hours, effective target-exposure throughput, optimizer-step throughput, and total token exposures; estimate dollar cost only if a defensible pricing basis exists.  
**Why:** Compute cost is part of the experimental objective, not merely an implementation detail.  
**Alternatives considered:** training time only; mandatory energy/FLOPs accounting.  
**Presentation relevance:** capability-vs-cost comparison.

## D-042 — Energy/FLOPs measurement
**Selected choice:** energy and theoretical FLOPs are optional, not mandatory.  
**Why:** Avoid measurement overhead becoming the project unless estimates can be added cleanly and defensibly.  
**Presentation relevance:** optional appendix metric.

---

## Frozen inputs from Notebook 04

| Metric | Model A | Model B | Model C |
|---|---:|---:|---:|
| Parameters | 7,407,872 | 16,913,280 | 33,497,600 |
| Best validation loss | 3.972054 | 3.776427 | 3.684501 |
| Best validation PPL | 53.09 | 43.66 | 39.83 |
| Best update | 3,663 | 3,663 | 3,663 |
| Wall time | 11.78 min | 23.55 min | 42.22 min |
| Peak memory | 4.94 GiB | 7.29 GiB | 10.62 GiB |

The official test split remained untouched until D-092 / Section 7.

---

## D-084 — Evaluation scope, frozen-input boundary, and evidence/analysis separation
**Selected choice:** Notebook 05 is a standalone reader-facing analytical report built from immutable Notebook 04 outputs. Models A/B/C, their training protocol, checkpoints, and histories are frozen inputs: no retraining, retuning, or hyperparameter changes occur in Notebook 05. Raw measurements, provenance, and audit details live in an appendix/reference layer; the main numbered sections contain the analytical narrative. The official test split stays sealed until the designated final-evaluation section.  
**Why:** Separating evidence from interpretation preserves provenance while making the experimental argument readable. Freezing Notebook 04 prevents post-hoc tuning from contaminating the controlled scaling comparison.  
**Alternatives considered:** interleave raw measurements and commentary chronologically; allow additional tuning during evaluation.  
**Presentation relevance:** stable numbered sections can be cited directly from the final presentation.

## D-085 — Evaluation output artifact contract
**Selected choice:** persist reusable machine-readable evaluation outputs under `results/evaluation/` and presentation-ready figures under `figures/evaluation/`, with stable filenames.  
**Why:** Prevents results from being trapped in notebook state and creates a reproducible bridge to the final presentation.  
**Alternatives considered:** notebook-only outputs; manual recreation of presentation figures.  
**Presentation relevance:** enables direct reuse of canonical figures and tables.  
**Refines:** D-084.

## D-086 — Evaluation metric and figure contract
**Selected choice:** before full-history inspection, freeze validation loss/perplexity, absolute/relative changes, parameter/time/memory multipliers, and marginal quality improvement per added million parameters, minute, and GiB. Important figures must be code-generated, labeled, independently understandable, and saved with stable filenames.  
**Why:** Reduces hindsight-driven metric selection and makes the scaling argument more defensible.  
**Alternatives considered:** choose metrics after viewing curves; report only endpoint perplexity.  
**Presentation relevance:** defines the quantitative and visual evidence used later.

## D-087 — Validation-history ingestion and audit contract
**Selected choice:** ingest complete persistent A/B/C history and run-summary artifacts and hard-gate them against the frozen Notebook 04 contract. Require history/summary/checkpoint artifacts; verify 3,663 updates, 3 epochs, 21 validation events at the canonical schedule, 256,512 targets/event, final validation at 3,663, `perplexity = exp(loss)`, best-loss/update agreement, compact-summary agreement, equal controls, and no prior test use. Normalize verified evidence under `results/evaluation/evidence/`.  
**Why:** Scaling conclusions require complete, mutually comparable artifacts.  
**Alternatives considered:** trust endpoint summaries alone; continue with partial histories.  
**Presentation relevance:** auditable foundation for every learning-curve and scaling claim.  
**Refines:** D-084, D-085, D-086.

## D-088 — Quality-scaling analysis
**Selected choice:** analyze all 21 shared validation checkpoints plus best/final endpoint quality for A/B/C, reporting absolute and relative loss/perplexity changes while keeping compute interpretation separate.  
**Evidence/result:** best validation loss improved monotonically A→B→C: 3.972054 → 3.776427 → 3.684501; perplexity 53.09 → 43.66 → 39.83. A→B loss reduction was 4.93% and B→C was 2.43%; all three models were still best at update 3,663.  
**Why:** Separates the quality effect of capacity from resource cost before combining them.  
**Presentation relevance:** establishes continued benefit from scale and an early quality-side diminishing-return signal.

## D-089 — Compute-cost scaling analysis
**Selected choice:** compare directly measured wall-clock time and peak allocated GPU memory, and derive GPU-hours, updates/sec, and effective target-exposures/sec from the recorded workload and time. Treat throughput as an end-to-end experiment measure rather than a pure kernel benchmark.  
**Evidence/result:** wall time 11.78 → 23.55 → 42.22 minutes; peak memory 4.94 → 7.29 → 10.62 GiB; effective throughput approximately 84,889 → 42,462 → 23,685 target exposures/sec. Across A→C, parameters increased 4.52×, wall time 3.58×, memory 2.15×, and effective throughput fell about 72%.  
**Why:** Makes the resource side of scaling explicit without conflating measured and derived quantities.  
**Presentation relevance:** shows the operational cost of capacity scaling.

## D-090 — Marginal return analysis
**Selected choice:** evaluate diminishing returns only through the observed incremental A→B and B→C steps, using the D-086 precommitted denominators; do not fit a universal scaling law from three model sizes.  
**Evidence/result:** B→C retained only about 26.9% / 23.3% of A→B loss/perplexity efficiency per added parameter, 29.6% / 25.6% per added training minute, and 33.2% / 28.7% per added GiB of peak memory. Model C remained best in absolute quality.  
**Why:** Distinguishes continued absolute improvement from declining marginal efficiency.  
**Presentation relevance:** directly answers the higher-level diminishing-return question within the experiment's scope.

## D-091 — Controlled generation execution contract
**Selected choice:** generate three deterministic validation-derived prompt continuations from each best A/B/C checkpoint using temperature 0.8, top-p 0.9, 96 new tokens, and identical per-prompt seeds; evaluate fluency/grammar, local coherence, topical continuity, and repetition/degeneration.  
**Evidence/result:** all three models produced recognizable WikiText-like prose, but the nine samples did not show a stable monotonic A→B→C qualitative ranking. Model C sometimes preserved topic better but also showed repetition and abrupt drift.  
**Why:** Human-visible generation is complementary to aggregate predictive metrics, not a replacement for them.  
**Presentation relevance:** demonstrates why lower perplexity does not guarantee every stochastic sample looks better.

## D-092 — Final untouched-test evaluation contract
**Selected choice:** open the official WikiText-103 test split only after all model/training/validation-based decisions are frozen; reconstruct/tokenize it with the same causal evaluation procedure and score each best validation-selected checkpoint once. No retuning or reselection follows test results.  
**Evidence/result:** test loss A/B/C = 3.955290 / 3.772079 / 3.680554; test perplexity = 52.210814 / 43.470355 / 39.668379 over 293,376 scored targets/model. The monotonic ranking survived intact, with small slightly negative validation-to-test gaps.  
**Why:** Provides genuinely held-out final generalization evidence without leakage.  
**Presentation relevance:** independently confirms the central scaling result.

## D-093 — Final test-prompt generation contract
**Selected choice:** after quantitative test scores are frozen, generate three fixed test-derived prompt continuations per model using the same decoding protocol as D-091. These outputs are presentation-oriented qualitative evidence only and cannot alter model selection or training.  
**Evidence/result:** the final nine test-prompt samples again showed noisy human-visible ranking; Model B was often more topically stable, while Model C still exhibited drift or repetition despite having the best test perplexity.  
**Why:** Completes the originally planned validation-vs-final-test qualitative protocol while preserving test integrity.  
**Presentation relevance:** supports a nuanced distinction between aggregate predictive quality and individual sampled behavior.

## D-094 — Notebook 05 final synthesis and closure
**Selected choice:** close the original controlled A/B/C experiment after final synthesis, limitations, presentation-ready findings, and a compact `results/evaluation/analysis/final_evaluation_summary.json` artifact. Treat any future Model C extended training as a separately labeled exploratory experiment that cannot replace the frozen three-epoch Model C result in the A/B/C comparison.  
**Evidence/result:** Notebook 05 final execution completed with no cell errors and emitted `Notebook 05 status: COMPLETE`. The central finding is monotonic predictive-quality improvement with capacity, coupled with a clear B→C decline in marginal efficiency; qualitative generation remains noisier than perplexity ranking.  
**Why:** Preserves the causal interpretation of the controlled scaling experiment while creating a clean handoff to follow-on exploration.  
**Presentation relevance:** provides the final bounded claims, limitations, and reusable summary for the presentation.

---

## Artifact recovery provenance note — 2026-09-07

The stable Notebook 05 repository artifacts required by D-085 were found to be absent from `main` even though the executed notebook had produced them successfully in Colab. Under the later D-097 recovery policy, Notebook 05 was deterministically re-executed from the same frozen production checkpoints/histories, pinned data pipeline, and unchanged evaluation procedures solely to recover those missing repository artifacts.

Before the recovered files were declared eligible for commit, the recovery verifier reproduced the frozen evidence at full precision and checked the registered invariants:
- validation best loss A/B/C = `3.9720542587919865` / `3.7764272480429764` / `3.6845006885642775`, each best at update 3,663, with 21 validation events/model and 256,512 scored targets/event;
- test loss A/B/C = `3.955289630989754` / `3.772079201476944` / `3.6805543749744354` over exactly 293,376 scored targets/model;
- the full canonical Notebook 05 output set contained 23 evidence/analysis/figure artifacts, each recorded with byte size and SHA-256 in `results/evaluation/artifact_manifest_sha256.json`.

The official test split was therefore re-scored on 2026-09-07 only as a **provenance-recovery re-measurement** under the unchanged D-092 procedure. No checkpoint selection, model ranking, retuning, hyperparameter choice, or downstream experimental decision was made from this second measurement. The original D-092 result remains the frozen final-test decision evidence; the recovery run demonstrated reproducibility and restored the missing repository artifacts.

This note records provenance only; it does not create a new Notebook 05 experimental decision or alter D-084 through D-094.

## Next decision ID

The next globally unique decision ID is **D-095**. Notebook 06A — Model C Extended-Training Probe must begin from D-095 and remain explicitly separate from the frozen Notebook 05 scaling comparison.
