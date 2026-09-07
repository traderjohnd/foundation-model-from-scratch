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
**Selected choice:** standardized generation probe with fixed prompts and decoding settings. Development prompts come from validation; final prompts come from untouched test only after model selection. Provisional decoding: temperature .8, top-p .9. Rubric: fluency/grammar, local coherence, topical continuity, repetition/degeneration.  
**Why:** Perplexity is quantitative but does not show what improvement looks like to a human. Fixed prompts/settings make outputs comparable across A/B/C.  
**Alternatives considered:** random ad-hoc prompts; factual-accuracy scoring as primary qualitative measure.  
**Presentation relevance:** side-by-side outputs may be one of the strongest visuals.

## D-040 — Factual accuracy metric
**Selected choice:** factual accuracy is **not** a primary evaluation metric.  
**Why:** These small models see only 20M WikiText tokens; the experiment is about language modeling and scaling, not reliable factual knowledge.  
**Presentation relevance:** demonstrates appropriate metric selection and avoids overclaiming capability.

## D-041 — Resource metrics
**Selected choice:** collect GPU type, peak GPU memory, wall-clock training time, GPU-hours, tokens/sec, optimizer-step throughput, total token exposures, estimated cost; after training, inference latency/throughput where practical.  
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

The official test split remains untouched at Notebook 05 start.

---

## D-084 — Evaluation scope, frozen-input boundary, and evidence/analysis separation
**Selected choice:** Notebook 05 is a standalone reader-facing analytical report built from immutable Notebook 04 outputs. Models A/B/C, their training protocol, checkpoints, and histories are frozen inputs: no retraining, retuning, or hyperparameter changes occur in Notebook 05. The notebook explicitly separates collected evidence from interpretation. Raw measurements, provenance, and audit details live in a dedicated appendix/reference layer; the main numbered sections contain the analytical narrative: evaluation framework, quality scaling, compute scaling, diminishing returns, controlled generation, final untouched-test evaluation, and synthesis. The official test split remains sealed until the designated final-evaluation section.  
**Why:** Mixing raw run output with interpretation makes the experiment harder to audit and harder to communicate. Separating the evidence layer from the analytical layer preserves provenance while allowing the reader to follow the experimental argument cleanly. Freezing Notebook 04 prevents post-hoc tuning from contaminating the controlled scaling comparison. A stable numbered structure also makes Notebook 05 directly referenceable from the final presentation.  
**Alternatives considered:** interleave raw measurements and commentary chronologically; treat Notebook 05 as a collection of analysis cells without a standalone narrative; allow additional tuning during evaluation.  
**Presentation relevance:** presentation figures and claims can cite stable Notebook 05 sections while the appendix supplies exact supporting measurements and provenance.

## D-085 — Evaluation output artifact contract
**Selected choice:** Persist reusable machine-readable evaluation outputs under `results/evaluation/` and reproducible presentation-ready figures under `figures/evaluation/`. Keep raw source measurements and provenance in the appendix/reference layer, while the main body contains only derived metrics, figures, interpretation, and conclusions. Important outputs receive stable filenames so later presentation work can reference generated artifacts rather than manually recreating them.  
**Why:** The separation established in D-084 needs a concrete repository contract. Persisting derived data and figures prevents results from being trapped in notebook state, improves reproducibility, and creates a direct bridge from analysis to the final presentation.  
**Alternatives considered:** notebook-only outputs; interleave source data and analysis; manually rebuild figures for the presentation.  
**Presentation relevance:** enables direct reuse of canonical figures and tables in the final deck and keeps slide claims traceable to reproducible notebook outputs.  
**Refines:** D-084.

## D-086 — Evaluation metric and figure contract
**Selected choice:** Freeze scaling-analysis conventions before inspecting the complete A/B/C validation histories. Quality scaling uses validation loss and perplexity. Comparisons report absolute and relative changes, parameter/time/memory multipliers, and marginal quality improvement per added million parameters, per added training minute, and per added GiB. Log-parameter views may be used where analytically useful. Important figures must be code-generated, clearly titled, have axes labeled with units, be independently understandable, and be saved with stable presentation-ready filenames.  
**Why:** Precommitting the comparison metrics reduces hindsight-driven metric selection and makes the scaling argument more defensible. Stable figure conventions improve reproducibility and presentation reuse.  
**Alternatives considered:** choose metrics after viewing curves; report only endpoint perplexity; create presentation graphics manually outside the analytical pipeline.  
**Presentation relevance:** directly defines the quantitative and visual evidence used to support the final scaling conclusions.

## D-087 — Validation-history ingestion and audit contract
**Selected choice:** Before any scaling interpretation, ingest the complete persistent A/B/C history and run-summary artifacts and hard-gate them against the frozen Notebook 04 contract. Require all history/summary/checkpoint artifacts to exist; verify model identity, 3,663 completed updates, 3 epochs, exactly 21 validation events at the canonical validation schedule, 256,512 targets per validation event, final validation at update 3,663, `perplexity = exp(loss)`, best-loss/update agreement with each run summary, compact production-summary agreement, equal production controls, and explicit confirmation that official test content was not used. Normalize the verified validation evidence into a canonical long-form table under `results/evaluation/evidence/`. Any failed assertion stops comparative analysis; no inconsistent evidence is repaired, interpolated, or silently omitted.  
**Why:** Scaling conclusions are only defensible if all three learning curves come from complete, mutually comparable artifacts under the same experiment. A hard evidence gate separates provenance validation from analysis and prevents accidental comparison of partial, stale, or mismatched runs.  
**Alternatives considered:** trust the compact endpoint summary alone; parse training logs without cross-checks; continue analysis with missing validation points.  
**Presentation relevance:** provides an auditable foundation for every learning-curve and scaling claim used later in the notebook and presentation.  
**Refines:** D-084, D-085, D-086.

## Next decision ID

The next new evaluation decision is **D-088**.
