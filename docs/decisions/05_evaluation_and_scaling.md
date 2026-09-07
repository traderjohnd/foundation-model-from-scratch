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

## Next decision ID

The first new evaluation decision must be **D-084**.
