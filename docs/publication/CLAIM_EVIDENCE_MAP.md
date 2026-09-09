# Publication Claim-to-Evidence Map

Version 1 | Prepared 2026-09-09 | Publication decision D-101

Evidence snapshot: [47c6973](https://github.com/traderjohnd/foundation-model-from-scratch/tree/47c6973a898e770eceb77f35ad91a6f2c1afc154) (47c6973a898e770eceb77f35ad91a6f2c1afc154).

This is the first publication and presentation assembly artifact. It connects proposed public wording to the frozen evidence and states what each result can support. C-01 through C-08 identify claims; D- identifiers refer to the project decision register. Source links are pinned to the evidence snapshot so later documentation changes cannot silently change their meaning.

Decision details: [global index][index], [project definition][definition], [training register][training-register], [Notebook 05 register][evaluation-register], and [Notebook 06A register][extension-register]. The new [D-101 assembly contract](../decisions/00_project_definition.md) accompanies this map.

## Central narrative

Across the three tested model sizes, predictive quality improved while the marginal gain per added resource declined. A separate continuation showed that Model C could improve further on the same corpus under the specified extension policy, before reaching the predefined validation plateau criterion. The fixed generation probes showed that better likelihood did not ensure consistently better sampled text or reliable factual content.

## Experimental scope

The original comparison trained three decoder-only Transformers from random initialization using an explicit PyTorch model and training loop, with a byte-level BPE tokenizer trained from scratch. PyTorch and Hugging Face Datasets/Tokenizers supplied infrastructure; the [implementation policy][definition] excluded pretrained model weights, a pretrained tokenizer, and HF Trainer. The [model module][model] exposes the architecture and initialization.

Models A/B/C used the same 20,000,000-token training corpus, tokenizer, 512-token context, seed convention, and three-epoch protocol. Packing admitted 19,999,744 scored targets per epoch, or 59,999,232 target exposures per model. This is a fixed corpus and training-exposure comparison; elapsed time and compute were allowed to vary with model size. The tokenizer was trained on the full normalized official training split, while the language-model corpus was the selected 20M-token subset. See [corpus metadata][corpus], [tokenizer metadata][tokenizer], [training results][production], and D-043/D-044, D-052, D-065/D-066, D-081–D-083.

## Claim-to-evidence table

| ID and kind | Proposed publication claim | Evidence and visual anchor | Decisions | Limit that must accompany interpretation |
|---|---|---|---|---|
| **C-01 - Measured outcome** | Larger models achieved lower validation and test loss across the three tested sizes under the shared protocol. | Validation PPL **53.09 → 43.66 → 39.83**; test PPL **52.21 → 43.47 → 39.67**. [Endpoints][quality], [test results][test], [history audit][ingestion]; [Figure 01][fig01]. | D-059; D-081–D-083; D-087/D-088; D-092 | One primary seed per size; compound depth/width changes; LR selected using Model A. These are observed results for this family and protocol, with no estimate of variation across training seeds. |
| **C-02 - Measured and derived resource outcome** | Better predictive quality required more recorded training time and peak GPU memory in these runs. | T4 time **11.78 → 23.55 → 42.22 min**; peak allocation **4.94 → 7.29 → 10.62 GiB**. [Compute summary][compute]; [Figures 03][fig03], [04][fig04], [05][fig05], and [06][fig06]. | D-041; D-079; D-089 | Time and peak allocation are recorded run measurements. Throughput and GPU-hours are derived. End-to-end throughput is specific to this workload and timing scope; it is not inference speed or a pure kernel benchmark. |
| **C-03 - Derived comparison** | B→C delivered a smaller validation-loss gain per added parameter, training minute, and GiB than A→B, while C retained the best absolute likelihood. | B→C retained **26.9%, 29.6%, and 33.2%**, respectively, of A→B validation-loss efficiency. [Marginal returns][marginal], [retention][retention]; [Figure 07][fig07]. | D-086; D-090; D-094 | Two observed increments establish local diminishing marginal efficiency. They do not locate a universal capacity ceiling or a compute-optimal model size. These efficiency numbers use validation loss. |
| **C-04 - Bounded inference from continuation** | Frozen three-epoch Model C was materially training-duration constrained under the implemented continuation policy. | Exact parent update **3,663**; selected extension update **12,210**. Validation loss **3.684501 → 3.599946**; test PPL **39.67 → 36.85**. [Resume gate][resume], [extension summary][extension], [exploratory test][extension-test]. Use the separate 06A table below. | D-095/D-096; D-099/D-100 | Only C received extra optimization. The extension held the terminal LR at **2e-4**. It establishes useful remaining learning under that policy; it does not isolate additional unique-data value, establish the best LR policy, or revise the frozen A/B/C ranking. |
| **C-05 - Protocol-defined classification** | Extended C improved, then met the precommitted noisy-validation-plateau stopping criterion. | Best update **12,210**; stop **13,263**; **55** validation events; patience **6**; min_delta **0.001**. [Summary][extension] and D-099's recorded final six events in the [06A register][extension-register]. | D-095; D-099/D-100 | “Saturation” describes the observed trajectory under the specified LR, cadence, and stopping rule. The register classifies it as a noisy plateau rather than sustained overfitting; no universal optimization limit is demonstrated. |
| **C-06 - Qualitative observation** | Lower perplexity did not produce a consistently better qualitative ranking in the fixed probes. Extended C improved some topical continuity, while substantial content errors remained. | Nine [D-091 A/B/C continuations][generation] and three [06A continuations][extension-generation]; identical prompts, seeds 43/44/45, temperature 0.8, top-p 0.9, and 96 new tokens for the before/after C comparison. | D-039/D-040; D-091; D-096; D-100 | Small, fixed qualitative sample; no blinded rating study or quantitative factuality benchmark. The geology continuation becomes less repetitive while still containing inconsistent measurements. Better topical flow does not establish factual accuracy. |
| **C-07 - Decision evidence** | A controlled LR probe changed the provisional production choice from 3e-4 to 2e-3. | At probe update 400, validation loss was **6.163024** at 3e-4, **5.488027** at 2e-3, and **5.741478** at 3e-3. [Probe results][lr]; use a compact table rather than a scaling figure. | D-023; D-071; D-077/D-078 | 2e-3 was best among the tested candidates under the 400-update Model A probe. This does not establish an exact global optimum or a separately tuned optimum for B/C. |
| **C-08 - Provenance and engineering evidence** | Continuation was gated on recorded artifact identity and resume checks; the process exposed and corrected an actual evidence-retention gap. | [Recovery record][recovery] and [resume gate][resume]: 23 canonical NB05 artifacts checked; parent validation **3.6845006885642775**; **74** AdamW tensor states checked with no reported step/shape mismatches. | D-087; D-096–D-098; D-100 | These records support specific checks and a disclosed recovery. Hashes identify bytes; behavioral and state checks provide additional evidence. They do not establish a universal correctness guarantee or independent end-to-end reproduction. |

## Numerical reference - frozen Notebook 05 A/B/C

These values describe the original equal-exposure experiment only. Parameter counts are exact. Loss is displayed to six decimal places; perplexity, minutes, and GiB to two. Full source precision remains in the linked artifacts.

| Metric | Model A | Model B | Model C |
|---|---:|---:|---:|
| Parameters | 7,407,872 | 16,913,280 | 33,497,600 |
| Target exposures | 59,999,232 | 59,999,232 | 59,999,232 |
| Best checkpoint update | 3,663 | 3,663 | 3,663 |
| Validation loss | 3.972054 | 3.776427 | 3.684501 |
| Validation perplexity | 53.09 | 43.66 | 39.83 |
| Test loss | 3.955290 | 3.772079 | 3.680554 |
| Test perplexity | 52.21 | 43.47 | 39.67 |
| Recorded training time (min) | 11.78 | 23.55 | 42.22 |
| Peak GPU allocation (GiB) | 4.94 | 7.29 | 10.62 |

Sources: [production][production], [validation endpoints][quality], [test results][test], [compute summary][compute]. Validation scored 256,512 targets per event; test scored 293,376 targets per model. Perplexity comparisons in this document share the same tokenizer and evaluation procedure.

For the publication plan's per-doubling wording, the descriptive calculation is (loss_before − loss_after) / log2(parameters_after / parameters_before): **0.164251** nats for A→B and **0.093241** for B→C, using total parameters and unrounded validation endpoints. These are arithmetic summaries of the two observed intervals, not a fitted scaling law. D-090's original per-resource analysis remains the primary diminishing-return evidence.

## Numerical reference - separate exploratory Notebook 06A

| Metric | Frozen three-epoch C | Validation-selected extended C |
|---|---:|---:|
| Checkpoint global update | 3,663 | 12,210 |
| Validation loss | 3.684501 | 3.599946 |
| Test loss | 3.680554 | 3.606928 |
| Test perplexity | 39.67 | 36.85 |

Sources: [parent validation endpoint][quality], [frozen test][test], [extension summary][extension], [one-time exploratory test][extension-test]. Test loss improved by **0.073627** nats per scored target; test perplexity decreased by approximately **7.10%**, calculated from the values recorded in the exploratory-test artifact.

The selected checkpoint and the stopping checkpoint are different. The complete continuation executed **9,600** additional updates and **157,250,560** additional target exposures, stopping at global update **13,263**. Those completed-run totals are not the training budget of the selected update-12,210 checkpoint.

### 06A figure source

The full 9,600-record extension history is preserved externally, with its identity recorded in the [manifest][extension-manifest] and [evidence package note][extension-package]. Its SHA-256 is **cdc0cb63dfcdab65f1718347a4e352b2764a9d11cd6900144cdb9e6d279ed380** and its size is **3,151,019 bytes**.

A publication figure showing the complete extension training/validation trajectory must be generated from those exact bytes. It has not been created in this chunk. The committed summary and D-099's final-six-event list support the endpoint and stop-classification claims above; they cannot stand in for the complete curve. The original A/B/C figures stay visually distinct from any later 06A curve.

## Test-use disclosure

The publication should describe the full sequence:

1. **D-092:** original official-test evaluation of the frozen, validation-selected A/B/C checkpoints.
2. **D-097:** deterministic test re-measurement of those same frozen checkpoints under the unchanged evaluation procedure, solely to recover missing repository artifacts. The [NB05 register's recovery note][evaluation-register] records that this occurred on 2026-09-07 without checkpoint reselection or retuning.
3. **D-099/D-100:** validation selection of the 06A checkpoint froze before its precommitted exploratory official-test evaluation. The [test record][extension-test] reports a scoring count of one for 06A and no post-test reselection.

“Test scored once” therefore needs the **06A** qualifier. The later extension uses an already-examined benchmark test split and is labeled exploratory. Its score is a descriptive generalization check under the precommitted policy.

## Wording rules for downstream artifacts

| Topic | Publication treatment |
|---|---|
| Punctuation | Do not use em dashes. Use ordinary hyphens, commas, colons, semicolons, or parentheses as appropriate. |
| Budget | Name the corpus size and processed-target exposures. Use “same corpus and training-exposure budget” for A/B/C; compute and elapsed time varied. |
| Capacity and duration | Attribute C's later gain to useful continued optimization under the frozen extension policy. Keep additional unique data and alternative schedules as untested questions. |
| Data/parameter ratios | Corpus positions per parameter are approximately A/B/C **2.70/1.18/0.60**; processed target exposures per parameter are **8.10/3.55/1.79**. Label the numerator. Neither ratio establishes this experiment's compute optimum. |
| Model capability | Describe Wikipedia-style continuation and the observed sample weaknesses. Factual question answering and broad foundation-model capability were not established by this evaluation. |
| Evidence versus judgment | Label endpoint/resource measurements, arithmetic comparisons, protocol classifications, and qualitative observations distinctly. |
| From-scratch contribution | State the use of PyTorch and Hugging Face Datasets/Tokenizers precisely. Describe the original learned artifacts and explicit training implementation. Add a factual human/AI contribution statement when drafting the narrative. |
| Metric comparison | Keep tokenizer, scoring procedure, and evaluation split explicit. The reported BPE-token perplexities are not interchangeable with published word-level WikiText perplexities. |
| Reproducibility | Describe the checks actually recorded and the exact artifact locations. Independent reconstruction from a future build specification remains a separate, unperformed validation. |

## Handoff for the next publication chunk

Draft the short executive narrative from C-01 through C-06, using Figure 06 for the original quality/resource comparison and the separate 06A endpoint table for the continuation. Use C-07 and C-08 as concrete examples of how decisions were made and checked. The README, presentation, article, and monograph should reuse these claim IDs and evidence links as working references.

The next chunk is narrative assembly. The completed experiment supplies its evidence; publication work adds explanation and presentation assets.

[context]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/docs/PROJECT_CONTEXT.md
[index]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/docs/DECISION_INDEX.md
[definition]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/docs/decisions/00_project_definition.md
[training-register]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/docs/decisions/04_training_pipeline.md
[evaluation-register]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/docs/decisions/05_evaluation_and_scaling.md
[extension-register]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/docs/decisions/06a_model_c_extended_training_probe.md
[corpus]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/corpus/corpus_summary.json
[tokenizer]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/tokenizer/tokenizer_metadata.json
[model]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/src/model.py
[production]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/training/production_scaling_summary.json
[quality]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/evaluation/analysis/quality_endpoints.csv
[test]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/evaluation/analysis/final_test_results.json
[compute]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/evaluation/analysis/compute_scaling_summary.csv
[marginal]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/evaluation/analysis/marginal_returns.csv
[retention]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/evaluation/analysis/marginal_efficiency_retention.csv
[ingestion]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/evaluation/evidence/validation_history_ingestion_audit.json
[generation]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/evaluation/analysis/controlled_generation_results.json
[lr]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/training/model_a_lr_probe_refined.json
[resume]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/extended_training/model_c/d096_resume_gate.json
[recovery]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/docs/provenance/05_evaluation_artifact_recovery.md
[extension]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/extended_training/model_c/final_evidence/extension_summary.json
[extension-test]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/extended_training/model_c/final_evidence/one_time_exploratory_test.json
[extension-generation]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/extended_training/model_c/final_evidence/fixed_d091_generation.json
[extension-manifest]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/extended_training/model_c/final_evidence/artifact_manifest_sha256.json
[extension-package]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/results/extended_training/model_c/final_evidence/EVIDENCE_PACKAGE_NOTE.md
[fig01]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/figures/evaluation/fig_01_validation_loss_learning_curves.png
[fig03]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/figures/evaluation/fig_03_training_wall_time_vs_parameters.png
[fig04]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/figures/evaluation/fig_04_peak_gpu_memory_vs_parameters.png
[fig05]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/figures/evaluation/fig_05_effective_throughput_vs_parameters.png
[fig06]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/figures/evaluation/fig_06_quality_cost_frontier.png
[fig07]: https://github.com/traderjohnd/foundation-model-from-scratch/blob/47c6973a898e770eceb77f35ad91a6f2c1afc154/figures/evaluation/fig_07_marginal_efficiency_retained.png
