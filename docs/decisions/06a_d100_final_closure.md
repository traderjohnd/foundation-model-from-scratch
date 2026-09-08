# D-100 — Notebook 06A final synthesis and project experimental closure

Notebook 06A is complete. This decision freezes the final exploratory conclusions after validation selection, the precommitted one-time official-test evaluation, the fixed D-091 generation probe, and final evidence packaging.

## Frozen quantitative outcome

The original three-epoch Model C remains part of the frozen Notebook 05 A/B/C comparison and is unchanged:

- validation loss: **3.6845006885642775** (reported in Notebook 05 as 3.684501)
- test loss: **3.6805543749744354**
- test perplexity: **39.668379**
- global update: **3,663**

The separate 06A continuation selected its best checkpoint exclusively by validation loss:

- best 06A global update: **12,210**
- extension update: **8,547**
- best validation loss: **3.599946362767629**
- final training stop: global update **13,263**
- extension updates executed: **9,600**
- stop reason: `early_stopping_patience_exhausted`
- classification: **continued improvement followed by saturation / noisy validation plateau**

The precommitted one-time exploratory official-test evaluation then scored only the frozen update-12,210 checkpoint:

- test loss: **3.606927575449253**
- test perplexity: **36.85265170399543**
- scored targets: **293,376**
- improvement in test loss vs frozen three-epoch C: **0.07362679952518247**
- official 06A test-scoring count: **1**
- checkpoint selection changed after test: **false**
- retuning permitted after test: **false**

The held-out test improvement confirms that the validation gain generalized. Therefore the frozen three-epoch Model C was materially **training-duration constrained**. The extension later reached the precommitted saturation criterion, so the result is not that Model C could improve indefinitely.

## Qualitative follow-up

The fixed D-091 validation-prompt generation probe reused the exact three frozen prompts, seeds 43/44/45, temperature 0.8, top-p 0.9, and 96 new tokens.

Observed interpretation:

- Prompt 1 showed better topical continuity around city/building/industrial-development material than the original Model C sample.
- Prompt 2 showed the clearest improvement: substantially less repetitive ridge-language and better topical continuity in geology/topography.
- Prompt 3 remained unreliable and fabricated biographical/achievement details.

The conservative conclusion is: **undertraining contributed to some of the original qualitative drift, but additional training did not make the small model reliably factual.** Likelihood improvement and qualitative stability are related but not interchangeable objectives.

## Numerical-runtime incident closure

During 06A continuation, the T4/FP16 GradScaler reached a loss scale of 1,048,576 and produced non-finite gradients on three logical updates. Diagnostic inspection found zero non-finite model tensors and zero non-finite optimizer-state tensors. The recovery runner replayed the same logical update with the same data order and restored RNG state while backing off only the FP16 loss scale.

Final numerical-recovery facts:

- overflow retries: **3**
- final loss scale: **524,288**
- failed overflow attempts did not advance optimizer counters or data position
- LR, batch semantics, model architecture, AdamW hyperparameters, and official-test policy were unchanged

This is classified as runtime numerical-stability recovery, not hyperparameter retuning.

## Evidence closure

The final evidence package validates:

- `extension_summary.json`
- `extension_history.json`
- `extension_progress.json`
- `one_time_exploratory_test.json`
- `fixed_d091_generation.json`
- `artifact_manifest_sha256.json`

The five small JSON artifacts and manifest are committed under `results/extended_training/model_c/final_evidence/`. The complete 9,600-record `extension_history.json` is 3,151,019 bytes with SHA-256 `cdc0cb63dfcdab65f1718347a4e352b2764a9d11cd6900144cdb9e6d279ed380`. Its exact bytes were verified against the committed manifest from the uploaded six-file evidence package. The connected repository-write interface did not expose a direct multi-megabyte local-file upload path, so the full history remains in persistent Drive and the verified package rather than being reconstructed or rounded for GitHub. This transport boundary is documented explicitly in `results/extended_training/model_c/final_evidence/EVIDENCE_PACKAGE_NOTE.md`.

External checkpoint hashes:

- best checkpoint: `d4ead0686e01ea6457f74d780b2dc3859bd3fe6d2f2d164d257bda8f87ef4086`
- latest checkpoint: `effe670ebfbdb8f738635939ac4426570f36b4481cd3964d343633e4fec6335b`

Large checkpoint binaries remain outside Git by design.

## Final scientific interpretation

The controlled Notebook 05 experiment and the 06A exploratory continuation answer different questions and must remain separate:

1. **Capacity scaling under equal budget:** A→B→C improved predictive quality monotonically, but B→C delivered weaker marginal efficiency per added parameter, training minute, and GiB.
2. **Training-duration sensitivity of the largest model:** frozen Model C had not exhausted useful learning at three epochs; additional optimization improved both validation and held-out test likelihood before reaching a noisy saturation plateau.
3. **Likelihood vs generated behavior:** more training improved some topical stability but did not eliminate hallucination or guarantee a monotonic human-visible quality ranking.

No 06A checkpoint, metric, compute cost, or generated sample replaces any frozen Notebook 05 A/B/C result.

## Project status

The experimental portion of **Building a Foundation Model from Scratch** is complete. No additional model training, test scoring, LR search, checkpoint selection, or scope expansion is required for the core project.

Remaining work is presentation/report assembly from the frozen repository evidence. Any future fine-tuning, quantization, architectural research, or additional scaling run is a new project or explicitly separate follow-on experiment.

## Presentation relevance

The final narrative should emphasize four lessons:

- larger models improved likelihood under equal data/training controls, but marginal efficiency declined;
- a fixed training budget can confound capacity with duration, which 06A exposed cleanly without rewriting the original experiment;
- disciplined test sealing and fail-closed provenance gates materially improved the credibility of the project;
- better perplexity and more training do not automatically produce factual or uniformly better generated language.

## Next decision ID

The next globally unique decision ID is **D-101**.
