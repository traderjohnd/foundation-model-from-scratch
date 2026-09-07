# Decision Register — 06A Model C Extended-Training Probe

Notebook 06A is a **separate exploratory experiment**. It begins only after the controlled A/B/C scaling experiment has been frozen by Notebook 05 / D-094.

The frozen Model C result remains:
- epoch: **3**
- optimizer update: **3,663**
- validation loss: **3.684501**
- validation perplexity: **39.83**

No 06A checkpoint or metric may replace that frozen Model C point in the Notebook 05 A/B/C comparison.

---

## D-095 — Model C extended-training experimental contract

**Research question:** Was Model C still materially training-duration constrained at the end of the frozen three-epoch experiment, or does continued training now reach validation saturation or begin to overfit?

**Selected choice:** Continue Model C from its exact epoch-3 / update-3,663 production checkpoint under a tightly controlled extension. Preserve the original model, data, optimizer, batch, packing, validation, regularization, and numerical semantics. Change only the training-duration boundary and the post-original-schedule learning-rate policy required to continue optimization.

### Frozen starting state
The extension must resume the exact Model C state at optimizer update 3,663, not reconstruct an equivalent-looking model from weights alone. Before any extension update is allowed, the notebook must verify the presence and compatibility of the resumable checkpoint state required by the production wrapper, including at minimum:
- Model C parameters at update 3,663.
- AdamW optimizer state.
- mixed-precision scaler state when applicable.
- global optimizer-update counter and epoch position.
- RNG state(s) persisted by the production checkpoint where available/required for exact continuation.
- architecture/config identity for the 33,497,600-parameter Model C.

If the checkpoint cannot satisfy the resume gate, the experiment must stop rather than silently restart from weights or reset optimizer state.

### Frozen controls
Hold unchanged from the original experiment:
- tokenizer and tokenizer hash.
- exact 20,000,000-token training corpus and corpus hash.
- Model C architecture and parameterization.
- context length 512.
- stride-512 deterministic causal packing.
- 39,062 examples and 19,999,744 scored targets per additional epoch.
- logical effective batch of 16,384 targets/update, including the 22-sequence final partial batch.
- 1,221 optimizer updates per additional epoch.
- AdamW parameter groups, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay 0.10 for `ndim>=2`, no decay for RMSNorm scales.
- global gradient clipping at 1.0.
- dropout 0.10.
- original seed conventions and deterministic data-order semantics.
- official validation split, reconstruction, packing, and deterministic no-shuffle scoring over 256,512 targets.
- physical/numerical execution semantics appropriate to the resumed production environment; on the canonical T4 this is FP16 autocast + GradScaler with FP32 master parameters.

### Extension learning-rate policy
Use a **constant learning rate of `2e-4`** for the extension, equal to the terminal learning rate reached at update 3,663 by the frozen Notebook 04 cosine schedule.

**Why:** The purpose of 06A is not to search for a better hyperparameter schedule. Holding the terminal LR constant creates a conservative continuation that asks whether useful learning remains after the original schedule exhausted its planned decay. Restarting warmup, increasing LR, or running a new LR sweep would confound training-duration evidence with retuning.

### Validation and history contract
- Preserve the original Model C training/validation history as immutable prefix evidence.
- Append extension history into a **separate 06A artifact namespace**; never rewrite Notebook 04/05 history files.
- Record training loss over optimizer updates throughout the extension.
- Record validation loss and perplexity every **200 optimizer updates plus each additional epoch end**.
- Preserve both the complete extension-only series and a joined view whose original prefix is clearly labeled frozen and whose continuation is clearly labeled exploratory.
- Save every new best 06A validation checkpoint separately from the frozen Notebook 04/05 checkpoints.

### Early-stopping rule
Use validation-loss early stopping with:
- metric: validation cross-entropy loss.
- mode: minimize.
- `min_delta = 0.001` absolute loss.
- `patience = 6` consecutive validation observations without an improvement of at least `0.001` over the best 06A-observed validation loss.
- the first worse validation point **does not** stop training.
- patience resets only when a new validation loss improves on the current best by at least `0.001`.

**Why six observations:** the canonical validation cadence is roughly six observations per epoch (five 200-update checkpoints plus epoch end, with exact spacing determined by the 1,221-update epoch length). This requires a sustained approximately one-epoch lack of meaningful validation improvement before stopping, filtering single-point noise while remaining responsive to saturation/overfitting.

### Safety ceiling
Allow at most **10 additional epochs** (12,210 additional optimizer updates; ending no later than global update 15,873) unless early stopping triggers first.

If validation loss is still improving through that ceiling, the correct conclusion is: **no saturation was observed within the tested extension range**. The experiment must not continue merely to manufacture an overfitting result.

### Outcome classification
At completion, classify the observed extension conservatively:
1. **Continued improvement:** validation loss achieves meaningful new best values and remains improving through the tested range or stopping point.
2. **Saturation:** validation loss fails to improve by `min_delta` for the full patience window without a sustained worsening trend sufficient to support overfitting language.
3. **Overfitting:** training loss continues to improve while validation loss shows a sustained degradation after the best validation point across the patience window.

The notebook must retain the underlying curves and event history so the classification is auditable rather than inferred from endpoints only.

### Separation from frozen A/B/C experiment
All 06A outputs must be explicitly labeled **exploratory extended-training evidence**. They may answer whether Model C was duration-constrained, but they must not be substituted into:
- Notebook 05 A/B/C validation or test tables.
- the frozen diminishing-return comparison.
- the original three-epoch compute/resource comparison.
- any claim that all three models were trained under an identical budget.

**Why:** Model C alone receives additional optimization and token exposures in 06A. Mixing the extension into Notebook 05 would destroy the controlled capacity comparison.

### Validation against canonical project evidence
The contract is consistent with the canonical repository state:
- Notebook 05 / D-094 freezes the original A/B/C experiment and explicitly requires future Model C extended training to remain separate.
- the production scaling summary records Model C at 33,497,600 parameters, 3 epochs, 3,663 optimizer updates, best validation loss 3.684501 at update 3,663, and verified persistent artifacts.
- the frozen LR schedule terminates at `2e-4`, making that value the least-confounded continuation LR.
- the frozen validation cadence is every 200 optimizer updates plus epoch end.

**Alternatives considered:** stop at the first validation increase; reset/restart the cosine schedule; restart warmup; run a new LR sweep; train a fixed arbitrary number of extra epochs with no early stopping; replace the frozen Model C point with the best extended checkpoint.

**Why rejected:** each alternative either overreacts to validation noise, retunes the experiment, fails to answer the saturation question efficiently, or contaminates the frozen A/B/C comparison.

**Presentation relevance:** 06A can demonstrate the distinction between **capacity-limited** and **training-duration-limited** performance and show why a controlled scaling study must separate follow-on optimization from its original causal comparison.

---

## D-096 — Hard resume/provenance gate and precommitted exploratory test policy

**Selected choice:** Before a single extension optimizer update is allowed, Notebook 06A must pass a fail-closed resume/provenance gate that proves the exact frozen Model C state, the continuity of AdamW optimizer state, the continuity of the canonical data order, and the availability of the canonical Notebook 05 evidence needed to construct the frozen-prefix/extension joined view. D-096 also precommits the post-training test and qualitative-generation policy so neither can be chosen after inspecting the extension validation curve.

### Gate 1 — canonical Notebook 05 artifacts must exist on `main`
The following Notebook 05 outputs are dependencies of 06A and must be committed in their canonical paths before training begins:
- `results/evaluation/evidence/validation_history_canonical.csv`
- `results/evaluation/evidence/validation_history_ingestion_audit.json`
- `results/evaluation/evidence/final_test_stream_audit.json`
- the small analysis CSV/JSON artifacts referenced by D-085/D-094, including `results/evaluation/analysis/final_evaluation_summary.json`
- the presentation figures emitted under `figures/evaluation/`

The gate must check that the canonical validation-history CSV is available from the repository and use that committed file as the immutable Model C prefix for 06A's joined-view artifact. It must not silently substitute a Drive copy or reconstruct the prefix from rounded summary values.

**Why:** D-085 made these files part of the repository-level output contract, and 06A depends on the canonical validation history as frozen evidence. A joined curve that reads its prefix from transient Colab/Drive state would weaken provenance and make reproduction dependent on private runtime storage.

### Gate 2 — exact checkpoint identity by re-evaluation
After loading the update-3,663 Model C checkpoint and before any optimizer step, run the complete D-072 validation procedure over the canonical 256,512 scored validation targets.

Required assertion:
- observed resumed validation loss must reproduce the frozen value **3.684501** to at least five decimal places under the canonical numerical environment; any small environment-dependent discrepancy must be recorded explicitly with the full-precision observed value and investigated before proceeding.

The notebook should record:
- expected frozen validation loss: `3.684501`
- observed resumed validation loss at update 3,663
- absolute difference
- pass/fail status under the chosen tolerance
- observed perplexity derived from the resumed loss

**Why:** Matching metadata can still point to the wrong or partially restored state. Reproducing the actual frozen validation objective is a direct behavioral identity check on the loaded model and validation pipeline.

### Gate 3 — optimizer-state continuity, not mere optimizer structure
The resumed AdamW state must be inspected parameter-by-parameter.

For every parameter with optimizer state:
- `state['step']` must equal **3,663**.
- `exp_avg` must exist, have the correct shape, and contain non-zero values.
- `exp_avg_sq` must exist, have the correct shape, and contain non-zero values.

The gate should summarize counts of:
- trainable parameters expected to carry optimizer state
- states checked
- step mismatches
- zero first-moment tensors
- zero second-moment tensors
- missing/malformed states

Any mismatch fails the gate. A weights-only resume, newly initialized AdamW state, or partially reset moment history is not an acceptable continuation.

**Why:** Reset optimizer moments at the frozen weights would create a cold AdamW restart at `2e-4`, changing the experiment while appearing superficially resumable.

### Gate 4 — scaler, counters, model/config, and provenance
The gate must additionally verify:
- Model C architecture/config identity and exact parameter count **33,497,600**.
- global optimizer update counter **3,663**.
- completed epoch counter **3** and correct next-epoch position.
- FP16 GradScaler state when running under the canonical T4/FP16 path.
- persisted RNG state(s) required by the production wrapper.
- tokenizer SHA-256 `6ec601a267cec7c843df47927f53c4dd108c85a1d059318aeec4442c7274604f`.
- corpus token-stream SHA-256 `4101d5b18c38558a58110f54a161763186ab5318111366486ebbfa0a3fe584fa`.
- corpus manifest SHA-256 `4a00196b39311a6c2e2790780e8fc43316f24a014d3d3649028b10a671f8d3fe`.
- official validation reconstruction/packing remains the D-072 256,512-target deterministic no-shuffle procedure.

### Gate 5 — epoch-4 data-order continuity
The extension's first additional epoch is **epoch 4**. Under the D-073 seed convention, the training-example permutation for that epoch must be generated using **`seed + 4`**, i.e. seed **46** when the base seed is 42.

Before training, the notebook must generate the epoch-4 permutation without consuming optimizer updates and record at least the first 10 example indices in a machine-readable gate artifact. The same indices must be reproducible on rerun.

**Why:** Exact checkpoint continuation is insufficient if the data-order convention changes at the boundary. Recording the first indices makes the extension shuffle auditable and proves continuity with the original engine's epoch-indexed seed semantics.

### Gate artifact
Persist a machine-readable gate record under a separate 06A namespace, for example:
- `results/extended_training/model_c/d096_resume_gate.json`

The gate record should include all expected/observed hashes, checkpoint counters, validation reproduction result, optimizer-state audit counts, scaler/RNG presence, epoch-4 shuffle seed, first recorded shuffle indices, and a single top-level `gate_passed` boolean.

Training code must assert `gate_passed is True` before exposing or calling the extension optimizer loop.

### Precommitted one-time exploratory test policy
After training is complete and the **best 06A validation-selected checkpoint is frozen**, evaluate that checkpoint **exactly once** on the same official test procedure used under D-092.

Rules:
- the checkpoint is selected exclusively by 06A validation loss; test results cannot alter checkpoint selection, early stopping, LR, or any training choice.
- the test result is labeled **exploratory Model C extended-training evidence**.
- it is stored in the 06A artifact namespace.
- it must never be inserted into or replace the frozen Notebook 05 A/B/C test table.
- compare it descriptively with frozen Model C's Notebook 05 test result only to ask whether the validation improvement generalized.
- no additional test-driven retuning or repeated test scoring is permitted.

**Why:** The extension creates a genuinely new checkpoint whose generalization should be checked, but that check must be precommitted and one-time so the test split does not become a tuning instrument.

### Precommitted qualitative-generation follow-up
After the best 06A validation checkpoint is frozen, run the **same D-091 validation-derived prompts and decoding settings** for extended Model C: temperature `0.8`, top-p `0.9`, 96 new tokens, and the identical per-prompt seeds. The purpose is a controlled before/after Model C comparison, not a new prompt search.

Keep visible the existing Notebook 05 result that Model B often appeared more topically stable than Model C despite Model C's better likelihood metrics. If extended Model C improves noticeably under the fixed D-091 probe while validation/test loss also improve, that is evidence consistent with undertraining having contributed to the original qualitative drift; it is not proof that capacity itself caused or resolved the generation behavior.

### Failure policy
D-096 is fail-closed. If any required canonical artifact is missing, checkpoint validation fails to reproduce the frozen objective within the accepted tolerance, optimizer moments/steps are inconsistent, required scaler/RNG/config state is absent or incompatible, or epoch-4 data order does not match the frozen convention, **no extension training may begin**.

**Alternatives considered:** trust checkpoint filenames/metadata; verify optimizer keys but not moment contents; regenerate Notebook 05 evidence from summaries; choose whether to test after seeing the validation curve; use new qualitative prompts for extended C.

**Why rejected:** each alternative weakens provenance, changes optimizer/data continuity, creates post-hoc evaluation discretion, or makes qualitative before/after comparison uninterpretable.

**Presentation relevance:** D-096 demonstrates what a true continuation experiment requires: behavioral checkpoint identity, optimizer-state continuity, deterministic data continuity, immutable evidence provenance, and precommitted downstream evaluation.

## Next decision ID

The next globally unique decision ID is **D-097**.
