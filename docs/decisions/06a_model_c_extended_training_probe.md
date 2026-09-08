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

---

## D-097 — Deterministic Notebook 05 artifact recovery and content-hash manifest

**Selected choice:** Gate 1 of D-096 is refined from a presence check into a content-addressed provenance check. First attempt to recover the exact Notebook 05 artifacts generated during the completed run. If those exact files are unavailable, recover the canonical artifact set by re-executing Notebook 05 from its frozen inputs and unchanged code/protocol, then verify the regenerated quantitative evidence against the already-frozen Notebook 05 results before any file is committed or accepted by 06A.

### Recovery order
1. Prefer the exact original artifact bytes if they can be located in persistent storage.
2. Do not reconstruct CSV/JSON evidence from rounded values copied out of the notebook, project context, or decision register.
3. If the original ephemeral Colab outputs are unavailable, re-execute the committed Notebook 05 under the same frozen inputs and protocol.

The connected Drive search did not surface the project folder, Notebook 05 evaluation artifacts, or Model C files by name, so the original generated evaluation files are treated as unavailable through the current recovery interface unless a direct persistent location is later supplied.

### Quantitative recovery assertions
Before recovered/re-executed artifacts may be committed, the recovery run must assert at minimum:
- validation loss A/B/C = **3.972054 / 3.776427 / 3.684501**;
- final test loss A/B/C = **3.955290 / 3.772079 / 3.680554**;
- final test scored targets/model = **293,376**;
- validation scored targets/event = **256,512**;
- the frozen A→B→C predictive-quality ranking remains intact;
- the generated final evaluation summary agrees with D-094 on the frozen A/B/C conclusions.

Any disagreement blocks artifact recovery and requires investigation. The purpose of re-execution is to reproduce the already-frozen evidence, not to update or reinterpret Notebook 05.

### D-092 test re-measurement disclosure
If Notebook 05 must be re-executed, the official test split will necessarily be scored again under the identical D-092 procedure. This is recorded as a **deterministic re-measurement for artifact recovery**, not a new model-selection or tuning pass.

No training, checkpoint selection, hyperparameter choice, scaling conclusion, or downstream Notebook 05 decision may change as a result of this re-measurement. The recovered test outputs are accepted only if they reproduce the previously frozen test evidence. This disclosure preserves the historical truth of D-092's original one-time final evaluation while making clear that a later provenance-recovery run re-scored the same fixed checkpoints.

### Qualitative outputs
Quantitative CSV/JSON evidence and deterministic figures must reproduce from the frozen inputs. Fixed-seed generation outputs should also be regenerated when Notebook 05 is re-executed, but small GPU-level nondeterminism in sampled text is not allowed to alter D-091/D-093 conclusions. Any regenerated generation artifact must be labeled as recovered by re-execution and must preserve the original prompts, decoding settings, and seeds.

### Content-addressed artifact manifest
After recovery, compute SHA-256 for every canonical Notebook 05 artifact committed under:
- `results/evaluation/evidence/`;
- `results/evaluation/analysis/`;
- `figures/evaluation/`.

Persist the path, byte size, and SHA-256 of each file in a machine-readable manifest, for example:
- `results/evaluation/artifact_manifest_sha256.json`.

D-096 Gate 1 is thereby refined: 06A must verify both artifact presence **and** SHA-256 agreement with the committed recovery manifest before using `validation_history_canonical.csv` as its immutable frozen prefix.

### Recovery provenance record
The recovery commit must document:
- recovery date;
- whether each artifact was recovered byte-for-byte or regenerated by deterministic re-execution;
- the Git commit of Notebook 05 used for re-execution;
- the frozen checkpoint/history sources consumed;
- the frozen-value assertion results;
- whether the test split was re-scored for recovery;
- the artifact SHA-256 manifest.

### Why
The missing Notebook 05 outputs are not merely a documentation inconvenience: D-085 defined them as stable repository artifacts, D-094 cites the final summary, and 06A requires the canonical validation history as immutable prefix evidence. Content hashes convert that dependency from a filename convention into a reproducible evidence contract.

**Alternatives considered:** continue searching indefinitely for ephemeral Colab files; synthesize the outputs from rounded register values; allow 06A to read private Drive copies; require presence without hashes.

**Why rejected:** those alternatives either cannot establish provenance, risk manufacturing evidence, make reproduction dependent on private state, or permit silent content drift under unchanged filenames.

**Presentation relevance:** the recovery is a useful governance/reproducibility example: a fail-closed gate exposed an actual evidence-retention gap, and the correction strengthens the experiment by making canonical outputs content-addressed and independently auditable.

---

## D-098 — Notebook 06A launch and exact-resume execution contract

Notebook 06A is an exploratory continuation of frozen Model C and remains separate from the controlled Notebook 05 A/B/C comparison.

### Selected choice

Authorize the Model C extended-training probe to begin only through the resumable 06A runner after the following conditions are all true:

- D-096 is formally closed with `gate_passed=true` in the persisted canonical gate artifact.
- The execution-locked 06A preflight reproduces the frozen update-3,663 boundary and validates the D-095 extension contract.
- The resumable runner preflight reports `Mode: FRESH EXTENSION START`, global update 3,663, epoch 4 position 0, and zero persisted extension updates before first launch.
- Training begins only when the resumable runner is invoked with the explicit `--execute` flag.

### Immutable launch boundary

The first extension optimizer update is global update **3,664**. Before that update, the frozen parent state is:

- Model: C
- parameters: 33,497,600
- completed epochs: 3
- global update: 3,663
- frozen validation loss reference: 3.684501
- full-precision reproduced validation loss: 3.6845006885642775
- extension learning rate: 2e-4 constant
- first extension epoch: 4
- epoch-4 effective shuffle seed: 46
- first 10 epoch-4 example indices: `[23330, 14720, 12892, 24465, 36182, 35545, 35235, 2313, 29923, 36592]`

No Notebook 04 or Notebook 05 checkpoint, metric, history, test result, or figure is overwritten by 06A.

### Exact 06A resume semantics

06A persistence lives only under the separate exploratory namespace:

`/content/drive/MyDrive/foundation-model-from-scratch/production/extended_training/model_c/`

The resumable runner may either:

1. start from the immutable frozen production checkpoint at update 3,663 when no 06A latest checkpoint exists; or
2. resume from `model_c_extension_latest.pt` when an interrupted 06A run has already persisted extension state.

A 06A resume must restore, at minimum:

- model state;
- AdamW optimizer state and moment history;
- GradScaler state;
- CPU/CUDA RNG state;
- global update;
- extension update count;
- extension epoch number;
- exact number of updates already completed in the current epoch;
- extension target-exposure count;
- extension training and validation histories;
- best validation loss/update;
- `min_delta` material-improvement reference;
- early-stopping patience count;
- accumulated elapsed time.

The runner must reconstruct the canonical epoch loader and deterministically skip exactly the already-persisted optimizer-update groups before processing the next group. This preserves the same epoch-indexed shuffle convention used by the frozen training engine.

### Frozen extension controls

D-095 remains unchanged:

- constant LR: 2e-4;
- validation every 200 extension updates plus each additional epoch end;
- `min_delta = 0.001`;
- patience = 6 validation events;
- maximum 10 additional epochs;
- maximum global update 15,873;
- official test split remains inaccessible during training;
- the eventual exploratory test is performed once only after validation selection is frozen;
- controlled qualitative generation reuses the precommitted D-091 protocol.

### Persistence and interruption policy

The 06A `latest` checkpoint is persisted at each validation event. A Colab interruption may therefore lose at most the uncheckpointed work since the most recent validation event. On restart, the experiment resumes from the most recent persisted 06A validation checkpoint rather than reconstructing intermediate optimizer updates from logs.

This is intentional: exact state can only be guaranteed from an actually persisted checkpoint. The experiment does not infer or replay unseen optimizer steps after a runtime loss.

### Provenance closure carried forward

Notebook 05 artifact recovery is closed by the canonical 23-artifact SHA-256 manifest on `main`. The deterministic recovery rerun originated from Notebook 05 source commit `e6bdd5ba27068659454873d28e6a2451810a815b`; the recovered artifact set was later persisted through PR #35 without changing any D-084–D-094 experimental conclusion. D-096 Gate 1 verifies those committed bytes before 06A continuation.

### Why

The research question is whether Model C was still materially training-duration constrained at the frozen three-epoch boundary. That question is only interpretable if continuation preserves optimizer, data-order, numerical, validation, and provenance state while changing only the allowed training duration. The explicit launch flag prevents accidental continuation, and the 06A-specific resumable checkpoint makes the experiment robust to Colab interruptions without contaminating frozen evidence.

### Presentation relevance

This provides a useful governance/reproducibility example: the continuation was not allowed to begin merely because weights existed. It required behavioral identity, optimizer-state continuity, deterministic data continuity, content-addressed evidence, an explicit execution boundary, and a resumable exploratory namespace.

### Status at decision time

- D-096: PASS / formally closed.
- 06A execution-locked preflight: PASS.
- 06A resumable preflight: PASS.
- mode before first launch: FRESH EXTENSION START.
- global update before first launch: 3,663.
- optimizer updates executed by 06A at decision time: 0.
- training has not yet begun.

The next optimizer update, if explicitly launched, is **3,664**.

---

## D-099 — Notebook 06A validation-selected checkpoint and stop classification

Notebook 06A extended training is complete. This decision freezes the validation-selected exploratory Model C checkpoint **before any 06A official-test evaluation is performed**.

### Selected checkpoint

Freeze the best 06A checkpoint at:

- model: C
- global update: **12,210**
- extension update: **8,547**
- validation loss: **3.599946362767629**
- parent frozen Model C validation loss: **3.6845006885642775**
- constant extension learning rate: **2e-4**

This checkpoint was selected exclusively from the precommitted validation-loss criterion. No 06A official-test result has been observed or used in selection.

### Training completion

The resumable 06A runner stopped at:

- final global update: **13,263**
- extension updates: **9,600**
- validation events: **55**
- patience count at stop: **6**
- stop reason: `early_stopping_patience_exhausted`
- official test split used during training: **false**

The final six validation events after the selected best checkpoint were:

- 12,263 → 3.607138
- 12,463 → 3.614607
- 12,663 → 3.611002
- 12,863 → 3.610469
- 13,063 → 3.607164
- 13,263 → 3.600369

The last point approached the best again but did not improve by the precommitted `min_delta = 0.001`, so patience reached six and training stopped exactly as specified by D-095.

### Outcome classification

Classify the observed extension as **continued improvement followed by saturation / a noisy validation plateau**, not sustained overfitting.

Why:

- Model C improved materially beyond the frozen three-epoch boundary, from 3.6845006885642775 to 3.599946362767629 validation loss.
- Meaningful new best values continued to appear through global update 12,210.
- After the best point, validation fluctuated above the best but did not show a sustained monotonic degradation; the final event recovered to 3.600369.
- Therefore the evidence supports that the original three-epoch Model C was training-duration constrained, while the later extension eventually reached the D-095 saturation criterion.

This classification is bounded to the observed validation trajectory and does not imply a universal optimization limit for Model C.

### FP16 overflow recovery provenance

During continuation, the T4/FP16 GradScaler reached a scale of 1,048,576 and produced non-finite gradients on three logical updates. The saved model and optimizer tensors were independently checked and contained zero non-finite values. The 06A runner then used deterministic same-update replay with loss-scale backoff; failed attempts did not advance optimizer counters or data position.

Recorded extension totals:

- FP16 overflow retries: **3**
- final FP16 loss scale: **524,288**

This numerical recovery changed neither the constant learning rate nor the logical training/update count.

### Test boundary

D-096's precommitted one-time exploratory test policy now becomes eligible.

The next step may evaluate **only the frozen update-12,210 06A best checkpoint**, exactly once, using the same official test procedure as D-092. The result must remain in the 06A exploratory namespace and must not replace the frozen Notebook 05 A/B/C test table.

No further Model C training is authorized by this decision.

### Presentation relevance

This result separates two effects that the original fixed-budget scaling experiment could not distinguish by itself:

1. Model C's better three-epoch likelihood was not yet its best attainable result under continued optimization; it was still materially duration constrained.
2. Additional training eventually produced a validation plateau, showing why capacity scaling and training-duration scaling must be analyzed separately.

The frozen Notebook 05 comparison remains unchanged because only Model C received the additional optimization budget in 06A.

---

## D-100 — Notebook 06A final synthesis and project experimental closure

Notebook 06A is complete. This decision freezes the final exploratory conclusions after validation selection, the precommitted one-time official-test evaluation, the fixed D-091 generation probe, and final evidence packaging.

### Frozen quantitative outcome

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

### Qualitative follow-up

The fixed D-091 validation-prompt generation probe reused the exact three frozen prompts, seeds 43/44/45, temperature 0.8, top-p 0.9, and 96 new tokens.

Observed interpretation:

- Prompt 1 showed better topical continuity around city/building/industrial-development material than the original Model C sample.
- Prompt 2 showed the clearest improvement: substantially less repetitive ridge-language and better topical continuity in geology/topography.
- Prompt 3 remained unreliable and fabricated biographical/achievement details.

The conservative conclusion is: **undertraining contributed to some of the original qualitative drift, but additional training did not make the small model reliably factual.** Likelihood improvement and qualitative stability are related but not interchangeable objectives.

### Numerical-runtime incident closure

During 06A continuation, the T4/FP16 GradScaler reached a loss scale of 1,048,576 and produced non-finite gradients on three logical updates. Diagnostic inspection found zero non-finite model tensors and zero non-finite optimizer-state tensors. The recovery runner replayed the same logical update with the same data order and restored RNG state while backing off only the FP16 loss scale.

Final numerical-recovery facts:

- overflow retries: **3**
- final loss scale: **524,288**
- failed overflow attempts did not advance optimizer counters or data position
- LR, batch semantics, model architecture, AdamW hyperparameters, and official-test policy were unchanged

This is classified as runtime numerical-stability recovery, not hyperparameter retuning.

### Evidence closure

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

### Final scientific interpretation

The controlled Notebook 05 experiment and the 06A exploratory continuation answer different questions and must remain separate:

1. **Capacity scaling under equal budget:** A→B→C improved predictive quality monotonically, but B→C delivered weaker marginal efficiency per added parameter, training minute, and GiB.
2. **Training-duration sensitivity of the largest model:** frozen Model C had not exhausted useful learning at three epochs; additional optimization improved both validation and held-out test likelihood before reaching a noisy saturation plateau.
3. **Likelihood vs generated behavior:** more training improved some topical stability but did not eliminate hallucination or guarantee a monotonic human-visible quality ranking.

No 06A checkpoint, metric, compute cost, or generated sample replaces any frozen Notebook 05 A/B/C result.

### Project status

The experimental portion of **Building a Foundation Model from Scratch** is complete. No additional model training, test scoring, LR search, checkpoint selection, or scope expansion is required for the core project.

Remaining work is presentation/report assembly from the frozen repository evidence. Any future fine-tuning, quantization, architectural research, or additional scaling run is a new project or explicitly separate follow-on experiment.

### Presentation relevance

The final narrative should emphasize four lessons:

- larger models improved likelihood under equal data/training controls, but marginal efficiency declined;
- a fixed training budget can confound capacity with duration, which 06A exposed cleanly without rewriting the original experiment;
- disciplined test sealing and fail-closed provenance gates materially improved the credibility of the project;
- better perplexity and more training do not automatically produce factual or uniformly better generated language.

## Next decision ID

The next globally unique decision ID is **D-101**.
