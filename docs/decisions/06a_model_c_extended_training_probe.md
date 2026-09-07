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

## Next decision ID

The next globally unique decision ID is **D-096**.
