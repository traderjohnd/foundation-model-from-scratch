# D-099 — Notebook 06A validation-selected checkpoint and stop classification

Notebook 06A extended training is complete. This decision freezes the validation-selected exploratory Model C checkpoint **before any 06A official-test evaluation is performed**.

## Selected checkpoint

Freeze the best 06A checkpoint at:

- model: C
- global update: **12,210**
- extension update: **8,547**
- validation loss: **3.599946362767629**
- parent frozen Model C validation loss: **3.6845006885642775**
- constant extension learning rate: **2e-4**

This checkpoint was selected exclusively from the precommitted validation-loss criterion. No 06A official-test result has been observed or used in selection.

## Training completion

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

## Outcome classification

Classify the observed extension as **continued improvement followed by saturation / a noisy validation plateau**, not sustained overfitting.

Why:

- Model C improved materially beyond the frozen three-epoch boundary, from 3.6845006885642775 to 3.599946362767629 validation loss.
- Meaningful new best values continued to appear through global update 12,210.
- After the best point, validation fluctuated above the best but did not show a sustained monotonic degradation; the final event recovered to 3.600369.
- Therefore the evidence supports that the original three-epoch Model C was training-duration constrained, while the later extension eventually reached the D-095 saturation criterion.

This classification is bounded to the observed validation trajectory and does not imply a universal optimization limit for Model C.

## FP16 overflow recovery provenance

During continuation, the T4/FP16 GradScaler reached a scale of 1,048,576 and produced non-finite gradients on three logical updates. The saved model and optimizer tensors were independently checked and contained zero non-finite values. The 06A runner then used deterministic same-update replay with loss-scale backoff; failed attempts did not advance optimizer counters or data position.

Recorded extension totals:

- FP16 overflow retries: **3**
- final FP16 loss scale: **524,288**

This numerical recovery changed neither the constant learning rate nor the logical training/update count.

## Test boundary

D-096's precommitted one-time exploratory test policy now becomes eligible.

The next step may evaluate **only the frozen update-12,210 06A best checkpoint**, exactly once, using the same official test procedure as D-092. The result must remain in the 06A exploratory namespace and must not replace the frozen Notebook 05 A/B/C test table.

No further Model C training is authorized by this decision.

## Presentation relevance

This result separates two effects that the original fixed-budget scaling experiment could not distinguish by itself:

1. Model C's better three-epoch likelihood was not yet its best attainable result under continued optimization; it was still materially duration constrained.
2. Additional training eventually produced a validation plateau, showing why capacity scaling and training-duration scaling must be analyzed separately.

The frozen Notebook 05 comparison remains unchanged because only Model C received the additional optimization budget in 06A.
