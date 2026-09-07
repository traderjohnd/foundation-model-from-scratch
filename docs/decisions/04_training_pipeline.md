# Decision Register — 04 Training Pipeline

Notebook 04 converts the frozen corpus/model contracts into a controlled, explicit PyTorch training experiment and records the three production runs.

Earlier planning decisions retain their original IDs. Notebook-04 execution decisions were renumbered canonically to **D-065 through D-083** so decision IDs remain globally unique. The old Notebook-04-local IDs are preserved in Git history; the mapping is documented at the end of this file.

---

## D-019 — Training objective
**Selected choice:** autoregressive causal language modeling / next-token prediction.  
**Why:** Standard objective for GPT-style decoder-only models and directly aligned with the scaling experiment.  
**Alternatives considered:** masked LM, encoder-decoder denoising, supervised task objectives.  
**Presentation relevance:** causal masking, autoregression, self-supervised learning.

## D-020 — Loss function
**Selected choice:** cross-entropy loss over the next token.  
**Why:** Directly penalizes insufficient probability assigned to the true next token.  
**Presentation relevance:** probability distribution, CE, backpropagation.

## D-022 — Optimizer
**Initial selected choice:** AdamW.  
**Why:** Strong standard Transformer optimizer with adaptive updates and decoupled weight decay.  
**Alternatives considered:** SGD, Adam, Adafactor, Lion, newer optimizers.  
**Historical outcome:** exact parameter-group/beta/epsilon policy refined in D-069.

## D-023 — Learning-rate experiment
**Initial selected choice:** controlled Model A probe at `1e-4`, `3e-4`, `1e-3`; provisional expectation ~`3e-4`.  
**Why:** Learning rate should be selected empirically rather than copied from another model.  
**Alternatives considered:** one untested LR; exhaustive search.  
**Historical outcome:** the provisional expectation was superseded by D-078 after controlled evidence selected `2e-3`.

## D-024 — Learning-rate schedule
**Initial selected choice:** warmup followed by cosine decay.  
**Why:** Avoid aggressive early updates from random initialization and reduce LR for later refinement.  
**Historical outcome:** exact update-clock schedule finalized in D-070.

## D-025 — Effective batch size
**Initial selected choice:** **16,384 prediction targets per optimizer update**.  
**Why:** Defines optimization behavior independently of physical GPU micro-batch capacity.  
**Historical outcome:** exact logical/micro-batch behavior refined and gradient-equivalence tested in D-067.

## D-026 — Gradient accumulation
**Initial selected choice:** use accumulation as needed to preserve the effective batch.  
**Why:** Allows larger models to use smaller physical batches without changing the optimization unit.  
**Historical outcome:** D-079 showed all A/B/C fit physical batch 32 on T4, so full updates need no accumulation; the generic engine still supports it.

## D-027 — Training duration
**Selected choice:** maximum **3 epochs**.  
**Why:** One pass may be insufficient for informative learning curves, while many repeated passes increase overfitting/memorization risk.  
**Alternatives considered:** one epoch; many epochs; unrelated fixed-step budgets.  
**Presentation relevance:** generalization/overfitting tradeoff.

## D-028 — Best-checkpoint policy
**Selected choice:** retain the checkpoint with best validation performance, not automatically the final epoch.  
**Why:** Training loss can continue improving after validation stops improving.  
**Historical implementation:** D-080.

## D-031 — Training precision
**Initial selected choice:** BF16 mixed precision when natively supported; FP16 fallback; FP32 where needed.  
**Why:** Reduces memory/increases throughput while retaining practical numerical stability.  
**Historical outcome:** hardware-aware native-BF16 policy finalized in D-068; Tesla T4 resolves to FP16 + GradScaler.

## D-033 — Validation cadence
**Selected choice:** every 200 optimizer updates plus end of each epoch.  
**Why:** Enough resolution for learning curves and model-selection evidence without excessive validation overhead.  
**Historical implementation:** 21 validation events over 3,663 updates/model.

## D-034 — Checkpoint save rule
**Selected choice:** save on a new best validation loss.  
**Why:** Preserves strongest held-out generalization without using test data.  
**Historical implementation:** D-080.

## D-035 — Random seed
**Selected choice:** seed **42**.  
**Why:** Controls initialization, epoch shuffle, dropout and related stochasticity for a controlled experiment.  
**Presentation relevance:** reproducibility vs statistical significance.

## D-038 — Gradient clipping
**Selected choice:** global gradient norm clip **1.0**.  
**Why:** Safety rail against destabilizing updates, especially early in mixed-precision training.

---

## D-065 — Deterministic 512-token causal packing
**Selected choice:** context/stride 512; `x=tokens[start:start+512]`, `y=tokens[start+1:start+513]`; admit complete fixed-length examples only; no padding, wraparound, duplicated tail, or variable-length final example.  
**Why:** Provides simple fixed-shape supervision while ensuring every admitted target is the actual next corpus token and appears once.  
**Supersedes/resolves:** D-058.  
**Evidence:** 39,062 causal examples; 19,999,744 scored targets/epoch; 19,999,745 distinct participating corpus positions; **255** potential next-token targets intentionally unused.  
**Presentation relevance:** causal shift means the arithmetic 256-token remainder does not equal dropped targets.

## D-066 — Partial final effective batch
**Selected choice:** flush the final 22 sequences as a smaller optimizer update.  
**Why:** Avoids dropping, duplicating, or carrying examples across epochs while ensuring every admitted target is seen exactly once/epoch.  
**Evidence:** 1,220 full updates + one 22-sequence/11,264-target tail = **1,221 updates/epoch**.

## D-067 — Micro-batching and gradient accumulation
**Selected choice:** keep logical effective batch fixed at 16,384 targets; physical micro-batch is hardware-dependent; normalize accumulated gradients by actual logical-update target count. Physical batches may cross logical boundaries.  
**Why:** Makes gradients invariant to physical batching and handles the smaller epoch-tail correctly.  
**Supersedes/refines:** D-025/D-026 implementation details.  
**Evidence:** exact-gradient-equivalence tests passed for multiple micro-batch sizes, including non-divisors of 32.

## D-068 — Mixed-precision runtime policy
**Selected choice:** CUDA + **native** BF16 → BF16 autocast/no scaler; CUDA without native BF16 → FP16 autocast + GradScaler; CPU → FP32 smoke only; master/model parameters FP32.  
**Why:** Native capability must be distinguished from emulation; precision policy should be hardware truthful and reproducible.  
**Supersedes/refines:** D-031.  
**Evidence:** Tesla T4 compute capability 7.5 has no native BF16 and therefore resolves to FP16 + GradScaler.

## D-069 — AdamW parameter groups
**Selected choice:** AdamW betas `(0.9,0.95)`, epsilon `1e-8`; weight decay .10 for `ndim>=2`, zero decay for RMSNorm scales; tied parameters deduplicated by identity.  
**Why:** Makes regularization explicit, avoids decaying norm scales, and prevents duplicate optimization of tied weights.  
**Supersedes/refines:** D-022.

## D-070 — Exact learning-rate schedule
**Selected choice:** optimizer-update clock; 3,663 total updates; 5% warmup = 183 updates; 3,480 cosine-decay updates; minimum/final LR = 10% of peak.  
**Why:** Converts the broad warmup+cosine design into an exact, micro-batch-independent schedule.  
**Supersedes/refines:** D-024.

## D-071 — Initial Model A LR-probe protocol
**Selected choice:** candidates `1e-4`, `3e-4`, `1e-3`; 400 optimizer updates/candidate; validation @200/@400; 20-update warmup; same initialization, data order, RNG restart, optimizer, precision, logical batch, clipping, and validation. Selection rule: lowest val loss @400, then @200, then lower LR.  
**Why:** Makes LR comparison causal and reproducible rather than confounded by initialization/data differences.  
**Supersedes/refines:** D-023 experiment design.

## D-072 — Validation protocol
**Selected choice:** official validation only; same pinned normalization/reconstruction/tokenizer; original article order; one boundary after every complete validation article; no shuffle; evaluate full summed CE divided by exact target count.  
**Why:** Every model must be scored on exactly the same held-out sequence and denominator.  
**Evidence:** 60 articles; 256,579 ordinary tokens + 60 boundaries = 256,639-token stream; 501 examples; **256,512 scored targets**. Batch-size invariance passed. Official test content not used.

## D-073 — Single-run training engine
**Selected choice:** explicit PyTorch engine supporting logical effective-update boundaries independent of DataLoader batch boundaries, mixed precision, AdamW, LR scheduling, target-normalized accumulation, clipping, validation hooks, and deterministic `seed+epoch` shuffling.  
**Why:** Centralizes the exact experiment semantics without relying on HF Trainer or fragile notebook-cell state.  
**Evidence:** synthetic and real-model smokes passed, including physical micro-batches crossing logical boundaries.

## D-074 — Canonical model-module integration
**Selected choice:** use the verified Notebook 03 architecture from `src/model.py`.  
**Why:** Reuse one canonical implementation rather than duplicate architecture code in training notebooks.  
**Evidence:** exact parameters, tying, and real Model A forward/backward smoke passed. Original merge was PR #22.

## D-075 — Canonical corpus reconstruction gate
**Selected choice:** fresh training runtimes rebuild train/validation streams from canonical provenance and refuse to proceed when counts/fingerprints differ. Article-permutation fingerprint uses little-endian signed 32-bit indices (`<i4`).  
**Why:** Prevents silent data drift across Colab sessions.  
**Evidence:** reproduced 20M train tokens, canonical stream SHA-256, 4,604 records, 19,995,397 text tokens, 4,603 boundaries, 39,062 train examples, 256,639 validation tokens, 501 val examples, 256,512 val targets; test unused.

## D-076 — Real Model A optimizer smoke
**Selected choice:** before LR evidence, run 3 real optimizer updates on canonical Model A/data plus one full validation pass.  
**Why:** Verifies the real optimizer/scaler/grad/tied-weight path before spending compute on experiments.  
**Evidence:** PASS on T4 FP16+GradScaler, physical batch32; validation loss after update3 9.586980; weights changed; gradients cleared; tying persisted.

## D-077 — Controlled original LR-probe execution
**Observed evidence:**
- `1e-4`: val@200 7.067461; val@400 6.799811; PPL 897.68; clip 28.7%
- `3e-4`: 6.394534; 6.163024; PPL 474.86; clip 10.5%
- `1e-3`: 5.988447; **5.659233**; PPL 286.93; clip 6.5%

**Why this created a new decision:** the best candidate was the upper boundary, so selecting it would not demonstrate that the useful region had been bracketed.  
**Outcome:** run a narrow upper-bound refinement instead of locking `1e-3`.

## D-078 — LR bracket refinement and production peak LR
**Selected choice:** test `2e-3` and `3e-3`; lock production peak LR at **`2e-3`**.  
**Why:** `2e-3` achieved the lowest val loss at update400 and `3e-3` was worse, providing upper-side bracketing.  
**Evidence:** `2e-3`: val@200 5.884712, val@400 **5.488027**, PPL 241.78, clip 5.0%; `3e-3`: 6.159744, 5.741478, PPL 311.52, clip 4.5%.  
**Supersedes:** D-023 provisional ~3e-4 expectation and completes D-071 selection process.  
**Presentation relevance:** evidence-based hyperparameter selection instead of copied defaults.

## D-079 — Production accelerator micro-batch preflight
**Selected choice/evidence:** on controlled Tesla T4, physical micro-batch **32 sequences** for all A/B/C. Peak allocations: A 4.88 GiB; B 7.16 GiB; C 10.37 GiB.  
**Why:** Measure hardware fit before launching expensive production runs while keeping logical batch fixed.  
**Historical consequence:** gradient accumulation is unnecessary for full updates on T4; the 22-sequence tail remains smaller by D-066.  
**Refines:** D-026 conditional accumulation plan.

## D-080 — Production wrapper, persistence, and resume contract
**Selected choice:** preserve all frozen controls; validate every 200 updates + epoch end; retain best/latest; persist JSON history/summary; record resource metrics; never access test. Persist artifacts to Google Drive. For B/C use checkpoint format v2 containing model/optimizer/GradScaler/RNG state for exact resume from validation boundaries.  
**Why:** Colab runtime resets are an operational failure mode; experimental evidence must survive them without silently changing the controlled run.  
**Evidence:** 21 validation events, 59,999,232 target exposures/model, checkpoint write/read and JSON write/read smokes passed.  
**Implements/refines:** D-028/D-034 checkpoint policy.

## D-081 — Model A production training
**Selected choice:** train A under frozen protocol and stop at the fixed 3-epoch budget even though the final validation point is best.  
**Why:** A/B/C must receive the same training budget for the scaling comparison.  
**Evidence:** 7,407,872 params; 3,663 updates; 59,999,232 exposures; best val loss **3.972054**; PPL **53.09**; best update 3,663; wall 11.78 min; peak 4.94 GiB; test NO; persistent artifacts verified.

## D-082 — Model B production training
**Selected choice:** same frozen protocol as A; do not retune LR by model size.  
**Why:** Model capacity is the experimental variable; retuning optimization per model would weaken the controlled comparison.  
**Evidence:** 16,913,280 params; 3,663 updates; 59,999,232 exposures; best val loss **3.776427**; PPL **43.66**; best update 3,663; wall 23.55 min; peak 7.29 GiB; test NO; persistent artifacts verified.

## D-083 — Model C production training and Notebook 04 freeze
**Selected choice:** same frozen protocol as A/B; after completion, freeze training and move interpretation to Notebook 05.  
**Why:** Comparative analysis should not feed back into post-hoc retuning after seeing production results.  
**Evidence:** 33,497,600 params; 3,663 updates; 59,999,232 exposures; best val loss **3.684501**; PPL **39.83**; best update 3,663; wall 42.22 min; peak 10.62 GiB; test NO; persistent artifacts verified.  
**Preliminary observation only:** validation improved with scale, but incremental loss gains shrank while time/memory rose; formal interpretation belongs to Notebook 05.

---

## Legacy Notebook-04 ID mapping

The temporary Notebook-04 continuation file restarted IDs at D-058, creating collisions with the already-existing project register and Notebook 03 addendum. Canonical IDs are now:

| Legacy NB04 ID | Canonical ID |
|---|---|
| D-058 | D-065 |
| D-059 | D-066 |
| D-060 | D-067 |
| D-061 | D-068 |
| D-062 | D-069 |
| D-063 | D-070 |
| D-064 | D-071 |
| D-065 | D-072 |
| D-066 | D-073 |
| D-067 | D-074 |
| D-068 | D-075 |
| D-069 | D-076 |
| D-070 | D-077 |
| D-071 | D-078 |
| D-072 | D-079 |
| D-073 | D-080 |
| D-074 | D-081 |
| D-075 | D-082 |
| D-076 | D-083 |

Git history preserves the legacy files and their original labels. From this normalization forward, only the canonical IDs above should be cited.
