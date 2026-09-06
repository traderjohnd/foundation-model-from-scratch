# Notebook 04 — Training Pipeline Decision Continuation

This file records Notebook 04 decisions that extend the canonical `docs/DECISION_REGISTER.md`. It exists so the training-pipeline evidence is preserved without rewriting earlier decisions. `docs/PROJECT_CONTEXT.md` points to this continuation as part of the current project record.

---

## D-058 — 512-token causal packing policy

**Selected choice**
- context length: 512
- stride: 512
- example `k`: `x=tokens[k*512:k*512+512]`, `y=tokens[k*512+1:k*512+513]`
- complete 512-input / 512-target examples only
- no padding, wraparound, duplicated tail, or variable-length final example

**Evidence**
- corpus tokens: 20,000,000
- full causal examples: 39,062
- scored targets/epoch: 19,999,744
- distinct corpus positions used: 19,999,745
- unused tail positions/targets: 255

**Status**: locked and validated.

---

## D-059 — Partial final effective batch

**Selected choice**
Flush the final 22 sequences as one smaller optimizer update rather than dropping, duplicating, or carrying them across epochs.

**Evidence**
- 1,220 full updates × 32 sequences
- final partial update: 22 sequences = 11,264 targets
- total optimizer updates/epoch: 1,221
- all 19,999,744 scored targets seen exactly once per epoch

**Status**: locked and validated.

---

## D-060 — Micro-batching and gradient accumulation

**Selected choice**
Keep the logical effective batch fixed at 16,384 targets/update while treating physical micro-batch size as a hardware parameter. Normalize accumulated gradients by the actual target count of the logical update.

**Evidence**
Exact-gradient-equivalence tests passed for multiple micro-batch sizes, including sizes that do not divide 32 exactly.

**Status**: locked and validated.

---

## D-061 — Mixed-precision runtime policy

**Selected choice**
- CUDA with native BF16 support → BF16 autocast, no GradScaler
- CUDA without native BF16 → FP16 autocast + GradScaler
- CPU → FP32 smoke-test fallback only
- FP32 master parameters remain intact

Native BF16 capability must exclude emulation.

**Evidence**
Tesla T4 reports compute capability 7.5 and no native BF16; Notebook 04 correctly resolves it to FP16 + GradScaler.

**Status**: locked and validated.

---

## D-062 — AdamW parameter groups

**Selected choice**
- weight decay 0.10 on parameters with `ndim >= 2`
- no weight decay on RMSNorm scale parameters (`ndim < 2`)
- AdamW betas `(0.9, 0.95)`
- epsilon `1e-8`
- identity-safe handling of tied embedding/LM-head parameters

**Status**: locked and validated.

---

## D-063 — Learning-rate schedule

**Selected choice**
Warmup followed by cosine decay, clocked on optimizer updates rather than micro-batches.

Production schedule:
- maximum epochs: 3
- updates/epoch: 1,221
- scheduled updates: 3,663
- warmup fraction: 5%
- warmup updates: 183
- cosine-decay updates: 3,480
- minimum LR: 10% of peak LR

**Status**: locked and validated.

---

## D-064 — Initial Model A learning-rate probe

**Selected choice**
Compare peak LRs `1e-4`, `3e-4`, and `1e-3` for 400 optimizer updates each, with validation at updates 200 and 400. Hold initialization, data order, RNG restart, optimizer, precision, batch, clipping, and validation data constant.

Selection rule:
1. lowest validation loss at update 400
2. tie → lowest validation loss at update 200
3. tie → lower LR

**Observed validation loss @400**
- `1e-4`: 6.799811
- `3e-4`: 6.163024
- `1e-3`: 5.659233

`1e-3` won the original grid but was the upper boundary, so the useful region was not yet bracketed.

**Status**: completed; followed by D-071 refinement.

---

## D-065 — Validation protocol

**Selected choice**
Use the official validation split only, reconstructed with the same pinned preprocessing and tokenizer. Preserve original validation article order and append one boundary token after every complete article. Evaluate deterministically with no shuffle.

**Evidence**
- validation articles: 60
- ordinary tokens: 256,579
- boundaries: 60
- validation stream: 256,639 tokens
- causal examples: 501
- scored validation targets: 256,512

Official test content remains excluded from tuning and validation.

**Status**: locked and validated.

---

## D-066 — Single-run training engine

**Selected choice**
Use an explicit PyTorch training engine that supports logical effective-update boundaries independently of physical DataLoader batch boundaries, mixed precision, AdamW, LR scheduling, target-count-normalized accumulation, clipping, validation hooks, and deterministic epoch shuffling.

**Evidence**
Synthetic and real-model smoke tests passed, including physical micro-batches that cross logical 32-sequence update boundaries.

**Status**: locked and validated.

---

## D-067 — Canonical model module integration

**Selected choice**
Move the validated Notebook 03 architecture into canonical reusable `src/model.py` without changing its behavior.

**Evidence**
- GitHub PR #22 merged
- Model A exact parameters: 7,407,872
- embedding/LM-head tying verified
- real Model A forward/backward smoke passed

**Status**: complete.

---

## D-068 — Canonical corpus reconstruction gate

**Selected choice**
Fresh training runtimes must rebuild the training and validation streams from canonical provenance and refuse to proceed if counts or fingerprints differ.

**Evidence**
Executed Notebook 04 reproduced:
- training stream: 20,000,000 tokens
- training SHA-256: `4101d5b18c38558a58110f54a161763186ab5318111366486ebbfa0a3fe584fa`
- selected article records: 4,604
- text tokens: 19,995,397
- boundary tokens: 4,603
- training causal examples: 39,062
- validation stream: 256,639
- validation examples: 501
- validation targets: 256,512

The canonical article-permutation fingerprint serializes permutation indices as little-endian signed 32-bit integers (`<i4`).

**Status**: complete and passed.

---

## D-069 — Real Model A optimizer smoke

**Selected choice**
Before LR experiments, run three real optimizer updates on Model A with the canonical corpus and one full validation pass.

**Evidence**
- PASS on Tesla T4
- FP16 + GradScaler
- physical micro-batch: 32 sequences
- targets/update: 16,384
- peak allocated GPU memory: approximately 4.94 GiB
- validation after update 3 completed on all 256,512 targets
- weights changed, gradients cleared, weight tying preserved

**Status**: complete and passed.

---

## D-070 — Controlled Model A LR probe execution

**Observed result**
The original candidate set completed successfully. Validation loss decreased monotonically through the upper tested boundary (`1e-3`), motivating one narrow refinement rather than prematurely locking the boundary winner.

**Status**: complete.

---

## D-071 — LR bracket refinement and production peak LR

**Decision**
Refine the upper boundary with `2e-3` and `3e-3` under the exact D-064 controls.

**Observed validation evidence**

| Peak LR | Val loss @200 | Val loss @400 | PPL @400 | Clip fraction |
|---:|---:|---:|---:|---:|
| `1e-4` | 7.067461 | 6.799811 | 897.68 | 28.7% |
| `3e-4` | 6.394534 | 6.163024 | 474.86 | 10.5% |
| `1e-3` | 5.988447 | 5.659233 | 286.93 | 6.5% |
| `2e-3` | 5.884712 | **5.488027** | **241.78** | 5.0% |
| `3e-3` | 6.159744 | 5.741478 | 311.52 | 4.5% |

**Selected production peak LR**
**`2e-3`**

**Rationale**
`2e-3` produced the lowest validation loss at update 400 and is below the largest tested value (`3e-3`), so the useful region is bracketed on the upper side.

Official test content was not used for selection.

**Status**: locked with experimental evidence.

---

## D-072 — Production accelerator micro-batch preflight

**Decision**
Before launching the three full model runs, probe the largest physical micro-batch that fits Models A/B/C on the active CUDA device while keeping the logical effective batch fixed at 32 sequences / 16,384 targets.

Model A has already demonstrated 32 sequences on the Tesla T4. Models B and C remain to be measured in the clean Notebook 04 continuation.

**Status**: next execution checkpoint; production training must not begin until this preflight is reviewed.
