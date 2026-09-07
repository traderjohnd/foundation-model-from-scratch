# D-098 — Notebook 06A launch and exact-resume execution contract

Notebook 06A is an exploratory continuation of frozen Model C and remains separate from the controlled Notebook 05 A/B/C comparison.

## Selected choice

Authorize the Model C extended-training probe to begin only through the resumable 06A runner after the following conditions are all true:

- D-096 is formally closed with `gate_passed=true` in the persisted canonical gate artifact.
- The execution-locked 06A preflight reproduces the frozen update-3,663 boundary and validates the D-095 extension contract.
- The resumable runner preflight reports `Mode: FRESH EXTENSION START`, global update 3,663, epoch 4 position 0, and zero persisted extension updates before first launch.
- Training begins only when the resumable runner is invoked with the explicit `--execute` flag.

## Immutable launch boundary

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

## Exact 06A resume semantics

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

## Frozen extension controls

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

## Persistence and interruption policy

The 06A `latest` checkpoint is persisted at each validation event. A Colab interruption may therefore lose at most the uncheckpointed work since the most recent validation event. On restart, the experiment resumes from the most recent persisted 06A validation checkpoint rather than reconstructing intermediate optimizer updates from logs.

This is intentional: exact state can only be guaranteed from an actually persisted checkpoint. The experiment does not infer or replay unseen optimizer steps after a runtime loss.

## Provenance closure carried forward

Notebook 05 artifact recovery is closed by the canonical 23-artifact SHA-256 manifest on `main`. The deterministic recovery rerun originated from Notebook 05 source commit `e6bdd5ba27068659454873d28e6a2451810a815b`; the recovered artifact set was later persisted through PR #35 without changing any D-084–D-094 experimental conclusion. D-096 Gate 1 verifies those committed bytes before 06A continuation.

## Why

The research question is whether Model C was still materially training-duration constrained at the frozen three-epoch boundary. That question is only interpretable if continuation preserves optimizer, data-order, numerical, validation, and provenance state while changing only the allowed training duration. The explicit launch flag prevents accidental continuation, and the 06A-specific resumable checkpoint makes the experiment robust to Colab interruptions without contaminating frozen evidence.

## Presentation relevance

This provides a useful governance/reproducibility example: the continuation was not allowed to begin merely because weights existed. It required behavioral identity, optimizer-state continuity, deterministic data continuity, content-addressed evidence, an explicit execution boundary, and a resumable exploratory namespace.

## Status at decision time

- D-096: PASS / formally closed.
- 06A execution-locked preflight: PASS.
- 06A resumable preflight: PASS.
- mode before first launch: FRESH EXTENSION START.
- global update before first launch: 3,663.
- optimizer updates executed by 06A at decision time: 0.
- training has not yet begun.

The next optimizer update, if explicitly launched, is **3,664**.
