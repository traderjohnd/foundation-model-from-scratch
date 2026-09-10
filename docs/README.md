# Documentation Map

The project documentation is organized by purpose so current state, historical decisions, and provenance records remain distinct.

## Canonical project documents

- `PROJECT_CONTEXT.md` - concise current-state and resume document. Read this first when continuing the project.
- `DECISION_INDEX.md` - chronological index of all globally unique project decisions.

## Decision registers

`docs/decisions/` contains **one canonical decision register per project phase/notebook**. Each register preserves the detailed rationale, alternatives, evidence, and presentation relevance for that phase.

Current registers:

- `00_project_definition.md`
- `01_data_preparation_and_corpus_audit.md`
- `02_tokenizer_training_and_corpus_construction.md`
- `03_model_architecture.md`
- `04_training_pipeline.md`
- `05_evaluation_and_scaling.md`
- `06a_model_c_extended_training_probe.md`

Decision IDs are globally unique and append-only. Consolidating decisions into a phase register changes only file organization; it does not rewrite their historical meaning.

## Provenance records

`docs/provenance/` contains audit/recovery records that are important to reproducibility but are not themselves the canonical decision register.

- `05_evaluation_artifact_recovery.md` - records the Notebook 05 repository-persistence discrepancy, fail-closed recovery rule, and final resolution.

## Evidence and outputs

Machine-readable experimental evidence is stored under `results/`. Presentation-ready figures are stored under `figures/`. Large model checkpoints remain outside Git by design when appropriate, with hashes recorded in repository manifests.

## Publication and presentation assembly

- [`publication/CLAIM_EVIDENCE_MAP.md`](publication/CLAIM_EVIDENCE_MAP.md) - D-101's map from proposed publication claims to pinned frozen evidence, decision IDs, figures, and interpretation limits. Read before drafting downstream narratives.
- [`publication/EXECUTIVE_NARRATIVE.md`](publication/EXECUTIVE_NARRATIVE.md) - the first STE-style narrative draft for the README, presentation, article, and monograph.

D-101 is recorded in the project-wide `decisions/00_project_definition.md` register. The claim map is a communication artifact; experimental evidence remains under `results/`.

Public summaries use **wall time**, **peak memory**, and validation-loss gain per approximate parameter doubling. The public comparison is **0.164 nats** for A→B and **0.093 nats** for B→C.

## Reading order for presentation/report work

1. `PROJECT_CONTEXT.md`
2. `publication/CLAIM_EVIDENCE_MAP.md`
3. `publication/EXECUTIVE_NARRATIVE.md`
4. `DECISION_INDEX.md`
5. the relevant phase register under `docs/decisions/`
6. canonical evidence under `results/`
7. presentation figures under `figures/`
8. provenance notes only when audit/recovery history is relevant
