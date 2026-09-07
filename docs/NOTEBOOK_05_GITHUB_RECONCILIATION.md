# Notebook 05 — GitHub Artifact Reconciliation

## Purpose

This note reconciles the completed Notebook 05 execution with the current GitHub repository before Notebook 06A proceeds beyond its experimental-contract stage.

## Verified canonical Notebook 05 state

The repository contains the completed executed notebook:

- `notebooks/05_evaluation_&_scaling.ipynb`

The canonical Notebook 05 decision register is complete through D-094:

- `docs/decisions/05_evaluation_and_scaling.md`

`docs/PROJECT_CONTEXT.md` records Notebook 05 as complete and freezes the A/B/C experiment before the 06A extension.

The global decision index has been reconciled on the 06A work branch to include:

- the Notebook 06A phase register;
- D-095 as the first Notebook 06A decision;
- D-096 as the next global decision ID.

## Generated Notebook 05 artifacts referenced by the executed notebook

The executed Notebook 05 explicitly reports creation of the following reusable outputs:

### Evidence
- `results/evaluation/evidence/validation_history_canonical.csv`
- `results/evaluation/evidence/validation_history_ingestion_audit.json`
- `results/evaluation/evidence/final_test_stream_audit.json`

### Analysis
- `results/evaluation/analysis/quality_endpoints.csv`
- `results/evaluation/analysis/quality_scaling_changes.csv`
- `results/evaluation/analysis/compute_scaling_summary.csv`
- `results/evaluation/analysis/compute_scaling_changes.csv`
- `results/evaluation/analysis/marginal_returns.csv`
- `results/evaluation/analysis/marginal_efficiency_retention.csv`
- `results/evaluation/analysis/controlled_generation_results.json`
- `results/evaluation/analysis/controlled_generation_results.csv`
- `results/evaluation/analysis/final_test_results.csv`
- `results/evaluation/analysis/final_test_results.json`
- `results/evaluation/analysis/final_test_controlled_generation.json`
- `results/evaluation/analysis/final_test_controlled_generation.csv`
- `results/evaluation/analysis/final_evaluation_summary.json`

### Figures
- `figures/evaluation/fig_01_validation_loss_learning_curves.png`
- `figures/evaluation/fig_02_validation_perplexity_learning_curves.png`
- additional Notebook 05 figures saved under `figures/evaluation/` by the executed notebook.

## Repository discrepancy

At reconciliation time, `main` contains neither `results/evaluation/` nor `figures/evaluation/`, despite the executed notebook showing successful writes to those paths.

This is a **repository-persistence discrepancy**, not an experimental-result discrepancy. The notebook outputs, decision register, and project context agree on the completed results, but the generated reusable files were not subsequently committed to GitHub.

## Reconciliation rule

Do **not** regenerate or hand-reconstruct these missing Notebook 05 artifacts from summary values alone. Their exact canonical versions should be recovered from the original executed Colab workspace or other persisted project storage and committed byte-for-byte when available. Recreating them from rounded values would risk introducing a second, noncanonical evidence set.

The frozen Notebook 05 experiment remains unchanged. Notebook 06A may define its experimental contract, but it must not mutate or replace Notebook 05 evidence while this persistence discrepancy is being resolved.

## 06A boundary

D-095 is documentation/design only and performs no training. The next implementation step, D-096, is the hard resume/provenance gate for the exact Model C update-3,663 checkpoint. No 06A extension update should occur until that gate passes.
