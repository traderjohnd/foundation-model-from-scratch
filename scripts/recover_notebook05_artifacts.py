#!/usr/bin/env python3
"""Deterministic recovery harness for Notebook 05 evaluation artifacts.

This script is intentionally fail-closed. It is meant to be run in the same
Colab/Drive environment used by Notebook 05, after the frozen production
checkpoints/histories are available. It does not train or modify Models A/B/C.

Workflow:
1. Execute Notebook 05 from the repository state in a fresh working tree.
2. Require the canonical evaluation artifacts to exist.
3. Assert regenerated validation/test invariants against the frozen register.
4. Compute SHA-256 and byte size for every canonical artifact.
5. Write results/evaluation/artifact_manifest_sha256.json.

The generated artifacts are eligible for commit only if every assertion passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

FROZEN_VALIDATION_LOSS = {
    "A": 3.972054,
    "B": 3.776427,
    "C": 3.684501,
}
FROZEN_TEST_LOSS = {
    "A": 3.955290,
    "B": 3.772079,
    "C": 3.680554,
}
EXPECTED_VALIDATION_TARGETS = 256_512
EXPECTED_TEST_TARGETS = 293_376
TOLERANCE = 5e-7  # sufficient to preserve the six-decimal frozen values

CANONICAL_ARTIFACTS = [
    "results/evaluation/evidence/validation_history_canonical.csv",
    "results/evaluation/evidence/validation_history_ingestion_audit.json",
    "results/evaluation/evidence/final_test_stream_audit.json",
    "results/evaluation/analysis/quality_endpoints.csv",
    "results/evaluation/analysis/quality_scaling_changes.csv",
    "results/evaluation/analysis/compute_scaling_summary.csv",
    "results/evaluation/analysis/compute_scaling_changes.csv",
    "results/evaluation/analysis/marginal_returns.csv",
    "results/evaluation/analysis/marginal_efficiency_retention.csv",
    "results/evaluation/analysis/controlled_generation_results.json",
    "results/evaluation/analysis/controlled_generation_results.csv",
    "results/evaluation/analysis/final_test_results.csv",
    "results/evaluation/analysis/final_test_results.json",
    "results/evaluation/analysis/final_test_controlled_generation.json",
    "results/evaluation/analysis/final_test_controlled_generation.csv",
    "results/evaluation/analysis/final_evaluation_summary.json",
    "figures/evaluation/fig_01_validation_loss_learning_curves.png",
    "figures/evaluation/fig_02_validation_perplexity_learning_curves.png",
    "figures/evaluation/fig_03_training_wall_time_vs_parameters.png",
    "figures/evaluation/fig_04_peak_gpu_memory_vs_parameters.png",
    "figures/evaluation/fig_05_effective_throughput_vs_parameters.png",
    "figures/evaluation/fig_06_quality_cost_frontier.png",
    "figures/evaluation/fig_07_marginal_efficiency_retained.png",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_key(obj: Any, candidate_keys: set[str]) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in candidate_keys:
                return value
        for value in obj.values():
            found = find_key(value, candidate_keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key(value, candidate_keys)
            if found is not None:
                return found
    return None


def load_test_results(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    records: list[dict[str, Any]] = []
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for key in ("results", "models", "test_results"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
            if isinstance(value, dict):
                records = [dict(v, model=k) if isinstance(v, dict) else {"model": k, "value": v}
                           for k, v in value.items()]
                break
        if not records:
            records = [payload]

    normalized: dict[str, dict[str, Any]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        model = find_key(rec, {"model", "model_key", "model_name"})
        if model is None:
            continue
        model = str(model).strip().upper().replace("MODEL ", "")
        if model in {"A", "B", "C"}:
            normalized[model] = rec
    return normalized


def assert_close(label: str, observed: float, expected: float) -> None:
    if not math.isfinite(observed) or abs(observed - expected) > TOLERANCE:
        raise AssertionError(
            f"{label}: observed={observed:.12f}, expected={expected:.12f}, "
            f"abs_diff={abs(observed - expected):.12g}, tolerance={TOLERANCE}"
        )


def validate_frozen_results(repo: Path) -> dict[str, Any]:
    validation_csv = repo / "results/evaluation/evidence/validation_history_canonical.csv"
    test_json = repo / "results/evaluation/analysis/final_test_results.json"

    with validation_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    validation_checks = {}
    for model, expected_loss in FROZEN_VALIDATION_LOSS.items():
        model_rows = [r for r in rows if str(r.get("model", r.get("model_key", ""))).strip().upper().replace("MODEL ", "") == model]
        if not model_rows:
            raise AssertionError(f"No validation-history rows found for Model {model}")

        loss_key = next((k for k in ("validation_loss", "val_loss", "loss") if k in model_rows[0]), None)
        update_key = next((k for k in ("update", "global_update", "optimizer_update") if k in model_rows[0]), None)
        targets_key = next((k for k in ("validation_targets", "scored_targets", "targets") if k in model_rows[0]), None)
        if loss_key is None:
            raise AssertionError("Validation history has no recognized loss column")

        losses = [float(r[loss_key]) for r in model_rows]
        observed_best = min(losses)
        assert_close(f"Model {model} best validation loss", observed_best, expected_loss)

        if update_key is not None:
            best_row = min(model_rows, key=lambda r: float(r[loss_key]))
            if int(float(best_row[update_key])) != 3663:
                raise AssertionError(f"Model {model} best validation update is not 3663")

        if targets_key is not None:
            bad = [r for r in model_rows if int(float(r[targets_key])) != EXPECTED_VALIDATION_TARGETS]
            if bad:
                raise AssertionError(f"Model {model} has validation events with noncanonical target counts")

        validation_checks[model] = {
            "best_validation_loss": observed_best,
            "expected_validation_loss": expected_loss,
            "events": len(model_rows),
        }

    test_records = load_test_results(test_json)
    if set(test_records) != {"A", "B", "C"}:
        raise AssertionError(f"Could not normalize A/B/C from {test_json}; got {sorted(test_records)}")

    test_checks = {}
    for model, expected_loss in FROZEN_TEST_LOSS.items():
        rec = test_records[model]
        observed_loss = find_key(rec, {"test_loss", "loss"})
        targets = find_key(rec, {"test_targets", "scored_targets", "targets", "n_targets"})
        if observed_loss is None or targets is None:
            raise AssertionError(f"Model {model} test record is missing loss or target count")
        observed_loss = float(observed_loss)
        targets = int(targets)
        assert_close(f"Model {model} test loss", observed_loss, expected_loss)
        if targets != EXPECTED_TEST_TARGETS:
            raise AssertionError(
                f"Model {model} test targets={targets}, expected={EXPECTED_TEST_TARGETS}"
            )
        test_checks[model] = {
            "test_loss": observed_loss,
            "expected_test_loss": expected_loss,
            "test_targets": targets,
        }

    return {
        "validation": validation_checks,
        "test": test_checks,
        "validation_targets_per_event": EXPECTED_VALIDATION_TARGETS,
        "test_targets_per_model": EXPECTED_TEST_TARGETS,
        "tolerance": TOLERANCE,
        "frozen_invariants_passed": True,
    }


def require_artifacts(repo: Path) -> list[Path]:
    paths = [repo / rel for rel in CANONICAL_ARTIFACTS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        formatted = "\n".join(f"  - {p.relative_to(repo)}" for p in missing)
        raise FileNotFoundError(f"Canonical Notebook 05 artifacts missing after execution:\n{formatted}")
    return paths


def build_manifest(repo: Path, paths: list[Path], verification: dict[str, Any]) -> dict[str, Any]:
    entries = []
    for path in sorted(paths):
        entries.append({
            "path": path.relative_to(repo).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "artifact_set": "Notebook 05 recovered canonical evaluation outputs",
        "recovery_method": "deterministic re-execution of Notebook 05 from frozen inputs",
        "d092_note": (
            "The official test split was re-scored solely to recover missing repository artifacts "
            "under the unchanged D-092 procedure. No model selection, retuning, or downstream "
            "experimental decision is permitted from this re-measurement."
        ),
        "verification": verification,
        "artifacts": entries,
    }


def execute_notebook(repo: Path, timeout: int) -> None:
    notebook = repo / "notebooks/05_evaluation_&_scaling.ipynb"
    if not notebook.exists():
        raise FileNotFoundError(notebook)
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(notebook),
    ]
    subprocess.run(cmd, cwd=repo, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--skip-execution", action="store_true", help="Validate artifacts already regenerated in the current working tree")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not args.skip_execution:
        execute_notebook(repo, args.timeout)

    paths = require_artifacts(repo)
    verification = validate_frozen_results(repo)
    manifest = build_manifest(repo, paths, verification)

    manifest_path = repo / "results/evaluation/artifact_manifest_sha256.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print("Notebook 05 artifact recovery verification: PASS")
    print(f"Manifest: {manifest_path.relative_to(repo)}")
    print(f"Artifacts hashed: {len(paths)}")
    print("Frozen validation/test invariants: PASS")
    print("Eligible for commit: YES")


if __name__ == "__main__":
    main()
