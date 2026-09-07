#!/usr/bin/env python3
"""CSV-based fail-closed verifier for recovered Notebook 05 artifacts.

Use after Notebook 05 has already been re-executed successfully:
    python scripts/recover_notebook05_artifacts_v3.py

This version avoids JSON-schema ambiguity by validating the canonical flat CSV
outputs and treats the JSON files as immutable artifacts to hash.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

FROZEN_VALIDATION_LOSS = {"A": 3.972054, "B": 3.776427, "C": 3.684501}
FROZEN_TEST_LOSS = {"A": 3.955290, "B": 3.772079, "C": 3.680554}
EXPECTED_VALIDATION_TARGETS = 256_512
EXPECTED_TEST_TARGETS = 293_376
TOLERANCE = 5e-7

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


def norm_model(value: Any) -> str:
    s = str(value).strip().upper().replace("MODEL ", "").replace("MODEL_", "")
    return s


def find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    lookup = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    raise AssertionError(f"Missing required column; tried {candidates}; available={fieldnames}")


def assert_close(label: str, observed: float, expected: float) -> None:
    if not math.isfinite(observed) or abs(observed - expected) > TOLERANCE:
        raise AssertionError(
            f"{label}: observed={observed:.12f}, expected={expected:.12f}, "
            f"abs_diff={abs(observed - expected):.12g}, tolerance={TOLERANCE}"
        )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames is None:
            raise AssertionError(f"No header in {path}")
        return list(reader.fieldnames), rows


def validate_validation(repo: Path) -> dict[str, Any]:
    path = repo / "results/evaluation/evidence/validation_history_canonical.csv"
    fields, rows = read_csv(path)
    model_col = find_column(fields, ("model", "model_key", "model_name"))
    loss_col = find_column(fields, ("validation_loss", "val_loss", "loss"))
    update_col = find_column(fields, ("update", "global_update", "optimizer_update"))
    target_col = find_column(fields, ("validation_targets", "scored_targets", "targets"))

    out: dict[str, Any] = {}
    for model, expected in FROZEN_VALIDATION_LOSS.items():
        mr = [r for r in rows if norm_model(r[model_col]) == model]
        if not mr:
            raise AssertionError(f"No validation rows for Model {model}")
        for r in mr:
            if int(float(r[target_col])) != EXPECTED_VALIDATION_TARGETS:
                raise AssertionError(f"Model {model} has noncanonical validation target count")
        best = min(mr, key=lambda r: float(r[loss_col]))
        observed = float(best[loss_col])
        assert_close(f"Model {model} best validation loss", observed, expected)
        if int(float(best[update_col])) != 3663:
            raise AssertionError(f"Model {model} best validation update is not 3663")
        out[model] = {"best_validation_loss": observed, "events": len(mr), "best_update": 3663}
    return out


def validate_test(repo: Path) -> dict[str, Any]:
    path = repo / "results/evaluation/analysis/final_test_results.csv"
    fields, rows = read_csv(path)

    # pandas to_csv on an indexed DataFrame often writes the index as the first
    # column (possibly named model/model_key, or as an unnamed field).
    model_candidates = ("model", "model_key", "model_name", "index", "unnamed: 0")
    model_col = None
    lower = {f.lower(): f for f in fields}
    for c in model_candidates:
        if c.lower() in lower:
            model_col = lower[c.lower()]
            break
    if model_col is None:
        # Last-resort structural check: identify a column whose normalized values
        # are exactly A/B/C across the three result rows.
        for f in fields:
            vals = {norm_model(r.get(f, "")) for r in rows}
            if {"A", "B", "C"}.issubset(vals):
                model_col = f
                break
    if model_col is None:
        raise AssertionError(f"Could not identify model column in {path}; fields={fields}")

    loss_col = find_column(fields, ("test_loss", "loss"))
    target_col = find_column(fields, ("test_targets", "scored_targets", "targets", "n_targets"))

    out: dict[str, Any] = {}
    for model, expected in FROZEN_TEST_LOSS.items():
        mr = [r for r in rows if norm_model(r[model_col]) == model]
        if len(mr) != 1:
            raise AssertionError(f"Expected exactly one test row for Model {model}; got {len(mr)}")
        observed = float(mr[0][loss_col])
        targets = int(float(mr[0][target_col]))
        assert_close(f"Model {model} test loss", observed, expected)
        if targets != EXPECTED_TEST_TARGETS:
            raise AssertionError(f"Model {model} test targets={targets}, expected={EXPECTED_TEST_TARGETS}")
        out[model] = {"test_loss": observed, "test_targets": targets}
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    repo = Path.cwd().resolve()
    paths = [repo / rel for rel in CANONICAL_ARTIFACTS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing canonical artifacts:\n" + "\n".join(str(p.relative_to(repo)) for p in missing))

    verification = {
        "validation": validate_validation(repo),
        "test": validate_test(repo),
        "validation_targets_per_event": EXPECTED_VALIDATION_TARGETS,
        "test_targets_per_model": EXPECTED_TEST_TARGETS,
        "tolerance": TOLERANCE,
        "frozen_invariants_passed": True,
    }

    entries = [
        {"path": p.relative_to(repo).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(paths)
    ]
    manifest = {
        "artifact_set": "Notebook 05 recovered canonical evaluation outputs",
        "recovery_method": "deterministic re-execution of Notebook 05 from frozen inputs",
        "d092_note": (
            "The official test split was re-scored solely to recover missing repository artifacts under the unchanged D-092 procedure. "
            "No model selection, retuning, or downstream experimental decision is permitted from this re-measurement."
        ),
        "verification": verification,
        "artifacts": entries,
    }
    out = repo / "results/evaluation/artifact_manifest_sha256.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Notebook 05 artifact recovery verification: PASS")
    print(f"Manifest: {out.relative_to(repo)}")
    print(f"Artifacts hashed: {len(paths)}")
    print("Frozen validation/test invariants: PASS")
    print("Eligible for commit: YES")


if __name__ == "__main__":
    main()
