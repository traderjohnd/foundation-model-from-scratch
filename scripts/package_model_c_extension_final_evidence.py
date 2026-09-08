#!/usr/bin/env python3
"""Package completed Notebook 06A evidence from persistent Drive into the repo.

Read-only with respect to training/checkpoints. Copies canonical JSON evidence,
validates frozen 06A outcomes, and writes a SHA-256 manifest including external
checkpoint hashes without copying large checkpoint binaries into git.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVE_ROOT = Path(
    "/content/drive/MyDrive/foundation-model-from-scratch/production/"
    "extended_training/model_c"
)
OUT_DIR = REPO_ROOT / "results/extended_training/model_c/final_evidence"

EXPECTED_BEST_UPDATE = 12_210
EXPECTED_BEST_VAL = 3.599946362767629
EXPECTED_FINAL_UPDATE = 13_263
EXPECTED_EXT_UPDATES = 9_600
EXPECTED_TEST_LOSS = 3.606927575449253
EXPECTED_TEST_TARGETS = 293_376
EXPECTED_TEST_STREAM_SHA = (
    "9578e1403a94bf085eb55372e76d4dd74e02c89f7085368c75a1d75f5537d188"
)
EXPECTED_FP16_RETRIES = 3

ARTIFACTS = [
    "extension_summary.json",
    "extension_history.json",
    "extension_progress.json",
    "one_time_exploratory_test.json",
    "fixed_d091_generation.json",
]
CHECKPOINTS = [
    "model_c_extension_best.pt",
    "model_c_extension_latest.pt",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    if not DRIVE_ROOT.exists():
        raise FileNotFoundError(f"06A Drive evidence root not found: {DRIVE_ROOT}")

    for name in ARTIFACTS + CHECKPOINTS:
        if not (DRIVE_ROOT / name).exists():
            raise FileNotFoundError(DRIVE_ROOT / name)

    summary = load_json(DRIVE_ROOT / "extension_summary.json")
    test = load_json(DRIVE_ROOT / "one_time_exploratory_test.json")
    generation = load_json(DRIVE_ROOT / "fixed_d091_generation.json")

    assert summary["experiment"] == "06A_model_c_extended_training_probe"
    assert int(summary["final_global_update"]) == EXPECTED_FINAL_UPDATE
    assert int(summary["extension_updates"]) == EXPECTED_EXT_UPDATES
    assert int(summary["best_extension_validation_update"]) == EXPECTED_BEST_UPDATE
    assert abs(float(summary["best_extension_validation_loss"]) - EXPECTED_BEST_VAL) < 1e-12
    assert summary["stop_reason"] == "early_stopping_patience_exhausted"
    assert summary["saturation_observed"] is True
    assert summary["official_test_split_content_used"] is False
    assert int(summary["fp16_overflow_retries_total"]) == EXPECTED_FP16_RETRIES

    assert test["evaluation"] == "one_time_exploratory_official_test"
    assert int(test["selected_checkpoint_update"]) == EXPECTED_BEST_UPDATE
    assert abs(float(test["selected_validation_loss"]) - EXPECTED_BEST_VAL) < 1e-12
    assert abs(float(test["test_loss"]) - EXPECTED_TEST_LOSS) < 1e-12
    assert int(test["test_targets"]) == EXPECTED_TEST_TARGETS
    assert test["test_stream_sha256"] == EXPECTED_TEST_STREAM_SHA
    assert int(test["official_test_scoring_count_for_06a"]) == 1
    assert test["checkpoint_selection_changed_after_test"] is False
    assert test["retuning_permitted_after_test"] is False

    # The generation artifact must represent the fixed D-091 protocol for three prompts.
    rows = generation if isinstance(generation, list) else generation.get("results", [])
    assert len(rows) == 3
    assert [int(r["seed"]) for r in rows] == [43, 44, 45]
    assert all(float(r["temperature"]) == 0.8 for r in rows)
    assert all(float(r["top_p"]) == 0.9 for r in rows)
    assert all(int(r["new_tokens"]) == 96 for r in rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in ARTIFACTS:
        src = DRIVE_ROOT / name
        dst = OUT_DIR / name
        shutil.copy2(src, dst)
        copied[str(dst.relative_to(REPO_ROOT))] = file_record(dst)

    external_checkpoints = {}
    for name in CHECKPOINTS:
        path = DRIVE_ROOT / name
        external_checkpoints[name] = {
            "persistent_path": str(path),
            **file_record(path),
        }

    manifest = {
        "experiment": "06A_model_c_extended_training_probe",
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(DRIVE_ROOT),
        "repo_evidence_files": copied,
        "external_checkpoint_files": external_checkpoints,
        "frozen_outcome": {
            "best_validation_update": EXPECTED_BEST_UPDATE,
            "best_validation_loss": EXPECTED_BEST_VAL,
            "final_global_update": EXPECTED_FINAL_UPDATE,
            "extension_updates": EXPECTED_EXT_UPDATES,
            "stop_reason": "early_stopping_patience_exhausted",
            "fp16_overflow_retries_total": EXPECTED_FP16_RETRIES,
            "one_time_test_loss": EXPECTED_TEST_LOSS,
            "one_time_test_targets": EXPECTED_TEST_TARGETS,
            "one_time_test_stream_sha256": EXPECTED_TEST_STREAM_SHA,
        },
        "large_checkpoints_copied_into_git": False,
    }
    manifest_path = OUT_DIR / "artifact_manifest_sha256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("NOTEBOOK 06A FINAL EVIDENCE PACKAGING: PASS")
    print("Copied JSON artifacts:", len(ARTIFACTS))
    print("Best validation checkpoint:", EXPECTED_BEST_UPDATE)
    print("One-time test loss:", EXPECTED_TEST_LOSS)
    print("Generation prompts:", len(rows))
    print("External checkpoints hashed:", len(CHECKPOINTS))
    print("Manifest:", manifest_path)
    print("No training, validation, generation, or test scoring executed.")


if __name__ == "__main__":
    main()
