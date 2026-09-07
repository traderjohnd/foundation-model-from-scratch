#!/usr/bin/env python3
"""Consolidate D-096 component gates into the canonical resume-gate artifact.

Read-only with respect to all training/checkpoint state. Verifies Notebook 05
content-addressed repository evidence (Gate 1), loads Gate 2-5 JSON outputs,
cross-checks frozen invariants, and writes one canonical machine-readable gate
record. No optimizer/scaler step is executed; update 3,664 is unreachable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "extended_training" / "model_c"
MANIFEST_PATH = REPO_ROOT / "results" / "evaluation" / "artifact_manifest_sha256.json"

GATE2_PATH = OUT_DIR / "d096_gate2_resume_identity.json"
GATE3_PATH = OUT_DIR / "d096_gate3_adamw_continuity.json"
GATE4_PATH = OUT_DIR / "d096_gate4_provenance.json"
GATE5_PATH = OUT_DIR / "d096_gate5_shuffle_continuity.json"
OUTPUT_PATH = OUT_DIR / "d096_resume_gate.json"

EXPECTED_VAL = 3.684501
EXPECTED_UPDATE = 3663
EXPECTED_PARAMETERS = 33_497_600
EXPECTED_VALIDATION_TARGETS = 256_512
EXPECTED_TOKENIZER_SHA = "6ec601a267cec7c843df47927f53c4dd108c85a1d059318aeec4442c7274604f"
EXPECTED_TRAIN_SHA = "4101d5b18c38558a58110f54a161763186ab5318111366486ebbfa0a3fe584fa"
EXPECTED_MANIFEST_SHA = "4a00196b39311a6c2e2790780e8fc43316f24a014d3d3649028b10a671f8d3fe"
EXPECTED_EPOCH4_FIRST10 = [23330, 14720, 12892, 24465, 36182, 35545, 35235, 2313, 29923, 36592]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    assert path.exists(), f"D-096 BLOCKED: missing component artifact: {path}"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    return data


def verify_gate1() -> dict:
    assert MANIFEST_PATH.exists(), "D-096 BLOCKED: Notebook 05 artifact manifest missing"
    manifest = load_json(MANIFEST_PATH)
    assert manifest["verification"]["frozen_invariants_passed"] is True
    assert len(manifest["artifacts"]) == 23

    checked = []
    for item in manifest["artifacts"]:
        path = REPO_ROOT / item["path"]
        assert path.exists(), f"D-096 BLOCKED: missing Notebook 05 artifact: {item['path']}"
        actual_bytes = path.stat().st_size
        actual_sha = sha256_file(path)
        assert actual_bytes == int(item["bytes"]), f"D-096 BLOCKED: byte-size mismatch: {item['path']}"
        assert actual_sha == item["sha256"], f"D-096 BLOCKED: SHA-256 mismatch: {item['path']}"
        checked.append(item["path"])

    c_val = float(manifest["verification"]["validation"]["C"]["best_validation_loss"])
    assert abs(c_val - 3.6845006885642775) < 1e-12
    assert int(manifest["verification"]["validation"]["C"]["best_update"]) == EXPECTED_UPDATE
    assert int(manifest["verification"]["validation_targets_per_event"]) == EXPECTED_VALIDATION_TARGETS

    return {
        "gate": "D-096 Gate 1 — Notebook 05 repository evidence",
        "gate_passed": True,
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "canonical_artifacts_checked": len(checked),
        "validation_history_path": "results/evaluation/evidence/validation_history_canonical.csv",
        "model_c_frozen_validation_loss_full_precision": c_val,
        "model_c_frozen_best_update": EXPECTED_UPDATE,
        "validation_targets_per_event": EXPECTED_VALIDATION_TARGETS,
    }


def main() -> None:
    gate1 = verify_gate1()
    gate2 = load_json(GATE2_PATH)
    gate3 = load_json(GATE3_PATH)
    gate4 = load_json(GATE4_PATH)
    gate5 = load_json(GATE5_PATH)

    for idx, gate in enumerate((gate2, gate3, gate4, gate5), start=2):
        assert gate.get("gate_passed") is True, f"D-096 BLOCKED: Gate {idx} did not pass"
        assert int(gate.get("training_updates_executed", -1)) == 0, (
            f"D-096 BLOCKED: Gate {idx} reports training updates"
        )

    # Gate 2 behavioral identity.
    assert gate2["checkpoint"]["global_update"] == EXPECTED_UPDATE
    assert gate2["checkpoint"]["completed_full_epochs"] == 3
    assert gate2["validation_targets"] == EXPECTED_VALIDATION_TARGETS
    assert round(float(gate2["observed_validation_loss"]), 5) == round(EXPECTED_VAL, 5)

    # Gate 3 AdamW continuity.
    assert gate3["checkpoint_global_update"] == EXPECTED_UPDATE
    assert gate3["states_checked"] == gate3["trainable_parameter_tensors"] == 74
    for key in (
        "missing_or_malformed_states",
        "step_mismatches",
        "shape_mismatches",
        "zero_exp_avg_tensors",
        "zero_exp_avg_sq_tensors",
    ):
        assert gate3[key] == 0, f"D-096 BLOCKED: Gate 3 {key}={gate3[key]}"

    # Gate 4 provenance/config.
    assert gate4["checkpoint"]["global_update"] == EXPECTED_UPDATE
    assert gate4["checkpoint"]["completed_full_epochs"] == 3
    assert gate4["model_parameters"] == EXPECTED_PARAMETERS
    assert gate4["runtime"]["gpu_name"] == "Tesla T4"
    assert gate4["runtime"]["precision"] == "fp16"
    assert gate4["grad_scaler_state_present"] is True
    assert gate4["rng_state"]["torch_cpu_rng_state_present"] is True
    assert gate4["rng_state"]["torch_cuda_rng_state_count"] >= 1
    assert gate4["hashes"]["tokenizer_sha256"] == EXPECTED_TOKENIZER_SHA
    assert gate4["hashes"]["train_stream_sha256"] == EXPECTED_TRAIN_SHA
    assert gate4["hashes"]["corpus_manifest_sha256"] == EXPECTED_MANIFEST_SHA
    assert gate4["validation_provenance"] == {
        "context_length": 512,
        "examples": 501,
        "targets": EXPECTED_VALIDATION_TARGETS,
        "shuffle": False,
    }

    # Gate 5 epoch-4 data-order continuity.
    assert gate5["base_seed"] == 42
    assert gate5["epoch_number"] == 4
    assert gate5["effective_shuffle_seed"] == 46
    assert gate5["training_examples"] == 39_062
    assert gate5["first_10_example_indices"] == EXPECTED_EPOCH4_FIRST10
    assert gate5["repeat_first_10_example_indices"] == EXPECTED_EPOCH4_FIRST10
    assert gate5["independent_first_10_example_indices"] == EXPECTED_EPOCH4_FIRST10

    consolidated = {
        "decision": "D-096",
        "artifact": "Model C hard resume/provenance gate",
        "gate_passed": True,
        "training_authorized_by_gate": True,
        "training_started": False,
        "next_optimizer_update_if_explicitly_started": 3664,
        "component_gates": {
            "gate1_repository_evidence": gate1,
            "gate2_behavioral_identity": gate2,
            "gate3_adamw_continuity": gate3,
            "gate4_provenance": gate4,
            "gate5_shuffle_continuity": gate5,
        },
        "frozen_boundary": {
            "model_key": "C",
            "global_update": EXPECTED_UPDATE,
            "completed_epochs": 3,
            "validation_loss_reference": EXPECTED_VAL,
            "model_parameters": EXPECTED_PARAMETERS,
        },
        "extension_contract": {
            "first_epoch": 4,
            "epoch4_shuffle_seed": 46,
            "constant_learning_rate": 2e-4,
            "min_delta": 0.001,
            "patience_validation_events": 6,
            "max_additional_epochs": 10,
            "max_global_update": 15_873,
        },
        "explicit_stop": "D-096 is closed, but update 3,664 has not been executed by this consolidator.",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(consolidated, indent=2) + "\n", encoding="utf-8")

    print("D-096 CONSOLIDATED RESUME GATE: PASS")
    print("Gate 1 Notebook 05 artifacts checked:", gate1["canonical_artifacts_checked"])
    print("Gate 2 observed validation loss:", f"{float(gate2['observed_validation_loss']):.12f}")
    print("Gate 3 AdamW states checked:", gate3["states_checked"])
    print("Gate 4 provenance hashes: PASS")
    print("Gate 5 epoch-4 first 10:", gate5["first_10_example_indices"])
    print("Canonical gate artifact:", OUTPUT_PATH.relative_to(REPO_ROOT))
    print("Training authorized by D-096: YES")
    print("Training started: NO")
    print("STOP: update 3,664 has NOT executed.")


if __name__ == "__main__":
    main()
