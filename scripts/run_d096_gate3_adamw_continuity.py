#!/usr/bin/env python3
"""D-096 Gate 3: AdamW optimizer-state continuity for frozen Model C.

Read-only audit. Loads the exact update-3,663 Model C v2 checkpoint and verifies
that every trainable parameter carries canonical AdamW state with step=3,663,
shape-matched exp_avg/exp_avg_sq tensors, and non-zero moment contents.

ZERO optimizer/scaler steps are executed. This script cannot reach update 3,664.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.model import MODEL_CONFIGS, DecoderOnlyLM, analytical_parameter_count
from src.training_pipeline import build_adamw_optimizer

EXPECTED_MODEL_KEY = "C"
EXPECTED_PARAMETERS = 33_497_600
EXPECTED_UPDATE = 3_663


def step_as_int(value) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persistent-root",
        type=Path,
        default=Path("/content/drive/MyDrive/foundation-model-from-scratch/production"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    checkpoint_path = (
        args.persistent_root.resolve()
        / "checkpoints"
        / "model_c_latest.pt"
    )
    assert checkpoint_path.exists(), (
        f"D-096 Gate 3 BLOCKED: missing checkpoint: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint.get("format_version") == 2
    assert checkpoint.get("model_key") == EXPECTED_MODEL_KEY
    assert int(checkpoint.get("global_update", -1)) == EXPECTED_UPDATE
    assert "optimizer_state_dict" in checkpoint
    assert "model_state_dict" in checkpoint

    cfg = MODEL_CONFIGS[EXPECTED_MODEL_KEY]
    assert analytical_parameter_count(cfg)["total"] == EXPECTED_PARAMETERS

    model = DecoderOnlyLM(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS

    optimizer = build_adamw_optimizer(model, learning_rate=2e-3)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    audit = {
        "gate": "D-096 Gate 3 — AdamW optimizer-state continuity",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_global_update": int(checkpoint["global_update"]),
        "training_updates_executed": 0,
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameters": sum(p.numel() for _, p in trainable),
        "states_checked": 0,
        "missing_or_malformed_states": 0,
        "step_mismatches": 0,
        "shape_mismatches": 0,
        "zero_exp_avg_tensors": 0,
        "zero_exp_avg_sq_tensors": 0,
        "failures": [],
    }

    for name, param in trainable:
        state = optimizer.state.get(param)
        if not state or not all(k in state for k in ("step", "exp_avg", "exp_avg_sq")):
            audit["missing_or_malformed_states"] += 1
            audit["failures"].append({"parameter": name, "reason": "missing_or_malformed_state"})
            continue

        audit["states_checked"] += 1

        step = step_as_int(state["step"])
        if step != EXPECTED_UPDATE:
            audit["step_mismatches"] += 1
            audit["failures"].append({
                "parameter": name,
                "reason": "step_mismatch",
                "observed_step": step,
                "expected_step": EXPECTED_UPDATE,
            })

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        if exp_avg.shape != param.shape or exp_avg_sq.shape != param.shape:
            audit["shape_mismatches"] += 1
            audit["failures"].append({
                "parameter": name,
                "reason": "moment_shape_mismatch",
                "parameter_shape": list(param.shape),
                "exp_avg_shape": list(exp_avg.shape),
                "exp_avg_sq_shape": list(exp_avg_sq.shape),
            })
            continue

        if not bool((exp_avg != 0).any().item()):
            audit["zero_exp_avg_tensors"] += 1
            audit["failures"].append({"parameter": name, "reason": "zero_exp_avg"})

        if not bool((exp_avg_sq != 0).any().item()):
            audit["zero_exp_avg_sq_tensors"] += 1
            audit["failures"].append({"parameter": name, "reason": "zero_exp_avg_sq"})

    audit["gate_passed"] = (
        audit["states_checked"] == audit["trainable_parameter_tensors"]
        and audit["missing_or_malformed_states"] == 0
        and audit["step_mismatches"] == 0
        and audit["shape_mismatches"] == 0
        and audit["zero_exp_avg_tensors"] == 0
        and audit["zero_exp_avg_sq_tensors"] == 0
    )

    assert audit["gate_passed"], (
        "D-096 Gate 3 BLOCKED: AdamW continuity audit failed: "
        + json.dumps({k: v for k, v in audit.items() if k != "failures"})
    )

    output = args.output or Path(
        "results/extended_training/model_c/d096_gate3_adamw_continuity.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    print("D-096 GATE 3: PASS")
    print("Checkpoint:", checkpoint_path)
    print("Global update:", audit["checkpoint_global_update"])
    print("Trainable parameter tensors:", audit["trainable_parameter_tensors"])
    print("States checked:", audit["states_checked"])
    print("Step mismatches:", audit["step_mismatches"])
    print("Shape mismatches:", audit["shape_mismatches"])
    print("Zero exp_avg tensors:", audit["zero_exp_avg_tensors"])
    print("Zero exp_avg_sq tensors:", audit["zero_exp_avg_sq_tensors"])
    print("Missing/malformed states:", audit["missing_or_malformed_states"])
    print("Optimizer updates executed: 0")
    print("STOP: Gate 4 has not been executed; update 3,664 is unreachable from this script.")


if __name__ == "__main__":
    main()
