#!/usr/bin/env python3
"""D-096 Gate 2: behavioral identity of frozen Model C at update 3,663.

Read-only validation gate. Loads the exact resumable Model C v2 `latest`
checkpoint from the canonical production root, reconstructs/loads the canonical
validation dataset through Notebook 04's training pipeline, runs the exact
no-shuffle D-072 validation evaluator over 256,512 targets, and fails closed
unless the loss reproduces frozen 3.684501 to at least five decimal places.

ZERO optimizer steps are executed. This script cannot reach update 3,664.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Ensure repository-root imports work when this file is executed directly as
# `python scripts/run_d096_gate2_resume_identity.py` from a fresh Colab clone.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.model import MODEL_CONFIGS, DecoderOnlyLM, analytical_parameter_count
from src.training_pipeline import (
    VALIDATION_TARGETS,
    build_adamw_optimizer,
    evaluate_language_model,
    load_or_build_canonical_datasets,
    make_grad_scaler,
    make_validation_dataloader,
    resolve_runtime_precision_policy,
)

EXPECTED_MODEL_KEY = "C"
EXPECTED_PARAMETERS = 33_497_600
EXPECTED_UPDATE = 3_663
EXPECTED_COMPLETED_EPOCHS = 3
EXPECTED_EPOCH_INDEX = 2
EXPECTED_UPDATES_IN_EPOCH = 1_221
EXPECTED_VAL_LOSS = 3.684501
REQUIRED_DECIMALS = 5


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"D-096 Gate 2 BLOCKED: missing resumable checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("D-096 Gate 2 BLOCKED: checkpoint payload is not a dict")
    return checkpoint


def audit_checkpoint_metadata(checkpoint: dict) -> dict:
    required = {
        "format_version", "model_key", "global_update", "epoch_index",
        "updates_completed_in_epoch", "completed_full_epochs",
        "model_state_dict", "optimizer_state_dict", "scaler_state_dict", "rng_state",
    }
    missing = sorted(required - set(checkpoint))
    assert not missing, f"D-096 Gate 2 BLOCKED: checkpoint missing keys: {missing}"
    assert int(checkpoint["format_version"]) == 2
    assert checkpoint["model_key"] == EXPECTED_MODEL_KEY
    assert int(checkpoint["global_update"]) == EXPECTED_UPDATE
    assert int(checkpoint["completed_full_epochs"]) == EXPECTED_COMPLETED_EPOCHS
    assert int(checkpoint["epoch_index"]) == EXPECTED_EPOCH_INDEX
    assert int(checkpoint["updates_completed_in_epoch"]) == EXPECTED_UPDATES_IN_EPOCH
    return {
        "format_version": int(checkpoint["format_version"]),
        "model_key": checkpoint["model_key"],
        "global_update": int(checkpoint["global_update"]),
        "completed_full_epochs": int(checkpoint["completed_full_epochs"]),
        "epoch_index": int(checkpoint["epoch_index"]),
        "updates_completed_in_epoch": int(checkpoint["updates_completed_in_epoch"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persistent-root",
        type=Path,
        default=Path("/content/drive/MyDrive/foundation-model-from-scratch/production"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "D-096 Gate 2 BLOCKED: CUDA GPU is required"
    gpu_name = torch.cuda.get_device_name(0)
    assert gpu_name == "Tesla T4", (
        f"D-096 Gate 2 BLOCKED: canonical runtime requires Tesla T4; got {gpu_name!r}"
    )

    persistent_root = args.persistent_root.resolve()
    checkpoint_path = persistent_root / "checkpoints" / "model_c_latest.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_audit = audit_checkpoint_metadata(checkpoint)

    cfg = MODEL_CONFIGS[EXPECTED_MODEL_KEY]
    assert analytical_parameter_count(cfg)["total"] == EXPECTED_PARAMETERS
    model = DecoderOnlyLM(cfg).cuda()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actual_parameters = sum(p.numel() for p in model.parameters())
    assert actual_parameters == EXPECTED_PARAMETERS

    # Prove compatibility with the exact canonical runtime state. We load these
    # states but deliberately never call optimizer.step() or scaler.step().
    policy = resolve_runtime_precision_policy()
    assert policy.precision == "fp16"
    assert policy.use_grad_scaler is True
    optimizer = build_adamw_optimizer(model, learning_rate=2e-3)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler = make_grad_scaler(policy)
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    datasets = load_or_build_canonical_datasets(persistent_root)
    validation_loader = make_validation_dataloader(
        datasets["validation_dataset"],
        batch_size=32,
    )

    # Behavioral identity measurement using the exact Notebook 04 evaluator.
    metrics = evaluate_language_model(model, validation_loader, policy)
    observed = float(metrics["validation_loss"])
    targets = int(metrics["validation_targets"])
    absolute_difference = abs(observed - EXPECTED_VAL_LOSS)
    rounded_match = (
        round(observed, REQUIRED_DECIMALS)
        == round(EXPECTED_VAL_LOSS, REQUIRED_DECIMALS)
    )

    assert targets == VALIDATION_TARGETS == 256_512, (
        f"D-096 Gate 2 BLOCKED: validation targets={targets}, expected=256512"
    )
    assert rounded_match, (
        f"D-096 Gate 2 BLOCKED: observed validation loss {observed:.12f} does not "
        f"reproduce {EXPECTED_VAL_LOSS:.6f} to {REQUIRED_DECIMALS} decimal places"
    )

    result = {
        "gate": "D-096 Gate 2 — behavioral checkpoint identity",
        "gate_passed": True,
        "training_updates_executed": 0,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint": checkpoint_audit,
        "runtime": {"gpu_name": gpu_name, "precision": policy.precision},
        "model_parameters": actual_parameters,
        "expected_validation_loss": EXPECTED_VAL_LOSS,
        "observed_validation_loss": observed,
        "absolute_difference": absolute_difference,
        "required_decimal_places": REQUIRED_DECIMALS,
        "validation_perplexity": math.exp(observed),
        "validation_targets": targets,
    }

    output = args.output or Path(
        "results/extended_training/model_c/d096_gate2_resume_identity.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("D-096 GATE 2: PASS")
    print("Checkpoint:", checkpoint_path)
    print("Global update:", checkpoint_audit["global_update"])
    print("Completed epochs:", checkpoint_audit["completed_full_epochs"])
    print("GPU / precision:", gpu_name, "/", policy.precision)
    print("Validation targets:", targets)
    print("Expected validation loss:", f"{EXPECTED_VAL_LOSS:.12f}")
    print("Observed validation loss:", f"{observed:.12f}")
    print("Absolute difference:", f"{absolute_difference:.12g}")
    print("Observed perplexity:", f"{math.exp(observed):.12f}")
    print("Optimizer updates executed: 0")
    print("STOP: Gate 3 has not been executed; update 3,664 is unreachable from this script.")


if __name__ == "__main__":
    main()
