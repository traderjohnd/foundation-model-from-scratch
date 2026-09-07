#!/usr/bin/env python3
"""Notebook 06A Model C extension preflight — execution locked.

This script prepares and verifies the exact continuation state required by
D-095/D-096 but deliberately contains no training call. It cannot execute
optimizer update 3,664.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import MODEL_CONFIGS, DecoderOnlyLM, analytical_parameter_count
from src.training_pipeline import (
    CANONICAL_TOKENIZER_SHA256,
    CANONICAL_TRAIN_STREAM_SHA256,
    OPTIMIZER_UPDATES_PER_EPOCH,
    PRODUCTION_MICRO_BATCH_SEQUENCES,
    SEED,
    VALIDATION_EXAMPLES,
    VALIDATION_TARGETS,
    build_adamw_optimizer,
    load_or_build_canonical_datasets,
    make_epoch_dataloader,
    make_grad_scaler,
    make_validation_dataloader,
    move_optimizer_state_to_device,
    resolve_runtime_precision_policy,
    restore_rng_state,
    set_optimizer_learning_rate,
)

MODEL_KEY = "C"
EXPECTED_PARAMETERS = 33_497_600
FROZEN_GLOBAL_UPDATE = 3_663
FROZEN_COMPLETED_EPOCHS = 3
FROZEN_EPOCH_INDEX = 2
FROZEN_UPDATES_IN_EPOCH = 1_221
EXTENSION_FIRST_EPOCH_NUMBER = 4
EXTENSION_FIRST_EPOCH_INDEX = 4  # canonical loader API is intentionally fed epoch number
EXTENSION_SHUFFLE_SEED = 46
EXTENSION_LR = 2e-4
MAX_ADDITIONAL_EPOCHS = 10
MAX_GLOBAL_UPDATE = 15_873
MIN_DELTA = 0.001
PATIENCE_VALIDATION_EVENTS = 6
EXPECTED_CORPUS_MANIFEST_SHA256 = (
    "4a00196b39311a6c2e2790780e8fc43316f24a014d3d3649028b10a671f8d3fe"
)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persistent-root",
        type=Path,
        default=Path("/content/drive/MyDrive/foundation-model-from-scratch/production"),
    )
    parser.add_argument(
        "--gate-artifact",
        type=Path,
        default=REPO_ROOT / "results/extended_training/model_c/d096_resume_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/extended_training/model_c/extension_preflight.json",
    )
    args = parser.parse_args()

    # D-096 is the hard authorization boundary.
    gate = load_json(args.gate_artifact)
    assert gate.get("decision") == "D-096"
    assert gate.get("gate_passed") is True
    assert gate.get("training_authorized_by_gate") is True
    assert gate.get("training_started") is False
    assert int(gate.get("next_optimizer_update_if_explicitly_started")) == 3_664

    # Freeze the D-095 extension contract before any mutable runtime work.
    contract = gate["extension_contract"]
    assert int(contract["first_epoch"]) == EXTENSION_FIRST_EPOCH_NUMBER
    assert int(contract["epoch4_shuffle_seed"]) == EXTENSION_SHUFFLE_SEED
    assert float(contract["constant_learning_rate"]) == EXTENSION_LR
    assert float(contract["min_delta"]) == MIN_DELTA
    assert int(contract["patience_validation_events"]) == PATIENCE_VALIDATION_EVENTS
    assert int(contract["max_additional_epochs"]) == MAX_ADDITIONAL_EPOCHS
    assert int(contract["max_global_update"]) == MAX_GLOBAL_UPDATE

    assert torch.cuda.is_available(), "06A preflight BLOCKED: CUDA required"
    gpu_name = torch.cuda.get_device_name(0)
    assert gpu_name == "Tesla T4", f"06A preflight BLOCKED: expected Tesla T4, got {gpu_name!r}"
    policy = resolve_runtime_precision_policy()
    assert policy.precision == "fp16" and policy.use_grad_scaler is True

    persistent_root = args.persistent_root.resolve()
    checkpoint_path = persistent_root / "checkpoints" / "model_c_latest.pt"
    assert checkpoint_path.exists(), f"06A preflight BLOCKED: missing {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Exact frozen boundary.
    assert checkpoint.get("format_version") == 2
    assert checkpoint.get("model_key") == MODEL_KEY
    assert int(checkpoint.get("global_update")) == FROZEN_GLOBAL_UPDATE
    assert int(checkpoint.get("completed_full_epochs")) == FROZEN_COMPLETED_EPOCHS
    assert int(checkpoint.get("epoch_index")) == FROZEN_EPOCH_INDEX
    assert int(checkpoint.get("updates_completed_in_epoch")) == FROZEN_UPDATES_IN_EPOCH
    assert int(checkpoint.get("seed")) == SEED == 42
    assert checkpoint.get("tokenizer_sha256") == CANONICAL_TOKENIZER_SHA256
    assert checkpoint.get("train_stream_sha256") == CANONICAL_TRAIN_STREAM_SHA256

    cfg = MODEL_CONFIGS[MODEL_KEY]
    assert analytical_parameter_count(cfg)["total"] == EXPECTED_PARAMETERS
    model = DecoderOnlyLM(cfg).cuda()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS

    optimizer = build_adamw_optimizer(model, learning_rate=EXTENSION_LR)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, torch.device("cuda"))
    set_optimizer_learning_rate(optimizer, EXTENSION_LR)
    assert all(float(group["lr"]) == EXTENSION_LR for group in optimizer.param_groups)

    scaler = make_grad_scaler(policy)
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint["rng_state"])

    datasets = load_or_build_canonical_datasets(persistent_root)
    training_dataset = datasets["training_dataset"]
    validation_dataset = datasets["validation_dataset"]
    assert len(training_dataset) == 39_062
    assert len(validation_dataset) == VALIDATION_EXAMPLES == 501

    # Prepare, but do not iterate, the exact epoch-4 training loader.
    epoch4_loader = make_epoch_dataloader(
        training_dataset,
        micro_batch_size=PRODUCTION_MICRO_BATCH_SEQUENCES[MODEL_KEY],
        epoch=EXTENSION_FIRST_EPOCH_INDEX,
        seed=SEED,
    )
    validation_loader = make_validation_dataloader(
        validation_dataset,
        batch_size=PRODUCTION_MICRO_BATCH_SEQUENCES[MODEL_KEY],
    )
    assert len(epoch4_loader) == (39_062 + 31) // 32
    assert sum(y.numel() for _x, y in validation_loader) == VALIDATION_TARGETS == 256_512
    assert OPTIMIZER_UPDATES_PER_EPOCH == 1_221

    # Corpus-manifest hash was already verified by D-096 Gate 4; assert the same
    # frozen value from the canonical consolidated gate artifact.
    gate4_hashes = gate["component_gates"]["gate4_provenance"]["hashes"]
    assert gate4_hashes["tokenizer_sha256"] == CANONICAL_TOKENIZER_SHA256
    assert gate4_hashes["train_stream_sha256"] == CANONICAL_TRAIN_STREAM_SHA256
    assert gate4_hashes["corpus_manifest_sha256"] == EXPECTED_CORPUS_MANIFEST_SHA256

    result = {
        "artifact": "Notebook 06A Model C extension preflight",
        "preflight_passed": True,
        "execution_locked": True,
        "training_started": False,
        "optimizer_updates_executed": 0,
        "next_optimizer_update_if_explicitly_started": 3_664,
        "checkpoint_path": str(checkpoint_path),
        "runtime": {"gpu_name": gpu_name, "precision": policy.precision},
        "frozen_boundary": {
            "global_update": FROZEN_GLOBAL_UPDATE,
            "completed_epochs": FROZEN_COMPLETED_EPOCHS,
            "model_parameters": EXPECTED_PARAMETERS,
        },
        "extension_contract": {
            "first_epoch_number": EXTENSION_FIRST_EPOCH_NUMBER,
            "effective_shuffle_seed": EXTENSION_SHUFFLE_SEED,
            "constant_learning_rate": EXTENSION_LR,
            "optimizer_updates_per_epoch": OPTIMIZER_UPDATES_PER_EPOCH,
            "max_additional_epochs": MAX_ADDITIONAL_EPOCHS,
            "max_global_update": MAX_GLOBAL_UPDATE,
            "min_delta": MIN_DELTA,
            "patience_validation_events": PATIENCE_VALIDATION_EVENTS,
        },
        "data": {
            "training_examples": len(training_dataset),
            "validation_examples": len(validation_dataset),
            "validation_targets": VALIDATION_TARGETS,
            "cache_status": datasets["cache_status"],
        },
        "explicit_stop": "Preflight only. This script contains no training call; update 3,664 cannot execute.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("NOTEBOOK 06A MODEL C EXTENSION PREFLIGHT: PASS")
    print("D-096 gate: PASS / persisted")
    print("Checkpoint update:", FROZEN_GLOBAL_UPDATE)
    print("Completed epochs:", FROZEN_COMPLETED_EPOCHS)
    print("GPU / precision:", gpu_name, "/", policy.precision)
    print("Extension LR:", EXTENSION_LR)
    print("Epoch 4 effective shuffle seed:", EXTENSION_SHUFFLE_SEED)
    print("Training examples / updates per epoch:", len(training_dataset), "/", OPTIMIZER_UPDATES_PER_EPOCH)
    print("Validation examples / targets:", len(validation_dataset), "/", VALIDATION_TARGETS)
    print("Optimizer updates executed: 0")
    print("STOP: runner is execution-locked; update 3,664 cannot execute from this script.")


if __name__ == "__main__":
    main()
