#!/usr/bin/env python3
"""D-096 Gate 4: scaler/counters/config/hash/RNG/validation provenance audit.

Read-only. Loads the exact frozen Model C v2 checkpoint and verifies the
remaining resume/provenance invariants required before epoch-4 shuffle audit.
No optimizer or scaler step is executed. Update 3,664 is unreachable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import MODEL_CONFIGS, DecoderOnlyLM, analytical_parameter_count
from src.training_pipeline import (
    CANONICAL_TOKENIZER_SHA256,
    CANONICAL_TRAIN_STREAM_SHA256,
    CONTEXT_LENGTH,
    MODEL_PARAMETERS,
    PRODUCTION_MICRO_BATCH_SEQUENCES,
    SEED,
    VALIDATION_EXAMPLES,
    VALIDATION_TARGETS,
    load_or_build_canonical_datasets,
    make_grad_scaler,
    make_validation_dataloader,
    resolve_runtime_precision_policy,
    sha256_bytes,
    sha256_file,
)

EXPECTED_MODEL_KEY = "C"
EXPECTED_PARAMETERS = 33_497_600
EXPECTED_UPDATE = 3_663
EXPECTED_COMPLETED_EPOCHS = 3
EXPECTED_EPOCH_INDEX = 2
EXPECTED_UPDATES_IN_EPOCH = 1_221
EXPECTED_TOKENIZER_SHA256 = "6ec601a267cec7c843df47927f53c4dd108c85a1d059318aeec4442c7274604f"
EXPECTED_TRAIN_STREAM_SHA256 = "4101d5b18c38558a58110f54a161763186ab5318111366486ebbfa0a3fe584fa"
EXPECTED_CORPUS_MANIFEST_SHA256 = "4a00196b39311a6c2e2790780e8fc43316f24a014d3d3649028b10a671f8d3fe"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persistent-root",
        type=Path,
        default=Path("/content/drive/MyDrive/foundation-model-from-scratch/production"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "D-096 Gate 4 BLOCKED: CUDA GPU required"
    gpu_name = torch.cuda.get_device_name(0)
    assert gpu_name == "Tesla T4", f"D-096 Gate 4 BLOCKED: expected Tesla T4, got {gpu_name!r}"

    checkpoint_path = args.persistent_root.resolve() / "checkpoints" / "model_c_latest.pt"
    assert checkpoint_path.exists(), f"D-096 Gate 4 BLOCKED: missing {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Counters and model identity.
    assert checkpoint.get("format_version") == 2
    assert checkpoint.get("model_key") == EXPECTED_MODEL_KEY
    assert int(checkpoint.get("global_update")) == EXPECTED_UPDATE
    assert int(checkpoint.get("completed_full_epochs")) == EXPECTED_COMPLETED_EPOCHS
    assert int(checkpoint.get("epoch_index")) == EXPECTED_EPOCH_INDEX
    assert int(checkpoint.get("updates_completed_in_epoch")) == EXPECTED_UPDATES_IN_EPOCH
    assert int(checkpoint.get("seed")) == SEED == 42
    assert int(checkpoint.get("micro_batch_sequences")) == PRODUCTION_MICRO_BATCH_SEQUENCES["C"] == 32

    cfg = MODEL_CONFIGS[EXPECTED_MODEL_KEY]
    assert analytical_parameter_count(cfg)["total"] == EXPECTED_PARAMETERS
    assert MODEL_PARAMETERS[EXPECTED_MODEL_KEY] == EXPECTED_PARAMETERS
    model = DecoderOnlyLM(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS

    # Canonical T4 mixed-precision / GradScaler continuity.
    policy = resolve_runtime_precision_policy()
    assert policy.precision == "fp16" and policy.use_grad_scaler is True
    scaler_state = checkpoint.get("scaler_state_dict")
    assert isinstance(scaler_state, dict) and scaler_state, "D-096 Gate 4 BLOCKED: missing/empty GradScaler state"
    scaler = make_grad_scaler(policy)
    scaler.load_state_dict(scaler_state)

    # Persisted RNG state required by Notebook 04 wrapper.
    rng_state = checkpoint.get("rng_state")
    assert isinstance(rng_state, dict), "D-096 Gate 4 BLOCKED: rng_state missing"
    cpu_rng = rng_state.get("torch_cpu_rng_state")
    cuda_rng = rng_state.get("torch_cuda_rng_state_all")
    assert torch.is_tensor(cpu_rng) and cpu_rng.numel() > 0
    assert isinstance(cuda_rng, (list, tuple)) and len(cuda_rng) > 0
    assert all(torch.is_tensor(x) and x.numel() > 0 for x in cuda_rng)

    # Tokenizer, corpus stream, and corpus-manifest provenance.
    tokenizer_path = REPO_ROOT / "results" / "tokenizer" / "tokenizer.json"
    manifest_path = REPO_ROOT / "results" / "corpus" / "corpus_manifest.jsonl"
    assert tokenizer_path.exists() and manifest_path.exists()
    tokenizer_sha = sha256_file(tokenizer_path)
    manifest_sha = sha256_file(manifest_path)
    assert tokenizer_sha == EXPECTED_TOKENIZER_SHA256 == CANONICAL_TOKENIZER_SHA256
    assert manifest_sha == EXPECTED_CORPUS_MANIFEST_SHA256
    assert checkpoint.get("tokenizer_sha256") == EXPECTED_TOKENIZER_SHA256
    assert checkpoint.get("train_stream_sha256") == EXPECTED_TRAIN_STREAM_SHA256

    datasets = load_or_build_canonical_datasets(args.persistent_root.resolve())
    train_stream = np.asarray(datasets["training_stream"], dtype="<u2")
    train_sha = sha256_bytes(train_stream.tobytes())
    assert train_sha == EXPECTED_TRAIN_STREAM_SHA256 == CANONICAL_TRAIN_STREAM_SHA256

    # D-072 validation provenance: same packing/counts/no-shuffle loader construction.
    validation_dataset = datasets["validation_dataset"]
    assert len(validation_dataset) == VALIDATION_EXAMPLES == 501
    assert CONTEXT_LENGTH == 512
    validation_loader = make_validation_dataloader(validation_dataset, batch_size=32)
    validation_targets = sum(int(y.numel()) for _x, y in validation_loader)
    assert validation_targets == VALIDATION_TARGETS == 256_512

    result = {
        "gate": "D-096 Gate 4 — scaler/counters/config/provenance",
        "gate_passed": True,
        "training_updates_executed": 0,
        "checkpoint_path": str(checkpoint_path),
        "runtime": {"gpu_name": gpu_name, "precision": policy.precision},
        "checkpoint": {
            "format_version": 2,
            "model_key": EXPECTED_MODEL_KEY,
            "global_update": EXPECTED_UPDATE,
            "completed_full_epochs": EXPECTED_COMPLETED_EPOCHS,
            "epoch_index": EXPECTED_EPOCH_INDEX,
            "updates_completed_in_epoch": EXPECTED_UPDATES_IN_EPOCH,
            "seed": SEED,
        },
        "model_parameters": EXPECTED_PARAMETERS,
        "grad_scaler_state_present": True,
        "rng_state": {
            "torch_cpu_rng_state_present": True,
            "torch_cuda_rng_state_count": len(cuda_rng),
        },
        "hashes": {
            "tokenizer_sha256": tokenizer_sha,
            "train_stream_sha256": train_sha,
            "corpus_manifest_sha256": manifest_sha,
        },
        "validation_provenance": {
            "context_length": CONTEXT_LENGTH,
            "examples": len(validation_dataset),
            "targets": validation_targets,
            "shuffle": False,
        },
    }

    output = args.output or Path("results/extended_training/model_c/d096_gate4_provenance.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("D-096 GATE 4: PASS")
    print("Checkpoint:", checkpoint_path)
    print("Global update:", EXPECTED_UPDATE)
    print("Completed epochs:", EXPECTED_COMPLETED_EPOCHS)
    print("Model parameters:", EXPECTED_PARAMETERS)
    print("GPU / precision:", gpu_name, "/", policy.precision)
    print("GradScaler state: PRESENT")
    print("CPU RNG state: PRESENT")
    print("CUDA RNG state count:", len(cuda_rng))
    print("Tokenizer SHA-256:", tokenizer_sha)
    print("Train stream SHA-256:", train_sha)
    print("Corpus manifest SHA-256:", manifest_sha)
    print("Validation examples / targets:", len(validation_dataset), "/", validation_targets)
    print("Optimizer updates executed: 0")
    print("STOP: Gate 5 has not been executed; update 3,664 is unreachable from this script.")


if __name__ == "__main__":
    main()
