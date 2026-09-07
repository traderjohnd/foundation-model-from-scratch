#!/usr/bin/env python3
"""Notebook 06A Model C extended-training probe.

D-095 / D-096 exploratory continuation of frozen Model C. The runner is
DISARMED by default: without --execute it performs exact-resume preflight and
exits before optimizer update 3,664. With --execute it resumes the exact v2
production checkpoint and trains only in the separate 06A namespace.

Frozen Notebook 05 A/B/C evidence is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
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
    SEQUENCES_PER_FULL_UPDATE,
    VALIDATION_TARGETS,
    atomic_torch_save,
    build_adamw_optimizer,
    capture_rng_state,
    evaluate_language_model,
    iter_effective_batch_pieces,
    load_or_build_canonical_datasets,
    make_epoch_dataloader,
    make_grad_scaler,
    make_validation_dataloader,
    move_optimizer_state_to_device,
    resolve_runtime_precision_policy,
    restore_rng_state,
    set_optimizer_learning_rate,
    train_one_optimizer_update,
    write_json_artifact,
)

MODEL_KEY = "C"
EXPECTED_PARAMETERS = 33_497_600
FROZEN_GLOBAL_UPDATE = 3_663
FROZEN_COMPLETED_EPOCHS = 3
FROZEN_VAL_LOSS = 3.684501
EXTENSION_LR = 2e-4
EXTENSION_FIRST_EPOCH_NUMBER = 4
EXTENSION_FIRST_LOADER_EPOCH = 4  # D-073 / D-096: seed 42 + 4 = 46
MAX_ADDITIONAL_EPOCHS = 10
MAX_GLOBAL_UPDATE = 15_873
VALIDATION_EVERY_EXTENSION_UPDATES = 200
MIN_DELTA = 0.001
PATIENCE = 6
EXPECTED_FIRST_10 = [23330, 14720, 12892, 24465, 36182, 35545, 35235, 2313, 29923, 36592]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def extension_paths(persistent_root: Path) -> dict[str, Path]:
    root = persistent_root / "extended_training" / "model_c"
    return {
        "root": root,
        "latest": root / "model_c_extension_latest.pt",
        "best": root / "model_c_extension_best.pt",
        "history": root / "extension_history.json",
        "summary": root / "extension_summary.json",
        "progress": root / "extension_progress.json",
    }


def make_extension_checkpoint(*, model, optimizer, scaler, global_update, extension_epoch_number,
                              updates_completed_in_epoch, extension_updates, history,
                              validation_history, best_validation_loss, best_validation_update,
                              material_reference_loss, patience_count, total_extension_targets,
                              elapsed_seconds, stop_reason, gpu_name, precision):
    return {
        "format_version": 1,
        "project": "foundation-model-from-scratch",
        "experiment": "06A_model_c_extended_training_probe",
        "model_key": MODEL_KEY,
        "frozen_parent_update": FROZEN_GLOBAL_UPDATE,
        "global_update": int(global_update),
        "extension_epoch_number": int(extension_epoch_number),
        "updates_completed_in_epoch": int(updates_completed_in_epoch),
        "extension_updates": int(extension_updates),
        "total_extension_targets": int(total_extension_targets),
        "constant_learning_rate": EXTENSION_LR,
        "best_validation_loss": float(best_validation_loss),
        "best_validation_update": int(best_validation_update),
        "material_reference_loss": float(material_reference_loss),
        "patience_count": int(patience_count),
        "history": history,
        "validation_history": validation_history,
        "elapsed_seconds": float(elapsed_seconds),
        "stop_reason": stop_reason,
        "gpu_name": gpu_name,
        "precision": precision,
        "tokenizer_sha256": CANONICAL_TOKENIZER_SHA256,
        "train_stream_sha256": CANONICAL_TRAIN_STREAM_SHA256,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize update 3,664 and later extension updates")
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
    args = parser.parse_args()

    gate = load_json(args.gate_artifact)
    assert gate.get("decision") == "D-096"
    assert gate.get("gate_passed") is True
    assert gate.get("training_authorized_by_gate") is True
    assert gate.get("training_started") is False
    assert int(gate.get("next_optimizer_update_if_explicitly_started")) == 3_664
    contract = gate["extension_contract"]
    assert float(contract["constant_learning_rate"]) == EXTENSION_LR
    assert int(contract["first_epoch"]) == EXTENSION_FIRST_EPOCH_NUMBER
    assert int(contract["epoch4_shuffle_seed"]) == 46
    assert float(contract["min_delta"]) == MIN_DELTA
    assert int(contract["patience_validation_events"]) == PATIENCE
    assert int(contract["max_additional_epochs"]) == MAX_ADDITIONAL_EPOCHS
    assert int(contract["max_global_update"]) == MAX_GLOBAL_UPDATE

    assert torch.cuda.is_available(), "06A extension BLOCKED: CUDA required"
    gpu_name = torch.cuda.get_device_name(0)
    assert gpu_name == "Tesla T4", f"06A extension BLOCKED: expected Tesla T4, got {gpu_name!r}"
    policy = resolve_runtime_precision_policy()
    assert policy.precision == "fp16" and policy.use_grad_scaler is True

    persistent_root = args.persistent_root.resolve()
    source_checkpoint_path = persistent_root / "checkpoints" / "model_c_latest.pt"
    assert source_checkpoint_path.exists(), source_checkpoint_path
    source = torch.load(source_checkpoint_path, map_location="cpu", weights_only=False)
    assert source.get("format_version") == 2
    assert source.get("model_key") == MODEL_KEY
    assert int(source.get("global_update")) == FROZEN_GLOBAL_UPDATE
    assert int(source.get("completed_full_epochs")) == FROZEN_COMPLETED_EPOCHS
    assert int(source.get("epoch_index")) == 2
    assert int(source.get("updates_completed_in_epoch")) == OPTIMIZER_UPDATES_PER_EPOCH == 1_221
    assert source.get("tokenizer_sha256") == CANONICAL_TOKENIZER_SHA256
    assert source.get("train_stream_sha256") == CANONICAL_TRAIN_STREAM_SHA256

    cfg = MODEL_CONFIGS[MODEL_KEY]
    assert analytical_parameter_count(cfg)["total"] == EXPECTED_PARAMETERS
    model = DecoderOnlyLM(cfg).cuda()
    model.load_state_dict(source["model_state_dict"], strict=True)
    optimizer = build_adamw_optimizer(model, learning_rate=EXTENSION_LR)
    optimizer.load_state_dict(source["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, torch.device("cuda"))
    set_optimizer_learning_rate(optimizer, EXTENSION_LR)
    scaler = make_grad_scaler(policy)
    scaler.load_state_dict(source["scaler_state_dict"])
    restore_rng_state(source["rng_state"])

    datasets = load_or_build_canonical_datasets(persistent_root)
    training_dataset = datasets["training_dataset"]
    validation_dataset = datasets["validation_dataset"]
    assert len(training_dataset) == 39_062
    validation_loader = make_validation_dataloader(validation_dataset, batch_size=32)
    assert sum(y.numel() for _x, y in validation_loader) == VALIDATION_TARGETS == 256_512

    # Behavioral identity is rechecked immediately before any possible training.
    baseline_metrics = evaluate_language_model(model, validation_loader, policy)
    baseline_loss = float(baseline_metrics["validation_loss"])
    assert round(baseline_loss, 5) == round(FROZEN_VAL_LOSS, 5)

    # Verify the exact first epoch order without mutating model/optimizer state.
    probe_loader = make_epoch_dataloader(training_dataset, micro_batch_size=32,
                                         epoch=EXTENSION_FIRST_LOADER_EPOCH, seed=SEED)
    first_batch_x, _first_batch_y = next(iter(probe_loader))
    # Gate 5 is canonical evidence for the exact example-index order; this runner
    # relies on the persisted gate rather than reverse-mapping token tensors.
    gate5 = gate["component_gates"]["gate5_shuffle_continuity"]
    assert gate5["first_10_example_indices"] == EXPECTED_FIRST_10
    assert int(gate5["effective_shuffle_seed"]) == 46
    assert first_batch_x.shape[0] == 32

    print("NOTEBOOK 06A MODEL C EXTENSION RUNNER: PREFLIGHT PASS")
    print("Frozen update:", FROZEN_GLOBAL_UPDATE)
    print("Baseline validation loss:", f"{baseline_loss:.12f}")
    print("Constant extension LR:", EXTENSION_LR)
    print("Early stopping: min_delta=0.001, patience=6 validation events")
    print("Ceiling: 10 additional epochs / global update 15873")

    if not args.execute:
        print("Execution flag: ABSENT")
        print("Optimizer updates executed: 0")
        print("STOP: runner is DISARMED by default; rerun with --execute only after explicit approval.")
        return

    print("Execution flag: PRESENT -- extension training is starting")

    paths = extension_paths(persistent_root)
    paths["root"].mkdir(parents=True, exist_ok=True)

    # 06A starts from the frozen checkpoint. If a 06A latest checkpoint exists,
    # fail closed for now rather than silently creating ambiguous resume semantics.
    assert not paths["latest"].exists(), (
        "06A extension latest checkpoint already exists. Refusing a second start; "
        "resume handling must be explicit."
    )

    global_update = FROZEN_GLOBAL_UPDATE
    extension_updates = 0
    total_extension_targets = 0
    history = []
    validation_history = []
    best_validation_loss = baseline_loss
    best_validation_update = FROZEN_GLOBAL_UPDATE
    material_reference_loss = baseline_loss
    patience_count = 0
    stop_reason = None
    start_time = time.perf_counter()

    for extra_epoch_offset in range(MAX_ADDITIONAL_EPOCHS):
        epoch_number = EXTENSION_FIRST_EPOCH_NUMBER + extra_epoch_offset
        loader_epoch_value = EXTENSION_FIRST_LOADER_EPOCH + extra_epoch_offset
        loader = make_epoch_dataloader(
            training_dataset,
            micro_batch_size=PRODUCTION_MICRO_BATCH_SEQUENCES[MODEL_KEY],
            epoch=loader_epoch_value,
            seed=SEED,
        )
        update_groups = iter_effective_batch_pieces(
            loader,
            sequences_per_full_update=SEQUENCES_PER_FULL_UPDATE,
        )
        updates_completed_in_epoch = 0

        for pieces in update_groups:
            global_update += 1
            extension_updates += 1
            updates_completed_in_epoch += 1
            assert global_update <= MAX_GLOBAL_UPDATE

            metrics = train_one_optimizer_update(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                policy=policy,
                batch_pieces=pieces,
                learning_rate=EXTENSION_LR,
            )
            total_extension_targets += int(metrics["training_targets"])
            record = {
                "epoch": epoch_number,
                "update": global_update,
                "extension_update": extension_updates,
                **metrics,
            }

            should_validate = (
                extension_updates % VALIDATION_EVERY_EXTENSION_UPDATES == 0
                or updates_completed_in_epoch == OPTIMIZER_UPDATES_PER_EPOCH
            )

            if should_validate:
                val = evaluate_language_model(model, validation_loader, policy)
                val_loss = float(val["validation_loss"])
                record.update(val)
                is_strict_best = val_loss < best_validation_loss
                if is_strict_best:
                    best_validation_loss = val_loss
                    best_validation_update = global_update

                material_improvement = val_loss <= (material_reference_loss - MIN_DELTA)
                if material_improvement:
                    material_reference_loss = val_loss
                    patience_count = 0
                else:
                    patience_count += 1

                event = {
                    "epoch": epoch_number,
                    "update": global_update,
                    "extension_update": extension_updates,
                    **val,
                    "strict_best": is_strict_best,
                    "material_improvement": material_improvement,
                    "material_reference_loss": material_reference_loss,
                    "patience_count": patience_count,
                }
                validation_history.append(event)
                history.append(record)

                elapsed = time.perf_counter() - start_time
                payload = make_extension_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    global_update=global_update,
                    extension_epoch_number=epoch_number,
                    updates_completed_in_epoch=updates_completed_in_epoch,
                    extension_updates=extension_updates,
                    history=history,
                    validation_history=validation_history,
                    best_validation_loss=best_validation_loss,
                    best_validation_update=best_validation_update,
                    material_reference_loss=material_reference_loss,
                    patience_count=patience_count,
                    total_extension_targets=total_extension_targets,
                    elapsed_seconds=elapsed,
                    stop_reason=None,
                    gpu_name=gpu_name,
                    precision=policy.precision,
                )
                atomic_torch_save(payload, paths["latest"])
                if is_strict_best:
                    atomic_torch_save(payload, paths["best"])

                write_json_artifact({
                    "status": "in_progress",
                    "global_update": global_update,
                    "extension_update": extension_updates,
                    "epoch": epoch_number,
                    "validation_loss": val_loss,
                    "validation_perplexity": float(val["validation_perplexity"]),
                    "best_validation_loss": best_validation_loss,
                    "best_validation_update": best_validation_update,
                    "patience_count": patience_count,
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                }, paths["progress"])

                print(
                    f"06A | epoch {epoch_number:2d} | global {global_update:5d} | "
                    f"ext {extension_updates:5d} | val={val_loss:.6f} | "
                    f"ppl={val['validation_perplexity']:.2f} | "
                    f"best={best_validation_loss:.6f}@{best_validation_update} | "
                    f"patience={patience_count}/{PATIENCE}"
                )

                if patience_count >= PATIENCE:
                    stop_reason = "early_stopping_patience_exhausted"
                    break
            else:
                history.append(record)

        if stop_reason is not None:
            break
        assert updates_completed_in_epoch == OPTIMIZER_UPDATES_PER_EPOCH

    if stop_reason is None:
        stop_reason = "maximum_extension_range_reached"

    elapsed = time.perf_counter() - start_time
    summary = {
        "experiment": "06A_model_c_extended_training_probe",
        "model_key": MODEL_KEY,
        "frozen_parent_update": FROZEN_GLOBAL_UPDATE,
        "final_global_update": global_update,
        "extension_updates": extension_updates,
        "additional_epochs_completed_equivalent": extension_updates / OPTIMIZER_UPDATES_PER_EPOCH,
        "total_extension_targets": total_extension_targets,
        "constant_learning_rate": EXTENSION_LR,
        "baseline_validation_loss": baseline_loss,
        "best_extension_validation_loss": best_validation_loss,
        "best_extension_validation_update": best_validation_update,
        "validation_events": len(validation_history),
        "patience_count_at_stop": patience_count,
        "stop_reason": stop_reason,
        "saturation_observed": stop_reason == "early_stopping_patience_exhausted",
        "no_saturation_within_tested_range": stop_reason == "maximum_extension_range_reached",
        "elapsed_seconds": elapsed,
        "official_test_split_content_used": False,
        "frozen_notebook05_results_modified": False,
    }
    write_json_artifact({"history": history, "validation_history": validation_history}, paths["history"])
    write_json_artifact(summary, paths["summary"])
    write_json_artifact({
        "status": "complete",
        "stop_reason": stop_reason,
        "global_update": global_update,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }, paths["progress"])

    print("NOTEBOOK 06A MODEL C EXTENSION: COMPLETE")
    print("Stop reason:", stop_reason)
    print("Final global update:", global_update)
    print("Best validation loss:", f"{best_validation_loss:.12f}", "@", best_validation_update)
    print("Official test split content used: NO")


if __name__ == "__main__":
    main()
