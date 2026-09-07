#!/usr/bin/env python3
"""Notebook 06A Model C extension runner with exact 06A resume semantics.

DISARMED by default. Without --execute, this validates whether the next run would
start from frozen update 3,663 or resume from the separate 06A latest checkpoint,
then exits with zero optimizer updates.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import MODEL_CONFIGS, DecoderOnlyLM, analytical_parameter_count
from src.training_pipeline import (
    CANONICAL_TOKENIZER_SHA256, CANONICAL_TRAIN_STREAM_SHA256,
    OPTIMIZER_UPDATES_PER_EPOCH, PRODUCTION_MICRO_BATCH_SEQUENCES, SEED,
    SEQUENCES_PER_FULL_UPDATE, VALIDATION_TARGETS, atomic_torch_save,
    build_adamw_optimizer, capture_rng_state, evaluate_language_model,
    iter_effective_batch_pieces, load_or_build_canonical_datasets,
    make_epoch_dataloader, make_grad_scaler, make_validation_dataloader,
    move_optimizer_state_to_device, resolve_runtime_precision_policy,
    restore_rng_state, set_optimizer_learning_rate, train_one_optimizer_update,
    write_json_artifact,
)

MODEL_KEY = "C"
EXPECTED_PARAMETERS = 33_497_600
FROZEN_UPDATE = 3_663
FROZEN_EPOCHS = 3
FROZEN_VAL_LOSS = 3.684501
EXT_LR = 2e-4
FIRST_EPOCH = 4
MAX_ADDITIONAL_EPOCHS = 10
MAX_GLOBAL_UPDATE = 15_873
VAL_EVERY = 200
MIN_DELTA = 0.001
PATIENCE = 6


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paths(root: Path) -> dict[str, Path]:
    d = root / "extended_training" / "model_c"
    return {
        "root": d,
        "latest": d / "model_c_extension_latest.pt",
        "best": d / "model_c_extension_best.pt",
        "history": d / "extension_history.json",
        "summary": d / "extension_summary.json",
        "progress": d / "extension_progress.json",
    }


def make_ckpt(*, model, optimizer, scaler, global_update, epoch_number,
              updates_in_epoch, ext_updates, history, val_history,
              best_loss, best_update, material_ref, patience_count,
              targets, elapsed, stop_reason, gpu_name, precision):
    return {
        "format_version": 2,
        "experiment": "06A_model_c_extended_training_probe",
        "model_key": MODEL_KEY,
        "frozen_parent_update": FROZEN_UPDATE,
        "global_update": int(global_update),
        "extension_epoch_number": int(epoch_number),
        "updates_completed_in_epoch": int(updates_in_epoch),
        "extension_updates": int(ext_updates),
        "total_extension_targets": int(targets),
        "constant_learning_rate": EXT_LR,
        "best_validation_loss": float(best_loss),
        "best_validation_update": int(best_update),
        "material_reference_loss": float(material_ref),
        "patience_count": int(patience_count),
        "history": history,
        "validation_history": val_history,
        "elapsed_seconds": float(elapsed),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--persistent-root", type=Path,
                    default=Path("/content/drive/MyDrive/foundation-model-from-scratch/production"))
    ap.add_argument("--gate-artifact", type=Path,
                    default=REPO_ROOT / "results/extended_training/model_c/d096_resume_gate.json")
    args = ap.parse_args()

    gate = load_json(args.gate_artifact)
    assert gate["decision"] == "D-096" and gate["gate_passed"] is True
    assert gate["training_authorized_by_gate"] is True
    c = gate["extension_contract"]
    assert float(c["constant_learning_rate"]) == EXT_LR
    assert int(c["first_epoch"]) == FIRST_EPOCH
    assert int(c["epoch4_shuffle_seed"]) == 46
    assert float(c["min_delta"]) == MIN_DELTA
    assert int(c["patience_validation_events"]) == PATIENCE
    assert int(c["max_additional_epochs"]) == MAX_ADDITIONAL_EPOCHS
    assert int(c["max_global_update"]) == MAX_GLOBAL_UPDATE

    assert torch.cuda.is_available(), "06A BLOCKED: CUDA required"
    gpu_name = torch.cuda.get_device_name(0)
    assert gpu_name == "Tesla T4", f"06A BLOCKED: expected Tesla T4, got {gpu_name!r}"
    policy = resolve_runtime_precision_policy()
    assert policy.precision == "fp16" and policy.use_grad_scaler

    root = args.persistent_root.resolve()
    p = paths(root)
    p["root"].mkdir(parents=True, exist_ok=True)
    source_path = root / "checkpoints" / "model_c_latest.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    assert source["format_version"] == 2 and source["model_key"] == MODEL_KEY
    assert int(source["global_update"]) == FROZEN_UPDATE
    assert int(source["completed_full_epochs"]) == FROZEN_EPOCHS
    assert source["tokenizer_sha256"] == CANONICAL_TOKENIZER_SHA256
    assert source["train_stream_sha256"] == CANONICAL_TRAIN_STREAM_SHA256

    cfg = MODEL_CONFIGS[MODEL_KEY]
    assert analytical_parameter_count(cfg)["total"] == EXPECTED_PARAMETERS
    model = DecoderOnlyLM(cfg).cuda()
    optimizer = build_adamw_optimizer(model, learning_rate=EXT_LR)
    scaler = make_grad_scaler(policy)

    datasets = load_or_build_canonical_datasets(root)
    train_ds = datasets["training_dataset"]
    val_ds = datasets["validation_dataset"]
    val_loader = make_validation_dataloader(val_ds, batch_size=32)
    assert len(train_ds) == 39_062
    assert sum(y.numel() for _x, y in val_loader) == VALIDATION_TARGETS == 256_512

    resume = p["latest"].exists()
    if resume:
        ck = torch.load(p["latest"], map_location="cpu", weights_only=False)
        assert ck.get("format_version") == 2
        assert ck.get("experiment") == "06A_model_c_extended_training_probe"
        assert ck.get("model_key") == MODEL_KEY
        assert int(ck.get("frozen_parent_update")) == FROZEN_UPDATE
        assert float(ck.get("constant_learning_rate")) == EXT_LR
        assert ck.get("tokenizer_sha256") == CANONICAL_TOKENIZER_SHA256
        assert ck.get("train_stream_sha256") == CANONICAL_TRAIN_STREAM_SHA256
        model.load_state_dict(ck["model_state_dict"], strict=True)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, torch.device("cuda"))
        scaler.load_state_dict(ck["scaler_state_dict"])
        set_optimizer_learning_rate(optimizer, EXT_LR)
        restore_rng_state(ck["rng_state"])

        global_update = int(ck["global_update"])
        epoch_number = int(ck["extension_epoch_number"])
        updates_in_epoch = int(ck["updates_completed_in_epoch"])
        ext_updates = int(ck["extension_updates"])
        targets = int(ck["total_extension_targets"])
        history = list(ck["history"])
        val_history = list(ck["validation_history"])
        best_loss = float(ck["best_validation_loss"])
        best_update = int(ck["best_validation_update"])
        material_ref = float(ck["material_reference_loss"])
        patience_count = int(ck["patience_count"])
        elapsed_before = float(ck.get("elapsed_seconds", 0.0))
        assert global_update == FROZEN_UPDATE + ext_updates
        assert 0 <= updates_in_epoch <= OPTIMIZER_UPDATES_PER_EPOCH
        if updates_in_epoch == OPTIMIZER_UPDATES_PER_EPOCH:
            epoch_number += 1
            updates_in_epoch = 0
    else:
        model.load_state_dict(source["model_state_dict"], strict=True)
        optimizer.load_state_dict(source["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, torch.device("cuda"))
        scaler.load_state_dict(source["scaler_state_dict"])
        set_optimizer_learning_rate(optimizer, EXT_LR)
        restore_rng_state(source["rng_state"])
        baseline = evaluate_language_model(model, val_loader, policy)
        baseline_loss = float(baseline["validation_loss"])
        assert round(baseline_loss, 5) == round(FROZEN_VAL_LOSS, 5)
        global_update, epoch_number, updates_in_epoch = FROZEN_UPDATE, FIRST_EPOCH, 0
        ext_updates, targets = 0, 0
        history, val_history = [], []
        best_loss, best_update = baseline_loss, FROZEN_UPDATE
        material_ref, patience_count, elapsed_before = baseline_loss, 0, 0.0

    print("NOTEBOOK 06A RESUMABLE EXTENSION: PREFLIGHT PASS")
    print("Mode:", "RESUME" if resume else "FRESH EXTENSION START")
    print("Current global update:", global_update)
    print("Current extension epoch / updates in epoch:", epoch_number, "/", updates_in_epoch)
    print("Extension updates already persisted:", ext_updates)
    print("Constant LR:", EXT_LR)
    if not args.execute:
        print("Execution flag: ABSENT")
        print("Optimizer updates executed this invocation: 0")
        print("STOP: resumable runner is DISARMED by default.")
        return

    print("Execution flag: PRESENT -- training/resume is starting")
    start = time.perf_counter()
    stop_reason = None

    while epoch_number <= FIRST_EPOCH + MAX_ADDITIONAL_EPOCHS - 1:
        loader = make_epoch_dataloader(train_ds, micro_batch_size=32,
                                       epoch=epoch_number, seed=SEED)
        groups = iter_effective_batch_pieces(loader, SEQUENCES_PER_FULL_UPDATE)
        for _ in range(updates_in_epoch):
            try:
                next(groups)
            except StopIteration as exc:
                raise RuntimeError("06A resume position exceeds epoch length") from exc

        for pieces in groups:
            global_update += 1
            ext_updates += 1
            updates_in_epoch += 1
            assert global_update == FROZEN_UPDATE + ext_updates
            assert global_update <= MAX_GLOBAL_UPDATE

            m = train_one_optimizer_update(model, optimizer, scaler, policy, pieces, EXT_LR)
            targets += int(m["training_targets"])
            rec = {"epoch": epoch_number, "update": global_update,
                   "extension_update": ext_updates, **m}

            validate = (ext_updates % VAL_EVERY == 0 or
                        updates_in_epoch == OPTIMIZER_UPDATES_PER_EPOCH)
            if validate:
                v = evaluate_language_model(model, val_loader, policy)
                val_loss = float(v["validation_loss"])
                rec.update(v)
                strict_best = val_loss < best_loss
                if strict_best:
                    best_loss, best_update = val_loss, global_update
                material = val_loss <= material_ref - MIN_DELTA
                if material:
                    material_ref, patience_count = val_loss, 0
                else:
                    patience_count += 1
                val_history.append({"epoch": epoch_number, "update": global_update,
                    "extension_update": ext_updates, **v,
                    "strict_best": strict_best, "material_improvement": material,
                    "material_reference_loss": material_ref,
                    "patience_count": patience_count})
                history.append(rec)
                elapsed = elapsed_before + (time.perf_counter() - start)
                payload = make_ckpt(model=model, optimizer=optimizer, scaler=scaler,
                    global_update=global_update, epoch_number=epoch_number,
                    updates_in_epoch=updates_in_epoch, ext_updates=ext_updates,
                    history=history, val_history=val_history, best_loss=best_loss,
                    best_update=best_update, material_ref=material_ref,
                    patience_count=patience_count, targets=targets, elapsed=elapsed,
                    stop_reason=None, gpu_name=gpu_name, precision=policy.precision)
                atomic_torch_save(payload, p["latest"])
                if strict_best:
                    atomic_torch_save(payload, p["best"])
                write_json_artifact({"status":"in_progress","global_update":global_update,
                    "extension_update":ext_updates,"epoch":epoch_number,
                    "updates_completed_in_epoch":updates_in_epoch,
                    "validation_loss":val_loss,"best_validation_loss":best_loss,
                    "best_validation_update":best_update,"patience_count":patience_count,
                    "saved_at_utc":datetime.now(timezone.utc).isoformat()}, p["progress"])
                print(f"06A | epoch {epoch_number:2d} | global {global_update:5d} | ext {ext_updates:5d} | val={val_loss:.6f} | best={best_loss:.6f}@{best_update} | patience={patience_count}/{PATIENCE}")
                if patience_count >= PATIENCE:
                    stop_reason = "early_stopping_patience_exhausted"
                    break
            else:
                history.append(rec)

        if stop_reason:
            break
        assert updates_in_epoch == OPTIMIZER_UPDATES_PER_EPOCH
        epoch_number += 1
        updates_in_epoch = 0

    if stop_reason is None:
        stop_reason = "maximum_extension_range_reached"

    elapsed = elapsed_before + (time.perf_counter() - start)
    summary = {
        "experiment":"06A_model_c_extended_training_probe",
        "model_key":MODEL_KEY,"frozen_parent_update":FROZEN_UPDATE,
        "final_global_update":global_update,"extension_updates":ext_updates,
        "total_extension_targets":targets,"constant_learning_rate":EXT_LR,
        "best_extension_validation_loss":best_loss,
        "best_extension_validation_update":best_update,
        "validation_events":len(val_history),"patience_count_at_stop":patience_count,
        "stop_reason":stop_reason,
        "saturation_observed":stop_reason=="early_stopping_patience_exhausted",
        "elapsed_seconds":elapsed,"official_test_split_content_used":False,
        "resume_capable":True,
    }
    write_json_artifact({"history":history,"validation_history":val_history}, p["history"])
    write_json_artifact(summary, p["summary"])
    write_json_artifact({"status":"complete","global_update":global_update,
        "stop_reason":stop_reason,"saved_at_utc":datetime.now(timezone.utc).isoformat()}, p["progress"])
    print("NOTEBOOK 06A EXTENSION: COMPLETE")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
