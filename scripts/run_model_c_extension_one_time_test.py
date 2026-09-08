#!/usr/bin/env python3
"""One-time official-test scorer for Notebook 06A Model C extension.

DISARMED by default. Without --execute, verifies the frozen validation-selected
checkpoint at global update 12,210 and that no prior 06A test artifact exists.
With --execute, reconstructs the canonical official WikiText-103 test stream,
asserts the frozen D-092 stream contract, scores the selected checkpoint exactly
once, and persists the result in the separate 06A namespace.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_pinned_wikitext, normalize_wikitext_text, reconstruct_articles
from src.model import MODEL_CONFIGS, DecoderOnlyLM, analytical_parameter_count
from src.training_pipeline import (
    BOUNDARY_TOKEN_ID, CANONICAL_TOKENIZER_SHA256, CausalTokenDataset,
    load_canonical_tokenizer, make_validation_dataloader,
    resolve_runtime_precision_policy, evaluate_language_model,
)

MODEL_KEY = "C"
EXPECTED_PARAMETERS = 33_497_600
SELECTED_UPDATE = 12_210
SELECTED_VAL_LOSS = 3.599946362767629
FROZEN_TEST_MODEL_C_LOSS = 3.6805543749744354
FROZEN_TEST_MODEL_C_PPL = 39.668379
EXPECTED_TEST_ROWS = 4_358
EXPECTED_TEST_ARTICLES = 60
EXPECTED_TEXT_TOKENS = 293_699
EXPECTED_BOUNDARIES = 60
EXPECTED_STREAM_TOKENS = 293_759
EXPECTED_STREAM_SHA256 = "9578e1403a94bf085eb55372e76d4dd74e02c89f7085368c75a1d75f5537d188"
EXPECTED_EXAMPLES = 573
EXPECTED_TARGETS = 293_376
EXPECTED_UNUSED_TAIL = 382


def sha256_u16(arr: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arr, dtype="<u2").tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--persistent-root", type=Path,
                    default=Path("/content/drive/MyDrive/foundation-model-from-scratch/production"))
    args = ap.parse_args()

    assert torch.cuda.is_available(), "06A one-time test BLOCKED: CUDA required"
    gpu_name = torch.cuda.get_device_name(0)
    assert gpu_name == "Tesla T4", f"Expected Tesla T4, got {gpu_name!r}"
    policy = resolve_runtime_precision_policy()

    root = args.persistent_root.resolve()
    ext_root = root / "extended_training" / "model_c"
    best_path = ext_root / "model_c_extension_best.pt"
    summary_path = ext_root / "extension_summary.json"
    output_path = ext_root / "one_time_exploratory_test.json"

    assert best_path.exists(), best_path
    assert summary_path.exists(), summary_path
    assert not output_path.exists(), (
        "06A one-time test artifact already exists. Refusing repeated official-test scoring."
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert int(summary["best_extension_validation_update"]) == SELECTED_UPDATE
    assert abs(float(summary["best_extension_validation_loss"]) - SELECTED_VAL_LOSS) < 1e-12
    assert summary["official_test_split_content_used"] is False
    assert summary["stop_reason"] == "early_stopping_patience_exhausted"

    ck = torch.load(best_path, map_location="cpu", weights_only=False)
    assert int(ck["global_update"]) == SELECTED_UPDATE
    assert abs(float(ck["best_validation_loss"]) - SELECTED_VAL_LOSS) < 1e-12
    assert ck["model_key"] == MODEL_KEY
    assert ck["tokenizer_sha256"] == CANONICAL_TOKENIZER_SHA256

    cfg = MODEL_CONFIGS[MODEL_KEY]
    assert analytical_parameter_count(cfg)["total"] == EXPECTED_PARAMETERS

    print("NOTEBOOK 06A ONE-TIME TEST: PREFLIGHT PASS")
    print("Frozen validation-selected checkpoint:", SELECTED_UPDATE)
    print("Selected validation loss:", f"{SELECTED_VAL_LOSS:.12f}")
    print("Prior 06A test artifact: ABSENT")
    print("Official test split accessed this invocation:", bool(args.execute))

    if not args.execute:
        print("Execution flag: ABSENT")
        print("STOP: official test remains unopened by this invocation.")
        return

    tokenizer = load_canonical_tokenizer(REPO_ROOT / "results/tokenizer/tokenizer.json")
    dataset = load_pinned_wikitext()
    raw_rows = list(dataset["test"]["text"])
    assert len(raw_rows) == EXPECTED_TEST_ROWS
    normalized = [normalize_wikitext_text(x) for x in raw_rows]
    articles = reconstruct_articles(raw_rows, normalized, split_name="test")
    assert len(articles) == EXPECTED_TEST_ARTICLES

    ids = []
    text_tokens = 0
    for article in articles:
        article_ids = tokenizer.encode(article["text"]).ids
        ids.extend(article_ids)
        text_tokens += len(article_ids)
        ids.append(BOUNDARY_TOKEN_ID)

    stream = np.asarray(ids, dtype="<u2")
    assert text_tokens == EXPECTED_TEXT_TOKENS
    assert int(np.count_nonzero(stream == BOUNDARY_TOKEN_ID)) == EXPECTED_BOUNDARIES
    assert len(stream) == EXPECTED_STREAM_TOKENS
    assert sha256_u16(stream) == EXPECTED_STREAM_SHA256

    test_ds = CausalTokenDataset(stream)
    assert len(test_ds) == EXPECTED_EXAMPLES
    assert len(stream) - (len(test_ds) * 512 + 1) == EXPECTED_UNUSED_TAIL
    test_loader = make_validation_dataloader(test_ds, batch_size=32)
    assert sum(y.numel() for _x, y in test_loader) == EXPECTED_TARGETS

    model = DecoderOnlyLM(cfg).cuda()
    model.load_state_dict(ck["model_state_dict"], strict=True)
    metrics = evaluate_language_model(model, test_loader, policy)
    loss = float(metrics["validation_loss"])
    ppl = float(metrics["validation_perplexity"])
    targets = int(metrics["validation_targets"])
    assert targets == EXPECTED_TARGETS

    result = {
        "experiment": "06A_model_c_extended_training_probe",
        "evaluation": "one_time_exploratory_official_test",
        "selected_checkpoint_update": SELECTED_UPDATE,
        "selected_validation_loss": SELECTED_VAL_LOSS,
        "test_loss": loss,
        "test_perplexity": ppl,
        "test_targets": targets,
        "frozen_three_epoch_model_c_test_loss": FROZEN_TEST_MODEL_C_LOSS,
        "frozen_three_epoch_model_c_test_perplexity": FROZEN_TEST_MODEL_C_PPL,
        "test_loss_change_vs_frozen": loss - FROZEN_TEST_MODEL_C_LOSS,
        "test_loss_improvement_vs_frozen": FROZEN_TEST_MODEL_C_LOSS - loss,
        "test_stream_sha256": EXPECTED_STREAM_SHA256,
        "official_test_scoring_count_for_06a": 1,
        "checkpoint_selection_changed_after_test": False,
        "retuning_permitted_after_test": False,
        "label": "exploratory Model C extended-training evidence",
        "gpu_name": gpu_name,
        "precision": policy.precision,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("NOTEBOOK 06A ONE-TIME TEST: COMPLETE")
    print(json.dumps(result, indent=2))
    print("LOCK: this artifact prevents repeated 06A official-test scoring.")


if __name__ == "__main__":
    main()
