#!/usr/bin/env python3
"""Run the precommitted D-091 qualitative probe on validation-selected 06A Model C.

This is a controlled before/after generation probe only. It reuses the exact three
validation-derived prompts, seeds, temperature, top-p, and 96-token generation
length frozen in Notebook 05. It does not access the official test split and does
not alter checkpoint selection or training.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import MODEL_CONFIGS, DecoderOnlyLM, analytical_parameter_count
from src.training_pipeline import (
    CANONICAL_TOKENIZER_SHA256,
    autocast_context,
    load_canonical_tokenizer,
    resolve_runtime_precision_policy,
    sha256_file,
    write_json_artifact,
)

MODEL_KEY = "C"
EXPECTED_PARAMETERS = 33_497_600
SELECTED_UPDATE = 12_210
SELECTED_VAL_LOSS = 3.599946362767629
TEMPERATURE = 0.8
TOP_P = 0.9
NEW_TOKENS = 96

PROMPTS = [
    {
        "prompt_id": 1,
        "validation_example_index": 20,
        "seed": 43,
        "prompt_text": "echsler School in 1894, two Carnegie libraries in 1913, and the Threefoot Building, Meridian's tallest skyscraper, in 1929.\nThe city continued to grow thanks to a commission government's efforts",
    },
    {
        "prompt_id": 2,
        "validation_example_index": 200,
        "seed": 44,
        "prompt_text": "af Dam.\nMount Elbert is composed largely of quartzite. However, the summit ridge consists of metamorphic basement rock, which is Pre-Cambrian in origin and about 1.7 billion years old.",
    },
    {
        "prompt_id": 3,
        "validation_example_index": 400,
        "seed": 45,
        "prompt_text": " with Ward Churchill: Historical and Current Perspectives \". Z Magazine.)\nIn 1996, Churchill moved to the new Ethnic Studies Department of the University of Colorado. In 1997, he was promoted to full professor. He was selected as",
    },
]


def sample_top_p(probs: torch.Tensor, top_p: float, generator: torch.Generator) -> int:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    total = sorted_probs.sum()
    if not torch.isfinite(total) or total <= 0:
        raise FloatingPointError("Invalid probability mass during top-p sampling")
    sorted_probs = sorted_probs / total
    sampled_sorted = torch.multinomial(sorted_probs, 1, generator=generator)
    return int(sorted_idx[sampled_sorted].item())


@torch.inference_mode()
def generate(model, tokenizer, policy, prompt_text: str, seed: int) -> tuple[str, list[int]]:
    prompt_ids = tokenizer.encode(prompt_text).ids
    if not prompt_ids:
        raise RuntimeError("Prompt tokenized to an empty sequence")

    ids = list(prompt_ids)
    generated = []
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))

    for _ in range(NEW_TOKENS):
        x = torch.tensor([ids[-512:]], dtype=torch.long, device="cuda")
        with autocast_context(policy):
            logits = model(x)
        next_logits = logits[0, -1].float() / TEMPERATURE
        probs = torch.softmax(next_logits, dim=-1)
        next_id = sample_top_p(probs, TOP_P, generator)
        generated.append(next_id)
        ids.append(next_id)

    return tokenizer.decode(generated), generated


def main():
    assert torch.cuda.is_available(), "06A generation requires CUDA"
    gpu_name = torch.cuda.get_device_name(0)
    assert gpu_name == "Tesla T4", f"Expected Tesla T4, got {gpu_name!r}"
    policy = resolve_runtime_precision_policy()
    assert policy.precision == "fp16"

    root = Path("/content/drive/MyDrive/foundation-model-from-scratch/production")
    ext_root = root / "extended_training" / "model_c"
    best_path = ext_root / "model_c_extension_best.pt"
    test_artifact = ext_root / "one_time_exploratory_test.json"
    output_path = ext_root / "fixed_d091_generation.json"

    assert best_path.exists(), f"Missing selected checkpoint: {best_path}"
    assert test_artifact.exists(), "Run the precommitted one-time 06A test before generation"

    test_result = json.loads(test_artifact.read_text(encoding="utf-8"))
    assert int(test_result["selected_checkpoint_update"]) == SELECTED_UPDATE
    assert int(test_result["official_test_scoring_count_for_06a"]) == 1
    assert test_result["checkpoint_selection_changed_after_test"] is False
    assert test_result["retuning_permitted_after_test"] is False

    ck = torch.load(best_path, map_location="cpu", weights_only=False)
    assert int(ck["global_update"]) == SELECTED_UPDATE
    assert abs(float(ck["best_validation_loss"]) - SELECTED_VAL_LOSS) < 1e-12
    assert ck["model_key"] == MODEL_KEY
    assert ck["tokenizer_sha256"] == CANONICAL_TOKENIZER_SHA256

    tokenizer_path = REPO_ROOT / "results/tokenizer/tokenizer.json"
    assert sha256_file(tokenizer_path) == CANONICAL_TOKENIZER_SHA256
    tokenizer = load_canonical_tokenizer(tokenizer_path)

    cfg = MODEL_CONFIGS[MODEL_KEY]
    assert analytical_parameter_count(cfg)["total"] == EXPECTED_PARAMETERS
    model = DecoderOnlyLM(cfg).cuda()
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()

    results = []
    print("NOTEBOOK 06A FIXED D-091 GENERATION PROBE")
    print("Checkpoint:", SELECTED_UPDATE)
    print("Temperature / top-p / new tokens:", TEMPERATURE, "/", TOP_P, "/", NEW_TOKENS)
    print("Official test split accessed by this script: False")

    for spec in PROMPTS:
        continuation, token_ids = generate(
            model, tokenizer, policy, spec["prompt_text"], spec["seed"]
        )
        row = {
            "model": "C_extended_06A",
            **spec,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "new_tokens": NEW_TOKENS,
            "selected_checkpoint_update": SELECTED_UPDATE,
            "selected_validation_loss": SELECTED_VAL_LOSS,
            "continuation_text": continuation,
            "continuation_token_ids": token_ids,
        }
        results.append(row)
        print("\n" + "=" * 72)
        print(f"PROMPT {spec['prompt_id']} | seed={spec['seed']}")
        print("PROMPT:")
        print(spec["prompt_text"])
        print("CONTINUATION:")
        print(continuation)

    payload = {
        "experiment": "06A_model_c_extended_training_probe",
        "evaluation": "fixed_D091_validation_prompt_generation",
        "checkpoint_update": SELECTED_UPDATE,
        "checkpoint_validation_loss": SELECTED_VAL_LOSS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "new_tokens": NEW_TOKENS,
        "prompt_count": len(PROMPTS),
        "prompt_source": "frozen Notebook 05 D-091 validation-derived prompts",
        "official_test_split_content_used_by_generation": False,
        "gpu_name": gpu_name,
        "precision": policy.precision,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    write_json_artifact(payload, output_path)
    print("\nNOTEBOOK 06A FIXED D-091 GENERATION: COMPLETE")
    print("Artifact:", output_path)


if __name__ == "__main__":
    main()
