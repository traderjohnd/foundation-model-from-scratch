#!/usr/bin/env python3
"""D-096 Gate 5: deterministic epoch-4 data-order continuity.

Read-only audit. Reproduces the exact Notebook 04 DataLoader shuffle path for
extension epoch 4 using the frozen seed convention `base_seed + epoch = 46`.
It records the first 10 training-example indices actually yielded by the
DataLoader and independently reproduces them while accounting for PyTorch's
DataLoader iterator base-seed draw from the same generator.

ZERO optimizer steps are executed. This script cannot reach update 3,664.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Make repository-root `src` imports work when this script is launched directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training_pipeline import (  # noqa: E402
    NUM_FULL_EXAMPLES,
    SEED,
    make_epoch_dataloader,
)

EXPECTED_NUM_EXAMPLES = 39_062
EXPECTED_EPOCH_NUMBER = 4
EXPECTED_SHUFFLE_SEED = 46
MICRO_BATCH_SIZE = 32
FIRST_N = 10


class IndexDataset(Dataset):
    """Dataset whose data value is its own example index."""

    def __init__(self, length: int):
        self.length = int(length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.tensor(index, dtype=torch.long)


def actual_loader_first_indices() -> list[int]:
    """Read first shuffled indices through the exact canonical loader helper."""
    dataset = IndexDataset(EXPECTED_NUM_EXAMPLES)
    loader = make_epoch_dataloader(
        dataset,
        micro_batch_size=MICRO_BATCH_SIZE,
        epoch=EXPECTED_EPOCH_NUMBER,
        seed=SEED,
    )
    first_batch = next(iter(loader))
    return [int(x) for x in first_batch[:FIRST_N].tolist()]


def manual_reproduction_first_indices() -> list[int]:
    """Independently reproduce DataLoader's generator-consumption semantics."""
    generator = torch.Generator()
    generator.manual_seed(EXPECTED_SHUFFLE_SEED)

    # DataLoader iterator creation consumes one int64 draw from loader.generator
    # to establish its internal base seed before RandomSampler consumes the same
    # generator for torch.randperm(...).
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    permutation = torch.randperm(
        EXPECTED_NUM_EXAMPLES,
        generator=generator,
    )
    return [int(x) for x in permutation[:FIRST_N].tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/extended_training/model_c/d096_gate5_shuffle_continuity.json"
        ),
    )
    args = parser.parse_args()

    assert NUM_FULL_EXAMPLES == EXPECTED_NUM_EXAMPLES
    assert SEED == 42
    assert SEED + EXPECTED_EPOCH_NUMBER == EXPECTED_SHUFFLE_SEED

    actual_first = actual_loader_first_indices()
    repeat_first = actual_loader_first_indices()
    manual_first = manual_reproduction_first_indices()

    assert len(actual_first) == FIRST_N
    assert actual_first == repeat_first, (
        "D-096 Gate 5 BLOCKED: canonical epoch-4 DataLoader order is not "
        "reproducible across identical reruns"
    )
    assert actual_first == manual_first, (
        "D-096 Gate 5 BLOCKED: DataLoader order does not match independent "
        "reproduction of seed-46 generator semantics"
    )
    assert len(set(actual_first)) == FIRST_N
    assert all(0 <= i < EXPECTED_NUM_EXAMPLES for i in actual_first)

    result = {
        "gate": "D-096 Gate 5 — epoch-4 data-order continuity",
        "gate_passed": True,
        "training_updates_executed": 0,
        "base_seed": SEED,
        "epoch_number": EXPECTED_EPOCH_NUMBER,
        "effective_shuffle_seed": EXPECTED_SHUFFLE_SEED,
        "training_examples": EXPECTED_NUM_EXAMPLES,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "first_10_example_indices": actual_first,
        "repeat_first_10_example_indices": repeat_first,
        "independent_first_10_example_indices": manual_first,
        "reproducible": actual_first == repeat_first,
        "independent_reproduction_match": actual_first == manual_first,
        "dataloader_semantics_note": (
            "PyTorch DataLoader consumes one int64 generator draw for iterator "
            "base_seed before RandomSampler draws randperm from the same generator."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("D-096 GATE 5: PASS")
    print("Base seed:", SEED)
    print("Epoch number:", EXPECTED_EPOCH_NUMBER)
    print("Effective shuffle seed:", EXPECTED_SHUFFLE_SEED)
    print("Training examples:", EXPECTED_NUM_EXAMPLES)
    print("First 10 example indices:", actual_first)
    print("Repeat reproduction:", repeat_first)
    print("Independent reproduction:", manual_first)
    print("Optimizer updates executed: 0")
    print("STOP: D-096 component gates are complete; update 3,664 has NOT executed.")


if __name__ == "__main__":
    main()
