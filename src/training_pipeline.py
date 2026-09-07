"""Canonical Notebook 04 training-pipeline implementation.

This module consolidates the validated training path from Notebook 04 without
changing the frozen experiment:

- canonical WikiText-103 training/validation reconstruction
- 512-token causal packing
- deterministic epoch shuffling
- AdamW + warmup/cosine schedule
- FP16 + GradScaler on Tesla T4
- 16,384 prediction targets per full optimizer update
- 3 epochs / 3,663 optimizer updates
- validation every 200 updates plus epoch end
- persistent best/latest checkpoints
- exact resume from checkpoint format v2 (model/optimizer/scaler/RNG state)
- official test split is never accessed here

The notebooks for Models A/B/C call the same implementation with only the
model key changed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

from src.data import (
    load_pinned_wikitext,
    normalize_wikitext_text,
    reconstruct_articles,
)
from src.model import (
    MODEL_CONFIGS,
    DecoderOnlyLM,
    analytical_parameter_count,
    initialize_model_weights,
)

SEED = 42
CORPUS_TOKENS = 20_000_000
CONTEXT_LENGTH = 512
NUM_FULL_EXAMPLES = (CORPUS_TOKENS - 1) // CONTEXT_LENGTH
PREDICTION_TARGETS = NUM_FULL_EXAMPLES * CONTEXT_LENGTH

EFFECTIVE_BATCH_TOKENS = 16_384
SEQUENCES_PER_FULL_UPDATE = EFFECTIVE_BATCH_TOKENS // CONTEXT_LENGTH
FULL_UPDATES_PER_EPOCH = NUM_FULL_EXAMPLES // SEQUENCES_PER_FULL_UPDATE
TAIL_SEQUENCES_PER_EPOCH = NUM_FULL_EXAMPLES % SEQUENCES_PER_FULL_UPDATE
TAIL_TARGETS_PER_EPOCH = TAIL_SEQUENCES_PER_EPOCH * CONTEXT_LENGTH
OPTIMIZER_UPDATES_PER_EPOCH = math.ceil(
    NUM_FULL_EXAMPLES / SEQUENCES_PER_FULL_UPDATE
)

MAX_EPOCHS = 3
PRODUCTION_TOTAL_UPDATES = OPTIMIZER_UPDATES_PER_EPOCH * MAX_EPOCHS
EXPECTED_PRODUCTION_TARGET_EXPOSURES = PREDICTION_TARGETS * MAX_EPOCHS

PRODUCTION_PEAK_LR = 2e-3
WARMUP_FRACTION = 0.05
MIN_LR_RATIO = 0.10
GRAD_CLIP_NORM = 1.0

PRODUCTION_VALIDATION_EVERY = 200
PRODUCTION_EPOCH_END_UPDATES = (1_221, 2_442, 3_663)
PRODUCTION_VALIDATION_UPDATES = tuple(sorted(
    set(range(
        PRODUCTION_VALIDATION_EVERY,
        PRODUCTION_TOTAL_UPDATES + 1,
        PRODUCTION_VALIDATION_EVERY,
    ))
    | set(PRODUCTION_EPOCH_END_UPDATES)
))

MODEL_PARAMETERS = {
    "A": 7_407_872,
    "B": 16_913_280,
    "C": 33_497_600,
}
PRODUCTION_MICRO_BATCH_SEQUENCES = {
    "A": 32,
    "B": 32,
    "C": 32,
}
DECISION_IDS = {
    "A": "D-074",
    "B": "D-075",
    "C": "D-076",
}

CANONICAL_TOKENIZER_SHA256 = (
    "6ec601a267cec7c843df47927f53c4dd108c85a1d059318aeec4442c7274604f"
)
CANONICAL_ARTICLE_PERMUTATION_SHA256 = (
    "d4e368c0c22c1ea044133f7648466201450e66dc170da8ba67235fc1cd3b836c"
)
CANONICAL_TRAIN_STREAM_SHA256 = (
    "4101d5b18c38558a58110f54a161763186ab5318111366486ebbfa0a3fe584fa"
)

ARTICLE_SHUFFLE_SEED = 42
TOKEN_BUDGET = 20_000_000
BOUNDARY_TOKEN_ID = 0

EXPECTED_TRAIN_ARTICLES = 28_472
EXPECTED_SELECTED_ARTICLE_RECORDS = 4_604
EXPECTED_BOUNDARY_TOKENS = 4_603
EXPECTED_TEXT_TOKENS = 19_995_397
EXPECTED_FINAL_INCLUDED_TEXT_TOKENS = 1_312
EXPECTED_FINAL_FULL_TEXT_TOKENS = 4_410

VALIDATION_ARTICLES = 60
VALIDATION_TEXT_TOKENS_NO_BOUNDARIES = 256_579
VALIDATION_STREAM_TOKENS = 256_639
VALIDATION_EXAMPLES = 501
VALIDATION_TARGETS = 256_512

ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
WEIGHT_DECAY = 0.10

assert NUM_FULL_EXAMPLES == 39_062
assert PREDICTION_TARGETS == 19_999_744
assert SEQUENCES_PER_FULL_UPDATE == 32
assert TAIL_SEQUENCES_PER_EPOCH == 22
assert TAIL_TARGETS_PER_EPOCH == 11_264
assert OPTIMIZER_UPDATES_PER_EPOCH == 1_221
assert PRODUCTION_TOTAL_UPDATES == 3_663
assert EXPECTED_PRODUCTION_TARGET_EXPOSURES == 59_999_232
assert len(PRODUCTION_VALIDATION_UPDATES) == 21


@dataclass(frozen=True)
class PrecisionPolicy:
    device_type: str
    precision: str
    autocast_dtype: Optional[torch.dtype]
    use_autocast: bool
    use_grad_scaler: bool


def native_cuda_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported(
            including_emulation=False
        )
    except TypeError:
        major, _minor = torch.cuda.get_device_capability(0)
        return major >= 8


def resolve_runtime_precision_policy() -> PrecisionPolicy:
    if not torch.cuda.is_available():
        return PrecisionPolicy(
            device_type="cpu",
            precision="fp32",
            autocast_dtype=None,
            use_autocast=False,
            use_grad_scaler=False,
        )
    if native_cuda_bf16_supported():
        return PrecisionPolicy(
            device_type="cuda",
            precision="bf16",
            autocast_dtype=torch.bfloat16,
            use_autocast=True,
            use_grad_scaler=False,
        )
    return PrecisionPolicy(
        device_type="cuda",
        precision="fp16",
        autocast_dtype=torch.float16,
        use_autocast=True,
        use_grad_scaler=True,
    )


def autocast_context(policy: PrecisionPolicy):
    if not policy.use_autocast:
        return nullcontext()
    return torch.autocast(
        device_type=policy.device_type,
        dtype=policy.autocast_dtype,
        enabled=True,
    )


def make_grad_scaler(policy: PrecisionPolicy):
    return torch.amp.GradScaler(
        "cuda",
        enabled=policy.use_grad_scaler,
    )


class CausalTokenDataset(Dataset):
    def __init__(self, token_ids, context_length: int = CONTEXT_LENGTH):
        token_ids = np.asarray(token_ids)
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be one-dimensional")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.token_ids = token_ids
        self.context_length = int(context_length)
        self.num_examples = max(
            0,
            (len(token_ids) - 1) // self.context_length,
        )

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        if index < 0:
            index += self.num_examples
        if index < 0 or index >= self.num_examples:
            raise IndexError("dataset index out of range")
        start = index * self.context_length
        stop = start + self.context_length
        x = torch.tensor(
            self.token_ids[start:stop],
            dtype=torch.long,
        )
        y = torch.tensor(
            self.token_ids[start + 1:stop + 1],
            dtype=torch.long,
        )
        return x, y


def make_epoch_dataloader(
    dataset,
    micro_batch_size: int,
    epoch: int,
    seed: int = SEED,
):
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch))
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def make_validation_dataloader(
    validation_dataset,
    batch_size: int,
):
    return DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def iter_effective_batch_pieces(
    dataloader,
    sequences_per_full_update: int = SEQUENCES_PER_FULL_UPDATE,
):
    pieces = []
    sequences_collected = 0

    for x_batch, y_batch in dataloader:
        offset = 0
        physical_batch_sequences = x_batch.size(0)

        while offset < physical_batch_sequences:
            remaining_for_update = (
                sequences_per_full_update
                - sequences_collected
            )
            take = min(
                remaining_for_update,
                physical_batch_sequences - offset,
            )
            pieces.append((
                x_batch[offset:offset + take],
                y_batch[offset:offset + take],
            ))
            sequences_collected += take
            offset += take

            if sequences_collected == sequences_per_full_update:
                yield pieces
                pieces = []
                sequences_collected = 0

    if pieces:
        yield pieces


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_canonical_tokenizer(
    tokenizer_path=Path("results/tokenizer/tokenizer.json"),
):
    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)

    actual_sha256 = sha256_file(tokenizer_path)
    if actual_sha256 != CANONICAL_TOKENIZER_SHA256:
        raise RuntimeError(
            "Tokenizer fingerprint mismatch: "
            f"{actual_sha256}"
        )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    assert tokenizer.get_vocab_size() == 16_384
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    return tokenizer


def build_validation_token_stream(validation_articles, tokenizer):
    token_ids = []
    text_token_count = 0
    assert len(validation_articles) == VALIDATION_ARTICLES

    for article in validation_articles:
        article_ids = tokenizer.encode(article["text"]).ids
        token_ids.extend(article_ids)
        text_token_count += len(article_ids)
        token_ids.append(BOUNDARY_TOKEN_ID)

    stream = np.asarray(token_ids, dtype="<u2")
    assert text_token_count == VALIDATION_TEXT_TOKENS_NO_BOUNDARIES
    assert len(stream) == VALIDATION_STREAM_TOKENS
    assert (
        int(np.count_nonzero(stream == BOUNDARY_TOKEN_ID))
        == VALIDATION_ARTICLES
    )
    return stream


def rebuild_canonical_streams(tokenizer):
    dataset = load_pinned_wikitext()

    raw_train_rows = list(dataset["train"]["text"])
    normalized_train_rows = [
        normalize_wikitext_text(text)
        for text in raw_train_rows
    ]
    train_articles = reconstruct_articles(
        raw_train_rows,
        normalized_train_rows,
        split_name="train",
    )
    assert len(train_articles) == EXPECTED_TRAIN_ARTICLES

    rng = np.random.default_rng(ARTICLE_SHUFFLE_SEED)
    article_permutation = rng.permutation(len(train_articles))

    permutation_sha256 = sha256_bytes(
        np.asarray(
            article_permutation,
            dtype="<i4",
        ).tobytes()
    )
    assert (
        permutation_sha256
        == CANONICAL_ARTICLE_PERMUTATION_SHA256
    )

    train_stream = np.empty(TOKEN_BUDGET, dtype="<u2")
    cursor = 0
    selected_records = 0
    boundary_tokens = 0
    text_tokens = 0
    final_record = None

    for article_index in article_permutation:
        article = train_articles[int(article_index)]
        article_ids = tokenizer.encode(article["text"]).ids

        remaining = TOKEN_BUDGET - cursor
        if remaining <= 0:
            break

        if len(article_ids) + 1 <= remaining:
            n = len(article_ids)
            train_stream[cursor:cursor + n] = np.asarray(
                article_ids,
                dtype="<u2",
            )
            cursor += n
            text_tokens += n

            train_stream[cursor] = BOUNDARY_TOKEN_ID
            cursor += 1
            boundary_tokens += 1
            selected_records += 1
        else:
            take = min(remaining, len(article_ids))
            train_stream[cursor:cursor + take] = np.asarray(
                article_ids[:take],
                dtype="<u2",
            )
            cursor += take
            text_tokens += take
            selected_records += 1
            final_record = {
                "included_text_tokens": take,
                "full_text_tokens": len(article_ids),
                "boundary_included": False,
            }
            break

    assert cursor == TOKEN_BUDGET
    train_sha256 = sha256_bytes(
        np.asarray(
            train_stream,
            dtype="<u2",
        ).tobytes()
    )

    assert train_sha256 == CANONICAL_TRAIN_STREAM_SHA256
    assert selected_records == EXPECTED_SELECTED_ARTICLE_RECORDS
    assert boundary_tokens == EXPECTED_BOUNDARY_TOKENS
    assert text_tokens == EXPECTED_TEXT_TOKENS
    assert final_record == {
        "included_text_tokens": EXPECTED_FINAL_INCLUDED_TEXT_TOKENS,
        "full_text_tokens": EXPECTED_FINAL_FULL_TEXT_TOKENS,
        "boundary_included": False,
    }

    raw_validation_rows = list(dataset["validation"]["text"])
    normalized_validation_rows = [
        normalize_wikitext_text(text)
        for text in raw_validation_rows
    ]
    validation_articles = reconstruct_articles(
        raw_validation_rows,
        normalized_validation_rows,
        split_name="validation",
    )
    validation_stream = build_validation_token_stream(
        validation_articles,
        tokenizer,
    )
    return train_stream, validation_stream


def load_or_build_canonical_datasets(
    persistent_root,
    tokenizer_path=Path("results/tokenizer/tokenizer.json"),
):
    persistent_root = Path(persistent_root)
    tokenizer = load_canonical_tokenizer(tokenizer_path)

    cache_dir = persistent_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_cache = cache_dir / "canonical_train_stream_u16.npy"
    val_cache = cache_dir / "canonical_validation_stream_u16.npy"

    cache_valid = False
    if train_cache.exists() and val_cache.exists():
        try:
            cached_train = np.load(train_cache, mmap_mode="r")
            cached_val = np.load(val_cache, mmap_mode="r")
            cache_valid = (
                cached_train.dtype == np.dtype("<u2")
                and cached_val.dtype == np.dtype("<u2")
                and len(cached_train) == TOKEN_BUDGET
                and len(cached_val) == VALIDATION_STREAM_TOKENS
                and sha256_bytes(
                    np.asarray(
                        cached_train,
                        dtype="<u2",
                    ).tobytes()
                ) == CANONICAL_TRAIN_STREAM_SHA256
            )
        except Exception:
            cache_valid = False

    if cache_valid:
        training_stream = cached_train
        validation_stream = cached_val
        cache_status = "loaded"
    else:
        training_stream, validation_stream = (
            rebuild_canonical_streams(tokenizer)
        )

        train_tmp = train_cache.with_suffix(".tmp")
        val_tmp = val_cache.with_suffix(".tmp")

        with train_tmp.open("wb") as f:
            np.save(
                f,
                np.asarray(training_stream, dtype="<u2"),
            )
        with val_tmp.open("wb") as f:
            np.save(
                f,
                np.asarray(validation_stream, dtype="<u2"),
            )

        os.replace(train_tmp, train_cache)
        os.replace(val_tmp, val_cache)

        training_stream = np.load(train_cache, mmap_mode="r")
        validation_stream = np.load(val_cache, mmap_mode="r")
        cache_status = "rebuilt"

    training_dataset = CausalTokenDataset(
        training_stream,
        context_length=CONTEXT_LENGTH,
    )
    validation_dataset = CausalTokenDataset(
        validation_stream,
        context_length=CONTEXT_LENGTH,
    )

    assert len(training_dataset) == 39_062
    assert len(validation_dataset) == 501

    return {
        "tokenizer": tokenizer,
        "training_stream": training_stream,
        "validation_stream": validation_stream,
        "training_dataset": training_dataset,
        "validation_dataset": validation_dataset,
        "cache_status": cache_status,
    }


def build_adamw_parameter_groups(model):
    decay_params = []
    no_decay_params = []
    seen = set()

    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))

        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    grouped_ids = [
        id(p)
        for group in (decay_params, no_decay_params)
        for p in group
    ]
    trainable_ids = [
        id(p)
        for p in model.parameters()
        if p.requires_grad
    ]

    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(trainable_ids)

    return [
        {
            "params": decay_params,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]


def build_adamw_optimizer(model, learning_rate: float):
    return torch.optim.AdamW(
        build_adamw_parameter_groups(model),
        lr=learning_rate,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
    )


def learning_rate_for_update(update_number: int) -> float:
    warmup_updates = max(
        1,
        int(PRODUCTION_TOTAL_UPDATES * WARMUP_FRACTION),
    )
    min_lr = PRODUCTION_PEAK_LR * MIN_LR_RATIO

    if update_number <= warmup_updates:
        return PRODUCTION_PEAK_LR * (
            update_number / warmup_updates
        )

    decay_updates = (
        PRODUCTION_TOTAL_UPDATES - warmup_updates
    )
    progress = (
        update_number - warmup_updates
    ) / decay_updates

    cosine = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
    return min_lr + (
        PRODUCTION_PEAK_LR - min_lr
    ) * cosine


def set_optimizer_learning_rate(optimizer, learning_rate: float):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def global_grad_norm(parameters) -> float:
    grads = [
        p.grad.detach()
        for p in parameters
        if p.grad is not None
    ]
    if not grads:
        return 0.0

    norms = [
        torch.linalg.vector_norm(g.float(), ord=2)
        for g in grads
    ]
    total = torch.linalg.vector_norm(
        torch.stack(norms),
        ord=2,
    )
    return float(total.cpu())


def train_one_optimizer_update(
    model,
    optimizer,
    scaler,
    policy,
    batch_pieces,
    learning_rate: float,
):
    set_optimizer_learning_rate(
        optimizer,
        learning_rate,
    )

    total_targets = sum(
        y.numel()
        for _, y in batch_pieces
    )
    total_loss_sum = 0.0

    for x, y in batch_pieces:
        x = x.to(
            policy.device_type,
            non_blocking=True,
        )
        y = y.to(
            policy.device_type,
            non_blocking=True,
        )

        with autocast_context(policy):
            logits = model(x)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )

        if not torch.isfinite(loss_sum):
            raise FloatingPointError(
                "Non-finite training loss encountered"
            )

        total_loss_sum += float(
            loss_sum.detach().float().cpu()
        )

        normalized_loss = loss_sum / total_targets

        if policy.use_grad_scaler:
            scaler.scale(normalized_loss).backward()
        else:
            normalized_loss.backward()

    if policy.use_grad_scaler:
        scaler.unscale_(optimizer)

    grad_norm_before_clip = global_grad_norm(
        model.parameters()
    )
    if not math.isfinite(grad_norm_before_clip):
        raise FloatingPointError(
            "Non-finite gradient norm encountered"
        )

    clipped = grad_norm_before_clip > GRAD_CLIP_NORM

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=GRAD_CLIP_NORM,
        error_if_nonfinite=True,
    )

    if policy.use_grad_scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    return {
        "training_loss": total_loss_sum / total_targets,
        "training_targets": total_targets,
        "grad_norm_before_clip": grad_norm_before_clip,
        "clipped": clipped,
        "learning_rate": learning_rate,
    }


def evaluate_language_model(
    model,
    validation_loader,
    policy,
):
    was_training = model.training
    model.eval()

    total_loss_sum = 0.0
    total_targets = 0

    try:
        with torch.inference_mode():
            for x, y in validation_loader:
                x = x.to(
                    policy.device_type,
                    non_blocking=True,
                )
                y = y.to(
                    policy.device_type,
                    non_blocking=True,
                )

                with autocast_context(policy):
                    logits = model(x)
                    loss_sum = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                        reduction="sum",
                    )

                if not torch.isfinite(loss_sum):
                    raise FloatingPointError(
                        "Non-finite validation loss encountered"
                    )

                total_loss_sum += float(
                    loss_sum.detach().float().cpu()
                )
                total_targets += y.numel()

    finally:
        model.train(was_training)

    mean_loss = total_loss_sum / total_targets
    return {
        "validation_loss": mean_loss,
        "validation_perplexity": math.exp(mean_loss),
        "validation_targets": total_targets,
    }


def atomic_torch_save(payload, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def write_json_artifact(payload, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            allow_nan=False,
        )
    os.replace(tmp, path)


def capture_rng_state():
    return {
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_rng_state(state):
    torch.set_rng_state(
        state["torch_cpu_rng_state"]
    )
    cuda_state = state.get(
        "torch_cuda_rng_state_all"
    )
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _artifact_paths(
    model_key: str,
    persistent_root,
):
    key = model_key.lower()
    persistent_root = Path(persistent_root)

    checkpoint_dir = persistent_root / "checkpoints"
    result_dir = persistent_root / "results" / "training"
    legacy_dir = persistent_root / "legacy_partial"

    return {
        "best": checkpoint_dir / f"model_{key}_best.pt",
        "latest": checkpoint_dir / f"model_{key}_latest.pt",
        "history": result_dir / f"model_{key}_training_history.json",
        "summary": result_dir / f"model_{key}_run_summary.json",
        "progress": result_dir / f"model_{key}_progress.json",
        "legacy_dir": legacy_dir,
    }


def _complete_summary_if_available(
    model_key: str,
    paths,
):
    required = [
        paths["best"],
        paths["latest"],
        paths["history"],
        paths["summary"],
    ]

    if not all(
        p.exists() and p.stat().st_size > 0
        for p in required
    ):
        return None

    try:
        with paths["summary"].open(
            "r",
            encoding="utf-8",
        ) as f:
            summary = json.load(f)
    except Exception:
        return None

    complete = (
        summary.get("model_key") == model_key
        and summary.get("completed_updates") == 3_663
        and summary.get("completed_full_epochs") == 3
        and summary.get("total_target_exposures")
            == EXPECTED_PRODUCTION_TARGET_EXPOSURES
        and summary.get("validation_events") == 21
        and summary.get(
            "official_test_split_content_used"
        ) is False
    )

    if not complete:
        return None

    if "best_validation_perplexity" not in summary:
        summary["best_validation_perplexity"] = math.exp(
            summary["best_validation_loss"]
        )

    return summary


def _load_resume_checkpoint_or_archive_legacy(
    model_key: str,
    paths,
):
    latest = paths["latest"]

    if not latest.exists():
        return None

    candidate = torch.load(
        latest,
        map_location="cpu",
        weights_only=False,
    )

    if (
        candidate.get("format_version") == 2
        and candidate.get("model_key") == model_key
        and "rng_state" in candidate
    ):
        return candidate

    paths["legacy_dir"].mkdir(
        parents=True,
        exist_ok=True,
    )
    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    for name in ("latest", "best"):
        path = paths[name]
        if path.exists():
            archived = (
                paths["legacy_dir"]
                / f"{path.stem}_{stamp}{path.suffix}"
            )
            os.replace(path, archived)

    return None


def _make_v2_checkpoint(
    *,
    model_key,
    model,
    optimizer,
    scaler,
    global_update,
    epoch_index,
    updates_completed_in_epoch,
    completed_full_epochs,
    total_target_exposures,
    best_validation_loss,
    best_validation_update,
    history,
    validation_history,
    cumulative_elapsed_seconds,
    peak_memory_gib_so_far,
    gpu_name,
    precision,
):
    return {
        "format_version": 2,
        "project": "foundation-model-from-scratch",
        "model_key": model_key,
        "global_update": int(global_update),
        "epoch_index": int(epoch_index),
        "updates_completed_in_epoch": int(
            updates_completed_in_epoch
        ),
        "completed_full_epochs": int(
            completed_full_epochs
        ),
        "total_target_exposures": int(
            total_target_exposures
        ),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "best_validation_update": int(
            best_validation_update
        ),
        "history": history,
        "validation_history": validation_history,
        "cumulative_elapsed_seconds": float(
            cumulative_elapsed_seconds
        ),
        "peak_memory_gib_so_far": float(
            peak_memory_gib_so_far
        ),
        "seed": SEED,
        "peak_lr": PRODUCTION_PEAK_LR,
        "micro_batch_sequences": (
            PRODUCTION_MICRO_BATCH_SEQUENCES[model_key]
        ),
        "effective_batch_targets": EFFECTIVE_BATCH_TOKENS,
        "gpu_name": gpu_name,
        "precision": precision,
        "tokenizer_sha256": CANONICAL_TOKENIZER_SHA256,
        "train_stream_sha256": CANONICAL_TRAIN_STREAM_SHA256,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "saved_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def run_production_model(
    model_key: str,
    persistent_root,
    *,
    require_t4: bool = True,
):
    model_key = str(model_key).upper()

    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            "model_key must be one of ('A', 'B', 'C')"
        )

    expected_parameters = MODEL_PARAMETERS[model_key]
    actual_parameters = analytical_parameter_count(
        MODEL_CONFIGS[model_key]
    )["total"]

    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"Model {model_key} parameter mismatch: "
            f"{actual_parameters:,} != {expected_parameters:,}"
        )

    policy = resolve_runtime_precision_policy()

    if policy.device_type != "cuda":
        raise RuntimeError(
            "Production training requires a CUDA runtime."
        )

    gpu_name = torch.cuda.get_device_name(0)

    if require_t4 and gpu_name != "Tesla T4":
        raise RuntimeError(
            "Controlled production hardware is Tesla T4; "
            f"active GPU is {gpu_name!r}."
        )

    if require_t4:
        assert policy.precision == "fp16"
        assert policy.use_grad_scaler

    paths = _artifact_paths(
        model_key,
        persistent_root,
    )

    complete_summary = _complete_summary_if_available(
        model_key,
        paths,
    )

    decision_id = DECISION_IDS[model_key]

    if complete_summary is not None:
        print("=" * 76)
        print(
            f"{decision_id} MODEL {model_key} "
            "PRODUCTION TRAINING: ALREADY COMPLETE"
        )
        print("=" * 76)
        print(
            f"Model {model_key} will NOT be retrained."
        )
        return complete_summary

    datasets = load_or_build_canonical_datasets(
        persistent_root
    )
    training_dataset = datasets["training_dataset"]
    validation_dataset = datasets["validation_dataset"]

    print(
        "Canonical corpus cache:",
        datasets["cache_status"].upper(),
    )
    print("Training examples:", len(training_dataset))
    print("Validation examples:", len(validation_dataset))
    print("Official test split content used: NO")

    resume_checkpoint = (
        _load_resume_checkpoint_or_archive_legacy(
            model_key,
            paths,
        )
    )

    cfg = MODEL_CONFIGS[model_key]

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = DecoderOnlyLM(cfg)
    initialize_model_weights(
        model,
        seed=SEED,
    )
    model = model.cuda()

    optimizer = build_adamw_optimizer(
        model,
        learning_rate=PRODUCTION_PEAK_LR,
    )
    scaler = make_grad_scaler(policy)
    optimizer.zero_grad(set_to_none=True)

    validation_loader = make_validation_dataloader(
        validation_dataset,
        batch_size=PRODUCTION_MICRO_BATCH_SEQUENCES[
            model_key
        ],
    )

    history = []
    validation_history = []
    global_update = 0
    completed_full_epochs = 0
    total_target_exposures = 0
    best_validation_loss = float("inf")
    best_validation_update = None
    cumulative_elapsed_before_session = 0.0
    peak_memory_gib_before_session = 0.0
    start_epoch = 0
    skip_updates_in_start_epoch = 0
    resume_count = 0

    if resume_checkpoint is not None:
        model.load_state_dict(
            resume_checkpoint["model_state_dict"],
            strict=True,
        )
        optimizer.load_state_dict(
            resume_checkpoint["optimizer_state_dict"]
        )
        move_optimizer_state_to_device(
            optimizer,
            torch.device("cuda"),
        )
        scaler.load_state_dict(
            resume_checkpoint["scaler_state_dict"]
        )

        history = list(resume_checkpoint["history"])
        validation_history = list(
            resume_checkpoint["validation_history"]
        )
        global_update = int(
            resume_checkpoint["global_update"]
        )
        completed_full_epochs = int(
            resume_checkpoint["completed_full_epochs"]
        )
        total_target_exposures = int(
            resume_checkpoint["total_target_exposures"]
        )
        best_validation_loss = float(
            resume_checkpoint["best_validation_loss"]
        )
        best_validation_update = int(
            resume_checkpoint["best_validation_update"]
        )
        cumulative_elapsed_before_session = float(
            resume_checkpoint["cumulative_elapsed_seconds"]
        )
        peak_memory_gib_before_session = float(
            resume_checkpoint["peak_memory_gib_so_far"]
        )

        checkpoint_epoch = int(
            resume_checkpoint["epoch_index"]
        )
        checkpoint_updates_in_epoch = int(
            resume_checkpoint[
                "updates_completed_in_epoch"
            ]
        )

        if (
            checkpoint_updates_in_epoch
            == OPTIMIZER_UPDATES_PER_EPOCH
        ):
            start_epoch = checkpoint_epoch + 1
            skip_updates_in_start_epoch = 0
        else:
            start_epoch = checkpoint_epoch
            skip_updates_in_start_epoch = (
                checkpoint_updates_in_epoch
            )

        restore_rng_state(
            resume_checkpoint["rng_state"]
        )
        resume_count = 1

        assert len(history) == global_update

        print("=" * 76)
        print(
            f"{decision_id} MODEL {model_key} "
            "PRODUCTION TRAINING: RESUMING"
        )
        print("=" * 76)
        print("Resume update:", global_update)

    else:
        print("=" * 76)
        print(
            f"{decision_id} MODEL {model_key} "
            "PRODUCTION TRAINING: STARTING"
        )
        print("=" * 76)
        print(
            "Model parameters:         ",
            f"{expected_parameters:,}",
        )
        print(
            "Peak learning rate:       ",
            PRODUCTION_PEAK_LR,
        )
        print("Maximum epochs:           ", MAX_EPOCHS)
        print(
            "Scheduled updates:        ",
            PRODUCTION_TOTAL_UPDATES,
        )
        print(
            "Effective batch targets:  ",
            EFFECTIVE_BATCH_TOKENS,
        )
        print(
            "Physical micro-batch:     ",
            PRODUCTION_MICRO_BATCH_SEQUENCES[
                model_key
            ],
            "sequences",
        )
        print(
            "Persistent output root:   ",
            Path(persistent_root),
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    session_start = time.perf_counter()

    for epoch_index in range(
        start_epoch,
        MAX_EPOCHS,
    ):
        loader = make_epoch_dataloader(
            training_dataset,
            micro_batch_size=(
                PRODUCTION_MICRO_BATCH_SEQUENCES[
                    model_key
                ]
            ),
            epoch=epoch_index,
            seed=SEED,
        )

        update_groups = iter_effective_batch_pieces(
            loader,
            sequences_per_full_update=(
                SEQUENCES_PER_FULL_UPDATE
            ),
        )

        skipped = (
            skip_updates_in_start_epoch
            if epoch_index == start_epoch
            else 0
        )

        for _ in range(skipped):
            try:
                next(update_groups)
            except StopIteration as exc:
                raise RuntimeError(
                    "Resume checkpoint points past "
                    "the end of its epoch."
                ) from exc

        updates_completed_in_epoch = skipped

        for pieces in update_groups:
            global_update += 1
            updates_completed_in_epoch += 1

            lr = learning_rate_for_update(
                global_update
            )

            metrics = train_one_optimizer_update(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                policy=policy,
                batch_pieces=pieces,
                learning_rate=lr,
            )

            total_target_exposures += int(
                metrics["training_targets"]
            )

            record = {
                "epoch": epoch_index + 1,
                "update": global_update,
                **metrics,
            }

            if (
                global_update
                in PRODUCTION_VALIDATION_UPDATES
            ):
                validation_metrics = (
                    evaluate_language_model(
                        model,
                        validation_loader,
                        policy,
                    )
                )

                record.update(
                    validation_metrics
                )

                validation_history.append({
                    "epoch": epoch_index + 1,
                    "update": global_update,
                    **validation_metrics,
                })

                improved = (
                    validation_metrics[
                        "validation_loss"
                    ]
                    < best_validation_loss
                )

                if improved:
                    best_validation_loss = float(
                        validation_metrics[
                            "validation_loss"
                        ]
                    )
                    best_validation_update = (
                        global_update
                    )

                history.append(record)

                checkpoint_completed_epochs = (
                    max(
                        completed_full_epochs,
                        epoch_index + 1,
                    )
                    if (
                        updates_completed_in_epoch
                        == OPTIMIZER_UPDATES_PER_EPOCH
                    )
                    else completed_full_epochs
                )

                torch.cuda.synchronize()

                session_elapsed = (
                    time.perf_counter()
                    - session_start
                )
                cumulative_elapsed = (
                    cumulative_elapsed_before_session
                    + session_elapsed
                )
                current_peak_gib = (
                    torch.cuda.max_memory_allocated()
                    / (1024 ** 3)
                )
                peak_memory_gib_so_far = max(
                    peak_memory_gib_before_session,
                    current_peak_gib,
                )

                checkpoint_payload = (
                    _make_v2_checkpoint(
                        model_key=model_key,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        global_update=global_update,
                        epoch_index=epoch_index,
                        updates_completed_in_epoch=(
                            updates_completed_in_epoch
                        ),
                        completed_full_epochs=(
                            checkpoint_completed_epochs
                        ),
                        total_target_exposures=(
                            total_target_exposures
                        ),
                        best_validation_loss=(
                            best_validation_loss
                        ),
                        best_validation_update=(
                            best_validation_update
                        ),
                        history=history,
                        validation_history=(
                            validation_history
                        ),
                        cumulative_elapsed_seconds=(
                            cumulative_elapsed
                        ),
                        peak_memory_gib_so_far=(
                            peak_memory_gib_so_far
                        ),
                        gpu_name=gpu_name,
                        precision=policy.precision,
                    )
                )

                atomic_torch_save(
                    checkpoint_payload,
                    paths["latest"],
                )

                if improved:
                    atomic_torch_save(
                        checkpoint_payload,
                        paths["best"],
                    )

                write_json_artifact(
                    {
                        "status": "in_progress",
                        "model_key": model_key,
                        "global_update": global_update,
                        "total_updates": (
                            PRODUCTION_TOTAL_UPDATES
                        ),
                        "validation_loss": (
                            validation_metrics[
                                "validation_loss"
                            ]
                        ),
                        "validation_perplexity": (
                            validation_metrics[
                                "validation_perplexity"
                            ]
                        ),
                        "best_validation_loss": (
                            best_validation_loss
                        ),
                        "best_validation_update": (
                            best_validation_update
                        ),
                        "checkpoint_format_version": 2,
                        "saved_at_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    },
                    paths["progress"],
                )

                print(
                    f"Model {model_key} | update "
                    f"{global_update:4d}/3663 | "
                    f"val_loss="
                    f"{validation_metrics['validation_loss']:.6f} | "
                    f"ppl="
                    f"{validation_metrics['validation_perplexity']:.2f} | "
                    f"best="
                    f"{best_validation_loss:.6f}"
                    f"@{best_validation_update} | "
                    f"elapsed="
                    f"{cumulative_elapsed/60:.1f} min | "
                    f"checkpoint=PERSISTED"
                )

            else:
                history.append(record)

        assert (
            updates_completed_in_epoch
            == OPTIMIZER_UPDATES_PER_EPOCH
        )
        completed_full_epochs = epoch_index + 1

    torch.cuda.synchronize()

    session_elapsed = time.perf_counter() - session_start
    cumulative_elapsed = (
        cumulative_elapsed_before_session
        + session_elapsed
    )
    peak_memory_gib = max(
        peak_memory_gib_before_session,
        torch.cuda.max_memory_allocated() / (1024 ** 3),
    )

    assert global_update == PRODUCTION_TOTAL_UPDATES
    assert completed_full_epochs == MAX_EPOCHS
    assert (
        total_target_exposures
        == EXPECTED_PRODUCTION_TARGET_EXPOSURES
    )
    assert len(validation_history) == 21
    assert best_validation_update is not None
    assert math.isfinite(best_validation_loss)

    summary = {
        "model_key": model_key,
        "model_parameters": expected_parameters,
        "completed_updates": global_update,
        "completed_full_epochs": completed_full_epochs,
        "peak_lr": PRODUCTION_PEAK_LR,
        "micro_batch_sequences": (
            PRODUCTION_MICRO_BATCH_SEQUENCES[
                model_key
            ]
        ),
        "effective_batch_targets": EFFECTIVE_BATCH_TOKENS,
        "total_target_exposures": (
            total_target_exposures
        ),
        "best_validation_loss": (
            best_validation_loss
        ),
        "best_validation_perplexity": math.exp(
            best_validation_loss
        ),
        "best_validation_update": (
            best_validation_update
        ),
        "validation_events": len(
            validation_history
        ),
        "elapsed_seconds": cumulative_elapsed,
        "gpu_hours": cumulative_elapsed / 3600.0,
        "updates_per_second": (
            global_update / cumulative_elapsed
        ),
        "targets_per_second": (
            total_target_exposures
            / cumulative_elapsed
        ),
        "peak_memory_gib": peak_memory_gib,
        "gpu_name": gpu_name,
        "precision": policy.precision,
        "seed": SEED,
        "tokenizer_sha256": CANONICAL_TOKENIZER_SHA256,
        "train_stream_sha256": CANONICAL_TRAIN_STREAM_SHA256,
        "official_test_split_content_used": False,
        "checkpoint_format_version": 2,
        "resume_count_this_runner": resume_count,
        "best_checkpoint": str(
            paths["best"]
        ),
        "latest_checkpoint": str(
            paths["latest"]
        ),
    }

    write_json_artifact(
        {
            "model_key": model_key,
            "history": history,
            "validation_history": validation_history,
        },
        paths["history"],
    )
    write_json_artifact(
        summary,
        paths["summary"],
    )
    write_json_artifact(
        {
            "status": "complete",
            "model_key": model_key,
            "global_update": global_update,
            "saved_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },
        paths["progress"],
    )

    print()
    print("=" * 76)
    print(
        f"{decision_id} MODEL {model_key} "
        "PRODUCTION TRAINING: PASS"
    )
    print("=" * 76)

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return summary


def audit_persistent_model_artifacts(
    model_key: str,
    persistent_root,
):
    model_key = str(model_key).upper()

    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            "model_key must be one of ('A', 'B', 'C')"
        )

    paths = _artifact_paths(
        model_key,
        persistent_root,
    )

    required = [
        paths["best"],
        paths["latest"],
        paths["history"],
        paths["summary"],
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size <= 0:
            raise RuntimeError(
                f"Empty artifact: {path}"
            )

    summary = _complete_summary_if_available(
        model_key,
        paths,
    )

    if summary is None:
        raise RuntimeError(
            f"Model {model_key} persistent run "
            "is not complete."
        )

    return {
        "summary": summary,
        "paths": {
            name: str(path)
            for name, path in {
                "best": paths["best"],
                "latest": paths["latest"],
                "history": paths["history"],
                "summary": paths["summary"],
            }.items()
        },
        "sizes_bytes": {
            path.name: path.stat().st_size
            for path in required
        },
    }
