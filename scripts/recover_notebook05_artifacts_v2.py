#!/usr/bin/env python3
"""Schema-compatible wrapper for Notebook 05 artifact recovery verification.

This fixes the D-097 recovery verifier's handling of pandas' default
DataFrame.to_json(...), whose default orient='columns' produces a nested
column-oriented JSON object rather than a row list.

Use this after Notebook 05 has already been re-executed successfully:
    python scripts/recover_notebook05_artifacts_v2.py --skip-execution

It delegates all invariant checks, artifact requirements, SHA-256 manifest
creation, and fail-closed behavior to recover_notebook05_artifacts.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import recover_notebook05_artifacts as base


def _normalize_model(value: Any) -> str:
    return str(value).strip().upper().replace("MODEL ", "")


def _rows_from_column_oriented(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert pandas default orient='columns' JSON into row dictionaries."""
    mappings = {k: v for k, v in payload.items() if isinstance(v, dict)}
    if not mappings:
        return []

    row_ids: set[str] = set()
    for mapping in mappings.values():
        row_ids.update(str(k) for k in mapping.keys())

    def row_sort_key(value: str):
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    rows: list[dict[str, Any]] = []
    for row_id in sorted(row_ids, key=row_sort_key):
        row: dict[str, Any] = {}
        for column, mapping in mappings.items():
            if row_id in mapping:
                row[column] = mapping[row_id]
            elif str(row_id) in mapping:
                row[column] = mapping[str(row_id)]
        rows.append(row)
    return rows


def load_test_results(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    records: list[dict[str, Any]] = []

    if isinstance(payload, list):
        records = [r for r in payload if isinstance(r, dict)]

    elif isinstance(payload, dict):
        # Notebook 05 writes final_test_results.reset_index().to_json(path, indent=2)
        # without orient=..., so pandas uses orient='columns'. Handle that first.
        column_rows = _rows_from_column_oriented(payload)
        if column_rows and any(
            any(k in row for k in ("model", "model_key", "model_name", "index"))
            for row in column_rows
        ):
            records = column_rows
        else:
            for key in ("results", "models", "test_results"):
                value = payload.get(key)
                if isinstance(value, list):
                    records = [r for r in value if isinstance(r, dict)]
                    break
                if isinstance(value, dict):
                    records = [
                        dict(v, model=k) if isinstance(v, dict)
                        else {"model": k, "value": v}
                        for k, v in value.items()
                    ]
                    break
            if not records:
                records = [payload]

    normalized: dict[str, dict[str, Any]] = {}
    for rec in records:
        model = None
        for key in ("model", "model_key", "model_name", "index"):
            if key in rec:
                candidate = _normalize_model(rec[key])
                if candidate in {"A", "B", "C"}:
                    model = candidate
                    break
        if model is None:
            candidate = base.find_key(rec, {"model", "model_key", "model_name"})
            if candidate is not None:
                candidate = _normalize_model(candidate)
                if candidate in {"A", "B", "C"}:
                    model = candidate
        if model is not None:
            normalized[model] = rec

    return normalized


base.load_test_results = load_test_results


if __name__ == "__main__":
    base.main()
