from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from .config import KNOWN_LOCAL_FEATURES


def _as_satoshi(value: object) -> int:
    """Convert a Blockbook amount to satoshi without using binary floats."""
    if value is None or value == "":
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return 0
    if "." not in text:
        return int(text)
    whole, fraction = text.split(".", 1)
    fraction = (fraction + "00000000")[:8]
    sign = -1 if whole.startswith("-") else 1
    whole_abs = whole.lstrip("+-") or "0"
    return sign * (int(whole_abs) * 100_000_000 + int(fraction))


def _addresses(io_item: dict) -> list[str]:
    raw = io_item.get("addresses") or []
    if isinstance(raw, str):
        return [raw]
    return [str(value) for value in raw if value]


def _values(items: Sequence[dict]) -> list[int]:
    return [_as_satoshi(item.get("value")) for item in items]


def _pearson(values: Sequence[int]) -> float:
    if len(values) < 2 or len(set(values)) < 2:
        return 0.0
    indices = np.arange(len(values), dtype=float)
    result = float(np.corrcoef(indices, np.asarray(values, dtype=float))[0, 1])
    return 0.0 if not math.isfinite(result) else result


def _rankdata(values: Sequence[int]) -> np.ndarray:
    """Return average ranks for ties, equivalent to scipy.stats.rankdata."""
    array = np.asarray(values)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(values: Sequence[int]) -> float:
    if len(values) < 2 or len(set(values)) < 2:
        return 0.0
    indices = np.arange(len(values), dtype=float)
    ranks = _rankdata(values)
    result = float(np.corrcoef(indices, ranks)[0, 1])
    return 0.0 if not math.isfinite(result) else result


def _stats(values: Sequence[int], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_pearson": 0.0,
            f"{prefix}_spearman": 0.0,
        }
    array = np.asarray(values, dtype=float)
    return {
        f"{prefix}_min": float(array.min()),
        f"{prefix}_max": float(array.max()),
        f"{prefix}_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_pearson": _pearson(values),
        f"{prefix}_spearman": _spearman(values),
    }


def transaction_features(transaction: dict) -> dict[str, object]:
    """Compute the 18 recovered transaction-scoped Elliptic features."""
    inputs = list(transaction.get("vin") or [])
    outputs = list(transaction.get("vout") or [])
    input_values = _values(inputs)
    output_values = _values(outputs)

    input_addresses = {address for item in inputs for address in _addresses(item)}
    output_addresses = {address for item in outputs for address in _addresses(item)}

    value = transaction.get("value")
    if value is None:
        value_satoshi = sum(output_values)
    else:
        value_satoshi = _as_satoshi(value)

    value_in = transaction.get("valueIn")
    if transaction.get("fees") is not None:
        fee_satoshi = _as_satoshi(transaction.get("fees"))
    elif value_in is not None:
        fee_satoshi = max(0, _as_satoshi(value_in) - value_satoshi)
    else:
        fee_satoshi = max(0, sum(input_values) - sum(output_values))

    result: dict[str, object] = {
        "txid": str(transaction.get("txid") or transaction.get("hash") or ""),
        "block_height": int(transaction.get("blockHeight") or 0),
        "block_time": int(transaction.get("blockTime") or 0),
        "value_satoshi": value_satoshi,
        "fee_satoshi": fee_satoshi,
        "input_count": len(inputs),
        "output_count": len(outputs),
        "unique_input_addresses": len(input_addresses),
        "unique_output_addresses": len(output_addresses),
        "input_addresses": "|".join(sorted(input_addresses)),
        "output_addresses": "|".join(sorted(output_addresses)),
        "predecessor_txids": "|".join(
            sorted({str(item["txid"]) for item in inputs if item.get("txid")})
        ),
    }
    result.update(_stats(input_values, "input_value"))
    result.update(_stats(output_values, "output_value"))
    return result


def transparent_feature_names() -> list[str]:
    return list(KNOWN_LOCAL_FEATURES.values())


def split_addresses(value: object) -> set[str]:
    if value is None:
        return set()
    return {address for address in str(value).split("|") if address}


def reconstruct_edges(rows: Iterable[dict[str, object]]) -> set[tuple[int, int]]:
    rows_list = list(rows)
    by_hash = {str(row["txid"]): int(row["txId"]) for row in rows_list}
    edges: set[tuple[int, int]] = set()
    for row in rows_list:
        destination = int(row["txId"])
        for predecessor in str(row.get("predecessor_txids") or "").split("|"):
            source = by_hash.get(predecessor)
            if source is not None and source != destination:
                edges.add((source, destination))
    return edges
