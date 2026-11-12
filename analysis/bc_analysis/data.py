from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .blockbook import BlockbookCache, BlockbookClient, fetch_transactions
from .config import (
    BLOCK_RANGES,
    CACHE_DIR,
    DEFAULT_BLOCKBOOK_URL,
    DEFAULT_SEED,
    PROCESSED_DIR,
    SELECTED_TIMESTEPS,
    TIMESTEP_SPLIT,
    ensure_runtime_directories,
)
from .features import transaction_features


FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_columns() -> list[str]:
    return ["txId", "timestep"] + [f"feature_{index}" for index in range(2, 167)]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def update_manifest(payload: dict[str, object]) -> Path:
    path = PROCESSED_DIR / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat()}
    manifest.update({
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "libraries": {
            name: version
            for name in ("numpy", "pandas", "scikit-learn", "matplotlib", "torch")
            if (version := _package_version(name)) is not None
        },
        "selected_timesteps": list(SELECTED_TIMESTEPS),
        "split": TIMESTEP_SPLIT,
        "seed": DEFAULT_SEED,
        "research_sources": {
            "paper_doi": "10.1016/j.fsidi.2026.302124",
            "paper_sha256": "db095b7f1b71fbedcde956125d1242e3055650bf4b97dbf99816788904fb63e5",
            "bitcoin_tx_subgraph_builder_commit": "8e01503c0f9f384402288509bdd5fc102590bcfb",
            "public_mapping_commit": "a5dbbcc655783003d942d0d54ad8404f0b7b3b93",
        },
    })
    manifest.update(payload)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def prepare(elliptic_dir: Path, mapping_path: Path) -> dict[str, int]:
    ensure_runtime_directories()
    source_paths = {
        "features": elliptic_dir / FEATURES_FILE,
        "classes": elliptic_dir / CLASSES_FILE,
        "edges": elliptic_dir / EDGES_FILE,
        "mapping": mapping_path,
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(missing))

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source_paths["features"],
        header=None,
        names=_feature_columns(),
        chunksize=20_000,
    ):
        selected = chunk[chunk["timestep"].isin(SELECTED_TIMESTEPS)]
        if not selected.empty:
            chunks.append(selected)
    features = pd.concat(chunks, ignore_index=True)
    features["txId"] = features["txId"].astype("int64")
    features["timestep"] = features["timestep"].astype("int16")

    classes = pd.read_csv(source_paths["classes"])
    classes["txId"] = classes["txId"].astype("int64")
    selected_ids = set(features["txId"].tolist())
    classes = classes[classes["txId"].isin(selected_ids)].copy()

    mapping = pd.read_csv(source_paths["mapping"])
    mapping = mapping.rename(columns={"transaction": "txid"})
    if not {"txId", "txid"}.issubset(mapping.columns):
        raise ValueError("Mapping must contain txId and transaction/txid columns")
    mapping["txId"] = mapping["txId"].astype("int64")
    mapping = mapping[mapping["txId"].isin(selected_ids)][["txId", "txid"]].copy()
    mapping["txid"] = mapping["txid"].astype(str).str.lower()

    edges = pd.read_csv(source_paths["edges"])
    edges = edges[
        edges["txId1"].isin(selected_ids) & edges["txId2"].isin(selected_ids)
    ].copy()

    features.to_csv(PROCESSED_DIR / "selected_features.csv.gz", index=False)
    classes.to_csv(PROCESSED_DIR / "selected_classes.csv.gz", index=False)
    mapping.to_csv(PROCESSED_DIR / "selected_mapping.csv.gz", index=False)
    edges.to_csv(PROCESSED_DIR / "selected_original_edges.csv.gz", index=False)

    merged = features[["txId", "timestep"]].merge(classes, on="txId", how="left")
    merged["class"] = merged["class"].fillna("unknown").astype(str)
    counts = {
        "nodes": int(len(features)),
        "mapped": int(mapping["txId"].nunique()),
        "unmapped": int(len(features) - mapping["txId"].nunique()),
        "edges": int(len(edges)),
        "licit": int((merged["class"] == "2").sum()),
        "illicit": int((merged["class"] == "1").sum()),
        "unknown": int((merged["class"] == "unknown").sum()),
    }
    counts["by_timestep"] = {
        str(step): {
            "nodes": int((merged["timestep"] == step).sum()),
            "licit": int(((merged["timestep"] == step) & (merged["class"] == "2")).sum()),
            "illicit": int(((merged["timestep"] == step) & (merged["class"] == "1")).sum()),
            "unknown": int(((merged["timestep"] == step) & (merged["class"] == "unknown")).sum()),
        }
        for step in SELECTED_TIMESTEPS
    }
    update_manifest(
        {
            "inputs": {
                name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for name, path in source_paths.items()
            },
            "counts": counts,
            "block_ranges": BLOCK_RANGES,
        }
    )
    return counts


def _client(base_url: str) -> BlockbookClient:
    cache = BlockbookCache(CACHE_DIR / "blockbook.sqlite3")
    return BlockbookClient(base_url=base_url, cache=cache)


def collect(
    mode: str,
    base_url: str = DEFAULT_BLOCKBOOK_URL,
    workers: int = 4,
    margin: int = 0,
    limit: int | None = None,
) -> dict[str, int]:
    ensure_runtime_directories()
    client = _client(base_url)
    transactions: list[dict] = []
    failures: list[tuple[str, str]] = []

    if mode == "selected":
        mapping_path = PROCESSED_DIR / "selected_mapping.csv.gz"
        if not mapping_path.exists():
            raise FileNotFoundError("Run prepare before collect")
        mapping = pd.read_csv(mapping_path)
        txids = mapping["txid"].astype(str).tolist()
        if limit is not None:
            txids = txids[:limit]
        fetched, failures = fetch_transactions(client, txids, workers=workers)
        timestep_by_hash = dict(zip(mapping["txid"], mapping["txId"], strict=False))
        features = pd.read_csv(PROCESSED_DIR / "selected_features.csv.gz", usecols=["txId", "timestep"])
        step_by_id = dict(zip(features["txId"], features["timestep"], strict=False))
        for tx in fetched:
            txid = str(tx.get("txid") or "")
            elliptic_id = timestep_by_hash.get(txid)
            tx["_timestep"] = int(step_by_id.get(elliptic_id, 0))
            transactions.append(tx)
    elif mode == "blocks":
        collected_blocks = 0
        for timestep, (first, last) in BLOCK_RANGES.items():
            for height in range(max(0, first - margin), last + margin + 1):
                if limit is not None and collected_blocks >= limit:
                    break
                try:
                    block_txs = client.block_transactions(height)
                    for tx in block_txs:
                        tx["_timestep"] = timestep
                    transactions.extend(block_txs)
                except Exception as error:
                    failures.append((str(height), str(error)))
                collected_blocks += 1
            if limit is not None and collected_blocks >= limit:
                break
    else:
        raise ValueError("mode must be 'selected' or 'blocks'")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for transaction in transactions:
        row = transaction_features(transaction)
        txid = str(row["txid"])
        if not txid or txid in seen:
            continue
        seen.add(txid)
        row["timestep"] = int(transaction.get("_timestep") or 0)
        rows.append(row)

    output = PROCESSED_DIR / (
        "block_candidates.csv.gz" if mode == "blocks" else "selected_transactions.csv.gz"
    )
    pd.DataFrame(rows).to_csv(output, index=False)
    if failures:
        pd.DataFrame(failures, columns=["item", "error"]).to_csv(
            PROCESSED_DIR / f"collect_{mode}_failures.csv", index=False
        )
    result = {"transactions": len(rows), "failures": len(failures)}
    update_manifest(
        {
            "collection": {
                "mode": mode,
                "base_url": base_url,
                "workers": workers,
                "block_margin": margin,
                "limit": limit,
                **result,
            }
        }
    )
    return result
