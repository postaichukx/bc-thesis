from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import (
    BLOCKCHAIN_DATA_DIR,
    KNOWN_LOCAL_FEATURES,
    MODEL_DATA_DIR,
    TIMESTEP_SPLIT,
    ensure_runtime_directories,
)
from ..kap_04_blockchain.transaction_collection import update_manifest
from ..kap_04_blockchain.transaction_features import reconstruct_edges, split_addresses


def _load_collected() -> pd.DataFrame:
    selected = BLOCKCHAIN_DATA_DIR / "selected_transactions.csv.gz"
    blocks = BLOCKCHAIN_DATA_DIR / "block_candidates.csv.gz"
    if blocks.exists():
        return pd.read_csv(blocks)
    if selected.exists():
        return pd.read_csv(selected)
    raise FileNotFoundError("Run collect before build-datasets")


def _label(value: object) -> int:
    text = str(value)
    if text == "1":
        return 1  # illicit
    if text == "2":
        return 0  # licit
    return -1


def _address_set(row: pd.Series) -> set[str]:
    return split_addresses(row.get("input_addresses")) | split_addresses(
        row.get("output_addresses")
    )


def _clean_split(nodes: pd.DataFrame) -> pd.Series:
    result = pd.Series("dropped", index=nodes.index, dtype="object")
    occupied: set[str] = set()
    for split in ("train", "validation", "test"):
        split_rows = nodes[nodes["split_temporal"] == split]
        accepted_addresses: set[str] = set()
        for index, row in split_rows.iterrows():
            if not bool(row["transparent_available"]):
                continue
            addresses = _address_set(row)
            if addresses & occupied:
                continue
            result.loc[index] = split
            accepted_addresses.update(addresses)
        occupied.update(accepted_addresses)
    return result


def build_datasets() -> dict[str, int]:
    ensure_runtime_directories()
    required = {
        "features": BLOCKCHAIN_DATA_DIR / "selected_features.csv.gz",
        "classes": BLOCKCHAIN_DATA_DIR / "selected_classes.csv.gz",
        "mapping": BLOCKCHAIN_DATA_DIR / "selected_mapping.csv.gz",
        "edges": BLOCKCHAIN_DATA_DIR / "selected_original_edges.csv.gz",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Run prepare first; missing: " + ", ".join(missing))

    features = pd.read_csv(required["features"])
    classes = pd.read_csv(required["classes"])
    mapping = pd.read_csv(required["mapping"])
    original_edges = pd.read_csv(required["edges"])
    collected = _load_collected()

    classes["label"] = classes["class"].map(_label).astype("int8")
    mapping["txid"] = mapping["txid"].astype(str).str.lower()
    collected["txid"] = collected["txid"].astype(str).str.lower()
    transparent = mapping.merge(collected, on="txid", how="left")
    transparent = transparent.drop_duplicates("txId", keep="first")

    nodes = features.merge(classes[["txId", "label"]], on="txId", how="left")
    nodes["label"] = nodes["label"].fillna(-1).astype("int8")
    keep_columns = [
        "txId",
        "txid",
        "block_height",
        "block_time",
        "input_addresses",
        "output_addresses",
        "predecessor_txids",
        *KNOWN_LOCAL_FEATURES.values(),
    ]
    available_columns = [column for column in keep_columns if column in transparent.columns]
    nodes = nodes.merge(transparent[available_columns], on="txId", how="left")
    nodes = nodes.copy()
    nodes["transparent_available"] = nodes["txid"].notna()
    nodes["split_temporal"] = nodes["timestep"].map(TIMESTEP_SPLIT)
    nodes["split_clean"] = _clean_split(nodes)

    reconstructed = reconstruct_edges(
        nodes[nodes["transparent_available"]][
            ["txId", "txid", "predecessor_txids"]
        ].fillna("").to_dict("records")
    )
    reconstructed_edges = pd.DataFrame(sorted(reconstructed), columns=["txId1", "txId2"])

    nodes.to_csv(MODEL_DATA_DIR / "model_nodes.csv.gz", index=False)
    original_edges.to_csv(MODEL_DATA_DIR / "model_original_edges.csv.gz", index=False)
    reconstructed_edges.to_csv(
        MODEL_DATA_DIR / "model_reconstructed_edges.csv.gz", index=False
    )

    clean_counts = nodes["split_clean"].value_counts().to_dict()
    result = {
        "nodes": int(len(nodes)),
        "transparent_nodes": int(nodes["transparent_available"].sum()),
        "original_edges": int(len(original_edges)),
        "reconstructed_edges": int(len(reconstructed_edges)),
        "clean_train": int(clean_counts.get("train", 0)),
        "clean_validation": int(clean_counts.get("validation", 0)),
        "clean_test": int(clean_counts.get("test", 0)),
        "clean_dropped": int(clean_counts.get("dropped", 0)),
    }
    update_manifest({"dataset_build": result})
    return result


def filter_edges_for_split(
    edges: pd.DataFrame, nodes: pd.DataFrame, split_column: str
) -> pd.DataFrame:
    split_by_id = nodes.set_index("txId")[split_column].to_dict()
    filtered = edges[
        edges["txId1"].isin(split_by_id) & edges["txId2"].isin(split_by_id)
    ].copy()
    if split_column == "split_clean":
        source_split = filtered["txId1"].map(split_by_id)
        target_split = filtered["txId2"].map(split_by_id)
        filtered = filtered[
            (source_split == target_split) & (source_split != "dropped")
        ].copy()
    return filtered
