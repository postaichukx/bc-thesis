from __future__ import annotations

import csv
import gzip
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASSES_FILE, EDGES_FILE, FEATURES_FILE, _feature_columns, sha256_file


LABELS = {"1": "illicit", "2": "licit", "unknown": "unknown"}
TARGETS = {"1": 1, "2": 0, "unknown": -1}


def _read_table(path: Path, columns: list[str], dtypes: dict) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=dtypes, keep_default_na=False)
    if list(table.columns) != columns:
        raise ValueError(f"{path.name}: expected columns {columns}")
    return table


def audit_elliptic(elliptic_dir: Path, output_dir: Path) -> dict:
    """Audit all original nodes; unknown nodes remain without a target class."""
    inputs = {
        name: elliptic_dir / filename
        for name, filename in (
            ("features", FEATURES_FILE), ("classes", CLASSES_FILE), ("edges", EDGES_FILE)
        )
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path}")
    if elliptic_dir.resolve() == output_dir.resolve():
        raise ValueError("Raw inputs and audit outputs must use different directories")

    with inputs["features"].open(newline="", encoding="utf-8") as handle:
        first = next(csv.reader(handle), [])
    if len(first) != 167:
        raise ValueError("Features must have 167 columns: txId and 166 features, without a header")

    classes = _read_table(inputs["classes"], ["txId", "class"], {"txId": "int64", "class": str})
    edges = _read_table(inputs["edges"], ["txId1", "txId2"], {"txId1": "int64", "txId2": "int64"})
    if classes["txId"].duplicated().any():
        raise ValueError("Duplicate class identifiers would multiply joined rows")
    invalid_classes = sorted(set(classes["class"]) - set(LABELS))
    if invalid_classes:
        raise ValueError(f"Unsupported class codes: {invalid_classes}")
    if (classes["txId"] <= 0).any() or (edges[["txId1", "txId2"]] <= 0).any().any():
        raise ValueError("Transaction identifiers must be positive integers")

    labels = classes.set_index("txId")["class"]
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".elliptic_nodes-", suffix=".csv.gz", dir=output_dir)
    os.close(descriptor)
    temporary_path = Path(temporary)
    metadata = []
    bad_values = bad_timesteps = bad_ids = 0
    rows = 0
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8", newline="", compresslevel=1) as handle:
            for chunk in pd.read_csv(
                inputs["features"], header=None, names=_feature_columns(),
                dtype={"txId": "int64"}, chunksize=20_000,
            ):
                numbers = chunk.drop(columns="txId").to_numpy(dtype="float64")
                bad_values += int((~np.isfinite(numbers)).sum())
                bad_timesteps += int((~chunk["timestep"].isin(range(1, 50))).sum())
                bad_ids += int((chunk["txId"] <= 0).sum())
                codes = chunk["txId"].map(labels)
                annotations = pd.DataFrame({
                    "class_code": codes,
                    "class": codes.map(LABELS),
                    "label": codes.map(TARGETS).astype("Int8"),
                    "is_labeled": codes.isin(["1", "2"]),
                }, index=chunk.index)
                chunk = pd.concat([chunk, annotations], axis=1)
                metadata.append(chunk[["txId", "timestep", "class"]].copy())
                chunk.to_csv(handle, index=False, header=rows == 0)
                rows += len(chunk)
        if not metadata:
            raise ValueError("Features file has no transaction rows")
        nodes = pd.concat(metadata, ignore_index=True)
        node_ids = set(nodes["txId"])
        class_ids = set(classes["txId"])
        edge_ids = set(edges["txId1"]) | set(edges["txId2"])
        duplicate_nodes = int(nodes["txId"].duplicated().sum())
        duplicate_edges = int(edges.duplicated().sum())
        self_loops = int((edges["txId1"] == edges["txId2"]).sum())
        missing_classes = sorted(node_ids - class_ids)
        extra_classes = sorted(class_ids - node_ids)
        missing_endpoints = sorted(edge_ids - node_ids)
        by_step = {}
        for step, group in nodes.groupby("timestep", sort=True):
            by_step[str(step)] = {
                "nodes": len(group),
                **{name: int((group["class"] == name).sum()) for name in LABELS.values()},
            }
        step_lookup = nodes.drop_duplicates("txId").set_index("txId")["timestep"]
        source_steps = edges["txId1"].map(step_lookup)
        target_steps = edges["txId2"].map(step_lookup)
        cross_step_edges = int((source_steps.notna() & target_steps.notna() & source_steps.ne(target_steps)).sum())
        errors = []
        for count, description in (
            (bad_values, "missing, infinite or nonnumeric feature values"),
            (bad_timesteps, "timesteps outside integers 1..49"),
            (bad_ids, "nonpositive feature identifiers"),
            (duplicate_nodes, "duplicate feature identifiers"),
            (duplicate_edges, "duplicate edges"),
            (self_loops, "self-loop edges"),
            (len(missing_classes), "nodes without a class record"),
            (len(extra_classes), "class records without features"),
            (len(missing_endpoints), "edge endpoints without features"),
        ):
            if count:
                errors.append(f"{count} {description}")
        provenance_path = elliptic_dir / "source.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {
            "acquisition": "not recorded; local inputs supplied by caller"
        }
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "all original timesteps; no mapping or blockchain queries",
            "valid": not errors,
            "validation_errors": errors,
            "provenance": provenance,
            "inputs": {name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for name, path in inputs.items()},
            "counts": {
                "nodes": rows, "feature_columns": 166, "classes": len(classes), "edges": len(edges),
                **{name: int((nodes["class"] == name).sum()) for name in LABELS.values()},
                "timesteps": int(nodes["timestep"].nunique()),
            },
            "checks": {
                "duplicate_nodes": duplicate_nodes, "duplicate_edges": duplicate_edges,
                "self_loops": self_loops, "invalid_feature_values": bad_values,
                "missing_class_ids": len(missing_classes), "extra_class_ids": len(extra_classes),
                "missing_edge_endpoints": len(missing_endpoints), "cross_timestep_edges": cross_step_edges,
            },
            "by_timestep": by_step,
            "label_policy": {
                "classes": {"0": "licit", "1": "illicit"},
                "unlabeled_sentinel": -1,
                "unknown_usage": "Retain features and graph context; exclude from supervised loss and ground-truth metrics. No unknown target class.",
                "is_labeled": "Label availability only; intersect with a later train/validation/test split.",
            },
            "limitations": [
                "Internal Elliptic identifiers are not public Bitcoin txids.",
                "Unknown labels provide no ground truth; supervised masks must not include label -1.",
                "The published graph is incomplete and feature definitions are partly anonymized.",
                "Address history and successor features may contain information unavailable at transaction time.",
                "Dataset labels are inherited annotations, not independently proven fraud.",
                "No normalization, splitting, model training, mapping or PostgreSQL migration is performed by this audit.",
            ],
            "outputs": {},
        }
        if not errors:
            destination = output_dir / "elliptic_nodes.csv.gz"
            temporary_path.replace(destination)
            report["outputs"]["nodes"] = str(destination.resolve())
        else:
            report["examples"] = {
                "missing_class_ids": missing_classes[:10], "extra_class_ids": extra_classes[:10],
                "missing_edge_endpoints": missing_endpoints[:10],
            }
        # Publish the report last so it never advertises an unfinished node file.
        report_path = output_dir / "audit.json"
        descriptor, name = tempfile.mkstemp(prefix=".audit-", suffix=".json", dir=output_dir)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        Path(name).replace(report_path)
        return report
    finally:
        temporary_path.unlink(missing_ok=True)
