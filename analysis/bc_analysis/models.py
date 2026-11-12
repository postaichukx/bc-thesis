from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from .config import (
    KNOWN_LOCAL_FEATURES,
    MODEL_SEEDS,
    PROCESSED_DIR,
    RESULTS_DIR,
    ensure_runtime_directories,
)
from .datasets import filter_edges_for_split
from .data import update_manifest


VARIANTS = {
    "original_features_original_graph": ("original", "original"),
    "transparent_features_original_graph": ("transparent", "original"),
    "transparent_features_reconstructed_graph": ("transparent", "reconstructed"),
}
SPLITS = {"temporal": "split_temporal", "leakage_resistant": "split_clean"}
MODELS = ("logistic_regression", "random_forest", "gcn")


@dataclass
class PreparedData:
    nodes: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray
    edges: pd.DataFrame
    train_mask: np.ndarray
    validation_mask: np.ndarray
    test_mask: np.ndarray


def _scale_from_training(features: np.ndarray, train_scope: np.ndarray) -> np.ndarray:
    """Standardize every row using statistics fitted on training rows only."""
    if not train_scope.any():
        raise ValueError("No training nodes available for feature scaling")
    scaler = StandardScaler()
    scaler.fit(features[train_scope])
    return scaler.transform(features).astype(np.float32)


def _best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(labels, predictions, zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "illicit_f1": float(f1_score(labels, predictions, zero_division=0)),
        "illicit_precision": float(precision_score(labels, predictions, zero_division=0)),
        "illicit_recall": float(recall_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _prepare_data(feature_kind: str, graph_kind: str, split_column: str) -> PreparedData:
    nodes_path = PROCESSED_DIR / "model_nodes.csv.gz"
    if not nodes_path.exists():
        raise FileNotFoundError("Run build-datasets before train")
    nodes = pd.read_csv(nodes_path)

    if feature_kind == "transparent":
        nodes = nodes[nodes["transparent_available"].astype(bool)].copy()
        feature_columns = list(KNOWN_LOCAL_FEATURES.values())
    else:
        feature_columns = [f"feature_{index}" for index in range(2, 167)]

    if split_column == "split_clean":
        nodes = nodes[nodes[split_column] != "dropped"].copy()
    nodes = nodes.reset_index(drop=True)

    edge_path = PROCESSED_DIR / f"model_{graph_kind}_edges.csv.gz"
    edges = pd.read_csv(edge_path)
    edges = filter_edges_for_split(edges, nodes, split_column)

    features = nodes[feature_columns].to_numpy(dtype=np.float64)
    labels = nodes["label"].to_numpy(dtype=np.int64)
    train_scope = (nodes[split_column] == "train").to_numpy()
    if not train_scope.any():
        raise ValueError(f"No training nodes for {feature_kind}/{split_column}")
    features = _scale_from_training(features, train_scope)

    train_mask = train_scope & (labels >= 0)
    validation_mask = (nodes[split_column] == "validation").to_numpy() & (labels >= 0)
    test_mask = (nodes[split_column] == "test").to_numpy() & (labels >= 0)
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError(
            f"A labeled train, validation, or test split is empty for {feature_kind}/{split_column}"
        )
    return PreparedData(nodes, features, labels, edges, train_mask, validation_mask, test_mask)


def _train_logistic(data: PreparedData) -> tuple[dict[str, object], list[dict[str, object]]]:
    best: tuple[float, float, LogisticRegression] | None = None
    for c_value in (0.1, 1.0, 10.0):
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2_000,
            random_state=42,
        )
        model.fit(data.features[data.train_mask], data.labels[data.train_mask])
        validation_probabilities = model.predict_proba(data.features[data.validation_mask])[:, 1]
        threshold, validation_f1 = _best_threshold(
            data.labels[data.validation_mask], validation_probabilities
        )
        if best is None or validation_f1 > best[0]:
            best = (validation_f1, threshold, model)
    assert best is not None
    test_probabilities = best[2].predict_proba(data.features[data.test_mask])[:, 1]
    metrics = _metrics(data.labels[data.test_mask], test_probabilities, best[1])
    metrics.update({"seed": 42, "parameters": json.dumps({"C": best[2].C})})
    return metrics, [metrics]


def _select_random_forest(data: PreparedData) -> dict[str, object]:
    best: tuple[float, dict[str, object]] | None = None
    for depth in (10, 20, None):
        for min_leaf in (1, 5):
            model = RandomForestClassifier(
                n_estimators=500,
                max_depth=depth,
                min_samples_leaf=min_leaf,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(data.features[data.train_mask], data.labels[data.train_mask])
            probabilities = model.predict_proba(data.features[data.validation_mask])[:, 1]
            _, validation_f1 = _best_threshold(data.labels[data.validation_mask], probabilities)
            params = {"max_depth": depth, "min_samples_leaf": min_leaf}
            if best is None or validation_f1 > best[0]:
                best = (validation_f1, params)
    assert best is not None
    return best[1]


def _train_random_forest(data: PreparedData) -> tuple[dict[str, object], list[dict[str, object]]]:
    params = _select_random_forest(data)
    runs: list[dict[str, object]] = []
    for seed in MODEL_SEEDS:
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(data.features[data.train_mask], data.labels[data.train_mask])
        validation_probabilities = model.predict_proba(data.features[data.validation_mask])[:, 1]
        threshold, _ = _best_threshold(data.labels[data.validation_mask], validation_probabilities)
        test_probabilities = model.predict_proba(data.features[data.test_mask])[:, 1]
        metrics = _metrics(data.labels[data.test_mask], test_probabilities, threshold)
        metrics.update({"seed": seed, "parameters": json.dumps(params)})
        runs.append(metrics)
    return runs[0], runs


def _normalized_adjacency(data: PreparedData, torch):
    id_to_index = {int(tx_id): index for index, tx_id in enumerate(data.nodes["txId"])}
    sources: list[int] = []
    destinations: list[int] = []
    for source, destination in data.edges[["txId1", "txId2"]].itertuples(index=False):
        source_index = id_to_index.get(int(source))
        destination_index = id_to_index.get(int(destination))
        if source_index is None or destination_index is None:
            continue
        sources.extend((source_index, destination_index))
        destinations.extend((destination_index, source_index))
    for index in range(len(data.nodes)):
        sources.append(index)
        destinations.append(index)
    indices = torch.tensor([sources, destinations], dtype=torch.long)
    values = torch.ones(len(sources), dtype=torch.float32)
    degree = torch.zeros(len(data.nodes), dtype=torch.float32)
    degree.scatter_add_(0, indices[0], values)
    inverse_sqrt = degree.clamp(min=1).pow(-0.5)
    normalized_values = inverse_sqrt[indices[0]] * values * inverse_sqrt[indices[1]]
    return torch.sparse_coo_tensor(
        indices, normalized_values, (len(data.nodes), len(data.nodes))
    ).coalesce()


def _train_gcn(data: PreparedData) -> tuple[dict[str, object], list[dict[str, object]]]:
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("GCN requires the optional 'gcn' dependencies") from error

    class GCN(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.first = nn.Linear(input_size, 128, bias=False)
            self.second = nn.Linear(128, 2, bias=False)
            self.dropout = nn.Dropout(0.5)

        def forward(self, features, adjacency):
            hidden = torch.sparse.mm(adjacency, features)
            hidden = torch.relu(self.first(hidden))
            hidden = self.dropout(hidden)
            hidden = torch.sparse.mm(adjacency, hidden)
            return self.second(hidden)

    adjacency = _normalized_adjacency(data, torch)
    features = torch.tensor(data.features, dtype=torch.float32)
    labels = torch.tensor(data.labels, dtype=torch.long)
    train_indices = torch.tensor(np.flatnonzero(data.train_mask), dtype=torch.long)
    validation_indices = torch.tensor(np.flatnonzero(data.validation_mask), dtype=torch.long)
    test_indices = torch.tensor(np.flatnonzero(data.test_mask), dtype=torch.long)
    train_labels = data.labels[data.train_mask]
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor(
        [len(train_labels) / (2 * max(1, count)) for count in class_counts],
        dtype=torch.float32,
    )

    runs: list[dict[str, object]] = []
    for seed in MODEL_SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = GCN(features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        best_state = None
        best_validation_f1 = -1.0
        best_threshold = 0.5
        stale_epochs = 0

        for _epoch in range(200):
            model.train()
            optimizer.zero_grad()
            logits = model(features, adjacency)
            loss = criterion(logits[train_indices], labels[train_indices])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                probabilities = torch.softmax(model(features, adjacency), dim=1)[:, 1]
                validation_probabilities = probabilities[validation_indices].cpu().numpy()
            threshold, validation_f1 = _best_threshold(
                data.labels[data.validation_mask], validation_probabilities
            )
            if validation_f1 > best_validation_f1 + 1e-8:
                best_validation_f1 = validation_f1
                best_threshold = threshold
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= 8:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(features, adjacency), dim=1)[:, 1]
            test_probabilities = probabilities[test_indices].cpu().numpy()
        metrics = _metrics(data.labels[data.test_mask], test_probabilities, best_threshold)
        metrics.update(
            {
                "seed": seed,
                "parameters": json.dumps(
                    {"hidden": 128, "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4}
                ),
            }
        )
        runs.append(metrics)
    return runs[0], runs


def train_all(
    variants: tuple[str, ...] = tuple(VARIANTS),
    splits: tuple[str, ...] = tuple(SPLITS),
    models: tuple[str, ...] = MODELS,
) -> dict[str, int]:
    ensure_runtime_directories()
    invalid = (
        set(variants) - set(VARIANTS)
        | set(splits) - set(SPLITS)
        | set(models) - set(MODELS)
    )
    if invalid:
        raise ValueError("Unknown experiment options: " + ", ".join(sorted(invalid)))

    rows: list[dict[str, object]] = []
    for variant in variants:
        feature_kind, graph_kind = VARIANTS[variant]
        for split in splits:
            data = _prepare_data(feature_kind, graph_kind, SPLITS[split])
            for model_name in models:
                if model_name == "logistic_regression":
                    _, runs = _train_logistic(data)
                elif model_name == "random_forest":
                    _, runs = _train_random_forest(data)
                else:
                    _, runs = _train_gcn(data)
                for run in runs:
                    rows.append(
                        {
                            "variant": variant,
                            "split": split,
                            "model": model_name,
                            "train_labeled": int(data.train_mask.sum()),
                            "validation_labeled": int(data.validation_mask.sum()),
                            "test_labeled": int(data.test_mask.sum()),
                            "graph_nodes": len(data.nodes),
                            "graph_edges": len(data.edges),
                            **run,
                        }
                    )

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "model_runs.csv", index=False)
    metric_columns = [
        "illicit_f1",
        "illicit_precision",
        "illicit_recall",
        "pr_auc",
        "balanced_accuracy",
    ]
    summary = (
        results.groupby(["variant", "split", "model"], as_index=False)[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary.to_csv(RESULTS_DIR / "model_summary.csv", index=False)
    result = {
        "configurations": int(results.groupby(["variant", "split", "model"]).ngroups),
        "runs": len(results),
    }
    update_manifest(
        {
            "training": {
                "variants": list(variants),
                "splits": list(splits),
                "models": list(models),
                "model_seeds": list(MODEL_SEEDS),
                "normalization": "StandardScaler fitted on training nodes only",
                **result,
            }
        }
    )
    return result
