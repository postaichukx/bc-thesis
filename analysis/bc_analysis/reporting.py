from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from .config import PROCESSED_DIR, RESULTS_DIR, THESIS_FIGURES_DIR, ensure_runtime_directories


COLORS = {"licit": "#377eb8", "illicit": "#e41a1c", "unknown": "#a6a6a6"}


def _save(figure: plt.Figure, name: str) -> None:
    THESIS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(THESIS_FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(THESIS_FIGURES_DIR / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _class_distribution() -> None:
    features = pd.read_csv(PROCESSED_DIR / "selected_features.csv.gz", usecols=["txId", "timestep"])
    classes = pd.read_csv(PROCESSED_DIR / "selected_classes.csv.gz")
    frame = features.merge(classes, on="txId", how="left")
    frame["class"] = frame["class"].fillna("unknown").astype(str)
    frame["class"] = frame["class"].replace({"1": "illicit", "2": "licit"})
    table = (
        frame.groupby(["timestep", "class"]).size().unstack(fill_value=0)
        .reindex(columns=["licit", "illicit", "unknown"], fill_value=0)
    )

    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    bottom = np.zeros(len(table))
    slovak_labels = {"licit": "licitné", "illicit": "ilicitné", "unknown": "neoznačené"}
    for column in table.columns:
        values = table[column].to_numpy()
        axis.bar(
            table.index.astype(str),
            values,
            bottom=bottom,
            label=slovak_labels[column],
            color=COLORS[column],
        )
        bottom += values
    axis.set_xlabel("Časový krok")
    axis.set_ylabel("Počet transakcií")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save(figure, "elliptic_class_distribution")


def _topological_layout(node_ids: np.ndarray, edges: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Create a deterministic layered layout without an external graph package."""
    ids = [int(value) for value in node_ids]
    node_set = set(ids)
    successors: dict[int, list[int]] = {node_id: [] for node_id in ids}
    indegree = {node_id: 0 for node_id in ids}
    for source, target in edges[["txId1", "txId2"]].itertuples(index=False):
        source_id, target_id = int(source), int(target)
        if source_id not in node_set or target_id not in node_set:
            continue
        successors[source_id].append(target_id)
        indegree[target_id] += 1

    ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    layer = {node_id: 0 for node_id in ready}
    cursor = 0
    while cursor < len(ready):
        source = ready[cursor]
        cursor += 1
        for target in successors[source]:
            layer[target] = max(layer.get(target, 0), layer[source] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)

    # Bitcoin payment edges are acyclic. The fallback keeps the figure usable
    # if an inconsistent edge ever appears in a derived input file.
    for node_id in ids:
        layer.setdefault(node_id, 0)

    by_layer: dict[int, list[int]] = {}
    for node_id in ids:
        by_layer.setdefault(layer[node_id], []).append(node_id)

    positions: dict[int, tuple[float, float]] = {}
    largest_layer = max(by_layer, default=0)
    for layer_number, members in by_layer.items():
        members.sort()
        x = layer_number / max(1, largest_layer)
        if len(members) == 1:
            positions[members[0]] = (x, 0.5)
            continue
        for index, node_id in enumerate(members):
            positions[node_id] = (x, index / (len(members) - 1))
    return positions


def _graph_overview() -> None:
    """Render every selected Elliptic component using its real nodes and edges."""
    nodes = pd.read_csv(
        PROCESSED_DIR / "model_nodes.csv.gz",
        usecols=["txId", "timestep", "label"],
    )
    edges = pd.read_csv(PROCESSED_DIR / "model_original_edges.csv.gz")
    steps = sorted(int(value) for value in nodes["timestep"].unique())

    figure, axes = plt.subplots(2, 3, figsize=(10.4, 7.4), constrained_layout=True)
    for axis, step in zip(axes.flat, steps):
        step_nodes = nodes[nodes["timestep"] == step].copy()
        node_ids = set(int(value) for value in step_nodes["txId"])
        step_edges = edges[
            edges["txId1"].isin(node_ids) & edges["txId2"].isin(node_ids)
        ].copy()
        positions = _topological_layout(step_nodes["txId"].to_numpy(), step_edges)

        segments = [
            [positions[int(source)], positions[int(target)]]
            for source, target in step_edges[["txId1", "txId2"]].itertuples(index=False)
            if int(source) in positions and int(target) in positions
        ]
        if segments:
            axis.add_collection(
                LineCollection(segments, colors="#6b7280", linewidths=0.20, alpha=0.10, zorder=1)
            )

        labels = step_nodes["label"].astype(int)
        for label, name in ((-1, "unknown"), (0, "licit"), (1, "illicit")):
            selected = step_nodes[labels == label]
            if selected.empty:
                continue
            coordinates = np.asarray([positions[int(node_id)] for node_id in selected["txId"]])
            axis.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                s=2.1 if label != -1 else 0.75,
                c=COLORS[name],
                alpha=0.90 if label != -1 else 0.42,
                linewidths=0,
                zorder=3 if label != -1 else 2,
            )

        axis.set_title(
            f"krok {step}: {len(step_nodes):,} vrcholov, {len(step_edges):,} hrán".replace(",", " "),
            fontsize=9,
        )
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#d1d5db")
            spine.set_linewidth(0.7)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["licit"],
               markeredgecolor="none", markersize=5, label="licitné označenie"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["illicit"],
               markeredgecolor="none", markersize=5, label="ilicitné označenie"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["unknown"],
               markeredgecolor="none", markersize=5, label="neoznačené"),
        Line2D([0], [0], color="#6b7280", linewidth=0.8, label="tok medzi transakciami"),
    ]
    figure.legend(handles=legend, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    figure.suptitle(
        "Skutočná štruktúra šiestich vybraných komponentov Elliptic",
        fontsize=12,
        fontweight="bold",
    )
    _save(figure, "elliptic_graph_overview")


def _mapping_accuracy(path: Path) -> None:
    summary = pd.read_csv(path)
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(summary))
    axis.bar(x - 0.18, summary["top1_rate"], width=0.36, label="top-1", color="#4daf4a")
    topk_column = next(column for column in summary if column.startswith("top") and column.endswith("_rate") and column != "top1_rate")
    axis.bar(x + 0.18, summary[topk_column], width=0.36, label=topk_column.replace("_rate", ""), color="#984ea3")
    axis.set_xticks(x, summary["stage"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Podiel správnych priradení")
    axis.set_xlabel("Použitá skupina príznakov")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save(figure, "mapping_holdout_accuracy")


def _split_retention() -> None:
    nodes = pd.read_csv(
        PROCESSED_DIR / "model_nodes.csv.gz",
        usecols=["split_temporal", "split_clean"],
    )
    order = ["train", "validation", "test"]
    original = nodes["split_temporal"].value_counts().reindex(order, fill_value=0)
    retained = nodes["split_clean"].value_counts().reindex(order, fill_value=0)
    x = np.arange(len(order))
    labels = ["tréning", "validácia", "test"]

    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    axis.bar(x - 0.2, original.to_numpy(), 0.4, label="časové rozdelenie", color="#377eb8")
    axis.bar(x + 0.2, retained.to_numpy(), 0.4, label="bez prekryvu adries", color="#ff7f00")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Počet transakcií")
    axis.set_xlabel("Časť dát")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    for offset, values in ((-0.2, original), (0.2, retained)):
        for index, value in enumerate(values):
            axis.text(index + offset, value, f"{int(value):,}".replace(",", " "), ha="center", va="bottom", fontsize=8)
    figure.tight_layout()
    _save(figure, "split_retention")


def _metric_profile(path: Path) -> None:
    runs = pd.read_csv(path)
    selected = runs[
        (runs["variant"] == "transparent_features_original_graph")
        & (runs["model"] == "random_forest")
    ].copy()
    metrics = [
        ("illicit_precision", "precíznosť"),
        ("illicit_recall", "úplnosť"),
        ("illicit_f1", "F1"),
        ("pr_auc", "PR-AUC"),
        ("balanced_accuracy", "vyvážená úspešnosť"),
    ]
    grouped = selected.groupby("split")[[name for name, _ in metrics]].mean()
    x = np.arange(len(metrics))
    figure, axis = plt.subplots(figsize=(9.2, 4.5))
    axis.bar(
        x - 0.2,
        grouped.loc["temporal", [name for name, _ in metrics]],
        0.4,
        label="časové rozdelenie",
        color="#377eb8",
    )
    axis.bar(
        x + 0.2,
        grouped.loc["leakage_resistant", [name for name, _ in metrics]],
        0.4,
        label="bez prekryvu adries",
        color="#ff7f00",
    )
    axis.set_xticks(x, [label for _, label in metrics], rotation=15, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Hodnota metriky")
    axis.set_xlabel("Metrika náhodného lesa, variant B")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save(figure, "transparent_rf_metric_profile")


def _model_comparison(path: Path) -> None:
    runs = pd.read_csv(path)
    grouped = (
        runs.groupby(["variant", "split", "model"], as_index=False)["illicit_f1"]
        .agg(["mean", "std"])
        .reset_index()
    )
    temporal = grouped[grouped["split"] == "temporal"].copy()
    clean = grouped[grouped["split"] == "leakage_resistant"].copy()
    joined = temporal.merge(
        clean,
        on=["variant", "model"],
        suffixes=("_temporal", "_clean"),
    )
    labels = [f"{model}\n{variant}" for model, variant in zip(joined["model"], joined["variant"])]
    x = np.arange(len(joined))
    figure, axis = plt.subplots(figsize=(11.5, 5.2))
    axis.bar(x - 0.2, joined["mean_temporal"], 0.4, label="časové rozdelenie", color="#377eb8")
    axis.bar(x + 0.2, joined["mean_clean"], 0.4, label="rozdelenie bez prekryvu", color="#ff7f00")
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_ylabel("F1 pre ilicitnú triedu")
    axis.set_ylim(0, 1.05)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save(figure, "model_f1_leakage_comparison")

    best = runs.sort_values("illicit_f1", ascending=False).head(1).iloc[0]
    matrix = np.array([[best["tn"], best["fp"]], [best["fn"], best["tp"]]])
    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    axis.set_xticks([0, 1], ["licitná", "ilicitná"])
    axis.set_yticks([0, 1], ["licitná", "ilicitná"])
    axis.set_xlabel("Predikovaná trieda")
    axis.set_ylabel("Skutočná trieda")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    _save(figure, "best_model_confusion_matrix")


def create_report() -> dict[str, bool]:
    ensure_runtime_directories()
    generated = {
        "class_distribution": False,
        "graph_overview": False,
        "split_retention": False,
        "mapping": False,
        "models": False,
    }
    if (PROCESSED_DIR / "selected_features.csv.gz").exists():
        _class_distribution()
        generated["class_distribution"] = True
    if (PROCESSED_DIR / "model_nodes.csv.gz").exists():
        _split_retention()
        if (PROCESSED_DIR / "model_original_edges.csv.gz").exists():
            _graph_overview()
            generated["graph_overview"] = True
        generated["split_retention"] = True
    mapping_path = RESULTS_DIR / "mapping_holdout_summary.csv"
    if mapping_path.exists():
        _mapping_accuracy(mapping_path)
        generated["mapping"] = True
    models_path = RESULTS_DIR / "model_runs.csv"
    if models_path.exists():
        _model_comparison(models_path)
        _metric_profile(models_path)
        generated["models"] = True
    return generated
