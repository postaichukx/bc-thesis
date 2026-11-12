from __future__ import annotations

import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

from .audit import LABELS, TARGETS
from .data import CLASSES_FILE, FEATURES_FILE, sha256_file


SEMANTICS_URL = "https://github.com/nesfit/DeanonymizedEllipticBitcoinDataset/blob/a5dbbcc655783003d942d0d54ad8404f0b7b3b93/README.md"
DATASET_URL = "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set"
STATS = (
    ("min", "minimum"), ("max", "maximum"), ("std", "smerodajná odchýlka"),
    ("mean", "priemer"), ("pearson", "Pearsonov koeficient voči indexom"),
    ("spearman", "Spearmanov koeficient voči indexom"),
)


def column_dictionary() -> list[dict]:
    """Keep feature indices traceable; recovered meanings are not official names."""
    columns = []

    def add(name, description, group, feature=None, time_scope="", status="obnovený význam"):
        is_value = feature is not None and feature != 1
        columns.append({
            "column_number": len(columns) + 1,
            "feature_number": feature,
            "column_name": name,
            "group_sk": group,
            "description_sk": description,
            "meaning_status_sk": status,
            "stored_scale_sk": "Pôvodná publikovaná číselná škála; nejde o surové jednotky." if is_value else "Identifikátor / kategória / celé číslo.",
            "time_scope_sk": time_scope,
            "source_url": SEMANTICS_URL if is_value else DATASET_URL,
        })

    for name, description in (
        ("elliptic_tx_id", "Interný identifikátor Elliptic; nejde o verejný bitcoinový txid."),
        ("class_code", "Pôvodný kód triedy: 1 = illicit, 2 = licit, unknown = bez označenia."),
        ("class_label", "Označenie prevzaté z Elliptic: licit, illicit alebo unknown."),
        ("target_label", "Cieľ: 0 = licit, 1 = illicit; -1 označuje chýbajúci cieľ, nie tretiu triedu."),
        ("is_labeled", "Dostupnosť cieľa, nie trénovacia maska: true pre licit/illicit, false pre unknown."),
    ):
        add(name, description, "Identifikácia a označenia", status="pôvodný údaj / odvodené označenie")

    add("time_step", "Časový krok Elliptic v rozsahu 1–49; nejde o výšku bloku.",
        "Časový krok", 1, "Poradie vzorkovaných časových okien.", "publikovaný význam")

    def feature(index, slug, description, group="Transakcia", time_scope="Samotná transakcia."):
        unknown = slug == "unknown_meaning"
        add(f"feature_{index:03d}__{slug}", description, group, index, time_scope,
            "neurčený význam" if unknown else "obnovený význam")

    for index, slug, description in (
        (2, "transaction_volume", "Objem transakcie."),
        (3, "fee", "Transakčný poplatok."),
        (4, "unknown_meaning", "Význam príznaku zostáva neurčený."),
        (5, "input_count", "Počet vstupov."),
        (6, "output_count", "Počet výstupov."),
        (7, "unique_input_address_count", "Počet rôznych vstupných adries."),
        (8, "unknown_meaning", "Význam príznaku zostáva neurčený."),
        (15, "unique_output_address_count", "Počet rôznych výstupných adries."),
        (16, "unknown_meaning", "Význam príznaku zostáva neurčený."),
    ):
        feature(index, slug, description)
    for start, side, label in ((9, "input", "Hodnoty vstupov"), (17, "output", "Hodnoty výstupov")):
        for offset, (stat, description) in enumerate(STATS):
            feature(start + offset, f"{side}_value_{stat}", f"{label}: {description}.")

    address_properties = (
        ("total_received", "Celková prijatá hodnota"),
        ("total_sent", "Celková odoslaná hodnota"),
        ("balance", "Dostupný zostatok"),
        ("incoming_tx_count", "Počet prichádzajúcich transakcií"),
        ("outgoing_tx_count", "Počet odchádzajúcich transakcií"),
        ("lifetime", "Životnosť adresy"),
    )
    for start, side, label in ((23, "input", "Vstupné adresy"), (59, "output", "Výstupné adresy")):
        for prop, (slug, description) in enumerate(address_properties):
            for offset, (stat, stat_description) in enumerate(STATS):
                feature(start + prop * 6 + offset, f"{side}_addresses_{slug}_{stat}",
                        f"{description}, {label.lower()}: {stat_description}.", label,
                        "História adries po blok 575 059; nie stav v čase transakcie.")

    neighbor_properties = (
        ("volume", "Objem"), ("block_height", "Výška bloku"), ("fee", "Poplatok"),
        ("unknown_meaning", "Neurčený význam"),
        ("input_count", "Počet vstupov"), ("output_count", "Počet výstupov"),
    )
    for start, side, label in ((95, "predecessor", "Predchodcovia"), (131, "successor", "Nasledovníci")):
        for prop, (slug, description) in enumerate(neighbor_properties):
            for offset, (stat, stat_description) in enumerate(STATS):
                unknown = slug == "unknown_meaning"
                feature(start + prop * 6 + offset,
                        slug if unknown else f"{side}_{slug}_{stat}",
                        "Význam príznaku zostáva neurčený." if unknown else f"{description}, {label.lower()}: {stat_description}.",
                        label, "Okolie môže presahovať uložený graf; nasledovníci vznikajú neskôr.")

    metadata, features = columns[:5], columns[5:]
    features.sort(key=lambda item: item["feature_number"])
    for position, column in enumerate(metadata + features, start=1):
        column["column_number"] = position
    return metadata + features


def export_table(elliptic_dir: Path, audit_dir: Path, output_dir: Path) -> dict:
    """Export every audited row, preserving the original numeric field strings."""
    report_path = audit_dir / "audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report["valid"]:
        raise ValueError("Export requires a successful audit")
    if output_dir.resolve() in {elliptic_dir.resolve(), audit_dir.resolve()}:
        raise ValueError("Export must use its own directory")
    for recorded in report["inputs"].values():
        path = elliptic_dir / Path(recorded["path"]).name
        if sha256_file(path) != recorded["sha256"]:
            raise ValueError(f"Input changed since audit: {path.name}")

    with (elliptic_dir / CLASSES_FILE).open(encoding="utf-8", newline="") as handle:
        classes = {row["txId"]: row["class"] for row in csv.DictReader(handle)}
    dictionary = column_dictionary()
    headers = [item["column_name"] for item in dictionary]
    counts = Counter()
    sample_rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "elliptic_all_transactions.csv"
    with tempfile.TemporaryDirectory(prefix=".table-export-", dir=output_dir) as temporary:
        pending = Path(temporary) / destination.name
        with pending.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(headers)
            with (elliptic_dir / FEATURES_FILE).open(encoding="utf-8", newline="") as source:
                for row in csv.reader(source):
                    if len(row) != 167:
                        raise ValueError("Feature row must contain txId and 166 features")
                    code = classes[row[0]]
                    label = LABELS[code]
                    exported = [row[0], code, label, str(TARGETS[code]),
                                "false" if code == "unknown" else "true", *row[1:]]
                    writer.writerow(exported)
                    counts[label] += 1
                    if counts[label] <= 10:
                        sample_rows.append(exported)
        if sum(counts.values()) != report["counts"]["nodes"] or any(
            counts[label] != report["counts"][label] for label in LABELS.values()
        ):
            raise ValueError("Export counts differ from audited counts")
        pending.replace(destination)

    dictionary_path = output_dir / "elliptic_column_dictionary.csv"
    with dictionary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(dictionary)
    manifest = {
        "rows": sum(counts.values()), "columns": len(headers), "class_counts": dict(counts),
        "feature_count": 166, "unresolved_features": [item["feature_number"] for item in dictionary if item["meaning_status_sk"] == "neurčený význam"],
        "csv_path": str(destination.resolve()), "csv_bytes": destination.stat().st_size,
        "csv_sha256": sha256_file(destination), "audit_path": str(report_path.resolve()),
        "inputs": report["inputs"], "dataset_url": DATASET_URL, "semantics_url": SEMANTICS_URL,
        "semantics_paper": "https://doi.org/10.1016/j.fsidi.2026.302124",
        "values": "Original numeric field strings, row order and internal IDs preserved. No inverse normalization or public txid mapping.",
        "sample_selection": "First ten rows of each class in original file order; illustrative, not representative or a data split.",
        "edge_file": report["inputs"]["edges"]["path"],
        "dictionary": dictionary, "sample_rows": sample_rows,
    }
    (output_dir / "table_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {key: manifest[key] for key in ("rows", "columns", "class_counts", "csv_path", "csv_bytes", "csv_sha256")}
