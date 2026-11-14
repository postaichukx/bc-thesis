import csv
import json

import pytest

from bc_analysis.kap_03_elliptic.dataset_audit import audit_elliptic
from bc_analysis.kap_04_blockchain.transaction_collection import CLASSES_FILE, EDGES_FILE, FEATURES_FILE
from bc_analysis.kap_03_elliptic.feature_table_export import column_dictionary, export_table


def test_dictionary_preserves_all_positions_and_uncertainty():
    columns = column_dictionary()
    assert len(columns) == 171
    assert len({item["column_name"] for item in columns}) == 171
    assert [item["feature_number"] for item in columns[5:]] == list(range(1, 167))
    assert [item["column_number"] for item in columns] == list(range(1, 172))
    assert [item["feature_number"] for item in columns if item["meaning_status_sk"] == "neurčený význam"] == [4, 8, 16, *range(113, 119), *range(149, 155)]
    by_index = {item["feature_number"]: item for item in columns[5:]}
    assert by_index[23]["column_name"].endswith("input_addresses_total_received_min")
    assert by_index[94]["column_name"].endswith("output_addresses_lifetime_spearman")
    assert by_index[95]["column_name"].endswith("predecessor_volume_min")
    assert by_index[166]["column_name"].endswith("successor_output_count_spearman")


def test_export_preserves_every_original_value_and_unknown(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [[str(101 + i), "1", *(["0.1234567890123456789"] * 165)] for i in range(3)]
    with (raw / FEATURES_FILE).open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    (raw / CLASSES_FILE).write_text("txId,class\n103,unknown\n101,1\n102,2\n")
    (raw / EDGES_FILE).write_text("txId1,txId2\n101,102\n102,103\n")
    audit = tmp_path / "audit"
    audit_elliptic(raw, audit)
    output = tmp_path / "tables"
    result = export_table(raw, audit, output)
    with (output / "elliptic_all_transactions.csv").open() as handle:
        exported = list(csv.reader(handle))
    assert result["rows"] == 3
    assert result["columns"] == 171
    assert result["class_counts"] == {"illicit": 1, "licit": 1, "unknown": 1}
    for original, row in zip(rows, exported[1:], strict=True):
        assert [row[0], *row[5:]] == original
    assert [row[3:5] for row in exported[1:]] == [["1", "true"], ["0", "true"], ["-1", "false"]]
    manifest = json.loads((output / "table_manifest.json").read_text())
    assert manifest["sample_rows"] == exported[1:]
    with pytest.raises(ValueError, match="own directory"):
        export_table(raw, audit, raw)
    (raw / CLASSES_FILE).write_text("txId,class\n103,2\n101,1\n102,2\n")
    with pytest.raises(ValueError, match="changed since audit"):
        export_table(raw, audit, output)


def test_export_rejects_failed_audit(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "audit.json").write_text('{"valid": false}')
    with pytest.raises(ValueError, match="successful audit"):
        export_table(tmp_path / "raw", audit, tmp_path / "tables")
