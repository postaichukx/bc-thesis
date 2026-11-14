import json
from pathlib import Path

import pandas as pd
import pytest

from bc_analysis.kap_03_elliptic.dataset_audit import audit_elliptic
from bc_analysis.kap_04_blockchain.transaction_collection import CLASSES_FILE, EDGES_FILE, FEATURES_FILE, sha256_file


def raw_files(directory: Path):
    directory.mkdir()
    pd.DataFrame([[101, 1, *([0.5] * 165)], [102, 2, *([1.5] * 165)], [103, 2, *([2.5] * 165)]]).to_csv(directory / FEATURES_FILE, header=False, index=False)
    pd.DataFrame({"txId": [103, 101, 102], "class": ["unknown", "1", "2"]}).to_csv(directory / CLASSES_FILE, index=False)
    pd.DataFrame({"txId1": [101, 102], "txId2": [102, 103]}).to_csv(directory / EDGES_FILE, index=False)


def test_full_audit_retains_unknown_features_and_joins_by_id(tmp_path):
    raw = tmp_path / "raw"
    raw_files(raw)
    hashes = {p.name: sha256_file(p) for p in raw.iterdir()}
    output = tmp_path / "audit"
    report = audit_elliptic(raw, output)
    nodes = pd.read_csv(output / "elliptic_nodes.csv.gz")
    assert report["valid"]
    assert report["counts"] == {"nodes": 3, "feature_columns": 166, "classes": 3, "edges": 2, "illicit": 1, "licit": 1, "unknown": 1, "timesteps": 2}
    assert nodes["label"].tolist() == [1, 0, -1]
    assert nodes["class"].tolist() == ["illicit", "licit", "unknown"]
    assert nodes["is_labeled"].tolist() == [True, True, False]
    assert nodes.iloc[-1]["feature_166"] == 2.5
    assert report["checks"]["cross_timestep_edges"] == 1
    assert hashes == {p.name: sha256_file(p) for p in raw.iterdir()}
    assert json.loads((output / "audit.json").read_text())["valid"]


@pytest.mark.parametrize("problem", ["missing_class", "extra_class", "missing_edge", "duplicate_node", "duplicate_edge", "self_loop", "missing_value", "invalid_timestep"])
def test_audit_reports_bad_data_without_publishing_nodes(tmp_path, problem):
    raw = tmp_path / "raw"
    raw_files(raw)
    if problem in {"missing_class", "extra_class"}:
        table = pd.read_csv(raw / CLASSES_FILE)
        table = table.iloc[1:] if problem == "missing_class" else pd.concat([table, pd.DataFrame({"txId": [999], "class": ["unknown"]})])
        table.to_csv(raw / CLASSES_FILE, index=False)
    elif problem in {"missing_edge", "duplicate_edge", "self_loop"}:
        table = pd.read_csv(raw / EDGES_FILE)
        if problem == "missing_edge":
            table.loc[0, "txId2"] = 999
        elif problem == "self_loop":
            table.loc[0, "txId2"] = 101
        else:
            table = pd.concat([table, table.iloc[[0]]])
        table.to_csv(raw / EDGES_FILE, index=False)
    else:
        table = pd.read_csv(raw / FEATURES_FILE, header=None)
        if problem == "duplicate_node":
            table = pd.concat([table, table.iloc[[0]]])
        elif problem == "missing_value":
            table.loc[0, 5] = float("nan")
        else:
            table[1] = table[1].astype(float)
            table.loc[0, 1] = 1.5
        table.to_csv(raw / FEATURES_FILE, header=False, index=False)
    output = tmp_path / "audit"
    report = audit_elliptic(raw, output)
    assert not report["valid"]
    assert report["validation_errors"]
    assert not (output / "elliptic_nodes.csv.gz").exists()


@pytest.mark.parametrize("problem", ["duplicate_class", "invalid_class", "wrong_header", "wrong_width"])
def test_audit_rejects_ambiguous_or_wrong_schema(tmp_path, problem):
    raw = tmp_path / "raw"
    raw_files(raw)
    if problem == "wrong_width":
        table = pd.read_csv(raw / FEATURES_FILE, header=None).iloc[:, :-1]
        table.to_csv(raw / FEATURES_FILE, header=False, index=False)
    else:
        table = pd.read_csv(raw / CLASSES_FILE)
        if problem == "duplicate_class":
            table = pd.concat([table, table.iloc[[0]]])
        elif problem == "invalid_class":
            table.loc[0, "class"] = "3"
        else:
            table = table.rename(columns={"class": "label"})
        table.to_csv(raw / CLASSES_FILE, index=False)
    with pytest.raises(ValueError):
        audit_elliptic(raw, tmp_path / "audit")


def test_audit_missing_file_and_raw_output_collision(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit_elliptic(tmp_path / "missing", tmp_path / "audit")
    raw = tmp_path / "raw"
    raw_files(raw)
    with pytest.raises(ValueError, match="different directories"):
        audit_elliptic(raw, raw)


def test_repeat_audit_preserves_values_and_raw_inputs(tmp_path):
    raw = tmp_path / "raw"
    raw_files(raw)
    output = tmp_path / "audit"
    first = audit_elliptic(raw, output)
    first_nodes = pd.read_csv(output / "elliptic_nodes.csv.gz")
    second = audit_elliptic(raw, output)
    pd.testing.assert_frame_equal(first_nodes, pd.read_csv(output / "elliptic_nodes.csv.gz"))
    assert first["inputs"] == second["inputs"]
    assert first["counts"] == second["counts"]
    assert not list(output.glob(".elliptic_nodes-*"))


def test_cli_exits_unsuccessfully_on_invalid_data(tmp_path, capsys):
    from bc_analysis.cli import main

    raw = tmp_path / "raw"
    raw_files(raw)
    table = pd.read_csv(raw / EDGES_FILE)
    table.loc[0, "txId2"] = 999
    table.to_csv(raw / EDGES_FILE, index=False)
    with pytest.raises(SystemExit) as error:
        main(["audit", "--elliptic-dir", str(raw), "--output-dir", str(tmp_path / "audit")])
    assert error.value.code == 1
    assert not json.loads(capsys.readouterr().out)["valid"]
