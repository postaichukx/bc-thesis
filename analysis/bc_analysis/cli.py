from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_elliptic
from .config import DEFAULT_BLOCKBOOK_URL, DEFAULT_SEED
from .data import collect, prepare
from .datasets import build_datasets
from .mapping import evaluate_mapping
from .models import MODELS, SPLITS, VARIANTS, train_all
from .reporting import create_report
from .table_export import export_table


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bc-analysis",
        description="Reproducible Elliptic/Bitcoin analysis for the bachelor thesis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit all original Elliptic records without mapping")
    audit_parser.add_argument("--elliptic-dir", type=Path, required=True)
    audit_parser.add_argument("--output-dir", type=Path, required=True)

    table_parser = subparsers.add_parser("export-table", help="Export all audited transactions with a column dictionary")
    table_parser.add_argument("--elliptic-dir", type=Path, required=True)
    table_parser.add_argument("--audit-dir", type=Path, required=True)
    table_parser.add_argument("--output-dir", type=Path, required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Select and validate Elliptic records")
    prepare_parser.add_argument("--elliptic-dir", type=Path, required=True)
    prepare_parser.add_argument("--mapping", type=Path, required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect public Bitcoin transactions")
    collect_parser.add_argument("--mode", choices=("selected", "blocks"), default="blocks")
    collect_parser.add_argument("--base-url", default=DEFAULT_BLOCKBOOK_URL)
    collect_parser.add_argument("--workers", type=int, default=4)
    collect_parser.add_argument("--margin", type=int, default=0)
    collect_parser.add_argument("--limit", type=int)

    mapping_parser = subparsers.add_parser(
        "mapping-eval", help="Evaluate mapping with a hidden 20 percent holdout"
    )
    mapping_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    mapping_parser.add_argument("--top-k", type=int, default=5)

    subparsers.add_parser("build-datasets", help="Build model variants and split metadata")

    train_parser = subparsers.add_parser("train", help="Train and evaluate all selected models")
    train_parser.add_argument("--variants", type=_csv_tuple, default=tuple(VARIANTS))
    train_parser.add_argument("--splits", type=_csv_tuple, default=tuple(SPLITS))
    train_parser.add_argument("--models", type=_csv_tuple, default=MODELS)

    subparsers.add_parser("report", help="Create tables and thesis figures")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        result = audit_elliptic(args.elliptic_dir, args.output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result["valid"]:
            raise SystemExit(1)
        return
    elif args.command == "export-table":
        result = export_table(args.elliptic_dir, args.audit_dir, args.output_dir)
    elif args.command == "prepare":
        result = prepare(args.elliptic_dir, args.mapping)
    elif args.command == "collect":
        result = collect(args.mode, args.base_url, args.workers, args.margin, args.limit)
    elif args.command == "mapping-eval":
        result = evaluate_mapping(args.seed, args.top_k)
    elif args.command == "build-datasets":
        result = build_datasets()
    elif args.command == "train":
        result = train_all(args.variants, args.splits, args.models)
    else:
        result = create_report()
    print(json.dumps(result, indent=2, ensure_ascii=False))
