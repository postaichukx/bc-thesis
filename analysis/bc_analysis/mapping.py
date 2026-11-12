from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_SEED,
    HOLDOUT_FRACTION,
    KNOWN_LOCAL_FEATURES,
    MATCHING_STAGES,
    PROCESSED_DIR,
    RESULTS_DIR,
    SELECTED_TIMESTEPS,
    ensure_runtime_directories,
)
from .data import update_manifest


INTEGER_FEATURES = {
    "value_satoshi",
    "fee_satoshi",
    "input_count",
    "output_count",
    "unique_input_addresses",
    "unique_output_addresses",
    "input_value_min",
    "input_value_max",
    "output_value_min",
    "output_value_max",
}


@dataclass(frozen=True)
class AffineCalibration:
    slope: float
    intercept: float
    residual_scale: float

    def predict(self, value: float) -> float:
        return self.slope * value + self.intercept


def fit_calibrations(
    elliptic: pd.DataFrame, raw: pd.DataFrame
) -> dict[str, AffineCalibration]:
    joined = elliptic.merge(raw, on="txid", how="inner", suffixes=("_elliptic", "_raw"))
    if len(joined) < 2:
        raise ValueError("At least two mapped calibration transactions are required")

    calibrations: dict[str, AffineCalibration] = {}
    for feature_number, raw_name in KNOWN_LOCAL_FEATURES.items():
        x = joined[f"feature_{feature_number}"].to_numpy(dtype=float)
        y = joined[raw_name].to_numpy(dtype=float)
        design = np.column_stack((x, np.ones(len(x))))
        slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        residuals = np.abs(y - (slope * x + intercept))
        residual_scale = float(np.quantile(residuals, 0.95)) if len(residuals) else 0.0
        calibrations[raw_name] = AffineCalibration(
            slope=float(slope),
            intercept=float(intercept),
            residual_scale=max(residual_scale, 1e-12),
        )
    return calibrations


def deanonymize_rows(
    rows: pd.DataFrame, calibrations: dict[str, AffineCalibration]
) -> pd.DataFrame:
    result = rows[["txId", "timestep"]].copy()
    for feature_number, raw_name in KNOWN_LOCAL_FEATURES.items():
        values = rows[f"feature_{feature_number}"].astype(float)
        calibration = calibrations[raw_name]
        predicted = values * calibration.slope + calibration.intercept
        if raw_name in INTEGER_FEATURES:
            predicted = predicted.round()
        result[raw_name] = predicted
    return result


def _holdout_ids(mapped: pd.DataFrame, seed: int) -> set[int]:
    rng = np.random.default_rng(seed)
    holdout: set[int] = set()
    for timestep in SELECTED_TIMESTEPS:
        values = mapped.loc[mapped["timestep"] == timestep, "txId"].to_numpy(
            dtype=np.int64, copy=True
        )
        rng.shuffle(values)
        size = max(1, int(round(len(values) * HOLDOUT_FRACTION)))
        holdout.update(int(value) for value in values[:size])
    return holdout


def _candidate_score(
    predicted: pd.Series,
    candidates: pd.DataFrame,
    names: tuple[str, ...],
    calibrations: dict[str, AffineCalibration],
) -> np.ndarray:
    score = np.zeros(len(candidates), dtype=float)
    for name in names:
        target = float(predicted[name])
        values = candidates[name].to_numpy(dtype=float)
        if name in {"input_count", "output_count", "unique_input_addresses", "unique_output_addresses"}:
            scale = 1.0
        else:
            scale = max(calibrations[name].residual_scale, abs(target) * 1e-9, 1e-9)
        score += np.abs(values - target) / scale
    return score / max(1, len(names))


def _rank_candidates(
    predicted: pd.Series,
    pool: pd.DataFrame,
    names: tuple[str, ...],
    calibrations: dict[str, AffineCalibration],
    top_k: int = 5,
) -> tuple[pd.DataFrame, int]:
    same_counts = pool[
        (pool["input_count"] == int(round(float(predicted["input_count"]))))
        & (pool["output_count"] == int(round(float(predicted["output_count"]))))
    ].copy()
    if same_counts.empty:
        return same_counts, 0
    same_counts["score"] = _candidate_score(predicted, same_counts, names, calibrations)
    ranked = same_counts.sort_values(["score", "txid"], kind="stable")
    return ranked.head(top_k), len(same_counts)


def evaluate_mapping(seed: int = DEFAULT_SEED, top_k: int = 5) -> dict[str, object]:
    ensure_runtime_directories()
    required = {
        "features": PROCESSED_DIR / "selected_features.csv.gz",
        "mapping": PROCESSED_DIR / "selected_mapping.csv.gz",
        "candidates": PROCESSED_DIR / "block_candidates.csv.gz",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prepared data: " + ", ".join(missing))

    features = pd.read_csv(required["features"])
    mapping = pd.read_csv(required["mapping"])
    candidates = pd.read_csv(required["candidates"])
    mapping["txid"] = mapping["txid"].astype(str).str.lower()
    candidates["txid"] = candidates["txid"].astype(str).str.lower()
    empty_pool = candidates.iloc[0:0].copy()
    candidate_groups = {
        (int(timestep), int(input_count), int(output_count)): group.copy()
        for (timestep, input_count, output_count), group in candidates.groupby(
            ["timestep", "input_count", "output_count"], sort=False
        )
    }

    mapped = features.merge(mapping, on="txId", how="inner")
    holdout_ids = _holdout_ids(mapped, seed)
    calibration_rows = mapped[~mapped["txId"].isin(holdout_ids)]
    holdout_rows = mapped[mapped["txId"].isin(holdout_ids)]

    calibrations = fit_calibrations(calibration_rows, candidates)
    predicted_holdout = deanonymize_rows(holdout_rows, calibrations).merge(
        holdout_rows[["txId", "txid"]], on="txId", how="left"
    )

    detail_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for stage, names in MATCHING_STAGES.items():
        top1 = topk = unmatched = ambiguous = 0
        for _, predicted in predicted_holdout.iterrows():
            pool = candidate_groups.get(
                (
                    int(predicted["timestep"]),
                    int(round(float(predicted["input_count"]))),
                    int(round(float(predicted["output_count"]))),
                ),
                empty_pool,
            )
            ranked, candidate_count = _rank_candidates(
                predicted, pool, names, calibrations, top_k=top_k
            )
            truth = str(predicted["txid"])
            ranked_txids = ranked["txid"].astype(str).tolist()
            is_unmatched = candidate_count == 0
            is_top1 = bool(ranked_txids and ranked_txids[0] == truth)
            is_topk = truth in ranked_txids
            is_ambiguous = bool(
                len(ranked) > 1
                and np.isclose(float(ranked.iloc[0]["score"]), float(ranked.iloc[1]["score"]))
            )
            top1 += int(is_top1)
            topk += int(is_topk)
            unmatched += int(is_unmatched)
            ambiguous += int(is_ambiguous)
            detail_rows.append(
                {
                    "stage": stage,
                    "txId": int(predicted["txId"]),
                    "timestep": int(predicted["timestep"]),
                    "truth_txid": truth,
                    "top1_txid": ranked_txids[0] if ranked_txids else "",
                    "top1_score": float(ranked.iloc[0]["score"]) if len(ranked) else np.nan,
                    "topk_txids": "|".join(ranked_txids),
                    "candidate_count": candidate_count,
                    "correct_top1": is_top1,
                    "correct_topk": is_topk,
                    "ambiguous": is_ambiguous,
                    "unmatched": is_unmatched,
                }
            )
        total = len(predicted_holdout)
        summaries.append(
            {
                "stage": stage,
                "total": total,
                "top1": top1,
                "top1_rate": top1 / total if total else 0.0,
                f"top{top_k}": topk,
                f"top{top_k}_rate": topk / total if total else 0.0,
                "ambiguous": ambiguous,
                "unmatched": unmatched,
            }
        )

    unmapped = features[~features["txId"].isin(mapping["txId"])]
    predicted_unmapped = deanonymize_rows(unmapped, calibrations)
    unmapped_rows: list[dict[str, object]] = []
    for _, predicted in predicted_unmapped.iterrows():
        pool = candidate_groups.get(
            (
                int(predicted["timestep"]),
                int(round(float(predicted["input_count"]))),
                int(round(float(predicted["output_count"]))),
            ),
            empty_pool,
        )
        ranked, candidate_count = _rank_candidates(
            predicted, pool, MATCHING_STAGES["full"], calibrations, top_k=top_k
        )
        unmapped_rows.append(
            {
                "txId": int(predicted["txId"]),
                "timestep": int(predicted["timestep"]),
                "candidate_txids": "|".join(ranked["txid"].astype(str).tolist()),
                "candidate_scores": "|".join(f"{score:.8g}" for score in ranked["score"]),
                "candidate_count": candidate_count,
                "status": "candidate_only",
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(RESULTS_DIR / "mapping_holdout_details.csv", index=False)
    pd.DataFrame(summaries).to_csv(RESULTS_DIR / "mapping_holdout_summary.csv", index=False)
    pd.DataFrame(unmapped_rows).to_csv(RESULTS_DIR / "unmapped_candidates.csv", index=False)
    calibration_json = {
        name: {
            "slope": value.slope,
            "intercept": value.intercept,
            "residual_scale": value.residual_scale,
        }
        for name, value in calibrations.items()
    }
    (RESULTS_DIR / "mapping_calibration.json").write_text(
        json.dumps(calibration_json, indent=2), encoding="utf-8"
    )
    result = {
        "holdout": len(predicted_holdout),
        "unmapped": len(predicted_unmapped),
        "summary": summaries,
    }
    update_manifest(
        {
            "mapping_evaluation": {
                "seed": seed,
                "holdout_fraction": HOLDOUT_FRACTION,
                "top_k": top_k,
                **result,
            }
        }
    )
    return result
