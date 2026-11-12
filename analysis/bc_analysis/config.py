from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
DATA_ROOT = ANALYSIS_ROOT / "data"
PROCESSED_DIR = DATA_ROOT / "processed"
CACHE_DIR = DATA_ROOT / "cache"
RESULTS_DIR = ANALYSIS_ROOT / "results"
THESIS_FIGURES_DIR = PROJECT_ROOT / "FEIstyle_v2-0-1" / "img" / "generated"

SELECTED_TIMESTEPS = (20, 29, 30, 32, 35, 42)
TIMESTEP_SPLIT = {
    20: "train",
    29: "train",
    30: "validation",
    32: "validation",
    35: "test",
    42: "test",
}

# Inclusive block ranges recovered by Safar et al. and published with the
# BitcoinTxSubgraphBuilder preprocessing code. A margin can be added by CLI.
BLOCK_RANGES = {
    20: (429_504, 429_515),
    29: (447_648, 447_655),
    30: (449_664, 449_671),
    32: (453_696, 453_706),
    35: (459_744, 459_753),
    42: (473_856, 473_868),
}

DEFAULT_BLOCKBOOK_URL = "https://btc1.trezor.io"
DEFAULT_USER_AGENT = "bc-thesis-analysis/0.1 (academic research; FEI STU)"
HOLDOUT_FRACTION = 0.20
DEFAULT_SEED = 42
MODEL_SEEDS = (42, 43, 44, 45, 46)

# Feature numbers include the explicit time-step column as feature 1.
KNOWN_LOCAL_FEATURES = {
    2: "value_satoshi",
    3: "fee_satoshi",
    5: "input_count",
    6: "output_count",
    7: "unique_input_addresses",
    9: "input_value_min",
    10: "input_value_max",
    11: "input_value_std",
    12: "input_value_mean",
    13: "input_value_pearson",
    14: "input_value_spearman",
    15: "unique_output_addresses",
    17: "output_value_min",
    18: "output_value_max",
    19: "output_value_std",
    20: "output_value_mean",
    21: "output_value_pearson",
    22: "output_value_spearman",
}

MATCHING_STAGES = {
    "counts": ("input_count", "output_count"),
    "value": (
        "input_count",
        "output_count",
        "value_satoshi",
        "fee_satoshi",
        "unique_input_addresses",
        "unique_output_addresses",
    ),
    "full": tuple(KNOWN_LOCAL_FEATURES.values()),
}


def ensure_runtime_directories() -> None:
    for path in (PROCESSED_DIR, CACHE_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
