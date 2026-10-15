"""Shared, leakage-free data loading and time-based splitting (Day 2), reused from Day 3 on."""

from pathlib import Path

import pandas as pd

from logagent.features import EARLY_FEATURES  # noqa: F401  (re-exported)

ROOT = Path(__file__).resolve().parents[2]
BLOCKS = ROOT / "data" / "processed" / "blocks.parquet"

EVENT_COLS = [f"E{i}" for i in range(1, 30)]
FEATURES = EVENT_COLS + ["n_events", "n_unique_events", "latency"]  # full block lifetime
TARGET = "label"


def load_blocks() -> pd.DataFrame:
    return pd.read_parquet(BLOCKS).sort_values("first_ts").reset_index(drop=True)


def time_split(df: pd.DataFrame, train=0.70, val=0.15):
    """Chronological train/val/test: the model is always evaluated on blocks it 'hasn't seen yet'."""
    df = df.sort_values("first_ts").reset_index(drop=True)
    n = len(df)
    i_val, i_test = int(n * train), int(n * (train + val))
    return df.iloc[:i_val], df.iloc[i_val:i_test], df.iloc[i_test:]


def xy(df: pd.DataFrame, features=FEATURES):
    return df[features], df[TARGET]
