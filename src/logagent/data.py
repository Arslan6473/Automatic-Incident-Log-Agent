"""Build the per-block modelling table from the Loghub HDFS_v1 archive.

One row per HDFS block: event-count features (E1..E29), a few derived features,
the block's first-seen timestamp (for time-based splits) and the anomaly label.

Also builds *early-warning* features: counts over only the first EARLY_K events of each
block (early_E1..early_E29) plus the seconds elapsed until that point (early_elapsed).
Full-lifetime counts make HDFS almost trivially separable; predicting from the first few
events is the realistic task for the incident agent ("how early can we warn?").

Run:  uv run python -m logagent.data
"""

import re
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from logagent.features import EARLY_K

ROOT = Path(__file__).resolve().parents[2]
RAW_ZIP = ROOT / "data" / "raw" / "HDFS_v1.zip"
PRE = ROOT / "data" / "raw" / "preprocessed"
OUT = ROOT / "data" / "processed" / "blocks.parquet"

BLOCK_RE = re.compile(rb"blk_-?\d+")


def first_seen_timestamps(zip_path: Path) -> pd.Series:
    """Stream HDFS.log out of the zip (no 1.5 GB extract) and record each block's first timestamp."""
    first: dict[bytes, bytes] = {}
    with zipfile.ZipFile(zip_path) as zf, zf.open("HDFS.log") as fh:
        for line in fh:
            for blk in BLOCK_RE.findall(line):
                if blk not in first:
                    first[blk] = line[:13]  # b"081109 203518"
    ts = {k.decode(): datetime.strptime(v.decode(), "%y%m%d %H%M%S") for k, v in first.items()}
    return pd.Series(ts, name="first_ts")


def early_features(traces: pd.DataFrame, k: int = EARLY_K) -> pd.DataFrame:
    """Event counts over the first k events of each block, plus seconds elapsed until the k-th event."""
    seqs = traces["Features"].str.strip("[]").str.split(",")
    gaps = traces["TimeInterval"].str.strip("[]").str.split(",")
    counts = pd.DataFrame([pd.Series(s[:k]).value_counts() for s in seqs]).fillna(0).astype("int16")
    counts = counts.reindex(columns=[f"E{i}" for i in range(1, 30)], fill_value=0).add_prefix("early_")
    counts["early_elapsed"] = [sum(float(x) for x in g[: k - 1] if x.strip()) for g in gaps]
    counts.insert(0, "BlockId", traces["BlockId"].values)
    return counts


def build() -> pd.DataFrame:
    counts = pd.read_csv(PRE / "Event_occurrence_matrix.csv").drop(columns=["Type"])
    traces = pd.read_csv(PRE / "Event_traces.csv", usecols=["BlockId", "Features", "TimeInterval", "Latency"])

    df = counts.merge(traces[["BlockId", "Latency"]], on="BlockId", how="left")
    df = df.merge(early_features(traces), on="BlockId", how="left")
    df["label"] = (df.pop("Label") == "Fail").astype("int8")

    event_cols = [f"E{i}" for i in range(1, 30)]
    df["n_events"] = df[event_cols].sum(axis=1)
    df["n_unique_events"] = (df[event_cols] > 0).sum(axis=1)

    df = df.join(first_seen_timestamps(RAW_ZIP), on="BlockId")
    df = df.rename(columns={"BlockId": "block_id", "Latency": "latency"})
    return df.sort_values("first_ts").reset_index(drop=True)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT} rows={len(df):,} anomaly_rate={df['label'].mean():.3%} "
          f"missing_ts={df['first_ts'].isna().sum()} span={df['first_ts'].min()} -> {df['first_ts'].max()}")
