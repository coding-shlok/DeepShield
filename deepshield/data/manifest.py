"""Manifest CSV handling and leakage-free 70/15/15 splitting (PRD Phase 1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import MANIFEST_COLUMNS


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sample_id": str, "group": str, "method": str, "path": str})
    df["method"] = df["method"].fillna("")
    df["group"] = df["group"].fillna(df["sample_id"])
    if "split" not in df:
        df["split"] = ""
    df["split"] = df["split"].fillna("")
    df.attrs["root"] = str(Path(path).resolve().parent)
    return df


def write_manifest(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in MANIFEST_COLUMNS if c in df.columns] + [c for c in df.columns if c not in MANIFEST_COLUMNS]
    df[cols].to_csv(path, index=False)


def assign_splits(df: pd.DataFrame, ratios=(0.70, 0.15, 0.15), seed: int = 42,
                  keep_existing: bool = True) -> pd.DataFrame:
    """Group-aware, label-stratified split.

    All samples sharing a `group` (identity / source video – e.g. a FF++ original and every
    manipulation of it) land in the same split, which prevents identity leakage that would
    otherwise inflate test accuracy. Rows that already have a split (official dataset splits)
    are kept when `keep_existing`.
    """
    df = df.copy()
    if "split" not in df:
        df["split"] = ""
    todo = df["split"].isin(["", None]) | df["split"].isna() if keep_existing else pd.Series(True, index=df.index)
    if not todo.any():
        return df
    sub = df[todo]
    # group's label = majority (fake if any fake – stratify by "contains fake")
    g = sub.groupby("group")["label"].max()
    rng = np.random.default_rng(seed)
    names = ["train", "val", "test"]
    assignment = {}
    for lbl in sorted(g.unique()):
        groups = g[g == lbl].index.to_numpy()
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = max(n_val, 1)
        cuts = [n_train, n_train + n_val]
        for i, grp in enumerate(groups):
            assignment[grp] = names[0] if i < cuts[0] else names[1] if i < cuts[1] else names[2]
    df.loc[todo, "split"] = sub["group"].map(assignment).values
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["dataset", "split", "label"]).size().unstack(fill_value=0)
