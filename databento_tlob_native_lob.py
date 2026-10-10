"""
databento_tlob_native_lob.py

Databento MBP-10 -> native-style TLOB LOB-only adapter.

Model input: 40 L10 features in the official TLOB/LOBSTER order
    ask_px, ask_sz, bid_px, bid_sz
for levels 0..9.

Optional TLOB-style quantity sampling:
    cumsum = size.cumsum()
    keep = (cumsum % quantity < size)

Native TLOB labels:
    smooth=10 by default
    horizon in {10,20,50,100}
    alpha = mean(abs(future/past - 1))/2
    class 0=up, 1=stationary, 2=down

Important: Databento MBP-10 action codes are not a one-to-one encoding
of LOBSTER event types, so no event-action filter is silently imposed.
Use keep_actions=("A","C","T") only if you intentionally want that
approximate LOBSTER-like subset.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from read_databento_new import dbn_to_df

LOB_COLS = []
SIZE_COLS = []
for level in range(10):
    s = f"{level:02d}"
    LOB_COLS += [
        f"ask_px_{s}", f"ask_sz_{s}",
        f"bid_px_{s}", f"bid_sz_{s}",
    ]
    SIZE_COLS += [f"ask_sz_{s}", f"bid_sz_{s}"]

REQUIRED_COLS = ["ts_event", "action", "size"] + LOB_COLS


@dataclass(frozen=True)
class LOBZScoreStats:
    mean_price: float
    std_price: float
    mean_size: float
    std_size: float


@dataclass
class PreparedDay:
    name: str
    raw_lob: np.ndarray
    pct_change: np.ndarray
    lob: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    n_raw: int = 0
    n_filtered: int = 0
    n_sampled: int = 0


def load_databento_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate one event-level Databento MBP-10 CSV."""
    path = Path(path)
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")

    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="raise", format="mixed")
    sort_cols = ["ts_event"] + (["sequence"] if "sequence" in df.columns else [])
    df = df.sort_values(sort_cols).reset_index(drop=True)

    for c in LOB_COLS + ["size"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    #'''
    if df[LOB_COLS].isna().any().any():
        bad = df[LOB_COLS].columns[df[LOB_COLS].isna().any()].tolist()
        raise ValueError(f"{path}: NaN/non-numeric LOB columns: {bad}")

    X = df[LOB_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: non-finite LOB values")
    
    #'''
    if (df[SIZE_COLS].to_numpy(dtype=np.float64) < 0).any():
        raise ValueError(f"{path}: negative book sizes")
    if (df["bid_px_00"] > df["ask_px_00"]).any():
        raise ValueError(f"{path}: crossed book rows found")

    return df

def load_databento_zst(path: str | Path) -> pd.DataFrame:
    """Load and validate one event-level Databento MBP-10 CSV."""
    path = Path(path)
    #df = pd.read_csv(path)
    fName = ''
    df = dbn_to_df(fName, path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")

    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="raise", format="mixed")
    sort_cols = ["ts_event"] + (["sequence"] if "sequence" in df.columns else [])
    df = df.sort_values(sort_cols).reset_index(drop=True)

    for c in LOB_COLS + ["size"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    '''
    if df[LOB_COLS].isna().any().any():
        bad = df[LOB_COLS].columns[df[LOB_COLS].isna().any()].tolist()
        raise ValueError(f"{path}: NaN/non-numeric LOB columns: {bad}")

    X = df[LOB_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: non-finite LOB values")
    '''
    if (df[SIZE_COLS].to_numpy(dtype=np.float64) < 0).any():
        raise ValueError(f"{path}: negative book sizes")
    if (df["bid_px_00"] > df["ask_px_00"]).any():
        raise ValueError(f"{path}: crossed book rows found")

    return df




def sample_quantity(
    df: pd.DataFrame,
    quantity: int = 500,
    keep_actions: Optional[Sequence[str]] = None,
):
    """
    Reproduce TLOB's quantity-sampling formula.

    keep_actions=None uses all Databento MBP-10 events.
    keep_actions=("A","C","T") is an optional approximate LOBSTER-like filter.
    """
    if quantity <= 0:
        raise ValueError("quantity must be > 0")

    work = df.copy()
    n_raw = len(work)

    if keep_actions is not None:
        keep_actions = tuple(map(str, keep_actions))
        work = work[work["action"].astype(str).isin(keep_actions)].copy()

    work = work.reset_index(drop=True)
    if work.empty:
        raise ValueError("No rows remain before sampling")

    sz = pd.to_numeric(work["size"], errors="coerce").fillna(0.0)
    cumsum = sz.cumsum()
    mask = (cumsum % quantity < sz)
    sampled = work.loc[mask].reset_index(drop=True)

    if sampled.empty:
        raise ValueError("Quantity sampling produced zero rows")

    info = {
        "n_raw": n_raw,
        "n_filtered": len(work),
        "n_sampled": len(sampled),
        "quantity": quantity,
        "keep_actions": keep_actions,
        "action_counts_raw": df["action"].astype(str).value_counts().to_dict(),
        "action_counts_sampled":
            sampled["action"].astype(str).value_counts().to_dict(),
    }
    return sampled, info


def extract_lob40(df: pd.DataFrame) -> np.ndarray:
    """Return [N,40] in exact TLOB LOB-only feature order."""
    return df[LOB_COLS].to_numpy(dtype=np.float64, copy=True)


def fit_zscore_stats(raw_days: Iterable[np.ndarray]) -> LOBZScoreStats:
    """Fit TLOB-style common price and size z-score statistics."""
    arrays = [np.asarray(x, dtype=np.float64) for x in raw_days]
    X = np.concatenate(arrays, axis=0)

    prices = X[:, 0::2]
    sizes = X[:, 1::2]

    stats = LOBZScoreStats(
        mean_price=float(prices.mean()),
        std_price=float(prices.std(ddof=1)),
        mean_size=float(sizes.mean()),
        std_size=float(sizes.std(ddof=1)),
    )
    if stats.std_price <= 0 or stats.std_size <= 0:
        raise ValueError("Invalid normalization std")
    return stats


def apply_zscore(X_raw: np.ndarray, stats: LOBZScoreStats) -> np.ndarray:
    X = np.asarray(X_raw, dtype=np.float64).copy()
    X[:, 0::2] = (X[:, 0::2] - stats.mean_price) / stats.std_price
    X[:, 1::2] = (X[:, 1::2] - stats.mean_size) / stats.std_size
    return X.astype(np.float32)


def tlob_percentage_change(
    X_raw: np.ndarray,
    len_smooth: int = 10,
    horizon: int = 10,
) -> np.ndarray:
    """
    Reproduce the percentage-change series used by TLOB labeling().

    TLOB feature order means:
      best ask = column 0
      best bid = column 2
    """
    if len_smooth <= 0 or horizon <= 0:
        raise ValueError("len_smooth and horizon must be > 0")

    smooth = min(len_smooth, horizon)
    X = np.asarray(X_raw, dtype=np.float64)

    if len(X) < smooth + horizon:
        return np.empty(0, dtype=np.float64)

    ask_w = np.lib.stride_tricks.sliding_window_view(X[:, 0], smooth)
    bid_w = np.lib.stride_tricks.sliding_window_view(X[:, 2], smooth)

    prev = (ask_w[:-horizon] + bid_w[:-horizon]) / 2.0
    fut = (ask_w[horizon:] + bid_w[horizon:]) / 2.0

    prev = prev.mean(axis=1)
    fut = fut.mean(axis=1)

    return (fut - prev) / prev


def alpha_from_changes(changes: Iterable[np.ndarray]) -> float:
    x = np.concatenate([np.asarray(a) for a in changes if len(a) > 0])
    return float(np.abs(x).mean() / 2.0)


def labels_from_change(change: np.ndarray, alpha: float) -> np.ndarray:
    """TLOB: 0=up, 1=stationary, 2=down."""
    x = np.asarray(change)
    return np.where(
        x < -alpha, 2,
        np.where(x > alpha, 0, 1)
    ).astype(np.int64)


class NativeTLOBWindowDataset(Dataset):
    """
    Day-safe TLOB windows.

    For x[i:i+seq_size], the native label index is
        j = i + seq_size - min(len_smooth, horizon)

    This matches the official lobster_load()/Dataset alignment while
    preventing windows from crossing trading-day boundaries.
    """

    def __init__(
        self,
        days: Sequence[PreparedDay],
        seq_size: int = 128,
        len_smooth: int = 10,
        horizon: int = 10,
    ):
        self.days = list(days)
        self.seq_size = seq_size
        self.smooth = min(len_smooth, horizon)
        self.horizon = horizon
        self.index = []

        for d, day in enumerate(self.days):
            if day.lob is None or day.labels is None:
                raise ValueError(f"{day.name}: split not finalized")

            n_examples = len(day.lob) - seq_size - horizon + 1
            for i in range(max(0, n_examples)):
                j = i + seq_size - self.smooth
                self.index.append((d, i, j))

        if not self.index:
            raise ValueError("No usable sequence windows")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        d, i, j = self.index[idx]
        day = self.days[d]
        x = day.lob[i:i + self.seq_size]
        y = day.labels[j]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def restrict_to_rth_et(df, ts_col_utc, start, end):
    # ts_event is tz-aware UTC per Databento
    df['ts_et'] = df[ts_col_utc].dt.tz_convert("America/New_York") 

    # Apply the filter
    df  = df[(df['ts_et'].dt.time >= start) & (df['ts_et'].dt.time <= end)]

    return df

def clean_sampled_l10(
    sampled,
    file_name="",
):
    sampled = sampled.copy()

    price_cols = []
    size_cols = []

    for level in range(10):

        s = f"{level:02d}"

        price_cols += [
            f"ask_px_{s}",
            f"bid_px_{s}",
        ]

        size_cols += [
            f"ask_sz_{s}",
            f"bid_sz_{s}",
        ]

    n0 = len(sampled)

    # ----------------------------
    # Valid best bid / ask
    # ----------------------------

    valid_bbo = (
        sampled["bid_px_00"].notna()
        &
        sampled["ask_px_00"].notna()
        &
        (sampled["bid_px_00"] > 0)
        &
        (sampled["ask_px_00"] > 0)
        &
        (
            sampled["bid_px_00"]
            <= sampled["ask_px_00"]
        )
    )

    n_bad_bbo = int((~valid_bbo).sum())

    # ----------------------------
    # Complete L10 state
    # ----------------------------

    complete_prices = (
        sampled[price_cols]
        .notna()
        .all(axis=1)
    )

    complete_sizes = (
        sampled[size_cols]
        .notna()
        .all(axis=1)
    )

    # positive prices
    positive_prices = (
        sampled[price_cols] > 0
    ).all(axis=1)

    # sizes may legitimately be zero,
    # but should not be negative
    valid_sizes = (
        sampled[size_cols] >= 0
    ).all(axis=1)

    valid = (
        valid_bbo
        & complete_prices
        & complete_sizes
        & positive_prices
        & valid_sizes
    )

    n_valid = int(valid.sum())
    n_drop = n0 - n_valid

    print(
        f"{file_name}: "
        f"sampled={n0:,} | "
        f"bad_BBO={n_bad_bbo:,} | "
        f"incomplete/invalid_L10={n_drop:,} | "
        f"kept={n_valid:,} "
        f"({100*n_valid/max(n0,1):.3f}%)"
    )

    return (
        sampled.loc[valid]
        .reset_index(drop=True)
    )
def prepare_raw_days(
    files: Sequence[str | Path],
    *,
    quantity: int = 500,
    keep_actions: Optional[Sequence[str]] = None,
    len_smooth: int = 10,
    horizon: int = 10,
):
    """Load -> quantity sample -> extract raw 40-dim LOB -> target changes."""
    days, diagnostics = [], []

    for file in files:
        file = Path(file)
        df = load_databento_zst(file)
        
        # "normal" eastern trading time
        tStart = datetime.time(9, 30)
        tEnd   = datetime.time(15, 30)
        
        time_stamm_column_utc="ts_event"    
        df = restrict_to_rth_et(df, time_stamm_column_utc , tStart,tEnd)
        
        
        sampled, info = sample_quantity(
            df,
            quantity=quantity,
            keep_actions=keep_actions,
        )
        
        sampled = clean_sampled_l10(
            sampled,
            file.name,
            )   
            
        raw = extract_lob40(sampled)
        pct = tlob_percentage_change(
            raw,
            len_smooth=len_smooth,
            horizon=horizon,
        )

        days.append(PreparedDay(
            name=file.stem,
            raw_lob=raw,
            pct_change=pct,
            n_raw=info["n_raw"],
            n_filtered=info["n_filtered"],
            n_sampled=info["n_sampled"],
        ))
        info["file"] = str(file)
        diagnostics.append(info)

    return days, diagnostics


def finalize_days(
    days: Sequence[PreparedDay],
    stats: LOBZScoreStats,
    *,
    alpha: Optional[float] = None,
):
    """Normalize and assign labels to a split."""
    days = list(days)

    if alpha is None:
        alpha = alpha_from_changes(day.pct_change for day in days)

    for day in days:
        day.lob = apply_zscore(day.raw_lob, stats)
        day.labels = labels_from_change(day.pct_change, alpha)

    return days, float(alpha)


def prepare_native_tlob_lob_only(
    train_files: Sequence[str | Path],
    val_files: Optional[Sequence[str | Path]] = None,
    *,
    seq_size: int = 128,
    len_smooth: int = 10,
    horizon: int = 10,
    quantity: int = 500,
    keep_actions: Optional[Sequence[str]] = None,
    alpha_policy: str = "official_split",
):
    """
    End-to-end adapter.

    alpha_policy:
      "official_split" -> train and validation each derive their own alpha,
                          closest to the official TLOB preprocessing.
      "train"          -> validation reuses training alpha; stricter OOS rule.
    """
    train_days, train_diag = prepare_raw_days(
        train_files,
        quantity=quantity,
        keep_actions=keep_actions,
        len_smooth=len_smooth,
        horizon=horizon,
    )

    stats = fit_zscore_stats(day.raw_lob for day in train_days)
    train_days, train_alpha = finalize_days(train_days, stats)

    train_dataset = NativeTLOBWindowDataset(
        train_days,
        seq_size=seq_size,
        len_smooth=len_smooth,
        horizon=horizon,
    )

    val_days = None
    val_dataset = None
    val_alpha = None
    val_diag = []

    if val_files:
        val_days, val_diag = prepare_raw_days(
            val_files,
            quantity=quantity,
            keep_actions=keep_actions,
            len_smooth=len_smooth,
            horizon=horizon,
        )

        if alpha_policy == "official_split":
            val_days, val_alpha = finalize_days(val_days, stats)
        elif alpha_policy == "train":
            val_days, val_alpha = finalize_days(
                val_days, stats, alpha=train_alpha
            )
        else:
            raise ValueError(
                "alpha_policy must be 'official_split' or 'train'"
            )

        val_dataset = NativeTLOBWindowDataset(
            val_days,
            seq_size=seq_size,
            len_smooth=len_smooth,
            horizon=horizon,
        )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_days": train_days,
        "val_days": val_days,
        "normalization": stats,
        "train_alpha": train_alpha,
        "val_alpha": val_alpha,
        "train_diagnostics": train_diag,
        "val_diagnostics": val_diag,
        "config": {
            "num_features": 40,
            "seq_size": seq_size,
            "len_smooth": len_smooth,
            "horizon": horizon,
            "quantity": quantity,
            "keep_actions": keep_actions,
            "alpha_policy": alpha_policy,
        },
    }


def class_distribution(dataset: Dataset):
    counts = np.zeros(3, dtype=np.int64)
    for i in range(len(dataset)):
        _, y = dataset[i]
        counts[int(y)] += 1
    return counts, counts / counts.sum()


def print_summary(prepared: dict):
    print("\nNative TLOB / Databento LOB-only")
    print("-" * 60)
    print("config:", prepared["config"])
    print("normalization:", prepared["normalization"])
    print(f"train alpha = {prepared['train_alpha']:.10g}")
    print(f"train windows = {len(prepared['train_dataset']):,}")
    c, p = class_distribution(prepared["train_dataset"])
    print("train [up/stat/down] =", c.tolist(), np.round(p, 4).tolist())

    print("\nTraining-day sampling")
    for d in prepared["train_diagnostics"]:
        print(
            f"{Path(d['file']).name}: "
            f"raw={d['n_raw']:,} filtered={d['n_filtered']:,} "
            f"sampled={d['n_sampled']:,}"
        )

    if prepared["val_dataset"] is not None:
        print(f"\nval alpha = {prepared['val_alpha']:.10g}")
        print(f"val windows = {len(prepared['val_dataset']):,}")
        c, p = class_distribution(prepared["val_dataset"])
        print("val [up/stat/down] =", c.tolist(), np.round(p, 4).tolist())


# Backward-compatible alias used in the accompanying runner/instructions.
def print_preparation_summary(prepared: dict):
    return print_summary(prepared)
