# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 11:23:55 2026

@author: vanio
"""
from __future__ import annotations
from read_databento_new import dbn_to_df, X_pred_Y, estimate_subsequence_counts,estimate_observed_subsequence_counts
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
from time_series_analysis import calculate_correlation, calculate_mutual_information,  calculate_transfer_entropy
#from time_series_analysis import compute_mi_discrete 
from plot_distributions import plotDistribution, plot_distributions_comparison_preserve_order
import datetime
import pickle
from integrate_day_distributions import integrate_distributions,  integrate_conditional_class_distributions

from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Sequence, Tuple
from plotting import plt_ts, overlay_intraday_series
from matplotlib.ticker import MaxNLocator
 
from collections import defaultdict
from itertools import product

# #############################################################################
#            Visualisations
# =============================================================================
def set_paper_style():
    """
    Compact publication style suitable for IEEE/CIFEr figures.
    """
    plt.rcParams.update({
        # Font / text
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 8.5,
        "axes.labelsize": 6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,

        # Figure / axes
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "lines.linewidth": 0.7,

        # Grid
        "axes.grid": True,
        "grid.linewidth": 0.25,
        "grid.alpha": 0.18,

        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,

        # Legend
        "legend.frameon": False,

        # Output
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })



def save_paper_figure(fig, filename_base):
    fig.savefig(f"{filename_base}.pdf")
    fig.savefig(f"{filename_base}.png")


import numpy as np
import matplotlib.pyplot as plt


def plot_three_rebased_prices(
    t_list,
    y_list,
    labels,
    xlabel="Normalized intraday time",
    ylabel=r"$\Delta \log(\mathrm{MP})$",
    title=None,
    figsize=(3.35, 2.1),
    lw=1.2,
    save_as=None,
):
    """
    Overlay daily log-mid-price trajectories after:

      1. rebasing each trajectory to zero;
      2. mapping each day's horizontal axis to [0, 1].

    Parameters
    ----------
    t_list : list
        Retained for consistency and validation. Actual calendar dates are
        not used on the horizontal axis.
    y_list : list of array-like
        Daily log-mid-price series.
    labels : list of str
        Day labels.
    """
    set_paper_style()
    if not (len(t_list) == len(y_list) == len(labels)):
        raise ValueError("t_list, y_list, and labels must have equal length.")

    fig, ax = plt.subplots(figsize=figsize)

    for t, y, label in zip(t_list, y_list, labels):
        y = np.asarray(y, dtype=float)

        if len(y) == 0:
            continue

        if len(t) != len(y):
            raise ValueError(
                f"Time and price arrays have different lengths for {label}: "
                f"{len(t)} and {len(y)}."
            )

        # y is assumed to already contain log-mid prices
        y_rebased = y - y[0]

        # Align every trading day on the same [0, 1] interval
        intraday_time = np.linspace(0.0, 1.0, len(y))

        ax.plot(
            intraday_time,
            y_rebased,
           #linewidth=lw,
            label=label,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.grid(True, linewidth=0.4, alpha=0.3)
    #ax.legend(frameon=False, fontsize=5,loc="upper center")
    ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 0.97),
    ncol=3,
    frameon=False,
    fontsize=6,
    columnspacing=1,
    handlelength=2.0,
)
    
    
    ax.set_xlim(0.0, 1.0)

    fig.tight_layout()

    if save_as is not None:
        fig.savefig(save_as, bbox_inches="tight")

    return fig, ax


# =============================================================================



def add_class_label(
    time_series: pd.DataFrame,
    class_name: str,
    fill_missing: bool = False,
) -> pd.Series:
    """
    Add a {-1,0,1} classification column to time_series and return it.

    Supported class_name:
      c{k}   : based on log_mid_return_fwd_k
      ca{k}  : based on log_mid_sum_fwd_k vs log_mid_sum_bwd_k

    Examples:
      add_class_label(time_series, "c1",  theta=0.0000415)
      add_class_label(time_series, "c2",  theta=0.0000415)
      add_class_label(time_series, "c4",  theta=0.000075)
      add_class_label(time_series, "ca1", theta=0.000007)
      add_class_label(time_series, "ca2", theta=0.000007)
      add_class_label(time_series, "ca4", theta=0.000007)
    """
    thetaDict = {}
    thetaDict["c1"] =0.0000415
    thetaDict["c2"] =0.0000415
    thetaDict["c4"] =0.000075
    thetaDict["ca1"]=0.000007
    thetaDict["ca2"]=0.000007
    thetaDict["ca4"]=0.000007
    
    theta = thetaDict[class_name]
    if class_name.startswith("ca"):
        k = int(class_name[2:])

        fwd_col = f"log_mid_sum_fwd_{k}"
        bwd_col = f"log_mid_sum_bwd_{k}"

        if fwd_col not in time_series.columns:
            raise KeyError(f"Missing column: {fwd_col}")
        if bwd_col not in time_series.columns:
            raise KeyError(f"Missing column: {bwd_col}")

        fwd = time_series[fwd_col]
        bwd = time_series[bwd_col]

        if fill_missing:
            fwd = fwd.ffill()
            bwd = bwd.bfill()

        # reproduce the original scaling:
        # ca1 -> theta
        # ca2 -> 1.5 * theta
        # ca4 -> 2.0 * theta
        theta_mult = {
            1: 1.0,
            2: 1.5,
            4: 2.0,
        }.get(k, 1.0)

        th = theta * theta_mult

        time_series[class_name] = np.select(
            [
                fwd > bwd * (1.0 + th),
                fwd < bwd * (1.0 - th),
            ],
            [1, -1],
            default=0,
        ).astype(np.int8)

    elif class_name.startswith("c"):
        k = int(class_name[1:])

        ret_col = f"log_mid_return_fwd_{k}"

        if ret_col not in time_series.columns:
            raise KeyError(f"Missing column: {ret_col}")

        r = time_series[ret_col]

        if fill_missing:
            r = r.ffill()

        time_series[class_name] = np.select(
            [
                r > theta,
                r < -theta,
            ],
            [1, -1],
            default=0,
        ).astype(np.int8)

    else:
        raise ValueError("class_name must start with 'c' or 'ca', e.g. 'c1', 'c2', 'ca1', 'ca4'.")

    return time_series[class_name]
#------------------------------------------------------------------------------
# Intended for VPIN classifiaction task
#------------------------------------------------------------------------------
def add_vpin_class_label(
    time_series: pd.DataFrame,
    class_name: str,
    vpin_col: str = "vpin_sym",
) -> pd.Series:
    """
    Add a binary VPIN behavioral class column.

    Class names:
        cvp{k} : classify the recent k-step VPIN-symbol history

    VPIN symbols:
        0, 1 -> low VPIN states
        2, 3 -> high VPIN states

    Classification:
        1 : number of {2,3} observations > number of {0,1}
        0 : otherwise

    The initial observations use progressively shorter windows:
        t=0 -> window length 1
        t=1 -> window length 2
        ...
        until the full window k is reached.

    Examples:
        add_vpin_class_label(time_series, "cvp4")
        add_vpin_class_label(time_series, "cvp5")
        add_vpin_class_label(time_series, "cvp10")
    """

    if not class_name.startswith("cvp"):
        raise ValueError(
            "class_name must have the form 'cvp{k}', "
            "e.g. 'cvp4', 'cvp5'."
        )

    try:
        k = int(class_name[3:])
    except ValueError:
        raise ValueError(
            "class_name must have the form 'cvp{k}', "
            "where k is an integer."
        )

    if k < 1:
        raise ValueError("VPIN history window k must be >= 1.")

    if vpin_col not in time_series.columns:
        raise KeyError(f"Missing column: {vpin_col}")

    vpin = time_series[vpin_col]

    # Indicator of whether the current VPIN symbol is in the high regime
    high = vpin.isin([2, 3]).astype(np.int8)

    # Indicator of whether it is in the low regime
    low = vpin.isin([0, 1]).astype(np.int8)

    # Number of high/low VPIN states in the trailing k observations.
    # min_periods=1 gives windows 1, 2, ..., k at the beginning.
    high_count = high.rolling(
        window=k,
        min_periods=1
    ).sum()

    low_count = low.rolling(
        window=k,
        min_periods=1
    ).sum()

    # High VPIN behavioral state iff high symbols dominate
    time_series[class_name] = (
        high_count > low_count
    ).astype(np.int8)

    return time_series[class_name]


#------------------------------------------------------------------------------
def estimate_subsequence_class_distributions(seq, classes, num_classes, max_subsequence_length):
    """
    Calculates the empirical class distribution at the terminal point 
    for all possible subsequences up to a maximum length.
    
    Parameters:
    - seq: List or array of symbols (e.g., z-encoded LOB mid-prices)
    - classes: Parallel list/array of 0-indexed integer classes for each point
    - num_classes: Total number of unique classes (n)
    - max_subsequence_length: Maximum window size (k)
    
    Returns:
    - all_subsequences: List of lists containing all possible subsequence tuples per length
    - distributions: List of lists containing [subsequence, class_counts_list, total_occurrences]
    """
    unique_symbols = sorted(set(seq))
    all_subsequences = []
    distributions = []
    
    for length in range(1, max_subsequence_length + 1):
        # Generate all possible permutations for this length
        possible_subseqs = list(product(unique_symbols, repeat=length))
        all_subsequences.append(possible_subseqs)
        
        # Map each subsequence to an array counting occurrences of each class index
        # Factory function initializes a list of zeros tracking [class_0_count, class_1_count, ...]
        subseq_class_counts = defaultdict(lambda: [0] * num_classes)
        
        # Sliding window over the parallel sequences
        for i in range(len(seq) - length + 1):
            subseq = tuple(seq[i:i + length])
            
            # The terminal class is at the last index of the current window
            terminal_index = i + length - 1
            terminal_class = classes[terminal_index]
            
            # Increment the counter for this specific class under this subsequence context
            subseq_class_counts[subseq][terminal_class] += 1

        # Format the final distributions structure
        length_distributions = []
        for subseq in possible_subseqs:
            class_counts = subseq_class_counts.get(subseq, [0] * num_classes)
            total_occurrences = sum(class_counts)
            
            # Appends the sequence, raw class frequencies, and total times observed
            length_distributions.append([subseq, class_counts, total_occurrences])
            
        distributions.append(length_distributions)
    
    return all_subsequences, distributions

def estimate_subsequence_class_probabilities_01(seq, classes, num_classes, max_subsequence_length, non_zero_only = True):

    if len(seq) != len(classes):
        raise ValueError("seq and classes must have the same length.")

    unique_symbols = sorted(set(seq))
    all_subsequences = []
    distributions = []

    for length in range(1, max_subsequence_length + 1):
        possible_subseqs = list(product(unique_symbols, repeat=length))
        all_subsequences.append(possible_subseqs)

        subseq_class_counts = defaultdict(lambda: [0] * num_classes)

        for i in range(len(seq) - length + 1):
            subseq = tuple(seq[i:i + length])
            terminal_class = classes[i + length - 1]
            subseq_class_counts[subseq][terminal_class] += 1

        length_distributions = []
        for subseq in possible_subseqs:
            class_counts = subseq_class_counts.get(subseq, [0] * num_classes)
            total_occurrences = sum(class_counts)

            if total_occurrences > 0:
                class_probs = [c / total_occurrences for c in class_counts]
            else:
                class_probs = [0.0] * num_classes
            
            if  not non_zero_only or total_occurrences > 0 :   
                length_distributions.append(
                    [subseq, class_counts, class_probs, total_occurrences]
                )
        if len(length_distributions)>0:
            distributions.append(length_distributions)

    return all_subsequences, distributions



def estimate_subsequence_class_probabilities(
    seq,
    classes,
    max_subsequence_length,
    class_values=(-1, 0, 1),
    non_zero_only=True,
):
    """
    Estimate P(class | subsequence).

    Distribution columns follow the explicit order in class_values.
    Default order:
        [P(-1), P(0), P(1)]
    """
    if len(seq) != len(classes):
        raise ValueError("seq and classes must have the same length.")

    class_values = tuple(class_values)
    num_classes = len(class_values)

    class_to_index = {
        label: index
        for index, label in enumerate(class_values)
    }

    unknown_classes = set(classes) - set(class_values)
    if unknown_classes:
        raise ValueError(
            f"Unknown class labels found: {unknown_classes}. "
            f"Expected labels: {class_values}."
        )

    unique_symbols = sorted(set(seq))
    all_subsequences = []
    distributions = []

    for length in range(1, max_subsequence_length + 1):
        possible_subseqs = list(
            product(unique_symbols, repeat=length)
        )
        all_subsequences.append(possible_subseqs)

        subseq_class_counts = defaultdict(
            lambda: [0] * num_classes
        )

        for i in range(len(seq) - length + 1):
            subseq = tuple(seq[i:i + length])
            terminal_class = classes.iloc[i + length - 1]

            class_index = class_to_index[terminal_class]
            subseq_class_counts[subseq][class_index] += 1

        length_distributions = []

        for subseq in possible_subseqs:
            class_counts = subseq_class_counts.get(
                subseq,
                [0] * num_classes,
            )

            total_occurrences = sum(class_counts)

            if total_occurrences > 0:
                class_probs = [
                    count / total_occurrences
                    for count in class_counts
                ]
            else:
                class_probs = [0.0] * num_classes

            if not non_zero_only or total_occurrences > 0:
                length_distributions.append(
                    [
                        subseq,
                        class_counts,
                        class_probs,
                        total_occurrences,
                    ]
                )

        if length_distributions:
            distributions.append(length_distributions)

    return all_subsequences,  distributions


Seq = Tuple[int, ...]

import matplotlib.pyplot as plt


def plot_lob_time_series(ts_dict, freq):
    # 1. Initialize the plot figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # 2. Loop through the dictionary and plot each time series
    for date, log_prices in ts_dict.items():
        # Reconstruct the event index (sampled every freq   events)
        event_index = [i * freq for i in range(len(log_prices))]

        # Plot the series with a descriptive label for the legend
        ax.plot(event_index, log_prices, label=f"Date: {date}", alpha=0.85)

    # 3. Add styling, labels, and titles
    ax.set_title(
        "LOB Log Mid-Price Comparison (100-Event Sampling)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Event Count (Ticks)", fontsize=12)
    ax.set_ylabel("Log Mid-Price", fontsize=12)

    # 4. Enhance readability
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=11)

    # Format numbers on y-axis clearly if they are very precise floats
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))

    # 5. Render the plot cleanly
    plt.tight_layout()
    plt.show()
#------------------------------------------------------------------------------
# Z-encoding of tume series
#------------------------------------------------------------------------------
# ---------- Basic building blocks ----------
def ewma_zscore(series: pd.Series, alpha: float) -> pd.Series:
    x = series.astype(float)
    mu = x.ewm(alpha=alpha, adjust=False).mean()
    sq = (x - mu)**2
    var = sq.ewm(alpha=alpha, adjust=False).mean()
    sigma = np.sqrt(var)
    z = (x - mu) / sigma
    return z.replace([np.inf, -np.inf], np.nan).dropna()
#-----------------------------------------------------------------------------
def ewma_zscore_predictive(series: pd.Series, alpha: float, min_periods: int = 2,
                           eps: float = 1e-12) -> pd.Series:
    """
    Predictive (causal) EWMA z-score:
        z[i] = (x[i] - mu[i-1]) / sigma[i-1]
    where mu, sigma are computed from past only.
    """
    x = series.astype("float64")

    # Past-only series for estimating mu_{i-1}, sigma_{i-1}
    x_lag = x.shift(1)

    # EWMA mean of past
    mu_prev = x_lag.ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()

    # EWMA variance of past (EWMA of squared deviations from past mean)
    dev_prev = x_lag - mu_prev
    var_prev = (dev_prev * dev_prev).ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()
    sigma_prev = np.sqrt(var_prev).clip(lower=eps)

    z = (x - mu_prev) / sigma_prev
    z = z.replace([np.inf, -np.inf], np.nan)
    return z

#-----------------------------------------------------------------------------

def discretize_z(z: pd.Series, bins: np.ndarray) -> pd.Series:
    """
    Map z-scores to symbols 0..m-1 using given bin edges in z-space.
    bins: array of shape (m-1,) with increasing thresholds.
    """
    symbols = np.digitize(z.values, bins=bins, right=False)  # 0..m-1
    return pd.Series(symbols, index=z.index)

def discretize_z_new(z: pd.Series, bins: np.ndarray) -> pd.Series:
    sym = np.digitize(z.values, bins=bins, right=False).astype(np.int32)  # 0..m-1
    return pd.Series(sym, index=z.index)


def z_encode_ts_current(ts: pd.Series, alpha: float, bins: np.ndarray):
    """
    Given a continuous time series, EWMA parameter alpha, and z-bins,
    return (z_scores, symbols).
    """
    z = ewma_zscore(ts, alpha=alpha)
    s = discretize_z(z, bins=bins)
    s = s.loc[z.index]  # ensure alignment
    return z, s

def z_encode_ts_predictive(ts: pd.Series, alpha: float, bins: np.ndarray,
                           min_periods: int = 2) -> tuple[pd.Series, pd.Series]:
    """
    Given a continuous time series, EWMA parameter alpha, and z-bins,
    return (z_scores, symbols).
    """
    z = ewma_zscore_predictive(ts, alpha=alpha, min_periods=min_periods)
    s = discretize_z(z, bins=bins)
    # keep symbols NaN wherever z is NaN (warmup)
    s = s.where(z.notna())
    
    z = z.bfill()
    z = z.ffill()
    
    s = s.bfill()
    s = s.ffill()
    
    return z, s


def fit_quantile_bins(z: pd.Series, m: int) -> np.ndarray:
    """
    Equal-frequency thresholds for m symbols => m-1 bin edges.
    Fitted on the provided z (e.g., train split only).
    """
    z = z.dropna().astype("float64").values
    qs = np.linspace(0, 1, m + 1)[1:-1]          # 1/m, 2/m, ..., (m-1)/m
    edges = np.quantile(z, qs)

    # Ensure strictly increasing edges (duplicates can happen if z has many ties)
    edges = np.asarray(edges, dtype="float64")
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = np.nextafter(edges[i-1], np.inf)  # tiny nudge upward
    return edges

def fit_quantile_bins_new(z: pd.Series, m: int) -> np.ndarray:
    z = z.dropna().astype("float64").values
    qs = np.linspace(0, 1, m + 1)[1:-1]  # 1/m .. (m-1)/m
    edges = np.quantile(z, qs)
    # ensure strictly increasing
    edges = np.asarray(edges, dtype="float64")
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = np.nextafter(edges[i-1], np.inf)
    return edges


def fit_bins_per_alpha(x_train: pd.Series, alphas, m: int, min_periods: int = 2):
    bins = {}
    for a in alphas:
        z_a = ewma_zscore_predictive(x_train, alpha=a, min_periods=min_periods)
        bins[a] = fit_quantile_bins(z_a, m)
    return bins

#------------------------------------------------------------------------------
def add_event_features_and_resample_time(
    df: pd.DataFrame,
    dt_sec: int = 1,                 # bucket size in seconds
    W_events: int = 300,             # trailing event-vol window on raw events (optional)
    k_fwdL = [1,2,3],        # forward horizons in *buckets*
    sort_cols=("ts_event", "sequence"),
    fill_empty_buckets: bool = True, # create a regular clock grid and ffill snapshot
) -> pd.DataFrame:
    """
    Clock-time bars from event stream:
      - Each dt_sec bucket is represented by the last event in the bucket (snapshot columns).
      - Block features are computed using the set of events inside that bucket.
      - Optional: fill empty buckets by forward-filling snapshot and setting event-based counts to 0.
Features:
'mid_price'
'log_mid' 
'log_mid_ret' 
'sigma_W' 
'mid_changed' 
'mid_change_count_n'   -number of midprice chages in the block - 0 ..n
'move_rate_per_event'   normalized number of chages per event
'ofi_L1'                order flow imbalance
'ofi_L1_n'              rolling OFI over last n events
'dt_block_sec'          seconds per block
'log_dt_block'
'event_intensity'       events per second
'log_event_intensity'
'move_rate_per_sec'     mid price movements per second
'trade_buy_vol_event'   the trade size at time t if the trade was buy-initiated
'trade_sell_vol_event'  the trade size at time t if the trade was sell-initiated  
'trade_buy_vol_n'       last n events
'trade_sell_vol_n'      last n events
'trade_ofi_n'           rolling signed trade volume imbalance over last n events
'vpin'                  volume-bucket VPIN on trades
'event_idx'
'sample_idx'
'log_mid_return_fwd_1'
'log_mid_sum_bwd_1'
'log_mid_sum_fwd_1'
'log_mid_sum_ratio_1'
'imbalance'
'micro_price'
'spread'
'rel_spread'
'log_spread'
'micro_mid'
'micro_mid_sign'
'mid_cross_prev_ask_up'
'mid_cross_prev_bid_dn'
'mid_ret'
'jump_gt_prev_spread'
    """

    df2 = df.sort_values(list(sort_cols)).copy()

    tcol = "ts_event"

    # --- forward-fill LOB snapshot columns at event level ---
    lob_prefixes = ("bid_px_", "ask_px_", "bid_sz_", "ask_sz_", "bid_ct_", "ask_ct_")
    lob_cols = [c for c in df2.columns if c.startswith(lob_prefixes)]
    df2[lob_cols] = df2[lob_cols].ffill()

    # --- event-level mid/log/returns ---
    df2["mid_price"] = (df2["bid_px_00"].astype("float64") + df2["ask_px_00"].astype("float64")) / 2.0
    df2["log_mid"] = np.log(df2["mid_price"].where(df2["mid_price"] > 0))
    # NOTE: this diff crosses bucket boundaries; for within-bucket vol we compute separately below
    df2["log_mid_ret"] = df2["log_mid"].diff()

    # --- optional trailing event-vol sigma_W on raw events (RMS over last W_events returns) ---
    if W_events is not None and W_events > 0:
        df2["sigma_W"] = (
            df2["log_mid_ret"]
               .rolling(W_events, min_periods=W_events)
               .apply(lambda x: np.sqrt(np.mean(x * x)), raw=True)
        )
    else:
        df2["sigma_W"] = np.nan

    # --- assign each event to a time bucket (bucket_start) ---
    freq = f"{int(dt_sec)}S"
    df2["bucket_start"] = df2[tcol].dt.floor(freq)

    # --- group by bucket and compute:
    #     - last snapshot row (rep bar)
    #     - within-bucket counts/vol/intensity
    g = df2.groupby("bucket_start", sort=True)

    # (A) representative snapshot = last event in bucket
    last_cols = [tcol] + lob_cols + ["mid_price", "log_mid", "sigma_W"]
    dfN = g[last_cols].last().copy()

    # store last-event timestamp explicitly
    dfN.rename(columns={tcol: "ts_last_event"}, inplace=True)

    # (B) within-bucket event_count
    dfN["event_count"] = g.size().astype(np.int32)

    # (C) within-bucket duration (span between first and last event in the bucket)
    #     (can be 0 if only 1 event; for empty buckets we'll set 0)
    t_first = g[tcol].first()
    t_last = g[tcol].last()
    dfN["dt_span_sec"] = (t_last - t_first).dt.total_seconds().astype("float64")

    # (D) within-bucket mid-price change count (changes *inside* bucket)
    #     use diffs inside each bucket to avoid boundary artifacts
    def _mid_change_count_in_bucket(s: pd.Series) -> int:
        v = s.to_numpy()
        if v.size <= 1:
            return 0
        return int(np.sum(v[1:] != v[:-1]))

    dfN["mid_change_count"] = g["mid_price"].apply(_mid_change_count_in_bucket).astype(np.int16)

    # (E) within-bucket realized vol proxy on log_mid, computed *inside* bucket
    #     returns = diff within bucket, RMS = sqrt(mean(r^2))
    def _sigma_bucket_from_logmid(s: pd.Series) -> float:
        v = s.to_numpy(dtype=float)
        if v.size <= 1:
            return np.nan
        r = np.diff(v)
        return float(np.sqrt(np.mean(r * r)))

    dfN["sigma_bucket"] = g["log_mid"].apply(_sigma_bucket_from_logmid).astype("float64")

    # (F) rates / intensity (use fixed dt_sec for a stable clock-time definition)
    eps = 1e-12
    dfN["dt_bucket_sec"] = float(dt_sec)
    dfN["event_intensity"] = dfN["event_count"] / dfN["dt_bucket_sec"]
    dfN["log_event_intensity"] = np.log(dfN["event_intensity"].clip(lower=eps))
    dfN["move_rate_per_event"] = dfN["mid_change_count"] / dfN["event_count"].replace(0, np.nan)
    dfN["move_rate_per_sec"] = dfN["mid_change_count"] / dfN["dt_bucket_sec"]

    # --- optionally fill empty buckets to get a regular clock grid ---
    if fill_empty_buckets:
        full_idx = pd.date_range(dfN.index.min(), dfN.index.max(), freq=freq, tz=df2[tcol].dt.tz)
        dfN = dfN.reindex(full_idx)

        # if no events in a bucket: event-based quantities become 0, snapshot forward-fills
        dfN["event_count"] = dfN["event_count"].fillna(0).astype(np.int32)
        dfN["mid_change_count"] = dfN["mid_change_count"].fillna(0).astype(np.int16)
        dfN["dt_span_sec"] = dfN["dt_span_sec"].fillna(0.0)

        # forward fill snapshot/state features from previous bucket
        dfN[lob_cols + ["mid_price", "log_mid", "sigma_W"]] = dfN[lob_cols + ["mid_price", "log_mid", "sigma_W"]].ffill()
        dfN["ts_last_event"] = dfN["ts_last_event"].ffill()

        # recompute derived rates safely after filling
        dfN["event_intensity"] = dfN["event_count"] / float(dt_sec)
        dfN["log_event_intensity"] = np.log(dfN["event_intensity"].clip(lower=eps))
        dfN["move_rate_per_event"] = dfN["mid_change_count"] / dfN["event_count"].replace(0, np.nan)
        dfN["move_rate_per_sec"] = dfN["mid_change_count"] / float(dt_sec)

    # index is bucket_start (clock grid)
    dfN.index.name = "bucket_start"
    dfN["sample_idx"] = np.arange(len(dfN), dtype=np.int64)

    # --- forward targets on the clock-time series (k_fwd in buckets) ---
    for k_fwd in k_fwdL:
        k = int(k_fwd)
        s = dfN["log_mid"]

        dfN[f"log_mid_return_fwd_{k}"] = s.shift(-k) - s
        dfN[f"log_mid_sum_bwd_{k}"] = s.rolling(window=k, min_periods=k).sum()
        dfN[f"log_mid_sum_fwd_{k}"] = s.shift(-1).rolling(window=k, min_periods=k).sum().shift(-(k-1))

        bwd = dfN[f"log_mid_sum_bwd_{k}"]
        dfN[f"log_mid_sum_ratio_{k}"] = (dfN[f"log_mid_sum_fwd_{k}"] - bwd) / bwd.replace(0, np.nan)

    # --- LOB derived features on the clock-time snapshots ---
    den = (dfN["bid_sz_00"] + dfN["ask_sz_00"]).replace(0, np.nan)
    dfN["imbalance"] = dfN["bid_sz_00"] / den
    dfN["micro_price"] = dfN["imbalance"] * dfN["ask_px_00"] + (1.0 - dfN["imbalance"]) * dfN["bid_px_00"]

    dfN["spread"] = dfN["ask_px_00"] - dfN["bid_px_00"]
    dfN["rel_spread"] = dfN["spread"] / dfN["mid_price"].replace(0, np.nan)

    eps_p = 1e-12
    dfN["log_spread"] = np.log(dfN["ask_px_00"].clip(lower=eps_p)) - np.log(dfN["bid_px_00"].clip(lower=eps_p))

    dfN["micro_mid"] = dfN["micro_price"] - dfN["mid_price"]
    dfN["micro_mid_sign"] = np.sign(dfN["micro_mid"])

    prev_bid = dfN["bid_px_00"].shift(1)
    prev_ask = dfN["ask_px_00"].shift(1)
    dfN["mid_cross_prev_ask_up"] = (dfN["mid_price"] > prev_ask).astype(np.int8)
    dfN["mid_cross_prev_bid_dn"] = (dfN["mid_price"] < prev_bid).astype(np.int8)

    dfN["mid_ret"] = dfN["mid_price"].diff()
    prev_spread = dfN["spread"].shift(1)
    dfN["jump_gt_prev_spread"] = (dfN["mid_ret"].abs() > prev_spread).astype(np.int8)

    return dfN

#-----------------------------------------------------------------------------


#------------------------------------------------------------------------------
def add_ofi_level_k(
    df2: pd.DataFrame,
    k: int = 0,
    n: int | None = None,
    prefix: str = "ofi",
    max_levels: int = 10,   # MBP-10
) -> pd.DataFrame:
    """
    Adds event-level OFI at book level k:
      - f"{prefix}_{k}"   : OFI_k(t)
      - f"{prefix}_{k}_n" : rolling sum over last n events (if n provided)

    Assumes bid_px_XX, ask_px_XX, bid_sz_XX, ask_sz_XX exist and are forward-filled.
    """
    if not (0 <= k < max_levels):
        raise ValueError(f"k must be in [0,{max_levels-1}] for MBP-{max_levels}.")

    bp = df2[f"bid_px_{k:02d}"].astype(float)
    ap = df2[f"ask_px_{k:02d}"].astype(float)
    bq = df2[f"bid_sz_{k:02d}"].astype(float)
    aq = df2[f"ask_sz_{k:02d}"].astype(float)

    bp_prev, ap_prev = bp.shift(1), ap.shift(1)
    bq_prev, aq_prev = bq.shift(1), aq.shift(1)

    db = np.where(bp > bp_prev,  bq,
         np.where(bp < bp_prev, -bq_prev,
                  bq - bq_prev))

    da = np.where(ap < ap_prev,  aq,
         np.where(ap > ap_prev, -aq_prev,
                  aq - aq_prev))

    ofi_k = pd.Series(db - da, index=df2.index, name=f"{prefix}_{k}")
    df2[f"{prefix}_{k}"] = ofi_k
    df2[f"{prefix}_{k}"] = df2[f"{prefix}_{k}"].fillna(0)
    if n is not None and n > 0:
        df2[f"{prefix}_{k}_n"] = ofi_k.rolling(n, min_periods=1).sum()
        df2[f"{prefix}_{k}_n"].fillna(0)
    return df2
           
def add_ofi_multi_level(
    df2: pd.DataFrame,
    L: int = 10,
    n: int | None = None,
    weights: str = "exp",
    tau: float = 3.0,
    prefix: str = "ofi",
) -> pd.DataFrame:
    """
    Adds:
      - f"{prefix}_L{L}"   : weighted sum of OFI_k for k=0..L-1
      - f"{prefix}_L{L}_n" : rolling sum over last n events (optional)
    """
    if L <= 0:
        raise ValueError("L must be positive.")
    L = min(int(L), 10)  # MBP-10

    # weights
    if weights == "uniform":
        w = np.ones(L, dtype=float)
    elif weights == "inv":
        w = 1.0 / (np.arange(L, dtype=float) + 1.0)
    elif weights == "exp":
        if tau <= 0:
            raise ValueError("tau must be > 0 for exp weights.")
        w = np.exp(-np.arange(L, dtype=float) / float(tau))
    else:
        raise ValueError("weights must be 'uniform', 'inv', or 'exp'.")

    w = w / w.sum()  # keep scale stable across L

    # compute weighted sum
    ofi_sum = np.zeros(len(df2), dtype=float)
    for k in range(L):
        add_ofi_level_k(df2, k=k, n=None, prefix=prefix, max_levels=10)
        ofi_sum += w[k] * df2[f"{prefix}_{k}"].to_numpy(dtype=float)

    name = f"{prefix}_L{L}"
    df2[name] = ofi_sum

    if n is not None and n > 0:
        df2[f"{name}_n"] = df2[name].rolling(n, min_periods=n).sum()

    return df2
#------------------------------------------------------------------------------
def make_ofi_weights(L: int = 10, weights: str = "exp", tau: float = 3.0):
    L = min(int(L), 10)

    if weights == "uniform":
        w = np.ones(L, dtype=float)

    elif weights == "inv":
        w = 1.0 / (np.arange(L, dtype=float) + 1.0)

    elif weights == "exp":
        if tau <= 0:
            raise ValueError("tau must be > 0 for exp weights.")
        w = np.exp(-np.arange(L, dtype=float) / float(tau))

    else:
        raise ValueError("weights must be 'uniform', 'inv', or 'exp'.")

    return w / w.sum()
#------------------------------------------------------------------------------
def add_normalized_ofi_multi_level(
    df2,
    L: int = 10,
    n: int | None = 10,
    weights: str = "exp",
    tau: float = 3.0,
    prefix: str = "ofi",
    eps: float = 1e-12,
):
    """
    Adds:
      - ofi_L10_norm   : weighted deep OFI divided by weighted depth
      - ofi_L10_norm_n : rolling weighted deep OFI divided by rolling weighted depth
    """

    L = min(int(L), 10)

    # First compute weighted deep OFI event-level:
    # df2[f"{prefix}_L{L}"] = sum_k w_k OFI_k
    df2 = add_ofi_multi_level(
        df2,
        L=L,
        n=None,
        weights=weights,
        tau=tau,
        prefix=prefix,
    )

    name = f"{prefix}_L{L}"
    ofi = df2[name].astype(float)

    # Same weights as numerator
    w = make_ofi_weights(L=L, weights=weights, tau=tau)

    bid_cols = [f"bid_sz_{k:02d}" for k in range(L)]
    ask_cols = [f"ask_sz_{k:02d}" for k in range(L)]

    weighted_bid_depth = df2[bid_cols].astype(float).mul(w, axis=1).sum(axis=1)
    weighted_ask_depth = df2[ask_cols].astype(float).mul(w, axis=1).sum(axis=1)

    depth = (weighted_bid_depth + weighted_ask_depth).clip(lower=eps)

    # event-level normalized weighted deep OFI
    df2[f"{name}_norm"] = ofi / depth

    if n is not None and n > 0:
        # rolling normalized weighted deep OFI
        # normalize after summing numerator and denominator over the same window
        ofi_n = ofi.rolling(n, min_periods=1).sum()
        depth_n = depth.rolling(n, min_periods=1).sum()

        df2[f"{name}_norm_n"] = ofi_n / depth_n.clip(lower=eps)

    return df2

#------------------------------------------------------------------------------
def add_event_features_and_resample(
    df: pd.DataFrame,
    n: int = 10,                          # events resampling window
    W: int = 300,                         # back window for volatility estimATION
    k_fwdL = [1,2,3],                     # forward  windows for the predicted feature
    sort_cols=("ts_event", "sequence"),
    take: str = "last",                   # "last" keeps 10th/20th/... ; "first" keeps 1st/11th/...
    # VPIN params
    compute_vpin: bool = True,
    vpin_bucket_vol: float = 10000, #50_000.0,    # volume bucket size (shares/contracts)
    vpin_mavg: int = 10 # 20, 50,                  # rolling avg over last m buckets
) -> pd.DataFrame:

    if take not in {"last", "first"}:
        raise ValueError("take must be 'last' or 'first'")

    df2 = df.sort_values(list(sort_cols)).copy()
    tcol = "ts_event"  # tz-aware UTC datetime per Databento


    # --- forward-fill LOB snapshot columns (so every row has a complete state) ---
    lob_prefixes = ("bid_px_", "ask_px_", "bid_sz_", "ask_sz_", "bid_ct_", "ask_ct_")
    lob_cols = [c for c in df2.columns if c.startswith(lob_prefixes)]
    df2[lob_cols] = df2[lob_cols].ffill()

    # --- mid / log-mid / event returns (raw event series) ---
    df2["mid_price"] = (df2["bid_px_00"].astype("float64") + df2["ask_px_00"].astype("float64")) / 2.0
    df2["log_mid"] = np.log(df2["mid_price"].where(df2["mid_price"] > 0))
    df2["log_mid_ret"] = df2["log_mid"].diff()

    # --- event-volatility sigma_W at every raw event (RMS over last W event-returns) ---
    df2["sigma_W"] = (
        df2["log_mid_ret"]
           .rolling(W, min_periods=W)
           .apply(lambda x: np.sqrt(np.mean(x * x)), raw=True)
    )
    

    # --- block features over last n raw events (aligned to each raw event) ---
    # Mid-price changes count over last n raw events
    dmid = df2["mid_price"].diff()
    df2["mid_changed"] = dmid.fillna(0).ne(0).astype(np.int8)
    
    df2["mid_change_count_n"] = (
        df2["mid_changed"]
           .rolling(n, min_periods=1)
           .sum()
           .astype(np.int16)
    )
    
    df2["move_rate_per_event"] = df2["mid_change_count_n"] / n
#------------------------------------------------------------------------------
    # --- L1 OFI (Cont-style) per event, then rolling sum over last n events ---
    bpx = df2["bid_px_00"].astype(float)
    apx = df2["ask_px_00"].astype(float)
    bsz = df2["bid_sz_00"].astype(float)
    asz = df2["ask_sz_00"].astype(float)

    bpx_prev, apx_prev = bpx.shift(1), apx.shift(1)
    bsz_prev, asz_prev = bsz.shift(1), asz.shift(1)

    db = np.where(bpx > bpx_prev,  bsz,
         np.where(bpx < bpx_prev, -bsz_prev,
                  bsz - bsz_prev))
    da = np.where(apx < apx_prev,  asz,
         np.where(apx > apx_prev, -asz_prev,
                  asz - asz_prev))
    df2["ofi_L1"] = (db - da)
    df2["ofi_L1"] = df2["ofi_L1"].fillna(0)
    # order flow imbalance: Flow (LOB): ofi_L1_n (rolling OFI over last n events)
#   df2["ofi_L1_n"] = pd.Series(df2["ofi_L1"], index=df2.index).rolling(n, min_periods=n).sum()
#    df2["ofi_L1_n"] = df2["ofi_L1"].rolling(window=n, min_periods=1).sum()
    


    # 1. Compute your clean rolling sum exactly as you have it
    df2["ofi_L1_n"] = df2["ofi_L1"].rolling(window=n, min_periods=1).sum()
    
    # 2. Compute the simultaneous total depth at Level 1 per event
    total_depth_event = df2["bid_sz_00"] + df2["ask_sz_00"]
    
    # 3. Get the average depth across that exact same backward window
    # Note: Match the window and min_periods parameters perfectly
    rolling_avg_depth = total_depth_event.rolling(window=n, min_periods=1).mean()
    
    # 4. Safe division to prevent dividing by zero if depth vanishes
    df2["ofi_L1_n_norm"] = df2["ofi_L1_n"] / rolling_avg_depth.replace(0, np.nan)
    df2["ofi_L1_n_norm"] = df2["ofi_L1_n_norm"].fillna(0.0)



    
    
#------------------------------------------------------------------------------
    # Physical duration of last n events: τ_t - τ_{t-n+1} 
    # Using diff(periods=n-1) gives τ_t - τ_{t-(n-1)}
    # duration/intensity over last n events
    eps_time = 1e-9
    df2["dt_block_sec"] = df2[tcol].diff(n - 1).dt.total_seconds()
    df2["dt_block_sec"] = df2["dt_block_sec"].clip(lower=eps_time)

    df2["log_dt_block"] = np.log(df2["dt_block_sec"])
    df2["event_intensity"] = n / df2["dt_block_sec"]
    df2["log_event_intensity"] = np.log(df2["event_intensity"])
    df2["move_rate_per_sec"] = df2["mid_change_count_n"] / df2["dt_block_sec"]

#------------------------------------------------------------------------------
    # --- Trade-based rolling OFI over last n events (uses action/side/size) ---
    a = df2["action"].astype(str).str.upper()
    side = df2["side"].astype(str).str.upper()
    is_trade = a.str.startswith("T")  # adjust if your trade action differs
    size = df2["size"].astype(float).to_numpy()
    price = df2["price"].astype(float).to_numpy()
    mid = df2["mid_price"].to_numpy()
    
    # aggressor sign: +1 buy, -1 sell
    sgn = np.zeros(len(df2), dtype=np.int8)
    sgn[is_trade.values & (side.values == "B")] = 1
    sgn[is_trade.values & (side.values == "A")] = -1

    # fallback for side == 'N' on trade rows: midpoint/tick rule
    unk = is_trade.values & (side.values == "N")
    if np.any(unk):
        s = np.sign(price[unk] - mid[unk]).astype(np.int8)
        tick = np.sign(np.diff(price, prepend=price[0]))[unk].astype(np.int8)
        s = np.where(s != 0, s, tick)
        # carry last nonzero within the *trade stream* (simple pass)
        last = 0
        for i, val in enumerate(s):
            if val == 0:
                s[i] = last
            else:
                last = val
        sgn[unk] = s

    buy_vol_event = np.where(is_trade.values & (sgn > 0), size, 0.0)
    sell_vol_event = np.where(is_trade.values & (sgn < 0), size, 0.0)

    df2["trade_buy_vol_event"] = buy_vol_event
    df2["trade_sell_vol_event"] = sell_vol_event

    df2["trade_buy_vol_n"] = pd.Series(buy_vol_event, index=df2.index).rolling(n, min_periods=n).sum()
    df2["trade_sell_vol_n"] = pd.Series(sell_vol_event, index=df2.index).rolling(n, min_periods=n).sum()
    tv = df2["trade_buy_vol_n"] + df2["trade_sell_vol_n"]
    # Flow (trades): trade_ofi_n (rolling signed trade volume imbalance over last n events)
    df2["trade_ofi_n"] = ((df2["trade_buy_vol_n"] - df2["trade_sell_vol_n"]) / tv.replace(0, np.nan)).fillna(0)
    df2["tvi_n"] =       ((df2["trade_buy_vol_n"] - df2["trade_sell_vol_n"]) / tv.replace(0, np.nan)).fillna(0)
    # --- VPIN (volume-bucket VPIN on trades), forward-filled to events ---
    df2["vpin"] = np.nan
    if compute_vpin:
        tr_idx = np.flatnonzero(is_trade.values)
        if tr_idx.size > 0:
            tr_size = size[tr_idx]
            tr_buy = buy_vol_event[tr_idx]
            tr_sell = sell_vol_event[tr_idx]

            cumv = np.cumsum(tr_size)
            bucket_id = (cumv // float(vpin_bucket_vol)).astype(np.int64)

            buy_b = pd.Series(tr_buy).groupby(bucket_id).sum()
            sell_b = pd.Series(tr_sell).groupby(bucket_id).sum()

            imb = (buy_b - sell_b).abs()
            vpin_bucket = (imb / float(vpin_bucket_vol)).rolling(vpin_mavg, min_periods=vpin_mavg).mean()

            # map bucket VPIN back to each trade, then ffill to all events
            vpin_trade = pd.Series(bucket_id).map(vpin_bucket).to_numpy()
            #df2.loc[df2.index[tr_idx], "vpin"] = vpin_trade
            col = df2.columns.get_loc("vpin")
            df2.iloc[tr_idx, col] = vpin_trade
            df2["vpin"] = df2["vpin"].ffill()
            df2["vpin"] = df2["vpin"].bfill()
            df2["dvpin"]= df2["vpin"].diff()
            df2["dvpin"]= df2["dvpin"].bfill()
#------------------------------------------------------------------------------
    # Optional: mid-change rate per second
    df2["move_rate_per_sec"] = df2["mid_change_count_n"] / df2["dt_block_sec"].clip(lower=eps_time)


    ofi_window = n         # For scale-invariance / fractal-style analysis
    # ofi_window = 10      # For predictive modeling  

    df2 = add_ofi_multi_level(df2, L=10, n=ofi_window, weights="exp", tau=3.0, prefix="ofi")
    #deep_ofi = df2["ofi_L10_n"]


    # noprmalized ofi
    df2 = add_normalized_ofi_multi_level(df2, L=1, n=ofi_window, weights="exp", tau=3.0, prefix="ofi")

    df2 = add_normalized_ofi_multi_level(df2, L=3, n=ofi_window, weights="exp", tau=3.0, prefix="ofi")

    df2 = add_normalized_ofi_multi_level(df2, L=10, n=ofi_window, weights="exp", tau=3.0, prefix="ofi")

    

    # --- downsample every n events (keep 'first' or 'last' in each n-block) ---
    ev = np.arange(len(df2), dtype=np.int64)
    if take == "first":
        mask = (ev % n) == 0
    else:
        mask = (ev % n) == (n - 1)
    dfN = df2.loc[mask].copy()
    
    dfN = dfN.sort_index()  # timestamp index
    
    dfN["event_idx"] = ev[mask]
    dfN["sample_idx"] = (dfN["event_idx"] // n).astype(np.int64)
    # -------------------------------------------------------------------------
    
    
    # k-step-ahead log return on the resampled series 
    # --- forward k-step return on the RESAMPLED series (optional) ---
    for k_fwd in k_fwdL:
        s = dfN["log_mid"]
    
        # forward k-step return: log_mid[t+k] - log_mid[t]
        dfN[f"log_mid_return_fwd_{k_fwd}"] = s.shift(-k_fwd) - s
        dfN[f"log_mid_return_fwd_{k_fwd}"].replace( np.nan,0)
        # backward sum: log_mid[t-k+1] + ... + log_mid[t]
        dfN[f"log_mid_sum_bwd_{k_fwd}"] = s.rolling(window=k_fwd, min_periods=k_fwd).sum()
    
        # forward sum: log_mid[t+1] + ... + log_mid[t+k]
        # rolling is backward-looking => shift result by -(k-1)
        dfN[f"log_mid_sum_fwd_{k_fwd}"] = (
            s.shift(-1).rolling(window=k_fwd, min_periods=k_fwd).sum().shift(-(k_fwd-1))
        )
    
        bwd = dfN[f"log_mid_sum_bwd_{k_fwd}"]
        dfN[f"log_mid_sum_ratio_{k_fwd}"] = (dfN[f"log_mid_sum_fwd_{k_fwd}"] - bwd) / bwd.replace(0, np.nan)

    
        # --- classification indicator (avoid divide-by-zero) ---
        bwd = dfN[f"log_mid_sum_bwd_{k_fwd}"].to_numpy()
        fwd = dfN[f"log_mid_sum_fwd_{k_fwd}"].to_numpy()
    
        denom = np.where(np.abs(bwd) < 1e-12, np.nan, bwd)
        dfN[f"log_mid_sum_ratio_{k_fwd}"] = (fwd - bwd) / denom
        dfN[f"log_mid_sum_ratio_{k_fwd}"].replace( np.nan,0)
    # ------------------------------------------------------------------------
    
    # --- imbalance / microprice / spread features on RESAMPLED series ---
    # 1. Protect against division by zero
    den = (dfN["bid_sz_00"] + dfN["ask_sz_00"]).replace(0, np.nan)
    
    # 2. Use a distinct name for the micro-price interpolation weight (0 to 1)
    dfN["bid_ratio_L1"] = dfN["bid_sz_00"] / den
    
    # 3. Calculate micro-price using the ratio
    dfN["micro_price"] = dfN["bid_ratio_L1"] * dfN["ask_px_00"] + (1.0 - dfN["bid_ratio_L1"]) * dfN["bid_px_00"]
    
    # 4. Calculate the standard Order Book Imbalance (OBI) metric (-1 to 1) for current model
    dfN["obi_L1"] = (dfN["bid_sz_00"] - dfN["ask_sz_00"]) / den
    dfN["obi_L1"] = dfN["obi_L1"].fillna(0.0) # Handle empty book cases safely

    dfN["imbalance"] = (dfN["bid_sz_00"] - dfN["ask_sz_00"]) / den
    dfN["imbalance"] = dfN["imbalance"].fillna(0.0)  # Neutral fallback for empty books



    dfN["spread"] = dfN["ask_px_00"] - dfN["bid_px_00"]
    dfN["rel_spread"] = dfN["spread"] / dfN["mid_price"]  # = 2*spread/(ask+bid)

    eps = 1e-12
    dfN["log_spread"] = np.log(dfN["ask_px_00"].clip(lower=eps)) - np.log(dfN["bid_px_00"].clip(lower=eps))

    dfN["micro_mid"] = dfN["micro_price"] - dfN["mid_price"]
    dfN["micro_mid_sign"] = np.sign(dfN["micro_mid"])


    # --- “mid crosses previous spread” on RESAMPLED series ---
    prev_bid = dfN["bid_px_00"].shift(1)
    prev_ask = dfN["ask_px_00"].shift(1)

    dfN["mid_cross_prev_ask_up"] = (dfN["mid_price"] > prev_ask).astype(np.int8)
    dfN["mid_cross_prev_bid_dn"] = (dfN["mid_price"] < prev_bid).astype(np.int8)

    # --- “mid jump relative to spread” (resampled) ---
    dfN["log_mid_ret"] = dfN["log_mid"].diff()
    dfN["mid_ret"]     = dfN["mid_price"].diff()
    prev_spread = dfN["spread"].shift(1)
    
    dfN["jump_gt_prev_spread"] = (dfN["mid_ret"].abs() > prev_spread).astype(np.int8)

    # Clean helper column if created
    if "_ts_event_dt" in dfN.columns:
        # keep it if you like; otherwise drop
        pass

    return dfN
#------------------------------------------------------------------------------
#-----VOLUME TIME RESAMPLING
#------------------------------------------------------------------------------
def bucket_sum(values, end_pos, prev_end_pos):
    x = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    cs = np.cumsum(x)

    prev = np.zeros(len(end_pos), dtype=float)
    ok = prev_end_pos >= 0
    prev[ok] = cs[prev_end_pos[ok]]

    return cs[end_pos] - prev

#------------------------------------------------------------------------------
def add_event_features_and_resample_volume(
    df: pd.DataFrame,
    n_shares: int = 10000,                   # volume resampling window
    W: int = 300,                         # back window for volatility estimATION
    k_fwdL = [1,2,3],                     # forward  windows for the predicted feature
    sort_cols=("ts_event", "sequence"),
    take: str = "last",                   # "last" keeps 10th/20th/... ; "first" keeps 1st/11th/...
    # VPIN params
    compute_vpin: bool = True,
    vpin_bucket_vol: float = 10000, #50_000.0,    # volume bucket size (shares/contracts)
    vpin_mavg: int = 10 # 20, 50,                  # rolling avg over last m buckets
) -> pd.DataFrame:

    #ofi_window = n         # For scale-invariance / fractal-style analysis
    ofi_window = 10        # For predictive modeling  
    n =   ofi_window
    
    if take not in {"last", "first"}:
        raise ValueError("take must be 'last' or 'first'")

    df2 = df.sort_values(list(sort_cols)).copy()
    tcol = "ts_event"  # tz-aware UTC datetime per Databento


    # --- forward-fill LOB snapshot columns (so every row has a complete state) ---
    lob_prefixes = ("bid_px_", "ask_px_", "bid_sz_", "ask_sz_", "bid_ct_", "ask_ct_")
    lob_cols = [c for c in df2.columns if c.startswith(lob_prefixes)]
    df2[lob_cols] = df2[lob_cols].ffill()

    # --- mid / log-mid / event returns (raw event series) ---
    df2["mid_price"] = (df2["bid_px_00"].astype("float64") + df2["ask_px_00"].astype("float64")) / 2.0
    df2["log_mid"] = np.log(df2["mid_price"].where(df2["mid_price"] > 0))
    df2["log_mid_ret"] = df2["log_mid"].diff()

    # --- event-volatility sigma_W at every raw event (RMS over last W event-returns) ---
    df2["sigma_W"] = (
        df2["log_mid_ret"]
           .rolling(W, min_periods=W)
           .apply(lambda x: np.sqrt(np.mean(x * x)), raw=True)
    )
    

    # --- block features over last n raw events (aligned to each raw event) ---
    # Mid-price changes count over last n raw events
    dmid = df2["mid_price"].diff()
    df2["mid_changed"] = dmid.fillna(0).ne(0).astype(np.int8)
    
#------------------------------------------------------------------------------
    # --- L1 OFI (Cont-style) per event, then rolling sum over last n events ---
    bpx = df2["bid_px_00"].astype(float)
    apx = df2["ask_px_00"].astype(float)
    bsz = df2["bid_sz_00"].astype(float)
    asz = df2["ask_sz_00"].astype(float)

    bpx_prev, apx_prev = bpx.shift(1), apx.shift(1)
    bsz_prev, asz_prev = bsz.shift(1), asz.shift(1)

    db = np.where(bpx > bpx_prev,  bsz,
         np.where(bpx < bpx_prev, -bsz_prev,
                  bsz - bsz_prev))
    da = np.where(apx < apx_prev,  asz,
         np.where(apx > apx_prev, -asz_prev,
                  asz - asz_prev))
    df2["ofi_L1"] = (db - da)
    # order flow imbalance: Flow (LOB): ofi_L1_n (rolling OFI over last n events)
    df2["ofi_L1_n"] = pd.Series(df2["ofi_L1"], index=df2.index).rolling(n, min_periods=n).sum()

#------------------------------------------------------------------------------
    # Physical duration of last n events: τ_t - τ_{t-n+1} 
    # Using diff(periods=n-1) gives τ_t - τ_{t-(n-1)}
    # duration/intensity over last n events
    eps_time = 1e-9
    df2["dt_block_sec"] = df2[tcol].diff(n - 1).dt.total_seconds()
    df2["dt_block_sec"] = df2["dt_block_sec"].clip(lower=eps_time)

    df2["log_dt_block"] = np.log(df2["dt_block_sec"])
    df2["event_intensity"] = n / df2["dt_block_sec"]
    df2["log_event_intensity"] = np.log(df2["event_intensity"])
    #df2["move_rate_per_sec"] = df2["mid_change_count_n"] / df2["dt_block_sec"]

#------------------------------------------------------------------------------
    # --- Trade-based rolling OFI over last n events (uses action/side/size) ---
    a = df2["action"].astype(str).str.upper()
    side = df2["side"].astype(str).str.upper()
    is_trade = a.str.startswith("T")  # adjust if your trade action differs
    size = df2["size"].astype(float).to_numpy()
    price = df2["price"].astype(float).to_numpy()
    mid = df2["mid_price"].to_numpy()
    
    # aggressor sign: +1 buy, -1 sell
    sgn = np.zeros(len(df2), dtype=np.int8)
    sgn[is_trade.values & (side.values == "B")] = 1
    sgn[is_trade.values & (side.values == "A")] = -1

    # fallback for side == 'N' on trade rows: midpoint/tick rule
    unk = is_trade.values & (side.values == "N")
    if np.any(unk):
        s = np.sign(price[unk] - mid[unk]).astype(np.int8)
        tick = np.sign(np.diff(price, prepend=price[0]))[unk].astype(np.int8)
        s = np.where(s != 0, s, tick)
        # carry last nonzero within the *trade stream* (simple pass)
        last = 0
        for i, val in enumerate(s):
            if val == 0:
                s[i] = last
            else:
                last = val
        sgn[unk] = s

    buy_vol_event = np.where(is_trade.values & (sgn > 0), size, 0.0)
    sell_vol_event = np.where(is_trade.values & (sgn < 0), size, 0.0)

    df2["trade_buy_vol_event"] = buy_vol_event
    df2["trade_sell_vol_event"] = sell_vol_event

    df2["trade_buy_vol_n"] = pd.Series(buy_vol_event, index=df2.index).rolling(n, min_periods=n).sum()
    df2["trade_sell_vol_n"] = pd.Series(sell_vol_event, index=df2.index).rolling(n, min_periods=n).sum()
    tv = df2["trade_buy_vol_n"] + df2["trade_sell_vol_n"]
    # Flow (trades): trade_ofi_n (rolling signed trade volume imbalance over last n events)
    df2["trade_ofi_n"] = (df2["trade_buy_vol_n"] - df2["trade_sell_vol_n"]) / tv.replace(0, np.nan)

    # --- VPIN (volume-bucket VPIN on trades), forward-filled to events ---
    df2["vpin"] = np.nan
    if compute_vpin:
        tr_idx = np.flatnonzero(is_trade.values)
        if tr_idx.size > 0:
            tr_size = size[tr_idx]
            tr_buy = buy_vol_event[tr_idx]
            tr_sell = sell_vol_event[tr_idx]

            cumv = np.cumsum(tr_size)
            bucket_id = (cumv // float(vpin_bucket_vol)).astype(np.int64)

            buy_b = pd.Series(tr_buy).groupby(bucket_id).sum()
            sell_b = pd.Series(tr_sell).groupby(bucket_id).sum()

            imb = (buy_b - sell_b).abs()
            vpin_bucket = (imb / float(vpin_bucket_vol)).rolling(vpin_mavg, min_periods=vpin_mavg).mean()

            # map bucket VPIN back to each trade, then ffill to all events
            vpin_trade = pd.Series(bucket_id).map(vpin_bucket).to_numpy()
            #df2.loc[df2.index[tr_idx], "vpin"] = vpin_trade
            col = df2.columns.get_loc("vpin")
            df2.iloc[tr_idx, col] = vpin_trade
            df2["vpin"] = df2["vpin"].ffill()
            df2["vpin"] = df2["vpin"].bfill()
            df2["dvpin"]= df2["vpin"].diff()
            df2["dvpin"]= df2["dvpin"].bfill()
#------------------------------------------------------------------------------
    # Optional: mid-change rate per second
    # df2["move_rate_per_sec"] = df2["mid_change_count_n"] / df2["dt_block_sec"].clip(lower=eps_time)



     # df2 = add_ofi_multi_level(df2, L=10, n=ofi_window, weights="exp", tau=3.0, prefix="ofi")
    #deep_ofi = df2["ofi_L10_n"]


    # df2 = add_normalized_ofi_multi_level(df2, L=10, n=ofi_window, weights="exp", tau=3.0, prefix="ofi")
    # main feature:
    # deep_nofi_n = df2["ofi_L10_norm_n"]
    


   
    # --- volume-time resampling: keep the event where cumulative volume crosses j*n_shares ---
    
    trade_vol_event = df2["trade_buy_vol_event"] + df2["trade_sell_vol_event"]
    cumv = trade_vol_event.cumsum().to_numpy()
    
    thresholds = np.arange(n_shares, cumv[-1] + 1e-12, n_shares)
    
    end_pos = np.searchsorted(cumv, thresholds, side="left")
    
    valid = end_pos < len(df2)
    end_pos = end_pos[valid]
    thresholds = thresholds[valid]
    
    dfN = df2.iloc[end_pos].copy()
    
    # If df2 is already sorted by ts_event/sequence, this is already chronological.
    # Use sort_index only if the index is truly a timestamp index.
    dfN = dfN.sort_index()
    
    dfN["event_idx"] = end_pos
    dfN["sample_idx"] = np.arange(len(dfN), dtype=np.int64)
    dfN["cum_trade_vol"] = cumv[end_pos]
    dfN["volume_threshold"] = thresholds
    
    prev_end_pos = np.r_[-1, end_pos[:-1]].astype(np.int64)

    dfN["mid_change_count_vol_bucket"] = bucket_sum(
        df2["mid_changed"].to_numpy(),
        end_pos,
        prev_end_pos
    ).astype(np.int32)
    
    dfN["bucket_trade_vol"] = bucket_sum(
        trade_vol_event,
        end_pos,
        prev_end_pos
    )
    
    dfN["events_in_bucket"] = end_pos - prev_end_pos
    
    dfN["move_rate_per_share"] = (
        dfN["mid_change_count_vol_bucket"]
        / dfN["bucket_trade_vol"].replace(0, np.nan)
    )
    
    dfN["move_rate_per_event_in_vol_bucket"] = (
        dfN["mid_change_count_vol_bucket"]
        / dfN["events_in_bucket"].replace(0, np.nan)
    )


    
    
    # -------------------------------------------------------------------------
    
    
    # k-step-ahead log return on the resampled series 
    # --- forward k-step return on the RESAMPLED series (optional) ---
    for k_fwd in k_fwdL:
        s = dfN["log_mid"]
    
        # forward k-step return: log_mid[t+k] - log_mid[t]
        dfN[f"log_mid_return_fwd_{k_fwd}"] = s.shift(-k_fwd) - s
        dfN[f"log_mid_return_fwd_{k_fwd}"].replace( np.nan,0)
        # backward sum: log_mid[t-k+1] + ... + log_mid[t]
        dfN[f"log_mid_sum_bwd_{k_fwd}"] = s.rolling(window=k_fwd, min_periods=k_fwd).sum()
    
        # forward sum: log_mid[t+1] + ... + log_mid[t+k]
        # rolling is backward-looking => shift result by -(k-1)
        dfN[f"log_mid_sum_fwd_{k_fwd}"] = (
            s.shift(-1).rolling(window=k_fwd, min_periods=k_fwd).sum().shift(-(k_fwd-1))
        )
    
        bwd = dfN[f"log_mid_sum_bwd_{k_fwd}"]
        dfN[f"log_mid_sum_ratio_{k_fwd}"] = (dfN[f"log_mid_sum_fwd_{k_fwd}"] - bwd) / bwd.replace(0, np.nan)

    
        # --- classification indicator (avoid divide-by-zero) ---
        bwd = dfN[f"log_mid_sum_bwd_{k_fwd}"].to_numpy()
        fwd = dfN[f"log_mid_sum_fwd_{k_fwd}"].to_numpy()
    
        denom = np.where(np.abs(bwd) < 1e-12, np.nan, bwd)
        dfN[f"log_mid_sum_ratio_{k_fwd}"] = (fwd - bwd) / denom
        dfN[f"log_mid_sum_ratio_{k_fwd}"].replace( np.nan,0)
    # ------------------------------------------------------------------------
    
    # --- imbalance / microprice / spread features on RESAMPLED series ---
    den = (dfN["bid_sz_00"] + dfN["ask_sz_00"]).replace(0, np.nan)
    dfN["imbalance"] = dfN["bid_sz_00"] / den

    dfN["micro_price"] = dfN["imbalance"] * dfN["ask_px_00"] + (1.0 - dfN["imbalance"]) * dfN["bid_px_00"]

    dfN["spread"] = dfN["ask_px_00"] - dfN["bid_px_00"]
    dfN["rel_spread"] = dfN["spread"] / dfN["mid_price"]  # = 2*spread/(ask+bid)

    eps = 1e-12
    dfN["log_spread"] = np.log(dfN["ask_px_00"].clip(lower=eps)) - np.log(dfN["bid_px_00"].clip(lower=eps))

    dfN["micro_mid"] = dfN["micro_price"] - dfN["mid_price"]
    dfN["micro_mid_sign"] = np.sign(dfN["micro_mid"])


    # --- “mid crosses previous spread” on RESAMPLED series ---
    prev_bid = dfN["bid_px_00"].shift(1)
    prev_ask = dfN["ask_px_00"].shift(1)

    dfN["mid_cross_prev_ask_up"] = (dfN["mid_price"] > prev_ask).astype(np.int8)
    dfN["mid_cross_prev_bid_dn"] = (dfN["mid_price"] < prev_bid).astype(np.int8)

    # --- “mid jump relative to spread” (resampled) ---
    dfN["log_mid_ret"] = dfN["log_mid"].diff()
    dfN["mid_ret"]     = dfN["mid_price"].diff()
    prev_spread = dfN["spread"].shift(1)
    
    dfN["jump_gt_prev_spread"] = (dfN["mid_ret"].abs() > prev_spread).astype(np.int8)

    # Clean helper column if created
    if "_ts_event_dt" in dfN.columns:
        # keep it if you like; otherwise drop
        pass

    return dfN

#-----------END VOLUME-TIME RESAMPLING ----------------------------------------
    # --- book/state imbalance + microprice/spreads on resampled series ---


def restrict_to_rth_et(df, ts_col_utc, start, end):
    # ts_event is tz-aware UTC per Databento
    df['ts_et'] = df[ts_col_utc].dt.tz_convert("America/New_York") 

    # Apply the filter
    df  = df[(df['ts_et'].dt.time >= start) & (df['ts_et'].dt.time <= end)]

    return df

def generate_timeseries(symbol, date, tStart,tEnd, frequency, k_fwdL, W, fName, fPath):
    # frequency -resampling frequency in events
    
    print('===================================================================')
    #print('START GENERATING TRAINING DATA FOR ',feature," AT ",frequency,'sec ON ',date)
    
    # future returns aligned with horizon 𝑘 - in trhe resampling frequency;
    # i.e.k*frequency events ahead

    # Output at frequency sampling:
     #'mid_price', 
     #'log_mid', 
     #log_mid_ret', 
     #'sigma_W', 
     #'mid_changed', - at evednt level->disregard
     #'mid_change_count_n', - number of midprice chages in the block - 0 ..n
     #'move_rate_per_event', - normalized number of chages per event
     #'dt_block_sec',        -- physical seconds per block (how long did it take to arrive and process n events) - 0.0 - 500   
     #'log_dt_block',        -- log(dt_block_sec) 
     #'event_intensity',     -- events per second 
     #'log_event_intensity', 
     #'move_rate_per_sec',   -- mid-price changes per second  
     #'event_idx_in_group',
     #'sample_idx', 
     #'log_mid_return_fwd_1',  -- k-step (1 block)-ahead log return on the resampled series
     #'log_mid_return_fwd_2'
     #'log_mid_return_fwd_3'
     #'imbalance',             -- dfN["bid_sz_00"]/(dfN["bid_sz_00"] + dfN["ask_sz_00"])
     #'micro_price',           -- dfN["imbalance"] * dfN["ask_px_00"] + (1.0 - dfN["imbalance"]) * dfN["bid_px_00"]
     #'spread',                -- dfN["ask_px_00"] - dfN["bid_px_00"]
     #'rel_spread', 
     #'log_spread', 
     #'micro_mid',             -- dfN["micro_mid"] = dfN["micro_price"] - dfN["mid_price"]
     #'micro_mid_sign',        -- dfN["micro_mid_sign"] = np.sign(dfN["micro_mid"])
     #'mid_cross_prev_ask_up', -- dfN["mid_cross_prev_ask_up"] = (dfN["mid_price"] > prev_ask).astype(np.int8)
     #'mid_cross_prev_bid_dn', -- dfN["mid_cross_prev_bid_dn"] = (dfN["mid_price"] < prev_bid).astype(np.int8)
     #'mid_ret',               -- dfN["mid_ret"]     = dfN.groupby(group_col, sort=False)["mid_price"].diff()
     #'jump_gt_prev_spread'    -- dfN["jump_gt_prev_spread"] = (dfN["mid_ret"].abs() > prev_spread).astype(np.int8)
     
     # ofi_L10: Weighted deep order-flow imbalance at the event level, combining OFI from book levels 0–9 (L1–L10) 
     # ofi_L10_n: Rolling sum of ofi_L10 over the last n events (cumulative deep OFI over the recent window).
     
     # ofi_L10_norm: Depth-normalized deep OFI: ofi_L10 divided by the total bid+ask size across levels 0–9 at that event (scale-free “pressure per available depth”).
     # ofi_L10_norm_n: Rolling depth-normalized deep OFI over the last n events: (sum of ofi_L10 over the window) divided by (sum of total depth over the window). 
     
     
     
    group_col = "instrument_id"   # or "symbol"
    sort_cols=("ts_event", "sequence")  
    take = "last"  
    df = dbn_to_df(fName, fPath)
    df = df[df["symbol"] == symbol]
    # restrict to normal trading time 09:30 - 15:30
    time_stamm_column_utc="ts_event"
    df = restrict_to_rth_et(df, time_stamm_column_utc , tStart,tEnd)
    
    dfr = add_event_features_and_resample(
            df,
            frequency,  # events resampling window 
            W,          # back window for volatility estimation
            k_fwdL,
            sort_cols,
            take # "last" keeps 10th/20th/... ; "first" keeps 1st/11th/...
        ) 
    
    return dfr


#------------------------------------------------------------------------------
         
def generate_timeseries_time(date, tStart,tEnd, frequency, k_fwdL, W, fName, fPath):
    group_col = "instrument_id"   # or "symbol"
    sort_cols=("ts_event", "sequence")  
    take = "last"  
    df = dbn_to_df(fName, fPath)

    # restrict to normal trading time 09:30 - 15:30
    time_stamm_column_utc="ts_event"    
    df = restrict_to_rth_et(df, time_stamm_column_utc , tStart,tEnd)
    
     
    dfr = add_event_features_and_resample_time(
        df,
        frequency,             # bucket size in seconds
        W ,             # trailing event-vol window on raw events (optional)
        k_fwdL    ,             # forward horizon in *buckets*
        sort_cols=sort_cols)
    return dfr
#------------------------------------------------------------------------------
                               
def generate_timeseries_volume(date, tStart,tEnd, frequency, k_fwdL, W, fName, fPath):
    group_col = "instrument_id"   # or "symbol"
    sort_cols=("ts_event", "sequence")  
    take = "last"  
    df = dbn_to_df(fName, fPath)

    # restrict to normal trading time 09:30 - 15:30
    time_stamm_column_utc="ts_event"    
    df = restrict_to_rth_et(df, time_stamm_column_utc , tStart,tEnd)
    
     
    dfr = add_event_features_and_resample_volume(
        df,
        frequency,             # bucket size in seconds
        W ,             # trailing event-vol window on raw events (optional)
        k_fwdL    ,             # forward horizon in *buckets*
        sort_cols=sort_cols)
    return dfr    



#-----------------------------------------------------------------------------
def z_encoding_old(time_series, feature, n_symbols,  alpha):
    bins = fit_bins_per_alpha(time_series[feature], [alpha], n_symbols, min_periods=2)
    z, s = z_encode_ts_predictive(time_series[feature], alpha=alpha, bins=bins[alpha], min_periods=2)

    z = z.bfill()
    z = z.ffill() 
    
    s = s.bfill()
    s = s.ffill()
    
    time_series[feature+"_z"]   = z 
    time_series[feature+"_sym"] =  s 
    
    return time_series

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from statistics import NormalDist


def make_symmetric_z_bins(m: int, zmax: float = 3.0) -> np.ndarray:
    # evenly spaced thresholds in [-zmax, zmax]
    return np.linspace(-zmax, zmax, m + 1)[1:-1]

def make_normal_quantile_z_bins(m: int, clip: float = 3.0) -> np.ndarray:
    nd = NormalDist()
    qs = [i / m for i in range(1, m)]
    bins = np.array([nd.inv_cdf(q) for q in qs], dtype=float)
    return np.clip(bins, -clip, clip)

#--------------------------------------------------------------------------
# Support for predictive z-encoding - binning
def expanding_quantile_edges(
    z: pd.Series,
    n_symbols: int,
    bin_min_periods: int = 1,
) -> pd.DataFrame:
    """
    For each time t, compute quantile bin edges from z[:t-1].

    Returns a DataFrame with n_symbols-1 columns:
        edge_1, ..., edge_{n_symbols-1}
    """
    if n_symbols < 2:
        raise ValueError("n_symbols must be >= 2")

    z_hist = z.shift(1)  # important: only past z-values are used

    qs = np.arange(1, n_symbols) / n_symbols

    edges = []
    for q in qs:
        e = z_hist.expanding(min_periods=bin_min_periods).quantile(q)
        edges.append(e)

    edges = pd.concat(edges, axis=1)
    edges.columns = [f"edge_{j}" for j in range(1, n_symbols)]

    return edges


def discretize_z_timevarying_bins(
    z: pd.Series,
    edges: pd.DataFrame,
    missing_symbol: int = -1,
) -> pd.Series:
    """
    Discretize z_t using time-varying bin edges at time t.

    Symbol is number of edges less than or equal to z_t.
    Produces symbols 0, ..., n_symbols-1.
    """
    z_arr = z.to_numpy(dtype=float)
    E = edges.to_numpy(dtype=float)

    sym = np.full(len(z_arr), missing_symbol, dtype=np.int32)

    valid = np.isfinite(z_arr) & np.all(np.isfinite(E), axis=1)

    if valid.any():
        # count how many thresholds z_t crosses
        sym[valid] = np.sum(z_arr[valid, None] >= E[valid], axis=1).astype(np.int32)

    return pd.Series(sym, index=z.index)

# -------------------------------------------------------------------------
"""
bins_mode options
-----------------

1) bins_mode="daily_quantile"

   Fits symbol thresholds from the empirical quantiles of the predictive
   z-scores over the entire provided series.

   Interpretation:
       Offline diagnostic encoding.

   Leakage status:
       Not strictly predictive if the same interval is used for evaluation,
       because the bin edges use future z-values.

   Typical use:
       Exploratory MI/TE diagnostics, full-day distribution studies.


2) bins_mode="prefix_quantile"

   Fits symbol thresholds from the empirical quantiles of the predictive
   z-scores over the first fit_frac fraction of the provided series.

   Interpretation:
       The initial prefix is treated as a calibration / burn-in region.

   Leakage status:
       Predictive only after the prefix. The prefix itself should not be
       treated as out-of-sample, because bin edges are fit using the full
       prefix.

   Typical use:
       Single-session experiments where the beginning of the day is used
       to calibrate symbolic bins before forecasting later periods.


3) bins_mode="history_quantile"

   Fits symbol thresholds from all observations with timestamp <= fit_until.
   The timestamp is taken either from time_col or from the DataFrame index.

   Interpretation:
       Uses all historically available data before the prediction-start time.

   Leakage status:
       Predictive for observations after fit_until, provided no later data
       are included in the calibration set.

   Typical use:
       Production-style or walk-forward experiments:
           previous sessions + current observed prefix -> fit bins
           future prediction interval                  -> encode/evaluate


4) bins_mode="fixed_uniform"

   Uses fixed z-score thresholds evenly spaced in [-zmax, zmax].

   Example:
       n_symbols=8, zmax=3 gives thresholds roughly
       [-2.25, -1.50, -0.75, 0.00, 0.75, 1.50, 2.25].

   Interpretation:
       Symbols represent fixed deviations from the predictive EWMA mean
       in units of predictive EWMA standard deviation.

   Leakage status:
       Fully predictive. No bin fitting is performed.

   Typical use:
       Strict online encoding, robust cross-day comparison, symbolic
       generative modeling where interpretability is preferred over balanced
       symbol frequencies.


5) bins_mode="fixed_normal"

   Uses fixed thresholds given by standard-normal quantiles, optionally
   clipped to [-zmax, zmax].

   Interpretation:
       Similar to fixed_uniform, but bins are closer to equal-probability
       under an approximately Gaussian z-score distribution.

   Leakage status:
       Fully predictive. No bin fitting is performed.

   Typical use:
       Default recommended fixed-bin scheme when one wants no lookahead but
       better occupancy balance than uniformly spaced z-thresholds.
"""
# 
# -------------------------------------------------------------------------
def z_encoding(
    time_series: pd.DataFrame,
    feature: str,
    n_symbols: int,
    alpha: float,
    # optional params
    min_periods: int = 2,
    bins_mode: str = "expanding_quantile",
    fit_frac: float = 0.5,
    fit_until=None,
    time_col: str | None = None,
    zmax: float = 3.0,
    fill_mode: str = "ffill",
    missing_symbol: int = -1,
    bin_min_periods: int = 1,   # new: for expanding_quantile
) -> pd.DataFrame:

    x = time_series[feature]

    # predictive z-score: z_t uses only x_{<t}
    z = ewma_zscore_predictive(x, alpha=alpha, min_periods=min_periods).bfill()
    #z.isna().sum()
    # ------------------------------------------------------------------
    # fixed-bin cases
    # ------------------------------------------------------------------
    if bins_mode == "daily_quantile":
        bins = fit_quantile_bins(z, n_symbols)
        s = discretize_z(z, bins=bins)

    elif bins_mode == "prefix_quantile":
        T = max(int(len(z) * fit_frac), 1)
        bins = fit_quantile_bins(z.iloc[:T], n_symbols)
        s = discretize_z(z, bins=bins)

    elif bins_mode == "history_quantile":
        if fit_until is None:
            raise ValueError("fit_until must be provided for bins_mode='history_quantile'.")

        if time_col is None:
            ts = time_series.index
        else:
            ts = pd.to_datetime(time_series[time_col])

        fit_mask = ts <= fit_until

        if fit_mask.sum() < n_symbols:
            raise ValueError("Not enough observations before fit_until.")

        bins = fit_quantile_bins(z.loc[fit_mask], n_symbols)
        s = discretize_z(z, bins=bins)

    elif bins_mode == "fixed_uniform":
        bins = make_symmetric_z_bins(n_symbols, zmax=zmax)
        s = discretize_z(z, bins=bins)

    elif bins_mode == "fixed_normal":
        bins = make_normal_quantile_z_bins(n_symbols, clip=zmax)
        s = discretize_z(z, bins=bins)

    # ------------------------------------------------------------------
    # fully online case: each t uses bins fitted on z_{<t}
    # ------------------------------------------------------------------
    elif bins_mode == "expanding_quantile":
        edges = expanding_quantile_edges(
            z,
            n_symbols=n_symbols,
            bin_min_periods=bin_min_periods,
        )
        s = discretize_z_timevarying_bins(
            z,
            edges,
            missing_symbol=missing_symbol,
        )

    else:
        raise ValueError(
            "bins_mode must be one of: "
            "daily_quantile, prefix_quantile, history_quantile, "
            "fixed_uniform, fixed_normal, expanding_quantile"
        )

    # ------------------------------------------------------------------
    # fill handling
    # ------------------------------------------------------------------
    if fill_mode == "ffill":
        z = z.ffill()
        s = s.ffill().fillna(missing_symbol).astype(np.int32)

    elif fill_mode == "bfill_ffill":
        # non-causal; use only for offline diagnostics
        z = z.bfill().ffill()
        s = s.bfill().ffill().astype(np.int32)

    elif fill_mode == "bfill":
        # non-causal; use only for offline diagnostics
        # s = s.replace(missing_symbol, np.nan).bfill().astype("int")
        s = s.replace(missing_symbol, np.nan).bfill().ffill().astype("int")
    elif fill_mode == "none":
        s = s.where(z.notna(), other=missing_symbol).astype("int")

    else:
        raise ValueError("fill_mode must be: ffill, bfill_ffill, none")

    if s.iloc[0]==missing_symbol: #bfill just the first symbol
        s.iloc[0] = s.iloc[1]
        
    time_series[feature + "_z"] = z
    time_series[feature + "_sym"] = s

    return time_series

#------------------------------------------------------------------------------
#             Statistics of One time series

def X_statistics(X, feature, frequency, units,  lagmax):
    lagmax = 20
    
    correlations = calculate_correlation(X, lagmax)
    # 2. Extract keys and values
    keys = list(correlations.keys())
    values = list(correlations.values())
    plt.title('Autocorrelation '+ feature+' Frequency '+str(frequency)+' '+units)
    plt.plot(keys, values, marker='o', linestyle='-')
    plt.show()
    
    mi_value = calculate_mutual_information(X, lagmax)
    keys = list(mi_value.keys())
    values = list(mi_value.values())
    plt.title('MI '+ feature+' Frequency '+str(frequency)+' '+units)
    plt.plot(keys, values, marker='o', linestyle='-')
    plt.show()
    
    te_vals = calculate_transfer_entropy(X, lagmax)
    keys = list(te_vals.keys())
    values = list(te_vals.values())
    plt.title('TE '+ feature+' Frequency '+str(frequency)+' '+units)
    plt.plot(keys, values, marker='o', linestyle='-')
    plt.show()

#------------------------------------------------------------------------------
# Retrive data frame with real-valued time series with ALL features
#------------------------------------------------------------------------------
def get_timeseries_by_date(symbol,fPath, date,resampling,frequency, frw_intervals, tStart,tEnd ):

    fName = 'xnas-itch-'+date+'.mbp-10.dbn.zst'
    
    if resampling == 'events':
        print('Timeseries in events ',frequency,'events')# seconds
        freq_units='evn'
        W = 3*frequency # backward window to calculate volatility
        time_series = generate_timeseries(symbol, date, tStart,tEnd, frequency,  frw_intervals, W, fName, fPath)
    elif resampling == 'seconds':
        print('Timeseries in seconds ',frequency,'sec')# seconds
        freq_units = 'sec'
        W = 3*frequency # backward window to calculate volatility
        time_series = generate_timeseries_time(date, tStart,tEnd, frequency,frw_intervals, W, fName, fPath)
    elif resampling == 'volume':
        print('Timeseries in shares ',frequency,'shr')# seconds
        freq_units = 'shr'
        W = 3*frequency # backward window to calculate volatility
        time_series = generate_timeseries_volume(date, tStart, tEnd, frequency,frw_intervals, W, fName, fPath)            
    return time_series



#------------------------------------------------------------------------------
def distribution_by_date(symbol,fPath, dates,resampling,frequency, variate, features, alpha, n_symbols,max_seq_length, frw_intervals):
    # "normal" trading time
    tStart = datetime.time(9, 30)
    tEnd   = datetime.time(15, 30)    

    distributions_dates = {}
    for date in dates:
        fName = 'xnas-itch-'+date+'.mbp-10.dbn.zst'
        
        if resampling == 'events':
            print('Timeseries in events ',frequency,'events')# seconds
            freq_units='evn'
            W = 3*frequency # backward window to calculate volatility
            time_series = generate_timeseries(date, tStart,tEnd, frequency,  frw_intervals, W, fName, fPath)
        elif resampling == 'seconds':
            print('Timeseries in seconds ',frequency,'sec')# seconds
            freq_units = 'sec'
            W = 3*frequency # backward window to calculate volatility
            time_series = generate_timeseries_time(date, tStart,tEnd, frequency,frw_intervals, W, fName, fPath)
        elif resampling == 'volume':
            print('Timeseries in shares ',frequency,'shr')# seconds
            freq_units = 'shr'
            W = 3*frequency # backward window to calculate volatility
            time_series = generate_timeseries_volume(date, tStart, tEnd, frequency,frw_intervals, W, fName, fPath)            
            
            
            
        #----------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        # generate z-encoded training data
        #------------------------------------------------------------------------------
        
        # here all the features of interest are z-encoded  
        if variate == "univariate":  
            #------------------------------------------------------------------------------
            #              z-encoding - extend the frame with _sym (z) of each time sereis
            generate_list=[]
            feature = features[0]
            # bins_mode: str = "daily_quantile",   # "daily_quantile" | "prefix_quantile" | "fixed_uniform" | "fixed_normal"   
            time_series = z_encoding(time_series, feature, n_symbols,  alpha, bins_mode = "expanding_quantile")

            feature = feature+'_sym'            
            list_series = time_series[feature][5:]
            all_subsequences, counts = estimate_subsequence_counts(list_series, max_seq_length)
                
            #samples = [''.join(map(str, t)) for sublist in all_subsequences for t in sublist]
            counts= [[np.array(t[0]).astype(int).tolist() ,t[1],t[2]] for sublist in counts for t in sublist]
            distrs = [[c[0], c[1]/c[2]] for c in counts]
            samples = [d[0] for d in distrs]
            
            distributions_dates[date] =  [distrs, samples ]
        #------------------------------------------------------------------------------
        # generate z-encoded BI-variate training data
        #------------------------------------------------------------------------------
        if variate == "bivariate":
            #------------------------------------------------------------------------------
            #              z-encoding for BI Variate
            # Encoding with 4 symbols per variate
            n_symbols = 4
            feature = features[0]
            predictor = feature
            predicted = features[1]

            time_series_2z = z_encoding(time_series, predictor, n_symbols,  alpha)
            time_series_2z = z_encoding(time_series, predicted, n_symbols,  alpha)

            predicted_series = time_series_2z[predicted]
            predictor_series = time_series_2z[predictor]
            
            #------------------------------------------------------------------
            z_series12 = pd.concat([predicted_series, predictor_series], axis=1)
            ni = np.asarray([n_symbols,n_symbols])
            weights = np.concatenate(([1], np.cumprod(ni[:-1])))
            z_series12 = (z_series12* weights).sum(axis=1)
              
            
            bi_series = list( z_series12.astype(int))
            
            
            #all_subsequences1, counts1 = estimate_subsequence_counts(bi_series, max_seq_length)
            
            
            all_subsequences, counts = estimate_observed_subsequence_counts(
                seq=bi_series,
                max_subsequence_length=max_seq_length,
                sample_size=1 , # 0.75,
                sample_after_length=30,
                random_state=42,
                sort="lexicographic",
                include_prob=True,
            ) 
           
            samples = [list(t) for sublist in all_subsequences for t in sublist]
            # samples = [''.join(map(str, t)) for sublist in all_subsequences for t in sublist]
            #counts= [[list(t[0]) ,t[1],t[2]] for sublist in counts for t in sublist]
            counts=  [[np.array(t[0]).astype(int).tolist() ,t[1],t[2]] for sublist in counts for t in sublist]
            distrs = [[c[0], c[1]/c[2]] for c in counts]
            samples = [d[0] for d in distrs]
            
            distributions_dates[date] =  [distrs, samples]
    return time_series, distributions_dates
#------------------------------------------------------------------------------
# Process distribution for particular date - givebn the time series
#------------------------------------------------------------------------------
def get_distribution_by_ts(time_series, variate, predicted, predictor, alpha, n_symbols,max_seq_length):
        #----------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        # generate z-encoded training data
        #------------------------------------------------------------------------------
        
        # all the features of interest will be z-encoded  
        if variate == "univariate":  
            #------------------------------------------------------------------------------
            #              z-encoding - extend the name with _sym (z) of each time sereis
            #              univariate: - use the first feature
            feature = predicted    
            # bins_mode: str = "daily_quantile",   # "daily_quantile" | "prefix_quantile" | "fixed_uniform" | "fixed_normal"   
            time_series_z = z_encoding(time_series, feature  , n_symbols,  alpha, bins_mode = "expanding_quantile")
         
            feature = feature+'_sym'            
            series_z = list(time_series_z[feature].astype(int))   
            
            all_subsequences, counts = estimate_observed_subsequence_counts(
                seq=series_z,
                max_subsequence_length=max_seq_length,
                sample_size=1 , # 0.75,
                sample_after_length=30,
                random_state=42,
                sort="lexicographic",
                include_prob=True,
            )
            return all_subsequences, counts, time_series_z[feature].astype(int)
            
        #------------------------------------------------------------------------------
        # generate z-encoded BI-variate training data
        #------------------------------------------------------------------------------
        if variate == "bivariate":
            #------------------------------------------------------------------------------
            #              z-encoding for BI Variate
            # Encoding with 4 symbols per variate
             
            time_series_2z = z_encoding(time_series, predictor, n_symbols,  alpha, fill_mode = "bfill", bins_mode = "expanding_quantile")
            time_series_2z = z_encoding(time_series, predicted, n_symbols,  alpha, fill_mode = "bfill", bins_mode = "expanding_quantile")

            predicted_series = time_series_2z[predicted+'_sym']
            predictor_series = time_series_2z[predictor+'_sym']
            
            #------------------------------------------------------------------
            z_series12 = pd.concat([predicted_series, predictor_series], axis=1)
            ni = np.asarray([n_symbols,n_symbols])
            weights = np.concatenate(([1], np.cumprod(ni[:-1])))
            z_series12 = (z_series12* weights).sum(axis=1)
              
            
            bi_series = list( z_series12.astype(int))
            
            
            #all_subsequences1, counts1 = estimate_subsequence_counts(bi_series, max_seq_length)
            
            
            all_subsequences, counts = estimate_observed_subsequence_counts(
                seq=bi_series,
                max_subsequence_length=max_seq_length,
                sample_size=1 , # 0.75,
                sample_after_length=30,
                random_state=42,
                sort="lexicographic",
                include_prob=True,
            ) 
            
        if variate == "multivariate":
            #------------------------------------------------------------------------------
            #              z-encoding for BI Variate
            # Encoding with 4 symbols per variate
            
            time_series_z = time_series.copy()
            
            
            if isinstance(predictor, str):
                predictor_list = [predictor]
            else:
                predictor_list = list(predictor)

            
            # All variables to z-encode.
            variables = [predicted] + predictor_list

            for var in variables:
                time_series_z = z_encoding(
                    time_series_z,
                    var,
                    n_symbols,
                    alpha,
                    fill_mode="bfill",
                    bins_mode="expanding_quantile",
                )


            sym_cols = [
                var + "_sym"
                for var in variables
            ]
        
            # Shape: [T, 1 + n_predictors]
            Z = time_series_z[sym_cols].astype(np.int64).to_numpy()
            
        
            # Mixed-radix weights. If all components have the same alphabet size:
            weights = n_symbols ** np.arange(len(sym_cols), dtype=np.int64)
        
            # Joint multivariate symbol.
            z_joint = (Z * weights).sum(axis=1).astype(np.int64)


            # Preserve old interface/name.
            z_series12 = pd.Series(
                z_joint,
                index=time_series_z.index,
                name="joint_sym",
            )
            
            '''
            all_subsequences, counts = estimate_observed_subsequence_counts(
                seq=z_joint,
                max_subsequence_length=max_seq_length,
                sample_size=1 , # 0.75,
                sample_after_length=30,
                random_state=42,
                sort="lexicographic",
                include_prob=True,
            )    
            '''
            
            all_subsequences, counts = estimate_observed_subsequence_counts(
                seq=list(z_series12.astype(int)),
                max_subsequence_length=max_seq_length,
                sample_size=1,
                sample_after_length=30,
                random_state=42,
                sort="lexicographic",
                include_prob=True,
            )
            
#------------------------------------------------------------------------------
        return all_subsequences, counts, z_series12.astype(int)
#--------------------------------------------------------------------------------
def get_bivariate_ts(time_series, predicted, predictor, alpha, n_symbols):
        #------------------------------------------------------------------------------
        #              z-encoding for BI Variate
        # Encoding with 4 symbols per variate
         
        time_series_2z = z_encoding(time_series, predictor, n_symbols,  alpha, fill_mode = "bfill", bins_mode = "expanding_quantile")
        time_series_2z = z_encoding(time_series, predicted, n_symbols,  alpha, fill_mode = "bfill", bins_mode = "expanding_quantile")

        predicted_series = time_series_2z[predicted+'_sym']
        predictor_series = time_series_2z[predictor+'_sym']
        
        #------------------------------------------------------------------
        z_series12 = pd.concat([predicted_series, predictor_series], axis=1)
        ni = np.asarray([n_symbols,n_symbols])
        weights = np.concatenate(([1], np.cumprod(ni[:-1])))
        z_series12 = (z_series12* weights).sum(axis=1)
          
        
        # bi_series = list( z_series12.astype(int))
        
        
        return z_series12.astype(int)

def get_variate_ts(time_series, predicted, predictor, alpha, n_symbols):

        time_series_z = time_series.copy()
        
        
        if isinstance(predictor, str):
            predictor_list = [predictor]
        else:
            predictor_list = list(predictor)

        
        # All variables to z-encode.
        variables = [predicted] + predictor_list

        for var in variables:
            time_series_z = z_encoding(
                time_series_z,
                var,
                n_symbols,
                alpha,
                fill_mode="bfill",
                bins_mode="expanding_quantile",
            )


        sym_cols = [
            var + "_sym"
            for var in variables
        ]
    
        # Shape: [T, 1 + n_predictors]
        Z = time_series_z[sym_cols].astype(np.int64).to_numpy()
        
    
        # Mixed-radix weights. If all components have the same alphabet size:
        weights = n_symbols ** np.arange(len(sym_cols), dtype=np.int64)
    
        # Joint multivariate symbol.
        z_joint = (Z * weights).sum(axis=1).astype(np.int64)


        # Preserve old interface/name.
        z_series12 = pd.Series(
            z_joint,
            index=time_series_z.index,
            name="joint_sym",
        )
         
        
        return z_series12.astype(int)






#--------------------------------------------------------------------------------
def filter_length(dist, len_list, min_prob=0.0):
    out = []
    for i in dist:
        if len(i[0]) in len_list and i[1] > min_prob:
            out.append(i)
    return out

#------------------------------------------------------------------------------
# End of distribution_by_date
#------------------------------------------------------------------------------
def plot_real_valued_features(symbol,dates,fPath,feature_list, resampling,frequency,frw_intervals ,tStart,tEnd  ): 
    for d in dates:     
        time_series = get_timeseries_by_date(symbol,fPath, d, resampling,frequency,frw_intervals ,tStart,tEnd )
    
        for f in feature_list:
            plt_ts(time_series[f], shw = True, title=f+'@'+dates[0], label='', color='blue', xlabel= 'x'+str(frequency)+' '+ resampling, ylabel=f)

#------------------------------------------------------------------------------
#==============================================================================
def compute_global_weights(sequences, seq_probs):
    length_counts = {}
    for seq in sequences:
        length = len(seq)
        length_counts[length] = length_counts.get(length, 0) + 1
        
    total_seqs = len(sequences)
    p_length = {l: count / total_seqs for l, count in length_counts.items()}
    
    
    weights = []
    
    for seq, prb in zip(sequences, seq_probs):
        length = len(seq)
        # Global probability: p(seq) = p(len) · p(seq|len)
        weight = p_length[length] * prb
        weights.append(weight)
    
    # Normalize to sum to 1
    total = sum(weights)
    weights = [w / total for w in weights]
    
    return weights
#==============================================================================

symbol= 'AAPL'
symbol= 'INTC'
symbol= 'NVDA'
symbol= 'AAPL'

alpha = 0.05  # z-encoding EWMA parameter

if symbol == 'AAPL':
    fPath = 'C:\\EXPIMP\\Vanio\\Projects\\Market Data Preparation\\Data'
    fPath = 'C:\\V\\Projects\\Market Data Preparation\\Data'
    fPath = 'C:\\Vanio\\Projects\\Market Data Preparation\\Data'
    
elif symbol in ['INTC','NVDA']:
    fPath = 'C:\\V\\Projects\\Market Data Preparation\\Data\\INTC_NVIDIA'
    fPath = 'C:\\Vanio\\Projects\\Market Data Preparation\\Data\\INTC_NVIDIA'
     
# "normal" eastern trading time
tStart = datetime.time(9, 30)
tEnd   = datetime.time(15, 30)
resampling= 'events' # "volume",  "seconds" 
frequency = 100      # events or 1000 shares or 10s 
frw_intervals = [1,2,3,4] # steps ahead price to be added to the time sereis

#------------------------------------------------------------------------------
# Encoder  generative model - Training Distributions
#------------------------------------------------------------------------------

variate  = "multivariate"
#variate  =  "univariate"
#variate  = "bivariate"

if variate  == "multivariate":
    predictor =  ['micro_price', 'vpin', 'ofi_L3_norm_n']
    predictor_name ="micro_vpin_ofi_l3" 



if variate=="bivariate":
    n_symbols=4       # observable symbols in one time series/feture
    alphabet = list(range(n_symbols**2))
elif variate =="univariate":
    n_symbols=8      # observable symbols in one time series/feture
    alphabet = list(range(n_symbols))
elif variate =="multivariate":
    n_symbols=4    # observable symbols in one time series/feture
    alphabet = list(range(n_symbols**len(predictor)))

    
max_seq_length=4
max_subsequence_length=max_seq_length


#features     = ['log_mid',"tvi_n" , 'obi_L1', "ofi_L1_n", "ofi_L1_n_norm",'ofi_L1_norm_n','ofi_L3_norm_n','ofi_L10_norm_n',"micro_price",'vpin', 'sigma_W' ]
features     = ['log_mid','micro_price',"tvi_n", 'sigma_W', 'vpin','ofi_L1_norm_n','ofi_L3_norm_n','ofi_L10_norm_n', 'log_spread','imbalance']
features     = ['log_mid','micro_price']

#-----------------------------------------------------------------------------
# Training dates - March, April 2025
#-----------------------------------------------------------------------------

price_pred_classes =  ['c2','c4', 'ca2','ca4']
vpin_pred_classes  =  ['cvp4', 'cvp5', 'cvp10']
clsNames =  price_pred_classes + vpin_pred_classes

clsNames = ['ca4']
predicted = features[0]

seq_prob_weight_file = 'SQ_PRB_WT_'
seq_prob_file = 'SQ_PRB_'
cls_dist_file = 'CLS_DISTR_'


dates_may = []

 # generate training or validaton data

training_data = False                       # generate validation  data             
training_data = True                        # generate training data

validation_data = not training_data
validation_days = 3

if validation_data:
    dates_april = ['20250401', '20250402', '20250403', '20250404','20250407', '20250408', '20250409', '20250410',
             '20250411', '20250414', '20250415', '20250416','20250417', '20250421', '20250422', '20250423',
             '20250424', '20250425', '20250428', '20250429','20250430'
             ]
    dates = dates_april[:validation_days]

    save_sequence_distributions       = True
    save_seq_prob_weight              = False
    daily_sequence_distributions      = True
    monthly_sequence_distributions    = False
    
    daily_class_distributions = True
    save_daily_class          = True
    
    monthly_class_calculation = False            # calculate class distribution by sub-sequence 
    save_class_distributions  = False

elif training_data:
    dates_march = ['20250303','20250304','20250305','20250306','20250307','20250310','20250311','20250312','20250313','20250314',
                   '20250317','20250318','20250319','20250320','20250321','20250324','20250325','20250326','20250327','20250328','20250331']
   
    
    dates = dates_march                     # daily data  to be used
    monthly_Date = dates[0][:6]             #monthly
    
    save_sequence_distributions       = True
    save_seq_prob_weight              = True
    daily_sequence_distributions      = False
    monthly_sequence_distributions    = True
    seq_prob_weight_file = 'SQ_PRB_WT_'
    seq_prob_file = 'SQ_PRB_'
    
    daily_class_distributions = False
    save_daily_class          = False
    
    monthly_class_calculation = True            # calculate class distribution by sub-sequence 
    save_class_distributions = True
    
#-----------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Real Valued Features - univariate - VISUALIZING ONLY
#------------------------------------------------------------------------------
plot_ind_real_valued = False
if plot_ind_real_valued:
    ftrs = features[:1] 
    '''
    plot_real_valued_features(symbol,['20250303'],fPath,ftrs, resampling,frequency,frw_intervals ,tStart,tEnd )
    plot_real_valued_features(symbol,['20250310'],fPath,ftrs, resampling,frequency,frw_intervals ,tStart,tEnd )
    plot_real_valued_features(symbol,['20250317'],fPath,ftrs, resampling,frequency,frw_intervals ,tStart,tEnd )
    '''
    t_list = []
    y_list = []
    from_file = True
    outfname = "three_days_price"
    for f in ftrs:
        if not from_file:
            ts1 = get_timeseries_by_date(symbol,fPath, '20250303', resampling,frequency,frw_intervals ,tStart,tEnd )
            ts2 = get_timeseries_by_date(symbol,fPath, '20250310', resampling,frequency,frw_intervals ,tStart,tEnd )
            ts3 = get_timeseries_by_date(symbol,fPath, '20250317', resampling,frequency,frw_intervals ,tStart,tEnd )       
            pickle.dump( [ts1, ts2 , ts3   ], open("three_days_price", "wb") )
        else:
           ts = pickle.load(open("three_days_price", "rb"))    
        
        t_list = [ts[0][f].index,  ts[1][f].index,  ts[2][f].index]
        y_list = [ts[0][f].values, ts[1][f].values, ts[2][f].values]

    labels= ['03/03/2025','03/10/2025','03/17/2025']
    
    fig, ax=plot_three_rebased_prices(
        t_list,
        y_list,
        labels,
        xlabel="Time",
        ylabel=r"$\Delta \log(\mathrm{MP})$",
        title=None,
        figsize=(3.35, 2.1),
        lw=1.2,
        save_as=None,
    )
    plt.show()
    filename = 'Three Days Log Mid Price'
    save_paper_figure(fig, filename)
    sys.exit()

calculated = {}
saved = {}
for prd in features[1:]:
    calculated[prd] = False
    saved[prd]      = False 
    

for clsName in clsNames:
    if clsName in price_pred_classes:
        num_classes=3

    if clsName in vpin_pred_classes:
        num_classes=2
        
    print('Start for class=',clsName, ' Predicted =', predicted)
    
    if variate == 'multivariate':
        
        #----------------------------------------------------------------------
        C = []  #classes distribution list by days
        L = [] # sequences distributions list by days
        i = 0
        print('Start for Multi Variate Predictor', predictor, ' Predicted =', predicted, 'Class =',clsName)
        for date in dates:
            print('---------',date,'-------------')
            time_series = get_timeseries_by_date(symbol,fPath, date, resampling,frequency,frw_intervals ,tStart,tEnd )

            #--------------------------------------------------------------------------------------------------------------------
            if (monthly_sequence_distributions  or daily_sequence_distributions) :
                all_subsequences, counts, z_series12 = get_distribution_by_ts(time_series, variate, predicted, predictor, alpha, n_symbols,max_seq_length)
                L.append(counts)
                if i == 0:
                    firstcounts = counts                                         # distribitions at the first date
                allcounts = integrate_distributions(L, max_seq_length, alphabet) # aggregated distribution during the month
                L = [allcounts]
                
                if daily_sequence_distributions:
                    sequences = [s for subsequence in all_subsequences for s in subsequence]
                    seq_probs = [i[3] for c in counts for i in c]
                    fName = seq_prob_file+symbol+'_'+predicted+'-'+predictor+'_'+date 
                    pickle.dump( [sequences, seq_probs ], open( fName , "wb") )
                    print('Dumped ',fName)
                    saved[predictor] =True

            if (monthly_class_calculation or daily_class_distributions):     # class distributions for the day
                 
                z_series12 = get_variate_ts  (time_series, predicted, predictor, alpha, n_symbols)
                
                
                if clsName in price_pred_classes:
                    cls_ts = add_class_label(time_series, clsName)
                    class_values=(-1, 0, 1)
                if clsName in vpin_pred_classes:
                    # adding z-encoding for vpin: vpin_sym
                    time_series = z_encoding(time_series, "vpin", n_symbols=4,  alpha=alpha, fill_mode = "bfill" , bins_mode = "expanding_quantile") 
                    cls_ts = add_vpin_class_label(time_series, clsName)
                    class_values=(0, 1)                          

                cl_distributions=estimate_subsequence_class_probabilities(z_series12, cls_ts, max_subsequence_length= max_subsequence_length, class_values=class_values)
                C.append(cl_distributions[1])
                all_cls_distr =integrate_conditional_class_distributions(C,  max_len=max_seq_length,  n_classes=num_classes, alphabet=alphabet) # aggregated class distribution during the month

                C=[all_cls_distr]
                
                if save_daily_class:
                    fName = cls_dist_file+symbol+'_'+predicted+'-'+predictor+'_'+ clsName+'_'+date 
                    
                    cls_distr = [[i[0],i[2]] for s in cl_distributions[1] for i in s]
                    pickle.dump( cls_distr, open( fName , "wb") )
                    print('Dumped ',fName)        
            i = i+1

        if monthly_sequence_distributions:
            # distributions and samples for the month - these will be used for training
            cntsall=  [[np.array(t[0]).astype(int).tolist() ,t[1],t[2]] for sublist in allcounts for t in sublist]
            distrsall = [[c[0], c[1]/c[2]] for c in cntsall]
            samplesall = [s[0] for s in cntsall]
            
            # Save monthly                    
            if save_sequence_distributions and not saved[predictor]:                   # saving monthly aggregated distributions for training
                outfname = seq_prob_file+symbol+'_' +predicted+'-'+predictor+'_'+monthly_Date    
                pickle.dump( [distrsall, samplesall ], open( outfname, "wb") )
                print('Dumped ',outfname)
                saved[predictor] =True

       
            if monthly_class_calculation and save_class_distributions:
                outfname = cls_dist_file+symbol+'_' + predicted+'-'+predictor+'_'+monthly_Date+'_'+ clsName
                cls_distr = [item for sublist in all_cls_distr for item in sublist]
                pickle.dump( cls_distr, open( outfname, "wb") )
                print('Dumped ',outfname)


        #----------------------------------------------------------------------

            #--------------------------------------------------------------------------------------------------------------------

        if monthly_sequence_distributions :
            # distributions and samples for the month - these will be used for training
            cntsall=  [[np.array(t[0]).astype(int).tolist() ,t[1],t[2]] for sublist in allcounts for t in sublist]
            distrsall = [[c[0], c[1]/c[2]] for c in cntsall]
            samplesall = [s[0] for s in cntsall]
            
            # distributions and samples for the last date of the month - for analysis/comparison
            cntslast=  [[np.array(t[0]).astype(int).tolist() ,t[1],t[2]] for sublist in counts for t in sublist]
            distrslast = [[c[0], c[1]/c[2]] for c in cntslast]
            sampleslast = [s[0] for s in cntslast]
            
            # distributions and samples for the first date of the month - for analysis/comparison
            cntsfirst =  [[np.array(t[0]).astype(int).tolist() ,t[1],t[2]] for sublist in firstcounts for t in sublist]
            distrsfirst = [[c[0], c[1]/c[2]] for c in cntsfirst]
            samplesfirst = [s[0] for s in cntsfirst]
            
            plot = False  #plot comaprison of first day, lst day, average monthly distributions
            distributions = [distrsall, distrsfirst, distrslast]
            if plot:
                plots = []
                seq_lens = [1,2]
                for d in distributions:
                    plots.append(filter_length(d,seq_lens, 1e-06))
                
                colors=[ "blue", "orange",'green']
                names = [dates[0][:6],dates[0], dates[-1]]  
                fig, ax, selected, P = plot_distributions_comparison_preserve_order(
                    distributions=plots,
                    names=names,
                    colors=colors,
                    reference_index=0,   # preserve order from first distribution
                    top_n=len(plots[0]), #  100,            # first 100 entries exactly as they appear in input
                    figsize=(18, 6),
                    bar_group_width=0.7, #0.72,
                    title=predictor+ " predicts "+predicted,
                    xlabel="Sequences",
                    ylabel="Probability",
                    rotation=90,
                    edgecolor = "white", # for bars
                    linewidth = 0.2,       # for bars
                    alpha = 0.9
                )
                plt.show() 
        
        
        if monthly_sequence_distributions  and save_sequence_distributions:                               # saving monthly aggregated distributions for training
            outfname = seq_prob_file + symbol+'_' +predicted+'-'+predictor_name+'_'+ monthly_Date    
            pickle.dump( [distrsall, samplesall ], open( outfname, "wb") )
            print('Dumped ',outfname)
            saved[predictor] =True
    elif variate == 'bivariate': # univariate or bivariate
    
        for predictor in features[1:]:
            C = []  #classes distribution list by days
            L = [] # sequences distributions list by days
            i = 0
            print('Start for Predictor', predictor, ' Predicted =', predicted, 'Class =',clsName)
            for date in dates:
                print('---------',date,'-------------')
                time_series = get_timeseries_by_date(symbol,fPath, date, resampling,frequency,frw_intervals ,tStart,tEnd )
                #--------------------------------------------------------------------------------------------------------------------
                if (monthly_sequence_distributions  or daily_sequence_distributions) and not calculated[predictor]:
                    all_subsequences, counts, z_series12 = get_distribution_by_ts(time_series, variate, predicted, predictor, alpha, n_symbols,max_seq_length)
                    L.append(counts)
                    if i == 0:
                        firstcounts = counts                                         # distribitions at the first date
                    allcounts = integrate_distributions(L, max_seq_length, alphabet) # aggregated distribution during the month
                    L = [allcounts]
                    
                    if daily_sequence_distributions:
                        sequences = [s for subsequence in all_subsequences for s in subsequence]
                        seq_probs = [i[3] for c in counts for i in c]
                        fName = seq_prob_file+symbol+'_'+predicted+'-'+predictor+'_'+date 
                        pickle.dump( [sequences, seq_probs ], open( fName , "wb") )
                        print('Dumped ',fName)
                        saved[predictor] =True
                # End sequence distributions for the day      
                        
                if (monthly_class_calculation or daily_class_distributions):     # class distributions for the day
                    z_series12 = get_bivariate_ts(time_series, predicted, predictor, alpha, n_symbols)
                    
                    if clsName in price_pred_classes:
                        cls_ts = add_class_label(time_series, clsName)
                        class_values=(-1, 0, 1)
                    if clsName in vpin_pred_classes:
                        # adding z-encoding for vpin: vpin_sym
                        time_series = z_encoding(time_series, "vpin", n_symbols=4,  alpha=alpha, fill_mode = "bfill" , bins_mode = "expanding_quantile") 
                        cls_ts = add_vpin_class_label(time_series, clsName)
                        class_values=(0, 1)                          

                    cl_distributions=estimate_subsequence_class_probabilities(z_series12, cls_ts, max_subsequence_length= max_subsequence_length, class_values=class_values)
                    C.append(cl_distributions[1])
                    all_cls_distr =integrate_conditional_class_distributions(C,  max_len=max_seq_length,  n_classes=num_classes, alphabet=alphabet) # aggregated class distribution during the month
                        
                    '''          
                        time_series = z_encoding(time_series, "vpin", n_symbols=4,  alpha=0.05, fill_mode = "bfill" , bins_mode = "expanding_quantile") 
                        cvp4 = add_vpin_class_label(time_series, "cvp4")
                        vpin_distributions=estimate_subsequence_class_probabilities(z_series12, cvp4, max_subsequence_length= max_subsequence_length, class_values=(0, 1))set()
                        add_vpin_class_label(time_series, "cvp5")
                        add_vpin_class_label(time_series, "cvp10")
                    '''
                    
                    C=[all_cls_distr]
                    
                    if save_daily_class:
                        fName = cls_dist_file+symbol+'_'+predicted+'-'+predictor+'_'+ clsName+'_'+date 
                        
                        cls_distr = [[i[0],i[2]] for s in cl_distributions[1] for i in s]
                        pickle.dump( cls_distr, open( fName , "wb") )
                        print('Dumped ',fName)

                   
                    
                i = i+1                      
            calculated[predictor] = True
        
            if monthly_sequence_distributions:
                # distributions and samples for the month - these will be used for training
                cntsall=  [[np.array(t[0]).astype(int).tolist() ,t[1],t[2]] for sublist in allcounts for t in sublist]
                distrsall = [[c[0], c[1]/c[2]] for c in cntsall]
                samplesall = [s[0] for s in cntsall]
                
                # distributions and samples for the last date of the month - for analysis/comparison
                # cntslast=  [[np.array(t[0]).astype(int).tolist() ,t[1],t[2]] for sublist in counts for t in sublist]
                # distrslast = [[c[0], c[1]/c[2]] for c in cntslast]
                # sampleslast = [s[0] for s in cntslast]
                
                # distributions and samples for the first date of the month - for analysis/comparison
                # cntsfirst =  [[np.array(t[0]).astype(int).tolist() ,t[1],t[2]] for sublist in firstcounts for t in sublist]
                # distrsfirst = [[c[0], c[1]/c[2]] for c in cntsfirst]
                # samplesfirst = [s[0] for s in cntsfirst]
                
                plot = False  #plot comaprison of first day, lst day, average monthly distributions
                #distributions = [distrsall, distrsfirst, distrslast]
                if plot:
                    plots = []
                    seq_lens = [1,2]
                    for d in distributions:
                        plots.append(filter_length(d,seq_lens, 1e-06))
                    
                    colors=[ "blue", "orange",'green']
                    names = [dates[0][:6],dates[0], dates[-1]]  
                    fig, ax, selected, P = plot_distributions_comparison_preserve_order(
                        distributions=plots,
                        names=names,
                        colors=colors,
                        reference_index=0,   # preserve order from first distribution
                        top_n=len(plots[0]), #  100,            # first 100 entries exactly as they appear in input
                        figsize=(18, 6),
                        bar_group_width=0.7, #0.72,
                        title=predictor+ " predicts "+predicted,
                        xlabel="Sequences",
                        ylabel="Probability",
                        rotation=90,
                        edgecolor = "white", # for bars
                        linewidth = 0.2,       # for bars
                        alpha = 0.9
                    )
                    plt.show()
                    
                # Save monthly                    
                if save_sequence_distributions and not saved[predictor]:                   # saving monthly aggregated distributions for training
                    outfname = seq_prob_file+symbol+'_' +predicted+'-'+predictor+'_'+monthly_Date    
                    pickle.dump( [distrsall, samplesall ], open( outfname, "wb") )
                    print('Dumped ',outfname)
                    saved[predictor] =True
    
           
                if monthly_class_calculation and save_class_distributions:
                    outfname = cls_dist_file+symbol+'_' + predicted+'-'+predictor+'_'+monthly_Date+'_'+ clsName
                    cls_distr = [item for sublist in all_cls_distr for item in sublist]
                    pickle.dump( cls_distr, open( outfname, "wb") )
                    print('Dumped ',outfname)

    elif variate == 'univariate':                    
         
        C = []  #classes distribution list by days
        L = [] # sequences distributions list by days
        i = 0
        print('Start for Asset=', predicted, 'Class =',clsName)
        for date in dates:
            print('---------',date,'-------------')
            time_series = get_timeseries_by_date(symbol,fPath, date, resampling,frequency,frw_intervals ,tStart,tEnd )
            #--------------------------------------------------------------------------------------------------------------------
            if monthly_sequence_distributions :
                all_subsequences, counts, z_series12 = get_distribution_by_ts(time_series, variate, predicted, predicted, alpha, n_symbols,max_seq_length)

                #-------------------------------------------------------------------------------------------------------------------- 
                if save_seq_prob_weight:
                    fName = seq_prob_weight_file+symbol+'_'+date
                    sequences = [s for subsequence in all_subsequences for s in subsequence]
                    seq_probs = [i[3] for c in counts for i in c]
                    global_weights = compute_global_weights(sequences, seq_probs)
                    
                    pickle.dump([sequences,seq_probs,global_weights], open( fName, "wb") )
                    print('Dumped ',fName)
                #--------------------------------------------------------------------------------------------------------------------
                       
                    
                L.append(counts)
                if i == 0:
                    firstcounts = counts                                         # distribitions at the first date
                allcounts = integrate_distributions(L, max_seq_length, alphabet) # aggregated distribution during the month
                L = [allcounts]
        
                
            if (monthly_class_calculation or daily_class_distributions):
                z_series12 = get_bivariate_ts(time_series, predicted, predictor, alpha, n_symbols)
                

                if clsName in price_pred_classes:
                    cls_ts = add_class_label(time_series, clsName)
                    class_values=(-1, 0, 1)
                    
                if clsName in vpin_pred_classes:
                    # adding z-encoding for vpin: vpin_sym
                    time_series = z_encoding(time_series, "vpin", n_symbols=4,  alpha=alpha, fill_mode = "bfill" , bins_mode = "expanding_quantile") 
                    cls_ts = add_vpin_class_label(time_series, clsName)
                    class_values=( 0, 1)


                cl_distributions=estimate_subsequence_class_probabilities(z_series12, cls_ts, max_subsequence_length= max_subsequence_length, class_values=class_values)
                C.append(cl_distributions[1])
                all_cls_distr =integrate_conditional_class_distributions(C,  max_len=max_seq_length,  n_classes=num_classes, alphabet=alphabet) # aggregated class distribution during the month
                               
                C=[all_cls_distr]
            i = i+1                      
        calculated[predictor] = True
    
        if monthly_sequence_distributions :
            # distributions and samples for the month - these will be used for training
            cntsall=  [[np.array(t[0]).astype(int).tolist() ,t[1],t[2]] for sublist in allcounts for t in sublist]
            distrsall = [[c[0], c[1]/c[2]] for c in cntsall]
            samplesall = [s[0] for s in cntsall]
            
            # distributions and samples for the last date of the month - for analysis/comparison
            cntslast=  [[np.array(t[0]).astype(int).tolist() ,t[1],t[2]] for sublist in counts for t in sublist]
            distrslast = [[c[0], c[1]/c[2]] for c in cntslast]
            sampleslast = [s[0] for s in cntslast]
            
            # distributions and samples for the first date of the month - for analysis/comparison
            cntsfirst =  [[np.array(t[0]).astype(int).tolist() ,t[1],t[2]] for sublist in firstcounts for t in sublist]
            distrsfirst = [[c[0], c[1]/c[2]] for c in cntsfirst]
            samplesfirst = [s[0] for s in cntsfirst]
            
            plot = False  #plot comaprison of first day, lst day, average monthly distributions
            #distributions = [distrsall, distrsfirst, distrslast]
            if plot:
                plots = []
                seq_lens = [1,2]
                for d in distributions:
                    plots.append(filter_length(d,seq_lens, 1e-06))
                
                colors=[ "blue", "orange",'green']
                names = [dates[0][:6],dates[0], dates[-1]]  
                fig, ax, selected, P = plot_distributions_comparison_preserve_order(
                    distributions=plots,
                    names=names,
                    colors=colors,
                    reference_index=0,   # preserve order from first distribution
                    top_n=len(plots[0]), #  100,            # first 100 entries exactly as they appear in input
                    figsize=(18, 6),
                    bar_group_width=0.7, #0.72,
                    title=predictor+ " predicts "+predicted,
                    xlabel="Sequences",
                    ylabel="Probability",
                    rotation=90,
                    edgecolor = "white", # for bars
                    linewidth = 0.2,       # for bars
                    alpha = 0.9
                )
                plt.show()             
    
        
        
        
        
#==========================================================================================================
# Processing Monthly Aggregation
#==========================================================================================================
        
            if monthly_sequence_distributions  and save_sequence_distributions and not saved[predictor]:                   # saving monthly aggregated distributions for training
                outfname = seq_prob_file+symbol+'_' + predicted+'-'+predictor+'_'+monthly_Date    
                pickle.dump( [distrsall, samplesall ], open( outfname, "wb") )
                print('Dumped ',outfname)
                saved[predictor] =True

       
            if monthly_class_calculation and save_class_distributions:
                outfname = cls_dist_file+symbol+'_'+predicted+'-'+predictor+'_'+ clsName+'_'+monthly_Date 
                cls_distr = [item for sublist in all_cls_distr for item in sublist]
                pickle.dump( cls_distr, open( outfname, "wb") )
                print('Dumped ',outfname)


sys.exit()

