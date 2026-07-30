# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:51:27 2026

@author: vanio
"""

from read_databento_new import dbn_to_df, X_pred_Y, estimate_subsequence_counts
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
from time_series_analysis import calculate_correlation, calculate_mutual_information,  calculate_transfer_entropy
from time_series_analysis import compute_mi_discrete 
from plot_distributions import plotDistribution
import datetime
import pickle


#-----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def plot_grouped_spectra_bars(
    all_spectra,
    kernel_order=None,
    title="",
    top_k=20,
    normalize_display="as_is",   # "as_is" or "top_k"
    show_cumulative=True,
    figsize=(14, 6),
    rotation=0,
    logy=False,
):
    """
    Plot multiple spectra on one grouped bar chart.

    Parameters
    ----------
    all_spectra : dict[str, array-like]
        Dictionary: kernel_name -> 1D normalized spectrum array (descending preferred).
    kernel_order : list[str] or None
        Order of kernels in the plot. If None, uses dict keys order.
    title : str
        Plot title, e.g. your Title variable.
    top_k : int
        Number of leading eigenvalues to show.
    normalize_display : str
        - "as_is": keep original normalization
        - "top_k": renormalize only the displayed top_k part to sum to 1
    show_cumulative : bool
        Whether to append captured mass in the legend.
    figsize : tuple
        Figure size.
    rotation : int
        Rotation for x tick labels.
    logy : bool
        Whether to use log scale on y-axis.
    """
    if kernel_order is None:
        kernel_order = list(all_spectra.keys())

    spectra = {}
    captured_mass = {}

    for name in kernel_order:
        lam = np.asarray(all_spectra[name], dtype=float).reshape(-1)
        lam = np.sort(lam)[::-1]
        lam = np.clip(lam, 0.0, None)

        k = min(top_k, len(lam))
        head = lam[:k].copy()

        captured_mass[name] = head.sum()

        if normalize_display == "top_k":
            s = head.sum()
            if s > 0:
                head = head / s
        elif normalize_display != "as_is":
            raise ValueError("normalize_display must be 'as_is' or 'top_k'.")

        # pad shorter spectra with zeros so all have same displayed length
        if k < top_k:
            head = np.pad(head, (0, top_k - k), mode="constant")

        spectra[name] = head

    n_kernels = len(kernel_order)
    x = np.arange(top_k)

    # bar width adapted to number of spectra
    width = 0.8 / max(n_kernels, 1)

    plt.figure(figsize=figsize)

    for i, name in enumerate(kernel_order):
        offset = (i - (n_kernels - 1) / 2) * width
        label = name
        if show_cumulative and normalize_display == "as_is":
            label = f"{name} (Σ₁:{top_k}={captured_mass[name]:.3f})"
        plt.bar(x + offset, spectra[name], width=width, label=label, alpha=0.9)

    plt.xlabel("Eigenvalue index")
    if normalize_display == "top_k":
        plt.ylabel(f"Top-{top_k} renormalized eigenvalue")
    else:
        plt.ylabel("Eigenvalue")
    plot_title = f"Spectral comparison: {title}"
    if normalize_display == "top_k":
        plot_title += f"  (top-{top_k} renormalized)"
    plt.title(plot_title)

    plt.xticks(x, [str(i + 1) for i in range(top_k)], rotation=rotation)

    if logy:
        plt.yscale("log")

    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
#------------------------------------------------------------------------------
def plot_tail_mass(
    all_spectra,
    kernel_order=None,
    title="",
    threshold=None,
    logy=True,
    figsize=(10, 6),
):
    """
    Plot tail cumulative mass T(r) = sum_{i>=r} lambda_i.
    Optionally threshold tiny values to zero before computing the tail.
    """
    if kernel_order is None:
        kernel_order = list(all_spectra.keys())

    plt.figure(figsize=figsize)

    for name in kernel_order:
        lam = np.asarray(all_spectra[name], dtype=float).reshape(-1)
        lam = np.sort(lam)[::-1]
        lam = np.clip(lam, 0.0, None)

        if threshold is not None:
            lam = lam.copy()
            lam[lam < threshold] = 0.0

        tail = np.cumsum(lam[::-1])[::-1]   # tail[r] = sum_{i>=r} lam_i
        x = np.arange(1, len(lam) + 1)
        plt.plot(x, tail, linewidth=2, label=name)

    plt.xlabel("Eigenvalue index")
    plt.ylabel("Tail cumulative mass")
    plt.title(f"Tail spectral mass: {title}")
    if logy:
        plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------
def plot_binned_spectra0(
    all_spectra,
    kernel_order=None,
    title="",
    bin_size=10,
    mode="sum",              # "sum", "mean", "last"
    normalize_bins=False,    # if True, renormalize displayed bins to sum to 1
    logy=False,
    figsize=(14, 6),
):
    """
    Plot spectra aggregated in bins of size `bin_size`.

    Parameters
    ----------
    all_spectra : dict[str, array-like]
        kernel_name -> normalized spectrum
    kernel_order : list[str] or None
        plotting order
    title : str
        plot title
    bin_size : int
        number of eigenvalues per bin
    mode : str
        "sum", "mean", or "last"
    normalize_bins : bool
        whether to renormalize the displayed bin values to sum to 1
    logy : bool
        log scale on y-axis
    figsize : tuple
        figure size
    """
    if kernel_order is None:
        kernel_order = list(all_spectra.keys())

    plt.figure(figsize=figsize)

    for name in kernel_order:
        lam = np.asarray(all_spectra[name], dtype=float).reshape(-1)
        lam = np.sort(lam)[::-1]
        lam = np.clip(lam, 0.0, None)

        n_bins = int(np.ceil(len(lam) / bin_size))
        vals = []

        for b in range(n_bins):
            block = lam[b * bin_size:(b + 1) * bin_size]
            if len(block) == 0:
                continue
            if mode == "sum":
                vals.append(block.sum())
            elif mode == "mean":
                vals.append(block.mean())
            elif mode == "last":
                vals.append(block[-1])
            else:
                raise ValueError("mode must be 'sum', 'mean', or 'last'")

        vals = np.asarray(vals, dtype=float)

        if normalize_bins and vals.sum() > 0:
            vals = vals / vals.sum()

        x = np.arange(1, len(vals) + 1)
        plt.plot(x, vals, marker="o", linewidth=2, label=name)

    plt.xlabel(f"Eigenvalue block index (block size = {bin_size})")
    if mode == "sum":
        ylabel = "Block spectral mass"
    elif mode == "mean":
        ylabel = "Mean eigenvalue in block"
    else:
        ylabel = "Last eigenvalue in block"
    if normalize_bins:
        ylabel += " (renormalized)"
    plt.ylabel(ylabel)

    plt.title(f"Binned spectral comparison: {title}")
    if logy:
        plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_binned_spectra(
    all_spectra,
    kernel_order=None,
    title="",
    bin_size=10,
    max_groups=None,         # <-- new
    mode="sum",              # "sum", "mean", "last"
    normalize_bins=False,
    logy=False,
    figsize=(14, 6),
):
    if kernel_order is None:
        kernel_order = list(all_spectra.keys())

    plt.figure(figsize=figsize)

    for name in kernel_order:
        lam = np.asarray(all_spectra[name], dtype=float).reshape(-1)
        lam = np.sort(lam)[::-1]
        lam = np.clip(lam, 0.0, None)

        n_bins = int(np.ceil(len(lam) / bin_size))
        vals = []

        for b in range(n_bins):
            block = lam[b * bin_size:(b + 1) * bin_size]
            if len(block) == 0:
                continue
            if mode == "sum":
                vals.append(block.sum())
            elif mode == "mean":
                vals.append(block.mean())
            elif mode == "last":
                vals.append(block[-1])
            else:
                raise ValueError("mode must be 'sum', 'mean', or 'last'")

        vals = np.asarray(vals, dtype=float)

        if max_groups is not None:
            vals = vals[:max_groups]

        if normalize_bins and vals.sum() > 0:
            vals = vals / vals.sum()

        x = np.arange(1, len(vals) + 1)
        plt.plot(x, vals, marker="o", linewidth=2, label=name)

    plt.xlabel(f"Eigenvalue block index (block size = {bin_size})")
    if mode == "sum":
        ylabel = "Block spectral mass"
    elif mode == "mean":
        ylabel = "Mean eigenvalue in block"
    else:
        ylabel = "Last eigenvalue in block"
    if normalize_bins:
        ylabel += " (renormalized)"
    plt.ylabel(ylabel)

    ttl = f"Binned spectral comparison: {title}"
    if max_groups is not None:
        ttl += f" (first {max_groups} groups)"
    plt.title(ttl)

    if logy:
        plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
#------------------------------------------------------------------------------
# Analysis
# -----------------------------
# Spectrum summary utilities
# -----------------------------


def participation_ratio(lams, eps=1e-15):
    """
    Participation ratio:
        PR = (sum lams)^2 / sum(lams^2)
    For normalized lams summing to 1, PR = 1 / sum(lams^2).
    """
    lams = np.asarray(lams, dtype=float).reshape(-1)
    lams = np.clip(lams, 0.0, None)
    s1 = lams.sum()
    s2 = np.sum(lams**2)
    if s1 <= eps or s2 <= eps:
        return 0.0
    return (s1 * s1) / s2


def entropy_effective_rank(lams, eps=1e-15):
    """
    Entropy effective rank:
        r_ent = exp( - sum_i p_i log p_i )
    where p_i = lams_i / sum(lams).
    """
    lams = np.asarray(lams, dtype=float).reshape(-1)
    lams = np.clip(lams, 0.0, None)
    s = lams.sum()
    if s <= eps:
        return 0.0
    p = lams / s
    p = p[p > eps]
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def effective_dimension(lams, gamma, eps=1e-15):
    """
    Kernel effective dimension at scale gamma:
        d_eff(gamma) = sum_i lambda_i / (lambda_i + gamma)
    gamma can be a scalar or array-like.
    """
    lams = np.asarray(lams, dtype=float).reshape(-1)
    lams = np.clip(lams, 0.0, None)

    gamma = np.asarray(gamma, dtype=float)
    if np.any(gamma <= 0):
        raise ValueError("gamma must be positive.")

    if gamma.ndim == 0:
        return float(np.sum(lams / (lams + gamma + eps)))

    return np.array([np.sum(lams / (lams + g + eps)) for g in gamma], dtype=float)


def spectrum_summary(lams, gammas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), eps=1e-15):
    """
    Convenience summary for one spectrum.
    """
    lams = np.asarray(lams, dtype=float).reshape(-1)
    lams = np.clip(lams, 0.0, None)

    out = {
        "trace": float(lams.sum()),
        "rank_numerical": int(np.sum(lams > eps)),
        "participation_ratio": participation_ratio(lams, eps=eps),
        "entropy_effective_rank": entropy_effective_rank(lams, eps=eps),
        "effective_dimension": {float(g): float(effective_dimension(lams, g, eps=eps)) for g in gammas},
    }
    return out

#-----------------------------------------------------------------------------


def compare_summaries(all_spectra, gammas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), eps=1e-15):
    """
    all_spectra: dict[name] -> spectrum array
    returns: dict[name] -> summary dict
    """
    return {
        name: spectrum_summary(lams, gammas=gammas, eps=eps)
        for name, lams in all_spectra.items()
    }


# -----------------------------
# Pretty print + plot
# -----------------------------
def print_summaries(summary_dict, kernel_order=None, digits=4):
    """
    Nicely print scalar spectral quality measures.
    """
    if kernel_order is None:
        kernel_order = list(summary_dict.keys())

    header = (
        f"{'Kernel':<20}"
        f"{'Trace':>12}"
        f"{'Rank':>10}"
        f"{'PR':>12}"
        f"{'EntropyRank':>16}"
    )
    print(header)
    print("-" * len(header))

    for name in kernel_order:
        s = summary_dict[name]
        print(
            f"{name:<20}"
            f"{s['trace']:>12.{digits}f}"
            f"{s['rank_numerical']:>10d}"
            f"{s['participation_ratio']:>12.{digits}f}"
            f"{s['entropy_effective_rank']:>16.{digits}f}"
        )

def plot_effective_dimensions(
    summary_dict,
    kernel_order=None,
    title="",
    figsize=(10, 6),
    logx=True,
    logy=False,
    marker="o",
):
    """
    Plot effective dimension curves d_eff(gamma) for all kernels on one figure.
    """
    if kernel_order is None:
        kernel_order = list(summary_dict.keys())

    plt.figure(figsize=figsize)

    for name in kernel_order:
        edict = summary_dict[name]["effective_dimension"]
        gammas = np.array(sorted(edict.keys()), dtype=float)
        vals = np.array([edict[g] for g in gammas], dtype=float)

        plt.plot(gammas, vals, marker=marker, linewidth=2, label=name)

    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$d_{\mathrm{eff}}(\gamma)$")
    plt.title("Kernel effective dimension" + (f": {title}" if title else ""))
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


#------------------------------------------------------------------------------
