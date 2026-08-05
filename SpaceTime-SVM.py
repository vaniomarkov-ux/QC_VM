# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import annotations
import pickle
import random
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional, Any
from joblib import parallel_backend, Parallel, delayed 
#-----------------------------------------------------------------------------
# Quanrtum SpaceTime Model
#------------------------------------------------------------------------------
def _to_complex_vector(
    x,
    dtype=np.complex64,
) -> np.ndarray:
    """
    Convert a PyTorch tensor or array-like object to a flat
    one-dimensional complex NumPy array.
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    return np.asarray(
        x,
        dtype=dtype,
    ).reshape(-1)


# ---------------------------
# Model: Kraus via stacked-isometry whitening
# ---------------------------
class KrausInstrument(nn.Module):
    def __init__(self, m: int, d: int, learn_rho0: bool = True, eps: float = 1e-8):
        super().__init__()
        self.m, self.d, self.eps = m, d, eps

        # Unconstrained complex A in R^{2} via (real, imag)
        # Shape: (m*d, d)
        self.A_re = nn.Parameter(torch.randn(m * d, d) * 0.01)
        self.A_im = nn.Parameter(torch.randn(m * d, d) * 0.01)

        # Initial state rho0: either fixed to |0><0| or learnable PSD/trace-1
        self.learn_rho0 = learn_rho0
        if learn_rho0:
            # Learn rho0 via Cholesky-like factor L, rho = L L† / Tr(...)
            self.L_re = nn.Parameter(torch.randn(d, d) * 0.01)
            self.L_im = nn.Parameter(torch.randn(d, d) * 0.01)

    def _make_rho0(self, device):
        d = self.d
        if not self.learn_rho0:
            # |0><0|
            rho0 = torch.zeros(d, d, dtype=torch.complex64, device=device)
            rho0[0, 0] = 1.0 + 0.0j
            return rho0
        else:
            L = torch.complex(self.L_re, self.L_im).to(device)
            rho = L @ L.conj().T
            tr = torch.real(torch.trace(rho)) + self.eps
            rho = rho / tr
            # Optional: ensure Hermitian numerically
            rho = 0.5 * (rho + rho.conj().T)
            return rho

    def kraus_operators(self):
        """
        Returns K: complex tensor [m, d, d] satisfying sum_y K_y† K_y = I
        """
        d, m, eps = self.d, self.m, self.eps
        A = torch.complex(self.A_re, self.A_im)  # [(m*d), d]

        # Compute G = A† A  [d, d]
        G = A.conj().T @ A
        # Make sure Hermitian (numerical)
        G = 0.5 * (G + G.conj().T)

        # Eigen-decomp for inverse sqrt: G = Q diag(w) Q†
        w, Q = torch.linalg.eigh(G)  # w: [d], Q: [d, d]
        w = torch.clamp(w, min=eps)
        inv_sqrt = (Q * (w.rsqrt())) @ Q.conj().T  # Q diag(w^-1/2) Q†

        V = A @ inv_sqrt  # [(m*d), d] with V†V = I

        K = V.reshape(m, d, d)  # [m, d, d]
        return K

    @torch.no_grad()
    def check_cptp(self):
        K = self.kraus_operators()
        S = torch.zeros(self.d, self.d, dtype=K.dtype, device=K.device)
        for y in range(self.m):
            S = S + K[y].conj().T @ K[y]
        return torch.max(torch.abs(S - torch.eye(self.d, dtype=S.dtype, device=S.device))).item()

    def sequence_prob_batch(self, seq_batch):
        """
        seq_batch: LongTensor [B, T]
        returns: FloatTensor [B] model probabilities
        """
        device = seq_batch.device
        B, T = seq_batch.shape
        d = self.d

        K = self.kraus_operators()  # [m, d, d]
        rho0 = self._make_rho0(device)  # [d, d]

        # Batch rho: [B, d, d]
        rho = rho0.unsqueeze(0).expand(B, d, d).clone()

        # Iterate over time; each step is batched matmuls
        PAD = -1  # same PAD as in collate_fn
        
        for t in range(T):
            sym = seq_batch[:, t]               # [B], dtype long
            active = (sym != PAD)             # [B]
            if not torch.any(active):
                break
        
            sym_a = sym[active]               # [B_active]
            rho_a = rho[active]               # [B_active, d, d]
        
            # IMPORTANT: sym_a must be LongTensor and non-negative
            sym_a = sym_a.long()
        
            Kt = K.index_select(0, sym_a)     # [B_active, d, d]
            rho_a = torch.bmm(torch.bmm(Kt, rho_a), Kt.conj().transpose(1, 2))
        
            rho[active] = rho_a


        # prob = Tr(rho), real part
        prob = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1))
        # numerical clamp
        prob = torch.clamp(prob, min=0.0)
        return prob
    @torch.inference_mode()
    def path_operator(self, seq, device=None, return_prob=False, eps=1e-12):
        if device is None:
            device = next(self.parameters()).device
        self.eval()

        # seq -> tensor
        if not torch.is_tensor(seq):
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
        else:
            seq_t = seq.to(device)

        K = self.kraus_operators()        # [m,d,d] complex64
        rho0 = self._make_rho0(device)    # [d,d]  complex64
        d = self.d

        # K_seq = K[a_T] ... K[a_1]
        Kseq = torch.eye(d, dtype=K.dtype, device=device)
        for s in seq_t.tolist():
            Kseq = K[s] @ Kseq

        if not return_prob:
            return (Kseq.detach().cpu().numpy(),
                    rho0.detach().cpu().numpy())

        # p(seq) = Tr(Kseq rho0 Kseq†)
        rhoT = Kseq @ rho0 @ Kseq.conj().transpose(-2, -1)
        p = torch.real(torch.trace(rhoT)).clamp_min(0.0)

        return (Kseq.detach().cpu().numpy(),
                rho0.detach().cpu().numpy(),
                float(p.detach().cpu().item()))
# ---------------------------
#------------------------------------------------------------------------------
# Compute the representation factors
#------------------------------------------------------------------------------
def compute_pi_factors(
    model,
    sequences,
    nqbts: int,
    *,
    normalize: bool = True,
    eps: float = 1e-12,
    n_jobs: int = 1,
    description: str = "Extracting Pi factors",
    show_progress: bool = True,
):
    """
    Return U and W such that

        Pi(sequence_i) = |u_i><w_i|.
    """

    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "model"):
        model.model.eval()

    def compute_one(sequence):
        with torch.inference_mode():
            u, w = RepUV(
                model,
                list(sequence),
                nqbts,
            )

        u = _to_complex_vector(u)
        w = _to_complex_vector(w)

        if normalize:
            norm_u = np.linalg.norm(u)
            norm_w = np.linalg.norm(w)

            if norm_u <= eps or norm_w <= eps:
                raise ValueError(
                    f"Zero-norm Pi factor for sequence {sequence}."
                )

            u = u / norm_u
            w = w / norm_w

        return u, w

    if n_jobs == 1:
        iterator = tqdm(
            sequences,
            total=len(sequences),
            desc=description,
            unit="sequence",
            disable=not show_progress,
        )

        results = [
            compute_one(sequence)
            for sequence in iterator
        ]

    else:
        results = Parallel(
            n_jobs=n_jobs,
            backend="threading",
            verbose=5 if show_progress else 0,
        )(
            delayed(compute_one)(sequence)
            for sequence in sequences
        )

    U = np.stack(
        [result[0] for result in results],
        axis=0,
    )

    W = np.stack(
        [result[1] for result in results],
        axis=0,
    )

    return U, W
def compute_pi_factors_old(
    model,
    sequences,
    nqbts: int,
    *,
    normalize: bool = True,
    eps: float = 1e-12,
    n_jobs: int = 1,
):
    """
    Return matrices

        U[i] = u(sequence_i)
        W[i] = w(sequence_i)

    Shapes
    ------
    U : (N, dim_u)
    W : (N, dim_w)
    """

    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "model"):
        model.model.eval()

    def compute_one(sequence):
        with torch.inference_mode():
            # Modify only this line if RepUV has a different signature.
            u, w = RepUV(
                model,
                list(sequence),
                nqbts,
            )

        u = _to_complex_vector(u)
        w = _to_complex_vector(w)

        if normalize:
            norm_u = np.linalg.norm(u)
            norm_w = np.linalg.norm(w)

            if norm_u <= eps or norm_w <= eps:
                raise ValueError(
                    f"Zero-norm Pi factor for sequence {sequence}."
                )

            u = u / norm_u
            w = w / norm_w

        return u, w

    if n_jobs == 1:
        results = [
            compute_one(sequence)
            for sequence in sequences
        ]
    else:
        # Threads avoid the Windows loky/gmpy2 process problem.
        results = Parallel(
            n_jobs=n_jobs,
            backend="threading",
        )(
            delayed(compute_one)(sequence)
            for sequence in sequences
        )

    U = np.stack(
        [result[0] for result in results],
        axis=0,
    )

    W = np.stack(
        [result[1] for result in results],
        axis=0,
    )

    return U, W
#------------------------------------------------------------------------------
# Construct valid SVM kernels
#------------------------------------------------------------------------------

def pi_kernel_from_factors(
    U_left: np.ndarray,
    W_left: np.ndarray,
    U_right: np.ndarray,
    W_right: np.ndarray,
    *,
    kernel_kind: str = "hs_real",
) -> np.ndarray:
    """
    Construct a Pi kernel without explicitly building Pi.

    Parameters
    ----------
    kernel_kind:
        "hs_real"
            Re Tr(Pi_i^† Pi_j)

        "hs_fidelity"
            |Tr(Pi_i^† Pi_j)|^2
    """

    U_left = np.asarray(
        U_left,
        dtype=np.complex128,
    )

    W_left = np.asarray(
        W_left,
        dtype=np.complex128,
    )

    U_right = np.asarray(
        U_right,
        dtype=np.complex128,
    )

    W_right = np.asarray(
        W_right,
        dtype=np.complex128,
    )

    # <u_i | u_j>
    overlap_u = (
        U_left.conj()
        @ U_right.T
    )

    # <w_j | w_i>
    overlap_w_reverse = (
        W_left
        @ W_right.conj().T
    )

    overlap_pi = (
        overlap_u
        * overlap_w_reverse
    )

    if kernel_kind == "hs_real":
        K = np.real(overlap_pi)

    elif kernel_kind == "hs_fidelity":
        K = np.abs(overlap_pi) ** 2

    else:
        raise ValueError(
            "kernel_kind must be 'hs_real' "
            "or 'hs_fidelity'."
        )

    return np.asarray(
        K,
        dtype=np.float64,
    )
#------------------------------------------------------------------------------
# SVM training with a precomputed kernel
#------------------------------------------------------------------------------
def train_validate_precomputed_svm(
    K_train: np.ndarray,
    y_train: np.ndarray,
    train_target_probabilities: np.ndarray,
    K_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_target_probabilities: np.ndarray,
    *,
    C_grid=(
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
    ),
    class_weight="balanced",
    sample_weight_mode=None,
    cv_splits: int = 5,
    seed: int = 1234,
    n_jobs: int = 1,
    cache_size_mb: float = 1024.0,
):
    """
    Train and validate an SVC using precomputed Gram matrices.

    Required shapes
    ---------------
    K_train : (N_train, N_train)
    K_valid : (N_valid, N_train)
    """

    K_train = np.asarray(
        K_train,
        dtype=np.float64,
    )

    K_valid = np.asarray(
        K_valid,
        dtype=np.float64,
    )

    y_train = np.asarray(
        y_train,
        dtype=np.int64,
    )

    y_valid = np.asarray(
        y_valid,
        dtype=np.int64,
    )

    n_train = len(y_train)
    n_valid = len(y_valid)

    if K_train.shape != (
        n_train,
        n_train,
    ):
        raise ValueError(
            "K_train must have shape "
            f"({n_train}, {n_train}), "
            f"received {K_train.shape}."
        )

    if K_valid.shape != (
        n_valid,
        n_train,
    ):
        raise ValueError(
            "K_valid must have shape "
            f"({n_valid}, {n_train}), "
            f"received {K_valid.shape}."
        )

    if np.any(~np.isfinite(K_train)):
        raise ValueError(
            "K_train contains NaN or infinity."
        )

    if np.any(~np.isfinite(K_valid)):
        raise ValueError(
            "K_valid contains NaN or infinity."
        )

    # Remove small numerical asymmetry.
    K_train = 0.5 * (
        K_train + K_train.T
    )

    unique_classes, class_counts = np.unique(
        y_train,
        return_counts=True,
    )

    if len(unique_classes) < 2:
        raise ValueError(
            "Training data contain fewer than two classes."
        )

    effective_cv_splits = min(
        int(cv_splits),
        int(class_counts.min()),
    )

    if effective_cv_splits < 2:
        raise ValueError(
            "At least two samples per class are required."
        )

    cv = StratifiedKFold(
        n_splits=effective_cv_splits,
        shuffle=True,
        random_state=seed,
    )

    svm = SVC(
        kernel="precomputed",
        class_weight=class_weight,
        decision_function_shape="ovr",
        cache_size=float(cache_size_mb),
    )

    n_candidates = len(C_grid)
    n_cv_fits = n_candidates * effective_cv_splits
    
    print(
        f"\nSVM grid search:"
        f"\n  candidates = {n_candidates}"
        f"\n  CV folds   = {effective_cv_splits}"
        f"\n  total fits = {n_cv_fits}"
        f"\n  workers    = {n_jobs}",
        flush=True,
    )
    
    search = GridSearchCV(
        estimator=svm,
        param_grid={
            "C": list(C_grid),
        },
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
        },
        refit="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
    
        # We already evaluate the final selected model on the
        # complete training set. Avoid recalculating train scores
        # inside every CV fold.
        return_train_score=False,
    
        error_score="raise",
    
        # 2: completion and duration of each fit
        # 3: additionally shows fold and score
        verbose=3,
    )

    '''
    search = GridSearchCV(
        estimator=svm,
        param_grid={
            "C": list(C_grid),
        },
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
        },
        refit="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
        return_train_score=True,
        error_score="raise",
        verbose=1,
    )
    '''
    sample_weights = make_training_sample_weights(
        train_target_probabilities,
        sample_weight_mode,
    )

    fit_arguments = {}

    if sample_weights is not None:
        fit_arguments["sample_weight"] = sample_weights

    start_time = time.perf_counter()

    if n_jobs == 1:
        search.fit(
            K_train,
            y_train,
            **fit_arguments,
        )
    else:
        # Explicit threading avoids Windows child-process failures.
        with parallel_backend(
            "threading",
            n_jobs=n_jobs,
        ):
            search.fit(
                K_train,
                y_train,
                **fit_arguments,
            )

    fit_seconds = (
        time.perf_counter()
        - start_time
    )

    best_model = search.best_estimator_

    y_train_pred = best_model.predict(
        K_train
    )

    y_valid_pred = best_model.predict(
        K_valid
    )

    train_metrics = evaluate_classification(
        y_train,
        y_train_pred,
        train_target_probabilities,
    )

    valid_metrics = evaluate_classification(
        y_valid,
        y_valid_pred,
        valid_target_probabilities,
    )

    best_index = int(
        search.best_index_
    )

    cv_results = search.cv_results_

    return {
        "model": best_model,
        "search": search,
        "best_params": search.best_params_,
        "best_cv_f1_macro": float(
            search.best_score_
        ),
        "best_cv_f1_macro_std": float(
            cv_results[
                "std_test_f1_macro"
            ][best_index]
        ),
        "best_cv_accuracy": float(
            cv_results[
                "mean_test_accuracy"
            ][best_index]
        ),
        "best_cv_balanced_accuracy": float(
            cv_results[
                "mean_test_balanced_accuracy"
            ][best_index]
        ),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "y_train_pred": y_train_pred,
        "y_valid_pred": y_valid_pred,
        "n_train": n_train,
        "n_valid": n_valid,
        "cv_splits": effective_cv_splits,
        "fit_seconds": fit_seconds,
        "sample_weight_mode": sample_weight_mode,
        "standardize": False,
    }
def pi_kernel_from_factors_blocked(
    U_left: np.ndarray,
    W_left: np.ndarray,
    U_right: np.ndarray,
    W_right: np.ndarray,
    *,
    kernel_kind: str = "hs_fidelity",
    block_size: int = 256,
    dtype=np.float64,
    description: str = "Constructing Pi kernel",
    show_progress: bool = True,
) -> np.ndarray:
    """
    Construct the Pi Gram matrix in row blocks.

    Output shape:
        (len(U_left), len(U_right))
    """
    U_left = np.asarray(
        U_left,
        dtype=np.complex128,
    )
    W_left = np.asarray(
        W_left,
        dtype=np.complex128,
    )
    U_right = np.asarray(
        U_right,
        dtype=np.complex128,
    )
    W_right = np.asarray(
        W_right,
        dtype=np.complex128,
    )

    n_left = U_left.shape[0]
    n_right = U_right.shape[0]

    K = np.empty(
        (n_left, n_right),
        dtype=dtype,
    )

    block_starts = range(
        0,
        n_left,
        block_size,
    )

    iterator = tqdm(
        block_starts,
        total=(n_left + block_size - 1) // block_size,
        desc=description,
        unit="block",
        disable=not show_progress,
    )

    for start in iterator:
        stop = min(
            start + block_size,
            n_left,
        )

        # <u_i | u_j>
        overlap_u = (
            U_left[start:stop].conj()
            @ U_right.T
        )

        # <w_j | w_i>
        overlap_w = (
            W_left[start:stop]
            @ W_right.conj().T
        )

        overlap_pi = overlap_u * overlap_w

        if kernel_kind == "hs_real":
            K[start:stop] = np.real(
                overlap_pi
            )

        elif kernel_kind == "hs_fidelity":
            K[start:stop] = (
                np.abs(overlap_pi) ** 2
            )

        else:
            raise ValueError(
                "kernel_kind must be 'hs_real' "
                "or 'hs_fidelity'."
            )

    return K
def run_pi_representation_svm(
    representation_name: str,
    model,
    nqbts: int,
    train_dset,
    valid_dset,
    *,
    n_classes: int = 3,
    kernel_kind: str = "hs_fidelity",
    normalize_factors: bool = True,
    factor_n_jobs: int = 1,
    svm_n_jobs: int = 1,
    class_weight="balanced",
    sample_weight_mode=None,
    C_grid=(
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
    ),
    cv_splits: int = 5,
    seed: int = 1234,
    cache_size_mb: float = 1024.0,
):
    """
    End-to-end Pi representation SVM using a precomputed kernel.
    """
    set_reproducible_seed(seed)

    (
        train_sequences,
        P_train,
        y_train,
    ) = parse_probability_dataset(
        train_dset,
        n_classes=n_classes,
    )

    (
        valid_sequences,
        P_valid,
        y_valid,
    ) = parse_probability_dataset(
        valid_dset,
        n_classes=n_classes,
    )

    extraction_start = time.perf_counter()

    print(
        f"\n[1/4] Extracting training Pi factors "
        f"({len(train_sequences)} sequences)",
        flush=True,
    )
    
    start = time.perf_counter()
    
    U_train, W_train = compute_pi_factors(
        model=model,
        sequences=train_sequences,
        nqbts=nqbts,
        normalize=normalize_factors,
        n_jobs=factor_n_jobs,
        description="Training Pi factors",
        show_progress=True,
    )
    
    print(
        f"Training-factor extraction completed in "
        f"{time.perf_counter() - start:.2f} seconds.",
        flush=True,
    )




    print(
        f"\n[2/4] Extracting validation Pi factors "
        f"({len(valid_sequences)} sequences)",
        flush=True,
    )
    
    start = time.perf_counter()
    
    U_valid, W_valid = compute_pi_factors(
        model=model,
        sequences=valid_sequences,
        nqbts=nqbts,
        normalize=normalize_factors,
        n_jobs=factor_n_jobs,
        description="Validation Pi factors",
        show_progress=True,
    )
    
    print(
        f"Validation-factor extraction completed in "
        f"{time.perf_counter() - start:.2f} seconds.",
        flush=True,
    )
    
    
    print(
        "\n[3/4] Constructing Pi Gram matrices",
        flush=True,
    )


    factor_extraction_seconds = (
        time.perf_counter()
        - extraction_start
    )

    

    start = time.perf_counter()
    
    K_train = pi_kernel_from_factors_blocked(
        U_train,
        W_train,
        U_train,
        W_train,
        kernel_kind=kernel_kind,
        block_size=256,
        description="Training Gram matrix",
        show_progress=True,
    )
    
    K_valid = pi_kernel_from_factors_blocked(
        U_valid,
        W_valid,
        U_train,
        W_train,
        kernel_kind=kernel_kind,
        block_size=256,
        description="Validation Gram matrix",
        show_progress=True,
    )
    
    kernel_seconds = (
        time.perf_counter() - start
    )
    
    print(
        f"Kernel construction completed in "
        f"{kernel_seconds:.2f} seconds.",
        flush=True,
    )
    
    
    print(
        "\n[4/4] Training and selecting the SVM",
        flush=True,
    )


    result = train_validate_precomputed_svm(
        K_train=K_train,
        y_train=y_train,
        train_target_probabilities=P_train,
        K_valid=K_valid,
        y_valid=y_valid,
        valid_target_probabilities=P_valid,
        C_grid=C_grid,
        class_weight=class_weight,
        sample_weight_mode=sample_weight_mode,
        cv_splits=cv_splits,
        seed=seed,
        n_jobs=svm_n_jobs,
        cache_size_mb=cache_size_mb,
    )

    dim_u = int(
        U_train.shape[1]
    )

    dim_w = int(
        W_train.shape[1]
    )

    implicit_complex_dimension = (
        dim_u * dim_w
    )

    result.update({
        "representation_name": representation_name,
        "kernel_kind": kernel_kind,
        "feature_extraction_seconds": (
            factor_extraction_seconds
            + kernel_seconds
        ),
        "factor_extraction_seconds": (
            factor_extraction_seconds
        ),
        "kernel_construction_seconds": (
            kernel_seconds
        ),
        "factor_dimension_u": dim_u,
        "factor_dimension_w": dim_w,
        "implicit_complex_feature_dimension": (
            implicit_complex_dimension
        ),
        "implicit_real_feature_dimension": (
            2 * implicit_complex_dimension
        ),

        # Maintains compatibility with the existing report function.
        "n_features": (
            2 * implicit_complex_dimension
        ),

        "train_class_counts": {
            int(class_label): int(count)
            for class_label, count in zip(
                *np.unique(
                    y_train,
                    return_counts=True,
                )
            )
        },
        "valid_class_counts": {
            int(class_label): int(count)
            for class_label, count in zip(
                *np.unique(
                    y_valid,
                    return_counts=True,
                )
            )
        },
        "K_train": K_train,
        "K_valid": K_valid,
        "U_train": U_train,
        "W_train": W_train,
        "U_valid": U_valid,
        "W_valid": W_valid,
    })

    return result

    
#------------------------------------------------------------------------------
#            Transformer Model
#------------------------------------------------------------------------------
@dataclass
class TrainedTransformerProcess:
    model: CausalTransformerNextSymbol
    k: int
    BOS: int
    PAD: int
    max_len: int
    device: torch.device

    @torch.no_grad()
    def next_dist(self, prefix: List[int], eps_mix: float = 0.0) -> torch.Tensor:
        self.model.eval()
        x = torch.tensor([self.BOS] + prefix, dtype=torch.long, device=self.device).unsqueeze(0)
        lengths = torch.tensor([x.size(1)], dtype=torch.long, device=self.device)
        logits = self.model(x, lengths).squeeze(0)
        p = torch.softmax(logits, dim=-1)
        if eps_mix > 0:
            p = (1 - eps_mix) * p + eps_mix * (torch.ones_like(p) / self.k)
        return p

    @torch.no_grad()
    def logprob_fixed_length(self, seq: List[int], eps_mix: float = 0.0) -> float:
        lp = 0.0
        prefix: List[int] = []
        for a in seq:
            p = self.next_dist(prefix, eps_mix=eps_mix)
            lp += float(torch.log(p[a] + 1e-30))
            prefix.append(a)
        return lp

    @torch.no_grad()
    def sample(self, length: int, temperature: float = 1.0) -> List[int]:
        self.model.eval()
        seq: List[int] = []
        for _ in range(length):
            x = torch.tensor([self.BOS] + seq, dtype=torch.long, device=self.device).unsqueeze(0)
            lengths = torch.tensor([x.size(1)], dtype=torch.long, device=self.device)
            logits = self.model(x, lengths).squeeze(0) / max(temperature, 1e-6)
            p = torch.softmax(logits, dim=-1)
            a = int(torch.multinomial(p, num_samples=1).item())
            seq.append(a)
        return seq

    @torch.no_grad()
    def embed_prefixes(self, prefixes: List[List[int]], layer: int = -1) -> torch.Tensor:
        """
        Extract a representation for each prefix: last-token hidden state at a chosen layer.

        layer index:
          -1 => final layernorm output (recommended)
          0..n_layers-1 => output after that block
        Returns:
          Z: (N, d_model)
        """
        self.model.eval()
        # batch by padding
        BOS = self.BOS
        PAD = self.PAD
        xs = [torch.tensor([BOS] + p, dtype=torch.long) for p in prefixes]
        lengths = torch.tensor([x.numel() for x in xs], dtype=torch.long, device=self.device)
        maxT = int(lengths.max().item())
        X = torch.full((len(xs), maxT), PAD, dtype=torch.long)
        for i, x in enumerate(xs):
            X[i, :x.numel()] = x
        X = X.to(self.device)

        logits, layer_states = self.model(X, lengths, return_layers=True)
        # layer_states includes per-block outputs + final ln_f output at the end
        h = layer_states[layer]  # (N,T,d)
        idx = (lengths - 1).view(-1, 1, 1).expand(h.size(0), 1, h.size(2))
        last = h.gather(1, idx).squeeze(1)  # (N,d)
        return last

def Rep(seq, gen_model):
    rep = gen_model.embed_prefixes([seq])
    return rep[0]



#------------------------------------------------------------------------------
FeatureExtractor = Callable[
    [Sequence[Sequence[int]]],
    np.ndarray,
]


def set_reproducible_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Some operations do not have deterministic GPU implementations.
    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


def parse_probability_dataset(
    dataset: Sequence[Any],
    *,
    n_classes: int = 3,
    eps: float = 1e-12,
) -> tuple[
    list[list[int]],
    np.ndarray,
    np.ndarray,
]:
    """
    Parse

        [sequence, [p_1, p_2, p_3]]

    into sequences, normalized target probabilities, and modal labels.
    """
    sequences: list[list[int]] = []
    target_probabilities: list[np.ndarray] = []

    for index, item in enumerate(dataset):
        if len(item) != 2:
            raise ValueError(
                f"Dataset item {index} must contain "
                "[sequence, probability_vector]."
            )

        sequence, probabilities = item

        sequence = [
            int(symbol)
            for symbol in sequence
        ]

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        ).reshape(-1)

        if len(probabilities) != n_classes:
            raise ValueError(
                f"Item {index} has {len(probabilities)} target "
                f"probabilities; expected {n_classes}."
            )

        if np.any(~np.isfinite(probabilities)):
            raise ValueError(
                f"Item {index} contains NaN or infinity."
            )

        if np.any(probabilities < -eps):
            raise ValueError(
                f"Item {index} contains negative probabilities."
            )

        probabilities = np.maximum(
            probabilities,
            0.0,
        )

        probability_sum = float(
            probabilities.sum()
        )

        if probability_sum <= eps:
            raise ValueError(
                f"Item {index} has zero probability mass."
            )

        probabilities = (
            probabilities / probability_sum
        )

        sequences.append(sequence)
        target_probabilities.append(probabilities)

    P = np.stack(
        target_probabilities,
        axis=0,
    )

    y = np.argmax(
        P,
        axis=1,
    ).astype(np.int64)

    return sequences, P, y


def to_real_feature_matrix(
    features: Any,
    *,
    dtype=np.float64,
) -> np.ndarray:
    """
    Convert real or complex representations into a real 2D matrix.

    Complex representations are mapped as

        z -> [Re(z), Im(z)].

    Matrix-valued representations are flattened first.
    """
    if isinstance(features, torch.Tensor):
        features = (
            features
            .detach()
            .cpu()
            .numpy()
        )

    X = np.asarray(features)

    if X.ndim == 0:
        raise ValueError(
            "Representation extractor returned a scalar."
        )

    if X.ndim == 1:
        X = X[:, None]

    elif X.ndim > 2:
        X = X.reshape(
            X.shape[0],
            -1,
        )

    if np.iscomplexobj(X):
        X = np.concatenate(
            [
                np.real(X),
                np.imag(X),
            ],
            axis=1,
        )

    X = np.asarray(
        X,
        dtype=dtype,
    )

    if np.any(~np.isfinite(X)):
        bad = np.argwhere(~np.isfinite(X))[:10]

        raise ValueError(
            "Feature matrix contains NaN or infinity. "
            f"First affected positions: {bad.tolist()}."
        )

    return X


def make_training_sample_weights(
    target_probabilities: np.ndarray,
    mode: str | None,
    *,
    eps: float = 1e-12,
) -> np.ndarray | None:
    """
    Optional use of soft-target confidence.

    mode=None
        Standard modal-label SVM.

    mode="confidence"
        Weight by max_c p_c.

    mode="margin"
        Weight by difference between the largest and second-largest
        target probabilities.
    """
    if mode is None:
        return None

    P = np.asarray(
        target_probabilities,
        dtype=np.float64,
    )

    if mode == "confidence":
        weights = np.max(
            P,
            axis=1,
        )

    elif mode == "margin":
        sorted_probabilities = np.sort(
            P,
            axis=1,
        )

        weights = (
            sorted_probabilities[:, -1]
            - sorted_probabilities[:, -2]
        )

        # Avoid exactly zero-weight samples.
        weights = np.maximum(
            weights,
            eps,
        )

    else:
        raise ValueError(
            "sample_weight_mode must be None, "
            "'confidence', or 'margin'."
        )

    # Keep the average sample weight equal to one.
    return weights / np.mean(weights)

#------------------------------------------------------------------------------
#    Classification Metrics
#------------------------------------------------------------------------------
def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_probabilities: np.ndarray,
    *,
    labels: Sequence[int] = (0, 1, 2),
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Standard hard-label metrics plus one soft-target diagnostic.
    """
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    P = np.asarray(
        target_probabilities,
        dtype=np.float64,
    )

    labels = list(labels)

    if class_names is None:
        class_names = [
            f"class_{label}"
            for label in labels
        ]

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    row_sums = confusion.sum(
        axis=1,
        keepdims=True,
    )

    normalized_confusion = np.divide(
        confusion,
        row_sums,
        out=np.zeros_like(
            confusion,
            dtype=np.float64,
        ),
        where=row_sums > 0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    # Expected correctness under the original soft targets:
    #
    # if class c is predicted, score p_c.
    soft_expected_accuracy = float(
        np.mean(
            P[
                np.arange(len(y_pred)),
                y_pred,
            ]
        )
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "matthews_corrcoef": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
        "cohen_kappa": float(
            cohen_kappa_score(
                y_true,
                y_pred,
            )
        ),
        "soft_expected_accuracy": (
            soft_expected_accuracy
        ),
        "soft_expected_error": (
            1.0 - soft_expected_accuracy
        ),
        "mean_modal_confidence": float(
            np.mean(
                np.max(P, axis=1)
            )
        ),
        "confusion_matrix": confusion,
        "normalized_confusion_matrix": (
            normalized_confusion
        ),
        "classification_report": report,
    }

#------------------------------------------------------------------------------
#  SVM training and validation
#------------------------------------------------------------------------------

def train_validate_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_target_probabilities: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_target_probabilities: np.ndarray,
    *,
    kernels: Sequence[str] = ("linear",),
    C_grid: Sequence[float] = (
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
    ),
    gamma_grid: Sequence[Any] = (
        "scale",
        1e-3,
        1e-2,
        1e-1,
        1.0,
    ),
    standardize: bool = True,
    class_weight: str | dict | None = "balanced",
    sample_weight_mode: str | None = None,
    cv_splits: int = 5,
    seed: int = 1234,
    n_jobs: int = -1,
    cache_size_mb: float = 4096.0,
) -> dict[str, Any]:
    """
    Tune the SVM by stratified CV on the training set and evaluate the
    selected model on the untouched validation set.

    Primary model-selection criterion: macro F1.
    """
    X_train = np.asarray(
        X_train,
        dtype=np.float64,
    )

    X_valid = np.asarray(
        X_valid,
        dtype=np.float64,
    )

    y_train = np.asarray(
        y_train,
        dtype=np.int64,
    )

    y_valid = np.asarray(
        y_valid,
        dtype=np.int64,
    )

    if X_train.ndim != 2 or X_valid.ndim != 2:
        raise ValueError(
            "X_train and X_valid must be two-dimensional."
        )

    if X_train.shape[1] != X_valid.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: "
            f"{X_train.shape[1]} and {X_valid.shape[1]}."
        )

    unique_classes, class_counts = np.unique(
        y_train,
        return_counts=True,
    )

    if len(unique_classes) < 2:
        raise ValueError(
            "The training set contains fewer than two classes."
        )

    smallest_class = int(
        class_counts.min()
    )

    effective_cv_splits = min(
        int(cv_splits),
        smallest_class,
    )

    if effective_cv_splits < 2:
        raise ValueError(
            "At least two training examples are required "
            "in every class for cross-validation."
        )

    steps = []

    if standardize:
        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    svm = SVC(
        class_weight=class_weight,
        decision_function_shape="ovr",
        probability=False,
        random_state=seed,
        cache_size=cache_size_mb,
    )

    steps.append(
        (
            "svm",
            svm,
        )
    )

    pipeline = Pipeline(steps)

    parameter_grid: list[dict[str, Any]] = []

    if "linear" in kernels:
        parameter_grid.append({
            "svm__kernel": ["linear"],
            "svm__C": list(C_grid),
        })

    if "rbf" in kernels:
        parameter_grid.append({
            "svm__kernel": ["rbf"],
            "svm__C": list(C_grid),
            "svm__gamma": list(gamma_grid),
        })

    unsupported = (
        set(kernels)
        - {"linear", "rbf"}
    )

    if unsupported:
        raise ValueError(
            f"Unsupported kernels: {sorted(unsupported)}."
        )

    if not parameter_grid:
        raise ValueError(
            "At least one kernel must be supplied."
        )

    cv = StratifiedKFold(
        n_splits=effective_cv_splits,
        shuffle=True,
        random_state=seed,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": (
            "balanced_accuracy"
        ),
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
        return_train_score=True,
        error_score="raise",
        verbose=0,
    )


    sample_weights = make_training_sample_weights(
        train_target_probabilities,
        sample_weight_mode,
    )

    fit_arguments = {}

    if sample_weights is not None:
        fit_arguments["svm__sample_weight"] = (
            sample_weights
        )

    start_time = time.perf_counter()

    with parallel_backend(
        "threading",
        n_jobs=n_jobs,
    ):
        search.fit(
            X_train,
            y_train,
            **fit_arguments,
        )

    fit_seconds = (
        time.perf_counter()
        - start_time
    )

    best_model = search.best_estimator_

    y_train_pred = best_model.predict(
        X_train
    )

    y_valid_pred = best_model.predict(
        X_valid
    )

    train_metrics = evaluate_classification(
        y_train,
        y_train_pred,
        train_target_probabilities,
    )

    valid_metrics = evaluate_classification(
        y_valid,
        y_valid_pred,
        valid_target_probabilities,
    )

    best_index = int(
        search.best_index_
    )

    cv_results = search.cv_results_

    return {
        "model": best_model,
        "search": search,
        "best_params": search.best_params_,
        "best_cv_f1_macro": float(
            search.best_score_
        ),
        "best_cv_f1_macro_std": float(
            cv_results[
                "std_test_f1_macro"
            ][best_index]
        ),
        "best_cv_accuracy": float(
            cv_results[
                "mean_test_accuracy"
            ][best_index]
        ),
        "best_cv_balanced_accuracy": float(
            cv_results[
                "mean_test_balanced_accuracy"
            ][best_index]
        ),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "y_train_pred": y_train_pred,
        "y_valid_pred": y_valid_pred,
        "n_train": int(len(y_train)),
        "n_valid": int(len(y_valid)),
        "n_features": int(
            X_train.shape[1]
        ),
        "cv_splits": effective_cv_splits,
        "fit_seconds": fit_seconds,
        "sample_weight_mode": (
            sample_weight_mode
        ),
        "standardize": standardize,
    }

# =============================================================================
# 3) SCALABLE CAUSAL TRANSFORMER (GPT-style)
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask, key_padding_mask):
        # pre-norm attention
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(y)

        # pre-norm FFN
        y = self.ln2(x)
        y = self.ff(y)
        x = x + self.drop2(y)
        return x


class CausalTransformerNextSymbol(nn.Module):
    """
    Decoder-only transformer that outputs logits over k symbols for the "next symbol"
    given a prefix. We take the hidden state at the last (non-pad) position.
    """
    def __init__(
        self,
        vocab_in: int,
        k: int,
        pad_idx: int,
        max_ctx: int,          # maximum input length (BOS + max prefix length)
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        tie_embeddings: bool = False,  # usually irrelevant here because output dim is k (not vocab_in)
    ):
        super().__init__()
        self.k = k
        self.pad_idx = pad_idx
        self.max_ctx = max_ctx
        self.d_model = d_model

        if d_ff is None:
            d_ff = 4 * d_model

        self.tok_emb = nn.Embedding(vocab_in, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_ctx, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, k)

        # precompute a max causal mask (bool) on CPU; we slice to T and move to device
        full = torch.triu(torch.ones(max_ctx, max_ctx, dtype=torch.bool), diagonal=1)
        self.register_buffer("_causal_mask_full", full, persistent=False)

    def forward(self, x: torch.LongTensor, lengths: torch.LongTensor, return_layers: bool = False):
        """
        x: (B,T) with PAD
        lengths: (B,) true lengths
        return_layers: if True, returns list of layer outputs (B,T,d_model) including final.
        """
        B, T = x.shape
        if T > self.max_ctx:
            raise ValueError(f"Input length T={T} exceeds max_ctx={self.max_ctx}. Increase max_ctx.")

        device = x.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B,T)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)

        key_padding_mask = (x == self.pad_idx)  # True = ignore
        attn_mask = self._causal_mask_full[:T, :T].to(device)  # True = mask

        layer_states = [] if return_layers else None
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            if return_layers:
                layer_states.append(h)

        h = self.ln_f(h)
        if return_layers:
            layer_states.append(h)

        # gather last valid token state per example
        idx = (lengths - 1).view(B, 1, 1).expand(B, 1, h.size(-1))
        last = h.gather(1, idx).squeeze(1)  # (B,d_model)
        logits = self.head(last)            # (B,k)

        if return_layers:
            return logits, layer_states
        return logits


def soft_ce(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return -(soft_targets * logp).sum(dim=-1)  # (B,)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#------------------------------------------------------------------------------
#   Transformer representation extractor
#------------------------------------------------------------------------------
def make_prefix_embedding_extractor(
    model,
    *,
    layer: int = -1,
    batch_size: int = 512,
) -> FeatureExtractor:
    """
    Works for the causal transformer and for any RNN exposing:

        model.embed_prefixes(prefixes, layer=...)
    """

    def extract(
        sequences: Sequence[Sequence[int]],
    ) -> np.ndarray:
        model.model.eval()

        batches: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(
                0,
                len(sequences),
                batch_size,
            ):
                batch_sequences = [
                    list(sequence)
                    for sequence in sequences[
                        start:start + batch_size
                    ]
                ]

                Z = model.embed_prefixes(
                    batch_sequences,
                    layer=layer,
                )

                if isinstance(Z, torch.Tensor):
                    Z = (
                        Z.detach()
                        .cpu()
                        .numpy()
                    )

                batches.append(
                    np.asarray(Z)
                )

        return np.concatenate(
            batches,
            axis=0,
        )

    return extract

#------------------------------------------------------------------------------
#   RNN representation extractor
#------------------------------------------------------------------------------
# def rnn_model.embed_prefixes(prefixes):
#     return None

def make_rnn_extractor(
    model,
    *,
    batch_size: int = 512,
) -> FeatureExtractor:

    def extract(sequences):
        outputs = []

        model.eval()

        with torch.inference_mode():
            for start in range(
                0,
                len(sequences),
                batch_size,
            ):
                batch = sequences[
                    start:start + batch_size
                ]

                Z = model.embed_prefixes(batch)

                if isinstance(Z, torch.Tensor):
                    Z = (
                        Z.detach()
                        .cpu()
                        .numpy()
                    )

                outputs.append(
                    np.asarray(Z)
                )

        return np.concatenate(
            outputs,
            axis=0,
        )

    return extract

#------------------------------------------------------------------------------
#   Quantum representation extractor
#------------------------------------------------------------------------------
def compute_omega_vectors(
    model,
    sequences,
    nqbt,
    epss,
    compute_in_parallel,
    n_jobs,
    parallel_backend):
    
    return None
    

def make_omega_extractor(
    model,
    nqbts: int,
    *,
    eps: float = 1e-12,
    compute_in_parallel: bool = False,
    n_jobs: int = 4,
    parallel_backend: str = "threading",
) -> FeatureExtractor:

    def extract(sequences):
        return compute_omega_vectors(
            model=model,
            sequences=sequences,
            nqbts=nqbts,
            eps=eps,
            compute_in_parallel=(
                compute_in_parallel
            ),
            n_jobs=n_jobs,
            parallel_backend=(
                parallel_backend
            ),
        )

    return extract

def RepUV(model, sequence, n, dtype=np.complex64):
    """
    Non-density spacetime representation factors for a path sequence:
      Pi(seq) ∝ |u><w|
    using column-major vec convention.

    Returns
    -------
    u, w : np.ndarray, shape (d*d,), complex
        u = vec(K_seq @ rho0) / sqrt(d)
        w = vec(K_seq)        / sqrt(d)

    Notes
    -----
    The 1/sqrt(d) factor is optional and only changes global scaling.
    """
    dim = 2 ** n

    Kseq, rho0, p_seq = model.path_operator(sequence, return_prob=True, device="cpu")

    K = np.asarray(Kseq, dtype=dtype)
    rho = np.asarray(rho0, dtype=dtype)

    # column-major vec convention
    u = (K @ rho).reshape(-1, order="F") / np.sqrt(dim)
    w = K.reshape(-1, order="F") / np.sqrt(dim)

    return u, w

    

def make_pi_extractor(
    model,
    nqbts: int,
) -> FeatureExtractor:
    """
    Explicit vectorization of

        Pi = |u><w|.

    With column-major vectorization,

        vec_F(Pi) = conjugate(w) kron u.

    Warning: the resulting dimension is quadratic in len(u).
    """

    def extract(sequences):
        vectors = []

        for sequence in sequences:
            u, w = RepUV(
                model,
                sequence,
                nqbts,
            )

            u = np.asarray(
                u,
                dtype=np.complex128,
            ).reshape(-1)

            w = np.asarray(
                w,
                dtype=np.complex128,
            ).reshape(-1)

            pi_vector = np.kron(
                np.conjugate(w),
                u,
            )

            vectors.append(
                pi_vector
            )

        return np.stack(
            vectors,
            axis=0,
        )

    return extract

#------------------------------------------------------------------------------
# Rpresentation - SVM training
#------------------------------------------------------------------------------
def run_representation_svm(
    representation_name: str,
    extractor: FeatureExtractor,
    train_dset,
    valid_dset,
    *,
    n_classes: int = 3,
    kernels: Sequence[str] = ("linear",),
    standardize: bool = True,
    class_weight: str | dict | None = "balanced",
    sample_weight_mode: str | None = None,
    seed: int = 1234,
    cv_splits: int = 5,
    svm_n_jobs: int = -1,
) -> dict[str, Any]:
    """
    Extract one representation and evaluate its SVM readout.
    """
    set_reproducible_seed(seed)

    (
        train_sequences,
        P_train,
        y_train,
    ) = parse_probability_dataset(
        train_dset,
        n_classes=n_classes,
    )

    (
        valid_sequences,
        P_valid,
        y_valid,
    ) = parse_probability_dataset(
        valid_dset,
        n_classes=n_classes,
    )

    extraction_start = time.perf_counter()

    X_train_raw = extractor(
        train_sequences
    )

    X_valid_raw = extractor(
        valid_sequences
    )

    extraction_seconds = (
        time.perf_counter()
        - extraction_start
    )

    X_train = to_real_feature_matrix(
        X_train_raw
    )

    X_valid = to_real_feature_matrix(
        X_valid_raw
    )

    if X_train.shape[0] != len(
        train_sequences
    ):
        raise ValueError(
            "Training representation row count does not "
            "match train_dset."
        )

    if X_valid.shape[0] != len(
        valid_sequences
    ):
        raise ValueError(
            "Validation representation row count does not "
            "match valid_dset."
        )

    result = train_validate_svm(
        X_train=X_train,
        y_train=y_train,
        train_target_probabilities=P_train,
        X_valid=X_valid,
        y_valid=y_valid,
        valid_target_probabilities=P_valid,
        kernels=kernels,
        standardize=standardize,
        class_weight=class_weight,
        sample_weight_mode=(
            sample_weight_mode
        ),
        cv_splits=cv_splits,
        seed=seed,
        n_jobs=svm_n_jobs,
    )

    result.update({
        "representation_name": (
            representation_name
        ),
        "feature_extraction_seconds": (
            extraction_seconds
        ),
        "train_class_counts": dict(
            zip(
                *np.unique(
                    y_train,
                    return_counts=True,
                )
            )
        ),
        "valid_class_counts": dict(
            zip(
                *np.unique(
                    y_valid,
                    return_counts=True,
                )
            )
        ),
    })

    return result
#------------------------------------------------------------------------------
# Reporting
#------------------------------------------------------------------------------

def print_representation_svm_report(
    result: Mapping[str, Any],
) -> None:
    name = result["representation_name"]

    train_metrics = result[
        "train_metrics"
    ]

    valid_metrics = result[
        "valid_metrics"
    ]

    print("\n" + "=" * 80)
    print(f"REPRESENTATION: {name}")
    print("=" * 80)

    print(
        f"Training samples       : {result['n_train']}"
    )
    print(
        f"Validation samples     : {result['n_valid']}"
    )
    print(
        f"Feature dimension      : {result['n_features']}"
    )
    print(
        f"Training class counts  : "
        f"{result['train_class_counts']}"
    )
    print(
        f"Validation class counts: "
        f"{result['valid_class_counts']}"
    )
    print(
        f"Best SVM parameters    : "
        f"{result['best_params']}"
    )
    print(
        f"CV macro F1            : "
        f"{result['best_cv_f1_macro']:.6f} "
        f"+/- {result['best_cv_f1_macro_std']:.6f}"
    )
    print(
        f"Feature extraction time: "
        f"{result['feature_extraction_seconds']:.3f} sec"
    )
    print(
        f"SVM fit/search time    : "
        f"{result['fit_seconds']:.3f} sec"
    )

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "matthews_corrcoef",
        "cohen_kappa",
        "soft_expected_accuracy",
    ]

    rows = []

    for metric in metric_names:
        rows.append({
            "metric": metric,
            "train": train_metrics[metric],
            "validation": valid_metrics[metric],
            "gap_train_minus_valid": (
                train_metrics[metric]
                - valid_metrics[metric]
            ),
        })

    metric_table = pd.DataFrame(rows)

    print("\nPerformance:")
    print(
        metric_table.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print("\nValidation confusion matrix:")
    print(
        valid_metrics[
            "confusion_matrix"
        ]
    )

    print(
        "\nValidation row-normalized "
        "confusion matrix:"
    )
    print(
        np.array2string(
            valid_metrics[
                "normalized_confusion_matrix"
            ],
            precision=4,
            suppress_small=True,
        )
    )

    class_report = pd.DataFrame(
        valid_metrics[
            "classification_report"
        ]
    ).T

    print("\nValidation classification report:")
    print(
        class_report.to_string(
            float_format=lambda value: (
                f"{value:.6f}"
            )
        )
    )
#------------------------------------------------------------------------------
#  Comparing Repreentations
#------------------------------------------------------------------------------
def compare_representation_svms(
    representation_extractors: Mapping[
        str,
        FeatureExtractor,
    ],
    train_dset,
    valid_dset,
    *,
    kernels: Sequence[str] = ("linear",),
    standardize: bool = True,
    class_weight: str | dict | None = "balanced",
    sample_weight_mode: str | None = None,
    seed: int = 1234,
    cv_splits: int = 5,
    svm_n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply exactly the same SVM protocol to every representation.
    """
    all_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for name, extractor in representation_extractors.items():
        print(
            f"\nEvaluating representation: {name}"
        )

        result = run_representation_svm(
            representation_name=name,
            extractor=extractor,
            train_dset=train_dset,
            valid_dset=valid_dset,
            kernels=kernels,
            standardize=standardize,
            class_weight=class_weight,
            sample_weight_mode=(
                sample_weight_mode
            ),
            seed=seed,
            cv_splits=cv_splits,
            svm_n_jobs=svm_n_jobs,
        )

        all_results[name] = result

        valid = result["valid_metrics"]
        train = result["train_metrics"]

        summary_rows.append({
            "representation": name,
            "n_features": result[
                "n_features"
            ],
            "best_params": str(
                result["best_params"]
            ),
            "cv_f1_macro": result[
                "best_cv_f1_macro"
            ],
            "train_accuracy": train[
                "accuracy"
            ],
            "valid_accuracy": valid[
                "accuracy"
            ],
            "valid_balanced_accuracy": valid[
                "balanced_accuracy"
            ],
            "train_f1_macro": train[
                "f1_macro"
            ],
            "valid_f1_macro": valid[
                "f1_macro"
            ],
            "valid_f1_weighted": valid[
                "f1_weighted"
            ],
            "valid_mcc": valid[
                "matthews_corrcoef"
            ],
            "valid_soft_expected_accuracy": valid[
                "soft_expected_accuracy"
            ],
            "generalization_gap_f1": (
                train["f1_macro"]
                - valid["f1_macro"]
            ),
        })

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            "valid_f1_macro",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return summary, all_results
#------------------------------------------------------------------------------
# Usage
#------------------------------------------------------------------------------

dPath = '..\data\\'
dPath = 'C:\V\Projects\RepComp\data\\'
dPath ='C:\EXPIMP\Vanio\Projects\ICML-2026\Learning\Data\\'
mPath = '..\models\\'
mPath = 'C:\V\Projects\RepComp\models\\'
mPath ='C:\EXPIMP\Vanio\Projects\ICML-2026\Learning\Models\\'

model = "Causal Transformer — final last token"
model = "Spacetime Pi — real HS kernel"
model = "Spacetime Pi — HS fidelity kernel"



dates_march = ['20250303','20250304','20250305','20250306','20250307','20250310','20250311','20250312','20250313','20250314',
               '20250317','20250318','20250319','20250320','20250321','20250324','20250325','20250326','20250327','20250328','20250331']

features_list =  ["log_mid",'sigma_W',"vpin",'imbalance',"mid_cross_prev_ask_up", 
                    'micro_price',"trade_ofi_n", "log_mid_return_fwd_1"]

symbol= 'AAPL'
clsName = 'ca4'

predicted = features_list[0]
predictor = features_list[1]

trn_date = dates_march[0]
vld_date = dates_march[1]

trn_file = dPath+'CLS_DISTR_'+symbol+'_'+trn_date+'_'+predicted+'_'+predictor+'_'+clsName
vld_file = dPath+'CLS_DISTR_'+symbol+'_'+vld_date+'_'+predicted+'_'+predictor+'_'+clsName

training   = pickle.load(open(trn_file, "rb")) 
validation = pickle.load(open(vld_file, "rb")) 
tSequences = [list(s[0]) for s in training]
vSequences = [list(v[0]) for v in validation]

if model == "Causal Transformer — final last token":
    trm_model_file =  mPath +  'TRNSF_MOD_'+symbol+'_'+trn_date+'_'+predicted+'_'+predictor+'.pt'
    
    device = torch.device("cpu")  # or "cuda"
                    
    trm_model = torch.load(
        trm_model_file,
        map_location=device,
        weights_only=False,
    )
                    
    trm_model.model.to(device)   
    trm_model.model.eval()
    
    '''                
    Z = trm_model.embed_prefixes(
        prefixes=tSequences,
        layer=-1,
    )
    
    print(Z.shape)
    '''
    
    
    
    transformer_extractor = (
        make_prefix_embedding_extractor(
            trm_model,
            layer=-1,
            batch_size=512,
        )
    )
    
    transformer_result = run_representation_svm(
        representation_name=(
            "Causal Transformer — final last token"
        ),
        extractor=transformer_extractor,
        train_dset=training ,
        valid_dset=validation,
        kernels=("linear", "rbf"),
        standardize=True,
        class_weight="balanced",
        sample_weight_mode=None,
        seed=1234,
        cv_splits=5,
        svm_n_jobs=14,  # -1
    )
    
    
    
    
    
    print_representation_svm_report(
        transformer_result
    )
if model == "Spacetime Pi — HS fidelity kernel":
    nQubits = 5
    
    sptm_model_file =  mPath +  'QMOD_'+symbol+'_'+trn_date+'_'+predicted+'_'+predictor+'_'+str(nQubits)+'q'
    sptm_model = pickle.load(open(sptm_model_file, "rb"))[0] 
    
    pi_result = run_pi_representation_svm(
        representation_name=(
            "Spacetime Pi — HS fidelity kernel"
        ),
        model=sptm_model,
        nqbts=nQubits,
        train_dset=training,
        valid_dset=validation,
        kernel_kind="hs_fidelity",
        normalize_factors=True,
        factor_n_jobs=8,
        svm_n_jobs=8,
        class_weight="balanced",
        sample_weight_mode=None,
        cv_splits=5,
        seed=1234,
    )
    
    print_representation_svm_report(
        pi_result
    )


    

if model == "Spacetime Pi — real HS kernel":
    pi_linear_result = run_pi_representation_svm(
        representation_name=(
            "Spacetime Pi — real HS kernel"
        ),
        model=quantum_model,
        nqbts=nqbts,
        train_dset=train_dset,
        valid_dset=valid_dset,
        kernel_kind="hs_real",
        normalize_factors=True,
        factor_n_jobs=8,
        svm_n_jobs=8,
        class_weight="balanced",
        sample_weight_mode=None,
        cv_splits=5,
        seed=1234,
    )
    
    print_representation_svm_report(
        pi_linear_result
    )    
    