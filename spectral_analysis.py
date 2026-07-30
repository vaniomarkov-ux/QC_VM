# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:31:49 2026

@author: vanio
"""

from __future__ import annotations
from typing import Callable, List, Sequence, Tuple, Optional, Dict, Any
from typing import  Literal 
import numpy as np

SeqT = Any
KernelKind = Literal["hs_cosine", "hs_cosine_abs2"]


SeqT = Any  # your sequence type (e.g. list[int], tuple[int], etc.)

def gram_centered_eigenspectrum(
    seq_prob: List[Tuple[SeqT, float]],
    Rep: Callable[[SeqT], np.ndarray],
    model, 
    eps: float = 1e-12,
    normalize_eigs: str = "trace",   # "trace" or "sum"
    return_matrices: bool = False,
) -> Dict[str, Any]:
    """
    Inputs
    ------
    seq_prob: list of (sequence, probability) pairs [(s1,p1),...,(sn,pn)]
              p_i should be >= 0; will be renormalized to sum to 1 if needed.
    Rep:      callable Rep(s) -> embedding vector z (1D array-like)
    eps:      numerical stability for norm and eigenvalue clipping
    normalize_eigs:
        - "trace": normalize by trace(K_centered) = sum eigenvalues (same as sum for PSD)
        - "sum":   normalize by sum of nonnegative eigenvalues (robust if tiny negatives appear)
    return_matrices: if True, also returns K and K_centered

    Returns
    -------
    dict with:
      - "eigenvalues": normalized eigenvalues (descending), nonnegative
      - "raw_eigenvalues": raw eigenvalues (descending), clipped at 0 for output
      - optionally "K", "K_centered", "p"
    """
    n = len(seq_prob)
    if n == 0:
        raise ValueError("seq_prob is empty.")

    # ---- probabilities
    p = np.array([float(pi) for _, pi in seq_prob], dtype=np.float32)
    if np.any(p < 0):
        raise ValueError("Probabilities must be nonnegative.")
    psum = p.sum()
    if psum <= 0:
        raise ValueError("Sum of probabilities must be > 0.")
    p = p / psum  # ensure sum(p)=1

    # ---- embeddings matrix Z: (n, d)
    Z_list = []
    for s, _ in seq_prob:
        z = np.asarray(Rep(s, model), dtype=np.float32).reshape(-1)
        Z_list.append(z)
    d = Z_list[0].shape[0]
    if any(z.shape[0] != d for z in Z_list):
        raise ValueError("All embeddings must have the same dimension.")
    Z = np.vstack(Z_list)  # (n,d)

    # ---- cosine Gram matrix K = (Zhat)(Zhat)^T
    norms = np.linalg.norm(Z, axis=1, keepdims=True)  # (n,1)
    # Guard against zero vectors
    norms = np.maximum(norms, eps)
    Zhat = Z / norms
    K = Zhat @ Zhat.T  # (n,n)
    # Force symmetry (numerical)
    K = 0.5 * (K + K.T)

    # ---- probability-weighted centering
    # Cp = I - 1 p^T ; use Cp K Cp^T to preserve symmetry
    one = np.ones((n, 1), dtype=np.float32)
    Cp = np.eye(n, dtype=np.float32) - one @ p.reshape(1, n)
    Kc = Cp @ K @ Cp.T
    Kc = 0.5 * (Kc + Kc.T)

    # ---- eigenvalues of centered Gram (symmetric)
    eigvals = np.linalg.eigh(Kc)[0]          # ascending
    eigvals = eigvals[::-1]                 # descending
    # Clip tiny negatives (centering can introduce tiny negative eigs numerically)
    eigvals_clipped = np.maximum(eigvals, 0.0)

    # ---- normalize
    if normalize_eigs == "trace":
        denom = float(np.trace(Kc))
        # trace should equal sum(eigs); may be tiny ~0 if Kc is ~0
        denom = max(denom, eps)
    elif normalize_eigs == "sum":
        denom = float(eigvals_clipped.sum())
        denom = max(denom, eps)
    else:
        raise ValueError("normalize_eigs must be 'trace' or 'sum'.")

    eigvals_norm = eigvals_clipped / denom

    out = {
        "eigenvalues": eigvals_norm,
        "raw_eigenvalues": eigvals_clipped,
        "p": p,
    }
    if return_matrices:
        out["K"] = K
        out["K_centered"] = Kc
    return out
#------------------------------------------------------------------------------
# Quantum HS Kernels
#------------------------------------------------------------------------------
KernelKind = Literal["hs_cosine", "hs_cosine_abs2"]

def hs_cosine_kernel(
    ops: List[np.ndarray],
    eps: float = 1e-12,
) -> np.ndarray:
    """
    HS-cosine Mercer kernel for complex operators A_i (not necessarily Hermitian/PSD).

    K_ij = Re Tr(A_i^† A_j) / (||A_i||_HS ||A_j||_HS)

    ops: list of (d,d) complex/real arrays
    returns: (n,n) real symmetric Gram matrix (PSD up to numerical noise)
    """
    n = len(ops)
    if n == 0:
        raise ValueError("ops is empty.")
    d = ops[0].shape
    if any(A.shape != d for A in ops):
        raise ValueError("All operators must have the same shape.")

    # HS norms
    norms = np.empty(n, dtype=np.float32)
    for i, A in enumerate(ops):
        # ||A||_HS^2 = Tr(A^† A) = sum |A_ij|^2
        norms[i] = float(np.sqrt(np.real(np.vdot(A.ravel(), A.ravel())) + eps))

    K = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        Ai = ops[i]
        AiH = Ai.conj().T
        for j in range(i, n):
            val = np.trace(AiH @ ops[j])  # complex
            val = float(np.real(val)) / (norms[i] * norms[j])
            K[i, j] = K[j, i] = val

    # symmetrize for numerical stability
    return 0.5 * (K + K.T)


def hs_cosine_abs2_kernel(
    ops: List[np.ndarray],
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Phase-robust Mercer kernel using squared magnitude of HS overlap.

    K_ij = |Tr(A_i^† A_j)|^2 / (||A_i||_HS^2 ||A_j||_HS^2)

    - Real, nonnegative
    - PSD (Gram of degree-2 polynomial features on normalized vec(A))
    """
    n = len(ops)
    if n == 0:
        raise ValueError("ops is empty.")
    d = ops[0].shape
    if any(A.shape != d for A in ops):
        raise ValueError("All operators must have the same shape.")

    # HS norms squared
    norm2 = np.empty(n, dtype=np.float32)
    for i, A in enumerate(ops):
        norm2[i] = float(np.real(np.vdot(A.ravel(), A.ravel())) + eps)

    K = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        Ai = ops[i]
        AiH = Ai.conj().T
        for j in range(i, n):
            ov = np.trace(AiH @ ops[j])   # complex overlap
            val = (np.abs(ov) ** 2) / (norm2[i] * norm2[j])
            K[i, j] = K[j, i] = float(val)

    return 0.5 * (K + K.T)


def center_kernel_prob(K: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Probability-weighted centering:
      Kc = Cp K Cp^T,  Cp = I - 1 p^T
    """
    n = K.shape[0]
    p = np.asarray(p, dtype=np.float32).reshape(-1)
    if p.shape[0] != n:
        raise ValueError("p must have length n.")
    if np.any(p < 0):
        raise ValueError("p must be nonnegative.")
    p = p / max(p.sum(), eps)

    one = np.ones((n, 1), dtype=np.float32)
    Cp = np.eye(n, dtype=np.float32) - one @ p.reshape(1, n)
    Kc = Cp @ K @ Cp.T
    return 0.5 * (Kc + Kc.T)


def kernel_eigenspectrum(Kc: np.ndarray, eps: float = 1e-12, normalize: Literal["sum", "trace"] = "sum") -> np.ndarray:
    """
    Eigenvalues of centered Gram matrix, sorted descending, clipped >=0, normalized.
    """
    eig = np.linalg.eigh(0.5 * (Kc + Kc.T))[0][::-1]  # descending
    eig = np.maximum(eig, 0.0)
    if normalize == "trace":
        denom = float(np.trace(Kc))
    else:
        denom = float(eig.sum())
    denom = max(denom, eps)
    return eig / denom


def quantum_spectrum_from_ops(
    ops: List[np.ndarray],
    p: np.ndarray,
    kind: KernelKind = "hs_cosine",
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    End-to-end:
      ops -> K -> centered Kc -> normalized eigenvalues
    """
    if kind == "hs_cosine":
        K = hs_cosine_kernel(ops, eps=eps)
    elif kind == "hs_cosine_abs2":
        K = hs_cosine_abs2_kernel(ops, eps=eps)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    Kc = center_kernel_prob(K, p, eps=eps)
    lams = kernel_eigenspectrum(Kc, eps=eps, normalize="sum")
    return {"K": K, "K_centered": Kc, "eigenvalues": lams}

 


def quantum_gram_centered_eigenspectrum(
    seq_prob: List[Tuple[SeqT, float]],
    Rep: Callable[[SeqT], np.ndarray],
    nqbts,
    mqbts,
    ccontr,
    model_parameters,                 
    controller, 
    kernel: KernelKind = "hs_cosine",
    eps: float = 1e-12,
    normalize_eigs: Literal["sum", "trace"] = "sum",
    return_matrices: bool = False,
) -> Dict[str, Any]:
    """
    Quantum spectral analysis with the same high-level interface as the classical version.

    Inputs
    ------
    seq_prob : list[(seq, p)]
        Sequences and their probabilities (used for centering). p_i >= 0.
    Rep : callable
        Rep(seq) -> A (complex or real matrix, e.g. spacetime operator/density), shape (d,d).
        You can close over `model` in Rep if needed.
    model : optional
        Passed for interface symmetry; you can ignore or use via Rep closure.
    kernel : "hs_cosine" or "hs_cosine_abs2"
        - hs_cosine: Re Tr(A_i^† A_j) / (||A_i||_HS ||A_j||_HS)
        - hs_cosine_abs2: |Tr(A_i^† A_j)|^2 / (||A_i||_HS^2 ||A_j||_HS^2)
    eps : float
        Stabilizer for norms / normalization.
    normalize_eigs : "sum" or "trace"
        How to normalize eigenvalues after clipping at 0.
    return_matrices : bool
        If True, also returns K and K_centered.

    Returns
    -------
    dict with:
      - eigenvalues: normalized, descending, nonnegative
      - raw_eigenvalues: clipped raw (descending)
      - p: normalized probabilities
      - optionally K, K_centered
    """
    n = len(seq_prob)
    if n == 0:
        raise ValueError("seq_prob is empty.")

    # ---- probabilities
    p = np.array([float(pi) for _, pi in seq_prob], dtype=np.float32)
    if np.any(p < 0):
        raise ValueError("Probabilities must be nonnegative.")
    psum = p.sum()
    if psum <= 0:
        raise ValueError("Sum of probabilities must be > 0.")
    p = p / psum

    # ---- compute operator representations
    reps: List[np.ndarray] = []
    for s, _ in seq_prob:
        A = np.asarray(Rep(nqbts, mqbts, ccontr, model_parameters, controller,s))   #n, m, c, model_parameters, controller, seq
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"Rep(seq) must return a square matrix. Got shape {A.shape}.")
        reps.append(A.astype(np.complex64, copy=False))

    d = reps[0].shape
    if any(A.shape != d for A in reps):
        raise ValueError("All Rep(seq) matrices must have the same shape.")

    # ---- HS norms (and norm^2) for normalization
    # ||A||_HS^2 = sum |A_ij|^2 = vdot(vec(A), vec(A))
    norm2 = np.array([np.real(np.vdot(A.ravel(), A.ravel())) for A in reps], dtype=np.float32) + eps
    norms = np.sqrt(norm2)

    # ---- build Gram matrix
    K = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        AiH = reps[i].conj().T
        for j in range(i, n):
            ov = np.trace(AiH @ reps[j])  # complex overlap Tr(A_i^† A_j)

            if kernel == "hs_cosine":
                val = float(np.real(ov) / (norms[i] * norms[j]))
            elif kernel == "hs_cosine_abs2":
                val = float((np.abs(ov) ** 2) / (norm2[i] * norm2[j]))
            else:
                raise ValueError(f"Unknown kernel: {kernel}")

            K[i, j] = K[j, i] = val

    K = 0.5 * (K + K.T)

    # ---- probability-weighted centering: Kc = Cp K Cp^T, Cp = I - 1 p^T
    one = np.ones((n, 1), dtype=np.float32)
    Cp = np.eye(n, dtype=np.float32) - one @ p.reshape(1, n)
    Kc = Cp @ K @ Cp.T
    Kc = 0.5 * (Kc + Kc.T)

    # ---- eigenvalues (symmetric), descending
    eigvals = np.linalg.eigh(Kc)[0][::-1]
    eigvals_clipped = np.maximum(eigvals, 0.0)

    # ---- normalize
    if normalize_eigs == "trace":
        denom = float(np.trace(Kc))
        denom = max(denom, eps)
    elif normalize_eigs == "sum":
        denom = float(eigvals_clipped.sum())
        denom = max(denom, eps)
    else:
        raise ValueError("normalize_eigs must be 'sum' or 'trace'.")

    eigvals_norm = eigvals_clipped / denom

    out: Dict[str, Any] = {
        "eigenvalues": eigvals_norm,
        "raw_eigenvalues": eigvals_clipped,
        "p": p,
    }
    if return_matrices:
        out["K"] = K
        out["K_centered"] = Kc
    return out