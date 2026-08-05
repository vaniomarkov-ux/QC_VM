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
from joblib import parallel_backend


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
        verbose=2,
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
    layer: int = -1,
    batch_size: int = 512,
    show_progress: bool = True,
):
    """
    Create an extractor returning one transformer embedding per sequence.

    Output shape:
        (number_of_sequences, embedding_dimension)
    """

    def extract(sequences):
        sequences = list(sequences)

        if len(sequences) == 0:
            raise ValueError("The sequence collection is empty.")

        if hasattr(model, "model"):
            model.model.eval()
        elif hasattr(model, "eval"):
            model.eval()

        batches = []

        batch_starts = range(
            0,
            len(sequences),
            batch_size,
        )

        total_batches = (
            len(sequences) + batch_size - 1
        ) // batch_size

        iterator = tqdm(
            batch_starts,
            total=total_batches,
            desc="Transformer embeddings",
            unit="batch",
            disable=not show_progress,
        )

        start_time = time.perf_counter()

        with torch.inference_mode():
            for start in iterator:
                stop = min(
                    start + batch_size,
                    len(sequences),
                )

                batch_sequences = [
                    list(sequence)
                    for sequence in sequences[start:stop]
                ]

                Z = model.embed_prefixes(
                    batch_sequences,
                    layer=layer,
                )

                if torch.is_tensor(Z):
                    Z = Z.detach().cpu().numpy()

                Z = np.asarray(Z)

                if Z.ndim != 2:
                    raise ValueError(
                        "embed_prefixes must return a matrix "
                        f"with shape (batch, dimension); received {Z.shape}."
                    )

                batches.append(Z)

                elapsed = time.perf_counter() - start_time
                completed = stop

                rate = completed / max(elapsed, 1e-12)

                iterator.set_postfix(
                    sequences=f"{completed}/{len(sequences)}",
                    rate=f"{rate:.1f}/s",
                )

        X = np.concatenate(
            batches,
            axis=0,
        )

        elapsed = time.perf_counter() - start_time

        print(
            "\nTransformer extraction completed:"
            f"\n  sequences = {len(sequences)}"
            f"\n  dimension = {X.shape[1]}"
            f"\n  time      = {elapsed:.2f} seconds",
            flush=True,
        )

        return X

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

def RepUV(
    model,
    sequence,
    nqbts,
    ):
    return None

    

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
    representation_name,
    extractor,
    train_dset,
    valid_dset,
    *,
    n_classes=3,
    kernels=("linear", "rbf"),
    standardize=True,
    class_weight="balanced",
    sample_weight_mode=None,
    seed=1234,
    cv_splits=5,
    svm_n_jobs=1,
    **svm_kwargs,
):
    import time

    set_reproducible_seed(seed)

    print(
        f"\n{'=' * 72}"
        f"\nRepresentation: {representation_name}"
        f"\n{'=' * 72}",
        flush=True,
    )

    print("\n[1/4] Parsing datasets", flush=True)

    train_sequences, P_train, y_train = (
        parse_probability_dataset(
            train_dset,
            n_classes=n_classes,
        )
    )

    valid_sequences, P_valid, y_valid = (
        parse_probability_dataset(
            valid_dset,
            n_classes=n_classes,
        )
    )

    print(
        f"  training examples   = {len(train_sequences)}"
        f"\n  validation examples = {len(valid_sequences)}",
        flush=True,
    )

    print(
        "\n[2/4] Extracting training representations",
        flush=True,
    )

    start = time.perf_counter()

    X_train_raw = extractor(
        train_sequences
    )

    train_extraction_seconds = (
        time.perf_counter() - start
    )

    print(
        f"Training extraction finished in "
        f"{train_extraction_seconds:.2f} seconds.",
        flush=True,
    )

    print(
        "\n[3/4] Extracting validation representations",
        flush=True,
    )

    start = time.perf_counter()

    X_valid_raw = extractor(
        valid_sequences
    )

    valid_extraction_seconds = (
        time.perf_counter() - start
    )

    print(
        f"Validation extraction finished in "
        f"{valid_extraction_seconds:.2f} seconds.",
        flush=True,
    )

    X_train = to_real_feature_matrix(
        X_train_raw
    )

    X_valid = to_real_feature_matrix(
        X_valid_raw
    )

    print(
        "\nRepresentation matrices:"
        f"\n  X_train = {X_train.shape}"
        f"\n  X_valid = {X_valid.shape}",
        flush=True,
    )

    print(
        "\n[4/4] Starting SVM cross-validation",
        flush=True,
    )

    start = time.perf_counter()

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
        sample_weight_mode=sample_weight_mode,
        seed=seed,
        cv_splits=cv_splits,
        n_jobs=svm_n_jobs,
        **svm_kwargs,
    )

    svm_seconds = time.perf_counter() - start

    print(
        f"\nSVM training completed in "
        f"{svm_seconds / 60.0:.2f} minutes.",
        flush=True,
    )

    result.update({
        "representation_name": representation_name,
        "n_features": int(X_train.shape[1]),
        "feature_extraction_seconds": (
            train_extraction_seconds
            + valid_extraction_seconds
        ),
        "train_extraction_seconds": (
            train_extraction_seconds
        ),
        "valid_extraction_seconds": (
            valid_extraction_seconds
        ),
        "svm_seconds": svm_seconds,
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
mPath = '..\models\\'
mPath = 'C:\V\Projects\RepComp\models\\'


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
        show_progress=True,
    )
)

transformer_result = run_representation_svm(
    representation_name=(
        "Causal Transformer — final last token"
    ),
    extractor=transformer_extractor,
    train_dset=training ,
    valid_dset=validation,
    n_classes=3,
    kernels=("linear", "rbf"),
    standardize=True,
    class_weight="balanced",
    sample_weight_mode=None,
    seed=1234,
    cv_splits=5,
    svm_n_jobs=-1,  # -1
)


print_representation_svm_report(
    transformer_result
)