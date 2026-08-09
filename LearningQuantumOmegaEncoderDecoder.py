# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:17:45 2026

@author: vanio
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import pickle
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
import time
import contextlib


try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
import numpy as np
from typing import Literal, Optional, Tuple, List

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
from plot_distributions import plotDistribution, plotDistributions

import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

#from processing_results import ExperimentTracker
#-----------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------
# Joint trining Data
#------------------------------------------------------------------------------
def extract_ensemble_arrays(joint_data, component_data=None):
    if "X" in joint_data:
        # Direct-sequence representation:
        # [N_joint, n_channels, k]
        X_joint = np.asarray(
            joint_data["X"],
            dtype=np.int16,
        )

    elif "local_id_vectors" in joint_data:
        if component_data is None:
            raise ValueError(
                "component_data is required when joint_data "
                "contains local dictionary indices."
            )

        local_ids = np.asarray(
            joint_data["local_id_vectors"],
            dtype=np.int64,
        )

        X_by_channel = []

        for m in range(local_ids.shape[1]):
            local_sequences = np.asarray(
                component_data[m]["sequences"],
                dtype=np.int16,
            )

            X_by_channel.append(
                local_sequences[local_ids[:, m]]
            )

        X_joint = np.stack(
            X_by_channel,
            axis=1,
        )

    else:
        raise KeyError(
            "joint_data contains neither 'X' nor "
            "'local_id_vectors'. Available keys are: "
            f"{list(joint_data.keys())}"
        )

    X_by_channel = [
        X_joint[:, m, :]
        for m in range(X_joint.shape[1])
    ]

    Y_dist = np.asarray(
        joint_data["target_distributions"],
        dtype=np.float32,
    )

    counts = np.asarray(
        joint_data["counts"],
        dtype=np.int64,
    )

    weights = np.asarray(
        joint_data["seq_probs"],
        dtype=np.float64,
    )

    class_values = np.asarray(
        joint_data.get("class_values", (-1, 0, 1))
    )

    Y_class = class_values[
        np.argmax(Y_dist, axis=1)
    ]

    return {
        "X_joint": X_joint,
        "X_by_channel": X_by_channel,
        "Y_dist": Y_dist,
        "Y_class": Y_class,
        "counts": counts,
        "weights": weights,
    }

import numpy as np


def get_channel_tdata(
    training_data,
    channel: int,
    preserve_order: bool = True,
    eps: float = 1e-12,
):
    """
    Extract and aggregate training data for one channel from joint ensemble data.

    Parameters
    ----------
    training_data
        Dictionary containing at least:

        - "X_joint":
            shape [N_joint, n_channels, sequence_length]

        - "Y_dist":
            shape [N_joint, n_classes]

        and preferably:

        - "counts":
            occurrence count of each unique joint sequence

        Optionally:

        - "class_counts":
            exact class counts for each joint sequence,
            shape [N_joint, n_classes]

        - "class_values":
            labels corresponding to the class-distribution columns

    channel
        Channel index, from 0 to n_channels - 1.

    preserve_order
        If True, unique local sequences are returned in order of first
        appearance in X_joint.

    Returns
    -------
    Dictionary containing:

        "X":
            unique local sequences, shape [N_unique, sequence_length]

        "target_distributions":
            aggregated p(class | local sequence),
            shape [N_unique, n_classes]

        "counts":
            total support of each local sequence

        "class_counts":
            aggregated class mass/counts

        "seq_probs":
            empirical probability of each local sequence

        "joint_to_local":
            maps each row of X_joint to its local-sequence row

        "class_values":
            class labels
    """
    X_joint = np.asarray(training_data["X_joint"])
    Y_dist = np.asarray(
        training_data["Y_dist"],
        dtype=np.float64,
    )

    if X_joint.ndim != 3:
        raise ValueError(
            "X_joint must have shape "
            "[N_joint, n_channels, sequence_length]."
        )

    n_joint, n_channels, _ = X_joint.shape

    if not 0 <= channel < n_channels:
        raise IndexError(
            f"channel must be in [0, {n_channels - 1}], "
            f"received {channel}."
        )

    if Y_dist.shape[0] != n_joint:
        raise ValueError(
            "X_joint and Y_dist have different numbers of rows."
        )

    # One local sequence for every joint-sequence row.
    X_channel = np.asarray(
        X_joint[:, channel, :],
        dtype=np.int16,
    )

    # Identify unique local sequences.
    unique_X, first_indices, inverse = np.unique(
        X_channel,
        axis=0,
        return_index=True,
        return_inverse=True,
    )

    n_unique = len(unique_X)
    n_classes = Y_dist.shape[1]

    # Use exact joint class counts when available.
    if "class_counts" in training_data:
        joint_class_counts = np.asarray(
            training_data["class_counts"],
            dtype=np.float64,
        )

        if joint_class_counts.shape != Y_dist.shape:
            raise ValueError(
                "class_counts must have the same shape as Y_dist."
            )

        if "counts" in training_data:
            joint_counts = np.asarray(
                training_data["counts"],
                dtype=np.float64,
            )
        else:
            joint_counts = joint_class_counts.sum(axis=1)

    else:
        if "counts" in training_data:
            joint_counts = np.asarray(
                training_data["counts"],
                dtype=np.float64,
            )
        elif "weights" in training_data:
            # Relative weights are sufficient for the conditional
            # distributions, although they are not integer support counts.
            joint_counts = np.asarray(
                training_data["weights"],
                dtype=np.float64,
            )
        else:
            raise KeyError(
                "training_data must contain either 'counts' or 'weights'."
            )

        # Reconstruct class mass from p(class | joint sequence).
        joint_class_counts = (
            joint_counts[:, None] * Y_dist
        )

    if joint_counts.shape != (n_joint,):
        raise ValueError(
            "counts must have shape [N_joint]."
        )

    local_counts = np.zeros(
        n_unique,
        dtype=np.float64,
    )

    local_class_counts = np.zeros(
        (n_unique, n_classes),
        dtype=np.float64,
    )

    # Sum over all joint rows containing the same local sequence.
    np.add.at(
        local_counts,
        inverse,
        joint_counts,
    )

    np.add.at(
        local_class_counts,
        inverse,
        joint_class_counts,
    )

    target_distributions = (
        local_class_counts
        / np.maximum(local_counts[:, None], eps)
    )

    # np.unique sorts lexicographically. Optionally restore order
    # of first appearance.
    if preserve_order:
        order = np.argsort(first_indices)

        unique_X = unique_X[order]
        local_counts = local_counts[order]
        local_class_counts = local_class_counts[order]
        target_distributions = target_distributions[order]

        # Remap old unique indices to the reordered indices.
        old_to_new = np.empty_like(order)
        old_to_new[order] = np.arange(n_unique)

        inverse = old_to_new[inverse]

    seq_probs = local_counts / local_counts.sum()

    class_values = np.asarray(
        training_data.get(
            "class_values",
            np.arange(n_classes),
        )
    )

    dominant_classes = class_values[
        np.argmax(target_distributions, axis=1)
    ]

    return {
        "X": unique_X,
        "target_distributions": target_distributions.astype(
            np.float32
        ),
        "dominant_classes": dominant_classes,
        "counts": local_counts,
        "class_counts": local_class_counts,
        "seq_probs": seq_probs,
        "joint_to_local": inverse,
        "class_values": class_values,
        "channel": channel,
    }
#------------------------------------------------------------------------------
def save_predictive_model(path, pred_model, meta=None):
    """
    Save predictive model properly
    
    Args:
        path: file path (e.g., 'model.pt')
        pred_model: PredictiveQuantumModel instance
        meta: optional metadata dict
    """
    payload = {
        # Model states
        'encoder_state': pred_model.encoder.state_dict(),
        'decoder_state': pred_model.decoder.state_dict(),
        
        # Model configurations (needed for reconstruction)
        'encoder_config': {
            'm': pred_model.encoder.m,
            'd': pred_model.encoder.d,
            'learn_rho0': pred_model.encoder.learn_rho0,
            'rho0_type': pred_model.encoder.rho0_type,
            'eps': pred_model.encoder.eps,
        },
        'decoder_config': {
            'd_in': pred_model.decoder.d_in,
            'd_out': pred_model.decoder.d_out,
            'use_unitary': pred_model.decoder.use_unitary,
            'eps': pred_model.decoder.eps,
        },
        
        # Optional metadata
        'meta': meta or {},
    }
    
    torch.save(payload, path)
    print(f"✓ Model saved to {path}")


def load_predictive_model(path, device="cpu"):
    """
    Load predictive model from file
    
    Args:
        path: file path
        device: 'cpu' or 'cuda'
    
    Returns:
        pred_model: PredictiveQuantumModel
        meta: metadata dict
    """
    payload = torch.load(path, map_location=device)
    
    # Reconstruct encoder
    enc_cfg = payload['encoder_config']
    encoder = KrausInstrument(
        m=enc_cfg['m'],
        d=enc_cfg['d'],
        learn_rho0=enc_cfg['learn_rho0'],
        rho0_type=enc_cfg['rho0_type'],
        eps=enc_cfg.get('eps', 1e-8)
    )
    encoder.load_state_dict(payload['encoder_state'])
    
    # Reconstruct decoder
    dec_cfg = payload['decoder_config']
    decoder = QuantumDecoder(
        d_in=dec_cfg['d_in'],
        d_out=dec_cfg['d_out'],
        use_unitary=dec_cfg['use_unitary'],
        eps=dec_cfg.get('eps', 1e-8)
    )
    decoder.load_state_dict(payload['decoder_state'])
    
    # Reconstruct full model
    pred_model = PredictiveQuantumModel(encoder, decoder, freeze_encoder=False)
    pred_model = pred_model.to(device)
    pred_model.eval()
    
    print(f"✓ Model loaded from {path}")
    
    return pred_model, payload.get('meta', {})



#------------------------------------------------------------------------------
# Interference and Utilities
@torch.no_grad()

def predict_single_sequence(model, sequence, device="cpu"):
    """
    Predict target distribution for a single sequence
    
    sequence: list[int] or tensor
    returns: numpy array [d_out] of probabilities
    """
    model.eval()
    
    if not torch.is_tensor(sequence):
        seq_t = torch.tensor([sequence], dtype=torch.long)  # [1, T]
    else:
        seq_t = sequence.unsqueeze(0) if sequence.dim() == 1 else sequence
    
    seq_t = seq_t.to(device)
    probs = model(seq_t)  # [1, d_out]
    
    return probs[0].cpu().numpy()


@torch.no_grad()
def get_embedding(model, sequence, device="cpu"):
    """
    Get the encoded density matrix for a sequence
    
    sequence: list[int] or tensor
    returns: numpy array [d, d] complex density matrix
    """
    model.eval()
    
    if not torch.is_tensor(sequence):
        seq_t = torch.tensor([sequence], dtype=torch.long)
    else:
        seq_t = sequence.unsqueeze(0) if sequence.dim() == 1 else sequence
    
    seq_t = seq_t.to(device)
    rho = model.encode_sequences(seq_t)  # [1, d, d]
    
    return rho[0].cpu().numpy()


def save_predictive_model(path, model, meta=None):
    """Save complete predictive model (encoder + decoder)"""
    payload = {
        "encoder_state": model.encoder.state_dict(),
        "decoder_state": model.decoder.state_dict(),
        "encoder_config": {
            "m": model.encoder.m,
            "d": model.encoder.d,
            "learn_rho0": model.encoder.learn_rho0,
        },
        "decoder_config": {
            "d_in": model.decoder.d_in,
            "d_out": model.decoder.d_out,
            "use_unitary": model.decoder.use_unitary,
        },
        "meta": meta or {},
    }
    torch.save(payload, path)
    print(f"Model saved to {path}")


def load_predictive_model(path, device="cpu"):
    """Load complete predictive model"""
    payload = torch.load(path, map_location=device)
    
    # Reconstruct encoder
    enc_cfg = payload["encoder_config"]
    encoder = KrausInstrument(
        m=enc_cfg["m"],
        d=enc_cfg["d"],
        learn_rho0=enc_cfg["learn_rho0"]
    )
    encoder.load_state_dict(payload["encoder_state"])
    
    # Reconstruct decoder
    dec_cfg = payload["decoder_config"]
    decoder = QuantumDecoder(
        d_in=dec_cfg["d_in"],
        d_out=dec_cfg["d_out"],
        use_unitary=dec_cfg["use_unitary"]
    )
    decoder.load_state_dict(payload["decoder_state"])
    
    # Combine
    model = PredictiveQuantumModel(encoder, decoder, freeze_encoder=False)
    model = model.to(device)
    
    return model, payload.get("meta", {})

class DynamicLossWeights:
    def __init__(self, alpha=0.99):
        """
        Dynamically balance encoder and prediction losses
        
        alpha: EMA smoothing factor (higher = slower adaptation)
        """
        self.alpha = alpha
        self.enc_ema = None
        self.pred_ema = None
    
    def update(self, enc_loss, pred_loss):
        """Update running averages of loss magnitudes"""
        enc_val = float(enc_loss.detach())
        pred_val = float(pred_loss.detach())
        
        if self.enc_ema is None:
            # Initialize
            self.enc_ema = enc_val
            self.pred_ema = pred_val
        else:
            # Exponential moving average
            self.enc_ema = self.alpha * self.enc_ema + (1 - self.alpha) * enc_val
            self.pred_ema = self.alpha * self.pred_ema + (1 - self.alpha) * pred_val
    
    def get_weights(self):
        """
        Return normalized weights such that weighted losses have similar scales
        
        Returns:
            lambda_enc, lambda_pred
        """
        if self.enc_ema is None or self.pred_ema is None:
            return 1.0, 1.0
        
        # Inverse of loss magnitude → larger loss gets smaller weight
        # This balances the contribution of each loss
        enc_weight = 1.0 / (self.enc_ema + 1e-8)
        pred_weight = 1.0 / (self.pred_ema + 1e-8)
        
        # Normalize so they sum to 2.0 (for interpretability)
        total = enc_weight + pred_weight
        lambda_enc = 2.0 * enc_weight / total
        lambda_pred = 2.0 * pred_weight / total
        
        return lambda_enc, lambda_pred



#------------------------------------------------------------------------------
def save_model_weights(path, model, meta=None):
    payload = {
        "model_state": model.state_dict(),
        "meta": meta or {},
    }
    torch.save(payload, path)

def read_model_weights(path, map_location="cpu"):    
    payload = torch.load(path, map_location=map_location)
    sd = payload["model_state"]  # state_dict: name -> tensor
    return sd
#------------------------------------------------------------------------------
def load_model_weights(path, m, n_qubits, learn_rho0=True, device="cpu"):
    d = 2 ** n_qubits
    model = KrausInstrument(m=m, d=d, learn_rho0=learn_rho0).to(device)

    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state"], strict=True)

    meta = payload.get("meta", {})
    return model, meta

#------------------------------------------------------------------------------
def predict_probs(model, sequences, batch_size=2048, device=None):
    """
    sequences: list[list[int]] (ragged lengths allowed)
    returns: list[float] probabilities in the same order
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    probs_out = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            B = len(batch)
            T = max(len(s) for s in batch)

            seq_pad = torch.full((B, T), PAD, dtype=torch.long, device=device)
            for j, s in enumerate(batch):
                seq_pad[j, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)

            p = model.sequence_prob_batch(seq_pad)  # FloatTensor [B]
            probs_out.extend(p.detach().cpu().tolist())

    return probs_out
# ---------------------------
# Dataset Preparation
# -----------------------------------------------------------------------------
# integrate sequences probabilities and sequences distributions
def integrate_data(class_distributions, distrs_samples):
    # Extract the actual data list from the first element of the wrapper
    sp = distrs_samples[0]   
    sd = class_distributions  

    # Fast, parallel extraction using list comprehensions
    sequences            = [item[0] for item in sp]
    seq_probs            = [item[1] for item in sp]
    target_distributions = [item[2] for item in sd]

    return sequences, target_distributions, seq_probs
#------------------------------------------------------------------------------
PAD = -1  # must be outside symbol range {0,...,m-1}


class PredictiveSeqDataset(Dataset):
    def __init__(self, sequences, targets, emp_probs=None):
        self.sequences = sequences
        self.targets = targets
        self.emp_probs = emp_probs if emp_probs is not None else [1.0]*len(sequences)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.emp_probs[idx]

def collate_predictive(batch):
    seqs, targets, probs = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    T = int(lens.max())
    
    seq_pad = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        seq_pad[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    
    targets = torch.tensor(targets, dtype=torch.long)
    probs = torch.tensor(probs, dtype=torch.float32)
    
    return seq_pad, lens, targets, probs

class SeqDataset(Dataset):
    def __init__(self, sequences, emp_probs):
        self.sequences = sequences              # list[list[int]] lengths 1..7
        self.emp_probs = [float(x) for x in emp_probs]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.emp_probs[idx]

def collate_pad(batch):
    seqs, probs = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    T = int(lens.max())

    seq_pad = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        seq_pad[i, :len(s)] = torch.tensor(s, dtype=torch.long)

    probs = torch.tensor(probs, dtype=torch.float32)
    return seq_pad, lens, probs


class QuantumDecoder(nn.Module):
    def __init__(
        self, 
        d_in: int, 
        d_out: int, 
        use_unitary: bool = True,
        normalization_point: str = "input",  # NEW: "input", "after_unitary", or "output"
        eps: float = 1e-8
    ):
        """
        Args:
            normalization_point: where to normalize density matrix
                - "input": normalize right after encoder (before unitary)
                - "after_unitary": normalize after unitary, before co-isometry
                - "output": normalize only at final output (Option C - current)
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.use_unitary = use_unitary
        self.normalization_point = normalization_point.lower()
        self.eps = eps
        
        # Validate normalization_point
        valid_points = ["input", "after_unitary", "output"]
        if self.normalization_point not in valid_points:
            raise ValueError(f"normalization_point must be one of {valid_points}, "
                           f"got '{normalization_point}'")
        
        if use_unitary:
            self.U_re = nn.Parameter(torch.randn(d_in, d_in) * 0.01)
            self.U_im = nn.Parameter(torch.randn(d_in, d_in) * 0.01)
        
        self.V_re = nn.Parameter(torch.randn(d_in, d_out) * 0.01)
        self.V_im = nn.Parameter(torch.randn(d_in, d_out) * 0.01)
    
    def get_unitary(self):
        if not self.use_unitary:
            return torch.eye(self.d_in, dtype=torch.complex64, device=self.U_re.device)
        
        A = torch.complex(self.U_re, self.U_im)
        G = A.conj().T @ A
        G = 0.5 * (G + G.conj().T)
        
        w, Q = torch.linalg.eigh(G)
        w = torch.clamp(w, min=self.eps)
        inv_sqrt = (Q * w.rsqrt()) @ Q.conj().T
        
        U = A @ inv_sqrt
        return U
    
    def get_coisometry(self):
        V = torch.complex(self.V_re, self.V_im)
        G = V.conj().T @ V
        G = 0.5 * (G + G.conj().T)
        
        w, Q = torch.linalg.eigh(G)
        w = torch.clamp(w, min=self.eps)
        inv_sqrt = (Q * w.rsqrt()) @ Q.conj().T
        
        V_normalized = V @ inv_sqrt
        return V_normalized
    
    def _normalize_density_matrix(self, rho_batch):
        """
        Normalize density matrices: ρ → ρ / Tr(ρ)
        
        Args:
            rho_batch: [B, d, d] density matrices
        
        Returns:
            rho_normalized: [B, d, d] normalized density matrices
        """
        # Compute traces
        traces = torch.real(torch.diagonal(rho_batch, dim1=-2, dim2=-1).sum(-1))  # [B]
        traces = torch.clamp(traces, min=self.eps)
        
        # Normalize
        rho_normalized = rho_batch / traces.unsqueeze(-1).unsqueeze(-1)
        
        return rho_normalized
    
    def forward(self, rho_batch):
        """
        rho_batch: [B, d_in, d_in] batch of density matrices (may be unnormalized)
        returns: [B, d_out] unnormalized prediction logits
        """
        U = self.get_unitary()          # [d_in, d_in]
        V = self.get_coisometry()       # [d_in, d_out]
        
        # === NORMALIZATION POINT: INPUT ===
        if self.normalization_point == "input":
            rho_batch = self._normalize_density_matrix(rho_batch)
        
        # Apply unitary: rho' = U rho U†
        rho_rot = torch.bmm(
            torch.bmm(U.unsqueeze(0).expand(rho_batch.size(0), -1, -1), rho_batch),
            U.conj().T.unsqueeze(0).expand(rho_batch.size(0), -1, -1)
        )
        
        # === NORMALIZATION POINT: AFTER_UNITARY ===
        if self.normalization_point == "after_unitary":
            rho_rot = self._normalize_density_matrix(rho_rot)
        
        # Apply co-isometry: rho_pred = V† rho' V
        rho_pred = torch.bmm(
            torch.bmm(V.conj().T.unsqueeze(0).expand(rho_rot.size(0), -1, -1), rho_rot),
            V.unsqueeze(0).expand(rho_rot.size(0), -1, -1)
        )
        
        # Extract diagonal (prediction logits)
        logits = torch.real(torch.diagonal(rho_pred, dim1=-2, dim2=-1))
        
        # === NORMALIZATION POINT: OUTPUT (Option C - default) ===
        if self.normalization_point == "output":
            # Don't normalize here - done in predict_probs
            pass
        
        return logits  # [B, d_out]
    
    def predict_probs(self, rho_batch):
        """
        Returns normalized probabilities over d_out outcomes
        """
        logits = self.forward(rho_batch)
        logits = torch.clamp(logits, min=0.0)
        
        # Always normalize at output for probabilities
        probs = logits / (logits.sum(dim=-1, keepdim=True) + self.eps)
        
        return probs

#------------------------------------------------------------------------------
#  Quantum OMEGA Decoder
#------------------------------------------------------------------------------


def batch_psd_sqrt(
    rho: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Hermitian PSD square root.

    rho:
        (..., d, d) complex tensor
    """
    rho = 0.5 * (
        rho + rho.conj().transpose(-2, -1)
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(rho)

    eigenvalues = torch.clamp(
        eigenvalues.real,
        min=0.0,
    )

    sqrt_eigenvalues = torch.sqrt(
        eigenvalues + eps
    )

    return (
        eigenvectors
        * sqrt_eigenvalues.unsqueeze(-2)
    ) @ eigenvectors.conj().transpose(-2, -1)


def omega_vector_from_path(
    K: torch.Tensor,
    rho0: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Construct normalized Omega vectors:

        omega = vec(K sqrt(rho0)) / sqrt(p_sequence)

    Parameters
    ----------
    K:
        (B, d, d), or (d, d)

    rho0:
        (B, d, d), or (d, d)

    Returns
    -------
    omega:
        (B, d*d), complex

    p_sequence:
        (B,), real
    """
    if K.ndim == 2:
        K = K.unsqueeze(0)

    if rho0.ndim == 2:
        rho0 = rho0.unsqueeze(0)

    if rho0.shape[0] == 1 and K.shape[0] > 1:
        rho0 = rho0.expand(
            K.shape[0],
            -1,
            -1,
        )

    sqrt_rho0 = batch_psd_sqrt(
        rho0,
        eps=eps,
    )

    purification_operator = (
        K @ sqrt_rho0
    )

    # Column-major vectorization:
    # vec_F(X) = flatten(X^T) in PyTorch.
    omega_unnormalized = (
        purification_operator
        .transpose(-2, -1)
        .reshape(K.shape[0], -1)
    )

    p_sequence = torch.sum(
        torch.abs(omega_unnormalized) ** 2,
        dim=-1,
    ).real

    omega = omega_unnormalized / torch.sqrt(
        p_sequence.clamp_min(eps)
    ).unsqueeze(-1)

    return omega, p_sequence
#------------------------------------------------------------------------------
class ComplexHouseholderUnitary(nn.Module):
    """
    Structured complex unitary represented by:

        diagonal phases
        followed by several Householder reflections.

    Complexity is O(B * D * n_reflections), not O(B * D^2).
    """

    def __init__(
        self,
        dimension: int,
        n_reflections: int = 16,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.dimension = dimension
        self.n_reflections = n_reflections
        self.eps = eps

        scale = 1.0 / math.sqrt(dimension)

        self.v_re = nn.Parameter(
            torch.randn(
                n_reflections,
                dimension,
            ) * scale
        )

        self.v_im = nn.Parameter(
            torch.randn(
                n_reflections,
                dimension,
            ) * scale
        )

        self.phases = nn.Parameter(
            torch.zeros(dimension)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, D), complex
        """
        phase = torch.exp(
            1j * self.phases
        ).to(dtype=x.dtype)

        z = x * phase.unsqueeze(0)

        for index in range(self.n_reflections):
            v = torch.complex(
                self.v_re[index],
                self.v_im[index],
            ).to(dtype=x.dtype)

            denominator = torch.sum(
                torch.abs(v) ** 2
            ).real.clamp_min(self.eps)

            # v^dagger z
            coefficient = torch.sum(
                z * v.conj().unsqueeze(0),
                dim=-1,
            ) / denominator

            z = (
                z
                - 2.0
                * coefficient.unsqueeze(-1)
                * v.unsqueeze(0)
            )

        return z
#------------------------------------------------------------------------------
class ComplexLowRankResidual(nn.Module):
    """
    General non-unitary transformation:

        A = I + alpha L R^dagger

    applied without explicitly constructing A.
    """

    def __init__(
        self,
        dimension: int,
        rank: int = 16,
        initial_alpha: float = 0.05,
    ):
        super().__init__()

        self.dimension = dimension
        self.rank = rank

        scale = 1.0 / math.sqrt(dimension)

        self.L_re = nn.Parameter(
            torch.randn(
                dimension,
                rank,
            ) * scale
        )
        self.L_im = nn.Parameter(
            torch.randn(
                dimension,
                rank,
            ) * scale
        )

        self.R_re = nn.Parameter(
            torch.randn(
                dimension,
                rank,
            ) * scale
        )
        self.R_im = nn.Parameter(
            torch.randn(
                dimension,
                rank,
            ) * scale
        )

        self.alpha = nn.Parameter(
            torch.tensor(
                float(initial_alpha)
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, D), complex

        Computes:
            x A^T
            = x + alpha (x R*) L^T
        """
        L = torch.complex(
            self.L_re,
            self.L_im,
        ).to(dtype=x.dtype)

        R = torch.complex(
            self.R_re,
            self.R_im,
        ).to(dtype=x.dtype)

        low_rank_coordinates = (
            x @ R.conj()
        )

        correction = (
            low_rank_coordinates @ L.transpose(0, 1)
        )

        return x + self.alpha * correction

#------------------------------------------------------------------------------
class ComplexCoisometry(nn.Module):
    """
    Learnable C with:

        C C^dagger = I_dout.

    Internally parameterizes Q in C^(D x d_out)
    with Q^dagger Q = I, then C = Q^dagger.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
    ):
        super().__init__()

        if d_out > d_in:
            raise ValueError(
                "d_out cannot exceed d_in "
                "for a co-isometry."
            )

        self.d_in = d_in
        self.d_out = d_out

        scale = 1.0 / math.sqrt(d_in)

        self.raw_re = nn.Parameter(
            torch.randn(
                d_in,
                d_out,
            ) * scale
        )

        self.raw_im = nn.Parameter(
            torch.randn(
                d_in,
                d_out,
            ) * scale
        )

    def matrix(self) -> torch.Tensor:
        raw = torch.complex(
            self.raw_re,
            self.raw_im,
        )

        # Q shape: (d_in, d_out)
        # Q^dagger Q = I
        Q, _ = torch.linalg.qr(
            raw,
            mode="reduced",
        )

        # C shape: (d_out, d_in)
        return Q.conj().transpose(0, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, d_in), complex
        returns: (B, d_out), complex
        """
        C = self.matrix().to(dtype=x.dtype)

        return torch.einsum(
            "od,bd->bo",
            C,
            x,
        )
#------------------------------------------------------------------------------
class OmegaQuantumDecoder(nn.Module):
    """
    Decoder operating on the Omega-vector representation.

    Architecture:

        omega
          -> optional structured unitary
          -> optional low-rank general map
          -> co-isometry
          -> normalized output density matrix
    """

    def __init__(
        self,
        omega_dimension: int,
        d_out: int,
        *,
        use_unitary: bool = True,
        n_reflections: int = 16,
        use_general_map: bool = True,
        general_rank: int = 16,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.omega_dimension = omega_dimension
        self.d_out = d_out
        self.d_in = omega_dimension
        self.eps = eps

        if use_unitary:
            self.unitary = ComplexHouseholderUnitary(
                dimension=omega_dimension,
                n_reflections=n_reflections,
                eps=eps,
            )
        else:
            self.unitary = nn.Identity()

        if use_general_map:
            self.general_map = ComplexLowRankResidual(
                dimension=omega_dimension,
                rank=general_rank,
            )
        else:
            self.general_map = nn.Identity()

        self.coisometry = ComplexCoisometry(
            d_in=omega_dimension,
            d_out=d_out,
        )

    def forward(
        self,
        omega: torch.Tensor,
    ):
        """
        omega:
            (B, omega_dimension), complex

        Returns
        -------
        rho_out:
            (B, d_out, d_out), complex

        probabilities:
            (B, d_out), real
        """
        if omega.ndim != 2:
            raise ValueError(
                "omega must have shape (batch, dimension)."
            )

        # Numerically enforce normalized input.
        omega = omega / torch.linalg.vector_norm(
            omega,
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

        z = self.unitary(omega)
        z = self.general_map(z)

        amplitudes = self.coisometry(z)

        output_mass = torch.sum(
            torch.abs(amplitudes) ** 2,
            dim=-1,
        ).real.clamp_min(self.eps)

        probabilities = (
            torch.abs(amplitudes) ** 2
        ) / output_mass.unsqueeze(-1)

        rho_subnormalized = (
            amplitudes.unsqueeze(-1)
            @ amplitudes.conj().unsqueeze(-2)
        )

        rho_out = (
            rho_subnormalized
            / output_mass[:, None, None]
        )

        return rho_out, probabilities
#------------------------------------------------------------------------------
# Encoder–decoder wrapper
#------------------------------------------------------------------------------
class OmegaPredictiveQuantumModel(nn.Module):
    def __init__(
        self,
        encoder,
        decoder: OmegaQuantumDecoder,
        *,
        freeze_encoder: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.freeze_encoder = freeze_encoder
        self.eps = eps

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def _as_complex_tensor(
        self,
        x,
        device,
    ):
        if torch.is_tensor(x):
            return x.to(
                device=device,
                dtype=torch.complex64,
            )

        if not self.freeze_encoder:
            raise TypeError(
                "A trainable encoder must return PyTorch "
                "tensors from path_operator. NumPy conversion "
                "would break the gradient graph."
            )

        return torch.as_tensor(
            x,
            dtype=torch.complex64,
            device=device,
        )

    def forward(
        self,
        sequences,
    ):
        device = next(
            self.decoder.parameters()
        ).device

        K_values = []
        rho0_values = []

        context = (
            torch.no_grad()
            if self.freeze_encoder
            else contextlib.nullcontext()
        )

        with context:
            for sequence in sequences:
                output = self.encoder.path_operator(
                    list(sequence),
                    return_prob=True,
                    device=device,
                )

                if output is None:
                    raise RuntimeError(
                        "path_operator returned None for "
                        f"sequence {sequence}."
                    )

                K_sequence, rho0, _ = output

                K_values.append(
                    self._as_complex_tensor(
                        K_sequence,
                        device,
                    )
                )

                rho0_values.append(
                    self._as_complex_tensor(
                        rho0,
                        device,
                    )
                )

            K_batch = torch.stack(
                K_values,
                dim=0,
            )

            rho0_batch = torch.stack(
                rho0_values,
                dim=0,
            )

            omega, sequence_probabilities = (
                omega_vector_from_path(
                    K=K_batch,
                    rho0=rho0_batch,
                    eps=self.eps,
                )
            )

        rho_prediction, prediction = self.decoder(
            omega
        )

        return {
            "probabilities": prediction,
            "rho_out": rho_prediction,
            "omega": omega,
            "sequence_probabilities": (
                sequence_probabilities
            ),
        }

#------------------------------------------------------------------------------
# Encoder
#------------------------------------------------------------------------------

def sequences_to_ragged(
    sequences,
    *,
    pad_value: int,
):
    """
    Convert either:

        list[list[int]]

    or:

        padded LongTensor[B, T]

    into a list of unpadded Python sequences.

    PAD is assumed to occur only after the final valid symbol.
    """
    if torch.is_tensor(sequences):
        if sequences.ndim != 2:
            raise ValueError(
                "A padded sequence tensor must have shape (B, T); "
                f"received {tuple(sequences.shape)}."
            )

        rows = sequences.detach().cpu().tolist()
        ragged_sequences = []

        for row_index, row in enumerate(rows):
            try:
                end = row.index(pad_value)
            except ValueError:
                end = len(row)

            # Validate right-padding.
            if any(symbol != pad_value for symbol in row[end:]):
                raise ValueError(
                    "Non-PAD symbol found after the first PAD at "
                    f"batch row {row_index}."
                )

            sequence = [
                int(symbol)
                for symbol in row[:end]
            ]

            if len(sequence) == 0:
                raise ValueError(
                    f"Empty sequence found at batch row {row_index}."
                )

            ragged_sequences.append(sequence)

        return ragged_sequences

    # Already ragged.
    ragged_sequences = []

    for sequence_index, sequence in enumerate(sequences):
        sequence = [
            int(symbol)
            for symbol in sequence
        ]

        # Also tolerate padded Python lists.
        if pad_value in sequence:
            end = sequence.index(pad_value)

            if any(
                symbol != pad_value
                for symbol in sequence[end:]
            ):
                raise ValueError(
                    "Non-PAD symbol found after the first PAD in "
                    f"sequence {sequence_index}."
                )

            sequence = sequence[:end]

        if len(sequence) == 0:
            raise ValueError(
                f"Empty sequence found at index {sequence_index}."
            )

        ragged_sequences.append(sequence)

    return ragged_sequences
#-----------------------------------------------------------------------------
import torch.nn.functional as F


def normalize_distributions(
    probabilities: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    probabilities = probabilities.clamp_min(0.0)

    return probabilities / probabilities.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)


def prediction_loss_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Returns one loss value per sequence.

    prediction, target:
        shape (B, d_out)

    output:
        shape (B,)
    """
    prediction = normalize_distributions(
        prediction,
        eps=eps,
    )

    target = normalize_distributions(
        target,
        eps=eps,
    )

    name = loss_type.lower().strip()

    if name in {
        "mse",
        "mseerror",
        "mse_error",
    }:
        return torch.mean(
            (prediction - target) ** 2,
            dim=-1,
        )

    if name in {
        "xentropy",
        "crossentropy",
        "cross_entropy",
        "ce",
    }:
        return -torch.sum(
            target * torch.log(
                prediction.clamp_min(eps)
            ),
            dim=-1,
        )

    if name in {
        "kldivergence",
        "kl",
        "kl_divergence",
    }:
        return torch.sum(
            target
            * (
                torch.log(target.clamp_min(eps))
                - torch.log(prediction.clamp_min(eps))
            ),
            dim=-1,
        )

    if name in {
        "jsdivergence",
        "js",
        "js_divergence",
    }:
        midpoint = 0.5 * (
            target + prediction
        )

        kl_target = torch.sum(
            target
            * (
                torch.log(target.clamp_min(eps))
                - torch.log(midpoint.clamp_min(eps))
            ),
            dim=-1,
        )

        kl_prediction = torch.sum(
            prediction
            * (
                torch.log(prediction.clamp_min(eps))
                - torch.log(midpoint.clamp_min(eps))
            ),
            dim=-1,
        )

        return 0.5 * (
            kl_target + kl_prediction
        )

    raise ValueError(
        f"Unknown prediction loss: {loss_type}"
    )


def encoder_loss_per_sample(
    model_probabilities: torch.Tensor,
    empirical_probabilities: torch.Tensor,
    loss_type: str = "mse",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compare p_model(sequence) with p_empirical(sequence).

    Returns shape (B,).
    """
    model_probabilities = (
        model_probabilities.real
        .reshape(-1)
        .clamp_min(eps)
    )

    empirical_probabilities = (
        empirical_probabilities
        .reshape(-1)
        .clamp_min(eps)
    )

    name = loss_type.lower().strip()

    if name in {
        "mse",
        "mseerror",
        "mse_error",
    }:
        return (
            model_probabilities
            - empirical_probabilities
        ) ** 2

    if name in {
        "logmse",
        "log_mse",
    }:
        return (
            torch.log(model_probabilities)
            - torch.log(empirical_probabilities)
        ) ** 2

    if name in {
        "relative_mse",
        "relativemse",
    }:
        return (
            (
                model_probabilities
                - empirical_probabilities
            )
            / empirical_probabilities.clamp_min(eps)
        ) ** 2

    raise ValueError(
        f"Unknown encoder loss: {loss_type}"
    )


def prepare_sample_weights(
    weights: torch.Tensor,
    batch_size: int,
    device,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return nonnegative sample weights of shape (B,).

    If all supplied weights are zero, uniform weights are used.
    """
    if weights is None:
        return torch.ones(
            batch_size,
            dtype=torch.float32,
            device=device,
        )

    weights = weights.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    ).reshape(-1)

    if weights.numel() != batch_size:
        raise ValueError(
            "Weight batch-size mismatch: "
            f"received {weights.numel()}, "
            f"expected {batch_size}."
        )

    weights = weights.clamp_min(0.0)

    if float(weights.sum().detach().cpu()) <= eps:
        weights = torch.ones_like(weights)

    return weights


def weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    values = values.reshape(-1)
    weights = weights.reshape(-1).to(values.dtype)

    return torch.sum(
        values * weights
    ) / weights.sum().clamp_min(eps)

#------------------------------------------------------------------------------
class OmegaRepresentationEncoder(nn.Module):
    """
    sequence -> normalized Omega vector

        omega(s) = vec(K_s sqrt(rho0)) / sqrt(p_s)
    """

    def __init__(
        self,
        encoder,
        pad_value: int,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.encoder = encoder
        self.pad_value = pad_value
        self.eps = eps

    def _psd_sqrt(
        self,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        rho = 0.5 * (
            rho + rho.conj().transpose(-2, -1)
        )

        eigenvalues, eigenvectors = torch.linalg.eigh(
            rho
        )

        eigenvalues = torch.clamp(
            eigenvalues.real,
            min=0.0,
        )

        sqrt_eigenvalues = torch.sqrt(
            eigenvalues + self.eps
        )

        return (
            eigenvectors
            * sqrt_eigenvalues.unsqueeze(-2)
        ) @ eigenvectors.conj().transpose(-2, -1)

    @staticmethod
    def _to_complex_tensor(
        value,
        device,
    ):
        if torch.is_tensor(value):
            return value.to(
                device=device,
                dtype=torch.complex64,
            )

        return torch.as_tensor(
            value,
            dtype=torch.complex64,
            device=device,
        )

    def forward(self, sequences):
        # NEW: accept either padded tensor or ragged lists.
        sequences = sequences_to_ragged(
            sequences,
            pad_value=self.pad_value,
        )

        device = next(
            self.encoder.parameters()
        ).device

        omega_values = []
        sequence_probabilities = []

        for sequence in sequences:
            result = self.encoder.path_operator(
                sequence,
                return_prob=True,
                device=device,
            )

            if result is None:
                raise RuntimeError(
                    "path_operator returned None for "
                    f"sequence {sequence}."
                )

            K_sequence, rho0, _ = result

            K_sequence = self._to_complex_tensor(
                K_sequence,
                device,
            )

            rho0 = self._to_complex_tensor(
                rho0,
                device,
            )

            sqrt_rho0 = self._psd_sqrt(
                rho0
            )

            amplitude_operator = (
                K_sequence @ sqrt_rho0
            )

            # Column-major vectorization.
            omega_raw = (
                amplitude_operator
                .transpose(-2, -1)
                .reshape(-1)
            )

            probability = torch.sum(
                torch.abs(omega_raw) ** 2
            ).real

            omega = omega_raw / torch.sqrt(
                probability.clamp_min(
                    self.eps
                )
            )

            omega_values.append(omega)
            sequence_probabilities.append(
                probability
            )

        return (
            torch.stack(
                omega_values,
                dim=0,
            ),
            torch.stack(
                sequence_probabilities,
                dim=0,
            ),
        )

    @staticmethod
    def _to_complex_tensor(
        value,
        device,
    ):
        if torch.is_tensor(value):
            return value.to(
                device=device,
                dtype=torch.complex64,
            )

        return torch.as_tensor(
            value,
            dtype=torch.complex64,
            device=device,
        )

    def forward(self, sequences):
        sequences = sequences_to_ragged(
            sequences,
            pad_value=self.pad_value,
            )
        device = next(
            self.encoder.parameters()
        ).device

        omega_values = []
        sequence_probabilities = []

        for sequence in sequences:
            result = self.encoder.path_operator(
                list(sequence),
                return_prob=True,
                device=device,
            )

            if result is None:
                raise RuntimeError(
                    "path_operator returned None for "
                    f"sequence {sequence}."
                )

            K_sequence, rho0, _ = result

            K_sequence = self._to_complex_tensor(
                K_sequence,
                device,
            )

            rho0 = self._to_complex_tensor(
                rho0,
                device,
            )

            sqrt_rho0 = self._psd_sqrt(rho0)

            amplitude_operator = (
                K_sequence @ sqrt_rho0
            )

            # Column-major vectorization:
            # vec_F(X) = flatten(X^T)
            omega_raw = (
                amplitude_operator
                .transpose(-2, -1)
                .reshape(-1)
            )

            probability = torch.sum(
                torch.abs(omega_raw) ** 2
            ).real

            omega = omega_raw / torch.sqrt(
                probability.clamp_min(self.eps)
            )

            omega_values.append(omega)
            sequence_probabilities.append(probability)

        return (
            torch.stack(omega_values, dim=0),
            torch.stack(sequence_probabilities, dim=0),
        )
#------------------------------------------------------------------------------
class OmegaPredictiveModel(nn.Module):
    """
    Drop-in predictive model using the Omega representation.

    Default forward interface:

        predicted_distributions, sequence_probabilities = model(sequences)

    This matches the original PredictiveQuantumModel interface.
    """

    def __init__(
        self,
        representation_encoder: nn.Module,
        decoder: nn.Module,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.representation_encoder = representation_encoder
        self.decoder = decoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for parameter in self.representation_encoder.parameters():
                parameter.requires_grad = False

    @property
    def encoder(self):
        """
        Backward-compatible access to the original Kraus encoder.

        Allows existing code such as:

            model.encoder.parameters()
        """
        return self.representation_encoder.encoder

    def forward(
        self,
        sequences,
        return_details: bool = False,
    ):
        """
        Parameters
        ----------
        sequences
            Either:
                padded LongTensor of shape (B, T), or
                ragged list of sequences.

        return_details
            When False, return the original two-object interface.

            When True, additionally expose Omega and the output density
            matrix for diagnostics.

        Returns
        -------
        Default:
            predicted_probabilities:
                Tensor of shape (B, d_out)

            sequence_probabilities:
                Tensor of shape (B,)

        With return_details=True:
            dictionary containing all intermediate values.
        """
        if self.freeze_encoder:
            with torch.no_grad():
                omega, sequence_probabilities = (
                    self.representation_encoder(sequences)
                )
        else:
            omega, sequence_probabilities = (
                self.representation_encoder(sequences)
            )

        # Decoder remains outside no_grad so that it can be trained
        # when the encoder is frozen.
        rho_prediction, predicted_probabilities = (
            self.decoder(omega)
        )

        sequence_probabilities = (
            sequence_probabilities.real.reshape(-1)
        )

        if return_details:
            return {
                "omega": omega,
                "sequence_probabilities": (
                    sequence_probabilities
                ),
                "rho_prediction": rho_prediction,
                "probabilities": predicted_probabilities,
            }

        # Backward-compatible interface.
        return (
            predicted_probabilities,
            sequence_probabilities,
        )
# -----------------------------------------------------------------------------
# Model: Kraus via stacked-isometry whitening
# ---------------------------
class KrausInstrument(nn.Module):
    def __init__(self, m: int, d: int, learn_rho0: bool = True, rho0_type: str = "mixed",  # NEW: "mixed" or "pure"
        eps: float = 1e-8
    ):
        """
        m: alphabet size
        d: system dimension
        learn_rho0: whether to learn initial state
        rho0_type: "mixed" (density matrix) or "pure" (state vector)
            - "mixed": learn as ρ₀ = L L† / Tr(L L†) [density matrix]
            - "pure": learn as |ψ⟩, then ρ₀ = |ψ⟩⟨ψ| [pure state]
        """
        super().__init__()
        self.m = m
        self.d = d
        self.eps = eps
        self.learn_rho0 = learn_rho0
        self.rho0_type = rho0_type.lower()
        
        if self.rho0_type not in ["mixed", "pure"]:
            raise ValueError(f"rho0_type must be 'mixed' or 'pure', got {rho0_type}")

        # Kraus operators: unconstrained A matrix
        # Unconstrained complex A in R^{2} via (real, imag)
        # Shape: (m*d, d)
        self.A_re = nn.Parameter(torch.randn(m * d, d) * 0.01)
        self.A_im = nn.Parameter(torch.randn(m * d, d) * 0.01)

        # Initial state parameterization
        if learn_rho0:
            if self.rho0_type == "mixed":
                # Learn ρ₀ via Cholesky factor L
                # ρ₀ = L L† / Tr(L L†)
                self.L_re = nn.Parameter(torch.randn(d, d) * 0.01)
                self.L_im = nn.Parameter(torch.randn(d, d) * 0.01)
                
            elif self.rho0_type == "pure":
                # Learn |ψ⟩ as complex state vector
                # ρ₀ = |ψ⟩⟨ψ|
                self.psi_re = nn.Parameter(torch.randn(d) * 0.01)
                self.psi_im = nn.Parameter(torch.randn(d) * 0.01)

    def _make_rho0(self, device):
        """
        Construct initial density matrix ρ₀
        
        Returns:
            rho0: [d, d] complex density matrix satisfying Tr(ρ₀) = 1
        """
        d = self.d
        
        if not self.learn_rho0:
            # Fixed |0⟩⟨0|
            rho0 = torch.zeros(d, d, dtype=torch.complex64, device=device)
            rho0[0, 0] = 1.0 + 0.0j
            return rho0
        
        # Learn initial state
        if self.rho0_type == "mixed":
            # Parameterize via Cholesky: ρ₀ = L L† / Tr(L L†)
            L = torch.complex(self.L_re, self.L_im).to(device)
            rho = L @ L.conj().T
            
            # Normalize trace to 1
            tr = torch.real(torch.trace(rho)) + self.eps
            rho = rho / tr
            
            # Ensure Hermitian (numerical stability)
            rho = 0.5 * (rho + rho.conj().T)
            
            return rho
        
        elif self.rho0_type == "pure":
            # Parameterize as pure state |ψ⟩, then ρ₀ = |ψ⟩⟨ψ|
            psi = torch.complex(self.psi_re, self.psi_im).to(device)  # [d]
            
            # Normalize |ψ⟩ to unit norm
            psi_norm = torch.linalg.norm(psi)
            psi = psi / (psi_norm + self.eps)
            
            # Construct density matrix: ρ₀ = |ψ⟩⟨ψ|
            rho = torch.outer(psi, psi.conj())  # [d, d]
            
            # Ensure Hermitian (numerical stability)
            rho = 0.5 * (rho + rho.conj().T)
            
            return rho
        
        else:
            raise ValueError(f"Unknown rho0_type: {self.rho0_type}")


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
#-----------------------------------------------------------------------------
def train_encoder(
    sequences, emp_probs,
    m: int, n_qubits: int,
    batch_size=4*512,
    lr=1e-3,
    epochs=50,
    learn_rho0=True,
    model=None,
    num_workers=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    optimizer_name="adam"):
    d = 2 ** n_qubits

    ds = SeqDataset(sequences, emp_probs)
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_pad,
        pin_memory=(device.startswith("cuda")),
    )

    if model is None:
        model = KrausInstrument(m=m, d=d, learn_rho0=learn_rho0)
        # New optimizer each session (weights-only restart)

    opt = make_optimizer(optimizer_name, model.parameters(), lr, weight_decay=1e-4)
        
    model = model.to(device)


    
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n_seen = 0

        for seq_pad, lens, p_emp in dataloader:
            seq_pad = seq_pad.to(device, non_blocking=True)
            p_emp = p_emp.to(device, non_blocking=True)

            p_mdl = model.sequence_prob_batch(seq_pad)
            loss = torch.mean(p_emp * (p_emp - p_mdl) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item() * seq_pad.size(0)
            n_seen += seq_pad.size(0)

        print(f"epoch {ep:3d} | loss {total / max(n_seen,1):.6e}")

    return model
#-----------------------------------------------------------------------------
class PredictiveSeqDatasetBilevel(Dataset):
    """
    Bilevel dataset: sequences with both sequence probs and conditional distributions
    """
    def __init__(self, sequences, target_distributions, seq_probs, global_weights):
        """
        sequences: list of sequences (one per unique prefix)
        target_distributions: list of [p₀, p₁, p₂, ...] for each sequence
        seq_probs: empirical p(sequence) from length-specific distribution
        """
        assert len(sequences) == len(target_distributions) == len(seq_probs) == len(global_weights)
        self.sequences = sequences
        self.target_dists = target_distributions
        self.seq_probs = seq_probs
        self.global_weights = global_weights
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.target_dists[idx],  # [d_out] distribution
            self.seq_probs[idx],      # scalar: p_emp(sequence)
            self.global_weights[idx] # scalar
        )


def collate_bilevel(batch):
    """
    Collate samples for Omega bilevel predictive training.

    Accepted dataset-item formats
    -----------------------------
    Four fields:
        sequence, target_distribution, seq_probability, global_weight

    Five fields:
        sequence, sequence_length, target_distribution,
        seq_probability, global_weight

    The explicit sequence length is discarded because the Omega
    representation encoder removes right-padding before calling
    path_operator.

    Returns
    -------
    seq_pad : LongTensor, shape (B, T_max)
    target_distributions : FloatTensor, shape (B, d_out)
    seq_probs : FloatTensor, shape (B,)
    global_weights : FloatTensor, shape (B,)
    """
    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch.")

    sequences = []
    targets = []
    seq_probabilities = []
    weights = []

    for sample_index, sample in enumerate(batch):
        if not isinstance(sample, (tuple, list)):
            raise TypeError(
                f"Sample {sample_index} must be a tuple or list; "
                f"received {type(sample).__name__}."
            )

        if len(sample) == 4:
            (
                sequence,
                target_distribution,
                seq_probability,
                global_weight,
            ) = sample

        elif len(sample) == 5:
            (
                sequence,
                _sequence_length,
                target_distribution,
                seq_probability,
                global_weight,
            ) = sample

        else:
            raise ValueError(
                f"Sample {sample_index} contains {len(sample)} fields. "
                "Expected either 4 fields:\n"
                "  (sequence, target_distribution, seq_probability, "
                "global_weight)\n"
                "or 5 fields:\n"
                "  (sequence, sequence_length, target_distribution, "
                "seq_probability, global_weight)."
            )

        sequence_tensor = torch.as_tensor(
            sequence,
            dtype=torch.long,
        ).reshape(-1)

        # Tolerate samples that are already right-padded.
        pad_locations = torch.nonzero(
            sequence_tensor == PAD,
            as_tuple=False,
        )

        if pad_locations.numel() > 0:
            first_pad = int(pad_locations[0].item())

            if torch.any(
                sequence_tensor[first_pad:] != PAD
            ):
                raise ValueError(
                    "A non-PAD symbol occurs after the first PAD in "
                    f"sample {sample_index}."
                )

            sequence_tensor = sequence_tensor[:first_pad]

        if sequence_tensor.numel() == 0:
            raise ValueError(
                f"Sample {sample_index} contains an empty sequence."
            )

        target_tensor = torch.as_tensor(
            target_distribution,
            dtype=torch.float32,
        ).reshape(-1)

        if target_tensor.numel() == 0:
            raise ValueError(
                f"Sample {sample_index} has an empty target distribution."
            )

        if not torch.all(torch.isfinite(target_tensor)):
            raise ValueError(
                f"Sample {sample_index} contains nonfinite target values."
            )

        sequences.append(sequence_tensor)
        targets.append(target_tensor)

        seq_probabilities.append(
            torch.as_tensor(
                seq_probability,
                dtype=torch.float32,
            ).reshape(())
        )

        weights.append(
            torch.as_tensor(
                global_weight,
                dtype=torch.float32,
            ).reshape(())
        )

    # Confirm a common target dimension.
    target_dimension = targets[0].numel()

    for sample_index, target in enumerate(targets):
        if target.numel() != target_dimension:
            raise ValueError(
                "All target distributions must have the same dimension. "
                f"Sample 0 has dimension {target_dimension}, while "
                f"sample {sample_index} has dimension {target.numel()}."
            )

    batch_size = len(sequences)
    maximum_length = max(
        sequence.numel()
        for sequence in sequences
    )

    seq_pad = torch.full(
        (batch_size, maximum_length),
        fill_value=PAD,
        dtype=torch.long,
    )

    for row, sequence in enumerate(sequences):
        seq_pad[row, :sequence.numel()] = sequence

    target_distributions = torch.stack(
        targets,
        dim=0,
    )

    seq_probs = torch.stack(
        seq_probabilities,
        dim=0,
    )

    global_weights = torch.stack(
        weights,
        dim=0,
    )

    return (
        seq_pad,
        target_distributions,
        seq_probs,
        global_weights,
    )


#-----------------------------------------------------------------------------
# Full predictive model Encoder+Decoder

class PredictiveQuantumModel(nn.Module):
    def __init__(self, encoder: KrausInstrument, decoder: QuantumDecoder, 
                 freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def encode_sequences_unnormalized(self, seq_pad):
        """
        Encode sequences WITHOUT normalization (Option C)
        
        Returns:
            rho_unnorm: [B, d, d] unnormalized density matrices
        """
        device = seq_pad.device
        B, T = seq_pad.shape
        d = self.encoder.d
        
        K = self.encoder.kraus_operators()  # [m, d, d]
        rho0 = self.encoder._make_rho0(device)  # [d, d]
        
        # Initialize batch of density matrices
        rho = rho0.unsqueeze(0).expand(B, d, d).clone()
        
        # Apply Kraus operators WITHOUT normalizing
        for t in range(T):
            sym = seq_pad[:, t]
            active = (sym != PAD)
            if not torch.any(active):
                break
            
            sym_a = sym[active].long()
            rho_a = rho[active]
            
            Kt = K.index_select(0, sym_a)
            # Apply K ρ K† but DO NOT normalize
            rho_a = torch.bmm(torch.bmm(Kt, rho_a), Kt.conj().transpose(1, 2))
            
            rho[active] = rho_a
        
        return rho  # [B, d, d]
    
    def forward(self, seq_pad):
        """
        Full pipeline: encode (unnormalized) → align → decode → normalize → predict
        
        Args:
            seq_pad: [B, T] padded sequences
        
        Returns:
            probs: [B, d_out] prediction probabilities
            traces: [B] sequence probabilities (traces of unnormalized states)
        """
        # Encode without normalization (Option C)
        rho_unnorm = self.encode_sequences_unnormalized(seq_pad)  # [B, d, d]
        
        # Get normalization factors (sequence probabilities)
        traces = torch.real(torch.diagonal(rho_unnorm, dim1=-2, dim2=-1).sum(-1))  # [B]
        traces = torch.clamp(traces, min=self.encoder.eps)
        
        # ===== DO NOT NORMALIZE YET - Keep unnormalized for phase preservation =====
        # rho_enc = rho_unnorm / traces.unsqueeze(-1).unsqueeze(-1)  ← REMOVED
        
        # Apply alignment unitary (on unnormalized state)
        U = self.decoder.get_unitary()  # [d_in, d_in]
        rho_align = torch.bmm(
            torch.bmm(U.unsqueeze(0).expand(rho_unnorm.size(0), -1, -1), rho_unnorm),
            U.conj().T.unsqueeze(0).expand(rho_unnorm.size(0), -1, -1)
        )  # [B, d_in, d_in] still unnormalized
        
        # Apply co-isometry to prediction register
        V = self.decoder.get_coisometry()  # [d_in, d_out]
        rho_pred_unnorm = torch.bmm(
            torch.bmm(V.conj().T.unsqueeze(0).expand(rho_align.size(0), -1, -1), rho_align),
            V.unsqueeze(0).expand(rho_align.size(0), -1, -1)
        )  # [B, d_out, d_out] still unnormalized
        
        # ===== NORMALIZE ONLY AT OUTPUT =====
        # Extract diagonal (prediction logits)
        logits = torch.real(torch.diagonal(rho_pred_unnorm, dim1=-2, dim2=-1))  # [B, d_out]
        logits = torch.clamp(logits, min=0.0)
        
        # Normalize to get probabilities
        probs = logits / (logits.sum(dim=-1, keepdim=True) + self.encoder.eps)  # [B, d_out]
        
        return probs, traces


    @torch.no_grad()
    def get_encoded_state(self, seq_pad):
        """
        Get the normalized encoded density matrix for a sequence
        
        Args:
            seq_pad: [B, T] padded sequences
        
        Returns:
            rho_enc: [B, d, d] normalized encoded states
            traces: [B] sequence probabilities
        """
        rho_unnorm = self.encode_sequences_unnormalized(seq_pad)
        traces = torch.real(torch.diagonal(rho_unnorm, dim1=-2, dim2=-1).sum(-1))
        traces = torch.clamp(traces, min=self.encoder.eps)
        rho_enc = rho_unnorm / traces.unsqueeze(-1).unsqueeze(-1)
        return rho_enc, traces
    
    @torch.no_grad()
    def get_sequence_probability(self, seq_pad):
        """
        Get ONLY the sequence generation probability p(sequence)
        
        Args:
            seq_pad: [B, T] padded sequences
        
        Returns:
            traces: [B] sequence probabilities
        """
        rho_unnorm = self.encode_sequences_unnormalized(seq_pad)
        traces = torch.real(torch.diagonal(rho_unnorm, dim1=-2, dim2=-1).sum(-1))
        traces = torch.clamp(traces, min=self.encoder.eps)
        return traces
    
    @torch.no_grad()
    def get_class_distribution(self, seq_pad):
        """
        Get ONLY the class distribution p(class|sequence)
        
        Args:
            seq_pad: [B, T] padded sequences
        
        Returns:
            probs: [B, d_out] class probabilities
        """
        probs, _ = self.forward(seq_pad)
        return probs

#-----------------------------------------------------------------------------
def train_predictive_model_bilevel_omega(
    sequences,
    target_distributions,
    seq_probs,
    global_weights,
    encoder,
    d_out: int,
    batch_size: int = 512,
    lr: float = 1e-3,
    epochs: int = 100,
    freeze_encoder: bool = False,
    use_unitary: bool = True,
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    ),
    optimizer_name: str = "adam",
    weight_decay: float = 1e-4,
    encoder_lr_multiplier: float = 0.1,
    lambda_enc: float = 0.1,
    lambda_pred: float = 1.0,
    prediction_loss: str = "mseError",
    encoder_loss: str = "mse",
    num_workers: int = 0,

    # Omega decoder configuration
    use_general_map: bool = True,
    general_rank: int = 16,
    n_reflections: int = 16,

    # Numerical/training controls
    eps: float = 1e-8,
    gradient_clip_norm: float | None = 5.0,
    show_progress: bool = True,
    restore_best_training_state: bool = False,
):
    """
    Train the Omega predictive model with bilevel loss:

        L = lambda_pred * L_prediction
            + lambda_enc * L_encoder

    Encoder objective:
        p_model(sequence) approximately p_empirical(sequence)

    Predictive objective:
        p_model(class | sequence)
            approximately target_distribution(sequence)

    Notes
    -----
    1. The Dataset and DataLoader structure are unchanged.
    2. The input batch may be a padded LongTensor or ragged lists,
       provided OmegaRepresentationEncoder supports both.
    3. When freeze_encoder=True, lambda_enc is effectively zero
       for optimization, although encoder loss is still reported.
    """

    device = str(device)

    if len(sequences) == 0:
        raise ValueError(
            "sequences cannot be empty."
        )

    if not (
        len(sequences)
        == len(target_distributions)
        == len(seq_probs)
        == len(global_weights)
    ):
        raise ValueError(
            "sequences, target_distributions, seq_probs, "
            "and global_weights must have equal lengths."
        )

    if epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    # ------------------------------------------------------------
    # Construct Omega representation model
    # ------------------------------------------------------------

    d_system = int(encoder.d)
    d_omega = d_system**2

    representation_encoder = OmegaRepresentationEncoder(
        encoder=encoder,
        # Uses the same global PAD symbol as collate_bilevel.
        pad_value=PAD,
        eps=eps,
    )


    decoder = OmegaQuantumDecoder(
        omega_dimension=d_omega,
        d_out=d_out,
        use_unitary=use_unitary,
        n_reflections=n_reflections,
        use_general_map=use_general_map,
        general_rank=general_rank,
        eps=eps,
    )

    pred_model = OmegaPredictiveModel(
        representation_encoder=representation_encoder,
        decoder=decoder,
        freeze_encoder=freeze_encoder,
    ).to(device)


    # ------------------------------------------------------------
    # Dataset and loader remain unchanged
    # ------------------------------------------------------------

    ds = PredictiveSeqDatasetBilevel(
        sequences,
        target_distributions,
        seq_probs,
        global_weights,
    )

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_bilevel,
        pin_memory=device.startswith("cuda"),
    )

    # ------------------------------------------------------------
    # Optimizer parameter groups
    # ------------------------------------------------------------

    encoder_parameters = [
        parameter
        for parameter in pred_model.encoder.parameters()
        if parameter.requires_grad
    ]

    decoder_parameters = [
        parameter
        for parameter in pred_model.decoder.parameters()
        if parameter.requires_grad
    ]

    param_groups = []

    if encoder_parameters:
        param_groups.append({
            "params": encoder_parameters,
            "lr": lr * encoder_lr_multiplier,
            "name": "encoder",
        })

    if decoder_parameters:
        param_groups.append({
            "params": decoder_parameters,
            "lr": lr,
            "name": "decoder",
        })

    if not param_groups:
        raise RuntimeError(
            "The model has no trainable parameters."
        )

    optimizer_name_normalized = (
        optimizer_name.lower().strip()
    )

    if optimizer_name_normalized == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=weight_decay,
        )

    elif optimizer_name_normalized == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(
            "optimizer_name must be 'adam' or 'adamw'; "
            f"received {optimizer_name!r}."
        )

    effective_lambda_enc = (
        0.0
        if freeze_encoder
        else float(lambda_enc)
    )

    trainable_encoder_count = sum(
        parameter.numel()
        for parameter in pred_model.encoder.parameters()
        if parameter.requires_grad
    )

    trainable_decoder_count = sum(
        parameter.numel()
        for parameter in pred_model.decoder.parameters()
        if parameter.requires_grad
    )

    print(
        "\nOmega predictive model"
        f"\n  system dimension          : {d_system}"
        f"\n  Omega-vector dimension    : {d_omega}"
        f"\n  output dimension          : {d_out}"
        f"\n  encoder frozen            : {freeze_encoder}"
        f"\n  trainable encoder params  : "
        f"{trainable_encoder_count:,}"
        f"\n  trainable decoder params  : "
        f"{trainable_decoder_count:,}"
        f"\n  lambda prediction         : {lambda_pred}"
        f"\n  lambda encoder requested  : {lambda_enc}"
        f"\n  lambda encoder effective  : "
        f"{effective_lambda_enc}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    history = {
        "total_loss": [],
        "prediction_loss": [],
        "encoder_loss": [],
        "epoch_seconds": [],
    }

    best_loss = float("inf")
    best_state = None
    best_epoch = None

    trainable_parameters = [
        parameter
        for parameter in pred_model.parameters()
        if parameter.requires_grad
    ]

    for epoch in range(epochs):
        pred_model.train()

        epoch_start = time.perf_counter()

        prediction_numerator = 0.0
        encoder_numerator = 0.0
        weight_denominator = 0.0

        batch_iterator = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            disable=not show_progress,
            leave=False,
        )

        for batch in batch_iterator:
            # Expected collate_bilevel output:
            #
            #   batch_sequences
            #   batch_target_distributions
            #   batch_seq_probs
            #   batch_global_weights

            if not isinstance(batch, (tuple, list)):
                raise TypeError(
                    "collate_bilevel must return a tuple or list."
                )

            if len(batch) != 4:
                raise ValueError(
                    "collate_bilevel must return four objects: "
                    "(sequences, target_distributions, "
                    "seq_probs, global_weights). "
                    f"Received {len(batch)} objects."
                )

            (
                batch_sequences,
                batch_targets,
                batch_empirical_seq_probs,
                batch_weights,
            ) = batch

            batch_targets = torch.as_tensor(
                batch_targets,
                dtype=torch.float32,
                device=device,
            )

            batch_empirical_seq_probs = torch.as_tensor(
                batch_empirical_seq_probs,
                dtype=torch.float32,
                device=device,
            ).reshape(-1)

            current_batch_size = (
                batch_targets.shape[0]
            )

            sample_weights = prepare_sample_weights(
                batch_weights,
                batch_size=current_batch_size,
                device=device,
                eps=eps,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred_model_output = pred_model(
                batch_sequences
            )

            predicted_distributions, model_sequence_probabilities =pred_model(
                batch_sequences
            )
            
            model_sequence_probabilities = (
                model_sequence_probabilities.real.reshape(-1)
            )



            if (
                predicted_distributions.shape
                != batch_targets.shape
            ):
                raise RuntimeError(
                    "Prediction-target shape mismatch: "
                    f"prediction="
                    f"{tuple(predicted_distributions.shape)}, "
                    f"target={tuple(batch_targets.shape)}."
                )

            if (
                model_sequence_probabilities.numel()
                != current_batch_size
            ):
                raise RuntimeError(
                    "Sequence probability batch-size mismatch: "
                    f"received "
                    f"{model_sequence_probabilities.numel()}, "
                    f"expected {current_batch_size}."
                )

            # Per-sequence predictive loss
            prediction_values = (
                prediction_loss_per_sample(
                    prediction=predicted_distributions,
                    target=batch_targets,
                    loss_type=prediction_loss,
                    eps=eps,
                )
            )

            # Per-sequence encoder/generative loss
            encoder_values = (
                encoder_loss_per_sample(
                    model_probabilities=(
                        model_sequence_probabilities
                    ),
                    empirical_probabilities=(
                        batch_empirical_seq_probs
                    ),
                    loss_type=encoder_loss,
                    eps=eps,
                )
            )

            loss_prediction = weighted_mean(
                prediction_values,
                sample_weights,
                eps=eps,
            )

            loss_encoder = weighted_mean(
                encoder_values,
                sample_weights,
                eps=eps,
            )

            loss_total = (
                float(lambda_pred)
                * loss_prediction
                + effective_lambda_enc
                * loss_encoder
            )

            if not torch.isfinite(loss_total):
                raise FloatingPointError(
                    "Nonfinite total loss encountered."
                    f"\nPrediction loss: "
                    f"{loss_prediction.detach().cpu().item()}"
                    f"\nEncoder loss: "
                    f"{loss_encoder.detach().cpu().item()}"
                )

            loss_total.backward()

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=gradient_clip_norm,
                )

            optimizer.step()

            # Weighted epoch statistics
            with torch.no_grad():
                batch_weight_sum = float(
                    sample_weights.sum()
                    .detach()
                    .cpu()
                    .item()
                )

                prediction_numerator += float(
                    torch.sum(
                        prediction_values.detach()
                        * sample_weights
                    )
                    .cpu()
                    .item()
                )

                encoder_numerator += float(
                    torch.sum(
                        encoder_values.detach()
                        * sample_weights
                    )
                    .cpu()
                    .item()
                )

                weight_denominator += batch_weight_sum

            batch_iterator.set_postfix({
                "total": (
                    f"{loss_total.detach().cpu().item():.5g}"
                ),
                "pred": (
                    f"{loss_prediction.detach().cpu().item():.5g}"
                ),
                "enc": (
                    f"{loss_encoder.detach().cpu().item():.5g}"
                ),
            })

        if weight_denominator <= eps:
            raise RuntimeError(
                "The epoch has zero total sample weight."
            )

        epoch_prediction_loss = (
            prediction_numerator
            / weight_denominator
        )

        epoch_encoder_loss = (
            encoder_numerator
            / weight_denominator
        )

        epoch_total_loss = (
            float(lambda_pred)
            * epoch_prediction_loss
            + effective_lambda_enc
            * epoch_encoder_loss
        )

        epoch_seconds = (
            time.perf_counter() - epoch_start
        )

        history["total_loss"].append(
            epoch_total_loss
        )

        history["prediction_loss"].append(
            epoch_prediction_loss
        )

        history["encoder_loss"].append(
            epoch_encoder_loss
        )

        history["epoch_seconds"].append(
            epoch_seconds
        )

        if epoch_total_loss < best_loss:
            best_loss = epoch_total_loss
            best_epoch = epoch + 1

            if restore_best_training_state:
                best_state = copy.deepcopy(
                    pred_model.state_dict()
                )

        print(
            f"Epoch {epoch + 1:4d}/{epochs}"
            f" | total={epoch_total_loss:.8g}"
            f" | pred={epoch_prediction_loss:.8g}"
            f" | enc={epoch_encoder_loss:.8g}"
            f" | time={epoch_seconds:.2f}s",
            flush=True,
        )

    if (
        restore_best_training_state
        and best_state is not None
    ):
        pred_model.load_state_dict(
            best_state
        )

    pred_model.eval()

    metadata = {
        "representation": "omega",
        "system_dimension": d_system,
        "omega_dimension": d_omega,
        "output_dimension": d_out,
        "freeze_encoder": freeze_encoder,
        "use_unitary": use_unitary,
        "use_general_map": use_general_map,
        "general_rank": general_rank,
        "n_reflections": n_reflections,
        "prediction_loss": prediction_loss,
        "encoder_loss": encoder_loss,
        "lambda_pred": float(lambda_pred),
        "lambda_enc_requested": float(lambda_enc),
        "lambda_enc_effective": (
            effective_lambda_enc
        ),
        "trainable_encoder_parameters": (
            trainable_encoder_count
        ),
        "trainable_decoder_parameters": (
            trainable_decoder_count
        ),
        "best_epoch": best_epoch,
        "best_training_loss": best_loss,
        "history": history,
    }

    return pred_model, metadata

#------------------------------------------------------------------------------
def train_predictive_model_bilevel(
    sequences,             # list of sequences  
    target_distributions,  # sequence induced class distributions
    seq_probs,             # sequence probabilities wrt same length distributions
    global_weights,        # ← GLOBAL Empirical weights
    encoder: KrausInstrument,
    d_out: int,
    batch_size: int = 512,
    lr: float = 1e-3,
    epochs: int = 100,
    freeze_encoder: bool = False,
    use_unitary: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    optimizer_name: str = "adam",
    weight_decay: float = 1e-4,
    encoder_lr_multiplier: float = 0.1,
    lambda_enc: float = 0.1,   # NEW: weight for encoder loss
    lambda_pred: float = 1.0,  # NEW: weight for prediction loss
    prediction_loss = 'mseError', # 'jsDivergence' # 'xEntropy', 'klDivergence', 'mseError'
    num_workers=0,
):
    
    """
      Train predictive model with bilevel loss:
      - Encoder loss: match p(sequence)
      - Prediction loss: match p(target|sequence)
      
      Args:
          sequences: list of sequences
          target_distributions: list of [p₀, p₁, ...] distributions per sequence
          seq_probs: list of p(sequence) empirical probabilities
          lambda_enc: weight for encoder/sequence loss
          lambda_pred: weight for prediction/conditional loss
      """
    
    d_system = encoder.d
    d_omega = d_system**2
    
    representation_encoder = OmegaRepresentationEncoder(
        encoder=encoder,
        pad_value=PAD,
        eps=1e-8,
    )
    
    decoder = OmegaQuantumDecoder(
    omega_dimension=d_omega,
    d_out=d_out,
    use_unitary=use_unitary,
    use_general_map=True,
    general_rank=16,
    )
    
    model = OmegaPredictiveModel(
    representation_encoder=representation_encoder,
    decoder=decoder,
    freeze_encoder=freeze_encoder,
    )

    model = model.to(device)
    
    # Dataset with bilevel structure
    ds = PredictiveSeqDatasetBilevel(sequences, target_distributions, seq_probs, global_weights,)
    dataloader = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        collate_fn=collate_bilevel,
        pin_memory=(device.startswith("cuda")),
    )
    
   
    # Separate parameter groups for different learning rates
    if not freeze_encoder:
        param_groups = [
            {
                'params': model.encoder.parameters(),
                'lr': lr * encoder_lr_multiplier,
                'name': 'encoder'
            },
            {
                'params': model.decoder.parameters(),
                'lr': lr,
                'name': 'decoder'
            }
        ]
        print(f"Joint training: Encoder LR={lr * encoder_lr_multiplier:.2e}, "
              f"Decoder LR={lr:.2e}")
        print(f"Loss weights: λ_enc={lambda_enc}, λ_pred={lambda_pred}")
        opt = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    else:
        param_groups = [{'params': model.decoder.parameters(), 'lr': lr}]
        print(f"Decoder-only training: Decoder LR={lr:.2e} (encoder frozen)")
        opt = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    
    # Training loop
    for ep in range(1, epochs + 1):
        model.train()
        total_enc_loss = 0.0
        total_pred_loss = 0.0
        total_loss = 0.0
        n_seen = 0
        

        for seq_pad, lens, target_dist, seq_prob, global_weight in dataloader:
            
            seq_pad = seq_pad.to(device, non_blocking=True)               # sequences
            target_dist = target_dist.to(device, non_blocking=True)       # [B, d_out] target distributions
            seq_prob = seq_prob.to(device, non_blocking=True)             # [B] generative probabilities
            global_weight = global_weight.to(device, non_blocking=True)   # [B] global importnace
            
       
        
       
            # Normalize empirical sequence importance to sum to 1 in batch 
            global_weight_normalized = global_weight / (global_weight.sum() + 1e-12)
            
            # Forward pass with Option C (unnormalized encoding)
            model_dist, traces = model(seq_pad)  # probs: [B, d_out], traces: [B]
            
            
            # ===== ENCODER LOSS: Match p(sequence) =====
            # traces = Tr(K_s ρ₀ K_s†) = model's sequence probability
            p_enc = torch.clamp(traces, min=1e-12)
            
            
            # Cross-entropy: H(p_emp, p_model) = -Σ p_emp log p_model
            enc_loss = -(global_weight_normalized * torch.log(p_enc)).sum()
            
            ###################################################################
            if prediction_loss == 'xEntropy':
                # ===== PREDICTION LOSS: Match p(target|sequence) =====
                # Cross-entropy between distributions
                # H(p_emp(·|s), p_model(·|s)) = -Σ_y p_emp(y|s) log p_model(y|s)
                pred_loss_per_sample = -(target_dist * torch.log(model_dist + 1e-12)).sum(dim=-1)  # [B]
            
            if prediction_loss == 'klDivergence':  
                # ===== KL DIVERGENCE =====
                # KL(p_emp || p_model) = Σ_y p_emp(y) [log p_emp(y) - log p_model(y)]
                #                      = Σ_y p_emp(y) log p_emp(y) - Σ_y p_emp(y) log p_model(y)
                #                      = H(p_emp || p_model) - H(p_emp)
                log_target = torch.log(target_dist + 1e-12)
                pred_loss_per_sample = (target_dist * (log_target - torch.log(model_dist + 1e-12))).sum(dim=-1)  # [B]
            
            if prediction_loss == 'jsDivergence':  
               # ===== JENSEN-SHANNON DIVERGENCE (Symmetric) =====
               # JS(p_emp, p_model) = 0.5 * KL(p_emp || m) + 0.5 * KL(p_model || m)
               # where m = 0.5 * (p_emp + p_model) is the mixture
               m         = 0.5 * (target_dist + model_dist)
               log_m     = torch.log(m + 1e-12)
               log_target = torch.log(target_dist + 1e-12)
               pred_loss_per_sample = (
                    0.5 * (target_dist * (log_target - log_m)).sum(dim=-1) +
                    0.5 * (model_dist  * (torch.log(model_dist + 1e-12) - log_m)).sum(dim=-1)
                )  # [B]
            if prediction_loss == 'mseError':  
                # ===== MEAN SQUARED ERROR =====
                # MSE = Σ_y (p_emp(y) - p_model(y))²
                pred_loss_per_sample = ((target_dist - model_dist) ** 2).sum(dim=-1)  # [B]
            
            
            # Weight by global sequence importance 
            pred_loss = (global_weight_normalized * pred_loss_per_sample).sum()
            
            
            
            # ===== COMBINED LOSS =====
            loss = lambda_enc * enc_loss + lambda_pred * pred_loss
            
            # Backward and optimize
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            # Track losses
            total_enc_loss += float(enc_loss.detach().cpu()) * seq_pad.size(0)
            total_pred_loss += float(pred_loss.detach().cpu()) * seq_pad.size(0)
            total_loss += float(loss.detach().cpu()) * seq_pad.size(0)
            n_seen += seq_pad.size(0)
        
        # Epoch summary
        avg_enc_loss = total_enc_loss / max(n_seen, 1)
        avg_pred_loss = total_pred_loss / max(n_seen, 1)
        avg_total_loss = total_loss / max(n_seen, 1)
        
        mode = "Joint" if not freeze_encoder else "Decoder-only"
            
        print(f"Epoch {ep:3d} | {mode} | "
              f"Total: {avg_total_loss:.6f} | "
              f"Enc: {avg_enc_loss:.6f} | "
              f"Pred: {avg_pred_loss:.6f}")              
              
              
    return model



    




#-----------------------------------------------------------------------------
# Prediction of sequence probability and sequence class distribution
#------------------------------------------------------------------------------

# Single sequence
@torch.no_grad()
def predict_from_sequence(
    model: PredictiveQuantumModel,
    sequence,
    device: str = "cpu"
):
    """
    Get model's probability and class distribution for ONE sequence
    
    Args:
        model: trained PredictiveQuantumModel
        sequence: list of symbols, e.g., [0, 1, 2]
        device: computation device
    
    Returns:
        dict with:
            - 'sequence': input sequence
            - 'p_sequence': p_model(s) - generative probability
            - 'p_classes': [p₀, p₁, p₂] - conditional distribution
    """
    model.eval()
    model = model.to(device)
    
    # Convert to tensor [1, T] (batch size 1)
    if not torch.is_tensor(sequence):
        seq_t = torch.tensor([sequence], dtype=torch.long)
    else:
        seq_t = sequence.unsqueeze(0) if sequence.dim() == 1 else sequence
    
    seq_t = seq_t.to(device)
    
    # Forward pass
    probs, traces = model(seq_t)  # probs: [1, d_out], traces: [1]
    
    # Extract single result
    p_sequence = float(traces[0].cpu().item())
    p_classes = probs[0].cpu().numpy()
    
    return {
        'sequence': sequence if isinstance(sequence, list) else sequence.tolist(),
        'p_sequence': p_sequence,
        'p_classes': p_classes,
    }


# multiple sequences
@torch.no_grad()
def predict_from_sequences(
    model: PredictiveQuantumModel,
    sequence,
    device: str = "cpu"
):
    """
    Get both generative and predictive outputs for a single sequence
    
    Args:
        model: trained PredictiveQuantumModel
        sequence: list or array of symbols, e.g., [0, 1, 2]
        device: computation device
    
    Returns:
        dict with:
            - 'p_sequence': float, model's probability of generating this sequence
            - 'p_classes': numpy array [d_out], conditional class distribution
            - 'sequence': input sequence (for reference)
    """
    model.eval()
    model = model.to(device)
    
    # Convert sequence to tensor
    if not torch.is_tensor(sequence):
        seq_t = torch.tensor([sequence], dtype=torch.long)  # [1, T]
    else:
        seq_t = sequence.unsqueeze(0) if sequence.dim() == 1 else sequence
    
    seq_t = seq_t.to(device)
    
    # Forward pass (Option C: returns probs and traces)
    probs, traces = model(seq_t)  # probs: [1, d_out], traces: [1]
    
    # Extract results
    p_sequence = float(traces[0].cpu().item())
    p_classes = probs[0].cpu().numpy()
    
    return {
        'sequence': sequence if isinstance(sequence, list) else sequence.tolist(),
        'p_sequence': p_sequence,
        'p_classes': p_classes,
    }


@torch.no_grad()
def predict_from_sequences_batch(
    model: PredictiveQuantumModel,
    sequences,
    batch_size: int = 512,
    device: str = "cpu"
):
    """
    Get predictions for multiple sequences efficiently
    
    Args:
        model: trained PredictiveQuantumModel
        sequences: list of sequences
        batch_size: batch size for processing
        device: computation device
    
    Returns:
        list of dicts, one per sequence
    """
    model.eval()
    model = model.to(device)
    
    results = []
    
    # Process in batches
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        
        # Pad sequences
        lens = [len(s) for s in batch_seqs]
        max_len = max(lens)
        
        seq_pad = torch.full((len(batch_seqs), max_len), PAD, dtype=torch.long)
        for j, s in enumerate(batch_seqs):
            seq_pad[j, :len(s)] = torch.tensor(s, dtype=torch.long)
        
        seq_pad = seq_pad.to(device)
        
        # Forward pass
        probs, traces = model(seq_pad)  # [B, d_out], [B]
        
        # Extract results
        for j in range(len(batch_seqs)):
            results.append({
                'sequence': batch_seqs[j],
                'p_sequence': float(traces[j].cpu().item()),
                'p_classes': probs[j].cpu().numpy(),
            })
    
    return results

@torch.no_grad()
def get_model_predictions_ordered(
    model,
    sequences,
    batch_size: int = 512,
    device=None,
):
    """
    Evaluate the predictive model on all sequences, preserving order.

    Parameters
    ----------
    model
        Trained PredictiveQuantumModel or OmegaPredictiveModel.

    sequences
        Ragged collection of symbolic sequences.

    batch_size
        Number of sequences evaluated per batch.

    device
        Device used for evaluation. When None, use the model's
        current device.

    Returns
    -------
    mod_seq_probs
        List containing p_model(sequence).

    mod_target_distributions
        List containing p_model(class | sequence).
    """
    sequences = list(sequences)

    if len(sequences) == 0:
        return [], []

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
        model = model.to(device)

    model.eval()

    mod_seq_probs = []
    mod_target_distributions = []

    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[
                start:start + batch_size
            ]

            batch_count = len(batch_sequences)
            maximum_length = max(
                len(sequence)
                for sequence in batch_sequences
            )

            seq_pad = torch.full(
                (batch_count, maximum_length),
                PAD,
                dtype=torch.long,
                device=device,
            )

            for index, sequence in enumerate(batch_sequences):
                sequence_tensor = torch.as_tensor(
                    sequence,
                    dtype=torch.long,
                    device=device,
                )

                seq_pad[
                    index,
                    :sequence_tensor.numel(),
                ] = sequence_tensor

            # Compatible model interface:
            #
            # predicted_distributions: [B, d_out]
            # model_seq_probs:         [B]
            predicted_distributions, model_seq_probs = model(
                seq_pad
            )

            model_seq_probs = (
                model_seq_probs.real.reshape(-1)
            )

            if predicted_distributions.shape[0] != batch_count:
                raise RuntimeError(
                    "Prediction batch-size mismatch: "
                    f"expected {batch_count}, received "
                    f"{predicted_distributions.shape[0]}."
                )

            if model_seq_probs.numel() != batch_count:
                raise RuntimeError(
                    "Sequence-probability batch-size mismatch: "
                    f"expected {batch_count}, received "
                    f"{model_seq_probs.numel()}."
                )

            mod_seq_probs.extend(
                model_seq_probs
                .detach()
                .cpu()
                .tolist()
            )

            mod_target_distributions.extend(
                predicted_distributions
                .detach()
                .cpu()
                .tolist()
            )

    return mod_seq_probs, mod_target_distributions
#==============================================================================
def compute_divergences_by_length(
    sequences,
    emp_target_dists,
    mod_target_dists,
    divergence_type: str = "kl"
):
    """
    Compute divergence between empirical and model distributions by sequence length
    
    Args:
        sequences: list of sequences
        emp_target_dists: list of empirical [p₀, p₁, p₂, ...]
        mod_target_dists: list of model [p₀, p₁, p₂, ...]
        divergence_type: "xEntropy", "klDivergence", "jsDivergence", or "mseError"
    
    Returns:
        dict: {length: [divergences list]}
    """
    assert len(sequences) == len(emp_target_dists) == len(mod_target_dists)
    
    divergences_by_length = {}
    divergence_type = divergence_type.lower()
    
    for seq, emp_dist, mod_dist in zip(sequences, emp_target_dists, mod_target_dists):
        length = len(seq)
        
        emp_dist = np.array(emp_dist, dtype=np.float64)
        mod_dist = np.array(mod_dist, dtype=np.float64)
        
        # Clamp to avoid log(0)
        emp_dist = np.clip(emp_dist, 1e-12, 1.0)
        mod_dist = np.clip(mod_dist, 1e-12, 1.0)
        
        # Compute divergence
        if divergence_type == 'xentropy':
            # Cross-entropy: H(p_emp || p_model) = -Σ_y p_emp(y) log p_model(y)
            div = -np.sum(emp_dist * np.log(mod_dist))
            
        elif divergence_type == 'kldivergence':
            # KL divergence: KL(p_emp || p_model) = Σ_y p_emp(y) [log p_emp(y) - log p_model(y)]
            div = np.sum(emp_dist * (np.log(emp_dist) - np.log(mod_dist)))
            
        elif divergence_type == 'jsdivergence':
            # Jensen-Shannon (symmetric): JS = 0.5*KL(p_emp||m) + 0.5*KL(p_model||m)
            # where m = 0.5*(p_emp + p_model)
            m = 0.5 * (emp_dist + mod_dist)
            div = (
                0.5 * np.sum(emp_dist * (np.log(emp_dist) - np.log(m))) +
                0.5 * np.sum(mod_dist * (np.log(mod_dist) - np.log(m)))
            )
            
        elif divergence_type == 'mseerror':
            # Mean squared error: MSE = Σ_y (p_emp(y) - p_model(y))²
            div = np.sum((emp_dist - mod_dist) ** 2)
            
        else:
            raise ValueError(f"Unknown divergence type: {divergence_type}. "
                           f"Must be one of: 'xEntropy', 'klDivergence', 'jsDivergence', 'mseError'")
        
        if length not in divergences_by_length:
            divergences_by_length[length] = []
        divergences_by_length[length].append(div)
    
    return divergences_by_length


def plot_divergence_by_length(
    sequences,
    emp_target_dists,
    mod_target_dists,
    mod_seq_probs=None,
    emp_seq_probs=None,
    divergence_type: str = "kl",
    figsize=(14, 6),
    title: str = None
):
    """
    Plot average divergence bars by sequence length
    
    Args:
        sequences: list of sequences
        emp_target_dists: empirical class distributions
        mod_target_dists: model class distributions
        mod_seq_probs: optional model sequence probabilities (for weighting)
        emp_seq_probs: optional empirical sequence probabilities (for weighting)
        divergence_type: "kl" or "js"
        figsize: figure size
        title: plot title
    """
    
    # Compute divergences by length
    divergences_by_length = compute_divergences_by_length(
        sequences, emp_target_dists, mod_target_dists, divergence_type
    )
    
    # Compute average and std for each length
    lengths = sorted(divergences_by_length.keys())
    avg_divs = []
    std_divs = []
    counts = []
    
    for length in lengths:
        divs = divergences_by_length[length]
        avg_divs.append(np.mean(divs))
        std_divs.append(np.std(divs))
        counts.append(len(divs))
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ========== SUBPLOT 1: Divergence with Error Bars ==========
    colors = plt.cm.viridis(np.linspace(0, 1, len(lengths)))
    
    bars = ax1.bar(range(len(lengths)), avg_divs, yerr=std_divs, 
                   capsize=5, alpha=0.7, color=colors, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'Average {divergence_type.upper()} Divergence', fontsize=12, fontweight='bold')
    ax1.set_title(f'Model vs Empirical Distribution\n({divergence_type.upper()} Divergence by Length)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(lengths)))
    ax1.set_xticklabels(lengths)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avg_divs)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ========== SUBPLOT 2: Count of Sequences per Length ==========
    bars2 = ax2.bar(range(len(lengths)), counts, alpha=0.7, 
                    color=colors, edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Sequences', fontsize=12, fontweight='bold')
    ax2.set_title('Sample Count by Sequence Length', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(lengths)))
    ax2.set_xticklabels(lengths)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count labels on bars
    for bar, count in zip(bars2, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Main title
    if title is None:
        title = f'Divergence Analysis: Model vs Empirical Distributions ({divergence_type.upper()})'
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig, (ax1, ax2), {
        'lengths': lengths,
        'avg_divs': avg_divs,
        'std_divs': std_divs,
        'counts': counts
    }








# ============================================================================
# Pretty printing utilities
# ============================================================================

def print_prediction(result, class_names=None):
    """
    Pretty print a single prediction result
    
    Args:
        result: dict from predict_from_sequence
        class_names: optional list of class names (e.g., ['down', 'same', 'up'])
    """
    print(f"Sequence: {result['sequence']}")
    print(f"  p(sequence) = {result['p_sequence']:.6f}")
    print(f"  p(class|sequence):")
    
    p_classes = result['p_classes']
    d_out = len(p_classes)
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(d_out)]
    
    for i, (name, prob) in enumerate(zip(class_names, p_classes)):
        bar_length = int(prob * 40)  # 40-char bar
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"    {name:12s} | {bar} | {prob:.4f}")

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
#------------------------------------------------------------------------------
#  Diagnostics
#------------------------------------------------------------------------------

def compute_prediction_agreement(
    sequences,
    emp_target_dists,
    mod_target_dists,
    class_names=None
):
    """
    Compute percentage of samples where model and empirical agree on most probable class

    Args:
        sequences: list of sequences
        emp_target_dists: empirical class distributions
        mod_target_dists: model class distributions
        class_names: optional list of class names (e.g., ['Down', 'Neutral', 'Up'])

    Returns:
        dict with overall and per-length agreement statistics
    """
    assert len(sequences) == len(emp_target_dists) == len(mod_target_dists)

    # Overall agreement
    total_agreements = 0
    total_samples = len(sequences)

    # Agreement by length
    agreements_by_length = {}
    totals_by_length = {}

    # Agreement by predicted class
    d_out = len(emp_target_dists[0])
    if class_names is None:
        class_names = [f"class_{i}" for i in range(d_out)]

    agreements_by_class = {name: 0 for name in class_names}
    totals_by_class = {name: 0 for name in class_names}

    # Detailed results
    detailed_results = []

    for seq, emp_dist, mod_dist in zip(sequences, emp_target_dists, mod_target_dists):
        length = len(seq)

        emp_dist = np.array(emp_dist)
        mod_dist = np.array(mod_dist)

        # Find argmax for each
        emp_argmax = np.argmax(emp_dist)
        mod_argmax = np.argmax(mod_dist)

        # Check agreement
        agrees = (emp_argmax == mod_argmax)

        # Track overall
        if agrees:
            total_agreements += 1

        # Track by length
        if length not in agreements_by_length:
            agreements_by_length[length] = 0
            totals_by_length[length] = 0

        totals_by_length[length] += 1
        if agrees:
            agreements_by_length[length] += 1

        # Track by empirical predicted class
        emp_class_name = class_names[emp_argmax]
        totals_by_class[emp_class_name] += 1
        if agrees:
            agreements_by_class[emp_class_name] += 1

        # Store detailed result
        detailed_results.append({
            'sequence': seq,
            'length': length,
            'emp_argmax': emp_argmax,
            'mod_argmax': mod_argmax,
            'emp_class': class_names[emp_argmax],
            'mod_class': class_names[mod_argmax],
            'agrees': agrees,
            'emp_prob': emp_dist[emp_argmax],
            'mod_prob': mod_dist[mod_argmax],
            'emp_dist': emp_dist.tolist(),
            'mod_dist': mod_dist.tolist(),
        })

    # Compute percentages
    overall_agreement_pct = 100.0 * total_agreements / total_samples

    agreement_pct_by_length = {
        length: 100.0 * agreements_by_length[length] / totals_by_length[length]
        for length in sorted(totals_by_length.keys())
    }

    agreement_pct_by_class = {
        class_name: 100.0 * agreements_by_class[class_name] / totals_by_class[class_name]
        if totals_by_class[class_name] > 0 else 0.0
        for class_name in class_names
    }

    return {
        'overall_agreement_pct': overall_agreement_pct,
        'total_agreements': total_agreements,
        'total_samples': total_samples,
        'agreement_pct_by_length': agreement_pct_by_length,
        'agreements_by_length': agreements_by_length,
        'totals_by_length': totals_by_length,
        'agreement_pct_by_class': agreement_pct_by_class,
        'agreements_by_class': agreements_by_class,
        'totals_by_class': totals_by_class,
        'detailed_results': detailed_results,
    }

def classification_metrics_from_labels(
    y_true,
    y_pred,
    class_values=(-1, 0, 1),
    class_names=None,
    sample_weight=None,
    eps=1e-12,
):
    """
    Compute accuracy, precision, recall, and F1 per class.

    y_true:
        empirical dominant classes, e.g. [-1, 0, 1]

    y_pred:
        model dominant classes, e.g. [-1, 0, 1]

    sample_weight:
        optional weights/counts per example.
        Use counts for occurrence-weighted empirical metrics.
        Use None for unique-sequence metrics.
    """

    class_values = np.asarray(class_values)

    if class_names is None:
        class_names = {
            -1: "Down",
             0: "Neutral",
             1: "Up",
        }

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if sample_weight is None:
        sample_weight = np.ones(len(y_true), dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)

    n_classes = len(class_values)

    label_to_idx = {
        int(label): i
        for i, label in enumerate(class_values)
    }

    confusion = np.zeros(
        (n_classes, n_classes),
        dtype=np.float64,
    )

    for yt, yp, w in zip(y_true, y_pred, sample_weight):
        i = label_to_idx[int(yt)]
        j = label_to_idx[int(yp)]
        confusion[i, j] += w

    per_class = {}

    for i, label in enumerate(class_values):
        label = int(label)
        name = class_names.get(label, str(label))

        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        support = confusion[i, :].sum()

        precision = tp / max(tp + fp, eps)
        recall = tp / max(tp + fn, eps)
        f1 = 2.0 * precision * recall / max(precision + recall, eps)

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    total = confusion.sum()
    accuracy = np.trace(confusion) / max(total, eps)

    f1_values = np.asarray([
        per_class[class_names.get(int(label), str(int(label)))]["f1"]
        for label in class_values
    ])

    supports = np.asarray([
        per_class[class_names.get(int(label), str(int(label)))]["support"]
        for label in class_values
    ])

    macro_f1 = f1_values.mean()

    weighted_f1 = (
        supports * f1_values
    ).sum() / max(supports.sum(), eps)

    return {
        "confusion": confusion,
        "per_class": per_class,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "class_values": class_values,
    }


def report( sequences, target_distributions, mod_target_distributions ):
    results = compute_prediction_agreement(
        sequences,
        target_distributions,
        mod_target_distributions,
        class_names=['Down', 'Neutral', 'Up']
    )
    
    # Print summary
    print("=" * 70)
    print("PREDICTION AGREEMENT ANALYSIS")
    print(symbol, date, predicted, predictor, str(n_qubits)+'q','Class ', clsName, prediction_loss)
    print("Sequences Lengths", seq_lengths)
    print("=" * 70)
    print(f"Overall Agreement: {results['overall_agreement_pct']:.2f}%")
    print(f"  ({results['total_agreements']}/{results['total_samples']} sequences)\n")
    
    print("Agreement by Sequence Length:")
    for length in sorted(results['agreement_pct_by_length'].keys()):
        pct = results['agreement_pct_by_length'][length]
        count = results['agreements_by_length'][length]
        total = results['totals_by_length'][length]
        print(f"  Length {length}: {pct:6.2f}% ({count}/{total})")
    
    print("\nAgreement by Empirical Predicted Class:")
    for class_name in ['Down', 'Neutral', 'Up']:
        pct = results['agreement_pct_by_class'][class_name]
        count = results['agreements_by_class'][class_name]
        total = results['totals_by_class'][class_name]
        print(f"  {class_name:8s}: {pct:6.2f}% ({count}/{total})")
    
    # Access detailed results for further analysis
    disagreements = [r for r in results['detailed_results'] if not r['agrees']]
    print(f"\nFound {len(disagreements)} disagreements")
    
    
    # Empirical dominant class
    class_values = np.asarray([-1, 0, 1])
    class_names = {
        -1: "Down",
         0: "Neutral",
         1: "Up",
    }
    
    y_true = class_values[
        np.argmax(target_distributions, axis=1)
    ]
    
    # Model dominant class
    y_pred = class_values[
        np.argmax(mod_target_distributions, axis=1)
    ]
    
    metrics = classification_metrics_from_labels(
    y_true=y_true,
    y_pred=y_pred,
    class_values=class_values,
    sample_weight=global_weights,
    )
    
    metrics_unweighted = classification_metrics_from_labels(
        y_true=y_true,
        y_pred=y_pred,
        class_values=class_values,
        sample_weight=None,
    )
    
    print("\nAgreement by Empirical Predicted Class:")
    for class_name in ["Down", "Neutral", "Up"]:
        pct = results["agreement_pct_by_class"][class_name]
        count = results["agreements_by_class"][class_name]
        total = results["totals_by_class"][class_name]
        print(f"  {class_name:8s}: {pct:6.2f}% ({count}/{total})")
    
    print("\nPrecision / Recall / F1 by Empirical Predicted Class:")
    for class_name in ["Down", "Neutral", "Up"]:
        m = metrics["per_class"][class_name]
    
        print(
            f"  {class_name:8s}: "
            f"precision={100*m['precision']:6.2f}%  "
            f"recall={100*m['recall']:6.2f}%  "
            f"F1={100*m['f1']:6.2f}%  "
            f"support={m['support']:.0f}"
        )
    
    print(
        f"\nOverall accuracy: {100*metrics['accuracy']:.2f}%"
    )
    
    print(
        f"Macro F1:        {100*metrics['macro_f1']:.2f}%"
    )
    
    print(
        f"Weighted F1:     {100*metrics['weighted_f1']:.2f}%"
    )
        


# ============================================================================
# USAGE
# ============================================================================
mode = 'validate'
mode = 'train'

clsNames = ['c2','c4','ca2','ca4']
clsName  = 'ca4'
nClasses = 3          #classifiaction griups
d_out = nClasses
# particular learning task parameters
dates = ['202503','20250303','20250304', '20250401', '20250402' ]  # training and validation
symbol= 'AAPL'
symbol= 'INTC'
symbol= 'NVDA'

features     = ['log_mid',"tvi_n" , 'obi_L1', "ofi_L1_n", "ofi_L1_n_norm",'ofi_L1_norm_n','ofi_L3_norm_n','ofi_L10_norm_n',"micro_price",'vpin', 'sigma_W' ]
features     = ['log_mid', "ofi_L1_n_norm",'ofi_L1_norm_n','ofi_L3_norm_n','ofi_L10_norm_n',"micro_price",'vpin', 'sigma_W' ]
features     = ['log_mid', 'sigma_W' ]
predicted  =  features[0]  # 'log_mid'
predictor  =  features[1]

seq_lengths = [1,2,3,4,5]



n_qubits = 5  # system dimension d = 2^n_qubits 
d = 2 ** n_qubits 
    

prediction_loss="mseError"
prediction_loss="xEntropy"
prediction_loss="klDivergence"
prediction_loss="jsDivergence"
prediction_loss = 'mseError'

monthly_data = True
if monthly_data:
    date = dates[0]   # the data is aggregated for 1 month
else:
    date = dates[1]    

date_validation = dates[-2]

#------------------------------------------------------------------
# Pre-trained models location
mPath = '.\\models\\'
#--------------------------------------------------------------------------
# Training Data Load
#--------------------------------------------------------------------------
fPath = '.\\data\\' 
# SEQ_DISTR_AAPL_bivariate_log_mid-micro_price_202504



# -----------------------------
# Step 1: Pretrain encoder on sequence distributions
# -----------------------------
print('Class', clsName, "Predictor", predictor)
print("=" * 60)
print("STEP 1: Pretraining encoder on sequences")
print("=" * 60)

title_model = "PRD_OMEGA_" +clsName+"_"+symbol+"_"+predicted+"-"+predictor+"_"+date+'_'+prediction_loss+'_'+str(n_qubits)+'q'
save_model_file = mPath+title_model

# Configuration
m = 16  # alphabet size (e.g., 4x4 for price+OFI encoding)
##########################################################################
# Encoder System Size
##########################################################################
 
  
# -------------------------------------------------------------------------



variate="bivariate"

n_symbols = 4   #observable symbols per feature

# resampling 
frequency = 100 #events
freq_units = 'evn'





# Load empirical sequence data
# sequences_all, emp_probs_all = load_your_empirical_data()
# Filter by length and probability



if monthly_data:
    # sequences distributions
    #       "SEQ_DISTR_  AAPL    _  bivariate_log_mid-sigma_W_202503"           
    title = "SEQ_DISTR_"+symbol+"_"+variate+"_"+predicted+"-"+predictor+"_"+date
    
    #---------------------------------------------------------------
    #   TEMP _Remove
    #--------------------------------------------------------------
    # title = "SQ_PRB_AAPL_20250303_log_mid_sigma_W"
    title = "SQ_PRB_NVDA_bivariate_log_mid-sigma_W_202503"
    #--------------------------------------------------------------
     
    infname = fPath+title
    distrs_samples =  pickle.load(open( infname, "rb") ) 
    sequences_all = distrs_samples[1]
    emp_probs_all = [i[1] for i in distrs_samples[0]] # empirical proabilities
    print('Loaded Sequence Disributions ', title) 

  
else:  # daily data
    title = "SQ_PRB_"+symbol+"_"+date+"_"+predicted+"_"+predictor
    infname = fPath+title
    distrs_samples =  pickle.load(open( infname, "rb") ) 
    sequences = [list(s) for s in distrs_samples[0]]
    emp_probs = distrs_samples[1]
    

#-----------------------------------------------------------------------------
# Loading Class Distributions - Training
#-----------------------------------------------------------------------------
if monthly_data:     # CLS_DISTR_AAPL__log_mid-sigma_W_202503_ca4
    title = "CLS_DISTR_"+symbol+"_"+"_"+predicted+"-"+predictor+"_"+date+"_"+clsName
    title = "CLS_DISTR_"+symbol+"_"+predicted+"-"+predictor+"_"+date+"_"+clsName
    infname = fPath+title
 
    #print(f"Number of unique sequences: {len(sequences)}")
else:       # CLS_DISTR_AAPL_20250303_log_mid_sigma_W_ca4
    title = "CLS_DISTR_"+symbol+"_"+date+'_'+predicted+"_"+predictor+"_"+clsName
    infname = fPath+title
   
emp_cls_dstrbs =  pickle.load(open( infname, "rb") )
print("Loaded Class Distribution:", title) 


sequences = []
emp_probs = []
emp_dstrb = []
min_seq_prob=0.0001
max_seq_len = 5

for i in range(len(sequences_all)):
    if len(sequences_all[i]) <= max_seq_len and emp_probs_all[i] > min_seq_prob:
        sequences.append(sequences_all[i])
        emp_probs.append(emp_probs_all[i])
        emp_dstrb.append(emp_cls_dstrbs[i][1]) 
        
        assert list(emp_cls_dstrbs[i][0])==sequences_all[i]

target_distributions = emp_dstrb
seq_probs = emp_probs


global_weights = compute_global_weights(sequences, emp_probs)   
training_sequences = sequences
training_seq_probs = emp_probs


  
print(f"Number of training sequences: {len(sequences)}")
print('Loaded from ',title )


# -----------------------------------------------------------------
# Location and name of trained encoder
#------------------------------------------------------------------

if monthly_data:
    mFname = 'QMOD_'+symbol +'_'+date+'_'+predicted+'_'+predictor+'_'+ str(n_qubits)+'q'   

    
    print("Loading pre-trained encoder:",mFname )
    # [model, sequences, emp_probs ]
    encoder = pickle.load( open( mPath + mFname , "rb") )[0]
    #model, sequences, emp_probs = result[0], result[1], result[2]
    #encoder, meta=load_model_weights(mPath + model_file_name, m, n_qubits, learn_rho0=True, device="cpu")
    # model test
    #mod_seq_probs = predict_probs(encoder, sequences)
    print("Loading pre-trained encoder:",mFname )
else:
    # QMOD_AAPL_20250303_log_mid_sigma_W_5q
    mFname = 'QMOD_'+symbol+'_'+date+'_'+predicted+"_"+predictor+'_'+str(n_qubits)+'q'
    print("Loading pre-trained encoder:",mFname )
    encoder =  pickle.load( open( mPath + mFname, "rb") )[0]  #  [model, sequences, emp_probs ]       

if not hasattr(encoder, "rho0_type"):
    encoder.rho0_type = "mixed"


#------------------------------------------------------------------------------
# Validation Data
#-------------------------------------------------------------------------------
validation_sequences = None
validation_targets = None
validation_weights = None

    
#   #--------------------------------------------------------------------------
# Joint Training Encoder-Decoder
#--------------------------------------------------------------------------
freeze_encoder = False                         #  train/no-train the encoder
if freeze_encoder:
    lambda_enc = 0.0                          #  weight for encoder loss
    lambda_pred  = 1                          #  weight for prediction loss
else:
    lambda_enc = 0.02                          #  weight for encoder loss - 0.02
    lambda_pred  = 0.98                        #  weight for prediction loss



batch_size = 8*512
epochs  = 300
print("Training ","Freeze Encoder", freeze_encoder, "Weight encoder ", lambda_enc, "Weight Prediction",lambda_pred )

gpu_id = 3
device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu" 

train_prediction_model = True
if train_prediction_model:
    pred_model = train_predictive_model_bilevel_omega(
        sequences=sequences,
        target_distributions=target_distributions,
        seq_probs=seq_probs,
        global_weights=global_weights,
        encoder=encoder,
        d_out=3,
        batch_size=512,
        lr=1e-3,
        epochs=epochs,
        freeze_encoder=True,
        use_unitary=True,
        lambda_enc=0.0,
        lambda_pred=1.0,
        prediction_loss="mseError",
        device=device,
    )[0] 

    save_prediction_model = False
    if save_prediction_model:
        # Save metadata
        meta = {
            'batch_size': batch_size,
            'epochs_trained': epochs,
            'lambda_enc': lambda_enc ,
            'lambda_pred':lambda_pred,
            'date'       :date, 
            'prediction_loss':prediction_loss,
            }
        
        # save trained model              
        save_predictive_model(save_model_file, pred_model, meta)
        print("Predictive Model saved as", save_model_file)
else:                         #Load prediction model
    pred_model, meta = load_predictive_model(save_model_file, device='cpu')
    print("Predictive Model loaded from", save_model_file)

#-------------------------------------------------------------------------------
mod_seq_probs, mod_target_distributions = (
    get_model_predictions_ordered(
        pred_model,
        sequences,
    )
)

#divergences_by_length = compute_divergences_by_length(sequences,  target_distributions,  mod_target_distributions, divergence_type = "kl")
plot_divergence_by_length(sequences, target_distributions, mod_target_distributions,  mod_seq_probs, emp_probs,
    divergence_type = prediction_loss, figsize=(14, 6),
    title="Performance: "+prediction_loss+" by Seq Length"
)

benchmark_loss = prediction_loss
plot_divergence_by_length(sequences, target_distributions, mod_target_distributions,  
                          mod_seq_probs, emp_probs,
                          divergence_type = benchmark_loss, figsize=(14, 6),
    title="Performance: "+benchmark_loss+" by Seq Length"
)
           

# Performance Evaluation

#-----------------------------------------------------------------------------
# Model Output - generative probabilities and class distributions for 
   
report( sequences, target_distributions, mod_target_distributions )
   
# ---------------------------------------------------------------------        
# Done with in-sample estimated
# Read out of sample data ------------------------------------------------
print('========================================================================')
print('Out of sampe test: Training date', date,' Validation date', date_validation)
print('========================================================================')

#-----------------------------------------------------------------------------
# Load sequences distributions -out of samples
#----------------------------------------------------------------------------
date = date_validation
title = "SQ_PRB_"+symbol+"_"+date+"_"+predicted+"_"+predictor
infname = fPath+title
distrs_samples =  pickle.load(open( infname, "rb") ) # distrs_samples[0] - sequences toules, distrs_samples[1] - local probs
print('Loaded Validation Sequence Disributions ', title) 
#-----------------------------------------------------------
# Load sequence CLASS distributions
#-----------------------------------------------------------
#Read out of sample class data
title = "CLS_DISTR_"+symbol+"_"+date+'_'+predicted+"_"+predictor+"_"+clsName
infname = fPath+title
   
target_distributions =  pickle.load(open( infname, "rb") )#[[(seq),[p0,p1,p2]]]

print('Loaded Validation Class Disributions ', title) 

emp_cls_dstrbs = target_distributions


print('-----------------------------------------------------------------------')
print('Validation for  max sequence length ',  max_seq_len)
print('-----------------------------------------------------------------------')
sequences = []
emp_probs = []
emp_dstrb = []

sequences_all = distrs_samples[0]
emp_probs_all = distrs_samples[1]


for i in range(len(sequences_all)):
    if len(sequences_all[i]) <= max_seq_len and emp_probs_all[i] > min_seq_prob:
        sequences.append(list(sequences_all[i]))
        emp_probs.append(emp_probs_all[i])
        emp_dstrb.append(emp_cls_dstrbs[i][1]) 
        
        assert emp_cls_dstrbs[i][0]==sequences_all[i]

       
global_weights = compute_global_weights(sequences, emp_probs)   




# Apply model out of sample      
mod_seq_probs, mod_target_distributions = get_model_predictions_ordered(pred_model, sequences)
        
print("Out of Sample Report for ", date)
report( sequences, emp_dstrb, mod_target_distributions )    


print('---------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print('Validation for  max sequence length ',  max_seq_len-1)
print('-----------------------------------------------------------------------')
sequences = []
emp_probs = []
emp_dstrb = []

sequences_all = distrs_samples[0]
emp_probs_all = distrs_samples[1]


for i in range(len(sequences_all)):
    if len(sequences_all[i]) <= max_seq_len-1 and emp_probs_all[i] > min_seq_prob:
        sequences.append(list(sequences_all[i]))
        emp_probs.append(emp_probs_all[i])
        emp_dstrb.append(emp_cls_dstrbs[i][1]) 
        
        assert emp_cls_dstrbs[i][0]==sequences_all[i]

       
global_weights = compute_global_weights(sequences, emp_probs)   




# Apply model out of sample      
mod_seq_probs, mod_target_distributions = get_model_predictions_ordered(pred_model, sequences)
        
print("Out of Sample Report for ", date)
report( sequences, emp_dstrb, mod_target_distributions )         



