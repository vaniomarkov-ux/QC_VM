# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:17:45 2026

@author: vanio
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
from typing import Literal, Optional, Tuple, List

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
from plot_distributions import plotDistribution, plotDistributions

import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

from processing_results import ExperimentTracker
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
class QuantumDecoder_old(nn.Module):
    def __init__(self, d_in: int, d_out: int, use_unitary: bool = True, 
                      eps: float = 1e-8):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.use_unitary = use_unitary
        self.eps = eps
        
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
        # Normalize columns to make V†V = I
        G = V.conj().T @ V
        G = 0.5 * (G + G.conj().T)
        
        w, Q = torch.linalg.eigh(G)
        w = torch.clamp(w, min=self.eps)
        inv_sqrt = (Q * w.rsqrt()) @ Q.conj().T
        
        V_normalized = V @ inv_sqrt
        return V_normalized
#------------------------------------------------------------------------------
# Forward pass and prediction
    def forward(self, rho_batch):
        """
        rho_batch: [B, d_in, d_in] batch of density matrices
        returns: [B, d_out] unnormalized prediction logits
        """
        U = self.get_unitary()          # [d_in, d_in]
        V = self.get_coisometry()       # [d_in, d_out]
        
        # Apply unitary: rho' = U rho U†
        rho_rot = torch.bmm(torch.bmm(U.unsqueeze(0).expand(rho_batch.size(0), -1, -1), 
                                       rho_batch),
                            U.conj().T.unsqueeze(0).expand(rho_batch.size(0), -1, -1))
        
        # Apply co-isometry: rho_pred = V† rho' V
        rho_pred = torch.bmm(torch.bmm(V.conj().T.unsqueeze(0).expand(rho_batch.size(0), -1, -1),
                                       rho_rot),
                             V.unsqueeze(0).expand(rho_batch.size(0), -1, -1))
        
        # Extract diagonal (prediction probabilities)
        logits = torch.real(torch.diagonal(rho_pred, dim1=-2, dim2=-1))
        return logits  # [B, d_out]
    
    def predict_probs(self, rho_batch):
        """
        Returns normalized probabilities over d_out outcomes
        """
        logits = self.forward(rho_batch)
        logits = torch.clamp(logits, min=0.0)
        probs = logits / (logits.sum(dim=-1, keepdim=True) + self.eps)
        return probs
#------------------------------------------------------------------------------
# Encoder
#------------------------------------------------------------------------------
# ---------------------------
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
    """Collate function for bilevel dataset with both prob types"""
    seqs, target_dists, seq_probs, global_weights = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    T = int(lens.max())
    
    # Pad sequences
    seq_pad = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        seq_pad[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    
    # Stack distributions and probabilities
    target_dists = torch.tensor(target_dists, dtype=torch.float32)    # [B, d_out]
    seq_probs = torch.tensor(seq_probs, dtype=torch.float32)          # [B] conditional
    global_weights = torch.tensor(global_weights, dtype=torch.float32) # [B] global
    
    return seq_pad, lens, target_dists, seq_probs, global_weights



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
    
    d_in = encoder.d
    decoder = QuantumDecoder(d_in=d_in, d_out=d_out, use_unitary=use_unitary)
    
    model = PredictiveQuantumModel(encoder, decoder, freeze_encoder=freeze_encoder)
    model = model.to(device)
    
    # Dataset with bilevel structure
    ds = PredictiveSeqDatasetBilevel(sequences, target_distributions, seq_probs, global_weights)
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



    



#------------------------------------------------------------------------------
# Train Predictive Model with Dynamic loss weight
def train_predictive_model_dynamic(
    sequences,
    target_distributions,
    seq_probs,
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
    lambda_enc: float = 1.0,       # Can be 'auto' for dynamic
    lambda_pred: float = 1.0,
    use_dynamic_weights: bool = False,  # NEW: enable dynamic weighting
):
    """
    Train with optional dynamic loss weighting
    """
    
    # ... (model setup as before) ...
    
    # Initialize dynamic weights if requested
    if use_dynamic_weights:
        loss_weights = DynamicLossWeights(alpha=0.95)
        print("Using dynamic loss weighting")
    else:
        loss_weights = None
        print(f"Using fixed weights: λ_enc={lambda_enc}, λ_pred={lambda_pred}")
    
    # Training loop
    for ep in range(1, epochs + 1):
        model.train()
        total_enc_loss = 0.0
        total_pred_loss = 0.0
        total_loss = 0.0
        n_seen = 0
        
        for seq_pad, lens, target_dist, seq_prob in dataloader:
            seq_pad = seq_pad.to(device, non_blocking=True)
            target_dist = target_dist.to(device, non_blocking=True)
            seq_prob = seq_prob.to(device, non_blocking=True)
            
            # Forward pass with Option C (unnormalized encoding)
            probs, traces = model(seq_pad)  # probs: [B, d_out], traces: [B]
            
            # ===== ENCODER LOSS =====
            p_enc = torch.clamp(traces, min=1e-12)
            seq_prob_normalized = seq_prob / (seq_prob.sum() + 1e-12)
            enc_loss = -torch.mean(seq_prob_normalized * torch.log(p_enc))
            
            # weights define the target distribution over the sampled sequences
            #enc_loss = (w * (-torch.log(p_mdl))).sum() / wsum
            
            
            
            
            
            # ===== PREDICTION LOSS =====
            # pred_loss_per_sample = -(target_dist * torch.log(probs + 1e-12)).sum(dim=-1)
            # pred_loss = (seq_prob * pred_loss_per_sample).sum() / (seq_prob.sum() + 1e-12)
            
            pred_loss_per_sample = -(target_dist * torch.log(probs + 1e-12)).sum(dim=-1)  # [B]
            pred_loss = (global_weights * pred_loss_per_sample).sum() / (global_weights.sum() + 1e-12)
            
            
            # ===== DYNAMIC WEIGHTING (if enabled) =====
            if use_dynamic_weights:
                loss_weights.update(enc_loss, pred_loss)
                lambda_enc_curr, lambda_pred_curr = loss_weights.get_weights()
            else:
                lambda_enc_curr = lambda_enc
                lambda_pred_curr = lambda_pred
            
            # ===== COMBINED LOSS =====
            loss = lambda_enc_curr * enc_loss + lambda_pred_curr * pred_loss
            
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
        
        if use_dynamic_weights:
            lambda_enc_curr, lambda_pred_curr = loss_weights.get_weights()
            print(f"Epoch {ep:3d} | {mode} | "
                  f"Total: {avg_total_loss:.6f} | "
                  f"Enc: {avg_enc_loss:.6f} (λ={lambda_enc_curr:.3f}) | "
                  f"Pred: {avg_pred_loss:.6f} (λ={lambda_pred_curr:.3f})")
        else:
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
    model: PredictiveQuantumModel,
    sequences,
    batch_size: int = 512,
    device: str = "cpu"
):
    """
    Get model predictions for all sequences in order
    
    Args:
        model: trained PredictiveQuantumModel
        sequences: list of sequences (in order)
        batch_size: batch size for processing
        device: computation device
    
    Returns:
        mod_seq_probs: list of p_model(sequence) in same order
        mod_target_distributions: list of p_model(class|sequence) in same order
    """
    model.eval()  # Set to evaluation mode
    model = model.to(device)
    
    mod_seq_probs = []
    mod_target_distributions = []
    
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
        probs, traces = model(seq_pad)  # probs: [B, d_out], traces: [B]
        
        # Extract and append (detach + numpy)
        for j in range(len(batch_seqs)):
            mod_seq_probs.append(float(traces[j].detach().cpu().item()))
            mod_target_distributions.append(probs[j].detach().cpu().numpy().tolist())
    
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


clsNames = ['c1','c2','c4','ca2','ca4']
clsNames = ['c2','c4','ca2','ca4']
class_type = 1 # #1 of 6 classes - one step ahead
clsName    = clsNames[class_type-1]


features     = ['log_mid',"tvi_n" , 'obi_L1', "ofi_L1_n", "ofi_L1_n_norm",'ofi_L1_norm_n','ofi_L3_norm_n','ofi_L10_norm_n',"micro_price",'vpin', 'sigma_W' ]
features     = ['log_mid', "ofi_L1_n_norm",'ofi_L1_norm_n','ofi_L3_norm_n','ofi_L10_norm_n',"micro_price",'vpin', 'sigma_W' ]
features     = ['log_mid', 'sigma_W' ]
predicted =  features[0]  # 'log_mid'

seq_lengths = [1,2,3,4,5]

# Save Performance Results
# 1. Initialize the tracker
db_name="QuantumEncoderDecoderPerformance"
tracker = ExperimentTracker(db_name)

clsNames = ['ca4']
    
for clsName in clsNames:
    for predictor in features[1:]:
            
        # -----------------------------
        # Step 1: Pretrain encoder on sequence distributions
        # -----------------------------
        print('Class', clsName, "Predictor", predictor)
        print("=" * 60)
        print("STEP 1: Pretraining encoder on sequences")
        print("=" * 60)
        
        # Configuration
        m = 16  # alphabet size (e.g., 4x4 for price+OFI encoding)
        ##########################################################################
        # Encoder System Size
        ##########################################################################
 
      
        # -------------------------------------------------------------------------
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
    title = "CLS_DISTR_"+symbol+"__"+date+'_'+predicted+"_"+predictor+"_"+clsName
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

learn_rho0 = True
device = "cuda" if torch.cuda.is_available() else "cpu"
rho0_type = "mixed"  # or  "pure"

# Initialize clean encoder
encoder = KrausInstrument(
    m=m, 
    d=d, 
    learn_rho0=learn_rho0,
    rho0_type=rho0_type
).to(device)

# Verify initialization
print(f"Clean Encoder initialized:")
print(f"  Alphabet size (m): {encoder.m}")
print(f"  System dimension (d): {encoder.d}")
print(f"  Initial state type: {encoder.rho0_type}")
print(f"  Learn ρ₀: {encoder.learn_rho0}")
  

# -----------------------------
# Step 2: Prepare predictive dataset
# -----------------------------
print("\n" + "=" * 60)
print("STEP 2: Preparing predictive dataset")
print("=" * 60)



# Load sequences with target labels
# Each sequence is a prefix, target is the next symbol to predict
# pred_sequences, pred_targets, pred_emp_probs = load_predictive_data()

# Example: synthetic data for illustration
# pred_sequences = [[0,1,2], [1,2,3], [2,3,4], ...]
# pred_targets = [3, 4, 5, ...]  # next mid-price symbol
# pred_emp_probs = [0.01, 0.015, 0.008, ...]


# include class type in the class distibutions file name
###########################################################################
# Training can be done using individual or joint training data
###########################################################################


if monthly_data:
    title = "CLS_DISTR_"+symbol+"_"+"_"+predicted+"-"+predictor+"_"+date+"_"+clsName
    infname = fPath+title
    
    class_distributions =  pickle.load(open( infname, "rb") ) 
    
    print("Loaded:", title) 
    sequences, target_distributions, seq_probs = integrate_data(class_distributions, distrs_samples)
    
    # calculate empirical weight of a sequence 
    global_weights = compute_global_weights(sequences, emp_probs)

    print(f"Number of unique sequences: {len(sequences)}")
else:
    title = "CLS_DISTR_"+symbol+"_"+date+'_'+predicted+"_"+predictor+"_"+clsName
    infname = fPath+title
   
    target_distributions =  pickle.load(open( infname, "rb") )
    
    target_distributions = [d[1] for d in target_distributions]
    
    seq_probs = emp_probs
    print("Loaded:", title) 
    

# calculate empirical weight of a sequence 
global_weights = compute_global_weights(sequences, seq_probs)
 
   
# -----------------------------
# Step 3: Train full predictive model
# -----------------------------
#prediction_loss in 'xEntropy', 'klDivergence', 'jsDivergence', 'mseError'"
prediction_loss = 'mseError'
print("\n" + "=" * 60)
print("STEP 3: Training predictive model (encoder + decoder)",  predictor, predicted,"Loss=", prediction_loss, 'Class-=', clsName)
print("Sequences Lengths", seq_lengths)
print("=" * 60)

d_out = 3  # prediction target dimension (e.g., 4 mid-price symbols)

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


pred_model =  train_predictive_model_bilevel(
sequences,             # list of sequences  
target_distributions,  # sequence induced class distributions
seq_probs,             # sequence probabilities wrt same length distributions
global_weights,        # ← GLOBAL Empirical weights
encoder = encoder,
d_out=d_out,
batch_size=batch_size,
lr = 1e-3,
epochs  = epochs,
#freeze_encoder = True,
freeze_encoder = freeze_encoder,
use_unitary = True,
device  = "cuda" if torch.cuda.is_available() else "cpu",
optimizer_name = "adam",
weight_decay = 1e-4,
encoder_lr_multiplier = 0.1,
lambda_enc = lambda_enc  ,   
lambda_pred  = lambda_pred,  
prediction_loss = prediction_loss,
num_workers = 0,
)

# Save metadata

meta = {
'batch_size': batch_size,
'epochs_trained': epochs,
'lambda_enc': lambda_enc ,
'lambda_pred':lambda_pred,
'date'       :date, 
'prediction_loss':prediction_loss,
}


#title = "PRD_" +str(class_type)+"_"+symbol+"_"+predicted+"-"+predictor+"_"+date+'_'+prediction_loss+'_'+str(n_qubits)+'q'
title = "PRD_" +clsName+"_"+symbol+"_"+predicted+"-"+predictor+"_"+date+'_'+prediction_loss+'_'+str(n_qubits)+'q'
save_file = mPath+title
save_predictive_model(save_file, pred_model)
print("Predictive Model saved as", save_file)

# Load
# pred_model_loaded, meta_loaded = load_predictive_model('pred_model_v1.pt', device='cuda')

# print(f"Loaded model trained for {meta_loaded['epochs_trained']} epochs")
# print(f"Final training loss: {meta_loaded['final_loss']}")

# Use for inference
# result = predict_from_sequence(pred_model_loaded, [0, 1, 2])


#-----------------------------------------------------------------------------
# Model Output - generative probabilities and class distributions for 

# apply model for set of sequences

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



