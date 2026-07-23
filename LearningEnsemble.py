# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:17:45 2026

@author: vanio
"""
import pickle
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
from typing import Literal, Optional, Tuple, List
from collections import OrderedDict

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
from plot_distributions import plotDistribution, plotDistributions

import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
#-----------------------------------------------------------------------------
# SAVE -- LOAD Full predictive model: Encoder + Decoder
#-----------------------------------------------------------------------------
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
#------------------------------------------------------------------------------
# Save/Load Locally pre-trained encoder
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
def predict_encoder_probs(model, sequences, batch_size=2048, device=None):
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

# -----------------------------------------------------------------------------
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
#------------------------------------------------------------------------------
# Related to the encoder - the generative model
#------------------------------------------------------------------------------
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


#------------------------------------------------------------------------------
# Decoder class
#------------------------------------------------------------------------------

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
    
    def get_coisometry0(self):
        V = torch.complex(self.V_re, self.V_im)
        G = V.conj().T @ V
        G = 0.5 * (G + G.conj().T)
        
        w, Q = torch.linalg.eigh(G)
        w = torch.clamp(w, min=self.eps)
        inv_sqrt = (Q * w.rsqrt()) @ Q.conj().T
        
        V_normalized = V @ inv_sqrt
        return V_normalized
    
    def get_coisometry(self):
        """
        Return V_iso with V_iso.conj().T @ V_iso = I.
    
        Shape:
            V_iso: [d_in, d_out]
        """
    
        V = torch.complex(self.V_re, self.V_im)
    
        if not torch.isfinite(V.real).all() or not torch.isfinite(V.imag).all():
            raise FloatingPointError("Non-finite entries found in decoder V.")
    
        # QR is more stable than eigendecomposition of V†V.
        Q, R = torch.linalg.qr(V, mode="reduced")
    
        # Q has shape [d_in, d_out] and Q†Q = I.
        return Q    

    def get_coisometry1(self):
        """
        SVD/polar version.
        Produces V_iso with V_iso.conj().T @ V_iso = I.
        """
    
        V = torch.complex(self.V_re, self.V_im)
    
        if not torch.isfinite(V.real).all() or not torch.isfinite(V.imag).all():
            raise FloatingPointError("Non-finite entries found in decoder V.")
    
        U, S, Vh = torch.linalg.svd(V, full_matrices=False)
    
        V_iso = U @ Vh
    
        return V_iso

    
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
        num_workers=0,
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
# Related to the Encoder + Decoder
#-----------------------------------------------------------------------------
class PredictiveSeqDataset(Dataset):
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
# Ensemble of encoders
class MultiEncoderQuantumEnsemble(nn.Module):
    def __init__(
        self,
        encoders: List[KrausInstrument],
        d_out: int,
        use_unitary: bool = True,
        normalization_point: str = "input",  # Where to normalize individual encoder outputs
        eps: float = 1e-8
    ):
        """
        Multi-encoder quantum ensemble
        
        Args:
            encoders: list of n pretrained/trainable KrausInstrument encoders
            d_out: output prediction dimension
            use_unitary: whether to use entangling unitary on product state
            normalization_point: "input" (normalize each encoder), "after_unitary", "output"
        """
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.n_encoders = len(encoders)
        self.d_out = d_out
        self.normalization_point = normalization_point
        self.eps = eps
        
        # Verify all encoders have same dimension
        self.d = encoders[0].d
        assert all(enc.d == self.d for enc in encoders), \
            "All encoders must have same system dimension"
        
        # Product state dimension
        self.d_product = self.d ** self.n_encoders
        
        # Decoder on product state
        self.decoder = QuantumDecoder(
            d_in=self.d_product,
            d_out=d_out,
            use_unitary=use_unitary,
            normalization_point="output",  # Decoder always normalizes at output
            eps=eps
        )
    
    def encode_sequences_multi(self, seq_pad):
        """
        Encode sequences with all encoders independently
        
        Args:
            seq_pad: [B, T] padded sequences
        
        Returns:
            rho_list: list of n density matrices, each [B, d, d]
            traces_list: list of n trace vectors, each [B]
        """
        rho_list = []
        traces_list = []
        
        for encoder in self.encoders:
            # Each encoder processes same sequence independently
            rho_unnorm = self._encode_with_single_encoder(encoder, seq_pad)  # [B, d, d]
            
            # Get traces (sequence probabilities from this encoder)
            traces = torch.real(torch.diagonal(rho_unnorm, dim1=-2, dim2=-1).sum(-1))  # [B]
            traces = torch.clamp(traces, min=self.eps)
            
            rho_list.append(rho_unnorm)
            traces_list.append(traces)
        
        return rho_list, traces_list
    
    def _encode_with_single_encoder(self, encoder, seq_pad):
        """
        Encode sequences with a single encoder (unnormalized)
        """
        device = seq_pad.device
        B, T = seq_pad.shape
        d = encoder.d
        
        K = encoder.kraus_operators()
        rho0 = encoder._make_rho0(device)
        
        rho = rho0.unsqueeze(0).expand(B, d, d).clone()
        
        for t in range(T):
            sym = seq_pad[:, t]
            active = (sym != -1)  # PAD = -1
            if not torch.any(active):
                break
            
            sym_a = sym[active].long()
            rho_a = rho[active]
            
            Kt = K.index_select(0, sym_a)
            rho_a = torch.bmm(torch.bmm(Kt, rho_a), Kt.conj().transpose(1, 2))
            
            rho[active] = rho_a
        
        return rho
    
    def normalize_encoder_outputs(self, rho_list, traces_list):
        """
        Normalize encoder outputs: ρ → ρ / Tr(ρ)
        
        Args:
            rho_list: list of [B, d, d] unnormalized density matrices
            traces_list: list of [B] traces
        
        Returns:
            rho_normalized_list: list of [B, d, d] normalized density matrices
        """
        rho_normalized_list = []
        
        for rho, traces in zip(rho_list, traces_list):
            rho_norm = rho / traces.unsqueeze(-1).unsqueeze(-1)
            rho_normalized_list.append(rho_norm)
        
        return rho_normalized_list
    
    def build_product_state(self, rho_list):
        """
        Build tensor product state: ρ_product = ρ₁ ⊗ ρ₂ ⊗ ... ⊗ ρₙ
        
        Args:
            rho_list: list of n density matrices, each [B, d, d]
        
        Returns:
            rho_product: [B, d^n, d^n] product state
        """
        B = rho_list[0].size(0)
        
        # Start with first encoder's state
        rho_product = rho_list[0]  # [B, d, d]
        
        # Tensor product with each subsequent encoder
        for i in range(1, self.n_encoders):
            rho_product = self._tensor_product_batch(rho_product, rho_list[i])
        
        return rho_product  # [B, d^n, d^n]
    
    def _tensor_product_batch(self, rho1, rho2):
        """
        Compute batched tensor product: ρ₁ ⊗ ρ₂
        
        Args:
            rho1: [B, d1, d1]
            rho2: [B, d2, d2]
        
        Returns:
            rho_product: [B, d1*d2, d1*d2]
        """
        B = rho1.size(0)
        d1 = rho1.size(1)
        d2 = rho2.size(1)
        
        # Reshape for Kronecker product
        # ρ₁ ⊗ ρ₂ = reshape(ρ₁[:,:,None,None] * ρ₂[None,None,:,:])
        rho_product = torch.einsum('bij,bkl->bikjl', rho1, rho2)
        rho_product = rho_product.reshape(B, d1*d2, d1*d2)
        
        return rho_product
    
    def forward(self, seq_pad):
        """
        Full pipeline: multi-encode → normalize → product state → decoder
        
        Args:
            seq_pad: [B, T] padded sequences
        
        Returns:
            probs: [B, d_out] prediction probabilities
            traces_list: list of [B] sequence probabilities from each encoder
        """
        # Step 1: Encode with all encoders independently
        rho_list, traces_list = self.encode_sequences_multi(seq_pad)
        
        # Step 2: Normalize individual encoder outputs (if normalization_point="input")
        if self.normalization_point == "input":
            rho_list = self.normalize_encoder_outputs(rho_list, traces_list)
        
        # Step 3: Build product state
        rho_product = self.build_product_state(rho_list)  # [B, d^n, d^n]
        
        # Step 4: Apply decoder (unitary + co-isometry)
        probs = self.decoder.predict_probs(rho_product)  # [B, d_out]

        
        return probs, traces_list
    
    def get_encoder_contributions(self, seq_pad):
        """
        Analyze individual encoder contributions
        
        Args:
            seq_pad: [B, T] padded sequences
        
        Returns:
            dict with individual encoder states and statistics
        """
        rho_list, traces_list = self.encode_sequences_multi(seq_pad)
        
        contributions = {}
        for i, (rho, traces) in enumerate(zip(rho_list, traces_list)):
            contributions[f'encoder_{i}'] = {
                'rho': rho.detach().cpu().numpy(),
                'traces': traces.detach().cpu().numpy(),
                'mean_trace': float(traces.mean().cpu()),
                'std_trace': float(traces.std().cpu()),
            }
        
        return contributions

    
    def freeze_encoders(self, freeze: bool = True):
        """Freeze/unfreeze all encoder parameters"""
        for encoder in self.encoders:
            for param in encoder.parameters():
                param.requires_grad = not freeze
        
        if freeze:
            print(f"✓ Froze {self.n_encoders} encoders")
        else:
            print(f"✓ Unfroze {self.n_encoders} encoders")
    
    def freeze_decoder(self, freeze: bool = True):
        """Freeze/unfreeze decoder parameters"""
        for param in self.decoder.parameters():
            param.requires_grad = not freeze
        
        if freeze:
            print("✓ Froze decoder")
        else:
            print("✓ Unfroze decoder")

    def get_model_info(self, print_info=True):
        """
        Return and optionally print a summary of the multi-encoder ensemble.
    
        Assumes:
            self.encoders
            self.n_encoders
            self.d_out
            self.use_unitary
            self.normalization_point
    
        but uses getattr(...) so it remains robust if some attributes are absent.
        """
    
        def count_params(module):
            if not hasattr(module, "parameters"):
                return 0, 0
    
            total = sum(
                p.numel()
                for p in module.parameters()
            )
    
            trainable = sum(
                p.numel()
                for p in module.parameters()
                if p.requires_grad
            )
    
            return total, trainable
    
        encoders = list(self.encoders)
    
        encoder_infos = []
        encoder_dims = []
        product_dim = 1
        product_dim_known = True
    
        for i, enc in enumerate(encoders):
    
            d = getattr(enc, "d", None)
            dim = getattr(enc, "dim", None)
            n_qubits = getattr(enc, "n_qubits", None)
    
            if d is None and dim is not None:
                d = dim
    
            if d is None and n_qubits is not None:
                d = 2 ** int(n_qubits)
    
            if d is not None:
                d = int(d)
                encoder_dims.append(d)
                product_dim *= d
            else:
                encoder_dims.append(None)
                product_dim_known = False
    
            total_params, trainable_params = count_params(enc)
    
            encoder_infos.append(
                {
                    "index": i,
                    "class": enc.__class__.__name__,
                    "d": d,
                    "n_qubits": n_qubits,
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "frozen": trainable_params == 0,
                }
            )
    
        total_params, trainable_params = count_params(self)
    
        info = {
            "class": self.__class__.__name__,
            "n_encoders": getattr(self, "n_encoders", len(encoders)),
            "d_out": getattr(self, "d_out", None),
            "use_unitary": getattr(self, "use_unitary", None),
            "normalization_point": getattr(self, "normalization_point", None),
            "encoder_dims": encoder_dims,
            "product_dim": product_dim if product_dim_known else None,
            "density_matrix_shape": (
                (product_dim, product_dim)
                if product_dim_known
                else None
            ),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "encoder_infos": encoder_infos,
        }
    
        if print_info:
            print("\nMulti-Encoder Quantum Ensemble")
            print("-" * 40)
            print(f"Number of encoders:     {info['n_encoders']}")
            print(f"Output dimension:       {info['d_out']}")
            print(f"Use unitary decoder:    {info['use_unitary']}")
            print(f"Normalization point:    {info['normalization_point']}")
    
            print("\nEncoder dimensions:")
            for enc_info in encoder_infos:
                print(
                    f"  Encoder {enc_info['index']}: "
                    f"class={enc_info['class']}, "
                    f"d={enc_info['d']}, "
                    f"n_qubits={enc_info['n_qubits']}, "
                    f"trainable_params={enc_info['trainable_params']:,}, "
                    f"total_params={enc_info['total_params']:,}, "
                    f"frozen={enc_info['frozen']}"
                )
    
            if product_dim_known:
                print("\nJoint decoder input:")
                print(f"  Product Hilbert dimension: {product_dim}")
                print(
                    f"  Density matrix shape:      "
                    f"{product_dim} x {product_dim}"
                )
            else:
                print("\nJoint decoder input:")
                print("  Product Hilbert dimension: unknown")
    
            print("\nParameters:")
            print(f"  Total parameters:     {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print("-" * 40)
    
        return info            
#------------------------------------------------------------------------------
class MultiEncoderDataset(Dataset):
    """Dataset for multi-encoder ensemble"""
    def __init__(self, sequences_list, target_distributions, global_weights):
        """
        sequences_list: list of n sequence lists (one per encoder)
        target_distributions: [N, d_out] class distributions
        global_weights: [N] global probabilities
        """
        self.n_encoders = len(sequences_list)
        self.sequences_list = sequences_list
        self.target_dists = target_distributions
        self.global_weights = global_weights
        
        # Verify consistency
        n = len(sequences_list[0])
        assert all(len(seqs) == n for seqs in sequences_list)
        assert len(target_distributions) == n
        assert len(global_weights) == n
        
    def __len__(self):
        return len(self.sequences_list[0])
    
    def __getitem__(self, idx):
        # Return sequences from all encoders + target + weight
        seqs = [self.sequences_list[i][idx] for i in range(self.n_encoders)]
        return (*seqs, self.target_dists[idx], self.global_weights[idx])


def collate_multi_encoder(batch, n_encoders):
    """Collate function for multi-encoder dataset"""
    # Unpack batch
    # Each item: (seq_enc0, seq_enc1, ..., seq_encN, target_dist, global_weight)
    
    # Separate by encoder
    sequences_by_encoder = [[] for _ in range(n_encoders)]
    target_dists = []
    global_weights = []
    
    for item in batch:
        for i in range(n_encoders):
            sequences_by_encoder[i].append(item[i])
        target_dists.append(item[n_encoders])
        global_weights.append(item[n_encoders + 1])
    
    # Pad sequences for each encoder
    seq_pads = []
    for seqs in sequences_by_encoder:
        lens = [len(s) for s in seqs]
        T = max(lens)
        
        seq_pad = torch.full((len(seqs), T), -1, dtype=torch.long)  # PAD = -1
        for i, s in enumerate(seqs):
            seq_pad[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        
        seq_pads.append(seq_pad)
    
    # Stack target distributions and weights
    target_dists = torch.tensor(target_dists, dtype=torch.float32)
    global_weights = torch.tensor(global_weights, dtype=torch.float32)
    
    return (*seq_pads, target_dists, global_weights)



#------------------------------------------------------------------------------
def train_multi_encoder_ensemble(
    ensemble_model: MultiEncoderQuantumEnsemble,
    sequences_list: List[List],  # List of n sequence lists (one per encoder)
    target_distributions,
    global_weights,
    d_out: int,
    batch_size: int = 512,
    lr: float = 1e-3,
    epochs: int = 100,
    freeze_encoders: bool = False,
    freeze_decoder: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    optimizer_name: str = "adam",
    weight_decay: float = 1e-4,
    encoder_lr_multiplier: float = 0.1,
    prediction_loss: str = "ce",  # "ce", "kl", "js", "mse"
    lambda_enc: float = 0.0,      # Weight for encoder loss (0 = ignore)
    lambda_pred: float = 1.0,
):
    """
    Train multi-encoder quantum ensemble
    
    Args:
        ensemble_model: MultiEncoderQuantumEnsemble instance
        sequences_list: list of n sequence lists, one per encoder
            e.g., [price_ofi_seqs, price_ovi_seqs, price_vol_seqs]
            All must have same length and order!
        target_distributions: empirical class distributions [N, d_out]
        global_weights: global probability weights [N]
        d_out: output dimension
        freeze_encoders: if True, only train decoder
        freeze_decoder: if True, only train encoders (unusual)
        encoder_lr_multiplier: slower LR for encoders if fine-tuning
        prediction_loss: loss type ("ce", "kl", "js", "mse")
        lambda_enc: weight for encoder loss (can be 0 for decoder-only)
        lambda_pred: weight for prediction loss
    
    Returns:
        trained ensemble_model
    """
    
    # Verify input consistency
    n_encoders = ensemble_model.n_encoders
    assert len(sequences_list) == n_encoders, \
        f"sequences_list must have {n_encoders} elements (one per encoder)"
    
    n_samples = len(sequences_list[0])
    assert all(len(seqs) == n_samples for seqs in sequences_list), \
        "All sequence lists must have same length"
    assert len(target_distributions) == n_samples
    assert len(global_weights) == n_samples
    
    # Set freeze states
    ensemble_model.freeze_encoders(freeze_encoders)
    ensemble_model.freeze_decoder(freeze_decoder)
    
    ensemble_model = ensemble_model.to(device)
    
    # Create dataset
    # We need to pad sequences from all encoders
    ds = MultiEncoderDataset(sequences_list, target_distributions, global_weights)
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_multi_encoder(batch, n_encoders),
        pin_memory=(device.startswith("cuda")),
    )
    
    # Setup optimizer
    if not freeze_encoders and not freeze_decoder:
        # Train both with different learning rates
        param_groups = [
            {
                'params': [p for enc in ensemble_model.encoders for p in enc.parameters()],
                'lr': lr * encoder_lr_multiplier,
                'name': 'encoders'
            },
            {
                'params': ensemble_model.decoder.parameters(),
                'lr': lr,
                'name': 'decoder'
            }
        ]
        print(f"Joint training: Encoder LR={lr * encoder_lr_multiplier:.2e}, Decoder LR={lr:.2e}")
    else:
        # Train only unfrozen parts
        trainable_params = [p for p in ensemble_model.parameters() if p.requires_grad]
        param_groups = [{'params': trainable_params, 'lr': lr}]
        
        if freeze_encoders:
            print(f"Decoder-only training: Decoder LR={lr:.2e}")
        elif freeze_decoder:
            print(f"Encoder-only training: Encoder LR={lr:.2e}")
    
    opt = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    
    # Training loop
    for ep in range(1, epochs + 1):
        ensemble_model.train()
        total_enc_loss = 0.0
        total_pred_loss = 0.0
        total_loss = 0.0
        n_seen = 0
        
        for batch_data in dataloader:
            # Unpack batch: seq_pads for each encoder + target_dist + global_weight
            *seq_pads, target_dist, global_weight = batch_data
            
            # Move to device
            seq_pads = [sp.to(device, non_blocking=True) for sp in seq_pads]
            target_dist = target_dist.to(device, non_blocking=True)  # [B, d_out]
            global_weight = global_weight.to(device, non_blocking=True)  # [B]
            
            # For multi-encoder, we need to use the SAME sequence for all encoders
            # (assuming all encoders see same sequence, just trained on different features)
            seq_pad = seq_pads[0]  # Use first sequence (all should be identical)
            
            # Forward pass
            probs, traces_list = ensemble_model(seq_pad)  # probs: [B, d_out]
            
            # ===== ENCODER LOSS =====
            enc_loss = torch.tensor(0.0, device=device)
            
            if lambda_enc > 0:
                # Average encoder loss across all encoders
                for traces in traces_list:
                    p_enc = torch.clamp(traces, min=1e-12)
                    global_weight_normalized = global_weight / (global_weight.sum() + 1e-12)
                    enc_loss += -(global_weight_normalized * torch.log(p_enc)).sum()
                
                enc_loss = enc_loss / len(traces_list)  # Average across encoders
            
            # ===== PREDICTION LOSS =====
            probs = torch.clamp(probs, min=1e-12, max=1.0)
            target_dist = torch.clamp(target_dist, min=1e-12, max=1.0)
            
            global_weight_normalized = global_weight / (global_weight.sum() + 1e-12)
            
            if prediction_loss == "ce":
                # Cross-entropy
                pred_loss_per_sample = -(target_dist * torch.log(probs)).sum(dim=-1)
                
            elif prediction_loss == "kl":
                # KL divergence
                log_target = torch.log(target_dist)
                log_probs = torch.log(probs)
                pred_loss_per_sample = (target_dist * (log_target - log_probs)).sum(dim=-1)
                
            elif prediction_loss == "js":
                # Jensen-Shannon
                m = 0.5 * (target_dist + probs)
                log_target = torch.log(target_dist)
                log_probs = torch.log(probs)
                log_m = torch.log(m)
                pred_loss_per_sample = (
                    0.5 * (target_dist * (log_target - log_m)).sum(dim=-1) +
                    0.5 * (probs * (log_probs - log_m)).sum(dim=-1)
                )
                
            elif prediction_loss == "mse":
                # Mean squared error
                pred_loss_per_sample = ((target_dist - probs) ** 2).sum(dim=-1)
            
            else:
                raise ValueError(f"Unknown prediction_loss: {prediction_loss}")
            
            pred_loss = (global_weight_normalized * pred_loss_per_sample).sum()
            
            # ===== COMBINED LOSS =====
            loss = lambda_enc * enc_loss + lambda_pred * pred_loss
            
            # Backward and optimize
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), 1.0)
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
        
        mode = "Joint" if (not freeze_encoders and not freeze_decoder) else \
               "Decoder-only" if freeze_encoders else "Encoder-only"
        
        print(f"Epoch {ep:3d} | {mode} | "
              f"Total: {avg_total_loss:.6f} | "
              f"Enc: {avg_enc_loss:.6f} | "
              f"Pred: {avg_pred_loss:.6f}")
    
    return ensemble_model




#------------------------------------------------------------------------------
def train_predictive_model(
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
    prediction_loss = 'mseError' # 'jsDivergence' # 'xEntropy', 'klDivergence', 'mseError'
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
    ds = PredictiveSeqDataset(sequences, target_distributions, seq_probs, global_weights)
    dataloader = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0, 
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
#==============================================================================
# Global importance/probability of a sequence - combining fixed lengrt distribution 
# probability and sequence-length probability
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
# X_joint.shape == [N_joint, n_channels, sequence_length]
# training_data["X_joint"][example sequence#][channel#][position in the sequence]
# training_data["X_joint"][example_sequence_index, channel_index, position]

# training_data["X_by_channel"][channel/feature#][position]
#-----------------------------------------------------------------------------
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
#------------------------------------------------------------------------------
def flatten_multilength_training_data(
    training_data_by_len,
    length_weighting="equal",
    class_values=(-1, 0, 1),
    eps=1e-12,
):
    """
    Flatten training_data[L] across sequence lengths.

    Returns a list-based structure because sequence lengths differ.

    Output
    ------
    flat_data["sequences"]
        list of joint multichannel sequences.
        Each element has shape conceptually [n_channels, sequence_length].

    flat_data["X_by_channel"]
        list over channels.
        flat_data["X_by_channel"][m][j] is the channel-m sequence
        for flattened example j.

    flat_data["Y_dist"]
        empirical class distributions, shape [N_total, n_classes].

    flat_data["Y_class"]
        dominant empirical class.

    flat_data["weights"]
        globally normalized training weights.

    flat_data["counts"]
        empirical counts.

    flat_data["seq_lengths"]
        sequence length of each flattened example.
    """

    lengths = sorted(training_data_by_len.keys())
    n_lengths = len(lengths)

    sequences = []
    Y_list = []
    weights_list = []
    counts_list = []
    seq_lengths = []

    class_values = np.asarray(class_values)

    n_channels = None
    X_by_channel_flat = None

    # ------------------------------------------------------------------
    # Total mass for global weighting
    # ------------------------------------------------------------------
    if length_weighting == "counts":
        total_count_all_lengths = sum(
            np.asarray(
                training_data_by_len[L]["counts"],
                dtype=np.float64,
            ).sum()
            for L in lengths
        )

    elif length_weighting == "equal":
        total_count_all_lengths = None

    else:
        raise ValueError(
            "length_weighting must be either 'equal' or 'counts'."
        )

    # ------------------------------------------------------------------
    # Flatten length by length
    # ------------------------------------------------------------------
    for L in lengths:
        data_L = training_data_by_len[L]

        X_L = np.asarray(data_L["X_joint"])
        Y_L = np.asarray(data_L["Y_dist"], dtype=np.float64)
        counts_L = np.asarray(data_L["counts"], dtype=np.float64)

        if X_L.ndim != 3:
            raise ValueError(
                f"training_data[{L}]['X_joint'] must have shape "
                "[N, n_channels, sequence_length]."
            )

        N_L, n_channels_L, seq_len_L = X_L.shape

        if seq_len_L != L:
            raise ValueError(
                f"Dictionary key says length {L}, "
                f"but X_joint has length {seq_len_L}."
            )

        if Y_L.shape[0] != N_L:
            raise ValueError(
                f"Length {L}: X_joint and Y_dist row mismatch."
            )

        if counts_L.shape[0] != N_L:
            raise ValueError(
                f"Length {L}: X_joint and counts row mismatch."
            )

        if n_channels is None:
            n_channels = n_channels_L
            X_by_channel_flat = [
                []
                for _ in range(n_channels)
            ]
        elif n_channels_L != n_channels:
            raise ValueError(
                "Different numbers of channels across lengths."
            )

        # --------------------------------------------------------------
        # Recalculate global weights
        # --------------------------------------------------------------
        if length_weighting == "equal":
            within_L_weights = counts_L / (
                counts_L.sum() + eps
            )

            global_weights_L = within_L_weights / n_lengths

        else:
            global_weights_L = counts_L / (
                total_count_all_lengths + eps
            )

        # --------------------------------------------------------------
        # Flatten X_joint as list of [channel][position]
        # --------------------------------------------------------------
        sequences_L = [
            X_L[j].tolist()
            for j in range(N_L)
        ]

        sequences.extend(sequences_L)

        # --------------------------------------------------------------
        # Flatten X_by_channel
        # X_by_channel_flat[m][j] is the sequence for channel m
        # in flattened example j.
        # --------------------------------------------------------------
        for m in range(n_channels):
            X_by_channel_flat[m].extend(
                X_L[:, m, :].tolist()
            )

        Y_list.append(Y_L)
        weights_list.append(global_weights_L)
        counts_list.append(counts_L)

        seq_lengths.extend([L] * N_L)

    Y_dist = np.concatenate(Y_list, axis=0)
    weights = np.concatenate(weights_list, axis=0)
    counts = np.concatenate(counts_list, axis=0)
    seq_lengths = np.asarray(seq_lengths, dtype=np.int64)

    # Remove tiny numerical drift.
    weights = weights / (weights.sum() + eps)

    Y_class = class_values[
        np.argmax(Y_dist, axis=1)
    ]

    return {
        "sequences": sequences,
        "X_by_channel": X_by_channel_flat,
        "Y_dist": Y_dist.astype(np.float32),
        "Y_class": Y_class,
        "weights": weights.astype(np.float64),
        "counts": counts.astype(np.float64),
        "seq_lengths": seq_lengths,
        "class_values": class_values,
        "lengths": lengths,
        "n_channels": n_channels,
        "length_weighting": length_weighting,
    }

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
#------------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------
# Estimated sequency probabilities: from original or flattened data
def same_length_joint_probs_from_original(training_data_by_len, L):
    data_L = training_data_by_len[L]

    X_L = np.asarray(data_L["X_joint"])
    counts_L = np.asarray(data_L["counts"], dtype=np.float64)

    probs_L = counts_L / counts_L.sum()

    sequences_L = [
        X_L[j].tolist()
        for j in range(X_L.shape[0])
    ]

    return {
        "sequences": sequences_L,
        "probs": probs_L,
        "counts": counts_L,
        "Y_dist": np.asarray(data_L["Y_dist"]),
        "Y_class": np.asarray(data_L["Y_class"]),
        "length": L,
    }

def same_length_joint_probs_from_flat(flat_data, L, use_counts=True):
    seq_lengths = np.asarray(flat_data["seq_lengths"])
    mask = seq_lengths == L

    indices = np.where(mask)[0]

    if use_counts:
        mass = np.asarray(flat_data["counts"], dtype=np.float64)[mask]
    else:
        mass = np.asarray(flat_data["weights"], dtype=np.float64)[mask]

    probs_L = mass / mass.sum()

    sequences_L = [
        flat_data["sequences"][j]
        for j in indices
    ]

    return {
        "indices": indices,
        "sequences": sequences_L,
        "probs": probs_L,
        "counts": np.asarray(flat_data["counts"])[mask],
        "weights": np.asarray(flat_data["weights"])[mask],
        "Y_dist": np.asarray(flat_data["Y_dist"])[mask],
        "Y_class": np.asarray(flat_data["Y_class"])[mask],
        "length": L,
    }
# ############################################################################
# Channel training data

def aggregate_flat_channel_data(
    flat_data,
    channel,
    restrict_length=None,
    eps=1e-12,
):
    """
    Aggregate flattened ensemble data into unique examples for one channel.

    Adds:
        - seq_probs: empirical probabilities over selected examples
        - seq_probs_same_length: empirical probabilities conditional on length
        - seq_lengths: length of each unique local sequence

    If restrict_length=L, then seq_probs and seq_probs_same_length coincide.
    """

    X_channel = flat_data["X_by_channel"][channel]
    Y = np.asarray(flat_data["Y_dist"], dtype=np.float64)
    weights = np.asarray(flat_data["weights"], dtype=np.float64)
    counts = np.asarray(flat_data["counts"], dtype=np.float64)
    seq_lengths_all = np.asarray(flat_data["seq_lengths"], dtype=np.int64)
    class_values = np.asarray(flat_data["class_values"])

    if restrict_length is None:
        indices = np.arange(len(X_channel))
    else:
        indices = np.where(seq_lengths_all == restrict_length)[0]

    table = OrderedDict()

    for j in indices:
        key = tuple(X_channel[j])
        L_j = int(seq_lengths_all[j])

        if key not in table:
            table[key] = {
                "weight": 0.0,
                "count": 0.0,
                "class_mass": np.zeros(Y.shape[1], dtype=np.float64),
                "length": L_j,
            }

        # Same local sequence should always have same length.
        if table[key]["length"] != L_j:
            raise ValueError(
                "Same sequence key appeared with different lengths. "
                "This should not happen."
            )

        w_j = weights[j]
        c_j = counts[j]

        table[key]["weight"] += w_j
        table[key]["count"] += c_j

        # Use empirical counts for the class distribution.
        # This estimates P(class | local channel sequence).
        table[key]["class_mass"] += c_j * Y[j]

    X_unique = []
    W_unique = []
    C_unique = []
    Y_unique = []
    L_unique = []

    for key, item in table.items():
        X_unique.append(list(key))
        W_unique.append(item["weight"])
        C_unique.append(item["count"])
        L_unique.append(item["length"])

        y = item["class_mass"] / max(item["count"], eps)
        Y_unique.append(y)

    W_unique = np.asarray(W_unique, dtype=np.float64)
    C_unique = np.asarray(C_unique, dtype=np.float64)
    Y_unique = np.asarray(Y_unique, dtype=np.float32)
    L_unique = np.asarray(L_unique, dtype=np.int64)

    # ------------------------------------------------------------
    # Training weights
    # ------------------------------------------------------------
    W_unique = W_unique / (W_unique.sum() + eps)

    # ------------------------------------------------------------
    # Empirical sequence probabilities over the selected set
    # ------------------------------------------------------------
    seq_probs = C_unique / (C_unique.sum() + eps)

    # ------------------------------------------------------------
    # Empirical sequence probabilities conditional on sequence length
    # ------------------------------------------------------------
    seq_probs_same_length = np.zeros_like(C_unique, dtype=np.float64)

    for L in np.unique(L_unique):
        mask_L = L_unique == L
        seq_probs_same_length[mask_L] = (
            C_unique[mask_L]
            / (C_unique[mask_L].sum() + eps)
        )

    Y_class = class_values[np.argmax(Y_unique, axis=1)]

    return {
        "X": X_unique,
        "Y_dist": Y_unique,
        "Y_class": Y_class,

        # Loss weights
        "weights": W_unique,

        # Raw empirical support
        "counts": C_unique,

        # Empirical probabilities
        "seq_probs": seq_probs,
        "seq_probs_same_length": seq_probs_same_length,

        # Metadata
        "seq_lengths": L_unique,
        "class_values": class_values,
        "channel": channel,
        "restrict_length": restrict_length,
    }
#-----------------------------------------------------------------------------
def get_channel_fixed_length_distribution(
    channel_data_k,
    L,
    eps=1e-12,
):
    """
    Extract lexicographically ordered fixed-length local sequence data
    for one channel.

    Parameters
    ----------
    channel_data_k:
        channel_data[k], output of aggregate_flat_channel_data(...)

    L:
        Desired sequence length.

    Returns
    -------
    result:
        {
            "X": list of lexicographically ordered unique sequences,
            "Y_dist": local class distributions,
            "Y_class": dominant local class,
            "seq_probs": empirical same-length sequence probabilities,
            "counts": empirical counts,
            "weights": training weights,
            "class_values": class labels,
            "length": L,
        }
    """

    X = channel_data_k["X"]
    Y_dist = np.asarray(channel_data_k["Y_dist"], dtype=np.float64)
    counts = np.asarray(channel_data_k["counts"], dtype=np.float64)
    weights = np.asarray(channel_data_k["weights"], dtype=np.float64)
    class_values = np.asarray(channel_data_k["class_values"])

    if "seq_lengths" in channel_data_k:
        seq_lengths = np.asarray(
            channel_data_k["seq_lengths"],
            dtype=np.int64,
        )
        mask = seq_lengths == L
    else:
        # Fallback: infer length directly from the sequence.
        mask = np.asarray(
            [len(seq) == L for seq in X],
            dtype=bool,
        )

    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError(
            f"No sequences of length {L} found."
        )

    # Lexicographic ordering.
    sorted_indices = sorted(
        indices,
        key=lambda i: tuple(X[i]),
    )

    X_sorted = [
        list(X[i])
        for i in sorted_indices
    ]

    Y_sorted = Y_dist[sorted_indices]
    counts_sorted = counts[sorted_indices]
    weights_sorted = weights[sorted_indices]

    # Same-length empirical sequence probabilities.
    seq_probs_sorted = counts_sorted / (
        counts_sorted.sum() + eps
    )

    Y_class_sorted = class_values[
        np.argmax(Y_sorted, axis=1)
    ]

    return {
        "X": X_sorted,
        "Y_dist": Y_sorted.astype(np.float32),
        "Y_class": Y_class_sorted,
        "seq_probs": seq_probs_sorted.astype(np.float64),
        "counts": counts_sorted.astype(np.float64),
        "weights": weights_sorted.astype(np.float64),
        "class_values": class_values,
        "length": L,
        "indices": np.asarray(sorted_indices, dtype=np.int64),
    }

# ============================================================================
# USAGE
# ============================================================================

mode = 'train'
mode = 'validate'
mode = 'train'

prediction_loss = 'mseError'

if __name__ == "__main__" and mode == 'train':

    # -------------------------------------------------------------------------
    # Learning task parameters
    # -------------------------------------------------------------------------
    features     = ['log_mid',"tvi_n" , 'obi_L1', "ofi_L1_n", "ofi_L1_n_norm",'ofi_L1_norm_n','ofi_L3_norm_n','ofi_L10_norm_n',"micro_price",'vpin', 'sigma_W' ]
    features     = ['log_mid','ofi_L10_norm_n',"micro_price",'vpin' ]
    dates = ['202504','20250501']  # training and validation
    date = dates[0]   # the data is aggregated for 1 month
    symbol= 'AAPL'
    date = dates[0]   # the data is aggregated for 1 month





    n_symbols = 4   #observable symbols per feature

    # resampling 
    frequency = 100 #events
    freq_units = 'evn'
    
    max_seq_len =  5          # max sequence length to be used for training 
    
    min_seq_prob = 0.000000   # threshold for rare events


    ################################################################################
    ################################################################################        
    #------------------------------------------------------------------------------
    # Ensemble of Models
    #------------------------------------------------------------------------------
    # Step 1: Prepare data for multi-encoder training
    # Each encoder sees same time interval of sequence on different features
    
    #-----------------------------------------------------------------------------
    #  Load of training data
    #------------------------------------------------------------------------------
    # [joint_data, component_data]
    
#    joint_data
#        Unique joint sequences and their empirical class distributions.

#        joint_data["X"] has shape:
#            [n_unique_joint_sequences, n_channels, sequence_length]

#    component_data
#        List of length n_channels.

#        component_data[i]["X"] has shape:
#            [n_unique_component_sequences_i, sequence_length]
    
    predicted = features[0]
    
    predictors = [ 'ofi_L10_norm_n',"micro_price",'vpin',
                  ['ofi_L10_norm_n',"micro_price",'vpin'],
                 ]
    
    predictors_names = [ 'ofi_L10_norm_n',"micro_price",'vpin','L10_micro_vpin']
    
    
    variates  =  ['bivariate','bivariate','bivariate','multivariate'] 
    qubits    =  [3,           3,          3         , 6            ]
    n_channels = len(predictors)
    
    seq_lens = [1,2,3,4,5]
    seq_lens = [1,2,3,4]
    
    clsNames = ['c1','c2','ca2','ca4']
    clsName    = 'c2'
    
    fPath = '..\\Data Preparation\\' 
    mPath = '..\\Models\\' 
    training_data = {}
    seq_probs_len = {}
    seq_probs_len_2 = {}


    #--------------------------------------------------------------------------
    # Ensemble data is generted and stored by sequence length 
    #--------------------------------------------------------------------------
    for seq_len in seq_lens:
        fName = "ENS_TD_"+symbol+"_"+date+"_SL_"+str(seq_len)+"_CL_"+clsName+"_"+predicted+"_ALL"
       
    
        # Shape: [N_joint, n_channels, sequence_length].
        # joint_data["X"] = np.asarray(
        # joint_data["keys"],
        # dtype=np.int16,
        # ): every element is a multidimensional - # features - sequence with length described by SL_ 
        # so every element contain #features sequences with same length-SL_ 
        # the j-th unique multivariate sequence is joint_data["X"][j]
        
        # the predicted variable is always joint_data["X"][j][0] and joint_data["X"][j][i], i-1..11
        # are the observations by sequence
        
        # joint_data["class_values"] = tuple(class_values):  (-1, 0, 1)
        # joint_data["sequence_length"] = sequence_length = SL_
        # joint_data["n_channels"] = n_channels     -> one predicted and 10 predictors
        # joint_data["daily_sample_counts"] = np.asarray(
        #     daily_sample_counts,
        #     dtype=np.int64,
        # )  - ifthe sample is aggregation ove n days - counts of observations per day
        
        # component_data[0] - information about predicted
        # component_data[1] - information about predictor i
    
        data  = pickle.load( open( fPath+fName, "rb") )

        joint_data, component_data = data[0], data[1]
        
        #    "X_joint": X_joint,
        #    "X_by_channel": X_by_channel,
        #    "Y_dist": Y_dist,
        #    "Y_class": Y_class,
        #    "counts": counts,
        #    "weights": weights,3333333
        print('Loaded')
        
        training_data[seq_len] = extract_ensemble_arrays(joint_data, component_data)
   
        seq_probs_len[seq_len] = same_length_joint_probs_from_original(
        training_data_by_len=training_data, L=seq_len )
#------=Extracted data by length   
    
    flat_training_data = flatten_multilength_training_data(
                            training_data_by_len=training_data,
                            length_weighting="equal",
                            class_values=(-1, 0, 1),
                            )
    # flat training data format
    
    #  {
    #      "sequences": sequences,
    #      "X_by_channel": X_by_channel_flat,
    #      "Y_dist": Y_dist.astype(np.float32),
    #      "Y_class": Y_class,
    #      "weights": weights.astype(np.float64),
    #      "counts": counts.astype(np.float64),
    #      "seq_lengths": seq_lengths,
    #      "class_values": class_values,
    #      "lengths": lengths,
    #      "n_channels": n_channels,
    #      "length_weighting": length_weighting,
    #  }

    print('Aggregated- all lengths')

#------------------------------------------------------------------------------
#   THIS IS NOT USED YET
#------------------------------------------------------------------------------
    for seq_len in seq_lens:
        seq_probs_len_2[seq_len ] = same_length_joint_probs_from_flat(
        flat_training_data,
        L=seq_len,
        use_counts=True,
    )
#---------------------------------------------------------------------------
# Output: for each seq_len in each item - 4 channels
#        "indices": indices,
#        "sequences": sequences_L,
#        "probs": probs_L,
#        "counts": np.asarray(flat_data["counts"])[mask],
#        "weights": np.asarray(flat_data["weights"])[mask],
#        "Y_dist": np.asarray(flat_data["Y_dist"])[mask],
#        "Y_class": np.asarray(flat_data["Y_class"])[mask],
#        "length": L,
##############################################################################
#  Format trainig data
#  Load pre-trained encoders
#----------------------------------------------------------------------------      
    channel_data = {}
    channel_seq_distributions = {}
    for channel in range(n_channels):
        channel_data[channel] = aggregate_flat_channel_data(
                                    flat_training_data, channel, )
        channel_seq_distributions[channel] = {} 
        for seq_len in seq_lens:
            channel_seq_distributions[channel][seq_len] = get_channel_fixed_length_distribution(
                channel_data[channel],
                L=seq_len,
            )
    # 
    # per Channel data format
    #  "X": X_unique,
    #  "Y_dist": Y_unique,
    #  "Y_class": Y_class,

    # Loss weights
    #  "weights": W_unique,

    # Raw empirical support
    #  "counts": C_unique,

    # Empirical probabilities
    #  "seq_probs": seq_probs,
    #  "seq_probs_same_length": seq_probs_same_length,

    # Metadata
    #  "seq_lengths": L_unique,
    #   "class_values": class_values,
    #  "channel": channel,
    #  "restrict_length": restrict_length,


    # Verification of Encoders by channel
    
    all_encoders = {}
    all_examples = {}
    all_probabil = {}
    all_cl_distr = {}
    all_classes  = {}
    all_weights  = {}
    for channel in range(n_channels):
        m = 1+max(channel_seq_distributions[channel][max(seq_lens)]["X"][-1])  # alphabet size (e.g., 4x4 for price+OFI encoding) - DEPENDS ON THE PREDICTOR
        variate   = variates[channel]
        predictor      = predictors[channel]
        predictor_name = predictors_names[channel]
        n_qubits  = qubits[channel]
        
        # Channel Configuration
        d = 2 ** n_qubits

        
        
        model_file_name = "WGHTS_"+symbol+'_'+variate+"_"+predicted+"-"+predictor_name+"_"+date+'_'+str(n_qubits)+'q'+'.pt'
                           
        
        encoder, meta=load_model_weights(mPath+model_file_name, m, n_qubits, learn_rho0=True, device="cpu")
        
        all_encoders[channel] = encoder # one encoder per channel
        all_examples[channel] =  [] 
        all_probabil[channel] =  []
        all_cl_distr[channel] =  []
        all_classes [channel] =  []
        all_weights [channel] =  []
        
        total_loss = 0
        # data aggregation by sequence length
        for seq_len in seq_lens:
            
            print('---------','Cahnnel', channel, 'Predicted',predicted, 'Predictor', predictor_name,'Qubits ', n_qubits )
            
            emp_probs = list(channel_seq_distributions[channel][seq_len]["seq_probs"])
            seqs      = channel_seq_distributions[channel][seq_len]["X"]      
            cls_distr = list(channel_seq_distributions[channel][seq_len]['Y_dist'])
            classes   = list(channel_seq_distributions[channel][seq_len]['Y_class'])
            weights   = list(channel_seq_distributions[channel][seq_len]['weights'])
            
            all_examples[channel] =  all_examples[channel] + seqs
            all_probabil[channel] =  all_probabil[channel] + emp_probs
            all_cl_distr[channel] =  all_cl_distr[channel] + cls_distr
            all_classes [channel] =  all_classes [channel] + classes
            all_weights [channel] =  all_weights [channel] + weights

            encoder_test = False
            if encoder_test: 
                mod_seq_probs = predict_encoder_probs(encoder, seqs)
    
                
                for i in range(len(mod_seq_probs)):
                    total_loss = total_loss+ emp_probs[i]*(emp_probs[i]-mod_seq_probs[i])**2
    
                plot = False
                if plot:
                    title = 'Chnl '+str(channel)+' SL '+str(seq_len)+' QB '+str(n_qubits)+' '+predictor
                    plotDistributions(emp_probs[:200], mod_seq_probs[:200], seqs[:200],
                                  title+' C='+f"{total_loss:.4e}", 'Target', 'Model', c1 = 'blue', c2='red' )
        if encoder_test: 
            print('Encoder loss for channel',channel,total_loss) 

        
        
                    



    print('Channels training data constructed')
    #sys.exit()
    
    # Training channel encoders/decoders
    train_channel_encoder_decoder = False
    if train_channel_encoder_decoder:
        channel_prediction_models = {}
        for channel in range(n_channels):
            #   all_examples[channel]  - sequences
            #   all_probabil[channel]  - sequence probabilities
            #   all_cl_distr[channel]  - class distributions
            #   all_classes [channel]  - dominating class
            #   all_weights [channel]  - global weights
    
            sequences            = all_examples[channel]             # list of sequences  
            target_distributions = all_cl_distr[channel]             # sequence induced class distributions
            seq_probs            = all_probabil[channel]             # sequence probabilities wrt same length distributions
            global_weights       = all_weights [channel]             # ← GLOBAL Empirical weights
    
    
            # -----------------------------
            #  Train full predictive model
            # -----------------------------
            #prediction_loss in 'xEntropy', 'klDivergence', 'jsDivergence', 'mseError'"
            #prediction_loss = 'mseError'
            predicted = features[0]
            predictor      = predictors[channel]
            predictor_name = predictors_names[channel]
            n_qubits  = qubits[channel]
            
            # Channel Configuration
            d = 2 ** n_qubits
    
            
            
            print("\n" + "=" * 60)
            print("Training predictive model (encoder + decoder)",  predictor_name, predicted,"Loss=", prediction_loss, 'Class-=', clsName)
            print("=" * 60)
            
            d_out = 3  # prediction target dimension (e.g., 4 mid-price symbols)
        
            pred_model =  train_predictive_model(
                sequences,             # list of sequences  
                target_distributions,  # sequence induced class distributions
                seq_probs,             # sequence probabilities wrt same length distributions
                global_weights,        # ← GLOBAL Empirical weights
                encoder = encoder,
                d_out=d_out,
                batch_size=8*512,
                lr = 1e-3,
                epochs  = 200,
                freeze_encoder = True,
                use_unitary = True,
                device  = "cuda" if torch.cuda.is_available() else "cpu",
                optimizer_name = "adam",
                weight_decay = 1e-4,
                encoder_lr_multiplier = 0.1,
                lambda_enc = 0  ,   # NEW: weight for encoder loss
                lambda_pred  = 1,  # NEW: weight for prediction loss
                prediction_loss = prediction_loss
            )
    
        # Save
            '''
                meta = {
                    'training_date': '2026-04-07',
                    'epochs_trained': 200,
                    'final_loss': 0.856,
                    'data_period': '2024-01-01 to 2024-02-01',
                }
            '''
            channel_predictor = "PRD_" +clsName+"_"+symbol+"_"+predicted+"-"+predictor_name+"_"+date+'_'+str(n_qubits)+'q'
            save_file = fPath+channel_predictor
        
            save_predictive_model(save_file, pred_model)
            channel_prediction_models[channel] = pred_model
            
    read_channel_encoder_decoder = False
    if read_channel_encoder_decoder:
        channel_prediction_models = {}
        for channel in range(n_channels):

            #prediction_loss = 'mseError'
            predicted = features[0]
            predictor      = predictors[channel]
            predictor_name = predictors_names[channel]
            n_qubits  = qubits[channel]
            
            # Channel Configuration
            d = 2 ** n_qubits
            print("\n" + "=" * 60)
            print("Loading predictive model (encoder + decoder)",  predictor_name, predicted,"Loss=", prediction_loss, 'Class-=', clsName)
            print("=" * 60)
            
            d_out = 3  # prediction target dimension (e.g., 4 mid-price symbols)
             

            channel_predictor = "PRD_" +clsName+"_"+symbol+"_"+predicted+"-"+predictor_name+"_"+date+'_'+str(n_qubits)+'q'
            save_file = fPath+channel_predictor
            pred_model, meta = load_predictive_model(save_file)
            channel_prediction_models[channel] = pred_model
    #-----------------------------------------------------------------------------
    # Channel Models Output - generative probabilities and class distributions for 

    perf_channel_encoder_decoder = False    # estimate the perfoprmance of a channel encoder/decoder
    if perf_channel_encoder_decoder:
        for channel in range(n_channels):
            pred_model = channel_prediction_models[channel]
            # for set of sequences
            sequences            = all_examples[channel]
            target_distributions = all_cl_distr[channel]             # sequence induced class distributions
            seq_probs            = all_probabil[channel]             # sequence probabilities wrt same length distributions
            global_weights       = all_weights [channel]             # ← GLOBAL Empirical weights
          
            
            
            mod_seq_probs, mod_target_distributions = get_model_predictions_ordered(pred_model, sequences)
        
        
        
        # quality of the predictive model / the decoder
        
        #divergences_by_length = compute_divergences_by_length(sequences,  target_distributions,  mod_target_distributions, divergence_type = "kl")
            plot_divergence_by_length(sequences, target_distributions, mod_target_distributions,  mod_seq_probs=None, emp_seq_probs=None,
                divergence_type = prediction_loss, figsize=(14, 6),
                title="Performance: Channel "+str(channel)+'_'+prediction_loss+" by Seq Length"
            )
    
            results = compute_prediction_agreement(
                sequences,
                target_distributions,
                mod_target_distributions,
                class_names=['Down', 'Neutral', 'Up']
            )
            
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
            sample_weight=channel_data[channel]['counts'],
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
    
    
    
            # Print summary
            print("=" * 70)
            print("PREDICTION AGREEMENT ANALYSIS")
            print(symbol, date, predicted, predictor, str(n_qubits)+'q','Class type', clsName, prediction_loss)
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
    


###############################################################################
# Quantum Ensemble of Quantum Encoders
###############################################################################
    train_ensemble = True
    if  train_ensemble:
        # currently encoders are in a dictionary - use the first three elements
        encoders_list =[all_encoders[channel] for channel in  range(n_channels-1) ]
        # Step 1: Create ensemble
        ensemble = MultiEncoderQuantumEnsemble(
            # encoders=[encoder_ofi, encoder_ovi, encoder_vol],
            encoders = encoders_list,
            d_out=3,
            use_unitary=True,
            normalization_point="input"
        )
        
        print(f"Ensemble created with {ensemble.n_encoders} encoders")
        ensemble.get_model_info()
        # Decoder training data
        
        # sequences_list = [x[:3] for x in flat_training_data["sequences"]] # use the first 3 encoders

        sequences_list = flat_training_data["X_by_channel"][0:3]



        target_dists   = flat_training_data["Y_dist"]
        global_weights = flat_training_data["weights"]

        # Train ensemble (decoder only, encoders frozen)
        ensemble_trained = train_multi_encoder_ensemble(
            ensemble_model=ensemble,
            sequences_list=sequences_list,
            target_distributions=target_dists,
            global_weights=global_weights,
            d_out=3,
            batch_size=256+128,
            lr=1e-3,   # 5e-3, too high causing convergence problems?
            epochs=5, #200,
            freeze_encoders=True,      # Keep pretrained encoders fixed
            freeze_decoder=False,      # Train decoder
            device  = "cuda" if torch.cuda.is_available() else "cpu",
            prediction_loss="kl",
            lambda_enc=0.0,            # Don't optimize encoder loss
            lambda_pred=1.0,
        )



sys.exit()


# -----------------------------
# Step 1: Pretrain or load trained encoders on sequence distributions
# -----------------------------
print("=" * 60)
print(" Step 1: Pretrain or load trained encoders on sequence distributions")
print("=" * 60)
   
# Configuration
ne = 3   # encoders number
m  = 16  # alphabet size (e.g., 4x4 for price+OFI encoding)
n_qubits = 5  # system dimension d = 2^n_qubits 
d = 2 ** n_qubits

    
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

# -------------------------------------------------------------------------
# particular learning task parameters
dates = ['202504','20250501']  # training and validation
symbol= 'AAPL'
date = dates[0]   # the data is aggregated for 1 month
features_list =  ['log_mid',"tvi_n" , 'obi_L1', "ofi_L1_n_norm",'ofi_L3_norm_n','ofi_L10_norm_n',"micro_price","ofi_L1_n", 'ofi_L1_norm_n']
variate="bivariate"

predicted =  features_list[0]  # 'log_mid_sym'
predictor =  features_list[1]  # "tvi_n"
predictor =  features_list[2]  # 'obi_L1'
predictor =  features_list[3]  # 'ofi_L1_n_norm'
n_symbols = 4   #observable symbols per feature

# resampling 
frequency = 100 #events
freq_units = 'evn'

max_seq_len =  6          # max sequence length to be used for training 
min_seq_prob = 0.000000   # threshold for rare events


# Load empirical sequence data
# sequences_all, emp_probs_all = load_your_empirical_data()
# Filter by length and probability

#--------------------------------------------------------------------------
# Training Data Load
#--------------------------------------------------------------------------
fPath = '..\\Data Preparation\\' 
# SEQ_DISTR_AAPL_bivariate_log_mid-micro_price_202504

# sequences distributions
title = "SEQ_DISTR_"+symbol+"_"+variate+"_"+predicted+"-"+predictor+"_"+date

infname = fPath+title
distrs_samples =  pickle.load(open( infname, "rb") ) 
sequences_all = distrs_samples[1]
emp_probs_all = [i[1] for i in distrs_samples[0]] # empirical proabilities
  
sequences = []
emp_probs = []
for i in range(len(sequences_all)):
    if len(sequences_all[i]) <= max_seq_len and emp_probs_all[i] > min_seq_prob:
        sequences.append(sequences_all[i])
        emp_probs.append(emp_probs_all[i])

print(f"Number of training sequences: {len(sequences)}")


# Train encoder (generative channel)
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Location and name of trained encoder
#------------------------------------------------------------------
save_file_name = 'MOD'+title[9:] +'_'+str(n_qubits)+'q'
model_file_name = "WGHTS_"+save_file_name[4:]+'.pt'

load_from_file = True

if load_from_file:
    print("Loading pre-trained encoder:",symbol+"_"+variate+"_"+predicted+"-"+predictor+"_"+date+'_'+str(n_qubits)+'q' )
    # [model, sequences, emp_probs ]
    result = pickle.load( open( save_file_name, "rb") )
    #model, sequences, emp_probs = result[0], result[1], result[2]
    encoder, meta=load_model_weights(model_file_name, m, n_qubits, learn_rho0=True, device="cpu")
    # model test
    mod_seq_probs = predict_encoder_probs(encoder, sequences)
    
    total_loss = 0
    for i in range(len(mod_seq_probs)):
        total_loss = total_loss+emp_probs[i]*(emp_probs[i]-mod_seq_probs[i])**2


    #Kseq, rho0, p = model.path_operator(sequences_all[-1], return_prob=True, device="cpu")

    plotDistributions(emp_probs[:200],mod_seq_probs[:200], sequences[:200],
                      title+' C='+f"{total_loss:.4e}", 'Target', 'Model', c1 = 'blue', c2='red' )

    
    
#--------------------------------------------------------------------------
# Local encoder pre-training - Not Implemented - Just a sketch
#--------------------------------------------------------------------------
elif False:                                     # train the model   
    print("Encoder learning")
    print('Number of examples ', len(sequences))
       
    ds = SeqDataset(sequences, emp_probs)

    encoder = train_encoder(ds.sequences, ds.emp_probs,
        m, n_qubits,
        batch_size= 4*512,
        lr=1e-3,
        epochs=2000,
        learn_rho0=True,
        model=modelM,
        num_workers=6,
        device="cuda" if torch.cuda.is_available() else "cpu",
        optimizer_name="adam")
 
    
 
    # Save pretrained encoder
    save_model_weights(
        model_file_name, 
        encoder, 
        meta={"m": m, "n_qubits": n_qubits, "max_len": max_seq_len}
    )




##############################################################################
##############################################################################


# Step 4: Evaluate
print("\n" + "=" * 70)
print("ENSEMBLE EVALUATION")
print("=" * 70)

mod_probs, mod_traces_list = ensemble_trained(seq_pad)

print(f"Model predictions: {mod_probs[0]}")
print(f"Individual encoder contributions:")
for i, traces in enumerate(mod_traces_list):
    print(f"  Encoder {i} trace: {float(traces[0]):.6f}")

# Step 5: Analyze encoder contributions
contributions = ensemble_trained.get_encoder_contributions(seq_pad)
for enc_name, contrib_data in contributions.items():
    print(f"{enc_name}:")
    print(f"  Mean trace: {contrib_data['mean_trace']:.6f}")
    print(f"  Std trace: {contrib_data['std_trace']:.6f}")

# Step 6: Save trained ensemble
def save_ensemble_model(path, ensemble_model, meta=None):
    """Save multi-encoder ensemble"""
    payload = {
        'encoder_states': [enc.state_dict() for enc in ensemble_model.encoders],
        'decoder_state': ensemble_model.decoder.state_dict(),
        'encoder_configs': [
            {
                'm': enc.m,
                'd': enc.d,
                'learn_rho0': enc.learn_rho0,
                'rho0_type': enc.rho0_type,
                'eps': enc.eps,
            }
            for enc in ensemble_model.encoders
        ],
        'decoder_config': {
            'd_in': ensemble_model.decoder.d_in,
            'd_out': ensemble_model.decoder.d_out,
            'use_unitary': ensemble_model.decoder.use_unitary,
            'normalization_point': ensemble_model.decoder.normalization_point,
            'eps': ensemble_model.decoder.eps,
        },
        'ensemble_config': {
            'n_encoders': ensemble_model.n_encoders,
            'd': ensemble_model.d,
            'd_product': ensemble_model.d_product,
            'normalization_point': ensemble_model.normalization_point,
        },
        'meta': meta or {},
    }
    torch.save(payload, path)
    print(f"✓ Ensemble saved to {path}")


def load_ensemble_model(path, device="cpu"):
    """Load multi-encoder ensemble"""
    payload = torch.load(path, map_location=device)
    
    # Reconstruct encoders
    encoders = []
    for enc_cfg in payload['encoder_configs']:
        enc = KrausInstrument(
            m=enc_cfg['m'],
            d=enc_cfg['d'],
            learn_rho0=enc_cfg['learn_rho0'],
            rho0_type=enc_cfg['rho0_type'],
            eps=enc_cfg.get('eps', 1e-8)
        )
        encoders.append(enc)
    
    # Load encoder states
    for enc, enc_state in zip(encoders, payload['encoder_states']):
        enc.load_state_dict(enc_state)
    
    # Reconstruct ensemble
    dec_cfg = payload['decoder_config']
    ensemble = MultiEncoderQuantumEnsemble(
        encoders=encoders,
        d_out=dec_cfg['d_out'],
        use_unitary=dec_cfg['use_unitary'],
        normalization_point=payload['ensemble_config']['normalization_point'],
        eps=dec_cfg.get('eps', 1e-8)
    )
    
    # Load decoder state
    ensemble.decoder.load_state_dict(payload['decoder_state'])
    ensemble = ensemble.to(device)
    ensemble.eval()
    
    print(f"✓ Ensemble loaded from {path}")
    return ensemble, payload.get('meta', {})


# Usage
meta = {
    'training_date': '2026-04-07',
    'epochs': 200,
    'final_loss': 0.865,
    'n_encoders': 3,
    'encoder_names': ['price_ofi', 'price_ovi', 'price_volume'],
}

save_ensemble_model('ensemble_v1.pt', ensemble_trained, meta=meta)

# Load later
ensemble_loaded, meta_loaded = load_ensemble_model('ensemble_v1.pt', device=device)
print(f"Loaded ensemble with {meta_loaded['n_encoders']} encoders")

# Step 7: Inference on test/validation data
@torch.no_grad()
def predict_ensemble(ensemble_model, sequences_list, batch_size=512, device="cpu"):
    """
    Get predictions from ensemble on test data
    
    Args:
        ensemble_model: trained MultiEncoderQuantumEnsemble
        sequences_list: list of n sequence lists (one per encoder)
        batch_size: batch size
        device: computation device
    
    Returns:
        probs: [N, d_out] prediction probabilities
        traces_list: list of n trace arrays [N]
    """
    ensemble_model.eval()
    ensemble_model = ensemble_model.to(device)
    
    n_encoders = len(sequences_list)
    n_samples = len(sequences_list[0])
    d_out = ensemble_model.d_out
    
    all_probs = []
    all_traces_list = [[] for _ in range(n_encoders)]
    
    # Process in batches
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_size_actual = end_idx - start_idx
        
        # Get batch sequences
        batch_seqs = [seqs[start_idx:end_idx] for seqs in sequences_list]
        
        # Pad sequences
        seq_pads = []
        for seqs in batch_seqs:
            lens = [len(s) for s in seqs]
            T = max(lens)
            
            seq_pad = torch.full((batch_size_actual, T), -1, dtype=torch.long)
            for i, s in enumerate(seqs):
                seq_pad[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            
            seq_pads.append(seq_pad.to(device))
        
        # Use first seq_pad for ensemble (all encoders see same sequence)
        seq_pad = seq_pads[0]
        
        # Forward pass
        probs, traces_list = ensemble_model(seq_pad)
        
        # Store results
        all_probs.append(probs.cpu().numpy())
        for i, traces in enumerate(traces_list):
            all_traces_list[i].append(traces.cpu().numpy())
    
    # Concatenate results
    all_probs = np.concatenate(all_probs, axis=0)
    all_traces_list = [np.concatenate(traces, axis=0) for traces in all_traces_list]
    
    return all_probs, all_traces_list


# Usage
test_seqs_ofi = test_sequences_ofi
test_seqs_ovi = test_sequences_ovi
test_seqs_vol = test_sequences_volume

test_sequences_list = [test_seqs_ofi, test_seqs_ovi, test_seqs_vol]

test_probs, test_traces_list = predict_ensemble(
    ensemble_trained,
    test_sequences_list,
    batch_size=8192,
    device=device
)

print(f"Predictions on {len(test_seqs_ofi)} test sequences")
print(f"  Shape: {test_probs.shape}")
print(f"  First prediction: {test_probs[0]}")

# Step 8: Evaluate ensemble performance
agreement_results = compute_prediction_agreement(
    test_seqs_ofi,  # Use any sequence list
    test_target_dists,
    test_probs,
    class_names=['Down', 'Neutral', 'Up']
)

print("\n" + "=" * 70)
print("ENSEMBLE TEST PERFORMANCE")
print("=" * 70)
print(f"Overall Agreement: {agreement_results['overall_agreement_pct']:.2f}%")
print(f"  ({agreement_results['total_agreements']}/{agreement_results['total_samples']} sequences)\n")

print("Agreement by Sequence Length:")
for length in sorted(agreement_results['agreement_pct_by_length'].keys()):
    pct = agreement_results['agreement_pct_by_length'][length]
    count = agreement_results['agreements_by_length'][length]
    total = agreement_results['totals_by_length'][length]
    print(f"  Length {length}: {pct:6.2f}% ({count}/{total})")

print("\nAgreement by Empirical Predicted Class:")
for class_name in ['Down', 'Neutral', 'Up']:
    pct = agreement_results['agreement_pct_by_class'][class_name]
    count = agreement_results['agreements_by_class'][class_name]
    total = agreement_results['totals_by_class'][class_name]
    print(f"  {class_name:8s}: {pct:6.2f}% ({count}/{total})")

# Step 9: Divergence analysis by entropy bins
print("\n" + "=" * 70)
print("DIVERGENCE ANALYSIS BY ENTROPY")
print("=" * 70)

entropy_results = compute_entropy_binned_divergences(
    test_seqs_ofi,
    test_target_dists,
    test_probs,
    divergence_type="kl",
    n_entropy_bins=5
)

print("\nKL Divergence by Entropy Bin:")
for bin_idx in sorted(entropy_results['stats_by_bin'].keys()):
    stats = entropy_results['stats_by_bin'][bin_idx]
    print(f"  Bin {bin_idx} ({stats['entropy_range']}):")
    print(f"    Mean KL: {stats['mean_div']:.6f}")
    print(f"    Std KL:  {stats['std_div']:.6f}")
    print(f"    Count:   {stats['count']}")

# Step 10: Compare individual encoders vs ensemble
print("\n" + "=" * 70)
print("ENCODER CONTRIBUTION ANALYSIS")
print("=" * 70)

print(f"\nIndividual encoder sequence probabilities (traces):")
encoder_names = ['OFI', 'OVI', 'Volume']

for seq_idx in range(min(5, len(test_seqs_ofi))):
    print(f"\nSequence {seq_idx}: {test_seqs_ofi[seq_idx]}")
    for enc_idx, (name, traces) in enumerate(zip(encoder_names, test_traces_list)):
        print(f"  {name:8s} encoder trace: {traces[seq_idx]:.6f}")

# Step 11: Save evaluation results
def save_evaluation_results(path, results_dict, meta=None):
    """Save evaluation results to file"""
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results_dict)
    
    payload = {
        'results': serializable_results,
        'meta': meta or {},
    }
    
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    
    print(f"✓ Results saved to {path}")


eval_results = {
    'agreement': agreement_results,
    'entropy_divergence': entropy_results,
    'test_size': len(test_seqs_ofi),
}

save_evaluation_results(
    'ensemble_eval_results.json',
    eval_results,
    meta={
        'date': '2026-04-07',
        'model': 'multi_encoder_ensemble',
        'n_encoders': 3,
        'encoder_types': ['OFI', 'OVI', 'Volume'],
        'test_set_size': len(test_seqs_ofi),
        'normalization_point': 'input',
        'prediction_loss': 'kl',
    }
)

# Step 12: Comparative analysis - single vs ensemble
print("\n" + "=" * 70)
print("SINGLE vs ENSEMBLE COMPARISON")
print("=" * 70)

# Get predictions from individual decoders
@torch.no_grad()
def predict_single_encoder(encoder, decoder, sequences, device="cpu"):
    """Get predictions using single encoder + decoder"""
    encoder.eval()
    decoder.eval()
    
    all_probs = []
    
    for seq in sequences:
        seq_t = torch.tensor([seq], dtype=torch.long).to(device)
        
        # Encode
        rho_unnorm = encoder.encode_sequences_unnormalized(seq_t)
        traces = torch.real(torch.diagonal(rho_unnorm, dim1=-2, dim2=-1).sum(-1))
        
        # Normalize
        rho_norm = rho_unnorm / traces.unsqueeze(-1).unsqueeze(-1)
        
        # Decode
        probs = decoder.predict_probs(rho_norm)
        all_probs.append(probs[0].cpu().numpy())
    
    return np.array(all_probs)


# Get single encoder predictions
single_ofi_probs = predict_single_encoder(
    ensemble_trained.encoders[0],
    QuantumDecoder(d_in=8, d_out=3),  # Would need to load proper decoder
    test_seqs_ofi,
    device=device
)

single_ovi_probs = predict_single_encoder(
    ensemble_trained.encoders[1],
    QuantumDecoder(d_in=8, d_out=3),
    test_seqs_ovi,
    device=device
)

# Compare agreements
single_ofi_agreement = compute_prediction_agreement(
    test_seqs_ofi,
    test_target_dists,
    single_ofi_probs,
    class_names=['Down', 'Neutral', 'Up']
)

single_ovi_agreement = compute_prediction_agreement(
    test_seqs_ovi,
    test_target_dists,
    single_ovi_probs,
    class_names=['Down', 'Neutral', 'Up']
)

ensemble_agreement = agreement_results

print("Agreement Comparison:")
print(f"  Single OFI:  {single_ofi_agreement['overall_agreement_pct']:.2f}%")
print(f"  Single OVI:  {single_ovi_agreement['overall_agreement_pct']:.2f}%")
print(f"  Ensemble:    {ensemble_agreement['overall_agreement_pct']:.2f}%")

improvement = (
    ensemble_agreement['overall_agreement_pct'] - 
    max(single_ofi_agreement['overall_agreement_pct'], 
        single_ovi_agreement['overall_agreement_pct'])
)
print(f"\nEnsemble improvement: {improvement:+.2f}%")

# Step 13: Summary report
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

summary = {
    'ensemble_config': {
        'n_encoders': ensemble_trained.n_encoders,
        'system_dimension': ensemble_trained.d,
        'product_dimension': ensemble_trained.d_product,
        'output_dimension': ensemble_trained.d_out,
    },
    'training_results': {
        'epochs_trained': 200,
        'batch_size': 8192,
        'learning_rate': 5e-3,
        'encoder_lr_multiplier': 0.1,
        'prediction_loss': 'kl',
        'freeze_encoders': True,
    },
    'test_performance': {
        'test_set_size': len(test_seqs_ofi),
        'ensemble_agreement_pct': ensemble_agreement['overall_agreement_pct'],
        'single_ofi_agreement_pct': single_ofi_agreement['overall_agreement_pct'],
        'single_ovi_agreement_pct': single_ovi_agreement['overall_agreement_pct'],
        'ensemble_improvement_pct': improvement,
    },
    'by_length': {
        length: {
            'agreement_pct': ensemble_agreement['agreement_pct_by_length'][length],
            'count': ensemble_agreement['totals_by_length'][length],
        }
        for length in sorted(ensemble_agreement['agreement_pct_by_length'].keys())
    },
    'by_class': {
        class_name: {
            'agreement_pct': ensemble_agreement['agreement_pct_by_class'][class_name],
            'count': ensemble_agreement['totals_by_class'][class_name],
        }
        for class_name in ['Down', 'Neutral', 'Up']
    },
}

# Print summary
print("\nEnsemble Configuration:")
for key, val in summary['ensemble_config'].items():
    print(f"  {key}: {val}")

print("\nTraining Configuration:")
for key, val in summary['training_results'].items():
    print(f"  {key}: {val}")

print("\nTest Performance:")
print(f"  Ensemble Agreement: {summary['test_performance']['ensemble_agreement_pct']:.2f}%")
print(f"  Single OFI Agreement: {summary['test_performance']['single_ofi_agreement_pct']:.2f}%")
print(f"  Single OVI Agreement: {summary['test_performance']['single_ovi_agreement_pct']:.2f}%")
print(f"  Improvement: {summary['test_performance']['ensemble_improvement_pct']:+.2f}%")

print("\nPerformance by Sequence Length:")
for length, stats in summary['by_length'].items():
    print(f"  Length {length}: {stats['agreement_pct']:.2f}% ({stats['count']} sequences)")

print("\nPerformance by Predicted Class:")
for class_name, stats in summary['by_class'].items():
    print(f"  {class_name}: {stats['agreement_pct']:.2f}% ({stats['count']} sequences)")

# Step 14: Save summary report
def save_summary_report(path, summary_dict):
    """Save summary report to file"""
    import json
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_summary = convert_to_serializable(summary_dict)
    
    with open(path, 'w') as f:
        json.dump(serializable_summary, f, indent=2)
    
    print(f"✓ Summary report saved to {path}")


save_summary_report('ensemble_summary.json', summary)

# Step 15: Create visualizations (optional)
def plot_ensemble_results(agreement_results, entropy_results=None, figsize=(16, 10)):
    """
    Create comprehensive visualization of ensemble results
    
    Args:
        agreement_results: from compute_prediction_agreement()
        entropy_results: from compute_entropy_binned_divergences()
        figsize: figure size
    """
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=figsize)
    
    # Plot 1: Agreement by sequence length
    ax1 = plt.subplot(2, 3, 1)
    lengths = sorted(agreement_results['agreement_pct_by_length'].keys())
    agreement_pcts = [agreement_results['agreement_pct_by_length'][l] for l in lengths]
    counts = [agreement_results['totals_by_length'][l] for l in lengths]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(lengths)))
    bars1 = ax1.bar(range(len(lengths)), agreement_pcts, color=colors, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Sequence Length', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Agreement %', fontsize=11, fontweight='bold')
    ax1.set_title('Prediction Agreement by Sequence Length', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(lengths)))
    ax1.set_xticklabels(lengths)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, pct, count in zip(bars1, agreement_pcts, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%\n(n={count})',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Agreement by class
    ax2 = plt.subplot(2, 3, 2)
    class_names = list(agreement_results['agreement_pct_by_class'].keys())
    class_pcts = [agreement_results['agreement_pct_by_class'][c] for c in class_names]
    class_counts = [agreement_results['totals_by_class'][c] for c in class_names]
    
    colors2 = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
    bars2 = ax2.bar(range(len(class_names)), class_pcts, color=colors2, 
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Agreement %', fontsize=11, fontweight='bold')
    ax2.set_title('Prediction Agreement by Class', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, pct, count in zip(bars2, class_pcts, class_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%\n(n={count})',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Sample count distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(range(len(lengths)), counts, color=colors, edgecolor='black', linewidth=1.5)
    
    ax3.set_xlabel('Sequence Length', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Sequences', fontsize=11, fontweight='bold')
    ax3.set_title('Sample Count Distribution', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(lengths)))
    ax3.set_xticklabels(lengths)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add count labels
    for i, count in enumerate(counts):
        ax3.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Entropy-based divergence (if available)
    if entropy_results is not None:
        ax4 = plt.subplot(2, 3, 4)
        
        bins = sorted(entropy_results['stats_by_bin'].keys())
        divergences = [entropy_results['stats_by_bin'][b]['mean_div'] for b in bins]
        stds = [entropy_results['stats_by_bin'][b]['std_div'] for b in bins]
        bin_labels = [f"Bin {b}" for b in bins]
        
        colors_entropy = plt.cm.plasma(np.linspace(0, 1, len(bins)))
        bars4 = ax4.bar(range(len(bins)), divergences, yerr=stds, capsize=5,
                       color=colors_entropy, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax4.set_xlabel('Entropy Bin', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Mean KL Divergence', fontsize=11, fontweight='bold')
        ax4.set_title('Divergence by Empirical Entropy', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(bins)))
        ax4.set_xticklabels(bin_labels, rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, div in zip(bars4, divergences):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{div:.4f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 5: Entropy bin sample counts
    if entropy_results is not None:
        ax5 = plt.subplot(2, 3, 5)
        
        bin_counts = [entropy_results['stats_by_bin'][b]['count'] for b in bins]
        bars5 = ax5.bar(range(len(bins)), bin_counts, color=colors_entropy,
                       edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax5.set_xlabel('Entropy Bin', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Number of Sequences', fontsize=11, fontweight='bold')
        ax5.set_title('Sample Count by Entropy Bin', fontsize=12, fontweight='bold')
        ax5.set_xticks(range(len(bins)))
        ax5.set_xticklabels(bin_labels, rotation=45)
        ax5.grid(axis='y', alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars5, bin_counts):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Overall statistics summary (text)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""
ENSEMBLE PERFORMANCE SUMMARY

Overall Agreement: {agreement_results['overall_agreement_pct']:.2f}%
  Total Agreements: {agreement_results['total_agreements']}/{agreement_results['total_samples']}

Best Length: {max(agreement_results['agreement_pct_by_length'].items(), key=lambda x: x[1])[0]}
  Agreement: {max(agreement_results['agreement_pct_by_length'].values()):.2f}%

Worst Length: {min(agreement_results['agreement_pct_by_length'].items(), key=lambda x: x[1])[0]}
  Agreement: {min(agreement_results['agreement_pct_by_length'].values()):.2f}%

Best Class: {max(agreement_results['agreement_pct_by_class'].items(), key=lambda x: x[1])[0]}
  Agreement: {max(agreement_results['agreement_pct_by_class'].values()):.2f}%

Worst Class: {min(agreement_results['agreement_pct_by_class'].items(), key=lambda x: x[1])[0]}
  Agreement: {min(agreement_results['agreement_pct_by_class'].values()):.2f}%
"""
    
    if entropy_results is not None:
        avg_divergence = np.mean([entropy_results['stats_by_bin'][b]['mean_div'] 
                                  for b in entropy_results['stats_by_bin'].keys()])
        summary_text += f"\nAverage KL Divergence: {avg_divergence:.6f}"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return fig


# Usage
fig = plot_ensemble_results(
    agreement_results,
    entropy_results=entropy_results,
    figsize=(16, 10)
)

plt.savefig('ensemble_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to ensemble_results.png")
plt.show()

# Step 16: Detailed error analysis
print("\n" + "=" * 70)
print("DETAILED ERROR ANALYSIS")
print("=" * 70)

# Find disagreements
disagreements = [r for r in agreement_results['detailed_results'] if not r['agrees']]
print(f"\nTotal disagreements: {len(disagreements)}")

if disagreements:
    print("\nFirst 10 disagreements:")
    print("-" * 70)
    print(f"{'Seq':20s} | {'Empirical':12s} | {'Model':12s} | {'Emp Prob':10s} | {'Mod Prob':10s}")
    print("-" * 70)
    
    for i, disagreement in enumerate(disagreements[:10]):
        seq_str = str(disagreement['sequence'])[:20]
        emp_class = disagreement['emp_class']
        mod_class = disagreement['mod_class']
        emp_prob = disagreement['emp_prob']
        mod_prob = disagreement['mod_prob']
        
        print(f"{seq_str:20s} | {emp_class:12s} | {mod_class:12s} | {emp_prob:10.4f} | {mod_prob:10.4f}")

# Step 17: Final statistics
print("\n" + "=" * 70)
print("FINAL STATISTICS")
print("=" * 70)

print(f"\nDataset Statistics:")
print(f"  Total test sequences: {agreement_results['total_samples']}")
print(f"  Correct predictions: {agreement_results['total_agreements']}")
print(f"  Wrong predictions: {agreement_results['total_samples'] - agreement_results['total_agreements']}")
print(f"  Overall accuracy: {agreement_results['overall_agreement_pct']:.2f}%")

print(f"\nSequence Length Statistics:")
for length in sorted(agreement_results['agreement_pct_by_length'].keys()):
    pct = agreement_results['agreement_pct_by_length'][length]
    count = agreement_results['totals_by_length'][length]
    print(f"  Length {length:2d}: {pct:6.2f}% accuracy ({count:5d} sequences)")

print(f"\nClass Distribution Statistics:")
for class_name in ['Down', 'Neutral', 'Up']:
    pct = agreement_results['agreement_pct_by_class'][class_name]
    count = agreement_results['totals_by_class'][class_name]
    print(f"  {class_name:8s}: {pct:6.2f}% accuracy ({count:5d} sequences)")

print(f"\nCohesion Metrics:")
print(f"  Std Dev of length accuracies: {np.std(list(agreement_results['agreement_pct_by_length'].values())):.2f}%")
print(f"  Std Dev of class accuracies: {np.std(list(agreement_results['agreement_pct_by_class'].values())):.2f}%")

# Step 18: Export detailed predictions for analysis
def export_predictions_to_csv(path, sequences, empirical_dists, model_dists, 
                              agreement_results, class_names=['Down', 'Neutral', 'Up']):
    """Export detailed predictions to CSV for further analysis"""
    import csv
    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['sequence', 'length', 'emp_class', 'mod_class', 'agrees']
        for cn in class_names:
            header.append(f'emp_{cn}')
        for cn in class_names:
            header.append(f'mod_{cn}')
        writer.writerow(header)
        
        # Data rows
        for detail in agreement_results['detailed_results']:
            row = [
                str(detail['sequence']),
                detail['length'],
                detail['emp_class'],
                detail['mod_class'],
                int(detail['agrees']),
            ]
            row.extend(detail['emp_dist'])
            row.extend(detail['mod_dist'])
            writer.writerow(row)
    
    print(f"✓ Predictions exported to {path}")


export_predictions_to_csv(
    'ensemble_predictions.csv',
    test_seqs_ofi,
    test_target_dists,
    test_probs,
    agreement_results
)

# Step 19: Performance comparison with baseline
print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

# Random predictor baseline
random_accuracy = 100.0 / 3  # 3 classes
print(f"Random classifier accuracy: {random_accuracy:.2f}%")

# Majority class baseline
majority_class = max(agreement_results['agreement_pct_by_class'].items(), 
                     key=lambda x: x[1])[0]
majority_count = agreement_results['totals_by_class'][majority_class]
majority_accuracy = 100.0 * majority_count / agreement_results['total_samples']
print(f"Majority class baseline ({majority_class}): {majority_accuracy:.2f}%")

# Ensemble improvement
ensemble_accuracy = agreement_results['overall_agreement_pct']
improvement_over_random = ensemble_accuracy - random_accuracy
improvement_over_majority = ensemble_accuracy - majority_accuracy

print(f"\nEnsemble Accuracy: {ensemble_accuracy:.2f}%")
print(f"  Improvement over random: +{improvement_over_random:.2f}%")
print(f"  Improvement over majority: {improvement_over_majority:+.2f}%")

# Step 20: Final report to file
final_report = f"""
{'=' * 80}
MULTI-ENCODER QUANTUM ENSEMBLE - FINAL EVALUATION REPORT
{'=' * 80}

ENSEMBLE CONFIGURATION
{'-' * 80}
Number of encoders: {ensemble_trained.n_encoders}
Individual encoder dimension: {ensemble_trained.d}
Product state dimension: {ensemble_trained.d_product}
Output dimension: {ensemble_trained.d_out}
Normalization point: {ensemble_trained.normalization_point}

TRAINING CONFIGURATION
{'-' * 80}
Learning rate (decoder): 5e-3
Learning rate (encoders): 5e-4
Epochs trained: 200
Batch size: 8192
Prediction loss: KL divergence
Freeze encoders: True
Freeze decoder: False

TEST SET STATISTICS
{'-' * 80}
Total test sequences: {agreement_results['total_samples']}
Correct predictions: {agreement_results['total_agreements']}
Wrong predictions: {agreement_results['total_samples'] - agreement_results['total_agreements']}
Overall accuracy: {agreement_results['overall_agreement_pct']:.2f}%

PERFORMANCE BY SEQUENCE LENGTH
{'-' * 80}
"""

for length in sorted(agreement_results['agreement_pct_by_length'].keys()):
    pct = agreement_results['agreement_pct_by_length'][length]
    count = agreement_results['totals_by_length'][length]
    final_report += f"  Length {length:2d}: {pct:6.2f}% ({count:5d} sequences)\n"

final_report += f"\nPERFORMANCE BY PREDICTED CLASS\n"
final_report += f"{'-' * 80}\n"

for class_name in ['Down', 'Neutral', 'Up']:
    pct = agreement_results['agreement_pct_by_class'][class_name]
    count = agreement_results['totals_by_class'][class_name]
    final_report += f"  {class_name:8s}: {pct:6.2f}% ({count:5d} sequences)\n"

final_report += f"\nBASELINE COMPARISON\n"
final_report += f"{'-' * 80}\n"
final_report += f"Random classifier: {random_accuracy:.2f}%\n"
final_report += f"Majority class ({majority_class}): {majority_accuracy:.2f}%\n"
final_report += f"Ensemble model: {ensemble_accuracy:.2f}%\n"
final_report += f"  Improvement over random: +{improvement_over_random:.2f}%\n"
final_report += f"  Improvement over majority: {improvement_over_majority:+.2f}%\n"

if entropy_results is not None:
    final_report += f"\nDIVERGENCE ANALYSIS BY ENTROPY\n"
    final_report += f"{'-' * 80}\n"
    
    for bin_idx in sorted(entropy_results['stats_by_bin'].keys()):
        stats = entropy_results['stats_by_bin'][bin_idx]
        final_report += f"  Bin {bin_idx} ({stats['entropy_range']}):\n"
        final_report += f"    Mean KL divergence: {stats['mean_div']:.6f}\n"
        final_report += f"    Std KL divergence: {stats['std_div']:.6f}\n"
        final_report += f"    Sample count: {stats['count']}\n"

final_report += f"\nKEY FINDINGS\n"
final_report += f"{'-' * 80}\n"
final_report += f"Best performing length: {max(agreement_results['agreement_pct_by_length'].items(), key=lambda x: x[1])[0]} " \
                f"({max(agreement_results['agreement_pct_by_length'].values()):.2f}%)\n"
final_report += f"Worst performing length: {min(agreement_results['agreement_pct_by_length'].items(), key=lambda x: x[1])[0]} " \
                f"({min(agreement_results['agreement_pct_by_length'].values()):.2f}%)\n"
final_report += f"Best performing class: {max(agreement_results['agreement_pct_by_class'].items(), key=lambda x: x[1])[0]} " \
                f"({max(agreement_results['agreement_pct_by_class'].values()):.2f}%)\n"
final_report += f"Worst performing class: {min(agreement_results['agreement_pct_by_class'].items(), key=lambda x: x[1])[0]} " \
                f"({min(agreement_results['agreement_pct_by_class'].values()):.2f}%)\n"

final_report += f"\nRECOMMENDATIONS\n"
final_report += f"{'-' * 80}\n"

if ensemble_accuracy > majority_accuracy + 5:
    final_report += f"✓ Ensemble shows strong improvement over baseline methods.\n"
    final_report += f"  Recommend deploying this model for production use.\n"
elif ensemble_accuracy > majority_accuracy:
    final_report += f"✓ Ensemble shows moderate improvement over baseline methods.\n"
    final_report += f"  Consider further optimization before production deployment.\n"
else:
    final_report += f"⚠ Ensemble does not improve over baseline methods.\n"
    final_report += f"  Consider retraining with different hyperparameters.\n"

if improvement_over_random > 20:
    final_report += f"✓ Model significantly outperforms random prediction.\n"
else:
    final_report += f"⚠ Model only marginally outperforms random prediction.\n"

# Identify weak areas
weak_lengths = [l for l, pct in agreement_results['agreement_pct_by_length'].items() 
                if pct < agreement_results['overall_agreement_pct'] - 5]
if weak_lengths:
    final_report += f"⚠ Performance drops for sequences of length: {weak_lengths}\n"
    final_report += f"  Consider additional training data or architecture changes for these lengths.\n"

final_report += f"\n{'=' * 80}\n"
final_report += f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
final_report += f"{'=' * 80}\n"

# Save report
with open('ensemble_final_report.txt', 'w') as f:
    f.write(final_report)

print(final_report)
print("✓ Final report saved to ensemble_final_report.txt")

# Step 21: Archive all results
def archive_results(archive_name='ensemble_results_archive'):
    """Create archive with all results"""
    import zipfile
    from pathlib import Path
    
    files_to_archive = [
        'ensemble_v1.pt',
        'ensemble_eval_results.json',
        'ensemble_summary.json',
        'ensemble_predictions.csv',
        'ensemble_results.png',
        'ensemble_final_report.txt',
    ]
    
    with zipfile.ZipFile(f'{archive_name}.zip', 'w') as zipf:
        for file in files_to_archive:
            if Path(file).exists():
                zipf.write(file)
                print(f"  ✓ Added {file}")
    
    print(f"\n✓ All results archived to {archive_name}.zip")

aarchive_results('ensemble_results_20260407')

print("\n" + "=" * 80)
print("ENSEMBLE TRAINING AND EVALUATION COMPLETE")
print("=" * 80)
print("\nGenerated Files:")
print("  ✓ ensemble_v1.pt - Trained model weights")
print("  ✓ ensemble_eval_results.json - Detailed evaluation metrics")
print("  ✓ ensemble_summary.json - Summary statistics")
print("  ✓ ensemble_predictions.csv - Individual predictions for all sequences")
print("  ✓ ensemble_results.png - Visualization plots")
print("  ✓ ensemble_final_report.txt - Comprehensive text report")
print("  ✓ ensemble_results_20260407.zip - Archive of all results")

print("\nNext Steps:")
print("  1. Review ensemble_final_report.txt for detailed analysis")
print("  2. Check ensemble_results.png for visualization insights")
print("  3. Load model with: ensemble_loaded, meta = load_ensemble_model('ensemble_v1.pt')")
print("  4. Use for production: predictions, traces = predict_ensemble(...)")

print("\nModel Information:")
print(f"  Encoders: {ensemble_trained.n_encoders}")
print(f"  System dimension: {ensemble_trained.d}")
print(f"  Product dimension: {ensemble_trained.d_product}")
print(f"  Output classes: {ensemble_trained.d_out}")

print("\nKey Metrics:")
print(f"  Overall accuracy: {agreement_results['overall_agreement_pct']:.2f}%")
print(f"  Total test sequences: {agreement_results['total_samples']:,}")
print(f"  Correct predictions: {agreement_results['total_agreements']:,}")

print("\n" + "=" * 80)
print("Thank you for using the Multi-Encoder Quantum Ensemble framework!")
print("=" * 80 + "\n")

'''    
#else:              #Validation mode
    # Configuration
    
    #prediction_loss in 'xEntropy', 'klDivergence', 'jsDivergence', 'mseError'"
    #prediction_loss = 'mseError'
    
    n_qubits = 5  # system dimension d = 2^n_qubits = 8
    d = 2 ** n_qubits
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rho0_type = "mixed"  # or  "pure"

    
    # -------------------------------------------------------------------------
    # particular Validaton task parameters


    variate = 'bivariate'
    
    features_list =  ['log_mid',"tvi_n" , 'obi_L1', "ofi_L1_n_norm",'ofi_L3_norm_n','ofi_L10_norm_n',"micro_price","ofi_L1_n", 'ofi_L1_norm_n']

    predicted =  features_list[0]  # 'log_mid_sym'
    predictor =  features_list[1]  # "tvi_n" 
    predictor =  features_list[2]  # 'obi_L1'
    predictor =  features_list[3]  # 'ofi_L1_n_norm'
    n_symbols = 4   #observable symbols per feature    
    
    m        = n_symbols*n_symbols  # alphabet size (e.g., 4x4 for price+OFI encoding) for bivariate only
    
    # resampling 
    frequency = 100 #events
    freq_units = 'evn'
    
    max_seq_len =  6          # max sequence length to be used for training 
    min_seq_prob = 0.000000   # threshold for rare events
    
    # Load empirical sequence data
    # sequences_all, emp_probs_all = load_your_empirical_data()
    # Filter by length and probability
    
    
    # Verify initialization
    print(f"  Alphabet size (m): {m}")
    print(f"  System dimension (d): {d}")
    print(f"  Initial state type: {rho0_type}")

    #--------------------------------------------------------------------------
    # Validation Data Load
    #--------------------------------------------------------------------------
    fPath = '..\\Data Preparation\\' 
    # SEQ_DISTR_AAPL_bivariate_log_mid-micro_price_202504

    # empirical sequences distributions
    title = "SEQ_DISTR_"+symbol+"_"+variate+"_"+predicted+"-"+predictor+"_"+date

    infname = fPath+title
    distrs_samples =  pickle.load(open( infname, "rb") ) 

    sequences_all = distrs_samples[1]
    emp_probs_all = [i[1] for i in distrs_samples[0]] # empirical proabilities
  
    sequences = []
    emp_probs = []
    for i in range(len(sequences_all)):
        if len(sequences_all[i]) <= max_seq_len and emp_probs_all[i] > min_seq_prob:
            sequences.append(sequences_all[i])
            emp_probs.append(emp_probs_all[i])
    
    print(f"Number of validation sequences: {len(sequences)}")
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
    # empirical class distributions conritioned on sequence prefix
    # include class_type in the name
    title = "CLS_DISTR_"+symbol+"_"+"_"+predicted+"-"+predictor+"_"+date+"_"+clsName
    infname = fPath+title
    
    class_distributions =  pickle.load(open( infname, "rb") ) 
    sequences, target_distributions, seq_probs = integrate_data(class_distributions, distrs_samples)
    
    # calculate empirical weight of a sequence 
    global_weights = compute_global_weights(sequences, seq_probs)
    
    
    print(f"Number of unique sequences: {len(sequences)}")

    # -----------------------------
    # Step 3: Load full predictive model
    # -----------------------------

    print("\n" + "=" * 60)
    print("STEP 3: Loading predictive model (encoder + decoder) Loss=", prediction_loss)
    print("=" * 60)
    d_out = 3  # prediction target dimension (e.g., 4 mid-price symbols)

    
    title = "PRD_" +clsName+"_"+symbol+"_"+predicted+"-"+predictor+"_"+date+'_'+prediction_loss+'_'+str(n_qubits)+'q'
    
    save_file = fPath+title
    
    #has been saved as:  save_predictive_model(save_file, pred_model)
    
    # Load
    pred_model, meta_loaded  = load_predictive_model(save_file, device='cpu')
    
    #print(f"Loaded model trained for {meta_loaded['epochs_trained']} epochs")
    #print(f"Final training loss: {meta_loaded['final_loss']}")
    
    # Use for inference
    result = predict_from_sequence(pred_model, [0, 1, 2])

    
    #-----------------------------------------------------------------------------
    # Model Output - generative probabilities and class distributions for 
    
    # for set of sequences
    mod_seq_probs, mod_target_distributions = get_model_predictions_ordered(pred_model, sequences)
    print('Predicted')
    
    # Compute agreement
    results = compute_prediction_agreement(
        sequences,
        target_distributions,
        mod_target_distributions,
        class_names=['Down', 'Neutral', 'Up']
    )
    
    # Print summary
    print("=" * 70)
    print("PREDICTION AGREEMENT ANALYSIS")
    print(symbol, date, predicted, predictor, str(n_qubits)+'q','Class type', class_type, prediction_loss)
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

sequences_price_tvi = sequences_all  # Sequences (price+OFI features)
sequences_price_ofi = sequences_all  # Same sequences (price+OVI features)
sequences_price_obi = sequences_all  # Same sequences (price+volume features)

sequences_list = [
    sequences_price_ofi,
    sequences_price_ovi,
    sequences_price_vol,
]

'''

