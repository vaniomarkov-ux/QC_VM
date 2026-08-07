# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 18:27:38 2026

@author: vanio
"""
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#-----------------------------------------------------------------------------
import pickle
#-----------------------------------------------------------------------------
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

PAD = -1  # must be outside symbol range {0,...,m-1}
device = 'cpu'
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
    
def sequences_to_ragged(
    sequences,
    pad_value=None,
):
    """
    Accept either:

        list[list[int]]

    or:

        padded LongTensor[B, T]

    and return a list of unpadded Python sequences.
    """
    if torch.is_tensor(sequences):
        if sequences.ndim != 2:
            raise ValueError(
                "Padded sequences must have shape (B, T)."
            )

        rows = sequences.detach().cpu().tolist()
    else:
        rows = [
            list(sequence)
            for sequence in sequences
        ]

    output = []

    for index, row in enumerate(rows):
        row = [int(symbol) for symbol in row]

        if pad_value is not None and pad_value in row:
            first_pad = row.index(pad_value)

            if any(
                symbol != pad_value
                for symbol in row[first_pad:]
            ):
                raise ValueError(
                    f"Non-PAD symbol after PAD in sequence {index}."
                )

            row = row[:first_pad]

        if len(row) == 0:
            raise ValueError(
                f"Empty sequence at index {index}."
            )

        output.append(row)

    return output  
#-----------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------
# Π-representation encoder
#-----------------------------------------------------------------------------
class PiRepresentationEncoder(nn.Module):
    """
    Frozen Kraus encoder producing factorized Pi representations:

        Pi(sequence) = |u><w|

        u = vec(K rho0) / sqrt(d)
        w = vec(K)      / sqrt(d)

    Column-major vectorization is used.
    """

    def __init__(
        self,
        encoder,
        pad_value=None,
        normalize_factors: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.encoder = encoder
        self.pad_value = pad_value
        self.normalize_factors = normalize_factors
        self.eps = eps

        self.system_dimension = int(encoder.d)
        self.factor_dimension = self.system_dimension**2

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

        scale = 1.0 / math.sqrt(
            self.system_dimension
        )

        u_values = []
        w_values = []
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

            expected_shape = (
                self.system_dimension,
                self.system_dimension,
            )

            if tuple(K_sequence.shape) != expected_shape:
                raise ValueError(
                    f"K has shape {tuple(K_sequence.shape)}; "
                    f"expected {expected_shape}."
                )

            if tuple(rho0.shape) != expected_shape:
                raise ValueError(
                    f"rho0 has shape {tuple(rho0.shape)}; "
                    f"expected {expected_shape}."
                )

            # Sequence probability:
            # p(sequence) = Tr(K rho0 K†)
            tau = (
                K_sequence
                @ rho0
                @ K_sequence.conj().transpose(-2, -1)
            )

            sequence_probability = (
                torch.trace(tau).real
                .clamp_min(self.eps)
            )

            # Column-major vec(X):
            # vec_F(X) = flatten(X^T)
            u = (
                (K_sequence @ rho0)
                .transpose(-2, -1)
                .reshape(-1)
                * scale
            )

            w = (
                K_sequence
                .transpose(-2, -1)
                .reshape(-1)
                * scale
            )

            if self.normalize_factors:
                norm_u = torch.linalg.vector_norm(u)
                norm_w = torch.linalg.vector_norm(w)

                if norm_u <= self.eps:
                    raise ValueError(
                        f"Zero-norm u for sequence {sequence}."
                    )

                if norm_w <= self.eps:
                    raise ValueError(
                        f"Zero-norm w for sequence {sequence}."
                    )

                u = u / norm_u
                w = w / norm_w

            u_values.append(u)
            w_values.append(w)

            sequence_probabilities.append(
                sequence_probability
            )

        return (
            torch.stack(u_values, dim=0),
            torch.stack(w_values, dim=0),
            torch.stack(
                sequence_probabilities,
                dim=0,
            ),
        )
#-----------------------------------------------------------------------------
# Optional structured unitary
#-----------------------------------------------------------------------------

class ComplexHouseholderUnitary(nn.Module):
    """
    Structured unitary using complex Householder reflections.

    Cost:
        O(batch * dimension * n_reflections)
    """

    def __init__(
        self,
        dimension: int,
        n_reflections: int = 8,
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

    def forward(self, x):
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


#-----------------------------------------------------------------------------
# Optional low rank general map
#-----------------------------------------------------------------------------

class ComplexLowRankResidual(nn.Module):
    """
    General non-unitary residual map:

        G = I + alpha L R†
    """

    def __init__(
        self,
        dimension: int,
        rank: int = 8,
        initial_alpha: float = 0.02,
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

        self.raw_alpha = nn.Parameter(
            torch.tensor(
                float(initial_alpha)
            )
        )

    def forward(self, x):
        L = torch.complex(
            self.L_re,
            self.L_im,
        ).to(dtype=x.dtype)

        R = torch.complex(
            self.R_re,
            self.R_im,
        ).to(dtype=x.dtype)

        # Bound the residual strength.
        alpha = torch.tanh(
            self.raw_alpha
        )

        coordinates = x @ R.conj()

        correction = (
            coordinates @ L.transpose(0, 1)
        )

        return x + alpha * correction
#-----------------------------------------------------------------------------
class PiBilinearDecoder(nn.Module):
    """
    Implicit decoder for:

        Pi = |u><w|.

    Class amplitude:

        a_y = u† M_y w

    with:

        M_y = A_y B_y†

    The explicit d_factor x d_factor Pi matrix is never created.
    """

    def __init__(
        self,
        factor_dimension: int,
        d_out: int,
        readout_rank: int = 8,
        *,
        use_unitary: bool = False,
        n_reflections: int = 8,
        use_general_map: bool = False,
        general_rank: int = 8,
        renormalize_after_transform: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.factor_dimension = factor_dimension
        self.d_out = d_out
        self.readout_rank = readout_rank
        self.eps = eps

        self.renormalize_after_transform = (
            renormalize_after_transform
        )

        if use_unitary:
            self.unitary = ComplexHouseholderUnitary(
                dimension=factor_dimension,
                n_reflections=n_reflections,
                eps=eps,
            )
        else:
            self.unitary = None

        if use_general_map:
            self.general_map = ComplexLowRankResidual(
                dimension=factor_dimension,
                rank=general_rank,
            )
        else:
            self.general_map = None

        scale = 1.0 / math.sqrt(
            factor_dimension
        )

        shape = (
            d_out,
            factor_dimension,
            readout_rank,
        )

        self.A_re = nn.Parameter(
            torch.randn(*shape) * scale
        )
        self.A_im = nn.Parameter(
            torch.randn(*shape) * scale
        )

        self.B_re = nn.Parameter(
            torch.randn(*shape) * scale
        )
        self.B_im = nn.Parameter(
            torch.randn(*shape) * scale
        )

    def _normalize(self, x):
        return x / torch.linalg.vector_norm(
            x,
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

    def forward(
        self,
        u,
        w,
        return_details: bool = False,
    ):
        u = self._normalize(u)
        w = self._normalize(w)

        if self.unitary is not None:
            u = self.unitary(u)
            w = self.unitary(w)

        if self.general_map is not None:
            u = self.general_map(u)
            w = self.general_map(w)

        if self.renormalize_after_transform:
            u = self._normalize(u)
            w = self._normalize(w)

        A = torch.complex(
            self.A_re,
            self.A_im,
        ).to(dtype=u.dtype)

        B = torch.complex(
            self.B_re,
            self.B_im,
        ).to(dtype=u.dtype)

        # u† A_y
        left = torch.einsum(
            "bd,odr->bor",
            u.conj(),
            A,
        )

        # B_y† w
        right = torch.einsum(
            "odr,bd->bor",
            B.conj(),
            w,
        )

        amplitudes = torch.sum(
            left * right,
            dim=-1,
        )

        powers = (
            torch.abs(amplitudes) ** 2
        ) + self.eps

        probabilities = powers / powers.sum(
            dim=-1,
            keepdim=True,
        )

        if return_details:
            return {
                "probabilities": probabilities,
                "amplitudes": amplitudes,
                "u_transformed": u,
                "w_transformed": w,
            }

        return probabilities

#-----------------------------------------------------------------------------
class PiPredictiveModel(nn.Module):
    """
    sequence
        -> frozen Kraus encoder
        -> Pi factors (u, w)
        -> bilinear decoder
        -> conditional target distribution
    """

    def __init__(
        self,
        representation_encoder,
        decoder,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.representation_encoder = (
            representation_encoder
        )

        self.decoder = decoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for parameter in (
                self.representation_encoder.parameters()
            ):
                parameter.requires_grad = False

    @property
    def encoder(self):
        return self.representation_encoder.encoder

    @property
    def pad_value(self):
        return self.representation_encoder.pad_value

    def forward(
        self,
        sequences,
        return_details: bool = False,
    ):
        if self.freeze_encoder:
            with torch.no_grad():
                (
                    u,
                    w,
                    sequence_probabilities,
                ) = self.representation_encoder(
                    sequences
                )
        else:
            (
                u,
                w,
                sequence_probabilities,
            ) = self.representation_encoder(
                sequences
            )

        if return_details:
            decoder_output = self.decoder(
                u,
                w,
                return_details=True,
            )

            return {
                "u": u,
                "w": w,
                "sequence_probabilities": (
                    sequence_probabilities
                ),
                **decoder_output,
            }

        predicted_probabilities = self.decoder(
            u,
            w,
        )

        return (
            predicted_probabilities,
            sequence_probabilities.real.reshape(-1),
        )
#------------------------------------------------------------------------------
@torch.inference_mode()
def precompute_pi_factors(
    representation_encoder,
    sequences,
    *,
    batch_size: int = 512,
    device=None,
    show_progress: bool = True,
):
    """
    Precompute u, w, and p_model(sequence) once.
    """
    sequences = list(sequences)

    if device is None:
        device = next(
            representation_encoder.parameters()
        ).device
    else:
        device = torch.device(device)

    representation_encoder = (
        representation_encoder.to(device)
    )

    representation_encoder.eval()

    u_values = []
    w_values = []
    probability_values = []

    batch_starts = range(
        0,
        len(sequences),
        batch_size,
    )

    iterator = tqdm(
        batch_starts,
        total=(
            len(sequences)
            + batch_size
            - 1
        ) // batch_size,
        desc="Precomputing Pi factors",
        unit="batch",
        disable=not show_progress,
    )

    for start in iterator:
        batch_sequences = sequences[
            start:start + batch_size
        ]

        u, w, probabilities = (
            representation_encoder(
                batch_sequences
            )
        )

        u_values.append(
            u.detach().cpu()
        )

        w_values.append(
            w.detach().cpu()
        )

        probability_values.append(
            probabilities.detach().cpu()
        )

    return (
        torch.cat(u_values, dim=0),
        torch.cat(w_values, dim=0),
        torch.cat(
            probability_values,
            dim=0,
        ),
    )
#------------------------------------------------------------------------------
class CachedPiDataset(Dataset):
    def __init__(
        self,
        u,
        w,
        target_distributions,
        global_weights,
    ):
        self.u = torch.as_tensor(
            u,
            dtype=torch.complex64,
        )

        self.w = torch.as_tensor(
            w,
            dtype=torch.complex64,
        )

        self.targets = torch.as_tensor(
            target_distributions,
            dtype=torch.float32,
        )

        self.weights = torch.as_tensor(
            global_weights,
            dtype=torch.float32,
        ).reshape(-1)

        number_of_samples = self.u.shape[0]

        if self.w.shape[0] != number_of_samples:
            raise ValueError(
                "u and w have different sample counts."
            )

        if self.targets.shape[0] != number_of_samples:
            raise ValueError(
                "Target sample count mismatch."
            )

        if self.weights.shape[0] != number_of_samples:
            raise ValueError(
                "Weight sample count mismatch."
            )

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, index):
        return (
            self.u[index],
            self.w[index],
            self.targets[index],
            self.weights[index],
        )

#------------------------------------------------------------------------------
def normalize_distributions(
    probabilities,
    eps=1e-8,
):
    probabilities = probabilities.clamp_min(0.0)

    return probabilities / probabilities.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
#------------------------------------------------------------------------------
def prediction_loss_per_sample(
    prediction,
    target,
    loss_type="mseError",
    eps=1e-8,
):
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
            target
            * torch.log(
                prediction.clamp_min(eps)
            ),
            dim=-1,
        )

    if name in {
        "kl",
        "kldivergence",
        "kl_divergence",
    }:
        return torch.sum(
            target
            * (
                torch.log(target.clamp_min(eps))
                - torch.log(
                    prediction.clamp_min(eps)
                )
            ),
            dim=-1,
        )

    if name in {
        "js",
        "jsdivergence",
        "js_divergence",
    }:
        midpoint = 0.5 * (
            prediction + target
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
                torch.log(
                    prediction.clamp_min(eps)
                )
                - torch.log(midpoint.clamp_min(eps))
            ),
            dim=-1,
        )

        return 0.5 * (
            kl_target + kl_prediction
        )

    raise ValueError(
        f"Unknown prediction loss {loss_type!r}."
    )

#------------------------------------------------------------------------------
def weighted_mean(
    values,
    weights,
    eps=1e-8,
):
    values = values.reshape(-1)
    weights = weights.reshape(-1)

    weights = weights.clamp_min(0.0)

    if weights.sum() <= eps:
        weights = torch.ones_like(weights)

    return torch.sum(
        values * weights
    ) / weights.sum().clamp_min(eps)
#------------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_pi_decoder(
    decoder,
    dataloader,
    device,
    prediction_loss="mseError",
    eps=1e-8,
):
    decoder.eval()

    loss_numerator = 0.0
    weight_denominator = 0.0

    true_labels = []
    predicted_labels = []

    for u, w, targets, weights in dataloader:
        u = u.to(device)
        w = w.to(device)

        targets = targets.to(
            device=device,
            dtype=torch.float32,
        )

        weights = weights.to(
            device=device,
            dtype=torch.float32,
        )

        predictions = decoder(u, w)

        losses = prediction_loss_per_sample(
            predictions,
            targets,
            loss_type=prediction_loss,
            eps=eps,
        )

        loss_numerator += float(
            torch.sum(losses * weights)
            .detach()
            .cpu()
        )

        weight_denominator += float(
            weights.sum().detach().cpu()
        )

        true_labels.extend(
            torch.argmax(
                targets,
                dim=-1,
            ).cpu().tolist()
        )

        predicted_labels.extend(
            torch.argmax(
                predictions,
                dim=-1,
            ).cpu().tolist()
        )

    mean_loss = (
        loss_numerator
        / max(weight_denominator, eps)
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    return {
        "loss": mean_loss,
        "macro_f1": macro_f1,
    }
#-----------------------------------------------------------------------------------------------------------
def get_model_predictions_ordered(
    model,
    sequences,
    batch_size: int = 512,
    device=None,
):
    """
    Return:

        p_model(sequence)
        p_model(class | sequence)

    in the original sequence order.
    """
    sequences = list(sequences)

    if len(sequences) == 0:
        return [], []

    if device is None:
        device = next(
            model.parameters()
        ).device
    else:
        device = torch.device(device)
        model = model.to(device)

    model.eval()

    mod_seq_probs = []
    mod_target_distributions = []

    with torch.inference_mode():
        for start in range(
            0,
            len(sequences),
            batch_size,
        ):
            batch_sequences = sequences[
                start:start + batch_size
            ]

            (
                predicted_distributions,
                model_sequence_probabilities,
            ) = model(batch_sequences)

            mod_seq_probs.extend(
                model_sequence_probabilities
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

    return (
        mod_seq_probs,
        mod_target_distributions,
    )

#-----------------------------------------------------------------------------------------------------------
def train_predictive_model_pi(
    sequences,
    target_distributions,
    seq_probs,
    global_weights,
    encoder,
    d_out: int,
    *,
    valid_sequences=None,
    valid_target_distributions=None,
    valid_global_weights=None,
    pad_value=None,
    batch_size: int = 512,
    representation_batch_size: int = 512,
    lr: float = 1e-3,
    epochs: int = 100,
    weight_decay: float = 1e-4,
    prediction_loss: str = "mseError",
    optimizer_name: str = "adam",
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    ),
    normalize_factors: bool = True,
    readout_rank: int = 8,
    use_unitary: bool = False,
    n_reflections: int = 8,
    use_general_map: bool = False,
    general_rank: int = 8,
    gradient_clip_norm: float = 5.0,
    patience: int = 15,
    num_workers: int = 0,
    eps: float = 1e-8,
):
    """
    Train only the Pi decoder.

    The encoder is frozen and Pi factors are cached once.
    """
    device = torch.device(device)

    # --------------------------------------------------------
    # Representation encoder
    # --------------------------------------------------------

    representation_encoder = PiRepresentationEncoder(
        encoder=encoder,
        pad_value=pad_value,
        normalize_factors=normalize_factors,
        eps=eps,
    ).to(device)

    for parameter in representation_encoder.parameters():
        parameter.requires_grad = False

    representation_encoder.eval()

    factor_dimension = (
        representation_encoder.factor_dimension
    )

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    decoder = PiBilinearDecoder(
        factor_dimension=factor_dimension,
        d_out=d_out,
        readout_rank=readout_rank,
        use_unitary=use_unitary,
        n_reflections=n_reflections,
        use_general_map=use_general_map,
        general_rank=general_rank,
        eps=eps,
    ).to(device)

    pred_model = PiPredictiveModel(
        representation_encoder=representation_encoder,
        decoder=decoder,
        freeze_encoder=True,
    ).to(device)

    # --------------------------------------------------------
    # Precompute training representations
    # --------------------------------------------------------

    print(
        "\nPrecomputing training Pi factors...",
        flush=True,
    )

    u_train, w_train, model_seq_probs = (
        precompute_pi_factors(
            representation_encoder,
            sequences,
            batch_size=representation_batch_size,
            device=device,
            show_progress=True,
        )
    )

    train_dataset = CachedPiDataset(
        u=u_train,
        w=w_train,
        target_distributions=target_distributions,
        global_weights=global_weights,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # --------------------------------------------------------
    # Optional validation cache
    # --------------------------------------------------------

    valid_loader = None

    if valid_sequences is not None:
        if valid_target_distributions is None:
            raise ValueError(
                "valid_target_distributions is required."
            )

        if valid_global_weights is None:
            valid_global_weights = np.ones(
                len(valid_sequences),
                dtype=np.float32,
            )

        print(
            "\nPrecomputing validation Pi factors...",
            flush=True,
        )

        u_valid, w_valid, _ = precompute_pi_factors(
            representation_encoder,
            valid_sequences,
            batch_size=representation_batch_size,
            device=device,
            show_progress=True,
        )

        valid_dataset = CachedPiDataset(
            u=u_valid,
            w=w_valid,
            target_distributions=(
                valid_target_distributions
            ),
            global_weights=valid_global_weights,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            decoder.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(
            "optimizer_name must be 'adam' or 'adamw'."
        )

    trainable_decoder_parameters = sum(
        parameter.numel()
        for parameter in decoder.parameters()
        if parameter.requires_grad
    )

    print(
        "\nPi predictive model"
        f"\n  system dimension         : {encoder.d}"
        f"\n  Pi factor dimension      : {factor_dimension}"
        f"\n  explicit Pi dimension    : "
        f"{factor_dimension**2:,}"
        f"\n  output dimension         : {d_out}"
        f"\n  normalize factors        : {normalize_factors}"
        f"\n  readout rank             : {readout_rank}"
        f"\n  use unitary              : {use_unitary}"
        f"\n  use general map          : {use_general_map}"
        f"\n  trainable decoder params : "
        f"{trainable_decoder_parameters:,}",
        flush=True,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_state = None
    best_score = -float("inf")
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "train_macro_f1": [],
        "valid_loss": [],
        "valid_macro_f1": [],
    }

    for epoch in range(epochs):
        decoder.train()

        epoch_start = time.perf_counter()

        iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=False,
        )

        for u, w, targets, weights in iterator:
            u = u.to(
                device,
                non_blocking=True,
            )

            w = w.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            weights = weights.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = decoder(u, w)

            sample_losses = (
                prediction_loss_per_sample(
                    predictions,
                    targets,
                    loss_type=prediction_loss,
                    eps=eps,
                )
            )

            loss = weighted_mean(
                sample_losses,
                weights,
                eps=eps,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Nonfinite decoder loss."
                )

            loss.backward()

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    decoder.parameters(),
                    gradient_clip_norm,
                )

            optimizer.step()

            iterator.set_postfix(
                loss=f"{loss.detach().item():.6g}"
            )

        train_metrics = evaluate_pi_decoder(
            decoder,
            train_loader,
            device=device,
            prediction_loss=prediction_loss,
            eps=eps,
        )

        history["train_loss"].append(
            train_metrics["loss"]
        )

        history["train_macro_f1"].append(
            train_metrics["macro_f1"]
        )

        if valid_loader is not None:
            valid_metrics = evaluate_pi_decoder(
                decoder,
                valid_loader,
                device=device,
                prediction_loss=prediction_loss,
                eps=eps,
            )

            history["valid_loss"].append(
                valid_metrics["loss"]
            )

            history["valid_macro_f1"].append(
                valid_metrics["macro_f1"]
            )

            monitored_score = (
                valid_metrics["macro_f1"]
            )

        else:
            valid_metrics = None

            monitored_score = (
                train_metrics["macro_f1"]
            )

        epoch_seconds = (
            time.perf_counter() - epoch_start
        )

        message = (
            f"Epoch {epoch + 1:3d}/{epochs}"
            f" | train loss={train_metrics['loss']:.6g}"
            f" | train F1={train_metrics['macro_f1']:.4f}"
        )

        if valid_metrics is not None:
            message += (
                f" | valid loss="
                f"{valid_metrics['loss']:.6g}"
                f" | valid F1="
                f"{valid_metrics['macro_f1']:.4f}"
            )

        message += (
            f" | time={epoch_seconds:.1f}s"
        )

        print(message, flush=True)

        if monitored_score > best_score:
            best_score = monitored_score

            best_state = copy.deepcopy(
                decoder.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            valid_loader is not None
            and epochs_without_improvement >= patience
        ):
            print(
                f"Early stopping after epoch {epoch + 1}.",
                flush=True,
            )
            break

    if best_state is not None:
        decoder.load_state_dict(best_state)

    pred_model.eval()

    pred_model.training_history = history

    pred_model.training_metadata = {
        "representation": "pi",
        "factor_dimension": factor_dimension,
        "normalize_factors": normalize_factors,
        "readout_rank": readout_rank,
        "use_unitary": use_unitary,
        "use_general_map": use_general_map,
        "best_macro_f1": best_score,
        "cached_model_sequence_probabilities": (
            model_seq_probs
        ),
    }

    return pred_model
#-------------------------------------------------------------------------------
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
from pathlib import Path
def save_predictive_model(
    path,
    model,
    meta=None,
):
    """
    Save a complete Pi predictive model:

        KrausInstrument
        + PiRepresentationEncoder configuration
        + PiBilinearDecoder
        + PiPredictiveModel configuration
    """
    path = Path(path)

    representation_encoder = model.representation_encoder
    encoder = model.encoder
    decoder = model.decoder

    use_unitary = decoder.unitary is not None
    use_general_map = decoder.general_map is not None

    if use_unitary:
        n_reflections = int(
            decoder.unitary.n_reflections
        )
    else:
        n_reflections = 0

    if use_general_map:
        general_rank = int(
            decoder.general_map.rank
        )
    else:
        general_rank = 0

    encoder_config = {
        "m": int(encoder.m),
        "d": int(encoder.d),
        "learn_rho0": bool(
            encoder.learn_rho0
        ),

        # Important for mixed/pure initial-state reconstruction.
        "rho0_type": getattr(
            encoder,
            "rho0_type",
            "mixed",
        ),

        "eps": float(
            getattr(
                encoder,
                "eps",
                1e-8,
            )
        ),
    }

    representation_config = {
        "pad_value": (
            None
            if representation_encoder.pad_value is None
            else int(
                representation_encoder.pad_value
            )
        ),
        "normalize_factors": bool(
            representation_encoder.normalize_factors
        ),
        "eps": float(
            representation_encoder.eps
        ),
    }

    decoder_config = {
        "factor_dimension": int(
            decoder.factor_dimension
        ),
        "d_out": int(
            decoder.d_out
        ),
        "readout_rank": int(
            decoder.readout_rank
        ),
        "use_unitary": use_unitary,
        "n_reflections": n_reflections,
        "use_general_map": use_general_map,
        "general_rank": general_rank,
        "renormalize_after_transform": bool(
            decoder.renormalize_after_transform
        ),
        "eps": float(
            decoder.eps
        ),
    }

    model_config = {
        "freeze_encoder": bool(
            model.freeze_encoder
        ),
    }

    supplied_meta = (
        {}
        if meta is None
        else dict(meta)
    )

    payload = {
        "format": "pi_predictive_model",
        "format_version": 1,

        "encoder_state": (
            encoder.state_dict()
        ),
        "decoder_state": (
            decoder.state_dict()
        ),

        "encoder_config": encoder_config,
        "representation_config": (
            representation_config
        ),
        "decoder_config": decoder_config,
        "model_config": model_config,

        "training_history": getattr(
            model,
            "training_history",
            None,
        ),
        "training_metadata": getattr(
            model,
            "training_metadata",
            None,
        ),

        "meta": supplied_meta,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        payload,
        path,
    )

    print(
        f"Pi predictive model saved to {path}",
        flush=True,
    )

def load_predictive_model(
    path,
    device="cpu",
    *,
    freeze_encoder=None,
    strict=True,
):
    """
    Reconstruct and load a complete Pi predictive model.

    Parameters
    ----------
    path
        Saved checkpoint path.

    device
        Loading device.

    freeze_encoder
        None:
            restore the saved setting.

        True/False:
            override the saved setting.

    strict
        Passed to load_state_dict.

    Returns
    -------
    model
        Reconstructed PiPredictiveModel.

    meta
        User metadata saved with the model.
    """
    device = torch.device(device)

    payload = torch.load(
        path,
        map_location=device,
    )

    checkpoint_format = payload.get(
        "format"
    )

    if checkpoint_format != "pi_predictive_model":
        raise ValueError(
            "This checkpoint is not identified as a "
            "Pi predictive model. "
            f"Found format={checkpoint_format!r}."
        )

    # ---------------------------------------------------------
    # Reconstruct the Kraus encoder
    # ---------------------------------------------------------

    enc_cfg = payload[
        "encoder_config"
    ]

    encoder = KrausInstrument(
        m=enc_cfg["m"],
        d=enc_cfg["d"],
        learn_rho0=enc_cfg[
            "learn_rho0"
        ],
        rho0_type=enc_cfg.get(
            "rho0_type",
            "mixed",
        ),
        eps=enc_cfg.get(
            "eps",
            1e-8,
        ),
    )

    encoder_load_result = (
        encoder.load_state_dict(
            payload["encoder_state"],
            strict=strict,
        )
    )

    # ---------------------------------------------------------
    # Reconstruct the Pi representation encoder
    # ---------------------------------------------------------

    rep_cfg = payload[
        "representation_config"
    ]

    representation_encoder = (
        PiRepresentationEncoder(
            encoder=encoder,
            pad_value=rep_cfg.get(
                "pad_value"
            ),
            normalize_factors=rep_cfg.get(
                "normalize_factors",
                True,
            ),
            eps=rep_cfg.get(
                "eps",
                1e-8,
            ),
        )
    )

    # ---------------------------------------------------------
    # Reconstruct the Pi decoder
    # ---------------------------------------------------------

    dec_cfg = payload[
        "decoder_config"
    ]

    decoder = PiBilinearDecoder(
        factor_dimension=dec_cfg[
            "factor_dimension"
        ],
        d_out=dec_cfg[
            "d_out"
        ],
        readout_rank=dec_cfg[
            "readout_rank"
        ],
        use_unitary=dec_cfg.get(
            "use_unitary",
            False,
        ),
        n_reflections=dec_cfg.get(
            "n_reflections",
            8,
        ),
        use_general_map=dec_cfg.get(
            "use_general_map",
            False,
        ),
        general_rank=dec_cfg.get(
            "general_rank",
            8,
        ),
        renormalize_after_transform=(
            dec_cfg.get(
                "renormalize_after_transform",
                True,
            )
        ),
        eps=dec_cfg.get(
            "eps",
            1e-8,
        ),
    )

    decoder_load_result = (
        decoder.load_state_dict(
            payload["decoder_state"],
            strict=strict,
        )
    )

    # ---------------------------------------------------------
    # Reconstruct the complete predictive model
    # ---------------------------------------------------------

    saved_freeze_encoder = payload.get(
        "model_config",
        {},
    ).get(
        "freeze_encoder",
        True,
    )

    effective_freeze_encoder = (
        saved_freeze_encoder
        if freeze_encoder is None
        else bool(freeze_encoder)
    )

    model = PiPredictiveModel(
        representation_encoder=(
            representation_encoder
        ),
        decoder=decoder,
        freeze_encoder=(
            effective_freeze_encoder
        ),
    )

    model = model.to(device)
    model.eval()

    # Restore optional diagnostics.
    if payload.get(
        "training_history"
    ) is not None:
        model.training_history = payload[
            "training_history"
        ]

    if payload.get(
        "training_metadata"
    ) is not None:
        model.training_metadata = payload[
            "training_metadata"
        ]

    if not strict:
        print(
            "Encoder missing keys:",
            encoder_load_result.missing_keys,
        )
        print(
            "Encoder unexpected keys:",
            encoder_load_result.unexpected_keys,
        )
        print(
            "Decoder missing keys:",
            decoder_load_result.missing_keys,
        )
        print(
            "Decoder unexpected keys:",
            decoder_load_result.unexpected_keys,
        )

    return model, payload.get(
        "meta",
        {},
    )

#-----------------------------------------------------------------------------
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

title_model = "PRD_PI_" +clsName+"_"+symbol+"_"+predicted+"-"+predictor+"_"+date+'_'+prediction_loss+'_'+str(n_qubits)+'q'
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

max_seq_len =  6          # max sequence length to be used for training 
min_seq_prob = 0.000000   # threshold for rare events


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
min_seq_prob=0.00001
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

train_prediction_model = True

if train_prediction_model:
    pred_model = train_predictive_model_pi(
        sequences = training_sequences,
        target_distributions = target_distributions,
        seq_probs = training_seq_probs,
        global_weights = global_weights,
        encoder = encoder,
        d_out=d_out,
    
        valid_sequences=validation_sequences,
        valid_target_distributions=validation_targets,
        valid_global_weights=validation_weights,
    
        pad_value=PAD,
    
        batch_size=512,
        representation_batch_size=512,
        lr=1e-3,
        epochs=1000,
        normalize_factors=True,
        readout_rank=8,
        use_unitary=False,
        use_general_map=True,
        general_rank = 8,
    
        prediction_loss="mseError",
        patience=15,
        device=device,
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
        
         
    # save trained model              
    save_predictive_model(save_model_file, pred_model, meta)
    print("Predictive Model saved as", save_model_file)
else:                         #Load prediction model
    pred_model, meta = load_predictive_model(save_model_file, device='cpu')
    
    print("Predictive Model loaded from", save_model_file)
    # print(f"Loaded model trained for {meta_loaded['epochs_trained']} epochs")
    # print(f"Final training loss: {meta_loaded['final_loss']}")
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
 # OUT OF SAMPLE VERIFICATION  
# Read out of sample data ------------------------------------------------
#-------------------------------------------------------------------------------
# Read out of sample data ------------------------------------------------
print('Out of sampe test: Training date', date,' Validation date', date_validation)

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
