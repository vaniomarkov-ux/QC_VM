from __future__ import annotations
import pickle
import numpy as np
from plot_distributions import plotDistribution,plotDistributions 
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional, Any
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from spectral_analysis import gram_centered_eigenspectrum
from PlotSpectra import compare_summaries,  print_summaries, plot_effective_dimensions
"""
E2E: Scalable causal Transformer for a process-language model from per-length distributions q_n.

Setting:
- Alphabet size: k (e.g., 4 or 16)
- Data given as per-length normalized distributions:
    for each length n: sum_{|s|=n} q_n(s) = 1
- Often provided as a flat list: [['0', p], ['00', p], ...]
  We group by length and normalize per length.

Objective:
- Learn p_theta(a_t | prefix) for a stochastic process, i.e. complete transitions:
    sum_a p_theta(a | prefix) = 1
- Targets derived from q_n:
    q(a | u) = q_t(u+a) / q_{t-1}(u)
  (with additive smoothing delta for sparsity/inconsistency)

Model:
- Decoder-only GPT-style causal Transformer
- We represent a prefix by its last hidden state (like the GRU baseline),
  and predict next-symbol distribution.

Outputs:
- trained model + helpers
- scaling runs: Tiny/Small/Base/Large with params + NLL

Requires: torch
"""

import os


def configure_cpu(n_threads: int = 96, n_interop: int = 4):
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)

    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_interop)

    print("torch num threads:", torch.get_num_threads())
    print("torch interop threads:", torch.get_num_interop_threads())


configure_cpu(n_threads=16, n_interop=4)


# =============================================================================
# 1) DATA REFORMAT (same input encoding format)
# =============================================================================
def flat_int_lists_to_by_len(
    flat: List[List[Any]],
    k: int,
    max_len: int,
    renorm_each_len: bool = True,
    tol: float = 1e-6,
) -> Dict[int, List[Tuple[List[int], float]]]:
    """
    Input format like:
      flat = [ [[0], p], [[1], p], [[0,1], p], ... ]
    where sequence is a list[int] and p is float.

    Output:
      data_by_len[n] = [([tokens], p_norm), ...]  with sum_{|s|=n} p_norm = 1 (if renorm_each_len)
    """
    by_len = defaultdict(list)

    for seq, p in flat:
        # seq is list[int], p is float
        if not isinstance(seq, list):
            raise TypeError(f"Expected seq as list[int], got {type(seq)}")
        if not (1 <= len(seq) <= max_len):
            continue
        p = float(p)
        if p < 0:
            raise ValueError(f"Negative probability p={p} for seq={seq}")
        for a in seq:
            a = int(a)
            if not (0 <= a < k):
                raise ValueError(f"Symbol {a} out of range [0,{k-1}] in seq={seq}")

        by_len[len(seq)].append((seq, p))

    out: Dict[int, List[Tuple[List[int], float]]] = {}
    for n, lst in by_len.items():
        total = sum(p for _, p in lst)
        if total <= 0:
            raise ValueError(f"Total mass is zero for length {n}")
        if abs(total - 1.0) > tol:
            if renorm_each_len:
                lst = [(seq, p / total) for seq, p in lst]
            else:
                raise ValueError(f"Length {n} sum={total}, expected 1.0")
        out[n] = lst

    if 1 not in out:
        raise ValueError("Need length-1 distribution present.")
    return out
def flat_strings_to_by_len(
    flat: List[List[Union[str, float]]],
    alphabet: str,
    max_len: int,
    renorm_each_len: bool = True,
    tol: float = 1e-6,
) -> Dict[int, List[Tuple[List[int], float]]]:
    """
    Convert flat [['0103', p], ...] into per-length dict:
        data_by_len[n] = [([token_ids...], prob), ...]

    Each symbol must be ONE CHARACTER from `alphabet`.
    Examples:
      - k=4:  alphabet="0123"
      - k=16: alphabet="0123456789ABCDEF"  (hex, single-character tokens)

    If your raw file is already lexicographically ordered, that's fine (order is irrelevant).
    """
    k = len(alphabet)
    char2id = {ch: i for i, ch in enumerate(alphabet)}
    by_len: Dict[int, List[Tuple[List[int], float]]] = defaultdict(list)

    for s, p in flat:
        if not isinstance(s, str):
            raise TypeError(f"Expected seq as str, got {type(s)}")
        if not (1 <= len(s) <= max_len):
            continue
        p = float(p)
        if p < 0:
            raise ValueError(f"Negative probability p={p} for seq={s}")

        seq = []
        for ch in s:
            if ch not in char2id:
                raise ValueError(f"Symbol '{ch}' not in alphabet '{alphabet}' for seq={s}")
            seq.append(char2id[ch])

        by_len[len(seq)].append((seq, p))

    out: Dict[int, List[Tuple[List[int], float]]] = {}
    for n, lst in by_len.items():
        total = sum(p for _, p in lst)
        if total <= 0:
            raise ValueError(f"Total probability mass is zero for length {n}")
        if abs(total - 1.0) > tol:
            if renorm_each_len:
                lst = [(seq, p / total) for seq, p in lst]
            else:
                raise ValueError(f"Length {n} sum={total}, expected 1.0")
        out[n] = lst

    if 1 not in out:
        raise ValueError("Need length-1 distribution present.")
    return out


def token_lists_to_by_len(
    token_prob: List[Tuple[List[int], float]],
    k: int,
    max_len: int,
    renorm_each_len: bool = True,
    tol: float = 1e-6,
) -> Dict[int, List[Tuple[List[int], float]]]:
    """
    If you already have tokenized sequences: [([0,1,2], p), ...]
    Convert to per-length dict and normalize per length.
    """
    by_len: Dict[int, List[Tuple[List[int], float]]] = defaultdict(list)

    for seq, p in token_prob:
        if not (1 <= len(seq) <= max_len):
            continue
        p = float(p)
        if p < 0:
            raise ValueError(f"Negative probability p={p} for seq={seq}")
        for a in seq:
            if not (0 <= a < k):
                raise ValueError(f"Symbol {a} out of range [0,{k-1}] in seq={seq}")
        by_len[len(seq)].append((seq, p))

    out: Dict[int, List[Tuple[List[int], float]]] = {}
    for n, lst in by_len.items():
        total = sum(p for _, p in lst)
        if total <= 0:
            raise ValueError(f"Total probability mass is zero for length {n}")
        if abs(total - 1.0) > tol:
            if renorm_each_len:
                lst = [(seq, p / total) for seq, p in lst]
            else:
                raise ValueError(f"Length {n} sum={total}, expected 1.0")
        out[n] = lst

    if 1 not in out:
        raise ValueError("Need length-1 distribution present.")
    return out


# =============================================================================
# 2) BUILD PREFIX -> NEXT-SYMBOL TRAINING SET
# =============================================================================

@dataclass
class PrefixExample:
    x: torch.LongTensor        # [BOS] + prefix
    length: int                # true length of x
    target: torch.FloatTensor  # soft target distribution over k symbols
    weight: float              # prefix mass q_{t-1}(prefix)


def _prob_maps(data_by_len: Dict[int, List[Tuple[List[int], float]]]) -> Dict[int, Dict[Tuple[int, ...], float]]:
    pm: Dict[int, Dict[Tuple[int, ...], float]] = {}
    for n, lst in data_by_len.items():
        m: Dict[Tuple[int, ...], float] = {}
        for s, p in lst:
            m[tuple(s)] = float(p)
        pm[n] = m
    return pm


def build_prefix_transition_examples(
    data_by_len: Dict[int, List[Tuple[List[int], float]]],
    k: int,
    max_len: int,
    delta: float = 1e-5,
) -> Tuple[List[PrefixExample], int, int, int]:
    """
    Tokens:
      symbols: 0..k-1
      BOS: k
      PAD: k+1

    Build examples:
      - t=1: input=[BOS], target=q_1(a1)
      - t>=2: for each u of length t-1 from q_{t-1}:
            target[a] = (q_t(u+a) + delta) / (q_{t-1}(u) + delta*k)
            weight = q_{t-1}(u)
    """
    BOS = k
    PAD = k + 1
    vocab_in = k + 2
    pm = _prob_maps(data_by_len)

    if 1 not in pm:
        raise ValueError("Missing length-1 distribution q_1.")

    examples: List[PrefixExample] = []

    # t=1: empty prefix
    q1 = pm[1]
    raw = [0.0] * k
    for (a,), p in q1.items():
        raw[a] += p
    denom = 1.0 + delta * k
    tgt = torch.tensor([(m + delta) / denom for m in raw], dtype=torch.float32)
    x = torch.tensor([BOS], dtype=torch.long)
    examples.append(PrefixExample(x=x, length=x.numel(), target=tgt, weight=1.0))

    # t=2..max_len
    for t in range(2, max_len + 1):
        if (t not in pm) or (t - 1 not in pm):
            continue
        q_t = pm[t]
        q_prev = pm[t - 1]

        for u, q_u in q_prev.items():
            if q_u <= 0:
                continue
            next_raw = [q_t.get(u + (a,), 0.0) for a in range(k)]
            denom = q_u + delta * k
            tgt = torch.tensor([(m + delta) / denom for m in next_raw], dtype=torch.float32)
            x = torch.tensor([BOS] + list(u), dtype=torch.long)
            examples.append(PrefixExample(x=x, length=x.numel(), target=tgt, weight=float(q_u)))

    return examples, BOS, PAD, vocab_in


def oracle_weighted_nll(examples: List[PrefixExample]) -> float:
    """
    Best achievable weighted NLL for these (smoothed) targets:
      sum_w H(target) / sum_w
    """
    num = 0.0
    den = 0.0
    for ex in examples:
        h = float(-(ex.target * torch.log(ex.target + 1e-30)).sum())
        num += ex.weight * h
        den += ex.weight
    return num / max(den, 1e-12)


def collate_batch(batch: List[PrefixExample], pad_token: int, device: torch.device):
    B = len(batch)
    maxT = max(ex.length for ex in batch)
    X = torch.full((B, maxT), pad_token, dtype=torch.long)
    L = torch.tensor([ex.length for ex in batch], dtype=torch.long)
    T = torch.stack([ex.target for ex in batch], dim=0)
    W = torch.tensor([ex.weight for ex in batch], dtype=torch.float32)

    for i, ex in enumerate(batch):
        X[i, :ex.length] = ex.x

    return X.to(device), L.to(device), T.to(device), W.to(device)

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


# =============================================================================
# 4) TRAINING + EVAL + REPRESENTATION EXTRACTION
# =============================================================================

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



def train_transformer_process(
    data_by_len: Dict[int, List[Tuple[List[int], float]]],
    k: int,
    max_len: int = 7,
    delta: float = 1e-5,
    # Transformer config
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    d_ff: Optional[int] = None,
    dropout: float = 0.1,
    # optimization
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    epochs: int = 200,
    batch_size: int = 512,
    grad_clip: float = 1.0,
    seed: int = 0,
    device: Optional[str] = None,
    print_every: int = 25,
) -> TrainedTransformerProcess:
    random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    examples, BOS, PAD, vocab_in = build_prefix_transition_examples(
        data_by_len=data_by_len, k=k, max_len=max_len, delta=delta
    )

    oracle = oracle_weighted_nll(examples)
    max_ctx = max_len+1  # BOS + prefix up to length (max_len-1) => max_len tokens

    model = CausalTransformerNextSymbol(
        vocab_in=vocab_in,
        k=k,
        pad_idx=PAD,
        max_ctx=max_ctx,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
    ).to(dev)


    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    n = len(examples)
    for ep in range(1, epochs + 1):
        idx = torch.randperm(n).tolist()
        total_loss_mass = 0.0
        total_mass = 0.0

        model.train()
        for i in range(0, n, batch_size):
            batch = [examples[j] for j in idx[i:i + batch_size]]
            X, L, T, W = collate_batch(batch, pad_token=PAD, device=dev)

            logits = model(X, L)            # (B,k)
            loss_vec = soft_ce(logits, T)   # (B,)
            loss = (W * loss_vec).sum() / (W.sum() + 1e-12)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            total_loss_mass += float((W * loss_vec).sum().detach().cpu())
            total_mass += float(W.sum().detach().cpu())

        if ep == 1 or (print_every and ep % print_every == 0):
            nll = total_loss_mass / max(total_mass, 1e-12)
            print(f"epoch {ep:4d}  weighted_NLL={nll:.8f}  (ppl={math.exp(nll):.4f})  gap_to_oracle={nll-oracle:+.6f}")

    return TrainedTransformerProcess(model=model, k=k, BOS=BOS, PAD=PAD, max_len=max_len, device=dev)
#------------------------------------------------------------------------------
def build_transformer_tensors(examples, PAD: int):
    """
    Convert PrefixExample list into dense tensors once.

    Returns:
      X_all: (N, maxT)
      L_all: (N,)
      T_all: (N, k)
      W_all: (N,)
    """
    N = len(examples)
    maxT = max(ex.length for ex in examples)
    k = examples[0].target.numel()

    X_all = torch.full((N, maxT), PAD, dtype=torch.long)
    L_all = torch.empty(N, dtype=torch.long)
    T_all = torch.empty((N, k), dtype=torch.float32)
    W_all = torch.empty(N, dtype=torch.float32)

    for i, ex in enumerate(examples):
        X_all[i, :ex.length] = ex.x
        L_all[i] = ex.length
        T_all[i] = ex.target
        W_all[i] = ex.weight

    return X_all, L_all, T_all, W_all
#-----------------------------------------------------------------------------
#------------------------------------------------------------------------------
def train_transformer_process_cpu_fast(
    data_by_len,
    k: int,
    max_len: int = 7,
    delta: float = 1e-5,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    d_ff=None,
    dropout: float = 0.1,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    epochs: int = 200,
    batch_size: int = 4096,          # larger batch for CPU
    grad_clip: float = 1.0,
    seed: int = 0,
    print_every: int = 25,
    use_compile: bool = False,
    use_bfloat16: bool = False,
):
    import random, math
    import torch
    import torch.nn as nn

    random.seed(seed)
    torch.manual_seed(seed)

    

    dev = torch.device("cpu")

    examples, BOS, PAD, vocab_in = build_prefix_transition_examples(
        data_by_len=data_by_len,
        k=k,
        max_len=max_len,
        delta=delta,
    )

    oracle = oracle_weighted_nll(examples)
    max_ctx = max_len + 1

    model = CausalTransformerNextSymbol(
        vocab_in=vocab_in,
        k=k,
        pad_idx=PAD,
        max_ctx=max_ctx,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
    ).to(dev)

    if use_compile:
        try:
            model = torch.compile(model, mode="max-autotune")
            print("[info] torch.compile enabled")
        except Exception as e:
            print("[warning] torch.compile failed:", e)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        foreach=True,
    )

    X_all, L_all, T_all, W_all = build_transformer_tensors(examples, PAD=PAD)

    N = X_all.shape[0]
    print(f"[data] examples={N:,}, maxT={X_all.shape[1]}, batch_size={batch_size}")
    print(
        f"[model] d_model={d_model}, layers={n_layers}, heads={n_heads}, "
        f"d_ff={d_ff or 4*d_model}"
    )
    print(
        f"[sanity] oracle weighted_NLL={oracle:.8f} "
        f"(ppl={math.exp(oracle):.4f})"
    )

    for ep in range(1, epochs + 1):
        perm = torch.randperm(N)

        total_loss_mass = 0.0
        total_mass = 0.0

        model.train()

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]

            X = X_all[idx]
            L = L_all[idx]
            T = T_all[idx]
            W = W_all[idx]

            opt.zero_grad(set_to_none=True)

            if use_bfloat16:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    logits = model(X, L)
                    loss_vec = soft_ce(logits.float(), T)
                    loss = (W * loss_vec).sum() / (W.sum() + 1e-12)
            else:
                logits = model(X, L)
                loss_vec = soft_ce(logits, T)
                loss = (W * loss_vec).sum() / (W.sum() + 1e-12)

            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            total_loss_mass += float((W * loss_vec).sum().detach())
            total_mass += float(W.sum().detach())

        if ep == 1 or (print_every and ep % print_every == 0):
            nll = total_loss_mass / max(total_mass, 1e-12)
            print(
                f"epoch {ep:4d}  weighted_NLL={nll:.8f} "
                f"(ppl={math.exp(nll):.4f})  gap_to_oracle={nll-oracle:+.6f}"
            )

    return TrainedTransformerProcess(
        model=model,
        k=k,
        BOS=BOS,
        PAD=PAD,
        max_len=max_len,
        device=dev,
    )

#------------------------------------------------------------------------------
# =============================================================================
# Save - Restore the model
def save_transformer_process(path, model, model_config, meta=None):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "meta": meta or {},
        },
        path,
    )


def load_transformer_process(
    path,
    model_class,
    device="cpu",
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model = model_class(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint.get("meta", {})
# =============================================================================
# 5) SCALING FAMILY RUNNER (for reviewer-proof baselines)
# =============================================================================

def transformer_scaling_suite(
    data_by_len: Dict[int, List[Tuple[List[int], float]]],
    k: int,
    max_len: int,
    delta: float,
    device: Optional[str] = None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Runs a standard scaling family and returns summary dicts.
    Adjust epochs/batch_size if you want tighter convergence per size.
    """
    configs = [
        dict(name="Tiny",  d_model=128, n_layers=2, n_heads=2, d_ff=512),
        dict(name="Small", d_model=256, n_layers=4, n_heads=4, d_ff=1024),
        dict(name="Base",  d_model=512, n_layers=6, n_heads=8, d_ff=2048),
        dict(name="Large", d_model=768, n_layers=8, n_heads=12, d_ff=3072),
    ]

    results = []
    for cfg in configs:
        print("\n" + "=" * 90)
        print(f"Scaling run: {cfg['name']}")
        trained = train_transformer_process(
            data_by_len=data_by_len,
            k=k,
            max_len=max_len,
            delta=delta,
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            dropout=0.1,
            lr=3e-4,
            weight_decay=1e-2,
            epochs=200,            # bump to 500+ if you want very tight convergence
            batch_size=512,
            grad_clip=1.0,
            seed=seed,
            device=device,
            print_every=50,
        )
        results.append(dict(
            name=cfg["name"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            params=count_params(trained.model),
        ))
    return results




if __name__ == "__main__":
    # IMPORTANT: For a true per-length dataset, all length-2 strings should be present and sum to 1,
    # likewise length-3, ..., length-7. If you provide only a subset, we will still learn, but the
    # implied conditionals may be inconsistent; smoothing delta then matters more.

#-----------------------------------------------------------------------------
# Generative Model for Mid-Price encoded by 4-symbols z encoding
# Data integrated for January 2025 at 10 seconds
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#   Generative Model Paramerters
#------------------------------------------------------------------------------
    

    fPath = 'C:\EXPIMP\Vanio\Projects\ICML-2026\Market Data Preparation\\' 
    fPath = '.\data\\'


    features_list =  ["log_mid_sym",'sigma_W_sym','imbalance_sym',"mid_cross_prev_ask_up_sym", 
                        'micro_price_sym',"trade_ofi_n_sym","vpin_sym", "log_mid_return_fwd_1_sym"]
    features_list =  ["log_mid_sym",'sigma_W_sym',"vpin_sym"]
    
    features =  ['log_mid','ofi_L10_norm_n',"micro_price",'vpin'] #, 'sigma_W']
    
    dates = ['20250303','20250304', '20250305', '20250306', '20250307',
             '20250310','20250311', '20250312', '20250313', '20250314',
             '20250317','20250318', '20250319', '20250320', '20250321',
             '20250324','20250325', '20250326', '20250327', '20250328',
             '20250331','20250401', '20250402' ]
    
    
    symbol= 'AAPL'
    date = '202503' # nonthly aggregated data

    
    frequency = 1 #sec
    freq_units = 'sec'
    
    frequency = 100 #events
    freq_units = 'evn'

    max_seq_len =       6 #max sequence length to be used for training 
    min_seq_prob = 0.000000 #threshold for rare events



    variate = 'bivariate'
    
    for feature in features[1:]:
    
        if variate == 'univariate':
            feature_full = feature
            n_symbols=8
            # SEQ_DISTR_AAPL_bivariate_log_mid-ofi_L10_norm_n_202503
            title = 'SEQ_DISTR_'+symbol+'_'+variate+'_'+feature_full+'_'+date
            infname = fPath+title
            target =  pickle.load(open( infname, "rb") ) 
        elif variate == 'bivariate':
            symbol = 'AAPL'
            n_symbols=16

            predicted =  features[0] #'log_mid' 
            predictor =  feature  
            


            feature_full = predicted +'-'+ predictor
           
            # SEQ_DISTR_AAPL_bivariate_log_mid-ofi_L10_norm_n_202503
            title = 'SEQ_DISTR_'+symbol+'_'+variate+'_'+feature_full+'_'+date
            
            infname = fPath+title
            
            target =  pickle.load(open( infname, "rb") )  
    
        #------------------------------------------------------------------------------
        # Read Target Distribution
        #------------------------------------------------------------------------------
       
        ''' 
        target1=[]
        target0=[]
        for i in range(len(target[0])):
            if target[0][i][1] > min_seq_prob and len(target[0][i][0]) <= max_seq_len:            # 0.00001:
                target1=target1+[target[1][i]]
                target0=target0+[target[0][i]]
    
        target[0] = target0
        target[1] = target1
        '''
        
        target_probs   = []
        
        for i in range(len(target[0])):
            target_probs.append(target[0][i][1])
    
    
    
        
        k = n_symbols             # number of observable symbols  
        maxlen = max_seq_len
        data = flat_int_lists_to_by_len(target[0], k=k, max_len=maxlen, renorm_each_len=True)


    
        n_layers = 4
        
        n_heads=4
        d_ff=1024
        dropout=0.1
        
        hidden_grid = [2048, 4096]
        
        hidden_grid = [4096]
        #hidden_grid = [128, 256, 512, 1024 ]
        
        results_rnn_dim = {}
        print('Transformer model', symbol, date, feature_full )
        for H in hidden_grid:
            print(f"\nTraining Transformer for feature {feature_full} with hidden dimension ={H}")
            d_model = H
        
            #increasing d_model increases parameters a lot
            #increasing n_layers increases params linearly
            #increasing d_ff increases params linearly
            '''
            trained_proc = train_transformer_process(
                data_by_len=data,
                k=k,
                max_len=maxlen,
                delta=1e-4,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                epochs=500,
                batch_size=512,
                device=None,
            )
            '''
            
            trained_proc = train_transformer_process_cpu_fast(
                data_by_len=data,
                k=k,
                max_len=maxlen,
                delta=1e-4,
                d_model=512,
                n_layers=2,
                n_heads=8,
                d_ff=2 * 512,
                dropout=0.1,
                epochs=500,
                batch_size=4096,
                lr=3e-4,
                use_compile=False,
                use_bfloat16=False,
            )
            
        
            n_params = count_params(trained_proc.model)
            print(f"[model] d_model={d_model} layers={n_layers} heads={n_heads} d_ff={d_ff or 4*d_model} "
                  f"dropout={dropout}  params={n_params:,}")
            # print(f"[sanity] oracle weighted_NLL={oracle:.8f}  (perplexity={math.exp(oracle):.4f})")
            # Rep(seq, gen_model) 
            
            sequences = target[1]
            model_distribution=[]
            divergence = 0.0
            for i in range(len(sequences)):
                sequence    = sequences[i]
                target_probability   = target_probs[i]
                seq = [int(c) for c in sequence]
                model_probability = np.exp(trained_proc.logprob_fixed_length(seq, eps_mix=1e-6))
                model_distribution = model_distribution+[model_probability]
                divergence = divergence+(model_probability-target_probability)**2    
        
            plot = True
            if plot:
                plotDistributions(target_probs[:600],model_distribution[:600], sequences[:600],
                                  title+' Cost='+str(divergence), 'Target', 'Model', c1 = 'blue', c2='red' )
            model_results_file = 'TRNSF_RES_'+ symbol+'_'+date+'_'+feature_full
            pickle.dump( [model_distribution, sequences, target_probs,   n_params, d_model], open( model_results_file, "wb") )
            model_file = 'TRNSF_MOD_'+ symbol+'_'+date+'_'+feature_full+'.pt'
            
        #----------------------------------------------------------------------
        # Save model
        save_model = True
        if save_model:
            model_config = {
                "k": k,
                "max_len": maxlen,
                "delta": 1e-4,
                "d_model": 512,
                "n_layers": 2,
                "n_heads": 8,
                "d_ff": 1024,
                "dropout": 0.1,
            }
            
            save_transformer_process(
                model_file ,
                trained_proc,
                model_config,
                meta={"epochs": 500},
            )
        
            restore_model = False
            if restore_model:
                trained_proc, meta = load_transformer_process(
                    "causal_transformer_process.pt",
                    CausalTransformerNextSymbol,
                    device="cpu",
                )
                
                
        #-----------------------------------------------------------------------------
        # Calculate the spectrum of representation
        #-----------------------------------------------------------------------------
        # seq_prob = [ (seq1, p1), (seq2, p2), ... ]
        
            seq_prob = target[0]
        
            print('Spectrum Estimation: #sequences',len(seq_prob) )
        
            seq_prob_filtered = [sp for sp in seq_prob if sp[1] > min_seq_prob]
            res = gram_centered_eigenspectrum(seq_prob_filtered, Rep, trained_proc, return_matrices=True)
            lams = res["eigenvalues"]
            Kc   = res["K_centered"]
        
            #print(lams[:10], lams.sum())  # should sum ~1
            import numpy as np
        
            def rank_for_mass(lams, mass=0.95):
                lams = np.asarray(lams, dtype=float)
                # (optional) ensure descending
                lams = np.sort(lams)[::-1]
                c = np.cumsum(lams)
                r = int(np.searchsorted(c, mass) + 1)  # +1 for 1-based count
                return r, c
            
            r95, cum = rank_for_mass(lams, 0.95)
            print("r95 =", r95, "cum[r95-1] =", cum[r95-1])
            lams_top = lams[:r95]
            # -----------------------------------------------------------------------------
            
            results_rnn_dim[f"TRM-{H}-{feature_full}"] = lams_top
            # Save top spectrum
            save_file_name = 'SPC_TRM_'+ date+'_'+feature + '_'+str(n_symbols)+'ss_'+str(frequency)+freq_units +"_"+str(H)+ '_TRM'
            
            pickle.dump( lams_top, open( save_file_name, "wb") )
            # -----------------------------------------------------------------------------
            save_file_name = 'KCN_TRM_'+ date+'_'+feature + '_'+str(n_symbols)+'ss_'+str(frequency)+freq_units +"_"+str(H)+ '_TRM'
            pickle.dump(Kc , open( save_file_name, "wb") )
            print('Kernel Centered Matrix Shape=',Kc.shape)
        
            
            plotDistribution(lams_top, range(r95), 'TRM Embedding: Eigenvalues '+title,'')
        
        summary_rnn_dim = compare_summaries(
        results_rnn_dim,
        gammas=np.logspace(-4, 0, 9),
        )
    
        print_summaries(summary_rnn_dim, kernel_order=list(results_rnn_dim.keys()))
        plot_effective_dimensions(summary_rnn_dim, kernel_order=list(results_rnn_dim.keys()))

    


    
    
    
    
    
    
