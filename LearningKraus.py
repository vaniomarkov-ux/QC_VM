# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:20:38 2026

@author: vanio
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from plot_distributions import plotDistribution, plotDistributions
import pickle
from typing import Literal, Optional

torch.set_num_threads(40)
torch.set_num_interop_threads(1)


# ---------------------------
# Dataset
# ---------------------------
PAD = -1  # must be outside symbol range {0,...,m-1}

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

def sequence_prob_batch(self, seq_pad):
    """
    seq_pad: LongTensor [B, T] with symbols in {0..m-1} or PAD=-1
    returns: FloatTensor [B]
    """
    device = seq_pad.device
    B, T = seq_pad.shape
    d = self.d

    K = self.kraus_operators()       # [m, d, d]
    rho0 = self._make_rho0(device)   # [d, d]
    rho = rho0.unsqueeze(0).expand(B, d, d).clone()  # [B,d,d]

    for t in range(T):
        sym = seq_pad[:, t]                 # [B]
        active = (sym != PAD)               # [B]
        if not torch.any(active):
            break

        sym_a = sym[active]                 # [B_active]
        rho_a = rho[active]                 # [B_active,d,d]

        Kt = K.index_select(0, sym_a)       # [B_active,d,d]
        rho_a = torch.bmm(torch.bmm(Kt, rho_a), Kt.conj().transpose(1, 2))

        rho[active] = rho_a

    prob = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1))
    return torch.clamp(prob, min=0.0)

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
# Training loop
# ---------------------------
# ---------------------------
# Training loop
# ---------------------------
def train_old(
    sequences, emp_probs,
    m: int, n_qubits: int,
    batch_size = 4*512,
    lr=1e-3,
    epochs=50,
    learn_rho0=True,
    model = None,
    num_workers=0,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    d = 2 ** n_qubits
    
    ds = SeqDataset(sequences, emp_probs)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                    collate_fn=collate_pad, pin_memory=True)
    if model is None:
        model = KrausInstrument(m=m, d=d, learn_rho0=learn_rho0).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        n = 0

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
            n += seq_pad.size(0)

        print(f"epoch {ep:3d} | loss {total/n:.6e}")
    return model
#-----------------------------------------------------------------------------
def make_optimizer(optimizer_name, params, lr, weight_decay=0.0):
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")
#-----------------------------------------------------------------------------
LossKind = Literal["mse_prob", "nll_seq"]

def compute_loss_from_seq_probs(
    p_emp: torch.Tensor,          # (B,) empirical weights/probabilities
    p_mdl: torch.Tensor,          # (B,) model sequence probabilities
    loss_kind: LossKind = "nll_seq",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Returns a scalar loss.

    - mse_prob:  mean( p_emp * (p_emp - p_mdl)^2 )
    - nll_seq:   weighted mean( -log p_mdl ) using weights p_emp
                (this matches the "distributional" NLL objective)
    """
    p_emp = p_emp.float()
    p_mdl = p_mdl.float()

    if loss_kind == "mse_prob":
        return torch.mean(p_emp * (p_emp - p_mdl) ** 2)

    if loss_kind == "nll_seq":
        # weights define the target distribution over the sampled sequences
        w = torch.clamp(p_emp, min=0.0)
        wsum = w.sum().clamp_min(eps)
        p_mdl = torch.clamp(p_mdl, min=eps)
        return (w * (-torch.log(p_mdl))).sum() / wsum

    raise ValueError(f"Unknown loss_kind: {loss_kind}")
#-----------------------------------------------------------------------------
def apply_length_mixture_weights(
    p_emp: torch.Tensor,      # (B,) per-length probs
    lens: torch.Tensor,       # (B,) lengths
    max_len: int,
    mixture: Literal["uniform", "geometric", "none"] = "uniform",
    alpha: float = 0.95,      # for geometric
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Returns reweighted p_emp suitable for pooled centering / NLL:
      p = pi_len * p_len(s), where sum_len pi_len = 1.

    - uniform:  pi_len = 1/max_len over lengths 1..max_len
    - geometric: pi_len ∝ alpha^len
    - none: returns p_emp unchanged
    """
    if mixture == "none":
        return p_emp

    lens = lens.long().clamp(min=1, max=max_len)
    device = p_emp.device

    if mixture == "uniform":
        pi = torch.full((max_len + 1,), 1.0 / max_len, device=device)  # pi[0] unused

    elif mixture == "geometric":
        w = torch.tensor([0.0] + [alpha ** l for l in range(1, max_len + 1)], device=device)
        pi = w / (w.sum().clamp_min(eps))

    else:
        raise ValueError("mixture must be 'uniform', 'geometric', or 'none'.")

    return p_emp * pi[lens]


#------------------------------------------------------------------------------
def train(
    sequences, emp_probs, max_len, 
    m: int, n_qubits: int,
    batch_size=4*512,
    lr=1e-3,
    epochs=50,
    learn_rho0=True,
    model=None,
    num_workers=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    optimizer_name="adam",
    loss_kind: LossKind = "nll_seq",        # <-- switch here
    length_mixture: Literal["uniform","geometric","none"] = "uniform",
    alpha: float = 0.95,
):
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
            #new
            lens = lens.to(device, non_blocking=True)
            #
            p_emp = p_emp.to(device, non_blocking=True)

            
            # new
            # optional: fix per-length normalization into a single pooled weighting
            p_w = apply_length_mixture_weights(
                p_emp=p_emp, lens=lens, max_len=max_len,
                mixture=length_mixture, alpha=alpha
            )
            #
            
            p_mdl = model.sequence_prob_batch(seq_pad)
            
            
            
            
            #replaced
            # loss = torch.mean(p_emp * (p_emp - p_mdl) ** 2)
            # by
            loss = compute_loss_from_seq_probs(
                p_emp=p_emp, p_mdl=p_mdl, loss_kind=loss_kind
                # or 
                # p_emp=p_w, p_mdl=p_mdl, loss_kind=loss_kind               
            )
            #
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            #replaced 
            # total += loss.item() * seq_pad.size(0)
            # by
            total += float(loss.detach().cpu()) * seq_pad.size(0)
            #
            n_seen += seq_pad.size(0)

        print(f"epoch {ep:3d} | loss {total / max(n_seen,1):.6e}")

    return model
#-----------------------------------------------------------------------------
def save_model_weights(path, model, meta=None):
    payload = {
        "model_state": model.state_dict(),
        "meta": meta or {},
    }
    torch.save(payload, path)
#------------------------------------------------------------------------------
def load_model_weights(path, m, n_qubits, learn_rho0=True, device="cpu"):
    d = 2 ** n_qubits
    model = KrausInstrument(m=m, d=d, learn_rho0=learn_rho0).to(device)

    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state"], strict=True)

    meta = payload.get("meta", {})
    return model, meta
#------------------------------------------------------------------------------
#torch.set_num_threads(4)          # physical cores
#torch.set_num_interop_threads(1)  # usually best on CPU

#-----------------------------------------------------------------------------
fPath = 'C:\EXPIMP\Vanio\Projects\ICML-2026\Market Data Preparation\\' 
fPath = ''
fPath = 'C:\Vanio\Projects\ICML-2026\RepresentationStability\Data\\'
fPath = 'C:\Vanio\Projects\ICML-2026\Learning\Data\\'

dates = ['20250303','20250304', '20250305', '20250306', '20250307',
         '20250310','20250311', '20250312', '20250313', '20250314',
         '20250317','20250318', '20250319', '20250320', '20250321',
         '20250324','20250325', '20250326', '20250327', '20250328',
         '20250331','20250401', '20250402' ]

dates_march = ['20250303','20250304','20250305','20250306','20250307','20250310','20250311','20250312','20250313','20250314',
               '20250317','20250318','20250319','20250320','20250321','20250324','20250325','20250326','20250327','20250328','20250331']

features_list =  ["log_mid_sym",'sigma_W_sym',"vpin_sym",'imbalance_sym',"mid_cross_prev_ask_up_sym", 
                    'micro_price_sym',"trade_ofi_n_sym", "log_mid_return_fwd_1_sym"] 
features_list =  ["log_mid",'sigma_W',"vpin",'imbalance',"mid_cross_prev_ask_up", 
                    'micro_price',"trade_ofi_n", "log_mid_return_fwd_1"]

univariate = True
univariate = False
symbol= 'AAPL'
symbol= 'INTC'
symbol= 'NVDA'
date=dates[0]
dates_march=['202503']

data_format_1 = True

for date in dates_march:
    if univariate:
    
        feature = features_list[2]
        n_symbols=8
        
        frequency = 1 #sec
        freq_units = 'sec'
        
        frequency = 100 #events
        freq_units = 'evn'
        
        max_seq_len =       6 #max sequence length to be used for training 
        min_seq_prob = 0.000000 #threshold for rare events
        
        m = 8               # number of observable symbols  
        n_qubits = 5        # size of system register
        
        # TD_AAPL_20250303_log_mid_sym_8ss_1sec
        if  data_format_1: 
            title = 'TD_'+symbol+'_'+date+'_'+feature + '_'+str(n_symbols)+'ss_'+str(frequency)+freq_units
        else:
            title = 'SQ_PRB_WT_'+symbol+'_'+date
    
        infname = fPath+title
        
       
        
        if data_format_1:
            distrs_samples =  pickle.load(open( infname, "rb") )
        else:
            sequences, emp_probs, weights =  pickle.load(open( infname, "rb") )
    else:
        n_symbols=16
        frequency = 1 #sec
        freq_units = 'sec'
       
     
        frequency = 100 #events
        freq_units = 'evn'
        
        max_seq_len =       6    # max sequence length to be used for training 
        min_seq_prob = 0.000000   # threshold for rare events
        
        m = 16            # number of observable symbols  
        n_qubits = 5     # size of system register
       
        predicted =  features_list[7]  # "log_mid_return_fwd_1_sym"
        predicted =  features_list[6]  # "vpin_sym"
        
        predicted =  features_list[0]  # "log_mid_sym"
        predictor =  features_list[1]  # 'sigma_W_sym'
        
        
        predicted =  features_list[0]  # "log_mid_sym"
        predictor =  features_list[1]  # 'sigma_W_sym'

        
        title = 'SQ_PRB_AAPL_20250331_log_mid_sigma_W'
        # TD_AAPL_20250303_log_mid_sym_8ss_1sec
        title = "TD_AAPL_20250303_imbalance_sym-log_mid_sum_ratio_1_sym_4ss_1sec"
        title = "SQ_PRB_"+symbol+"_"+date+"_"+predicted+"_"+predictor
        title = "SEQ_DISTR_AAPL_bivariate_log_mid-sigma_W_202503"
        title = "SQ_PRB_INTC_bivariate_log_mid-sigma_W_202503"
        title = "SQ_PRB_NVDA_bivariate_log_mid-sigma_W_202503"
        
        infname = fPath+title
        
        distrs_samples =  pickle.load(open( infname, "rb") ) 
        sequences = distrs_samples[1]
        emp_probs = distrs_samples[0]
    if data_format_1:
        sequences_all = distrs_samples[1]
        emp_probs_all = [i[1] for i in distrs_samples[0]]
        
        sequences = []
        emp_probs = []
        
        for i in range(len(sequences_all)):
            if len(sequences_all[i])<=max_seq_len and emp_probs_all[i] > min_seq_prob :
                sequences.append(sequences_all[i])
                emp_probs.append(emp_probs_all[i])
    
    print('Number of examples ', len(sequences))
    
    ds = SeqDataset(sequences, emp_probs)
    
    
    continue_optimization = False
    if continue_optimization:
        save_file_name = 'MOD'+title[8:]+'_'+str(n_qubits)+'q'
        model_file_name = "WGHTS_"+save_file_name+'.pt'
    
        modelM, meta = load_model_weights(model_file_name, m, n_qubits, learn_rho0=True, device="cpu")
        print(meta)
    else:
        modelM = None
    
    model = train(ds.sequences, ds.emp_probs, max_seq_len,
        m, n_qubits,
        batch_size= 6*512, #4*512,
        lr=1e-3,
        epochs=2000,
        learn_rho0=True,
        model=modelM,
        num_workers=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        optimizer_name="adam", loss_kind="nll_seq")
        #                    , loss_kind="mse_prob")
    
    
    
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
    
    
    
    
    p_model = predict_probs(model, sequences, batch_size=2*1024)
    
    total_loss = 0
    
    for i in range(len(p_model)):
        total_loss = total_loss+emp_probs[i]*(emp_probs[i]-p_model[i])**2
    
    #Kseq, rho0, p = model.path_operator(sequences_all[-1], return_prob=True, device="cpu")
    
    plotDistributions(emp_probs[:200],p_model[:200], sequences[:200],
                      title[8:]+' Cost='+str(total_loss), 'Target', 'Model', c1 = 'blue', c2='red' )
    
    print('Completed:', total_loss)
    print(title+'_'+str(n_qubits)+'q')
    save_file_name = 'QMOD_'+symbol+"_"+date+"_"+predicted+"_"+predictor+'_'+str(n_qubits)+'q'
    model_file_name = "WGHTS_"+save_file_name+'.pt'
    
    pickle.dump( [model, sequences, emp_probs ], open( save_file_name, "wb") )
    
    save_model_weights(
        model_file_name,
        model,
        meta={"m": m, "n_qubits": n_qubits, "d": 2**n_qubits, "learn_rho0": True}
    )            
    print('Dumped', save_file_name, model_file_name)
                
                
    
