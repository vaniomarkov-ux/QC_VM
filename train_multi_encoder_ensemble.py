# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:51:05 2026

@author: vanio
"""

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
    checkpoint_every=None,
    checkpoint_path_prefix=None,
    checkpoint_meta=None,
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
    device = torch.device(device)
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
        pin_memory=(device.type == "cuda"),
        
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
    
    meta_epoch = dict(checkpoint_meta or {})
    # Training loop
    for ep in range(1, epochs + 1):
        ensemble_model.train()
        total_enc_loss = 0.0
        total_pred_loss = 0.0
        total_loss = 0.0
        n_seen = 0
        
        for batch_data in dataloader:
            *seq_pads, target_dist, global_weight = batch_data
        
            seq_pads = [
                sp.to(device, non_blocking=True)
                for sp in seq_pads
            ]
        
            target_dist = target_dist.to(device, non_blocking=True)
            global_weight = global_weight.to(device, non_blocking=True)
        
            batch_n = target_dist.size(0)
        
            global_weight_normalized = (
                global_weight / global_weight.sum().clamp_min(1e-12)
            )
        
            # Correct: pass all encoder-specific sequence batches.
            probs, traces_list = ensemble_model(seq_pads)
        
            # ===== ENCODER LOSS =====
            enc_loss = torch.tensor(0.0, device=device)
        
            if lambda_enc > 0:
                for traces in traces_list:
                    p_enc = traces.clamp_min(1e-12)
                    enc_loss = enc_loss - (
                        global_weight_normalized * torch.log(p_enc)
                    ).sum()
        
                enc_loss = enc_loss / len(traces_list)
        
            # ===== PREDICTION LOSS =====
            eps = 1e-12
        
            probs = probs.clamp_min(eps)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        
            target_dist = target_dist.clamp_min(eps)
            target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp_min(eps)
        
            if prediction_loss == "ce":
                pred_loss_per_sample = -(
                    target_dist * torch.log(probs)
                ).sum(dim=-1)
        
            elif prediction_loss == "kl":
                pred_loss_per_sample = (
                    target_dist * (torch.log(target_dist) - torch.log(probs))
                ).sum(dim=-1)
        
            elif prediction_loss == "js":
                m = 0.5 * (target_dist + probs)
                m = m.clamp_min(eps)
        
                pred_loss_per_sample = (
                    0.5 * (
                        target_dist * (torch.log(target_dist) - torch.log(m))
                    ).sum(dim=-1)
                    +
                    0.5 * (
                        probs * (torch.log(probs) - torch.log(m))
                    ).sum(dim=-1)
                )
        
            elif prediction_loss == "mse":
                pred_loss_per_sample = (
                    (target_dist - probs) ** 2
                ).sum(dim=-1)
        
            else:
                raise ValueError(f"Unknown prediction_loss: {prediction_loss}")
        
            pred_loss = (
                global_weight_normalized * pred_loss_per_sample
            ).sum()
        
            loss = lambda_enc * enc_loss + lambda_pred * pred_loss
        
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {ep}: {loss.item()}"
                )
        
            if not loss.requires_grad:
                raise RuntimeError(
                    "Loss does not require grad. "
                    "Check freezing and accidental detach/no_grad."
                )
        
            opt.zero_grad(set_to_none=True)
            loss.backward()
        
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                1.0,
            )
        
            opt.step()
        
            total_enc_loss += float(enc_loss.detach().cpu()) * batch_n
            total_pred_loss += float(pred_loss.detach().cpu()) * batch_n
            total_loss += float(loss.detach().cpu()) * batch_n
            n_seen += batch_n
            
            
        
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
        
        # ===== CHECKPOINT =====
        if checkpoint_every is not None and checkpoint_path_prefix is not None:
            if ep % checkpoint_every == 0:
                ckpt_path = f"{checkpoint_path_prefix}_epoch_{ep:04d}.pt"
        
               
                meta_epoch.update({
                    "epoch": ep,
                    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "lr": lr,
                    "batch_size": batch_size,
                    "prediction_loss": prediction_loss,
                    "lambda_enc": lambda_enc,
                    "lambda_pred": lambda_pred,
                    "freeze_encoders": freeze_encoders,
                    "freeze_decoder": freeze_decoder,
                    "avg_enc_loss": avg_enc_loss,
                    "avg_pred_loss": avg_pred_loss,
                    "avg_total_loss": avg_total_loss,
                })
        
                save_ensemble_model(
                    ckpt_path,
                    ensemble_model,
                    meta=meta_epoch,
                )
        
                print(f"Saved checkpoint: {ckpt_path}")

    return ensemble_model,meta_epoch
