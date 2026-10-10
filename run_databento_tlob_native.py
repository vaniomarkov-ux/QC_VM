from __future__ import annotations

import argparse
import copy
import glob
import os
import random
import re
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset

import constants as cst
from models.tlob import TLOB
from databento_tlob_native_lob import (
    NativeTLOBWindowDataset,
    prepare_native_tlob_lob_only,
    print_summary,
)

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["up", "stationary", "down"]


def seed_everything(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None):
    if requested is None:
        requested = str(cst.DEVICE)

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({requested}) but CUDA is not available.")

    # TLOB uses constants.DEVICE internally during construction.
    cst.DEVICE = str(device)
    return device


def expand_file_patterns(items, name="input"):
    """
    Expand explicit file names and glob patterns.

    Examples:
        AAPL/AAPL_20250303_1000.csv
        AAPL/AAPL_202503*_1000.csv
        AAPL/AAPL_2025030[3-5]_1000.csv

    Patterns are expanded inside Python, so quoted patterns work
    consistently across shells. Results are sorted and deduplicated.
    """
    expanded = []

    for item in items:
        item = os.path.expanduser(item)
        matches = glob.glob(item)

        if not matches:
            raise FileNotFoundError(
                f"No files matched {name} input: {item!r}"
            )

        expanded.extend(matches)

    expanded = sorted(dict.fromkeys(str(Path(p)) for p in expanded))

    if not expanded:
        raise FileNotFoundError(f"No files found for {name}.")

    return expanded

def day_name(path):
    stem = Path(path).stem
    m = re.search(r"(20\d{6})", stem)
    return m.group(1) if m else stem


def evenly_spaced_subset(dataset, max_n: int):
    n = len(dataset)
    if max_n <= 0 or max_n >= n:
        return dataset

    idx = np.linspace(0, n - 1, num=max_n).round().astype(np.int64)
    return Subset(dataset, idx.tolist())


def make_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def metrics_from_predictions(y_true, y_pred, loss=None):
    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average=None,
        zero_division=0,
    )

    return {
        "n": len(y_true),
        "loss": loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        ),
        "per_class_f1": per_class_f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_n = 0
    ys, preds = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        n = y.size(0)
        total_loss += loss.item() * n
        total_n += n

        ys.append(y.cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())

    if total_n == 0:
        raise RuntimeError("Evaluation loader is empty.")

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)

    return metrics_from_predictions(
        y_true,
        y_pred,
        loss=total_loss / total_n,
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        n = y.size(0)
        total_loss += loss.item() * n
        total_n += n

    if total_n == 0:
        raise RuntimeError("Training loader is empty.")

    return total_loss / total_n


def pool_results(results):
    y_true = np.concatenate([m["y_true"] for m in results.values()])
    y_pred = np.concatenate([m["y_pred"] for m in results.values()])

    total_n = sum(m["n"] for m in results.values())
    pooled_loss = sum(m["loss"] * m["n"] for m in results.values()) / total_n

    return metrics_from_predictions(
        y_true,
        y_pred,
        loss=pooled_loss,
    )


def daily_summary(results):
    acc = np.array([m["accuracy"] for m in results.values()], dtype=float)
    f1 = np.array([m["macro_f1"] for m in results.values()], dtype=float)

    return {
        "accuracy_mean": acc.mean(),
        "accuracy_std": acc.std(ddof=0),
        "macro_f1_mean": f1.mean(),
        "macro_f1_std": f1.std(ddof=0),
    }


def metric_line(prefix, m):
    f = m["per_class_f1"]
    return (
        f"{prefix:<20s} | "
        f"N={m['n']:5d} | "
        f"loss={m['loss']:.6f} | "
        f"acc={m['accuracy']:.4f} | "
        f"macro_f1={m['macro_f1']:.4f} | "
        f"F1(up/stat/down)={f[0]:.4f}/{f[1]:.4f}/{f[2]:.4f}"
    )


def print_full_report(title, m):
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"N={m['n']} | loss={m['loss']:.6f} | "
        f"accuracy={m['accuracy']:.4f} | "
        f"macro_f1={m['macro_f1']:.4f}"
    )

    print(
        classification_report(
            m["y_true"],
            m["y_pred"],
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("confusion matrix [up, stationary, down]:")
    print(confusion_matrix(m["y_true"], m["y_pred"], labels=CLASS_LABELS))


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--train",
        nargs="+",
        required=True,
        help="training CSV files and/or glob patterns",
    )
    p.add_argument(
        "--val",
        nargs="*",
        default=[],
        help=(
            "optional validation CSV files and/or glob patterns; "
            "omit --val for training-only mode"
        ),
    )

    p.add_argument("--seq-size", type=int, default=128)
    p.add_argument(
        "--horizon",
        type=int,
        default=10,
        choices=[10, 20, 50, 100],
        help="native TLOB prediction horizon",
    )
    p.add_argument(
        "--len-smooth",
        type=int,
        default=10,
        help="native TLOB smoothing length",
    )
    p.add_argument(
        "--quantity",
        type=int,
        default=500,
        help="TLOB-style cumulative-size sampling quantity",
    )
    p.add_argument(
        "--alpha-policy",
        choices=["official_split", "train"],
        default="official_split",
        help=(
            "official_split: train and validation each derive their own alpha; "
            "train: validation reuses training alpha"
        ),
    )
    p.add_argument(
        "--keep-actions",
        nargs="*",
        default=None,
        help=(
            "optional Databento action filter applied before quantity sampling, "
            'e.g. --keep-actions A C T; omit to use all MBP-10 actions'
        ),
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run full train/validation evaluation every N epochs; epoch 1 and final epoch are always evaluated",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=40)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument(
        "--device",
        default=None,
        help="cpu, cuda, cuda:0, etc.; default uses constants.DEVICE",
    )

    p.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="0 = all training windows",
    )
    p.add_argument(
        "--max-val",
        type=int,
        default=0,
        help="0 = all windows; otherwise limit EACH validation day",
    )

    p.add_argument("--save", default="tlob_databento_smoke.pt")

    args = p.parse_args()

    # Expand explicit paths and wildcard patterns inside Python.
    args.train = expand_file_patterns(args.train, name="training")
    args.val = (
        expand_file_patterns(args.val, name="validation")
        if args.val
        else []
    )

    # Overlap is allowed, but warn clearly because validation metrics on
    # overlapping files are not out-of-sample.
    train_abs = {str(Path(p).resolve()) for p in args.train}
    val_abs = {str(Path(p).resolve()) for p in args.val}
    overlap = sorted(train_abs & val_abs)

    if overlap:
        print(
            "\nWARNING: training and validation inputs overlap. "
            "Validation metrics for these files are IN-SAMPLE, not OOS."
        )
        print("Overlapping file(s):")
        for path in overlap:
            print("  ", path)

    print("\nTraining files:")
    for path in args.train:
        print("  ", path)

    if args.val:
        print("\nValidation files:")
        for path in args.val:
            print("  ", path)
    else:
        print("\nNo validation files supplied: TRAINING-ONLY mode.")
        print(
            "The final epoch will be saved because no validation set is "
            "available for checkpoint selection."
        )

    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"device = {device}")

    # -----------------------------
    # Native-style TLOB data preparation
    # -----------------------------
    prepared = prepare_native_tlob_lob_only(
        train_files=args.train,
        val_files=args.val if args.val else None,
        seq_size=args.seq_size,
        len_smooth=args.len_smooth,
        horizon=args.horizon,
        quantity=args.quantity,
        keep_actions=args.keep_actions,
        alpha_policy=args.alpha_policy,
    )

    print_summary(prepared)

    train_full = prepared["train_dataset"]
    stats = prepared["normalization"]

    # Keep validation days separate for reporting, while preserving the
    # single validation alpha generated by the adapter for the whole split.
    val_full_by_day = OrderedDict()

    if args.val:
        for path, day in zip(args.val, prepared["val_days"]):
            name = day_name(path)

            if name in val_full_by_day:
                k = 2
                while f"{name}_{k}" in val_full_by_day:
                    k += 1
                name = f"{name}_{k}"

            val_full_by_day[name] = NativeTLOBWindowDataset(
                [day],
                seq_size=args.seq_size,
                len_smooth=args.len_smooth,
                horizon=args.horizon,
            )

    print(f"train windows available = {len(train_full):,}")
    for day, ds in val_full_by_day.items():
        print(f"val {day} windows available = {len(ds):,}")

    print("normalization stats:", stats)
    print(f"train alpha = {prepared['train_alpha']:.10g}")
    if prepared["val_alpha"] is not None:
        print(f"validation alpha = {prepared['val_alpha']:.10g}")

    train_ds = evenly_spaced_subset(train_full, args.max_train)

    val_by_day = OrderedDict(
        (day, evenly_spaced_subset(ds, args.max_val))
        for day, ds in val_full_by_day.items()
    )

    print(f"train windows used = {len(train_ds):,}")
    for day, ds in val_by_day.items():
        print(f"val {day} windows used = {len(ds):,}")

    train_loader = make_loader(
        train_ds,
        args.batch_size,
        True,
        args.num_workers,
        device,
    )

    train_eval_loader = make_loader(
        train_ds,
        args.batch_size,
        False,
        args.num_workers,
        device,
    )

    val_loaders = OrderedDict(
        (
            day,
            make_loader(
                ds,
                args.batch_size,
                False,
                args.num_workers,
                device,
            ),
        )
        for day, ds in val_by_day.items()
    )

    # -----------------------------
    # Official TLOB
    # -----------------------------
    model = TLOB(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seq_size=args.seq_size,
        num_features=40,
        num_heads=args.num_heads,
        is_sin_emb=True,
        dataset_type="DATABENTO",
    ).to(device)

    print(
        "parameters = "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        eps=1e-8,
    )

    has_validation = len(val_loaders) > 0

    best_val_loss = float("inf")
    best_epoch = None
    best_state = None

    # -----------------------------
    # Train
    # -----------------------------
    if args.eval_every < 1:
        raise ValueError("--eval-every must be >= 1")

    for epoch in range(1, args.epochs + 1):
        update_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        do_eval = (
            epoch == 1
            or epoch % args.eval_every == 0
            or epoch == args.epochs
        )

        if not do_eval:
            print(f"epoch {epoch:02d} | update_loss={update_loss:.6f}")

            # In training-only mode we keep the latest state so the final
            # epoch is saved even when no evaluation is requested here.
            if not has_validation:
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            continue

        # Clean in-sample performance, recomputed AFTER the epoch in eval mode.
        train_metrics = evaluate(
            model,
            train_eval_loader,
            criterion,
            device,
        )

        print(f"\nepoch {epoch:02d} | update_loss={update_loss:.6f}")
        print(metric_line("TRAIN (in-sample)", train_metrics))

        if has_validation:
            # Each validation day separately.
            val_results = OrderedDict(
                (
                    day,
                    evaluate(model, loader, criterion, device),
                )
                for day, loader in val_loaders.items()
            )

            pooled = pool_results(val_results)
            daily = daily_summary(val_results)

            for day, m in val_results.items():
                print(metric_line(f"VAL {day}", m))

            print(metric_line("VAL POOLED", pooled))
            print(
                "VAL DAILY            | "
                f"acc={daily['accuracy_mean']:.4f} +/- {daily['accuracy_std']:.4f} | "
                f"macro_f1={daily['macro_f1_mean']:.4f} +/- "
                f"{daily['macro_f1_std']:.4f}"
            )

            if pooled["loss"] < best_val_loss:
                best_val_loss = pooled["loss"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        else:
            # No validation set: the latest epoch is the checkpoint.
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No model state recorded.")

    # -----------------------------
    # Best-checkpoint reports
    # -----------------------------
    model.load_state_dict(best_state)

    final_train = evaluate(
        model,
        train_eval_loader,
        criterion,
        device,
    )

    print("\n" + "=" * 80)

    if has_validation:
        final_val = OrderedDict(
            (
                day,
                evaluate(model, loader, criterion, device),
            )
            for day, loader in val_loaders.items()
        )
        final_pooled = pool_results(final_val)
        final_daily = daily_summary(final_val)

        print(
            f"BEST EPOCH = {best_epoch} | "
            f"pooled validation loss = {best_val_loss:.6f}"
        )
    else:
        final_val = OrderedDict()
        final_pooled = None
        final_daily = None
        best_val_loss = None
        print(
            f"FINAL EPOCH = {best_epoch} | TRAINING-ONLY mode "
            "(no validation checkpoint selection)"
        )

    print("=" * 80)

    train_title = (
        "TRAIN (IN-SAMPLE) -- BEST CHECKPOINT"
        if has_validation
        else "TRAIN (IN-SAMPLE) -- FINAL EPOCH"
    )
    print_full_report(train_title, final_train)

    for day, m in final_val.items():
        print_full_report(
            f"VALIDATION {day} -- BEST CHECKPOINT",
            m,
        )

    if len(final_val) > 1:
        print_full_report(
            "VALIDATION POOLED -- BEST CHECKPOINT",
            final_pooled,
        )

        print(
            "\nVALIDATION DAILY SUMMARY\n"
            "------------------------\n"
            f"accuracy = {final_daily['accuracy_mean']:.4f} +/- "
            f"{final_daily['accuracy_std']:.4f}\n"
            f"macro-F1 = {final_daily['macro_f1_mean']:.4f} +/- "
            f"{final_daily['macro_f1_std']:.4f}"
        )

    # -----------------------------
    # Save
    # -----------------------------
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "seq_size": args.seq_size,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "num_features": 40,
            "normalization": stats.__dict__,
            "native_tlob": {
                "horizon": args.horizon,
                "len_smooth": args.len_smooth,
                "quantity": args.quantity,
                "alpha_policy": args.alpha_policy,
                "keep_actions": args.keep_actions,
                "train_alpha": prepared["train_alpha"],
                "val_alpha": prepared["val_alpha"],
            },
            "label_mapping": {
                "0": "up",
                "1": "stationary",
                "2": "down",
            },
            "train_files": [str(x) for x in args.train],
            "validation_files": [str(x) for x in args.val],
            "validation_days": list(final_val.keys()),
            "training_only": not has_validation,
            "eval_every": args.eval_every,
        },
        args.save,
    )

    print(f"\nsaved: {args.save}")


if __name__ == "__main__":
    main()

'''
python run_databento_tlob.py 
--train "data/databento/AAPL/xnas-itch-202503*.mbp-10.dbn.zst"    
--val "data/databento/AAPL/xnas-itch-2025040[123].mbp-10.dbn.zst" 
--eval-every 20 --epochs 200 
--batch-size 512 
--save "tlob_databento_AAPL_March.pt"
--horizon 50

--device cuda:0

--horizon 10   # default
--len-smooth 10
--quantity 500                        # volume resampling

TLOB uses a fixed smoothing length L=10

horizons h\in\{10,20,50,100\}
measured in the quantity-sampled series, not directly in raw events. 
These values are hard-coded as LEN_SMOOTH = 10 
and LOBSTER_HORIZONS = [10,20,50,100]

Comparison to event-level resampling

the current 100-event sampling + ca_4 task is roughly 
a 400-event horizon, which is quite close in raw-event scale to TLOB's \(h=50\) 
task under the observed quantity sampling.






'''
