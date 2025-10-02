import argparse
import json
import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from sklearn.metrics import f1_score

from src.utils import set_seed, get_device
from src.dataset import SeqDataset
from models.lstm import LSTMClassifier


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)  # X:(B,L,F), Y:(B,H)


def main(cfg):
    # ----- setup -----
    set_seed(int(cfg["train"]["seed"]))
    device = get_device()
    print("Device:", device)

    # ----- data -----
    train_df = pd.read_parquet(cfg["data"]["processed_train"])
    valid_df = pd.read_parquet(cfg["data"]["processed_valid"])

    lookback = int(cfg["features"]["lookback_days"])
    H = int(cfg["features"].get("target_horizons", 1))
    target_cols = [f"heatwave_h{h}" for h in range(1, H + 1)]
    feature_cols = [c for c in train_df.columns if c not in target_cols]
    input_size = len(feature_cols)
    out_dim = H

    train_ds = SeqDataset(train_df, feature_cols, target_cols, lookback)
    valid_ds = SeqDataset(valid_df, feature_cols, target_cols, lookback)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    )

    # ----- model -----
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        out_dim=out_dim,
    ).to(device)

    # ----- loss (per-horizon pos_weight supported) -----
    pw = cfg["train"].get("pos_weight_per_horizon", [cfg["train"]["class_weight_positive"]] * H)
    pos_w = torch.tensor([float(x) for x in pw], dtype=torch.float32, device=device)  # (H,)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # ----- optimizer + lr scheduler -----
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # ----- early stopping controls -----
    monitor = cfg["train"].get("monitor", "val_loss")  # "val_loss" or "val_f1"
    min_delta = float(cfg["train"].get("early_stopping_min_delta", 0.0))
    patience_limit = int(cfg["train"]["early_stopping_patience"])
    epochs = int(cfg["train"]["epochs"])
    ckpt_path = cfg["train"]["ckpt_path"]

    best_metric = None
    patience = 0

    # ----- training history (for plots) -----
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}

    # ----- training loop -----
    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - train"):
            xb, yb = xb.to(device), yb.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)                    # (B,H)
            loss = criterion(logits, yb)          # scalar
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in tqdm(valid_loader, desc="valid"):
                xb, yb = xb.to(device), yb.float().to(device)
                logits = model(xb)                # (B,H)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                ps.append(torch.sigmoid(logits).cpu().numpy())
                ys.append(yb.cpu().numpy())
        val_loss /= len(valid_loader.dataset)

        # macro-F1 across horizons at 0.5 (only if monitoring F1)
        val_f1 = None
        if monitor == "val_f1":
            p = np.concatenate(ps)   # (N,H)
            y = np.concatenate(ys)   # (N,H)
            f1s = []
            for h in range(p.shape[1]):
                f1s.append(
                    f1_score(y[:, h].astype(int), (p[:, h] >= 0.5).astype(int))
                )
            val_f1 = float(np.mean(f1s))

        # choose metric & compare with best
        if monitor == "val_loss":
            current_metric = val_loss
            improved = (best_metric is None) or (best_metric - current_metric > min_delta)
            sched_value = val_loss
        elif monitor == "val_f1":
            current_metric = val_f1
            improved = (best_metric is None) or (current_metric - best_metric > min_delta)
            sched_value = val_loss  # keep scheduler tied to loss for stability
        else:
            raise ValueError("train.monitor must be 'val_loss' or 'val_f1'")

        # log
        msg = f"Epoch {epoch}: train {train_loss:.4f} | valid {val_loss:.4f}"
        if val_f1 is not None:
            msg += f" | valF1 {val_f1:.3f}"
        print(msg)

        # record history
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_f1"].append(float(val_f1) if val_f1 is not None else None)

        # step LR scheduler
        scheduler.step(sched_value)

        # checkpoint + early stopping
        if improved:
            best_metric = current_metric
            patience = 0
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(
                {"model_state": model.state_dict(), "input_size": input_size, "cfg": cfg},
                ckpt_path,
            )
            print("Saved best:", ckpt_path, "| metric =", best_metric)
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"Early stopping on '{monitor}'. Best metric: {best_metric}")
                break

    # ----- save training history -----
    hist_path = os.path.join(os.path.dirname(ckpt_path), "train_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print("Wrote training history to", hist_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
