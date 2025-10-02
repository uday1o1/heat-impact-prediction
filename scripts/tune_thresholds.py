import argparse, json, os, yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_curve
from tqdm import tqdm

from src.utils import get_device
from src.dataset import SeqDataset
from models.lstm import LSTMClassifier

def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)

def main(cfg):
    device = get_device()
    ckpt = torch.load(cfg["train"]["ckpt_path"], map_location=device)
    H = int(cfg["features"].get("target_horizons", 1))

    model = LSTMClassifier(
        input_size=ckpt["input_size"],
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        out_dim=H,
    ).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    df = pd.read_parquet(cfg["data"]["processed_valid"])
    target_cols = [f"heatwave_h{h}" for h in range(1, H+1)]
    feature_cols = [c for c in df.columns if c not in target_cols]
    ds = SeqDataset(df, feature_cols, target_cols, int(cfg["features"]["lookback_days"]))
    dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)

    Y, P = [], []
    with torch.no_grad():
        for xb, yb in tqdm(dl, desc="valid"):
            prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
            Y.append(yb.numpy()); P.append(prob)
    Y = np.concatenate(Y)  # (N,H)
    P = np.concatenate(P)  # (N,H)

    thresholds = {}
    for h in range(H):
        y, p = Y[:, h].astype(int), P[:, h]
        # Search best F1 along PR curve (robust and fast)
        prec, rec, ths = precision_recall_curve(y, p)
        f1 = (2 * prec * rec) / (prec + rec + 1e-12)
        # exclude degenerate where threshold is nan or arrays misalign
        best_idx = int(np.nanargmax(f1))
        best_t = float(ths[max(best_idx-1, 0)]) if best_idx < len(ths) else 0.5
        thresholds[f"h{h+1}"] = best_t

    os.makedirs(os.path.dirname(cfg["train"]["ckpt_path"]), exist_ok=True)
    out_path = os.path.join(os.path.dirname(cfg["train"]["ckpt_path"]), "thresholds.json")
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print("Saved thresholds to", out_path)
    print(thresholds)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    main(cfg)
