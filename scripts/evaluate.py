import argparse, yaml, torch, pandas as pd
from torch.utils.data import DataLoader
from src.dataset import SeqDataset
from src.utils import get_device
from models.lstm import LSTMClassifier
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
import numpy as np

def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)

def run(cfg):
    device = get_device()
    ckpt = torch.load(cfg["train"]["ckpt_path"], map_location=device)
    H = int(cfg["features"].get("target_horizons", 1))

    model = LSTMClassifier(input_size=ckpt["input_size"],
                           hidden_size=cfg["model"]["hidden_size"],
                           num_layers=cfg["model"]["num_layers"],
                           dropout=cfg["model"]["dropout"],
                           out_dim=H).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    df = pd.read_parquet(cfg["data"]["processed_test"])
    target_cols = [f"heatwave_h{h}" for h in range(1, H+1)]
    feature_cols = [c for c in df.columns if c not in target_cols]
    ds = SeqDataset(df, feature_cols, target_cols, cfg["features"]["lookback_days"])
    dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)

    Y, P = [], []
    with torch.no_grad():
        for xb, yb in dl:
            logits = model(xb.to(device))
            prob = torch.sigmoid(logits).cpu().numpy()
            Y.append(yb.numpy()); P.append(prob)
    Y = np.concatenate(Y)  # (N,H)
    P = np.concatenate(P)  # (N,H)

    for h in range(H):
        y, p = Y[:,h], P[:,h]
        roc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        yhat = (p >= 0.5).astype(int)
        f1 = f1_score(y, yhat)
        print(f"H+{h+1}: ROC-AUC={roc:.3f} | PR-AUC={ap:.3f} | F1={f1:.3f} | prevalence={y.mean():.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    run(cfg)
