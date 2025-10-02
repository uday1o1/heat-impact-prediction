import argparse, os, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    f1_score, confusion_matrix, classification_report
)

from src.utils import get_device
from src.dataset import SeqDataset
from models.lstm import LSTMClassifier

def collate(batch):
    xs, ys = zip(*batch)
    import torch
    return torch.stack(xs, 0), torch.stack(ys, 0)

def main(args):
    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.metdir, exist_ok=True)

    # load config
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    H = int(cfg["features"].get("target_horizons", 1))

    # load model
    ckpt = torch.load(cfg["train"]["ckpt_path"], map_location=device)
    model = LSTMClassifier(
        input_size=ckpt["input_size"],
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        out_dim=H
    ).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    # test data
    test = pd.read_parquet(cfg["data"]["processed_test"])
    target_cols = [f"heatwave_h{h}" for h in range(1, H+1)]
    feature_cols = [c for c in test.columns if c not in target_cols]
    ds = SeqDataset(test, feature_cols, target_cols, int(cfg["features"]["lookback_days"]))
    dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)

    # get probabilities
    Ys, Ps = [], []
    with torch.no_grad():
        for xb, yb in dl:
            prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
            Ys.append(yb.numpy()); Ps.append(prob)
    Y = np.concatenate(Ys)  # (N, H)
    P = np.concatenate(Ps)  # (N, H)

    # load tuned thresholds if present
    th_path = os.path.join(os.path.dirname(cfg["train"]["ckpt_path"]), "thresholds.json")
    if os.path.exists(th_path):
        with open(th_path) as f:
            tuned = json.load(f)
        Ts = np.array([float(tuned.get(f"h{i+1}", 0.5)) for i in range(H)])
    else:
        Ts = np.full(H, 0.5, dtype=float)

    # metrics per horizon
    rows = []
    for h in range(H):
        y, p, t = Y[:, h].astype(int), P[:, h], Ts[h]

        roc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)

        # F1 at tuned threshold
        yhat = (p >= t).astype(int)
        f1 = f1_score(y, yhat)

        # Confusion matrix figure (optional)
        cm = confusion_matrix(y, yhat)
        fig_cm = plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title(f"H+{h+1} Confusion Matrix @ {t:.2f}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        cm_path = os.path.join(args.figdir, f"cm_h{h+1}.png")
        plt.savefig(cm_path, bbox_inches="tight", dpi=160); plt.close(fig_cm)

        # ROC curve
        fpr, tpr, _ = roc_curve(y, p)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc:.3f}")
        plt.plot([0,1],[0,1],"--",linewidth=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — Horizon H+{h+1}")
        plt.legend()
        roc_path = os.path.join(args.figdir, f"roc_h{h+1}.png")
        plt.savefig(roc_path, bbox_inches="tight", dpi=160); plt.close()

        # PR curve
        prec, rec, _ = precision_recall_curve(y, p)
        plt.figure()
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Precision–Recall — Horizon H+{h+1}")
        plt.legend()
        pr_path = os.path.join(args.figdir, f"pr_h{h+1}.png")
        plt.savefig(pr_path, bbox_inches="tight", dpi=160); plt.close()

        rows.append({
            "horizon": f"H+{h+1}",
            "prevalence": float(y.mean()),
            "roc_auc": float(roc),
            "pr_auc": float(ap),
            "threshold": float(t),
            "f1_at_threshold": float(f1),
            "roc_curve_png": os.path.relpath(roc_path),
            "pr_curve_png": os.path.relpath(pr_path),
            "conf_mat_png": os.path.relpath(cm_path)
        })

    # Save metrics table
    dfm = pd.DataFrame(rows)
    met_path = os.path.join(args.metdir, "test_metrics.csv")
    dfm.to_csv(met_path, index=False)
    print("Saved curves to:", args.figdir)
    print("Saved metrics to:", met_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--figdir", default="reports/figures")
    ap.add_argument("--metdir", default="reports/metrics")
    args = ap.parse_args()
    main(args)
