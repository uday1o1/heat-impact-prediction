import argparse, yaml, torch, pandas as pd, numpy as np, json, os
from datetime import timedelta
from src.utils import get_device
from models.lstm import LSTMClassifier

def main(cfg):
    device = get_device()
    ckpt = torch.load(cfg["train"]["ckpt_path"], map_location=device)
    H = int(cfg["features"].get("target_horizons", 1))

    model = LSTMClassifier(
        input_size=ckpt["input_size"],
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        out_dim=H
    ).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    df = pd.read_parquet(cfg["data"]["processed_test"])
    target_cols = [f"heatwave_h{h}" for h in range(1, H+1)]
    feature_cols = [c for c in df.columns if c not in target_cols]
    lookback = int(cfg["features"]["lookback_days"])

    X = df[feature_cols].values.astype(np.float32)
    x_seq = torch.tensor(X[-lookback:], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(x_seq)).cpu().numpy().ravel()  # (H,)

    # Load thresholds if available
    th_path = os.path.join(os.path.dirname(cfg["train"]["ckpt_path"]), "thresholds.json")
    if os.path.exists(th_path):
        with open(th_path) as f:
            th = json.load(f)
        ts = np.array([float(th.get(f"h{i+1}", 0.5)) for i in range(H)])
    else:
        ts = np.full(H, 0.5, dtype=float)

    last_day = df.index[-1]
    dates = [last_day + timedelta(days=h) for h in range(1, H+1)]
    labels = (probs >= ts).astype(int)

    out = pd.DataFrame({
        "date": dates,
        "p_heatwave": probs,
        "threshold": ts,
        "pred": labels
    })
    print(out.to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    main(cfg)
