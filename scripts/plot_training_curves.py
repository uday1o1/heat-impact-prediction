import argparse, json, os
import matplotlib.pyplot as plt

def main(args):
    hist_path = args.history
    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    with open(hist_path) as f:
        H = json.load(f)

    epochs = H["epoch"]
    train_loss = H["train_loss"]
    val_loss = H["val_loss"]
    val_f1 = H.get("val_f1", None)

    # 1) Loss curves
    plt.figure()
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("BCE loss"); plt.title("Training / Validation Loss")
    plt.legend()
    p1 = os.path.join(out_dir, "train_val_loss.png")
    plt.savefig(p1, bbox_inches="tight", dpi=160)
    plt.close()

    # 2) Optional: F1 curve (if present)
    if val_f1 and any(v is not None for v in val_f1):
        vf = [v if v is not None else float("nan") for v in val_f1]
        plt.figure()
        plt.plot(epochs, vf, label="Val F1 (macro across horizons)")
        plt.xlabel("Epoch"); plt.ylabel("F1"); plt.ylim(0, 1)
        plt.title("Validation F1")
        plt.legend()
        p2 = os.path.join(out_dir, "val_f1.png")
        plt.savefig(p2, bbox_inches="tight", dpi=160)
        plt.close()

    print("Saved figures to:", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="models/train_history.json")
    ap.add_argument("--outdir", default="reports/figures")
    args = ap.parse_args()
    main(args)
