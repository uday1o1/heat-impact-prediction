#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc
from heat.io import load_npz
from heat.vuln import compute_vulnerability, min_max_norm

DATA = Path("data/heat_synth.npz")
MODEL = Path("data/best_cnn_lstm.keras")
OUT = Path("docs/figs"); OUT.mkdir(parents=True, exist_ok=True)

def save(path):
    plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close(); print("Saved:", path)

def main():
    assert DATA.exists(), f"Missing {DATA}. Run: python scripts/build_dataset.py"
    D = load_npz(DATA)
    X_tr, y_tr = D["X_tr"], D["y_tr"]
    X_va, y_va = D["X_va"], D["y_va"]
    X_te, y_te = D["X_te"], D["y_te"]
    exposure, sensitivity = D["exposure"], D["sensitivity"]

    # 1) Class distribution
    for split_name, y in [("train", y_tr), ("val", y_va), ("test", y_te)]:
        classes, counts = np.unique(y, return_counts=True)
        plt.figure(figsize=(4,3))
        plt.bar([str(int(c)) for c in classes], counts)
        plt.xlabel("Class (0=normal, 1=extreme heat)"); plt.ylabel("Count")
        plt.title(f"Class distribution ({split_name})")
        save(OUT / f"class_dist_{split_name}.png")

    # 2) Sample frames
    plt.figure(figsize=(8,3))
    t0 = X_tr[0, :, :, :, 0]  # channel 0 across time
    for i, day in enumerate([0, len(t0)//2, -1]):
        plt.subplot(1,3,i+1); plt.imshow(t0[day], cmap="inferno"); plt.axis("off"); plt.title(f"Day {day}")
    save(OUT / "sample_frames.png")

    # 3) If trained model exists, make eval figs
    if MODEL.exists():
        model = tf.keras.models.load_model(MODEL)
        p = model.predict(X_te, verbose=0).ravel()
        yhat = (p >= 0.5).astype(int)

        # Confusion matrix
        cm = confusion_matrix(y_te, yhat, labels=[0,1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["normal","extreme"])
        plt.figure(figsize=(4,4)); disp.plot(cmap="Blues", colorbar=False)
        plt.title("Confusion Matrix (Test)"); plt.gca().images[-1].colorbar=None
        save(OUT / "cm_test.png")

        # PR curve
        prec, rec, _ = precision_recall_curve(y_te, p)
        plt.figure(figsize=(4,3)); plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve (Test)")
        save(OUT / "pr_test.png")

        # ROC
        fpr, tpr, _ = roc_curve(y_te, p); roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(4,3)); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (Test), AUC={roc_auc:.3f}")
        save(OUT / "roc_test.png")

        # 4) Vulnerability example (use last timestep prob map as hazard)
        # Make a prob map by averaging model logits per cell over test batch for demo
        # (In a real workflow, you'd predict per grid cell; here we craft a toy map)
        hazard_prob = min_max_norm(X_te[..., 0].mean(axis=(0,1)))  # (H,W) proxy from temp channel
        V = compute_vulnerability(hazard_prob, exposure, sensitivity)
        plt.figure(figsize=(9,3))
        for i,(name,arr) in enumerate([("Hazard prob", hazard_prob), ("Exposure", exposure), ("Sensitivity", sensitivity)]):
            plt.subplot(1,3,i+1); plt.imshow(arr, cmap="inferno"); plt.title(name); plt.axis("off")
        save(OUT / "vuln_inputs.png")
        plt.figure(figsize=(4,3)); plt.imshow(V, cmap="inferno"); plt.title("Vulnerability Index"); plt.axis("off")
        save(OUT / "vulnerability.png")
    else:
        print("Model not found; run: python scripts/train_model.py")

if __name__ == "__main__":
    main()
