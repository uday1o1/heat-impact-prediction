#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from heat.io import load_npz

DATA = Path("data/heat_synth.npz")
MODEL = Path("data/best_cnn_lstm.keras")

def main():
    assert DATA.exists() and MODEL.exists(), "Run build_dataset.py and train_model.py first."
    D = load_npz(DATA)
    X_te, y_te = D["X_te"], D["y_te"]

    model = tf.keras.models.load_model(MODEL)
    p = model.predict(X_te, verbose=0).ravel()
    yhat = (p >= 0.5).astype(int)

    auroc = roc_auc_score(y_te, p)
    auprc = average_precision_score(y_te, p)
    print(f"TEST AUROC: {auroc:.3f} | AUPRC: {auprc:.3f}")
    print("\nClassification Report:\n", classification_report(y_te, yhat, digits=3))

if __name__ == "__main__":
    main()
