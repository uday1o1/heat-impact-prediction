from pathlib import Path
import os
import numpy as np
import tensorflow as tf

from heat.io import load_npz
from heat.model import build_cnn_lstm

DATA = Path("data/heat_synth.npz")
CKPT = Path("data/best_cnn_lstm.keras")

EPOCHS = 20
BATCH_SIZE = 32
LABEL_SMOOTH_EPS = 0.05

def setup_tf():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

def smooth_labels(y, eps=LABEL_SMOOTH_EPS):
    y = y.astype("float32")
    return y * (1.0 - eps) + 0.5 * eps

def main():
    setup_tf()
    assert DATA.exists(), f"Missing {DATA}. Run: python scripts/build_dataset.py"

    D = load_npz(DATA)
    X_tr, y_tr = D["X_tr"], D["y_tr"]
    X_va, y_va = D["X_va"], D["y_va"]

    print("[load] X_tr", X_tr.shape, "y_tr", y_tr.shape)
    print("[load] X_va", X_va.shape, "y_va", y_va.shape)

    model = build_cnn_lstm(
        input_shape=X_tr.shape[1:],
        n_classes=1,
    )
    model.summary()

    y_tr_s = smooth_labels(y_tr)
    y_va_s = smooth_labels(y_va)

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CKPT),
        monitor="val_auroc",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_auroc",
        mode="max",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    )
    rlrop_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auroc",
        mode="max",
        factor=0.5,
        patience=1,
        verbose=1,
        min_lr=1e-5,
    )

    history = model.fit(
        X_tr, y_tr_s,
        validation_data=(X_va, y_va_s),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[ckpt_cb, es_cb, rlrop_cb],
        verbose=1,
    )

    print("\nTraining complete.")
    if CKPT.exists():
        print(f"Best model saved to: {CKPT.resolve()}")
    else:
        print("Checkpoint file not found; best weights kept in memory (EarlyStopping restore).")
        model.save(CKPT)
        print(f"Saved current model to: {CKPT.resolve()}")

if __name__ == "__main__":
    main()
