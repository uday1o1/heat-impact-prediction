#!/usr/bin/env python3
"""
Builds a synthetic spatiotemporal dataset for extreme heat prediction.
X: (N, T, H, W, C), y: (N,) binary labels
Also saves simple exposure/sensitivity rasters for vulnerability demo.
"""

from pathlib import Path
import numpy as np
from heat.io import save_npz
from heat.preprocess import normalize_channels, make_splits
from heat.utils import ensure_dir

OUT = Path("data/heat_synth.npz")
SEED = 7

def make_synthetic(N=4000, T=7, H=32, W=32, C=3, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(N, T, H, W, C)).astype(np.float32)

    # ~25% positives so it’s imbalanced like real events
    y = (rng.random(N) < 0.25).astype(np.int64)

    def t_trend(lo, hi, noise=0.15):
        base = np.linspace(lo, hi, T, dtype=np.float32).reshape(T, 1, 1)
        return base + rng.normal(0, noise, (T, H, W)).astype(np.float32)

    # Spatial heterogeneity (regional baselines that change per sample)
    def spatial_field(scale=0.5):
        gx = rng.normal(0, 1, (H, 1)).astype(np.float32)
        gy = rng.normal(0, 1, (1, W)).astype(np.float32)
        f = (gx + gy)
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)  # 0..1
        return scale * (f - 0.5)  # centered

    for i in range(N):
        # per-sample spatial bias (e.g., urban heat island / coastal effects)
        bias0 = spatial_field(0.25)  # temp bias
        bias1 = spatial_field(0.20)  # humidity bias
        bias2 = spatial_field(0.20)  # soil moisture bias

        if y[i] == 1:
            # heatwave pattern but *subtle* and noisy
            trend0 = t_trend(-0.1, 0.5, noise=0.20)   # temp anomaly smaller
            trend1 = t_trend(0.1, 0.5, noise=0.20)    # humidity moderate
            trend2 = t_trend(0.0, -0.2, noise=0.10)   # soil moisture slightly lower
        else:
            # non-event overlaps with event distribution
            trend0 = t_trend(-0.05, 0.3, noise=0.25)
            trend1 = t_trend(0.05, 0.3, noise=0.25)
            trend2 = t_trend(-0.05, 0.05, noise=0.12)

        # add temporal jitter: randomly shift the “peak” day
        shift = rng.integers(-1, 2)  # -1,0,1
        trend0 = np.roll(trend0, shift, axis=0)
        trend1 = np.roll(trend1, shift, axis=0)
        trend2 = np.roll(trend2, shift, axis=0)

        # inject into channels + spatial biases; add correlated noise
        corr_noise = rng.normal(0, 0.08, (T, H, W)).astype(np.float32)
        X[i, :, :, :, 0] += trend0 + bias0 + corr_noise
        X[i, :, :, :, 1] += trend1 + bias1 + 0.8 * corr_noise + rng.normal(0, 0.05, (T, H, W))
        X[i, :, :, :, 2] += trend2 + bias2 + 0.5 * corr_noise + rng.normal(0, 0.05, (T, H, W))

        # small label noise (real data has annotation errors)
        if rng.random() < 0.03:
            y[i] = 1 - y[i]

    # Normalize per-channel across dataset
    Xn, mean, std = normalize_channels(X)

    # Exposure/sensitivity (0..1) with non-trivial patterns
    gx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    gy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    exposure = (0.6 * gy + 0.4 * gx)  # weighted gradient
    exposure = (exposure - exposure.min()) / (exposure.max() - exposure.min() + 1e-8)
    sensitivity = (0.6 * np.flipud(gy) + 0.4 * gx)
    sensitivity = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min() + 1e-8)

    return Xn, y, exposure.astype(np.float32), sensitivity.astype(np.float32), \
           mean.astype(np.float32), std.astype(np.float32)

def main():
    ensure_dir(Path("data"))
    X, y, exposure, sensitivity, mean, std = make_synthetic()

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = make_splits(X, y, test_size=0.2, val_size=0.2, seed=13)

    save_npz(OUT,
             X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, X_te=X_te, y_te=y_te,
             exposure=exposure, sensitivity=sensitivity, mean=mean, std=std)
    print(f"Saved {OUT.resolve()}",
          f"\nShapes: X_tr{X_tr.shape}, X_va{X_va.shape}, X_te{X_te.shape}, y_tr{y_tr.shape}")

if __name__ == "__main__":
    main()
