import numpy as np

def compute_vulnerability(hazard_prob, exposure, sensitivity, clip=True):
    v = hazard_prob * exposure * sensitivity
    if clip:
        v = np.clip(v, 0, 1)
    return v

def min_max_norm(arr, eps=1e-8):
    m, M = arr.min(), arr.max()
    return (arr - m) / (M - m + eps)
