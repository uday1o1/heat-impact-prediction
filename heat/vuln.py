import numpy as np

def compute_vulnerability(hazard_prob, exposure, sensitivity, clip=True):
    """
    hazard_prob: (H, W) or (N,H,W) predicted probabilities in [0,1]
    exposure:    same shape, e.g., normalized population density
    sensitivity: same shape, e.g., % elderly or low-income normalized
    Returns: vulnerability index in [0,1] (if clip=True)
    """
    v = hazard_prob * exposure * sensitivity
    if clip:
        v = np.clip(v, 0, 1)
    return v

def min_max_norm(arr, eps=1e-8):
    m, M = arr.min(), arr.max()
    return (arr - m) / (M - m + eps)
