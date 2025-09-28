import numpy as np
from sklearn.model_selection import train_test_split

def normalize_channels(X, eps=1e-8):
    mean = X.mean(axis=(0,1,2,3), keepdims=True)
    std  = X.std(axis=(0,1,2,3), keepdims=True)
    return (X - mean) / (std + eps), mean, std

def make_splits(X, y, test_size=0.2, val_size=0.2, seed=13):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=val_size, random_state=seed, stratify=y_tr)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)
