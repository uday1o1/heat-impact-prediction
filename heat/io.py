from pathlib import Path
import numpy as np

def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)

def load_npz(path: Path):
    with np.load(path) as D:
        return {k: D[k] for k in D.files}
