"""
Minimal loader for AIPPD (Ancient Italy Pottery Profiles Dataset). Needs numpy and pandas (pyarrow and pillow for the
parquet file).

    import aippd
    masks, ids = aippd.load_masks()          # (21782, 256, 256) bool, axis of the vessel on the right edge
    meta = aippd.load_metadata()             # one row per vessel, same order as masks
    pot = aippd.real_size(masks[0], meta.height_cm[0])   # the profile cropped and resampled to 1 px = 1 mm
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"


def load_masks(path=DATA / "aippd_profiles_256.npz"):
    """Profiles as (N, 256, 256) bool arrays (True = the vessel's section) and their ids."""
    z = np.load(path)
    return np.unpackbits(z["masks"], axis=-1).astype(bool), z["ids"]


def load_metadata(path=DATA / "aippd_metadata.csv"):
    return pd.read_csv(path, dtype={"grave": str, "page": str, "figure_or_plate": str, "item_number": str})


def load_parquet(path=DATA / "aippd.parquet", decode=True):
    """The whole dataset as a DataFrame; with decode=True the column `mask` holds the (256, 256) bool arrays."""
    df = pd.read_parquet(path)
    if decode:
        import io
        from PIL import Image
        df["mask"] = [np.array(Image.open(io.BytesIO(b))) > 0 for b in df.pop("mask_png")]
    return df


def load_bibliography(path=DATA / "aippd_bibliography.csv"):
    return pd.read_csv(path)


def crop(mask):
    """The mask cut to the vessel's bounding box (its height in px is the vessel's height)."""
    rows, cols = np.nonzero(mask)
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def real_size(mask, height_cm, mm_per_px=1.0):
    """The profile at its real size: cropped and resampled so that one pixel is `mm_per_px` millimetres."""
    from scipy.ndimage import zoom
    c = crop(mask).astype(np.float32)
    factor = height_cm * 10 / mm_per_px / c.shape[0]
    return zoom(c, factor, order=1) > 0.5
