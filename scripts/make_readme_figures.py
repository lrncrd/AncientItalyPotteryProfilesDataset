"""
Figures of the README (assets/). Vessels are drawn in 3D by revolving the profile around its axis, with the renderer of
the Reconstructor project (potfm/render3d.py, https://github.com/lrncrd/Reconstructor); the map uses the Natural Earth
admin-0 countries (1:50m, public domain).

    python scripts/make_readme_figures.py --reconstructor ../Reconstructor --naturalearth ne_50m_admin_0_countries.geojson
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
PERIODS = ["EBA", "MBA", "RBA", "FBA", "EIA1", "EIA2", "OP", "AP"]
PERIOD_NAMES = {"EBA": "Early Bronze Age", "MBA": "Middle Bronze Age", "RBA": "Recent (Late) Bronze Age", "FBA": "Final Bronze Age",
                "EIA1": "Early Iron Age 1", "EIA2": "Early Iron Age 2", "OP": "Orientalising", "AP": "Archaic"}
INK, INK_2, GRID = "#1b1b1b", "#5a5955", "#e6e5e1"
CLAY = "#c98f6a"
TYPE_COL = {"Funerary": "#2a78d6", "Settlement": "#eb6834", "Cultual": "#1baf7a"}


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.labelcolor": INK, "savefig.dpi": 200})


def load():
    meta = pd.read_csv(ROOT / "data" / "aippd_metadata.csv")
    z = np.load(ROOT / "data" / "aippd_profiles_256.npz")
    masks = np.unpackbits(z["masks"], axis=-1).astype(bool)
    return meta, masks


def physical(mask, height_cm, mm_per_px=1.5):
    """The 256 mask cropped and resampled to its real size (mm_per_px), as Reconstructor's corpus."""
    from potfm.frame import minimum_image, to_physical
    crop = minimum_image(mask.astype(np.float32)) > 0.5
    return to_physical(crop, height_cm / crop.shape[0])


def typical(meta, sel, n, seed=0):
    """n vessels of `sel`, spread over proportions (height / width) and close to the median size."""
    d = meta[sel]
    rng = np.random.default_rng(seed)
    ratio = np.log(d["height_cm"] / d["max_width_cm"])
    size = np.log(d["height_cm"])
    out = []
    for q in np.linspace(0.08, 0.92, n):
        score = np.abs(ratio - ratio.quantile(q)) + 0.4 * np.abs(size - size.median()) + 0.01 * rng.random(len(d))
        score[d.index.isin(out)] = np.inf
        out.append(int(score.idxmin()))
    return out


def lineup(meta, masks):
    """Each period, five vessels at one scale (the same in every panel): the hero image."""
    from PIL import Image
    from potfm.render3d import render_pot
    ppc = 7.0  # pixels per cm, common to all panels
    panels = []
    for per in PERIODS:
        ids = typical(meta, (meta["period"] == per).values, 5, seed=PERIODS.index(per))
        imgs = []
        for i in ids:
            img = render_pot(physical(masks[i], meta["height_cm"][i]), size=2.4, dpi=160)
            w = max(int(round(meta["max_width_cm"][i] * ppc)), 8)
            imgs.append(np.asarray(Image.fromarray(img).resize((w, int(round(img.shape[0] * w / img.shape[1]))), Image.LANCZOS)))
        panels.append(imgs)
    gap = 10
    H = max(im.shape[0] for ims in panels for im in ims) + 4
    W = max(sum(im.shape[1] for im in ims) + gap * (len(ims) - 1) for ims in panels) + 8
    fig, axes = plt.subplots(4, 2, figsize=(2 * W / 95, 4 * (H + 60) / 95))
    for ax, per, ims in zip(axes.ravel(), PERIODS, panels):
        canvas = np.zeros((H, W, 4), np.uint8)
        x = (W - (sum(im.shape[1] for im in ims) + gap * (len(ims) - 1))) // 2
        for im in ims:
            canvas[H - im.shape[0] - 2:H - 2, x:x + im.shape[1]] = im
            x += im.shape[1] + gap
        ax.imshow(canvas)
        ax.axhline(H - 2, color=GRID, lw=0.8)
        ax.axis("off")
        ax.set_title(f"{per}  ·  {PERIOD_NAMES[per]}", fontsize=10, color=INK, loc="left", pad=4)
    ax = axes.ravel()[-1]
    bar = 20 * ppc
    ax.plot([W - bar - 4, W - 4], [8, 8], color=INK, lw=1.6)
    ax.text(W - bar / 2 - 4, 14, "20 cm", ha="center", va="top", fontsize=8, color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, wspace=0.04, hspace=0.25)
    fig.savefig(ASSETS / "aippd_lineup.png", facecolor="white")
    plt.close(fig)


def drawing_and_vessel(meta, masks):
    """What a record is: the profile drawing (section on the left of the axis) and the vessel it describes."""
    from scipy import ndimage
    from potfm.frame import CM_PER_PX
    from potfm.render3d import outline, render_pot
    sel = ((meta["height_cm"] > 7) & (meta["height_cm"] < 40)).values
    ids = [typical(meta, sel & (meta["height_cm"].between(lo, hi)).values, 1, seed=k)[0]
           for k, (lo, hi) in enumerate(((7, 9), (9, 12), (12, 16), (16, 22), (22, 30), (30, 40)))]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 4.6))
    for ax, i in zip(axes.ravel(), ids):
        ax.axis("off")
        pot = physical(masks[i], meta["height_cm"][i])
        pts = outline(pot)
        rows_ = np.nonzero(pot.any(axis=1))[0]
        outer = ndimage.gaussian_filter1d(np.array([(pot.shape[1] - np.nonzero(pot[r])[0].min()) * CM_PER_PX for r in rows_]), 1.5)
        y = (pot.shape[0] - rows_ - 0.5) * CM_PER_PX
        H, R = pts[:, 1].max(), pts[:, 0].max()
        ax.fill(-pts[:, 0], pts[:, 1], color=INK, lw=0)
        ax.plot(outer, y, color=INK, lw=0.7)
        ax.plot([outer[0], 0], [y[0], y[0]], color=INK, lw=0.7)
        ax.plot([0, outer[-1]], [y[-1], y[-1]], color=INK, lw=0.7)
        ax.plot([0, 0], [-0.04 * H, 1.04 * H], color=INK_2, lw=0.5, ls=(0, (6, 2, 1, 2)))
        img = render_pot(pot, size=1.8, dpi=200)
        w = H * img.shape[1] / img.shape[0]
        ax.imshow(img, extent=(R * 1.3, R * 1.3 + w, 0, H))
        ax.set_xlim(-R * 1.1, R * 1.3 + w)
        ax.set_ylim(-0.2 * H, 1.04 * H)
        ax.set_aspect("equal")
        ax.text(0, -0.12 * H, f"{meta['period'][i]} · {meta['height_cm'][i]:.0f} cm", ha="center", va="top", fontsize=8,
                color=INK_2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05, hspace=0.02)
    fig.savefig(ASSETS / "aippd_drawing_to_vessel.png", facecolor="white")
    plt.close(fig)


def masks_grid(meta, masks):
    """The data as a model sees it: 256 x 256 binary profiles, drawn through the smoothing network of Reconstructor
    (potfm/superres.py: the outline at 4x the resolution, without the 1-pixel steps; display only, the data are binary)."""
    from potfm import superres
    net = superres.load()
    rng = np.random.default_rng(1)
    ids = rng.choice(len(masks), 24, replace=False)
    fig, axes = plt.subplots(3, 8, figsize=(12, 4.6))
    for ax, i in zip(axes.ravel(), ids):
        # at the corpus' physical scale (1.5 mm/px), where the network was trained, then fitted back to a square cell
        f = superres.field(physical(masks[i], meta["height_cm"][i]), net)
        img = np.clip(f * superres.SCALE + 0.5, 0, 1)
        side = int(max(img.shape) * 1.02)
        cell = np.zeros((side, side), np.float32)
        y0, x0 = (side - img.shape[0]) // 2, side - img.shape[1] - (side - img.shape[0]) // 2 // 2
        cell[y0:y0 + img.shape[0], max(x0, 0):max(x0, 0) + img.shape[1]] = img[:, :side - max(x0, 0)]
        ax.imshow(cell, cmap="gray_r", interpolation="bilinear", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GRID)
        ax.set_title(meta["id"][i], fontsize=6.5, color=INK_2, pad=2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.08, hspace=0.18)
    fig.savefig(ASSETS / "aippd_masks.png", facecolor="white")
    plt.close(fig)


def site_map(meta, ne_path):
    sites = json.load(open(ROOT / "data" / "aippd_sites.geojson"))["features"]
    ctx_type = meta.groupby("context_id")["context_type"].first()
    world = json.load(open(ne_path))["features"]
    fig, ax = plt.subplots(figsize=(6.2, 7.0))
    polys = []
    for f in world:
        g = f["geometry"]
        parts = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for p in parts:
            ring = np.array(p[0])
            if ring[:, 0].max() > 4 and ring[:, 0].min() < 22 and ring[:, 1].max() > 35 and ring[:, 1].min() < 49:
                polys.append(ring)
    ax.add_collection(PolyCollection(polys, facecolor="#f1efe9", edgecolor="#b8b7b0", linewidth=0.5))
    for t, col in TYPE_COL.items():
        pts = [(s["geometry"]["coordinates"], s["properties"]["n_vessels"]) for s in sites
               if ctx_type.get(s["properties"]["context_id"]) == t and s["properties"]["n_vessels"] > 0]
        if not pts:
            continue
        xy = np.array([p for p, _ in pts])
        n = np.array([v for _, v in pts])
        lab = {"Cultual": "Cult place"}.get(t, t)
        ax.scatter(xy[:, 0], xy[:, 1], s=6 + 2.2 * np.sqrt(n) ** 1.3, c=col, alpha=0.7, edgecolors="white", linewidths=0.4,
                   label=f"{lab} ({len(xy)})", zorder=3)
    for n in (10, 100, 1000):
        ax.scatter([], [], s=6 + 2.2 * np.sqrt(n) ** 1.3, c="#9a9993", edgecolors="white", label=f"{n} vessels")
    ax.set_xlim(6.4, 19.0)
    ax.set_ylim(37.6, 47.3)
    ax.set_aspect(1 / np.cos(np.radians(42.5)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor("#eaf2f8")
    ax.legend(frameon=False, loc="lower left", fontsize=8, labelspacing=0.9, borderpad=0.6)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(ASSETS / "aippd_map.png", facecolor="white")
    plt.close(fig)


def overview(meta):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3), gridspec_kw={"width_ratios": [1.2, 1, 1], "wspace": 0.32})
    ax = axes[0]
    n = meta["period"].value_counts().reindex(PERIODS)
    ax.bar(range(len(n)), n.values, color="#2a78d6", alpha=0.9, width=0.72)
    for k, v in enumerate(n.values):
        ax.text(k, v + 80, f"{v:,}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(n)))
    ax.set_xticklabels(n.index)
    ax.set_ylabel("Vessels")
    ax.set_title("Vessels per period", fontsize=10, loc="left")
    for ax, key, lab, hi, col in ((axes[1], "height_cm", "Height (cm)", 60, "#c98f6a"), (axes[2], "rim_diameter_cm", "Rim diameter (cm)", 50, "#7d5ba6")):
        v = meta[key].values
        ax.hist(np.clip(v, 0, hi), bins=np.arange(0, hi + 2, 2), color=col, alpha=0.9, edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(v), color=INK, lw=0.8, ls="--")
        ax.text(np.median(v) + 1.2, ax.get_ylim()[1] * 0.92, f"median {np.median(v):.1f} cm", fontsize=7.5, va="top")
        ax.set_xlabel(lab)
        ax.set_title(lab.split(" (")[0], fontsize=10, loc="left")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.17)
    fig.savefig(ASSETS / "aippd_overview.png", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reconstructor", type=Path, required=True)
    ap.add_argument("--naturalearth", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.reconstructor.resolve()))
    style()
    ASSETS.mkdir(exist_ok=True)
    meta, masks = load()
    overview(meta)
    site_map(meta, args.naturalearth)
    masks_grid(meta, masks)
    drawing_and_vessel(meta, masks)
    lineup(meta, masks)
    for f in sorted(ASSETS.glob("*.png")):
        print(f.name, f"{f.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
