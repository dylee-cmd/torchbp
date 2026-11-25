"""Experiments of the paper on simulated data.

    python -m reconstruction.demo [--plot out.png] [--quick]

A. Point scatterer model (Section IV-A): reconstruction rate and RMSE
   versus the number of views.
B. Simulated ship (Section IV-B): direct reconstruction, with occluded
   footprint completion, and with error region deletion; then the number of
   views at a fixed 10 deg interval and the angle interval with four views.
C. The same ship with speckle and sea clutter, segmented with the default
   fine segmentation (no ground truth masks).
"""
import argparse

import numpy as np

from .geometry import ImageGrid, views_on_circle
from .segmentation import segment_target, complete_footprint, radon_azimuth
from .simulate import (
    ShipModel, point_scatterer_model, render_points, simulate_ship_images,
)
from .voxel import (
    VoxelGrid, reconstruct, remove_error_regions, cluster_centroids,
    match_points, principal_dimensions, voxel_scores,
)


def point_experiment(n_views_list, voxel=0.5):
    pts = point_scatterer_model()
    lo, hi = pts.min(0) - 3, pts.max(0) + 3
    grid = VoxelGrid.from_bounds(lo, hi, voxel)
    rows = []
    for n in n_views_list:
        views = views_on_circle(n, 30.0, 10.0)
        masks, grids = [], []
        for v in views:
            g = ImageGrid.covering(v, lo, hi, 0.5)
            img = render_points(pts, v, g, psf_sigma=1.0)
            masks.append(img > 0.3 * img.max())
            grids.append(g)
        occ = reconstruct(grid, views, masks, grids)
        est = cluster_centroids(occ, grid)
        rate, rmse = match_points(est, pts, 2.0)
        rows.append((n, len(est), rate, rmse))
    return rows


def ship_reconstruction(ship, views, seg_kwargs=None, voxel=0.5, speckle=False,
                        clutter=0.0, seed=0):
    """Returns dict of (precision, recall, IoU, (L, W, H)) per stage."""
    seg_kwargs = seg_kwargs or {}
    imgs, grids = simulate_ship_images(ship, views, speckle=speckle,
                                       clutter=clutter, seed=seed)
    masks = [segment_target(im, **seg_kwargs) for im in imgs]
    lo, hi = ship.bbox()
    vg = VoxelGrid.from_bounds(lo, hi, voxel)
    truth = ship.contains(vg.centers().reshape(*vg.shape, 3))
    out = {}
    occ = reconstruct(vg, views, masks, grids)
    out["direct"] = occ
    cm = [complete_footprint(m, v, g, ship.deck_outline(), 0.0, ship.freeboard)
          for m, v, g in zip(masks, views, grids)]
    occ = reconstruct(vg, views, cm, grids)
    out["+ footprint completion"] = occ
    theta = min(v.theta for v in views)
    out["+ error region deletion"] = remove_error_regions(occ, vg, theta,
                                                         heading=ship.heading)
    res = {}
    for k, o in out.items():
        dims = principal_dimensions(vg.points(o), ship.heading) if o.any() else (0, 0, 0)
        res[k] = (*voxel_scores(o, truth), dims)
    heading_err = []
    for m, g, v in zip(masks, grids, views):
        h = np.mod(radon_azimuth(m, g) + v.alpha, np.pi)
        heading_err.append(np.rad2deg(np.mod(h - ship.heading + np.pi / 2, np.pi) - np.pi / 2))
    return res, out, vg, truth, float(np.median(np.abs(heading_err)))


def _print_ship(title, res):
    print(f"\n{title}")
    print(f"  {'stage':26s} {'prec':>6s} {'recall':>6s} {'IoU':>6s}   L / W / H [m]")
    for k, (p, r, iou, dims) in res.items():
        print(f"  {k:26s} {p:6.3f} {r:6.3f} {iou:6.3f}   "
              f"{dims[0]:.1f} / {dims[1]:.1f} / {dims[2]:.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", type=str, default=None)
    ap.add_argument("--quick", action="store_true", help="skip the sweeps")
    args = ap.parse_args(argv)

    print("A. Point scatterer model (18 scatterers, incidence 30 deg, 10 deg steps)")
    print(f"  {'views':>5s} {'clusters':>8s} {'rate':>6s} {'RMSE [m]':>9s}")
    for n, ncl, rate, rmse in point_experiment([3, 5, 10, 15, 25] if not args.quick else [5, 25]):
        print(f"  {n:5d} {ncl:8d} {rate:6.3f} {rmse:9.3f}")

    ship = ShipModel(heading=np.deg2rad(20.0))
    print(f"\nB. Simulated ship: length {ship.length} m, width {ship.width} m, "
          f"mast top {ship.mast_top} m (true extents on the voxel grid differ "
          f"slightly by quantization)")
    clean = dict(smooth=1, open_size=1, ellipse_scale=None)
    views = views_on_circle(36, 30.0)
    res, out, vg, truth, herr = ship_reconstruction(ship, views, clean)
    tdims = principal_dimensions(vg.points(truth), ship.heading)
    print(f"  true voxel model L / W / H: {tdims[0]:.1f} / {tdims[1]:.1f} / {tdims[2]:.1f} m;"
          f" Radon heading error (median over views): {herr:.2f} deg")
    _print_ship("  36 views, 10 deg interval", res)

    if not args.quick:
        for n in (8, 12, 16):
            r, *_ = ship_reconstruction(ship, views_on_circle(n, 30.0, 10.0), clean)
            _print_ship(f"  {n} views, 10 deg interval", r)
        for step in (20, 30, 60, 90):
            r, *_ = ship_reconstruction(ship, views_on_circle(4, 30.0, step), clean)
            _print_ship(f"  4 views, {step} deg interval", r)

    print("\nC. Ship with speckle and sea clutter (default fine segmentation)")
    rn, *_ = ship_reconstruction(ship, views, None, speckle=True, clutter=0.05, seed=1)
    _print_ship("  36 views, 10 deg interval", rn)

    if args.plot:
        _plot(args.plot, out, vg, truth)


def _plot(path, out, vg, truth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 5))
    items = [("ground truth", truth)] + list(out.items())
    for i, (title, occ) in enumerate(items):
        ax = fig.add_subplot(1, len(items), i + 1, projection="3d")
        p = vg.points(occ)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=p[:, 2], s=1, cmap="jet")
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_box_aspect((np.ptp(p[:, 0]) or 1, np.ptp(p[:, 1]) or 1, np.ptp(p[:, 2]) or 1))
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
