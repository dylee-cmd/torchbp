"""Coherent scatterer change detection on two amplitude images.

Usage
-----
Your own coregistered amplitude images (2D arrays saved with numpy)::

    python -m change_detection.examples.amplitude_pair_example \\
        --img1 before.npy --img2 after.npy --spacing 0.5 0.5 --plot result.png

Without ``--img1/--img2`` a simulated pair is used: amplitudes of images 2
and 4 of :func:`change_detection.make_slc_time_series`. Between them a
building was demolished, a new building appeared, a container stack
appeared, and one static building is covered by snow in image 2.

Images must be coregistered, radiometrically calibrated amplitudes
(``|SLC|``, not dB) with the same geometry. ``--spacing`` is the pixel
spacing (azimuth, range) in meters, used by the clustering.

See :mod:`change_detection.cs_amplitude` for what replaces the phase-based
CS detection and the coherence when only amplitudes are available.
"""
import argparse

import numpy as np

from change_detection.cs_amplitude import (
    amplitude_pair_change_detection, detect_bright_scatterers,
)
from change_detection.cs_simulation import make_slc_time_series


def load_or_simulate(args):
    if args.img1 and args.img2:
        a1 = np.load(args.img1)
        a2 = np.load(args.img2)
        if a1.shape != a2.shape:
            raise SystemExit(f"image shapes differ: {a1.shape} vs {a2.shape}")
        return np.abs(a1), np.abs(a2), None
    scene = make_slc_time_series(seed=args.seed)
    i, j = 2, 4
    a1 = np.abs(scene.stack[i])
    a2 = np.abs(scene.stack[j])
    # Expected pair change of every simulated object between images i and j.
    truth = []
    for o in scene.objects:
        p1, p2 = o.a <= i <= o.b, o.a <= j <= o.b
        if p1 and not p2:
            truth.append((o, "disappeared"))
        elif p2 and not p1:
            truth.append((o, "appeared"))
        elif p1 and p2 and o.transient in (i, j):
            truth.append((o, "transient (snow)"))
    return a1, a2, truth


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img1", help="amplitude image at t1 (.npy)")
    ap.add_argument("--img2", help="amplitude image at t2 (.npy)")
    ap.add_argument("--spacing", type=float, nargs=2, default=(1.0, 1.0),
                    metavar=("AZ", "RG"), help="pixel spacing in meters")
    ap.add_argument("--scr-db", type=float, default=10.0,
                    help="point scatterer threshold above the local median [dB]")
    ap.add_argument("--gamma-t", type=float, default=0.5,
                    help="amplitude similarity threshold (high = unchanged)")
    ap.add_argument("--seed", type=int, default=0, help="simulation seed")
    ap.add_argument("--plot", help="save a figure to this file")
    args = ap.parse_args(argv)

    a1, a2, truth = load_or_simulate(args)
    res = amplitude_pair_change_detection(a1, a2, spacing=tuple(args.spacing),
                                          scr_db=args.scr_db, gamma_t=args.gamma_t)

    print(f"point scatterers: {res.cs[0].sum()} (t1), {res.cs[1].sum()} (t2); "
          f"{len(res.clusters)} change clusters")
    for idx, (c, cat) in enumerate(zip(res.clusters, res.categories)):
        print(f"  #{idx}: {cat:11s} {c.n_cs:4d} scatterers  {c.area_m2:7.0f} m^2  "
              f"center (row, col) = ({c.center[0]:.0f}, {c.center[1]:.0f})  "
              f"segmented {c.mask.sum()} px")
    if truth is not None:
        print("\nsimulated truth:")
        for o, kind in truth:
            hit = [cat for c, cat in zip(res.clusters, res.categories)
                   if o.contains(*c.center, pad=5)]
            print(f"  {o.name:26s} {kind:17s} -> detected as {hit if hit else 'nothing'}")

    if args.plot:
        plot(args.plot, a1, a2, res)


def plot(path, a1, a2, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    db = lambda a: 20 * np.log10(np.abs(a) + 1e-3)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(db(a1), cmap="gray", vmin=-10, vmax=30)
    ax[0].set_title("t1 amplitude (dB)")
    ax[1].imshow(db(a2), cmap="gray", vmin=-10, vmax=30)
    ax[1].set_title("t2 amplitude (dB)")
    ax[2].imshow(res.f[0], vmin=0, vmax=1)
    ax[2].set_title("amplitude similarity (1 - NCB)")
    base = np.clip((db(a2) + 10) / 40, 0, 1)
    rgb = np.repeat(base[..., None], 3, -1) * 0.6
    for c, cat in zip(res.clusters, res.categories):
        rgb[c.mask] = (1, 0, 0) if cat == "appeared" else (0, 0.8, 0)
    ax[3].imshow(rgb)
    ax[3].set_title("changes (red appeared, green disappeared)")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
