"""Run both change detection methods on synthetic SAR scenes.

    python -m change_detection.demo [--seeds 3] [--plot out.png]

Prints object-level precision / recall / F1 (DBSCAN clustering, same
protocol as the papers) of both methods, their stage / fusion ablations and
two simple global threshold baselines.
"""
import argparse
import time

import numpy as np

from .common import (
    lee_filter, log_ratio, ncb_image, otsu_threshold, morphological_opening,
    evaluate_objects, make_synthetic_pair, ObjectScores,
)
from .fusion import fusion_change_detection
from .superpixel_classification import superpixel_change_detection


def _sum(scores):
    return ObjectScores(sum(s.tp for s in scores), sum(s.fp for s in scores),
                        sum(s.fn for s in scores))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--targets", type=int, default=16)
    ap.add_argument("--plot", type=str, default=None,
                    help="save a figure of the first scene to this file")
    args = ap.parse_args(argv)

    rows = {}

    def add(name, score):
        rows.setdefault(name, []).append(score)

    for seed in range(args.seeds):
        img1, img2, gt = make_synthetic_pair((args.height, args.width),
                                             n_targets=args.targets, seed=seed)
        a, b = lee_filter(img1, 5), lee_filter(img2, 5)

        # Global Otsu baselines on single difference images.
        lr = np.abs(log_ratio(a, b))
        add("baseline |log-ratio| + Otsu",
            evaluate_objects(morphological_opening(lr > otsu_threshold(lr)), gt, 7, 11))
        ncb = ncb_image(a, b, 3)
        add("baseline NCB + Otsu",
            evaluate_objects(morphological_opening(ncb > otsu_threshold(ncb)), gt, 7, 11))

        t = time.time()
        fr = fusion_change_detection(img1, img2)
        t_f = time.time() - t
        add("fusion (proposed)", evaluate_objects(fr.change_map, gt, 7, 11))
        fr2 = fusion_change_detection(img1, img2, tail_estimation=False)
        add("fusion, full-range GG estimate", evaluate_objects(fr2.change_map, gt, 7, 11))

        t = time.time()
        sr = superpixel_change_detection(img1, img2)
        t_s = time.time() - t
        add("superpixel two-stage (proposed)", evaluate_objects(sr.change_map, gt, 7, 20))
        add("superpixel coarse only",
            evaluate_objects(morphological_opening(sr.coarse_map), gt, 7, 20))
        rr = superpixel_change_detection(img1, img2, stages="refinement")
        add("superpixel refinement only", evaluate_objects(rr.change_map, gt, 7, 20))
        print(f"seed {seed}: fusion {t_f:.2f} s, superpixel {t_s:.2f} s "
              f"({sr.labels.max() + 1} superpixels, "
              f"{len(sr.mixture_selected)} refined)")

        if args.plot and seed == 0:
            _plot(args.plot, img1, img2, gt, fr, sr)

    print(f"\n{'method':34s} {'P':>7s} {'R':>7s} {'F1':>7s}   (summed over "
          f"{args.seeds} scenes)")
    for name, scores in rows.items():
        s = _sum(scores)
        print(f"{name:34s} {s.precision:7.4f} {s.recall:7.4f} {s.f1:7.4f}")


def _plot(path, img1, img2, gt, fr, sr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    db = lambda x: 10 * np.log10(x + 1e-6)
    panels = [
        (db(img1), "T1 (dB)"), (db(img2), "T2 (dB)"), (gt, "ground truth"),
        (fr.ncb, "NCB"), (fr.fuzzy, "fuzzified log-ratio"), (fr.fused, "fused"),
        (fr.change_map, "fusion change map"), (sr.change_map, "superpixel change map"),
    ]
    for a, (im, title) in zip(ax.ravel(), panels):
        a.imshow(im, cmap="gray")
        a.set_title(title)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
