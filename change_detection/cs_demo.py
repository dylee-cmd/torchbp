"""Coherent scatterer change detection on simulated SLC time series.

    python -m change_detection.cs_demo [--seeds 3] [--plot out.png]

For every scene, the detected change clusters are matched to the simulated
objects: a cluster is correct when its center lies inside a changed object
(5 pixel tolerance) and its lifetime [a, b] equals the object's. Reported:
object-level precision / recall, lifetime and category accuracy, mask IoU
of the segmented changes, and whether the transient (snow) building and
the vegetation produced false changes.
"""
import argparse

import numpy as np

from .coherent_scatterers import cs_change_detection, categorize
from .cs_simulation import make_slc_time_series

#: Expected category of every simulated object (delta_t = 30 days).
EXPECTED = {
    "new building": "new",
    "demolished building": "removed",
    "container stack": "moving",
    "parked aircraft": "moving",
}


def evaluate(scene, res):
    changed = scene.change_objects()
    matched = set()
    tp = fp = cat_ok = 0
    ious = []
    for c, cat in zip(res.clusters, res.categories):
        hit = [o for o in changed if o.contains(*c.center, pad=5)
               and (o.a, o.b) == (c.a, c.b) and o.name not in matched]
        if hit:
            o = hit[0]
            matched.add(o.name)
            tp += 1
            cat_ok += EXPECTED.get(o.name) == cat
            r0, c0, h, w = o.rect
            gt = np.zeros_like(c.mask)
            gt[r0:r0 + h, c0:c0 + w] = True
            ious.append((c.mask & gt).sum() / max((c.mask | gt).sum(), 1))
        else:
            fp += 1
    fn = len(changed) - len(matched)
    # False changes on the transient (snow) building or in vegetation.
    snow = [o for o in scene.objects if o.transient is not None]
    snow_fa = sum(1 for c in res.clusters for o in snow if o.contains(*c.center, pad=5))
    veg_fa = sum(1 for c in res.clusters
                 if scene.vegetation[int(c.center[0]), int(c.center[1])])
    return dict(tp=tp, fp=fp, fn=fn, cat_ok=cat_ok, ious=ious,
                snow_fa=snow_fa, veg_fa=veg_fa, missed=sorted(
                    o.name for o in changed if o.name not in matched))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--r", type=int, default=1, help="CD metric transient robustness")
    ap.add_argument("--plot", type=str, default=None)
    args = ap.parse_args(argv)
    tot = dict(tp=0, fp=0, fn=0, cat_ok=0, snow_fa=0, veg_fa=0)
    ious = []
    for seed in range(args.seeds):
        sc = make_slc_time_series(seed=seed)
        res = cs_change_detection(sc.stack, sc.times, band_fraction=0.8,
                                  cs_threshold=0.05, r=args.r, delta_t=30.0)
        e = evaluate(sc, res)
        for k in tot:
            tot[k] += e[k]
        ious += e["ious"]
        print(f"seed {seed}: {len(res.clusters)} change clusters, TP={e['tp']} FP={e['fp']} "
              f"FN={e['fn']} {('missed: ' + ', '.join(e['missed'])) if e['missed'] else ''}")
        for c, cat in zip(res.clusters, res.categories):
            print(f"    images [{c.a}, {c.b}]  {c.n_cs:4d} CSs  {c.area_m2:7.0f} m^2  "
                  f"center {tuple(round(v) for v in c.center)}  -> {cat}")
        if args.plot and seed == 0:
            _plot(args.plot, sc, res)
    p = tot["tp"] / max(tot["tp"] + tot["fp"], 1)
    r = tot["tp"] / max(tot["tp"] + tot["fn"], 1)
    print(f"\nobject precision {p:.3f}, recall {r:.3f}; category correct "
          f"{tot['cat_ok']}/{tot['tp']}; mean mask IoU {np.mean(ious):.3f}; "
          f"false changes on the snow building {tot['snow_fa']}, in vegetation {tot['veg_fa']}")


def _plot(path, sc, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .coherent_scatterers import coherence
    n = len(sc.stack)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    db = lambda s: 20 * np.log10(np.abs(s) + 1e-3)
    ax[0, 0].imshow(db(sc.stack[-1]), cmap="gray", vmin=-10, vmax=30)
    ax[0, 0].set_title("last SLC amplitude (dB)")
    ax[0, 1].imshow(res.cs.sum(0), cmap="magma")
    ax[0, 1].set_title("number of images with a detected CS")
    colors = {"new": (1, 0, 0), "removed": (0, 0.8, 0), "moving": (1, 0.8, 0),
              "temporary": (0, 0.6, 1), "static": (0.5, 0.5, 0.5)}
    rgb = np.repeat(np.clip((db(sc.stack[-1]) + 10) / 40, 0, 1)[..., None], 3, -1) * 0.6
    for c, cat in zip(res.clusters, res.categories):
        rgb[c.mask] = colors[cat]
    ax[0, 2].imshow(rgb)
    ax[0, 2].set_title("segmented changes (red new, green removed, yellow moving)")
    for o in sc.objects:
        r0, c0, h, w = o.rect
        for a in ax[0]:
            a.add_patch(plt.Rectangle((c0, r0), w, h, fill=False, ec="c", lw=0.8))
    # Coherence matrices of a pixel of the snow building and of the new one.
    for a, name in zip(ax[1, :2], ["static building B (snow)", "new building"]):
        o = next(o for o in sc.objects if o.name == name)
        r0, c0, h, w = o.rect
        sub = res.cs[:, r0:r0 + h, c0:c0 + w].sum(0)
        pr, pc = np.unravel_index(np.argmax(sub), sub.shape)
        pr += r0
        pc += c0
        M = np.ones((n, n))
        for j in range(n):
            for k in range(j + 1, n):
                g = coherence(sc.stack[j][pr - 10:pr + 11, pc - 10:pc + 11],
                              sc.stack[k][pr - 10:pr + 11, pc - 10:pc + 11])[10, 10]
                M[j, k] = M[k, j] = g
        im = a.imshow(M, vmin=0, vmax=1)
        a.set_title(f"coherence matrix, {name}")
        plt.colorbar(im, ax=a)
    ax[1, 2].plot(sc.times[:-1], res.f[:, :, :].reshape(n - 1, -1)[:, res.change_map.ravel()].mean(1),
                  "o-", label="changed pixels")
    ax[1, 2].plot(sc.times[:-1], res.f.reshape(n - 1, -1)[:, (res.cs.all(0)).ravel()].mean(1),
                  "s-", label="CSs present in all images")
    ax[1, 2].axhline(0.5, color="k", ls="--")
    ax[1, 2].set_title("mean CD metric f_i")
    ax[1, 2].set_xlabel("t_i (days)")
    ax[1, 2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
