# Unsupervised SAR change detection

Implementations of three papers. Two are by G. Lee and K.-T. Kim (IEEE JSTARS, vol. 19, 2026) on small objects in image pairs, and one is by C. Villamil Lopez and U. Stilla (IEEE JSTARS, vol. 15, 2022) on SLC time series:

| Module | Paper |
| --- | --- |
| `superpixel_classification.py` | *Unsupervised SAR Change Detection of Small Objects via Superpixel Classification*, pp. 22726–22740 |
| `fusion.py` | *Unsupervised SAR Change Detection of Small Objects via Fusion of Difference Images*, pp. 10394–10409 |
| `superpixel_emeans.py` | Improved superpixel classification variant with [e-means](https://github.com/sacredvoid/emeans) coarse clustering |
| `coherent_scatterers.py` | *Monitoring of Construction Activity by Change Detection on SAR Time Series Using Coherent Scatterers*, pp. 7499–7514 (2022) |
| `cs_amplitude.py`, `examples/amplitude_pair_example.py` | Amplitude-only image-pair variant of the coherent-scatterer method and a runnable example |
| `cs_simulation.py`, `cs_demo.py` | SLC time-series simulator and demo for the coherent-scatterer method |
| `common.py` | Shared parts: Lee filter, log-ratio and NCB (Berger) difference images, Otsu, morphological opening, SLIC, k-means++, PCA, DBSCAN, object-level evaluation, synthetic data |
| `demo.py` | Runs both methods and their ablations on synthetic scenes |

Dependencies: NumPy and SciPy only (SLIC, k-means++, PCA and DBSCAN are implemented here).

## Usage

```python
from change_detection import (
    superpixel_change_detection, fusion_change_detection,
    evaluate_objects, make_synthetic_pair,
)

img1, img2, gt = make_synthetic_pair((400, 600), n_targets=16, seed=0)

# Paper 1: superpixel classification
res = superpixel_change_detection(img1, img2)            # Lee 5x5, NCB 11x11, SLIC S=40
print(evaluate_objects(res.change_map, gt, eps=7, min_pts=20))

# Paper 2: fusion of difference images
res = fusion_change_detection(img1, img2)                # Lee 5x5, NCB 3x3
print(evaluate_objects(res.change_map, gt, eps=7, min_pts=11))
```

Inputs are coregistered SAR images (intensity, amplitude or complex; complex inputs use the magnitude). The result objects also hold the intermediate products (NCB, superpixel labels and classes, fuzzified log-ratio, fused image, thresholds).

Demo with ablations and an optional figure:

```bash
python -m change_detection.demo --seeds 4 --plot demo.png
```

### e-means variant

`superpixel_emeans.py` needs the [emeans](https://github.com/sacredvoid/emeans) package, which brings scikit-learn and threadpoolctl with it:

```bash
pip install git+https://github.com/sacredvoid/emeans    # or: pip install -e .[emeans]
```

```python
from change_detection import superpixel_emeans_change_detection
res = superpixel_emeans_change_detection(img1, img2)     # SLIC S=20, 10 e-means runs, consensus
```

It changes only the coarse stage of the superpixel method; the EM/BIC refinement is unchanged.

- **e-means clustering.** The 3-class clustering of the standardized, PCA-decorrelated superpixel features uses `EMeans(n_clusters=3)` (mini-batch k-means with EMA centroids, a cosine step schedule and a last-pass mean; Bessa et al., arXiv:2611.09501) instead of Lloyd's k-means.
- **Consensus of restarts.** Each of the 10 runs names its clusters by mean NCB. Every superpixel then takes the class most runs agree on; ties go to the lowest-inertia run. `consensus=False` keeps only the lowest-inertia run, as in the paper.
- **Finer superpixels.** The default SLIC spacing is 20 (the paper uses 40), so small targets are less diluted by background inside a superpixel. This gives about 4× more superpixels to cluster.

Object-level F1 on synthetic scenes (`python -m change_detection.demo`):

| Scenes | Paper setting (k-means, S=40) | e-means variant |
| --- | --- | --- |
| standard, `--seeds 4` | 0.96 | 1.00 |
| hard (3×5 px targets at 8 dB, decorrelated clutter), `--hard --targets 20 --seeds 6` | 0.70 (P 0.80, R 0.63) | 0.78 (P 0.92, R 0.68) |

Where the gain on the hard scenes comes from:

| Configuration | F1 |
| --- | --- |
| k-means, S=20 | 0.76 |
| k-means consensus, S=20 | 0.76 |
| e-means consensus, S=20 | 0.78 |

So most of the improvement comes from the finer superpixel scale. e-means itself mainly raises precision (0.88 → 0.92). With only a few hundred superpixels, e-means is not faster than k-means; its speed advantage only appears for much larger sample counts.

### Coherent scatterer change detection on SLC time series

```python
from change_detection import cs_change_detection, make_slc_time_series

scene = make_slc_time_series(seed=0)                     # 8 SLCs, 11 days apart
res = cs_change_detection(scene.stack, scene.times, band_fraction=0.8,
                          cs_threshold=0.05, r=1, delta_t=30.0)
for c, cat in zip(res.clusters, res.categories):
    print(c.a, c.b, c.n_cs, cat)          # lifetime [a, b] (image indices), category
res.change_map                            # segmented changes
```

Inputs are coregistered complex SLC images `[azimuth, range]`. The method looks for changes in the strong point scatterers (CSs) of man-made objects instead of amplitude or coherence over all pixels, so vegetation and water are ignored.

1. **CS detection.** Range sub-looks are formed (10, 75 % overlap). A pixel contains a CS when its consecutive sub-look phase differences have a variance below `T`. Of those pixels, a CS is kept only if its total sub-look phase change is at most π, which removes the spread of the CS into neighbouring pixels.
2. **CD metric (eq. 2).** `f_i = max γ_{j,k}` over `j ∈ [i-r, i]`, `k ∈ [i+1, i+1+r]`. With `r ≥ 1` a transient change (e.g. snow in one image) does not look like a change.
3. **Pixelwise lifetimes.**
   - A forward/backward consistency check carries CS presence across high-coherence intervals (`f ≥ γ_t = 0.5`), which corrects CS detection false negatives.
   - The CS lifetime `[a, b]` is the run of images where the CS is present with high coherence between consecutive images.
   - CSs detected in fewer than `k (b - a + 1)` of their images are discarded (eq. 3, `k = 0.1`).
4. **Clustering.** DBSCAN in meters (`ε = 15 m`, `minPts = 20`), separately for each lifetime `(a, b)`. Clusters with fewer than 30 CSs or a convex hull under 20 m² are dropped.
5. **Segmentation.** This runs inside the cluster rectangle enlarged by 50 %.
   - Objects present in several images: the pixel changes in `(t_{a-1}, t_a)` and in `(t_b, t_{b+1})` and stays coherent during `[t_a, t_b]`.
   - Objects present in a single image: amplitude constraints instead (|ΔA| ≥ 3 dB towards both neighbouring images, `A ≥ -15 dB`, after despeckling).
   - Then a closing (radius 5) and removal of components that lie mostly outside the cluster.
6. **Change analysis.** Durations use the lower bound `t_b - t_a`. The categories are:
   - new (appeared, still present, at least `ΔT`);
   - removed;
   - moving (shorter than `ΔT`);
   - temporary;
   - static.

**Two amplitude images (no phase).** `cs_change_detection` needs SLC phase twice: for sub-look CS detection and for coherence. For amplitude-only pairs, `cs_amplitude.py` swaps in amplitude-based substitutes and reuses the rest of the pipeline through the `cs_maps=` / `metric=` arguments:
- **CS proxy:** bright point scatterers, i.e. local maxima at least 10 dB above the local median intensity.
- **CD metric:** amplitude similarity `1 - NCB`.

```python
import numpy as np
from change_detection.cs_amplitude import amplitude_pair_change_detection

res = amplitude_pair_change_detection(np.load("before.npy"), np.load("after.npy"),
                                      spacing=(0.5, 0.5))   # pixel spacing (az, rg) in m
for c, cat in zip(res.clusters, res.categories):            # "appeared" / "disappeared"
    print(cat, c.n_cs, c.center)
```

The runnable example is `python -m change_detection.examples.amplitude_pair_example [--img1 a.npy --img2 b.npy --spacing AZ RG] [--plot out.png]`. Without input files it uses images 2 and 4 of the simulated series.
- **Simulated pair, 3 seeds:** the demolished building, the new building and the container stack are always found with the right type, and there are no false alarms in vegetation.
- **Snow-covered building:** reported as a change in 2 of 3 seeds. A pair cannot tell a transient from a lasting change; that needs a third image (the `r ≥ 1` metric).

Further limitations without phase:
- bright speckle peaks can pass as point scatterers (about 1e-3 of single-look clutter at 10 dB);
- changes that keep the amplitude (sub-pixel motion) are invisible.

`pair_change_classes` gives the image-pair interpretation of Table I: irrelevant, unchanged, disappeared, appeared, changed/moved.

**Results on simulated series** (`python -m change_detection.cs_demo --seeds 4`). Each scene has 8 images, 2 static buildings (one covered by snow in image 2), a new building, a demolished building, a container stack and a parked aircraft, plus vegetation and water. Ground decorrelates with the temporal baseline.

| CD metric | Object precision | Recall | Category correct | Mask IoU | False changes on the snow building |
| --- | --- | --- | --- | --- | --- |
| `r = 1` | 1.00 | 1.00 | 16/16 | 0.49 | 0 |
| `r = 0` (consecutive pairs only) | 0.62 | 1.00 | 16/16 | 0.50 | 10 |

The `r = 0` row reproduces the paper's point that `r > 0` suppresses transient changes (Fig. 8). The coherence matrices of a snow pixel and a new-building pixel (`--plot`) show the same patterns as the paper's Figs. 6 and 7. The masks bleed a few pixels beyond the objects because of the coherence window and the closing, which keeps the IoU around 0.5.

**Notes**
- **CS threshold.** `T = 0.125` (the paper's value for 300 MHz TerraSAR-X with 10 sub-looks) gives about 1.7 % false CSs in the simulated clutter. With 75 % sub-look overlap, consecutive sub-look phases of distributed clutter are strongly correlated. The paper says `T` trades false positives against misses and should be chosen for few false positives; the demo uses `T = 0.05` (0.2 % false CSs). The function default stays 0.125.
- **Coherence window.** The default is 9 × 9 pixels for the simulator's square pixels. The paper uses 9 × 23 for its anisotropic TerraSAR-X pixels.
- **Not reproduced.** The paper's custom full-resolution despeckling filter is replaced by a Lee filter plus 3 × 3 multilooking. The 49-image TerraSAR-X Munich data set is not available here.

## Method summaries

### Superpixel classification (paper 1)

1. Lee speckle filter (5×5), then the NCB image `1 - 2Σ|I1||I2| / (Σ|I1|² + Σ|I2|²)` over an 11×11 window.
2. SLIC superpixels of the NCB image (initial spacing 40).
3. **Coarse stage.** Each superpixel gets the features mean, variance, third quartile (Q3), kurtosis and `r25`: the number of pixels above Q3 divided by the area of their minimal convex hull. The features are standardized and decorrelated with PCA, then clustered by k-means (K = 3, k-means++, 10 restarts). Clusters are named by their mean NCB: highest → changed, lowest → homogeneous unchanged (HU), the other → heterogeneous unchanged (HeU). Changed superpixels become a pixel mask through Otsu's threshold of their NCB values.
4. **Refinement stage.** Each unchanged superpixel is fitted two ways:
   - a single distribution: truncated lognormal on (0, 1) for HU, truncated gamma for HeU;
   - a mixture `(1-π) f_u + π f_c` of that background with a reflected truncated lognormal `f_c` (lognormal of `1 - x`).

   The mixture is fitted by EM with MAP updates: a Dirichlet prior on π (eq. 5), an inverse-gamma prior on the change component's variance, a variance floor, and the change mean of `1-x` constrained to at most 0.5. It is kept when its BIC is not larger than the single model's and its change component lies above the background. Its pixels with posterior change probability > 0.5 (the intersection of the weighted densities) are detected.
5. The final map is the union of both stages, followed by a 3×3 opening.

`stages="coarse"` and `stages="refinement"` give the paper's stage ablations.

### Fusion of difference images (paper 2)

1. Lee filter (5×5), log-ratio `log(|I2|/|I1|)` and NCB over a 3×3 window.
2. The log-ratio histogram is modeled with three generalized Gaussian (GG) classes: decreasing change, unchanged, increasing change. Shapes are estimated from the moment ratio with a lookup table.
   - The threshold pair `τ1 < 0 < τ2` minimizes the expected cost `J = Σ h(x) c(x)` (eq. 10–13).
   - The change classes use the tail estimators (eq. 17, 18): symmetric GGs centered at `τ1`/`τ2`, estimated from their outer halves.
   - The unchanged class is estimated from the residual histogram (eq. 19). Candidates with a non-positive unchanged prior are discarded.
3. A fuzzy membership function from the class posteriors (eq. 6) is split at the Bayes boundaries `k1`, `k2` (eq. 7, 8, outermost roots). It maps the log-ratio to [0, 1].
4. The fused image is the Hadamard product of the fuzzified log-ratio and the NCB (eq. 21). It is thresholded with Otsu's method and cleaned with a 3×3 opening.

`tail_estimation=False` uses the conventional full-range estimates (eq. 16) instead.

### Object-level evaluation

As in both papers, detected pixels are clustered with DBSCAN (radius 7; minPts 20 for paper 1, 11 for paper 2). Each cluster center is matched one-to-one with a ground-truth object (an 8-connected component of the GT mask):

- TP: the center falls inside a not-yet-matched object.
- FP: any other cluster.
- FN: an object that no cluster matched.

## Deviations from the papers

- **Threshold-pair cost (paper 2).** Eq. (12) writes the cost as `-2 ln P(ω|x)`, the class *posterior*. Taken literally, the expected posterior cost is minimized when the classes stop overlapping. That pushes both change classes to the extreme bins of the histogram, and on the synthetic scenes it detects none of the targets. The default `cost_kind="joint"` instead uses the Kittler–Illingworth minimum-error cost `-2 ln(P(ω) p(x|ω))`, the KI algorithm the paper says it follows. `cost_kind="posterior"` reproduces eq. (12) for reference.
- **Guards on the tail estimators (paper 2).** A tail of a single histogram bin has zero variance and an unbounded density. Tails with fewer than `min_tail_pixels` (10) samples are therefore rejected, and all standard deviations are floored at one bin width.
- **Unspecified constants (paper 1).** The paper gives no prior hyperparameters or variance floor. This implementation uses:
  - Dirichlet prior `(2, 2)` on π;
  - inverse-gamma prior `(1, 0.01)` on the change component's log variance;
  - log-σ floor 0.1;
  - SLIC compactness 0.1 on the [0, 1] NCB scale.

  All are function arguments.
- **M-step approximations (paper 1).** The EM M-step uses closed-form weighted moment / Minka gamma updates that ignore the truncation. The likelihoods used for EM and BIC do include the truncation.
- **Fallback background model (paper 1).** In the `stages="refinement"` ablation there is no coarse HU/HeU label, so the background family with the lower single-model BIC is used.

## Tests

```bash
pytest tests/test_change_detection.py
```
