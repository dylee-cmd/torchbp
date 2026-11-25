# Unsupervised SAR change detection of small objects

Implementations of two papers by G. Lee and K.-T. Kim (IEEE JSTARS, vol. 19, 2026):

| Module | Paper |
| --- | --- |
| `superpixel_classification.py` | *Unsupervised SAR Change Detection of Small Objects via Superpixel Classification*, pp. 22726–22740 |
| `fusion.py` | *Unsupervised SAR Change Detection of Small Objects via Fusion of Difference Images*, pp. 10394–10409 |
| `superpixel_emeans.py` | Improved superpixel classification variant with [e-means](https://github.com/sacredvoid/emeans) coarse clustering |
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
