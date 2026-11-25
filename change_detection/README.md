# Unsupervised SAR change detection of small objects

Implementations of two papers by G. Lee and K.-T. Kim (IEEE JSTARS, vol. 19, 2026):

| Module | Paper |
| --- | --- |
| `superpixel_classification.py` | *Unsupervised SAR Change Detection of Small Objects via Superpixel Classification*, pp. 22726–22740 |
| `fusion.py` | *Unsupervised SAR Change Detection of Small Objects via Fusion of Difference Images*, pp. 10394–10409 |
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
