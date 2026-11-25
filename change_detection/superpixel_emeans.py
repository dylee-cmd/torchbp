"""Superpixel classification change detection with e-means clustering.

Improved variant of :mod:`change_detection.superpixel_classification`
(G. Lee and K.-T. Kim, IEEE JSTARS 2026). The coarse three class
(HU / HeU / changed) clustering of superpixel features uses e-means
(Bessa et al., "e-means: outperforming k-means (in one-tenth the epochs)",
arXiv:2611.09501) through the ``EMeans`` estimator of
https://github.com/sacredvoid/emeans instead of Lloyd's k-means.

What changes compared with the paper's setting:

- **e-means coarse clustering.** Mini-batch k-means with exponential moving
  average centroids, a cosine step schedule and a final full-pass mean, with
  k-means++ seeding on a subsample.
- **Consensus of restarts.** The paper keeps the lowest inertia of 10 k-means
  restarts. Here each of ``n_init`` e-means runs is named by its cluster
  means (changed / HU / HeU), and every superpixel takes the class most runs
  agree on. This removes the dependence on a single restart.
- **Finer superpixels.** The default SLIC spacing is 20 instead of 40.
  Smaller superpixels keep small targets from being diluted by background
  (the main failure mode of the coarse stage), at the cost of many more
  superpixels to cluster, which e-means handles cheaply.

The refinement stage (EM mixture models with BIC selection) is unchanged.

On harder synthetic scenes (3 x 5 pixel targets at 8 dB, strongly
decorrelated clutter, 6 scenes of 20 targets;
``python -m change_detection.demo --hard --targets 20 --seeds 6``) the
object-level F1 changes as follows:

==========================================  ======
configuration                               F1
==========================================  ======
paper setting (k-means, S = 40)             0.70
k-means, S = 20                             0.76
k-means consensus, S = 20                   0.76
e-means consensus, S = 20 (this module)     0.78
==========================================  ======

The finer superpixel scale accounts for most of the gain. e-means mainly
raises precision (0.88 -> 0.92 against k-means consensus).
"""
from typing import Optional

import numpy as np

from .common import pca
from .superpixel_classification import (
    HU, HEU, CHANGED, SuperpixelResult, superpixel_change_detection,
)

__all__ = ["emeans_coarse_classification", "superpixel_emeans_change_detection"]


def _require_emeans():
    try:
        from emeans import EMeans
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "superpixel_emeans needs the emeans package: "
            "pip install git+https://github.com/sacredvoid/emeans") from e
    return EMeans


def _name_clusters(labels: np.ndarray, mean_ncb: np.ndarray, k: int) -> np.ndarray:
    """Map cluster ids to classes by the cluster-wise mean NCB: highest ->
    changed, lowest -> HU, the remaining one -> HeU."""
    means = np.array([mean_ncb[labels == j].mean() if (labels == j).any() else -np.inf
                      for j in range(k)])
    names = [HU, HEU, CHANGED][-k:] if k < 3 else [HU, HEU, CHANGED]
    cls = np.empty_like(labels)
    for rank, j in enumerate(np.argsort(means)):
        cls[labels == j] = names[rank]
    return cls


def emeans_coarse_classification(features: np.ndarray, seed: Optional[int] = 0,
                                 n_init: int = 10, consensus: bool = True,
                                 n_epochs: float = 3, var_ratio: float = 0.95,
                                 **emeans_kwargs) -> np.ndarray:
    """Three class e-means clustering of superpixel features.

    Features are standardized and decorrelated with PCA as in the paper,
    then clustered ``n_init`` times with ``EMeans(n_clusters=3)`` and
    seeds ``seed, seed + 1, ...``. Each run's clusters are named by their
    mean NCB. ``consensus=True`` assigns every superpixel the class most
    runs agree on (ties go to the lowest inertia run); ``False`` keeps the
    lowest inertia run like the paper's k-means restarts.

    Extra keyword arguments are passed to ``EMeans`` (e.g. ``batch_size``,
    ``init``).
    """
    EMeans = _require_emeans()
    X = np.asarray(features, dtype=np.float64)
    std = X.std(0)
    Z = pca((X - X.mean(0)) / np.where(std > 0, std, 1.0), var_ratio)
    k = min(3, X.shape[0])
    runs = []
    for i in range(n_init):
        rs = None if seed is None else seed + i
        km = EMeans(n_clusters=k, n_epochs=n_epochs, random_state=rs,
                    **emeans_kwargs).fit(Z)
        runs.append((km.inertia_, _name_clusters(km.labels_, X[:, 0], k)))
    runs.sort(key=lambda r: r[0])
    if not consensus or n_init == 1:
        return runs[0][1]
    classes = np.array([HU, HEU, CHANGED])
    V = np.stack([r[1] for r in runs])  # ordered by inertia
    counts = np.stack([(V == c).sum(0) for c in classes])  # [3, n_superpixels]
    winners = counts == counts.max(0)
    majority = classes[np.argmax(counts, axis=0)]
    # Ties: keep the lowest inertia run's class when it is among the winners.
    first = runs[0][1]
    first_wins = winners[np.searchsorted(classes, first), np.arange(len(first))]
    tied = winners.sum(0) > 1
    return np.where(tied & first_wins, first, majority)


def superpixel_emeans_change_detection(img1, img2, *, slic_spacing: int = 20,
                                       n_init: int = 10, consensus: bool = True,
                                       n_epochs: float = 3,
                                       seed: Optional[int] = 0,
                                       **kwargs) -> SuperpixelResult:
    """Superpixel classification change detection with e-means coarse
    clustering (see the module docstring).

    Parameters
    ----------
    img1, img2 : array_like
        Coregistered SAR images at T1 and T2.
    slic_spacing : int
        SLIC initialization spacing (20; the paper uses 40).
    n_init, consensus, n_epochs : int, bool, float
        e-means restarts, consensus voting and epochs per run.
    seed : int
        Base random seed of the e-means runs.
    **kwargs
        Any other argument of
        :func:`change_detection.superpixel_change_detection`.
    """
    def classify(features, seed=seed):
        return emeans_coarse_classification(features, seed=seed, n_init=n_init,
                                            consensus=consensus, n_epochs=n_epochs)

    return superpixel_change_detection(img1, img2, slic_spacing=slic_spacing,
                                       seed=seed, coarse_classifier=classify,
                                       **kwargs)
