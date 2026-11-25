"""Unsupervised SAR change detection of small objects via superpixel
classification.

Implementation of G. Lee and K.-T. Kim, "Unsupervised SAR Change Detection
of Small Objects via Superpixel Classification", IEEE J. Sel. Topics Appl.
Earth Observ. Remote Sens., vol. 19, pp. 22726-22740, 2026.

Pipeline (Fig. 6 of the paper):

1. Lee speckle filter on both images and the noncoherent change detection
   image with Berger's operator (NCB, eq. (1)).
2. SLIC superpixels of the NCB image.
3. Coarse classification: superpixel features (mean, variance, third
   quartile, kurtosis and the compactness r25 of the pixels above the third
   quartile) are standardized, decorrelated with PCA and clustered with
   k-means (K = 3, k-means++, 10 restarts) into homogeneous unchanged (HU),
   heterogeneous unchanged (HeU) and changed superpixels. Changed
   superpixels are refined to a pixel mask with Otsu's threshold of their
   NCB values.
4. Refined classification: every unchanged superpixel is fitted with a
   single distribution (truncated lognormal for HU, truncated gamma for
   HeU) and with a two component mixture of that background and a
   reflected truncated lognormal change component (EM with MAP updates,
   eq. (3)-(5)). The mixture is kept when its BIC (eq. (6)) is not larger,
   and its pixels whose posterior change probability exceeds 1/2 (the
   intersection of the weighted component densities) are detected.
5. The change map is the union of both stages followed by a morphological
   opening.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import optimize, special, stats
from scipy.spatial import ConvexHull, QhullError

from .common import (
    as_intensity, lee_filter, ncb_image, otsu_threshold, morphological_opening,
    slic, kmeans, pca,
)

__all__ = [
    "HU", "HEU", "CHANGED",
    "SuperpixelResult",
    "superpixel_features",
    "coarse_classification",
    "fit_single",
    "fit_mixture",
    "superpixel_change_detection",
]

#: Superpixel classes of the coarse stage.
HU, HEU, CHANGED = 0, 1, 2

_EPS = 1e-4
_LOG2PI = np.log(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Superpixel features and coarse classification
# ---------------------------------------------------------------------------

def _r25(ys: np.ndarray, xs: np.ndarray) -> float:
    """Ratio of the number of pixels to the area of their minimal convex
    hull (pixel footprints, i.e. the hull of the pixel corners)."""
    n = ys.size
    if n == 0:
        return 0.0
    off = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    pts = (np.stack([ys, xs], 1)[:, None, :] + off[None]).reshape(-1, 2)
    try:
        area = ConvexHull(pts).volume
    except (QhullError, ValueError):
        area = float(n)
    return float(n / max(area, n))


def superpixel_features(ncb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per superpixel feature vectors [mean, variance, Q3, kurtosis, r25]."""
    nlab = int(labels.max()) + 1
    feats = np.zeros((nlab, 5))
    flat_l = labels.ravel()
    flat_v = ncb.ravel()
    order = np.argsort(flat_l, kind="stable")
    bounds = np.searchsorted(flat_l[order], np.arange(nlab + 1))
    W = labels.shape[1]
    for k in range(nlab):
        idx = order[bounds[k]:bounds[k + 1]]
        v = flat_v[idx]
        if v.size == 0:
            continue
        m = v.mean()
        var = v.var()
        q3 = np.percentile(v, 75)
        kurt = ((v - m) ** 4).mean() / var ** 2 if var > 0 else 0.0
        hi = idx[v > q3]
        feats[k] = (m, var, q3, kurt, _r25(hi // W, hi % W))
    return feats


def coarse_classification(features: np.ndarray, seed: Optional[int] = 0,
                          n_init: int = 10, var_ratio: float = 0.95) -> np.ndarray:
    """Three class k-means of the standardized, PCA decorrelated superpixel
    features. Cluster identities follow the cluster-wise mean NCB: highest
    mean -> changed, lowest -> HU, the remaining one -> HeU.
    """
    X = features.copy()
    std = X.std(0)
    X = (X - X.mean(0)) / np.where(std > 0, std, 1.0)
    Z = pca(X, var_ratio)
    k = min(3, X.shape[0])
    lab, _ = kmeans(Z, k, n_init=n_init, seed=seed)
    means = np.array([features[lab == j, 0].mean() if (lab == j).any() else -np.inf
                      for j in range(k)])
    order = np.argsort(means)
    cls = np.empty_like(lab)
    names = [HU, HEU, CHANGED][-k:] if k < 3 else [HU, HEU, CHANGED]
    for rank, j in enumerate(order):
        cls[lab == j] = names[rank]
    return cls


# ---------------------------------------------------------------------------
# Distributions on (0, 1)
# ---------------------------------------------------------------------------

def _logpdf_tlognorm(x, mu, sigma):
    """Lognormal truncated to (0, 1)."""
    lx = np.log(x)
    logz = special.log_ndtr(-mu / sigma)
    return -lx - np.log(sigma) - 0.5 * _LOG2PI - (lx - mu) ** 2 / (2 * sigma ** 2) - logz


def _logpdf_tgamma(x, k, theta):
    """Gamma truncated to (0, 1)."""
    z = special.gammainc(k, 1.0 / theta)
    logz = np.log(max(z, 1e-300))
    return ((k - 1) * np.log(x) - x / theta - k * np.log(theta)
            - special.gammaln(k) - logz)


def _logpdf_background(kind, x, p):
    if kind == "lognormal":
        return _logpdf_tlognorm(x, p[0], p[1])
    return _logpdf_tgamma(x, p[0], p[1])


def _logpdf_change(x, p):
    """Reflected truncated lognormal: lognormal of y = 1 - x on (0, 1)."""
    return _logpdf_tlognorm(1.0 - x, p[0], p[1])


def _lognormal_moments(x, w=None):
    lx = np.log(x)
    if w is None:
        w = np.ones_like(x)
    sw = max(w.sum(), 1e-12)
    mu = (w * lx).sum() / sw
    var = (w * (lx - mu) ** 2).sum() / sw
    return mu, np.sqrt(max(var, 1e-6))


def _gamma_moments(x, w=None):
    """Weighted gamma fit (Minka's closed-form approximation of the ML
    shape)."""
    if w is None:
        w = np.ones_like(x)
    sw = max(w.sum(), 1e-12)
    m = (w * x).sum() / sw
    s = np.log(m) - (w * np.log(x)).sum() / sw
    s = max(s, 1e-8)
    k = (3.0 - s + np.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
    k = float(np.clip(k, 1e-2, 1e4))
    return k, m / k


def fit_single(x: np.ndarray, kind: str) -> Tuple[np.ndarray, float]:
    """Maximum likelihood fit of a truncated lognormal (``kind ==
    "lognormal"``) or truncated gamma (``"gamma"``) on (0, 1).

    Moment-based initialization refined by numerically maximizing the
    log-likelihood. Returns (params, log-likelihood).
    """
    if kind == "lognormal":
        p0 = np.array(_lognormal_moments(x))
        z0 = np.array([p0[0], np.log(p0[1])])

        def unpack(z):
            return np.array([z[0], np.exp(z[1])])
    else:
        p0 = np.array(_gamma_moments(x))
        z0 = np.log(p0)

        def unpack(z):
            return np.exp(z)

    def nll(z):
        v = -_logpdf_background(kind, x, unpack(z)).sum()
        return v if np.isfinite(v) else 1e300

    res = optimize.minimize(nll, z0, method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 400})
    p = unpack(res.x if res.fun <= nll(z0) else z0)
    return p, float(_logpdf_background(kind, x, p).sum())


@dataclass
class MixtureFit:
    pi: float
    theta_u: np.ndarray
    theta_c: np.ndarray
    loglik: float
    resp: np.ndarray
    n_iter: int


def fit_mixture(x: np.ndarray, kind: str, init_change: Optional[np.ndarray] = None,
                pi0: float = 0.2, alpha: Tuple[float, float] = (2.0, 2.0),
                ig_prior: Tuple[float, float] = (1.0, 0.01),
                sigma_floor: float = 0.1, max_mean_change: float = 0.5,
                max_iter: int = 200, tol: float = 1e-7) -> MixtureFit:
    """EM fit of ``(1 - pi) f_u(x) + pi f_c(x)`` (eq. (3)).

    ``f_u`` is the truncated background distribution of the superpixel
    class (``kind``) and ``f_c`` a reflected truncated lognormal change
    component. MAP updates: Dirichlet(``alpha``) prior on the mixture
    weight (eq. (5)) and an inverse-gamma(``ig_prior``) prior on the log
    variance of the change component. The change component is constrained
    to a log standard deviation of at least ``sigma_floor`` and to a mean of
    ``y = 1 - x`` of at most ``max_mean_change``.

    Parameters
    ----------
    init_change : ndarray of bool, optional
        Initial change pixel set for the moment initialization. Defaults to
        the pixels above the superpixel's Otsu threshold.
    """
    x = np.asarray(x, dtype=np.float64)
    M = x.size
    if init_change is None:
        init_change = x > otsu_threshold(x)
    if init_change.sum() < 3 or (~init_change).sum() < 3:
        init_change = x >= np.quantile(x, 0.9)
    y = 1.0 - x
    fit_u = _lognormal_moments if kind == "lognormal" else _gamma_moments
    th_u = np.array(fit_u(x[~init_change]))
    th_c = np.array(_lognormal_moments(y[init_change]))
    pi = pi0
    a1, a2 = alpha
    a0, b0 = ig_prior

    def constrain(th):
        mu, sig = th
        sig = max(sig, sigma_floor)
        mu = min(mu, np.log(max_mean_change) - 0.5 * sig ** 2)
        return np.array([mu, sig])

    th_c = constrain(th_c)
    prev = -np.inf
    it = 0
    for it in range(1, max_iter + 1):
        lu = np.log1p(-pi) + _logpdf_background(kind, x, th_u)
        lc = np.log(pi) + _logpdf_change(x, th_c)
        lse = np.logaddexp(lu, lc)
        resp = np.exp(lc - lse)
        ll = lse.sum()
        # M-step.
        n2 = resp.sum()
        pi = float(np.clip((n2 + a2 - 1.0) / (M + a1 + a2 - 2.0), 1e-6, 1 - 1e-6))
        ly = np.log(y)
        sw = max(n2, 1e-12)
        mu_c = (resp * ly).sum() / sw
        s2 = (resp * (ly - mu_c) ** 2).sum()
        var_c = (b0 + 0.5 * s2) / (a0 + 1.0 + 0.5 * n2)
        th_c = constrain(np.array([mu_c, np.sqrt(var_c)]))
        th_u = np.array(fit_u(x, 1.0 - resp))
        if abs(ll - prev) <= tol * max(1.0, abs(ll)):
            break
        prev = ll
    lu = np.log1p(-pi) + _logpdf_background(kind, x, th_u)
    lc = np.log(pi) + _logpdf_change(x, th_c)
    lse = np.logaddexp(lu, lc)
    return MixtureFit(pi, th_u, th_c, float(lse.sum()), np.exp(lc - lse), it)


def bic(loglik: float, k: int, n: int) -> float:
    """Bayesian information criterion, eq. (6)."""
    return -2.0 * loglik + k * np.log(n)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

@dataclass
class SuperpixelResult:
    change_map: np.ndarray
    coarse_map: np.ndarray
    refined_map: np.ndarray
    ncb: np.ndarray
    labels: np.ndarray
    classes: np.ndarray
    features: np.ndarray
    mixture_selected: Dict[int, MixtureFit] = field(default_factory=dict)


def superpixel_change_detection(img1, img2, *, lee_size: int = 5,
                                looks: float = 1.0, ncb_size: int = 11,
                                slic_spacing: int = 40,
                                slic_compactness: float = 0.1,
                                stages: str = "two_stage",
                                opening_size: int = 3,
                                min_refine_pixels: int = 30,
                                pi0: float = 0.2,
                                alpha: Tuple[float, float] = (2.0, 2.0),
                                seed: Optional[int] = 0) -> SuperpixelResult:
    """Two stage superpixel classification change detection.

    Parameters
    ----------
    img1, img2 : array_like
        Coregistered SAR images (intensity or complex) at times T1 and T2.
    lee_size : int
        Lee speckle filter window (0 or 1 disables the filter).
    looks : float
        Equivalent number of looks used by the Lee filter.
    ncb_size : int
        NCB sliding window size.
    slic_spacing, slic_compactness : int, float
        SLIC initialization spacing and compactness.
    stages : {"two_stage", "coarse", "refinement"}
        "coarse": coarse classification only. "refinement": the mixture
        refinement applied to every superpixel without the coarse
        classification (HU/HeU background chosen by the superpixel's
        BIC). "two_stage": the full method.
    opening_size : int
        Size of the final morphological opening.
    min_refine_pixels : int
        Superpixels with fewer samples are not refined.
    pi0, alpha : float, tuple
        EM initial mixture weight and Dirichlet prior.
    seed : int
        k-means random seed.
    """
    a = as_intensity(img1)
    b = as_intensity(img2)
    if lee_size and lee_size > 1:
        a = lee_filter(a, lee_size, looks)
        b = lee_filter(b, lee_size, looks)
    ncb = ncb_image(a, b, ncb_size)
    labels = slic(ncb, slic_spacing, slic_compactness)
    feats = superpixel_features(ncb, labels)
    nlab = feats.shape[0]

    flat_l = labels.ravel()
    order = np.argsort(flat_l, kind="stable")
    bounds = np.searchsorted(flat_l[order], np.arange(nlab + 1))
    flat_ncb = ncb.ravel()

    if stages == "refinement":
        classes = np.full(nlab, -1)
    else:
        classes = coarse_classification(feats, seed=seed)

    coarse = np.zeros(ncb.size, dtype=bool)
    refined = np.zeros(ncb.size, dtype=bool)
    selected = {}
    for k in range(nlab):
        idx = order[bounds[k]:bounds[k + 1]]
        v = flat_ncb[idx]
        if classes[k] == CHANGED:
            coarse[idx[v > otsu_threshold(v)]] = True
            continue
        if stages == "coarse" or v.size < min_refine_pixels:
            continue
        x = np.clip(v, _EPS, 1.0 - _EPS)
        if classes[k] == HU:
            kinds = ["lognormal"]
        elif classes[k] == HEU:
            kinds = ["gamma"]
        else:
            kinds = ["lognormal", "gamma"]
        best = None
        for kind in kinds:
            p_s, ll_s = fit_single(x, kind)
            b_s = bic(ll_s, 2, x.size)
            if best is None or b_s < best[0]:
                best = (b_s, kind)
        b_single, kind = best
        mix = fit_mixture(x, kind, pi0=pi0, alpha=alpha)
        b_mix = bic(mix.loglik, 5, x.size)
        # The change component must describe the upper tail.
        mean_c = 1.0 - np.exp(mix.theta_c[0] + 0.5 * mix.theta_c[1] ** 2)
        mean_u = (np.exp(mix.theta_u[0] + 0.5 * mix.theta_u[1] ** 2)
                  if kind == "lognormal" else mix.theta_u[0] * mix.theta_u[1])
        if b_mix <= b_single and mean_c > mean_u:
            det = mix.resp > 0.5
            if det.any():
                refined[idx[det]] = True
                selected[k] = mix

    shape = ncb.shape
    coarse = coarse.reshape(shape)
    refined = refined.reshape(shape)
    change = morphological_opening(coarse | refined, opening_size)
    return SuperpixelResult(change, coarse, refined, ncb, labels, classes, feats,
                            selected)
