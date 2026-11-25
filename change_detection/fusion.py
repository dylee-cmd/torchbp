"""Unsupervised SAR change detection of small objects via fusion of
difference images.

Implementation of G. Lee and K.-T. Kim, "Unsupervised SAR Change Detection
of Small Objects via Fusion of Difference Images", IEEE J. Sel. Topics
Appl. Earth Observ. Remote Sens., vol. 19, pp. 10394-10409, 2026.

Pipeline (Fig. 8 of the paper):

1. Lee speckle filter on both images.
2. Difference images: log-ratio (eq. (4)) and NCB with Berger's operator
   (eq. (5)).
3. The log-ratio histogram is modeled with three generalized Gaussian (GG)
   classes (decreasing change, unchanged, increasing change). The threshold
   pair (tau1, tau2) minimizes the Kittler-Illingworth style expected cost
   (eq. (10)-(14)); the class statistics come from the low-overlap tails
   (eq. (17)-(19)): the change classes are treated as symmetric GGs
   centered at tau1 and tau2 and estimated from their outer halves only.
4. A fuzzy membership function built from the class posteriors (eq. (6)),
   split at the Bayes decision boundaries k1 and k2 (eq. (7), (8)), maps
   the log-ratio image to [0, 1] (eq. (20)).
5. The fused image is the Hadamard product of the fuzzified log-ratio and
   the NCB images (eq. (21)), thresholded with Otsu's method and cleaned
   with a morphological opening.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import special

from .common import (
    as_intensity, lee_filter, log_ratio, ncb_image, otsu_threshold,
    morphological_opening,
)

__all__ = [
    "GGParams",
    "gg_logpdf",
    "gg_shape_from_ratio",
    "estimate_thresholds",
    "fuzzy_membership",
    "FusionResult",
    "fusion_change_detection",
]


# ---------------------------------------------------------------------------
# Generalized Gaussian model
# ---------------------------------------------------------------------------

def gg_logpdf(x, mu, sigma, beta):
    """Log density of the generalized Gaussian distribution, eq. (14)-(15):
    ``a exp(-(b |x - mu|)^beta)`` with ``a = b beta / (2 Gamma(1/beta))``
    and ``b = sqrt(Gamma(3/beta) / Gamma(1/beta)) / sigma``."""
    lb = 0.5 * (special.gammaln(3.0 / beta) - special.gammaln(1.0 / beta)) - np.log(sigma)
    la = lb + np.log(beta) - np.log(2.0) - special.gammaln(1.0 / beta)
    return la - np.exp(beta * (lb + np.log(np.abs(x - mu) + 1e-300)))


_BETA_GRID = np.exp(np.linspace(np.log(0.2), np.log(20.0), 2000))
_RATIO_GRID = np.exp(special.gammaln(1.0 / _BETA_GRID) + special.gammaln(3.0 / _BETA_GRID)
                     - 2.0 * special.gammaln(2.0 / _BETA_GRID))


def gg_shape_from_ratio(ratio):
    """GG shape parameter from the moment ratio ``E[(x-mu)^2] /
    E[|x-mu|]^2 = Gamma(1/b) Gamma(3/b) / Gamma(2/b)^2`` with a lookup
    table. The ratio decreases monotonically with the shape."""
    r = np.asarray(ratio, dtype=np.float64)
    # np.interp needs increasing x: reverse the decreasing ratio grid.
    return np.interp(r, _RATIO_GRID[::-1], _BETA_GRID[::-1],
                     left=_BETA_GRID[-1], right=_BETA_GRID[0])


@dataclass
class GGParams:
    prior: float
    mu: float
    sigma: float
    beta: float

    def logpdf(self, x):
        return gg_logpdf(x, self.mu, self.sigma, self.beta)


# ---------------------------------------------------------------------------
# Threshold pair estimation
# ---------------------------------------------------------------------------

def _tail_stats(x, h, side, idx, tail_estimation):
    """Statistics of a change class from the histogram tail.

    side +1: increasing change, tail x >= x[idx]; side -1: decreasing
    change, tail x <= x[idx]. Returns arrays (prior, mu, var, mad) for
    every idx in ``idx``.
    """
    n = x.size
    if side > 0:
        S0 = np.cumsum(h[::-1])[::-1]
        S1 = np.cumsum((h * x)[::-1])[::-1]
        S2 = np.cumsum((h * x * x)[::-1])[::-1]
    else:
        S0 = np.cumsum(h)
        S1 = np.cumsum(h * x)
        S2 = np.cumsum(h * x * x)
    s0, s1, s2 = S0[idx], S1[idx], S2[idx]
    t = x[idx]
    if tail_estimation:
        # Eq. (17)/(18): symmetric GG centered at the threshold, estimated
        # from its outer half.
        prior = 2.0 * s0
        mu = t
        with np.errstate(divide="ignore", invalid="ignore"):
            var = 2.0 * (s2 - 2 * t * s1 + t * t * s0) / prior
            mad = 2.0 * side * (s1 - t * s0) / prior
    else:
        # Eq. (16): conventional estimate from the whole region.
        prior = s0
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = s1 / prior
            var = s2 / prior - mu * mu
            mad = np.full_like(mu, np.nan)
    return prior, mu, var, mad


def _shape(var, mad):
    """GG shape from the variance and mean absolute deviation."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return gg_shape_from_ratio(var / (mad * mad))


def estimate_thresholds(lr: np.ndarray, bins: int = 256,
                        center: float = 0.0,
                        quantiles: Tuple[float, float] = (5e-4, 1 - 5e-4),
                        tail_estimation: bool = True,
                        min_tail_pixels: int = 10,
                        sigma_floor_bins: float = 1.0,
                        cost_kind: str = "joint"):
    """Optimal threshold pair (tau1, tau2) of a log-ratio image.

    Minimizes the expected classification cost ``J(tau1, tau2) =
    sum_x h(x) c(x, tau1, tau2)`` (eq. (11), (12)) over candidate pairs
    ``tau1 <= center <= tau2``, with GG class conditionals whose parameters
    come from the tail estimators (eq. (17)-(19)). Candidates that give an
    invalid model (e.g. a non-positive unchanged prior) are discarded.

    The tail estimators are ill-posed for tails of a handful of samples:
    a tail of a single histogram bin has zero variance and an unbounded
    density, which drives the cost to minus infinity. Candidates whose tail
    holds fewer than ``min_tail_pixels`` samples are therefore discarded
    and all class standard deviations are floored at ``sigma_floor_bins``
    histogram bin widths.

    ``cost_kind`` selects the per-bin cost. ``"joint"`` (default) is the
    Kittler-Illingworth minimum error cost ``-2 ln(P(w) p(x | w))`` of the
    KI algorithm the paper builds on. ``"posterior"`` is eq. (12) taken
    literally, ``-2 ln P(w | x)``: the expected posterior cost is
    minimized by classes that do not overlap at all, which pushes both
    change classes to the extremes of the histogram (on simulated data it
    misses every target), so it is kept only for reference.

    Returns
    -------
    (tau1, tau2), (dc, u, ic), (L1, L2) : tuple, tuple of GGParams, tuple
        The thresholds, the class models (decreasing change, unchanged,
        increasing change) and the histogram range.
    """
    v = np.asarray(lr, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    L1, L2 = np.quantile(v, quantiles)
    L1 = min(L1, center - 1e-6)
    L2 = max(L2, center + 1e-6)
    hist, edges = np.histogram(np.clip(v, L1, L2), bins=bins, range=(L1, L2))
    h = hist / hist.sum()
    x = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]
    min_mass = min_tail_pixels / v.size
    var_floor = (sigma_floor_bins * dx) ** 2
    lo_idx = np.nonzero(x <= center)[0]
    hi_idx = np.nonzero(x >= center)[0]
    if lo_idx.size == 0:
        lo_idx = np.array([0])
    if hi_idx.size == 0:
        hi_idx = np.array([bins - 1])

    pic, muic, varic, madic = _tail_stats(x, h, +1, hi_idx, tail_estimation)
    pdc, mudc, vardc, maddc = _tail_stats(x, h, -1, lo_idx, tail_estimation)
    if tail_estimation:
        bic_ = _shape(varic, madic)
        bdc_ = _shape(vardc, maddc)
    else:
        bic_ = np.full(hi_idx.size, 2.0)
        bdc_ = np.full(lo_idx.size, 2.0)
    tail_scale = 2.0 if tail_estimation else 1.0
    enough_ic = pic / tail_scale >= min_mass
    enough_dc = pdc / tail_scale >= min_mass
    varic = np.where(enough_ic, np.maximum(varic, var_floor), np.nan)
    vardc = np.where(enough_dc, np.maximum(vardc, var_floor), np.nan)

    best = (np.inf, None)
    xr = x[None, :]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Increasing change log densities for all tau2 candidates: [n2, n].
        ok_ic = (pic > 0) & (varic > 0) & np.isfinite(bic_)
        lf_ic = gg_logpdf(xr, muic[:, None], np.sqrt(np.where(ok_ic, varic, 1.0))[:, None],
                          bic_[:, None])
        for a, i1 in enumerate(lo_idx):
            ok_dc = pdc[a] > 0 and vardc[a] > 0 and np.isfinite(bdc_[a])
            if not ok_dc:
                # Every candidate keeps x <= tau1 in the decreasing class.
                continue
            if ok_dc:
                lf_dc = gg_logpdf(x, mudc[a], np.sqrt(vardc[a]), bdc_[a])
            else:
                lf_dc = np.full_like(x, -np.inf)
            p_dc = pdc[a] if ok_dc else 0.0
            # Residual histogram of the unchanged class on (tau1, tau2].
            res = h[None, :] - pic[:, None] * np.exp(lf_ic) * dx
            if ok_dc:
                res = res - p_dc * np.exp(lf_dc)[None, :] * dx
            res = np.maximum(res, 0.0)
            band = (x[None, :] > x[i1]) & (x[None, :] <= x[hi_idx][:, None])
            res = np.where(band, res, 0.0)
            pu = 1.0 - pic - p_dc
            mu_u = (res * xr).sum(1) / pu
            var_u = np.maximum((res * (xr - mu_u[:, None]) ** 2).sum(1) / pu, var_floor)
            mad_u = (res * np.abs(xr - mu_u[:, None])).sum(1) / pu
            bu = _shape(var_u, mad_u) if tail_estimation else np.full_like(mu_u, 2.0)
            valid = (ok_ic & (pu > 0) & (var_u > 0) & np.isfinite(bu)
                     & (x[hi_idx] > x[i1]))
            if not valid.any():
                continue
            lf_u = gg_logpdf(xr, mu_u[:, None], np.sqrt(np.where(valid, var_u, 1.0))[:, None],
                             bu[:, None])
            l_ic = np.log(np.maximum(pic, 1e-300))[:, None] + lf_ic
            l_u = np.log(np.maximum(pu, 1e-300))[:, None] + lf_u
            l_dc = (np.log(p_dc) + lf_dc)[None, :] if ok_dc else np.full_like(l_u, -np.inf)
            lse = np.logaddexp(np.logaddexp(l_ic, l_u), l_dc)
            post = np.where(xr <= x[i1], l_dc,
                            np.where(xr <= x[hi_idx][:, None], l_u, l_ic))
            if cost_kind == "posterior":
                post = post - lse
            cost = -2.0 * np.maximum(post, -700.0)
            J = np.where(h[None, :] > 0, h[None, :] * cost, 0.0).sum(1)
            J = np.where(valid, J, np.inf)
            j = int(np.argmin(J))
            if J[j] < best[0]:
                best = (J[j], (i1, hi_idx[j], p_dc, mudc[a], vardc[a], bdc_[a],
                               pu[j], mu_u[j], var_u[j], bu[j]), j)
    if best[1] is None:
        raise RuntimeError("No valid threshold pair found")
    i1, i2, p_dc, m_dc, v_dc, b_dc, p_u, m_u, v_u, b_u = best[1]
    j = best[2]
    dc = GGParams(float(p_dc), float(m_dc), float(np.sqrt(v_dc)) if v_dc > 0 else 1.0,
                  float(b_dc) if np.isfinite(b_dc) else 2.0)
    u = GGParams(float(p_u), float(m_u), float(np.sqrt(v_u)), float(b_u))
    ic = GGParams(float(pic[j]), float(muic[j]), float(np.sqrt(varic[j])), float(bic_[j]))
    return (float(x[i1]), float(x[i2])), (dc, u, ic), (L1, L2)


# ---------------------------------------------------------------------------
# Fuzzy membership
# ---------------------------------------------------------------------------

def _decision_boundary(xs, d, outermost):
    """Root of ``d(x)`` on the grid ``xs``. ``outermost="low"`` returns the
    smallest root, ``"high"`` the largest; without a sign change the point
    of the smallest |d|."""
    s = np.sign(d)
    ch = np.nonzero(s[:-1] * s[1:] < 0)[0]
    if ch.size == 0:
        return float(xs[np.argmin(np.abs(d))])
    i = ch[0] if outermost == "low" else ch[-1]
    # Linear interpolation inside the bracketing interval.
    x0, x1, d0, d1 = xs[i], xs[i + 1], d[i], d[i + 1]
    return float(x0 - d0 * (x1 - x0) / (d1 - d0))


def fuzzy_membership(classes, limits, center: float = 0.0, n_grid: int = 4096):
    """Posterior probability fuzzy membership function, eq. (6)-(9).

    Returns (phi, k1, k2) where ``phi(x)`` maps log-ratio values to [0, 1].
    """
    dc, u, ic = classes
    L1, L2 = limits
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if dc.prior > 0:
            g = np.linspace(L1, center, n_grid)
            d = dc.prior * np.exp(dc.logpdf(g)) - u.prior * np.exp(u.logpdf(g))
            k1 = _decision_boundary(g, d, "low")
        else:
            k1 = -np.inf
        g = np.linspace(center, L2, n_grid)
        d = ic.prior * np.exp(ic.logpdf(g)) - u.prior * np.exp(u.logpdf(g))
        k2 = _decision_boundary(g, d, "high")

    def phi(x):
        x = np.asarray(x, dtype=np.float64)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            l_u = np.log(u.prior) + u.logpdf(x)
            l_ic = np.log(ic.prior) + ic.logpdf(x)
            l_dc = (np.log(dc.prior) + dc.logpdf(x)) if dc.prior > 0 else np.full_like(x, -np.inf)
            lse = np.logaddexp(np.logaddexp(l_u, l_ic), l_dc)
            post_dc = np.exp(l_dc - lse)
            post_u = np.exp(l_u - lse)
            post_ic = np.exp(l_ic - lse)
        out = np.where(x <= k1, post_dc, np.where(x >= k2, post_ic, 1.0 - post_u))
        # Far outside the modeled range all densities underflow: such
        # extreme ratios are changes.
        out = np.where(np.isfinite(out), out, 1.0)
        return np.clip(out, 0.0, 1.0)

    return phi, k1, k2


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

@dataclass
class FusionResult:
    change_map: np.ndarray
    fused: np.ndarray
    fuzzy: np.ndarray
    log_ratio: np.ndarray
    ncb: np.ndarray
    threshold: float
    taus: Tuple[float, float]
    k1: float
    k2: float
    classes: tuple


def fusion_change_detection(img1, img2, *, lee_size: int = 5,
                            looks: float = 1.0, ncb_size: int = 3,
                            bins: int = 256, center: float = 0.0,
                            tail_estimation: bool = True,
                            cost_kind: str = "joint",
                            opening_size: int = 3) -> FusionResult:
    """Fusion of the fuzzified log-ratio and NCB difference images.

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
    bins : int
        Number of log-ratio histogram bins of the threshold pair search.
    center : float
        Log-ratio value separating the decreasing and increasing change
        search ranges (0 for radiometrically matched images).
    tail_estimation : bool
        Use the tail based GG estimators (eq. (17)-(19)); False uses the
        conventional full range estimates (eq. (16)) with Gaussian shape.
    cost_kind : {"joint", "posterior"}
        Per-bin cost of the threshold pair search, see
        :func:`estimate_thresholds`.
    opening_size : int
        Size of the final morphological opening.
    """
    a = as_intensity(img1)
    b = as_intensity(img2)
    if lee_size and lee_size > 1:
        a = lee_filter(a, lee_size, looks)
        b = lee_filter(b, lee_size, looks)
    lr = log_ratio(a, b)
    ncb = ncb_image(a, b, ncb_size)
    taus, classes, limits = estimate_thresholds(lr, bins=bins, center=center,
                                                tail_estimation=tail_estimation,
                                                cost_kind=cost_kind)
    phi, k1, k2 = fuzzy_membership(classes, limits, center=center)
    fuzzy = phi(lr)
    fused = fuzzy * ncb
    thr = otsu_threshold(fused)
    change = morphological_opening(fused > thr, opening_size)
    return FusionResult(change, fused, fuzzy, lr, ncb, thr, taus, k1, k2, classes)
