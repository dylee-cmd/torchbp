"""Building blocks shared by the unsupervised SAR change detection methods.

Difference images (log-ratio, noncoherent change detection with Berger's
operator), Lee speckle filter, Otsu's threshold, morphological opening,
SLIC superpixels, k-means++, PCA, DBSCAN and the object-level evaluation
protocol used in both papers. Everything is implemented with NumPy and
SciPy only.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage, signal
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

__all__ = [
    "as_intensity",
    "lee_filter",
    "log_ratio",
    "ncb_image",
    "otsu_threshold",
    "morphological_opening",
    "slic",
    "kmeans",
    "pca",
    "dbscan",
    "ObjectScores",
    "evaluate_objects",
    "make_synthetic_pair",
]


def as_intensity(img) -> np.ndarray:
    """Real non-negative float64 array from a real or complex image (or a
    torch tensor)."""
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if np.iscomplexobj(img):
        img = np.abs(img)
    return np.abs(img.astype(np.float64))


# ---------------------------------------------------------------------------
# Preprocessing and difference images
# ---------------------------------------------------------------------------

def lee_filter(img: np.ndarray, size: int = 5, looks: float = 1.0) -> np.ndarray:
    """Lee speckle filter (Lee 1981) with a ``size`` x ``size`` window.

    ``looks`` is the equivalent number of looks of the input intensity
    image; the speckle coefficient of variation is ``1/sqrt(looks)``.
    """
    x = as_intensity(img)
    cu2 = 1.0 / looks
    m = ndimage.uniform_filter(x, size, mode="reflect")
    m2 = ndimage.uniform_filter(x * x, size, mode="reflect")
    var = np.maximum(m2 - m * m, 0.0)
    # Variance of the underlying (speckle-free) signal.
    var_x = np.maximum((var - m * m * cu2) / (1.0 + cu2), 0.0)
    k = np.where(var > 0, var_x / np.maximum(var_x + m * m * cu2, 1e-300), 0.0)
    return m + k * (x - m)


def log_ratio(img1: np.ndarray, img2: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Log-ratio difference image ``log(|I2| / |I1|)``."""
    a = as_intensity(img1)
    b = as_intensity(img2)
    return np.log(b + eps) - np.log(a + eps)


def ncb_image(img1: np.ndarray, img2: np.ndarray, size: int = 3) -> np.ndarray:
    """Noncoherent change detection image with Berger's operator.

    ``1 - 2 sum(|I1||I2|) / (sum|I1|^2 + sum|I2|^2)`` over a ``size`` x
    ``size`` sliding window. Values are in [0, 1]; 0 means no change.
    """
    a = as_intensity(img1)
    b = as_intensity(img2)
    s12 = ndimage.uniform_filter(a * b, size, mode="reflect")
    s11 = ndimage.uniform_filter(a * a, size, mode="reflect")
    s22 = ndimage.uniform_filter(b * b, size, mode="reflect")
    den = s11 + s22
    ncb = np.where(den > 0, 1.0 - 2.0 * s12 / np.maximum(den, 1e-300), 0.0)
    return np.clip(ncb, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Thresholding and morphology
# ---------------------------------------------------------------------------

def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold maximizing the between-class variance."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.inf
    lo, hi = v.min(), v.max()
    if hi <= lo:
        return hi
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu_t * w0 - mu) ** 2 / (w0 * w1)
    sb[~np.isfinite(sb)] = -1.0
    i = int(np.argmax(sb[:-1]))
    # Threshold between bin i and i + 1: values > edges[i + 1] are class 1.
    return float(edges[i + 1])


def morphological_opening(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Binary opening with a ``size`` x ``size`` square structuring element."""
    if size <= 1:
        return mask.astype(bool)
    st = np.ones((size, size), dtype=bool)
    return ndimage.binary_opening(mask.astype(bool), structure=st)


# ---------------------------------------------------------------------------
# SLIC superpixels
# ---------------------------------------------------------------------------

def slic(img: np.ndarray, spacing: int = 40, compactness: float = 0.1,
         n_iter: int = 10, min_size_factor: float = 0.25) -> np.ndarray:
    """SLIC superpixels (Achanta et al. 2012) of a single channel image.

    Parameters
    ----------
    img : ndarray
        2D image, typically the NCB image in [0, 1].
    spacing : int
        Initialization grid spacing S in pixels.
    compactness : float
        Weight m of the spatial distance, in units of the image intensity
        per grid interval: ``D = sqrt(dc^2 + (ds / S)^2 * m^2)``.
    n_iter : int
        Number of assignment/update iterations.
    min_size_factor : float
        Connected fragments smaller than ``min_size_factor * S^2`` pixels
        are merged into an adjacent superpixel.

    Returns
    -------
    labels : ndarray of int
        Superpixel label of each pixel, 0 ... nlabels-1.
    """
    x = np.asarray(img, dtype=np.float64)
    H, W = x.shape
    S = max(1, int(spacing))
    ys = np.arange(S // 2, H, S)
    xs = np.arange(S // 2, W, S)
    if ys.size == 0:
        ys = np.array([H // 2])
    if xs.size == 0:
        xs = np.array([W // 2])
    cy, cx = np.meshgrid(ys, xs, indexing="ij")
    cy = cy.ravel().astype(np.float64)
    cx = cx.ravel().astype(np.float64)
    # Move the seeds to the lowest gradient position in a 3x3 neighborhood.
    gy, gx = np.gradient(ndimage.uniform_filter(x, 3))
    grad = gx * gx + gy * gy
    for k in range(cy.size):
        y0, x0 = int(cy[k]), int(cx[k])
        ya, yb = max(0, y0 - 1), min(H, y0 + 2)
        xa, xb = max(0, x0 - 1), min(W, x0 + 2)
        sub = grad[ya:yb, xa:xb]
        j = np.unravel_index(np.argmin(sub), sub.shape)
        cy[k], cx[k] = ya + j[0], xa + j[1]
    cc = x[cy.astype(int), cx.astype(int)].copy()

    labels = -np.ones((H, W), dtype=np.int64)
    w_s = (compactness / S) ** 2
    for _ in range(n_iter):
        dist = np.full((H, W), np.inf)
        for k in range(cy.size):
            ya = max(0, int(cy[k] - S))
            yb = min(H, int(cy[k] + S) + 1)
            xa = max(0, int(cx[k] - S))
            xb = min(W, int(cx[k] + S) + 1)
            yy = np.arange(ya, yb)[:, None]
            xx = np.arange(xa, xb)[None, :]
            d = (x[ya:yb, xa:xb] - cc[k]) ** 2 + w_s * ((yy - cy[k]) ** 2 + (xx - cx[k]) ** 2)
            sub = dist[ya:yb, xa:xb]
            better = d < sub
            sub[better] = d[better]
            labels[ya:yb, xa:xb][better] = k
        # Update the cluster centers.
        valid = labels >= 0
        lab = labels[valid]
        cnt = np.bincount(lab, minlength=cy.size).astype(np.float64)
        Y, X = np.nonzero(valid)
        nz = cnt > 0
        cy[nz] = (np.bincount(lab, Y, minlength=cy.size) / np.maximum(cnt, 1))[nz]
        cx[nz] = (np.bincount(lab, X, minlength=cy.size) / np.maximum(cnt, 1))[nz]
        cc[nz] = (np.bincount(lab, x[valid], minlength=cy.size) / np.maximum(cnt, 1))[nz]

    if (labels < 0).any():
        # Pixels out of every search window: nearest labeled pixel.
        idx = ndimage.distance_transform_edt(labels < 0, return_distances=False,
                                             return_indices=True)
        labels = labels[idx[0], idx[1]]
    return _enforce_connectivity(labels, int(min_size_factor * S * S))


def _enforce_connectivity(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Split disconnected superpixels into connected components, merge
    components smaller than ``min_size`` into an adjacent component and
    relabel consecutively."""
    out = np.zeros_like(labels)
    nxt = 0
    for lab in np.unique(labels):
        comp, n = ndimage.label(labels == lab)
        out[comp > 0] = comp[comp > 0] + nxt - 1
        nxt += n
    sizes = np.bincount(out.ravel())
    small = np.nonzero(sizes < min_size)[0]
    st = np.ones((3, 3), dtype=bool)
    # Merge small fragments, smallest first, into the most common
    # neighboring label.
    for lab in small[np.argsort(sizes[small])]:
        m = out == lab
        if not m.any():
            continue
        ring = ndimage.binary_dilation(m, structure=st) & ~m
        nb = out[ring]
        if nb.size == 0:
            continue
        vals, counts = np.unique(nb, return_counts=True)
        out[m] = vals[np.argmax(counts)]
    _, inv = np.unique(out, return_inverse=True)
    return inv.reshape(out.shape)


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def kmeans(X: np.ndarray, k: int, n_init: int = 10, max_iter: int = 300,
           seed: Optional[int] = 0, tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """k-means with k-means++ initialization and random restarts.

    Returns (labels, centers) of the restart with the lowest inertia.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    best = (None, None, np.inf)
    for _ in range(n_init):
        # k-means++ seeding.
        centers = [X[rng.integers(n)]]
        for _ in range(1, k):
            d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
            centers.append(X[rng.choice(n, p=p)])
        C = np.array(centers)
        prev = np.inf
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)
            lab = np.argmin(d2, axis=1)
            inertia = d2[np.arange(n), lab].sum()
            for j in range(k):
                if (lab == j).any():
                    C[j] = X[lab == j].mean(0)
            if prev - inertia <= tol * max(prev, 1.0):
                break
            prev = inertia
        if inertia < best[2]:
            best = (lab.copy(), C.copy(), inertia)
    return best[0], best[1]


def pca(X: np.ndarray, var_ratio: float = 0.95) -> np.ndarray:
    """Project standardized features onto the principal components that
    explain ``var_ratio`` of the variance."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = s ** 2
    if ev.sum() <= 0:
        return Xc[:, :1]
    ncomp = int(np.searchsorted(np.cumsum(ev) / ev.sum(), var_ratio) + 1)
    ncomp = min(max(ncomp, 1), Vt.shape[0])
    return Xc @ Vt[:ncomp].T


def dbscan(points: np.ndarray, eps: float = 7.0, min_pts: int = 20) -> np.ndarray:
    """DBSCAN (Ester et al. 1996) of 2D points.

    A point is a core point when at least ``min_pts`` points (itself
    included) lie within distance ``eps``. Border points join the cluster
    of their nearest core point. Returns the cluster label of every point,
    -1 for noise.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    labels = -np.ones(n, dtype=np.int64)
    if n == 0:
        return labels
    tree = cKDTree(pts)
    counts = np.asarray(tree.query_ball_point(pts, eps, return_length=True))
    core = counts >= min_pts
    ci = np.nonzero(core)[0]
    if ci.size == 0:
        return labels
    ctree = cKDTree(pts[ci])
    pairs = ctree.query_pairs(eps, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs)
                       else (np.zeros(0), (np.zeros(0, int), np.zeros(0, int))),
                       shape=(ci.size, ci.size))
    _, comp = connected_components(graph, directed=False)
    labels[ci] = comp
    bi = np.nonzero(~core)[0]
    if bi.size:
        d, j = ctree.query(pts[bi], distance_upper_bound=eps)
        ok = np.isfinite(d)
        labels[bi[ok]] = comp[j[ok]]
    return labels


# ---------------------------------------------------------------------------
# Object-level evaluation
# ---------------------------------------------------------------------------

@dataclass
class ObjectScores:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def __str__(self) -> str:
        return (f"TP={self.tp} FP={self.fp} FN={self.fn} "
                f"precision={self.precision:.4f} recall={self.recall:.4f} "
                f"F1={self.f1:.4f}")


def evaluate_objects(pred: np.ndarray, gt: np.ndarray, eps: float = 7.0,
                     min_pts: int = 20) -> ObjectScores:
    """Object-level evaluation of a binary change map.

    Detected pixels are clustered with DBSCAN and ground truth objects are
    the 8-connected components of ``gt``. Each predicted cluster center is
    matched one-to-one: a cluster whose center falls inside a not yet
    matched ground truth object is a TP, any other cluster is an FP and
    unmatched ground truth objects are FNs.
    """
    pred = np.asarray(pred, dtype=bool)
    gt_lab, n_gt = ndimage.label(np.asarray(gt, dtype=bool), structure=np.ones((3, 3)))
    pts = np.argwhere(pred)
    lab = dbscan(pts, eps, min_pts)
    matched = np.zeros(n_gt + 1, dtype=bool)
    tp = fp = 0
    for c in np.unique(lab[lab >= 0]):
        cy, cx = pts[lab == c].mean(0)
        iy = int(np.clip(round(cy), 0, gt_lab.shape[0] - 1))
        ix = int(np.clip(round(cx), 0, gt_lab.shape[1] - 1))
        g = gt_lab[iy, ix]
        if g > 0 and not matched[g]:
            matched[g] = True
            tp += 1
        else:
            fp += 1
    fn = int(n_gt - matched[1:].sum())
    return ObjectScores(tp, fp, fn)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def make_synthetic_pair(shape=(400, 600), n_targets: int = 12,
                        target_size=(5, 9), looks: float = 1.0,
                        hetero_fraction: float = 0.5,
                        hetero_correlation: float = 0.6,
                        target_tcr_db: float = 12.0,
                        seed: Optional[int] = 0):
    """Synthetic multitemporal SAR intensity pair with small changes.

    The scene has a homogeneous, temporally stable background and a
    heterogeneous textured region (gamma texture with partial temporal
    decorrelation, like moving vegetation) that produces clutter driven
    false alarms. Small rectangular targets appear in the second image
    (and a few disappear from the first). Both images get ``looks``-look
    gamma speckle.

    Returns
    -------
    img1, img2 : ndarray
        Intensity images.
    gt : ndarray of bool
        Ground truth change mask.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    refl = np.full((H, W), 1.0)
    # Heterogeneous region: right part of the scene with gamma texture.
    het = np.zeros((H, W), dtype=bool)
    het[:, int(W * (1 - hetero_fraction)):] = True
    nu = 2.0
    tex1 = rng.gamma(nu, 1.0 / nu, size=(H, W))
    tex_new = rng.gamma(nu, 1.0 / nu, size=(H, W))
    tex1 = ndimage.uniform_filter(tex1, 3) * 3.0
    tex_new = ndimage.uniform_filter(tex_new, 3) * 3.0
    rho = hetero_correlation
    tex2 = rho * tex1 + (1.0 - rho) * tex_new
    r1 = np.where(het, refl * tex1, refl)
    r2 = np.where(het, refl * tex2, refl)
    gt = np.zeros((H, W), dtype=bool)
    amp = 10.0 ** (target_tcr_db / 10.0)
    th, tw = target_size
    placed = 0
    tries = 0
    while placed < n_targets and tries < 100 * n_targets:
        tries += 1
        y = rng.integers(10, H - th - 10)
        x = rng.integers(10, W - tw - 10)
        if gt[max(0, y - 15):y + th + 15, max(0, x - 15):x + tw + 15].any():
            continue
        # Bright point-like scatterers on a rectangular footprint.
        patch = amp * (0.5 + rng.random((th, tw)))
        base = r1[y:y + th, x:x + tw].mean()
        if placed % 4 == 3:
            r1[y:y + th, x:x + tw] = base * patch  # disappearing target
        else:
            r2[y:y + th, x:x + tw] = base * patch  # appearing target
        gt[y:y + th, x:x + tw] = True
        placed += 1
    s1 = rng.gamma(looks, 1.0 / looks, size=(H, W))
    s2 = rng.gamma(looks, 1.0 / looks, size=(H, W))
    return r1 * s1, r2 * s2, gt
