"""Change detection on SAR time series using coherent scatterers.

Implementation of C. Villamil Lopez and U. Stilla, "Monitoring of
Construction Activity by Change Detection on SAR Time Series Using
Coherent Scatterers", IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.,
vol. 15, pp. 7499-7514, 2022.

Instead of amplitude changes or coherence loss of all pixels, the method
tracks the strong point scatterers of man-made objects (coherent
scatterers, CSs). Natural targets (vegetation, water) produce no CSs and are
ignored.

Processing chain (Fig. 1):

1. CS detection in every coregistered SLC image (Section II-A): range
   sub-looks; a CS has a phase that varies linearly with the sub-look
   frequency. Pixels with a low variance of consecutive sub-look phase
   differences contain a CS. Of those, only pixels whose total phase
   change across the sub-looks is at most pi keep the CS, which removes
   the spread of the CS into neighbors of the lower-resolution sub-looks.
2. CD metric (Section II-C): the coherence between image pairs, eq. (2):
   ``f_i = max_{j in [i-r, i], k in [i+1, i+1+r]} gamma_{j,k}``. It is low
   only if every pair across the interval (t_i, t_{i+1}) decorrelated, so
   transient changes (e.g. snow in one image) are ignored.
3. Pixelwise CD for CS pixels: a forward/backward consistency check
   propagates CS presence through high coherence intervals (CS detection
   false negatives). The CS lifetimes [a, b] then follow from where the
   coherence drops. CSs detected in too few of their images are discarded,
   eq. (3).
4. Spatiotemporal clustering (Section II-D): DBSCAN, in meters, applied
   separately to the CSs of every (t_start, t_end) combination. Small
   clusters are discarded.
5. Change segmentation (Section II-E): within the enlarged cluster
   rectangle, pixels that change in (t_{a-1}, t_a), change in
   (t_b, t_{b+1}) and stay coherent within [t_a, t_b]. Objects seen in a
   single image use amplitude constraints instead of the third one.
   Followed by a closing and removal of components mostly outside the
   cluster.
6. Change analysis (Section II-F): e.g. new buildings (appear after t_1,
   still present at t_n, static for at least dT) or moving objects
   (present for less than dT).

Image indices are 0-based here: image i is acquired at ``times[i]``,
``f[i]`` is the CD metric of the interval (t_i, t_{i+1}) and a CS lifetime
[a, b] means it is present in images a through b.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from .common import dbscan, lee_filter

__all__ = [
    "sublooks",
    "detect_coherent_scatterers",
    "coherence",
    "cd_metric",
    "pair_change_classes",
    "CSEvents",
    "cs_time_intervals",
    "ChangeCluster",
    "cluster_events",
    "segment_cluster",
    "categorize",
    "CSChangeResult",
    "cs_change_detection",
]


# ---------------------------------------------------------------------------
# Coherent scatterer detection (Section II-A)
# ---------------------------------------------------------------------------

def sublooks(slc: np.ndarray, n: int = 10, overlap: float = 0.75,
             axis: int = -1, band_fraction: float = 1.0) -> np.ndarray:
    """Full resolution sub-look images along ``axis`` (range).

    The occupied part of the spectrum (``band_fraction`` of the sampled
    band, centered) is split into ``n`` equally spaced sub-bands that
    overlap by ``overlap`` (10 sub-looks with 75 % overlap of a 300 MHz
    band give the paper's 92.3 MHz sub-looks spaced by 23.1 MHz).

    Returns an array of shape ``(n, *slc.shape)``.
    """
    x = np.asarray(slc)
    N = x.shape[axis]
    spec = np.fft.fftshift(np.fft.fft(x, axis=axis), axes=axis)
    B = band_fraction * N
    sub_bw = B / (1.0 + (n - 1) * (1.0 - overlap))
    step = sub_bw * (1.0 - overlap)
    f = np.arange(N) - N / 2.0
    lo = -B / 2.0
    shape = [1] * x.ndim
    shape[axis] = N
    out = []
    for i in range(n):
        a = lo + i * step
        mask = ((f >= a) & (f < a + sub_bw)).astype(np.float64).reshape(shape)
        out.append(np.fft.ifft(np.fft.ifftshift(spec * mask, axes=axis), axis=axis))
    return np.stack(out)


def detect_coherent_scatterers(slc: np.ndarray, n_sublooks: int = 10,
                               overlap: float = 0.75, threshold: float = 0.125,
                               axis: int = -1, band_fraction: float = 1.0,
                               return_stats: bool = False):
    """Binary CS map of one SLC image (Section II-A).

    The phase difference between consecutive sub-looks of a CS pixel is
    nearly constant. Its variance (after removing the circular mean, which
    compensates 2 pi jumps) is compared with ``threshold`` (T). Pixels with
    ``var < T`` whose total phase change across the sub-looks, ``(n - 1)``
    times the mean difference, is at most pi keep the CS; the others are
    the spread of a CS of a neighboring pixel.
    """
    sl = sublooks(slc, n_sublooks, overlap, axis, band_fraction)
    dphi = np.angle(sl[1:] * np.conj(sl[:-1]))
    mean = np.angle(np.exp(1j * dphi).sum(0))
    dev = np.angle(np.exp(1j * (dphi - mean[None])))
    var = (dev * dev).mean(0)
    slope_ok = np.abs(mean) * (n_sublooks - 1) <= np.pi
    cs = (var < threshold) & slope_ok
    if return_stats:
        return cs, var, mean
    return cs


# ---------------------------------------------------------------------------
# Coherence and CD metric (Section II-B, II-C)
# ---------------------------------------------------------------------------

def coherence(s1: np.ndarray, s2: np.ndarray, window=(9, 9)) -> np.ndarray:
    """Interferometric coherence magnitude over a sliding window, at the
    full pixel spacing."""
    c = s1 * np.conj(s2)
    num = ndimage.uniform_filter(c.real, window) + 1j * ndimage.uniform_filter(c.imag, window)
    p1 = ndimage.uniform_filter(np.abs(s1) ** 2, window)
    p2 = ndimage.uniform_filter(np.abs(s2) ** 2, window)
    den = np.sqrt(p1 * p2)
    return np.where(den > 0, np.abs(num) / np.maximum(den, 1e-300), 0.0)


def cd_metric(stack: Sequence[np.ndarray], r: int = 1, window=(9, 9)) -> np.ndarray:
    """CD metric ``f_i`` of eq. (2) for i = 0 .. n-2.

    ``f_i = max gamma_{j,k}`` over ``j in [i-r, i]`` and
    ``k in [i+1, i+1+r]`` (indices clipped to the series). ``r = 0`` gives
    the coherence of consecutive images. Every coherence image is computed
    once.
    """
    n = len(stack)
    cache: Dict[Tuple[int, int], np.ndarray] = {}

    def gamma(j, k):
        if (j, k) not in cache:
            cache[(j, k)] = coherence(stack[j], stack[k], window)
        return cache[(j, k)]

    f = []
    for i in range(n - 1):
        best = None
        for j in range(max(0, i - r), i + 1):
            for k in range(i + 1, min(n - 1, i + 1 + r) + 1):
                g = gamma(j, k)
                best = g if best is None else np.maximum(best, g)
        f.append(best)
    return np.stack(f)


#: Change classes of an image pair (Table I).
PAIR_IRRELEVANT, PAIR_UNCHANGED, PAIR_DISAPPEARED, PAIR_APPEARED, PAIR_CHANGED = 0, 1, 2, 3, 4


def pair_change_classes(cs1: np.ndarray, cs2: np.ndarray, coh: np.ndarray,
                        gamma_t: float = 0.5) -> np.ndarray:
    """Change type of every pixel of an image pair (Table I): irrelevant
    (clutter in both), unchanged (a CS and high coherence), disappeared
    (CS only in image 1, low coherence), appeared (CS only in image 2) or
    changed / moved (CSs in both, low coherence)."""
    out = np.full(cs1.shape, PAIR_IRRELEVANT, dtype=np.int8)
    any_cs = cs1 | cs2
    high = coh >= gamma_t
    out[any_cs & high] = PAIR_UNCHANGED
    low = any_cs & ~high
    out[low & cs1 & ~cs2] = PAIR_DISAPPEARED
    out[low & ~cs1 & cs2] = PAIR_APPEARED
    out[low & cs1 & cs2] = PAIR_CHANGED
    return out


# ---------------------------------------------------------------------------
# Pixelwise CD on a time series (Section II-C)
# ---------------------------------------------------------------------------

@dataclass
class CSEvents:
    """Lifetimes of the detected CSs, one entry per (pixel, CS).

    ``a`` and ``b`` are the first and last images where the CS is present
    (0-based, inclusive). ``a == 0`` means it appeared before the series
    started, ``b == n - 1`` that it is still present at the end.
    """
    rows: np.ndarray
    cols: np.ndarray
    a: np.ndarray
    b: np.ndarray

    def __len__(self):
        return len(self.rows)

    def select(self, m: np.ndarray) -> "CSEvents":
        return CSEvents(self.rows[m], self.cols[m], self.a[m], self.b[m])


def cs_time_intervals(cs: np.ndarray, f: np.ndarray, gamma_t: float = 0.5,
                      k: float = 0.1) -> CSEvents:
    """CS lifetimes from the CS maps ``cs`` [n, H, W] and the CD metric
    ``f`` [n-1, H, W].

    Only pixels with a CS in at least one image are processed. The forward
    pass (i = 0 .. n-2) and the backward pass (i = n-1 .. 1) mark a CS as
    present in the neighbor image whenever the coherence of that interval
    is high (``f >= gamma_t``): an unchanged CS is present in both images,
    and a missing detection is a CS detection false negative. Each run of
    images [a, b] with a CS and high coherence between consecutive images is
    one CS. Runs where the CS was originally detected in fewer than
    ``k * (b - a + 1)`` images are discarded as false positives, eq. (3).
    """
    cs = np.asarray(cs, dtype=bool)
    n = cs.shape[0]
    rr, cc = np.nonzero(cs.any(0))
    C = cs[:, rr, cc]                      # [n, P]
    high = np.asarray(f)[:, rr, cc] >= gamma_t  # [n-1, P]
    Cp = C.copy()
    for i in range(n - 1):
        Cp[i + 1] |= Cp[i] & high[i]
    for i in range(n - 1, 0, -1):
        Cp[i - 1] |= Cp[i] & high[i - 1]
    rows, cols, A, B = [], [], [], []
    for p in range(Cp.shape[1]):
        i = 0
        while i < n:
            if not Cp[i, p]:
                i += 1
                continue
            a = i
            while i + 1 < n and Cp[i + 1, p] and high[i, p]:
                i += 1
            b = i
            if C[a:b + 1, p].sum() >= k * (b - a + 1):
                rows.append(rr[p])
                cols.append(cc[p])
                A.append(a)
                B.append(b)
            i += 1
    return CSEvents(np.array(rows, dtype=int), np.array(cols, dtype=int),
                    np.array(A, dtype=int), np.array(B, dtype=int))


# ---------------------------------------------------------------------------
# Spatiotemporal clustering (Section II-D)
# ---------------------------------------------------------------------------

@dataclass
class ChangeCluster:
    a: int
    b: int
    rows: np.ndarray
    cols: np.ndarray
    area_m2: float
    mask: Optional[np.ndarray] = None   # segmented change (full image)

    @property
    def n_cs(self) -> int:
        return len(self.rows)

    @property
    def center(self) -> Tuple[float, float]:
        return float(self.rows.mean()), float(self.cols.mean())

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (int(self.rows.min()), int(self.rows.max()),
                int(self.cols.min()), int(self.cols.max()))


def _hull_area(rows, cols, spacing):
    pts = np.stack([rows * spacing[0], cols * spacing[1]], 1)
    try:
        return float(ConvexHull(pts).volume)
    except (QhullError, ValueError):
        return 0.0


def cluster_events(events: CSEvents, spacing=(1.0, 1.0), eps: float = 15.0,
                   min_pts: int = 20, min_cs: int = 30,
                   min_area: float = 20.0, n_images: Optional[int] = None,
                   include_static: bool = False) -> List[ChangeCluster]:
    """DBSCAN of the CSs, in meters, separately for every (a, b)
    combination. Clusters with fewer than ``min_cs`` CSs or a convex hull
    smaller than ``min_area`` m^2 are discarded. CSs present during the
    whole series (unchanged) are skipped unless ``include_static``."""
    out = []
    keys = np.stack([events.a, events.b], 1)
    for a, b in np.unique(keys, axis=0) if len(events) else []:
        if not include_static and n_images is not None and a == 0 and b == n_images - 1:
            continue
        sel = (events.a == a) & (events.b == b)
        r, c = events.rows[sel], events.cols[sel]
        pts = np.stack([r * spacing[0], c * spacing[1]], 1)
        lab = dbscan(pts, eps, min_pts)
        for L in np.unique(lab[lab >= 0]):
            m = lab == L
            if m.sum() < min_cs:
                continue
            area = _hull_area(r[m], c[m], spacing)
            if area < min_area:
                continue
            out.append(ChangeCluster(int(a), int(b), r[m], c[m], area))
    return out


# ---------------------------------------------------------------------------
# Segmentation of the changes (Section II-E)
# ---------------------------------------------------------------------------

def _amplitude_db(stack, lee_size=5, multilook=3):
    out = []
    for s in stack:
        inten = lee_filter(np.abs(s) ** 2, lee_size) if lee_size > 1 else np.abs(s) ** 2
        if multilook > 1:
            inten = ndimage.uniform_filter(inten, multilook)
        out.append(10.0 * np.log10(np.maximum(inten, 1e-12)))
    return np.stack(out)


def segment_cluster(cl: ChangeCluster, f: np.ndarray, gamma_t: float = 0.5,
                    amp_db: Optional[np.ndarray] = None, delta_a: float = 3.0,
                    a_min: float = -15.0, margin: float = 0.5,
                    closing_radius: int = 5, min_size: int = 20,
                    inside_fraction: float = 0.5) -> np.ndarray:
    """Dense change mask of one cluster.

    Within the cluster rectangle enlarged by ``margin`` (50 %), a pixel
    belongs to the change if it changed in (t_{a-1}, t_a), changed in
    (t_b, t_{b+1}) and stayed coherent during [t_a, t_b]. Constraints that
    fall outside the series are dropped. Objects present in one image only
    (a == b) additionally need an amplitude change of at least ``delta_a``
    dB towards both neighbor images and an amplitude of at least ``a_min``
    dB in image a. A closing with radius ``closing_radius`` fills holes;
    connected components smaller than ``min_size`` or mostly outside the
    cluster rectangle are discarded.
    """
    n = f.shape[0] + 1
    H, W = f.shape[1:]
    r0, r1, c0, c1 = cl.bbox
    hr = (r1 - r0 + 1) * margin / 2
    hc = (c1 - c0 + 1) * margin / 2
    R0, R1 = max(0, int(r0 - hr)), min(H, int(np.ceil(r1 + hr)) + 1)
    C0, C1 = max(0, int(c0 - hc)), min(W, int(np.ceil(c1 + hc)) + 1)
    fp = f[:, R0:R1, C0:C1]
    a, b = cl.a, cl.b
    m = np.ones((R1 - R0, C1 - C0), dtype=bool)
    if a > 0:
        m &= fp[a - 1] < gamma_t
    if b < n - 1:
        m &= fp[b] < gamma_t
    if b > a:
        m &= fp[a:b].min(0) >= gamma_t
    else:
        if amp_db is None:
            raise ValueError("objects seen in one image need amplitude images")
        A = amp_db[:, R0:R1, C0:C1]
        if a > 0:
            m &= np.abs(A[a] - A[a - 1]) >= delta_a
        if a < n - 1:
            m &= np.abs(A[a] - A[a + 1]) >= delta_a
        m &= A[a] >= a_min
    if closing_radius > 0:
        y, x = np.ogrid[-closing_radius:closing_radius + 1, -closing_radius:closing_radius + 1]
        disk = x * x + y * y <= closing_radius * closing_radius
        m = ndimage.binary_closing(np.pad(m, closing_radius), disk)[
            closing_radius:-closing_radius, closing_radius:-closing_radius]
    lab, nl = ndimage.label(m)
    keep = np.zeros_like(m)
    inner = np.zeros_like(m)
    inner[r0 - R0:r1 - R0 + 1, c0 - C0:c1 - C0 + 1] = True
    for L in range(1, nl + 1):
        comp = lab == L
        size = comp.sum()
        if size < min_size or (comp & inner).sum() < inside_fraction * size:
            continue
        keep |= comp
    full = np.zeros((H, W), dtype=bool)
    full[R0:R1, C0:C1] = keep
    return full


# ---------------------------------------------------------------------------
# Change analysis (Section II-F)
# ---------------------------------------------------------------------------

def categorize(cl: ChangeCluster, times: Sequence[float], delta_t: float) -> str:
    """Temporal category of a change cluster.

    The exact appearance and disappearance times are only known to lie in
    (t_{a-1}, t_a) and (t_b, t_{b+1}); durations use the lower bound
    t_b - t_a.

    - "new": appeared during the series, still present at the end and
      present for at least ``delta_t`` (new or renovated building).
    - "removed": present at the start, disappeared during the series.
    - "moving": appeared and disappeared during the series, present for
      less than ``delta_t`` (parked cars, cranes, scaffolds...).
    - "temporary": appeared and disappeared, present for at least
      ``delta_t``.
    - "static": present during the whole series.
    """
    n = len(times)
    appeared = cl.a > 0
    present_end = cl.b == n - 1
    dur = times[cl.b] - times[cl.a]
    if not appeared and present_end:
        return "static"
    if appeared and present_end:
        return "new" if dur >= delta_t else "moving"
    if not appeared:
        return "removed"
    return "moving" if dur < delta_t else "temporary"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

@dataclass
class CSChangeResult:
    cs: np.ndarray                 # [n, H, W] detected CSs
    f: np.ndarray                  # [n-1, H, W] CD metric
    events: CSEvents
    clusters: List[ChangeCluster]
    categories: List[str]
    change_map: np.ndarray         # union of the segmented changes
    label_map: np.ndarray          # cluster index + 1 of every changed pixel


def cs_change_detection(stack: Sequence[np.ndarray], times: Optional[Sequence[float]] = None,
                        *, spacing=(1.0, 1.0), range_axis: int = -1,
                        n_sublooks: int = 10, overlap: float = 0.75,
                        band_fraction: float = 1.0, cs_threshold: float = 0.125,
                        coh_window=(9, 9), r: int = 1, gamma_t: float = 0.5,
                        k: float = 0.1, eps: float = 15.0, min_pts: int = 20,
                        min_cs: int = 30, min_area: float = 20.0,
                        delta_a: float = 3.0, a_min: float = -15.0,
                        margin: float = 0.5, closing_radius: int = 5,
                        delta_t: float = 60.0, segment: bool = True) -> CSChangeResult:
    """Coherent scatterer change detection of a coregistered SLC series.

    Parameters
    ----------
    stack : sequence of complex ndarray
        n >= 2 coregistered SLC images [azimuth, range].
    times : sequence of float, optional
        Acquisition times (e.g. days), used by the change analysis.
        Defaults to 0, 1, 2, ...
    spacing : tuple
        Pixel spacing (rows, cols) in meters, for clustering.
    range_axis : int
        Axis of the images along which the sub-looks are formed.
    n_sublooks, overlap, band_fraction, cs_threshold
        CS detection parameters (10 sub-looks, 75 % overlap, T = 0.125).
    coh_window : tuple
        Coherence estimation window (the paper uses 9 x 23 pixels for its
        anisotropic TerraSAR-X pixels).
    r : int
        Transient change robustness of the CD metric, eq. (2).
    gamma_t : float
        Coherence threshold (0.5).
    k : float
        False positive control of eq. (3) (0.1).
    eps, min_pts, min_cs, min_area
        DBSCAN radius [m] and minimum points, minimum cluster size and
        convex hull area [m^2] (15 m, 20, 30, 20 m^2).
    delta_a, a_min, margin, closing_radius
        Segmentation parameters (3 dB, -15 dB, 50 %, 5 pixels).
    delta_t : float
        Duration threshold of the change analysis, in ``times`` units.
    """
    stack = [np.asarray(s) for s in stack]
    n = len(stack)
    if n < 2:
        raise ValueError("need at least two images")
    times = np.arange(n, dtype=float) if times is None else np.asarray(times, dtype=float)
    cs = np.stack([detect_coherent_scatterers(s, n_sublooks, overlap, cs_threshold,
                                              range_axis, band_fraction) for s in stack])
    f = cd_metric(stack, r, coh_window)
    events = cs_time_intervals(cs, f, gamma_t, k)
    clusters = cluster_events(events, spacing, eps, min_pts, min_cs, min_area, n)
    cats = [categorize(c, times, delta_t) for c in clusters]
    change = np.zeros(cs.shape[1:], dtype=bool)
    labels = np.zeros(cs.shape[1:], dtype=np.int32)
    if segment and clusters:
        amp = _amplitude_db(stack) if any(c.a == c.b for c in clusters) else None
        for idx, c in enumerate(clusters):
            c.mask = segment_cluster(c, f, gamma_t, amp, delta_a, a_min, margin,
                                     closing_radius)
            change |= c.mask
            labels[c.mask & (labels == 0)] = idx + 1
    return CSChangeResult(cs, f, events, clusters, cats, change, labels)
