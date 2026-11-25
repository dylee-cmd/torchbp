"""Voxel back-projection voting reconstruction (Section III-C), error
region deletion (Section III-D) and evaluation metrics (eq. (15), (16))."""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .geometry import SARView, ImageGrid

__all__ = [
    "VoxelGrid",
    "vote",
    "reconstruct",
    "remove_error_regions",
    "cluster_centroids",
    "match_points",
    "principal_dimensions",
    "voxel_scores",
]


@dataclass
class VoxelGrid:
    """Regular grid of candidate scatterer positions: voxel (i, j, k) is
    centered at ``origin + (i, j, k) * spacing``."""
    origin: np.ndarray
    spacing: float
    shape: Tuple[int, int, int]

    @staticmethod
    def from_bounds(lo, hi, spacing: float) -> "VoxelGrid":
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        shape = tuple(int(n) for n in np.floor((hi - lo) / spacing).astype(int) + 1)
        return VoxelGrid(lo, float(spacing), shape)

    def centers(self, idx: Optional[np.ndarray] = None) -> np.ndarray:
        if idx is None:
            idx = np.indices(self.shape).reshape(3, -1).T
        return self.origin + np.asarray(idx) * self.spacing

    def points(self, occ: np.ndarray) -> np.ndarray:
        return self.centers(np.argwhere(occ))

    def index_of(self, pts: np.ndarray) -> np.ndarray:
        return np.rint((np.asarray(pts) - self.origin) / self.spacing).astype(int)


def vote(grid: VoxelGrid, views: Sequence[SARView], masks: Sequence[np.ndarray],
         image_grids: Sequence[ImageGrid], chunk: int = 1 << 20) -> np.ndarray:
    """Voting accumulation: each candidate voxel gets one vote per view in
    which its theoretical projection (eq. (8)) falls inside the target
    segmentation area. Returns the vote count array of ``grid.shape``."""
    n = int(np.prod(grid.shape))
    votes = np.zeros(n, dtype=np.int32)
    idx_all = np.indices(grid.shape).reshape(3, -1).T
    for s in range(0, n, chunk):
        pts = grid.centers(idx_all[s:s + chunk])
        for v, m, g in zip(views, masks, image_grids):
            rc = np.rint(g.to_pixel(v.project(pts))).astype(int)
            ok = (rc[:, 0] >= 0) & (rc[:, 0] < g.nx) & (rc[:, 1] >= 0) & (rc[:, 1] < g.ny)
            hit = np.zeros(len(pts), dtype=bool)
            hit[ok] = m[rc[ok, 0], rc[ok, 1]]
            votes[s:s + chunk] += hit
    return votes.reshape(grid.shape)


def reconstruct(grid: VoxelGrid, views, masks, image_grids,
                threshold: Optional[int] = None) -> np.ndarray:
    """Voxels whose votes reach ``threshold`` (default: every view, the
    strictest multiview consistency)."""
    votes = vote(grid, views, masks, image_grids)
    thr = len(views) if threshold is None else threshold
    return votes >= thr


def _roof_clip(v_idx: np.ndarray, z_top: np.ndarray, slope: float, spacing: float,
               tol: float, min_extent: int) -> np.ndarray:
    """Clip heights of one slice's top profile.

    The top error region of a side-view reconstruction is a roof whose
    flanks fall with slope tan(theta) (eq. (14)). Starting from the highest
    unprocessed column (the vertex), the profile is followed outwards while
    it stays on the line ``z0 - tan(theta) |v - v0|``; when it stops
    satisfying eq. (14), the height of the last point on the line is the
    estimated true top and the roof columns are clipped to it.
    """
    n = len(v_idx)
    clip = z_top.copy()
    done = np.zeros(n, dtype=bool)
    order = np.argsort(-z_top, kind="stable")
    for i0 in order:
        if done[i0]:
            continue
        z0 = z_top[i0]
        ends = []
        for step in (1, -1):
            j = i0
            while True:
                k = j + step
                if k < 0 or k >= n or done[k] or abs(v_idx[k] - v_idx[j]) != 1:
                    break
                expect = z0 - slope * abs(v_idx[k] - v_idx[i0]) * spacing
                if abs(z_top[k] - expect) > tol:
                    break
                j = k
            ends.append(j)
        lo, hi = min(ends), max(ends)
        ext = [abs(e - i0) for e in ends]
        if max(ext) >= min_extent:
            h = min(z_top[e] for e, x in zip(ends, ext) if x > 0)
            clip[lo:hi + 1] = np.minimum(clip[lo:hi + 1], h)
        done[lo:hi + 1] = True
    return clip


def remove_error_regions(occ: np.ndarray, grid: VoxelGrid, theta: float,
                         heading: Optional[float] = None,
                         tol: Optional[float] = None, min_extent: int = 2) -> np.ndarray:
    """Error region deletion criterion (Section III-D).

    1. Take the reconstructed voxel region.
    2. Rotate it by the ship heading (estimated from the principal axis of
       the XY point cloud when not given) so the long axis is along X.
    3. Slice it into planes parallel to YOZ.
    4. In each slice remove the conical / sloped top error region whose
       flanks satisfy eq. (14) with the radar incidence angle ``theta``.
    5. Rotate back (the occupancy keeps the original grid).
    """
    pts_idx = np.argwhere(occ)
    if len(pts_idx) == 0:
        return occ.copy()
    pts = grid.centers(pts_idx)
    if heading is None:
        xy = pts[:, :2] - pts[:, :2].mean(0)
        w, V = np.linalg.eigh(np.cov(xy.T))
        a = V[:, np.argmax(w)]
        heading = float(np.arctan2(a[1], a[0]))
    c, s = np.cos(heading), np.sin(heading)
    u = c * pts[:, 0] + s * pts[:, 1]
    v = -s * pts[:, 0] + c * pts[:, 1]
    h = grid.spacing
    ui = np.floor((u - u.min()) / h).astype(int)
    vi = np.floor((v - v.min()) / h).astype(int)
    z = pts[:, 2]
    tol = h if tol is None else tol
    slope = np.tan(theta)
    keep = np.ones(len(pts), dtype=bool)
    order = np.lexsort((vi, ui))
    ui_s, vi_s = ui[order], vi[order]
    bounds = np.flatnonzero(np.diff(ui_s)) + 1
    for seg in np.split(np.arange(len(order)), bounds):
        sel = order[seg]
        cols, inv = np.unique(vi_s[seg], return_inverse=True)
        ztop = np.full(len(cols), -np.inf)
        np.maximum.at(ztop, inv, z[sel])
        clip = _roof_clip(cols, ztop, slope, h, tol, min_extent)
        keep[sel] = z[sel] <= clip[inv] + 1e-9
    out = np.zeros_like(occ)
    k = pts_idx[keep]
    out[k[:, 0], k[:, 1], k[:, 2]] = True
    return out


def cluster_centroids(occ: np.ndarray, grid: VoxelGrid,
                      weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Centroids of the 26-connected components of a reconstruction (one
    reconstructed scatterer per component)."""
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    if n == 0:
        return np.zeros((0, 3))
    w = np.ones(occ.shape) if weights is None else weights
    idx = np.array(ndimage.center_of_mass(w * (lab > 0), lab, np.arange(1, n + 1)))
    return grid.origin + idx * grid.spacing


def match_points(est: np.ndarray, true: np.ndarray, max_dist: float):
    """Reconstruction rate (eq. (15)) and RMSE (eq. (16)) of reconstructed
    scatterers. Each true scatterer is matched to the nearest estimate
    within ``max_dist`` (one-to-one, greedy by distance)."""
    true = np.asarray(true, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    if len(est) == 0:
        return 0.0, np.nan
    d = np.linalg.norm(true[:, None, :] - est[None, :, :], axis=-1)
    used_t = np.zeros(len(true), bool)
    used_e = np.zeros(len(est), bool)
    errs = []
    for flat in np.argsort(d, axis=None):
        i, j = np.unravel_index(flat, d.shape)
        if d[i, j] > max_dist:
            break
        if used_t[i] or used_e[j]:
            continue
        used_t[i] = used_e[j] = True
        errs.append(d[i, j])
    rate = used_t.sum() / len(true)
    rmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan
    return float(rate), rmse


def principal_dimensions(points: np.ndarray, heading: Optional[float] = None,
                         z_ref: float = 0.0, q: float = 0.0):
    """Length, width and height of a point cloud: extents along its
    principal horizontal axis, across it, and above ``z_ref``. ``q`` trims
    that quantile at both ends (robust extents)."""
    p = np.asarray(points, dtype=np.float64)
    xy = p[:, :2] - p[:, :2].mean(0)
    if heading is None:
        w, V = np.linalg.eigh(np.cov(xy.T))
        a = V[:, np.argmax(w)]
    else:
        a = np.array([np.cos(heading), np.sin(heading)])
    u = xy @ a
    v = xy @ np.array([-a[1], a[0]])

    def ext(x):
        return float(np.quantile(x, 1 - q) - np.quantile(x, q))

    return ext(u), ext(v), float(np.quantile(p[:, 2], 1 - q) - z_ref)


def voxel_scores(est: np.ndarray, truth: np.ndarray):
    """Voxel precision, recall and IoU of a reconstruction against the true
    occupancy on the same grid."""
    tp = np.logical_and(est, truth).sum()
    fp = np.logical_and(est, ~truth).sum()
    fn = np.logical_and(~est, truth).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return float(prec), float(rec), float(iou)
