"""Simulated multi-angle SAR data for the reconstruction experiments.

Two kinds of scenes, matching the paper's simulated experiments:

- point scatterer models (Section IV-A): isolated scatterers without
  occlusion, imaged as small blobs;
- a solid ship model (Section IV-B): hull with a pointed bow,
  superstructure and mast, voxelized. Every view images only the surface
  voxels that the radar sees (self-occlusion by ray marching towards the
  radar), projected with the side-view SAR projection model, so the images
  have the layover / overlap and the occluded footprints of real side-view
  imaging. Optional speckle and sea clutter.

The images are formed directly in the image plane of each view's projection
model (single backscatter, no physical scattering, as in the paper's
simulated image data).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .geometry import SARView, ImageGrid

__all__ = [
    "ShipModel",
    "point_scatterer_model",
    "render_points",
    "voxelize",
    "visible_surface_points",
    "simulate_ship_images",
]


@dataclass
class ShipModel:
    """Simple ship: hull, superstructure and mast (Fig. 11(a)-like).

    The ship frame has u along the ship's main axis (bow at +u), v across
    and z up from the sea surface. ``center`` and ``heading`` place it in
    the global frame.
    """
    length: float = 40.0
    width: float = 10.0
    freeboard: float = 3.0
    bow_length: float = 6.0
    super_u: Tuple[float, float] = (-14.0, -2.0)
    super_width: float = 7.0
    super_height: float = 5.0
    mast_u: float = 6.0
    mast_size: float = 1.0
    mast_top: float = 16.0
    center: Tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0

    def to_ship(self, pts: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.heading), np.sin(self.heading)
        x = pts[..., 0] - self.center[0]
        y = pts[..., 1] - self.center[1]
        return np.stack([c * x + s * y, -s * x + c * y, pts[..., 2]], -1)

    def half_width(self, u: np.ndarray) -> np.ndarray:
        """Hull half width along the axis (pointed bow)."""
        L2 = 0.5 * self.length
        hw = np.full_like(u, 0.5 * self.width)
        bow = u > L2 - self.bow_length
        hw = np.where(bow, 0.5 * self.width * (L2 - u) / self.bow_length, hw)
        return np.where(np.abs(u) <= L2, np.maximum(hw, 0.0), -1.0)

    def contains(self, pts: np.ndarray) -> np.ndarray:
        q = self.to_ship(pts)
        u, v, z = q[..., 0], q[..., 1], q[..., 2]
        hull = (z >= 0) & (z <= self.freeboard) & (np.abs(v) <= self.half_width(u))
        sup = ((z >= self.freeboard) & (z <= self.freeboard + self.super_height)
               & (u >= self.super_u[0]) & (u <= self.super_u[1])
               & (np.abs(v) <= 0.5 * self.super_width))
        mast = ((z >= self.freeboard) & (z <= self.mast_top)
                & (np.abs(u - self.mast_u) <= 0.5 * self.mast_size)
                & (np.abs(v) <= 0.5 * self.mast_size))
        return hull | sup | mast

    def deck_outline(self, n_bow: int = 1) -> np.ndarray:
        """Global XY polygon of the hull footprint."""
        L2 = 0.5 * self.length
        w2 = 0.5 * self.width
        ub = L2 - self.bow_length
        uv = np.array([[-L2, -w2], [ub, -w2], [L2, 0.0], [ub, w2], [-L2, w2]])
        c, s = np.cos(self.heading), np.sin(self.heading)
        x = c * uv[:, 0] - s * uv[:, 1] + self.center[0]
        y = s * uv[:, 0] + c * uv[:, 1] + self.center[1]
        return np.stack([x, y], -1)

    def bbox(self, margin: float = 1.0):
        r = 0.5 * np.hypot(self.length, self.width) + margin
        cx, cy = self.center
        return (np.array([cx - r, cy - r, -margin]),
                np.array([cx + r, cy + r, self.mast_top + margin]))


def point_scatterer_model() -> np.ndarray:
    """18 scatterers on four layers (heights 0, 10, 20 and 30 m) inside a
    100 m x 20 m x 30 m volume, similar to Fig. 8(a)."""
    pts = [
        # z = 0
        (-10, -40, 0), (10, -40, 0), (-10, 60, 0), (10, 60, 0), (0, -40, 0), (0, 60, 0),
        # z = 10
        (-10, 0, 10), (10, 0, 10), (0, 40, 10), (0, -20, 10), (0, 20, 10),
        # z = 20
        (-5, 10, 20), (5, 10, 20), (0, -10, 20), (0, 30, 20),
        # z = 30
        (0, 0, 30), (0, -30, 30), (0, 50, 30),
    ]
    return np.array(pts, dtype=np.float64)


def render_points(points: np.ndarray, view: SARView, grid: ImageGrid,
                  weights: Optional[np.ndarray] = None, psf_sigma: float = 0.7,
                  clutter: float = 0.0, speckle: bool = False,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Image of point scatterers: projected with the view's projection
    model, accumulated with bilinear splatting and blurred by a Gaussian
    point spread function (``psf_sigma`` pixels). Optional multiplicative
    exponential speckle and additive exponential clutter of mean
    ``clutter``."""
    rng = np.random.default_rng() if rng is None else rng
    img = np.zeros(grid.shape)
    rc = grid.to_pixel(view.project(points))
    w = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=np.float64)
    r0 = np.floor(rc[:, 0]).astype(int)
    c0 = np.floor(rc[:, 1]).astype(int)
    fr = rc[:, 0] - r0
    fc = rc[:, 1] - c0
    for dr, dc, ww in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc),
                       (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
        r = r0 + dr
        c = c0 + dc
        ok = (r >= 0) & (r < grid.nx) & (c >= 0) & (c < grid.ny)
        np.add.at(img, (r[ok], c[ok]), (w * ww)[ok])
    if psf_sigma > 0:
        img = ndimage.gaussian_filter(img, psf_sigma) * (2 * np.pi * psf_sigma ** 2)
    if speckle:
        img = img * rng.exponential(1.0, img.shape)
    if clutter > 0:
        img = img + rng.exponential(clutter, img.shape)
    return img


@dataclass
class Voxels:
    origin: np.ndarray
    spacing: float
    occ: np.ndarray

    def centers(self, idx=None) -> np.ndarray:
        if idx is None:
            idx = np.argwhere(self.occ)
        return self.origin + idx * self.spacing


def voxelize(model: ShipModel, spacing: float = 0.5) -> Voxels:
    """Occupancy grid of the ship model (voxel centers inside the model)."""
    lo, hi = model.bbox()
    shape = np.ceil((hi - lo) / spacing).astype(int) + 1
    axes = [lo[i] + np.arange(shape[i]) * spacing for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    occ = model.contains(np.stack([X, Y, Z], -1))
    return Voxels(lo, spacing, occ)


def visible_surface_points(vox: Voxels, direction: np.ndarray,
                           step: float = 0.5, chunk: int = 20000) -> np.ndarray:
    """Centers of the surface voxels visible from ``direction`` (unit
    vector towards the radar).

    A voxel is visible when one of its exposed faces that faces the radar
    (outward normal n with n . d > 0, empty neighbor across it) has an
    unobstructed ray towards the radar. The ray starts just outside the
    face, so it does not graze the solid's own wall.
    """
    occ = vox.occ
    shape = np.array(occ.shape)
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    nsteps = int(np.ceil(np.linalg.norm(occ.shape) / step)) + 2
    ts = np.arange(nsteps) * step
    padded = np.pad(occ, 1)
    visible = np.zeros(occ.shape, dtype=bool)
    for axis in range(3):
        for sgn in (1, -1):
            if sgn * d[axis] <= 0:
                continue
            nb = np.roll(padded, -sgn, axis=axis)[1:-1, 1:-1, 1:-1]
            idx = np.argwhere(occ & ~nb & ~visible)
            if idx.size == 0:
                continue
            n = np.zeros(3)
            n[axis] = sgn
            for s in range(0, len(idx), chunk):
                p0 = idx[s:s + chunk].astype(np.float64) + 0.501 * n
                q = np.rint(p0[:, None, :] + ts[None, :, None] * d[None, None, :]).astype(int)
                inside = np.all((q >= 0) & (q < shape), axis=-1)
                qc = np.where(inside[..., None], q, 0)
                hit = occ[qc[..., 0], qc[..., 1], qc[..., 2]] & inside
                ok = ~np.any(hit, axis=1)
                vi = idx[s:s + chunk][ok]
                visible[vi[:, 0], vi[:, 1], vi[:, 2]] = True
    return vox.centers(np.argwhere(visible))


def simulate_ship_images(model: ShipModel, views: List[SARView],
                         pixel: float = 0.5, voxel: float = 0.5,
                         speckle: bool = False, clutter: float = 0.0,
                         seed: Optional[int] = 0):
    """Simulated SAR images of a ship model for every view.

    Returns
    -------
    images : list of ndarray
    grids : list of ImageGrid
    """
    rng = np.random.default_rng(seed)
    vox = voxelize(model, voxel)
    lo, hi = model.bbox()
    images, grids = [], []
    for v in views:
        pts = visible_surface_points(vox, v.radar_direction())
        grid = ImageGrid.covering(v, lo, hi, pixel, margin=3.0)
        img = render_points(pts, v, grid, psf_sigma=0.6 * voxel / pixel,
                            speckle=speckle, clutter=clutter, rng=rng)
        images.append(img)
        grids.append(grid)
    return images, grids
