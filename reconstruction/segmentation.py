"""Ship target segmentation, azimuth estimation and occluded footprint
completion (Section III-A, III-B of the paper)."""
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

from .geometry import SARView, ImageGrid

__all__ = [
    "segment_target",
    "radon_azimuth",
    "slant_to_ground_azimuth",
    "rasterize_polygon",
    "deck_outline",
    "complete_footprint",
]


def segment_target(img: np.ndarray, factor: float = 3.0, smooth: int = 3,
                   open_size: int = 3, close_size: int = 5,
                   ellipse_scale: Optional[float] = 2.5) -> np.ndarray:
    """Fine segmentation of a SAR ship target image.

    Threshold at ``factor`` times the image grayscale mean (after a
    ``smooth`` x ``smooth`` mean filter), morphological opening and
    closing, keep the largest connected component, fill its holes and
    apply an elliptical shape constraint: pixels outside the ellipse of
    ``ellipse_scale`` standard deviations of the component's second
    moments are removed (edge artifacts and side-lobe streaks).
    """
    x = np.asarray(img, dtype=np.float64)
    if smooth and smooth > 1:
        x = ndimage.uniform_filter(x, smooth)
    m = x > factor * x.mean()
    if open_size > 1:
        m = ndimage.binary_opening(m, np.ones((open_size, open_size), bool))
    if close_size > 1:
        m = ndimage.binary_closing(m, np.ones((close_size, close_size), bool))
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum_labels(m, lab, np.arange(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    m = ndimage.binary_fill_holes(m)
    if ellipse_scale:
        rc = np.argwhere(m).astype(np.float64)
        mu = rc.mean(0)
        C = np.cov((rc - mu).T) + 1e-6 * np.eye(2)
        R, C_ = np.mgrid[0:m.shape[0], 0:m.shape[1]]
        d = np.stack([R - mu[0], C_ - mu[1]], -1)
        md2 = np.einsum("...i,ij,...j->...", d, np.linalg.inv(C), d)
        m = m & (md2 <= ellipse_scale ** 2)
    return m


def radon_azimuth(mask: np.ndarray, grid: Optional[ImageGrid] = None,
                  step_deg: float = 0.25, criterion: str = "energy") -> float:
    """Azimuth angle of the target's main axis by the Radon transform
    (eq. (10)-(12)).

    For every angle psi the line integrals R_psi(X) of the (segmented)
    image along the direction psi are computed. ``criterion="max"`` picks
    the angle with the largest maximum of R_psi, eq. (12) literally; for a
    wide hull this favors the diagonal (a 40 m x 10 m rectangle has a
    41.2 m diagonal) and is biased by up to atan(width / length).
    ``criterion="energy"`` (default) picks the angle of the most
    concentrated projection, ``sum R^2 / sum R``, which peaks exactly on
    the main axis of an elongated target. With ``grid`` the angle is
    measured in the image plane (x_i, y_i) coordinates, otherwise in pixel
    (row, col) coordinates. Returns psi in [0, pi).
    """
    rc = np.argwhere(mask).astype(np.float64)
    w = np.asarray(mask, dtype=np.float64)[mask > 0]
    if grid is not None:
        rc = rc * np.array([grid.dx, grid.dy])
        bin_w = min(grid.dx, grid.dy)
    else:
        bin_w = 1.0

    def score(psi):
        # Coordinate across the lines (normal of direction psi).
        t = -rc[:, 0] * np.sin(psi) + rc[:, 1] * np.cos(psi)
        b = np.floor((t - t.min()) / bin_w).astype(int)
        R = np.bincount(b, weights=w)
        if criterion == "max":
            return R.max()
        return (R * R).sum() / R.sum()

    angles = np.deg2rad(np.arange(0.0, 180.0, step_deg))
    s = np.array([score(a) for a in angles])
    best = angles[int(np.argmax(s))]
    # Local refinement.
    fine = best + np.deg2rad(np.linspace(-step_deg, step_deg, 21))
    sf = np.array([score(a) for a in fine])
    return float(np.mod(fine[int(np.argmax(sf))], np.pi))


def slant_to_ground_azimuth(phi_sar: float, theta: float) -> float:
    """Geometric correction of an azimuth angle measured in a slant range
    image (eq. (13)): ``arctan(tan(phi_SAR) sin(theta))``."""
    return float(np.arctan(np.tan(phi_sar) * np.sin(theta)))


def rasterize_polygon(poly_rc: np.ndarray, shape) -> np.ndarray:
    """Pixels whose centers are inside a polygon given in (row, col)
    coordinates (even-odd rule)."""
    poly = np.asarray(poly_rc, dtype=np.float64)
    r0 = max(0, int(np.floor(poly[:, 0].min())))
    r1 = min(shape[0] - 1, int(np.ceil(poly[:, 0].max())))
    c0 = max(0, int(np.floor(poly[:, 1].min())))
    c1 = min(shape[1] - 1, int(np.ceil(poly[:, 1].max())))
    out = np.zeros(shape, dtype=bool)
    if r1 < r0 or c1 < c0:
        return out
    R, C = np.mgrid[r0:r1 + 1, c0:c1 + 1].astype(np.float64)
    inside = np.zeros(R.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cond = (a[0] > R) != (b[0] > R)
        with np.errstate(divide="ignore", invalid="ignore"):
            cx = a[1] + (R - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
        inside ^= cond & (C < cx)
    out[r0:r1 + 1, c0:c1 + 1] = inside
    return out


def deck_outline(center_xy: Sequence[float], heading: float, length: float,
                 width: float, bow_length: float = 0.0) -> np.ndarray:
    """Global XY polygon of a ship deck with constant width (the deck width
    prior), optionally with a pointed bow of length ``bow_length`` at the
    +axis end."""
    L2, w2 = 0.5 * length, 0.5 * width
    if bow_length > 0:
        ub = L2 - bow_length
        uv = np.array([[-L2, -w2], [ub, -w2], [L2, 0.0], [ub, w2], [-L2, w2]])
    else:
        uv = np.array([[-L2, -w2], [L2, -w2], [L2, w2], [-L2, w2]])
    c, s = np.cos(heading), np.sin(heading)
    x = c * uv[:, 0] - s * uv[:, 1] + center_xy[0]
    y = s * uv[:, 0] + c * uv[:, 1] + center_xy[1]
    return np.stack([x, y], -1)


def complete_footprint(mask: np.ndarray, view: SARView, grid: ImageGrid,
                       outline_xy: np.ndarray, z_bottom: float, z_top: float,
                       n_levels: Optional[int] = None) -> np.ndarray:
    """Occluded footprint completion.

    The hull between ``z_bottom`` (waterline) and ``z_top`` (deck plane) is
    a prism with the prior deck outline. Its complete projection in the
    view is the union of the outline projected at every height in between;
    the self-occluded part of it (far side hull wall, deck areas shadowed
    by the superstructure) is missing from the segmented ``mask`` and is
    added back here.
    """
    kx, ky = view.k
    shift_px = np.hypot(kx * (z_top - z_bottom) / grid.dx, ky * (z_top - z_bottom) / grid.dy)
    if n_levels is None:
        n_levels = int(np.ceil(shift_px)) + 2
    out = np.asarray(mask, dtype=bool).copy()
    for z in np.linspace(z_bottom, z_top, n_levels):
        pts = np.concatenate([outline_xy, np.full((len(outline_xy), 1), z)], 1)
        out |= rasterize_polygon(grid.to_pixel(view.project(pts)), out.shape)
    return out
