"""Multi-angle SAR projection model (Section II of the paper).

Global frame O-XYZ: the sea surface is the XY plane and Z points up. Each
view i has a local frame O_i-X_iY_iZ_i that shares the vertical axis: it is
the global frame rotated by ``alpha_i`` about Z and translated by
``dO_i = (dx_O, dy_O, 0)`` (eq. (1), (2)). Y_i points from the
sub-satellite point towards the scene (ground range), X_i is the azimuth
direction.

In the far field the SAR projection direction is the normal of the slant
range plane, so a scatterer at height z appears shifted in the image plane
by ``(k_x z, k_y z)`` (eq. (5), (6)):

    k_x = (sin(phi) tan(beta) cot(theta) + tan(beta)) / cos(phi)
    k_y = -cot(theta)

with incidence angle theta, platform pitch beta and horizontal angle phi.
The projection matrix ``E_i`` (eq. (8)) maps homogeneous global points
``[x, y, z, 1]`` to image plane coordinates ``[x_i, y_i, z]``.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = ["SARView", "ImageGrid", "rotation_z", "views_on_circle"]


def rotation_z(alpha: float) -> np.ndarray:
    """3x3 rotation of eq. (1): global -> local coordinates."""
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class SARView:
    """Geometry of one SAR view.

    Parameters
    ----------
    alpha : float
        Rotation angle between the global and local frames [rad].
    theta : float
        Incidence angle [rad], measured from the vertical.
    d_origin : tuple
        Translation (dx_O, dy_O) of the local frame origin.
    beta : float
        Platform pitch angle [rad] (0 for a level platform).
    phi : float
        Horizontal angle of the platform velocity [rad].
    """
    alpha: float
    theta: float
    d_origin: Tuple[float, float] = (0.0, 0.0)
    beta: float = 0.0
    phi: float = 0.0

    @property
    def k(self) -> Tuple[float, float]:
        """Height projection factors (k_x, k_y) of eq. (6)."""
        tb = np.tan(self.beta)
        cot = 1.0 / np.tan(self.theta)
        kx = (np.sin(self.phi) * tb * cot + tb) / np.cos(self.phi)
        ky = -cot
        return float(kx), float(ky)

    @property
    def R(self) -> np.ndarray:
        """4x4 global -> local transform R_i of eq. (2)."""
        m = np.eye(4)
        m[:3, :3] = rotation_z(self.alpha)
        m[0, 3], m[1, 3] = self.d_origin
        return m

    @property
    def E(self) -> np.ndarray:
        """3x4 multi-angle SAR projection matrix E_i of eq. (8)."""
        kx, ky = self.k
        c, s = np.cos(self.alpha), np.sin(self.alpha)
        dx, dy = self.d_origin
        return np.array([[c, s, kx, dx],
                         [-s, c, ky, dy],
                         [0.0, 0.0, 1.0, 0.0]])

    def project(self, pts: np.ndarray) -> np.ndarray:
        """Image plane coordinates (x_i, y_i) of global points [..., 3]
        (eq. (9))."""
        pts = np.asarray(pts, dtype=np.float64)
        E = self.E
        return pts @ E[:2, :3].T + E[:2, 3]

    def radar_direction(self) -> np.ndarray:
        """Unit vector from the scene towards the radar in the global frame.

        In the local frame the slant range vector is ``(0, -sin theta,
        cos theta)`` (eq. (3)).
        """
        r_local = np.array([0.0, -np.sin(self.theta), np.cos(self.theta)])
        return rotation_z(self.alpha).T @ r_local


@dataclass
class ImageGrid:
    """Pixel grid of a view's image plane.

    Row index follows the local X_i (azimuth) axis and column index the
    local Y_i (ground range) axis: pixel (r, c) is centered at
    ``x_i = x0 + r * dx``, ``y_i = y0 + c * dy``.
    """
    x0: float
    dx: float
    y0: float
    dy: float
    nx: int
    ny: int

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nx, self.ny)

    def to_pixel(self, xy: np.ndarray) -> np.ndarray:
        """Fractional pixel indices (row, col) of image plane points."""
        xy = np.asarray(xy, dtype=np.float64)
        return np.stack([(xy[..., 0] - self.x0) / self.dx,
                         (xy[..., 1] - self.y0) / self.dy], -1)

    def to_plane(self, rc: np.ndarray) -> np.ndarray:
        rc = np.asarray(rc, dtype=np.float64)
        return np.stack([self.x0 + rc[..., 0] * self.dx,
                         self.y0 + rc[..., 1] * self.dy], -1)

    @staticmethod
    def covering(view: SARView, bbox_min, bbox_max, spacing: float,
                 margin: float = 2.0) -> "ImageGrid":
        """Grid that covers the projection of a 3D bounding box."""
        lo = np.asarray(bbox_min, dtype=np.float64)
        hi = np.asarray(bbox_max, dtype=np.float64)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        p = view.project(corners)
        pmin = p.min(0) - margin
        pmax = p.max(0) + margin
        n = np.ceil((pmax - pmin) / spacing).astype(int) + 1
        return ImageGrid(float(pmin[0]), spacing, float(pmin[1]), spacing,
                         int(n[0]), int(n[1]))


def views_on_circle(n: int, theta_deg: float = 30.0, step_deg: Optional[float] = None,
                    start_deg: float = 0.0):
    """``n`` views at a constant incidence angle and evenly stepped azimuth
    (``step_deg`` defaults to 360 / n)."""
    step = 360.0 / n if step_deg is None else step_deg
    th = np.deg2rad(theta_deg)
    return [SARView(alpha=np.deg2rad(start_deg + i * step), theta=th) for i in range(n)]
