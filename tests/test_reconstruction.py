"""Tests of the multi-angle SAR ship 3D reconstruction in
``reconstruction/``."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reconstruction import (  # noqa: E402
    SARView, ImageGrid, views_on_circle, VoxelGrid, reconstruct,
    remove_error_regions, cluster_centroids, match_points, principal_dimensions,
    voxel_scores, ShipModel, point_scatterer_model, render_points,
    simulate_ship_images, segment_target, complete_footprint, radon_azimuth,
    rasterize_polygon, voxelize, visible_surface_points,
)


class TestGeometry(unittest.TestCase):
    def test_projection_matrix_matches_eq9(self):
        v = SARView(alpha=0.3, theta=np.deg2rad(40), d_origin=(1.5, -2.0),
                    beta=0.05, phi=0.2)
        p = np.array([3.0, -4.0, 7.0])
        kx, ky = v.k
        c, s = np.cos(0.3), np.sin(0.3)
        expect = [c * 3 + s * -4 + kx * 7 + 1.5, -s * 3 + c * -4 + ky * 7 - 2.0]
        np.testing.assert_allclose(v.project(p), expect)
        np.testing.assert_allclose(v.E @ np.r_[p, 1.0], [*expect, 7.0])
        # E = H R (eq. (8)).
        H = np.array([[1, 0, kx, 0], [0, 1, ky, 0], [0, 0, 1, 0]])
        np.testing.assert_allclose(H @ v.R, v.E)

    def test_level_platform_factors(self):
        v = SARView(alpha=0.0, theta=np.deg2rad(30))
        kx, ky = v.k
        self.assertAlmostEqual(kx, 0.0)
        self.assertAlmostEqual(ky, -1 / np.tan(np.deg2rad(30)))

    def test_projection_is_along_slant_normal(self):
        # In the far-field side-view model the projection direction is the
        # normal of the slant range plane: points along it coincide.
        v = SARView(alpha=0.7, theta=np.deg2rad(35))
        n_local = np.array([0.0, np.cos(v.theta), np.sin(v.theta)])
        from reconstruction.geometry import rotation_z
        n = rotation_z(v.alpha).T @ n_local
        p = np.array([2.0, 1.0, 0.0])
        np.testing.assert_allclose(v.project(p + 5 * n), v.project(p), atol=1e-12)


class TestPointTargets(unittest.TestCase):
    def test_reconstructs_all_scatterers(self):
        pts = point_scatterer_model()
        lo, hi = pts.min(0) - 3, pts.max(0) + 3
        views = views_on_circle(12, 30.0, 10.0)
        masks, grids = [], []
        for v in views:
            g = ImageGrid.covering(v, lo, hi, 0.5)
            img = render_points(pts, v, g, psf_sigma=1.0)
            masks.append(img > 0.3 * img.max())
            grids.append(g)
        vg = VoxelGrid.from_bounds(lo, hi, 0.5)
        est = cluster_centroids(reconstruct(vg, views, masks, grids), vg)
        rate, rmse = match_points(est, pts, 2.0)
        self.assertEqual(rate, 1.0)
        self.assertLess(rmse, 0.5)


class TestSegmentation(unittest.TestCase):
    def test_rasterize_polygon(self):
        m = rasterize_polygon(np.array([[2, 2], [2, 8], [6, 8], [6, 2]]), (10, 10))
        self.assertEqual(m.sum(), 4 * 6)
        self.assertTrue(m[3, 5] and not m[1, 5])

    def test_radon_azimuth_of_rectangle(self):
        for ang in (0.0, 25.0, 100.0):
            a = np.deg2rad(ang)
            L, W = 60.0, 12.0
            corners = np.array([[-L, -W], [L, -W], [L, W], [-L, W]]) / 2
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            m = rasterize_polygon(corners @ R.T + 50, (100, 100))
            est = np.rad2deg(radon_azimuth(m))
            err = (est - ang + 90) % 180 - 90
            self.assertLess(abs(err), 1.5, (ang, est))

    def test_segmentation_keeps_target_rejects_clutter(self):
        rng = np.random.default_rng(0)
        img = rng.exponential(0.05, (120, 120))
        img[40:80, 55:65] += 2.0 * rng.exponential(1.0, (40, 10))
        m = segment_target(img)
        self.assertGreater(m[40:80, 55:65].mean(), 0.8)
        self.assertLess(m.sum(), 1.5 * 400)


class TestShip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ship = ShipModel(heading=np.deg2rad(20))
        cls.views = views_on_circle(18, 30.0, 20.0)
        imgs, cls.grids = simulate_ship_images(cls.ship, cls.views)
        cls.masks = [segment_target(im, smooth=1, open_size=1, ellipse_scale=None)
                     for im in imgs]
        lo, hi = cls.ship.bbox()
        cls.vg = VoxelGrid.from_bounds(lo, hi, 0.5)
        cls.truth = cls.ship.contains(cls.vg.centers().reshape(*cls.vg.shape, 3))

    def test_visibility_self_occlusion(self):
        vox = voxelize(self.ship, 0.5)
        v = self.views[0]
        vis = visible_surface_points(vox, v.radar_direction())
        q = self.ship.to_ship(vis)
        # The deck is seen, the hull bottom never.
        self.assertGreater((np.abs(q[:, 2] - 3.0) < 0.3).sum(), 100)
        bottom = (q[:, 2] < 0.2) & (np.abs(q[:, 1]) < 2) & (np.abs(q[:, 0]) < 12)
        self.assertEqual(bottom.sum(), 0)

    def test_footprint_completion_and_error_deletion(self):
        direct = reconstruct(self.vg, self.views, self.masks, self.grids)
        cm = [complete_footprint(m, v, g, self.ship.deck_outline(), 0.0,
                                 self.ship.freeboard)
              for m, v, g in zip(self.masks, self.views, self.grids)]
        comp = reconstruct(self.vg, self.views, cm, self.grids)
        clean = remove_error_regions(comp, self.vg, np.deg2rad(30),
                                     heading=self.ship.heading)
        p0, r0, i0 = voxel_scores(direct, self.truth)
        p1, r1, i1 = voxel_scores(comp, self.truth)
        p2, r2, i2 = voxel_scores(clean, self.truth)
        # Footprint completion recovers the self-occluded hull ...
        self.assertGreater(r1, 0.95)
        self.assertGreater(i1, i0 + 0.2)
        # ... and error region deletion removes false top voxels.
        self.assertGreater(p2, p1)
        self.assertGreater(i2, i1)
        L, W, H = principal_dimensions(self.vg.points(clean), self.ship.heading)
        self.assertLess(abs(L - 40.0), 1.5)
        self.assertLess(abs(H - 16.0), 1.0)


if __name__ == "__main__":
    unittest.main()
