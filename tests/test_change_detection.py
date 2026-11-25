"""Tests of the unsupervised SAR change detection methods in
``change_detection/``."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from change_detection import (  # noqa: E402
    ncb_image, log_ratio, otsu_threshold, slic, dbscan, kmeans,
    evaluate_objects, make_synthetic_pair, superpixel_change_detection,
    fusion_change_detection,
)
from change_detection import fusion, superpixel_classification as spc  # noqa: E402


class TestCommon(unittest.TestCase):
    def test_ncb_identical_and_changed(self):
        rng = np.random.default_rng(0)
        a = rng.exponential(size=(30, 30))
        np.testing.assert_allclose(ncb_image(a, a, 3), 0.0, atol=1e-12)
        # A uniform gain change keeps NCB close to zero only for small gains,
        # a disjoint support gives the maximum 1.
        b = np.zeros_like(a)
        b[:, 15:] = a[:, 15:]
        a2 = a.copy()
        a2[:, 15:] = 0.0
        self.assertAlmostEqual(ncb_image(a2, b, 3)[15, 3], 1.0)

    def test_log_ratio(self):
        a = np.full((4, 4), 2.0)
        np.testing.assert_allclose(log_ratio(a, 2 * a), math.log(2.0), rtol=1e-9)

    def test_otsu_bimodal(self):
        rng = np.random.default_rng(1)
        v = np.concatenate([rng.normal(0, 0.1, 5000), rng.normal(1, 0.1, 5000)])
        t = otsu_threshold(v)
        # Any threshold in the gap between the modes is optimal.
        self.assertLess(((v[:5000] > t).sum() + (v[5000:] <= t).sum()) / v.size, 0.01)

    def test_slic_covers_image(self):
        rng = np.random.default_rng(2)
        img = rng.random((80, 100)) * 0.1
        img[20:40, 30:60] += 0.8
        lab = slic(img, spacing=20, compactness=0.1)
        self.assertEqual(lab.shape, img.shape)
        self.assertGreaterEqual(lab.min(), 0)
        n = lab.max() + 1
        self.assertTrue(10 <= n <= 40, n)
        self.assertEqual(len(np.unique(lab)), n)

    def test_kmeans(self):
        rng = np.random.default_rng(3)
        X = np.concatenate([rng.normal(c, 0.1, (50, 2)) for c in (0, 3, 6)])
        lab, C = kmeans(X, 3, seed=0)
        self.assertEqual(len(np.unique(lab)), 3)
        np.testing.assert_allclose(np.sort(C[:, 0]), [0, 3, 6], atol=0.1)

    def test_dbscan(self):
        pts = np.concatenate([np.argwhere(np.ones((5, 5))),
                              np.argwhere(np.ones((5, 5))) + 50,
                              [[100, 100]]])
        lab = dbscan(pts, eps=2.0, min_pts=5)
        self.assertEqual(len(np.unique(lab[lab >= 0])), 2)
        self.assertEqual(lab[-1], -1)

    def test_evaluate_objects_rules(self):
        gt = np.zeros((100, 100), bool)
        gt[10:16, 10:16] = True
        gt[50:56, 50:56] = True
        pred = np.zeros_like(gt)
        pred[10:16, 10:16] = True      # TP
        pred[80:86, 80:86] = True      # FP
        s = evaluate_objects(pred, gt, eps=2.0, min_pts=5)
        self.assertEqual((s.tp, s.fp, s.fn), (1, 1, 1))
        self.assertAlmostEqual(s.f1, 0.5)


class TestSuperpixelClassification(unittest.TestCase):
    def test_mixture_beats_single_on_mixed_sample(self):
        rng = np.random.default_rng(0)
        u = np.clip(rng.lognormal(np.log(0.08), 0.5, 900), 1e-4, 1 - 1e-4)
        c = np.clip(1 - rng.lognormal(np.log(0.15), 0.3, 100), 1e-4, 1 - 1e-4)
        x = np.concatenate([u, c])
        _, ll_s = spc.fit_single(x, "lognormal")
        mix = spc.fit_mixture(x, "lognormal")
        self.assertLess(spc.bic(mix.loglik, 5, x.size), spc.bic(ll_s, 2, x.size))
        self.assertAlmostEqual(mix.pi, 0.1, delta=0.03)
        det = mix.resp > 0.5
        self.assertGreater(det[900:].mean(), 0.95)
        self.assertLess(det[:900].mean(), 0.01)

    def test_single_preferred_without_change(self):
        rng = np.random.default_rng(1)
        x = np.clip(rng.gamma(3.0, 0.04, 1000), 1e-4, 1 - 1e-4)
        p, ll_s = spc.fit_single(x, "gamma")
        self.assertAlmostEqual(p[0] * p[1], x.mean(), delta=0.01)
        mix = spc.fit_mixture(x, "gamma")
        self.assertGreaterEqual(spc.bic(mix.loglik, 5, x.size),
                                spc.bic(ll_s, 2, x.size))

    def test_end_to_end(self):
        img1, img2, gt = make_synthetic_pair((300, 400), n_targets=10, seed=4)
        res = superpixel_change_detection(img1, img2)
        s = evaluate_objects(res.change_map, gt, 7, 20)
        self.assertGreaterEqual(s.f1, 0.8, str(s))
        self.assertEqual(set(np.unique(res.classes)), {spc.HU, spc.HEU, spc.CHANGED})


class TestFusion(unittest.TestCase):
    def test_gg_shape_lookup(self):
        for beta in (0.7, 1.0, 2.0, 4.0):
            r = math.exp(math.lgamma(1 / beta) + math.lgamma(3 / beta)
                         - 2 * math.lgamma(2 / beta))
            self.assertAlmostEqual(float(fusion.gg_shape_from_ratio(r)), beta, delta=0.01)

    def test_gg_density_normalized(self):
        x = np.linspace(-20, 20, 200001)
        for beta in (0.8, 1.5, 2.0, 5.0):
            p = np.exp(fusion.gg_logpdf(x, 0.3, 1.2, beta))
            self.assertAlmostEqual(np.trapezoid(p, x), 1.0, places=3)
            self.assertAlmostEqual(np.trapezoid(p * (x - 0.3) ** 2, x), 1.44, places=2)

    def test_membership_range(self):
        img1, img2, gt = make_synthetic_pair((200, 300), n_targets=6, seed=5)
        res = fusion_change_detection(img1, img2)
        self.assertTrue(np.all((res.fuzzy >= 0) & (res.fuzzy <= 1)))
        self.assertLess(res.taus[0], res.taus[1])
        self.assertLessEqual(res.k1, 0.0)
        self.assertGreaterEqual(res.k2, 0.0)
        # Changed pixels get a much higher membership than unchanged ones.
        self.assertGreater(np.median(res.fuzzy[gt]), 10 * np.median(res.fuzzy[~gt]) + 0.1)

    def test_end_to_end(self):
        img1, img2, gt = make_synthetic_pair((300, 400), n_targets=10, seed=4)
        res = fusion_change_detection(img1, img2)
        s = evaluate_objects(res.change_map, gt, 7, 11)
        self.assertGreaterEqual(s.f1, 0.8, str(s))


if __name__ == "__main__":
    unittest.main()
