"""Tests of the coherent scatterer time series change detection in
``change_detection/coherent_scatterers.py``."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from change_detection.coherent_scatterers import (  # noqa: E402
    sublooks, detect_coherent_scatterers, coherence, cd_metric,
    cs_time_intervals, pair_change_classes, cs_change_detection, categorize,
    ChangeCluster, PAIR_APPEARED, PAIR_DISAPPEARED, PAIR_UNCHANGED,
    PAIR_CHANGED, PAIR_IRRELEVANT,
)
from change_detection.cs_simulation import make_slc_time_series  # noqa: E402


def _point_image(H=64, W=64, pts=(), band=0.8, clutter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    fr = np.fft.fftfreq(H) * H
    fc = np.fft.fftfreq(W) * W
    spec = np.zeros((H, W), complex)
    for r, c, a in pts:
        spec += a * np.exp(-2j * np.pi * (fr[:, None] * r / H + fc[None, :] * c / W))
    if clutter:
        spec += np.fft.fft2(clutter * (rng.standard_normal((H, W))
                                       + 1j * rng.standard_normal((H, W))) / np.sqrt(2))
    mask = (np.abs(fr)[:, None] < band * H / 2) & (np.abs(fc)[None, :] < band * W / 2)
    return np.fft.ifft2(spec * mask)


class TestCSDetection(unittest.TestCase):
    def test_sublook_bandwidths(self):
        s = _point_image(pts=[(32, 32, 1.0)], band=1.0)
        sl = sublooks(s, n=10, overlap=0.75)
        self.assertEqual(sl.shape, (10, 64, 64))
        # Sub-look bandwidth B / (1 + 9 * 0.25) of the band.
        spec = np.fft.fftshift(np.fft.fft(sl[0], axis=-1), axes=-1)
        nz = np.abs(spec[32]) > 1e-9
        self.assertAlmostEqual(nz.sum() / 64, 1 / 3.25, delta=2 / 64)

    def test_point_detected_at_its_pixel_only(self):
        s = _point_image(pts=[(20.3, 30.4, 30.0)], clutter=1.0)
        cs, var, mean = detect_coherent_scatterers(s, band_fraction=0.8,
                                                   threshold=0.05, return_stats=True)
        self.assertTrue(cs[20, 30])
        self.assertLess(var[20, 30], 0.01)
        # The sub-look spread of the CS into its range neighbors is removed
        # by the slope constraint.
        self.assertFalse(cs[20, 33] or cs[20, 27])
        # Few detections in pure clutter.
        self.assertLess(cs.mean(), 0.01)

    def test_coherence_and_metric(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((40, 40)) + 1j * rng.standard_normal((40, 40))
        b = rng.standard_normal((40, 40)) + 1j * rng.standard_normal((40, 40))
        self.assertGreater(coherence(a, a * np.exp(0.3j)).min(), 0.999)
        self.assertLess(coherence(a, b).mean(), 0.3)
        # Transient outlier image 2: r = 0 reports changes around it, r = 1
        # bridges it.
        stack = [a, a, b, a, a]
        f0 = cd_metric(stack, r=0)
        f1 = cd_metric(stack, r=1)
        self.assertLess(f0[1].mean(), 0.3)
        self.assertLess(f0[2].mean(), 0.3)
        self.assertGreater(f1[1].min(), 0.99)
        self.assertGreater(f1[2].min(), 0.99)


class TestTimeSeriesLogic(unittest.TestCase):
    def test_intervals_consistency_and_false_positive_check(self):
        n = 6
        cs = np.zeros((n, 1, 4), bool)
        f = np.ones((n - 1, 1, 4))
        # Pixel 0: CS present 2..5, detection missed in image 3.
        cs[[2, 4, 5], 0, 0] = True
        f[1, 0, 0] = 0.1
        # Pixel 1: CS 0..1 replaced by another CS 3..5.
        cs[[0, 1, 3, 4, 5], 0, 1] = True
        f[1, 0, 1] = 0.1
        f[2, 0, 1] = 0.1
        # Pixel 2: a single false detection in a long coherent interval.
        cs[0, 0, 2] = True
        ev = cs_time_intervals(cs, f, 0.5, k=0.2)
        got = sorted(zip(ev.cols.tolist(), ev.a.tolist(), ev.b.tolist()))
        self.assertEqual(got, [(0, 2, 5), (1, 0, 1), (1, 3, 5)])

    def test_pair_classes(self):
        cs1 = np.array([[0, 1, 0, 1, 1]], bool)
        cs2 = np.array([[0, 0, 1, 1, 1]], bool)
        coh = np.array([[0.1, 0.1, 0.1, 0.1, 0.9]])
        np.testing.assert_array_equal(
            pair_change_classes(cs1, cs2, coh)[0],
            [PAIR_IRRELEVANT, PAIR_DISAPPEARED, PAIR_APPEARED, PAIR_CHANGED,
             PAIR_UNCHANGED])

    def test_categories(self):
        times = np.arange(8) * 11.0
        mk = lambda a, b: ChangeCluster(a, b, np.zeros(1), np.zeros(1), 0.0)
        self.assertEqual(categorize(mk(4, 7), times, 30), "new")
        self.assertEqual(categorize(mk(0, 2), times, 30), "removed")
        self.assertEqual(categorize(mk(3, 4), times, 30), "moving")
        self.assertEqual(categorize(mk(1, 6), times, 30), "temporary")
        self.assertEqual(categorize(mk(0, 7), times, 30), "static")


class TestEndToEnd(unittest.TestCase):
    def test_simulated_series(self):
        sc = make_slc_time_series(seed=0)
        res = cs_change_detection(sc.stack, sc.times, band_fraction=0.8,
                                  cs_threshold=0.05, delta_t=30.0)
        found = {}
        for c, cat in zip(res.clusters, res.categories):
            for o in sc.change_objects():
                if o.contains(*c.center, pad=5):
                    found[o.name] = (c.a, c.b, cat)
        self.assertEqual(found.get("new building"), (4, 7, "new"))
        self.assertEqual(found.get("demolished building"), (0, 2, "removed"))
        self.assertEqual(found.get("container stack"), (3, 4, "moving"))
        self.assertEqual(len(res.clusters), len(found))  # no false changes
        o = next(o for o in sc.objects if o.name == "new building")
        r0, c0, h, w = o.rect
        self.assertGreater(res.change_map[r0:r0 + h, c0:c0 + w].mean(), 0.8)
        self.assertFalse(res.change_map[sc.vegetation].any())


if __name__ == "__main__":
    unittest.main()


class TestAmplitudePair(unittest.TestCase):
    def test_amplitude_only_pair(self):
        from change_detection.cs_amplitude import (
            amplitude_pair_change_detection, amplitude_similarity,
        )
        sc = make_slc_time_series(seed=1)
        a1, a2 = np.abs(sc.stack[2]), np.abs(sc.stack[4])
        sim = amplitude_similarity(a1, a1)
        self.assertGreater(sim.min(), 0.999)
        res = amplitude_pair_change_detection(a1, a2)
        found = {}
        for c, cat in zip(res.clusters, res.categories):
            for o in sc.objects:
                if o.contains(*c.center, pad=5):
                    found.setdefault(o.name, []).append(cat)
        self.assertEqual(found.get("new building"), ["appeared"])
        self.assertEqual(found.get("demolished building"), ["disappeared"])
        self.assertEqual(found.get("container stack"), ["appeared"])
        self.assertFalse(res.change_map[sc.vegetation].any())
