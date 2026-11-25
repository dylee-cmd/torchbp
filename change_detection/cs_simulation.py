"""Simulated coregistered SLC time series for the coherent scatterer change
detection (:mod:`change_detection.coherent_scatterers`).

Each image is band-limited in both axes (``band_fraction`` of the sampled
band), so point scatterers at subpixel positions spread like a real SLC
impulse response and the range sub-looks behave as in Section II-A.

Scene content:

- distributed clutter (circular complex Gaussian): urban ground with
  partial temporal coherence, vegetation that decorrelates completely and
  changes its backscatter between acquisitions (seasonal / wind
  effects), and dark water;
- man-made objects: rectangles populated with bright point scatterers
  (CSs) whose complex amplitudes stay identical while the object is
  present. Each object has a lifetime [a, b] in image indices: static
  buildings (whole series), new buildings, demolished buildings, short
  lived objects (parked vehicles, containers);
- an optional transient change: one static building whose scatterers are
  attenuated and decorrelated in a single image (e.g. snow cover).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["SimObject", "SLCScene", "make_slc_time_series"]


@dataclass
class SimObject:
    name: str
    rect: Tuple[int, int, int, int]   # r0, c0, height, width (pixels)
    a: int                            # first image present
    b: int                            # last image present
    density: float = 0.12             # CSs per pixel
    amplitude_db: Tuple[float, float] = (20.0, 30.0)
    transient: Optional[int] = None   # image with a transient change

    def contains(self, r, c, pad=0):
        r0, c0, h, w = self.rect
        return (r >= r0 - pad) & (r < r0 + h + pad) & (c >= c0 - pad) & (c < c0 + w + pad)


@dataclass
class SLCScene:
    stack: List[np.ndarray]
    times: np.ndarray
    objects: List[SimObject]
    vegetation: np.ndarray
    water: np.ndarray

    def change_objects(self) -> List[SimObject]:
        n = len(self.stack)
        return [o for o in self.objects if not (o.a == 0 and o.b == n - 1)]

    def change_mask(self) -> np.ndarray:
        H, W = self.stack[0].shape
        m = np.zeros((H, W), dtype=bool)
        for o in self.change_objects():
            r0, c0, h, w = o.rect
            m[r0:r0 + h, c0:c0 + w] = True
        return m


def _default_objects(n: int) -> List[SimObject]:
    return [
        SimObject("static building A", (20, 30, 40, 60), 0, n - 1),
        SimObject("static building B (snow)", (150, 40, 50, 50), 0, n - 1,
                  transient=2 if n > 3 else None),
        SimObject("new building", (40, 200, 50, 70), min(4, n - 1), n - 1),
        SimObject("demolished building", (170, 250, 45, 60), 0, min(2, n - 2)),
        SimObject("container stack", (110, 140, 14, 30), min(3, n - 2), min(4, n - 2)),
        SimObject("parked aircraft", (210, 140, 16, 26), min(5, n - 2), min(5, n - 2),
                  density=0.2),
    ]


def make_slc_time_series(shape=(256, 384), n: int = 8,
                         times: Optional[Sequence[float]] = None,
                         objects: Optional[List[SimObject]] = None,
                         band_fraction: float = 0.8,
                         ground_coherence: float = 0.4,
                         decorrelation_time: float = 20.0,
                         seed: Optional[int] = 0) -> SLCScene:
    """Simulate a coregistered SLC time series (see the module docstring).

    Clutter has unit mean intensity on urban ground, 3 dB more with full
    decorrelation and +-3 dB backscatter changes on vegetation, and -15 dB
    on water. Urban ground decorrelates with the temporal baseline:
    coherence ``ground_coherence * exp(-dt / decorrelation_time)`` (an AR(1)
    process on top of a persistent fraction), as X-band distributed
    scatterers do. The default ``times`` are 11 days apart.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    times = np.arange(n) * 11.0 if times is None else np.asarray(times, dtype=float)
    objects = _default_objects(n) if objects is None else objects

    rr, cc = np.mgrid[0:H, 0:W]
    vegetation = (cc > W * 0.72) & (rr < H * 0.55)
    water = (rr > H * 0.88)
    sigma = np.ones((H, W))
    sigma[vegetation] = 10 ** 0.3
    sigma[water] = 10 ** -1.5

    # Band-limiting mask (separable), unit gain in the band.
    fr = np.fft.fftfreq(H) * H
    fc = np.fft.fftfreq(W) * W
    mr = np.abs(fr) < band_fraction * H / 2
    mc = np.abs(fc) < band_fraction * W / 2
    mask = mr[:, None] & mc[None, :]

    # Scatterers of every object: fixed positions and complex amplitudes.
    scat = []
    for o in objects:
        r0, c0, h, w = o.rect
        m = rng.poisson(o.density * h * w)
        pr = r0 + rng.random(m) * h
        pc = c0 + rng.random(m) * w
        amp = 10 ** (rng.uniform(*o.amplitude_db, m) / 20)
        ph = np.exp(2j * np.pi * rng.random(m))
        scat.append((pr, pc, amp * ph))

    def cnoise():
        return (rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W))) / np.sqrt(2)

    stack = []
    ar = cnoise()
    for t in range(n):
        if t > 0:
            q = np.exp(-(times[t] - times[t - 1]) / decorrelation_time)
            ar = q * ar + np.sqrt(1 - q * q) * cnoise()
        new = cnoise()
        rho = np.where(vegetation | water, 0.0, ground_coherence)
        z = np.sqrt(rho) * ar + np.sqrt(1 - rho) * new
        gain = np.where(vegetation, 10 ** (rng.uniform(-0.3, 0.3) / 2), 1.0)
        clutter = z * np.sqrt(sigma) * gain
        spec = np.fft.fft2(clutter)
        P, Q, A = [], [], []
        for o, (pr, pc, a) in zip(objects, scat):
            if not (o.a <= t <= o.b):
                continue
            if o.transient == t:
                # Snow: attenuated, decorrelated scatterers in this image.
                a = 0.3 * np.abs(a) * np.exp(2j * np.pi * rng.random(a.size))
            P.append(pr)
            Q.append(pc)
            A.append(a)
        if P:
            pr = np.concatenate(P)
            pc = np.concatenate(Q)
            a = np.concatenate(A)
            Er = np.exp(-2j * np.pi * np.outer(pr, fr) / H)    # [P, H]
            Ec = np.exp(-2j * np.pi * np.outer(pc, fc) / W)    # [P, W]
            spec = spec + (Er * a[:, None]).T @ Ec
        stack.append(np.fft.ifft2(spec * mask))
    return SLCScene(stack, times, objects, vegetation, water)
