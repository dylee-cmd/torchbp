"""Coherent scatterer change detection for amplitude-only image pairs.

:func:`~change_detection.coherent_scatterers.cs_change_detection` needs
complex SLC data twice: the CS detection uses the phase of range sub-looks
and the CD metric is the interferometric coherence. Amplitude (detected)
images have neither, so both are replaced here by amplitude analogues and
passed to the unchanged rest of the pipeline (consistency check, lifetimes,
clustering, segmentation, change analysis):

- **CS proxy (bright point scatterers).** A pixel is a point scatterer
  candidate when it is a local amplitude maximum, at least
  ``scr_db`` above the local clutter median, and above ``a_min_db``. This
  keeps the "strong point scatterers of man-made objects" idea, but
  without the phase linearity test bright speckle peaks of distributed
  clutter can pass. ``scr_db`` controls that rate: for single-look
  speckle P(I > m * median) = 2^-m, about 1e-3 for 10 dB.
- **CD metric (amplitude similarity).** ``1 - NCB`` with Berger's operator,
  ``2 sum(a1 a2) / (sum a1^2 + sum a2^2)`` over a window: 1 for identical
  amplitudes, about 0.8 for unchanged speckled clutter, low where a
  dominant scatterer appeared or disappeared. Unlike coherence it cannot
  see changes that keep the amplitude (sub-pixel motion, phase-only
  changes) and it does not decorrelate over vegetation, so it only
  replaces the coherence where a CS is present, as in the original
  method.

With two images the lifetimes reduce to the image pair interpretation of
Table I: (a, b) = (0, 0) disappeared, (1, 1) appeared, (0, 1) unchanged.
Objects present in a single image are segmented with the amplitude
constraints of Section II-E.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy import ndimage

from .common import as_intensity
from .coherent_scatterers import CSChangeResult, cs_change_detection

__all__ = [
    "detect_bright_scatterers",
    "amplitude_similarity",
    "amplitude_pair_change_detection",
]


def detect_bright_scatterers(amp: np.ndarray, scr_db: float = 10.0,
                             a_min_db: float = -15.0, peak_size: int = 3,
                             background: int = 15) -> np.ndarray:
    """Point scatterer candidates of an amplitude image.

    Parameters
    ----------
    amp : ndarray
        Amplitude image (``|SLC|``; complex input uses the magnitude).
    scr_db : float
        Minimum signal to clutter ratio (in power dB) against the local
        median of the intensity in a ``background`` x ``background``
        window.
    a_min_db : float
        Minimum intensity in dB (``20 log10(amp)``).
    peak_size : int
        Neighborhood of the local maximum test.
    """
    a = as_intensity(amp)
    inten = a * a
    peak = inten >= ndimage.maximum_filter(inten, peak_size)
    med = ndimage.median_filter(inten, background)
    scr = inten >= med * 10 ** (scr_db / 10)
    bright = inten >= 10 ** (a_min_db / 10)
    return peak & scr & bright


def amplitude_similarity(amp1: np.ndarray, amp2: np.ndarray,
                         window=(9, 9)) -> np.ndarray:
    """Amplitude similarity ``1 - NCB`` in [0, 1] (high = unchanged)."""
    a = as_intensity(amp1)
    b = as_intensity(amp2)
    s12 = ndimage.uniform_filter(a * b, window)
    s11 = ndimage.uniform_filter(a * a, window)
    s22 = ndimage.uniform_filter(b * b, window)
    den = s11 + s22
    return np.where(den > 0, 2.0 * s12 / np.maximum(den, 1e-300), 1.0)


#: Pair change type of a cluster lifetime (Table I).
PAIR_LABELS = {(0, 0): "disappeared", (1, 1): "appeared", (0, 1): "unchanged"}


def amplitude_pair_change_detection(amp1: np.ndarray, amp2: np.ndarray, *,
                                    spacing=(1.0, 1.0), scr_db: float = 10.0,
                                    a_min_db: float = -15.0,
                                    window=(9, 9), gamma_t: float = 0.5,
                                    eps: float = 15.0, min_pts: int = 10,
                                    min_cs: int = 10, min_area: float = 20.0,
                                    delta_a: float = 3.0, a_min: float = -15.0,
                                    closing_radius: int = 5,
                                    **kwargs) -> CSChangeResult:
    """Coherent scatterer change detection of two amplitude images.

    The CS maps come from :func:`detect_bright_scatterers` and the CD metric
    from :func:`amplitude_similarity`; everything else is
    :func:`~change_detection.coherent_scatterers.cs_change_detection`. The
    returned ``categories`` are the pair change types "appeared" /
    "disappeared". The clustering defaults (``min_pts``, ``min_cs``) are
    lower than for SLC data because the local maximum test keeps at most
    one candidate per resolution cell. Other keyword arguments go to
    ``cs_change_detection``.
    """
    stack = [as_intensity(amp1), as_intensity(amp2)]
    cs = np.stack([detect_bright_scatterers(s, scr_db, a_min_db) for s in stack])
    metric = amplitude_similarity(stack[0], stack[1], window)[None]
    res = cs_change_detection(stack, [0.0, 1.0], spacing=spacing, gamma_t=gamma_t,
                              eps=eps, min_pts=min_pts, min_cs=min_cs,
                              min_area=min_area, delta_a=delta_a, a_min=a_min,
                              closing_radius=closing_radius, cs_maps=cs,
                              metric=metric, **kwargs)
    res.categories = [PAIR_LABELS[(c.a, c.b)] for c in res.clusters]
    return res
