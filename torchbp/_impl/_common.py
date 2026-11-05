"""Shared helpers of the pure-PyTorch operator implementations.

Everything here is plain differentiable PyTorch: the interpolation kernels
(Lanczos, Knab, polynomial), tap gathering for windowed interpolation,
bilinear lookups with the edge conventions of the original kernels, chunking
helpers and the generic vector-Jacobian product used by the ``*_grad`` ops.
"""
import math
from typing import Callable, Iterable, List, Optional, Sequence

import torch
from torch import Tensor

#: Speed of light in vacuum [m/s].
C0 = 299792458.0
PI = math.pi

# Smallest normal float32, threshold of the antenna weight normalization.
FLT_MIN = 1.17549435e-38

# Working precision of the geometry (distances) and of the phase arguments.
# Radar phases are large (4*pi*fc*R/c is ~1e5 rad at typical ranges), float32
# loses several mrad there. float64 is exact enough on any device; it can be
# switched to float32 for speed on GPUs with slow double precision.
_GEOM_DTYPE = torch.float64

# Approximate number of elements of the largest temporary of a chunked
# computation. Bounds the peak memory of the vectorized kernels.
_CHUNK_ELEMS = 1 << 22


def set_geometry_dtype(dtype: torch.dtype) -> None:
    """Set the precision used for geometry and phase computations.

    Parameters
    ----------
    dtype : torch.dtype
        ``torch.float64`` (default, accurate) or ``torch.float32`` (faster on
        GPUs with slow double precision, matches the precision of the original
        compiled kernels).
    """
    global _GEOM_DTYPE
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Unsupported geometry dtype {dtype}")
    _GEOM_DTYPE = dtype


def geometry_dtype() -> torch.dtype:
    return _GEOM_DTYPE


def set_chunk_elements(n: int) -> None:
    """Set the target element count of the largest chunked temporary."""
    global _CHUNK_ELEMS
    _CHUNK_ELEMS = max(1024, int(n))


def chunk_elements() -> int:
    return _CHUNK_ELEMS


def chunk_size(total: int, per_item: int) -> int:
    """Number of items per chunk such that ``chunk * per_item`` fits the
    chunk element budget."""
    return max(1, min(total, _CHUNK_ELEMS // max(1, per_item)))


def chunks(total: int, size: int) -> Iterable[tuple]:
    for s in range(0, total, size):
        yield s, min(total, s + size)


def cplx_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def expj(phase: Tensor) -> Tensor:
    """exp(1j * phase) in the complex dtype matching ``phase``."""
    return torch.polar(torch.ones_like(phase), phase)


def expj_pi(x: Tensor) -> Tensor:
    """exp(1j * pi * x) with exact half-cycle range reduction.

    ``x`` is in half-cycle units like the CUDA ``sincospif`` intrinsic. The
    reduction ``x - 2*round(x/2)`` is exact in floating point, so large
    arguments keep their accuracy.
    """
    r = x - 2.0 * torch.round(0.5 * x)
    return expj(PI * r)


def to_complex64(x: Tensor) -> Tensor:
    if x.dtype == torch.complex64:
        return x
    if x.dtype == torch.complex32:
        return x.to(torch.complex64)
    return x.to(torch.complex64)


def is_undef(t: Optional[Tensor]) -> bool:
    return t is None or t.numel() == 0


# ---------------------------------------------------------------------------
# Interpolation kernels
# ---------------------------------------------------------------------------

def lanczos_kernel(x: Tensor, a: float) -> Tensor:
    """Lanczos kernel sinc(x) * sinc(x / a). Taps with |x| > a are excluded
    by the callers' window."""
    return torch.sinc(x) * torch.sinc(x / a)


def knab_norm(order: int, v: float) -> float:
    a = 0.5 * order
    return math.exp(-2.0 * a * PI * v)


def knab_kernel(x: Tensor, a: float, v: float) -> Tensor:
    """Knab windowed sinc in the overflow-free exponential form.

    Equals sinc(x) * cosh(pi*v*a*sqrt(1-(x/a)^2)) / cosh(pi*v*a) for
    |x| < a and zero outside.
    """
    norm = math.exp(-2.0 * a * PI * v)
    xa = x / a
    s = torch.sqrt(torch.clamp(1.0 - xa * xa, min=0.0))
    n = torch.exp(PI * a * v * (s - 1.0))
    w = torch.sinc(x) * (norm / (n * (norm + 1.0)) + n / (norm + 1.0))
    return torch.where(x.abs() >= a, torch.zeros_like(w), w)


def poly_kernel(x: Tensor, a: float, coefs: Sequence[float]) -> Tensor:
    """Polynomial approximation kernel 1 + t*(c1 + c2*t + ...), t = x^2/a^2."""
    t = x * x / (a * a)
    inner = torch.full_like(t, float(coefs[-1]))
    for c in reversed(coefs[:-1]):
        inner = inner * t + float(c)
    return 1.0 + t * inner


def window_taps(pos: Tensor, n: int, half: float, ntaps: int):
    """Integer tap positions of a windowed interpolator.

    Taps cover ``ceil(pos - half) .. floor(pos + half)`` clamped to
    ``[0, n-1]``, the convention of all windowed kernels.

    Returns
    -------
    idx : Tensor
        [..., ntaps] long tap indices, clamped into the valid range.
    off : Tensor
        [..., ntaps] ``pos - idx`` offsets (float, the dtype of ``pos``).
    valid : Tensor
        [..., ntaps] bool tap validity.
    """
    start = torch.clamp(torch.ceil(pos - half), min=0)
    end = torch.clamp(torch.floor(pos + half), max=n - 1)
    k = torch.arange(ntaps, device=pos.device, dtype=pos.dtype)
    idxf = start.unsqueeze(-1) + k
    valid = idxf <= end.unsqueeze(-1)
    off = pos.unsqueeze(-1) - idxf
    idx = torch.clamp(idxf, 0, max(n - 1, 0)).long()
    return idx, off, valid


def ntaps_for(half: float) -> int:
    return int(math.floor(2.0 * half)) + 1


def tap_weights(kind: str, off: Tensor, order: int, *, v: float = 0.0,
                coefs: Optional[Sequence[float]] = None,
                cutoff: float = 1.0) -> Tensor:
    a = 0.5 * order
    if kind == "lanczos":
        return cutoff * lanczos_kernel(cutoff * off, a)
    if kind == "knab":
        return cutoff * knab_kernel(cutoff * off, a, v)
    if kind == "poly":
        return poly_kernel(off, a, coefs)
    raise ValueError(kind)


def interp_1d_rows(rows: Tensor, row_idx: Tensor, pos: Tensor, n: int,
                   kind: str, order: int, *, v: float = 0.0,
                   coefs=None, cutoff: float = 1.0) -> Tensor:
    """Windowed interpolation of 1D signals.

    Parameters
    ----------
    rows : Tensor
        [R, n] signals.
    row_idx : Tensor
        Long tensor of row indices into ``rows``, broadcastable with ``pos``.
    pos : Tensor
        Continuous sample positions.
    """
    half = 0.5 * order / cutoff
    T = ntaps_for(half)
    idx, off, valid = window_taps(pos, n, half, T)
    w = tap_weights(kind, off, order, v=v, coefs=coefs, cutoff=cutoff)
    w = torch.where(valid, w, torch.zeros_like(w))
    flat = rows.reshape(-1)
    lin = row_idx.unsqueeze(-1) * n + idx
    vals = torch.take(flat, lin)
    return (vals * w.to(vals.dtype if not vals.is_complex() else
                        (torch.float64 if vals.dtype == torch.complex128
                         else torch.float32))).sum(-1)


def interp_2d(img: Tensor, nx: int, ny: int, x: Tensor, y: Tensor,
              kind: str, order: int, *, v: float = 0.0, coefs=None,
              fmod_x: Optional[Tensor] = None,
              base: Optional[Tensor] = None) -> Tensor:
    """Separable windowed 2D interpolation.

    Parameters
    ----------
    img : Tensor
        Flattened images, ``img.reshape(-1)`` is indexed with
        ``base + ix * ny + iy``.
    x, y : Tensor
        Continuous positions (same shape), in samples of axis 0 and 1.
    fmod_x : Tensor, optional
        Demodulation of the x taps (radians per sample, broadcastable with
        ``x``): interpolates ``s(n) * exp(-1j*fmod_x*(n - x))``.
    base : Tensor, optional
        Long element offset of each position's image (batch offset).
    """
    a = 0.5 * order
    T = ntaps_for(a)
    ix, offx, vx = window_taps(x, nx, a, T)
    iy, offy, vy = window_taps(y, ny, a, T)
    wx = tap_weights(kind, offx, order, v=v, coefs=coefs)
    wy = tap_weights(kind, offy, order, v=v, coefs=coefs)
    wx = torch.where(vx, wx, torch.zeros_like(wx))
    wy = torch.where(vy, wy, torch.zeros_like(wy))
    flat = img.reshape(-1)
    lin = ix.unsqueeze(-1) * ny + iy.unsqueeze(-2)
    if base is not None:
        lin = lin + base.unsqueeze(-1).unsqueeze(-1)
    vals = torch.take(flat, lin)
    rdt = torch.float64 if vals.dtype in (torch.complex128, torch.float64) else torch.float32
    rows = (vals * wy.to(rdt).unsqueeze(-2)).sum(-1)
    wxc = wx.to(rdt)
    if fmod_x is not None:
        rot = expj(-fmod_x.unsqueeze(-1).to(offx.dtype) * (-offx))
        wxc = wxc * rot.to(cplx_dtype(rdt))
    return (rows * wxc).sum(-1)


def bilinear(img: Tensor, nx: int, ny: int, xi: Tensor, xf: Tensor,
             yi: Tensor, yf: Tensor, base: Optional[Tensor] = None) -> Tensor:
    """Bilinear interpolation at integer cell (xi, yi) with fractions
    (xf, yf). Indices must already be valid (``xi+1 < nx``, ``yi+1 < ny``)
    for the selected outputs; others are clamped and should be masked by the
    caller."""
    flat = img.reshape(-1)
    dx = 1 if nx > 1 else 0
    dy = 1 if ny > 1 else 0
    xi = torch.clamp(xi, 0, max(nx - 2, 0))
    yi = torch.clamp(yi, 0, max(ny - 2, 0))
    i00 = xi * ny + yi
    if base is not None:
        i00 = i00 + base
    v00 = torch.take(flat, i00)
    v01 = torch.take(flat, i00 + dy)
    v10 = torch.take(flat, i00 + dx * ny)
    v11 = torch.take(flat, i00 + (dx * ny + dy))
    if v00.is_complex():
        rdt = torch.float64 if v00.dtype == torch.complex128 else torch.float32
    else:
        rdt = v00.dtype
    xf = xf.to(rdt)
    yf = yf.to(rdt)
    a = v00 + yf * (v01 - v00)
    b = v10 + yf * (v11 - v10)
    return a + xf * (b - a)


def dem_sample(dem: Tensor, fr: Tensor, ft: Tensor) -> Tensor:
    """Edge-clamped bilinear DEM lookup at fractional cell (fr, ft).

    Same convention as the coherent kernels: ``i0 = min(int(f), n-1)``,
    ``i1 = min(i0+1, n-1)``. ``dem`` has shape [..., nr, nt]; the leading
    dims (channels) are kept.
    """
    nr, nt = dem.shape[-2], dem.shape[-1]
    ir0 = torch.clamp(fr.long(), max=nr - 1)
    ir1 = torch.clamp(ir0 + 1, max=nr - 1)
    wr = fr - ir0.to(fr.dtype)
    it0 = torch.clamp(ft.long(), max=nt - 1)
    it1 = torch.clamp(it0 + 1, max=nt - 1)
    wt = ft - it0.to(ft.dtype)
    d = dem.reshape(*dem.shape[:-2], nr * nt).to(fr.dtype)

    def g(i, j):
        return d[..., i * nt + j]

    za = g(ir0, it0) + wt * (g(ir0, it1) - g(ir0, it0))
    zb = g(ir1, it0) + wt * (g(ir1, it1) - g(ir1, it0))
    return za + wr * (zb - za)


def antenna_gain(g: Tensor, el: Tensor, az: Tensor, g_el0: float,
                 g_del: float, g_az0: float, g_daz: float):
    """Bilinear antenna gain lookup.

    Returns (gain, valid). Samples below the first row/column or past the
    last one are invalid and get zero gain.
    """
    g_nel, g_naz = g.shape
    el_idx = (el - g_el0) / g_del
    az_idx = (az - g_az0) / g_daz
    el_int = torch.trunc(el_idx)
    az_int = torch.trunc(az_idx)
    valid = ((el_idx >= 0) & (el_int + 1 < g_nel)
             & (az_idx >= 0) & (az_int + 1 < g_naz))
    ei = torch.where(valid, el_int, torch.zeros_like(el_int)).long()
    ai = torch.where(valid, az_int, torch.zeros_like(az_int)).long()
    ef = torch.where(valid, el_idx - el_int, torch.zeros_like(el_idx))
    af = torch.where(valid, az_idx - az_int, torch.zeros_like(az_idx))
    gain = bilinear(g.to(el.dtype), g_nel, g_naz, ei, ef, ai, af)
    gain = torch.where(valid, gain, torch.zeros_like(gain))
    return gain, valid


def pitch_rotate(el: Tensor, az: Tensor, pitch: Tensor):
    """Rotate the line of sight (el, az) about the antenna boresight x axis
    by ``pitch``. Zero pitch leaves the angles unchanged."""
    if not bool((pitch != 0).any()):
        return el, az
    ce = torch.cos(el)
    ux = ce * torch.cos(az)
    uy = ce * torch.sin(az)
    uz = torch.sin(el)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    uyp = cp * uy - sp * uz
    uzp = sp * uy + cp * uz
    el2 = torch.asin(torch.clamp(uzp, -1.0, 1.0))
    az2 = torch.atan2(uyp, ux)
    nz = pitch != 0
    return torch.where(nz, el2, el), torch.where(nz, az2, az)


def safe_sqrt(x: Tensor) -> Tensor:
    """sqrt(max(x, 0)) with a finite gradient at zero."""
    x = torch.clamp(x, min=0.0)
    pos = x > 0
    xs = torch.where(pos, x, torch.ones_like(x))
    return torch.where(pos, torch.sqrt(xs), torch.zeros_like(x))


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

def vjp(fn: Callable, inputs: List[Optional[Tensor]], wrt: List[bool],
        grad_out) -> List[Optional[Tensor]]:
    """Vector-Jacobian product of ``fn(*inputs)`` with ``grad_out``.

    ``wrt[i]`` selects the inputs to differentiate. Returns a list with the
    gradient of every selected input (zeros if it does not influence the
    output) and None for the others.
    """
    with torch.enable_grad():
        xs = []
        for x, w in zip(inputs, wrt):
            if isinstance(x, Tensor):
                x = x.detach()
                if w:
                    x = x.requires_grad_(True)
            xs.append(x)
        out = fn(*xs)
        targets = [x for x, w in zip(xs, wrt) if w and isinstance(x, Tensor)]
        if not targets:
            return [None] * len(inputs)
        if isinstance(out, (list, tuple)):
            outs = [o for o in out]
            gos = list(grad_out)
            pairs = [(o, g) for o, g in zip(outs, gos)
                     if o is not None and g is not None and o.requires_grad]
            outs = [p[0] for p in pairs]
            gos = [p[1] for p in pairs]
        else:
            outs, gos = [out], [grad_out]
            if not out.requires_grad:
                outs, gos = [], []
        if outs:
            grads = torch.autograd.grad(outs, targets, gos, allow_unused=True)
        else:
            grads = [None] * len(targets)
    res = []
    it = iter(grads)
    for x, w in zip(xs, wrt):
        if w and isinstance(x, Tensor):
            gx = next(it)
            if gx is None:
                gx = torch.zeros_like(x)
            res.append(gx.detach())
        else:
            res.append(None)
    return res
