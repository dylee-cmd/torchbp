"""Pure-PyTorch kernels of the image-domain utility operators: entropy,
CFAR, coherence, speckle filtering, resampling, element-wise interpolated
combination, subpixel correlation, the cfbp merge and the afbp spectrum
fusion."""
import math

import torch
from torch import Tensor

from . import register
from ._common import (
    PI, vjp, interp_2d, interp_1d_rows, knab_norm, expj_pi, geometry_dtype,
    chunk_size, chunks, is_undef,
)


def _batched(x: Tensor, nbatch: int, n0: int, n1: int) -> Tensor:
    return x.reshape(nbatch, n0, n1)


# ---------------------------------------------------------------------------
# Window sums
# ---------------------------------------------------------------------------

def _prefix_2d(x: Tensor) -> Tensor:
    """Zero-padded 2D inclusive prefix sum over the last two dims."""
    p = torch.cumsum(torch.cumsum(x, dim=-2), dim=-1)
    return torch.nn.functional.pad(p, (1, 0, 1, 0)) if not p.is_complex() else \
        torch.complex(torch.nn.functional.pad(p.real, (1, 0, 1, 0)),
                      torch.nn.functional.pad(p.imag, (1, 0, 1, 0)))


def _range_sum_2d(x: Tensor, lo0: Tensor, hi0: Tensor, lo1: Tensor,
                  hi1: Tensor) -> Tensor:
    """Sum of x[..., lo0:hi0, lo1:hi1] for every output pixel.

    ``lo0``/``hi0`` are per-row bounds (length N0 output rows), ``lo1``/
    ``hi1`` per-column bounds, exclusive upper ends. Empty ranges give 0.
    """
    p = _prefix_2d(x)
    hi0 = torch.maximum(hi0, lo0)
    hi1 = torch.maximum(hi1, lo1)
    a = p[..., hi0, :][..., :, hi1]
    b = p[..., lo0, :][..., :, hi1]
    c = p[..., hi0, :][..., :, lo1]
    d = p[..., lo0, :][..., :, lo1]
    return a - b - c + d


def _window_bounds(n: int, w: int, device, inclusive_hi: bool = True):
    i = torch.arange(n, device=device)
    lo = torch.clamp(i - w, min=0)
    if inclusive_hi:
        hi = torch.clamp(i + w, max=n - 1) + 1
    else:
        hi = torch.clamp(i + w, max=n - 1)
    return lo, hi


def _box(x: Tensor, w0: int, w1: int) -> Tensor:
    """Truncated (edge-clipped) box sum with half widths w0 (axis -2) and
    w1 (axis -1)."""
    n0, n1 = x.shape[-2], x.shape[-1]
    lo0, hi0 = _window_bounds(n0, w0, x.device)
    lo1, hi1 = _window_bounds(n1, w1, x.device)
    return _range_sum_2d(x, lo0, hi0, lo1, hi1)


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

@register("abs_sum")
def abs_sum(data: Tensor, nbatch: int) -> Tensor:
    x = data.reshape(nbatch, -1).abs()
    return x.sum(dim=1, keepdim=True, dtype=torch.float64).float()


@register("abs_sum_grad")
def abs_sum_grad(data: Tensor, grad: Tensor, nbatch: int) -> Tensor:
    d = data.reshape(nbatch, -1)
    g = grad.reshape(nbatch, 1).to(torch.float32)
    return (g * torch.sgn(d)).reshape(data.shape).to(data.dtype)


def _entropy_fwd(data: Tensor, norm: Tensor, nbatch: int) -> Tensor:
    x = data.reshape(nbatch, -1).abs() / norm.reshape(nbatch, 1)
    pos = x != 0
    xs = torch.where(pos, x, torch.ones_like(x))
    y = torch.where(pos, -xs * torch.log(xs), torch.zeros_like(x))
    return y.sum(dim=1)


@register("entropy")
def entropy(data: Tensor, norm: Tensor, nbatch: int) -> Tensor:
    return _entropy_fwd(data, norm, nbatch).float()


@register("entropy_grad")
def entropy_grad(data: Tensor, norm: Tensor, grad: Tensor, nbatch: int):
    return vjp(lambda d, n: _entropy_fwd(d, n, nbatch), [data, norm],
               [True, True], grad.reshape(nbatch))


# ---------------------------------------------------------------------------
# CFAR
# ---------------------------------------------------------------------------

@register("cfar_2d")
def cfar_2d(img: Tensor, nbatch: int, N0: int, N1: int, Navg0: int,
            Navg1: int, Nguard0: int, Nguard1: int, threshold: float,
            peaks_only: int) -> Tensor:
    x = _batched(img, nbatch, N0, N1)
    xd = x.to(torch.float64)
    ones = torch.ones((N0, N1), dtype=torch.float64, device=img.device)
    # Navg0/Nguard0 extend along the second (column) axis, Navg1/Nguard1
    # along the first (row) axis.
    s_out = _box(xd, Navg1, Navg0)
    s_in = _box(xd, Nguard1, Nguard0)
    c_out = _box(ones, Navg1, Navg0)
    c_in = _box(ones, Nguard1, Nguard0)
    avg = ((s_out - s_in) / (c_out - c_in)).to(torch.float32)
    det = torch.where(x > avg * threshold, x / avg, torch.zeros_like(x))
    if peaks_only:
        pad = torch.nn.functional.pad(x.unsqueeze(1), (1, 1, 1, 1),
                                      mode="replicate").squeeze(1)
        left = pad[:, 1:-1, :-2]
        right = pad[:, 1:-1, 2:]
        up = pad[:, :-2, 1:-1]
        down = pad[:, 2:, 1:-1]
        notpeak = (x < left) | (x < right) | (x < up) | (x < down)
        det = torch.where(notpeak, torch.zeros_like(det), det)
    return det.reshape(img.shape).clone()


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------

def _coherence_fwd(img0, img1, nbatch, N0, N1, w0, w1):
    a = _batched(img0, nbatch, N0, N1).to(torch.complex128)
    b = _batched(img1, nbatch, N0, N1).to(torch.complex128)
    # w0 is the half window along the column axis, w1 along the rows.
    corr = _box(a * b.conj(), w1, w0)
    p0 = _box(a.real ** 2 + a.imag ** 2, w1, w0)
    p1 = _box(b.real ** 2 + b.imag ** 2, w1, w0)
    denom = p0 * p1
    ok = denom > 0
    ds = torch.sqrt(torch.where(ok, denom, torch.ones_like(denom)))
    ca = torch.sqrt(corr.real ** 2 + corr.imag ** 2 + 1e-300)
    return torch.where(ok, ca / ds, torch.zeros_like(ds)).float()


@register("coherence_2d")
def coherence_2d(img0, img1, nbatch, N0, N1, w0, w1):
    return _coherence_fwd(img0, img1, nbatch, N0, N1, w0, w1)


@register("coherence_2d_grad")
def coherence_2d_grad(grad, img0, img1, nbatch, N0, N1, w0, w1):
    g = grad.reshape(nbatch, N0, N1)
    return vjp(lambda a, b: _coherence_fwd(a, b, nbatch, N0, N1, w0, w1),
               [img0, img1], [img0.requires_grad, img1.requires_grad], g)


@register("power_coherence_2d")
def power_coherence_2d(img0, img1, nbatch, N0, N1, w0, w1, corr_output):
    a = _batched(img0, nbatch, N0, N1).to(torch.complex128)
    b = _batched(img1, nbatch, N0, N1).to(torch.complex128)
    a2 = a.real ** 2 + a.imag ** 2
    b2 = b.real ** 2 + b.imag ** 2
    corr = _box(a2 * b2, w1, w0)
    p0 = _box(a2 * a2, w1, w0)
    p1 = _box(b2 * b2, w1, w0)
    denom = torch.sqrt(p0 * p1)
    v = torch.where(denom > 0, corr / torch.where(denom > 0, denom,
                    torch.ones_like(denom)), torch.zeros_like(denom))
    if corr_output:
        v = torch.where(v > 0.5, torch.sqrt(torch.clamp(2 * v - 1, min=0)),
                        torch.zeros_like(v))
    return v.float()


# ---------------------------------------------------------------------------
# Lee speckle filter
# ---------------------------------------------------------------------------

@register("lee_filter")
def lee_filter(img: Tensor, nbatch: int, Nx: int, Ny: int, wx: int, wy: int,
               cu: float) -> Tensor:
    x = _batched(img, nbatch, Nx, Ny)
    v = x.abs() if x.is_complex() else x
    v = v.to(torch.float64)
    nan = torch.isnan(v)
    v0 = torch.where(nan, torch.zeros_like(v), v)
    valid = (~nan).to(torch.float64)
    # Window [max(i-w, 0), min(i+w, N-1)) with exclusive upper bound.
    lo0, hi0 = _window_bounds(Nx, wx, img.device, inclusive_hi=False)
    lo1, hi1 = _window_bounds(Ny, wy, img.device, inclusive_hi=False)
    ones = torch.ones((Nx, Ny), dtype=torch.float64, device=img.device)
    count = _range_sum_2d(ones, lo0, hi0, lo1, hi1)
    n = _range_sum_2d(valid, lo0, hi0, lo1, hi1)
    s1 = _range_sum_2d(v0, lo0, hi0, lo1, hi1)
    nn = torch.clamp(n, min=1)
    mean = s1 / nn
    # Second moment about the window mean, two-pass form for accuracy.
    s2 = _range_sum_2d(v0 * v0, lo0, hi0, lo1, hi1)
    m2 = torch.clamp(s2 - s1 * mean, min=0)
    var = m2 / torch.clamp(count, min=1)
    ci = torch.sqrt(var) / mean
    w = torch.where(ci < cu, torch.zeros_like(ci), 1.0 - (cu * cu) / (ci * ci))
    out = mean + w * (v - mean)
    out = torch.where(count == 0, v, out)
    return out.float()


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _resample_2d(img, shift_r, shift_az, nbatch, Nr, Naz, order, kind, v):
    x = img.reshape(nbatch, Nr, Naz)
    gd = torch.float32
    ir = torch.arange(Nr, device=img.device, dtype=gd)[:, None]
    iaz = torch.arange(Naz, device=img.device, dtype=gd)[None, :]
    src_r = (ir + shift_r.reshape(Nr, Naz).to(gd))
    src_az = (iaz + shift_az.reshape(Nr, Naz).to(gd))
    a = 0.5 * order
    valid = (src_r >= -a) & (src_r < Nr + a) & (src_az >= -a) & (src_az < Naz + a)
    out = torch.zeros_like(x)
    sr = src_r.reshape(-1)
    sa = src_az.reshape(-1)
    T = order + 1
    step = chunk_size(Nr * Naz, T * T * nbatch)
    for s, e in chunks(Nr * Naz, step):
        for b in range(nbatch):
            base = torch.full((e - s,), b * Nr * Naz, dtype=torch.long,
                              device=img.device)
            val = interp_2d(x, Nr, Naz, sr[s:e], sa[s:e], kind, order, v=v,
                            base=base)
            out.view(nbatch, -1)[b, s:e] = val.to(x.dtype)
    out = torch.where(valid, out, torch.zeros_like(out))
    return out


@register("resample_2d_lanczos")
def resample_2d_lanczos(img, shift_r, shift_az, nbatch, Nr, Naz, order):
    return _resample_2d(img, shift_r, shift_az, nbatch, Nr, Naz, order,
                        "lanczos", 0.0)


@register("resample_2d_knab")
def resample_2d_knab(img, shift_r, shift_az, nbatch, Nr, Naz, order,
                     oversample):
    v = 1.0 - 1.0 / oversample
    return _resample_2d(img, shift_r, shift_az, nbatch, Nr, Naz, order,
                        "knab", v)


def _resample_1d(img, nbatch, N, M, order, kind, v):
    x = img.reshape(nbatch, N)
    step = torch.tensor(float(N), dtype=torch.float32) / float(M)
    src = torch.arange(M, device=img.device, dtype=torch.float32) * step.to(img.device)
    cutoff = 1.0 if M >= N else float(M) / float(N)
    out = torch.empty((nbatch, M), dtype=x.dtype, device=img.device)
    T = int(math.floor(order / cutoff)) + 1
    cs = chunk_size(M, T * nbatch)
    rows = torch.arange(nbatch, device=img.device)[:, None]
    for s, e in chunks(M, cs):
        val = interp_1d_rows(x, rows, src[None, s:e].expand(nbatch, -1), N,
                             kind, order, v=v, cutoff=cutoff)
        out[:, s:e] = val.to(x.dtype)
    return out


@register("resample_1d_lanczos")
def resample_1d_lanczos(img, nbatch, N, M, order):
    return _resample_1d(img, nbatch, N, M, order, "lanczos", 0.0)


@register("resample_1d_knab")
def resample_1d_knab(img, nbatch, N, M, order, oversample):
    v = 1.0 - 1.0 / oversample
    return _resample_1d(img, nbatch, N, M, order, "knab", v)


# ---------------------------------------------------------------------------
# Element-wise combination with a bilinearly resampled operand
# ---------------------------------------------------------------------------

def _interp2d_combine(a, b, nbatch, Na0, Na1, Nb0, Nb1, is_div):
    if a.is_complex() is False and b.is_complex():
        raise RuntimeError("Unsupported dtype combination for interp2d combine")
    dev = a.device
    f32 = torch.float32
    b0f = torch.arange(Na0, device=dev, dtype=f32) * float(Nb0 - 1) / float(Na0 - 1)
    b1f = torch.arange(Na1, device=dev, dtype=f32) * float(Nb1 - 1) / float(Na1 - 1)
    b0i = torch.floor(b0f)
    b1i = torch.floor(b1f)
    b0fr = (b0f - b0i)[:, None]
    b1fr = (b1f - b1i)[None, :]
    b0i = torch.clamp(torch.clamp(b0i.long(), max=Nb0 - 2), min=0)[:, None]
    b1i = torch.clamp(torch.clamp(b1i.long(), max=Nb1 - 2), min=0)[None, :]
    bb = b.reshape(nbatch, Nb0, Nb1)
    b1i1 = b1i + 1
    b0i1 = b0i + 1
    v = (bb[:, b0i, b1i] * (1 - b0fr) * (1 - b1fr)
         + bb[:, b0i, b1i1] * (1 - b0fr) * b1fr
         + bb[:, b0i1, b1i] * b0fr * (1 - b1fr)
         + bb[:, b0i1, b1i1] * b0fr * b1fr)
    aa = a.reshape(nbatch, Na0, Na1)
    v = v.to(aa.dtype)
    return aa / v if is_div else aa * v


@register("div_2d_interp_linear")
def div_2d_interp_linear(a, b, nbatch, Na0, Na1, Nb0, Nb1):
    return _interp2d_combine(a, b, nbatch, Na0, Na1, Nb0, Nb1, True)


@register("mul_2d_interp_linear")
def mul_2d_interp_linear(a, b, nbatch, Na0, Na1, Nb0, Nb1):
    return _interp2d_combine(a, b, nbatch, Na0, Na1, Nb0, Nb1, False)


# ---------------------------------------------------------------------------
# Subpixel correlation
# ---------------------------------------------------------------------------

@register("subpixel_correlation")
def subpixel_correlation(im_m, im_s, mean_m, mean_s, nbatch, N0, N1):
    m = im_m.reshape(nbatch, N0, N1).to(torch.complex128)
    s = im_s.reshape(nbatch, N0, N1).to(torch.complex128)
    mm = mean_m.reshape(nbatch, 1, 1).to(torch.complex128)
    ms = mean_s.reshape(nbatch, 1, 1).to(torch.complex128)
    s00 = s[:, :-1, :-1] - ms
    s01 = s[:, :-1, 1:] - ms
    s10 = s[:, 1:, :-1] - ms
    s11 = s[:, 1:, 1:] - ms
    vm = m[:, :-1, :-1] - mm
    a0 = s00
    a1 = s00 - s01
    a2 = s00 - s10
    a3 = s00 - s10 - s01 + s11

    def rsum(x):
        return x.real.sum(dim=(-2, -1))

    def cross(p, q):
        return 2.0 * (p * q.conj()).real

    b = torch.stack([(vm * a.conj()).sum(dim=(-2, -1))
                     for a in (a0, a1, a2, a3)], dim=1)
    r = torch.stack([
        rsum(a0 * a0.conj()),
        cross(a0, a1).sum(dim=(-2, -1)),
        cross(a0, a2).sum(dim=(-2, -1)),
        rsum(a1 * a1.conj()),
        rsum(a2 * a2.conj()),
        (cross(a0, a3) + cross(a1, a2)).sum(dim=(-2, -1)),
        cross(a1, a3).sum(dim=(-2, -1)),
        cross(a2, a3).sum(dim=(-2, -1)),
        rsum(a3 * a3.conj()),
    ], dim=1)
    c = (vm.real ** 2 + vm.imag ** 2).sum(dim=(-2, -1))
    return [r.float(), b.to(torch.complex64), c.float()]


# ---------------------------------------------------------------------------
# Cartesian factorized backprojection merge
# ---------------------------------------------------------------------------

@register("cfbp_merge2")
def cfbp_merge2(img0, img1, w0, idx0, w1, idx1, nbatch, Nx, Ny0, Ny1, Nyout,
                order0, order1, dx, dy, ox0, oy0, z0, ox1, oy1, z1, oxp, oyp,
                zp, ref_phase):
    dev = img0.device
    gd = geometry_dtype()
    ix = torch.arange(Nx, device=dev, dtype=gd)[:, None]
    iy = torch.arange(Nyout, device=dev, dtype=gd)[None, :]

    def dist(ox, oy, z):
        x = ox + ix * dx
        y = oy + iy * dy
        return torch.sqrt(x * x + z * z + y * y)

    dp = dist(oxp, oyp, zp)
    out = torch.zeros((nbatch, Nx, Nyout), dtype=torch.complex64, device=dev)
    for img, w, idx, order, ny, ox, oy, z in (
            (img0, w0, idx0, order0, Ny0, ox0, oy0, z0),
            (img1, w1, idx1, order1, Ny1, ox1, oy1, z1)):
        ph = expj_pi(ref_phase * (dist(ox, oy, z) - dp)).to(torch.complex64)
        im = img.reshape(nbatch, Nx, ny)
        taps = idx.reshape(Nyout, 1).long() + torch.arange(order, device=dev)
        taps = torch.clamp(taps, 0, ny - 1)
        ww = w.reshape(Nyout, order).to(torch.float32)
        acc = torch.zeros((nbatch, Nx, Nyout), dtype=torch.complex64, device=dev)
        for t in range(order):
            acc = acc + im[:, :, taps[:, t]] * ww[:, t]
        out = out + acc * ph
    return out


# ---------------------------------------------------------------------------
# afbp spectrum fusion
# ---------------------------------------------------------------------------

@register("afbp_fuse")
def afbp_fuse(S, nua, xs, inv_kr, x_half, band, x_c, x_taper, kmax):
    nsub, nrf, n_c = S.shape
    n_fine = nua.numel()
    dev = S.device
    x_eq = nua.reshape(1, -1) * inv_kr.reshape(-1, 1)
    bnd = band.reshape(-1, 1)
    off = x_c - 0.5 * bnd
    x = x_eq - off
    x_eq = x - bnd * torch.floor(x / bnd) + off
    xh = x_half.reshape(-1, 1)
    u0 = torch.bucketize(x_eq - xh, xs.contiguous())
    cols = (torch.arange(n_fine, device=dev) % n_c)[None, :]
    rows = torch.arange(nrf, device=dev)[:, None]
    taper = xh > x_taper
    span = torch.clamp(xh - x_taper, min=1e-9)
    fine = S.new_zeros((nrf, n_fine))
    for k in range(kmax):
        uk = u0 + k
        u = uk.clamp(max=nsub - 1)
        dist = (x_eq - xs[u]).abs()
        t = torch.clamp((dist - x_taper) / span, 0.0, 1.0)
        w_t = 0.5 * (1.0 + torch.cos(PI * t))
        w_h = (dist <= xh).to(w_t.dtype)
        w = torch.where(taper, w_t, w_h) * (uk < nsub)
        fine = fine + S[u, rows, cols] * w
    return fine
