"""Pure-PyTorch kernels of the backprojection family: polar and Cartesian
backprojection (linear, Lanczos and Knab range interpolation, antenna
pattern weighting, DEM), their gradients, GPGA target backprojection, the
block SVD autofocus inner products, forward projection, transmit power
normalization maps, illumination moments and the range dealias carrier."""
import math
from typing import Optional

import torch
from torch import Tensor

from . import register
from ._common import (
    C0, PI, FLT_MIN, geometry_dtype, chunk_size, chunks, expj, expj_pi,
    to_complex64, interp_1d_rows, antenna_gain, pitch_rotate, dem_sample,
    safe_sqrt, vjp, is_undef, ntaps_for,
)


# ---------------------------------------------------------------------------
# Pixel grids
# ---------------------------------------------------------------------------

def _polar_pixels(Nr, Ntheta, r0, dr, theta0, dtheta, dem, device):
    """Pixel coordinates of a polar grid, flattened [Nr*Ntheta].

    Returns x, y, z, extra, idr where ``extra = r^2 - x^2 - y^2`` is the
    correction that turns the Cartesian distance into the polar-form
    distance of the original kernels (zero inside |theta| <= 1, it continues
    the azimuth signal smoothly into the guard band past the fold).
    """
    gd = geometry_dtype()
    idr = torch.arange(Nr, device=device, dtype=gd)
    idt = torch.arange(Ntheta, device=device, dtype=gd)
    r = (r0 + idr * dr)[:, None]
    t = (theta0 + idt * dtheta)[None, :]
    ct2 = 1.0 - t * t
    ct = torch.sqrt(torch.clamp(ct2, min=0.0))
    x = r * ct
    y = r * t
    extra = r * r * (1.0 - ct * ct - t * t)
    if not is_undef(dem):
        dem_nr, dem_nt = dem.shape[-2], dem.shape[-1]
        fr = idr[:, None] * (dem_nr / Nr)
        ft = idt[None, :] * (dem_nt / Ntheta)
        z = dem_sample(dem, fr.expand(Nr, Ntheta), ft.expand(Nr, Ntheta))
    else:
        z = torch.zeros_like(x)
    x = x.expand(Nr, Ntheta)
    y = y.expand(Nr, Ntheta)
    extra = extra.expand(Nr, Ntheta)
    return (x.reshape(-1), y.reshape(-1), z.reshape(-1), extra.reshape(-1),
            r.expand(Nr, Ntheta).reshape(-1),
            idr[:, None].expand(Nr, Ntheta).reshape(-1))


# ---------------------------------------------------------------------------
# Backprojection core
# ---------------------------------------------------------------------------

def _range_sample(flat, rows, sx, N, method, order, knab_v):
    """Interpolate range compressed sweeps at fractional sample ``sx``.

    Returns (samples, valid)."""
    if method == "linear":
        id0 = torch.trunc(sx)
        valid = (sx >= 0) & (id0 <= N - 2)
        id0c = torch.where(valid, id0, torch.zeros_like(id0))
        frac = (sx - id0c).to(torch.float32)
        lin = rows * N + id0c.long()
        s0 = torch.take(flat, lin)
        s1 = torch.take(flat, lin + 1)
        s = s0 + frac * (s1 - s0)
    else:
        valid = (sx >= 0) & (sx < N - 1)
        sxc = torch.where(valid, sx, torch.zeros_like(sx))
        s = interp_1d_rows(flat, rows, sxc, N, method, order, v=knab_v)
    return s, valid


def _bp_accumulate(data, pos, att, x, y, z, extra, *, d0, r_res, fc,
                   data_fmod, method="linear", order=2, knab_v=0.0,
                   gain=None, gpos_valid_only=False):
    """Coherent sum over sweeps for a set of pixels.

    Parameters
    ----------
    data : Tensor
        [B, S, N] complex range compressed data.
    pos : Tensor
        [B, S, 3] sweep positions.
    att : Tensor or None
        [B, S, 3] antenna attitude, needed with ``gain``.
    x, y, z, extra : Tensor
        [P] pixel coordinates and the polar-form distance correction.
    gain : tuple, optional
        (g, g_el0, g_del, g_az0, g_daz) antenna pattern.

    Returns
    -------
    acc : Tensor
        [B, P] complex64 accumulation.
    w1, w2 : Tensor or None
        [B, P] sums of the (data window masked) gains and squared gains.
    """
    B, S, N = data.shape
    gd = geometry_dtype()
    flat = to_complex64(data).reshape(-1)
    pos = pos.reshape(B, S, 3).to(gd)
    if gain is not None:
        att = att.reshape(B, S, 3).to(gd)
    P = x.numel()
    dev = data.device
    taps = 2 if method == "linear" else ntaps_for(0.5 * order)
    ref_phase = 4.0 * fc / C0
    acc_parts, w1_parts, w2_parts = [], [], []
    per = B * (taps + 8)
    pc = chunk_size(P, per)
    brow = torch.arange(B, device=dev).reshape(B, 1, 1) * S
    for p0, p1 in chunks(P, pc):
        xs, ys, zs, es = (t[p0:p1].reshape(1, 1, -1) for t in (x, y, z, extra))
        n = p1 - p0
        acc = torch.zeros((B, n), dtype=torch.complex64, device=dev)
        w1 = w2 = None
        if gain is not None:
            w1 = torch.zeros((B, n), dtype=gd, device=dev)
            w2 = torch.zeros((B, n), dtype=gd, device=dev)
        sc = chunk_size(S, B * n * (taps + 8))
        for s0, s1 in chunks(S, sc):
            p = pos[:, s0:s1, :]
            pxs = p[..., 0:1]
            pys = p[..., 1:2]
            pzs = p[..., 2:3]
            dx = xs - pxs
            dy = ys - pys
            dz = zs - pzs
            d = safe_sqrt(dx * dx + dy * dy + dz * dz + es)
            sx = (d + d0) / r_res
            rows = brow + torch.arange(s0, s1, device=dev).reshape(1, -1, 1)
            samp, valid = _range_sample(flat, rows, sx, N, method, order,
                                        knab_v)
            phase = PI * ref_phase * d - data_fmod * sx
            contrib = samp * expj(phase).to(torch.complex64)
            if gain is not None:
                g, g_el0, g_del, g_az0, g_daz = gain
                dpos = d > 0
                ds = torch.where(dpos, d, torch.ones_like(d))
                sin_l = torch.where(dpos, dz / ds, torch.zeros_like(d))
                look = torch.asin(torch.clamp(sin_l, -1.0, 1.0))
                el = look - att[:, s0:s1, 0:1]
                az = torch.atan2(dy, dx) - att[:, s0:s1, 2:3]
                gv, _ = antenna_gain(g, el, az, g_el0, g_del, g_az0, g_daz)
                w = torch.where(valid, gv, torch.zeros_like(gv))
                contrib = contrib * w.to(torch.float32)
                w1 = w1 + w.sum(1)
                w2 = w2 + (w * w).sum(1)
            contrib = torch.where(valid, contrib, torch.zeros_like(contrib))
            acc = acc + contrib.sum(1)
        acc_parts.append(acc)
        if gain is not None:
            w1_parts.append(w1)
            w2_parts.append(w2)
    acc = torch.cat(acc_parts, dim=1)
    if gain is None:
        return acc, None, None
    return acc, torch.cat(w1_parts, 1), torch.cat(w2_parts, 1)


def _gain_args(g, g_az0, g_el0, g_daz, g_del):
    if is_undef(g):
        return None
    return (g, g_el0, g_del, g_az0, g_daz)


def _bp_polar(data, pos, att, nbatch, sweep_samples, nsweeps, fc, r_res, r0,
              dr, theta0, dtheta, Nr, Ntheta, d0, dealias, z0, g, g_az0,
              g_el0, g_daz, g_del, g_naz, g_nel, data_fmod, alias_fmod,
              normalize, dem, method="linear", order=2, knab_v=0.0):
    dev = data.device
    data = data.reshape(nbatch, nsweeps, sweep_samples)
    pos = pos.reshape(nbatch, nsweeps, 3)
    gain = _gain_args(g, g_az0, g_el0, g_daz, g_del)
    if gain is not None:
        att = att.reshape(nbatch, nsweeps, 3)
    x, y, z, extra, r, idr = _polar_pixels(Nr, Ntheta, r0, dr, theta0,
                                           dtheta, dem, dev)
    acc, w1, w2 = _bp_accumulate(
        data, pos, att, x, y, z, extra, d0=d0, r_res=r_res, fc=fc,
        data_fmod=data_fmod, method=method, order=order, knab_v=knab_v,
        gain=gain)
    if gain is not None and normalize:
        ok = w2 >= FLT_MIN
        scale = torch.where(ok, w1 / torch.where(ok, w2, torch.ones_like(w2)),
                            torch.ones_like(w2))
        acc = acc * scale.to(torch.float32)
    if dealias:
        ref_phase = 4.0 * fc / C0
        zz = z0 - z
        dd = torch.sqrt(r * r + zz * zz)
        ph = -PI * ref_phase * dd + alias_fmod * idr
        acc = acc * expj(ph).to(torch.complex64)
    img = acc.reshape(nbatch, Nr, Ntheta)
    if w1 is not None:
        w1 = w1.reshape(nbatch, Nr, Ntheta).float()
        w2 = w2.reshape(nbatch, Nr, Ntheta).float()
    return img, w1, w2


@register("backprojection_polar_2d")
def backprojection_polar_2d(data, pos, att, nbatch, sweep_samples, nsweeps,
                            fc, r_res, r0, dr, theta0, dtheta, Nr, Ntheta, d0,
                            dealias, z0, g, g_az0, g_el0, g_daz, g_del, g_naz,
                            g_nel, data_fmod, alias_fmod, normalize, dem):
    return _bp_polar(data, pos, att, nbatch, sweep_samples, nsweeps, fc,
                     r_res, r0, dr, theta0, dtheta, Nr, Ntheta, d0, dealias,
                     z0, g, g_az0, g_el0, g_daz, g_del, g_naz, g_nel,
                     data_fmod, alias_fmod, normalize, dem)[0]


@register("backprojection_polar_2d_illum")
def backprojection_polar_2d_illum(data, pos, att, nbatch, sweep_samples,
                                  nsweeps, fc, r_res, r0, dr, theta0, dtheta,
                                  Nr, Ntheta, d0, dealias, z0, g, g_az0,
                                  g_el0, g_daz, g_del, g_naz, g_nel,
                                  data_fmod, alias_fmod, normalize, dem):
    if is_undef(g):
        raise RuntimeError(
            "backprojection_polar_2d_illum requires an antenna pattern")
    return list(_bp_polar(data, pos, att, nbatch, sweep_samples, nsweeps, fc,
                          r_res, r0, dr, theta0, dtheta, Nr, Ntheta, d0,
                          dealias, z0, g, g_az0, g_el0, g_daz, g_del, g_naz,
                          g_nel, data_fmod, alias_fmod, normalize, dem))


@register("backprojection_polar_2d_lanczos")
def backprojection_polar_2d_lanczos(data, pos, att, nbatch, sweep_samples,
                                    nsweeps, fc, r_res, r0, dr, theta0,
                                    dtheta, Nr, Ntheta, d0, dealias, z0,
                                    order, g, g_az0, g_el0, g_daz, g_del,
                                    g_naz, g_nel, data_fmod, alias_fmod,
                                    normalize, dem):
    return _bp_polar(data, pos, att, nbatch, sweep_samples, nsweeps, fc,
                     r_res, r0, dr, theta0, dtheta, Nr, Ntheta, d0, dealias,
                     z0, g, g_az0, g_el0, g_daz, g_del, g_naz, g_nel,
                     data_fmod, alias_fmod, normalize, dem,
                     method="lanczos", order=order)[0]


def _knab_v(oversample, order=2):
    if not (2 <= order <= 16 and order % 2 == 0):
        raise RuntimeError(
            f"backprojection_polar_2d_knab: order must be even and in [2, 16], got {order}")
    if oversample < 1.0:
        raise RuntimeError(f"oversample should be >= 1, got {oversample}")
    return 1.0 - 1.0 / oversample


@register("backprojection_polar_2d_knab")
def backprojection_polar_2d_knab(data, pos, att, nbatch, sweep_samples,
                                 nsweeps, fc, r_res, r0, dr, theta0, dtheta,
                                 Nr, Ntheta, d0, dealias, z0, order,
                                 oversample, g, g_az0, g_el0, g_daz, g_del,
                                 g_naz, g_nel, data_fmod, alias_fmod,
                                 normalize, dem):
    return _bp_polar(data, pos, att, nbatch, sweep_samples, nsweeps, fc,
                     r_res, r0, dr, theta0, dtheta, Nr, Ntheta, d0, dealias,
                     z0, g, g_az0, g_el0, g_daz, g_del, g_naz, g_nel,
                     data_fmod, alias_fmod, normalize, dem, method="knab",
                     order=order, knab_v=_knab_v(oversample, order))[0]


@register("backprojection_polar_2d_knab_illum")
def backprojection_polar_2d_knab_illum(data, pos, att, nbatch, sweep_samples,
                                       nsweeps, fc, r_res, r0, dr, theta0,
                                       dtheta, Nr, Ntheta, d0, dealias, z0,
                                       order, oversample, g, g_az0, g_el0,
                                       g_daz, g_del, g_naz, g_nel, data_fmod,
                                       alias_fmod, normalize, dem):
    if is_undef(g):
        raise RuntimeError(
            "backprojection_polar_2d_knab_illum requires an antenna pattern")
    return list(_bp_polar(data, pos, att, nbatch, sweep_samples, nsweeps, fc,
                          r_res, r0, dr, theta0, dtheta, Nr, Ntheta, d0,
                          dealias, z0, g, g_az0, g_el0, g_daz, g_del, g_naz,
                          g_nel, data_fmod, alias_fmod, normalize, dem,
                          method="knab", order=order,
                          knab_v=_knab_v(oversample, order)))


@register("backprojection_polar_2d_grad")
def backprojection_polar_2d_grad(grad, data, pos, att, nbatch, sweep_samples,
                                 nsweeps, fc, r_res, r0, dr, theta0, dtheta,
                                 Nr, Ntheta, d0, dealias, z0, g, g_az0, g_el0,
                                 g_daz, g_del, g_naz, g_nel, data_fmod,
                                 alias_fmod, normalize, dem):
    def fwd(data_, pos_):
        return _bp_polar(data_, pos_, att, nbatch, sweep_samples, nsweeps,
                         fc, r_res, r0, dr, theta0, dtheta, Nr, Ntheta, d0,
                         dealias, z0, g, g_az0, g_el0, g_daz, g_del, g_naz,
                         g_nel, data_fmod, alias_fmod, normalize, dem)[0]

    gr = grad.reshape(nbatch, Nr, Ntheta).to(torch.complex64)
    return vjp(fwd, [data, pos], [data.requires_grad, pos.requires_grad], gr)


# ---------------------------------------------------------------------------
# Cartesian backprojection
# ---------------------------------------------------------------------------

def _cart_pixels(Nx, Ny, x0, dx, y0, dy, device):
    gd = geometry_dtype()
    ix = torch.arange(Nx, device=device, dtype=gd)
    iy = torch.arange(Ny, device=device, dtype=gd)
    x = (x0 + ix * dx)[:, None].expand(Nx, Ny).reshape(-1)
    y = (y0 + iy * dy)[None, :].expand(Nx, Ny).reshape(-1)
    return x, y


def _bp_cart(data, pos, nbatch, sweep_samples, nsweeps, fc, r_res, x0, dx,
             y0, dy, Nx, Ny, d0, data_fmod):
    x, y = _cart_pixels(Nx, Ny, x0, dx, y0, dy, data.device)
    z = torch.zeros_like(x)
    acc, _, _ = _bp_accumulate(
        data.reshape(nbatch, nsweeps, sweep_samples),
        pos.reshape(nbatch, nsweeps, 3), None, x, y, z, z,
        d0=d0, r_res=r_res, fc=fc, data_fmod=data_fmod)
    return acc.reshape(nbatch, Nx, Ny)


@register("backprojection_cart_2d")
def backprojection_cart_2d(data, pos, nbatch, sweep_samples, nsweeps, fc,
                           r_res, x0, dx, y0, dy, Nx, Ny, beamwidth, d0,
                           data_fmod):
    return _bp_cart(data, pos, nbatch, sweep_samples, nsweeps, fc, r_res, x0,
                    dx, y0, dy, Nx, Ny, d0, data_fmod)


@register("backprojection_cart_2d_grad")
def backprojection_cart_2d_grad(grad, data, pos, nbatch, sweep_samples,
                                nsweeps, fc, r_res, x0, dx, y0, dy, Nx, Ny,
                                beamwidth, d0, data_fmod):
    def fwd(data_, pos_):
        return _bp_cart(data_, pos_, nbatch, sweep_samples, nsweeps, fc,
                        r_res, x0, dx, y0, dy, Nx, Ny, d0, data_fmod)

    gr = grad.reshape(nbatch, Nx, Ny).to(torch.complex64)
    return vjp(fwd, [data, pos], [data.requires_grad, pos.requires_grad], gr)


# ---------------------------------------------------------------------------
# Range dealias carrier
# ---------------------------------------------------------------------------

@register("polar_range_dealias")
def polar_range_dealias(img, dem, nbatch, Nr, Ntheta, fc, r0, dr, theta0,
                        dtheta, ox, oy, oz, alias_fmod):
    x, y, z, extra, r, idr = _polar_pixels(Nr, Ntheta, r0, dr, theta0,
                                           dtheta, dem, img.device)
    zz = oz - z
    d2 = r * r - 2.0 * (x * ox + y * oy) + ox * ox + oy * oy + zz * zz
    d = torch.sqrt(torch.clamp(d2, min=0.0))
    ref_phase = 4.0 * fc / C0
    ref = expj(-PI * ref_phase * d + alias_fmod * idr).to(torch.complex64)
    return (img.reshape(-1, Nr * Ntheta) * ref).reshape(img.shape)


# ---------------------------------------------------------------------------
# GPGA
# ---------------------------------------------------------------------------

def _gpga(target_pos, data, pos, sweep_samples, nsweeps, fc, r_res, Ntarget,
          d0, data_fmod, method, order):
    gd = geometry_dtype()
    dev = data.device
    flat = to_complex64(data).reshape(-1)
    tp = target_pos.reshape(Ntarget, 3).to(gd)
    ps = pos.reshape(nsweeps, 3).to(gd)
    ref_phase = 4.0 * fc / C0
    out = torch.zeros((Ntarget, nsweeps), dtype=torch.complex64, device=dev)
    taps = 2 if method == "linear" else order + 1
    tc = chunk_size(Ntarget, nsweeps * (taps + 6))
    rows = torch.arange(nsweeps, device=dev)[None, :]
    for t0, t1 in chunks(Ntarget, tc):
        diff = tp[t0:t1, None, :] - ps[None, :, :]
        d = torch.sqrt((diff * diff).sum(-1))
        sx = (d + d0) / r_res
        id0 = torch.trunc(sx)
        valid = (sx >= 0) & (id0 + 1 < sweep_samples)
        if method == "linear":
            s, _ = _range_sample(flat, rows, sx, sweep_samples, "linear", 2,
                                 0.0)
        else:
            sxc = torch.where(valid, sx, torch.zeros_like(sx))
            s = interp_1d_rows(flat, rows.expand_as(sxc), sxc, sweep_samples,
                               "lanczos", order)
        v = s * expj(PI * ref_phase * d - data_fmod * sx).to(torch.complex64)
        out[t0:t1] = torch.where(valid, v, torch.zeros_like(v))
    return out


@register("gpga_backprojection_2d")
def gpga_backprojection_2d(target_pos, data, pos, sweep_samples, nsweeps, fc,
                           r_res, Ntarget, d0, data_fmod):
    return _gpga(target_pos, data, pos, sweep_samples, nsweeps, fc, r_res,
                 Ntarget, d0, data_fmod, "linear", 2)


@register("gpga_backprojection_2d_lanczos")
def gpga_backprojection_2d_lanczos(target_pos, data, pos, sweep_samples,
                                   nsweeps, fc, r_res, Ntarget, d0, order,
                                   data_fmod):
    return _gpga(target_pos, data, pos, sweep_samples, nsweeps, fc, r_res,
                 Ntarget, d0, data_fmod, "lanczos", order)


# ---------------------------------------------------------------------------
# Block SVD autofocus inner products
# ---------------------------------------------------------------------------

@register("blocksvd_alpha")
def blocksvd_alpha(img, data, pos, blocks, sweep_samples, nsweeps, nblocks,
                   Ntheta, fc, r_res, r0, dr, theta0, dtheta, d0, data_fmod,
                   dem):
    gd = geometry_dtype()
    dev = data.device
    flat = to_complex64(data).reshape(-1)
    Nr_img = img.shape[0]
    im = img.reshape(Nr_img, Ntheta)
    ps = pos.reshape(nsweeps, 3).to(gd)
    ref_phase = 4.0 * fc / C0
    alpha = torch.zeros((nblocks, nsweeps), dtype=torch.complex64, device=dev)
    bl = blocks.reshape(nblocks, 6).tolist()
    for ib, (ri0, ri1, ti0, ti1, slo, shi) in enumerate(bl):
        if ri1 <= ri0 or ti1 <= ti0 or shi <= slo:
            continue
        idr = torch.arange(ri0, ri1, device=dev, dtype=gd)
        idt = torch.arange(ti0, ti1, device=dev, dtype=gd)
        r = (r0 + idr * dr)[:, None]
        t = (theta0 + idt * dtheta)[None, :]
        ct = torch.sqrt(torch.clamp(1.0 - t * t, min=0.0))
        x = (r * ct).reshape(-1)
        y = (r * t).expand(ri1 - ri0, ti1 - ti0).reshape(-1)
        if not is_undef(dem):
            dem_nr, dem_nt = dem.shape
            fr = (idr[:, None] * (dem_nr / Nr_img)).expand(ri1 - ri0, ti1 - ti0)
            ft = (idt[None, :] * (dem_nt / Ntheta)).expand(ri1 - ri0, ti1 - ti0)
            z = dem_sample(dem, fr, ft).reshape(-1)
        else:
            z = torch.zeros_like(x)
        w = im[ri0:ri1, ti0:ti1].reshape(-1).conj().to(torch.complex64)
        npix = x.numel()
        sc = chunk_size(shi - slo, npix * 8)
        for s0, s1 in chunks(shi - slo, sc):
            s0 += slo
            s1 += slo
            p = ps[s0:s1, None, :]
            dxp = x[None, :] - p[..., 0]
            dyp = y[None, :] - p[..., 1]
            dzp = z[None, :] - p[..., 2]
            d = torch.sqrt(dxp * dxp + dyp * dyp + dzp * dzp)
            sx = (d + d0) / r_res
            rows = torch.arange(s0, s1, device=dev)[:, None]
            s, valid = _range_sample(flat, rows, sx, sweep_samples, "linear",
                                     2, 0.0)
            v = s * expj(PI * ref_phase * d - data_fmod * sx).to(torch.complex64)
            v = torch.where(valid, v, torch.zeros_like(v))
            alpha[ib, s0:s1] = v @ w
    return alpha


# ---------------------------------------------------------------------------
# Forward projection
# ---------------------------------------------------------------------------

def _projection_geometry(x, y, z, p, at, gain, normalization):
    """Per (sweep, pixel) distance and amplitude weight of the forward
    projection. ``p``/``at`` are [B, s, 3], pixels [P]."""
    dpx = x - p[..., 0:1]
    dpy = y - p[..., 1:2]
    dpz = z - p[..., 2:3]
    d2 = dpx * dpx + dpy * dpy + dpz * dpz
    d = torch.sqrt(d2)
    norm = 1.0 / d2
    if gain is not None:
        gg, g_el0, g_del, g_az0, g_daz = gain
        look = torch.asin(torch.clamp(dpz / d, min=-1.0))
        el = look - at[..., 0:1]
        az = torch.atan2(dpy, dpx) - at[..., 2:3]
        gv, gvalid = antenna_gain(gg, el, az, g_el0, g_del, g_az0, g_daz)
        norm = torch.where(gvalid, norm * gv, torch.zeros_like(norm))
    if normalization == 1:
        norm = norm * torch.sqrt(-dpz / d)
    return dpx, dpy, dpz, d, norm


def _kb_kernel(t, W, beta):
    """Kaiser-Bessel gridding kernel, support |t| < W/2 (grid samples)."""
    u = 1.0 - (2.0 * t / W) ** 2
    v = torch.i0(beta * torch.sqrt(torch.clamp(u, min=0.0)))
    return torch.where(u > 0, v, torch.zeros_like(v))


_NUFFT_W = 10
_NUFFT_SIGMA = 2


def _nufft_type1(c, xf, N):
    """f[..., j] = sum_p c[..., p] * exp(-2j*pi*xf[..., p]*j), j = 0..N-1.

    Kaiser-Bessel gridding on a 2x oversampled periodic grid, FFT and
    deconvolution. ``xf`` is in cycles per output sample (float64), ``c``
    complex. Leading dims are batched.
    """
    W = _NUFFT_W
    sig = _NUFFT_SIGMA
    M = sig * N
    beta = PI * math.sqrt((W / sig) ** 2 * (sig - 0.5) ** 2 - 0.8)
    lead = c.shape[:-1]
    R = int(torch.tensor(lead).prod()) if len(lead) else 1
    P = c.shape[-1]
    c = c.reshape(R, P).to(torch.complex128)
    xf = xf.reshape(R, P).to(torch.float64)
    # Center the outputs: k = j - N//2 in [-N/2, N/2).
    h = N // 2
    frac = xf - torch.floor(xf)
    c = c * expj(-2.0 * PI * ((frac * h) - torch.floor(frac * h)))
    u = frac * M
    i0 = torch.floor(u).long()
    off = torch.arange(-(W // 2) + 1, W // 2 + 1, device=c.device)
    idx = i0.unsqueeze(-1) + off
    wt = _kb_kernel(u.unsqueeze(-1) - idx.to(torch.float64), W, beta)
    idx = torch.remainder(idx, M)
    rows = torch.arange(R, device=c.device).reshape(R, 1, 1) * M
    flat_idx = (rows + idx).reshape(-1)
    vals = (c.unsqueeze(-1) * wt).reshape(-1)
    grid = torch.zeros(R * M, dtype=torch.float64, device=c.device)
    gi = torch.zeros(R * M, dtype=torch.float64, device=c.device)
    grid.index_add_(0, flat_idx, vals.real)
    gi.index_add_(0, flat_idx, vals.imag)
    g = torch.complex(grid, gi).reshape(R, M)
    F = torch.fft.fft(g, dim=-1)
    k = torch.arange(N, device=c.device) - h
    Fk = F[:, torch.remainder(k, M)]
    # Continuous Fourier transform of the kernel at k/M, by quadrature.
    nq = 4096
    tq = (torch.arange(nq, dtype=torch.float64, device=c.device) + 0.5) / nq * (W / 2)
    phiq = _kb_kernel(tq, W, beta)
    xi = k.to(torch.float64) / M
    phihat = 2.0 * (phiq[None, :] * torch.cos(2.0 * PI * xi[:, None] * tq[None, :])).sum(-1) * (W / 2) / nq
    f = Fk / phihat
    return f.reshape(*lead, N)


def _projection(img, dem, pos, vel, att, nbatch, sweep_samples, nsweeps, fc,
                fs, gamma, x0, dx, y0, dy, Nx, Ny, d0, g, g_az0, g_el0, g_daz,
                g_del, g_naz, g_nel, use_rvp, normalization, use_nufft=False):
    gd = geometry_dtype()
    dev = img.device
    P = Nx * Ny
    N = sweep_samples
    x, y = _cart_pixels(Nx, Ny, x0, dx, y0, dy, dev)
    if not is_undef(dem):
        z = dem.reshape(-1).to(gd)
    else:
        z = torch.zeros_like(x)
    w_img = to_complex64(img).reshape(nbatch, 1, P)
    ps = pos.reshape(nbatch, nsweeps, 3).to(gd)
    has_vel = not is_undef(vel)
    if has_vel:
        vs = vel.reshape(nbatch, nsweeps, 3).to(gd)
    gain = _gain_args(g, g_az0, g_el0, g_daz, g_del)
    at = att.reshape(nbatch, nsweeps, 3).to(gd) if gain is not None else None
    j = torch.arange(N, device=dev, dtype=gd)
    j_half = j - 0.5 * N
    out = torch.zeros((nbatch, nsweeps, N), dtype=torch.complex64, device=dev)
    per_sweep = nbatch * P * (_NUFFT_W * 3 if use_nufft else 12)
    sc = chunk_size(nsweeps, per_sweep)
    for s0, s1 in chunks(nsweeps, sc):
        p = ps[:, s0:s1, :]
        a = at[:, s0:s1, :] if at is not None else None
        dpx, dpy, dpz, d, norm = _projection_geometry(x, y, z, p, a, gain,
                                                      normalization)
        w = w_img * norm.to(torch.float32)  # [B, s, P]
        if has_vel:
            v = vs[:, s0:s1, :]
            bproj = (dpx * v[..., 0:1] + dpy * v[..., 1:2]
                     + dpz * v[..., 2:3]) / d / fs
            slope = -2.0 * gamma * j / fs - 2.0 * fc
            pc = chunk_size(P, nbatch * (s1 - s0) * N * 3)
            for q0, q1 in chunks(P, pc):
                tau_s = 2.0 * (d[..., q0:q1, None] + d0
                               + bproj[..., q0:q1, None] * j_half) / C0
                ph = tau_s * slope
                if use_rvp:
                    ph = ph + gamma * tau_s * tau_s
                e = expj_pi(ph).to(torch.complex64)
                out[:, s0:s1, :] += torch.einsum("bsp,bspj->bsj", w[..., q0:q1], e)
            continue
        # Without velocity the phase separates into a per-pixel constant and
        # a per-pixel frequency: phase(j) = tau*(-2*fc) + gamma*tau^2 [rvp]
        # - 2*gamma*tau/fs * j (half cycles).
        tau = 2.0 * (d + d0) / C0
        ph0 = -2.0 * fc * tau
        if use_rvp:
            ph0 = ph0 + gamma * tau * tau
        c = w * expj_pi(ph0).to(torch.complex64)
        xf = gamma * tau / fs  # cycles per sample, exp(-2j*pi*xf*j)
        if use_nufft:
            out[:, s0:s1, :] = _nufft_type1(c, xf, N).to(torch.complex64)
            continue
        pc = chunk_size(P, nbatch * (s1 - s0) * N * 3)
        for q0, q1 in chunks(P, pc):
            cyc = xf[..., q0:q1, None] * j
            cyc = cyc - torch.round(cyc)
            e = expj(-2.0 * PI * cyc).to(torch.complex64)
            out[:, s0:s1, :] += torch.einsum("bsp,bspj->bsj", c[..., q0:q1], e)
    return out


@register("projection_cart_2d")
def projection_cart_2d(img, dem, pos, vel, att, nbatch, sweep_samples,
                       nsweeps, fc, fs, gamma, x0, dx, y0, dy, Nx, Ny, d0, g,
                       g_az0, g_el0, g_daz, g_del, g_naz, g_nel, use_rvp,
                       normalization):
    return _projection(img, dem, pos, vel, att, nbatch, sweep_samples,
                       nsweeps, fc, fs, gamma, x0, dx, y0, dy, Nx, Ny, d0, g,
                       g_az0, g_el0, g_daz, g_del, g_naz, g_nel, use_rvp,
                       normalization)


@register("projection_cart_2d_nufft")
def projection_cart_2d_nufft(img, dem, pos, att, nbatch, sweep_samples,
                             nsweeps, fc, fs, gamma, x0, dx, y0, dy, Nx, Ny,
                             d0, g, g_az0, g_el0, g_daz, g_del, g_naz, g_nel,
                             use_rvp, normalization):
    # Type-1 NUFFT over the pixel beat frequencies: O(P*W + M log M) per
    # sweep instead of the O(P*M) direct sum.
    return _projection(img, dem, pos, None, att, nbatch, sweep_samples,
                       nsweeps, fc, fs, gamma, x0, dx, y0, dy, Nx, Ny, d0, g,
                       g_az0, g_el0, g_daz, g_del, g_naz, g_nel, use_rvp,
                       normalization, use_nufft=True)


# ---------------------------------------------------------------------------
# Transmit power normalization
# ---------------------------------------------------------------------------

def _tx_power_moments(px_base, py_base, use_h_fixed, h_fixed, z_base, dzdx,
                      dzdy, pos, att, g, g_az0, g_el0, g_daz, g_del, wa,
                      normalization, min_sin2):
    """Vectorized ``tx_power_pixel_moments``.

    px_base, py_base, z_base, dzdx, dzdy, min_sin2 : [P] (z_base None
    without a DEM). pos, att : [S, 3]. wa : [S].

    Returns S, W, mean, M2 : [P].
    """
    # Amplitude-only computation: float32 geometry is accurate enough here
    # (no carrier phase), the accumulators are kept in float64.
    gd = torch.float32
    has_dem = z_base is not None
    px_base = px_base.to(gd)
    py_base = py_base.to(gd)
    min_sin2 = min_sin2.to(gd)
    if has_dem:
        z_base, dzdx, dzdy = z_base.to(gd), dzdx.to(gd), dzdy.to(gd)
    pos = pos.to(gd)
    att = att.to(gd)
    wa = wa.to(gd)
    P = px_base.numel()
    S = pos.shape[0]
    acc = torch.zeros(P, dtype=torch.float64, device=pos.device)
    sw = torch.zeros_like(acc)
    swp = torch.zeros_like(acc)
    swp2 = torch.zeros_like(acc)
    if has_dem:
        inv_N = 1.0 / torch.sqrt(1.0 + dzdx * dzdx + dzdy * dzdy)
        sin_floor = torch.sqrt(min_sin2)
    sc = chunk_size(S, P * 24)
    for s0, s1 in chunks(S, sc):
        p = pos[s0:s1, :, None]
        px = px_base[None, :] - p[:, 0]
        py = py_base[None, :] - p[:, 1]
        if use_h_fixed:
            h = torch.full_like(px, h_fixed)
        else:
            h = p[:, 2].expand_as(px)
        if has_dem:
            h = h - z_base[None, :]
        d = torch.sqrt(px * px + py * py + h * h)
        look = torch.asin(torch.clamp(-h / d, min=-1.0))
        psi = torch.atan2(py, px)
        el = look - att[s0:s1, 0, None]
        az = psi - att[s0:s1, 2, None]
        el, az = pitch_rotate(el, az, att[s0:s1, 1, None].expand_as(el))
        gi, gvalid = antenna_gain(g, el, az, g_el0, g_del, g_az0, g_daz)
        sinl = torch.ones_like(d)
        if has_dem:
            if normalization in (1, 2):
                Rg2 = px * px + py * py
                Rg = torch.clamp(torch.sqrt(Rg2), min=1e-6)
                s = px * dzdx[None, :] + py * dzdy[None, :]
                if normalization == 1:
                    sinl = torch.maximum(sin_floor[None, :].expand_as(d),
                                         (Rg2 - s * h) * inv_N[None, :] / (Rg * d))
                else:
                    floor_g = sin_floor[None, :] * d / torch.clamp(h, min=1e-3)
                    den = Rg * torch.maximum(s + h, 1e-3 * d)
                    sinl = torch.maximum(floor_g, (Rg2 - s * h) / den)
            elif normalization == 3:
                sinl = d
        else:
            if normalization == 1:
                sinl = torch.sqrt(torch.maximum(min_sin2[None, :].expand_as(d),
                                                1.0 - (h * h) / (d * d)))
            elif normalization == 2:
                sinl = torch.sqrt(torch.maximum(min_sin2[None, :].expand_as(d),
                                                1.0 - (h * h) / (d * d))) * d / h
            elif normalization == 3:
                sinl = d
        w = wa[s0:s1, None]
        wi = gi * gi * w * w / (d * d * d)
        wi = torch.where(gvalid, wi, torch.zeros_like(wi))
        acc = acc + torch.where(gvalid, wi / sinl, torch.zeros_like(wi)).sum(0, dtype=torch.float64)
        pw = wi > 0
        wi = torch.where(pw, wi, torch.zeros_like(wi)).double()
        psi = psi.double()
        sw = sw + wi.sum(0)
        swp = swp + (wi * psi).sum(0)
        swp2 = swp2 + (wi * psi * psi).sum(0)
    has_w = sw > 0
    swd = torch.where(has_w, sw, torch.ones_like(sw))
    mean = torch.where(has_w, swp / swd, torch.zeros_like(sw))
    m2 = torch.where(has_w, swp2 - swp * swp / swd, torch.zeros_like(sw))
    m2 = torch.clamp(m2, min=0.0)
    return acc, sw, mean, m2


def _tx_power_dem(dem, Nr, Ntheta, device):
    gd = geometry_dtype()
    dem_nr, dem_nt = dem.shape[-2], dem.shape[-1]
    fr = (torch.arange(Nr, device=device, dtype=gd) * (dem_nr / Nr))[:, None]
    ft = (torch.arange(Ntheta, device=device, dtype=gd) * (dem_nt / Ntheta))[None, :]
    zz = dem_sample(dem.reshape(3, dem_nr, dem_nt), fr.expand(Nr, Ntheta),
                    ft.expand(Nr, Ntheta))
    return zz[0].reshape(-1), zz[1].reshape(-1), zz[2].reshape(-1)


def _polar_ground(Nr, Ntheta, r0, dr, theta0, dtheta, altitude, theta_psi,
                  device):
    gd = geometry_dtype()
    r = (r0 + torch.arange(Nr, device=device, dtype=gd) * dr)[:, None]
    tc = (theta0 + torch.arange(Ntheta, device=device, dtype=gd) * dtheta)[None, :]
    t = torch.sin(tc) if theta_psi else tc
    cos2 = 1.0 - t * t
    if altitude > 0:
        r2cos2 = r * r * cos2
        H2 = altitude * altitude
        shadow = r2cos2 < H2
        px = torch.sqrt(torch.clamp(r2cos2 - H2, min=0.0))
    else:
        shadow = torch.zeros_like(r * t, dtype=torch.bool)
        px = r * torch.sqrt(torch.clamp(cos2, min=0.0))
    py = r * t
    px = px.expand(Nr, Ntheta).reshape(-1)
    py = py.expand(Nr, Ntheta).reshape(-1)
    return px, py, shadow.expand(Nr, Ntheta).reshape(-1)


def _tx_power_finish(pixel, m_w, m_s, px, py, azimuth_resolution):
    if azimuth_resolution:
        Rg = torch.sqrt(px * px + py * py)
        var = torch.where(m_w > 0, m_s / torch.where(m_w > 0, m_w,
                          torch.ones_like(m_w)), torch.zeros_like(m_w))
        sigma = torch.sqrt(torch.clamp(var, min=0.0))
        ok = (sigma > 0) & (Rg > 0)
        pixel = torch.where(ok, pixel / torch.where(ok, sigma * Rg,
                            torch.ones_like(Rg)),
                            torch.full_like(pixel, math.inf))
    return torch.sqrt(pixel)


def _polar_tx_power(wa, pos, att, g, nbatch, g_az0, g_el0, g_daz, g_del,
                    nsweeps, dr, r0, theta0, dtheta, Nr, Ntheta,
                    normalization, azimuth_resolution, altitude, dem):
    dev = pos.device
    gd = geometry_dtype()
    px, py, shadow = _polar_ground(Nr, Ntheta, r0, dr, theta0, dtheta,
                                   altitude, False, dev)
    has_dem = not is_undef(dem)
    if has_dem:
        z_pix, dzdx, dzdy = _tx_power_dem(dem, Nr, Ntheta, dev)
    else:
        z_pix = dzdx = dzdy = None
    pos = pos.reshape(nbatch, nsweeps, 3)
    att = att.reshape(nbatch, nsweeps, 3)
    wa = wa.reshape(nbatch, nsweeps)
    out = []
    for b in range(nbatch):
        h_ref = altitude if altitude > 0 else float(pos[b, nsweeps // 2, 2])
        h_ref = torch.full((Nr * Ntheta,), h_ref, dtype=gd, device=dev)
        if has_dem:
            h_ref = torch.clamp(h_ref - z_pix, min=1e-3)
        min_sin2 = 2.0 * dr / h_ref
        pixel, m_w, _, m_s = _tx_power_moments(
            px, py, altitude > 0, altitude, z_pix, dzdx, dzdy, pos[b], att[b],
            g, g_az0, g_el0, g_daz, g_del, wa[b], normalization, min_sin2)
        img = _tx_power_finish(pixel, m_w, m_s, px, py, azimuth_resolution)
        img = torch.where(shadow, torch.zeros_like(img), img)
        out.append(img.reshape(Nr, Ntheta))
    return torch.stack(out).float()


@register("backprojection_polar_2d_tx_power")
def backprojection_polar_2d_tx_power(wa, pos, att, g, nbatch, g_az0, g_el0,
                                     g_daz, g_del, g_naz, g_nel, nsweeps,
                                     r_res, r0, dr, theta0, dtheta, Nr,
                                     Ntheta, normalization,
                                     azimuth_resolution, dem):
    if not is_undef(dem) and (dem.dim() != 3 or dem.shape[0] != 3):
        raise RuntimeError("dem must be a [3, dem_nr, dem_ntheta] tensor")
    return _polar_tx_power(wa, pos, att, g, nbatch, g_az0, g_el0, g_daz,
                           g_del, nsweeps, dr, r0, theta0, dtheta, Nr, Ntheta,
                           normalization, azimuth_resolution, 0.0, dem)


@register("backprojection_polar_2d_tx_power_slant")
def backprojection_polar_2d_tx_power_slant(wa, pos, att, g, nbatch, g_az0,
                                           g_el0, g_daz, g_del, g_naz, g_nel,
                                           nsweeps, r_res, r0, dr, theta0,
                                           dtheta, Nr, Ntheta, normalization,
                                           azimuth_resolution, altitude):
    return _polar_tx_power(wa, pos, att, g, nbatch, g_az0, g_el0, g_daz,
                           g_del, nsweeps, dr, r0, theta0, dtheta, Nr, Ntheta,
                           normalization, azimuth_resolution, altitude, None)


@register("backprojection_cart_2d_tx_power")
def backprojection_cart_2d_tx_power(wa, pos, att, g, nbatch, g_az0, g_el0,
                                    g_daz, g_del, g_naz, g_nel, nsweeps,
                                    r_res, x0, dx, y0, dy, Nx, Ny,
                                    normalization, azimuth_resolution):
    dev = pos.device
    gd = geometry_dtype()
    px, py = _cart_pixels(Nx, Ny, x0, dx, y0, dy, dev)
    pos = pos.reshape(nbatch, nsweeps, 3)
    att = att.reshape(nbatch, nsweeps, 3)
    wa = wa.reshape(nbatch, nsweeps)
    out = []
    for b in range(nbatch):
        h_ref = float(pos[b, nsweeps // 2, 2])
        min_sin2 = torch.full((Nx * Ny,), 2.0 * dx / h_ref, dtype=gd,
                              device=dev)
        pixel, m_w, _, m_s = _tx_power_moments(
            px, py, False, 0.0, None, None, None, pos[b], att[b], g, g_az0,
            g_el0, g_daz, g_del, wa[b], normalization, min_sin2)
        img = _tx_power_finish(pixel, m_w, m_s, px, py, azimuth_resolution)
        out.append(img.reshape(Nx, Ny))
    return torch.stack(out).float()


@register("backprojection_polar_2d_tx_power_accum")
def backprojection_polar_2d_tx_power_accum(wa, pos, att, g, g_az0, g_el0,
                                           g_daz, g_del, g_naz, g_nel,
                                           nsweeps, r0, dr, theta0, dtheta,
                                           Nr, Ntheta, normalization, dr_ref,
                                           h_ref, altitude, theta_psi, dem):
    dev = pos.device
    gd = geometry_dtype()
    has_dem = not is_undef(dem)
    if has_dem:
        if dem.dim() != 3 or dem.shape[0] != 3:
            raise RuntimeError("dem must be a [3, dem_nr, dem_ntheta] tensor")
        if altitude != 0.0:
            raise RuntimeError("dem is not supported with a slant grid")
    px, py, shadow = _polar_ground(Nr, Ntheta, r0, dr, theta0, dtheta,
                                   altitude, theta_psi, dev)
    if has_dem:
        z_pix, dzdx, dzdy = _tx_power_dem(dem, Nr, Ntheta, dev)
        h_pix = torch.clamp(h_ref - z_pix, min=1e-3)
    else:
        z_pix = dzdx = dzdy = None
        h_pix = torch.full((Nr * Ntheta,), float(h_ref), dtype=gd, device=dev)
    min_sin2 = 2.0 * dr_ref / h_pix
    pixel, m_w, m_mean, m_s = _tx_power_moments(
        px, py, altitude > 0, altitude, z_pix, dzdx, dzdy,
        pos.reshape(nsweeps, 3), att.reshape(nsweeps, 3), g, g_az0, g_el0,
        g_daz, g_del, wa.reshape(nsweeps), normalization, min_sin2)
    out = torch.stack([pixel, m_w, m_w * m_mean, m_s])
    out = torch.where(shadow[None, :], torch.zeros_like(out), out)
    return out.reshape(4, Nr, Ntheta).float()


@register("backprojection_cart_2d_tx_power_accum")
def backprojection_cart_2d_tx_power_accum(wa, pos, att, g, g_az0, g_el0,
                                          g_daz, g_del, g_naz, g_nel, nsweeps,
                                          x0, dx, y0, dy, Nx, Ny,
                                          normalization, dx_ref, h_ref):
    dev = pos.device
    gd = geometry_dtype()
    px, py = _cart_pixels(Nx, Ny, x0, dx, y0, dy, dev)
    min_sin2 = torch.full((Nx * Ny,), 2.0 * dx_ref / h_ref, dtype=gd,
                          device=dev)
    pixel, m_w, m_mean, m_s = _tx_power_moments(
        px, py, False, 0.0, None, None, None, pos.reshape(nsweeps, 3),
        att.reshape(nsweeps, 3), g, g_az0, g_el0, g_daz, g_del,
        wa.reshape(nsweeps), normalization, min_sin2)
    out = torch.stack([pixel, m_w, m_w * m_mean, m_s])
    return out.reshape(4, Nx, Ny).float()


def tx_power_merge_sample(acc, nx, ny, xi, xf, yi, yf, inside, state):
    """Interpolate the 4 accumulator channels of one input map and combine
    them into the running totals with Chan's parallel variance formula.
    ``state`` is a list [S, W, P1, M2] that is updated in place."""
    from ._common import bilinear
    S, W, P1, M2 = state
    a = acc.reshape(4, nx * ny).to(xf.dtype)
    xi = torch.where(inside, xi, torch.zeros_like(xi))
    yi = torch.where(inside, yi, torch.zeros_like(yi))
    xf = torch.where(inside, xf, torch.zeros_like(xf))
    yf = torch.where(inside, yf, torch.zeros_like(yf))
    vals = [bilinear(a[c], nx, ny, xi, xf, yi, yf) for c in range(4)]
    s, w, p1, m2 = vals
    use = inside & (w > 0)
    wd = torch.where(use, w, torch.ones_like(w))
    Wd = torch.where(W > 0, W, torch.ones_like(W))
    delta = P1 / Wd - p1 / wd
    comb = m2 + delta * delta * W * w / torch.where(use, W + w,
                                                    torch.ones_like(W))
    dM2 = torch.where(W > 0, comb, m2)
    zero = torch.zeros_like(S)
    state[3] = M2 + torch.where(use, dM2, zero)
    state[0] = S + torch.where(use, s, zero)
    state[1] = W + torch.where(use, w, zero)
    state[2] = P1 + torch.where(use, p1, zero)


@register("cart_tx_power_merge2")
def cart_tx_power_merge2(acc0, acc1, x0_0, dx_0, y0_0, dy_0, Nx_0, Ny_0,
                         x0_1, dx_1, y0_1, dy_1, Nx_1, Ny_1, x1, dx1, y1, dy1,
                         Nx1, Ny1):
    dev = acc0.device
    gd = geometry_dtype()
    X, Y = _cart_pixels(Nx1, Ny1, x1, dx1, y1, dy1, dev)
    z = torch.zeros_like(X)
    state = [z, z, z, z]
    for acc, ix0, idx_, iy0, idy_, nx, ny in (
            (acc0, x0_0, dx_0, y0_0, dy_0, Nx_0, Ny_0),
            (acc1, x0_1, dx_1, y0_1, dy_1, Nx_1, Ny_1)):
        if is_undef(acc):
            continue
        dxi = (X - ix0) / idx_
        dyi = (Y - iy0) / idy_
        inside = (dxi >= 0) & (dxi < nx - 1) & (dyi >= 0) & (dyi < ny - 1)
        xi = torch.trunc(dxi)
        yi = torch.trunc(dyi)
        tx_power_merge_sample(acc, nx, ny, xi.long(), dxi - xi, yi.long(),
                              dyi - yi, inside, state)
    state[3] = torch.clamp(state[3], min=0.0)
    return torch.stack(state).reshape(4, Nx1, Ny1).float()


# ---------------------------------------------------------------------------
# Illumination moments
# ---------------------------------------------------------------------------

@register("compute_illumination")
def compute_illumination(pos, att, g, g_az0, g_el0, g_daz, g_del, r0, dr,
                         theta0, dtheta, nr, ntheta, decimation, dem):
    dev = pos.device
    gd = geometry_dtype()
    dec = decimation if decimation > 0 else 1
    out_nr = (nr + dec - 1) // dec
    out_nt = (ntheta + dec - 1) // dec
    idr = torch.arange(out_nr, device=dev, dtype=gd) * dec
    idt = torch.arange(out_nt, device=dev, dtype=gd) * dec
    r = (r0 + dr * idr)[:, None]
    t = (theta0 + dtheta * idt)[None, :]
    ct = torch.sqrt(torch.clamp(1.0 - t * t, min=0.0))
    x = (r * ct).reshape(-1)
    y = (r * t).expand(out_nr, out_nt).reshape(-1)
    r2 = (r * r).expand(out_nr, out_nt).reshape(-1)
    if not is_undef(dem):
        dem_nr, dem_nt = dem.shape
        fr = (idr[:, None] * (dem_nr / nr)).expand(out_nr, out_nt)
        ft = (idt[None, :] * (dem_nt / ntheta)).expand(out_nr, out_nt)
        z = dem_sample(dem, fr, ft).reshape(-1)
    else:
        z = torch.zeros_like(x)
    nsweeps = pos.shape[0]
    ps = pos.reshape(nsweeps, 3).to(gd)
    has_att = not is_undef(att)
    if has_att:
        at = att.reshape(nsweeps, 3).to(gd)
    P = x.numel()
    w1 = torch.zeros(P, dtype=gd, device=dev)
    w2 = torch.zeros(P, dtype=gd, device=dev)
    sc = chunk_size(nsweeps, P * 16)
    for s0, s1 in chunks(nsweeps, sc):
        p = ps[s0:s1, :, None]
        px, py, pz = p[:, 0], p[:, 1], p[:, 2]
        d2 = (r2 + z * z)[None, :] + px * px + py * py + pz * pz \
            - 2.0 * (x[None, :] * px + y[None, :] * py + z[None, :] * pz)
        d = torch.sqrt(torch.clamp(d2, min=0.0))
        look = torch.asin(torch.clamp((z[None, :] - pz) / d, -1.0, 1.0))
        az = torch.atan2(y[None, :] - py, x[None, :] - px)
        if has_att:
            look = look - at[s0:s1, 0, None]
            az = az - at[s0:s1, 2, None]
        gv, _ = antenna_gain(g, look, az, g_el0, g_del, g_az0, g_daz)
        w1 = w1 + gv.sum(0)
        w2 = w2 + (gv * gv).sum(0)
    return [w1.reshape(out_nr, out_nt).float(),
            w2.reshape(out_nr, out_nt).float()]
