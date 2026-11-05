"""Pure-PyTorch kernels of the polar image interpolation family: polar to
polar interpolation (origin shift and rotation), the FFBP subaperture merges
(Lanczos, Knab, polynomial and antenna weighted polynomial kernels),
polar <-> Cartesian conversion and the transmit power accumulator merge."""
from typing import Optional

import torch
from torch import Tensor

from . import register
from ._common import (
    C0, PI, geometry_dtype, chunk_size, chunks, expj, to_complex64,
    interp_2d, bilinear, dem_sample, vjp, is_undef, ntaps_for,
)
from .backproj import tx_power_merge_sample


def _cplx(img: Tensor) -> Tensor:
    return to_complex64(img) if img.is_complex() else img.to(torch.float32)


def _bilinear_trunc(flat, n0, n1, fi, fj, valid, base=None):
    """Bilinear sample with truncating integer conversion (the kernels'
    ``(int)x`` convention), masked by ``valid``."""
    ii = torch.trunc(fi)
    jj = torch.trunc(fj)
    ii = torch.where(valid, ii, torch.zeros_like(ii))
    jj = torch.where(valid, jj, torch.zeros_like(jj))
    fr_i = torch.where(valid, fi - ii, torch.zeros_like(fi))
    fr_j = torch.where(valid, fj - jj, torch.zeros_like(fj))
    return bilinear(flat, n0, n1, ii.long(), fr_i, jj.long(), fr_j, base=base)


# ---------------------------------------------------------------------------
# Polar to polar interpolation
# ---------------------------------------------------------------------------

def _polar_interp(img, dorigin, nbatch, rotation, fc, r0, dr0, theta0,
                  dtheta0, Nr0, Ntheta0, r1, dr1, theta1, dtheta1, Nr1,
                  Ntheta1, z1, alias_fmod, method, order):
    gd = geometry_dtype()
    dev = img.device
    im = _cplx(img).reshape(nbatch, Nr0 * Ntheta0)
    do = dorigin.reshape(nbatch, 3).to(gd)
    ref_phase = 4.0 * fc / C0
    idr = torch.arange(Nr1, device=dev, dtype=gd)[:, None]
    d = r1 + dr1 * idr
    t = theta1 + dtheta1 * torch.arange(Ntheta1, device=dev, dtype=gd)[None, :]
    if rotation != 0.0:
        t = torch.sin(torch.asin(t) - rotation)
    tin = (t >= -1.0) & (t <= 1.0)
    tc = torch.where(tin, t, torch.zeros_like(t))
    sint = tc
    cost = torch.sqrt(1.0 - tc * tc)
    out = []
    for b in range(nbatch):
        o0, o1, o2 = do[b, 0], do[b, 1], do[b, 2]
        rp = torch.sqrt(d * d + o0 * o0 + o1 * o1 + 2 * d * (o0 * cost + o1 * sint))
        arg = (d * sint + o1) / (d * cost + o0)
        tp = arg / torch.sqrt(1.0 + arg * arg)
        dri = (rp - r0) / dr0
        dti = (tp - theta0) / dtheta0
        if method == "linear":
            ii = torch.trunc(dri)
            jj = torch.trunc(dti)
            valid = tin & (ii >= 0) & (ii < Nr0 - 1) & (jj >= 0) & (jj < Ntheta0 - 1)
            v = _bilinear_trunc(im[b], Nr0, Ntheta0, dri, dti, valid)
        else:
            valid = tin & (dri >= 0) & (dri < Nr0 - 1) & (dti >= 0) & (dti < Ntheta0 - 1)
            v = _interp_chunked(im[b], Nr0, Ntheta0,
                                torch.where(valid, dri, torch.zeros_like(dri)),
                                torch.where(valid, dti, torch.zeros_like(dti)),
                                method, order)
        z0 = z1 + o2
        dz = torch.sqrt(z1 * z1 + d * d)
        rpz = torch.sqrt(z0 * z0 + rp * rp)
        ph = PI * ref_phase * (rpz - dz) - alias_fmod * (dri - idr)
        val = v * expj(ph).to(torch.complex64)
        out.append(torch.where(valid, val, torch.zeros_like(val)))
    return torch.stack(out)


def _interp_chunked(flat, n0, n1, x, y, method, order, v=0.0, coefs=None,
                    fmod=None):
    """interp_2d over a grid of positions, chunked to bound memory."""
    shape = x.shape
    xf = x.reshape(-1)
    yf = y.reshape(-1)
    ff = None if fmod is None else torch.broadcast_to(fmod, shape).reshape(-1)
    T = ntaps_for(0.5 * order)
    n = xf.numel()
    cs = chunk_size(n, T * T * 2)
    parts = []
    for s, e in chunks(n, cs):
        parts.append(interp_2d(flat, n0, n1, xf[s:e], yf[s:e], method, order,
                               v=v, coefs=coefs,
                               fmod_x=None if ff is None else ff[s:e]))
    if not parts:
        return torch.zeros(shape, dtype=flat.dtype, device=flat.device)
    return torch.cat(parts).reshape(shape)


@register("polar_interp_linear")
def polar_interp_linear(img, dorigin, nbatch, rotation, fc, r0, dr0, theta0,
                        dtheta0, Nr0, Ntheta0, r1, dr1, theta1, dtheta1, Nr1,
                        Ntheta1, z1, alias_fmod):
    return _polar_interp(img, dorigin, nbatch, rotation, fc, r0, dr0, theta0,
                         dtheta0, Nr0, Ntheta0, r1, dr1, theta1, dtheta1, Nr1,
                         Ntheta1, z1, alias_fmod, "linear", 2)


@register("polar_interp_linear_grad")
def polar_interp_linear_grad(grad, img, dorigin, nbatch, rotation, fc, r0,
                             dr0, theta0, dtheta0, Nr0, Ntheta0, r1, dr1,
                             theta1, dtheta1, Nr1, Ntheta1, z1, alias_fmod):
    def fwd(img_, dorigin_):
        return _polar_interp(img_, dorigin_, nbatch, rotation, fc, r0, dr0,
                             theta0, dtheta0, Nr0, Ntheta0, r1, dr1, theta1,
                             dtheta1, Nr1, Ntheta1, z1, alias_fmod, "linear", 2)

    g = grad.reshape(nbatch, Nr1, Ntheta1).to(torch.complex64)
    return vjp(fwd, [img, dorigin], [img.requires_grad, dorigin.requires_grad], g)


@register("polar_interp_lanczos")
def polar_interp_lanczos(img, dorigin, nbatch, rotation, fc, r0, dr0, theta0,
                         dtheta0, Nr0, Ntheta0, r1, dr1, theta1, dtheta1, Nr1,
                         Ntheta1, z1, order, alias_fmod):
    return _polar_interp(img, dorigin, nbatch, rotation, fc, r0, dr0, theta0,
                         dtheta0, Nr0, Ntheta0, r1, dr1, theta1, dtheta1, Nr1,
                         Ntheta1, z1, alias_fmod, "lanczos", order)


# ---------------------------------------------------------------------------
# FFBP merges
# ---------------------------------------------------------------------------

def _range_fmod(m2, rp, tp, rpz, ref_phase, dr_c):
    """Local range spectrum shift of a subaperture image at a lookup, see
    the ffbp merge kernels. Radians per range sample."""
    rpz2 = rpz * rpz
    fpp = -(rp / (rpz2 * rpz)) * (1.0 + tp * tp * (2.0 - 3.0 * rp * rp / rpz2))
    return PI * ref_phase * 0.5 * m2 * fpp * dr_c


def _wmap_sample(wmap, w_r0, w_dr, w_t0, w_dt, w_nr, w_nt, rp, tp):
    w_ri = (rp - w_r0) / w_dr
    w_ti = (tp - w_t0) / w_dt
    ri = torch.clamp(torch.clamp(torch.trunc(w_ri), max=w_nr - 2), min=0)
    ti = torch.clamp(torch.clamp(torch.trunc(w_ti), max=w_nt - 2), min=0)
    rf = torch.clamp(w_ri - ri, 0.0, 1.0)
    tf = torch.clamp(w_ti - ti, 0.0, 1.0)
    return bilinear(wmap.reshape(-1).to(rp.dtype), w_nr, w_nt, ri.long(), rf,
                    ti.long(), tf)


def _merge2(imgs, dorigin, fc, r0, dr0, theta0, dtheta0, Nr0, Ntheta0, r1,
            dr1, theta1, dtheta1, Nr1, Ntheta1, z1, method, order, alias,
            alias_fmod, dem, *, knab_v=0.0, coefs=None, m2=(0.0, 0.0),
            weights=None, output_weight_map=False, decimation=1,
            out_dtype=None):
    gd = geometry_dtype()
    dev = imgs[0].device
    in_dtype = imgs[0].dtype
    r0 = [float(v) for v in r0.reshape(-1).tolist()]
    dr0 = [float(v) for v in dr0.reshape(-1).tolist()]
    theta0 = [float(v) for v in theta0.reshape(-1).tolist()]
    dtheta0 = [float(v) for v in dtheta0.reshape(-1).tolist()]
    Nr0 = [int(v) for v in Nr0.reshape(-1).tolist()]
    Ntheta0 = [int(v) for v in Ntheta0.reshape(-1).tolist()]
    do = dorigin.reshape(-1, 3).to(gd)
    ref_phase = 4.0 * fc / C0
    P = Nr1 * Ntheta1
    idr_all = torch.arange(Nr1, device=dev, dtype=gd)[:, None].expand(Nr1, Ntheta1).reshape(-1)
    idt_all = torch.arange(Ntheta1, device=dev, dtype=gd)[None, :].expand(Nr1, Ntheta1).reshape(-1)
    has_dem = not is_undef(dem)
    flats = []
    for k, im in enumerate(imgs):
        if is_undef(im):
            flats.append(None)
        else:
            flats.append(to_complex64(im).reshape(-1))
    T = 2 if method == "linear" else ntaps_for(0.5 * order)
    pc = chunk_size(P, 2 * T * T + 16)
    out = torch.empty(P, dtype=torch.complex64, device=dev)
    W1o = torch.zeros(P, dtype=gd, device=dev) if weights is not None else None
    W2o = torch.zeros(P, dtype=gd, device=dev) if weights is not None else None
    for p0, p1 in chunks(P, pc):
        idr = idr_all[p0:p1]
        idt = idt_all[p0:p1]
        d = r1 + dr1 * idr
        t = theta1 + dtheta1 * idt
        sint = t
        cost = torch.sqrt(torch.clamp(1.0 - t * t, min=0.0))
        if has_dem:
            dem_nr, dem_nt = dem.shape
            zp = dem_sample(dem, idr * (dem_nr / Nr1), idt * (dem_nt / Ntheta1))
        else:
            zp = torch.zeros_like(d)
        z1p = z1 - zp
        dz = torch.sqrt(z1p * z1p + d * d)
        A = torch.zeros(p1 - p0, dtype=torch.complex64, device=dev)
        W1 = torch.zeros_like(d)
        W2 = torch.zeros_like(d)
        for k in range(2):
            flat = flats[k]
            if flat is None:
                continue
            o0, o1, o2 = do[k, 0], do[k, 1], do[k, 2]
            rp2 = d * d + o0 * o0 + o1 * o1 + 2 * d * (o0 * cost + o1 * sint)
            rp = torch.sqrt(torch.clamp(rp2, min=0.0))
            tp = (d * sint + o1) / rp
            dri = (rp - r0[k]) / dr0[k]
            dti = (tp - theta0[k]) / dtheta0[k]
            valid = (dri >= 0) & (dri < Nr0[k] - 1) & (dti >= 0) & (dti < Ntheta0[k] - 1)
            z0 = z1 + o2 - zp
            rpz = torch.sqrt(z0 * z0 + rp * rp)
            xs = torch.where(valid, dri, torch.zeros_like(dri))
            ys = torch.where(valid, dti, torch.zeros_like(dti))
            nu = None
            if method in ("knab", "poly") and m2[k] != 0.0:
                nu = _range_fmod(m2[k], rp, tp, rpz, ref_phase, dr0[k])
                nu = torch.where(valid, nu, torch.zeros_like(nu))
            v = interp_2d(flat, Nr0[k], Ntheta0[k], xs, ys, method, order,
                          v=knab_v, coefs=coefs, fmod_x=nu)
            ph = PI * ref_phase * (rpz - dz) - alias_fmod * (dri - idr)
            val = v.to(torch.complex64) * expj(ph).to(torch.complex64)
            if weights is not None:
                wm = weights[k]
                if wm is None:
                    A = A + torch.where(valid, val, torch.zeros_like(val))
                else:
                    w1m, w2m, w_r0, w_dr, w_t0, w_dt, w_nr, w_nt = wm
                    w1 = _wmap_sample(w1m, w_r0, w_dr, w_t0, w_dt, w_nr, w_nt, rp, tp)
                    w2 = _wmap_sample(w2m, w_r0, w_dr, w_t0, w_dt, w_nr, w_nt, rp, tp)
                    use = valid & (w1 > 0) & (w2 > 0)
                    A = A + torch.where(use, val, torch.zeros_like(val))
                    W1 = W1 + torch.where(use, w1, torch.zeros_like(w1))
                    W2 = W2 + torch.where(use, w2, torch.zeros_like(w2))
            else:
                A = A + torch.where(valid, val, torch.zeros_like(val))
        if weights is not None and not output_weight_map:
            ok = W2 > 0
            scale = torch.where(ok, W1 / torch.where(ok, W2, torch.ones_like(W2)),
                                torch.ones_like(W2))
            A = A * scale.to(torch.float32)
        if alias:
            af = 0.0 if alias == 2 else alias_fmod
            rph = 0.0 if alias == 3 else ref_phase
            A = A * expj(PI * rph * dz - af * idr).to(torch.complex64)
        out[p0:p1] = A
        if weights is not None:
            W1o[p0:p1] = W1
            W2o[p0:p1] = W2
    st = out_dtype if out_dtype is not None else in_dtype
    out = out.reshape(Nr1, Ntheta1)
    if st != torch.complex64:
        out = out.to(st)
    if weights is None:
        return out
    if output_weight_map:
        dec = decimation if decimation > 0 else 1
        w1o = W1o.reshape(Nr1, Ntheta1)[::dec, ::dec].float().contiguous()
        w2o = W2o.reshape(Nr1, Ntheta1)[::dec, ::dec].float().contiguous()
        return [out, w1o, w2o]
    return [out, None, None]


def _check_merge_inputs(img0, img1, Nr0, Ntheta0):
    if not img0.is_complex():
        raise RuntimeError("ffbp merge: img must be complex")


@register("ffbp_merge2_lanczos")
def ffbp_merge2_lanczos(img0, img1, dorigin, fc, r0, dr0, theta0, dtheta0,
                        Nr0, Ntheta0, r1, dr1, theta1, dtheta1, Nr1, Ntheta1,
                        z1, order, alias, alias_fmod, dem):
    _check_merge_inputs(img0, img1, Nr0, Ntheta0)
    return _merge2([img0, img1], dorigin, fc, r0, dr0, theta0, dtheta0, Nr0,
                   Ntheta0, r1, dr1, theta1, dtheta1, Nr1, Ntheta1, z1,
                   "lanczos", order, alias, alias_fmod, dem)


@register("ffbp_merge2_knab")
def ffbp_merge2_knab(img0, img1, dorigin, fc, r0, dr0, theta0, dtheta0, Nr0,
                     Ntheta0, r1, dr1, theta1, dtheta1, Nr1, Ntheta1, z1,
                     order, oversample, alias, alias_fmod, dem, m2_0=0.0,
                     m2_1=0.0):
    _check_merge_inputs(img0, img1, Nr0, Ntheta0)
    v = 1.0 - 1.0 / oversample
    return _merge2([img0, img1], dorigin, fc, r0, dr0, theta0, dtheta0, Nr0,
                   Ntheta0, r1, dr1, theta1, dtheta1, Nr1, Ntheta1, z1,
                   "knab", order, alias, alias_fmod, dem, knab_v=v,
                   m2=(m2_0, m2_1))


def _poly_coefs(poly_coefs, order):
    if order < 2 or order > 8:
        raise RuntimeError(f"ffbp_merge2_poly requires order in [2, 8], got {order}")
    coefs = [float(c) for c in poly_coefs.reshape(-1).tolist()]
    if len(coefs) > 15:
        raise RuntimeError(
            f"ffbp_merge2_poly: poly_coefs has {len(coefs)} coefficients, max is 15")
    if not 4 <= len(coefs) <= 14:
        raise RuntimeError(
            f"ffbp_merge2_poly: n_coefs must be 4-14, got {len(coefs)}")
    return coefs


def _out_dtype(img0, img1, out_dtype):
    if img0.dtype not in (torch.complex64, torch.complex32):
        raise RuntimeError("ffbp_merge2_poly: img dtype must be complex64 or complex32")
    if img1.numel() > 0 and img1.dtype != img0.dtype:
        raise RuntimeError("ffbp_merge2_poly: img0 and img1 dtypes must match")
    st = out_dtype if out_dtype is not None else img0.dtype
    if st not in (torch.complex64, torch.complex32):
        raise RuntimeError("ffbp_merge2_poly: out_dtype must be complex64 or complex32")
    return st


@register("ffbp_merge2_poly")
def ffbp_merge2_poly(img0, img1, dorigin, fc, r0, dr0, theta0, dtheta0, Nr0,
                     Ntheta0, r1, dr1, theta1, dtheta1, Nr1, Ntheta1, z1,
                     order, poly_coefs, alias, alias_fmod, dem, m2_0=0.0,
                     m2_1=0.0, out_dtype=None):
    st = _out_dtype(img0, img1, out_dtype)
    coefs = _poly_coefs(poly_coefs, order)
    return _merge2([img0, img1], dorigin, fc, r0, dr0, theta0, dtheta0, Nr0,
                   Ntheta0, r1, dr1, theta1, dtheta1, Nr1, Ntheta1, z1,
                   "poly", order, alias, alias_fmod, dem, coefs=coefs,
                   m2=(m2_0, m2_1), out_dtype=st)


@register("ffbp_merge2_poly_weighted")
def ffbp_merge2_poly_weighted(img0, img1, dorigin, fc, r0, dr0, theta0,
                              dtheta0, Nr0, Ntheta0, r1, dr1, theta1, dtheta1,
                              Nr1, Ntheta1, z1, order, poly_coefs, alias,
                              alias_fmod, w1_map0, w2_map0, w_r0_0, w_dr0,
                              w_theta0_0, w_dtheta0, w_nr0, w_ntheta0,
                              w1_map1, w2_map1, w_r0_1, w_dr1, w_theta0_1,
                              w_dtheta1, w_nr1, w_ntheta1, output_weight_map,
                              output_weight_decimation, dem, m2_0=0.0,
                              m2_1=0.0, out_dtype=None):
    st = _out_dtype(img0, img1, out_dtype)
    coefs = _poly_coefs(poly_coefs, order)

    def wm(w1m, w2m, w_r0, w_dr, w_t0, w_dt, w_nr, w_nt):
        if is_undef(w1m) or is_undef(w2m):
            return None
        return (w1m, w2m, w_r0, w_dr, w_t0, w_dt, w_nr, w_nt)

    weights = [wm(w1_map0, w2_map0, w_r0_0, w_dr0, w_theta0_0, w_dtheta0,
                  w_nr0, w_ntheta0),
               wm(w1_map1, w2_map1, w_r0_1, w_dr1, w_theta0_1, w_dtheta1,
                  w_nr1, w_ntheta1)]
    return _merge2([img0, img1], dorigin, fc, r0, dr0, theta0, dtheta0, Nr0,
                   Ntheta0, r1, dr1, theta1, dtheta1, Nr1, Ntheta1, z1,
                   "poly", order, alias, alias_fmod, dem, coefs=coefs,
                   m2=(m2_0, m2_1), weights=weights,
                   output_weight_map=bool(output_weight_map),
                   decimation=output_weight_decimation, out_dtype=st)


# ---------------------------------------------------------------------------
# Polar <-> Cartesian
# ---------------------------------------------------------------------------

def _polar_to_cart(img, origin, nbatch, rotation, fc, r0, dr, theta0, dtheta,
                   Nr, Ntheta, x0, y0, dx, dy, Nx, Ny, alias_fmod, method,
                   order):
    gd = geometry_dtype()
    dev = img.device
    is_c = img.is_complex()
    im = _cplx(img).reshape(nbatch, Nr * Ntheta)
    og = origin.reshape(nbatch, 3).to(gd)
    ref_phase = 4.0 * fc / C0
    x = x0 + dx * torch.arange(Nx, device=dev, dtype=gd)[:, None]
    y = y0 + dy * torch.arange(Ny, device=dev, dtype=gd)[None, :]
    rs = float(torch.sin(torch.tensor(rotation, dtype=torch.float64)))
    rc = float(torch.cos(torch.tensor(rotation, dtype=torch.float64)))
    out = []
    for b in range(nbatch):
        o0, o1, o2 = og[b, 0], og[b, 1], og[b, 2]
        d = torch.sqrt((x - o0) ** 2 + (y - o1) ** 2)
        dz = torch.sqrt(d * d + o2 * o2)
        t = (y - o1) / d
        tc = (x - o0) / d
        cosa = t * rs + tc * rc
        if rotation != 0.0:
            t = rc * t - rs * tc
        dri = (d - r0) / dr
        dti = (t - theta0) / dtheta
        valid = ((cosa >= 0) & (dri >= 0) & (torch.trunc(dri) < Nr - 1)
                 & (dti >= 0) & (torch.trunc(dti) < Ntheta - 1))
        if method == "linear":
            v = _bilinear_trunc(im[b], Nr, Ntheta, dri, dti, valid)
        else:
            v = _interp_chunked(im[b], Nr, Ntheta,
                                torch.where(valid, dri, torch.zeros_like(dri)),
                                torch.where(valid, dti, torch.zeros_like(dti)),
                                method, order)
        if is_c:
            v = v.to(torch.complex64) * expj(PI * ref_phase * dz - alias_fmod * dri).to(torch.complex64)
        out.append(torch.where(valid, v, torch.zeros_like(v)))
    res = torch.stack(out)
    return res if is_c else res.to(img.dtype)


@register("polar_to_cart_linear")
def polar_to_cart_linear(img, origin, nbatch, rotation, fc, r0, dr, theta0,
                         dtheta, Nr, Ntheta, x0, y0, dx, dy, Nx, Ny,
                         alias_fmod):
    return _polar_to_cart(img, origin, nbatch, rotation, fc, r0, dr, theta0,
                          dtheta, Nr, Ntheta, x0, y0, dx, dy, Nx, Ny,
                          alias_fmod, "linear", 2)


@register("polar_to_cart_linear_grad")
def polar_to_cart_linear_grad(grad, img, origin, nbatch, rotation, fc, r0, dr,
                              theta0, dtheta, Nr, Ntheta, x0, y0, dx, dy, Nx,
                              Ny, alias_fmod):
    def fwd(img_, origin_):
        return _polar_to_cart(img_, origin_, nbatch, rotation, fc, r0, dr,
                              theta0, dtheta, Nr, Ntheta, x0, y0, dx, dy, Nx,
                              Ny, alias_fmod, "linear", 2)

    g = grad.reshape(nbatch, Nx, Ny)
    return vjp(fwd, [img, origin], [img.requires_grad, origin.requires_grad], g)


@register("polar_to_cart_lanczos")
def polar_to_cart_lanczos(img, origin, nbatch, rotation, fc, r0, dr, theta0,
                          dtheta, Nr, Ntheta, x0, y0, dx, dy, Nx, Ny,
                          alias_fmod, order):
    return _polar_to_cart(img, origin, nbatch, rotation, fc, r0, dr, theta0,
                          dtheta, Nr, Ntheta, x0, y0, dx, dy, Nx, Ny,
                          alias_fmod, "lanczos", order)


def _cart_to_polar(img, origin, nbatch, rotation, fc, x0, y0, dx, dy, Nx, Ny,
                   r0, dr, theta0, dtheta, Nr, Ntheta, alias_fmod, method,
                   order):
    gd = geometry_dtype()
    dev = img.device
    is_c = img.is_complex()
    im = _cplx(img).reshape(nbatch, Nx, Ny)
    og = origin.reshape(nbatch, 3).to(gd)
    ref_phase = 4.0 * fc / C0
    d = r0 + dr * torch.arange(Nr, device=dev, dtype=gd)[:, None]
    t = theta0 + dtheta * torch.arange(Ntheta, device=dev, dtype=gd)[None, :]
    tin = (t >= -1.0) & (t <= 1.0)
    cost = torch.sqrt(torch.clamp(1.0 - t * t, min=0.0))
    rs = float(torch.sin(torch.tensor(rotation, dtype=torch.float64)))
    rc = float(torch.cos(torch.tensor(rotation, dtype=torch.float64)))
    sina = rc * t + rs * cost
    cosa = rc * cost - rs * t
    xk = x0 + dx * torch.arange(Nx, device=dev, dtype=gd)[:, None]
    yk = y0 + dy * torch.arange(Ny, device=dev, dtype=gd)[None, :]
    out = []
    for b in range(nbatch):
        o0, o1, o2 = og[b, 0], og[b, 1], og[b, 2]
        x = o0 + d * cosa
        y = o1 + d * sina
        xi = (x - x0) / dx
        yi = (y - y0) / dy
        valid = (tin & (xi >= 0) & (torch.trunc(xi) < Nx - 1)
                 & (yi >= 0) & (torch.trunc(yi) < Ny - 1))
        src = im[b]
        if is_c:
            # The Cartesian samples are re-referenced to the polar carrier
            # before interpolation (the carrier depends on the sample
            # position only).
            dk = torch.sqrt((xk - o0) ** 2 + (yk - o1) ** 2)
            dzk = torch.sqrt(dk * dk + o2 * o2)
            ph = -(PI * ref_phase * dzk - alias_fmod * (dk - r0) / dr)
            src = src * expj(ph).to(torch.complex64)
        flat = src.reshape(-1)
        if method == "linear":
            v = _bilinear_trunc(flat, Nx, Ny, xi, yi, valid)
        else:
            v = _interp_chunked(flat, Nx, Ny,
                                torch.where(valid, xi, torch.zeros_like(xi)),
                                torch.where(valid, yi, torch.zeros_like(yi)),
                                method, order)
        out.append(torch.where(valid, v, torch.zeros_like(v)))
    res = torch.stack(out)
    return res if is_c else res.to(img.dtype)


@register("cart_to_polar_linear")
def cart_to_polar_linear(img, origin, nbatch, rotation, fc, x0, y0, dx, dy,
                         Nx, Ny, r0, dr, theta0, dtheta, Nr, Ntheta,
                         alias_fmod):
    return _cart_to_polar(img, origin, nbatch, rotation, fc, x0, y0, dx, dy,
                          Nx, Ny, r0, dr, theta0, dtheta, Nr, Ntheta,
                          alias_fmod, "linear", 2)


@register("cart_to_polar_linear_grad")
def cart_to_polar_linear_grad(grad, img, origin, nbatch, rotation, fc, x0, y0,
                              dx, dy, Nx, Ny, r0, dr, theta0, dtheta, Nr,
                              Ntheta, alias_fmod):
    def fwd(img_, origin_):
        return _cart_to_polar(img_, origin_, nbatch, rotation, fc, x0, y0, dx,
                              dy, Nx, Ny, r0, dr, theta0, dtheta, Nr, Ntheta,
                              alias_fmod, "linear", 2)

    g = grad.reshape(nbatch, Nr, Ntheta)
    return vjp(fwd, [img, origin], [img.requires_grad, origin.requires_grad], g)


@register("cart_to_polar_lanczos")
def cart_to_polar_lanczos(img, origin, nbatch, rotation, fc, x0, y0, dx, dy,
                          Nx, Ny, r0, dr, theta0, dtheta, Nr, Ntheta,
                          alias_fmod, order):
    return _cart_to_polar(img, origin, nbatch, rotation, fc, x0, y0, dx, dy,
                          Nx, Ny, r0, dr, theta0, dtheta, Nr, Ntheta,
                          alias_fmod, "lanczos", order)


# ---------------------------------------------------------------------------
# Transmit power accumulator merge
# ---------------------------------------------------------------------------

@register("ffbp_tx_power_merge2")
def ffbp_tx_power_merge2(acc0, acc1, dorigin, r0, dr0, theta0, dtheta0, Nr0,
                         Ntheta0, r1, dr1, theta1, dtheta1, Nr1, Ntheta1,
                         altitude, in_psi, out_psi):
    gd = geometry_dtype()
    dev = acc0.device
    r0 = r0.reshape(-1).tolist()
    dr0 = dr0.reshape(-1).tolist()
    theta0 = theta0.reshape(-1).tolist()
    dtheta0 = dtheta0.reshape(-1).tolist()
    Nr0 = [int(v) for v in Nr0.reshape(-1).tolist()]
    Ntheta0 = [int(v) for v in Ntheta0.reshape(-1).tolist()]
    do = dorigin.reshape(-1, 3).to(gd)
    r = (r1 + dr1 * torch.arange(Nr1, device=dev, dtype=gd))[:, None]
    tc = (theta1 + dtheta1 * torch.arange(Ntheta1, device=dev, dtype=gd))[None, :]
    t = torch.sin(tc) if out_psi else tc
    r = r.expand(Nr1, Ntheta1).reshape(-1)
    t = t.expand(Nr1, Ntheta1).reshape(-1)
    ok = (t >= -1.0) & (t <= 1.0)
    tcl = torch.clamp(t, -1.0, 1.0)
    H2 = altitude * altitude
    if altitude > 0:
        q = r * r * (1.0 - tcl * tcl) - H2
        ok = ok & (q >= 0)
        gx = torch.sqrt(torch.clamp(q, min=0.0))
    else:
        gx = r * torch.sqrt(1.0 - tcl * tcl)
    gy = r * tcl
    z = torch.zeros_like(r)
    state = [z, z, z, z]
    tlim = 1.5707964 if in_psi else 1.0
    for k, acc in enumerate((acc0, acc1)):
        if is_undef(acc):
            continue
        ddx = gx + do[k, 0]
        ddy = gy + do[k, 1]
        front = ddx > 0
        ri = torch.sqrt(ddx * ddx + ddy * ddy + (H2 if altitude > 0 else 0.0))
        ti = ddy / ri
        tq = torch.asin(torch.clamp(ti, -1.0, 1.0)) if in_psi else ti
        dri = (ri - r0[k]) / dr0[k]
        dti = (tq - theta0[k]) / dtheta0[k]
        nri, nti = Nr0[k], Ntheta0[k]
        t_hi_g = theta0[k] + dtheta0[k] * nti
        if t_hi_g >= tlim - 1e-6:
            dti = torch.where(dti > nti - 1.001, torch.full_like(dti, nti - 1.001), dti)
        if theta0[k] <= -tlim + 1e-6:
            dti = torch.where(dti < 1e-3, torch.full_like(dti, 1e-3), dti)
        inside = ok & front & (dri >= 0) & (dri < nri - 1) & (dti >= 0) & (dti < nti - 1)
        ri_i = torch.trunc(dri)
        ti_i = torch.trunc(dti)
        tx_power_merge_sample(acc, nri, nti, ri_i.long(), dri - ri_i,
                              ti_i.long(), dti - ti_i, inside, state)
    state[3] = torch.clamp(state[3], min=0.0)
    out = torch.stack(state)
    out = torch.where(ok[None, :], out, torch.zeros_like(out))
    return out.reshape(4, Nr1, Ntheta1).float()
