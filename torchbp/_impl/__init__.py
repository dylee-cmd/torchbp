"""Pure-PyTorch implementation of the ``torchbp`` operator library.

The operators keep the schemas of the former compiled C++/CUDA extension, so
``torch.ops.torchbp.<name>`` works unchanged and the fake (meta) and autograd
registrations in :mod:`torchbp.ops` attach to them as before. The kernels
are written with standard PyTorch tensor operations only and run on any
device PyTorch supports (CPU, CUDA, MPS, ...). Operators that returned
possibly undefined tensors in a list (gradients of inputs that do not
require grad, optional weight maps) now declare ``Tensor?[]`` and return
``None`` for those entries.
"""
import torch

from ._common import set_geometry_dtype, set_chunk_elements

__all__ = ["set_geometry_dtype", "set_chunk_elements", "SCHEMAS"]

SCHEMAS = [
    "backprojection_polar_2d(Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod, bool normalize, Tensor dem) -> Tensor",
    "backprojection_polar_2d_grad(Tensor grad, Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod, bool normalize, Tensor dem) -> Tensor?[]",
    "backprojection_polar_2d_lanczos(Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, int order, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod, bool normalize, Tensor dem) -> Tensor",
    "backprojection_polar_2d_knab(Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, int order, float oversample, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod, bool normalize, Tensor dem) -> Tensor",
    "backprojection_polar_2d_illum(Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod, bool normalize, Tensor dem) -> Tensor?[]",
    "backprojection_polar_2d_knab_illum(Tensor data, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float d0, int dealias, float z0, int order, float oversample, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, float data_fmod, float alias_fmod, bool normalize, Tensor dem) -> Tensor?[]",
    "backprojection_cart_2d(Tensor data, Tensor pos, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float x0, float dx, float y0, float dy, int Nx, int Ny, float beamwidth, float d0, float data_fmod) -> Tensor",
    "backprojection_cart_2d_grad(Tensor grad, Tensor data, Tensor pos, int nbatch, int sweep_samples, int nsweeps, float fc, float r_res, float x0, float dx, float y0, float dy, int Nx, int Ny, float beamwidth, float d0, float data_fmod) -> Tensor?[]",
    "polar_range_dealias(Tensor img, Tensor dem, int nbatch, int Nr, int Ntheta, float fc, float r0, float dr, float theta0, float dtheta, float ox, float oy, float oz, float alias_fmod) -> Tensor",
    "gpga_backprojection_2d(Tensor target_pos, Tensor data, Tensor pos, int sweep_samples, int nsweeps, float fc, float r_res, int Ntarget, float d0, float data_fmod) -> Tensor",
    "gpga_backprojection_2d_lanczos(Tensor target_pos, Tensor data, Tensor pos, int sweep_samples, int nsweeps, float fc, float r_res, int Ntarget, float d0, int order, float data_fmod) -> Tensor",
    "blocksvd_alpha(Tensor img, Tensor data, Tensor pos, Tensor blocks, int sweep_samples, int nsweeps, int nblocks, int Ntheta, float fc, float r_res, float r0, float dr, float theta0, float dtheta, float d0, float data_fmod, Tensor dem) -> Tensor",
    "cfar_2d(Tensor img, int nbatch, int N0, int N1, int Navg0, int Navg1, int Nguard0, int Nguard1, float threshold, int peaks_only) -> Tensor",
    "polar_interp_linear(Tensor img, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr0, float theta0, float dtheta0, int Nr0, int Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, float alias_fmod) -> Tensor",
    "polar_interp_linear_grad(Tensor grad, Tensor img, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr0, float theta0, float dtheta0, int Nr0, int Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, float alias_fmod) -> Tensor?[]",
    "polar_interp_lanczos(Tensor img, Tensor dorigin, int nbatch, float rotation, float fc, float r0, float dr0, float theta0, float dtheta0, int Nr0, int Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, float alias_fmod) -> Tensor",
    "ffbp_merge2_lanczos(Tensor img0, Tensor img1, Tensor dorigin, float fc, Tensor r0, Tensor dr0, Tensor theta0, Tensor dtheta0, Tensor Nr0, Tensor Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, int alias, float alias_fmod, Tensor dem) -> Tensor",
    "ffbp_merge2_knab(Tensor img0, Tensor img1, Tensor dorigin, float fc, Tensor r0, Tensor dr0, Tensor theta0, Tensor dtheta0, Tensor Nr0, Tensor Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, float oversample, int alias, float alias_fmod, Tensor dem, float m2_0=0., float m2_1=0.) -> Tensor",
    "ffbp_merge2_poly(Tensor img0, Tensor img1, Tensor dorigin, float fc, Tensor r0, Tensor dr0, Tensor theta0, Tensor dtheta0, Tensor Nr0, Tensor Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, Tensor poly_coefs, int alias, float alias_fmod, Tensor dem, float m2_0=0., float m2_1=0., ScalarType? out_dtype=None) -> Tensor",
    "ffbp_merge2_poly_weighted(Tensor img0, Tensor img1, Tensor dorigin, float fc, Tensor r0, Tensor dr0, Tensor theta0, Tensor dtheta0, Tensor Nr0, Tensor Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float z1, int order, Tensor poly_coefs, int alias, float alias_fmod, Tensor w1_map0, Tensor w2_map0, float w_r0_0, float w_dr0, float w_theta0_0, float w_dtheta0, int w_nr0, int w_ntheta0, Tensor w1_map1, Tensor w2_map1, float w_r0_1, float w_dr1, float w_theta0_1, float w_dtheta1, int w_nr1, int w_ntheta1, int output_weight_map, int output_weight_decimation, Tensor dem, float m2_0=0., float m2_1=0., ScalarType? out_dtype=None) -> Tensor?[]",
    "cfbp_merge2(Tensor img0, Tensor img1, Tensor w0, Tensor idx0, Tensor w1, Tensor idx1, int nbatch, int Nx, int Ny0, int Ny1, int Nyout, int order0, int order1, float dx, float dy, float ox0, float oy0, float z0, float ox1, float oy1, float z1, float oxp, float oyp, float zp, float ref_phase) -> Tensor",
    "polar_to_cart_linear(Tensor img, Tensor origin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny, float alias_fmod) -> Tensor",
    "polar_to_cart_linear_grad(Tensor grad, Tensor img, Tensor origin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny, float alias_fmod) -> Tensor?[]",
    "polar_to_cart_lanczos(Tensor img, Tensor origin, int nbatch, float rotation, float fc, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float x0, float y0, float dx, float dy, int Nx, int Ny, float alias_fmod, int order) -> Tensor",
    "cart_to_polar_linear(Tensor img, Tensor origin, int nbatch, float rotation, float fc, float x0, float y0, float dx, float dy, int Nx, int Ny, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float alias_fmod) -> Tensor",
    "cart_to_polar_linear_grad(Tensor grad, Tensor img, Tensor origin, int nbatch, float rotation, float fc, float x0, float y0, float dx, float dy, int Nx, int Ny, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float alias_fmod) -> Tensor?[]",
    "cart_to_polar_lanczos(Tensor img, Tensor origin, int nbatch, float rotation, float fc, float x0, float y0, float dx, float dy, int Nx, int Ny, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, float alias_fmod, int order) -> Tensor",
    "backprojection_polar_2d_tx_power(Tensor wa, Tensor pos, Tensor att, Tensor g, int nbatch, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int nsweeps, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, int normalization, int azimuth_resolution, Tensor dem) -> Tensor",
    "backprojection_polar_2d_tx_power_slant(Tensor wa, Tensor pos, Tensor att, Tensor g, int nbatch, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int nsweeps, float r_res, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, int normalization, int azimuth_resolution, float altitude) -> Tensor",
    "backprojection_cart_2d_tx_power(Tensor wa, Tensor pos, Tensor att, Tensor g, int nbatch, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int nsweeps, float r_res, float x0, float dx, float y0, float dy, int Nx, int Ny, int normalization, int azimuth_resolution) -> Tensor",
    "backprojection_polar_2d_tx_power_accum(Tensor wa, Tensor pos, Tensor att, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int nsweeps, float r0, float dr, float theta0, float dtheta, int Nr, int Ntheta, int normalization, float dr_ref, float h_ref, float altitude, int theta_psi, Tensor dem) -> Tensor",
    "backprojection_cart_2d_tx_power_accum(Tensor wa, Tensor pos, Tensor att, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int nsweeps, float x0, float dx, float y0, float dy, int Nx, int Ny, int normalization, float dx_ref, float h_ref) -> Tensor",
    "cart_tx_power_merge2(Tensor acc0, Tensor acc1, float x0_0, float dx_0, float y0_0, float dy_0, int Nx_0, int Ny_0, float x0_1, float dx_1, float y0_1, float dy_1, int Nx_1, int Ny_1, float x1, float dx1, float y1, float dy1, int Nx1, int Ny1) -> Tensor",
    "ffbp_tx_power_merge2(Tensor acc0, Tensor acc1, Tensor dorigin, Tensor r0, Tensor dr0, Tensor theta0, Tensor dtheta0, Tensor Nr0, Tensor Ntheta0, float r1, float dr1, float theta1, float dtheta1, int Nr1, int Ntheta1, float altitude, int in_psi, int out_psi) -> Tensor",
    "compute_illumination(Tensor pos, Tensor att, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, float r0, float dr, float theta0, float dtheta, int nr, int ntheta, int decimation, Tensor dem) -> Tensor?[]",
    "entropy(Tensor data, Tensor norm, int nbatch) -> Tensor",
    "entropy_grad(Tensor data, Tensor norm, Tensor grad, int nbatch) -> Tensor?[]",
    "abs_sum(Tensor data, int nbatch) -> Tensor",
    "abs_sum_grad(Tensor data, Tensor grad, int nbatch) -> Tensor",
    "lee_filter(Tensor img, int nbatch, int Nx, int Ny, int wx, int wy, float cu) -> Tensor",
    "coherence_2d(Tensor img0, Tensor img1, int nbatch, int N0, int N1, int w0, int w1) -> Tensor",
    "coherence_2d_grad(Tensor grad, Tensor img0, Tensor img1, int nbatch, int N0, int N1, int w0, int w1) -> Tensor?[]",
    "power_coherence_2d(Tensor img0, Tensor img1, int nbatch, int N0, int N1, int w0, int w1, int corr_output) -> Tensor",
    "projection_cart_2d(Tensor img, Tensor dem, Tensor pos, Tensor vel, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float fs, float gamma, float x0, float dx, float y0, float dy, int Nx, int Ny, float d0, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int use_rvp, int normalization) -> Tensor",
    "projection_cart_2d_nufft(Tensor img, Tensor dem, Tensor pos, Tensor att, int nbatch, int sweep_samples, int nsweeps, float fc, float fs, float gamma, float x0, float dx, float y0, float dy, int Nx, int Ny, float d0, Tensor g, float g_az0, float g_el0, float g_daz, float g_del, int g_naz, int g_nel, int use_rvp, int normalization) -> Tensor",
    "subpixel_correlation(Tensor im_m, Tensor im_s, Tensor mean_m, Tensor mean_s, int nbatch, int N0, int N1) -> Tensor?[]",
    "div_2d_interp_linear(Tensor a, Tensor b, int nbatch, int Na0, int Na1, int Nb0, int Nb1) -> Tensor",
    "mul_2d_interp_linear(Tensor a, Tensor b, int nbatch, int Na0, int Na1, int Nb0, int Nb1) -> Tensor",
    "resample_2d_lanczos(Tensor img, Tensor shift_r, Tensor shift_az, int nbatch, int Nr, int Naz, int order) -> Tensor",
    "resample_2d_knab(Tensor img, Tensor shift_r, Tensor shift_az, int nbatch, int Nr, int Naz, int order, float oversample) -> Tensor",
    "resample_1d_lanczos(Tensor img, int nbatch, int N, int M, int order) -> Tensor",
    "resample_1d_knab(Tensor img, int nbatch, int N, int M, int order, float oversample) -> Tensor",
    "afbp_fuse(Tensor S, Tensor nua, Tensor xs, Tensor inv_kr, Tensor x_half, Tensor band, float x_c, float x_taper, int kmax) -> Tensor",
]

_lib = torch.library.Library("torchbp", "DEF")
for _schema in SCHEMAS:
    _lib.define(_schema)


def register(name):
    """Register a device-independent kernel for operator ``name``."""
    def deco(fn):
        _lib.impl(name, fn, "CompositeExplicitAutograd")
        return fn
    return deco


from . import simple, backproj, polar_interp  # noqa: E402,F401
