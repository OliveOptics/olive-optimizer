"""
Shared constants and CUDA device functions for the GPU lens pipeline.

Used by both phase1_sweep.py and phase2_refine.py.
"""

import sys
import os
import math

# CUDA setup
_nvcc_pkg = os.path.join(sys.prefix, 'Lib', 'site-packages',
                         'nvidia', 'cuda_nvcc')
if os.path.isdir(_nvcc_pkg) and 'CUDA_HOME' not in os.environ:
    os.environ['CUDA_HOME'] = _nvcc_pkg

from numba import cuda, float64, float32

# ── Constants ─────────────────────────────────────────────────────────

# Element count — change this to resize the system
N_ELEM = 6

# Derived constants (computed at import time -> Numba sees them as literals)
K_SURF = 2 * N_ELEM           # surfaces
N_GAPS = 2 * N_ELEM - 1       # gaps (glass + air interleaved)
N_AIR  = N_ELEM - 1           # air gaps
N_SPHER = K_SURF + N_ELEM + N_AIR + N_ELEM  # spherical-only count
N_XVARS = N_SPHER + 3 * K_SURF  # + k, A4, A6 per surface

X_DG   = K_SURF               # x-vector offset: d_glass
X_DA   = K_SURF + N_ELEM      # x-vector offset: d_air
X_ND   = K_SURF + N_ELEM + N_AIR  # x-vector offset: n_d
X_K    = N_SPHER               # x-vector offset: conic constants
X_A4   = N_SPHER + K_SURF      # x-vector offset: A4 coefficients
X_A6   = N_SPHER + 2 * K_SURF  # x-vector offset: A6 coefficients

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


# ── Device functions ──────────────────────────────────────────────────

@cuda.jit(device=True)
def d_ynu_trace(sc, sn, snp, gd, gn, h0, nu0, h, nu):
    h[0] = h0
    nu[0] = nu0 - h0 * (snp[0] - sn[0]) * sc[0]
    for i in range(1, K_SURF):
        h[i] = h[i - 1] + (gd[i - 1] / gn[i - 1]) * nu[i - 1]
        nu[i] = nu[i - 1] - h[i] * (snp[i] - sn[i]) * sc[i]


@cuda.jit(device=True)
def d_sag(c, h, k=0.0, A4=0.0, A6=0.0):
    """Conic + even asphere sag: c*h^2/(1+sqrt(1-(1+k)*c^2*h^2)) + A4*h^4 + A6*h^6."""
    if abs(c) < 1e-15:
        return A4 * h**4 + A6 * h**6
    arg = 1.0 - (1.0 + k) * c * c * h * h
    if arg <= 0.0:
        return c * h * h + A4 * h**4 + A6 * h**6
    return c * h * h / (1.0 + math.sqrt(arg)) + A4 * h**4 + A6 * h**6


@cuda.jit(device=True)
def d_find_chief(sc, sn, snp, gd, gn, stop_idx, nu0, ha, na):
    d_ynu_trace(sc, sn, snp, gd, gn, 0.0, nu0, ha, na)
    hs_a = ha[stop_idx]
    d_ynu_trace(sc, sn, snp, gd, gn, 1.0, nu0, ha, na)
    hs_b = ha[stop_idx]
    denom = hs_b - hs_a
    if abs(denom) < 1e-15:
        return 0.0, False
    return -hs_a / denom, True


@cuda.jit(device=True)
def d_build(c1, c2, dg, da, nd, sc, sn, snp, gd, gn):
    for i in range(N_ELEM):
        sc[2 * i] = c1[i]
        sc[2 * i + 1] = c2[i]
        sn[2 * i] = 1.0
        snp[2 * i] = nd[i]
        sn[2 * i + 1] = nd[i]
        snp[2 * i + 1] = 1.0
    gi = 0
    for i in range(N_ELEM):
        gd[gi] = dg[i]
        gn[gi] = nd[i]
        gi += 1
        if i < N_AIR:
            gd[gi] = da[i]
            gn[gi] = 1.0
            gi += 1


@cuda.jit(device=True)
def d_ca(c1, c2, dg, e_min, k1=0.0, k2=0.0,
         A4_1=0.0, A4_2=0.0, A6_1=0.0, A6_2=0.0):
    ac1, ac2 = abs(c1), abs(c2)
    if ac1 < 1e-12 and ac2 < 1e-12:
        return 1e6
    hs = 1e6
    if ac1 > 1e-12:
        hs = min(hs, 0.999 / ac1)
    if ac2 > 1e-12:
        hs = min(hs, 0.999 / ac2)
    if dg - d_sag(c1, hs, k1, A4_1, A6_1) + d_sag(c2, hs, k2, A4_2, A6_2) >= e_min:
        return hs
    if dg < e_min:
        return 0.0
    lo, hi = 0.0, hs
    for _ in range(20):
        mid = (lo + hi) * 0.5
        if dg - d_sag(c1, mid, k1, A4_1, A6_1) + d_sag(c2, mid, k2, A4_2, A6_2) >= e_min:
            lo = mid
        else:
            hi = mid
    return lo


@cuda.jit(device=True)
def d_vignetting(hm, hc, sd):
    rlo, rhi = -1.0, 1.0
    for i in range(K_SURF):
        if abs(hm[i]) < 1e-15:
            continue
        a = (-sd[i] - hc[i]) / hm[i]
        b = (sd[i] - hc[i]) / hm[i]
        if hm[i] < 0.0:
            a, b = b, a
        if a > rlo:
            rlo = a
        if b < rhi:
            rhi = b
    if rlo < -1.0:
        rlo = -1.0
    if rhi > 1.0:
        rhi = 1.0
    v = (rhi - rlo) * 0.5
    return max(v, 0.0)
