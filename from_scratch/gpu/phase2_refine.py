"""
Phase 2: Multi-start GPU refinement (exact ray tracing + spot size).

Reads sweep results JSON from phase1_sweep.py, refines top designs,
and exports the best result.

Usage: python phase2_refine.py sweep_20260414_151155.json [--out result.json]
"""

import sys
import os
import time
import math
import json
import argparse
import numpy as np
from datetime import datetime

from numba import cuda, float64, float32
from numba.cuda.random import (create_xoroshiro128p_states,
                                xoroshiro128p_uniform_float64)

from host import (
    pack_x, unpack_x, build_system,
    ynu_trace, find_chief_ray_initial, seidel_coefficients, exact_trace,
    _thin_lens_ri, _thin_lens_ca,
)
from common import (
    N_ELEM, K_SURF, N_GAPS, N_AIR, N_SPHER, N_XVARS,
    X_DG, X_DA, X_ND, X_K, X_A4, X_A6,
    DEG2RAD, RAD2DEG,
    d_ynu_trace, d_sag,
    d_find_chief, d_build, d_ca, d_vignetting,
)

# Params array layout for refinement kernel
P_EFL      = 0
P_FNUM     = 1
P_STOP     = 2
P_NFIELDS  = 3
P_FIELDS   = 4    # 3 field angles in degrees (indices 4-6)
P_FWEIGHTS = 7    # 3 field weights (indices 7-9)
P_WEFL     = 10
P_WBFL     = 11
P_BFL_T    = 12
P_WTTL     = 13
P_TTL_T    = 14
P_WCRA     = 15
P_CRA_M    = 16
P_WRI      = 17
P_RI_M     = 18
P_WET      = 19
P_DEDGE    = 20
P_DAIR     = 21
P_EMIN     = 22
P_WSPOT    = 23
N_PARAMS   = 24


# ══════════════════════════════════════════════════════════════════════
#  DEVICE FUNCTIONS (phase 2 only)
# ══════════════════════════════════════════════════════════════════════

@cuda.jit(device=True)
def d_sag_intersect(c, h_v, tan_U, k=0.0, A4=0.0, A6=0.0):
    """Newton iteration to find where a ray hits a conic+aspheric surface.

    Ray arrives at the vertex plane at height h_v traveling at angle
    atan(tan_U).  Returns (h_hit, z_sag, ok).
    """
    h_hit = h_v
    for _it in range(12):
        ch2 = (1.0 + k) * c * c * h_hit * h_hit
        if ch2 >= 0.99:
            return h_hit, 0.0, False
        R = math.sqrt(1.0 - ch2)
        h2 = h_hit * h_hit
        z_s = c * h2 / (1.0 + R) + A4 * h2 * h2 + A6 * h2 * h2 * h2
        dzdh = c * h_hit / R + 4.0 * A4 * h_hit * h2 + 6.0 * A6 * h_hit * h2 * h2
        if abs(tan_U) > 1e-12:
            f_val = z_s - (h_hit - h_v) / tan_U
            df = dzdh - 1.0 / tan_U
        else:
            f_val = z_s * tan_U - (h_hit - h_v)
            df = dzdh * tan_U - 1.0
        if abs(df) < 1e-30:
            break
        dh = f_val / df
        h_hit -= dh
        if abs(dh) < 1e-13:
            break
        if abs(h_hit) > 200.0:
            return h_hit, 0.0, False
    ch2 = (1.0 + k) * c * c * h_hit * h_hit
    if ch2 >= 0.99:
        return h_hit, 0.0, False
    R = math.sqrt(1.0 - ch2)
    h2 = h_hit * h_hit
    z_sag = c * h2 / (1.0 + R) + A4 * h2 * h2 + A6 * h2 * h2 * h2
    return h_hit, z_sag, True


@cuda.jit(device=True)
def d_exact_trace(sc, sn, snp, gd, h0, u0, h, nu, sk, sA4, sA6):
    """Exact meridional ray trace with sag-corrected ray-surface intersection.

    At each surface, Newton iteration finds the true intersection height
    on the conic+aspheric surface.  Transfer between surfaces accounts for
    departure sag (ray leaves from the surface, not the vertex plane).

    sk, sA4, sA6: per-surface conic constant, A4, A6 arrays.
    Stores nu[i] = n'*sin(U'), propagates via tan(U').
    """
    # ── First surface ──
    tan_U = math.tan(u0)
    h_hit, z_sag, ok = d_sag_intersect(sc[0], h0, tan_U,
                                        sk[0], sA4[0], sA6[0])
    if not ok:
        nu[K_SURF - 1] = 2.0
        return

    h[0] = h_hit
    sin_a = h_hit * sc[0]
    if sin_a > 0.9999 or sin_a < -0.9999:
        nu[K_SURF - 1] = 2.0
        return
    alpha = math.asin(sin_a)
    sin_I = math.sin(u0 + alpha)
    sin_Ip = (sn[0] / snp[0]) * sin_I
    if sin_Ip > 0.9999 or sin_Ip < -0.9999:
        nu[K_SURF - 1] = 2.0
        return
    Ip = math.asin(sin_Ip)
    U_post = Ip - alpha
    nu[0] = snp[0] * math.sin(U_post)

    # ── Subsequent surfaces ──
    for i in range(1, K_SURF):
        tan_Up = math.tan(U_post)
        h_v = h[i - 1] + (gd[i - 1] - z_sag) * tan_Up

        h_hit, z_sag, ok = d_sag_intersect(sc[i], h_v, tan_Up,
                                            sk[i], sA4[i], sA6[i])
        if not ok:
            nu[K_SURF - 1] = 2.0
            return

        h[i] = h_hit
        sin_a = h_hit * sc[i]
        if sin_a > 0.9999 or sin_a < -0.9999:
            nu[K_SURF - 1] = 2.0
            return
        alpha = math.asin(sin_a)
        sin_I = math.sin(U_post + alpha)
        sin_Ip = (sn[i] / snp[i]) * sin_I
        if sin_Ip > 0.9999 or sin_Ip < -0.9999:
            nu[K_SURF - 1] = 2.0
            return
        Ip = math.asin(sin_Ip)
        U_post = Ip - alpha
        nu[i] = snp[i] * math.sin(U_post)


@cuda.jit(device=True)
def d_merit_eval(x, params, sc, sn, snp, gd, gn,
                 h1, n1, hm, nm, hc, nc):
    """
    Exact-raytrace spot-size merit for design vector x[N_XVARS].
    Traces 5 pupil rays at up to 3 field angles, computes image-plane
    spot error.  Includes penalties for EFL, BFL, TTL, CRA, RI, edge.
    Supports conic + even asphere (k, A4, A6) per surface.
    """
    efl_target = params[P_EFL]
    f_number   = params[P_FNUM]
    stop_idx   = int(params[P_STOP])
    n_fields   = int(params[P_NFIELDS])
    w_efl      = params[P_WEFL]
    w_bfl      = params[P_WBFL]
    bfl_tgt    = params[P_BFL_T]
    w_ttl      = params[P_WTTL]
    ttl_tgt    = params[P_TTL_T]
    w_cra      = params[P_WCRA]
    cra_mx     = params[P_CRA_M]
    w_ri       = params[P_WRI]
    ri_mn      = params[P_RI_M]
    w_et       = params[P_WET]
    d_edge_min = params[P_DEDGE]
    d_air_min  = params[P_DAIR]
    e_min      = params[P_EMIN]
    w_spot     = params[P_WSPOT]

    # Unpack aspheric coefficients from x into local arrays
    sk  = cuda.local.array(K_SURF, float64)
    sA4 = cuda.local.array(K_SURF, float64)
    sA6 = cuda.local.array(K_SURF, float64)
    for i in range(K_SURF):
        sk[i]  = x[X_K + i]
        sA4[i] = x[X_A4 + i]
        sA6[i] = x[X_A6 + i]

    # Build system from x
    for i in range(N_ELEM):
        sc[2 * i]     = x[2 * i]
        sc[2 * i + 1] = x[2 * i + 1]
        sn[2 * i]     = 1.0
        snp[2 * i]    = x[X_ND + i]
        sn[2 * i + 1] = x[X_ND + i]
        snp[2 * i + 1] = 1.0
    gi = 0
    for i in range(N_ELEM):
        gd[gi] = x[X_DG + i]
        gn[gi] = x[X_ND + i]
        gi += 1
        if i < N_AIR:
            gd[gi] = x[X_DA + i]
            gn[gi] = 1.0
            gi += 1

    # Paraxial setup: EFL, SA (paraxial trace is curvature-only, aspherics don't affect it)
    d_ynu_trace(sc, sn, snp, gd, gn, 1.0, 0.0, h1, n1)
    if abs(n1[K_SURF - 1]) < 1e-12:
        return 1e12
    efl = -1.0 / n1[K_SURF - 1]
    if efl < 0.0:
        return 1e12
    sa = efl / (2.0 * f_number)

    # Pass 1: trace all rays, find global best-focus BFL across all fields
    S_ab = 0.0
    S_bb = 0.0
    n_tir = 0

    h_max = cuda.local.array(K_SURF, float32)
    for i in range(K_SURF):
        h_max[i] = 0.0

    # Store per-ray results: a_i and b_i for up to 3*5=15 rays
    ray_a = cuda.local.array(15, float64)
    ray_b = cuda.local.array(15, float64)
    ray_fw = cuda.local.array(15, float64)
    ray_fi = cuda.local.array(15, float64)  # field index per ray
    n_good = 0

    # Cache chief ray for max field (reused in CRA penalty)
    h0c_max = 0.0
    u0c_max = 0.0
    ok_max = False

    for fi in range(n_fields):
        field_d = params[P_FIELDS + fi]
        fw      = params[P_FWEIGHTS + fi]
        if fw <= 0.0:
            continue

        if field_d > 0.01:
            nu0_f = math.tan(field_d * DEG2RAD)
            h0c, ok_c = d_find_chief(sc, sn, snp, gd, gn, stop_idx,
                                     nu0_f, h1, n1)
            if not ok_c:
                return 1e12
            u0c = math.atan(nu0_f)
            if fi == n_fields - 1:
                h0c_max = h0c
                u0c_max = u0c
                ok_max = True
        else:
            h0c = 0.0
            u0c = 0.0

        y_ideal = efl * math.tan(field_d * DEG2RAD)

        for ri in range(5):
            rho = -1.0 + 0.5 * ri
            h0_ray = h0c + rho * sa
            d_exact_trace(sc, sn, snp, gd, h0_ray, u0c, h1, n1,
                          sk, sA4, sA6)

            for s in range(K_SURF):
                ah = abs(h1[s])
                if ah > h_max[s]:
                    h_max[s] = ah

            nu_last = n1[K_SURF - 1]
            nu2 = nu_last * nu_last
            if nu2 >= 0.9999:
                n_tir += 1
                continue
            tan_U = nu_last / math.sqrt(1.0 - nu2)
            c_last = sc[K_SURF - 1]
            h_last = h1[K_SURF - 1]
            z_sag_last = d_sag(c_last, h_last,
                               sk[K_SURF - 1], sA4[K_SURF - 1], sA6[K_SURF - 1])
            a = h_last - tan_U * z_sag_last - y_ideal
            b = tan_U

            # Store ray result
            if n_good < 15:
                ray_a[n_good] = a
                ray_b[n_good] = b
                ray_fw[n_good] = fw
                ray_fi[n_good] = fi
                n_good += 1

            S_ab += fw * a * b
            S_bb += fw * b * b

    if n_tir >= n_fields * 3:
        return 1e12

    # Global best-focus BFL (one sensor plane for all fields)
    if S_bb < 1e-30:
        return 1e12
    bfl = -S_ab / S_bb

    # Pass 2: compute per-field RMS at the shared best-focus plane
    merit = 0.0
    for fi in range(n_fields):
        fw = params[P_FWEIGHTS + fi]
        if fw <= 0.0:
            continue
        sum_err_sq = 0.0
        cnt = 0
        for k in range(n_good):
            if int(ray_fi[k]) == fi:
                err = ray_a[k] + bfl * ray_b[k]
                sum_err_sq += err * err
                cnt += 1
        if cnt > 0:
            rms_sq = sum_err_sq / cnt
        else:
            rms_sq = 100.0  # penalty for no valid rays
        merit += w_spot * fw * rms_sq

    # TIR penalty
    if n_tir > 0:
        merit += n_tir * 100.0

    # EFL penalty
    efl_err = efl - efl_target
    merit += w_efl * efl_err * efl_err

    # BFL one-sided (sensor clearance)
    if w_bfl > 0.0:
        v = bfl_tgt - bfl
        if v > 0.0:
            merit += w_bfl * v * v

    # TTL one-sided
    ttl = bfl
    for i in range(N_GAPS):
        ttl += gd[i]
    if w_ttl > 0.0:
        v = ttl - ttl_tgt
        if v > 0.0:
            merit += w_ttl * v * v

    # CRA one-sided (exact trace at max field)
    max_field = params[P_FIELDS + n_fields - 1]
    if w_cra > 0.0 and max_field > 0.01 and ok_max:
        d_exact_trace(sc, sn, snp, gd, h0c_max, u0c_max, hc, nc,
                      sk, sA4, sA6)
        cra = math.asin(nc[K_SURF - 1]) * RAD2DEG
        v = abs(cra) - cra_mx
        if v > 0.0:
            merit += w_cra * v * v

    # RI one-sided (paraxial vignetting)
    if w_ri > 0.0:
        sd = cuda.local.array(K_SURF, float32)
        for e in range(N_ELEM):
            ca_e = d_ca(sc[2*e], sc[2*e+1], gd[2*e], e_min,
                        sk[2*e], sk[2*e+1],
                        sA4[2*e], sA4[2*e+1],
                        sA6[2*e], sA6[2*e+1])
            sd[2*e]   = ca_e
            sd[2*e+1] = ca_e

        d_ynu_trace(sc, sn, snp, gd, gn, sa, 0.0, hm, nm)

        max_fd = params[P_FIELDS + n_fields - 1]
        nu0_c = math.tan(max_fd * DEG2RAD)
        if ok_max:
            d_ynu_trace(sc, sn, snp, gd, gn, h0c_max, nu0_c, hc, nc)
            nu0_ref = math.tan(0.1 * DEG2RAD)
            h0_ref, ok_r = d_find_chief(sc, sn, snp, gd, gn, stop_idx,
                                        nu0_ref, h1, n1)
            if ok_r:
                d_ynu_trace(sc, sn, snp, gd, gn, h0_ref, nu0_ref, h1, n1)
                vf = d_vignetting(hm, hc, sd)
                vr = d_vignetting(hm, h1, sd)
                cf = math.cos(max_fd * DEG2RAD)
                cr = math.cos(0.1 * DEG2RAD)
                c4f = cf * cf * cf * cf
                c4r = cr * cr * cr * cr
                if vr < 1e-15:
                    ri = 0.0
                else:
                    ri = (c4f * vf) / (c4r * vr)
                v = ri_mn - ri
                if v > 0.0:
                    merit += w_ri * v * v

    # Edge thickness using max beam heights from exact traces
    if w_et > 0.0:
        for gi2 in range(N_GAPS):
            c_f = sc[gi2]
            c_b = sc[gi2 + 1]
            y_f = h_max[gi2] * 1.05
            y_b = h_max[gi2 + 1] * 1.05
            s_f = d_sag(c_f, y_f, sk[gi2], sA4[gi2], sA6[gi2])
            s_b = d_sag(c_b, y_b, sk[gi2 + 1], sA4[gi2 + 1], sA6[gi2 + 1])
            n_gap = gn[gi2]
            edge_min = d_edge_min if n_gap > 1.001 else 0.0
            et = gd[gi2] - s_f + s_b - edge_min
            if n_gap <= 1.001 and d_air_min > 0.0:
                et -= d_air_min
            if et < 0.0:
                merit += w_et * et * et

    return merit


# ══════════════════════════════════════════════════════════════════════
#  REFINEMENT KERNEL — SPGD
# ══════════════════════════════════════════════════════════════════════

@cuda.jit
def spgd_kernel(starts, bounds_lo, bounds_hi, rng_states,
                n_iters, delta, params, out_merit, out_x):
    """
    Coordinate descent with random steps and merit-adaptive perturbation.

    Grid:  n_starts blocks  x  TPB threads
    Each thread independently optimizes starts[blockIdx.x].

    Each iteration cycles through all N_XVARS variables:
      - Pick a random step for variable i
      - Evaluate merit with that one variable changed
      - If better, keep. If worse, revert that variable.
      - 1 eval per variable, N_XVARS evals per full pass.

    n_iters = number of full passes through all variables.

    Perturbation scales with sqrt(current merit):
        delta_eff = delta * clamp(sqrt(best), 0.01, 10.0)
    """
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    gid = bid * cuda.blockDim.x + tid

    sc  = cuda.local.array(K_SURF, float64)
    sn  = cuda.local.array(K_SURF, float64)
    snp = cuda.local.array(K_SURF, float64)
    gd  = cuda.local.array(N_GAPS, float64)
    gn  = cuda.local.array(N_GAPS, float64)
    h1  = cuda.local.array(K_SURF, float64)
    n1  = cuda.local.array(K_SURF, float64)
    hm  = cuda.local.array(K_SURF, float64)
    nm  = cuda.local.array(K_SURF, float64)
    hc  = cuda.local.array(K_SURF, float64)
    nc  = cuda.local.array(K_SURF, float64)
    x   = cuda.local.array(N_XVARS, float64)

    for i in range(N_XVARS):
        x[i] = starts[bid, i]

    best = d_merit_eval(x, params, sc, sn, snp, gd, gn,
                        h1, n1, hm, nm, hc, nc)

    for _pass in range(n_iters):
        # Adaptive perturbation scale from current merit
        scale = math.sqrt(best) if best > 0.0 else 0.01
        if scale < 0.01:
            scale = 0.01
        if scale > 10.0:
            scale = 10.0
        delta_eff = delta * scale

        for i in range(N_XVARS):
            rng_i = bounds_hi[i] - bounds_lo[i]
            if rng_i < 1e-15:
                continue

            # Random step for this one variable
            r = xoroshiro128p_uniform_float64(rng_states, gid)
            d = delta_eff * rng_i * (2.0 * r - 1.0)

            old_val = x[i]
            new_val = old_val + d
            if new_val > bounds_hi[i]:
                new_val = bounds_hi[i]
            if new_val < bounds_lo[i]:
                new_val = bounds_lo[i]

            x[i] = new_val
            m_new = d_merit_eval(x, params, sc, sn, snp, gd, gn,
                                 h1, n1, hm, nm, hc, nc)

            if m_new < best:
                best = m_new  # keep the change
            else:
                x[i] = old_val  # revert

    out_merit[gid] = best
    for i in range(N_XVARS):
        out_x[gid, i] = x[i]



# ══════════════════════════════════════════════════════════════════════
#  HOST: PHASE 2
# ══════════════════════════════════════════════════════════════════════

def load_sweep_results(path, n_top=20):
    """Load sweep results JSON from phase 1, keeping only top n_top."""
    with open(path) as f:
        data = json.load(f)
    config = data['config']
    designs = data['designs'][:n_top]
    print(f"  Loaded top {len(designs)} designs from {path}")
    return config, designs


def phase2_refine(designs, efl_target, f_number, field_angle_deg,
                  stop_idx, w_spot, w_efl, w_et, d_edge_min,
                  bfl_target, w_bfl, ttl_target, w_ttl,
                  cra_max, w_cra, ri_min, w_ri,
                  c_bounds, d_glass_bounds, d_air_bounds, n_d_bounds,
                  fields_deg=None, field_weights=None,
                  aspheric_surfaces=None,
                  k_bounds=(-5.0, 5.0),
                  a4_bounds=(-0.01, 0.01),
                  a6_bounds=(-0.001, 0.001),
                  tpb=256, n_iters=20, n_rounds=20,
                  delta=0.003):
    """Multi-start GPU refinement: SPGD scatter + select.

    Each round scatters random perturbations around each starting design,
    evaluates exact ray-trace merit, and keeps the best. Decreasing scale
    narrows the search progressively.

    aspheric_surfaces : set of surface indices (0-based) with free k/A4/A6.
        All other surfaces have k=A4=A6=0 (bounds locked).
    """
    print()
    print("=" * 80)
    print("PHASE 2 -- MULTI-START GPU REFINEMENT (exact ray trace)")
    print("=" * 80)

    if fields_deg is None:
        fields_deg = [0.0, 15.0, 30.0]
    if field_weights is None:
        field_weights = [1.0, 1.0, 1.0]
    asph_set = set(aspheric_surfaces) if aspheric_surfaces is not None else set()
    if asph_set:
        print(f"  Aspheric surfaces: {sorted(asph_set)}")

    n_starts = len(designs)
    if n_starts == 0:
        return []

    # Convert sweep results to x-vectors (spherical part from sweep,
    # aspheric slots initialized to zero)
    starts = np.zeros((n_starts, N_XVARS), dtype=np.float64)
    for i, r in enumerate(designs):
        full = pack_x(r['c1_list'], r['c2_list'],
                       r['d_glass'], r['d_air'], r['n_d_list'])
        n = min(len(full), N_SPHER)
        starts[i, :n] = full[:n]
        # k, A4, A6 slots stay at 0.0

    # Build bounds
    bounds_lo = np.zeros(N_XVARS, dtype=np.float64)
    bounds_hi = np.zeros(N_XVARS, dtype=np.float64)
    for j in range(K_SURF):
        bounds_lo[j], bounds_hi[j] = c_bounds
    for j in range(N_ELEM):
        bounds_lo[X_DG + j], bounds_hi[X_DG + j] = d_glass_bounds
    for j in range(N_AIR):
        bounds_lo[X_DA + j], bounds_hi[X_DA + j] = d_air_bounds
    for j in range(N_ELEM):
        bounds_lo[X_ND + j], bounds_hi[X_ND + j] = n_d_bounds
    # Aspheric bounds: only unlocked for surfaces in asph_set
    for j in range(K_SURF):
        if j in asph_set:
            bounds_lo[X_K + j], bounds_hi[X_K + j] = k_bounds
            bounds_lo[X_A4 + j], bounds_hi[X_A4 + j] = a4_bounds
            bounds_lo[X_A6 + j], bounds_hi[X_A6 + j] = a6_bounds
        # else: stays (0, 0) — locked to spherical

    # Pack optimizer params
    n_fields = min(3, len(fields_deg))
    params = np.zeros(N_PARAMS, dtype=np.float64)
    params[P_EFL]     = efl_target
    params[P_FNUM]    = f_number
    params[P_STOP]    = stop_idx
    params[P_NFIELDS] = n_fields
    for k in range(n_fields):
        params[P_FIELDS + k]   = fields_deg[k]
        params[P_FWEIGHTS + k] = field_weights[k]
    params[P_WEFL]  = w_efl
    params[P_WBFL]  = w_bfl
    params[P_BFL_T] = bfl_target
    params[P_WTTL]  = w_ttl
    params[P_TTL_T] = ttl_target
    params[P_WCRA]  = w_cra
    params[P_CRA_M] = cra_max
    params[P_WRI]   = w_ri
    params[P_RI_M]  = ri_min
    params[P_WET]   = w_et
    params[P_DEDGE] = d_edge_min
    params[P_DAIR]  = 0.3
    params[P_EMIN]  = 1.0
    params[P_WSPOT] = w_spot

    # GPU arrays
    d_bounds_lo = cuda.to_device(bounds_lo)
    d_bounds_hi = cuda.to_device(bounds_hi)
    d_params    = cuda.to_device(params)

    TPB = tpb
    KEEP_K = max(4, TPB // 16)  # keep top ~16 threads per design
    n_threads = n_starts * TPB
    rng_states = create_xoroshiro128p_states(n_threads, seed=123)

    current_starts = starts.copy()

    for rd in range(n_rounds):
        t0 = time.perf_counter()

        d_starts = cuda.to_device(current_starts)
        out_merit = np.full(n_threads, 1e12, dtype=np.float64)
        out_x     = np.zeros((n_threads, N_XVARS), dtype=np.float64)
        d_merit   = cuda.to_device(out_merit)
        d_x       = cuda.to_device(out_x)

        spgd_kernel[n_starts, TPB](
            d_starts, d_bounds_lo, d_bounds_hi, rng_states,
            n_iters, delta, d_params, d_merit, d_x)
        cuda.synchronize()

        out_merit = d_merit.copy_to_host()
        out_x     = d_x.copy_to_host()
        t_round = time.perf_counter() - t0

        # Keep top-K threads per design, duplicate to fill TPB slots
        merits_2d = out_merit.reshape(n_starts, TPB)
        xs_2d     = out_x.reshape(n_starts, TPB, N_XVARS)
        best_per_design = merits_2d.min(axis=1)

        for i in range(n_starts):
            top_k_idx = merits_2d[i].argsort()[:KEEP_K]
            # Next round starts from these top-K, cycled across TPB slots
            # The kernel reads starts[bid] so all threads in a block get
            # the same start — cycle through top-K so different rounds
            # start from different elite points
            winner = top_k_idx[rd % KEEP_K]
            current_starts[i] = xs_2d[i, winner]

        # Estimate avg per-field RMS from best merit
        # merit ≈ w_spot * sum(fw_i * rms_i²) + penalties
        # With equal fw=1: avg_rms ≈ sqrt(merit / w_spot / n_fields)
        best_merit = best_per_design.min()
        w_spot_val = params[P_WSPOT]
        nf = len(fields_deg)
        rms_est = np.sqrt(max(0, best_merit) / max(w_spot_val, 1) / max(nf, 1))
        print(f"  R{rd+1:2d} delta={delta:.4f}: "
              f"{t_round:.2f}s  "
              f"best={best_merit:.4e}  "
              f"median={np.median(best_per_design):.4e}  "
              f"RMS~{rms_est*1000:.0f}um")

    # Final ranking: pick best thread per design
    merits_2d = out_merit.reshape(n_starts, TPB)
    xs_2d     = out_x.reshape(n_starts, TPB, N_XVARS)
    best_thread = merits_2d.argmin(axis=1)
    final_merits = merits_2d[np.arange(n_starts), best_thread]
    final_xs     = xs_2d[np.arange(n_starts), best_thread]
    rank = final_merits.argsort()

    refined = []
    for k in range(n_starts):
        i = rank[k]
        x = final_xs[i]
        c1 = x[0:K_SURF:2]
        c2 = x[1:K_SURF:2]
        dg = x[X_DG:X_DG + N_ELEM]
        da = x[X_DA:X_DA + N_AIR]
        nd = x[X_ND:X_ND + N_ELEM]
        k_vals  = x[X_K:X_K + K_SURF]
        a4_vals = x[X_A4:X_A4 + K_SURF]
        a6_vals = x[X_A6:X_A6 + K_SURF]
        entry = {
            'merit': float(final_merits[i]),
            'x': x,
            'c1_list': list(c1), 'c2_list': list(c2),
            'd_glass': list(dg), 'd_air': list(da),
            'n_d_list': list(nd),
            'k_list': list(k_vals),
            'a4_list': list(a4_vals),
            'a6_list': list(a6_vals),
            'pattern': designs[i]['pattern'],
            'signs': designs[i]['signs'],
        }
        refined.append(entry)

    top5 = [f"{refined[i]['merit']:.4e}" for i in range(min(5, len(refined)))]
    print(f"  Top 5 merit values: {top5}")

    return refined


def _sag_sphere_cpu(c, h_val):
    """Spherical sag (CPU helper)."""
    if abs(c) < 1e-15:
        return 0.0
    arg = 1.0 - c * c * h_val * h_val
    if arg <= 0.0:
        return c * h_val * h_val
    return c * h_val * h_val / (1.0 + np.sqrt(arg))


def _spot_rms_cpu(x, f_number, stop_idx, fields_deg, field_weights):
    """Compute spot RMS (mm) from a design vector on CPU."""
    surfs, gaps = build_system(x[:N_SPHER], N_ELEM)
    try:
        h_u, nu_u = ynu_trace(surfs, gaps, 1.0, 0.0)
        if abs(nu_u[-1]) < 1e-12:
            return 999.0
        efl = -1.0 / nu_u[-1]
        if efl < 0:
            return 999.0
        sa = efl / (2 * f_number)
    except Exception:
        return 999.0

    c_last = surfs[-1][0]
    errors_sq = []
    S_ab, S_bb = 0.0, 0.0

    for fd, fw in zip(fields_deg, field_weights):
        if fw <= 0:
            continue
        if fd > 0.01:
            h0c, nu0f = find_chief_ray_initial(surfs, gaps, stop_idx, fd)
            u0c = np.arctan(nu0f)
        else:
            h0c, u0c = 0.0, 0.0
        y_ideal = efl * np.tan(np.radians(fd))
        for rho in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            h0r = h0c + rho * sa
            try:
                h_r, nu_r = exact_trace(surfs, gaps, h0r, u0c)
            except Exception:
                continue
            nu_l = nu_r[-1]
            nu2 = nu_l * nu_l
            if nu2 >= 0.9999:
                continue
            tan_U = nu_l / np.sqrt(1 - nu2)
            h_last = h_r[-1]
            z_sag = _sag_sphere_cpu(c_last, h_last)
            a = h_last - tan_U * z_sag - y_ideal
            b = tan_U
            S_ab += fw * a * b
            S_bb += fw * b * b

    if S_bb < 1e-30:
        return 999.0
    bfl = -S_ab / S_bb

    for fd, fw in zip(fields_deg, field_weights):
        if fw <= 0:
            continue
        if fd > 0.01:
            h0c, nu0f = find_chief_ray_initial(surfs, gaps, stop_idx, fd)
            u0c = np.arctan(nu0f)
        else:
            h0c, u0c = 0.0, 0.0
        y_ideal = efl * np.tan(np.radians(fd))
        for rho in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            h0r = h0c + rho * sa
            try:
                h_r, nu_r = exact_trace(surfs, gaps, h0r, u0c)
            except Exception:
                continue
            nu_l = nu_r[-1]
            nu2 = nu_l * nu_l
            if nu2 >= 0.9999:
                continue
            tan_U = nu_l / np.sqrt(1 - nu2)
            h_last = h_r[-1]
            z_sag = _sag_sphere_cpu(c_last, h_last)
            y_img = h_last - tan_U * z_sag + bfl * tan_U
            errors_sq.append((y_img - y_ideal) ** 2)

    if len(errors_sq) == 0:
        return 999.0
    return np.sqrt(np.mean(errors_sq))


# ══════════════════════════════════════════════════════════════════════
#  FINAL REPORT + EXPORT
# ══════════════════════════════════════════════════════════════════════


def final_report(refined, efl_target, f_number, field_angle_deg,
                 stop_idx, fields_deg, field_weights, w_spot, outpath):
    """Print detailed report and export best design to JSON."""
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    N = N_ELEM
    fields_deg_list = fields_deg
    field_weights_list = field_weights

    def _best_focus_bfl(surfs, gaps, efl_val, sa_val, stop_i):
        c_last = surfs[-1][0]
        S_ab, S_bb = 0.0, 0.0
        for fd, fw in zip(fields_deg_list, field_weights_list):
            if fw <= 0:
                continue
            if fd > 0.01:
                h0c, nu0f = find_chief_ray_initial(surfs, gaps, stop_i, fd)
                u0c = np.arctan(nu0f)
            else:
                h0c, u0c = 0.0, 0.0
            y_ideal = efl_val * np.tan(np.radians(fd))
            for rho in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                h0r = h0c + rho * sa_val
                try:
                    h_r, nu_r = exact_trace(surfs, gaps, h0r, u0c)
                except Exception:
                    continue
                nu_l = nu_r[-1]
                nu2 = nu_l * nu_l
                if nu2 >= 0.9999:
                    continue
                tan_U = nu_l / np.sqrt(1.0 - nu2)
                h_last = h_r[-1]
                z_sag = _sag_sphere_cpu(c_last, h_last)
                a = h_last - tan_U * z_sag - y_ideal
                b = tan_U
                S_ab += fw * a * b
                S_bb += fw * b * b
        if S_bb < 1e-30:
            return 0.0
        return -S_ab / S_bb

    n_show = min(10, len(refined))
    # w_spot passed as parameter
    print(f"\n  {'#':>3s} {'Pattern':>8s} {'Merit':>10s} {'RMS~um':>8s} {'EFL':>7s} "
          f"{'BFL':>6s} {'TTL':>6s} {'CRA':>7s}")
    for rank in range(n_show):
        r = refined[rank]
        x = r['x']
        rms_est = np.sqrt(max(0, r['merit']) / max(w_spot, 1) / max(len(fields_deg), 1))
        surfs, gaps = build_system(x, N)
        h_u, nu_u = ynu_trace(surfs, gaps, 1.0, 0.0)
        efl = -1.0 / nu_u[-1] if abs(nu_u[-1]) > 1e-12 else 0
        sa = efl / (2 * f_number)
        bfl = _best_focus_bfl(surfs, gaps, efl, sa, stop_idx)
        ttl = sum(g[0] for g in gaps) + bfl
        h0c, nu0c = find_chief_ray_initial(surfs, gaps, stop_idx,
                                           field_angle_deg)
        h_c, nu_c = ynu_trace(surfs, gaps, h0c, nu0c)
        cra_v = np.degrees(np.arctan(nu_c[-1]))
        print(f"  {rank+1:3d} {r['pattern']:>8s} {r['merit']:10.4e} "
              f"{rms_est*1000:8.1f} {efl:7.2f} {bfl:6.2f} {ttl:6.2f} {cra_v:+7.1f}")

    # Detail for top 5
    print()
    for rank in range(min(5, n_show)):
        r = refined[rank]
        x = r['x']
        surfs, gaps = build_system(x, N)
        h_u, nu_u = ynu_trace(surfs, gaps, 1.0, 0.0)
        efl = -1.0 / nu_u[-1]
        sa = efl / (2 * f_number)
        h_m, nu_m = ynu_trace(surfs, gaps, sa, 0.0)
        h0c, nu0c = find_chief_ray_initial(surfs, gaps, stop_idx,
                                           field_angle_deg)
        h_c, nu_c = ynu_trace(surfs, gaps, h0c, nu0c)
        bfl = _best_focus_bfl(surfs, gaps, efl, sa, stop_idx)
        ttl = sum(g[0] for g in gaps) + bfl
        cra_v = np.degrees(np.arctan(nu_c[-1]))

        S_I, S_II, S_III, S_IV, S_V = seidel_coefficients(
            surfs, h_m, nu_m, h_c, nu_c, 0.0, nu0c)
        S = [float(S_I.sum()), float(S_II.sum()), float(S_III.sum()),
             float(S_IV.sum()), float(S_V.sum())]
        srss = float(np.sqrt(sum(s**2 for s in S)))

        c1_l = r['c1_list']
        c2_l = r['c2_list']
        ca_list = [_thin_lens_ca(c1_l[j], c2_l[j], r['d_glass'][j],
                                 r['n_d_list'][j], 0.5)
                   for j in range(N)]
        try:
            ri, _, _ = _thin_lens_ri(surfs, gaps, stop_idx, f_number,
                                     field_angle_deg, ca_list)
        except Exception:
            ri = 0.0

        rms = _spot_rms_cpu(x, f_number, stop_idx, fields_deg, field_weights)
        rms_est = np.sqrt(max(0, r['merit']) / max(w_spot, 1) / max(len(fields_deg), 1))
        print(f"  #{rank+1} [{r['pattern']}]  merit={r['merit']:.4e}  "
              f"RMS~{rms_est*1000:.0f}um (sph:{rms*1000:.0f}um)")
        print(f"    EFL={efl:.3f}  BFL={bfl:.2f}  TTL={ttl:.2f}  "
              f"CRA={cra_v:+.1f}  RI={ri:.3f}")
        print(f"    S_I={S[0]:+.6f}  S_II={S[1]:+.6f}  "
              f"S_III={S[2]:+.6f}  S_IV={S[3]:+.6f}  "
              f"S_V={S[4]:+.6f}  RSS={srss:.6f}")
        for j in range(N):
            c1v = c1_l[j]
            c2v = c2_l[j]
            r_f = 1/c1v if abs(c1v) > 1e-12 else float('inf')
            r_b = 1/c2v if abs(c2v) > 1e-12 else float('inf')
            sg = '+' if r['signs'][j] > 0 else '-'
            line = (f"    L{j+1}({sg}) R_f={r_f:+8.2f} R_b={r_b:+8.2f} "
                    f"d={r['d_glass'][j]:.3f} n_d={r['n_d_list'][j]:.4f}")
            # Show aspheric coefficients for surfaces that have them
            k_l = r.get('k_list', [0.0] * K_SURF)
            a4_l = r.get('a4_list', [0.0] * K_SURF)
            a6_l = r.get('a6_list', [0.0] * K_SURF)
            for si, label in [(2*j, 'f'), (2*j+1, 'b')]:
                if abs(k_l[si]) > 1e-12 or abs(a4_l[si]) > 1e-12 or abs(a6_l[si]) > 1e-12:
                    line += (f"\n      S{si}({label}): k={k_l[si]:+.4f} "
                             f"A4={a4_l[si]:+.6f} A6={a6_l[si]:+.6f}")
            print(line)
        print(f"    Air: {['%.2f' % d for d in r['d_air']]}")
        print()

    # Export best design to JSON
    best = refined[0]
    bx = best['x']
    surfs_b, gaps_b = build_system(bx, N)
    h_u, nu_u = ynu_trace(surfs_b, gaps_b, 1.0, 0.0)
    efl_b = -1.0 / nu_u[-1]
    sa_b = efl_b / (2 * f_number)
    h_m, nu_m = ynu_trace(surfs_b, gaps_b, sa_b, 0.0)
    bfl_b = _best_focus_bfl(surfs_b, gaps_b, efl_b, sa_b, stop_idx)
    ttl_b = sum(g[0] for g in gaps_b) + bfl_b
    h0c, nu0c = find_chief_ray_initial(surfs_b, gaps_b, stop_idx,
                                       field_angle_deg)
    h_c, nu_c = ynu_trace(surfs_b, gaps_b, h0c, nu0c)
    cra_b = float(np.degrees(np.arctan(nu_c[-1])))

    S_I, S_II, S_III, S_IV, S_V = seidel_coefficients(
        surfs_b, h_m, nu_m, h_c, nu_c, 0.0, nu0c)
    S_vals = [float(S_I.sum()), float(S_II.sum()), float(S_III.sum()),
              float(S_IV.sum()), float(S_V.sum())]

    c1_b = best['c1_list']
    c2_b = best['c2_list']
    ca_list_b = [_thin_lens_ca(c1_b[j], c2_b[j], best['d_glass'][j],
                               best['n_d_list'][j], 0.5) for j in range(N)]
    try:
        ri_b, _, _ = _thin_lens_ri(surfs_b, gaps_b, stop_idx, f_number,
                                   field_angle_deg, ca_list_b)
    except Exception:
        ri_b = 0.0

    data = {
        'n_elements': N,
        'efl_target': efl_target,
        'f_number': f_number,
        'field_angle_deg': field_angle_deg,
        'stop_idx': stop_idx,
        'result': {
            'merit': float(best['merit']),
            'efl': float(efl_b),
            'bfl': float(bfl_b),
            'ttl': float(ttl_b),
            'cra': cra_b,
            'ri': float(ri_b),
            'seidel': {
                'S_I': S_vals[0], 'S_II': S_vals[1],
                'S_III': S_vals[2], 'S_IV': S_vals[3],
                'S_V': S_vals[4],
            },
        },
        'elements': [],
        'air_gaps': [float(d) for d in best['d_air']],
    }
    for j in range(N):
        c1v = c1_b[j]
        c2v = c2_b[j]
        r_f = 1/c1v if abs(c1v) > 1e-12 else float('inf')
        r_b = 1/c2v if abs(c2v) > 1e-12 else float('inf')
        k_l = best.get('k_list', [0.0] * K_SURF)
        a4_l = best.get('a4_list', [0.0] * K_SURF)
        a6_l = best.get('a6_list', [0.0] * K_SURF)
        elem = {
            'label': f'L{j+1}',
            'c_front': float(c1v),
            'c_back': float(c2v),
            'R_front': float(r_f),
            'R_back': float(r_b),
            'd_glass': float(best['d_glass'][j]),
            'n_d': float(best['n_d_list'][j]),
            'k_front': float(k_l[2*j]),
            'k_back': float(k_l[2*j+1]),
            'A4_front': float(a4_l[2*j]),
            'A4_back': float(a4_l[2*j+1]),
            'A6_front': float(a6_l[2*j]),
            'A6_back': float(a6_l[2*j+1]),
        }
        data['elements'].append(elem)

    with open(outpath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Exported best design: {outpath}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    # ══════════════════════════════════════════════════════════════════
    #  ALL PARAMETERS — edit here
    # ══════════════════════════════════════════════════════════════════

    # --- I/O ---
    SWEEP_JSON = os.path.join(os.path.dirname(__file__), "sweep_6elem.json")
    OUT_JSON   = None                 # None = auto-name with timestamp

    # --- Merit function weights ---
    w_spot          = 1000.0           # spot size weight (makes ~0.1mm² spot → ~50 merit)
    w_efl           = 20.0            # EFL error weight
    w_et            = 30.0            # edge thickness penalty weight
    d_edge_min      = 1.0             # minimum edge thickness (mm)
    bfl_target      = 6.0             # minimum back focal length (mm)
    w_bfl           = 10.0            # BFL penalty weight
    ttl_target      = 40.0            # maximum total track length (mm)
    w_ttl           = 10.0            # TTL penalty weight
    w_cra           = 5.0             # chief ray angle penalty weight
    w_ri            = 20.0            # relative illumination penalty weight
    fields_deg      = [0.0, 15.0, 30.0]   # field angles (degrees)
    field_weights   = [1.0, 1.0, 1.0]     # weight per field

    # --- Variable bounds ---
    c_bounds        = (-0.3, 0.3)     # curvature range
    d_glass_bounds  = (1.0, 8.0)     # glass thickness range (mm)
    d_air_bounds    = (0.3, 10.0)     # air gap range (mm)
    n_d_bounds      = (1.45, 1.95)    # refractive index range

    # --- Aspheric surfaces ---
    # Front surface of last element (L6 front = surface index 2*5 = 10)
    aspheric_surfaces = {9, 10}
    k_bounds        = (-5.0, 5.0)
    a4_bounds       = (-0.1, 0.1)
    a6_bounds       = (-0.01, 0.01)

    # --- SPGD optimizer ---
    n_top           = 30              # only refine top N designs from sweep
    tpb             = 256             # threads per design (exploration)
    n_iters         = 10              # full passes through all variables per round
    n_rounds        = 50              # number of rounds
    delta           = 0.02           # base perturbation (scaled by sqrt(merit) in kernel)

    # ══════════════════════════════════════════════════════════════════

    parser = argparse.ArgumentParser(
        description="Phase 2: GPU refinement from sweep results")
    parser.add_argument('sweep_json', type=str, nargs='?', default=SWEEP_JSON,
                        help='Path to sweep results JSON')
    parser.add_argument('--out', type=str, default=OUT_JSON,
                        help='Output JSON path')
    args = parser.parse_args()

    dev = cuda.get_current_device()
    gpu_name = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
    print(f"GPU: {gpu_name}")
    print()

    # Load sweep results (top N only)
    config, designs = load_sweep_results(args.sweep_json, n_top=n_top)

    efl_target      = config['efl_target']
    f_number        = config['f_number']
    field_angle_deg = config['field_angle_deg']
    stop_idx        = config['stop_idx']
    cra_max         = config.get('cra_max', 13.0)
    ri_min          = config.get('ri_min', 0.45)

    refined = phase2_refine(
        designs, efl_target, f_number, field_angle_deg,
        stop_idx, w_spot, w_efl, w_et, d_edge_min,
        bfl_target, w_bfl, ttl_target, w_ttl,
        cra_max, w_cra, ri_min, w_ri,
        c_bounds, d_glass_bounds, d_air_bounds, n_d_bounds,
        fields_deg, field_weights,
        aspheric_surfaces=aspheric_surfaces,
        k_bounds=k_bounds, a4_bounds=a4_bounds, a6_bounds=a6_bounds,
        tpb=tpb, n_iters=n_iters, n_rounds=n_rounds,
        delta=delta)

    if not refined:
        print("Refinement produced no results.")
        return

    if args.out:
        outpath = args.out
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outpath = os.path.join(os.path.dirname(__file__),
                               f"refined_{timestamp}.json")

    final_report(refined, efl_target, f_number, field_angle_deg,
                 stop_idx, fields_deg, field_weights, w_spot, outpath)


if __name__ == "__main__":
    main()
