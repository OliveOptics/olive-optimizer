"""
Optimize the L_033 lens system (us04235519).

Starting point: L_033.zmx scaled to EFL=16mm.
Design variables: 13 curvatures + 12 thicknesses = 25 total.
Glass types (refractive indices) are fixed.

Settings:
    EFL = 16mm (equality constraint)
    f/# = 1.3
    FOV = 30 deg half-field

Penalties (one-sided):
    TTL <= 40mm
    BFL >= 5mm
    CRA <= 18 deg at 30 deg field
    RI  >= 30% at 30 deg field

Merit: Seidel aberration sum (minimize)
Constraints: edge thickness >= 0 (physical)
"""

import sys as _sys
import os

_project_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(_project_dir))
if _root_dir not in _sys.path:
    _sys.path.insert(0, _root_dir)

import numpy as np
from scipy.optimize import minimize
from system import load_system
from parse_zmx import (parse_zmx, write_modified_zmx, zmx_to_trace_map,
                       zmx_to_normalized, build_system_from_zmx,
                       ynu_trace, find_chief_ray_initial)
from gradient import (full_merit_with_grad, efl_with_grad,
                      edge_thickness_with_grad, trace_marginal_fnum,
                      geometry_sd_with_grad, sag_derivs)


# ── ZMX helpers ──────────────────────────────────────────────────────────────

def _update_zmx_settings(zmx_path, new_enpd, fov_half=None):
    """Update ENPD, PUPD, fields, and clear stale vignetting/ray-aiming."""
    import re
    try:
        with open(zmx_path, 'r', encoding='utf-16') as f:
            text = f.read()
        encoding = 'utf-16'
    except (UnicodeError, UnicodeDecodeError):
        with open(zmx_path, 'r', encoding='utf-8') as f:
            text = f.read()
        encoding = 'utf-8'

    # Update ENPD
    text = re.sub(r'(ENPD\s+)\S+', rf'\g<1>{new_enpd:.6f}', text)

    # Remove PUPD line — let Zemax recompute pupil position
    text = re.sub(r'PUPD\s+.*\n?', '', text)

    # Turn off ray aiming (set first param to 0)
    text = re.sub(r'RAIM\s+\S+', 'RAIM 0', text, count=1)

    # Clear vignetting factors (set to zero for all fields)
    text = re.sub(r'ZVDX\s+.*', 'ZVDX 0 0 0', text)
    text = re.sub(r'ZVDY\s+.*', 'ZVDY 0 0 0', text)
    text = re.sub(r'ZVCX\s+.*', 'ZVCX 0 0 0', text)
    text = re.sub(r'ZVCY\s+.*', 'ZVCY 0 0 0', text)

    # Update field values if specified
    if fov_half is not None:
        text = re.sub(r'YFLD\s+.*', f'YFLD 0 {fov_half * 0.7:.1f} {fov_half:.1f}', text)

    with open(zmx_path, 'w', encoding=encoding) as f:
        f.write(text)


# ── Configuration ────────────────────────────────────────────────────────────

ZMX_SOURCE = os.path.join(_project_dir, 'F_019.zmx')
EFL_TARGET = 16.0
F_NUMBER = 1.3
FOV_HALF = 30.0          # deg

# Penalty weights
W_BFL = 1.0
W_TTL = 0.1
W_CRA = 1.0
W_RI = 10.0              # RI is hardest to satisfy, weight higher

BFL_MIN = 5.0             # mm
TTL_MAX = 40.0            # mm
CRA_MAX = 13.0            # deg at FOV_HALF
RI_MIN = 0.40             # relative, at FOV_HALF

D_EDGE_MIN = 0.3          # mm minimum edge thickness for glass elements
SEIDEL_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0])


# ── Pack / unpack design vector ──────────────────────────────────────────────

def pack(surfaces, gaps):
    """Pack surfaces and gaps into design vector x = [c0..c12, d0..d11]."""
    K = len(surfaces)
    Kd = K - 1
    x = np.zeros(K + Kd)
    for i in range(K):
        x[i] = surfaces[i][0]
    for j in range(Kd):
        x[K + j] = gaps[j][0]
    return x


def unpack(x, surfaces_ref, gaps_ref):
    """Unpack design vector back to surfaces and gaps."""
    K = len(surfaces_ref)
    Kd = K - 1
    surfaces = []
    for i in range(K):
        _, n, np_ = surfaces_ref[i]
        surfaces.append((x[i], n, np_))
    gaps = []
    for j in range(Kd):
        _, n_gap = gaps_ref[j]
        gaps.append((x[K + j], n_gap))
    return surfaces, gaps


def grad_to_x(dM_dc, dM_dd, K):
    """Combine curvature and gap gradients into x-space gradient."""
    return np.concatenate([dM_dc, dM_dd])


# ── Objective ────────────────────────────────────────────────────────────────

def objective(x, surfaces_ref, gaps_ref, stop_info):
    """Returns (merit, gradient)."""
    surfaces, gaps = unpack(x, surfaces_ref, gaps_ref)
    K = len(surfaces)
    nu0_chief = np.tan(np.radians(FOV_HALF))

    r = full_merit_with_grad(
        surfaces, gaps, stop_info, nu0_chief,
        f_number=F_NUMBER,
        weights=SEIDEL_WEIGHTS,
        bfl_target=BFL_MIN, w_bfl=W_BFL,
        ttl_target=TTL_MAX, w_ttl=W_TTL,
        cra_max=CRA_MAX, w_cra=W_CRA, cra_field_deg=FOV_HALF,
        ri_min=RI_MIN, w_ri=W_RI, ri_field_deg=FOV_HALF)

    grad = grad_to_x(r['dM_dc'], r['dM_dd'], K)
    return r['merit'], grad, r


# ── EFL constraint ───────────────────────────────────────────────────────────

def efl_constraint_val(x, surfaces_ref, gaps_ref):
    surfaces, gaps = unpack(x, surfaces_ref, gaps_ref)
    efl, dEFL_dc, dEFL_dd = efl_with_grad(surfaces, gaps)
    return efl - EFL_TARGET


def efl_constraint_jac(x, surfaces_ref, gaps_ref):
    surfaces, gaps = unpack(x, surfaces_ref, gaps_ref)
    _, dEFL_dc, dEFL_dd = efl_with_grad(surfaces, gaps)
    return grad_to_x(dEFL_dc, dEFL_dd, len(surfaces))


# ── Edge thickness constraints ───────────────────────────────────────────────

def edge_val(x, idx, surfaces_ref, gaps_ref):
    surfaces, gaps = unpack(x, surfaces_ref, gaps_ref)
    K = len(surfaces)
    (h_m, nu_m, dh_dc, dnu_dc, dh_dd, dnu_dd,
     efl, sa, dEFL_dc, dEFL_dd) = trace_marginal_fnum(surfaces, gaps, F_NUMBER)
    ET, _, _ = edge_thickness_with_grad(
        surfaces, gaps, h_m, dh_dc, dh_dd, D_EDGE_MIN)
    return ET[idx]


def edge_jac(x, idx, surfaces_ref, gaps_ref):
    surfaces, gaps = unpack(x, surfaces_ref, gaps_ref)
    K = len(surfaces)
    (h_m, nu_m, dh_dc, dnu_dc, dh_dd, dnu_dd,
     efl, sa, dEFL_dc, dEFL_dd) = trace_marginal_fnum(surfaces, gaps, F_NUMBER)
    _, dET_dc, dET_dd = edge_thickness_with_grad(
        surfaces, gaps, h_m, dh_dc, dh_dd, D_EDGE_MIN)
    return grad_to_x(dET_dc[idx], dET_dd[idx], K)


# ── Optimizer ────────────────────────────────────────────────────────────────

def run_optimization(verbose=True):
    # Load from ZMX, normalize, scale to target EFL
    zmx_data_src = parse_zmx(ZMX_SOURCE)
    norm_s, norm_g, norm_stop, efl_phys, meta = zmx_to_normalized(zmx_data_src)
    from parse_zmx import normalized_to_physical
    surfaces_ref, gaps_ref, stop_info = normalized_to_physical(
        norm_s, norm_g, norm_stop, EFL_TARGET)
    surfaces_ref = list(surfaces_ref)
    gaps_ref = list(gaps_ref)
    K = len(surfaces_ref)
    Kd = K - 1

    x0 = pack(surfaces_ref, gaps_ref)

    # ── Bounds ──
    # Wide bounds to allow the optimizer freedom with the new starting point
    bounds = []
    for i in range(K):
        bounds.append((-0.3, 0.3))
    for j in range(Kd):
        d = gaps_ref[j][0]
        n_gap = gaps_ref[j][1]
        if n_gap > 1.001:  # glass
            bounds.append((0.1, max(d * 3, 10.0)))
        else:  # air
            bounds.append((0.05, max(d * 5, 20.0)))

    # ── Constraints ──
    constraints = [
        {'type': 'eq',
         'fun': lambda x: efl_constraint_val(x, surfaces_ref, gaps_ref),
         'jac': lambda x: efl_constraint_jac(x, surfaces_ref, gaps_ref)},
    ]
    for idx in range(Kd):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=idx: edge_val(x, i, surfaces_ref, gaps_ref),
            'jac': lambda x, i=idx: edge_jac(x, i, surfaces_ref, gaps_ref),
        })

    # ── Callback ──
    iteration = [0]
    def callback(x):
        iteration[0] += 1
        if verbose and iteration[0] % 5 == 0:
            _, _, r = objective(x, surfaces_ref, gaps_ref, stop_info)
            cra_s = f"{r['cra']:.1f}" if r['cra'] is not None else "---"
            ri_s = f"{r['ri']:.3f}" if r['ri'] is not None else "---"
            print(f"  iter {iteration[0]:4d}: merit={r['merit']:.4e}"
                  f"  EFL={r['efl']:.2f}  BFL={r['bfl']:.2f}"
                  f"  TTL={r['ttl']:.2f}  CRA={cra_s}  RI={ri_s}")

    # ── Initial state ──
    _, _, r0 = objective(x0, surfaces_ref, gaps_ref, stop_info)
    if verbose:
        print("=" * 70)
        print("L_033 OPTIMIZATION")
        print("=" * 70)
        print(f"  EFL target:  {EFL_TARGET} mm")
        print(f"  f/#:         {F_NUMBER}")
        print(f"  FOV:         {FOV_HALF} deg half")
        print(f"  Variables:   {len(x0)} ({K} curvatures + {Kd} thicknesses)")
        print()
        print(f"  Initial state:")
        print(f"    Merit = {r0['merit']:.6e}")
        print(f"    EFL   = {r0['efl']:.4f} mm")
        print(f"    BFL   = {r0['bfl']:.4f} mm  (min {BFL_MIN})")
        print(f"    TTL   = {r0['ttl']:.4f} mm  (max {TTL_MAX})")
        print(f"    CRA   = {r0['cra']:.2f} deg  (max {CRA_MAX})")
        print(f"    RI    = {r0['ri']:.4f}     (min {RI_MIN})")
        print(f"    Seidel: {r0['S_total']}")
        print()

    # ── Run SLSQP ──
    result = minimize(
        lambda x: objective(x, surfaces_ref, gaps_ref, stop_info)[0],
        x0,
        jac=lambda x: objective(x, surfaces_ref, gaps_ref, stop_info)[1],
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={'maxiter': 1000, 'ftol': 1e-15, 'disp': verbose},
    )

    # ── Report ──
    x_opt = result.x
    surfaces_opt, gaps_opt = unpack(x_opt, surfaces_ref, gaps_ref)
    _, _, r_opt = objective(x_opt, surfaces_ref, gaps_ref, stop_info)

    if verbose:
        print()
        print("=" * 70)
        print("OPTIMIZATION RESULT")
        print("=" * 70)
        print(f"  Success:    {result.success}")
        print(f"  Message:    {result.message}")
        print(f"  Iterations: {result.nit}")
        print()
        print(f"  Merit  = {r_opt['merit']:.6e}")
        print(f"  EFL    = {r_opt['efl']:.4f} mm")
        print(f"  f/#    = {F_NUMBER}")
        print(f"  BFL    = {r_opt['bfl']:.4f} mm  (min {BFL_MIN})")
        print(f"  TTL    = {r_opt['ttl']:.4f} mm  (max {TTL_MAX})")
        print(f"  CRA    = {r_opt['cra']:.2f} deg  (max {CRA_MAX})")
        print(f"  RI     = {r_opt['ri']:.4f}     (min {RI_MIN})")
        print()
        print(f"  Seidel totals:")
        for name, val in zip(['S_I', 'S_II', 'S_III', 'S_IV', 'S_V'],
                             r_opt['S_total']):
            print(f"    {name:5s} = {val:+.8f}")
        print()

        # Edge thicknesses
        (h_m, _, dh_dc, _, dh_dd, _, _, _, _, _) = trace_marginal_fnum(
            surfaces_opt, gaps_opt, F_NUMBER)
        ET, _, _ = edge_thickness_with_grad(
            surfaces_opt, gaps_opt, h_m, dh_dc, dh_dd, 0.0)
        print(f"  Edge thicknesses (min ET for feasibility):")
        for j in range(Kd):
            n_gap = gaps_opt[j][1]
            medium = 'glass' if n_gap > 1.001 else 'air'
            flag = " !!!" if ET[j] < 0 else ""
            print(f"    Gap {j:2d} ({medium:5s}): d={gaps_opt[j][0]:.4f}"
                  f"  ET={ET[j]:.4f}{flag}")
        print()

        # Design changes
        print(f"  Curvature changes:")
        print(f"  {'Surf':>4s} {'Original':>12s} {'Optimized':>12s}"
              f" {'Change%':>10s}")
        for i in range(K):
            c_old = surfaces_ref[i][0]
            c_new = surfaces_opt[i][0]
            pct = ((c_new - c_old) / abs(c_old) * 100
                   if abs(c_old) > 1e-12 else 0)
            print(f"  {i:4d} {c_old:12.6f} {c_new:12.6f} {pct:+10.2f}%")
        print()

        print(f"  Thickness changes:")
        print(f"  {'Gap':>4s} {'Original':>10s} {'Optimized':>10s}"
              f" {'Change':>10s}")
        for j in range(Kd):
            d_old = gaps_ref[j][0]
            d_new = gaps_opt[j][0]
            print(f"  {j:4d} {d_old:10.4f} {d_new:10.4f}"
                  f" {d_new - d_old:+10.4f}")

    # ── Write optimized ZMX ──
    zmx_src = ZMX_SOURCE
    base_name = os.path.splitext(os.path.basename(zmx_src))[0]
    zmx_dst = os.path.join(_project_dir, f'{base_name}_opt.zmx')

    zmx_data = parse_zmx(zmx_src)
    _, _, _, efl_phys_w, _ = zmx_to_normalized(zmx_data)
    z2t, t2z = zmx_to_trace_map(zmx_data)

    # Write at EFL=16mm scale (optimized values are already at this scale).
    # Optical surfaces: curvatures and thicknesses from optimizer directly.
    # Non-optical surfaces (stop/dummy): scale from original ZMX.
    scale_factor = EFL_TARGET / efl_phys_w

    curv_updates = {}
    thick_updates = {}

    # Optical surfaces: curvatures
    for trace_idx in range(K):
        zmx_num = t2z[trace_idx]
        curv_updates[zmx_num] = surfaces_opt[trace_idx][0]

    # Thicknesses: need to handle merged gaps (where dummy surfaces
    # were folded into a single trace gap).
    # For each trace gap j, check if the original ZMX gap was merged
    # with adjacent dummy surfaces.
    surfaces_phys, gaps_phys, _ = build_system_from_zmx(zmx_data)

    for trace_gap_idx in range(Kd):
        zmx_num = t2z[trace_gap_idx]
        zmx_thick_orig = zmx_data['surfaces'][zmx_num]['thickness']
        trace_thick_orig = gaps_phys[trace_gap_idx][0]

        if abs(trace_thick_orig - zmx_thick_orig) < 1e-6:
            # No merging — direct mapping
            thick_updates[zmx_num] = gaps_opt[trace_gap_idx][0]
        else:
            # Merged gap: trace gap includes dummy surface thicknesses.
            # Scale all contributing ZMX surfaces proportionally.
            ratio = gaps_opt[trace_gap_idx][0] / (trace_thick_orig * scale_factor)
            # Find which ZMX surfaces contribute to this merged gap
            zmx_next = t2z[trace_gap_idx + 1]
            for sn in range(zmx_num, zmx_next):
                s = zmx_data['surfaces'][sn]
                if s['thickness'] != float('inf') and s['thickness'] > 0:
                    thick_updates[sn] = s['thickness'] * scale_factor * ratio

    # Last optical surface → image distance = BFL
    last_zmx_surf = t2z[K - 1]
    thick_updates[last_zmx_surf] = r_opt['bfl']

    # ── Compute clear apertures: as large as physically possible ──
    # For each gap, find max h where exact edge thickness >= 0.
    # Each surface's CA = min of its two adjacent gap limits.
    # Uses exact spherical sag (not paraxial approximation).
    sa_opt = EFL_TARGET / (2 * F_NUMBER)
    h_m_opt, _ = ynu_trace(surfaces_opt, gaps_opt, sa_opt, 0.0)

    def exact_et(c_f, c_b, d, h):
        """Exact edge thickness at height h."""
        sf, _, _, _, _ = sag_derivs(c_f, h)
        sb, _, _, _, _ = sag_derivs(c_b, h)
        return d - sf + sb

    LARGE_H = 20.0  # mm, generous upper bound

    def max_h_for_gap(c_f, c_b, d):
        """Find max h where ET >= 0 using bisection with exact sag.
        Also limited by sphere radius of each surface (h < 1/|c|)."""
        if d <= 0:
            return 0.0
        # h can't exceed the sphere radius of either surface
        h_sphere = LARGE_H
        if abs(c_f) > 1e-10:
            h_sphere = min(h_sphere, 0.999 / abs(c_f))
        if abs(c_b) > 1e-10:
            h_sphere = min(h_sphere, 0.999 / abs(c_b))

        # Check if ET is positive at the sphere limit — no geometric limit
        if exact_et(c_f, c_b, d, h_sphere) >= 0:
            return h_sphere
        # Bisect between 0 (ET=d>0) and h_sphere (ET<0)
        h_lo = 0.0
        h_hi = h_sphere
        for _ in range(60):
            h_mid = (h_lo + h_hi) / 2
            if exact_et(c_f, c_b, d, h_mid) >= 0:
                h_lo = h_mid
            else:
                h_hi = h_mid
        return h_lo

    h_max_gap = np.zeros(Kd)
    for j in range(Kd):
        c_f = surfaces_opt[j][0]
        c_b = surfaces_opt[j + 1][0]
        d = gaps_opt[j][0]
        h_max_gap[j] = max_h_for_gap(c_f, c_b, d)

    # Each surface's CA must satisfy ALL adjacent gaps.
    # For gap j, both surface j and j+1 must have CA <= h_max_gap[j].
    # So CA(i) = min over all gaps that surface i borders.
    max_ca_limit = np.max(np.abs(h_m_opt)) * 3.0  # cap for infinite cases
    ca = np.full(K, max_ca_limit)
    for j in range(Kd):
        limit = h_max_gap[j]
        ca[j] = min(ca[j], limit)
        ca[j + 1] = min(ca[j + 1], limit)
    ca = ca * 0.95

    # Semi-diameter updates for ZMX (map trace surfaces to ZMX surfaces)
    semi_dia_updates = {}
    for trace_idx in range(K):
        zmx_num = t2z[trace_idx]
        semi_dia_updates[zmx_num] = ca[trace_idx]

    # Non-optical surfaces: set CA appropriately
    optical_zmx_nums = set(t2z[ti] for ti in range(K))
    stop_zmx_num = zmx_data['stop_index']

    for s in zmx_data['surfaces']:
        sn = s['num']
        if sn in optical_zmx_nums:
            continue
        if sn == 0:
            continue  # object surface

        if sn == stop_zmx_num:
            # Stop surface: set to 0 (let Zemax determine from ENPD)
            semi_dia_updates[sn] = 0
        elif sn < len(zmx_data['surfaces']) - 1:
            # Other dummy surfaces: use max of adjacent optical CAs
            adj_cas = []
            if sn - 1 in semi_dia_updates:
                adj_cas.append(semi_dia_updates[sn - 1])
            if sn + 1 in semi_dia_updates:
                adj_cas.append(semi_dia_updates[sn + 1])
            if adj_cas:
                semi_dia_updates[sn] = max(adj_cas)
        else:
            # Image surface
            semi_dia_updates[sn] = ca[-1]

    write_modified_zmx(zmx_src, zmx_dst,
                       curvature_updates=curv_updates,
                       thickness_updates=thick_updates,
                       semi_dia_updates=semi_dia_updates)

    # Update ENPD, PUPD, and field values
    _update_zmx_settings(zmx_dst, EFL_TARGET / F_NUMBER, fov_half=FOV_HALF)

    if verbose:
        print()
        print(f"  Clear apertures (semi-diameter):")
        for trace_idx in range(K):
            zmx_num = t2z[trace_idx]
            print(f"    S{zmx_num:2d}: h_m={abs(h_m_opt[trace_idx]):.4f}"
                  f"  CA={ca[trace_idx]:.4f}")
        print()
        print(f"  Written optimized ZMX to: {zmx_dst}")

    return result, surfaces_opt, gaps_opt


if __name__ == '__main__':
    run_optimization()
