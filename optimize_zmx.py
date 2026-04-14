"""
Optimize the sc_dbga1_opt2.zmx system to a new EFL target.

Pulls correct glass indices from Zemax, optimizes curvatures using SLSQP
with analytic gradients, writes result to a new ZMX file.
"""

import os
import numpy as np
from scipy.optimize import minimize
from parse_zmx import (parse_zmx, build_system_from_zmx, write_modified_zmx,
                        zmx_to_trace_map, ynu_trace, _ray_height_at_stop)
from gradient import (ynu_trace_with_grad, merit_with_grad, merit_with_grad_fnum,
                      efl_with_grad, edge_thickness_with_grad,
                      trace_marginal_fnum, full_merit_with_grad)


# ── Zemax index pulling ─────────────────────────────────────────────────────

def get_zemax_indices(zmx_file):
    """Pull actual glass indices from Zemax catalog via ZOS-API."""
    import zospy as zp
    from compare_with_zemax import get_refractive_indices_from_zemax

    zos = zp.ZOS()
    oss = zos.connect(mode='standalone')
    oss.load(os.path.abspath(zmx_file))
    indices = get_refractive_indices_from_zemax(oss)
    zos.disconnect()
    return indices


def load_system(zmx_file, index_overrides=None):
    """Parse ZMX and optionally apply corrected glass indices."""
    zmx_data = parse_zmx(zmx_file)
    if index_overrides:
        for s in zmx_data['surfaces']:
            if s['num'] in index_overrides:
                s['n_d'] = index_overrides[s['num']]['n']
    surfaces, gaps, stop_info = build_system_from_zmx(zmx_data)
    return zmx_data, surfaces, gaps, stop_info


# ── General optimizer ────────────────────────────────────────────────────────

def optimize_general(surfaces, gaps, stop_info, nu0_chief,
                     efl_target, free_c, c_bounds,
                     semi_aperture=None, f_number=None,
                     weights=None, d_edge_min=0.5,
                     bfl_target=None, w_bfl=1.0,
                     ttl_target=None, w_ttl=1.0,
                     verbose=True):
    """
    Optimize a general optical system by varying curvatures.

    Supports fixed semi_aperture or f-number mode (specify exactly one).
    Optional BFL and TTL soft penalties in the merit function.
    """
    if (semi_aperture is None) == (f_number is None):
        raise ValueError("Specify exactly one of semi_aperture or f_number")

    use_fnum = f_number is not None
    K = len(surfaces)
    Kd = K - 1
    N = len(free_c)

    if weights is None:
        weights = np.ones(5)

    # Merit kwargs for full_merit_with_grad
    merit_kw = dict(
        weights=weights,
        bfl_target=bfl_target, w_bfl=w_bfl,
        ttl_target=ttl_target, w_ttl=w_ttl,
    )
    if use_fnum:
        merit_kw['f_number'] = f_number
    else:
        merit_kw['semi_aperture'] = semi_aperture

    def unpack(x):
        surfs = list(surfaces)
        for i, ci in enumerate(free_c):
            _, n, np_ = surfaces[ci]
            surfs[ci] = (x[i], n, np_)
        return surfs

    def grad_to_x(g_full):
        return np.array([g_full[ci] for ci in free_c])

    x0 = np.array([surfaces[ci][0] for ci in free_c])

    # ── Objective (Seidel + BFL + CRA) ──
    def _eval(x):
        surfs = unpack(x)
        return full_merit_with_grad(surfs, gaps, stop_info, nu0_chief, **merit_kw)

    def obj_val(x):
        return _eval(x)['merit']

    def obj_jac(x):
        return grad_to_x(_eval(x)['dM_dc'])

    # ── EFL constraint ──
    def efl_val(x):
        surfs = unpack(x)
        efl, _, _ = efl_with_grad(surfs, gaps)
        return efl - efl_target

    def efl_jac(x):
        surfs = unpack(x)
        _, dE_dc, _ = efl_with_grad(surfs, gaps)
        return grad_to_x(dE_dc)

    # ── Edge thickness constraints ──
    def _get_marginal(surfs):
        if use_fnum:
            (h_m, nu_m, dh_dc, dnu_dc, dh_dd, dnu_dd,
             efl, sa, dEFL_dc, dEFL_dd) = trace_marginal_fnum(
                surfs, gaps, f_number)
        else:
            h_m, nu_m, dh_dc, dnu_dc, dh_dd, dnu_dd = ynu_trace_with_grad(
                surfs, gaps, semi_aperture, 0.0)
        return h_m, dh_dc, dh_dd

    def edge_val(x, idx):
        surfs = unpack(x)
        h_m, dh_dc, dh_dd = _get_marginal(surfs)
        ET, _, _ = edge_thickness_with_grad(surfs, gaps, h_m, dh_dc, dh_dd, d_edge_min)
        return ET[idx]

    def edge_jac(x, idx):
        surfs = unpack(x)
        h_m, dh_dc, dh_dd = _get_marginal(surfs)
        _, dET_dc, _ = edge_thickness_with_grad(surfs, gaps, h_m, dh_dc, dh_dd, d_edge_min)
        return grad_to_x(dET_dc[idx])

    bounds = [c_bounds[ci] for ci in free_c]

    constraints = [
        {'type': 'eq', 'fun': efl_val, 'jac': efl_jac},
    ]
    for idx in range(Kd):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=idx: edge_val(x, i),
            'jac': lambda x, i=idx: edge_jac(x, i),
        })

    # ── Callback ──
    iteration = [0]
    def callback(x):
        iteration[0] += 1
        if verbose and iteration[0] % 10 == 0:
            r = _eval(x)
            print(f"  iter {iteration[0]:4d}: merit={r['merit']:.6e}"
                  f"  EFL={r['efl']:.2f}  BFL={r['bfl']:.2f}  TTL={r['ttl']:.2f}")

    if verbose:
        r0 = _eval(x0)
        print(f"\nStarting optimization...")
        print(f"  EFL target:    {efl_target} mm (current: {r0['efl']:.4f})")
        if use_fnum:
            print(f"  f-number:      {f_number}")
            print(f"  Semi-aperture: {r0['sa']:.4f} mm (design-dependent)")
        else:
            print(f"  Semi-aperture: {semi_aperture} mm (fixed)")
        if bfl_target is not None:
            print(f"  BFL target:    {bfl_target} mm (w={w_bfl}), current: {r0['bfl']:.4f}")
        if ttl_target is not None:
            print(f"  TTL target:    {ttl_target} mm (w={w_ttl}), current: {r0['ttl']:.4f}")
        print(f"  Design vars:   {N} curvatures")
        print(f"  Initial merit: {r0['merit']:.6e}")
        print()

    result = minimize(
        obj_val, x0, jac=obj_jac, method='SLSQP',
        bounds=bounds, constraints=constraints, callback=callback,
        options={'maxiter': 500, 'ftol': 1e-15, 'disp': verbose},
    )

    opt_surfaces = unpack(result.x)
    return result, opt_surfaces


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ZMX_FILE = 'sc_dbga1_opt2.zmx'
    EFL_TARGET = 75.0       # keep original EFL
    F_NUMBER = 4.0           # target f-number (was f/3)
    BFL_TARGET = 50.0        # target BFL (mm), original ~50.5
    W_BFL = 0.1              # BFL penalty weight
    TTL_TARGET = 100.0       # target TTL (mm), original ~107.7
    W_TTL = 0.01             # TTL penalty weight
    OUTPUT_FILE = 'sc_dbga1_opt2_full.zmx'

    # ── 1. Pull glass indices from Zemax ──
    print("Pulling glass indices from Zemax...")
    indices = get_zemax_indices(ZMX_FILE)
    for snum, info in sorted(indices.items()):
        print(f"  Surf {snum} ({info['material']}): n = {info['n']:.10f}")

    # ── 2. Load system with corrected indices ──
    zmx_data, surfaces, gaps, stop_info = load_system(ZMX_FILE, indices)
    K = len(surfaces)

    # Compute field angle from max image height
    max_field = max(zmx_data['fields_y'])
    efl_current, _, _ = efl_with_grad(surfaces, gaps)
    field_angle = np.degrees(np.arctan(max_field / efl_current))
    nu0_chief = np.tan(np.radians(field_angle))
    sa_current = zmx_data['enpd'] / 2.0
    fnum_current = efl_current / (2 * sa_current)
    sa_target = EFL_TARGET / (2 * F_NUMBER)

    print(f"\nSystem: {K} surfaces, stop = {stop_info}")
    print(f"  EFL      = {efl_current:.4f} mm  (target: {EFL_TARGET})")
    print(f"  f/#      = {fnum_current:.4f}      (target: {F_NUMBER})")
    print(f"  ENPD     = {zmx_data['enpd']} mm   (new: {EFL_TARGET/F_NUMBER:.2f})")
    print(f"  Semi-ap  = {sa_current:.4f} mm   (new: {sa_target:.4f})")
    print(f"  Field    = {field_angle:.4f} deg")

    # ── 3. Set up bounds: ±50% or ±0.01 (generous for f/# change) ──
    free_c = list(range(K))
    c_bounds = {}
    for i in range(K):
        c = surfaces[i][0]
        margin = max(abs(c) * 0.5, 0.01)
        c_bounds[i] = (c - margin, c + margin)

    print(f"\nBounds (50% or ±0.01):")
    for i in range(K):
        c = surfaces[i][0]
        lo, hi = c_bounds[i]
        print(f"  c[{i:2d}] = {c:+.8f}  bounds=[{lo:+.8f}, {hi:+.8f}]")

    # ── 4. Optimize with f-number mode + BFL/CRA penalties ──
    result, opt_surfaces = optimize_general(
        surfaces, gaps, stop_info, nu0_chief,
        EFL_TARGET, free_c, c_bounds,
        f_number=F_NUMBER,
        bfl_target=BFL_TARGET, w_bfl=W_BFL,
        ttl_target=TTL_TARGET, w_ttl=W_TTL)

    # ── 5. Report ──
    r_opt = full_merit_with_grad(
        opt_surfaces, gaps, stop_info, nu0_chief,
        f_number=F_NUMBER,
        bfl_target=BFL_TARGET, w_bfl=W_BFL,
        ttl_target=TTL_TARGET, w_ttl=W_TTL)

    efl_opt = r_opt['efl']
    sa_opt = r_opt['sa']
    S_opt = r_opt['S_total']

    print(f"\n{'=' * 60}")
    print(f"OPTIMIZATION RESULT")
    print(f"{'=' * 60}")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Iterations: {result.nit}")
    print()
    print(f"  EFL       = {efl_opt:.4f} mm (target {EFL_TARGET})")
    print(f"  f/#       = {F_NUMBER}")
    print(f"  Semi-ap   = {sa_opt:.4f} mm (was {sa_current:.4f})")
    print(f"  ENPD      = {2*sa_opt:.4f} mm (was {zmx_data['enpd']})")
    print(f"  BFL       = {r_opt['bfl']:.4f} mm (target {BFL_TARGET})")
    print(f"  TTL       = {r_opt['ttl']:.4f} mm (target {TTL_TARGET})")
    print(f"  Merit     = {result.fun:.6e}")
    print()

    print(f"  Seidel totals:")
    for name, val in zip(['S_I', 'S_II', 'S_III', 'S_IV', 'S_V'], S_opt):
        print(f"    {name:5s} = {val:+.8f}")
    print()

    print(f"  Ray heights at f/{F_NUMBER}:")
    for i in range(K):
        print(f"    Surf {i:2d}: h = {r_opt['h_m'][i]:.4f} mm")
    print()

    print(f"  Curvature changes:")
    print(f"  {'Surf':>4s} {'Original':>14s} {'Optimized':>14s} {'Change%':>10s} {'At bound?':>10s}")
    for i in range(K):
        c_old = surfaces[i][0]
        c_new = opt_surfaces[i][0]
        lo, hi = c_bounds[i]
        pct = (c_new - c_old) / abs(c_old) * 100 if abs(c_old) > 1e-12 else 0
        at_bound = 'LO' if abs(c_new - lo) < 1e-10 else ('HI' if abs(c_new - hi) < 1e-10 else '')
        print(f"  {i:4d} {c_old:14.8f} {c_new:14.8f} {pct:+10.2f}% {at_bound:>10s}")

    # ── 6. Write new ZMX ──
    _, t2z = zmx_to_trace_map(zmx_data)
    curv_updates = {}
    for trace_idx in range(K):
        zmx_num = t2z[trace_idx]
        curv_updates[zmx_num] = opt_surfaces[trace_idx][0]

    write_modified_zmx(ZMX_FILE, OUTPUT_FILE, curvature_updates=curv_updates)
    print(f"\nWritten optimized system to: {OUTPUT_FILE}")

    # ── 7. Verify with re-parse ──
    print(f"\nVerifying {OUTPUT_FILE}...")
    _, surfs_check, gaps_check, _ = load_system(OUTPUT_FILE, indices)
    efl_check, _, _ = efl_with_grad(surfs_check, gaps_check)
    print(f"  Re-parsed EFL = {efl_check:.4f} mm")
    max_curv_err = max(abs(surfs_check[i][0] - opt_surfaces[i][0]) for i in range(K))
    print(f"  Max curvature error: {max_curv_err:.2e}")

    # ── 8. Verify with Zemax ZOS-API ──
    print(f"\nVerifying with Zemax...")
    import zospy as zp
    from zospy.analyses.reports import CardinalPoints

    zos = zp.ZOS()
    oss = zos.connect(mode='standalone')
    oss.load(os.path.abspath(OUTPUT_FILE))

    for wl_num in [1, 2, 3]:
        cp = CardinalPoints(wavelength=wl_num)
        r = cp.run(oss)
        print(f"  WL {wl_num} ({r.data.wavelength:.4f} um): EFL = {r.data.cardinal_points.focal_length.image:.4f} mm")

    # Verify curvatures
    lde = oss.LDE
    mismatch = False
    for zmx_num, expected_c in curv_updates.items():
        row = lde.GetSurfaceAt(zmx_num)
        actual_r = row.Radius
        actual_c = 1/actual_r if abs(actual_r) > 1e-6 and abs(actual_r) < 1e15 else 0.0
        if abs(actual_c - expected_c) / max(abs(expected_c), 1e-15) > 0.001:
            print(f"  MISMATCH surf {zmx_num}: expected c={expected_c:.10f}, got c={actual_c:.10f}")
            mismatch = True
    if not mismatch:
        print("  All curvatures match.")

    zos.disconnect()


if __name__ == '__main__':
    main()
