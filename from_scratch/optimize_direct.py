"""
Direct optimizer for thick-lens systems from sweep results.

Takes a fully defined system (14 surfaces, glass thicknesses, air gaps, n_d per element)
and optimizes all variables simultaneously with L-BFGS-B.

Design vector:
    x = [c1, c2, ..., c14,           # 14 curvatures
         d_glass_1, ..., d_glass_7,   # 7 glass thicknesses
         d_air_1, ..., d_air_6,       # 6 air gaps
         n_d_1, ..., n_d_7]           # 7 refractive indices
"""

import sys
import os
import json
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parse_zmx import ynu_trace, find_chief_ray_initial
from gradient import (full_merit_with_grad, edge_thickness_with_grad,
                      _ri_forward)


def _merit_forward(surfaces, gaps, params, w_efl):
    """Lightweight merit evaluation — no gradients, for finite-diff only."""
    from parse_zmx import seidel_coefficients
    h_u, nu_u = ynu_trace(surfaces, gaps, 1.0, 0.0)
    if abs(nu_u[-1]) < 1e-12:
        return 1e12
    efl = -1.0 / nu_u[-1]
    if efl < 0:
        return 1e12
    sa = efl / (2 * params['f_number'])
    h_m, nu_m = ynu_trace(surfaces, gaps, sa, 0.0)

    nu0_c = params['nu0_chief']
    field_deg = np.degrees(np.arctan(nu0_c))
    h0_c, nu0_c_val = find_chief_ray_initial(
        surfaces, gaps, params['stop_idx'], field_deg)
    h_c, nu_c = ynu_trace(surfaces, gaps, h0_c, nu0_c_val)

    S_I, S_II, S_III, S_IV, S_V = seidel_coefficients(
        surfaces, h_m, nu_m, h_c, nu_c, 0.0, nu0_c)
    S = np.array([S_I.sum(), S_II.sum(), S_III.sum(), S_IV.sum(), S_V.sum()])
    merit = float(np.sum(params['weights'] * S ** 2))

    # EFL penalty
    merit += w_efl * (efl - params['efl_target']) ** 2

    # BFL penalty
    bfl = -h_m[-1] / nu_m[-1] if abs(nu_m[-1]) > 1e-12 else 0.0
    bt = params.get('bfl_target')
    if bt is not None and params.get('w_bfl', 0) > 0:
        v = bt - bfl
        if v > 0:
            merit += params['w_bfl'] * v ** 2

    # TTL penalty
    ttl = sum(g[0] for g in gaps) + bfl
    tt = params.get('ttl_target')
    if tt is not None and params.get('w_ttl', 0) > 0:
        v = ttl - tt
        if v > 0:
            merit += params['w_ttl'] * v ** 2

    # CRA penalty
    cm = params.get('cra_max')
    if cm is not None and params.get('w_cra', 0) > 0:
        cra = np.degrees(np.arctan(nu_c[-1]))
        v = abs(cra) - cm
        if v > 0:
            merit += params['w_cra'] * v ** 2

    # RI penalty
    rm = params.get('ri_min')
    if rm is not None and params.get('w_ri', 0) > 0:
        ri = _ri_forward(surfaces, gaps, params['stop_idx'],
                         field_deg, params['f_number'])
        v = rm - ri
        if v > 0:
            merit += params['w_ri'] * v ** 2

    return merit


def build_system(x, n_elements):
    """Unpack x into surfaces and gaps."""
    K = 2 * n_elements
    n_gaps = K - 1

    curvatures = x[:K]
    d_glass = x[K:K + n_elements]
    d_air = x[K + n_elements:K + n_elements + (n_elements - 1)]
    n_d = x[K + n_elements + (n_elements - 1):]

    surfaces = []
    gaps = []
    ai = 0
    for i in range(n_elements):
        surfaces.append((curvatures[2*i], 1.0, n_d[i]))
        surfaces.append((curvatures[2*i+1], n_d[i], 1.0))
        gaps.append((d_glass[i], n_d[i]))
        if i < n_elements - 1:
            gaps.append((d_air[ai], 1.0))
            ai += 1

    return surfaces, gaps


def pack_x(c1_list, c2_list, d_glass_list, d_air_list, n_d_list):
    """Pack all variables into design vector."""
    curvatures = []
    for c1, c2 in zip(c1_list, c2_list):
        curvatures.append(c1)
        curvatures.append(c2)
    return np.array(curvatures + list(d_glass_list) +
                    list(d_air_list) + list(n_d_list))


def unpack_x(x, n_elements):
    """Unpack design vector into components."""
    K = 2 * n_elements
    curvatures = x[:K]
    d_glass = x[K:K + n_elements]
    d_air = x[K + n_elements:K + n_elements + (n_elements - 1)]
    n_d = x[K + n_elements + (n_elements - 1):]
    c1_list = curvatures[0::2]
    c2_list = curvatures[1::2]
    return c1_list, c2_list, d_glass, d_air, n_d


def build_bounds(n_elements, c_bounds=(-0.3, 0.3),
                 d_glass_bounds=(0.5, 15.0),
                 d_air_bounds=(0.3, 15.0),
                 n_d_bounds=(1.45, 1.90)):
    """Build bounds for all variables."""
    K = 2 * n_elements
    bounds = []
    # Curvatures
    for _ in range(K):
        bounds.append(c_bounds)
    # Glass thicknesses
    for _ in range(n_elements):
        bounds.append(d_glass_bounds)
    # Air gaps
    for _ in range(n_elements - 1):
        bounds.append(d_air_bounds)
    # Refractive indices
    for _ in range(n_elements):
        bounds.append(n_d_bounds)
    return bounds


def objective(x, n_elements, params):
    """
    Combined merit function with analytic gradients for curvatures/gaps
    and finite-difference gradients for n_d.
    """
    K = 2 * n_elements
    n_gaps = K - 1

    surfaces, gaps = build_system(x, n_elements)

    try:
        result = full_merit_with_grad(
            surfaces, gaps,
            stop_info=params['stop_idx'],
            nu0_chief=params['nu0_chief'],
            f_number=params['f_number'],
            weights=params['weights'],
            bfl_target=params.get('bfl_target'),
            w_bfl=params.get('w_bfl', 0.0),
            ttl_target=params.get('ttl_target'),
            w_ttl=params.get('w_ttl', 0.0),
            cra_max=params.get('cra_max'),
            w_cra=params.get('w_cra', 0.0),
            cra_field_deg=params.get('cra_field_deg'),
            ri_min=params.get('ri_min'),
            w_ri=params.get('w_ri', 0.0),
            ri_field_deg=params.get('ri_field_deg'),
        )
    except Exception:
        return 1e12, np.zeros_like(x), None

    merit = result['merit']
    dM_dc = result['dM_dc']  # (K,) gradient w.r.t. curvatures
    dM_dd = result['dM_dd']  # (K-1,) gradient w.r.t. gap thicknesses

    # EFL penalty
    efl = result['efl']
    efl_err = efl - params['efl_target']
    w_efl = params['w_efl']
    merit += w_efl * efl_err ** 2
    dEFL_dc = result['dEFL_dc']
    dEFL_dd = result['dEFL_dd']

    # Edge thickness penalty
    w_et = params.get('w_et', 0.0)
    d_edge_min = params.get('d_edge_min', 0.2)
    if w_et > 0:
        ET, dET_dc, dET_dd = edge_thickness_with_grad(
            surfaces, gaps, result['h_m'],
            result['dh_m_dc'], result['dh_m_dd'], d_edge_min)
        for idx in range(n_gaps):
            if ET[idx] < 0:
                merit += w_et * ET[idx] ** 2
                dM_dc += 2 * w_et * ET[idx] * dET_dc[idx]
                dM_dd += 2 * w_et * ET[idx] * dET_dd[idx]

    # Add EFL gradient
    dM_dc += 2 * w_efl * efl_err * dEFL_dc
    dM_dd += 2 * w_efl * efl_err * dEFL_dd

    # Map gap gradients to x-space
    # Gap layout: glass_0, air_0, glass_1, air_1, ...
    grad = np.zeros_like(x)

    # Curvature gradients
    grad[:K] = dM_dc

    # Glass thickness gradients
    for i in range(n_elements):
        gap_idx = 2 * i  # glass gap for element i
        grad[K + i] = dM_dd[gap_idx]

    # Air gap gradients
    for i in range(n_elements - 1):
        gap_idx = 2 * i + 1  # air gap after element i
        grad[K + n_elements + i] = dM_dd[gap_idx]

    # n_d gradients via finite difference (lightweight forward-only eval)
    eps = 1e-6
    nd_offset = K + n_elements + (n_elements - 1)
    for i in range(n_elements):
        x_p = x.copy()
        x_p[nd_offset + i] += eps
        s_p, g_p = build_system(x_p, n_elements)
        try:
            m_p = _merit_forward(s_p, g_p, params, w_efl)
            grad[nd_offset + i] = (m_p - merit) / eps
        except Exception:
            grad[nd_offset + i] = 0.0

    # Build result info
    info = {
        'efl': efl, 'bfl': result['bfl'], 'ttl': result['ttl'],
        'cra': result['cra'], 'ri': result['ri'],
        'S_total': result['S_total'],
    }

    return merit, grad, info


def optimize_direct(n_elements, efl_target, f_number, field_angle_deg,
                    c1_list, c2_list, d_glass_list, d_air_list, n_d_list,
                    stop_idx,
                    weights=None,
                    c_bounds=(-0.3, 0.3),
                    d_glass_bounds=(0.5, 15.0),
                    d_air_bounds=(0.3, 15.0),
                    n_d_bounds=(1.45, 1.90),
                    d_edge_min=0.2,
                    w_efl=1000.0, w_et=100.0,
                    bfl_target=None, w_bfl=0.0,
                    ttl_target=None, w_ttl=0.0,
                    cra_max=None, w_cra=0.0,
                    ri_min=None, w_ri=0.0,
                    maxiter=1000, verbose=True):
    """
    Direct L-BFGS-B optimization of a thick-lens system.
    """
    if weights is None:
        weights = np.ones(5)

    params = {
        'stop_idx': stop_idx,
        'nu0_chief': np.tan(np.radians(field_angle_deg)),
        'f_number': f_number,
        'weights': weights,
        'efl_target': efl_target,
        'w_efl': w_efl,
        'w_et': w_et,
        'd_edge_min': d_edge_min,
        'bfl_target': bfl_target,
        'w_bfl': w_bfl,
        'ttl_target': ttl_target,
        'w_ttl': w_ttl,
        'cra_max': cra_max,
        'w_cra': w_cra,
        'cra_field_deg': field_angle_deg if cra_max is not None else None,
        'ri_min': ri_min,
        'w_ri': w_ri,
        'ri_field_deg': field_angle_deg if ri_min is not None else None,
    }

    x = pack_x(c1_list, c2_list, d_glass_list, d_air_list, n_d_list)
    bounds = build_bounds(n_elements, c_bounds, d_glass_bounds,
                          d_air_bounds, n_d_bounds)

    if verbose:
        m0, _, info0 = objective(x, n_elements, params)
        print(f"Initial: merit={m0:.4e}  EFL={info0['efl']:.2f}  "
              f"BFL={info0['bfl']:.2f}  TTL={info0['ttl']:.2f}  "
              f"CRA={info0['cra']:.1f}  RI={info0['ri']:.3f}")
        S = info0['S_total']
        print(f"  S_I={S[0]:+.4f}  S_II={S[1]:+.4f}  S_III={S[2]:+.4f}  "
              f"S_IV={S[3]:+.4f}  S_V={S[4]:+.4f}")
        print()

    last = [None]  # store last evaluation result

    def obj_and_grad(xv):
        m, g, info = objective(xv, n_elements, params)
        last[0] = (m, info)
        return m, g

    iteration = [0]
    def callback(xv):
        iteration[0] += 1
        if verbose and iteration[0] % 20 == 0 and last[0] is not None:
            m, info = last[0]
            if info:
                ri_str = f"{info['ri']:.3f}" if info['ri'] is not None else "n/a"
                print(f"  iter {iteration[0]:4d}: merit={m:.4e}  "
                      f"EFL={info['efl']:.2f}  BFL={info['bfl']:.1f}  "
                      f"TTL={info['ttl']:.1f}  CRA={info['cra']:.1f}  "
                      f"RI={ri_str}", flush=True)

    result = minimize(
        obj_and_grad, x,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        callback=callback,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10},
    )

    x_opt = result.x
    m_final, _, info_final = objective(x_opt, n_elements, params)
    surfaces, gaps = build_system(x_opt, n_elements)
    c1, c2, d_glass, d_air, n_d = unpack_x(x_opt, n_elements)

    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL RESULT  (iters={result.nit}, ok={result.success})")
        print(f"{'='*60}")
        print(f"  Merit = {m_final:.6e}")
        print(f"  EFL = {info_final['efl']:.4f} mm (target {efl_target})")
        print(f"  BFL = {info_final['bfl']:.2f} mm")
        print(f"  TTL = {info_final['ttl']:.2f} mm")
        print(f"  CRA = {info_final['cra']:.2f} deg")
        if info_final['ri'] is not None:
            print(f"  RI  = {info_final['ri']:.4f}")
        print()

        S = info_final['S_total']
        labels = ['spherical', 'coma', 'astigmatism', 'Petzval', 'distortion']
        for i, lbl in enumerate(labels):
            print(f"  S_{['I','II','III','IV','V'][i]:>3s} = {S[i]:+.8f}  ({lbl})")
        print()

        for i in range(n_elements):
            r_f = 1/c1[i] if abs(c1[i]) > 1e-12 else float('inf')
            r_b = 1/c2[i] if abs(c2[i]) > 1e-12 else float('inf')
            print(f"  L{i+1}: R_f={r_f:+8.2f}  R_b={r_b:+8.2f}  "
                  f"d={d_glass[i]:.3f}  n_d={n_d[i]:.4f}")
        print()
        for i in range(n_elements - 1):
            print(f"  Air gap {i+1}-{i+2}: {d_air[i]:.3f} mm")

    # Export to JSON
    outpath = os.path.join(os.path.dirname(__file__),
                           f"optimized_{''.join('+' if p > 0 else '-' for p in c1_list)}.json")
    S = info_final['S_total']
    data = {
        'n_elements': n_elements,
        'efl_target': efl_target,
        'f_number': f_number,
        'field_angle_deg': field_angle_deg,
        'stop_idx': stop_idx,
        'result': {
            'merit': float(m_final),
            'efl': float(info_final['efl']),
            'bfl': float(info_final['bfl']),
            'ttl': float(info_final['ttl']),
            'cra': float(info_final['cra']),
            'ri': float(info_final['ri']) if info_final['ri'] is not None else None,
            'seidel': {
                'S_I': float(S[0]), 'S_II': float(S[1]),
                'S_III': float(S[2]), 'S_IV': float(S[3]),
                'S_V': float(S[4]),
            },
        },
        'elements': [],
        'air_gaps': [float(d) for d in d_air],
    }
    for i in range(n_elements):
        r_f = 1/c1[i] if abs(c1[i]) > 1e-12 else float('inf')
        r_b = 1/c2[i] if abs(c2[i]) > 1e-12 else float('inf')
        data['elements'].append({
            'label': f'L{i+1}',
            'c_front': float(c1[i]),
            'c_back': float(c2[i]),
            'R_front': float(r_f),
            'R_back': float(r_b),
            'd_glass': float(d_glass[i]),
            'n_d': float(n_d[i]),
        })

    with open(outpath, 'w') as f:
        json.dump(data, f, indent=2)
    if verbose:
        print(f"\nExported: {outpath}")

    return x_opt, surfaces, gaps, info_final


def main():
    # Sweep result #2 [----+++] from sweep_top20.md
    c1_list = [+0.001065, -0.016955, +0.004432, -0.008114,
               +0.032755, -0.005061, +0.040962]
    c2_list = [+0.013793, -0.001196, +0.031685, +0.001066,
               -0.013251, -0.036399, -0.016623]
    d_glass_list = [1.0, 1.0, 1.0, 1.0, 2.89, 2.47, 4.25]
    d_air_list = [1.33, 1.33, 1.33, 5.33, 1.33, 1.33]
    n_d_list = [1.7, 1.7, 1.7, 1.7, 1.8, 1.8, 1.8]

    x_opt, surfs, gaps, info = optimize_direct(
        n_elements=7,
        efl_target=16.0,
        f_number=1.2,
        field_angle_deg=30.0,
        c1_list=c1_list,
        c2_list=c2_list,
        d_glass_list=d_glass_list,
        d_air_list=d_air_list,
        n_d_list=n_d_list,
        stop_idx=7,
        c_bounds=(-0.3, 0.3),
        d_glass_bounds=(0.5, 15.0),
        d_air_bounds=(0.3, 15.0),
        n_d_bounds=(1.45, 1.90),
        d_edge_min=0.3,
        w_efl=1000.0,
        w_et=100.0,
        bfl_target=5.0, w_bfl=10.0,
        ttl_target=40.0, w_ttl=10.0,
        cra_max=13.0, w_cra=5.0,
        ri_min=0.40, w_ri=50.0,
        maxiter=1000,
        verbose=True,
    )


if __name__ == "__main__":
    main()
