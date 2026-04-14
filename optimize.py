"""
Gradient-based optimization of a 2-singlet lens system using SLSQP.

Design variables: c1, c2, c3, c4 (curvatures), d_air (air gap)
Constraint: EFL = target (equality)
Bounds: curvature limits, d_air > 0, edge thickness > 0

Uses analytic gradients from gradient.py.
f-number input: semi_aperture = EFL(x) / (2 * f_number), design-dependent.
Edge thickness uses actual marginal ray heights with analytic gradients.
"""

import numpy as np
from scipy.optimize import minimize
from parse_zmx import ynu_trace
from gradient import (merit_with_grad_fnum, efl_with_grad,
                      trace_marginal_fnum, edge_thickness_with_grad)


# ── System builder ───────────────────────────────────────────────────────────

def build_system(x, n1=1.5, n2=1.5, d1=5.0, d2=5.0):
    """
    Build surfaces and gaps from design vector x = [c1, c2, c3, c4, d_air].
    """
    c1, c2, c3, c4, d_air = x
    n_air = 1.0
    surfaces = [
        (c1, n_air, n1),
        (c2, n1, n_air),
        (c3, n_air, n2),
        (c4, n2, n_air),
    ]
    gaps = [
        (d1, n1),
        (d_air, n_air),
        (d2, n2),
    ]
    return surfaces, gaps


# ── Objective: Seidel merit with analytic gradient ───────────────────────────

def objective(x, f_number, nu0_chief, stop_idx, n1, n2, d1, d2, weights):
    """Returns (merit, gradient) for SLSQP."""
    surfaces, gaps = build_system(x, n1, n2, d1, d2)

    (merit, dM_dc, dM_dd, S_total,
     efl, sa, h_m, dh_m_dc, dh_m_dd,
     dEFL_dc, dEFL_dd) = merit_with_grad_fnum(
        surfaces, gaps, stop_idx, f_number, nu0_chief, weights)

    # Map gradient back to x = [c1, c2, c3, c4, d_air]
    grad = np.zeros(5)
    grad[0:4] = dM_dc
    grad[4] = dM_dd[1]      # d_air is gap index 1

    return merit, grad


# ── EFL constraint: g(x) = EFL(x) - EFL_target = 0 ─────────────────────────

def efl_constraint(x, efl_target, n1, n2, d1, d2):
    """Returns (EFL - target, gradient)."""
    surfaces, gaps = build_system(x, n1, n2, d1, d2)

    efl, dEFL_dc, dEFL_dd = efl_with_grad(surfaces, gaps)

    grad = np.zeros(5)
    grad[0:4] = dEFL_dc
    grad[4] = dEFL_dd[1]

    return efl - efl_target, grad


# ── Edge thickness constraints with analytic gradients ───────────────────────

def edge_constraints(x, f_number, n1, n2, d1, d2, d_edge_min):
    """
    Returns edge thickness values and Jacobian.
    Uses actual marginal ray heights from f#-coupled trace.
    """
    surfaces, gaps = build_system(x, n1, n2, d1, d2)

    # Get marginal ray from f#-coupled trace
    (h_m, nu_m, dh_m_dc, dnu_m_dc, dh_m_dd, dnu_m_dd,
     efl, sa, dEFL_dc, dEFL_dd) = trace_marginal_fnum(surfaces, gaps, f_number)

    ET, dET_dc, dET_dd = edge_thickness_with_grad(
        surfaces, gaps, h_m, dh_m_dc, dh_m_dd, d_edge_min)

    # Map to x-space: [c1, c2, c3, c4, d_air]
    n_et = len(ET)
    dET_dx = np.zeros((n_et, 5))
    dET_dx[:, 0:4] = dET_dc
    dET_dx[:, 4] = dET_dd[:, 1]  # d_air is gap index 1

    return ET, dET_dx


# ── Optimizer ────────────────────────────────────────────────────────────────

def optimize_two_singlets(x0, efl_target=80.0, f_number=4.0,
                          field_angle_deg=3.0, stop_idx=0,
                          n1=1.5, n2=1.5, d1=5.0, d2=5.0,
                          weights=None, c_bounds=(-0.05, 0.05),
                          d_air_bounds=(2.0, 100.0), d_edge_min=0.5,
                          verbose=True):
    """
    Optimize a 2-singlet system using SLSQP with analytic gradients.

    Parameters
    ----------
    x0 : array-like, [c1, c2, c3, c4, d_air] initial design
    efl_target : target effective focal length (mm)
    f_number : target f-number (semi_aperture = EFL / (2 * f#))
    """
    if weights is None:
        weights = np.ones(5)

    nu0_chief = np.tan(np.radians(field_angle_deg))

    # Objective
    def obj_func(x):
        m, g = objective(x, f_number, nu0_chief, stop_idx,
                         n1, n2, d1, d2, weights)
        return m

    def obj_jac(x):
        m, g = objective(x, f_number, nu0_chief, stop_idx,
                         n1, n2, d1, d2, weights)
        return g

    # EFL equality constraint
    def efl_func(x):
        val, _ = efl_constraint(x, efl_target, n1, n2, d1, d2)
        return val

    def efl_jac(x):
        _, grad = efl_constraint(x, efl_target, n1, n2, d1, d2)
        return grad

    # Edge thickness inequality constraints with analytic Jacobians
    def edge_func_i(i):
        def func(x):
            ET, _ = edge_constraints(x, f_number, n1, n2, d1, d2, d_edge_min)
            return ET[i]
        return func

    def edge_jac_i(i):
        def jac(x):
            _, dET_dx = edge_constraints(x, f_number, n1, n2, d1, d2, d_edge_min)
            return dET_dx[i]
        return jac

    # Bounds
    bounds = [
        c_bounds,       # c1
        c_bounds,       # c2
        c_bounds,       # c3
        c_bounds,       # c4
        d_air_bounds,   # d_air
    ]

    constraints = [
        {'type': 'eq', 'fun': efl_func, 'jac': efl_jac},
        {'type': 'ineq', 'fun': edge_func_i(0), 'jac': edge_jac_i(0)},  # lens 1
        {'type': 'ineq', 'fun': edge_func_i(1), 'jac': edge_jac_i(1)},  # air gap
        {'type': 'ineq', 'fun': edge_func_i(2), 'jac': edge_jac_i(2)},  # lens 2
    ]

    # Iteration callback
    iteration = [0]
    def callback(x):
        iteration[0] += 1
        if verbose and iteration[0] % 10 == 0:
            m, _ = objective(x, f_number, nu0_chief, stop_idx,
                             n1, n2, d1, d2, weights)
            surfaces, gaps = build_system(x, n1, n2, d1, d2)
            efl, _, _ = efl_with_grad(surfaces, gaps)
            print(f"  iter {iteration[0]:4d}: merit={m:.6e}  EFL={efl:.2f}")

    if verbose:
        print("Starting SLSQP optimization...")
        print(f"  EFL target: {efl_target} mm")
        print(f"  f-number: {f_number}")
        semi_ap_init = efl_target / (2 * f_number)
        print(f"  Semi-aperture (at target EFL): {semi_ap_init:.2f} mm")
        print(f"  Field angle: {field_angle_deg} deg")
        print(f"  x0 = {x0}")
        m0, _ = objective(x0, f_number, nu0_chief, stop_idx,
                          n1, n2, d1, d2, weights)
        print(f"  Initial merit: {m0:.6e}")
        print()

    result = minimize(
        obj_func, x0,
        method='SLSQP',
        jac=obj_jac,
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={'maxiter': 500, 'ftol': 1e-15, 'disp': verbose},
    )

    if verbose:
        print()
        print("=" * 60)
        print("OPTIMIZATION RESULT")
        print("=" * 60)
        print(f"  Success: {result.success}")
        print(f"  Message: {result.message}")
        print(f"  Iterations: {result.nit}")
        print(f"  Function evals: {result.nfev}")
        print()

        x_opt = result.x
        c1, c2, c3, c4, d_air = x_opt
        print(f"  c1 = {c1:+.8f}  (R1 = {1/c1:+.2f} mm)" if abs(c1) > 1e-10 else f"  c1 = {c1:+.8f}  (flat)")
        print(f"  c2 = {c2:+.8f}  (R2 = {1/c2:+.2f} mm)" if abs(c2) > 1e-10 else f"  c2 = {c2:+.8f}  (flat)")
        print(f"  c3 = {c3:+.8f}  (R3 = {1/c3:+.2f} mm)" if abs(c3) > 1e-10 else f"  c3 = {c3:+.8f}  (flat)")
        print(f"  c4 = {c4:+.8f}  (R4 = {1/c4:+.2f} mm)" if abs(c4) > 1e-10 else f"  c4 = {c4:+.8f}  (flat)")
        print(f"  d_air = {d_air:.4f} mm")
        print()

        surfaces, gaps = build_system(x_opt, n1, n2, d1, d2)
        efl, _, _ = efl_with_grad(surfaces, gaps)
        sa = efl / (2 * f_number)
        print(f"  EFL = {efl:.4f} mm (target {efl_target})")
        print(f"  f/# = {f_number}")
        print(f"  Semi-aperture = {sa:.4f} mm")
        print(f"  Merit = {result.fun:.6e}")
        print()

        # Seidel breakdown
        (m, _, _, S_total,
         _, _, h_m, _, _, _, _) = merit_with_grad_fnum(
            surfaces, gaps, stop_idx, f_number, nu0_chief, weights)
        print(f"  S_I   = {S_total[0]:+.8f}  (spherical)")
        print(f"  S_II  = {S_total[1]:+.8f}  (coma)")
        print(f"  S_III = {S_total[2]:+.8f}  (astigmatism)")
        print(f"  S_IV  = {S_total[3]:+.8f}  (Petzval)")
        print(f"  S_V   = {S_total[4]:+.8f}  (distortion)")
        print()

        # Edge thicknesses
        ET, _ = edge_constraints(x_opt, f_number, n1, n2, d1, d2, 0)
        print(f"  Ray heights: {['%.4f' % h for h in h_m]}")
        print(f"  Edge thickness L1:  {ET[0]:.4f} mm")
        print(f"  Edge thickness air: {ET[1]:.4f} mm")
        print(f"  Edge thickness L2:  {ET[2]:.4f} mm")

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    efl_target = 80.0
    f_number = 4.0
    field_angle_deg = 3.0

    # Test several starting points
    starts = [
        ("Biconvex + Biconvex",
         [0.02, -0.01, 0.015, -0.005, 20.0]),
        ("Plano-convex + Plano-convex (near best from scan)",
         [0.01, 0.0, 0.017, 0.0, 12.0]),
        ("Symmetric biconvex pair",
         [0.015, -0.015, 0.015, -0.015, 15.0]),
        ("Meniscus + Biconvex",
         [0.03, 0.01, 0.02, -0.01, 10.0]),
    ]

    results = []
    for name, x0 in starts:
        print("\n" + "=" * 60)
        print(f"START: {name}")
        print("=" * 60)
        res = optimize_two_singlets(
            x0, efl_target=efl_target, f_number=f_number,
            field_angle_deg=field_angle_deg)
        results.append((name, res))

    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY — ALL STARTING POINTS")
    print("=" * 60)
    print(f"  {'Start':40s} {'Merit':>12s} {'EFL':>8s} {'OK':>4s}")
    print(f"  {'-----':40s} {'-----':>12s} {'---':>8s} {'--':>4s}")
    for name, res in sorted(results, key=lambda r: r[1].fun):
        x = res.x
        surfaces, gaps = build_system(x)
        efl, _, _ = efl_with_grad(surfaces, gaps)
        ok = "Y" if res.success else "N"
        print(f"  {name:40s} {res.fun:12.6e} {efl:8.2f} {ok:>4s}")


if __name__ == "__main__":
    main()
