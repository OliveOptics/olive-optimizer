"""
Analytic gradients for the Seidel surrogate merit function.

Computes dM/dc_i (curvatures) and dM/dd_i (thicknesses/gaps) via chain rule
through the y-nu trace and Seidel coefficient formulas.

Gradient verification against central finite differences is included.
"""

import numpy as np
from parse_zmx import (parse_zmx, build_system_from_zmx, ynu_trace,
                        find_chief_ray_initial, seidel_coefficients)


# ── Sag helpers ─────────────────────────────────────────────────────────────

def sag_derivs(c, y, k=0.0, A4=0.0):
    """
    Exact conic + even asphere sag and partial derivatives.

    sag = c*y^2 / (1 + sqrt(1 - (1+k)*c^2*y^2)) + A4*y^4

    where R = sqrt(1 - (1+k)*c^2*y^2).

    The derivatives dsag/dc and dsag/dy have the elegant property of being
    independent of k in their functional form (only R changes):
        dsag/dc = y^2 / (R*(1+R))
        dsag/dy = c*y / R + 4*A4*y^3

    Additional derivatives for aspheric parameters:
        dsag/dk = c^3*y^4 / (2*R*(1+R)^2)
        dsag/dA4 = y^4

    Returns
    -------
    s, dsdc, dsdy, dsdk, dsdA4
    """
    if abs(y) < 1e-15:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if abs(c) < 1e-15:
        s = c * y**2 / 2 + A4 * y**4
        return s, y**2 / 2, c * y + 4 * A4 * y**3, 0.0, y**4
    arg = 1 - (1 + k) * c**2 * y**2
    if arg < 0:
        # Ray beyond surface — paraxial fallback
        s = c * y**2 / 2 + A4 * y**4
        return s, y**2 / 2, c * y + 4 * A4 * y**3, 0.0, y**4
    R = np.sqrt(arg)
    D = 1 + R
    s = c * y**2 / D + A4 * y**4
    dsdc = y**2 / (R * D)
    dsdy = c * y / R + 4 * A4 * y**3
    dsdk = c**3 * y**4 / (2 * R * D**2)
    dsdA4 = y**4
    return s, dsdc, dsdy, dsdk, dsdA4


# ── y-nu trace with gradient ────────────────────────────────────────────────

def ynu_trace_with_grad(surfaces, gaps, h0, nu0):
    """
    Paraxial y-nu trace with analytic derivatives w.r.t. curvatures c and gaps d.

    The y-nu recurrence:
        Refraction:   nu[i] = nu_pre[i] - h[i] * phi[i],  phi[i] = (n'-n)*c[i]
        Propagation:  h[i+1] = h[i] + (d[i]/n_gap[i]) * nu[i]

    Derivatives propagate forward: a perturbation at surface j affects all
    downstream surfaces through the linear chain.

    Returns
    -------
    h, nu : (K,) ray heights and post-refraction reduced angles
    dh_dc, dnu_dc : (K, K) partial derivatives w.r.t. curvatures
    dh_dd, dnu_dd : (K, K-1) partial derivatives w.r.t. gap thicknesses
    """
    K = len(surfaces)
    h = np.zeros(K)
    nu = np.zeros(K)
    dh_dc = np.zeros((K, K))
    dnu_dc = np.zeros((K, K))
    dh_dd = np.zeros((K, K - 1))
    dnu_dd = np.zeros((K, K - 1))

    # Surface 0: refraction only
    h[0] = h0
    c0, n0, np0 = surfaces[0]
    phi0 = (np0 - n0) * c0
    nu[0] = nu0 - h0 * phi0
    # dnu[0]/dc[0] = -h0 * (n'0 - n0)
    dnu_dc[0, 0] = -h0 * (np0 - n0)

    for i in range(1, K):
        d_gap, n_gap = gaps[i - 1]
        t = d_gap / n_gap

        # Propagation: h[i] = h[i-1] + t * nu[i-1]
        h[i] = h[i - 1] + t * nu[i - 1]
        dh_dc[i] = dh_dc[i - 1] + t * dnu_dc[i - 1]
        dh_dd[i] = dh_dd[i - 1] + t * dnu_dd[i - 1]
        dh_dd[i, i - 1] += nu[i - 1] / n_gap  # direct: d(d[i-1]/n * nu[i-1])/dd[i-1]

        # Refraction: nu[i] = nu[i-1] - h[i] * phi[i]
        ci, ni, npi = surfaces[i]
        phi_i = (npi - ni) * ci
        nu[i] = nu[i - 1] - h[i] * phi_i
        dnu_dc[i] = dnu_dc[i - 1] - dh_dc[i] * phi_i
        dnu_dc[i, i] -= h[i] * (npi - ni)  # direct: -h[i] * d(phi)/dc[i]
        dnu_dd[i] = dnu_dd[i - 1] - dh_dd[i] * phi_i

    return h, nu, dh_dc, dnu_dc, dh_dd, dnu_dd


# ── Chief ray with gradient ─────────────────────────────────────────────────

def _stop_height_and_grad(h, nu, dh_dc, dnu_dc, dh_dd, dnu_dd,
                          stop_info, gaps):
    """
    Compute ray height at the stop and its gradient w.r.t. c and d.

    If stop is at a surface: h_stop = h[idx], gradients are dh_dc[idx], etc.
    If stop is inside a gap: h_stop = h[idx] + (d_before/n) * nu[idx],
    and gradients include the propagation.
    """
    if isinstance(stop_info, int):
        stop_info = {'type': 'at_surface', 'surface_idx': stop_info}

    if stop_info['type'] == 'at_surface':
        idx = stop_info['surface_idx']
        return h[idx], dh_dc[idx], dh_dd[idx]
    else:
        idx = stop_info['gap_idx']
        d_before = stop_info['distance_before_stop']
        n_gap = gaps[idx][1]
        t = d_before / n_gap
        h_stop = h[idx] + t * nu[idx]
        dh_stop_dc = dh_dc[idx] + t * dnu_dc[idx]
        dh_stop_dd = dh_dd[idx] + t * dnu_dd[idx]
        # d_before is part of gap[idx], so dh_stop/dd[idx] gets
        # an extra nu[idx]/n_gap term from d(d_before)/dd[idx].
        # But d_before is a fixed offset within the gap (not a
        # separate design variable), so no extra term needed.
        return h_stop, dh_stop_dc, dh_stop_dd


def chief_ray_with_grad(surfaces, gaps, stop_info, nu0_chief):
    """
    Compute chief ray trace with total derivatives w.r.t. c and d,
    including the implicit dependency of h0_chief on design variables.

    stop_info: dict from build_system_from_zmx or integer index (legacy).

    h0_chief is found by linear interpolation:
        trace with h0=0 -> h_stop_a
        trace with h0=1 -> h_stop_b
        h0_chief = -h_stop_a / (h_stop_b - h_stop_a)

    Total derivative uses chain rule:
        dh_c/dx = (dh_c/dx)|_{h0 fixed} + (dh_c/dh0) * (dh0/dx)
    """
    if isinstance(stop_info, int):
        stop_info = {'type': 'at_surface', 'surface_idx': stop_info}

    # Two auxiliary traces to find h0_chief and its gradient
    h_a, nu_a, dha_dc, dnua_dc, dha_dd, dnua_dd = ynu_trace_with_grad(
        surfaces, gaps, 0.0, nu0_chief)
    h_b, nu_b, dhb_dc, dnub_dc, dhb_dd, dnub_dd = ynu_trace_with_grad(
        surfaces, gaps, 1.0, nu0_chief)

    h_stop_a, dh_stop_a_dc, dh_stop_a_dd = _stop_height_and_grad(
        h_a, nu_a, dha_dc, dnua_dc, dha_dd, dnua_dd, stop_info, gaps)
    h_stop_b, dh_stop_b_dc, dh_stop_b_dd = _stop_height_and_grad(
        h_b, nu_b, dhb_dc, dnub_dc, dhb_dd, dnub_dd, stop_info, gaps)

    D = h_stop_b - h_stop_a
    h0_chief = -h_stop_a / D

    # Gradient of h0_chief via quotient rule
    dD_dc = dh_stop_b_dc - dh_stop_a_dc
    dh0_dc = (-dh_stop_a_dc * D + h_stop_a * dD_dc) / D**2

    dD_dd = dh_stop_b_dd - dh_stop_a_dd
    dh0_dd = (-dh_stop_a_dd * D + h_stop_a * dD_dd) / D**2

    # Trace chief ray with actual h0_chief (partial derivatives at fixed h0)
    h_c, nu_c, dhc_dc, dnuc_dc, dhc_dd, dnuc_dd = ynu_trace_with_grad(
        surfaces, gaps, h0_chief, nu0_chief)

    # Sensitivity of trace to h0: trace (h0=1, nu0=0) through same system
    # Since the trace is linear in (h0, nu0), dh[i]/dh0 = h_unit[i]
    h_unit, nu_unit = ynu_trace(surfaces, gaps, 1.0, 0.0)

    # Total derivatives via chain rule
    dhc_dc_total = dhc_dc + np.outer(h_unit, dh0_dc)
    dnuc_dc_total = dnuc_dc + np.outer(nu_unit, dh0_dc)
    dhc_dd_total = dhc_dd + np.outer(h_unit, dh0_dd)
    dnuc_dd_total = dnuc_dd + np.outer(nu_unit, dh0_dd)

    return h0_chief, h_c, nu_c, dhc_dc_total, dnuc_dc_total, dhc_dd_total, dnuc_dd_total


# ── Marginal ray with f-number coupling ──────────────────────────────────────

def trace_marginal_fnum(surfaces, gaps, f_number):
    """
    Trace marginal ray with semi_aperture = EFL(x) / (2 * f_number).

    Uses linearity: trace a unit ray (h0=1), then scale by semi_aperture.
    The coupling through EFL is handled via the product rule.

    Returns
    -------
    h_m, nu_m : (K,) marginal ray heights and angles
    dh_m_dc, dnu_m_dc : (K, K) total derivatives w.r.t. curvatures
    dh_m_dd, dnu_m_dd : (K, K-1) total derivatives w.r.t. gaps
    efl : effective focal length
    sa : semi-aperture
    dEFL_dc, dEFL_dd : EFL derivatives
    """
    # Unit marginal ray
    h_u, nu_u, dh_u_dc, dnu_u_dc, dh_u_dd, dnu_u_dd = ynu_trace_with_grad(
        surfaces, gaps, 1.0, 0.0)

    # EFL from unit trace: EFL = -1 / nu_u[-1]
    nu_f = nu_u[-1]
    efl = -1.0 / nu_f
    sa = efl / (2 * f_number)

    # EFL gradient: dEFL/dx = (1/nu_f^2) * dnu_f/dx
    dEFL_dc = (1.0 / nu_f**2) * dnu_u_dc[-1]
    dEFL_dd = (1.0 / nu_f**2) * dnu_u_dd[-1]

    # Semi-aperture gradient
    dsa_dc = dEFL_dc / (2 * f_number)
    dsa_dd = dEFL_dd / (2 * f_number)

    # Scale to actual marginal ray
    h_m = sa * h_u
    nu_m = sa * nu_u

    # Total derivatives (product rule)
    dh_m_dc = np.outer(h_u, dsa_dc) + sa * dh_u_dc
    dnu_m_dc = np.outer(nu_u, dsa_dc) + sa * dnu_u_dc
    dh_m_dd = np.outer(h_u, dsa_dd) + sa * dh_u_dd
    dnu_m_dd = np.outer(nu_u, dsa_dd) + sa * dnu_u_dd

    return (h_m, nu_m, dh_m_dc, dnu_m_dc, dh_m_dd, dnu_m_dd,
            efl, sa, dEFL_dc, dEFL_dd)


# ── Seidel coefficients with gradient ────────────────────────────────────────

def seidel_with_grad(surfaces,
                     h_m, nu_m, dh_m_dc, dnu_m_dc, dh_m_dd, dnu_m_dd,
                     h_c, nu_c, dh_c_dc, dnu_c_dc, dh_c_dd, dnu_c_dd,
                     nu0_m, nu0_c,
                     aspheres=None):
    """
    Compute Seidel coefficients and their gradients w.r.t. c and d.

    aspheres : list of (k, A4) per surface, or None for all-spherical.

    At each surface i:
        A       = nu_pre_m + n * h_m * c
        A_bar   = nu_pre_c + n * h_c * c
        delta   = nu_m/(n')^2 - nu_pre_m/n^2
        Q       = -h_m * delta
        S_I     = A^2 * Q
        S_II    = A * A_bar * Q
        S_III   = A_bar^2 * Q
        S_IV    = H^2 * phi / (n * n')
        S_V     = (S_III + S_IV) * A_bar / A

    Returns
    -------
    S : (5, K) Seidel coefficients per surface
    dS_dc : (5, K, K) derivatives w.r.t. curvatures
    dS_dd : (5, K, K-1) derivatives w.r.t. gaps
    """
    K = len(surfaces)
    Kd = K - 1

    # Lagrange invariant: H = nu0_m * h_c[0] - nu0_c * h_m[0]
    # In f-number mode h_m[0] depends on design variables, so dH/dx != 0
    H = nu0_m * h_c[0] - nu0_c * h_m[0]
    dH_dc = nu0_m * dh_c_dc[0] - nu0_c * dh_m_dc[0]
    dH_dd = nu0_m * dh_c_dd[0] - nu0_c * dh_m_dd[0]

    # Pre-refraction nu and their derivatives
    nu_pre_m = np.zeros(K)
    nu_pre_c = np.zeros(K)
    nu_pre_m[0] = nu0_m
    nu_pre_c[0] = nu0_c
    dnu_pre_m_dc = np.zeros((K, K))
    dnu_pre_m_dd = np.zeros((K, Kd))
    dnu_pre_c_dc = np.zeros((K, K))
    dnu_pre_c_dd = np.zeros((K, Kd))
    for i in range(1, K):
        nu_pre_m[i] = nu_m[i - 1]
        nu_pre_c[i] = nu_c[i - 1]
        dnu_pre_m_dc[i] = dnu_m_dc[i - 1]
        dnu_pre_m_dd[i] = dnu_m_dd[i - 1]
        dnu_pre_c_dc[i] = dnu_c_dc[i - 1]
        dnu_pre_c_dd[i] = dnu_c_dd[i - 1]

    S = np.zeros((5, K))
    dS_dc = np.zeros((5, K, K))
    dS_dd = np.zeros((5, K, Kd))

    for i in range(K):
        c, n, n_prime = surfaces[i]
        phi = (n_prime - n) * c

        # Forward quantities
        A = nu_pre_m[i] + n * h_m[i] * c
        A_bar = nu_pre_c[i] + n * h_c[i] * c
        delta = nu_m[i] / n_prime**2 - nu_pre_m[i] / n**2
        Q = -h_m[i] * delta

        S[0, i] = A**2 * Q
        S[1, i] = A * A_bar * Q
        S[2, i] = A_bar**2 * Q
        S[3, i] = H**2 * phi / (n * n_prime)
        S[4, i] = (S[2, i] + S[3, i]) * (A_bar / A) if abs(A) > 1e-12 else 0.0

        # Indicator vector for dc[i]/dc[j] = delta(i,j)
        dc_vec = np.zeros(K)
        dc_vec[i] = 1.0

        # ── Derivatives w.r.t. curvatures (vectorized over j) ──
        dA = dnu_pre_m_dc[i] + n * dh_m_dc[i] * c + n * h_m[i] * dc_vec
        dA_bar = dnu_pre_c_dc[i] + n * dh_c_dc[i] * c + n * h_c[i] * dc_vec
        d_delta = dnu_m_dc[i] / n_prime**2 - dnu_pre_m_dc[i] / n**2
        dQ = -dh_m_dc[i] * delta - h_m[i] * d_delta

        dS_dc[0, i] = 2 * A * dA * Q + A**2 * dQ
        dS_dc[1, i] = (dA * A_bar + A * dA_bar) * Q + A * A_bar * dQ
        dS_dc[2, i] = 2 * A_bar * dA_bar * Q + A_bar**2 * dQ
        dS_dc[3, i] = (2 * H * dH_dc * phi + H**2 * (n_prime - n) * dc_vec) / (n * n_prime)
        if abs(A) > 1e-12:
            ratio = A_bar / A
            d_ratio = (dA_bar * A - A_bar * dA) / A**2
            dS_dc[4, i] = ((dS_dc[2, i] + dS_dc[3, i]) * ratio
                           + (S[2, i] + S[3, i]) * d_ratio)

        # ── Derivatives w.r.t. gaps (vectorized over j) ──
        dA = dnu_pre_m_dd[i] + n * dh_m_dd[i] * c
        dA_bar = dnu_pre_c_dd[i] + n * dh_c_dd[i] * c
        d_delta = dnu_m_dd[i] / n_prime**2 - dnu_pre_m_dd[i] / n**2
        dQ = -dh_m_dd[i] * delta - h_m[i] * d_delta

        dS_dd[0, i] = 2 * A * dA * Q + A**2 * dQ
        dS_dd[1, i] = (dA * A_bar + A * dA_bar) * Q + A * A_bar * dQ
        dS_dd[2, i] = 2 * A_bar * dA_bar * Q + A_bar**2 * dQ
        dS_dd[3, i] = 2 * H * dH_dd * phi / (n * n_prime)
        if abs(A) > 1e-12:
            ratio = A_bar / A
            d_ratio = (dA_bar * A - A_bar * dA) / A**2
            dS_dd[4, i] = ((dS_dd[2, i] + dS_dd[3, i]) * ratio
                           + (S[2, i] + S[3, i]) * d_ratio)

    # ── Aspheric contributions ──
    # Φ_i = (n'-n) * (k_i * c_i³ + 8*A4_i)
    # ΔS_I  = Φ * h_m⁴,  ΔS_II = Φ * h_m³*h_c,  ΔS_III = Φ * h_m²*h_c²
    # ΔS_IV = 0,          ΔS_V  = Φ * h_m*h_c³
    dS_dk = np.zeros((5, K))     # gradient w.r.t. conic constants
    dS_dA4 = np.zeros((5, K))    # gradient w.r.t. A4 coefficients

    if aspheres is not None:
        for i in range(K):
            ki, A4i = aspheres[i]

            c, n, n_prime = surfaces[i]
            dn = n_prime - n
            Phi = dn * (ki * c**3 + 8 * A4i)

            hm = h_m[i]
            hc = h_c[i]
            hm2 = hm * hm
            hm3 = hm2 * hm
            hm4 = hm3 * hm
            hc2 = hc * hc
            hc3 = hc2 * hc

            # Aspheric Seidel values
            S[0, i] += Phi * hm4
            S[1, i] += Phi * hm3 * hc
            S[2, i] += Phi * hm2 * hc2
            # S[3, i] += 0  (aspherics don't affect Petzval)
            S[4, i] += Phi * hm * hc3

            # ── Gradients w.r.t. k_i and A4_i (direct, no chain rule) ──
            dPhi_dk = dn * c**3
            dPhi_dA4 = dn * 8

            dS_dk[0, i] = dPhi_dk * hm4
            dS_dk[1, i] = dPhi_dk * hm3 * hc
            dS_dk[2, i] = dPhi_dk * hm2 * hc2
            dS_dk[4, i] = dPhi_dk * hm * hc3

            dS_dA4[0, i] = dPhi_dA4 * hm4
            dS_dA4[1, i] = dPhi_dA4 * hm3 * hc
            dS_dA4[2, i] = dPhi_dA4 * hm2 * hc2
            dS_dA4[4, i] = dPhi_dA4 * hm * hc3

            # ── Gradients w.r.t. curvatures (Φ depends on c_i) ──
            # dΦ/dc_i = dn * 3*k_i*c_i²
            dPhi_dc_i = dn * 3 * ki * c**2
            dS_dc[0, i, i] += dPhi_dc_i * hm4
            dS_dc[1, i, i] += dPhi_dc_i * hm3 * hc
            dS_dc[2, i, i] += dPhi_dc_i * hm2 * hc2
            dS_dc[4, i, i] += dPhi_dc_i * hm * hc3

            # Also: Φ * d(h_m^4)/dc_j = Φ * 4*h_m³ * dh_m/dc_j (for all j)
            # and similarly for h_c terms
            dS_dc[0, i] += Phi * 4 * hm3 * dh_m_dc[i]
            dS_dc[1, i] += Phi * (3 * hm2 * dh_m_dc[i] * hc +
                                   hm3 * dh_c_dc[i])
            dS_dc[2, i] += Phi * (2 * hm * dh_m_dc[i] * hc2 +
                                   hm2 * 2 * hc * dh_c_dc[i])
            dS_dc[4, i] += Phi * (dh_m_dc[i] * hc3 +
                                   hm * 3 * hc2 * dh_c_dc[i])

            # ── Gradients w.r.t. gaps ──
            dS_dd[0, i] += Phi * 4 * hm3 * dh_m_dd[i]
            dS_dd[1, i] += Phi * (3 * hm2 * dh_m_dd[i] * hc +
                                   hm3 * dh_c_dd[i])
            dS_dd[2, i] += Phi * (2 * hm * dh_m_dd[i] * hc2 +
                                   hm2 * 2 * hc * dh_c_dd[i])
            dS_dd[4, i] += Phi * (dh_m_dd[i] * hc3 +
                                   hm * 3 * hc2 * dh_c_dd[i])

    return S, dS_dc, dS_dd, dS_dk, dS_dA4


# ── Merit function with gradient ────────────────────────────────────────────

def merit_with_grad(surfaces, gaps, stop_idx, semi_aperture, nu0_chief,
                    weights=None):
    """
    Compute merit function M = sum_k w_k * (sum_i S_k[i])^2
    and its gradient dM/dc, dM/dd.

    Returns
    -------
    merit : float
    dM_dc : (K,) gradient w.r.t. curvatures
    dM_dd : (K-1,) gradient w.r.t. gap thicknesses
    S_total : (5,) total Seidel aberrations
    """
    K = len(surfaces)
    if weights is None:
        weights = np.ones(5)

    # Marginal ray
    h_m, nu_m, dh_m_dc, dnu_m_dc, dh_m_dd, dnu_m_dd = ynu_trace_with_grad(
        surfaces, gaps, semi_aperture, 0.0)

    # Chief ray (with h0 dependency)
    h0_chief, h_c, nu_c, dh_c_dc, dnu_c_dc, dh_c_dd, dnu_c_dd = (
        chief_ray_with_grad(surfaces, gaps, stop_idx, nu0_chief))

    # Seidel coefficients with gradients
    S, dS_dc, dS_dd, dS_dk, dS_dA4 = seidel_with_grad(
        surfaces,
        h_m, nu_m, dh_m_dc, dnu_m_dc, dh_m_dd, dnu_m_dd,
        h_c, nu_c, dh_c_dc, dnu_c_dc, dh_c_dd, dnu_c_dd,
        0.0, nu0_chief)

    # Merit function
    S_total = S.sum(axis=1)  # (5,)
    merit = np.sum(weights * S_total**2)

    # Gradient: dM/dx_j = 2 * sum_k w_k * S_total_k * dS_total_k/dx_j
    dS_total_dc = dS_dc.sum(axis=1)  # (5, K)
    dS_total_dd = dS_dd.sum(axis=1)  # (5, K-1)

    dM_dc = 2 * np.sum(weights[:, None] * S_total[:, None] * dS_total_dc, axis=0)
    dM_dd = 2 * np.sum(weights[:, None] * S_total[:, None] * dS_total_dd, axis=0)

    return merit, dM_dc, dM_dd, S_total


# ── Merit function with f-number coupling ──────────────────────────────────

def merit_with_grad_fnum(surfaces, gaps, stop_idx, f_number, nu0_chief,
                         weights=None):
    """
    Compute merit function with f-number-coupled semi-aperture.
    semi_aperture = EFL(x) / (2 * f_number), so it varies with design.

    Returns
    -------
    merit, dM_dc, dM_dd, S_total : same as merit_with_grad
    efl : effective focal length
    sa : semi-aperture
    h_m : (K,) marginal ray heights (needed for edge thickness)
    dh_m_dc, dh_m_dd : marginal ray height derivatives
    dEFL_dc, dEFL_dd : EFL derivatives
    """
    K = len(surfaces)
    if weights is None:
        weights = np.ones(5)

    # Marginal ray with f# coupling
    (h_m, nu_m, dh_m_dc, dnu_m_dc, dh_m_dd, dnu_m_dd,
     efl, sa, dEFL_dc, dEFL_dd) = trace_marginal_fnum(surfaces, gaps, f_number)

    # Chief ray (with h0 dependency)
    h0_chief, h_c, nu_c, dh_c_dc, dnu_c_dc, dh_c_dd, dnu_c_dd = (
        chief_ray_with_grad(surfaces, gaps, stop_idx, nu0_chief))

    # Seidel coefficients with gradients
    S, dS_dc, dS_dd, dS_dk, dS_dA4 = seidel_with_grad(
        surfaces,
        h_m, nu_m, dh_m_dc, dnu_m_dc, dh_m_dd, dnu_m_dd,
        h_c, nu_c, dh_c_dc, dnu_c_dc, dh_c_dd, dnu_c_dd,
        0.0, nu0_chief)

    # Merit function
    S_total = S.sum(axis=1)  # (5,)
    merit = np.sum(weights * S_total**2)

    # Gradient
    dS_total_dc = dS_dc.sum(axis=1)
    dS_total_dd = dS_dd.sum(axis=1)

    dM_dc = 2 * np.sum(weights[:, None] * S_total[:, None] * dS_total_dc, axis=0)
    dM_dd = 2 * np.sum(weights[:, None] * S_total[:, None] * dS_total_dd, axis=0)

    return (merit, dM_dc, dM_dd, S_total,
            efl, sa, h_m, dh_m_dc, dh_m_dd, dEFL_dc, dEFL_dd)


# ── Edge thickness with gradient ─────────────────────────────────────────────

def edge_thickness_with_grad(surfaces, gaps, h_m, dh_m_dc, dh_m_dd,
                              d_edge_min=0.5):
    """
    Compute edge thickness constraints using actual marginal ray heights.

    ET[i] = d_gap[i] - sag(c[i], |h[i]|) + sag(c[i+1], |h[i+1]|) - edge_min

    Glass gaps get d_edge_min subtracted; air gaps get 0.
    Each ET must be >= 0 for physical realizability.

    Returns
    -------
    ET : (K-1,) edge thickness values
    dET_dc : (K-1, K) derivatives w.r.t. curvatures
    dET_dd : (K-1, K-1) derivatives w.r.t. gap thicknesses
    """
    K = len(surfaces)
    Kd = K - 1

    ET = np.zeros(Kd)
    dET_dc = np.zeros((Kd, K))
    dET_dd = np.zeros((Kd, Kd))

    for idx in range(Kd):
        c_f = surfaces[idx][0]
        c_b = surfaces[idx + 1][0]
        y_f = abs(h_m[idx])
        y_b = abs(h_m[idx + 1])
        d_gap = gaps[idx][0]
        n_gap = gaps[idx][1]

        s_f, dsdc_f, dsdy_f, _, _ = sag_derivs(c_f, y_f)
        s_b, dsdc_b, dsdy_b, _, _ = sag_derivs(c_b, y_b)

        # Glass gap -> apply edge min; air gap -> don't
        edge_min = d_edge_min if n_gap > 1.001 else 0.0
        ET[idx] = d_gap - s_f + s_b - edge_min

        # d|h|/dh = sign(h)
        sgn_f = np.sign(h_m[idx]) if abs(h_m[idx]) > 1e-15 else 0.0
        sgn_b = np.sign(h_m[idx + 1]) if abs(h_m[idx + 1]) > 1e-15 else 0.0

        # Gradient w.r.t. curvatures (vectorized over j)
        ds_f_dc = dsdy_f * sgn_f * dh_m_dc[idx].copy()
        ds_f_dc[idx] += dsdc_f
        ds_b_dc = dsdy_b * sgn_b * dh_m_dc[idx + 1].copy()
        ds_b_dc[idx + 1] += dsdc_b
        dET_dc[idx] = -ds_f_dc + ds_b_dc

        # Gradient w.r.t. gaps (vectorized over j)
        ds_f_dd = dsdy_f * sgn_f * dh_m_dd[idx].copy()
        ds_b_dd = dsdy_b * sgn_b * dh_m_dd[idx + 1].copy()
        dET_dd[idx] = -ds_f_dd + ds_b_dd
        dET_dd[idx, idx] += 1.0  # d(d_gap)/dd_idx

    return ET, dET_dc, dET_dd


# ── BFL with gradient ────────────────────────────────────────────────────────

def bfl_with_grad(h_m, nu_m, dh_m_dc, dnu_m_dc, dh_m_dd, dnu_m_dd):
    """
    Back focal length and its gradient.

    BFL = -h_m[-1] / nu_m[-1]
    Quotient rule: dBFL/dx = (-dh*nu + h*dnu) / nu^2
    """
    h_f = h_m[-1]
    nu_f = nu_m[-1]
    bfl = -h_f / nu_f

    dBFL_dc = (-dh_m_dc[-1] * nu_f + h_f * dnu_m_dc[-1]) / nu_f**2
    dBFL_dd = (-dh_m_dd[-1] * nu_f + h_f * dnu_m_dd[-1]) / nu_f**2

    return bfl, dBFL_dc, dBFL_dd


# ── TTL with gradient ────────────────────────────────────────────────────────

def ttl_with_grad(gaps, bfl, dBFL_dc, dBFL_dd):
    """
    Total track length (first surface to paraxial focus) and its gradient.

    TTL = sum(gap thicknesses) + BFL
    dTTL/dc = dBFL/dc       (gaps don't depend on curvatures)
    dTTL/dd[j] = 1 + dBFL/dd[j]  (gap j contributes 1 plus BFL effect)
    """
    gap_sum = sum(g[0] for g in gaps)
    ttl = gap_sum + bfl

    dTTL_dc = dBFL_dc.copy()
    dTTL_dd = np.ones_like(dBFL_dd) + dBFL_dd

    return ttl, dTTL_dc, dTTL_dd


# ── Full merit with BFL/TTL penalties ────────────────────────────────────────

def full_merit_with_grad(surfaces, gaps, stop_info, nu0_chief,
                         semi_aperture=None, f_number=None,
                         weights=None,
                         bfl_target=None, w_bfl=1.0,
                         ttl_target=None, w_ttl=1.0,
                         cra_max=None, w_cra=1.0, cra_field_deg=None,
                         ri_min=None, w_ri=1.0, ri_field_deg=None,
                         aspheres=None):
    """
    Combined merit function:
        M = Σ wk*Sk² + penalties

    Penalties (one-sided, only active when violated):
        + w_bfl * max(0, bfl_target - BFL)²    (BFL >= target)
        + w_ttl * max(0, TTL - ttl_target)²     (TTL <= target)
        + w_cra * max(0, CRA - cra_max)²        (CRA <= max)
        + w_ri  * max(0, ri_min - RI)²           (RI >= min)

    Supports fixed semi_aperture or f-number mode.
    Penalties are only applied when targets are not None.

    Returns dict with all computed values and gradients.
    """
    if (semi_aperture is None) == (f_number is None):
        raise ValueError("Specify exactly one of semi_aperture or f_number")

    K = len(surfaces)
    if weights is None:
        weights = np.ones(5)

    # ── Marginal ray ──
    if f_number is not None:
        (h_m, nu_m, dh_m_dc, dnu_m_dc, dh_m_dd, dnu_m_dd,
         efl, sa, dEFL_dc, dEFL_dd) = trace_marginal_fnum(
            surfaces, gaps, f_number)
    else:
        h_m, nu_m, dh_m_dc, dnu_m_dc, dh_m_dd, dnu_m_dd = ynu_trace_with_grad(
            surfaces, gaps, semi_aperture, 0.0)
        efl = -semi_aperture / nu_m[-1]
        sa = semi_aperture
        dEFL_dc = semi_aperture / nu_m[-1]**2 * dnu_m_dc[-1]
        dEFL_dd = semi_aperture / nu_m[-1]**2 * dnu_m_dd[-1]

    # ── Chief ray ──
    h0_chief, h_c, nu_c, dh_c_dc, dnu_c_dc, dh_c_dd, dnu_c_dd = (
        chief_ray_with_grad(surfaces, gaps, stop_info, nu0_chief))

    # ── Seidel coefficients ──
    S, dS_dc, dS_dd, dS_dk, dS_dA4 = seidel_with_grad(
        surfaces,
        h_m, nu_m, dh_m_dc, dnu_m_dc, dh_m_dd, dnu_m_dd,
        h_c, nu_c, dh_c_dc, dnu_c_dc, dh_c_dd, dnu_c_dd,
        0.0, nu0_chief,
        aspheres=aspheres)

    S_total = S.sum(axis=1)
    merit = np.sum(weights * S_total**2)

    dS_total_dc = dS_dc.sum(axis=1)    # (5, K) -> sum over surfaces -> wrong
    dS_total_dd = dS_dd.sum(axis=1)
    # dc and dd: dS_dc[s, i, j] = dS_s_at_surface_i / dc_j
    # dS_total_dc[s, j] = sum_i dS_dc[s, i, j]  — correct, sum over surfaces
    dM_dc = 2 * np.sum(weights[:, None] * S_total[:, None] * dS_total_dc, axis=0)
    dM_dd = 2 * np.sum(weights[:, None] * S_total[:, None] * dS_total_dd, axis=0)
    # dk and dA4: dS_dk[s, i] = dS_s_at_surface_i / dk_i (only affects own surface)
    # dS_total / dk_j = dS_dk[s, j]  — NO sum over surfaces needed
    dM_dk = 2 * np.sum(weights[:, None] * S_total[:, None] * dS_dk, axis=0)  # (K,)
    dM_dA4 = 2 * np.sum(weights[:, None] * S_total[:, None] * dS_dA4, axis=0)  # (K,)

    # ── BFL ── (one-sided: penalize if BFL < target)
    bfl, dBFL_dc, dBFL_dd = bfl_with_grad(
        h_m, nu_m, dh_m_dc, dnu_m_dc, dh_m_dd, dnu_m_dd)

    if bfl_target is not None and w_bfl > 0:
        bfl_viol = bfl_target - bfl  # positive when violated
        if bfl_viol > 0:
            merit += w_bfl * bfl_viol**2
            dM_dc += -2 * w_bfl * bfl_viol * dBFL_dc
            dM_dd += -2 * w_bfl * bfl_viol * dBFL_dd

    # ── TTL ── (one-sided: penalize if TTL > target)
    ttl, dTTL_dc, dTTL_dd = ttl_with_grad(gaps, bfl, dBFL_dc, dBFL_dd)

    if ttl_target is not None and w_ttl > 0:
        ttl_viol = ttl - ttl_target  # positive when violated
        if ttl_viol > 0:
            merit += w_ttl * ttl_viol**2
            dM_dc += 2 * w_ttl * ttl_viol * dTTL_dc
            dM_dd += 2 * w_ttl * ttl_viol * dTTL_dd

    # ── CRA ── (one-sided: penalize if |CRA| > max at specified field)
    cra_val = None
    if cra_max is not None and w_cra > 0 and cra_field_deg is not None:
        cra_val, dCRA_dc, dCRA_dd = cra_with_grad(
            surfaces, gaps, stop_info, cra_field_deg)
        abs_cra = abs(cra_val)
        cra_sign = np.sign(cra_val) if abs(cra_val) > 1e-12 else 0.0
        cra_viol = abs_cra - cra_max  # positive when violated
        if cra_viol > 0:
            merit += w_cra * cra_viol**2
            # d|CRA|/dx = sign(CRA) * dCRA/dx
            dM_dc += 2 * w_cra * cra_viol * cra_sign * dCRA_dc
            dM_dd += 2 * w_cra * cra_viol * cra_sign * dCRA_dd

    # ── RI ── (one-sided: penalize if RI < min at specified field)
    ri_val = None
    if ri_min is not None and w_ri > 0 and ri_field_deg is not None:
        ri_val, dRI_dc, dRI_dd = ri_with_grad(
            surfaces, gaps, stop_info, ri_field_deg, f_number)
        ri_viol = ri_min - ri_val  # positive when violated
        if ri_viol > 0:
            merit += w_ri * ri_viol**2
            dM_dc += -2 * w_ri * ri_viol * dRI_dc
            dM_dd += -2 * w_ri * ri_viol * dRI_dd

    return {
        'merit': merit, 'dM_dc': dM_dc, 'dM_dd': dM_dd,
        'dM_dk': dM_dk, 'dM_dA4': dM_dA4,
        'S_total': S_total, 'efl': efl, 'sa': sa,
        'bfl': bfl, 'ttl': ttl,
        'cra': cra_val, 'ri': ri_val,
        'h_m': h_m, 'nu_m': nu_m,
        'dh_m_dc': dh_m_dc, 'dh_m_dd': dh_m_dd,
        'dEFL_dc': dEFL_dc, 'dEFL_dd': dEFL_dd,
    }


# ── Forward-only merit (for finite differences) ─────────────────────────────

def eval_merit_forward(surfaces, gaps, stop_info, semi_aperture, nu0_chief,
                       weights=None):
    """Evaluate merit function without computing gradients."""
    from parse_zmx import _ray_height_at_stop

    if isinstance(stop_info, int):
        stop_info = {'type': 'at_surface', 'surface_idx': stop_info}
    if weights is None:
        weights = np.ones(5)

    h_m, nu_m = ynu_trace(surfaces, gaps, semi_aperture, 0.0)

    # Find chief ray initial conditions
    h_a, nu_a = ynu_trace(surfaces, gaps, 0.0, nu0_chief)
    h_b, nu_b = ynu_trace(surfaces, gaps, 1.0, nu0_chief)
    h_stop_a = _ray_height_at_stop(h_a, nu_a, stop_info, gaps)
    h_stop_b = _ray_height_at_stop(h_b, nu_b, stop_info, gaps)
    h0_chief = -h_stop_a / (h_stop_b - h_stop_a)

    h_c, nu_c = ynu_trace(surfaces, gaps, h0_chief, nu0_chief)

    S_I, S_II, S_III, S_IV, S_V = seidel_coefficients(
        surfaces, h_m, nu_m, h_c, nu_c, 0.0, nu0_chief)

    S_total = np.array([S_I.sum(), S_II.sum(), S_III.sum(), S_IV.sum(), S_V.sum()])
    return np.sum(weights * S_total**2)


# ── Gradient verification ────────────────────────────────────────────────────

def verify_gradient(surfaces, gaps, stop_idx, semi_aperture, nu0_chief,
                    weights=None, eps=1e-7):
    """Verify analytic gradient against central finite differences."""
    K = len(surfaces)
    Kd = K - 1

    merit, dM_dc, dM_dd, S_total = merit_with_grad(
        surfaces, gaps, stop_idx, semi_aperture, nu0_chief, weights)

    print("=" * 70)
    print("GRADIENT VERIFICATION (central finite differences)")
    print("=" * 70)
    print(f"  eps = {eps:.0e}")
    print(f"  Merit = {merit:.10e}")
    print(f"  S_total = {S_total}")
    print()
    print(f"  {'Var':>8s} {'Analytic':>14s} {'FD':>14s} {'Abs Err':>12s} {'Rel Err':>12s}")
    print(f"  {'---':>8s} {'--------':>14s} {'--':>14s} {'-------':>12s} {'-------':>12s}")

    max_rel_err = 0

    # Curvatures
    for j in range(K):
        surfs_plus = list(surfaces)
        c, n, np_ = surfaces[j]
        surfs_plus[j] = (c + eps, n, np_)
        surfs_minus = list(surfaces)
        surfs_minus[j] = (c - eps, n, np_)

        m_plus = eval_merit_forward(surfs_plus, gaps, stop_idx, semi_aperture, nu0_chief, weights)
        m_minus = eval_merit_forward(surfs_minus, gaps, stop_idx, semi_aperture, nu0_chief, weights)

        fd = (m_plus - m_minus) / (2 * eps)
        abs_err = abs(dM_dc[j] - fd)
        rel_err = abs_err / max(abs(fd), abs(dM_dc[j]), 1e-15)
        max_rel_err = max(max_rel_err, rel_err)
        flag = " <<<" if rel_err > 1e-4 else ""
        print(f"  c[{j:2d}]  {dM_dc[j]:14.6e} {fd:14.6e} {abs_err:12.2e} {rel_err:12.2e}{flag}")

    # Gaps
    for j in range(Kd):
        gps_plus = list(gaps)
        d_gap, n_gap = gaps[j]
        gps_plus[j] = (d_gap + eps, n_gap)
        gps_minus = list(gaps)
        gps_minus[j] = (d_gap - eps, n_gap)

        m_plus = eval_merit_forward(surfaces, gps_plus, stop_idx, semi_aperture, nu0_chief, weights)
        m_minus = eval_merit_forward(surfaces, gps_minus, stop_idx, semi_aperture, nu0_chief, weights)

        fd = (m_plus - m_minus) / (2 * eps)
        abs_err = abs(dM_dd[j] - fd)
        rel_err = abs_err / max(abs(fd), abs(dM_dd[j]), 1e-15)
        max_rel_err = max(max_rel_err, rel_err)
        flag = " <<<" if rel_err > 1e-4 else ""
        print(f"  d[{j:2d}]  {dM_dd[j]:14.6e} {fd:14.6e} {abs_err:12.2e} {rel_err:12.2e}{flag}")

    print()
    if max_rel_err < 1e-4:
        print(f"  PASS: Max relative error = {max_rel_err:.2e}")
    else:
        print(f"  FAIL: Max relative error = {max_rel_err:.2e}")

    return max_rel_err


# ── Gradient verification for f-number mode ─────────────────────────────────

def verify_gradient_fnum(surfaces, gaps, stop_idx, f_number, nu0_chief,
                         weights=None, eps=1e-7):
    """Verify analytic gradients in f-number mode against central FD."""
    K = len(surfaces)
    Kd = K - 1

    (merit, dM_dc, dM_dd, S_total,
     efl, sa, h_m, dh_m_dc, dh_m_dd,
     dEFL_dc, dEFL_dd) = merit_with_grad_fnum(
        surfaces, gaps, stop_idx, f_number, nu0_chief, weights)

    ET, dET_dc, dET_dd = edge_thickness_with_grad(
        surfaces, gaps, h_m, dh_m_dc, dh_m_dd)

    print("=" * 70)
    print("GRADIENT VERIFICATION — f-number mode")
    print("=" * 70)
    print(f"  f/# = {f_number}, EFL = {efl:.4f}, sa = {sa:.4f}")
    print(f"  Merit = {merit:.10e}")
    print()

    def eval_fnum_forward(surfs, gps):
        """Forward-only merit in f# mode."""
        h_u, nu_u = ynu_trace(surfs, gps, 1.0, 0.0)
        efl_ = -1.0 / nu_u[-1]
        sa_ = efl_ / (2 * f_number)
        return eval_merit_forward(surfs, gps, stop_idx, sa_, nu0_chief, weights)

    def eval_et_forward(surfs, gps):
        """Forward-only edge thickness in f# mode."""
        h_u, nu_u = ynu_trace(surfs, gps, 1.0, 0.0)
        efl_ = -1.0 / nu_u[-1]
        sa_ = efl_ / (2 * f_number)
        h_m_, _ = ynu_trace(surfs, gps, sa_, 0.0)
        et = []
        for idx in range(len(surfs) - 1):
            s_f = sag_derivs(surfs[idx][0], abs(h_m_[idx]), 0.0, 0.0)[0]
            s_b = sag_derivs(surfs[idx + 1][0], abs(h_m_[idx + 1]), 0.0, 0.0)[0]
            n_gap = gps[idx][1]
            edge_min = 0.5 if n_gap > 1.001 else 0.0
            et.append(gps[idx][0] - s_f + s_b - edge_min)
        return et

    print("  --- Merit gradient ---")
    print(f"  {'Var':>8s} {'Analytic':>14s} {'FD':>14s} {'Abs Err':>12s} {'Rel Err':>12s}")
    print(f"  {'---':>8s} {'--------':>14s} {'--':>14s} {'-------':>12s} {'-------':>12s}")

    max_rel_err = 0

    for j in range(K):
        c, n, np_ = surfaces[j]
        sp = list(surfaces); sp[j] = (c + eps, n, np_)
        sm = list(surfaces); sm[j] = (c - eps, n, np_)
        fd = (eval_fnum_forward(sp, gaps) - eval_fnum_forward(sm, gaps)) / (2 * eps)
        abs_err = abs(dM_dc[j] - fd)
        rel_err = abs_err / max(abs(fd), abs(dM_dc[j]), 1e-15)
        max_rel_err = max(max_rel_err, rel_err)
        flag = " <<<" if rel_err > 1e-4 else ""
        print(f"  c[{j:2d}]  {dM_dc[j]:14.6e} {fd:14.6e} {abs_err:12.2e} {rel_err:12.2e}{flag}")

    for j in range(Kd):
        d_gap, n_gap = gaps[j]
        gp = list(gaps); gp[j] = (d_gap + eps, n_gap)
        gm = list(gaps); gm[j] = (d_gap - eps, n_gap)
        fd = (eval_fnum_forward(surfaces, gp) - eval_fnum_forward(surfaces, gm)) / (2 * eps)
        abs_err = abs(dM_dd[j] - fd)
        rel_err = abs_err / max(abs(fd), abs(dM_dd[j]), 1e-15)
        max_rel_err = max(max_rel_err, rel_err)
        flag = " <<<" if rel_err > 1e-4 else ""
        print(f"  d[{j:2d}]  {dM_dd[j]:14.6e} {fd:14.6e} {abs_err:12.2e} {rel_err:12.2e}{flag}")

    print()
    print("  --- Edge thickness gradient ---")
    print(f"  {'Var':>8s} {'ET':>4s} {'Analytic':>14s} {'FD':>14s} {'Rel Err':>12s}")

    for et_idx in range(Kd):
        for j in range(K):
            c, n, np_ = surfaces[j]
            sp = list(surfaces); sp[j] = (c + eps, n, np_)
            sm = list(surfaces); sm[j] = (c - eps, n, np_)
            fd = (eval_et_forward(sp, gaps)[et_idx] - eval_et_forward(sm, gaps)[et_idx]) / (2 * eps)
            anal = dET_dc[et_idx, j]
            rel_err = abs(anal - fd) / max(abs(fd), abs(anal), 1e-15)
            max_rel_err = max(max_rel_err, rel_err)
            flag = " <<<" if rel_err > 1e-4 else ""
            print(f"  c[{j:2d}]  ET{et_idx} {anal:14.6e} {fd:14.6e} {rel_err:12.2e}{flag}")

        for j in range(Kd):
            d_gap, n_gap = gaps[j]
            gp = list(gaps); gp[j] = (d_gap + eps, n_gap)
            gm = list(gaps); gm[j] = (d_gap - eps, n_gap)
            fd = (eval_et_forward(surfaces, gp)[et_idx] - eval_et_forward(surfaces, gm)[et_idx]) / (2 * eps)
            anal = dET_dd[et_idx, j]
            rel_err = abs(anal - fd) / max(abs(fd), abs(anal), 1e-15)
            max_rel_err = max(max_rel_err, rel_err)
            flag = " <<<" if rel_err > 1e-4 else ""
            print(f"  d[{j:2d}]  ET{et_idx} {anal:14.6e} {fd:14.6e} {rel_err:12.2e}{flag}")

    print()
    if max_rel_err < 1e-4:
        print(f"  PASS: Max relative error = {max_rel_err:.2e}")
    else:
        print(f"  FAIL: Max relative error = {max_rel_err:.2e}")

    return max_rel_err


# ── EFL and its gradient (for constraint) ────────────────────────────────────

def efl_with_grad(surfaces, gaps, semi_aperture=None):
    """
    Compute effective focal length and its gradient w.r.t. c and d.

    EFL = -h0 / nu[-1]. Any h0 gives the same result (linearity).
    semi_aperture is optional — defaults to unit ray.
    """
    h0 = semi_aperture if semi_aperture is not None else 1.0
    h_m, nu_m, dh_m_dc, dnu_m_dc, dh_m_dd, dnu_m_dd = ynu_trace_with_grad(
        surfaces, gaps, h0, 0.0)

    nu_final = nu_m[-1]
    efl = -h0 / nu_final

    dEFL_dc = h0 / nu_final**2 * dnu_m_dc[-1]
    dEFL_dd = h0 / nu_final**2 * dnu_m_dd[-1]

    return efl, dEFL_dc, dEFL_dd


# ── CRA with gradient ────────────────────────────────────────────────────────

def cra_with_grad(surfaces, gaps, stop_info, field_angle_deg):
    """
    Compute chief ray angle at the image plane and its gradient.

    CRA = arctan(nu_c[-1]) in radians, returned in degrees.
    Gradients are in degrees per unit change in c or d.

    Returns
    -------
    cra_deg : float, CRA in degrees
    dCRA_dc : (K,) gradient w.r.t. curvatures
    dCRA_dd : (K-1,) gradient w.r.t. gaps
    """
    nu0_chief = np.tan(np.radians(field_angle_deg))
    h0, h_c, nu_c, dhc_dc, dnuc_dc, dhc_dd, dnuc_dd = (
        chief_ray_with_grad(surfaces, gaps, stop_info, nu0_chief))

    nu_last = nu_c[-1]
    cra_rad = np.arctan(nu_last)

    # d(arctan(x))/dx = 1/(1+x^2), then convert to degrees
    d_atan = np.degrees(1.0) / (1.0 + nu_last**2)
    dCRA_dc = d_atan * dnuc_dc[-1]
    dCRA_dd = d_atan * dnuc_dd[-1]

    return np.degrees(cra_rad), dCRA_dc, dCRA_dd


# ── Geometry-based clear semi-diameters with gradient ────────────────────────

def _exact_et_at_h(c_f, c_b, d, h):
    """Edge thickness at height h using exact spherical sag."""
    sf, _, _, _, _ = sag_derivs(c_f, h)
    sb, _, _, _, _ = sag_derivs(c_b, h)
    return d - sf + sb


def _max_h_for_gap_exact(c_f, c_b, d):
    """Find max h where exact ET >= 0, also limited by sphere radii.
    Returns h_max via bisection."""
    LARGE_H = 50.0
    if d <= 0:
        return 0.0
    # h can't exceed sphere radius of either surface
    h_limit = LARGE_H
    if abs(c_f) > 1e-10:
        h_limit = min(h_limit, 0.999 / abs(c_f))
    if abs(c_b) > 1e-10:
        h_limit = min(h_limit, 0.999 / abs(c_b))
    # Check if ET is positive at the limit
    if _exact_et_at_h(c_f, c_b, d, h_limit) >= 0:
        return h_limit
    # Bisect
    h_lo, h_hi = 0.0, h_limit
    for _ in range(60):
        h_mid = (h_lo + h_hi) / 2
        if _exact_et_at_h(c_f, c_b, d, h_mid) >= 0:
            h_lo = h_mid
        else:
            h_hi = h_mid
    return h_lo


def geometry_sd_with_grad(surfaces, gaps):
    """
    Compute max clear semi-diameter per surface from edge thickness geometry,
    using exact spherical sag (not paraxial approximation).

    For each gap j, finds h_max where ET(h) = d - sag(c_f, h) + sag(c_b, h) = 0
    via bisection, also capped at sphere radii.

    Gradients via implicit function theorem:
        ET(h_max, c_f, c_b, d) = 0 defines h_max implicitly.
        dh_max/dx = -(dET/dx) / (dET/dh)  at h = h_max.

    SD(i) = min(h_max from adjacent gaps).
    Gradient flows through the limiting gap.

    Returns
    -------
    sd : (K,) clear semi-diameter per surface
    dsd_dc : (K, K) gradient w.r.t. curvatures
    dsd_dd : (K, K-1) gradient w.r.t. gap thicknesses
    """
    K = len(surfaces)
    Kd = K - 1
    LARGE = 1e6

    h_max_gap = np.full(Kd, LARGE)
    dh_gap_dc = np.zeros((Kd, K))
    dh_gap_dd = np.zeros((Kd, Kd))

    for j in range(Kd):
        c_f = surfaces[j][0]
        c_b = surfaces[j + 1][0]
        d_gap = gaps[j][0]

        h_max = _max_h_for_gap_exact(c_f, c_b, d_gap)
        if h_max >= LARGE or h_max <= 1e-10:
            continue

        h_max_gap[j] = h_max

        # Check if limited by sphere radius or by ET=0
        h_sphere = 50.0
        if abs(c_f) > 1e-10:
            h_sphere = min(h_sphere, 0.999 / abs(c_f))
        if abs(c_b) > 1e-10:
            h_sphere = min(h_sphere, 0.999 / abs(c_b))

        limited_by_sphere_f = (abs(c_f) > 1e-10 and
                               abs(h_max - 0.999 / abs(c_f)) < 1e-6)
        limited_by_sphere_b = (abs(c_b) > 1e-10 and
                               abs(h_max - 0.999 / abs(c_b)) < 1e-6)

        if limited_by_sphere_f:
            # h_max = 0.999 / |c_f|, dh_max/dc_f = -0.999*sign(c_f) / c_f^2
            dh_gap_dc[j, j] = -0.999 * np.sign(c_f) / c_f**2
            # No dependence on c_b or d
        elif limited_by_sphere_b:
            # h_max = 0.999 / |c_b|
            dh_gap_dc[j, j + 1] = -0.999 * np.sign(c_b) / c_b**2
        else:
            # Limited by ET=0: implicit differentiation
            _, dsf_dc, dsf_dh, _, _ = sag_derivs(c_f, h_max)
            _, dsb_dc, dsb_dh, _, _ = sag_derivs(c_b, h_max)

            dET_dh = -dsf_dh + dsb_dh
            if abs(dET_dh) < 1e-15:
                continue

            dh_gap_dc[j, j] = dsf_dc / dET_dh
            dh_gap_dc[j, j + 1] = -dsb_dc / dET_dh
            dh_gap_dd[j, j] = -1.0 / dET_dh

    # SD per surface = min of adjacent gap limits
    sd = np.full(K, LARGE)
    dsd_dc = np.zeros((K, K))
    dsd_dd = np.zeros((K, Kd))

    for i in range(K):
        limiting_gap = -1
        min_val = LARGE

        if i > 0 and h_max_gap[i - 1] < min_val:
            min_val = h_max_gap[i - 1]
            limiting_gap = i - 1
        if i < Kd and h_max_gap[i] < min_val:
            min_val = h_max_gap[i]
            limiting_gap = i

        sd[i] = min_val
        if limiting_gap >= 0:
            dsd_dc[i] = dh_gap_dc[limiting_gap]
            dsd_dd[i] = dh_gap_dd[limiting_gap]

    return sd, dsd_dc, dsd_dd


# ── 1D vignetting with gradient ─────────────────────────────────────────────

def vignetting_1d_with_grad(h_m, h_c, sd,
                             dh_m_dc, dh_m_dd,
                             dh_c_dc, dh_c_dd,
                             dsd_dc, dsd_dd):
    """
    Compute 1D tangential vignetting factor and its gradient.

    V = (rho_hi - rho_lo) / 2

    rho_hi = min over surfaces of (SD(i) - h_c(i)) / h_m(i)
    rho_lo = max over surfaces of (-SD(i) - h_c(i)) / h_m(i)
    (with sign flip when h_m(i) < 0)

    Gradient flows through the limiting surface at each bound.

    Returns
    -------
    V : vignetting factor in [0, 1]
    dV_dc : (K,) gradient w.r.t. curvatures
    dV_dd : (K-1,) gradient w.r.t. gaps
    """
    K = len(h_m)
    Kd = K - 1

    rho_lo = -1.0
    rho_hi = 1.0
    lim_lo = -1
    lim_hi = -1

    for i in range(K):
        if abs(h_m[i]) < 1e-15:
            continue

        lo_i = (-sd[i] - h_c[i]) / h_m[i]
        hi_i = (sd[i] - h_c[i]) / h_m[i]
        if h_m[i] < 0:
            lo_i, hi_i = hi_i, lo_i

        if lo_i > rho_lo:
            rho_lo = lo_i
            lim_lo = i
        if hi_i < rho_hi:
            rho_hi = hi_i
            lim_hi = i

    clamped_lo = max(rho_lo, -1.0)
    clamped_hi = min(rho_hi, 1.0)
    V = max(0.0, clamped_hi - clamped_lo) / 2.0

    dV_dc = np.zeros(K)
    dV_dd = np.zeros(Kd)

    if V <= 0.0:
        return V, dV_dc, dV_dd

    # Gradient of rho_hi if limited by a surface (not pupil edge)
    drho_hi_dc = np.zeros(K)
    drho_hi_dd = np.zeros(Kd)
    if lim_hi >= 0 and rho_hi <= 1.0:
        i = lim_hi
        hm = h_m[i]
        # rho_hi = (SD(i) - h_c(i)) / h_m(i)  [or swapped if h_m<0]
        if hm > 0:
            num = sd[i] - h_c[i]
        else:
            num = -sd[i] - h_c[i]

        # d(num/hm)/dx = (dnum/dx * hm - num * dhm/dx) / hm^2
        if hm > 0:
            dnum_dc = dsd_dc[i] - dh_c_dc[i]
            dnum_dd = dsd_dd[i] - dh_c_dd[i]
        else:
            dnum_dc = -dsd_dc[i] - dh_c_dc[i]
            dnum_dd = -dsd_dd[i] - dh_c_dd[i]

        drho_hi_dc = (dnum_dc * hm - num * dh_m_dc[i]) / hm**2
        drho_hi_dd = (dnum_dd * hm - num * dh_m_dd[i]) / hm**2

    # Gradient of rho_lo if limited by a surface (not pupil edge)
    drho_lo_dc = np.zeros(K)
    drho_lo_dd = np.zeros(Kd)
    if lim_lo >= 0 and rho_lo >= -1.0:
        i = lim_lo
        hm = h_m[i]
        if hm > 0:
            num = -sd[i] - h_c[i]
        else:
            num = sd[i] - h_c[i]

        if hm > 0:
            dnum_dc = -dsd_dc[i] - dh_c_dc[i]
            dnum_dd = -dsd_dd[i] - dh_c_dd[i]
        else:
            dnum_dc = dsd_dc[i] - dh_c_dc[i]
            dnum_dd = dsd_dd[i] - dh_c_dd[i]

        drho_lo_dc = (dnum_dc * hm - num * dh_m_dc[i]) / hm**2
        drho_lo_dd = (dnum_dd * hm - num * dh_m_dd[i]) / hm**2

    # V = (rho_hi - rho_lo) / 2
    dV_dc = (drho_hi_dc - drho_lo_dc) / 2.0
    dV_dd = (drho_hi_dd - drho_lo_dd) / 2.0

    return V, dV_dc, dV_dd


def _ri_forward(surfaces, gaps, stop_info, field_angle_deg, f_number):
    """Compute RI value only (no gradients), used for finite differences."""
    theta_ref = 0.1

    # Marginal ray
    h_u, nu_u = ynu_trace(surfaces, gaps, 1.0, 0.0)
    efl = -1.0 / nu_u[-1]
    sa = efl / (2 * f_number)
    h_m, _ = ynu_trace(surfaces, gaps, sa, 0.0)

    # Geometry-based SD (exact)
    sd, _, _ = geometry_sd_with_grad(surfaces, gaps)

    # 1D vignetting at reference
    nu0_ref = np.tan(np.radians(theta_ref))
    from parse_zmx import find_chief_ray_initial
    h0_ref, nu0_ref_val = find_chief_ray_initial(
        surfaces, gaps, stop_info, theta_ref)
    h_c_ref, _ = ynu_trace(surfaces, gaps, h0_ref, nu0_ref_val)
    V_ref, _, _, _ = _vignetting_1d_forward(h_m, h_c_ref, sd)
    cos4_ref = np.cos(np.radians(theta_ref))**4
    ri_ref = cos4_ref * V_ref

    # Target field
    h0, nu0_val = find_chief_ray_initial(
        surfaces, gaps, stop_info, field_angle_deg)
    h_c, _ = ynu_trace(surfaces, gaps, h0, nu0_val)
    V, _, _, _ = _vignetting_1d_forward(h_m, h_c, sd)
    cos4 = np.cos(np.radians(field_angle_deg))**4
    ri_raw = cos4 * V

    if ri_ref < 1e-15:
        return 0.0
    return ri_raw / ri_ref


def _vignetting_1d_forward(h_m, h_c, sd):
    """1D vignetting value only (no gradients)."""
    K = len(h_m)
    rho_lo, rho_hi = -1.0, 1.0
    lim_lo, lim_hi = -1, -1
    for i in range(K):
        if abs(h_m[i]) < 1e-15:
            continue
        lo_i = (-sd[i] - h_c[i]) / h_m[i]
        hi_i = (sd[i] - h_c[i]) / h_m[i]
        if h_m[i] < 0:
            lo_i, hi_i = hi_i, lo_i
        if lo_i > rho_lo:
            rho_lo = lo_i
            lim_lo = i
        if hi_i < rho_hi:
            rho_hi = hi_i
            lim_hi = i
    rho_lo = max(rho_lo, -1.0)
    rho_hi = min(rho_hi, 1.0)
    V = max(0.0, rho_hi - rho_lo) / 2.0
    return V, rho_lo, rho_hi, (lim_lo, lim_hi)


def ri_with_grad(surfaces, gaps, stop_info, field_angle_deg, f_number):
    """
    Compute relative illumination at a field angle with gradients.

    Uses exact-sag geometry SD for vignetting. Gradients computed by
    central finite differences (robust to the discontinuities in the
    min/max vignetting logic).

    Returns
    -------
    ri_rel : float, relative RI
    dRI_dc : (K,) gradient w.r.t. curvatures
    dRI_dd : (K-1,) gradient w.r.t. gaps
    """
    K = len(surfaces)
    Kd = K - 1
    eps = 1e-7

    ri_val = _ri_forward(surfaces, gaps, stop_info, field_angle_deg, f_number)

    # Finite difference gradients
    dRI_dc = np.zeros(K)
    for j in range(K):
        c, n, np_ = surfaces[j]
        sp = list(surfaces); sp[j] = (c + eps, n, np_)
        sm = list(surfaces); sm[j] = (c - eps, n, np_)
        ri_p = _ri_forward(sp, gaps, stop_info, field_angle_deg, f_number)
        ri_m = _ri_forward(sm, gaps, stop_info, field_angle_deg, f_number)
        dRI_dc[j] = (ri_p - ri_m) / (2 * eps)

    dRI_dd = np.zeros(Kd)
    for j in range(Kd):
        d, n_gap = gaps[j]
        gp = list(gaps); gp[j] = (d + eps, n_gap)
        gm = list(gaps); gm[j] = (d - eps, n_gap)
        ri_p = _ri_forward(surfaces, gp, stop_info, field_angle_deg, f_number)
        ri_m = _ri_forward(surfaces, gm, stop_info, field_angle_deg, f_number)
        dRI_dd[j] = (ri_p - ri_m) / (2 * eps)

    return ri_val, dRI_dc, dRI_dd


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Test on the ZMX file
    zmx_data = parse_zmx('sc_dbga1_opt2.zmx')
    surfaces, gaps, stop_info = build_system_from_zmx(zmx_data)
    K = len(surfaces)
    semi_aperture = zmx_data['enpd'] / 2.0

    # Compute field angle
    max_field_value = max(zmx_data['fields_y'])
    field_type = zmx_data.get('field_type', 0)
    if field_type == 0:
        max_field_angle = max_field_value
    elif field_type == 2:
        h_m_tmp, nu_m_tmp = ynu_trace(surfaces, gaps, semi_aperture, 0.0)
        efl_tmp = -semi_aperture / (nu_m_tmp[-1])
        max_field_angle = np.degrees(np.arctan(max_field_value / efl_tmp))
    else:
        max_field_angle = max_field_value

    nu0_chief = np.tan(np.radians(max_field_angle))

    print(f"System: {K} surfaces, stop info: {stop_info}")
    print(f"Semi-aperture: {semi_aperture} mm")
    print(f"Max field angle: {max_field_angle:.4f} deg")
    print(f"nu0_chief: {nu0_chief:.6f}")
    print()

    # Compute merit and gradient
    merit, dM_dc, dM_dd, S_total = merit_with_grad(
        surfaces, gaps, stop_info, semi_aperture, nu0_chief)

    print(f"Merit = {merit:.6e}")
    print(f"S_total = [{', '.join(f'{s:.6f}' for s in S_total)}]")
    print()

    print("GRADIENT dM/dc:")
    for i in range(K):
        c, n, np_ = surfaces[i]
        print(f"  c[{i:2d}] = {c:+.8f}  dM/dc = {dM_dc[i]:+.6e}")
    print()

    print("GRADIENT dM/dd:")
    for i in range(K - 1):
        d_gap, n_gap = gaps[i]
        print(f"  d[{i:2d}] = {d_gap:8.4f}  dM/dd = {dM_dd[i]:+.6e}")
    print()

    # EFL gradient
    efl, dEFL_dc, dEFL_dd = efl_with_grad(surfaces, gaps, semi_aperture)
    print(f"EFL = {efl:.4f} mm")
    print("GRADIENT dEFL/dc:")
    for i in range(K):
        print(f"  c[{i:2d}]  dEFL/dc = {dEFL_dc[i]:+.6e}")
    print()

    # Verify
    print()
    verify_gradient(surfaces, gaps, stop_info, semi_aperture, nu0_chief)

    # Also test on a simple singlet
    print("\n\n")
    print("=" * 70)
    print("SINGLET TEST")
    print("=" * 70)
    singlet_surfaces = [
        (0.02, 1.0, 1.5),
        (-0.01, 1.5, 1.0),
    ]
    singlet_gaps = [(5.0, 1.5)]
    singlet_stop = 0
    singlet_sa = 10.0
    singlet_nu0c = np.tan(np.radians(2.0))

    merit_s, dM_dc_s, dM_dd_s, S_s = merit_with_grad(
        singlet_surfaces, singlet_gaps, singlet_stop, singlet_sa, singlet_nu0c)
    print(f"Merit = {merit_s:.6e}")
    print(f"dM/dc = {dM_dc_s}")
    print(f"dM/dd = {dM_dd_s}")
    print()
    verify_gradient(singlet_surfaces, singlet_gaps, singlet_stop,
                    singlet_sa, singlet_nu0c)

    # ── f-number mode tests ──
    print("\n\n")
    print("=" * 70)
    print("F-NUMBER MODE — 2-singlet test")
    print("=" * 70)
    doublet_surfaces = [
        (0.02, 1.0, 1.5),
        (-0.01, 1.5, 1.0),
        (0.015, 1.0, 1.5),
        (-0.005, 1.5, 1.0),
    ]
    doublet_gaps = [(5.0, 1.5), (10.0, 1.0), (5.0, 1.5)]
    doublet_stop = 0
    doublet_fnum = 4.0
    doublet_nu0c = np.tan(np.radians(3.0))

    verify_gradient_fnum(doublet_surfaces, doublet_gaps, doublet_stop,
                         doublet_fnum, doublet_nu0c)

    # ── Full merit (Seidel + BFL + TTL) verification ──
    print("\n\n")
    print("=" * 70)
    print("FULL MERIT (Seidel + BFL + TTL) — 2-singlet test")
    print("=" * 70)

    bfl_target = 30.0
    ttl_target = 50.0
    w_bfl = 0.1
    w_ttl = 0.05

    res = full_merit_with_grad(
        doublet_surfaces, doublet_gaps, doublet_stop, doublet_nu0c,
        f_number=doublet_fnum,
        bfl_target=bfl_target, w_bfl=w_bfl,
        ttl_target=ttl_target, w_ttl=w_ttl)

    print(f"  Merit = {res['merit']:.10e}")
    print(f"  BFL   = {res['bfl']:.4f} mm (target {bfl_target})")
    print(f"  TTL   = {res['ttl']:.4f} mm (target {ttl_target})")
    print()

    def eval_full_forward(surfs, gps):
        h_u, nu_u = ynu_trace(surfs, gps, 1.0, 0.0)
        efl_ = -1.0 / nu_u[-1]
        sa_ = efl_ / (2 * doublet_fnum)
        h_m_, nu_m_ = ynu_trace(surfs, gps, sa_, 0.0)
        bfl_ = -h_m_[-1] / nu_m_[-1]
        ttl_ = sum(g[0] for g in gps) + bfl_

        h_a, nu_a = ynu_trace(surfs, gps, 0.0, doublet_nu0c)
        h_b, nu_b = ynu_trace(surfs, gps, 1.0, doublet_nu0c)
        h0c = -h_a[doublet_stop] / (h_b[doublet_stop] - h_a[doublet_stop])
        h_c, nu_c = ynu_trace(surfs, gps, h0c, doublet_nu0c)

        S_I, S_II, S_III, S_IV, S_V = seidel_coefficients(
            surfs, h_m_, nu_m_, h_c, nu_c, 0.0, doublet_nu0c)
        S_tot = np.array([S_I.sum(), S_II.sum(), S_III.sum(), S_IV.sum(), S_V.sum()])
        m = float(np.sum(S_tot**2))
        m += w_bfl * (bfl_ - bfl_target)**2
        m += w_ttl * (ttl_ - ttl_target)**2
        return m

    print(f"  {'Var':>8s} {'Analytic':>14s} {'FD':>14s} {'Rel Err':>12s}")
    max_rel = 0
    K = len(doublet_surfaces)
    Kd = K - 1
    eps = 1e-7
    for j in range(K):
        c, n, np_ = doublet_surfaces[j]
        sp = list(doublet_surfaces); sp[j] = (c + eps, n, np_)
        sm = list(doublet_surfaces); sm[j] = (c - eps, n, np_)
        fd = (eval_full_forward(sp, doublet_gaps) - eval_full_forward(sm, doublet_gaps)) / (2*eps)
        anal = res['dM_dc'][j]
        rel = abs(anal - fd) / max(abs(fd), abs(anal), 1e-15)
        max_rel = max(max_rel, rel)
        flag = " <<<" if rel > 1e-4 else ""
        print(f"  c[{j:2d}]  {anal:14.6e} {fd:14.6e} {rel:12.2e}{flag}")

    for j in range(Kd):
        d_gap, n_gap = doublet_gaps[j]
        gp = list(doublet_gaps); gp[j] = (d_gap + eps, n_gap)
        gm = list(doublet_gaps); gm[j] = (d_gap - eps, n_gap)
        fd = (eval_full_forward(doublet_surfaces, gp) - eval_full_forward(doublet_surfaces, gm)) / (2*eps)
        anal = res['dM_dd'][j]
        rel = abs(anal - fd) / max(abs(fd), abs(anal), 1e-15)
        max_rel = max(max_rel, rel)
        flag = " <<<" if rel > 1e-4 else ""
        print(f"  d[{j:2d}]  {anal:14.6e} {fd:14.6e} {rel:12.2e}{flag}")

    print()
    if max_rel < 1e-4:
        print(f"  PASS: Max relative error = {max_rel:.2e}")
    else:
        print(f"  FAIL: Max relative error = {max_rel:.2e}")


if __name__ == "__main__":
    main()
