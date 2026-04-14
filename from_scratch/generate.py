"""
Generate an initial lens design from scratch (no existing structure needed).

Given only:
  - Number of singlet elements (N)
  - Refractive index (n_d, fixed for all elements)
  - EFL target, f-number, field angle
  - Optional: glass thicknesses, TTL budget

Produces a valid starting design (surfaces, gaps) that can be fed
directly into the optimizer.

Strategy:
  1. Split total power equally across N elements.
  2. Bend each element as equi-convex (shape factor q = 0).
  3. Space elements equally within a TTL budget.
  4. Estimate physical CA per element from sag budget.
  5. Predict thin-lens RI from beam footprint vs CA.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parse_zmx import ynu_trace, find_chief_ray_initial


def _sag(c, h):
    """Signed spherical sag for curvature c at height h."""
    if abs(c) < 1e-15:
        return 0.0
    arg = 1 - c**2 * h**2
    if arg <= 0:
        return c * h**2  # paraxial fallback
    return c * h**2 / (1 + np.sqrt(arg))


def _thin_lens_ca(c1, c2, d_glass, n_d, e_min=0.0):
    """
    Estimate the max clear aperture of a singlet with curvatures c1, c2.

    Edge thickness at height h:
        ET(h) = d_glass - sag(c1, h) + sag(c2, h)

    Finds the max h where ET(h) >= e_min using bisection.
    """
    ac1, ac2 = abs(c1), abs(c2)
    if ac1 < 1e-12 and ac2 < 1e-12:
        return 1e6  # flat lens, unlimited CA

    # h can't exceed either sphere radius
    h_limits = []
    if ac1 > 1e-12:
        h_limits.append(0.999 / ac1)
    if ac2 > 1e-12:
        h_limits.append(0.999 / ac2)
    h_sphere = min(h_limits)

    def edge_thickness(h):
        return d_glass - _sag(c1, h) + _sag(c2, h)

    # Check if ET is still above e_min at sphere limit
    if edge_thickness(h_sphere) >= e_min:
        return h_sphere

    # Check that ET at h=0 is valid (center thickness > e_min)
    if edge_thickness(0.0) < e_min:
        return 0.0

    # Bisect to find h where ET = e_min
    h_lo, h_hi = 0.0, h_sphere
    for _ in range(60):
        h_mid = (h_lo + h_hi) / 2
        if edge_thickness(h_mid) >= e_min:
            h_lo = h_mid
        else:
            h_hi = h_mid
    return h_lo


def _thin_lens_ri(surfaces, gaps, stop_idx, f_number, field_angle_deg, ca_list):
    """
    Predict RI at the given field angle using thin-lens CA limits.

    Traces marginal and chief rays, then checks vignetting at each
    element using the physical CA estimate.

    Returns
    -------
    ri : float, estimated relative illumination
    v_field : float, vignetting factor at field
    beam_info : list of dicts per surface with h_m, h_c, ca, margin
    """
    K = len(surfaces)

    # Marginal ray (f-number coupled)
    h_u, nu_u = ynu_trace(surfaces, gaps, 1.0, 0.0)
    efl = -1.0 / nu_u[-1]
    sa = efl / (2 * f_number)
    h_m, nu_m = ynu_trace(surfaces, gaps, sa, 0.0)

    # Chief ray at field
    h0_c, nu0_c = find_chief_ray_initial(
        surfaces, gaps, stop_idx, field_angle_deg)
    h_c, nu_c = ynu_trace(surfaces, gaps, h0_c, nu0_c)

    # Chief ray at reference (near on-axis)
    theta_ref = 0.1
    h0_ref, nu0_ref = find_chief_ray_initial(
        surfaces, gaps, stop_idx, theta_ref)
    h_c_ref, _ = ynu_trace(surfaces, gaps, h0_ref, nu0_ref)

    # 1D vignetting using CA as semi-diameter
    def vignetting_1d(h_marginal, h_chief, sd):
        rho_lo, rho_hi = -1.0, 1.0
        for i in range(K):
            if abs(h_marginal[i]) < 1e-15:
                continue
            lo_i = (-sd[i] - h_chief[i]) / h_marginal[i]
            hi_i = (sd[i] - h_chief[i]) / h_marginal[i]
            if h_marginal[i] < 0:
                lo_i, hi_i = hi_i, lo_i
            rho_lo = max(rho_lo, lo_i)
            rho_hi = min(rho_hi, hi_i)
        rho_lo = max(rho_lo, -1.0)
        rho_hi = min(rho_hi, 1.0)
        return max(0.0, rho_hi - rho_lo) / 2.0

    # SD array: each surface pair in an element shares the same CA
    sd = np.zeros(K)
    for i in range(K):
        elem_idx = i // 2  # which element this surface belongs to
        sd[i] = ca_list[elem_idx]

    v_field = vignetting_1d(h_m, h_c, sd)
    v_ref = vignetting_1d(h_m, h_c_ref, sd)

    cos4_field = np.cos(np.radians(field_angle_deg))**4
    cos4_ref = np.cos(np.radians(theta_ref))**4

    if v_ref < 1e-15:
        ri = 0.0
    else:
        ri = (cos4_field * v_field) / (cos4_ref * v_ref)

    # Per-surface beam info
    beam_info = []
    for i in range(K):
        elem_idx = i // 2
        beam_info.append({
            'h_m': h_m[i],
            'h_c': h_c[i],
            'h_max': abs(h_m[i]) + abs(h_c[i]),
            'ca': ca_list[elem_idx],
            'margin': ca_list[elem_idx] - (abs(h_m[i]) + abs(h_c[i])),
        })

    return ri, v_field, beam_info


def _power_layout(n_elements):
    """
    Return sign pattern for each element's power.

    Wide-angle fast lens layout (retrofocus-influenced):
      - Negative front elements: diverge beam, widen FOV acceptance
      - Positive middle elements: do the focusing
      - Negative near stop: correct Petzval / flatten field
      - Positive rear: final convergence
      - Negative last (if enough elements): fine-tune field curvature

    Returns list of +1/-1 per element.
    """
    if n_elements == 1:
        return [+1]
    elif n_elements == 2:
        return [-1, +1]
    elif n_elements == 3:
        return [-1, +1, +1]
    elif n_elements == 4:
        return [-1, +1, +1, -1]
    elif n_elements == 5:
        return [-1, +1, +1, -1, +1]
    elif n_elements == 6:
        return [-1, -1, +1, +1, -1, +1]
    elif n_elements == 7:
        return [-1, -1, +1, +1, -1, +1, +1]
    else:  # 8+
        # --, ++, -, ++, -
        signs = [-1, -1, +1, +1, -1, +1, +1, -1]
        # Extend with alternating +/- if N > 8
        while len(signs) < n_elements:
            signs.append(+1 if len(signs) % 2 == 0 else -1)
        return signs[:n_elements]


def _distribute_power(phi_total, signs, neg_strength=0.5):
    """
    Distribute total power across elements with given sign pattern.

    neg_strength : fraction of phi_total carried by negative elements
        e.g. 0.5 means negative group total = -0.5 * phi_total,
        so positive group must carry 1.5 * phi_total.

    Each group's power is split equally among its members.
    """
    n_pos = sum(1 for s in signs if s > 0)
    n_neg = sum(1 for s in signs if s < 0)

    if n_neg == 0:
        # All positive
        phi_each = phi_total / n_pos
        return [phi_each] * len(signs)

    phi_neg_total = -neg_strength * phi_total
    phi_pos_total = phi_total - phi_neg_total  # = (1 + neg_strength) * phi_total

    phi_pos = phi_pos_total / n_pos
    phi_neg = phi_neg_total / n_neg  # negative value

    return [phi_pos if s > 0 else phi_neg for s in signs]


def generate_initial_design(n_elements, n_d, efl_target, f_number,
                            field_angle_deg, d_glass=None, d_air=None,
                            ttl_factor=1.5, neg_strength=0.5, signs=None):
    """
    Generate an N-singlet starting design from first principles.

    Parameters
    ----------
    signs : list of +1/-1, optional
        Power sign per element. If None, uses default layout.
    d_air : list of float, optional
        Air gap thicknesses. If None, auto-computed.
    neg_strength : float
        Fraction of phi_total carried by negative elements.

    Returns
    -------
    surfaces, gaps, design_vector, meta
    """
    n_air = 1.0

    # --- Power layout ---
    if signs is None:
        signs = _power_layout(n_elements)
    phi_total = 1.0 / efl_target
    phi_list = _distribute_power(phi_total, signs, neg_strength)

    # --- Equi-convex/concave bending per element ---
    dn = n_d - n_air
    curvatures = []
    c_half_list = []
    for i in range(n_elements):
        # phi = (n-1)*(c_front - c_back)
        # equi-convex: c_front = -c_back = phi / (2*(n-1))
        c_half = phi_list[i] / (2.0 * dn)
        c_half_list.append(c_half)
        curvatures.append(+c_half)   # front
        curvatures.append(-c_half)   # back

    # --- Initial glass thickness guess ---
    if d_glass is None:
        d_g = np.clip(efl_target * 0.05, 2.0, 10.0)
        d_glass_list = [d_g] * n_elements
    elif np.isscalar(d_glass):
        d_glass_list = [float(d_glass)] * n_elements
    else:
        d_glass_list = list(d_glass)

    # --- Air gap spacing ---
    if d_air is not None:
        d_air_list = list(d_air)
    else:
        ttl_budget = efl_target * ttl_factor
        total_glass = sum(d_glass_list)
        bfl_estimate = efl_target * 0.5
        air_budget = max(ttl_budget - total_glass - bfl_estimate,
                         n_elements * 2.0)
        if n_elements > 1:
            d_air_each = air_budget / (n_elements - 1)
            d_air_each = max(d_air_each, 2.0)
            d_air_list = [d_air_each] * (n_elements - 1)
        else:
            d_air_list = []

    # --- Build initial system ---
    def _build(d_glass_list, d_air_list):
        surfaces = []
        gaps = []
        air_idx = 0
        for i in range(n_elements):
            surfaces.append((curvatures[2 * i], n_air, n_d))
            surfaces.append((curvatures[2 * i + 1], n_d, n_air))
            gaps.append((d_glass_list[i], n_d))
            if i < n_elements - 1:
                gaps.append((d_air_list[air_idx], n_air))
                air_idx += 1
        return surfaces, gaps

    # --- Size glass for adequate CA ---
    # Trace rays with initial guess, check beam footprint, increase
    # glass thickness if CA is too small
    surfaces, gaps = _build(d_glass_list, d_air_list)

    # Compute beam footprint
    h_u, nu_u = ynu_trace(surfaces, gaps, 1.0, 0.0)
    efl_actual = -1.0 / nu_u[-1]
    sa = efl_actual / (2 * f_number)
    h_m, _ = ynu_trace(surfaces, gaps, sa, 0.0)

    # Stop between L3 and L4 (surface index 5 for N=6)
    stop_idx = min(2 * (n_elements // 2) - 1, len(surfaces) - 1)
    h0_c, nu0_c = find_chief_ray_initial(surfaces, gaps, stop_idx,
                                          field_angle_deg)
    h_c, _ = ynu_trace(surfaces, gaps, h0_c, nu0_c)

    # For each element, check if CA >= beam footprint with margin
    for elem in range(n_elements):
        i_front = 2 * elem
        i_back = 2 * elem + 1
        h_max = max(abs(h_m[i_front]) + abs(h_c[i_front]),
                    abs(h_m[i_back]) + abs(h_c[i_back]))
        h_max *= 1.1  # 10% margin

        # Current CA with current glass thickness
        c1_e = curvatures[2 * elem]
        c2_e = curvatures[2 * elem + 1]
        ca = _thin_lens_ca(c1_e, c2_e, d_glass_list[elem], n_d)

        if ca < h_max:
            # Need thicker glass: d_center - sag(c1,h) + sag(c2,h) >= 0.5
            sag_diff = _sag(c1_e, h_max) - _sag(c2_e, h_max)
            d_glass_list[elem] = max(2.0, sag_diff + 0.5)

    # Rebuild with adjusted glass thicknesses
    surfaces, gaps = _build(d_glass_list, d_air_list)

    # --- Compute thin-lens CA and RI prediction ---
    ca_list = []
    for elem in range(n_elements):
        c1_e = curvatures[2 * elem]
        c2_e = curvatures[2 * elem + 1]
        ca = _thin_lens_ca(c1_e, c2_e, d_glass_list[elem], n_d)
        ca_list.append(ca)

    ri_pred, v_field, beam_info = _thin_lens_ri(
        surfaces, gaps, stop_idx, f_number, field_angle_deg, ca_list)

    # --- Design vector ---
    all_gap_d = [g[0] for g in gaps]
    design_vector = np.array(curvatures + all_gap_d)

    meta = {
        'n_elements': n_elements,
        'n_d': n_d,
        'efl_target': efl_target,
        'f_number': f_number,
        'field_angle_deg': field_angle_deg,
        'd_glass': d_glass_list,
        'phi_list': phi_list,
        'signs': signs,
        'c_half_list': c_half_list,
        'ca_list': ca_list,
        'ri_predicted': ri_pred,
        'beam_info': beam_info,
    }

    return surfaces, gaps, design_vector, meta


def print_design(surfaces, gaps, meta):
    """Pretty-print a generated design."""
    print("=" * 60)
    print("GENERATED INITIAL DESIGN")
    print("=" * 60)
    print(f"  Elements:    {meta['n_elements']}")
    print(f"  n_d:         {meta['n_d']}")
    print(f"  EFL target:  {meta['efl_target']} mm")
    print(f"  f-number:    {meta['f_number']}")
    print(f"  Field angle: {meta['field_angle_deg']} deg")
    signs_str = ''.join('+' if s > 0 else '-' for s in meta['signs'])
    print(f"  Power layout: [{signs_str}]")
    for i, phi in enumerate(meta['phi_list']):
        print(f"    L{i+1}: phi={phi:+.6f} mm^-1  "
              f"(f={1/phi:+.1f} mm)" if abs(phi) > 1e-12 else
              f"    L{i+1}: phi={phi:+.6f} mm^-1  (afocal)")
    print()

    K = len(surfaces)
    print(f"  {'Surf':>4s} {'c':>14s} {'n':>8s} {'n_prime':>8s}")
    print(f"  {'----':>4s} {'------':>14s} {'---':>8s} {'-------':>8s}")
    for i, (c, n, np_) in enumerate(surfaces):
        print(f"  {i:4d} {c:14.8f} {n:8.4f} {np_:8.4f}")
    print()

    print(f"  {'Gap':>4s} {'d (mm)':>10s} {'n_gap':>8s} {'type':>6s}")
    print(f"  {'---':>4s} {'------':>10s} {'-----':>8s} {'----':>6s}")
    for i, (d, n_gap) in enumerate(gaps):
        gtype = "glass" if n_gap > 1.001 else "air"
        print(f"  {i:4d} {d:10.4f} {n_gap:8.4f} {gtype:>6s}")
    print()

    # CA and beam info
    if 'ca_list' in meta:
        print(f"  {'Elem':>4s} {'CA':>8s} {'h_m(f)':>8s} {'h_c(f)':>8s} "
              f"{'h_max':>8s} {'margin':>8s}")
        print(f"  {'----':>4s} {'--':>8s} {'------':>8s} {'------':>8s} "
              f"{'-----':>8s} {'------':>8s}")
        for elem in range(meta['n_elements']):
            ca = meta['ca_list'][elem]
            bi = meta['beam_info'][2 * elem]  # front surface
            print(f"  {elem+1:4d} {ca:8.2f} {bi['h_m']:+8.2f} {bi['h_c']:+8.2f} "
                  f"{bi['h_max']:8.2f} {bi['margin']:+8.2f}")
        print()
        print(f"  Predicted RI at {meta['field_angle_deg']} deg: "
              f"{meta['ri_predicted']:.4f}")
        print()


if __name__ == "__main__":
    # Test: f/1.2, 30 deg, EFL=16
    print("=== f/1.2, 30 deg half-FOV, EFL=16mm ===\n")
    for N in range(1, 9):
        surfaces, gaps, x0, meta = generate_initial_design(
            n_elements=N, n_d=1.5, efl_target=16.0,
            f_number=1.2, field_angle_deg=30.0, ttl_factor=2.5)
        print_design(surfaces, gaps, meta)
        print()
