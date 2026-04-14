"""
L_033 lens system — loader, scaler, and analysis.

This is an f/1.2, 7-element lens with 13 optical surfaces.
The system is stored normalized (EFL=1) and scaled to any target EFL.

Usage:
    from system import load_system, analyze
    sys = load_system()              # normalized, EFL=1
    sys = load_system(efl=16.0)      # scaled to 16mm
    analyze(sys)                     # print all parameters
"""

import sys as _sys
import os

# Add parent dirs to path so we can import the core library
_project_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(_project_dir))
if _root_dir not in _sys.path:
    _sys.path.insert(0, _root_dir)

import numpy as np
from parse_zmx import (parse_zmx, zmx_to_normalized, normalized_to_physical,
                        scale_system, scale_stop_info, ynu_trace,
                        find_chief_ray_initial, zmx_to_trace_map)
from gradient import (ynu_trace_with_grad, efl_with_grad, bfl_with_grad,
                       ttl_with_grad, edge_thickness_with_grad,
                       full_merit_with_grad, sag_derivs)


ZMX_FILE = os.path.join(_project_dir, 'L_033.zmx')


# ── System container ─────────────────────────────────────────────────────────

class LensSystem:
    """Container for a lens system at a specific scale."""

    def __init__(self, surfaces, gaps, stop_info, efl, f_number,
                 norm_surfaces, norm_gaps, norm_stop, meta,
                 zmx_sd=None):
        self.surfaces = surfaces
        self.gaps = gaps
        self.stop_info = stop_info
        self.efl = efl
        self.f_number = f_number
        self.sa = efl / (2 * f_number)      # semi-aperture
        self.enpd = 2 * self.sa              # entrance pupil diameter
        self.K = len(surfaces)               # number of optical surfaces

        # Normalized system (EFL=1), always kept for rescaling
        self.norm_surfaces = norm_surfaces
        self.norm_gaps = norm_gaps
        self.norm_stop = norm_stop
        self.meta = meta

        # Clear aperture semi-diameters from ZMX (scaled to current EFL)
        self.zmx_sd = zmx_sd

    def rescale(self, efl_new):
        """Return a new LensSystem scaled to a different EFL."""
        s, g, stop = normalized_to_physical(
            self.norm_surfaces, self.norm_gaps, self.norm_stop, efl_new)
        return LensSystem(s, g, stop, efl_new, self.f_number,
                          self.norm_surfaces, self.norm_gaps,
                          self.norm_stop, self.meta)

    def trace_marginal(self):
        """Trace the marginal ray. Returns h, nu arrays."""
        return ynu_trace(self.surfaces, self.gaps, self.sa, 0.0)

    def compute_bfl(self):
        h_m, nu_m = self.trace_marginal()
        return -h_m[-1] / nu_m[-1]

    def compute_ttl(self):
        bfl = self.compute_bfl()
        return sum(g[0] for g in self.gaps) + bfl

    def compute_edge_thicknesses(self):
        h_m, _, dh_dc, _, dh_dd, _ = ynu_trace_with_grad(
            self.surfaces, self.gaps, self.sa, 0.0)
        ET, _, _ = edge_thickness_with_grad(
            self.surfaces, self.gaps, h_m, dh_dc, dh_dd, d_edge_min=0.0)
        return ET

    def compute_max_sd(self):
        """Compute clear semi-diameter per surface.
        Uses ZMX clear apertures if available, otherwise geometry-based."""
        if self.zmx_sd is not None:
            return self.zmx_sd.copy()
        # Fallback to geometry-based computation
        return max_clear_semi_diameters(self.surfaces, self.gaps)

    def trace_chief(self, field_angle_deg):
        """Trace chief ray at a given field angle. Returns h, nu arrays."""
        h0, nu0 = find_chief_ray_initial(
            self.surfaces, self.gaps, self.stop_info, field_angle_deg)
        return ynu_trace(self.surfaces, self.gaps, h0, nu0)

    def compute_cra(self, field_angles_deg=None):
        """
        Compute chief ray angle at the image plane for each field.

        CRA = arctan(nu_c[-1]), since the last medium is air (n=1).

        Returns
        -------
        dict with 'angles' (input field angles) and 'cra' (CRA in degrees)
        """
        if field_angles_deg is None:
            max_field = max(self.meta['fields_y'])
            field_angles_deg = np.linspace(0, max_field, 11)

        cra = np.zeros(len(field_angles_deg))
        for j, theta in enumerate(field_angles_deg):
            if abs(theta) < 1e-10:
                cra[j] = 0.0
                continue
            h_c, nu_c = self.trace_chief(theta)
            cra[j] = np.degrees(np.arctan(nu_c[-1]))

        return {'angles': np.array(field_angles_deg), 'cra': cra}

    def compute_ri(self, field_angles_deg=None, n_pupil=200):
        """
        Compute relative illumination at multiple field angles (2D circular pupil).

        RI is normalized relative to a near-zero reference angle (0.1 deg)
        so that the baseline vignetting from ZMX clear apertures is factored
        out.  RI_rel(theta) = RI_raw(theta) / RI_raw(theta_ref).

        Parameters
        ----------
        field_angles_deg : array-like or None
            Field angles in degrees. If None, uses 0 to max field in 10 steps.
        n_pupil : int
            Number of integration points along rho_y for area calculation.

        Returns
        -------
        dict with 'angles', 'ri', 'vignetting', 'cos4', 'ri_raw', 'sd'
        """
        if field_angles_deg is None:
            max_field = max(self.meta['fields_y'])
            field_angles_deg = np.linspace(0, max_field, 11)

        h_m, nu_m = self.trace_marginal()
        sd = self.compute_max_sd()

        # Reference: compute raw RI at a tiny angle for normalization
        theta_ref = 0.1  # degrees
        h_c_ref, _ = self.trace_chief(theta_ref)
        V_ref = vignetting_2d(h_m, h_c_ref, sd, n_pupil)
        cos4_ref = np.cos(np.radians(theta_ref))**4
        ri_ref = cos4_ref * V_ref

        n_fields = len(field_angles_deg)
        results = {
            'angles': np.array(field_angles_deg),
            'ri': np.zeros(n_fields),
            'ri_raw': np.zeros(n_fields),
            'vignetting': np.zeros(n_fields),
            'cos4': np.zeros(n_fields),
            'sd': sd,
            'ri_ref': ri_ref,
            'theta_ref': theta_ref,
        }

        for j, theta in enumerate(field_angles_deg):
            theta_rad = np.radians(theta)
            cos4 = np.cos(theta_rad)**4
            results['cos4'][j] = cos4

            if abs(theta) < 1e-10:
                # On-axis: use same V as reference (linear extrapolation)
                results['vignetting'][j] = V_ref
                ri_raw = cos4 * V_ref
            else:
                h_c, nu_c = self.trace_chief(theta)
                V = vignetting_2d(h_m, h_c, sd, n_pupil)
                results['vignetting'][j] = V
                ri_raw = cos4 * V

            results['ri_raw'][j] = ri_raw
            results['ri'][j] = ri_raw / ri_ref if ri_ref > 1e-15 else 0.0

        return results


# ── Vignetting helpers ───────────────────────────────────────────────────────

# def max_clear_semi_diameters(surfaces, gaps):
#     """
#     Compute the maximum clear semi-diameter for each surface, determined
#     by the geometry of adjacent elements (edge thickness going to zero).
#
#     For each gap j (between surfaces j and j+1):
#         ET(h) = d_j - sag(c_j, h) + sag(c_{j+1}, h)
#     Using paraxial sag: sag = c*h^2/2
#         ET(h) = d_j - (c_j - c_{j+1}) * h^2 / 2
#
#     Max h where ET=0:
#         h_max = sqrt(2*d / (c_j - c_{j+1}))  when c_j > c_{j+1}
#         h_max = inf                           when c_j <= c_{j+1}
#
#     Each surface's SD is the min of h_max from its adjacent gaps.
#
#     Returns
#     -------
#     sd : (K,) max clear semi-diameter per surface
#     """
#     K = len(surfaces)
#     Kd = K - 1
#
#     # Max SD from each gap
#     h_max_gap = np.full(Kd, np.inf)
#     for j in range(Kd):
#         c_front = surfaces[j][0]
#         c_back = surfaces[j + 1][0]
#         d_gap = gaps[j][0]
#         delta_c = c_front - c_back
#         if delta_c > 1e-15:
#             h_max_gap[j] = np.sqrt(2.0 * d_gap / delta_c)
#
#     # Each surface's SD = min of adjacent gap limits
#     sd = np.full(K, np.inf)
#     for i in range(K):
#         if i > 0:
#             sd[i] = min(sd[i], h_max_gap[i - 1])
#         if i < Kd:
#             sd[i] = min(sd[i], h_max_gap[i])
#
#     return sd


def vignetting_2d(h_m, h_c, sd, n_pts=200):
    """
    Compute the 2D (circular pupil) vignetting factor.

    In pupil space (rho_x, rho_y), each surface constrains rays to a circle:
        rho_x^2 + (rho_y + s_i)^2 <= R_i^2
    where s_i = h_c(i)/h_m(i), R_i = SD(i)/|h_m(i)|.

    The physical pupil is the unit disk: rho_x^2 + rho_y^2 <= 1.

    Since all constraint circles are centered on the rho_y axis, integrate
    over rho_y: at each rho_y, the allowed |rho_x| is the min over all
    constraints.

    V = vignetted_area / pi  (normalized so V=1 when no vignetting).

    Parameters
    ----------
    h_m : (K,) marginal ray heights
    h_c : (K,) chief ray heights at the field angle
    sd : (K,) clear semi-diameters per surface
    n_pts : int, integration points along rho_y

    Returns
    -------
    V : vignetting factor in [0, 1]
    """
    K = len(h_m)

    # Build constraint circles in pupil space
    # Each surface gives a circle centered at (0, -s_i) with radius R_i
    circles = []
    for i in range(K):
        if abs(h_m[i]) < 1e-15:
            # Marginal height ~0: check if chief ray alone exceeds SD
            if abs(h_c[i]) > sd[i]:
                return 0.0  # fully vignetted
            continue  # no pupil constraint from this surface
        s_i = h_c[i] / h_m[i]
        R_i = sd[i] / abs(h_m[i])
        circles.append((s_i, R_i))

    if not circles:
        return 1.0

    # Find rho_y range: intersection of all circle y-extents with [-1, 1]
    rho_y_lo = -1.0
    rho_y_hi = 1.0
    for s_i, R_i in circles:
        rho_y_lo = max(rho_y_lo, -s_i - R_i)
        rho_y_hi = min(rho_y_hi, -s_i + R_i)

    if rho_y_lo >= rho_y_hi:
        return 0.0

    rho_y = np.linspace(rho_y_lo, rho_y_hi, n_pts)

    # At each rho_y, compute max allowed |rho_x|
    # From the unit pupil: rho_x_max = sqrt(1 - rho_y^2)
    # From circle i: rho_x_max = sqrt(R_i^2 - (rho_y + s_i)^2)
    rho_x_max = np.sqrt(np.maximum(0.0, 1.0 - rho_y**2))

    for s_i, R_i in circles:
        constraint = np.sqrt(np.maximum(0.0, R_i**2 - (rho_y + s_i)**2))
        rho_x_max = np.minimum(rho_x_max, constraint)

    # Area = integral of 2 * rho_x_max drho_y
    # V = area / pi (unit disk area = pi)
    area = np.trapz(2.0 * rho_x_max, rho_y)
    V = area / np.pi

    return min(V, 1.0)


# ── Loader ───────────────────────────────────────────────────────────────────

def load_system(efl=None, index_overrides=None):
    """
    Load L_033.zmx and return a LensSystem.

    Parameters
    ----------
    efl : float or None
        Target EFL in mm. If None, returns normalized system (EFL=1).
    index_overrides : dict or None
        {zemax_surf_num: {'n': float}} to override glass indices.

    Returns
    -------
    LensSystem
    """
    zmx_data = parse_zmx(ZMX_FILE)
    norm_s, norm_g, norm_stop, efl_phys, meta = zmx_to_normalized(
        zmx_data, index_overrides)

    # Extract ZMX semi-diameters and map to trace surfaces
    z2t, t2z = zmx_to_trace_map(zmx_data)
    K_trace = len(t2z)
    zmx_sd_phys = np.zeros(K_trace)
    for trace_idx in range(K_trace):
        zmx_num = t2z[trace_idx]
        zmx_sd_phys[trace_idx] = zmx_data['surfaces'][zmx_num]['semi_dia']

    # Normalize SD the same way as the system (divide by efl_phys)
    zmx_sd_norm = zmx_sd_phys / efl_phys

    if efl is None:
        efl_target = 1.0
        surfaces, gaps, stop_info = norm_s, norm_g, norm_stop
        zmx_sd = zmx_sd_norm
    else:
        efl_target = efl
        surfaces, gaps, stop_info = normalized_to_physical(
            norm_s, norm_g, norm_stop, efl_target)
        zmx_sd = zmx_sd_norm * efl_target

    return LensSystem(
        surfaces, gaps, stop_info, efl_target, meta['f_number'],
        norm_s, norm_g, norm_stop, meta, zmx_sd=zmx_sd)


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze(sys, verbose=True):
    """Compute and optionally print all system parameters."""
    h_m, nu_m = sys.trace_marginal()
    efl = -sys.sa / nu_m[-1]
    bfl = -h_m[-1] / nu_m[-1]
    ttl = sum(g[0] for g in sys.gaps) + bfl
    ET = sys.compute_edge_thicknesses()

    results = {
        'efl': efl,
        'f_number': sys.f_number,
        'enpd': sys.enpd,
        'sa': sys.sa,
        'bfl': bfl,
        'ttl': ttl,
        'ttl_over_efl': ttl / efl,
        'bfl_over_efl': bfl / efl,
        'h_m': h_m,
        'nu_m': nu_m,
        'edge_thicknesses': ET,
        'surfaces': sys.surfaces,
        'gaps': sys.gaps,
    }

    # Field info
    meta = sys.meta
    if meta['field_type'] == 0:
        max_field = max(meta['fields_y'])
        img_h = efl * np.tan(np.radians(max_field))
        results['half_fov_deg'] = max_field
        results['image_height'] = img_h
        results['image_circle_dia'] = 2 * img_h

    # Relative illumination and CRA
    ri_data = sys.compute_ri()
    cra_data = sys.compute_cra(ri_data['angles'])
    results['ri_data'] = ri_data
    results['cra_data'] = cra_data

    if verbose:
        print(f"{'=' * 60}")
        print(f"L_033 @ EFL = {efl:.4f} mm")
        print(f"{'=' * 60}")
        print(f"  EFL       = {efl:.4f} mm")
        print(f"  f/#       = {sys.f_number:.4f}")
        print(f"  ENPD      = {sys.enpd:.4f} mm")
        print(f"  BFL       = {bfl:.4f} mm  ({bfl/efl:.4f} x EFL)")
        print(f"  TTL       = {ttl:.4f} mm  ({ttl/efl:.4f} x EFL)")
        if 'half_fov_deg' in results:
            print(f"  Half-FOV  = {results['half_fov_deg']:.1f} deg")
            print(f"  Img circle= {results['image_circle_dia']:.4f} mm dia")
        print()

        print(f"  {'Surf':>4s} {'c':>12s} {'R':>10s} {'n':>7s} {'n_p':>7s}"
              f" {'h_m':>8s} {'nu_m':>10s} {'SD':>8s}")
        sd = ri_data['sd']
        for i in range(sys.K):
            c, n, np_ = sys.surfaces[i]
            R = 1/c if abs(c) > 1e-10 else float('inf')
            R_str = f"{R:.3f}" if abs(R) < 1e6 else "inf"
            sd_str = f"{sd[i]:.4f}" if sd[i] < 1e6 else "inf"
            print(f"  {i:4d} {c:12.6f} {R_str:>10s} {n:7.4f} {np_:7.4f}"
                  f" {h_m[i]:8.4f} {nu_m[i]:10.6f} {sd_str:>8s}")
        print()

        print(f"  {'Gap':>4s} {'d':>8s} {'n':>7s} {'type':>6s} {'ET':>8s}")
        for i, (d, n) in enumerate(sys.gaps):
            medium = 'glass' if n > 1.01 else 'air'
            et_str = f"{ET[i]:.4f}" if i < len(ET) else "---"
            flag = " !!!" if i < len(ET) and ET[i] < 0 else ""
            print(f"  {i:4d} {d:8.4f} {n:7.4f} {medium:>6s} {et_str:>8s}{flag}")
        print()

        # RI + CRA table
        print(f"  RELATIVE ILLUMINATION & CRA (2D circular pupil)")
        print(f"  ref @ {ri_data['theta_ref']:.1f} deg: RI_raw={ri_data['ri_ref']:.4f}")
        print(f"  {'Field':>7s} {'cos4':>7s} {'V':>7s} {'RI_raw':>7s} {'RI_rel':>7s} {'CRA':>8s}")
        for j in range(len(ri_data['angles'])):
            ang = ri_data['angles'][j]
            c4 = ri_data['cos4'][j]
            V = ri_data['vignetting'][j]
            ri_raw = ri_data['ri_raw'][j]
            ri_rel = ri_data['ri'][j]
            cra = cra_data['cra'][j]
            print(f"  {ang:7.1f} {c4:7.4f} {V:7.4f} {ri_raw:7.4f} {ri_rel:7.4f} {cra:8.2f}")
        print()

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Normalized (EFL=1) ===")
    sys_norm = load_system()
    analyze(sys_norm)

    print("\n=== Scaled to EFL=16mm ===")
    sys_16 = load_system(efl=16.0)
    analyze(sys_16)
