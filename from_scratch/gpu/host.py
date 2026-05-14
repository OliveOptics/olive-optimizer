"""
Host-side helpers for the GPU lens-design pipeline.

Contents
--------
1. Paraxial y-nu trace, exact meridional trace, chief-ray finder, Seidel coeffs
2. Design-vector pack / unpack / build
3. Air-gap distribution presets
4. Thin-lens clear-aperture and RI estimators
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════
#  1. Optics — paraxial trace, exact trace, chief ray, Seidel
# ══════════════════════════════════════════════════════════════════════

def ynu_trace(surfaces, gaps, h0, nu0):
    """Trace a paraxial ray through the system."""
    K = len(surfaces)
    h = np.zeros(K)
    nu = np.zeros(K)

    h[0] = h0
    c, n, n_prime = surfaces[0]
    phi = (n_prime - n) * c
    nu[0] = nu0 - h[0] * phi

    for i in range(1, K):
        d, n_gap = gaps[i - 1]
        h[i] = h[i - 1] + (d / n_gap) * nu[i - 1]

        c, n, n_prime = surfaces[i]
        phi = (n_prime - n) * c
        nu[i] = nu[i - 1] - h[i] * phi

    return h, nu


def _sag_sphere(c, h_val):
    """Spherical sag at height h for curvature c."""
    if abs(c) < 1e-15:
        return 0.0
    arg = 1.0 - c * c * h_val * h_val
    if arg <= 0.0:
        return c * h_val * h_val
    return c * h_val * h_val / (1.0 + np.sqrt(arg))


def exact_trace(surfaces, gaps, h0, u0):
    """Exact meridional ray trace with sag-corrected ray-surface intersection.

    At each surface, Newton iteration finds the true intersection height on
    the sphere.  Transfer between surfaces accounts for departure sag.

    Parameters: h0 = initial height, u0 = initial angle (radians).
    Returns: h, nu arrays where nu[i] = n'_i * sin(U'_i).
    """
    K = len(surfaces)
    h = np.zeros(K)
    nu = np.zeros(K)

    c, n, n_prime = surfaces[0]
    h_hit = float(h0)
    tan_U = np.tan(u0)
    for _ in range(8):
        z_s = _sag_sphere(c, h_hit)
        ch2 = c * c * h_hit * h_hit
        if ch2 >= 1.0:
            break
        dzdh = c * h_hit / np.sqrt(1.0 - ch2)
        if abs(tan_U) > 1e-12:
            f_val = z_s - (h_hit - h0) / tan_U
            df = dzdh - 1.0 / tan_U
        else:
            f_val = z_s * tan_U - (h_hit - h0)
            df = dzdh * tan_U - 1.0
        if abs(df) < 1e-30:
            break
        dh = f_val / df
        h_hit -= dh
        if abs(dh) < 1e-14:
            break

    h[0] = h_hit
    z_sag = _sag_sphere(c, h_hit)

    sin_alpha = h_hit * c
    alpha = np.arcsin(np.clip(sin_alpha, -0.9999, 0.9999))
    sin_I = np.sin(u0 + alpha)
    sin_Ip = (n / n_prime) * sin_I
    Ip = np.arcsin(np.clip(sin_Ip, -0.9999, 0.9999))
    U_post = Ip - alpha
    nu[0] = n_prime * np.sin(U_post)

    for i in range(1, K):
        d_gap, n_gap = gaps[i - 1]
        tan_Up = np.tan(U_post)
        h_v = h[i - 1] + (d_gap - z_sag) * tan_Up

        c, n, n_prime = surfaces[i]
        h_hit = float(h_v)
        tan_U = tan_Up
        for _ in range(8):
            z_s = _sag_sphere(c, h_hit)
            ch2 = c * c * h_hit * h_hit
            if ch2 >= 1.0:
                break
            dzdh = c * h_hit / np.sqrt(1.0 - ch2)
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
            if abs(dh) < 1e-14:
                break

        h[i] = h_hit
        z_sag = _sag_sphere(c, h_hit)

        sin_alpha = h_hit * c
        alpha = np.arcsin(np.clip(sin_alpha, -0.9999, 0.9999))
        sin_I = np.sin(U_post + alpha)
        sin_Ip = (n / n_prime) * sin_I
        Ip = np.arcsin(np.clip(sin_Ip, -0.9999, 0.9999))
        U_post = Ip - alpha
        nu[i] = n_prime * np.sin(U_post)

    return h, nu


def _ray_height_at_stop(h, nu, stop_info, gaps):
    """Ray height at the stop location.

    If the stop is at a surface, return h[surface_idx].  If it's inside a
    gap, propagate from the surface before the gap to the stop position.
    """
    if stop_info['type'] == 'at_surface':
        return h[stop_info['surface_idx']]
    else:
        idx = stop_info['gap_idx']
        d_before = stop_info['distance_before_stop']
        n_gap = gaps[idx][1]
        return h[idx] + (d_before / n_gap) * nu[idx]


def find_chief_ray_initial(surfaces, gaps, stop_info, field_angle):
    """Find initial chief-ray conditions so that h = 0 at the stop.

    stop_info: dict ({'type','surface_idx'} or {'type','gap_idx',
    'distance_before_stop'}) or integer index (stop at surface).

    Linear in h0: trace with h0=0 and h0=1, interpolate.
    """
    if isinstance(stop_info, int):
        stop_info = {'type': 'at_surface', 'surface_idx': stop_info}

    n_air = 1.0
    nu0 = n_air * np.tan(np.radians(field_angle))

    h_a, nu_a = ynu_trace(surfaces, gaps, 0.0, nu0)
    h_stop_a = _ray_height_at_stop(h_a, nu_a, stop_info, gaps)

    h_b, nu_b = ynu_trace(surfaces, gaps, 1.0, nu0)
    h_stop_b = _ray_height_at_stop(h_b, nu_b, stop_info, gaps)

    h0 = -h_stop_a / (h_stop_b - h_stop_a)
    return h0, nu0


def seidel_coefficients(surfaces, h_marginal, nu_marginal, h_chief, nu_chief,
                        nu0_marginal, nu0_chief, conics=None):
    """Seidel aberration coefficients at each surface.

    conics : array-like of float or None
        Conic constant per surface (k).  None or all-zeros gives the
        spherical-only result.  Standard Buchdahl/Welford correction
            Q = (n' - n) * k * c^3 * h_m^4
        is added to S_I..S_V (Petzval excluded — depends only on paraxial power).
    """
    K = len(surfaces)
    S_I = np.zeros(K)
    S_II = np.zeros(K)
    S_III = np.zeros(K)
    S_IV = np.zeros(K)
    S_V = np.zeros(K)

    H = nu0_marginal * h_chief[0] - nu0_chief * h_marginal[0]

    nu_pre_marginal = np.zeros(K)
    nu_pre_chief = np.zeros(K)
    nu_pre_marginal[0] = nu0_marginal
    nu_pre_chief[0] = nu0_chief
    for i in range(1, K):
        nu_pre_marginal[i] = nu_marginal[i - 1]
        nu_pre_chief[i] = nu_chief[i - 1]

    for i in range(K):
        c, n, n_prime = surfaces[i]
        phi = (n_prime - n) * c

        nu_m = nu_pre_marginal[i]
        nu_c = nu_pre_chief[i]

        A = nu_m + n * h_marginal[i] * c
        A_bar = nu_c + n * h_chief[i] * c

        u_before = nu_m / n
        u_after = nu_marginal[i] / n_prime
        delta_u_over_n = u_after / n_prime - u_before / n

        S_I[i] = -A**2 * h_marginal[i] * delta_u_over_n
        S_II[i] = -A * A_bar * h_marginal[i] * delta_u_over_n
        S_III[i] = -A_bar**2 * h_marginal[i] * delta_u_over_n
        S_IV[i] = H**2 * phi / (n * n_prime)
        S_V[i] = (S_III[i] + S_IV[i]) * (A_bar / A) if abs(A) > 1e-12 else 0.0

        if conics is not None and conics[i] != 0.0:
            kc = conics[i]
            delta_n = n_prime - n
            hm4 = h_marginal[i] ** 4
            Q = delta_n * kc * c**3 * hm4
            hc_hm = h_chief[i] / h_marginal[i] if abs(h_marginal[i]) > 1e-15 else 0.0
            S_I[i] += Q
            S_II[i] += Q * hc_hm
            S_III[i] += Q * hc_hm**2
            S_V[i] += Q * hc_hm**3

    return S_I, S_II, S_III, S_IV, S_V


# ══════════════════════════════════════════════════════════════════════
#  2. Design vector — pack / unpack / build
# ══════════════════════════════════════════════════════════════════════
#
# Layout (sphere-only block, then optional aspheric block):
#     x = [c_1..c_K,                       # K = 2*n_elements curvatures
#          d_glass_1..d_glass_n_elements,
#          d_air_1..d_air_{n_elements-1},
#          n_d_1..n_d_n_elements,
#          k_1..k_K, A4_1..A4_K, A6_1..A6_K]   # optional aspheric tail

def build_system(x, n_elements):
    """Unpack x into (surfaces, gaps) tuples for the host-side trace."""
    K = 2 * n_elements

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


def pack_x(c1_list, c2_list, d_glass_list, d_air_list, n_d_list,
           k_list=None, a4_list=None, a6_list=None):
    """Pack all variables into a single design vector.

    k_list, a4_list, a6_list: per-surface aspheric coefficients.
    If None, zeros are appended (spherical).
    """
    curvatures = []
    for c1, c2 in zip(c1_list, c2_list):
        curvatures.append(c1)
        curvatures.append(c2)
    K = len(curvatures)
    base = curvatures + list(d_glass_list) + list(d_air_list) + list(n_d_list)
    kc  = list(k_list)  if k_list  is not None else [0.0] * K
    a4  = list(a4_list) if a4_list is not None else [0.0] * K
    a6  = list(a6_list) if a6_list is not None else [0.0] * K
    return np.array(base + kc + a4 + a6)


def unpack_x(x, n_elements):
    """Unpack design vector into components.

    Returns (c1, c2, d_glass, d_air, n_d) for spherical-only vectors, or
    (c1, c2, d_glass, d_air, n_d, k, a4, a6) when aspheric data is present.
    """
    K = 2 * n_elements
    n_spher = K + n_elements + (n_elements - 1) + n_elements
    curvatures = x[:K]
    d_glass = x[K:K + n_elements]
    d_air = x[K + n_elements:K + n_elements + (n_elements - 1)]
    n_d = x[K + n_elements + (n_elements - 1):n_spher]
    c1_list = curvatures[0::2]
    c2_list = curvatures[1::2]
    if len(x) > n_spher:
        k_list  = x[n_spher:n_spher + K]
        a4_list = x[n_spher + K:n_spher + 2 * K]
        a6_list = x[n_spher + 2 * K:n_spher + 3 * K]
        return c1_list, c2_list, d_glass, d_air, n_d, k_list, a4_list, a6_list
    return c1_list, c2_list, d_glass, d_air, n_d


# ══════════════════════════════════════════════════════════════════════
#  3. Air-gap distribution presets
# ══════════════════════════════════════════════════════════════════════

def make_air_gaps(signs, total_air, style):
    """Distribute `total_air` across the (N-1) air gaps of an N-element layout.

    style ∈ {'uniform', 'retrofocus', 'front_heavy', 'back_heavy', 'middle'}.
    `signs` is the per-element power sign list; used only by 'retrofocus' to
    find the first negative→positive boundary.
    """
    N = len(signs)
    n_air = N - 1
    if n_air == 0:
        return []

    if style == 'uniform':
        return [total_air / n_air] * n_air
    elif style == 'retrofocus':
        g = [1.5] * n_air
        for i in range(n_air):
            if signs[i] < 0 and signs[i+1] > 0:
                g[i] = 6.0
                break
        s = total_air / sum(g)
        return [x * s for x in g]
    elif style == 'front_heavy':
        w = [n_air - i for i in range(n_air)]
        s = total_air / sum(w)
        return [x * s for x in w]
    elif style == 'back_heavy':
        w = [i + 1 for i in range(n_air)]
        s = total_air / sum(w)
        return [x * s for x in w]
    elif style == 'middle':
        w = [1.0] * n_air
        w[n_air // 2] = 4.0
        s = total_air / sum(w)
        return [x * s for x in w]


# ══════════════════════════════════════════════════════════════════════
#  4. Thin-lens clear aperture and relative-illumination estimators
# ══════════════════════════════════════════════════════════════════════

def _sag(c, h, k=0.0, asph=None):
    """Signed sag for curvature c at height h with optional conic and aspheric.

    asph : sequence [A4, A6, A8, ...] or None.
    """
    if abs(c) < 1e-15:
        z = 0.0
    else:
        arg = 1 - (1 + k) * c**2 * h**2
        if arg <= 0:
            z = c * h**2
        else:
            z = c * h**2 / (1 + np.sqrt(arg))
    if asph is not None:
        h2 = h * h
        hp = h2 * h2
        for ai in asph:
            z += ai * hp
            hp *= h2
    return z


def _thin_lens_ca(c1, c2, d_glass, n_d, e_min=0.0,
                  k1=0.0, k2=0.0, asph1=None, asph2=None):
    """Max clear aperture of a singlet with curvatures c1, c2.

    ET(h) = d_glass - sag(c1, h) + sag(c2, h).  Bisects for the largest h
    where ET(h) >= e_min.
    """
    ac1, ac2 = abs(c1), abs(c2)
    if ac1 < 1e-12 and ac2 < 1e-12:
        return 1e6

    h_limits = []
    if ac1 > 1e-12:
        h_limits.append(0.999 / ac1)
    if ac2 > 1e-12:
        h_limits.append(0.999 / ac2)
    h_sphere = min(h_limits)

    def edge_thickness(h):
        return d_glass - _sag(c1, h, k1, asph1) + _sag(c2, h, k2, asph2)

    if edge_thickness(h_sphere) >= e_min:
        return h_sphere
    if edge_thickness(0.0) < e_min:
        return 0.0

    h_lo, h_hi = 0.0, h_sphere
    for _ in range(60):
        h_mid = (h_lo + h_hi) / 2
        if edge_thickness(h_mid) >= e_min:
            h_lo = h_mid
        else:
            h_hi = h_mid
    return h_lo


def _thin_lens_ri(surfaces, gaps, stop_idx, f_number, field_angle_deg, ca_list):
    """Predict RI at field_angle_deg using thin-lens CA limits.

    Traces marginal and chief rays, then checks 1D vignetting at each
    element using the physical CA estimate.

    Returns
    -------
    ri : float, estimated relative illumination
    v_field : float, vignetting factor at field
    beam_info : list of dicts per surface with h_m, h_c, ca, margin
    """
    K = len(surfaces)

    h_u, nu_u = ynu_trace(surfaces, gaps, 1.0, 0.0)
    efl = -1.0 / nu_u[-1]
    sa = efl / (2 * f_number)
    h_m, nu_m = ynu_trace(surfaces, gaps, sa, 0.0)

    h0_c, nu0_c = find_chief_ray_initial(
        surfaces, gaps, stop_idx, field_angle_deg)
    h_c, nu_c = ynu_trace(surfaces, gaps, h0_c, nu0_c)

    theta_ref = 0.1
    h0_ref, nu0_ref = find_chief_ray_initial(
        surfaces, gaps, stop_idx, theta_ref)
    h_c_ref, _ = ynu_trace(surfaces, gaps, h0_ref, nu0_ref)

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

    sd = np.zeros(K)
    for i in range(K):
        elem_idx = i // 2
        sd[i] = ca_list[elem_idx]

    v_field = vignetting_1d(h_m, h_c, sd)
    v_ref = vignetting_1d(h_m, h_c_ref, sd)

    cos4_field = np.cos(np.radians(field_angle_deg))**4
    cos4_ref = np.cos(np.radians(theta_ref))**4

    if v_ref < 1e-15:
        ri = 0.0
    else:
        ri = (cos4_field * v_field) / (cos4_ref * v_ref)

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
