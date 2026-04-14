"""
Parse a Zemax .zmx file and run y-nu trace + Seidel calculation.
"""

import numpy as np


def parse_zmx(filepath):
    """
    Parse a Zemax .zmx file (handles UTF-16 encoding with spaced characters).
    Returns system data: surfaces, entrance pupil, fields, stop index.
    """
    # Try reading as UTF-16 first, fall back to UTF-8
    try:
        with open(filepath, 'r', encoding='utf-16') as f:
            content = f.read()
    except (UnicodeError, UnicodeDecodeError):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

    lines = content.split('\n')

    surfaces = []  # list of dicts: {curv, thickness, glass, n_d, v_d}
    current_surf = None
    stop_index = None
    enpd = None  # entrance pupil diameter
    fields_y = []
    field_type = 0  # 0=angle(deg), 1=obj height, 2=paraxial img height, 3=real img height

    for line in lines:
        line = line.strip()

        # Entrance pupil diameter
        if line.startswith('ENPD'):
            parts = line.split()
            enpd = float(parts[1])

        # Field type
        if line.startswith('FTYP'):
            parts = line.split()
            field_type = int(parts[1])

        # Field values (Y) — YFLN (new format) or YFLD (old format)
        if line.startswith('YFLN') or line.startswith('YFLD'):
            parts = line.split()
            fields_y = [float(x) for x in parts[1:]]

        # New surface
        if line.startswith('SURF'):
            parts = line.split()
            surf_num = int(parts[1])
            current_surf = {
                'num': surf_num,
                'curv': 0.0,
                'thickness': 0.0,
                'glass': None,
                'n_d': 1.0,  # default air
                'v_d': 0.0,
                'is_stop': False,
                'semi_dia': 0.0,
            }
            surfaces.append(current_surf)

        if current_surf is None:
            continue

        # Stop surface
        if line == 'STOP':
            current_surf['is_stop'] = True
            stop_index = current_surf['num']

        # Curvature
        if line.startswith('CURV'):
            parts = line.split()
            current_surf['curv'] = float(parts[1])

        # Thickness
        if line.startswith('DISZ'):
            parts = line.split()
            if parts[1] == 'INFINITY':
                current_surf['thickness'] = float('inf')
            else:
                current_surf['thickness'] = float(parts[1])

        # Semi-diameter (clear aperture)
        if line.startswith('DIAM'):
            parts = line.split()
            current_surf['semi_dia'] = float(parts[1])

        # Glass
        if line.startswith('GLAS'):
            parts = line.split()
            current_surf['glass'] = parts[1]
            current_surf['n_d'] = float(parts[4])
            current_surf['v_d'] = float(parts[5])

    return {
        'surfaces': surfaces,
        'enpd': enpd,
        'fields_y': fields_y,
        'field_type': field_type,
        'stop_index': stop_index,
    }


# ── System scaling ──────────────────────────────────────────────────────────

def scale_system(surfaces, gaps, scale):
    """
    Scale an optical system by a linear factor.

    All lengths multiply by scale, curvatures divide by scale.
    Refractive indices and angles (f/#, field angle, nu) are invariant.

    To normalize to EFL=1:  scale = 1 / EFL_current
    To restore from normalized: scale = EFL_physical

    Parameters
    ----------
    surfaces : list of (c, n, n')
    gaps : list of (d, n)
    scale : float, linear scale factor

    Returns
    -------
    scaled_surfaces, scaled_gaps
    """
    scaled_surfaces = [(c / scale, n, np_) for c, n, np_ in surfaces]
    scaled_gaps = [(d * scale, n_gap) for d, n_gap in gaps]
    return scaled_surfaces, scaled_gaps


def scale_stop_info(stop_info, scale):
    """Scale stop_info distances by the linear scale factor."""
    if stop_info['type'] == 'at_surface':
        return stop_info  # no distance to scale
    return {
        'type': stop_info['type'],
        'gap_idx': stop_info['gap_idx'],
        'distance_before_stop': stop_info['distance_before_stop'] * scale,
    }


def zmx_to_normalized(zmx_data, index_overrides=None):
    """
    Parse ZMX data into a normalized optical system (EFL = 1).

    Returns
    -------
    norm_surfaces, norm_gaps : scaled system with EFL = 1
    stop_info : scaled stop info
    scale_factor : the physical EFL (multiply to restore physical units)
    meta : dict with physical quantities and ZMX metadata
    """
    # Apply index overrides
    if index_overrides:
        for s in zmx_data['surfaces']:
            if s['num'] in index_overrides:
                s['n_d'] = index_overrides[s['num']]['n']

    surfaces, gaps, stop_info = build_system_from_zmx(zmx_data)

    # Compute physical EFL
    h_m, nu_m = ynu_trace(surfaces, gaps, 1.0, 0.0)
    efl_physical = -1.0 / nu_m[-1]

    # Normalize: scale = 1/EFL so that EFL_norm = 1
    s = 1.0 / efl_physical
    norm_surfaces, norm_gaps = scale_system(surfaces, gaps, s)
    norm_stop = scale_stop_info(stop_info, s)

    # Compute normalized quantities for verification
    sa_physical = zmx_data['enpd'] / 2.0
    h_m_n, nu_m_n = ynu_trace(norm_surfaces, norm_gaps, sa_physical * s, 0.0)
    bfl_n = -h_m_n[-1] / nu_m_n[-1]
    ttl_n = sum(g[0] for g in norm_gaps) + bfl_n

    meta = {
        'efl_physical': efl_physical,
        'enpd': zmx_data['enpd'],
        'f_number': efl_physical / zmx_data['enpd'],
        'fields_y': zmx_data['fields_y'],
        'field_type': zmx_data['field_type'],
        'zmx_data': zmx_data,
        # Normalized quantities (EFL=1 units)
        'bfl_norm': bfl_n,
        'ttl_norm': ttl_n,
    }

    return norm_surfaces, norm_gaps, norm_stop, efl_physical, meta


def normalized_to_physical(norm_surfaces, norm_gaps, norm_stop, efl_target):
    """
    Scale a normalized system (EFL=1) to a physical EFL.

    Returns surfaces, gaps, stop_info in physical units.
    """
    surfaces, gaps = scale_system(norm_surfaces, norm_gaps, efl_target)
    stop_info = scale_stop_info(norm_stop, efl_target)
    return surfaces, gaps, stop_info


def write_modified_zmx(src_path, dst_path, curvature_updates=None,
                        thickness_updates=None, semi_dia_updates=None):
    """
    Read a ZMX file, update specified curvatures/thicknesses/semi-diameters,
    write new file.

    curvature_updates : dict {zemax_surface_number: new_curvature_value}
    thickness_updates : dict {zemax_surface_number: new_thickness_value}
    semi_dia_updates  : dict {zemax_surface_number: new_semi_diameter_value}
    """
    import re

    try:
        with open(src_path, 'r', encoding='utf-16') as f:
            text = f.read()
        encoding = 'utf-16'
    except (UnicodeError, UnicodeDecodeError):
        with open(src_path, 'r', encoding='utf-8') as f:
            text = f.read()
        encoding = 'utf-8'

    lines = text.split('\n')
    current_surf = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith('SURF'):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    current_surf = int(parts[1])
                except ValueError:
                    pass

        if current_surf is None:
            continue

        if curvature_updates and current_surf in curvature_updates:
            if stripped.startswith('CURV'):
                m = re.match(r'(\s*CURV\s+)(\S+)\s+(.*)', line)
                if m:
                    prefix, old_val, flags = m.groups()
                    new_c = curvature_updates[current_surf]
                    # Replace any curvature solve (type > 1) with
                    # simple variable (type 1) so Zemax keeps our value
                    flag_parts = flags.split()
                    if len(flag_parts) >= 1:
                        solve_type = int(flag_parts[0])
                        if solve_type > 1:
                            flag_parts[0] = '1'
                            # Clear pickup/solve params
                            for j in range(1, min(4, len(flag_parts))):
                                flag_parts[j] = '0'
                    new_flags = ' '.join(flag_parts)
                    lines[i] = f'{prefix}{new_c:.18E} {new_flags}'

        if thickness_updates and current_surf in thickness_updates:
            if stripped.startswith('DISZ') and 'INFINITY' not in stripped:
                m = re.match(r'(\s*DISZ\s+)(\S+)(.*)', line)
                if m:
                    prefix, old_val, suffix = m.groups()
                    new_d = thickness_updates[current_surf]
                    lines[i] = f'{prefix}{new_d:.18E}{suffix}'

        if semi_dia_updates and current_surf in semi_dia_updates:
            if stripped.startswith('DIAM'):
                m = re.match(r'(\s*DIAM\s+)(\S+)(.*)', line)
                if m:
                    prefix, old_val, rest = m.groups()
                    new_sd = semi_dia_updates[current_surf]
                    if new_sd == 0:
                        lines[i] = f'{prefix}0 0 0'
                    else:
                        lines[i] = f'{prefix}{new_sd:.6f}{rest}'

    with open(dst_path, 'w', encoding=encoding) as f:
        f.write('\n'.join(lines))


def zmx_to_trace_map(zmx_data):
    """
    Build mapping between Zemax surface numbers and trace surface indices.

    Returns
    -------
    zmx_to_trace : dict {zemax_surf_num: trace_idx}
    trace_to_zmx : dict {trace_idx: zemax_surf_num}
    """
    zmx_surfaces = zmx_data['surfaces']
    n_media = []
    for s in zmx_surfaces:
        n_media.append(s['n_d'] if s['glass'] is not None else 1.0)

    first_optical = 1
    last_optical = len(zmx_surfaces) - 2

    z2t = {}
    trace_idx = 0
    for i in range(first_optical, last_optical + 1):
        n_before = n_media[i - 1]
        n_after = n_media[i]
        if abs(n_before - n_after) < 1e-10:
            continue  # dummy surface
        z2t[i] = trace_idx
        trace_idx += 1

    t2z = {v: k for k, v in z2t.items()}
    return z2t, t2z


def build_system_from_zmx(zmx_data):
    """
    Convert parsed ZMX data into the format our y-nu trace expects.

    Dummy surfaces (n_before == n_after, e.g. aperture stops) are skipped.
    Their thickness is merged into the preceding gap.  The stop position
    is recorded as a fractional distance within the merged gap so that the
    chief ray constraint can be applied correctly.

    Returns
    -------
    surfaces_trace : list of (c, n, n')
    gaps_trace : list of (d, n)
    stop_info : dict with 'surface_idx' (trace index of the surface
        immediately after the gap that contains the stop) and
        'distance_before_stop' (distance from the preceding surface to
        the stop within that gap).  If the stop coincides with a real
        surface, 'surface_idx' is that surface's trace index and
        'distance_before_stop' is None.
    """
    zmx_surfaces = zmx_data['surfaces']

    # Medium after each Zemax surface
    n_media = []
    for s in zmx_surfaces:
        if s['glass'] is not None:
            n_media.append(s['n_d'])
        else:
            n_media.append(1.0)

    first_optical = 1
    last_optical = len(zmx_surfaces) - 2  # exclude image plane

    surfaces_trace = []
    gaps_trace = []

    # Map from Zemax surface index to trace surface index
    zmx_to_trace = {}
    # Accumulates thickness of skipped (dummy) surfaces
    pending_gap = 0.0
    pending_gap_n = None
    # Track where the stop falls
    stop_zmx_idx = zmx_data['stop_index']
    stop_in_gap_after_trace_surf = None  # trace surface index before the gap
    stop_distance_before = None          # distance within that gap

    for i in range(first_optical, last_optical + 1):
        s = zmx_surfaces[i]
        n_before = n_media[i - 1]
        n_after = n_media[i]

        is_dummy = (abs(n_before - n_after) < 1e-10)

        if is_dummy:
            # This surface has no optical effect — merge its thickness
            # into the pending gap
            if i == stop_zmx_idx:
                # The stop is at this dummy surface — record position
                # within the current pending gap
                stop_in_gap_after_trace_surf = len(surfaces_trace) - 1
                stop_distance_before = pending_gap
            pending_gap += s['thickness']
            continue

        # Real optical surface
        trace_idx = len(surfaces_trace)
        zmx_to_trace[i] = trace_idx

        if i == stop_zmx_idx:
            # Stop is at a real surface
            stop_in_gap_after_trace_surf = None
            stop_distance_before = None
            zmx_to_trace[i] = trace_idx

        surfaces_trace.append((s['curv'], n_before, n_after))

        # Emit gap from previous surface to this one (includes pending)
        if trace_idx > 0:
            gaps_trace.append((pending_gap, pending_gap_n))

        # Start accumulating the gap after this surface
        pending_gap = s['thickness'] if i < last_optical else 0.0
        pending_gap_n = n_media[i]

    # Build stop_info
    if stop_in_gap_after_trace_surf is not None:
        # Stop is inside a gap (dummy surface was removed)
        stop_info = {
            'type': 'in_gap',
            'gap_idx': stop_in_gap_after_trace_surf,
            'distance_before_stop': stop_distance_before,
        }
    elif stop_zmx_idx in zmx_to_trace:
        # Stop is at a real optical surface
        stop_info = {
            'type': 'at_surface',
            'surface_idx': zmx_to_trace[stop_zmx_idx],
        }
    else:
        # Fallback
        stop_info = {
            'type': 'at_surface',
            'surface_idx': 0,
        }

    return surfaces_trace, gaps_trace, stop_info


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


def ynu_trace_real(surfaces, gaps, h0, u0):
    """
    Exact meridional ray trace with real Snell's law.

    Uses exact sin/arcsin for refraction and tan for propagation.
    Ray intersection is at the vertex plane (no sag correction).

    Parameters
    ----------
    h0 : initial ray height (mm)
    u0 : initial ray angle (radians), 0 for collimated

    Returns
    -------
    h : (K,) ray heights at each surface
    u : (K,) real ray angles after refraction (radians)
    """
    K = len(surfaces)
    h = np.zeros(K)
    u = np.zeros(K)

    h[0] = h0
    c, n, n_prime = surfaces[0]

    if abs(c) < 1e-15 or abs(h0) < 1e-15:
        # Flat surface or on-axis: use paraxial for this surface
        if abs(n - n_prime) > 1e-10:
            sin_u0 = np.sin(u0)
            u[0] = np.arcsin((n / n_prime) * sin_u0)
        else:
            u[0] = u0
    else:
        alpha = np.arcsin(np.clip(h0 * c, -1, 1))
        I = u0 + alpha
        sin_I_prime = (n / n_prime) * np.sin(I)
        I_prime = np.arcsin(np.clip(sin_I_prime, -1, 1))
        u[0] = I_prime - alpha

    for i in range(1, K):
        d_gap, n_gap = gaps[i - 1]

        # Propagation: exact straight line
        h[i] = h[i - 1] + d_gap * np.tan(u[i - 1])

        # Refraction
        c, n, n_prime = surfaces[i]

        if abs(c) < 1e-15:
            # Flat surface
            if abs(n - n_prime) > 1e-10:
                sin_u = np.sin(u[i - 1])
                u[i] = np.arcsin(np.clip((n / n_prime) * sin_u, -1, 1))
            else:
                u[i] = u[i - 1]
        else:
            arg = h[i] * c
            if abs(arg) >= 1:
                u[i] = u[i - 1]  # ray beyond sphere
                continue
            alpha = np.arcsin(arg)
            I = u[i - 1] + alpha
            sin_I_prime = (n / n_prime) * np.sin(I)
            if abs(sin_I_prime) >= 1:
                u[i] = u[i - 1]  # TIR
                continue
            I_prime = np.arcsin(sin_I_prime)
            u[i] = I_prime - alpha

    return h, u


def _ray_height_at_stop(h, nu, stop_info, gaps):
    """
    Compute the ray height at the stop location.

    If the stop is at a surface, return h[surface_idx].
    If the stop is inside a gap (removed dummy surface), propagate
    from the surface before the gap to the stop position.
    """
    if stop_info['type'] == 'at_surface':
        return h[stop_info['surface_idx']]
    else:
        # Stop is in gap after surface gap_idx
        idx = stop_info['gap_idx']
        d_before = stop_info['distance_before_stop']
        n_gap = gaps[idx][1]
        return h[idx] + (d_before / n_gap) * nu[idx]


def find_chief_ray_initial(surfaces, gaps, stop_info, field_angle):
    """
    Find initial chief ray conditions so that h = 0 at the stop.

    stop_info can be a dict (new format from build_system_from_zmx) or
    an integer index (legacy format, stop at a surface).

    Linear in h0: trace with h0=0 and h0=1, interpolate.
    """
    # Support legacy integer stop_index
    if isinstance(stop_info, int):
        stop_info = {'type': 'at_surface', 'surface_idx': stop_info}

    n_air = 1.0
    nu0 = n_air * np.tan(np.radians(field_angle))

    # Trace with h0 = 0
    h_a, nu_a = ynu_trace(surfaces, gaps, 0.0, nu0)
    h_stop_a = _ray_height_at_stop(h_a, nu_a, stop_info, gaps)

    # Trace with h0 = 1
    h_b, nu_b = ynu_trace(surfaces, gaps, 1.0, nu0)
    h_stop_b = _ray_height_at_stop(h_b, nu_b, stop_info, gaps)

    # Linear interpolation: h_stop = h_stop_a + h0 * (h_stop_b - h_stop_a) = 0
    h0 = -h_stop_a / (h_stop_b - h_stop_a)

    return h0, nu0


def seidel_coefficients(surfaces, h_marginal, nu_marginal, h_chief, nu_chief,
                        nu0_marginal, nu0_chief):
    """Compute Seidel aberration coefficients at each surface."""
    K = len(surfaces)
    S_I = np.zeros(K)
    S_II = np.zeros(K)
    S_III = np.zeros(K)
    S_IV = np.zeros(K)
    S_V = np.zeros(K)

    # Lagrange invariant (using pre-refraction values at surface 0)
    H = nu0_marginal * h_chief[0] - nu0_chief * h_marginal[0]

    # Pre-refraction nu at each surface
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

        # Refraction invariants
        A = nu_m + n * h_marginal[i] * c
        A_bar = nu_c + n * h_chief[i] * c

        # Delta(u/n)
        u_before = nu_m / n
        u_after = nu_marginal[i] / n_prime
        delta_u_over_n = u_after / n_prime - u_before / n

        # Seidel contributions
        S_I[i] = -A**2 * h_marginal[i] * delta_u_over_n
        S_II[i] = -A * A_bar * h_marginal[i] * delta_u_over_n
        S_III[i] = -A_bar**2 * h_marginal[i] * delta_u_over_n
        S_IV[i] = H**2 * phi / (n * n_prime)
        S_V[i] = (S_III[i] + S_IV[i]) * (A_bar / A) if abs(A) > 1e-12 else 0.0

    return S_I, S_II, S_III, S_IV, S_V


def main():
    # Parse the ZMX file
    zmx_data = parse_zmx('sc_dbga1_opt2.zmx')

    field_type = zmx_data['field_type']
    field_type_names = {0: 'Angle (deg)', 1: 'Object height', 2: 'Paraxial image height (mm)', 3: 'Real image height'}

    print("=" * 70)
    print("ZEMAX FILE: sc_dbga1_opt2.zmx")
    print("=" * 70)
    print(f"  Entrance pupil diameter: {zmx_data['enpd']} mm")
    print(f"  Field type: {field_type} ({field_type_names.get(field_type, 'unknown')})")
    print(f"  Fields (Y): {zmx_data['fields_y']}")
    print(f"  Stop at surface: {zmx_data['stop_index']}")
    print()

    # Print surface table
    print(f"  {'Surf':>4s} {'Curv':>14s} {'Thick':>10s} {'Glass':>8s} {'n_d':>8s}")
    print(f"  {'----':>4s} {'----':>14s} {'-----':>10s} {'-----':>8s} {'---':>8s}")
    for s in zmx_data['surfaces']:
        t_str = "INF" if s['thickness'] == float('inf') else f"{s['thickness']:.4f}"
        g_str = s['glass'] if s['glass'] else "air"
        print(f"  {s['num']:4d} {s['curv']:14.8f} {t_str:>10s} {g_str:>8s} {s['n_d']:8.5f}")
    print()

    # Build trace system
    surfaces, gaps, stop_info = build_system_from_zmx(zmx_data)
    K = len(surfaces)

    print(f"  Optical surfaces: {K}")
    print(f"  Stop info: {stop_info}")
    print()

    # === Marginal ray ===
    # Object at infinity, collimated input at semi-aperture
    semi_aperture = zmx_data['enpd'] / 2.0
    h0_marginal = semi_aperture
    nu0_marginal = 0.0

    h_m, nu_m = ynu_trace(surfaces, gaps, h0_marginal, nu0_marginal)

    # Focal length and BFL
    u_final = nu_m[-1] / 1.0  # final medium is air
    f_trace = -h0_marginal / u_final
    bfl = -h_m[-1] / u_final

    print("MARGINAL RAY TRACE:")
    for i in range(K):
        c, n, np_ = surfaces[i]
        print(f"  Surf {i+1:2d}: h = {h_m[i]:10.4f}, nu = {nu_m[i]:12.6f}")
    print()
    print(f"  EFL (from trace): {f_trace:.4f} mm")
    print(f"  BFL (from trace): {bfl:.4f} mm")
    print()

    # === ABCD matrix verification ===
    M = np.eye(2)
    for i in range(K):
        c, n, n_prime = surfaces[i]
        phi = (n_prime - n) * c
        R = np.array([[1, 0], [-phi, 1]])
        M = R @ M
        if i < K - 1:
            d, n_gap = gaps[i]
            T = np.array([[1, d / n_gap], [0, 1]])
            M = T @ M

    f_matrix = -1.0 / M[1, 0]
    det_M = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    print(f"  EFL (ABCD matrix): {f_matrix:.4f} mm")
    print(f"  Trace vs Matrix diff: {abs(f_trace - f_matrix):.2e} mm")
    print(f"  det(M) = {det_M:.10f}")
    print()

    # === Chief ray (field = max field) ===
    max_field_value = max(zmx_data['fields_y'])

    if field_type == 0:
        # Field is angle in degrees — use directly
        max_field_angle = max_field_value
    elif field_type == 2:
        # Field is paraxial image height in mm
        # angle = arctan(image_height / EFL)
        max_field_angle = np.degrees(np.arctan(max_field_value / f_trace))
        print(f"  Field type 2: image height {max_field_value} mm -> angle {max_field_angle:.4f} deg")
    else:
        # Fallback: treat as angle
        max_field_angle = max_field_value

    h0_chief, nu0_chief = find_chief_ray_initial(surfaces, gaps, stop_info, max_field_angle)

    h_c, nu_c = ynu_trace(surfaces, gaps, h0_chief, nu0_chief)

    h_at_stop = _ray_height_at_stop(h_c, nu_c, stop_info, gaps)
    print(f"CHIEF RAY TRACE (field = {max_field_angle:.4f} deg, value = {max_field_value}):")
    print(f"  Initial: h0 = {h0_chief:.4f}, nu0 = {nu0_chief:.6f}")
    print(f"  h at stop: {h_at_stop:.2e} (should be ~0)")
    print()
    for i in range(K):
        print(f"  Surf {i+1:2d}: h = {h_c[i]:10.4f}, nu = {nu_c[i]:12.6f}")
    print()

    # === Seidel coefficients ===
    S_I, S_II, S_III, S_IV, S_V = seidel_coefficients(
        surfaces, h_m, nu_m, h_c, nu_c, nu0_marginal, nu0_chief
    )

    print("SEIDEL COEFFICIENTS (per surface):")
    print(f"  {'Surf':>4s} {'S_I':>12s} {'S_II':>12s} {'S_III':>12s} {'S_IV':>12s} {'S_V':>12s}")
    print(f"  {'----':>4s} {'---':>12s} {'----':>12s} {'-----':>12s} {'----':>12s} {'---':>12s}")
    for i in range(K):
        print(f"  {i+1:4d} {S_I[i]:12.6f} {S_II[i]:12.6f} {S_III[i]:12.6f} {S_IV[i]:12.6f} {S_V[i]:12.6f}")
    print(f"  {'SUM':>4s} {S_I.sum():12.6f} {S_II.sum():12.6f} {S_III.sum():12.6f} {S_IV.sum():12.6f} {S_V.sum():12.6f}")
    print()

    # Lagrange invariant
    H = nu0_marginal * h_c[0] - nu0_chief * h_m[0]
    print(f"  Lagrange invariant H = {H:.6f}")
    print()

    # Merit function
    weights = np.ones(5)
    S_total = np.array([S_I.sum(), S_II.sum(), S_III.sum(), S_IV.sum(), S_V.sum()])
    merit = np.sum(weights * S_total**2)
    print(f"MERIT FUNCTION: {merit:.6e}")
    for name, val in zip(['S_I', 'S_II', 'S_III', 'S_IV', 'S_V'], S_total):
        print(f"  {name:5s} = {val:+.6f}")


if __name__ == "__main__":
    main()
