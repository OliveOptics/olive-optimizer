"""
Test: y-nu paraxial trace + Seidel aberrations for a singlet lens.

System: plano-convex singlet in air
  Surface 1: curved (c1 = 1/R1), air -> glass (n=1.5)
  Gap: thickness d in glass
  Surface 2: flat (c2 = 0), glass -> air

Object at infinity (collimated input), so marginal ray enters parallel to axis.
"""

import numpy as np
from parse_zmx import ynu_trace, seidel_coefficients


def main():
    # === Define a plano-convex singlet ===
    # Focal length target: 100 mm
    # Glass: n = 1.5
    # Front surface: convex, back surface: flat
    # For a plano-convex lens: 1/f = (n-1) * c1
    # So c1 = 1 / ((n-1) * f) = 1 / (0.5 * 100) = 0.02 mm^-1

    n_glass = 1.5
    n_air = 1.0
    c1 = 0.02  # mm^-1
    c2 = -0.01  # mm^-1 (biconvex lens)
    d = 5.0  # lens thickness in mm

    # Lensmaker's equation (thick lens):
    # P = (n-1)[c1 - c2 + (n-1)*d*c1*c2/n]
    # where P = 1/f (total power)
    dn = n_glass - n_air
    P_thick = dn * (c1 - c2 + dn * d * c1 * c2 / n_glass)
    f_lensmaker = 1.0 / P_thick

    print("=" * 60)
    print("SINGLET LENS: Biconvex")
    print("=" * 60)
    print(f"  c1 = {c1:.6f} mm^-1  (R1 = {1/c1:.1f} mm)")
    print(f"  c2 = {c2:.6f} mm^-1  (R2 = {1/c2:.1f} mm)")
    print(f"  thickness = {d} mm")
    print(f"  n_glass = {n_glass}")
    print(f"  Focal length (lensmaker's eq): {f_lensmaker:.4f} mm")
    print()

    # Surface definitions: (curvature, n_before, n_after)
    surfaces = [
        (c1, n_air, n_glass),   # Surface 1: air -> glass
        (c2, n_glass, n_air),   # Surface 2: glass -> air
    ]

    # Gaps: (thickness, index of medium)
    gaps = [
        (d, n_glass),  # gap between surface 1 and 2, in glass
    ]

    # === Marginal ray ===
    # Object at infinity: ray enters parallel to axis (u=0, nu=0)
    # at the edge of the aperture (h = aperture semi-diameter)
    aperture = 10.0  # mm semi-diameter (f/5 system)
    h0_marginal = aperture
    nu0_marginal = 0.0  # collimated input

    h_m, nu_m = ynu_trace(surfaces, gaps, h0_marginal, nu0_marginal)

    print("MARGINAL RAY TRACE:")
    print(f"  Surface 1: h = {h_m[0]:.4f}, nu_after = {nu_m[0]:.6f}")
    print(f"  Surface 2: h = {h_m[1]:.4f}, nu_after = {nu_m[1]:.6f}")

    # Focal length from trace: f = -h_input / u_final
    # (collimated ray at height h converges to focus at distance f from principal plane)
    u_final = nu_m[-1] / n_air
    f_from_trace = -h0_marginal / u_final

    print()
    print("FOCAL LENGTH COMPARISON:")
    print(f"  From y-nu trace:       f = {f_from_trace:.4f} mm")
    print(f"  From lensmaker's eq:   f = {f_lensmaker:.4f} mm")
    print(f"  Difference:            {abs(f_from_trace - f_lensmaker):.6f} mm")
    print()

    # === Chief ray ===
    # For object at infinity, chief ray passes through center of stop.
    # Assume stop is at surface 1 (front stop).
    # Chief ray: h = 0 at stop, angle = field angle
    field_angle = 0.05  # radians (~2.9 degrees)
    h0_chief = 0.0  # passes through center of stop at surface 1
    nu0_chief = n_air * field_angle  # nu = n * u

    h_c, nu_c = ynu_trace(surfaces, gaps, h0_chief, nu0_chief)

    print("CHIEF RAY TRACE:")
    print(f"  Surface 1: h = {h_c[0]:.4f}, nu_after = {nu_c[0]:.6f}")
    print(f"  Surface 2: h = {h_c[1]:.4f}, nu_after = {nu_c[1]:.6f}")
    print()

    # === Seidel coefficients ===
    S_I, S_II, S_III, S_IV, S_V = seidel_coefficients(
        surfaces, h_m, nu_m, h_c, nu_c, nu0_marginal, nu0_chief
    )

    print("SEIDEL COEFFICIENTS (per surface):")
    print(f"  {'':12s} {'Surface 1':>12s} {'Surface 2':>12s} {'Total':>12s}")
    print(f"  {'S_I (sph)':12s} {S_I[0]:12.6f} {S_I[1]:12.6f} {S_I.sum():12.6f}")
    print(f"  {'S_II (coma)':12s} {S_II[0]:12.6f} {S_II[1]:12.6f} {S_II.sum():12.6f}")
    print(f"  {'S_III (ast)':12s} {S_III[0]:12.6f} {S_III[1]:12.6f} {S_III.sum():12.6f}")
    print(f"  {'S_IV (petz)':12s} {S_IV[0]:12.6f} {S_IV[1]:12.6f} {S_IV.sum():12.6f}")
    print(f"  {'S_V (dist)':12s} {S_V[0]:12.6f} {S_V[1]:12.6f} {S_V.sum():12.6f}")
    print()

    # === Sanity checks ===
    print("SANITY CHECKS:")

    # 1. Both surfaces contribute to S_I (both have nonzero curvature and refraction)
    print(f"  S_I at surface 1: {S_I[0]:.6f}")
    print(f"  S_I at surface 2: {S_I[1]:.6f}")

    # 2. For plano-convex with stop at front, chief ray height at surface 1 is 0
    #    so S_II should come only from surface 2 (where h_bar != 0)
    print(f"  Chief ray h at surface 1: {h_c[0]:.4f} (expect 0)")
    print(f"  Chief ray h at surface 2: {h_c[1]:.4f} (expect > 0)")

    # 3. Lagrange invariant
    H = nu0_marginal * h_c[0] - nu0_chief * h_m[0]
    print(f"  Lagrange invariant H: {H:.6f}")

    # 4. S_IV (Petzval) — depends only on power and indices, not ray heights
    #    For a thin lens: Petzval sum = 1/f. For thick lens it's approximate.
    petzval_sum = 0
    for i, (c, n, n_prime) in enumerate(surfaces):
        phi = (n_prime - n) * c
        petzval_sum += phi / (n * n_prime)
    print(f"  Petzval sum Σ φ/(n·n'): {petzval_sum:.6f}")
    print(f"  1/f (from trace):       {1/f_from_trace:.6f}")
    print(f"  1/f (lensmaker):        {1/f_lensmaker:.6f}")
    print()

    # === Merit function ===
    # Equal weights on all aberrations
    weights = np.ones(5)
    S_total = np.array([S_I.sum(), S_II.sum(), S_III.sum(), S_IV.sum(), S_V.sum()])
    merit = np.sum(weights * S_total**2)
    print(f"MERIT FUNCTION: {merit:.6e}")
    print(f"  (dominated by S_I = {S_I.sum():.6f} — spherical aberration)")


def test_two_singlets():
    """
    Two singlet lenses separated by an air gap.
    4 surfaces total: glass1_front, glass1_back, glass2_front, glass2_back.

    Verify focal length from y-nu trace against the thick-lens combination formula:
        P_total = P1 + P2 - t * P1 * P2
    where t is the reduced distance between the rear principal plane of lens 1
    and the front principal plane of lens 2.

    For simplicity, we use the surface-by-surface power approach:
    the y-nu trace automatically handles everything correctly, so we just
    compare against the formula for combined power of two thick lenses.
    """
    n_glass1 = 1.5
    n_glass2 = 1.7  # different glass for lens 2
    n_air = 1.0

    # Lens 1: biconvex
    c1 = 0.02    # front surface
    c2 = -0.01   # back surface
    d1 = 5.0     # thickness

    # Lens 2: meniscus
    c3 = 0.015   # front surface
    c4 = 0.005   # back surface (same sign = meniscus)
    d2 = 4.0     # thickness

    # Air gap between lenses
    gap = 20.0  # mm

    print("\n")
    print("=" * 60)
    print("TWO SINGLETS SEPARATED BY AIR GAP")
    print("=" * 60)
    print(f"  Lens 1: c1={c1}, c2={c2}, d={d1} mm, n={n_glass1}")
    print(f"  Lens 2: c3={c3}, c4={c4}, d={d2} mm, n={n_glass2}")
    print(f"  Air gap: {gap} mm")
    print()

    # === Individual lens powers (lensmaker's thick lens) ===
    dn1 = n_glass1 - n_air
    P1 = dn1 * (c1 - c2 + dn1 * d1 * c1 * c2 / n_glass1)
    f1 = 1.0 / P1

    dn2 = n_glass2 - n_air
    P2 = dn2 * (c3 - c4 + dn2 * d2 * c3 * c4 / n_glass2)
    f2 = 1.0 / P2

    print(f"  Lens 1 power: P1 = {P1:.6f}, f1 = {f1:.2f} mm")
    print(f"  Lens 2 power: P2 = {P2:.6f}, f2 = {f2:.2f} mm")

    # === Combined system: use y-nu trace ===
    surfaces = [
        (c1, n_air, n_glass1),    # Lens 1 front
        (c2, n_glass1, n_air),    # Lens 1 back
        (c3, n_air, n_glass2),    # Lens 2 front
        (c4, n_glass2, n_air),    # Lens 2 back
    ]

    gaps = [
        (d1, n_glass1),   # inside lens 1
        (gap, n_air),     # air gap between lenses
        (d2, n_glass2),   # inside lens 2
    ]

    # Marginal ray: collimated input
    aperture = 10.0
    h0_marginal = aperture
    nu0_marginal = 0.0

    h_m, nu_m = ynu_trace(surfaces, gaps, h0_marginal, nu0_marginal)

    print()
    print("MARGINAL RAY TRACE:")
    for i in range(4):
        print(f"  Surface {i+1}: h = {h_m[i]:.4f}, nu_after = {nu_m[i]:.6f}")

    # Focal length from trace
    u_final = nu_m[-1] / n_air
    f_from_trace = -h0_marginal / u_final

    # === Analytical combined focal length ===
    # For two thick lenses separated by distance t (measured between principal planes),
    # the exact approach is to use the system matrix (ABCD matrix).
    #
    # System matrix = R4 * T3 * R3 * T2 * R2 * T1 * R1
    # where R = refraction matrix, T = transfer matrix
    # Then P_system = -C element (for the ray transfer matrix [A B; C D])
    # and f = -1/C

    # Build ABCD matrix
    M = np.eye(2)
    # Convention: state vector is [y, nu], matrix acts on it
    # Refraction: [[1, 0], [-phi, 1]]
    # Transfer:   [[1, t], [0, 1]]  where t = d/n

    all_elements = []
    for i in range(4):
        c, n, n_prime = surfaces[i]
        phi = (n_prime - n) * c
        all_elements.append(('refract', phi))
        if i < 3:
            d, n_gap = gaps[i]
            all_elements.append(('transfer', d / n_gap))

    for elem_type, val in all_elements:
        if elem_type == 'refract':
            R = np.array([[1, 0], [-val, 1]])
            M = R @ M
        else:
            T = np.array([[1, val], [0, 1]])
            M = T @ M

    # System matrix M maps [y_in, nu_in] -> [y_out, nu_out]
    # For collimated input (nu_in=0): nu_out = M[1,0] * y_in
    # So P_system = -M[1,0] and f = 1/P = -1/M[1,0]...
    # Actually: nu_out = -P * y_in for a system, so P = -M[1,0]
    # Wait, let's just read it from the matrix properly:
    # [y_out]   [A  B] [y_in ]
    # [nu_out] = [C  D] [nu_in]
    # For nu_in = 0: nu_out = C * y_in
    # And f = -y_in / u_out = -y_in / (nu_out/n_out) = -y_in*n_out / nu_out
    # With n_out = 1 (air): f = -y_in / nu_out = -1/C
    C_mat = M[1, 0]
    f_from_matrix = -1.0 / C_mat
    det_M = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]

    print()
    print("FOCAL LENGTH COMPARISON:")
    print(f"  From y-nu trace:       f = {f_from_trace:.4f} mm")
    print(f"  From ABCD matrix:      f = {f_from_matrix:.4f} mm")
    print(f"  Difference:            {abs(f_from_trace - f_from_matrix):.2e} mm")
    print()
    print(f"  System matrix:")
    print(f"    [{M[0,0]:.6f}  {M[0,1]:.6f}]")
    print(f"    [{M[1,0]:.6f}  {M[1,1]:.6f}]")
    print(f"    det(M) = {det_M:.6f} (expect 1.0)")
    print()

    # === Chief ray + Seidel ===
    field_angle = 0.03  # radians
    h0_chief = 0.0  # stop at surface 1
    nu0_chief = n_air * field_angle

    h_c, nu_c = ynu_trace(surfaces, gaps, h0_chief, nu0_chief)

    print("CHIEF RAY TRACE:")
    for i in range(4):
        print(f"  Surface {i+1}: h = {h_c[i]:.4f}, nu_after = {nu_c[i]:.6f}")
    print()

    # Seidel coefficients
    S_I, S_II, S_III, S_IV, S_V = seidel_coefficients(
        surfaces, h_m, nu_m, h_c, nu_c, nu0_marginal, nu0_chief
    )

    print("SEIDEL COEFFICIENTS (per surface):")
    header = f"  {'':12s}" + "".join(f" {'Surf '+str(i+1):>10s}" for i in range(4)) + f" {'Total':>10s}"
    print(header)
    for name, S in [("S_I (sph)", S_I), ("S_II (coma)", S_II),
                    ("S_III (ast)", S_III), ("S_IV (petz)", S_IV), ("S_V (dist)", S_V)]:
        vals = "".join(f" {S[i]:10.6f}" for i in range(4)) + f" {S.sum():10.6f}"
        print(f"  {name:12s}{vals}")
    print()

    # Merit function
    weights = np.ones(5)
    S_total = np.array([S_I.sum(), S_II.sum(), S_III.sum(), S_IV.sum(), S_V.sum()])
    merit = np.sum(weights * S_total**2)
    print(f"MERIT FUNCTION: {merit:.6e}")


def test_four_lenses():
    """
    4-lens system (8 surfaces): resembles a double-Gauss or Tessar-like layout.
    Mix of positive/negative lenses, different glasses, varying air gaps.
    """
    n_air = 1.0

    # Lens 1: positive crown (n=1.517)
    # Lens 2: negative flint (n=1.620) — close to lens 1 (cemented-like gap)
    # Lens 3: negative flint (n=1.620)
    # Lens 4: positive crown (n=1.517)
    # Symmetric-ish layout for partial aberration cancellation

    # Surface prescriptions: (curvature c, n_before, n_after)
    #   R (mm)     c (mm^-1)
    #   40         0.025
    #  -200       -0.005
    #  -50        -0.020
    #   80         0.0125
    #  -80        -0.0125
    #   50         0.020
    #   200        0.005
    #  -40        -0.025

    surfaces = [
        ( 0.025,  n_air, 1.517),   # L1 front
        (-0.005,  1.517, n_air),   # L1 back
        (-0.020,  n_air, 1.620),   # L2 front
        ( 0.0125, 1.620, n_air),   # L2 back
        (-0.0125, n_air, 1.620),   # L3 front
        ( 0.020,  1.620, n_air),   # L3 back
        ( 0.005,  n_air, 1.517),   # L4 front
        (-0.025,  1.517, n_air),   # L4 back
    ]

    # Thicknesses and air gaps
    gaps = [
        (6.0, 1.517),   # L1 thickness
        (2.0, n_air),   # air gap L1-L2
        (3.0, 1.620),   # L2 thickness
        (12.0, n_air),  # air gap L2-L3 (larger — stop region)
        (3.0, 1.620),   # L3 thickness
        (2.0, n_air),   # air gap L3-L4
        (6.0, 1.517),   # L4 thickness
    ]

    K = len(surfaces)

    print("\n")
    print("=" * 60)
    print("FOUR-LENS SYSTEM (8 surfaces)")
    print("=" * 60)
    print()
    print(f"  {'Surf':>4s} {'R (mm)':>10s} {'c (mm^-1)':>10s} {'n':>6s} {'n_prime':>8s}")
    print(f"  {'----':>4s} {'------':>10s} {'---------':>10s} {'---':>6s} {'-------':>8s}")
    for i, (c, n, np_) in enumerate(surfaces):
        R_str = f"{1/c:.1f}" if c != 0 else "inf"
        print(f"  {i+1:4d} {R_str:>10s} {c:>10.4f} {n:>6.3f} {np_:>8.3f}")
    print()
    print(f"  {'Gap':>4s} {'d (mm)':>8s} {'n_medium':>8s} {'Element':>12s}")
    print(f"  {'---':>4s} {'------':>8s} {'-------':>8s} {'-------':>12s}")
    labels = ["L1 glass", "air L1-L2", "L2 glass", "air (stop)", "L3 glass", "air L3-L4", "L4 glass"]
    for i, (d, n) in enumerate(gaps):
        print(f"  {i+1:4d} {d:>8.1f} {n:>8.3f} {labels[i]:>12s}")
    print()

    # === Marginal ray (collimated input) ===
    aperture = 8.0  # mm semi-diameter
    h0_marginal = aperture
    nu0_marginal = 0.0

    h_m, nu_m = ynu_trace(surfaces, gaps, h0_marginal, nu0_marginal)

    # Focal length from trace
    u_final = nu_m[-1] / n_air
    f_from_trace = -h0_marginal / u_final

    # === ABCD matrix for verification ===
    M = np.eye(2)
    all_elements = []
    for i in range(K):
        c, n, n_prime = surfaces[i]
        phi = (n_prime - n) * c
        all_elements.append(('refract', phi))
        if i < K - 1:
            d, n_gap = gaps[i]
            all_elements.append(('transfer', d / n_gap))

    for elem_type, val in all_elements:
        if elem_type == 'refract':
            R = np.array([[1, 0], [-val, 1]])
            M = R @ M
        else:
            T = np.array([[1, val], [0, 1]])
            M = T @ M

    C_mat = M[1, 0]
    f_from_matrix = -1.0 / C_mat
    det_M = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]

    print("MARGINAL RAY TRACE:")
    for i in range(K):
        print(f"  Surface {i+1}: h = {h_m[i]:8.4f}, nu_after = {nu_m[i]:10.6f}")

    # BFL = distance from last surface to focus
    # For collimated input: marginal ray crosses axis at BFL after last surface
    # BFL = -h_last / u_last
    h_last = h_m[-1]
    u_last = nu_m[-1] / n_air  # final medium is air
    bfl = -h_last / u_last

    print()
    print("FOCAL LENGTH COMPARISON:")
    print(f"  From y-nu trace:       f = {f_from_trace:.4f} mm")
    print(f"  From ABCD matrix:      f = {f_from_matrix:.4f} mm")
    print(f"  Difference:            {abs(f_from_trace - f_from_matrix):.2e} mm")
    print(f"  det(M) = {det_M:.10f} (expect 1.0)")
    print()
    print(f"  BFL (back focal length): {bfl:.4f} mm")
    print(f"    (h_last = {h_last:.4f}, u_last = {u_last:.6f})")
    print()

    # === Chief ray ===
    # Place stop in the air gap between L2 and L3 (gap index 3, d=12mm)
    # Chief ray must have h=0 at the stop location.
    # Stop is 6mm into the 12mm gap (midpoint).
    # We need to find initial chief ray conditions such that h=0 at the stop.
    # Approach: trace a ray from surface 1 to the stop and solve for h0 given angle.
    #
    # Simpler: set stop at surface 1 for now (same as before).
    field_angle = 0.04  # radians (~2.3 degrees)
    h0_chief = 0.0  # stop at surface 1
    nu0_chief = n_air * field_angle

    h_c, nu_c = ynu_trace(surfaces, gaps, h0_chief, nu0_chief)

    print("CHIEF RAY TRACE:")
    for i in range(K):
        print(f"  Surface {i+1}: h = {h_c[i]:8.4f}, nu_after = {nu_c[i]:10.6f}")
    print()

    # === Seidel coefficients ===
    S_I, S_II, S_III, S_IV, S_V = seidel_coefficients(
        surfaces, h_m, nu_m, h_c, nu_c, nu0_marginal, nu0_chief
    )

    print("SEIDEL COEFFICIENTS (per surface):")
    header = f"  {'':12s}" + "".join(f" {'S'+str(i+1):>9s}" for i in range(K)) + f" {'Total':>10s}"
    print(header)
    for name, S in [("S_I (sph)", S_I), ("S_II (coma)", S_II),
                    ("S_III (ast)", S_III), ("S_IV (petz)", S_IV), ("S_V (dist)", S_V)]:
        vals = "".join(f" {S[i]:9.5f}" for i in range(K)) + f" {S.sum():10.6f}"
        print(f"  {name:12s}{vals}")
    print()

    # Petzval sum
    petzval_sum = 0
    for c, n, n_prime in surfaces:
        phi = (n_prime - n) * c
        petzval_sum += phi / (n * n_prime)
    print(f"  Petzval sum Σ φ/(n·n'): {petzval_sum:.6f}")
    print(f"  1/f (from trace):       {1/f_from_trace:.6f}")
    print()

    # Merit function
    weights = np.ones(5)
    S_total = np.array([S_I.sum(), S_II.sum(), S_III.sum(), S_IV.sum(), S_V.sum()])
    merit = np.sum(weights * S_total**2)
    print(f"MERIT FUNCTION: {merit:.6e}")
    print(f"  S_I  = {S_total[0]:+.6f}")
    print(f"  S_II = {S_total[1]:+.6f}")
    print(f"  S_III= {S_total[2]:+.6f}")
    print(f"  S_IV = {S_total[3]:+.6f}")
    print(f"  S_V  = {S_total[4]:+.6f}")


if __name__ == "__main__":
    main()
    test_two_singlets()
    test_four_lenses()
