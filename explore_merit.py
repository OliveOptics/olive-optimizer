"""
Explore the merit function landscape for a 2-singlet system.

System layout:
    [Lens 1: S1--S2]  --air gap--  [Lens 2: S3--S4]

Fixed: glass thicknesses, refractive indices, c2=c4=0 (plano back surfaces)
Free:  c1, c3 (front curvatures), d_air (air gap)

Visualizes the full (c1 x c3 x d_air) space as a single heatmap:
  X = all (c1, c3) combinations
  Y = d_air
  Color = log10(merit + EFL penalty)
"""

import numpy as np
import matplotlib.pyplot as plt
from parse_zmx import ynu_trace, seidel_coefficients


# ── System builder ───────────────────────────────────────────────────────────

def build_two_singlet_system(c1, c2, c3, c4, d_air,
                              n1=1.5, n2=1.5, d1=5.0, d2=5.0):
    """Build a 2-singlet system from design variables."""
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


# ── Merit function ───────────────────────────────────────────────────────────

def compute_merit(c1, c2, c3, c4, d_air,
                  semi_aperture=10.0, field_angle_deg=3.0,
                  n1=1.5, n2=1.5, d1=5.0, d2=5.0,
                  stop_idx=0, weights=None):
    """
    Compute the Seidel merit function for the 2-singlet system.
    Returns merit, S_total, efl (or None if system is invalid).
    """
    if weights is None:
        weights = np.ones(5)

    surfaces, gaps = build_two_singlet_system(c1, c2, c3, c4, d_air, n1, n2, d1, d2)

    h_m, nu_m = ynu_trace(surfaces, gaps, semi_aperture, 0.0)

    if abs(nu_m[-1]) < 1e-12:
        return None, None, None

    efl = -semi_aperture / nu_m[-1]

    nu0_chief = np.tan(np.radians(field_angle_deg))
    h_a, _ = ynu_trace(surfaces, gaps, 0.0, nu0_chief)
    h_b, _ = ynu_trace(surfaces, gaps, 1.0, nu0_chief)
    denom = h_b[stop_idx] - h_a[stop_idx]
    if abs(denom) < 1e-12:
        return None, None, None
    h0_chief = -h_a[stop_idx] / denom
    h_c, nu_c = ynu_trace(surfaces, gaps, h0_chief, nu0_chief)

    S_I, S_II, S_III, S_IV, S_V = seidel_coefficients(
        surfaces, h_m, nu_m, h_c, nu_c, 0.0, nu0_chief)

    S_total = np.array([S_I.sum(), S_II.sum(), S_III.sum(), S_IV.sum(), S_V.sum()])
    merit = np.sum(weights * S_total**2)

    return merit, S_total, efl


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # System parameters
    common = dict(semi_aperture=10.0, field_angle_deg=3.0,
                  n1=1.5, n2=1.5, d1=5.0, d2=5.0, stop_idx=0)
    EFL_TARGET = 80.0
    EFL_PENALTY = 100.0

    c2_fixed = 0.0
    c4_fixed = 0.0

    Nc = 50
    Nd = 50
    c1_range = np.linspace(-0.01, 0.02, Nc)
    c3_range = np.linspace(0.005, 0.035, Nc)
    d_range = np.linspace(3, 20, Nd)

    # X axis: all (c1, c3) combos = Nc^2 points
    # ordering: c3 is outer loop, c1 varies within each c3 block
    Nx = Nc * Nc
    Z = np.full((Nd, Nx), np.nan)

    print(f"Sweeping {Nc}x{Nc}x{Nd} = {Nc*Nc*Nd} evaluations...")
    for di in range(Nd):
        for i3 in range(Nc):
            for i1 in range(Nc):
                xi = i3 * Nc + i1
                m, S, efl = compute_merit(c1_range[i1], c2_fixed,
                                          c3_range[i3], c4_fixed,
                                          d_range[di], **common)
                if m is not None:
                    mt = m + EFL_PENALTY * (efl - EFL_TARGET)**2
                    Z[di, xi] = np.log10(max(mt, 1e-12))

    # Find local minima (valleys) by looking at each d_air row
    # For each column of d_air, find local minima in the X direction
    from scipy.ndimage import minimum_filter, label

    # Smooth slightly to avoid noise, then find local minima
    Z_filled = np.where(np.isnan(Z), np.nanmax(Z) + 1, Z)
    local_min = (Z_filled == minimum_filter(Z_filled, size=(5, 5)))
    # Only keep minima that are actually low (within 2 decades of global best)
    global_min = np.nanmin(Z)
    local_min &= (Z_filled < global_min + 2)

    # Cluster nearby minima
    labeled, n_features = label(local_min)
    valleys = []
    for i in range(1, n_features + 1):
        mask = labeled == i
        # Find the pixel with lowest merit in this cluster
        cluster_Z = np.where(mask, Z_filled, np.inf)
        min_idx = np.unravel_index(np.argmin(cluster_Z), cluster_Z.shape)
        di, xi = min_idx
        i3, i1 = divmod(xi, Nc)
        valleys.append(dict(
            di=di, xi=xi, i3=i3, i1=i1,
            c1=c1_range[i1], c3=c3_range[i3], d_air=d_range[di],
            merit=10**Z[di, xi], log_merit=Z[di, xi]))

    # Sort by merit
    valleys.sort(key=lambda v: v['merit'])

    # Keep top N distinct valleys (deduplicate by distance in parameter space)
    unique_valleys = []
    for v in valleys:
        is_dup = False
        for u in unique_valleys:
            if (abs(v['c1'] - u['c1']) < 0.003 and
                abs(v['c3'] - u['c3']) < 0.003 and
                abs(v['d_air'] - u['d_air']) < 3):
                is_dup = True
                break
        if not is_dup:
            unique_valleys.append(v)
        if len(unique_valleys) >= 10:
            break

    print(f"\nFound {len(unique_valleys)} distinct valleys:")
    for i, v in enumerate(unique_valleys):
        print(f"  #{i+1}: c1={v['c1']:.4f}, c3={v['c3']:.4f}, "
              f"d_air={v['d_air']:.1f}, merit={v['merit']:.4e}")

    # Clip color range: from best to best+4 decades
    vmin = np.nanmin(Z)
    vmax = min(vmin + 4, np.nanmax(Z))

    fig, ax = plt.subplots(figsize=(20, 6))
    im = ax.imshow(Z, aspect='auto', origin='lower', cmap='viridis_r',
                   extent=[0, Nx, d_range[0], d_range[-1]],
                   interpolation='nearest', vmin=vmin, vmax=vmax)

    # Mark c3 group boundaries
    for i3 in range(1, Nc):
        ax.axvline(i3 * Nc, color='white', lw=0.3, alpha=0.4)

    # c3 group labels
    for i3 in range(0, Nc, 5):
        cx = i3 * Nc + Nc // 2
        ax.text(cx, d_range[-1] + 2, f"c3={c3_range[i3]:.3f}",
                ha='center', va='bottom', fontsize=5, rotation=90)

    # X ticks
    xtick_pos = []
    xtick_labels = []
    for i3 in [0, Nc // 4, Nc // 2, 3 * Nc // 4, Nc - 1]:
        for i1 in [0, Nc // 2, Nc - 1]:
            xtick_pos.append(i3 * Nc + i1)
            xtick_labels.append(f"c3={c3_range[i3]:.2f}\nc1={c1_range[i1]:.2f}")

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=5)

    # Label valleys with numbers
    for i, v in enumerate(unique_valleys):
        ax.plot(v['xi'], v['d_air'], 'o', color='red', ms=8, mew=1.5,
                markerfacecolor='none')
        ax.annotate(f"#{i+1}\n{v['merit']:.1e}",
                    xy=(v['xi'], v['d_air']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=7, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    ax.set_ylabel('d_air (mm)', fontsize=12)
    ax.set_xlabel('(c1, c3) combinations — c1 varies within each c3 block',
                  fontsize=10)
    plt.colorbar(im, ax=ax, label='log10(merit + EFL penalty)', shrink=0.9)
    ax.set_title(
        f"Full (c1 x c3 x d_air) sweep — c2=c4=0 (plano)  |  "
        f"EFL target={EFL_TARGET}mm  |  {Nc}x{Nc}x{Nd} = {Nc*Nc*Nd} evals",
        fontsize=11)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
