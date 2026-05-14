"""Milestone 4: Cooke triplet LM benchmark.

A classical f/4, 50 mm EFL Cooke triplet — three elements (crown / flint /
crown), six surfaces, two glasses (SK16 / F2 / SK16). Prescription is a
standard reference design (see "Modern Lens Design" by W. J. Smith,
chapter on Cooke triplets).

This module is both a runnable demo and a builder used by
`tests/test_lm_cooke.py`. The builders are at module scope so they can be
imported; the demo is in `main()`.
"""
import math
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)

from lens_opt.tracer.surfaces import make_surface, Ray
from lens_opt.tracer.glass import Air, SK16, F2
from lens_opt.tracer.system import System, forward_trace
from lens_opt.tracer.sources import pupil_fan
from lens_opt.optim.pack import DesignTemplate, unpack


# Sampling layout for the merit. The flat residual vector is ordered
# field-major: residuals for field 0 come first, then field 1, then field 2.
# Within each field, wavelengths are in WAVELENGTHS_UM order, and within
# each (field, wavelength) the rays sweep -1 → +1 normalized pupil.
FIELD_ANGLES_DEG = (0.0, 10.0, 14.0)
WAVELENGTHS_UM   = (0.4861, 0.5876, 0.6563)   # F, d, C
N_RAYS_PER_FAN   = 7
RAYS_PER_FIELD   = N_RAYS_PER_FAN * len(WAVELENGTHS_UM)
N_RAYS_TOTAL     = RAYS_PER_FIELD * len(FIELD_ANGLES_DEG)


# ──────────────────────────────────────────────────────────────────────────
# Cooke triplet prescription
# ──────────────────────────────────────────────────────────────────────────
# Standard f/4 50 mm reference (Smith, "Modern Lens Design"). Approximately:
#
#   surf   R (mm)      t after (mm)   glass after
#   1      +22.014     3.259          SK16
#   2      -435.760    6.008          air
#   3      -22.213     1.000          F2
#   4      +20.292     4.750          air
#   5      +79.684     2.952          SK16
#   6      -18.395     41.225         air (BFL — to image plane)
#
# Aperture stop is between surfaces 2 and 3 (the air gap before the flint).
# We don't model an explicit stop here; the pupil sampling implicitly sets
# the aperture by ray height at the lens.

PRESCRIPTION = {
    "radii":          (22.014, -435.760, -22.213, 20.292, 79.684, -18.395),
    "thicknesses":    (3.259, 6.008, 1.000, 4.750, 2.952, 41.225),
    "glass_after":    ("SK16", "air", "F2", "air", "SK16", "air"),
    # Per-surface semi-diameters (mm) — the physical glass rim, also acting
    # as the clear aperture for vignetting. Outer elements get ~30% margin
    # above EPD/2 = 6.25 mm to admit off-axis bundles. The middle element
    # is tighter — its rim *is* the de-facto aperture stop in this
    # simplified model (we don't have an explicit AS surface).
    "semi_diameters": (8.0, 8.0, 6.5, 6.5, 8.0, 8.0),
    "epd_mm":         12.5,   # entrance pupil diameter (f/4 at 50 mm EFL)
    "efl_mm":         50.0,
}

_GLASS_MAP = {"air": Air, "SK16": SK16, "F2": F2}


def cooke_triplet_system(prescription=PRESCRIPTION, *, device=None) -> System:
    """Build the Cooke triplet System from a prescription dict."""
    dtype = torch.float64
    surfaces = tuple(
        make_surface(c=1.0 / R, device=device) for R in prescription["radii"]
    )
    gap_thicknesses = tuple(
        torch.tensor(t, dtype=dtype, device=device)
        for t in prescription["thicknesses"]
    )
    gap_materials = tuple(
        _GLASS_MAP[g](device=device) for g in prescription["glass_after"]
    )
    semi_diameters = tuple(
        torch.tensor(a, dtype=dtype, device=device)
        for a in prescription["semi_diameters"]
    )
    return System(
        entrance_material=Air(device=device),
        surfaces=surfaces,
        gap_thicknesses=gap_thicknesses,
        gap_materials=gap_materials,
        semi_diameters=semi_diameters,
    )


def cooke_triplet_template(system: System) -> DesignTemplate:
    """8 free variables: all 6 curvatures + the two inter-element air gaps.

    Element thicknesses (gaps 0, 2, 4) and the BFL (gap 5) are frozen.
    """
    return DesignTemplate(
        fixed_system=system,
        c_indices=(0, 1, 2, 3, 4, 5),
        t_indices=(1, 3),
    )


def cooke_triplet_rays(
    pupil_radius: float = PRESCRIPTION["epd_mm"] / 2.0,
    *,
    field_angles_deg=FIELD_ANGLES_DEG,
    wavelengths_um=WAVELENGTHS_UM,
    n_rays_per_fan: int = N_RAYS_PER_FAN,
    device=None,
) -> Ray:
    """Build the full sampling bundle.

    Shape: a flat Ray of length len(fields) * len(wavelengths) * n_rays_per_fan.
    Ordering is field-major (see module-level note). The chief ray of each
    field is at the center of its pupil fan.
    """
    bundles = []
    for u_deg in field_angles_deg:
        for wl in wavelengths_um:
            bundles.append(pupil_fan(
                n_rays_per_fan,
                pupil_radius=pupil_radius,
                field_angle_rad=math.radians(u_deg),
                wavelength_um=wl,
                chief_y=0.0,
                chief_z=0.0,
                device=device,
            ))
    return Ray(
        z=torch.cat([b.z for b in bundles]),
        y=torch.cat([b.y for b in bundles]),
        dz=torch.cat([b.dz for b in bundles]),
        dy=torch.cat([b.dy for b in bundles]),
        wavelength=torch.cat([b.wavelength for b in bundles]),
        valid=torch.cat([b.valid for b in bundles]),
    )


def per_field_residuals(
    v: Tensor, template: DesignTemplate, rays: Ray,
    rays_per_field: int = RAYS_PER_FIELD,
) -> Tensor:
    """Per-field centroid merit. Returns a flat residual vector.

    For each field group of `rays_per_field` rays (spanning all wavelengths
    and pupil heights for that field), the centroid of valid landings is
    computed; residuals are each ray's signed y-deviation from that field's
    centroid. Invalid rays contribute 0.

    Per-field centroiding (as opposed to per-(field, wavelength)) lets the
    merit penalise axial and lateral chromatic aberrations as additional
    spread within a field — the right behaviour for spot-RMS optimisation.
    Distortion (the chief ray's image height as a function of field) is by
    construction invisible to this merit; treat as a separate constraint
    if needed.
    """
    system = unpack(v, template)
    out = forward_trace(rays, system)
    # Reshape into [n_fields, rays_per_field]
    y = out.y.reshape(-1, rays_per_field)
    valid = out.valid.reshape(-1, rays_per_field)
    valid_f = valid.to(y.dtype)
    zero = torch.zeros_like(y)
    y_clean = torch.where(valid, y, zero)
    n_valid = valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
    y_centroid = y_clean.sum(dim=-1, keepdim=True) / n_valid       # [n_fields, 1]
    res = torch.where(valid, y - y_centroid, zero)
    return res.reshape(-1)


def per_field_rms_um(residuals: Tensor,
                     rays_per_field: int = RAYS_PER_FIELD) -> Tensor:
    """Convenience: per-field RMS in µm from the flat residual vector."""
    r2 = (residuals.reshape(-1, rays_per_field) ** 2).mean(dim=-1)
    return torch.sqrt(r2) * 1e3


def main():
    from lens_opt.optim.pack import pack
    from lens_opt.optim.lm import lm_optimize

    system = cooke_triplet_system()
    template = cooke_triplet_template(system)
    v0 = pack(template).detach().clone()

    print("Cooke triplet — starting design")
    print("=" * 64)
    for i, (R, t, g) in enumerate(zip(
        PRESCRIPTION["radii"],
        PRESCRIPTION["thicknesses"],
        PRESCRIPTION["glass_after"],
    )):
        print(f"  surf {i+1}:  R = {R:+10.4f}   t_after = {t:8.4f} mm   "
              f"glass_after = {g}")
    print()
    print(f"  EPD       : {PRESCRIPTION['epd_mm']} mm  (f/4 at 50 mm EFL)")
    print(f"  variables : {template.n_vars} (6 curvatures + 2 air gaps)")
    print(f"  fields    : {FIELD_ANGLES_DEG} deg")
    print(f"  wavelens  : {WAVELENGTHS_UM} um")
    print(f"  rays/fan  : {N_RAYS_PER_FAN}  ->  {N_RAYS_TOTAL} total residuals")
    print(f"  image plane at z = {sum(PRESCRIPTION['thicknesses']):.3f} mm")

    rays = cooke_triplet_rays()
    r0 = per_field_residuals(v0, template, rays)
    rms0 = per_field_rms_um(r0)
    valid_count = int(forward_trace(rays, system).valid.sum())

    print()
    print(f"  valid rays   : {valid_count} / {N_RAYS_TOTAL}")
    print(f"  RMS per field:")
    for u_deg, rms in zip(FIELD_ANGLES_DEG, rms0.tolist()):
        print(f"    {u_deg:5.1f} deg  ->  {rms:7.2f} um")
    phi0 = 0.5 * float((r0 * r0).sum())
    print(f"  merit       : {phi0:.6e}  ({math.sqrt(2*phi0/N_RAYS_TOTAL)*1e3:.2f} um overall RMS)")

    print()
    print("Optimizing (8 variables, 63 residuals)...")

    def residual(v):
        return per_field_residuals(v, template, rays)

    result = lm_optimize(v0, residual, max_iter=100, tol=1e-7)

    print()
    print("=" * 64)
    print("Result")
    print("=" * 64)
    print(f"  iterations  : {result.n_iter}")
    print(f"  exit reason : {result.exit_reason}")
    print()

    r_final = per_field_residuals(result.v, template, rays)
    rms_final = per_field_rms_um(r_final)
    overall_rms_final = math.sqrt(2 * result.merit / N_RAYS_TOTAL) * 1e3

    print(f"  RMS per field (start -> final):")
    for u_deg, rs, rf in zip(FIELD_ANGLES_DEG, rms0.tolist(), rms_final.tolist()):
        print(f"    {u_deg:5.1f} deg : {rs:7.2f} -> {rf:7.2f} um   "
              f"(x{rs/rf:.1f} shrink)")
    print(f"  Overall RMS : {math.sqrt(2*phi0/N_RAYS_TOTAL)*1e3:.2f} -> "
          f"{overall_rms_final:.2f} um")
    print(f"  Merit       : {phi0:.4e} -> {result.merit:.4e}  "
          f"(x{phi0/result.merit:.1f} smaller)")

    # Final design
    final_sys = unpack(result.v, template)
    print()
    print(f"  Final curvatures (1/R, in 1/mm):")
    for i in range(6):
        c_i = float(final_sys.surfaces[i].c)
        R_i = 1.0 / c_i if abs(c_i) > 1e-12 else float('inf')
        print(f"    surf {i+1}: c = {c_i:+.6f}   R = {R_i:+10.4f}")
    print(f"  Final air gaps:")
    print(f"    gap 1 (after surf 2): {float(final_sys.gap_thicknesses[1]):.4f} mm")
    print(f"    gap 3 (after surf 4): {float(final_sys.gap_thicknesses[3]):.4f} mm")

    # ────────────────────────────────────────────────────────────────────
    # Plots: ray aberration before/after, merit, damping
    # ────────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    def transverse_aberration(sys_):
        """Return [n_fields, n_wl, n_rays] of (y_ray - y_chief)."""
        out = forward_trace(rays, sys_)
        y = out.y.detach().reshape(
            len(FIELD_ANGLES_DEG), len(WAVELENGTHS_UM), N_RAYS_PER_FAN
        )
        # Chief ray is the middle pupil ray (index n_rays//2) of each (field, wl)
        chief = y[:, :, N_RAYS_PER_FAN // 2:N_RAYS_PER_FAN // 2 + 1]
        return (y - chief) * 1e3   # convert to um

    rho = torch.linspace(-1.0, 1.0, N_RAYS_PER_FAN).numpy()
    ab_start = transverse_aberration(system)
    ab_final = transverse_aberration(final_sys)
    wl_labels = ("F (486)", "d (588)", "C (656)")
    wl_colors = ("royalblue", "seagreen", "crimson")

    # Common y-limit for ray-aberration panels: use start-data envelope so the
    # before/after panels share a scale (makes shrinkage visually obvious).
    y_lim = float(ab_start.abs().max()) * 1.05

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.30)

    for fi, u_deg in enumerate(FIELD_ANGLES_DEG):
        ax_b = fig.add_subplot(gs[0, fi])
        ax_a = fig.add_subplot(gs[1, fi])
        for wi, (wl, lab, col) in enumerate(zip(WAVELENGTHS_UM, wl_labels, wl_colors)):
            ax_b.plot(rho, ab_start[fi, wi].numpy(), color=col, label=lab,
                      linewidth=1.2, marker="o", markersize=3)
            ax_a.plot(rho, ab_final[fi, wi].numpy(), color=col, label=lab,
                      linewidth=1.2, marker="o", markersize=3)
        for ax, title in ((ax_b, f"before  {u_deg:.0f}°"),
                          (ax_a, f"after   {u_deg:.0f}°")):
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-y_lim, y_lim)
            ax.set_xlabel("normalized pupil")
            ax.set_ylabel("Δy (μm)")
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.25)
        if fi == 0:
            ax_b.legend(fontsize=8, loc="best")

    iters = [h[0] for h in result.history]
    merits = [h[1] for h in result.history]
    lams = [h[2] for h in result.history]

    ax_merit = fig.add_subplot(gs[2, 0:2])
    ax_merit.semilogy(iters, merits, marker="o", linewidth=1.2)
    ax_merit.set_xlabel("iteration")
    ax_merit.set_ylabel("merit  0.5 ||r||²")
    ax_merit.set_title(f"Merit vs iter   start={merits[0]:.3e}, "
                       f"final={merits[-1]:.3e}", fontsize=10)
    ax_merit.grid(True, alpha=0.3, which="both")

    ax_lam = fig.add_subplot(gs[2, 2])
    ax_lam.semilogy(iters, lams, marker="o", color="darkorange", linewidth=1.2)
    ax_lam.set_xlabel("iteration")
    ax_lam.set_ylabel("λ")
    ax_lam.set_title("Damping vs iter", fontsize=10)
    ax_lam.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"M4  Cooke triplet  f/4, 50 mm EFL   3 fields × 3 wavelengths × "
        f"{N_RAYS_PER_FAN} rays = {N_RAYS_TOTAL} residuals,  8 variables   "
        f"RMS {math.sqrt(2*phi0/N_RAYS_TOTAL)*1e3:.0f} → "
        f"{overall_rms_final:.1f} μm  in {result.n_iter} iters",
        fontsize=11,
    )

    out_path = os.path.join(_HERE, "cooke_triplet.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
