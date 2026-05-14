"""Compare unconstrained vs constrained Cooke triplet optimization.

The unconstrained run (the M4 baseline) minimizes spot spread only; EFL
and edge thicknesses can drift freely. This script runs both versions
and prints the diagnostics side-by-side so the trade-off is visible:

    unconstrained: best RMS, but EFL may be far from 50 mm
    constrained  : enforces EFL = 50 mm and min edge thickness >= 0.5 mm
                   — slightly larger RMS but a manufacturable lens

Constraints are added as extra residual entries in the LM merit vector
(soft penalties). LM treats them identically to spot residuals.

Run from repo root:
    C:\\Users\\bwyan\\.venvs\\lens-opt\\Scripts\\python.exe lens_opt\\examples\\cooke_constrained.py
"""
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

from lens_opt.optim.pack import pack, unpack
from lens_opt.optim.lm import lm_optimize
from lens_opt.tracer.system import forward_trace
from lens_opt.tracer.sources import pupil_fan

from lens_opt.constraints import (
    paraxial_efl,
    element_edge_thickness,
    efl_residual,
    min_edge_residual,
)
from lens_opt.examples.cooke_triplet import (
    cooke_triplet_system, cooke_triplet_template, cooke_triplet_rays,
    per_field_residuals, N_RAYS_TOTAL, N_RAYS_PER_FAN,
    FIELD_ANGLES_DEG, WAVELENGTHS_UM, PRESCRIPTION,
)
from lens_opt.visualize import draw_system, plot_rays


TARGET_EFL_MM = 50.0
T_MIN_EDGE_MM = 1.8
EVAL_HEIGHT_MM = 7.0   # just outside EPD/2 = 6.25 to be conservative
W_EFL = 5.0            # weight for the EFL equality residual
W_EDGE = 50.0          # weight for each min-edge inequality residual

# Cooke triplet elements occupy surface pairs (0,1), (2,3), (4,5)
ELEMENTS = ((0, 1), (2, 3), (4, 5))
ELEMENT_NAMES = ("element 1 (SK16, +)", "element 2 (F2, -)", "element 3 (SK16, +)")


def report_design(system, label):
    print(f"  {label}")
    efl = float(paraxial_efl(system))
    print(f"    EFL (d-line)            : {efl:.4f} mm   "
          f"(target {TARGET_EFL_MM:.1f} mm, drift {efl - TARGET_EFL_MM:+.3f} mm)")
    print(f"    Edge thickness @ h={EVAL_HEIGHT_MM} mm:")
    for (f, b), name in zip(ELEMENTS, ELEMENT_NAMES):
        et = float(element_edge_thickness(system, f, b, EVAL_HEIGHT_MM))
        if et >= T_MIN_EDGE_MM - 1e-4:
            flag = "    at limit" if abs(et - T_MIN_EDGE_MM) < 1e-3 else ""
        else:
            flag = f"    BELOW MIN by {T_MIN_EDGE_MM - et:.4f} mm"
        print(f"      {name:30s}: {et:+.4f} mm{flag}")


def main():
    print("=" * 72)
    print("  Cooke triplet: unconstrained vs constrained LM")
    print("=" * 72)
    print(f"  Constraints:")
    print(f"    EFL target               : {TARGET_EFL_MM} mm   (weight {W_EFL})")
    print(f"    Min edge thickness       : {T_MIN_EDGE_MM} mm at "
          f"h={EVAL_HEIGHT_MM} mm   (weight {W_EDGE} each, 3 elements)")
    print()

    system = cooke_triplet_system()
    template = cooke_triplet_template(system)
    rays = cooke_triplet_rays()
    v0 = pack(template).detach().clone()

    print("-" * 72)
    report_design(system, "Starting design (published prescription)")
    print()

    # ──────────────────────────────────────────────────────────────────
    # Unconstrained run (M4 baseline)
    # ──────────────────────────────────────────────────────────────────

    def residual_unconstrained(v):
        return per_field_residuals(v, template, rays)

    r0 = residual_unconstrained(v0)
    phi0 = 0.5 * float((r0 * r0).sum())
    rms0_um = math.sqrt(2 * phi0 / N_RAYS_TOTAL) * 1e3

    result_u = lm_optimize(v0, residual_unconstrained, max_iter=100, tol=1e-7)
    system_u = unpack(result_u.v, template)
    rms_u_um = math.sqrt(2 * result_u.merit / N_RAYS_TOTAL) * 1e3

    print("-" * 72)
    print(f"  UNCONSTRAINED RESULT  ({result_u.n_iter} iters, {result_u.exit_reason})")
    print(f"    Spot RMS                 : {rms0_um:.2f} -> {rms_u_um:.2f} um")
    report_design(system_u, "")
    print()

    # ──────────────────────────────────────────────────────────────────
    # Constrained run: spot residuals + EFL + 3 edge residuals
    # ──────────────────────────────────────────────────────────────────

    def residual_constrained(v):
        sys_ = unpack(v, template)
        spot_r = per_field_residuals(v, template, rays)
        efl_r = efl_residual(sys_, TARGET_EFL_MM, weight=W_EFL).unsqueeze(0)
        edge_rs = torch.stack([
            min_edge_residual(sys_, f, b, T_MIN_EDGE_MM, EVAL_HEIGHT_MM,
                              weight=W_EDGE)
            for f, b in ELEMENTS
        ])
        return torch.cat([spot_r, efl_r, edge_rs])

    result_c = lm_optimize(v0, residual_constrained, max_iter=100, tol=1e-7)
    system_c = unpack(result_c.v, template)
    # Pull only the spot residuals back out for an apples-to-apples RMS
    spot_r_c = per_field_residuals(result_c.v, template, rays)
    spot_phi_c = 0.5 * float((spot_r_c * spot_r_c).sum())
    rms_c_um = math.sqrt(2 * spot_phi_c / N_RAYS_TOTAL) * 1e3

    print("-" * 72)
    print(f"  CONSTRAINED RESULT    ({result_c.n_iter} iters, {result_c.exit_reason})")
    print(f"    Spot RMS                 : {rms0_um:.2f} -> {rms_c_um:.2f} um   "
          f"(spot residuals only, constraint penalties stripped)")
    report_design(system_c, "")
    print()

    # ──────────────────────────────────────────────────────────────────
    # Comparison summary
    # ──────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  Comparison")
    print("=" * 72)
    print(f"  {'':28s}  {'unconstrained':>15s}  {'constrained':>15s}")
    print(f"  {'spot RMS (um)':28s}  {rms_u_um:>15.2f}  {rms_c_um:>15.2f}")
    efl_u = float(paraxial_efl(system_u))
    efl_c = float(paraxial_efl(system_c))
    print(f"  {'EFL drift from 50 mm (mm)':28s}  {efl_u - TARGET_EFL_MM:>+15.3f}  "
          f"{efl_c - TARGET_EFL_MM:>+15.3f}")
    def _tag(et):
        if et < T_MIN_EDGE_MM - 1e-4:
            return " v "
        if abs(et - T_MIN_EDGE_MM) < 1e-3:
            return " = "
        return "   "
    for (f, b), name in zip(ELEMENTS, ELEMENT_NAMES):
        et_u = float(element_edge_thickness(system_u, f, b, EVAL_HEIGHT_MM))
        et_c = float(element_edge_thickness(system_c, f, b, EVAL_HEIGHT_MM))
        print(f"  {name[:28]:28s}  {et_u:>+15.4f}{_tag(et_u)}  {et_c:>+15.4f}{_tag(et_c)}")
    print(f"    legend:  v = below min   = = at limit")

    # ──────────────────────────────────────────────────────────────────
    # Figure: lens layouts + ray-aberration plots, side-by-side
    # ──────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import numpy as np

    def transverse_aberration(sys_):
        """[n_fields, n_wl, n_rays] of (y - y_chief) at image plane, in um.

        Vignetted rays are replaced with NaN so matplotlib skips them — the
        plot then shows only the rays that physically reach the image.
        """
        out = forward_trace(rays, sys_)
        shape = (len(FIELD_ANGLES_DEG), len(WAVELENGTHS_UM), N_RAYS_PER_FAN)
        y = out.y.detach().reshape(shape)
        valid = out.valid.detach().reshape(shape)
        chief = y[:, :, N_RAYS_PER_FAN // 2 : N_RAYS_PER_FAN // 2 + 1]
        aber = (y - chief) * 1e3
        return torch.where(valid, aber, torch.full_like(aber, float("nan")))

    # Report vignetting stats per design
    print()
    print("Vignetting summary (rays surviving / total):")
    for name, sys_, _ in [("start", system, None),
                          ("unconstrained", system_u, None),
                          ("constrained", system_c, None)]:
        out = forward_trace(rays, sys_)
        valid = out.valid.detach().reshape(
            len(FIELD_ANGLES_DEG), len(WAVELENGTHS_UM), N_RAYS_PER_FAN
        )
        per_field = valid.reshape(len(FIELD_ANGLES_DEG), -1).sum(dim=-1)
        total = valid.numel() // len(FIELD_ANGLES_DEG)
        details = "   ".join(
            f"{u_deg:.0f}deg: {int(v)}/{total}"
            for u_deg, v in zip(FIELD_ANGLES_DEG, per_field.tolist())
        )
        print(f"  {name:14s}  {details}")

    designs = [
        ("Start (published)",         system,    rms0_um),
        ("Unconstrained",             system_u,  rms_u_um),
        (f"Constrained (EFL={int(TARGET_EFL_MM)}, edge>={T_MIN_EDGE_MM})",
                                      system_c,  rms_c_um),
    ]
    aberrations = [transverse_aberration(s).numpy() for _, s, _ in designs]

    rho = np.linspace(-1.0, 1.0, N_RAYS_PER_FAN)
    wl_colors = ("royalblue", "seagreen", "crimson")
    wl_labels = ("F (486)", "d (588)", "C (656)")

    # Layout rays: just d-line at 0 and 14 deg for clarity.
    layout_rays_axis = pupil_fan(7, pupil_radius=PRESCRIPTION["epd_mm"] / 2.0,
                                 field_angle_rad=0.0, wavelength_um=0.5876)
    layout_rays_field = pupil_fan(7, pupil_radius=PRESCRIPTION["epd_mm"] / 2.0,
                                  field_angle_rad=math.radians(14.0),
                                  wavelength_um=0.5876)

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(4, 3, hspace=0.50, wspace=0.28,
                          height_ratios=[1.4, 1, 1, 1])

    # Common y-limit per FIELD row so columns are comparable.
    y_limits_per_field = [
        max(np.nanmax(np.abs(aberrations[k][fi])) for k in range(3)) * 1.10
        for fi in range(len(FIELD_ANGLES_DEG))
    ]

    z_image = sum(PRESCRIPTION["thicknesses"])

    for col, (name, sys_, rms_um) in enumerate(designs):
        # Top row: lens layout — draw_system reads system.semi_diameters
        # by default, which is the same number we vignette against.
        ax = fig.add_subplot(gs[0, col])
        draw_system(ax, sys_, glass_pairs=[(0, 1), (2, 3), (4, 5)])
        plot_rays(ax, layout_rays_axis, sys_, color="royalblue",
                  linewidth=0.8, alpha=0.6)
        plot_rays(ax, layout_rays_field, sys_, color="crimson",
                  linewidth=0.8, alpha=0.6)
        ax.axvline(z_image, color="black", linestyle=":", linewidth=0.7,
                   alpha=0.5)
        ax.set_xlim(-2, z_image + 2)
        ax.set_ylim(-11, 11)
        ax.set_aspect("equal")
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("y (mm)")
        efl = float(paraxial_efl(sys_))
        ax.set_title(f"{name}\nEFL={efl:.2f}, RMS={rms_um:.2f} um",
                     fontsize=10)

        # Bottom three rows: ray aberration at each field
        for fi, u_deg in enumerate(FIELD_ANGLES_DEG):
            ax = fig.add_subplot(gs[1 + fi, col])
            for wi, (lab, c) in enumerate(zip(wl_labels, wl_colors)):
                ax.plot(rho, aberrations[col][fi, wi], color=c,
                        linewidth=1.2, marker="o", markersize=3, label=lab)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-y_limits_per_field[fi], y_limits_per_field[fi])
            ax.set_xlabel("normalized pupil")
            ax.set_ylabel("dy (um)")
            ax.set_title(f"aberration @ {u_deg:.0f} deg", fontsize=9)
            ax.grid(True, alpha=0.25)
            if col == 0 and fi == 0:
                ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Cooke triplet: unconstrained vs constrained   "
        f"(EFL={TARGET_EFL_MM} mm, min edge={T_MIN_EDGE_MM} mm @ h={EVAL_HEIGHT_MM} mm)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = os.path.join(_HERE, "cooke_constrained.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print()
    print(f"  Saved: {out_path}")
    print()


if __name__ == "__main__":
    main()
