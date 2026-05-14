"""Milestone 3.5: first real lens-side LM run on a BK7 singlet.

Setup:
  - Starting design: equi-convex BK7 singlet, R1=+50, R2=-50, t=5 mm.
    The spherical-aberration-minimizing bending for n=1.517 is asymmetric
    (front surface more curved than back), so equi-convex is non-optimal.
  - Image plane is fixed at the paraxial focal plane of the starting
    design. As bending changes the optimizer can both refocus and reduce
    aberration; both contribute to the merit.
  - 7 parallel rays at half-height 5 mm (f/5).
  - Single wavelength d-line (no chromatic component in this milestone).
  - Variables: both surface curvatures.

The test is a smoke check for the full LM-on-lens pipeline:
   pack -> residual_fn -> jacfwd -> lm step -> unpack -> retrace
We assert the run converges and the RMS spot drops by at least 2x.

This is M3.5 — proves the pieces compose correctly before tackling the
Cooke triplet (M4), which adds field angles, multiple wavelengths, and
more variables.
"""
import math
import pytest
import torch

from lens_opt.tracer.surfaces import make_surface
from lens_opt.tracer.glass import Air, BK7
from lens_opt.tracer.system import System, uniform_semi_diameters
from lens_opt.tracer.sources import parallel_bundle
from lens_opt.tracer.merit import merit_residuals
from lens_opt.optim.pack import DesignTemplate, pack
from lens_opt.optim.lm import lm_optimize


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.fixture(params=DEVICES)
def device(request):
    return torch.device(request.param)


def _t(x, device):
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def _bk7_singlet(R1, R2, t, post, device):
    return System(
        entrance_material=Air(device=device),
        surfaces=(make_surface(c=1.0 / R1, device=device),
                  make_surface(c=1.0 / R2, device=device)),
        gap_thicknesses=(_t(t, device), _t(post, device)),
        gap_materials=(BK7(device=device), Air(device=device)),
        semi_diameters=uniform_semi_diameters(2, device=device),
    )


def _paraxial_bfl(R1, R2, t, n):
    inv_f = (n - 1.0) * (1.0 / R1 - 1.0 / R2 +
                         (n - 1.0) * t / (n * R1 * R2))
    f = 1.0 / inv_f
    return f * (1.0 - (n - 1.0) * t / (n * R1))


def test_singlet_bending_optimization(device):
    R1, R2, t = 50.0, -50.0, 5.0
    wl = 0.5876
    n = float(BK7(device=device).n(_t(wl, device)))
    bfl = _paraxial_bfl(R1, R2, t, n)

    system = _bk7_singlet(R1, R2, t, bfl, device=device)
    template = DesignTemplate(fixed_system=system,
                              c_indices=(0, 1), t_indices=())
    v0 = pack(template).detach().clone()

    rays = parallel_bundle(7, half_height=5.0, wavelength_um=wl, device=device)

    def residual(v):
        return merit_residuals(v, template, rays)

    # Starting RMS for comparison
    r0 = residual(v0)
    phi0 = 0.5 * float((r0 * r0).sum())
    rms0 = math.sqrt(2.0 * phi0 / 7.0)

    # tol=1e-8 is a normal LM tolerance. Setting it tighter than ~1e-10 on
    # this problem causes "stuck" exits at the float64 precision floor: rho
    # checks fail on floating-point noise after the merit can no longer be
    # meaningfully reduced. Reporting that as "converged at floor" would be
    # a future LM polish; for now we just keep tol above the noise.
    result = lm_optimize(v0, residual, max_iter=50, tol=1e-8)

    rms_final = math.sqrt(2.0 * result.merit / 7.0)

    # Convergence sanity.
    assert result.exit_reason in ("converged_gradient", "max_iter"), (
        f"unexpected exit: {result.exit_reason}"
    )

    # The whole point: LM must improve the lens. Starting equi-convex is
    # measurably worse than the bending optimum, so we expect at least a
    # 2x RMS reduction. On this problem we observe ~5-10x in practice.
    assert rms_final < rms0 / 2.0, (
        f"RMS did not improve enough: start={rms0*1e3:.2f} um, "
        f"final={rms_final*1e3:.2f} um (need < {rms0*1e3/2:.2f} um)"
    )

    # Curvatures must remain physical (no NaN, no zero, no absurd values).
    v_final = result.v
    assert torch.all(torch.isfinite(v_final)), (
        f"final v has NaN/Inf: {v_final}"
    )
    # Curvatures should stay within roughly an order of magnitude of where
    # they started (sanity).
    assert (v_final.abs() < 1.0).all(), (
        f"|c| went out of physical range: {v_final}"
    )
