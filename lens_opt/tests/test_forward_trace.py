"""Milestone 1 tests: differentiable forward tracer correctness.

Four tests (asphere was removed from scope):
  1. Plano refraction at normal incidence
  2. Snell's law numerical check (oblique flat surface)
  3. Spherical paraxial focus (BK7 singlet, lensmaker BFL to 0.1%)
  4. Reversibility (1e-9 mm round-trip)

Each test runs on every available device (CPU, plus CUDA if present).
"""
import math
import pytest
import torch
from dataclasses import replace

from lens_opt.tracer.surfaces import Ray, make_surface
from lens_opt.tracer.glass import Air, BK7
from lens_opt.tracer.system import (
    System, forward_trace, reverse_system, uniform_semi_diameters,
)
from lens_opt.tracer.sources import single_ray, parallel_bundle


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.fixture(params=DEVICES)
def device(request):
    return torch.device(request.param)


def _tensor(x, device, dtype=torch.float64):
    return torch.as_tensor(x, device=device, dtype=dtype)


# ──────────────────────────────────────────────────────────────────────────
# Test 1: Plano refraction at normal incidence
# ──────────────────────────────────────────────────────────────────────────

def test_plano_normal_incidence(device):
    """A ray going straight along +z through a flat air/glass interface
    must exit with unchanged direction, regardless of index ratio."""
    surf = make_surface(c=0.0, device=device)
    system = System(
        entrance_material=Air(device=device),
        surfaces=(surf,),
        gap_thicknesses=(_tensor(10.0, device),),
        gap_materials=(BK7(device=device),),
        semi_diameters=uniform_semi_diameters(1, device=device),
    )
    ray = single_ray(y=0.0, u_rad=0.0, device=device)
    out = forward_trace(ray, system)
    assert torch.allclose(out.dz, _tensor(1.0, device), atol=1e-14)
    assert torch.allclose(out.dy, _tensor(0.0, device), atol=1e-14)
    assert torch.allclose(out.y, _tensor(0.0, device), atol=1e-14)
    assert bool(out.valid.item())


# ──────────────────────────────────────────────────────────────────────────
# Test 2: Snell's law on an oblique flat surface
# ──────────────────────────────────────────────────────────────────────────

def test_snell_numerical_check(device):
    """For air → BK7 at a flat surface and an oblique incidence angle,
    sin(theta_i) / sin(theta_t) must equal n_BK7 / n_air to 10 decimals."""
    u_in = 0.3  # radians, well below TIR
    wl = 0.5876

    surf = make_surface(c=0.0, device=device)
    bk7 = BK7(device=device)
    n_glass = bk7.n(_tensor(wl, device)).item()

    # Trace only the entrance surface: use a 2-surface system with the second
    # surface acting as a passive plano AT z=0 with same material — but simpler
    # to just stop after one surface. Use a single-surface system with image
    # plane right at z=0.
    system = System(
        entrance_material=Air(device=device),
        surfaces=(surf,),
        gap_thicknesses=(_tensor(0.0, device),),
        gap_materials=(bk7,),
        semi_diameters=uniform_semi_diameters(1, device=device),
    )
    ray = single_ray(y=0.0, u_rad=u_in, wavelength_um=wl, device=device)
    out = forward_trace(ray, system)

    sin_i = math.sin(u_in)
    sin_t_expected = sin_i / n_glass
    sin_t_actual = out.dy.item()  # at flat surface, dy == sin(theta_t)

    assert abs(sin_t_expected - sin_t_actual) < 1e-10, (
        f"Snell ratio fails: expected sin_t={sin_t_expected:.15e}, "
        f"got {sin_t_actual:.15e}, |err|={abs(sin_t_expected - sin_t_actual):.2e}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Test 3: Spherical paraxial focus — BK7 singlet
# ──────────────────────────────────────────────────────────────────────────

def test_paraxial_focus_BK7_singlet(device):
    """Trace parallel rays at small heights through a BK7 singlet
    (R1=50, R2=-50, t=5). The back-focal length where rays cross the axis
    must match the thick-lens lensmaker prediction to within 0.1%."""
    R1, R2, t = 50.0, -50.0, 5.0
    wl = 0.5876

    bk7 = BK7(device=device)
    n = bk7.n(_tensor(wl, device)).item()

    # Thick-lens lensmaker
    inv_f = (n - 1.0) * (1.0 / R1 - 1.0 / R2 + (n - 1.0) * t / (n * R1 * R2))
    f = 1.0 / inv_f
    BFL_expected = f * (1.0 - (n - 1.0) * t / (n * R1))

    s0 = make_surface(c=1.0 / R1, device=device)
    s1 = make_surface(c=1.0 / R2, device=device)
    system = System(
        entrance_material=Air(device=device),
        surfaces=(s0, s1),
        gap_thicknesses=(_tensor(t, device), _tensor(100.0, device)),  # past focus
        gap_materials=(bk7, Air(device=device)),
        semi_diameters=uniform_semi_diameters(2, device=device),
    )

    # Use 10 rays so y=0 is not in the set (linspace -h..h with even n excludes 0).
    rays = parallel_bundle(10, half_height=0.5, wavelength_um=wl, device=device)
    out = forward_trace(rays, system)

    # Each non-axial ray crosses the optical axis where y(z*) = 0, i.e.
    # z* = z_out - y_out * (dz_out / dy_out).
    z_cross = out.z - out.y * (out.dz / out.dy)
    BFL_actual = (z_cross.mean() - t).item()

    rel_err = abs(BFL_actual - BFL_expected) / BFL_expected
    assert rel_err < 1e-3, (
        f"BFL mismatch: expected {BFL_expected:.4f}, got {BFL_actual:.4f}, "
        f"rel_err={rel_err:.4e}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Test 4: Reversibility — trace through, reverse, trace back
# ──────────────────────────────────────────────────────────────────────────

def test_reversibility(device):
    """A ray traced forward then reversed and traced back must return to
    the starting (y, dz, -dy) within 1e-9 mm."""
    R1, R2, t = 50.0, -50.0, 5.0
    s0 = make_surface(c=1.0 / R1, device=device)
    s1 = make_surface(c=1.0 / R2, device=device)
    system = System(
        entrance_material=Air(device=device),
        surfaces=(s0, s1),
        gap_thicknesses=(_tensor(t, device), _tensor(50.0, device)),
        gap_materials=(BK7(device=device), Air(device=device)),
        semi_diameters=uniform_semi_diameters(2, device=device),
    )

    ray = single_ray(y=0.5, u_rad=0.02, device=device)
    out = forward_trace(ray, system)

    rev_sys, pre_thick = reverse_system(system)
    rev_ray = Ray(
        z=-pre_thick,
        y=out.y,
        dz=out.dz,
        dy=-out.dy,
        wavelength=out.wavelength,
        valid=out.valid,
    )
    back = forward_trace(rev_ray, rev_sys)

    assert torch.abs(back.y - ray.y).item() < 1e-9, (
        f"y round-trip error {torch.abs(back.y - ray.y).item():.3e}"
    )
    assert torch.abs(back.dz - ray.dz).item() < 1e-9, (
        f"dz round-trip error {torch.abs(back.dz - ray.dz).item():.3e}"
    )
    assert torch.abs(back.dy + ray.dy).item() < 1e-9, (
        f"dy round-trip error {torch.abs(back.dy + ray.dy).item():.3e}"
    )


