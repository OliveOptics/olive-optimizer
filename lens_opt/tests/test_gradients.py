"""Milestone 2 tests: merit-function gradient correctness.

Four tests per the plan:
  1. Finite-diff scalar gradient agrees with autograd at ~6 decimals.
  2. Full Jacobian (jacfwd) vs central differences at ~6 decimals.
  3. Residual-scalar consistency: 0.5 * sum(r^2) == merit_scalar.
  4. NaN absence: a near-TIR ray bundle still produces finite gradients.

These are the highest-risk milestone of the project — gradient bugs here
masquerade as LM bugs in M3. Don't proceed until all four pass.
"""
import math
import pytest
import torch
from torch.func import jacfwd

from lens_opt.tracer.surfaces import make_surface
from lens_opt.tracer.glass import Air, BK7
from lens_opt.tracer.system import System, uniform_semi_diameters
from lens_opt.tracer.sources import parallel_bundle, pupil_fan
from lens_opt.tracer.merit import merit_scalar, merit_residuals
from lens_opt.optim.pack import DesignTemplate, pack


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.fixture(params=DEVICES)
def device(request):
    return torch.device(request.param)


def _t(x, device):
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def _make_singlet(R1=50.0, R2=-50.0, t=5.0, post=50.0, device=None):
    return System(
        entrance_material=Air(device=device),
        surfaces=(make_surface(c=1.0 / R1, device=device),
                  make_surface(c=1.0 / R2, device=device)),
        gap_thicknesses=(_t(t, device), _t(post, device)),
        gap_materials=(BK7(device=device), Air(device=device)),
        semi_diameters=uniform_semi_diameters(2, device=device),
    )


# ──────────────────────────────────────────────────────────────────────────
# Test 1: finite-diff scalar gradient
# ──────────────────────────────────────────────────────────────────────────

def test_finite_diff_scalar_gradient(device):
    base = _make_singlet(device=device)
    template = DesignTemplate(fixed_system=base, c_indices=(0,), t_indices=())
    v0 = pack(template).detach().clone()
    rays = parallel_bundle(10, half_height=2.0, device=device)

    # Autograd
    v = v0.clone().requires_grad_(True)
    m = merit_scalar(v, template, rays)
    m.backward()
    grad_ag = v.grad.item()

    # Central difference
    eps = 1e-5
    m_plus = merit_scalar(v0 + eps, template, rays).item()
    m_minus = merit_scalar(v0 - eps, template, rays).item()
    grad_fd = (m_plus - m_minus) / (2.0 * eps)

    rel_err = abs(grad_ag - grad_fd) / max(abs(grad_fd), 1e-12)
    assert rel_err < 1e-6, (
        f"autograd={grad_ag:.10e}  fd={grad_fd:.10e}  rel_err={rel_err:.3e}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Test 2: full Jacobian via jacfwd matches central differences
# ──────────────────────────────────────────────────────────────────────────

def test_jacobian_jacfwd_vs_finite_diff(device):
    base = _make_singlet(device=device)
    template = DesignTemplate(fixed_system=base,
                              c_indices=(0, 1), t_indices=(0,))
    v0 = pack(template).detach().clone()
    rays = parallel_bundle(10, half_height=2.0, device=device)

    def fn(v_):
        return merit_residuals(v_, template, rays)

    J_fwd = jacfwd(fn)(v0)  # [n_rays, n_vars]

    eps = 1e-5
    J_fd = torch.zeros_like(J_fwd)
    for j in range(v0.numel()):
        v_plus = v0.clone()
        v_plus[j] = v_plus[j] + eps
        v_minus = v0.clone()
        v_minus[j] = v_minus[j] - eps
        J_fd[:, j] = (fn(v_plus) - fn(v_minus)) / (2.0 * eps)

    err = (J_fwd - J_fd).abs().max().item()
    scale = J_fd.abs().max().item()
    rel_err = err / max(scale, 1e-12)
    assert rel_err < 1e-6, (
        f"max |J_fwd - J_fd| = {err:.3e}, scale = {scale:.3e}, "
        f"rel_err = {rel_err:.3e}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Test 3: residual-scalar consistency
# ──────────────────────────────────────────────────────────────────────────

def test_residual_scalar_consistency(device):
    base = _make_singlet(device=device)
    template = DesignTemplate(fixed_system=base,
                              c_indices=(0, 1), t_indices=(0,))
    v = pack(template).detach().clone()
    rays = parallel_bundle(10, half_height=2.0, device=device)

    r = merit_residuals(v, template, rays)
    m_from_r = 0.5 * (r * r).sum()
    m_direct = merit_scalar(v, template, rays)

    assert torch.allclose(m_from_r, m_direct, atol=0.0, rtol=0.0), (
        f"merit_scalar={m_direct.item():.18e}  "
        f"0.5*sum(r^2)={m_from_r.item():.18e}  "
        f"diff={(m_from_r - m_direct).item():.3e}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Test 4: NaN-free gradients near TIR
# ──────────────────────────────────────────────────────────────────────────

def test_nan_absence_near_tir(device):
    """With a strong lens + extreme oblique pupil fan, some rays will TIR.
    The merit gradient w.r.t. design variables must stay finite."""
    # Strong-power singlet so back-surface incidence is high.
    base = _make_singlet(R1=10.0, R2=-10.0, t=4.0, device=device)
    template = DesignTemplate(fixed_system=base,
                              c_indices=(0, 1), t_indices=())
    v = pack(template).detach().clone().requires_grad_(True)

    # Pupil fan at 30 deg field, large pupil — guaranteed to drive some
    # rays past the critical angle at the BK7->air back surface.
    rays = pupil_fan(13, pupil_radius=4.0,
                     field_angle_rad=math.radians(30), device=device)

    m = merit_scalar(v, template, rays)
    m.backward()

    assert torch.all(torch.isfinite(v.grad)), (
        f"v.grad has NaN or Inf: {v.grad}"
    )
    # Also verify the gradient is meaningful (not all zero).
    assert v.grad.abs().sum().item() > 0, (
        f"v.grad is all zeros: {v.grad}"
    )
