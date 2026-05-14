"""Constraint helpers for lens-design optimization.

These functions compute scalar physical quantities of a System and turn
them into single-entry residuals that can be concatenated onto the spot
merit vector. LM then drives those entries toward zero alongside the
spot residuals.

  paraxial_efl(system, wavelength_um) -> scalar tensor
      EFL in mm via ABCD matrix product in reduced (y, n·u) coordinates.

  element_edge_thickness(system, front_surf, back_surf, eval_height)
      Edge thickness at the given semi-diameter, in mm. Computed as
      t_axial + sag(c_back, h) - sag(c_front, h). Negative values mean
      the surfaces have crossed at the rim (impossible glass).

  efl_residual(system, target_efl, weight, wavelength_um) -> scalar
      Equality penalty: weight * (efl - target).

  min_edge_residual(system, front_surf, back_surf, t_min, eval_height,
                    weight) -> scalar
      Inequality penalty: weight * relu(t_min - actual). Zero when the
      constraint is satisfied; linear in the violation when not.

All functions are differentiable through curvatures, thicknesses, and
glass dispersion. They are vmap-compatible — the batched LM can run a
constrained merit just as it runs the unconstrained one.
"""
import torch
from torch import Tensor

from lens_opt.tracer.system import System


# ──────────────────────────────────────────────────────────────────────────
# Paraxial EFL via ABCD product
# ──────────────────────────────────────────────────────────────────────────

def _refraction_matrix(power: Tensor) -> Tensor:
    """[[1, 0], [-power, 1]] in reduced coordinates."""
    zero = torch.zeros_like(power)
    one = torch.ones_like(power)
    return torch.stack([
        torch.stack([one, zero]),
        torch.stack([-power, one]),
    ])


def _translation_matrix(t_over_n: Tensor) -> Tensor:
    """[[1, t/n], [0, 1]] in reduced coordinates."""
    zero = torch.zeros_like(t_over_n)
    one = torch.ones_like(t_over_n)
    return torch.stack([
        torch.stack([one, t_over_n]),
        torch.stack([zero, one]),
    ])


def paraxial_efl(system: System, wavelength_um: float = 0.5876) -> Tensor:
    """Paraxial effective focal length in mm at the given wavelength.

    EFL = -1 / M[1, 0] where M is the system ABCD matrix in reduced
    coordinates, taken between the first-surface vertex and the
    last-surface vertex (no back-focal translation).
    """
    sample_c = system.surfaces[0].c
    dtype = sample_c.dtype
    device = sample_c.device
    wl = torch.as_tensor(wavelength_um, dtype=dtype, device=device)

    M = torch.eye(2, dtype=dtype, device=device)
    n_before = system.entrance_material.n(wl)
    K = len(system.surfaces)

    for i in range(K):
        c = system.surfaces[i].c
        n_after = system.gap_materials[i].n(wl)
        power = (n_after - n_before) * c
        M = _refraction_matrix(power) @ M

        # Translate to the next surface; skip the back focal gap.
        if i < K - 1:
            t = system.gap_thicknesses[i]
            M = _translation_matrix(t / n_after) @ M

        n_before = n_after

    return -1.0 / M[1, 0]


# ──────────────────────────────────────────────────────────────────────────
# Edge thickness via sag math
# ──────────────────────────────────────────────────────────────────────────

def _sag(c: Tensor, h: Tensor) -> Tensor:
    """Spherical sag at height h. Same sign convention as System.surfaces[i].c.

    sag(c, h) > 0 when c > 0 (surface bulges toward +z at non-axial heights).
    Clamps the radicand at eps to stay differentiable when |c*h| approaches 1.
    """
    eps = torch.finfo(c.dtype).eps
    ch2 = (c * h) ** 2
    arg = (1.0 - ch2).clamp(min=eps)
    return c * h * h / (1.0 + torch.sqrt(arg))


def element_edge_thickness(
    system: System,
    front_surf_idx: int,
    back_surf_idx: int,
    eval_height: float,
) -> Tensor:
    """Edge thickness of a lens element at semi-diameter `eval_height`.

    Assumes the element occupies consecutive surface indices
    (back_surf_idx == front_surf_idx + 1). For non-consecutive surfaces
    (e.g. a cemented doublet treated as separate elements), call this
    for each adjacent pair.
    """
    if back_surf_idx != front_surf_idx + 1:
        raise ValueError(
            "element_edge_thickness expects consecutive surfaces "
            f"(got front={front_surf_idx}, back={back_surf_idx})"
        )
    c_f = system.surfaces[front_surf_idx].c
    c_b = system.surfaces[back_surf_idx].c
    t = system.gap_thicknesses[front_surf_idx]
    h = torch.as_tensor(eval_height, dtype=c_f.dtype, device=c_f.device)
    return t + _sag(c_b, h) - _sag(c_f, h)


# ──────────────────────────────────────────────────────────────────────────
# Residual wrappers (for merit-vector composition)
# ──────────────────────────────────────────────────────────────────────────

def efl_residual(
    system: System,
    target_efl: float,
    weight: float = 1.0,
    wavelength_um: float = 0.5876,
) -> Tensor:
    """Single-entry equality residual: weight * (efl - target). Shape []."""
    return weight * (paraxial_efl(system, wavelength_um) - target_efl)


def min_edge_residual(
    system: System,
    front_surf_idx: int,
    back_surf_idx: int,
    t_min: float,
    eval_height: float,
    weight: float = 1.0,
) -> Tensor:
    """Single-entry inequality residual: weight * relu(t_min - edge). Shape [].

    Zero when the edge thickness meets `t_min`; positive (linear in the
    violation) when it does not.
    """
    actual = element_edge_thickness(system, front_surf_idx, back_surf_idx,
                                    eval_height)
    return weight * (t_min - actual).clamp(min=0.0)
