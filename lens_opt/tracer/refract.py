"""2D vector Snell's law with TIR detection.

For a sphere with signed curvature c, the unit normal at the intersection
point (z_int, y_int) pointing back toward the source (medium 1) is

    n_z = -sqrt(1 - c^2 * y_int^2)
    n_y = c * y_int

This is already unit length: n_z^2 + n_y^2 = 1.
"""
import torch
from torch import Tensor

from lens_opt.tracer.surfaces import Surface

_EPS = 1e-30


def surface_normal(y_int: Tensor, surf: Surface) -> tuple[Tensor, Tensor]:
    """Source-facing unit normal at the intersection point."""
    c = surf.c
    arg = 1.0 - c * c * y_int * y_int
    n_z = -torch.sqrt(torch.clamp(arg, min=_EPS))
    n_y = c * y_int
    return n_z, n_y


def refract(
    dz: Tensor, dy: Tensor, y_int: Tensor, surf: Surface,
    n_before: Tensor, n_after: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply Snell's law in 2D vector form.

    Returns (new_dz, new_dy, valid). `valid` is False where total internal
    reflection occurs; the returned direction is clamped to avoid NaN but
    is not physically meaningful for invalid rays.
    """
    n_z, n_y = surface_normal(y_int, surf)
    cos_i = -(dz * n_z + dy * n_y)
    ratio = n_before / n_after
    sin_t_sq = ratio * ratio * (1.0 - cos_i * cos_i)
    valid = sin_t_sq <= 1.0
    cos_t = torch.sqrt(torch.clamp(1.0 - sin_t_sq, min=_EPS))
    factor = ratio * cos_i - cos_t
    new_dz = ratio * dz + factor * n_z
    new_dy = ratio * dy + factor * n_y
    return new_dz, new_dy, valid
