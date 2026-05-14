"""Closed-form ray-sphere intersection (2D meridional).

For a ray that has been propagated to the surface's vertex plane (z=z_v,
y=y_vp) with unit direction (dz, dy), the near-side intersection with the
sphere  c*(z^2 + y^2) - 2z = 0  is given by

    E = dz - c*y_vp*dy
    F = c*y_vp**2
    t = F / (E + sqrt(E^2 - c*F))

This is the numerically clean form: F -> 0 as y_vp -> 0 gives t -> 0 (the
trivial intersection at the vertex), and c -> 0 gives F -> 0 (flat surface
is hit at the vertex plane).
"""
import torch
from torch import Tensor

from lens_opt.tracer.surfaces import Surface

_EPS = 1e-30


def sag(y: Tensor, c: Tensor) -> Tensor:
    """Sphere sag at height y (signed z-displacement from vertex)."""
    arg = 1.0 - c * c * y * y
    sqrt_arg = torch.sqrt(torch.clamp(arg, min=_EPS))
    return c * y * y / (1.0 + sqrt_arg)


def intersect_surface(
    y_vp: Tensor, dz: Tensor, dy: Tensor, surf: Surface,
) -> tuple[Tensor, Tensor, Tensor]:
    """Find ray-sphere intersection.

    Preconditions: ray has been propagated to the surface's vertex plane,
    so the y coordinate is `y_vp` and z (relative to vertex) is 0.
    Direction (dz, dy) is a unit vector.

    Returns:
        y_int     : y coordinate at the intersection.
        z_offset  : z displacement from the vertex plane (= t * dz).
        valid     : bool mask, True where the intersection is well-defined
                    (sphere discriminant non-negative).
    """
    c = surf.c
    E = dz - c * y_vp * dy
    F = c * y_vp * y_vp
    disc = E * E - c * F
    valid = disc >= 0
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=_EPS))
    denom = E + sqrt_disc
    denom_safe = torch.where(torch.abs(denom) < _EPS, torch.full_like(denom, _EPS), denom)
    t = F / denom_safe
    y_int = y_vp + t * dy
    z_offset = t * dz
    return y_int, z_offset, valid
