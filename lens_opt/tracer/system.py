"""Sequential optical system and forward trace.

Conventions:
- The first surface vertex is at z = 0.
- A Ray's `z` is its absolute z position; trace propagates it through.
- gap_thicknesses[i] is the axial distance from surface i to surface i+1,
  or from the last surface to the image plane for i = K-1.
- gap_materials[i] is the material in the gap after surface i.
- The image plane is at z = sum(gap_thicknesses).
"""
from dataclasses import dataclass, replace
from typing import Tuple
import torch

from lens_opt.tracer.surfaces import Surface, Ray
from lens_opt.tracer.glass import Material
from lens_opt.tracer.intersect import intersect_surface
from lens_opt.tracer.refract import refract


@dataclass(frozen=True)
class System:
    """A sequential optical system.

    Each surface has a `semi_diameter` (the radius of the glass rim) which
    is BOTH the drawing diameter AND the clear aperture for vignetting —
    in our model these are the same number. Rays whose intersection with
    a surface exceeds its semi-diameter are vignetted (marked invalid)
    and contribute zero to the merit.
    """
    entrance_material: Material
    surfaces: Tuple[Surface, ...]
    gap_thicknesses: Tuple[torch.Tensor, ...]
    gap_materials: Tuple[Material, ...]
    semi_diameters: Tuple[torch.Tensor, ...]


def uniform_semi_diameters(n_surfaces: int, value: float = 100.0,
                           *, device=None,
                           dtype=torch.float64) -> Tuple[torch.Tensor, ...]:
    """Helper: build a tuple of `n_surfaces` identical semi-diameters.

    Defaults to 100 mm — large enough that paraxial / small-pupil setups
    are effectively un-vignetted. Use a real per-surface tuple when the
    rim matters (e.g. fast lenses, off-axis fields).
    """
    return tuple(
        torch.as_tensor(value, dtype=dtype, device=device)
        for _ in range(n_surfaces)
    )


def forward_trace(ray: Ray, system: System) -> Ray:
    """Trace ray through `system`, returning the ray at the image plane.

    The ray's starting z may be anything; it's first propagated to the
    first surface's vertex plane (z = 0).
    """
    n_before = system.entrance_material.n(ray.wavelength)

    z = ray.z
    y = ray.y
    dz = ray.dz
    dy = ray.dy
    valid = ray.valid

    z_vertex = torch.zeros_like(ray.z)
    for i, surf in enumerate(system.surfaces):
        # Propagate to vertex plane
        t_prop = (z_vertex - z) / dz
        y = y + t_prop * dy
        z = z_vertex
        # Intersect with surface
        y_int, z_off, valid_isect = intersect_surface(y, dz, dy, surf)
        y = y_int
        z = z + z_off
        # Vignette: rays that hit outside the surface semi-diameter are blocked.
        valid_isect = valid_isect & (y_int.abs() <= system.semi_diameters[i])
        # Refract
        n_after = system.gap_materials[i].n(ray.wavelength)
        new_dz, new_dy, valid_refract = refract(dz, dy, y, surf, n_before, n_after)
        dz = new_dz
        dy = new_dy
        valid = valid & valid_isect & valid_refract
        # Advance vertex for next surface
        z_vertex = z_vertex + system.gap_thicknesses[i]
        n_before = n_after

    # Final propagate to image plane (at current z_vertex)
    t_prop = (z_vertex - z) / dz
    y = y + t_prop * dy
    z = z_vertex

    return replace(ray, z=z, y=y, dz=dz, dy=dy, valid=valid)


def reverse_system(system: System) -> Tuple[System, torch.Tensor]:
    """Build the system that traces light backwards through the original.

    Returns (reversed_system, pre_thickness). The reversed system has its
    first surface at z = 0 (per the standard convention); the caller must
    start the reverse ray at z = -pre_thickness so that propagation covers
    the original last gap (the distance between the original last surface
    and the original image plane).

    Surface flip rule: c -> -c.

    For K surfaces with original thicknesses t[0..K-1] (t[K-1] is the
    distance from the last surface to the image plane), the reversed
    thicknesses are
        new_t[i] = t[K-2-i] for i in 0..K-2,
        new_t[K-1] = 0,
        pre_thickness = t[K-1].
    Materials shift: new_entrance = g[K-1]; new_g[i] = g[K-2-i] for i in
    0..K-2; new_g[K-1] = original entrance.
    """
    K = len(system.surfaces)
    new_surfaces = tuple(
        replace(s, c=-s.c) for s in reversed(system.surfaces)
    )
    zero_t = torch.zeros_like(system.gap_thicknesses[0])
    new_thicknesses = tuple(list(reversed(system.gap_thicknesses[:-1])) + [zero_t])
    new_entrance = system.gap_materials[-1]
    new_gap_materials = tuple(
        list(reversed(system.gap_materials[:-1])) + [system.entrance_material]
    )
    pre_thickness = system.gap_thicknesses[-1]
    new_system = System(
        entrance_material=new_entrance,
        surfaces=new_surfaces,
        gap_thicknesses=new_thicknesses,
        gap_materials=new_gap_materials,
        semi_diameters=tuple(reversed(system.semi_diameters)),
    )
    return new_system, pre_thickness
