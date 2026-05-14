"""Meridional ray-bundle generation.

Helper functions to construct Ray bundles for tests and Cooke triplet
benchmarks. For real optimization runs, ray sources are usually built
inline from the system's aperture stop position and field angles.
"""
import math
import torch

from lens_opt.tracer.surfaces import Ray


def pupil_fan(
    n_rays: int,
    pupil_radius: float,
    field_angle_rad: float = 0.0,
    *,
    wavelength_um: float = 0.5876,
    chief_y: float = 0.0,
    chief_z: float = 0.0,
    device=None,
    dtype=torch.float64,
) -> Ray:
    """A 1D fan of meridional rays at one field angle.

    Rays sweep pupil heights uniformly from -pupil_radius to +pupil_radius.
    All rays share the same direction and wavelength.
    """
    rho = torch.linspace(-1.0, 1.0, n_rays, device=device, dtype=dtype)
    y0 = chief_y + rho * pupil_radius
    u = torch.full((n_rays,), float(field_angle_rad), device=device, dtype=dtype)
    return Ray(
        z=torch.full((n_rays,), float(chief_z), device=device, dtype=dtype),
        y=y0,
        dz=torch.cos(u),
        dy=torch.sin(u),
        wavelength=torch.full((n_rays,), float(wavelength_um), device=device, dtype=dtype),
        valid=torch.ones(n_rays, dtype=torch.bool, device=device),
    )


def parallel_bundle(
    n_rays: int,
    half_height: float,
    *,
    wavelength_um: float = 0.5876,
    chief_z: float = 0.0,
    device=None,
    dtype=torch.float64,
) -> Ray:
    """Parallel rays along +z, spanning [-half_height, +half_height] in y.

    Useful for testing paraxial focus of a singlet.
    """
    y0 = torch.linspace(-half_height, half_height, n_rays, device=device, dtype=dtype)
    return Ray(
        z=torch.full((n_rays,), float(chief_z), device=device, dtype=dtype),
        y=y0,
        dz=torch.ones(n_rays, device=device, dtype=dtype),
        dy=torch.zeros(n_rays, device=device, dtype=dtype),
        wavelength=torch.full((n_rays,), float(wavelength_um), device=device, dtype=dtype),
        valid=torch.ones(n_rays, dtype=torch.bool, device=device),
    )


def single_ray(
    y: float, u_rad: float,
    *,
    z: float = 0.0,
    wavelength_um: float = 0.5876,
    device=None,
    dtype=torch.float64,
) -> Ray:
    """A single meridional ray at given (y, angle)."""
    return Ray(
        z=torch.as_tensor(z, device=device, dtype=dtype),
        y=torch.as_tensor(y, device=device, dtype=dtype),
        dz=torch.as_tensor(math.cos(u_rad), device=device, dtype=dtype),
        dy=torch.as_tensor(math.sin(u_rad), device=device, dtype=dtype),
        wavelength=torch.as_tensor(wavelength_um, device=device, dtype=dtype),
        valid=torch.tensor(True, device=device),
    )
