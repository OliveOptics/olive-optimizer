"""Surface and Ray dataclasses for the 2D meridional tracer.

Surfaces are spherical (or plano when c=0). Asphere / conic terms have
been removed as out-of-scope for now.
"""
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class Surface:
    """A spherical refracting surface.

    Sag formula:  z = c*y^2 / (1 + sqrt(1 - c^2*y^2))
    Plano is the special case c = 0.
    """
    c: Tensor      # signed curvature (1/mm)


def make_surface(c=0.0, *, device=None, dtype=torch.float64) -> Surface:
    """Construct a Surface from a Python float or tensor."""
    return Surface(c=torch.as_tensor(c, device=device, dtype=dtype))


@dataclass(frozen=True)
class Ray:
    """A meridional ray bundle.

    All fields are tensors with a common batch shape `[...]`. Direction
    components are stored as a unit vector (dz, dy); dz**2 + dy**2 == 1.

    Validity is a boolean mask: a ray that suffers TIR or misses a surface
    is marked invalid and excluded from the merit function, but its tensor
    values keep flowing (with clamped numerics) so autograd stays NaN-free.
    """
    z: Tensor           # absolute z position (mm)
    y: Tensor           # transverse position (mm)
    dz: Tensor          # direction z-component (cos U)
    dy: Tensor          # direction y-component (sin U)
    wavelength: Tensor  # wavelength in micrometers
    valid: Tensor       # bool
