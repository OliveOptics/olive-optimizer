"""Pack/unpack the variable parameters of a System into a flat design vector.

The merit function and LM optimizer want to differentiate w.r.t. a single
1D tensor `v`. This module provides:
    DesignTemplate  — metadata: which slots of a System are variable
    pack(template)  — current values of variable slots as a flat tensor
    unpack(v, template) -> System  — plug v into the variable slots

Convention for v's layout:
    v[0:len(c_indices)]   = curvatures for c_indices, in given order
    v[len(c_indices):]    = thicknesses for t_indices, in given order

The round-trip  unpack(pack(template), template).surfaces[i].c == template
.fixed_system.surfaces[i].c  must hold exactly (no float drift).
"""
from dataclasses import dataclass
from typing import Tuple
import torch
from torch import Tensor

from lens_opt.tracer.surfaces import Surface
from lens_opt.tracer.system import System


@dataclass(frozen=True)
class DesignTemplate:
    """Describes which slots of a fixed System are variable.

    fixed_system : a System with all slots filled in.
    c_indices    : surface indices whose curvature is variable.
    t_indices    : gap indices whose thickness is variable.
    """
    fixed_system: System
    c_indices: Tuple[int, ...]
    t_indices: Tuple[int, ...]

    @property
    def n_vars(self) -> int:
        return len(self.c_indices) + len(self.t_indices)


def pack(template: DesignTemplate) -> Tensor:
    """Extract current variable values from template.fixed_system."""
    parts = []
    for i in template.c_indices:
        parts.append(template.fixed_system.surfaces[i].c)
    for i in template.t_indices:
        parts.append(template.fixed_system.gap_thicknesses[i])
    if not parts:
        return torch.empty(0, dtype=torch.float64)
    return torch.stack(parts)


def unpack(v: Tensor, template: DesignTemplate) -> System:
    """Build a new System with v plugged into the variable slots.

    Non-variable slots come straight from template.fixed_system. The new
    System shares tensor objects with the template for the fixed pieces.
    """
    n_c = len(template.c_indices)
    new_surfaces = list(template.fixed_system.surfaces)
    for k, surf_i in enumerate(template.c_indices):
        new_surfaces[surf_i] = Surface(c=v[k])
    new_thicknesses = list(template.fixed_system.gap_thicknesses)
    for k, gap_i in enumerate(template.t_indices):
        new_thicknesses[gap_i] = v[n_c + k]
    return System(
        entrance_material=template.fixed_system.entrance_material,
        surfaces=tuple(new_surfaces),
        gap_thicknesses=tuple(new_thicknesses),
        gap_materials=template.fixed_system.gap_materials,
        semi_diameters=template.fixed_system.semi_diameters,
    )
