"""RMS transverse-ray-error merit, in two forms required by LM.

merit_residuals(v, template, rays) -> Tensor[n_rays]
    Per-ray signed y-deviation from the bundle centroid. The Jacobian of
    this vector w.r.t. v is the Jacobian LM needs.

merit_scalar(v, template, rays) -> Tensor[]
    0.5 * sum(residuals**2). Suitable for .backward().

By construction:
    0.5 * merit_residuals(v).pow(2).sum() == merit_scalar(v)

Invalid rays (TIR or geometry failure) contribute zero residual so they
do not drive the optimization; the centroid is computed over valid rays
only. The computation is NaN-safe via torch.where masking.
"""
import torch
from torch import Tensor

from lens_opt.tracer.surfaces import Ray
from lens_opt.tracer.system import forward_trace
from lens_opt.optim.pack import DesignTemplate, unpack


def merit_residuals(v: Tensor, template: DesignTemplate, rays: Ray) -> Tensor:
    """Per-ray y-deviation from the valid-ray centroid (flat 1D)."""
    system = unpack(v, template)
    out = forward_trace(rays, system)
    valid_f = out.valid.to(out.y.dtype)
    zero = torch.zeros_like(out.y)
    y_clean = torch.where(out.valid, out.y, zero)
    n_valid = valid_f.sum().clamp(min=1.0)
    y_centroid = y_clean.sum() / n_valid
    return torch.where(out.valid, out.y - y_centroid, zero)


def merit_scalar(v: Tensor, template: DesignTemplate, rays: Ray) -> Tensor:
    """0.5 * sum(residuals**2). Differentiable, returns 0-dim tensor."""
    r = merit_residuals(v, template, rays)
    return 0.5 * (r * r).sum()
