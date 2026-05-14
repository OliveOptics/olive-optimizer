"""Lens-layout and ray visualization for the 2D meridional tracer.

Reusable helpers:
    trace_history(ray, system)  -> [(z, y), ...]   one ray, all vertices + image plane
    draw_system(ax, system, semi_diameters)        draw all surface outlines
    plot_rays(ax, ray_bundle, system, **kwargs)    trace and plot each ray
"""
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch

from lens_opt.tracer.surfaces import Ray
from lens_opt.tracer.system import System
from lens_opt.tracer.intersect import intersect_surface
from lens_opt.tracer.refract import refract


def trace_history(ray: Ray, system: System) -> List[Tuple[float, float]]:
    """Trace a single ray through `system`, returning (z, y) at every key point.

    Output: [start, intersection at surface 0, ..., intersection at surface K-1,
    image plane]. Each entry is a 2-tuple of Python floats.
    """
    n_before = system.entrance_material.n(ray.wavelength)
    z = ray.z.clone()
    y = ray.y.clone()
    dz = ray.dz.clone()
    dy = ray.dy.clone()
    pts: List[Tuple[float, float]] = [(float(z), float(y))]

    z_vertex = torch.zeros_like(z)
    for i, surf in enumerate(system.surfaces):
        t_prop = (z_vertex - z) / dz
        y = y + t_prop * dy
        z = z_vertex
        y_int, z_off, _ = intersect_surface(y, dz, dy, surf)
        y = y_int
        z = z + z_off
        pts.append((float(z), float(y)))
        n_after = system.gap_materials[i].n(ray.wavelength)
        new_dz, new_dy, _ = refract(dz, dy, y, surf, n_before, n_after)
        dz, dy = new_dz, new_dy
        n_before = n_after
        z_vertex = z_vertex + system.gap_thicknesses[i]

    t_prop = (z_vertex - z) / dz
    y = y + t_prop * dy
    z = z_vertex
    pts.append((float(z), float(y)))
    return pts


def _sphere_z(c: float, y: np.ndarray, z_vertex: float) -> np.ndarray:
    """Compute z(y) on a sphere with curvature c and vertex at z_vertex."""
    if abs(c) < 1e-15:
        return np.full_like(y, z_vertex)
    arg = np.clip(1.0 - c * c * y * y, 0.0, None)
    return z_vertex + c * y * y / (1.0 + np.sqrt(arg))


def draw_system(ax, system: System, semi_diameters=None, *,
                glass_pairs: List[Tuple[int, int]] = None,
                element_color='lightsteelblue', edge_color='navy'):
    """Draw surface outlines and fill glass elements.

    Parameters
    ----------
    semi_diameters : optional override. By default uses system.semi_diameters
        (the physical glass rim, which is also the clear aperture). Pass a
        list of floats to draw at a different scale (e.g. cosmetic margin).
    glass_pairs : list of (front_surf, back_surf) tuples describing each glass
        element. If None, assume consecutive pairs (0,1), (2,3), ...
    """
    K = len(system.surfaces)
    if semi_diameters is None:
        semi_diameters = [float(sd) for sd in system.semi_diameters]
    z_vertex = [0.0]
    for i in range(K - 1):
        z_vertex.append(z_vertex[-1] + float(system.gap_thicknesses[i]))

    if glass_pairs is None:
        glass_pairs = [(2 * i, 2 * i + 1) for i in range(K // 2)]

    for front, back in glass_pairs:
        h_f = semi_diameters[front]
        h_b = semi_diameters[back]
        h = min(h_f, h_b)
        y = np.linspace(-h, h, 80)
        z_f = _sphere_z(float(system.surfaces[front].c), y, z_vertex[front])
        z_b = _sphere_z(float(system.surfaces[back].c), y, z_vertex[back])
        z_outline = np.concatenate([z_f, z_b[::-1]])
        y_outline = np.concatenate([y, y[::-1]])
        ax.fill(z_outline, y_outline, color=element_color, alpha=0.4)
        ax.plot(z_f, y, color=edge_color, linewidth=1.2)
        ax.plot(z_b, y, color=edge_color, linewidth=1.2)
        ax.plot([z_f[0], z_b[0]], [-h, -h], color=edge_color, linewidth=0.8)
        ax.plot([z_f[-1], z_b[-1]], [h, h], color=edge_color, linewidth=0.8)

    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':', alpha=0.6)
    return z_vertex


def plot_rays(ax, ray_bundle: Ray, system: System, *, extend_before: float = 5.0,
              **plot_kwargs):
    """Trace and plot every ray in `ray_bundle` (Ray with batch shape [N])."""
    n = int(ray_bundle.y.shape[0])
    for i in range(n):
        single = Ray(
            z=ray_bundle.z[i],
            y=ray_bundle.y[i],
            dz=ray_bundle.dz[i],
            dy=ray_bundle.dy[i],
            wavelength=ray_bundle.wavelength[i],
            valid=ray_bundle.valid[i],
        )
        pts = trace_history(single, system)
        zs = [pts[0][0] - extend_before] + [p[0] for p in pts]
        ys = [pts[0][1] - extend_before * float(ray_bundle.dy[i] / ray_bundle.dz[i])] + [p[1] for p in pts]
        ax.plot(zs, ys, **plot_kwargs)
