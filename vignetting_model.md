# Vignetting and Relative Illumination Model

## Physical Setup

At the max field, a ray at pupil coordinate ρ (ranging from -1 to +1) hits surface i at height:

    h(ρ) = ρ · h_m[i] + h_c[i]

where:
- h_m[i] = marginal ray height at surface i (beam half-width)
- h_c[i] = chief ray height at surface i (off-axis beam center offset)

The beam at surface i spans from `h_c[i] - h_m[i]` to `h_c[i] + h_m[i]`.

## Vignetting Condition

Surface i has a physical clear semi-aperture CA[i]. A ray passes through if:

    -CA[i] ≤ ρ · h_m[i] + h_c[i] ≤ +CA[i]

Solving for ρ (assuming h_m[i] > 0):

    ρ_min[i] = (-CA[i] - h_c[i]) / h_m[i]
    ρ_max[i] = (+CA[i] - h_c[i]) / h_m[i]

Clamped to [-1, +1] since ρ cannot exceed the full pupil.

If CA[i] ≥ |h_m[i]| + |h_c[i]|, the surface does not vignette (full beam passes through).

## Overall Unvignetted Pupil

Each surface clips a different range of ρ. The overall unvignetted range is the intersection across all surfaces:

    ρ_min_total = max(ρ_min[i]) for all i
    ρ_max_total = min(ρ_max[i]) for all i

If ρ_min_total ≥ ρ_max_total, the beam is fully blocked at this field.

## Relative Illumination

For a circular pupil (2D), ρ is the 1D coordinate through the pupil center. The unvignetted fraction of the pupil area depends on how the clipping interacts with the circular geometry.

For the 1D (slit) approximation:

    RI ≈ (ρ_max_total - ρ_min_total) / 2

For a circular pupil with symmetric upper/lower clipping, the exact RI involves the area of intersection of circles, but the 1D linear model gives a practical differentiable approximation.

## As an Optimizer Constraint

The RI constraint for the optimizer:

    RI(max field) ≥ RI_min

where RI_min is a design requirement (e.g., 0.5 for 50% relative illumination at the field edge).

This is an inequality constraint that depends on:
- h_m[i], h_c[i] from the paraxial trace (already computed)
- CA[i] for each surface (design parameter or derived from lens geometry)

Since h_m and h_c flow through the y-ν trace, the gradient of RI w.r.t. design variables (curvatures, gaps) is available through the same chain rule machinery used for the Seidel gradient.

## Integration with Edge Thickness

CA[i] is not a free parameter — it is determined by the lens geometry:
- For a lens element, CA is typically set by the beam footprint plus some margin
- Or CA is fixed by the mechanical housing diameter
- The edge thickness constraint already couples CA with curvatures and center thickness:

      d_edge = d_center - sag(c_front, CA_front) + sag(c_back, CA_back)

So vignetting, edge thickness, and the optical design are all coupled through the clear apertures.
