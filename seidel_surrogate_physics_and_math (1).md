# Seidel Wavefront Surrogate: Physics and Math Model

## 1. The Physical Problem

We want to find the shape and arrangement of optical surfaces that transform input wavefronts into desired output wavefronts.

A point source emits a diverging spherical wavefront. A perfect lens converts it into a converging sphere focused to a point. In reality, the lens introduces errors — the converging wavefront isn't a perfect sphere. The deviation from a perfect sphere is the **wavefront error**, and it directly determines image quality.

Each refracting surface is a boundary between two media. What it does physically: it adds an optical path length (OPL) that varies across the aperture. The OPL contribution of a surface with sag profile z(x,y) going from index n₁ to n₂ is:

    ΔW(x,y) = (n₂ - n₁) · z(x,y)

This is Fermat's principle. The wavefront change equals the index difference times the surface shape.

For a system with K surfaces, the total wavefront error is:

    W_total = Σᵢ (n'ᵢ - nᵢ) · zᵢ(yᵢ) - W_reference

where W_reference is the ideal converging sphere to the image point.

The hard part: yᵢ (the ray height at surface i) depends on what all previous surfaces did. This coupling is what makes optical optimization difficult.

## 2. The Single Surface Case

A point source at distance s emits a diverging wavefront that hits a spherical surface with curvature c = 1/R, at the boundary between index n₁ and n₂.

The incoming wavefront OPL variation up to 4th order:

    W_in(y) = n₁ · (y²/2s + y⁴/8s³ + ...)

The spherical surface sag (Taylor expansion):

    z(y) = cy²/2 + c³y⁴/8 + ...

So the surface contributes:

    ΔW(y) = (n₂ - n₁) · (cy²/2 + c³y⁴/8 + ...)

The outgoing reference sphere:

    W_out(y) = n₂ · (y²/2s' + y⁴/8s'³ + ...)

Setting W_in + ΔW = W_out at each order:

**2nd order (y² terms) → paraxial imaging equation:**

    n₁/s + (n₂ - n₁)c = n₂/s'

This determines where the image forms. Always solvable. No aberration at this order.

**4th order (y⁴ terms) → spherical aberration:**

    W_error = (n₁/8s³ + (n₂-n₁)c³/8 - n₂/8s'³) · y⁴

This is the residual wavefront error. It depends on y⁴, so edge rays get it worst. For a single surface with fixed conjugates, the curvature c is determined by the paraxial equation, so you cannot independently zero this term.

## 3. Multiple Surfaces and the Coupling Problem

As surfaces increase:

- **1 surface:** Curvature determined by paraxial condition. Zero free parameters. Stuck with whatever aberration the physics gives you.
- **2 surfaces (single lens):** Two curvatures, one paraxial power constraint. One free parameter (bending). Can minimize but generally not zero spherical aberration.
- **5+ surfaces:** More free parameters than 3rd-order aberrations (there are 5 Seidel types). Many solutions can zero the 3rd-order terms, but they differ in higher-order residuals.

The fundamental difficulty: the total wavefront error is a nested composition of coupled nonlinear functions. Each surface's aberration contribution depends on the ray height yᵢ, which depends on all preceding surfaces. Changing surface 1 changes where light hits surface 2, which changes surface 2's contribution.

## 4. The Seidel Approximation

The Seidel model makes one key simplification: replace the real ray height yᵢ = hᵢ + δᵢ with the paraxial ray height hᵢ, dropping the aberration-induced deviation δᵢ.

This means each surface's aberration contribution depends only on paraxial ray data, computed once upfront. The nonlinear nested composition becomes a linear chain (paraxial trace) followed by a simple sum (aberration contributions).

## 5. The y-nu Paraxial Ray Trace

The y-nu trace tracks a ray through the system using two numbers at each surface:
- y: ray height
- ν = n·u: product of refractive index and ray angle

We use ν = n·u instead of u because it makes the recurrence cleaner (ν is the natural variable for paraxial Snell's law).

**At a surface (refraction with power φᵢ = (n'ᵢ - nᵢ) · cᵢ):**

    ν' = ν - y · φ

**Between surfaces (propagation through gap of thickness d in index n):**

    y_next = y + (d/n) · ν

These two equations alternate: refract, propagate, refract, propagate...

### Why Two Rays

Aberrations depend on two independent things: where in the aperture the ray is, and where in the field the object point is. Two rays span this 2D space at the paraxial level.

**Marginal ray** — represents the aperture:
- Starts on-axis at the object, goes to the edge of the aperture stop
- Its height hᵢ at each surface tells you the beam footprint size on that surface
- Initial conditions: determined by object distance and aperture size

**Chief ray** — represents the field:
- Starts at the edge of the field of view, passes through the center of the aperture stop
- Its height h̄ᵢ at each surface tells you how far off-center an off-axis beam lands
- Initial conditions: determined by field angle and stop position

Both rays are traced using the same y-nu recurrence. This gives six numbers at each surface: hᵢ, νᵢ (marginal) and h̄ᵢ, ν̄ᵢ (chief), plus the known surface parameters cᵢ, nᵢ, n'ᵢ.

### Chain Dependency

The y-nu trace is a chain: h and ν at surface i depend on all preceding curvatures and thicknesses.

    ν'₁ = ν₁ - h₁ · φ₁           (refract at surface 1)
    h₂ = h₁ + (d₁/n'₁) · ν'₁     (propagate through gap in post-refraction medium)
    ν'₂ = ν'₁ - h₂ · φ₂          (refract at surface 2)
    h₃ = h₂ + (d₂/n'₂) · ν'₂     (propagate)
    ...

Changing c₁ changes φ₁, which changes ν'₁, which changes h₂, and so on downstream. This means the Seidel aberrations at later surfaces depend on earlier design variables. The coupling is real but **linear** — much simpler than the nonlinear coupling in real raytracing.

## 6. Seidel Aberration Coefficients

At each surface i, define the refraction invariants:

    Aᵢ = nᵢ · uᵢ + nᵢ · hᵢ · cᵢ     (equivalently: νᵢ + nᵢ · hᵢ · cᵢ)
    Āᵢ = nᵢ · ūᵢ + nᵢ · h̄ᵢ · cᵢ     (equivalently: ν̄ᵢ + nᵢ · h̄ᵢ · cᵢ)

And the change in u/n across the surface:

    Δ(u/n)ᵢ = u'ᵢ/n'ᵢ - uᵢ/nᵢ

And the Lagrange invariant (constant throughout the system):

    H = nᵢ · uᵢ · h̄ᵢ - nᵢ · ūᵢ · hᵢ

The five Seidel contributions from surface i:

    S_Iᵢ   = -Aᵢ² · hᵢ · Δ(u/n)ᵢ           — spherical aberration
    S_IIᵢ  = -Aᵢ · Āᵢ · hᵢ · Δ(u/n)ᵢ        — coma
    S_IIIᵢ = -Āᵢ² · hᵢ · Δ(u/n)ᵢ            — astigmatism
    S_IVᵢ  = -H² · φᵢ / (nᵢ · n'ᵢ)                — Petzval field curvature
    S_Vᵢ   = (S_IIIᵢ + S_IVᵢ) · (Āᵢ / Aᵢ)   — distortion

Every quantity here is a known number from the paraxial trace and the design prescription.

### How Aperture and Field Combine

The five aberrations correspond to different combinations of aperture (h) and field (h̄):

    Spherical aberration:  h⁴         — pure aperture
    Coma:                  h³ · h̄     — mostly aperture, some field
    Astigmatism:           h² · h̄²   — equal mix
    Field curvature:       h² · h̄²   — equal mix (different symmetry)
    Distortion:            h  · h̄³   — mostly field

This is why we need both the marginal and chief rays — they span the aperture and field dimensions respectively.

### Total System Aberrations

The total Seidel aberrations are sums over surfaces:

    S_I_total   = Σᵢ S_Iᵢ
    S_II_total  = Σᵢ S_IIᵢ
    S_III_total = Σᵢ S_IIIᵢ
    S_IV_total  = Σᵢ S_IVᵢ
    S_V_total   = Σᵢ S_Vᵢ

The summation is simple, but remember each term shares the underlying y-nu chain so they are not fully independent with respect to design variables.

## 7. Aspheric Surface Contributions

An aspheric surface has sag:

    z(y) = cy²/2 · 1/(1 + sqrt(1-(1+k)c²y²)) + A₄y⁴ + A₆y⁶ + ...

The base sphere is handled by the standard Seidel formulas above. The aspheric departure adds extra terms to the Seidel sums.

**Conic constant k — modifies spherical aberration:**

    S_I_conic = -(n'ᵢ - nᵢ) · kᵢ · cᵢ³ · hᵢ⁴

**Even polynomial A₄ — adds to spherical aberration:**

    S_I_A4 = -8(n'ᵢ - nᵢ) · A₄ᵢ · hᵢ⁴

These add directly to S_I at that surface. The aspheric terms do not affect the paraxial ray trace (which depends only on the base curvature).

Important: aspherics placed away from the stop also contribute to off-axis aberrations (coma, astigmatism). The off-axis aspheric contributions are:

    S_II_asph  = S_I_asph · (h̄ᵢ / hᵢ)
    S_III_asph = S_I_asph · (h̄ᵢ / hᵢ)²

This means an aspheric at the stop (where h̄ = 0) only corrects spherical aberration. An aspheric away from the stop also affects coma and astigmatism. This is a key design lever.

## 8. Wavefront Error from Seidel Coefficients

The Seidel sums map to wavefront error as a function of normalized pupil coordinate ρ (0 to 1) and normalized field coordinate η (0 to 1):

    W(ρ, η, θ) = S_I · ρ⁴
               + S_II · η · ρ³ · cos θ
               + S_III · η² · ρ² · cos²θ
               + S_IV · η² · ρ²
               + S_V · η³ · ρ · cos θ

where θ is the azimuthal angle in the pupil.

### Mapping to Zernike Coefficients

Each Seidel term corresponds to specific Zernike polynomials:

    a₁₁ (spherical)     ∝ S_I
    a₇, a₈ (coma)       ∝ S_II
    a₅, a₆ (astigmatism) ∝ S_III
    a₄ (defocus/field curvature) ∝ S_III + S_IV
    tilt terms           ∝ S_V

The Zernike representation is useful because each coefficient is a named, physically meaningful aberration, and the merit function has a clean form.

## 9. Merit Function

The merit function is a weighted sum of squared Seidel aberrations, evaluated over the desired field points:

    M = Σ_field_points Σ_aberrations wⱼ · Sⱼ²

Or equivalently in Zernike form:

    M = Σ_field_points Σⱼ wⱼ · aⱼ²

The weights wⱼ encode design priorities — e.g., weight spherical aberration more heavily for an on-axis system, or weight astigmatism more for a wide-field system.

## 10. Gradients

Every Seidel coefficient is an explicit function of the design variables (cᵢ, dᵢ, kᵢ, A₄ᵢ, nᵢ). The dependence flows through:

1. The y-nu recurrence: hᵢ and νᵢ depend on all preceding c and d values
2. The refraction invariant: Aᵢ depends on hᵢ, uᵢ, cᵢ, nᵢ
3. The Seidel formula: S depends on Aᵢ, hᵢ, and the refraction change

The gradient ∂M/∂(any variable) follows by the chain rule through this entire path. Because every step is polynomial arithmetic, the gradients are exact and cheap — comparable in cost to one forward evaluation.

For the merit function M = Σ wⱼ Sⱼ²:

    ∂M/∂c₁ = 2 · Σⱼ wⱼ · Sⱼ · (∂Sⱼ/∂c₁)

where ∂Sⱼ/∂c₁ includes both the direct effect at surface 1 and the indirect effect through the y-nu chain on all downstream surfaces.

## 11. Accuracy and Validity

**What this model captures:**
- All five 3rd-order monochromatic aberrations
- Correct field dependence (through the chief ray)
- Correct surface-by-surface contributions
- Aspheric effects (conic and even polynomial)
- The linear coupling between surfaces through the y-nu chain

**What this model misses:**
- Higher-order aberrations (5th, 7th order) — important for fast systems below ~f/2
- Chromatic aberrations — unless you extend by running the trace at multiple wavelengths with wavelength-dependent nᵢ
- Nonlinear ray height coupling — the real ray deviation δᵢ from paraxial, which matters for very steep rays
- Vignetting — rays that miss surfaces entirely

**Accuracy range:** Reliable for systems f/2 and slower. The model finds the correct optimization basins even when absolute aberration values have some error, because the relative ranking of designs is preserved.

## 12. Design Variables and Constraints

**Variables the optimizer can change:**
- cᵢ: surface curvatures (continuous)
- dᵢ: surface spacings / thicknesses (continuous)
- kᵢ: conic constants (continuous, only on aspheric surfaces)
- A₄ᵢ, A₆ᵢ: polynomial aspheric coefficients (continuous, only on aspheric surfaces)
- nᵢ: refractive indices (discrete — from glass catalog, or continuous if treating as a variable to be matched later)

**Constraints:**
- Total system power (focal length): determined by the paraxial imaging equation
- Physical thickness: dᵢ > 0
- Edge/center thickness: surfaces must not intersect
- Glass availability: nᵢ must correspond to real glasses (handled in refinement phase)

## 13. Summary of Computational Pipeline

```
INPUT: Design prescription (curvatures, thicknesses, indices, conics, aspherics)
       System specs (object distance, field angle, aperture)

STEP 1: y-nu trace
  - Trace marginal ray → hᵢ, νᵢ at each surface
  - Trace chief ray → h̄ᵢ, ν̄ᵢ at each surface
  - Cost: O(K) arithmetic operations, K = number of surfaces

STEP 2: Seidel coefficients
  - At each surface, compute Aᵢ, Āᵢ, Δ(u/n)ᵢ
  - Compute S_I through S_V for each surface
  - Add aspheric contributions where applicable
  - Cost: O(K) arithmetic operations

STEP 3: Sum over surfaces
  - S_total = Σᵢ Sᵢ for each aberration type
  - Cost: O(K) additions

STEP 4: Merit function
  - M = Σ wⱼ Sⱼ² (optionally over multiple field points)
  - Cost: O(1)

STEP 5: Gradients (if needed)
  - Differentiate through steps 1-4 by chain rule
  - Cost: O(K × V) where V = number of design variables

TOTAL: O(K × V) per evaluation with gradients — pure arithmetic, no raytracing
```
