# Differentiable Lens Optimizer with Levenberg–Marquardt — Implementation Plan (2D Meridional)

## Goal

Build a GPU-accelerated, differentiable **2D meridional** sequential ray tracer in PyTorch, with a Levenberg–Marquardt optimizer that minimizes RMS transverse ray error. Match Zemax-style DLS behavior on a Cooke triplet benchmark.

The system is assumed rotationally symmetric; rays are confined to the meridional (y–z) plane. This is sufficient for tangential aberration analysis and keeps the Jacobian small.

## Stack

- Python 3.10+
- PyTorch 2.1+ (`torch`, `torch.func`)
- NumPy (for test data, comparisons)
- Matplotlib (spot diagrams, convergence plots)
- pytest (unit tests)

GPU optional but supported via `.to(device)`. All code must run on CPU as well for tests.

## Conventions (fixed up front — don't change later)

- **Coordinate system:** light propagates in +z direction. y is the transverse coordinate. All rays live in the meridional (y–z) plane — no x component, no skew rays.
- **Curvature sign:** `c = 1/R`, positive when the center of curvature is to the image side of the vertex (standard lens-design convention).
- **Surface normal:** points back toward the medium the ray came from (medium 1).
- **Units:** millimeters for lengths, micrometers for wavelengths (convert internally when needed).
- **Tensor shapes:** ray position and direction carry batch shape `[..., 2]` (z, y components), scalar quantities are `[...]`. Standard batch order: `[design, field, wavelength, pupil_ray]` when all are present.
- **Dtype:** float64 throughout for geometry. Mixed precision is a later optimization, not a starting choice.

## Scope (what 2D meridional buys and costs)

- Captures: spherical aberration, coma (tangential), field curvature (tangential focal shift), axial color, lateral color (tangential), distortion (image height vs field).
- Does NOT capture: astigmatism (no sagittal rays), sagittal coma, anything requiring skew rays. If those become important, the natural extension is to add an x-component (Milestone 6+), not to redesign.

---

## Milestone 1: Differentiable ray tracer (forward pass)

### Deliverables

- `tracer/surfaces.py` — surface types (Spherical, Asphere, Plano)
- `tracer/refract.py` — 2D vector Snell's law with TIR handling
- `tracer/intersect.py` — sphere intersection (closed form in y–z), asphere intersection (Newton, fixed iteration count)
- `tracer/glass.py` — Sellmeier dispersion `n(wavelength)`
- `tracer/system.py` — sequential trace through a surface list
- `tracer/sources.py` — meridional ray bundle generation (1D pupil fan: linear, oversampled-near-edges, randomized)

### Key requirements

- All operations on PyTorch tensors, no NumPy in the trace itself.
- Every operation that could produce NaN must be guarded: `sqrt` arguments clamped to `min=eps`, divisions guarded against zero curvature.
- No Python `if`/`else` on tensor values. Use `torch.where` for any conditional behavior.
- Newton iteration for asphere intersection uses a **fixed** iteration count (8 is fine). No while-loops with tensor-valued exit conditions.
- Validity is tracked as a boolean tensor alongside the rays. Invalid rays continue propagating (with garbage values) but are masked from the merit function. **Never branch the computation on validity** — masks are metadata.

### Tests (must pass before moving on)

1. **Plano refraction at normal incidence:** ray entering glass perpendicular to a flat surface should exit unchanged in direction. Index ratio doesn't matter.
2. **Snell's law numerical check:** for an oblique ray and a flat surface, compare `sin(θ_i)/sin(θ_t)` to `n2/n1` to 10 decimals.
3. **Spherical paraxial focus:** trace parallel rays through a single positive lens (BK7, R1=50mm, R2=-50mm, t=5mm) and check the back focal length matches the lensmaker's equation to within 0.1%.
4. **Reversibility:** trace a ray through the system, reverse the output direction, trace back. The reversed ray should hit the original starting point to within 1e-9 mm.
5. **Asphere reduces to sphere:** an asphere with k=0 and all aspheric coefficients = 0 must produce identical intersection points to the spherical intersection routine.

### Acceptance criterion

All 5 tests pass on CPU and GPU. Run `pytest tests/test_forward_trace.py -v` and see green.

---

## Milestone 2: Gradient correctness

### Deliverables

- `tracer/merit.py` — RMS transverse-ray-error merit function returning **two forms**:
  - `merit_scalar(v)` — single scalar suitable for `.backward()`
  - `merit_residuals(v)` — flat residual vector suitable for LM (one entry per ray: signed y deviation from the chief-ray image point, or from the field-weighted centroid)

### Key requirements

- The residual vector must satisfy `0.5 * (r**2).sum() == merit_scalar` to machine precision. This is non-negotiable; LM depends on it.
- The merit function takes the **design vector** `v` as a flat 1D tensor and internally unpacks it into surface parameters. This decoupling matters for `jacfwd`.
- Build a `pack(surfaces) -> v` and `unpack(v) -> surfaces` pair. Round-trip must be exact.

### Tests (must pass before moving on)

1. **Finite-difference gradient check:** for a single variable (one curvature), compute `(merit(v+eps) - merit(v-eps)) / (2*eps)` and compare to the autograd gradient. Should agree to ~6 decimal places with `eps=1e-5`.
2. **Full Jacobian finite-difference check:** for a small problem (3 variables, 10 rays), build the Jacobian both by `jacfwd(merit_residuals)(v)` and by central differences over each variable. They should agree to ~6 decimal places.
3. **Residual-scalar consistency:** `torch.allclose(0.5 * merit_residuals(v).pow(2).sum(), merit_scalar(v))` returns True.
4. **NaN absence:** intentionally configure a near-TIR ray (high incidence angle, large index ratio). Verify the gradient w.r.t. all variables is finite (not NaN).

### Acceptance criterion

All 4 tests pass. This milestone is the highest-risk one — bugs here will mimic LM bugs later. Don't proceed until gradients are verified.

---

## Milestone 3: Levenberg–Marquardt optimizer

### Deliverables

- `optim/lm.py` — LM step function and outer driver

### Algorithm

```
Inputs: v0 (initial design), residual_fn, max_iter, tol
λ ← 1e-3      (initial damping)
v ← v0

for iter in 1..max_iter:
    r ← residual_fn(v)
    J ← jacfwd(residual_fn)(v)
    g ← Jᵀ r                              # gradient = Jᵀr
    Φ ← 0.5 ‖r‖²
    
    if ‖g‖_∞ < tol: return v, "converged on gradient"
    
    # Marquardt: damping scaled by diag(JᵀJ)
    H ← Jᵀ J
    D ← diag(H)
    
    # Step search loop
    while λ < 1e10:
        A ← H + λ · D
        Δv ← solve(A, -g)                 # use lstsq for stability
        
        v_trial ← v + Δv
        Φ_trial ← 0.5 ‖residual_fn(v_trial)‖²
        
        # Gain ratio
        predicted_reduction ← 0.5 · Δvᵀ (λ·D·Δv - g)
        actual_reduction ← Φ - Φ_trial
        ρ ← actual_reduction / predicted_reduction
        
        if ρ > 0:
            v ← v_trial
            λ ← λ · max(1/3, 1 - (2ρ - 1)³)    # Nielsen update
            ν ← 2
            break  # accept step, go to next iter
        else:
            λ ← λ · ν
            ν ← ν · 2
            continue  # reject, try larger λ
    
    if step rejection bailed out: return v, "stuck"

return v, "max iterations reached"
```

### Key requirements

- **Solve via `torch.linalg.lstsq`** on the augmented system, not by forming `JᵀJ` explicitly. This is the numerically clean version:
  ```
  [    J   ]         [ -r ]
  [ √(λD)  ] · Δv ≈  [  0 ]
  ```
- Adaptive damping uses the Nielsen update rule shown above. Don't use naive ×10/÷10.
- Outer driver returns: final `v`, final merit, iteration history (list of merits per accepted step), exit reason.
- All operations stay on the same device as `v` (no CPU↔GPU shuffling inside the loop).

### Tests (must pass before benchmarking)

1. **Linear least-squares problem:** define `residuals(v) = A @ v - b` for known A, b. LM should converge in 1–2 iterations to the analytic solution `v = (AᵀA)⁻¹ Aᵀ b`.
2. **Rosenbrock-as-residuals:** classic test problem `r₁ = 10·(x₂ - x₁²), r₂ = 1 - x₁`. LM should converge to `(1, 1)` from `(-1.2, 1.0)` in fewer than 20 iterations.
3. **Damping behavior:** start with a deliberately bad initial design where the first Gauss-Newton step would overshoot. Verify λ increases, the step gets rejected, a smaller step is tried, and eventually accepted.

### Acceptance criterion

All 3 tests pass. LM works on synthetic problems before being applied to lenses.

---

## Milestone 4: Cooke triplet benchmark

### Deliverables

- `examples/cooke_triplet.py` — full pipeline: define starting design, optimize, plot spot diagram before/after, print convergence history

### Setup

Standard Cooke triplet (three elements, six surfaces, three glasses). Prescription is widely published — start from any reasonable design (e.g., from "Modern Lens Design" by Smith, or Zemax's sample files).

- 3 fields: 0°, 10°, 14° (or whatever the reference design uses)
- 3 wavelengths: F (486.1 nm), d (587.6 nm), C (656.3 nm)
- Pupil sampling: 1D meridional fan, 21 rays uniformly from normalized pupil −1 to +1 per field/wavelength (≈ 189 residuals total: 3·3·21)
- Variables to optimize: 6 curvatures + 2 air gaps = 8 variables (keep glasses and element thicknesses fixed for first run)

### Tests / validation

1. **Forward trace sanity:** the starting design should produce RMS transverse ray error in the 30–100 µm range across fields. If wildly different, the trace has a bug.
2. **Convergence:** LM from a slightly perturbed starting design (multiply each curvature by `1 + 0.01·randn`) should return to within 1% of the original merit in fewer than 20 iterations.
3. **Optimization from a worse start:** perturb by `1 + 0.05·randn` and verify LM converges to a sensible design with RMS transverse error < 20 µm.

### Plots to generate

- Ray-aberration plot (transverse error vs normalized pupil coordinate, per field, per wavelength) before and after optimization
- Merit vs iteration (semilogy)
- Damping λ vs iteration (semilogy)

### Acceptance criterion

Cooke triplet optimization runs end-to-end, converges to a sensible result, and the ray-aberration plot visibly tightens.

---

## Milestone 5: GPU batching for multiple starts

### Deliverables

- Modify the pipeline to accept a batched design vector of shape `[B, P]` where B is the number of designs and P is the variable count.
- Run LM on all B designs in parallel (each with its own λ, its own convergence state).

### Key requirements

- The trace must broadcast over the leading B dimension naturally — most of this is free if shapes were set up right in Milestone 1.
- The damping loop becomes per-design: some designs accept their step while others reject. Handle this with per-design λ tensors and `torch.where` to selectively update.
- Return shape: `[B, P]` final designs, `[B]` final merits, `[B]` exit reasons.

### Tests

1. **Equivalence:** running B=1 batched should produce identical results to the un-batched version.
2. **Independence:** running B=4 identical copies should produce 4 identical results.
3. **Speedup:** running B=100 random perturbations on GPU should take less than 5× the wall-clock of B=1.

### Acceptance criterion

100 randomized starts of the Cooke triplet optimize in parallel on a single GPU in under a minute.

---

## Out of scope (don't build yet)

- CMA-ES / global search wrapper — comes after the local optimizer is solid
- Glass selection (discrete) — needs a separate strategy
- Constraint handling (edge thickness, focal length targets) — add as soft penalty terms later
- `torch.compile` — apply only after correctness is established
- Freeform / Zernike surfaces — extension of the asphere code, not a Milestone-1 concern
- Skew rays / full 3D ray tracing — add an x-component to ray vectors when sagittal aberrations matter, treat as Milestone 6+

---

## File layout

```
lens_opt/
  tracer/
    __init__.py
    surfaces.py        # Surface dataclasses
    intersect.py       # Sphere + asphere intersection
    refract.py         # Vector Snell, TIR handling
    glass.py           # Sellmeier
    system.py          # Sequential trace
    sources.py         # Ray bundle generation
    merit.py           # Spot merit (scalar and residual forms)
  optim/
    __init__.py
    lm.py              # Levenberg-Marquardt
    pack.py            # pack/unpack design vector
  tests/
    test_forward_trace.py
    test_gradients.py
    test_lm_synthetic.py
    test_cooke.py
  examples/
    cooke_triplet.py
  pyproject.toml
  README.md
```

## Order of operations for Claude Code

1. Set up the project skeleton, `pyproject.toml`, install deps.
2. Build Milestone 1 entirely, including all 5 tests. Don't move on until they pass.
3. Build Milestone 2, including all 4 tests. **Take this milestone seriously** — gradient bugs masquerade as optimizer bugs later.
4. Build Milestone 3 with synthetic tests only. No lens problems yet.
5. Build Milestone 4 — first real optimization.
6. Build Milestone 5 — batching.

Each milestone ends with a green test suite. Don't stack new code on unverified foundations.
