"""Milestone 5 demo: 100 randomly perturbed Cooke triplet starts in parallel.

Builds B=100 perturbed copies of the published Cooke triplet prescription
and runs batched LM on the GPU (CPU also works). Reports:
  - per-design exit reasons (converged / max_iter / stuck)
  - distribution of final RMS spots
  - best/worst designs found
  - wall-clock

Saves a 2-panel figure:
  - merit history of all designs overlaid (semilogy)
  - histogram of final RMS spots

Run from repo root:
    C:\\Users\\bwyan\\.venvs\\lens-opt\\Scripts\\python.exe lens_opt\\examples\\cooke_multistart.py
"""
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

from lens_opt.optim.pack import pack
from lens_opt.optim.lm import lm_optimize_batched

from lens_opt.examples.cooke_triplet import (
    cooke_triplet_system, cooke_triplet_template, cooke_triplet_rays,
    per_field_residuals, N_RAYS_TOTAL,
)


B = 100                 # number of designs to run in parallel
PERTURB_FRAC = 0.20     # 20% randn perturbation: large enough that some
                        # designs land in different basins or fail to
                        # converge — makes the demo visually interesting.
                        # Drop to 0.05 to see all 100 finding the same answer.
SEED = 0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    system = cooke_triplet_system(device=device)
    template = cooke_triplet_template(system)
    rays = cooke_triplet_rays(device=device)
    v_published = pack(template).detach().clone()

    # B random perturbations around the published prescription.
    gen = torch.Generator(device=device).manual_seed(SEED)
    perturb = 1.0 + PERTURB_FRAC * torch.randn(B, v_published.numel(),
                                               generator=gen,
                                               dtype=v_published.dtype,
                                               device=device)
    v0_batch = v_published.unsqueeze(0) * perturb

    def residual(v):
        return per_field_residuals(v, template, rays)

    # Warmup so first-run autotune does not pollute the timing.
    _ = lm_optimize_batched(v0_batch[:2], residual, max_iter=3)
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"\nRunning batched LM:  B={B}  variables={v_published.numel()}  "
          f"residuals={N_RAYS_TOTAL}")
    t0 = time.perf_counter()
    result = lm_optimize_batched(v0_batch, residual, max_iter=80, tol=1e-7)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    print(f"Wall-clock: {wall*1000:.1f} ms  "
          f"({wall*1000/B:.2f} ms per design effective)")

    # Per-design RMS in um
    rms_um = torch.sqrt(2.0 * result.merit / N_RAYS_TOTAL) * 1e3
    rms_um_np = rms_um.detach().cpu().numpy()

    # Summary
    reasons = result.exit_reasons
    n_converged = sum(1 for r in reasons if r == "converged_gradient")
    n_maxiter   = sum(1 for r in reasons if r == "max_iter")
    n_stuck     = sum(1 for r in reasons if r == "stuck")

    print()
    print("=" * 64)
    print("  Summary")
    print("=" * 64)
    print(f"  converged_gradient : {n_converged}/{B}")
    print(f"  max_iter           : {n_maxiter}/{B}")
    print(f"  stuck              : {n_stuck}/{B}")
    print()
    print(f"  Final RMS spot (um):")
    print(f"    min     : {rms_um_np.min():.3f}")
    print(f"    median  : {np.median(rms_um_np):.3f}")
    print(f"    mean    : {rms_um_np.mean():.3f}")
    print(f"    max     : {rms_um_np.max():.3f}")
    print(f"    p90     : {np.percentile(rms_um_np, 90):.3f}")

    best_i = int(rms_um.argmin())
    print(f"\n  Best design (#{best_i}): RMS = {rms_um_np[best_i]:.3f} um, "
          f"{result.n_iter[best_i]} iters")

    # ────────────────────────────────────────────────────────────────────
    # Figure: merit history overlay + final-RMS histogram
    # ────────────────────────────────────────────────────────────────────
    merit_hist = result.merit_history.detach().cpu().numpy()  # [iter+1, B]

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    ax = axs[0]
    iters = np.arange(merit_hist.shape[0])
    # Plot 100 traces in muted color so the cluster is visible.
    ax.semilogy(iters, merit_hist, color="royalblue", alpha=0.15, linewidth=0.8)
    # Highlight the best design.
    ax.semilogy(iters, merit_hist[:, best_i], color="crimson", linewidth=2.0,
                label=f"best (#{best_i})")
    ax.semilogy(iters, np.median(merit_hist, axis=1), color="black",
                linewidth=1.5, linestyle="--", label="median")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("merit  0.5 ||r||²")
    ax.set_title(f"Merit history, {B} parallel designs")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    ax = axs[1]
    ax.hist(rms_um_np, bins=30, color="steelblue", edgecolor="black",
            alpha=0.8)
    ax.axvline(rms_um_np[best_i], color="crimson", linewidth=1.5,
               label=f"best = {rms_um_np[best_i]:.2f} µm")
    ax.axvline(np.median(rms_um_np), color="black", linestyle="--",
               linewidth=1.2, label=f"median = {np.median(rms_um_np):.2f} µm")
    ax.set_xlabel("final RMS spot (μm)")
    ax.set_ylabel("count")
    ax.set_title(f"Final RMS distribution, {B} starts at ±{int(PERTURB_FRAC*100)}%")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"M5  Cooke triplet × {B} parallel LM   "
        f"device={device}   "
        f"wall {wall*1000:.0f} ms ({wall*1000/B:.1f} ms/design effective)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = os.path.join(_HERE, "cooke_multistart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
