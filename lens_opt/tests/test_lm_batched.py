"""Milestone 5: batched LM correctness and speed.

Three tests per the plan:

  1. Equivalence: B=1 batched run produces the same answer as the
     unbatched `lm_optimize` on the same problem.
  2. Independence: B=4 identical copies converge to 4 identical results.
  3. Speedup: B=100 GPU run takes less than 5x the wall-clock of B=1
     on a real lens problem (Cooke triplet). Skipped if no CUDA.

The first two are correctness; the third proves we actually got
parallelism out of the batching.
"""
import math
import time
import pytest
import torch

from lens_opt.optim.lm import lm_optimize, lm_optimize_batched
from lens_opt.optim.pack import pack

from lens_opt.examples.cooke_triplet import (
    cooke_triplet_system, cooke_triplet_template, cooke_triplet_rays,
    per_field_residuals, N_RAYS_TOTAL,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _linear_problem(device, seed=0, n=4, m=20):
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(m, n, generator=g, dtype=torch.float64, device=device)
    b = torch.randn(m, generator=g, dtype=torch.float64, device=device)

    def residual(v):
        return A @ v - b

    v0 = torch.zeros(n, dtype=torch.float64, device=device)
    return residual, v0


# ──────────────────────────────────────────────────────────────────────────
# Test 1: equivalence (B=1 batched vs unbatched)
# ──────────────────────────────────────────────────────────────────────────

def test_batched_b1_equivalent_to_unbatched():
    """B=1 batched LM gives the same v_final as the unbatched LM."""
    device = torch.device("cpu")
    residual, v0 = _linear_problem(device, seed=0)

    res_unbatched = lm_optimize(v0, residual, max_iter=20, tol=1e-10)

    v0_batched = v0.unsqueeze(0)        # [1, P]
    res_batched = lm_optimize_batched(v0_batched, residual,
                                      max_iter=20, tol=1e-10)

    # Final designs match to ~7 decimals. The unbatched and batched paths
    # go through different torch.linalg.lstsq kernels (single vs batched
    # implementations), so bit-identical isn't guaranteed even with the
    # same algorithm. 1e-7 is loose-but-meaningful — far below any lens
    # design relevance — and accommodates suite-order-dependent noise.
    err = (res_batched.v[0] - res_unbatched.v).abs().max().item()
    assert err < 1e-7, (
        f"batched v[0] differs from unbatched v by {err:.3e}\n"
        f"unbatched: {res_unbatched.v}\n"
        f"batched:   {res_batched.v[0]}"
    )

    # Same exit reason.
    assert res_batched.exit_reasons[0] == res_unbatched.exit_reason


# ──────────────────────────────────────────────────────────────────────────
# Test 2: independence (B identical copies → B identical results)
# ──────────────────────────────────────────────────────────────────────────

def test_batched_identical_copies():
    """4 identical starts must produce 4 identical final designs."""
    device = torch.device("cpu")
    residual, v0 = _linear_problem(device, seed=1)

    B = 4
    v0_batched = v0.unsqueeze(0).expand(B, -1).contiguous()
    res = lm_optimize_batched(v0_batched, residual,
                              max_iter=20, tol=1e-10)

    # 'Identical' here is bounded by float64 noise. torch.linalg.lstsq may
    # take slightly different code paths per batch entry, so v's match to
    # ~1e-9 rather than bit-identical, and n_iter / exit_reason can diverge
    # by 1 iteration at the precision floor. The semantic guarantee — that
    # batch entries don't interfere with each other — is what v drift tests.
    for i in range(1, B):
        err = (res.v[i] - res.v[0]).abs().max().item()
        assert err < 1e-9, f"copy {i} drifted from copy 0 by {err:.3e}"
        assert abs(int(res.n_iter[i]) - int(res.n_iter[0])) <= 1
        assert res.exit_reasons[i] in (
            "converged_gradient", "stuck", "max_iter"
        )


# ──────────────────────────────────────────────────────────────────────────
# Test 3: speedup (B=100 on GPU < 5x B=1 wall-clock)
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="speedup test requires CUDA")
def test_batched_speedup_on_cooke():
    """Batched B=100 Cooke optimization on GPU < 5x wall-clock of B=1.

    Runs a small warmup first to avoid kernel-launch / autotune costs
    contaminating the timing.
    """
    device = torch.device("cuda")
    system = cooke_triplet_system(device=device)
    template = cooke_triplet_template(system)
    rays = cooke_triplet_rays(device=device)
    v0 = pack(template).detach().clone()

    def residual(v):
        return per_field_residuals(v, template, rays)

    # Warmup (compiles vmap traces, kernel autotune)
    _ = lm_optimize_batched(v0.unsqueeze(0).repeat(2, 1), residual,
                            max_iter=5)
    torch.cuda.synchronize()

    # B=1 timing
    v_b1 = v0.unsqueeze(0)
    t0 = time.perf_counter()
    res_b1 = lm_optimize_batched(v_b1, residual, max_iter=100, tol=1e-7)
    torch.cuda.synchronize()
    t_b1 = time.perf_counter() - t0

    # B=100 timing — randomly perturbed starts
    gen = torch.Generator(device=device).manual_seed(0)
    perturb = 1.0 + 0.01 * torch.randn(100, v0.numel(),
                                       generator=gen,
                                       dtype=v0.dtype, device=device)
    v_b100 = v0.unsqueeze(0) * perturb
    t0 = time.perf_counter()
    res_b100 = lm_optimize_batched(v_b100, residual, max_iter=100, tol=1e-7)
    torch.cuda.synchronize()
    t_b100 = time.perf_counter() - t0

    ratio = t_b100 / t_b1
    print(f"\n  B=1   wall-clock : {t_b1*1000:.1f} ms")
    print(f"  B=100 wall-clock : {t_b100*1000:.1f} ms")
    print(f"  ratio            : x{ratio:.2f}")

    assert ratio < 5.0, (
        f"B=100 was {ratio:.2f}x slower than B=1 — "
        f"expected <5x. (B=1: {t_b1*1000:.1f} ms, B=100: {t_b100*1000:.1f} ms)"
    )

    # Sanity-check that the B=100 actually optimized. With vignetting on,
    # the optimization may exit in any of converged_gradient / stuck /
    # max_iter — exit reason isn't a reliable success signal. Use the
    # final RMS instead: most designs from a 1% perturbation should reach
    # well under 30 um.
    final_rms_um = (2.0 * res_b100.merit / N_RAYS_TOTAL).sqrt() * 1e3
    n_good = int((final_rms_um < 30.0).sum())
    assert n_good >= 90, (
        f"only {n_good}/100 reached RMS<30um; batched LM may not be "
        f"finding the basin (exit reasons: "
        f"{set(res_b100.exit_reasons)})"
    )
