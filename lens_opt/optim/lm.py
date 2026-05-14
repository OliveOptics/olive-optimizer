"""Levenberg-Marquardt optimizer for the differentiable ray tracer.

The user-facing entry point is `lm_optimize`. It calls a residual function
`r(v) -> Tensor[m]` repeatedly, builds the Jacobian via `torch.func.jacfwd`,
and adjusts the design vector `v` to minimize 0.5 * ||r||^2.

Algorithm: Nielsen-damped Levenberg-Marquardt.

  At each iteration, solve the augmented linear least-squares problem
      [    J    ]         [ -r ]
      [ sqrt(lD) ] * dv ~=  [  0 ]
  via torch.linalg.lstsq. D = diag(J.T J) is the Marquardt scaling.

  The step is accepted iff the actual reduction in 0.5*||r||^2 is positive.
  Lambda is then updated by the Nielsen rule
      accept: lambda *= max(1/3, 1 - (2*rho - 1)**3),  nu = 2
      reject: lambda *= nu,                            nu *= 2

The augmented-lstsq formulation is numerically cleaner than forming
J.T J explicitly: it avoids squaring the condition number of J.
"""
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
from torch import Tensor
from torch.func import jacfwd, vmap


@dataclass(frozen=True)
class LMResult:
    """Outcome of an `lm_optimize` run.

    v          : final design vector
    merit      : final 0.5 * ||r||^2
    history    : list of (iter, merit, lambda, grad_inf_norm) per accepted step,
                 starting with iter=0 at v0
    exit_reason: 'converged_gradient' | 'max_iter' | 'stuck'
    n_iter     : number of accepted iterations performed
    """
    v: Tensor
    merit: float
    history: List[Tuple[int, float, float, float]]
    exit_reason: str
    n_iter: int


def _solve_augmented(J: Tensor, r: Tensor, lam: float, D: Tensor) -> Tensor:
    """Solve [J; sqrt(lam D)] dv = [-r; 0] in least squares.

    Args:
        J:   [m, n] Jacobian.
        r:   [m]    residual vector.
        lam: scalar damping.
        D:   [n]    diag(J.T J).

    Returns:
        dv:  [n]    proposed step.
    """
    m, n = J.shape
    sqrt_lD = torch.sqrt(lam * D)              # [n]
    top = J
    bot = torch.diag(sqrt_lD)                  # [n, n]
    A = torch.cat([top, bot], dim=0)           # [m+n, n]
    b = torch.cat([-r, torch.zeros(n, dtype=r.dtype, device=r.device)], dim=0)
    sol = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
    return sol


def lm_optimize(
    v0: Tensor,
    residual_fn: Callable[[Tensor], Tensor],
    max_iter: int = 100,
    tol: float = 1e-8,
    lambda_init: float = 1e-3,
    lambda_max: float = 1e10,
) -> LMResult:
    """Minimize 0.5 * ||residual_fn(v)||^2 by Nielsen-damped LM.

    Args:
        v0:          [n] initial design vector. Must be float (no grad needed).
        residual_fn: v -> r(v), where r has shape [m]. Must be differentiable
                     through torch.func.jacfwd.
        max_iter:    maximum number of OUTER iterations (accepted steps + final
                     bail). The inner step-search loop is bounded by lambda_max.
        tol:         convergence tolerance on ||J.T r||_inf (the gradient).
        lambda_init: initial damping. 1e-3 is the standard default.
        lambda_max:  if damping exceeds this without an accepted step, return
                     exit_reason='stuck'.

    Returns:
        LMResult.
    """
    device = v0.device
    dtype = v0.dtype

    v = v0.detach().clone()
    lam = float(lambda_init)
    nu = 2.0

    r = residual_fn(v).detach()
    phi = 0.5 * float((r * r).sum())

    history: List[Tuple[int, float, float, float]] = []

    # Record the starting point. Compute grad-norm at v0 for logging.
    J0 = jacfwd(residual_fn)(v).detach()
    g0_inf = float((J0.T @ r).abs().max())
    history.append((0, phi, lam, g0_inf))

    if g0_inf < tol:
        return LMResult(v=v, merit=phi, history=history,
                        exit_reason="converged_gradient", n_iter=0)

    n_accepted = 0

    for it in range(1, max_iter + 1):
        # Fresh Jacobian at current v
        J = jacfwd(residual_fn)(v).detach()
        g = J.T @ r                                # [n]
        g_inf = float(g.abs().max())

        if g_inf < tol:
            return LMResult(v=v, merit=phi, history=history,
                            exit_reason="converged_gradient", n_iter=n_accepted)

        # Marquardt scaling: diag(J.T J), floored so a column with zero
        # sensitivity does not give 0/0.
        D = (J * J).sum(dim=0)                     # [n]
        D = torch.clamp(D, min=torch.finfo(dtype).eps)

        accepted = False
        while lam < lambda_max:
            dv = _solve_augmented(J, r, lam, D)
            v_trial = v + dv
            r_trial = residual_fn(v_trial).detach()
            phi_trial = 0.5 * float((r_trial * r_trial).sum())

            # Predicted reduction from the quadratic model.
            # rho = (phi - phi_trial) / (0.5 * dv.T (lam*D*dv - g))
            pred = 0.5 * float((dv * (lam * D * dv - g)).sum())
            actual = phi - phi_trial

            if pred > 0.0 and actual > 0.0:
                rho = actual / pred
                # Nielsen accept-update
                v = v_trial
                r = r_trial
                phi = phi_trial
                lam = lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0
                n_accepted += 1
                history.append((it, phi, lam, g_inf))
                accepted = True
                break
            else:
                # Reject: increase damping, try again.
                lam = lam * nu
                nu = nu * 2.0

        if not accepted:
            return LMResult(v=v, merit=phi, history=history,
                            exit_reason="stuck", n_iter=n_accepted)

    return LMResult(v=v, merit=phi, history=history,
                    exit_reason="max_iter", n_iter=n_accepted)


# ──────────────────────────────────────────────────────────────────────────
# Batched LM — many designs in parallel
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LMBatchedResult:
    """Outcome of `lm_optimize_batched`.

    v             : [B, P] final designs
    merit         : [B]    final 0.5 * ||r||^2 per design
    exit_reasons  : list of length B; per-design exit ('converged_gradient',
                    'max_iter', 'stuck')
    n_iter        : [B] int   accepted-step count per design
    merit_history : [n_outer + 1, B] merit at each outer iteration. Frozen
                    designs keep their last merit. NaN never appears.
    lambda_history: [n_outer + 1, B] lambda per design per outer iter.
    """
    v: Tensor
    merit: Tensor
    exit_reasons: List[str]
    n_iter: Tensor
    merit_history: Tensor
    lambda_history: Tensor


# Exit codes used internally. Kept small/contiguous for tensor representation.
_EXIT_CONVERGED = 0
_EXIT_MAX_ITER = 1
_EXIT_STUCK = 2
_EXIT_REASON_MAP = {
    _EXIT_CONVERGED: "converged_gradient",
    _EXIT_MAX_ITER: "max_iter",
    _EXIT_STUCK: "stuck",
}


def lm_optimize_batched(
    v0: Tensor,
    residual_fn: Callable[[Tensor], Tensor],
    max_iter: int = 100,
    tol: float = 1e-7,
    lambda_init: float = 1e-3,
    lambda_max: float = 1e10,
) -> LMBatchedResult:
    """Run B independent LM optimizations in parallel.

    Args:
        v0:          [B, P] initial designs.
        residual_fn: SINGLE-design function v[P] -> r[M]. Must be
                     vmap- and jacfwd-compatible. The batched version is
                     built internally via `torch.vmap`.
        max_iter:    maximum number of OUTER iterations. Each outer iter
                     performs one step attempt per active design (may be
                     accepted or rejected).
        tol:         per-design gradient convergence tolerance on
                     ||J^T r||_inf.
        lambda_init: initial damping (shared across designs).
        lambda_max:  per-design ceiling. Exceeding it marks the design
                     'stuck' and freezes its state.

    Semantics:
        - Each outer iteration solves the per-design augmented system once.
        - Per-design accept/reject via Nielsen rule. Rejected designs do
          not advance v; their lambda grows for the next attempt.
        - Frozen designs (converged or stuck) keep their last v, r, phi,
          lambda; the global iteration still recomputes J for them, which
          is wasted work but harmless. The loop exits once every design
          is frozen, or max_iter is reached.

    Returns:
        LMBatchedResult.
    """
    if v0.dim() != 2:
        raise ValueError(f"v0 must be [B, P]; got shape {tuple(v0.shape)}")
    B, P = v0.shape
    device = v0.device
    dtype = v0.dtype
    eps = torch.finfo(dtype).eps

    # Lift the single-design residual + Jacobian to operate on [B, *].
    batched_residual = vmap(residual_fn)
    batched_jacfwd = vmap(jacfwd(residual_fn))

    v = v0.detach().clone()
    r = batched_residual(v).detach()                       # [B, M]
    M = r.shape[-1]
    phi = 0.5 * (r * r).sum(dim=-1)                        # [B]

    lam = torch.full((B,), float(lambda_init), dtype=dtype, device=device)
    nu = torch.full((B,), 2.0, dtype=dtype, device=device)

    active = torch.ones(B, dtype=torch.bool, device=device)
    exit_codes = torch.full((B,), -1, dtype=torch.int64, device=device)
    n_iter = torch.zeros(B, dtype=torch.int64, device=device)

    merit_hist = [phi.detach().clone()]
    lam_hist = [lam.detach().clone()]

    for outer in range(max_iter):
        if not bool(active.any()):
            break

        J = batched_jacfwd(v).detach()                     # [B, M, P]
        # g = J^T r per design  → einsum 'bmp,bm->bp'
        g = torch.einsum("bmp,bm->bp", J, r)               # [B, P]
        g_inf = g.abs().amax(dim=-1)                       # [B]

        # Mark newly converged BEFORE attempting a step.
        newly_converged = active & (g_inf < tol)
        exit_codes = torch.where(newly_converged,
                                 torch.full_like(exit_codes, _EXIT_CONVERGED),
                                 exit_codes)
        active = active & ~newly_converged
        if not bool(active.any()):
            merit_hist.append(phi.detach().clone())
            lam_hist.append(lam.detach().clone())
            break

        # Marquardt scaling per design.
        D = (J * J).sum(dim=-2).clamp(min=eps)             # [B, P]

        # Augmented-system lstsq, batched.
        # Top block: J [B, M, P]. Bottom block: diag(sqrt(lam*D)) [B, P, P].
        sqrt_lD = torch.sqrt(lam.unsqueeze(-1) * D)        # [B, P]
        bot = torch.diag_embed(sqrt_lD)                    # [B, P, P]
        A = torch.cat([J, bot], dim=-2)                    # [B, M+P, P]
        rhs = torch.cat(
            [-r, torch.zeros(B, P, dtype=dtype, device=device)],
            dim=-1,
        ).unsqueeze(-1)                                    # [B, M+P, 1]
        dv = torch.linalg.lstsq(A, rhs).solution.squeeze(-1)   # [B, P]

        v_trial = v + dv
        r_trial = batched_residual(v_trial).detach()
        phi_trial = 0.5 * (r_trial * r_trial).sum(dim=-1)  # [B]

        pred = 0.5 * (dv * (lam.unsqueeze(-1) * D * dv - g)).sum(dim=-1)  # [B]
        actual = phi - phi_trial                            # [B]

        accept = active & (pred > 0) & (actual > 0)
        reject = active & ~accept

        # Update v, r, phi only where accepted.
        v = torch.where(accept.unsqueeze(-1), v_trial, v)
        r = torch.where(accept.unsqueeze(-1), r_trial, r)
        phi = torch.where(accept, phi_trial, phi)

        # Nielsen lambda updates.
        # rho is meaningful only where pred > 0; guard the division.
        safe_pred = torch.where(pred > 0, pred, torch.ones_like(pred))
        rho = actual / safe_pred
        shrink = (1.0 - (2.0 * rho - 1.0) ** 3).clamp(min=1.0 / 3.0)
        lam = torch.where(accept, lam * shrink, lam)
        lam = torch.where(reject, lam * nu, lam)
        nu = torch.where(accept, torch.full_like(nu, 2.0), nu)
        nu = torch.where(reject, nu * 2.0, nu)

        n_iter = torch.where(accept, n_iter + 1, n_iter)

        # Designs whose lambda blew past lambda_max are 'stuck'.
        newly_stuck = reject & (lam > lambda_max)
        exit_codes = torch.where(newly_stuck,
                                 torch.full_like(exit_codes, _EXIT_STUCK),
                                 exit_codes)
        active = active & ~newly_stuck

        merit_hist.append(phi.detach().clone())
        lam_hist.append(lam.detach().clone())

    # Anything still active hit max_iter.
    exit_codes = torch.where(active,
                             torch.full_like(exit_codes, _EXIT_MAX_ITER),
                             exit_codes)
    exit_reasons = [_EXIT_REASON_MAP[int(c)] for c in exit_codes.tolist()]

    merit_history = torch.stack(merit_hist, dim=0)         # [iter+1, B]
    lambda_history = torch.stack(lam_hist, dim=0)

    return LMBatchedResult(
        v=v,
        merit=phi,
        exit_reasons=exit_reasons,
        n_iter=n_iter,
        merit_history=merit_history,
        lambda_history=lambda_history,
    )
