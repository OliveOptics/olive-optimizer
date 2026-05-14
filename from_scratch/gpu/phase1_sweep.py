"""
Phase 1: Coarse GPU sweep (paraxial + Seidel) -- broad exploration.

Outputs sweep results to a JSON file that phase2_refine.py can read.

Usage: python phase1_sweep.py [--out sweep_results.json]
"""

import sys
import os
import itertools
import time
import math
import json
import argparse
import numpy as np
from datetime import datetime

from numba import cuda, float64, float32, int32

from host import make_air_gaps
from common import (
    N_ELEM, K_SURF, N_GAPS, N_AIR, N_XVARS,
    X_DG, X_DA, X_ND, DEG2RAD, RAD2DEG,
    d_ynu_trace, d_sag, d_find_chief, d_build, d_ca, d_vignetting,
)


# ══════════════════════════════════════════════════════════════════════
#  SWEEP KERNEL
# ══════════════════════════════════════════════════════════════════════

@cuda.jit
def sweep_kernel(phi_in, sigma_in, nd_in, dair_in,
                 efl_target, f_number, field_deg, stop_idx,
                 r_min, e_min, t_min, out, reject):
    """
    reject codes: 0=pending, 1=radius_too_small, 2=zero_efl_first,
    3=neg_efl_first, 4=efl_no_converge, 5=zero_efl_rescale,
    6=neg_efl_rescale, 7=air_clearance, 8=neg_margin,
    9=chief_ray_fail, 99=passed
    """
    tid = cuda.grid(1)
    if tid >= phi_in.shape[0]:
        return
    out[tid, 0] = 0.0
    reject[tid] = 0

    c1  = cuda.local.array(N_ELEM, float32)
    c2  = cuda.local.array(N_ELEM, float32)
    dg  = cuda.local.array(N_ELEM, float32)
    da  = cuda.local.array(N_AIR,  float32)
    nd  = cuda.local.array(N_ELEM, float32)
    sc  = cuda.local.array(K_SURF, float32)
    sn  = cuda.local.array(K_SURF, float32)
    snp = cuda.local.array(K_SURF, float32)
    gd  = cuda.local.array(N_GAPS, float32)
    gn  = cuda.local.array(N_GAPS, float32)
    h1  = cuda.local.array(K_SURF, float32)
    n1  = cuda.local.array(K_SURF, float32)
    hm  = cuda.local.array(K_SURF, float32)
    nm  = cuda.local.array(K_SURF, float32)
    hc  = cuda.local.array(K_SURF, float32)
    nc  = cuda.local.array(K_SURF, float32)
    sd  = cuda.local.array(K_SURF, float32)

    for i in range(N_ELEM):
        nd[i] = nd_in[tid, i]
        dg[i] = t_min
    for i in range(N_AIR):
        da[i] = dair_in[tid, i]
    for i in range(N_ELEM):
        dn = nd[i] - 1.0
        c1[i] = phi_in[tid, i] * (1.0 + sigma_in[tid, i]) / (2.0 * dn)
        c2[i] = phi_in[tid, i] * (sigma_in[tid, i] - 1.0) / (2.0 * dn)
    for i in range(N_ELEM):
        cv = c1[i]
        if abs(cv) > 1e-12 and 1.0 / abs(cv) < r_min:
            reject[tid] = 1  # radius_too_small
            return
        cv = c2[i]
        if abs(cv) > 1e-12 and 1.0 / abs(cv) < r_min:
            reject[tid] = 1  # radius_too_small
            return

    d_build(c1, c2, dg, da, nd, sc, sn, snp, gd, gn)
    d_ynu_trace(sc, sn, snp, gd, gn, 1.0, 0.0, h1, n1)
    if abs(n1[K_SURF - 1]) < 1e-12:
        reject[tid] = 2  # zero_efl_first
        return
    efl = -1.0 / n1[K_SURF - 1]
    if efl < 0.0:
        reject[tid] = 3  # neg_efl_first
        return
    sa = efl / (2.0 * f_number)
    d_ynu_trace(sc, sn, snp, gd, gn, sa, 0.0, hm, nm)
    nu0_c = math.tan(field_deg * DEG2RAD)
    h0_c, ok = d_find_chief(sc, sn, snp, gd, gn, stop_idx,
                            nu0_c, h1, n1)
    if not ok:
        reject[tid] = 9  # chief_ray_fail
        return
    d_ynu_trace(sc, sn, snp, gd, gn, h0_c, nu0_c, hc, nc)

    for e in range(N_ELEM):
        i_f, i_b = 2 * e, 2 * e + 1
        hx = abs(hm[i_f]) + abs(hc[i_f])
        v  = abs(hm[i_b]) + abs(hc[i_b])
        if v > hx:
            hx = v
        hx *= 1.15
        dg[e] = d_sag(c1[e], hx) - d_sag(c2[e], hx) + e_min
        if dg[e] < t_min:
            dg[e] = t_min

    efl_err = 1.0
    rej_rescale = 0
    for _ in range(3):
        d_build(c1, c2, dg, da, nd, sc, sn, snp, gd, gn)
        d_ynu_trace(sc, sn, snp, gd, gn, 1.0, 0.0, h1, n1)
        if abs(n1[K_SURF - 1]) < 1e-12:
            reject[tid] = 5  # zero_efl_rescale
            return
        efl = -1.0 / n1[K_SURF - 1]
        if efl < 0.0:
            reject[tid] = 6  # neg_efl_rescale
            return
        efl_err = abs(efl - efl_target) / efl_target
        if efl_err < 0.01:
            break
        s = efl / efl_target
        for i in range(N_ELEM):
            c1[i] *= s
            c2[i] *= s
    if efl_err > 0.05:
        reject[tid] = 4  # efl_no_converge
        return

    sa = efl / (2.0 * f_number)
    d_build(c1, c2, dg, da, nd, sc, sn, snp, gd, gn)
    d_ynu_trace(sc, sn, snp, gd, gn, sa, 0.0, hm, nm)
    h0_c, ok = d_find_chief(sc, sn, snp, gd, gn, stop_idx,
                            nu0_c, h1, n1)
    if not ok:
        reject[tid] = 9  # chief_ray_fail
        return
    d_ynu_trace(sc, sn, snp, gd, gn, h0_c, nu0_c, hc, nc)

    for i in range(N_AIR):
        ib  = 2 * i + 1
        if2 = 2 * (i + 1)
        hg = abs(hm[ib]) + abs(hc[ib])
        v  = abs(hm[if2]) + abs(hc[if2])
        if v > hg:
            hg = v
        hg *= 1.15
        clr = da[i] - d_sag(c2[i], hg) + d_sag(c1[i + 1], hg)
        if clr < 0.3:
            reject[tid] = 7  # air_clearance
            return

    bfl = -hm[K_SURF-1] / nm[K_SURF-1] if abs(nm[K_SURF-1]) > 1e-12 else 0.0
    ttl = bfl
    for i in range(N_GAPS):
        ttl += gd[i]
    cra = math.atan(nc[K_SURF - 1]) * RAD2DEG

    min_margin = 1e6
    max_glass  = 0.0
    for e in range(N_ELEM):
        ca_e = d_ca(c1[e], c2[e], dg[e], e_min)
        sd[2 * e]     = ca_e
        sd[2 * e + 1] = ca_e
        hx = abs(hm[2 * e]) + abs(hc[2 * e])
        v  = abs(hm[2 * e + 1]) + abs(hc[2 * e + 1])
        if v > hx:
            hx = v
        m = ca_e - hx
        if m < min_margin:
            min_margin = m
        if dg[e] > max_glass:
            max_glass = dg[e]
    if min_margin < 0.0:
        reject[tid] = 8  # neg_margin
        return

    nu0_ref = math.tan(0.1 * DEG2RAD)
    h0_ref, ok_r = d_find_chief(sc, sn, snp, gd, gn, stop_idx,
                                nu0_ref, h1, n1)
    if not ok_r:
        ri = 0.0
    else:
        d_ynu_trace(sc, sn, snp, gd, gn, h0_ref, nu0_ref, h1, n1)
        vf = d_vignetting(hm, hc, sd)
        vr = d_vignetting(hm, h1, sd)
        cf = math.cos(field_deg * DEG2RAD)
        cr = math.cos(0.1 * DEG2RAD)
        c4f = cf * cf * cf * cf
        c4r = cr * cr * cr * cr
        if vr < 1e-15:
            ri = 0.0
        else:
            ri = (c4f * vf) / (c4r * vr)

    S1 = 0.0; S2 = 0.0; S3 = 0.0; S4 = 0.0; S5 = 0.0
    H_inv = -nu0_c * hm[0]
    npm = 0.0
    npc = nu0_c
    for i in range(K_SURF):
        cs  = sc[i]; ns = sn[i]; nps = snp[i]
        phi_s = (nps - ns) * cs
        A  = npm + ns * hm[i] * cs
        Ab = npc + ns * hc[i] * cs
        ub = npm / ns
        ua = nm[i] / nps
        dun = ua / nps - ub / ns
        si   = -A * A * hm[i] * dun
        sii  = -A * Ab * hm[i] * dun
        siii = -Ab * Ab * hm[i] * dun
        siv  = H_inv * H_inv * phi_s / (ns * nps)
        if abs(A) > 1e-12:
            sv = (siii + siv) * (Ab / A)
        else:
            sv = 0.0
        S1 += si; S2 += sii; S3 += siii; S4 += siv; S5 += sv
        npm = nm[i]; npc = nc[i]

    srss = math.sqrt(S1*S1 + S2*S2 + S3*S3 + S4*S4 + S5*S5)
    reject[tid] = 99  # passed
    out[tid, 0] = 1.0
    out[tid, 1] = bfl
    out[tid, 2] = ttl
    out[tid, 3] = cra
    out[tid, 4] = ri
    out[tid, 5] = min_margin
    out[tid, 6] = max_glass
    out[tid, 7] = srss


# ══════════════════════════════════════════════════════════════════════
#  HOST: PHASE 1
# ══════════════════════════════════════════════════════════════════════

def phase1_sweep(efl_target, f_number, field_angle_deg, N, stop_idx,
                 bfl_min, ttl_max, cra_max, ri_min,
                 n_power_samples, n_sigma_samples, sigma_max,
                 r_min, e_min, t_min,
                 glass_combos, total_airs, styles):
    """Run coarse GPU sweep, return top feasible as result dicts.

    Discrete glass combos (crown/flint pairings) and discrete air gap
    styles (uniform, retrofocus, etc.) combined with continuous power
    and bending sampling.
    """
    print("=" * 80)
    print("PHASE 1 -- COARSE SWEEP")
    print("=" * 80)

    rng = np.random.default_rng(42)
    phi_total = 1.0 / efl_target
    n_per_power = 1 + n_sigma_samples

    # Build all discrete combos: signs x glass x air
    t0 = time.perf_counter()
    combos = []
    for combo in itertools.product([+1, -1], repeat=N):
        signs = list(combo)
        if sum(1 for s in signs if s > 0) == 0:
            continue
        for n_d_neg, n_d_pos in glass_combos:
            n_d_list = [n_d_pos if s > 0 else n_d_neg for s in signs]
            for ta in total_airs:
                for style in styles:
                    d_air = make_air_gaps(signs, ta, style)
                    combos.append((signs, n_d_list, d_air))

    chunks_phi, chunks_sigma, chunks_nd, chunks_dair, chunks_signs = \
        [], [], [], [], []
    rejected_power = 0

    for signs, n_d_list, d_air in combos:
        signs_a = np.array(signs, dtype=np.float32)
        nd_a    = np.array(n_d_list, dtype=np.float32)
        dair_a  = np.array(d_air, dtype=np.float32)
        n_pos = int((signs_a > 0).sum())
        n_neg = N - n_pos

        # ── Power sampling (Dirichlet) ──
        if n_neg == 0:
            w = rng.dirichlet(np.ones(N), size=n_power_samples).astype(np.float32)
            phi_batch = w * np.float32(phi_total)
        else:
            neg_strength = rng.uniform(0.1, 0.8, size=n_power_samples).astype(np.float32)
            phi_neg_total = neg_strength * np.float32(phi_total)
            phi_pos_total = (1.0 + neg_strength) * np.float32(phi_total)
            w_pos = rng.dirichlet(np.ones(n_pos), size=n_power_samples).astype(np.float32)
            w_neg = rng.dirichlet(np.ones(n_neg), size=n_power_samples).astype(np.float32)
            phi_batch = np.empty((n_power_samples, N), dtype=np.float32)
            ip, in_ = 0, 0
            for j in range(N):
                if signs[j] > 0:
                    phi_batch[:, j] = w_pos[:, ip] * phi_pos_total
                    ip += 1
                else:
                    phi_batch[:, j] = -w_neg[:, in_] * phi_neg_total
                    in_ += 1

        # Power ratio filter
        abs_phis = np.abs(phi_batch)
        mean_phi = abs_phis.mean(axis=1)
        valid = mean_phi > 1e-15
        ratios = np.full(n_power_samples, np.inf, dtype=np.float32)
        ratios[valid] = abs_phis[valid].max(axis=1) / mean_phi[valid]
        power_ok = ratios <= 3.0
        rejected_power += int((~power_ok).sum())
        phi_valid = phi_batch[power_ok]
        n_valid = phi_valid.shape[0]
        if n_valid == 0:
            continue

        n_jobs_combo = n_valid * n_per_power

        # ── Expand phi with sigma variants ──
        phi_exp = np.repeat(phi_valid, n_per_power, axis=0)
        sigma_exp = rng.uniform(-sigma_max, sigma_max,
                                size=(n_jobs_combo, N)).astype(np.float32)
        equi_idx = np.arange(n_valid) * n_per_power
        sigma_exp[equi_idx] = 0.0

        chunks_phi.append(phi_exp)
        chunks_sigma.append(sigma_exp)
        chunks_nd.append(np.tile(nd_a, (n_jobs_combo, 1)))
        chunks_dair.append(np.tile(dair_a, (n_jobs_combo, 1)))
        chunks_signs.append(np.tile(signs_a, (n_jobs_combo, 1)))

    phi_arr   = np.concatenate(chunks_phi)
    sigma_arr = np.concatenate(chunks_sigma)
    nd_arr    = np.concatenate(chunks_nd)
    dair_arr  = np.concatenate(chunks_dair)
    signs_arr = np.concatenate(chunks_signs)
    n_jobs    = phi_arr.shape[0]
    out_arr32 = np.zeros((n_jobs, 8), dtype=np.float32)
    rej_arr   = np.zeros(n_jobs, dtype=np.int32)
    t_gen = time.perf_counter() - t0
    print(f"  Jobs: {n_jobs}  ({rejected_power} rejected by power ratio)")
    print(f"  Job generation: {t_gen:.2f}s")

    # GPU sweep
    d_phi   = cuda.to_device(phi_arr)
    d_sigma = cuda.to_device(sigma_arr)
    d_nd    = cuda.to_device(nd_arr)
    d_dair  = cuda.to_device(dair_arr)
    d_out   = cuda.to_device(out_arr32)
    d_rej   = cuda.to_device(rej_arr)
    TPB = 256
    blocks = (n_jobs + TPB - 1) // TPB

    t0 = time.perf_counter()
    sweep_kernel[blocks, TPB](d_phi, d_sigma, d_nd, d_dair,
                              efl_target, f_number, field_angle_deg,
                              stop_idx, r_min, e_min, t_min, d_out, d_rej)
    cuda.synchronize()
    t_kern = time.perf_counter() - t0
    out_arr = d_out.copy_to_host().astype(np.float64)
    rej_arr = d_rej.copy_to_host()
    print(f"  Kernel: {t_kern:.3f}s  ({n_jobs/t_kern/1e6:.2f} M evals/s)")

    # ── Rejection diagnostics ──
    REJECT_NAMES = {
        0: 'unknown/not_reached',
        1: 'radius_too_small',
        2: 'zero_efl_first_trace',
        3: 'negative_efl_first_trace',
        4: 'efl_no_converge',
        5: 'zero_efl_after_rescale',
        6: 'negative_efl_after_rescale',
        7: 'air_gap_clearance',
        8: 'negative_margin',
        9: 'chief_ray_fail',
        99: 'passed',
    }
    print()
    print("  KERNEL REJECTION BREAKDOWN:")
    for code in sorted(REJECT_NAMES.keys()):
        count = int((rej_arr == code).sum())
        if count > 0:
            pct = 100.0 * count / n_jobs
            print(f"    {REJECT_NAMES[code]:30s}  {count:>10d}  ({pct:5.1f}%)")

    # ── Per-pattern breakdown ──
    # Build pattern string for each job
    pattern_strs = []
    for i in range(n_jobs):
        pattern_strs.append(
            ''.join('+' if signs_arr[i, j] > 0 else '-' for j in range(N)))
    pattern_strs = np.array(pattern_strs)
    unique_patterns = sorted(set(pattern_strs))

    print()
    print(f"  PER-PATTERN BREAKDOWN ({len(unique_patterns)} patterns):")
    print(f"    {'Pattern':>8s}  {'Total':>8s}  {'Passed':>8s}  "
          f"{'%Pass':>6s}  {'TopReject':>30s}  {'%TopRej':>7s}")
    for pat in unique_patterns:
        mask = pattern_strs == pat
        total = int(mask.sum())
        codes = rej_arr[mask]
        n_pass = int((codes == 99).sum())
        pct_pass = 100.0 * n_pass / total if total > 0 else 0.0
        # Find top rejection reason (excluding passed)
        fail_codes = codes[codes != 99]
        if len(fail_codes) > 0:
            vals, counts = np.unique(fail_codes, return_counts=True)
            top_idx = counts.argmax()
            top_code = int(vals[top_idx])
            top_count = int(counts[top_idx])
            top_pct = 100.0 * top_count / total
            top_name = REJECT_NAMES.get(top_code, f'code_{top_code}')
        else:
            top_name = '-'
            top_pct = 0.0
        print(f"    {pat:>8s}  {total:>8d}  {n_pass:>8d}  "
              f"{pct_pass:>5.1f}%  {top_name:>30s}  {top_pct:>6.1f}%")

    valid_mask = out_arr[:, 0] > 0.5

    # ── Soft feasibility filter with per-reason tracking ──
    t0 = time.perf_counter()
    SOFT = 0.10
    bfl_ok = out_arr[:, 1] >= bfl_min * (1.0 - SOFT)
    ttl_ok = out_arr[:, 2] <= ttl_max * (1.0 + SOFT)
    cra_ok = np.abs(out_arr[:, 3]) <= cra_max * (1.0 + SOFT)
    ri_ok  = out_arr[:, 4] >= ri_min * (1.0 - SOFT)
    feasible_mask = valid_mask & bfl_ok & ttl_ok & cra_ok & ri_ok
    feasible_idx = np.where(feasible_mask)[0]

    n_valid = int(valid_mask.sum())
    n_fail_bfl = int((valid_mask & ~bfl_ok).sum())
    n_fail_ttl = int((valid_mask & ~ttl_ok).sum())
    n_fail_cra = int((valid_mask & ~cra_ok).sum())
    n_fail_ri  = int((valid_mask & ~ri_ok).sum())
    print()
    print(f"  SOFT FILTER (on {n_valid} valid designs):")
    print(f"    BFL < {bfl_min*(1-SOFT):.1f}mm:   {n_fail_bfl:>8d} rejected")
    print(f"    TTL > {ttl_max*(1+SOFT):.1f}mm:   {n_fail_ttl:>8d} rejected")
    print(f"    |CRA| > {cra_max*(1+SOFT):.1f}deg: {n_fail_cra:>8d} rejected")
    print(f"    RI < {ri_min*(1-SOFT):.3f}:       {n_fail_ri:>8d} rejected")
    print(f"    Feasible:             {len(feasible_idx):>8d}")
    print(f"    Filter: {(time.perf_counter() - t0)*1000:.1f}ms")

    if len(feasible_idx) == 0:
        return []

    # Score feasible designs
    t0 = time.perf_counter()
    FLOOR = 0.05
    scores = np.zeros(len(feasible_idx))
    for k, ji in enumerate(feasible_idx):
        o = out_arr[ji]
        abs_p = np.abs(phi_arr[ji])
        mean_p = abs_p.mean()
        phi_sys = 1.0 / efl_target

        m_bfl = (o[1] - bfl_min) / bfl_min
        m_ttl = (ttl_max - o[2]) / ttl_max
        m_cra = (cra_max - abs(o[3])) / cra_max
        m_ri  = (o[4] - ri_min) / (1.0 - ri_min)
        m_mrg = max(o[5], 0) / 5.0
        m_pwr = min(1.0, 1.0 / (abs_p.max() / phi_sys))
        m_afo = min(1.0, (abs_p.min() / mean_p) / 0.3) if mean_p > 1e-15 else 0
        m_bnd = min(1.0, np.std(np.abs(sigma_arr[ji])) / 0.5)
        m_gls = min(1.0, max(0.0, (6.0 - o[6]) / 4.0))
        m_sei = 1.0 / (1.0 + o[7])

        vals = [max(min(v, 1.0), FLOOR) for v in
                [m_bfl, m_ttl, m_cra, m_ri, m_mrg,
                 m_pwr, m_afo, m_bnd, m_gls]]
        m_sei = max(min(m_sei, 1.0), FLOOR)
        log_score = sum(math.log(v) for v in vals) + 3.0 * math.log(m_sei)
        scores[k] = math.exp(log_score / (len(vals) + 3.0))

    rank_order = (-scores).argsort()
    print(f"  Scoring: {(time.perf_counter() - t0)*1000:.1f}ms")

    # Build JSON directly from GPU arrays — no CPU re-eval needed
    n_top = min(2000, len(rank_order))
    designs = []
    for k in range(n_top):
        ki = rank_order[k]
        ji = feasible_idx[ki]
        o = out_arr[ji]
        s = signs_arr[ji]
        nd_list = nd_arr[ji]
        p = phi_arr[ji]
        sg = sigma_arr[ji]
        da = dair_arr[ji]

        # Reconstruct curvatures from phi/sigma/nd
        c1 = [float(p[i] * (1.0 + sg[i]) / (2.0 * (nd_list[i] - 1.0))) for i in range(N_ELEM)]
        c2 = [float(p[i] * (sg[i] - 1.0) / (2.0 * (nd_list[i] - 1.0))) for i in range(N_ELEM)]

        designs.append({
            'phi':    [float(v) for v in p],
            'sigma':  [float(v) for v in sg],
            'c1':     c1,
            'c2':     c2,
            'n_d':    [float(v) for v in nd_list],
            'd_air':  [float(v) for v in da],
            'signs':  [int(v) for v in s],
            'pattern': ''.join('+' if v > 0 else '-' for v in s),
            'score':  float(scores[ki]),
            'bfl':    float(o[1]),
            'ttl':    float(o[2]),
            'cra':    float(o[3]),
            'ri':     float(o[4]),
            'seidel': float(o[7]),
        })

    print(f"  Top {len(designs)} designs ranked")
    if designs:
        print(f"  Best: score={designs[0]['score']:.3f}  pattern={designs[0]['pattern']}")
    return designs


def save_sweep_results(designs, config, outpath):
    """Save sweep results to JSON for phase 2 consumption."""
    data = {
        'config': config,
        'n_designs': len(designs),
        'designs': designs,
    }
    with open(outpath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(designs)} designs to {outpath}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 1: GPU coarse sweep")
    parser.add_argument('--out', type=str, default=None,
                        help='Output JSON path (default: sweep_YYYYMMDD_HHMMSS.json)')
    args = parser.parse_args()

    # System parameters
    efl_target      = 16.0
    f_number        = 1.2
    field_angle_deg = 30.0
    N               = N_ELEM
    stop_idx        = N_ELEM + 1

    # Sweep constraints
    bfl_min  = 6.0
    ttl_max  = 40.0
    cra_max  = 13.0
    ri_min   = 0.45

    # Sweep sampling (reduced for diagnostic run)
    n_power_samples = 100
    n_sigma_samples = 3
    sigma_max       = 1.5
    r_min           = 3.0
    e_min           = 1.0
    t_min           = 1.0
    glass_combos    = [(1.5, 1.5), (1.5, 1.8), (1.5, 1.9),
                       (1.45, 1.9), (1.7, 1.5), (1.7, 1.8)]
    total_airs      = [12, 18, 24]
    styles          = ['uniform', 'retrofocus', 'front_heavy',
                       'back_heavy', 'middle']

    dev = cuda.get_current_device()
    gpu_name = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
    print(f"GPU: {gpu_name}")
    print()

    top_results = phase1_sweep(
        efl_target, f_number, field_angle_deg, N, stop_idx,
        bfl_min, ttl_max, cra_max, ri_min,
        n_power_samples, n_sigma_samples, sigma_max,
        r_min, e_min, t_min,
        glass_combos, total_airs, styles)

    if not top_results:
        print("No feasible designs found.")
        return

    # Save results
    config = {
        'efl_target': efl_target,
        'f_number': f_number,
        'field_angle_deg': field_angle_deg,
        'n_elements': N,
        'stop_idx': stop_idx,
        'bfl_min': bfl_min,
        'ttl_max': ttl_max,
        'cra_max': cra_max,
        'ri_min': ri_min,
    }
    if args.out:
        outpath = args.out
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outpath = os.path.join(os.path.dirname(__file__),
                               f"sweep_{timestamp}.json")
    save_sweep_results(top_results, config, outpath)


if __name__ == "__main__":
    main()
