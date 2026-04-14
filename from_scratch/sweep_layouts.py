"""
Sweep all +/- power sign patterns for N-element thin-lens systems.
Parallelized with multiprocessing.

Power distribution: Dirichlet random sampling within each sign group,
filtered by max_power_ratio to prevent any single lens from dominating.
"""

import sys
import os
import itertools
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parse_zmx import ynu_trace, find_chief_ray_initial, seidel_coefficients
from generate import _sag, _thin_lens_ca, _thin_lens_ri


def _sample_power(phi_total, signs, max_power_ratio, rng):
    """
    Sample a random power distribution using Dirichlet within each sign group.

    Returns phi_list or None if rejected by max_power_ratio.

    max_power_ratio : max(|phi_i|) / mean(|phi_i|) must be <= this.
    """
    n_pos = sum(1 for s in signs if s > 0)
    n_neg = sum(1 for s in signs if s < 0)
    N = len(signs)

    if n_neg == 0:
        # All positive: Dirichlet over all elements
        w = rng.dirichlet(np.ones(N))
        phi_list = [w[i] * phi_total for i in range(N)]
    else:
        # Sample neg_strength uniformly in [0.1, 0.8]
        neg_strength = rng.uniform(0.1, 0.8)
        phi_neg_total = neg_strength * phi_total   # positive number
        phi_pos_total = (1.0 + neg_strength) * phi_total

        # Dirichlet within each group
        w_pos = rng.dirichlet(np.ones(n_pos))
        w_neg = rng.dirichlet(np.ones(n_neg))

        phi_list = []
        ip, in_ = 0, 0
        for s in signs:
            if s > 0:
                phi_list.append(w_pos[ip] * phi_pos_total)
                ip += 1
            else:
                phi_list.append(-w_neg[in_] * phi_neg_total)
                in_ += 1

    # Check max power ratio
    abs_phis = [abs(p) for p in phi_list]
    mean_phi = sum(abs_phis) / N
    if mean_phi < 1e-15:
        return None
    ratio = max(abs_phis) / mean_phi
    if ratio > max_power_ratio:
        return None

    return phi_list


def _build(curvatures, d_glass_list, d_air_list, n_d_list):
    n_air = 1.0
    n_elements = len(d_glass_list)
    surfaces, gaps = [], []
    ai = 0
    for i in range(n_elements):
        nd = n_d_list[i]
        surfaces.append((curvatures[2*i], n_air, nd))
        surfaces.append((curvatures[2*i+1], nd, n_air))
        gaps.append((d_glass_list[i], nd))
        if i < n_elements - 1:
            gaps.append((d_air_list[ai], n_air))
            ai += 1
    return surfaces, gaps


def evaluate(args):
    """Evaluate one layout. Takes a single tuple for Pool.map."""
    (signs, n_d_list, efl_target, f_number, field_angle_deg,
     phi_list, sigma_list, d_air_list, stop_idx,
     r_min, e_min, t_min) = args

    n_elements = len(signs)
    curvatures = []
    c1_list = []
    c2_list = []
    for i in range(n_elements):
        dn = n_d_list[i] - 1.0
        sigma = sigma_list[i]
        phi = phi_list[i]
        c1 = phi * (1 + sigma) / (2.0 * dn)
        c2 = phi * (sigma - 1) / (2.0 * dn)
        c1_list.append(c1)
        c2_list.append(c2)
        curvatures.append(c1)
        curvatures.append(c2)

    # R_min early rejection
    for i in range(n_elements):
        for c in (c1_list[i], c2_list[i]):
            if abs(c) > 1e-12 and 1.0 / abs(c) < r_min:
                return None

    d_glass_list = [t_min] * n_elements

    surfaces, gaps = _build(curvatures, d_glass_list, d_air_list, n_d_list)

    try:
        h_u, nu_u = ynu_trace(surfaces, gaps, 1.0, 0.0)
        if abs(nu_u[-1]) < 1e-12:
            return None
        efl = -1.0 / nu_u[-1]
        if efl < 0:
            return None
        sa = efl / (2 * f_number)
        h_m, nu_m = ynu_trace(surfaces, gaps, sa, 0.0)
        h0_c, nu0_c = find_chief_ray_initial(surfaces, gaps, stop_idx,
                                              field_angle_deg)
        h_c, nu_c = ynu_trace(surfaces, gaps, h0_c, nu0_c)
    except Exception:
        return None

    # Size glass for beam
    for elem in range(n_elements):
        i_f, i_b = 2*elem, 2*elem+1
        h_max = max(abs(h_m[i_f]) + abs(h_c[i_f]),
                    abs(h_m[i_b]) + abs(h_c[i_b])) * 1.15
        sag_diff = _sag(c1_list[elem], h_max) - _sag(c2_list[elem], h_max)
        d_glass_list[elem] = max(t_min, sag_diff + e_min)

    # Rebuild
    surfaces, gaps = _build(curvatures, d_glass_list, d_air_list, n_d_list)
    try:
        h_u, nu_u = ynu_trace(surfaces, gaps, 1.0, 0.0)
        efl = -1.0 / nu_u[-1]
        if efl < 0: return None
        sa = efl / (2 * f_number)
        h_m, nu_m = ynu_trace(surfaces, gaps, sa, 0.0)
        h0_c, nu0_c = find_chief_ray_initial(surfaces, gaps, stop_idx,
                                              field_angle_deg)
        h_c, nu_c = ynu_trace(surfaces, gaps, h0_c, nu0_c)
    except Exception:
        return None

    # Check air gap clearance — lenses must not overlap
    min_air_clearance = 0.3  # mm
    for i in range(n_elements - 1):
        # Beam height at back of elem i and front of elem i+1
        i_back = 2*i + 1
        i_front = 2*(i+1)
        h_at_gap = max(abs(h_m[i_back]) + abs(h_c[i_back]),
                       abs(h_m[i_front]) + abs(h_c[i_front])) * 1.15
        # Clearance = air_gap - sag_back(elem_i) + sag_front(elem_{i+1})
        sag_back = _sag(c2_list[i], h_at_gap)
        sag_front = _sag(c1_list[i+1], h_at_gap)
        clearance = d_air_list[i] - sag_back + sag_front
        if clearance < min_air_clearance:
            return None

    bfl = -h_m[-1] / nu_m[-1] if abs(nu_m[-1]) > 1e-12 else 0
    ttl = sum(g[0] for g in gaps) + bfl
    cra = np.degrees(np.arctan(nu_c[-1]))

    ca_list, margins = [], []
    for elem in range(n_elements):
        ca = _thin_lens_ca(c1_list[elem], c2_list[elem],
                           d_glass_list[elem], n_d_list[elem], e_min)
        ca_list.append(ca)
        i_f, i_b = 2*elem, 2*elem+1
        h_max = max(abs(h_m[i_f]) + abs(h_c[i_f]),
                    abs(h_m[i_b]) + abs(h_c[i_b]))
        margins.append(ca - h_max)

    try:
        ri, _, _ = _thin_lens_ri(surfaces, gaps, stop_idx, f_number,
                                  field_angle_deg, ca_list)
    except Exception:
        ri = 0.0

    # Seidel aberration coefficients
    try:
        nu0_m = 0.0  # marginal ray starts parallel
        S_I, S_II, S_III, S_IV, S_V = seidel_coefficients(
            surfaces, h_m, nu_m, h_c, nu_c, nu0_m, nu0_c)
        seidel = {
            'S_I': float(S_I.sum()),
            'S_II': float(S_II.sum()),
            'S_III': float(S_III.sum()),
            'S_IV': float(S_IV.sum()),
            'S_V': float(S_V.sum()),
        }
        seidel_rss = float(np.sqrt(sum(v**2 for v in seidel.values())))
    except Exception:
        seidel = {'S_I': 99., 'S_II': 99., 'S_III': 99.,
                  'S_IV': 99., 'S_V': 99.}
        seidel_rss = 99.

    abs_phis = [abs(p) for p in phi_list]
    mean_phi = sum(abs_phis) / len(abs_phis)
    max_phi_ratio = max(abs_phis) / mean_phi
    phi_total = 1.0 / efl
    # How much the strongest element dominates vs system power
    max_phi_sys_ratio = max(abs_phis) / phi_total
    # Fraction of weakest element vs mean (low = wasted element)
    min_phi_ratio = min(abs_phis) / mean_phi
    # Bending diversity: std of |sigma|
    sigma_std = float(np.std([abs(s) for s in sigma_list]))
    # Max glass thickness
    max_glass = max(d_glass_list)

    return {
        'pattern': ''.join('+' if s > 0 else '-' for s in signs),
        'signs': signs,
        'efl': efl, 'bfl': bfl, 'ttl': ttl,
        'cra': cra, 'ri': ri,
        'min_margin': min(margins), 'margins': margins,
        'ca_list': ca_list, 'd_air': list(d_air_list),
        'd_glass': d_glass_list, 'phi_list': phi_list,
        'sigma_list': sigma_list,
        'c1_list': c1_list, 'c2_list': c2_list,
        'n_d_list': list(n_d_list),
        'max_phi_ratio': max_phi_ratio,
        'max_phi_sys_ratio': max_phi_sys_ratio,
        'min_phi_ratio': min_phi_ratio,
        'sigma_std': sigma_std,
        'max_glass': max_glass,
        'seidel': seidel,
        'seidel_rss': seidel_rss,
        'n_pos': sum(1 for s in signs if s > 0),
        'n_neg': sum(1 for s in signs if s < 0),
    }


def make_air_gaps(signs, total_air, style):
    N = len(signs)
    n_air = N - 1
    if n_air == 0:
        return []

    if style == 'uniform':
        return [total_air / n_air] * n_air
    elif style == 'retrofocus':
        g = [1.5] * n_air
        for i in range(n_air):
            if signs[i] < 0 and signs[i+1] > 0:
                g[i] = 6.0
                break
        s = total_air / sum(g)
        return [x * s for x in g]
    elif style == 'front_heavy':
        w = [n_air - i for i in range(n_air)]
        s = total_air / sum(w)
        return [x * s for x in w]
    elif style == 'back_heavy':
        w = [i + 1 for i in range(n_air)]
        s = total_air / sum(w)
        return [x * s for x in w]
    elif style == 'middle':
        w = [1.0] * n_air
        w[n_air // 2] = 4.0
        s = total_air / sum(w)
        return [x * s for x in w]


def _sample_sigma(n_elements, sigma_max, rng):
    """Sample a random shape-factor vector. Uniform in [-sigma_max, +sigma_max]."""
    return rng.uniform(-sigma_max, sigma_max, size=n_elements).tolist()


def main():
    efl_target = 16.0
    f_number = 1.2
    field_angle_deg = 30.0
    N = 7
    stop_idx = 7  # between L4 and L5

    bfl_min = 6.0
    ttl_max = 40.0
    cra_max = 13.0
    ri_min = 0.45

    max_power_ratio = 3.0   # max(|phi|) / mean(|phi|) limit
    n_power_samples = 10    # random power draws per (pattern, spacing)
    n_sigma_samples = 3     # random sigma draws per power sample
    sigma_max = 1.5         # |sigma| <= sigma_max
    r_min = 3.0             # mm, minimum radius of curvature
    e_min = 0.5             # mm, minimum edge thickness
    t_min = 1.0             # mm, minimum center thickness

    # Glass combos: (n_d_neg, n_d_pos) per sign group
    glass_combos = [
        (1.5, 1.5),
        (1.5, 1.8),
        (1.7, 1.5),
        (1.7, 1.8),
    ]

    print("=" * 100)
    print(f"N={N}  EFL={efl_target}, f/{f_number}, +/-{field_angle_deg} deg  "
          f"stop=surf {stop_idx}")
    print(f"Specs: BFL>{bfl_min}, TTL<{ttl_max}, |CRA|<{cra_max}, RI>{ri_min}")
    print(f"Power sampling: {n_power_samples} Dirichlet draws/combo, "
          f"max_power_ratio<={max_power_ratio}")
    print(f"Bending: sigma_max={sigma_max}, {n_sigma_samples} samples + "
          f"equi-convex baseline")
    print(f"Glass combos: {glass_combos}")
    print(f"Physical: R_min={r_min}mm, e_min={e_min}mm, t_min={t_min}mm")
    print("=" * 100)

    total_airs = [12, 18, 24]
    styles = ['uniform', 'retrofocus', 'front_heavy', 'back_heavy', 'middle']

    rng = np.random.default_rng(42)
    phi_total = 1.0 / efl_target

    # Build all evaluation jobs
    jobs = []
    rejected_power = 0
    rejected_sigma = 0
    for combo in itertools.product([+1, -1], repeat=N):
        signs = list(combo)
        if sum(1 for s in signs if s > 0) == 0:
            continue

        for n_d_neg, n_d_pos in glass_combos:
            n_d_list = [n_d_pos if s > 0 else n_d_neg for s in signs]

            for ta in total_airs:
                for style in styles:
                    d_air = make_air_gaps(signs, ta, style)

                    for _ in range(n_power_samples):
                        phi_list = _sample_power(phi_total, signs,
                                                 max_power_ratio, rng)
                        if phi_list is None:
                            rejected_power += 1
                            continue

                        # Always include equi-convex baseline (sigma=0)
                        jobs.append((signs, n_d_list, efl_target, f_number,
                                     field_angle_deg, phi_list, [0.0] * N,
                                     d_air, stop_idx, r_min, e_min, t_min))

                        # Random sigma samples
                        for _ in range(n_sigma_samples):
                            sigma_list = _sample_sigma(N, sigma_max, rng)

                            # R_min pre-check (use per-element n_d)
                            skip = False
                            for k in range(N):
                                dn_k = n_d_list[k] - 1.0
                                c1 = phi_list[k] * (1 + sigma_list[k]) / (2.0 * dn_k)
                                c2 = phi_list[k] * (sigma_list[k] - 1) / (2.0 * dn_k)
                                for c in (c1, c2):
                                    if abs(c) > 1e-12 and 1.0 / abs(c) < r_min:
                                        skip = True
                                        break
                                if skip:
                                    break
                            if skip:
                                rejected_sigma += 1
                                continue

                            jobs.append((signs, n_d_list, efl_target, f_number,
                                         field_angle_deg, phi_list, sigma_list,
                                         d_air, stop_idx, r_min, e_min, t_min))

    print(f"Total jobs: {len(jobs)} ({rejected_power} rejected by power ratio, "
          f"{rejected_sigma} rejected by R_min), using {cpu_count()} cores")

    # Run in parallel
    with Pool() as pool:
        raw = pool.map(evaluate, jobs)

    # Filter valid results
    results = []
    for info in raw:
        if info is not None:
            info['meets_bfl'] = info['bfl'] >= bfl_min
            info['meets_ttl'] = info['ttl'] <= ttl_max
            info['meets_cra'] = abs(info['cra']) <= cra_max
            info['meets_ri'] = info['ri'] >= ri_min
            info['all_met'] = (info['meets_bfl'] and info['meets_ttl']
                               and info['meets_cra'] and info['meets_ri'])
            results.append(info)

    print(f"Valid results: {len(results)}")

    feasible = [r for r in results if r['all_met']]
    print(f"Feasible: {len(feasible)}")
    print()

    if feasible:
        def score(r):
            # Constraint headroom (how comfortably specs are met)
            m_bfl = (r['bfl'] - bfl_min) / 5.0
            m_ttl = (ttl_max - r['ttl']) / 10.0
            m_cra = (cra_max - abs(r['cra'])) / 10.0
            m_ri = (r['ri'] - ri_min) / 0.2
            m_mrg = max(r['min_margin'], 0) / 5.0
            # Power distribution quality
            m_pwr_sys = min(1.0, 1.0 / r['max_phi_sys_ratio'])
            m_afocal = min(1.0, r['min_phi_ratio'] / 0.3)
            # Bending diversity
            m_bend = min(1.0, r['sigma_std'] / 0.5)
            # Glass thickness
            m_glass = min(1.0, max(0.0, (6.0 - r['max_glass']) / 4.0))
            # Aberration quality (lower RSS = better)
            # RSS of ~0.5 is good, ~2.0 is poor
            m_seidel = min(1.0, max(0.0, 1.0 / (1.0 + r['seidel_rss'])))
            vals = [min(max(v, 0.0), 1.0) for v in
                    [m_bfl, m_ttl, m_cra, m_ri, m_mrg,
                     m_pwr_sys, m_afocal, m_bend, m_glass, m_seidel]]
            return np.prod(vals) ** (1.0/len(vals))

        feasible.sort(key=lambda r: -score(r))

        seen = set()
        unique = []
        for r in feasible:
            if r['pattern'] not in seen:
                seen.add(r['pattern'])
                unique.append(r)

        n_show = min(20, len(unique))
        print(f"TOP {n_show} FEASIBLE (unique patterns, best config each):")
        print(f"  {'#':>3s} {'Pattern':>8s} {'Score':>6s} {'RI':>6s} "
              f"{'BFL':>6s} {'TTL':>6s} {'CRA':>7s} "
              f"{'S_RSS':>7s} {'S_I':>7s} {'S_IV':>7s} "
              f"{'PhiSys':>7s} {'MaxGls':>7s}")

        for i, r in enumerate(unique[:n_show]):
            sc = score(r)
            print(f"  {i+1:3d} {r['pattern']:>8s} {sc:6.3f} {r['ri']:6.3f} "
                  f"{r['bfl']:6.1f} {r['ttl']:6.1f} {r['cra']:+7.1f} "
                  f"{r['seidel_rss']:7.3f} {r['seidel']['S_I']:+7.3f} "
                  f"{r['seidel']['S_IV']:+7.3f} "
                  f"{r['max_phi_sys_ratio']:7.2f} {r['max_glass']:7.1f}")

        print("\nDETAIL:")
        for i, r in enumerate(unique[:5]):
            sc = score(r)
            print(f"\n  #{i+1} [{r['pattern']}] score={sc:.3f}")
            print(f"    RI={r['ri']:.3f}  BFL={r['bfl']:.1f}  "
                  f"TTL={r['ttl']:.1f}  CRA={r['cra']:+.1f}")
            s = r['seidel']
            print(f"    Seidel: S_I={s['S_I']:+.3f} S_II={s['S_II']:+.3f} "
                  f"S_III={s['S_III']:+.3f} S_IV={s['S_IV']:+.3f} "
                  f"S_V={s['S_V']:+.3f}  RSS={r['seidel_rss']:.3f}")
            print(f"    Air: {['%.1f' % d for d in r['d_air']]}")
            print(f"    Glass: {['%.1f' % d for d in r['d_glass']]}")
            for j in range(N):
                sg = '+' if r['signs'][j] > 0 else '-'
                phi = r['phi_list'][j]
                sig = r['sigma_list'][j]
                f_e = 1/phi if abs(phi) > 1e-12 else float('inf')
                print(f"    L{j+1}({sg}) phi={phi:+.5f} sig={sig:+.3f} "
                      f"f={f_e:+.1f}  CA={r['ca_list'][j]:.1f}  "
                      f"mrg={r['margins'][j]:+.1f}")

        # Write top 20 to file
        outpath = os.path.join(os.path.dirname(__file__), 'sweep_top20.md')
        with open(outpath, 'w', encoding='utf-8') as f:
            f.write(f"# Sweep Results - Top {n_show} Feasible Layouts\n\n")
            f.write(f"N={N}  EFL={efl_target}mm  f/{f_number}  "
                    f"+/-{field_angle_deg} deg  stop=surf {stop_idx}\n\n")
            f.write(f"Constraints: BFL>{bfl_min}  TTL<{ttl_max}  "
                    f"|CRA|<{cra_max}  RI>{ri_min}\n\n")
            f.write(f"Sampling: {n_power_samples} power x "
                    f"(1 equi-convex + {n_sigma_samples} sigma) per combo  "
                    f"sigma_max={sigma_max}  R_min={r_min}mm  "
                    f"e_min={e_min}mm  t_min={t_min}mm\n\n")
            f.write(f"Total jobs: {len(jobs)}  "
                    f"Valid: {len(results)}  Feasible: {len(feasible)}\n\n")

            # Summary table
            f.write("## Summary\n\n")
            f.write(f"| # | Pattern | Score |   RI |  BFL |  TTL |   CRA | "
                    f"MrgMin | S_RSS | S_I | S_IV | PhiSys | MaxGls |\n")
            f.write(f"|--:|--------:|------:|-----:|-----:|-----:|------:|"
                    f"------:|------:|----:|-----:|-------:|-------:|\n")
            for i, r in enumerate(unique[:n_show]):
                sc = score(r)
                f.write(f"| {i+1} | {r['pattern']} | {sc:.3f} | "
                        f"{r['ri']:.3f} | {r['bfl']:.1f} | {r['ttl']:.1f} | "
                        f"{r['cra']:+.1f} | {r['min_margin']:+.1f} | "
                        f"{r['seidel_rss']:.3f} | {r['seidel']['S_I']:+.3f} | "
                        f"{r['seidel']['S_IV']:+.3f} | "
                        f"{r['max_phi_sys_ratio']:.2f} | "
                        f"{r['max_glass']:.1f} |\n")

            # Detail for all 20
            f.write("\n## Detail\n")
            for i, r in enumerate(unique[:n_show]):
                sc = score(r)
                f.write(f"\n### #{i+1} [{r['pattern']}] score={sc:.3f}\n\n")
                f.write(f"- RI={r['ri']:.3f}  BFL={r['bfl']:.1f}  "
                        f"TTL={r['ttl']:.1f}  CRA={r['cra']:+.1f}\n")
                sd = r['seidel']
                f.write(f"- Seidel: S_I={sd['S_I']:+.3f}  "
                        f"S_II={sd['S_II']:+.3f}  S_III={sd['S_III']:+.3f}  "
                        f"S_IV={sd['S_IV']:+.3f}  S_V={sd['S_V']:+.3f}  "
                        f"RSS={r['seidel_rss']:.3f}\n")
                f.write(f"- Air gaps: {['%.1f' % d for d in r['d_air']]}\n")
                f.write(f"- Glass: {['%.1f' % d for d in r['d_glass']]}\n\n")
                f.write(f"| Elem | Sign | phi | sigma | f (mm) | "
                        f"CA | margin |\n")
                f.write(f"|-----:|-----:|----:|------:|-------:|"
                        f"--:|-------:|\n")
                for j in range(N):
                    s = '+' if r['signs'][j] > 0 else '-'
                    phi = r['phi_list'][j]
                    sig = r['sigma_list'][j]
                    f_e = 1/phi if abs(phi) > 1e-12 else float('inf')
                    f.write(f"| L{j+1} | {s} | {phi:+.5f} | {sig:+.3f} | "
                            f"{f_e:+.1f} | {r['ca_list'][j]:.1f} | "
                            f"{r['margins'][j]:+.1f} |\n")

                # Surface-level prescription (all 14 surfaces)
                nd_str = ', '.join(f'{nd:.2f}' for nd in r['n_d_list'])
                f.write(f"\nSurface prescription (n_d=[{nd_str}]):\n\n")
                f.write(f"| Surf | c (1/mm) | R (mm) | Type | "
                        f"Gap (mm) | Medium |\n")
                f.write(f"|-----:|---------:|-------:|-----:|"
                        f"--------:|-------:|\n")
                for j in range(N):
                    c1 = r['c1_list'][j]
                    c2 = r['c2_list'][j]
                    r1 = 1/c1 if abs(c1) > 1e-12 else float('inf')
                    r2 = 1/c2 if abs(c2) > 1e-12 else float('inf')
                    si = 2 * j
                    f.write(f"| S{si+1} | {c1:+.6f} | {r1:+.2f} | "
                            f"L{j+1} front | {r['d_glass'][j]:.2f} | "
                            f"glass |\n")
                    if j < N - 1:
                        f.write(f"| S{si+2} | {c2:+.6f} | {r2:+.2f} | "
                                f"L{j+1} back | {r['d_air'][j]:.2f} | "
                                f"air |\n")
                    else:
                        f.write(f"| S{si+2} | {c2:+.6f} | {r2:+.2f} | "
                                f"L{j+1} back | - | image |\n")

        print(f"\nWrote top {n_show} details to {outpath}")
    else:
        def near_score(r):
            fails = 0
            if not r['meets_bfl']: fails += abs(bfl_min - r['bfl'])
            if not r['meets_ttl']: fails += abs(r['ttl'] - ttl_max)
            if not r['meets_cra']: fails += abs(abs(r['cra']) - cra_max)
            if not r['meets_ri']: fails += 10 * abs(ri_min - r['ri'])
            return fails

        results.sort(key=near_score)

        n_show = min(30, len(results))
        print(f"TOP {n_show} NEAR-FEASIBLE:")
        print(f"  {'#':>3s} {'Pattern':>8s} {'RI':>6s} {'BFL':>6s} "
              f"{'TTL':>6s} {'CRA':>7s} {'MrgMin':>7s} {'PwrRat':>7s} "
              f"{'Fails':s}")

        for i, r in enumerate(results[:n_show]):
            flag = ""
            if not r['meets_bfl']: flag += f" BFL({r['bfl']:.1f})"
            if not r['meets_ttl']: flag += f" TTL({r['ttl']:.1f})"
            if not r['meets_cra']: flag += f" CRA({r['cra']:+.1f})"
            if not r['meets_ri']: flag += f" RI({r['ri']:.2f})"
            print(f"  {i+1:3d} {r['pattern']:>8s} {r['ri']:6.3f} "
                  f"{r['bfl']:6.1f} {r['ttl']:6.1f} {r['cra']:+7.1f} "
                  f"{r['min_margin']:+7.1f} {r['max_phi_ratio']:7.2f}"
                  f"  {flag}")

        print("\nDETAIL (top 5):")
        for i, r in enumerate(results[:5]):
            print(f"\n  #{i+1} [{r['pattern']}]")
            print(f"    RI={r['ri']:.3f}  BFL={r['bfl']:.1f}  "
                  f"TTL={r['ttl']:.1f}  CRA={r['cra']:+.1f}")
            print(f"    Air: {['%.1f' % d for d in r['d_air']]}")
            print(f"    Glass: {['%.1f' % d for d in r['d_glass']]}")
            for j in range(N):
                s = '+' if r['signs'][j] > 0 else '-'
                phi = r['phi_list'][j]
                sig = r['sigma_list'][j]
                f_e = 1/phi if abs(phi) > 1e-12 else float('inf')
                print(f"    L{j+1}({s}) phi={phi:+.5f} sig={sig:+.3f} "
                      f"f={f_e:+.1f}  CA={r['ca_list'][j]:.1f}  "
                      f"mrg={r['margins'][j]:+.1f}")


if __name__ == "__main__":
    main()
