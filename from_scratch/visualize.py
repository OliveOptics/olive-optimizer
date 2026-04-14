"""
Visualize a lens system: draw elements and trace rays.
Works with sweep results or optimizer JSON output.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parse_zmx import ynu_trace, find_chief_ray_initial


def sag(c, h):
    """Spherical sag at height h for curvature c."""
    if abs(c) < 1e-15:
        return 0.0
    arg = 1 - c**2 * h**2
    if arg <= 0:
        return c * h**2
    return c * h**2 / (1 + np.sqrt(arg))


def draw_surface(ax, z_vertex, c, h_max, n_pts=50, **kwargs):
    """Draw a spherical surface arc from -h_max to +h_max at z_vertex."""
    h_arr = np.linspace(-h_max, h_max, n_pts)
    z_arr = np.array([z_vertex + sag(c, abs(h)) for h in h_arr])
    ax.plot(z_arr, h_arr, **kwargs)
    return z_arr, h_arr


def draw_system(ax, surfaces, gaps, beam_sd=None, stop_idx=None):
    """Draw all lens elements and the stop.

    beam_sd: semi-diameter per element based on actual beam footprint.
    """
    K = len(surfaces)
    n_elements = K // 2

    # Compute z positions of each surface (vertex positions)
    z = np.zeros(K)
    z[0] = 0.0
    for i in range(1, K):
        z[i] = z[i-1] + gaps[i-1][0]

    # Semi-diameters for drawing
    if beam_sd is not None:
        sd = beam_sd
    else:
        sd = [8.0] * n_elements

    # Draw each element
    colors = plt.cm.Set2(np.linspace(0, 1, n_elements))
    for elem in range(n_elements):
        i_f = 2 * elem
        i_b = 2 * elem + 1
        c_f = surfaces[i_f][0]
        c_b = surfaces[i_b][0]
        h = sd[elem]

        n_pts = 50
        h_arr = np.linspace(-h, h, n_pts)
        z_f = np.array([z[i_f] + sag(c_f, abs(hv)) for hv in h_arr])
        z_b = np.array([z[i_b] + sag(c_b, abs(hv)) for hv in h_arr])

        # Fill the lens
        z_outline = np.concatenate([z_f, z_b[::-1]])
        h_outline = np.concatenate([h_arr, h_arr[::-1]])
        ax.fill(z_outline, h_outline, alpha=0.25, color=colors[elem])
        ax.plot(z_f, h_arr, color=colors[elem], linewidth=1.5)
        ax.plot(z_b, h_arr, color=colors[elem], linewidth=1.5)

        # Top and bottom edges
        ax.plot([z_f[0], z_b[-1]], [-h, -h], color=colors[elem], linewidth=1.0)
        ax.plot([z_f[-1], z_b[0]], [h, h], color=colors[elem], linewidth=1.0)

        # Label
        z_mid = (z[i_f] + z[i_b]) / 2
        ax.text(z_mid, h + 0.5, f'L{elem+1}', ha='center', va='bottom',
                fontsize=7, color=colors[elem])

    # Draw stop
    if stop_idx is not None:
        z_stop = z[stop_idx] if stop_idx < K else z[-1]
        h_stop = max(sd) * 0.5
        ax.plot([z_stop, z_stop], [-h_stop, h_stop], 'k--', linewidth=1.0, alpha=0.5)
        ax.text(z_stop, -h_stop - 0.5, 'STOP', ha='center', va='top', fontsize=7)

    # Draw optical axis
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')

    return z


def trace_and_draw_ray(ax, surfaces, gaps, h0, nu0, z_positions, color='blue',
                       alpha=0.5, linewidth=0.8):
    """Trace a ray and draw it on the plot, with ray-surface intersections at sag."""
    try:
        h, nu = ynu_trace(surfaces, gaps, h0, nu0)
    except Exception:
        return

    K = len(surfaces)

    # Compute z of ray-surface intersection (vertex + sag at ray height)
    z_hit = np.zeros(K)
    for i in range(K):
        c = surfaces[i][0]
        z_hit[i] = z_positions[i] + sag(c, abs(h[i]))

    # Before first surface — short lead-in only
    z_start = z_hit[0] - 2.0
    n_before = surfaces[0][1]
    h_start = h[0] - (nu0 / n_before) * (z_hit[0] - z_start)
    ax.plot([z_start, z_hit[0]], [h_start, h[0]],
            color=color, alpha=alpha, linewidth=linewidth)

    # Between surfaces
    for i in range(K - 1):
        ax.plot([z_hit[i], z_hit[i+1]], [h[i], h[i+1]],
                color=color, alpha=alpha, linewidth=linewidth)

    # After last surface — extend toward image, capped
    if abs(nu[-1]) > 1e-12:
        bfl = -h[-1] / nu[-1]
        z_img = z_hit[-1] + bfl
        # Cap extension to reasonable range
        max_extend = 20.0
        if abs(z_img - z_hit[-1]) > max_extend:
            z_end = z_hit[-1] + max_extend * np.sign(bfl)
            n_after = surfaces[-1][2]
            h_end = h[-1] + nu[-1] / n_after * (z_end - z_hit[-1])
            ax.plot([z_hit[-1], z_end], [h[-1], h_end],
                    color=color, alpha=alpha, linewidth=linewidth)
        else:
            ax.plot([z_hit[-1], z_img], [h[-1], 0.0],
                    color=color, alpha=alpha, linewidth=linewidth)


def visualize_system(surfaces, gaps, stop_idx, f_number, field_angle_deg,
                     ca_list=None, title=None):
    """Main visualization function."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    # Trace marginal ray to determine beam footprint for drawing
    h_u, nu_u = ynu_trace(surfaces, gaps, 1.0, 0.0)
    efl = -1.0 / nu_u[-1]
    sa = efl / (2 * f_number)
    h_m, _ = ynu_trace(surfaces, gaps, sa, 0.0)

    # Include off-axis beam for sizing elements
    n_elements = len(surfaces) // 2
    try:
        h0_c, nu0_c = find_chief_ray_initial(surfaces, gaps, stop_idx,
                                              field_angle_deg)
        h_c, _ = ynu_trace(surfaces, gaps, h0_c, nu0_c)
    except Exception:
        h_c = np.zeros_like(h_m)

    # Beam semi-diameter per element: max of (marginal + chief) on either surface
    beam_sd = []
    for elem in range(n_elements):
        i_f, i_b = 2*elem, 2*elem+1
        h_max = max(abs(h_m[i_f]) + abs(h_c[i_f]),
                    abs(h_m[i_b]) + abs(h_c[i_b]))
        beam_sd.append(h_max * 1.15)

    # Draw elements sized to beam
    z_pos = draw_system(ax, surfaces, gaps, beam_sd, stop_idx)

    # On-axis ray fan (blue)
    for frac in [1.0, 0.7, 0.3]:
        h0 = sa * frac
        trace_and_draw_ray(ax, surfaces, gaps, h0, 0.0, z_pos,
                           color='blue', alpha=0.5)
        trace_and_draw_ray(ax, surfaces, gaps, -h0, 0.0, z_pos,
                           color='blue', alpha=0.5)

    # Full field ray fan (red)
    try:
        h0_c, nu0_c = find_chief_ray_initial(surfaces, gaps, stop_idx,
                                              field_angle_deg)
        trace_and_draw_ray(ax, surfaces, gaps, h0_c, nu0_c, z_pos,
                           color='red', alpha=0.6, linewidth=1.2)
        for frac in [1.0, 0.7, 0.3]:
            trace_and_draw_ray(ax, surfaces, gaps, h0_c + sa * frac, nu0_c,
                               z_pos, color='red', alpha=0.3)
            trace_and_draw_ray(ax, surfaces, gaps, h0_c - sa * frac, nu0_c,
                               z_pos, color='red', alpha=0.3)
    except Exception:
        pass

    # Half field ray fan (green)
    try:
        half_field = field_angle_deg / 2
        h0_h, nu0_h = find_chief_ray_initial(surfaces, gaps, stop_idx, half_field)
        trace_and_draw_ray(ax, surfaces, gaps, h0_h, nu0_h, z_pos,
                           color='green', alpha=0.5, linewidth=1.0)
        for frac in [1.0, 0.7, 0.3]:
            trace_and_draw_ray(ax, surfaces, gaps, h0_h + sa * frac, nu0_h,
                               z_pos, color='green', alpha=0.25)
            trace_and_draw_ray(ax, surfaces, gaps, h0_h - sa * frac, nu0_h,
                               z_pos, color='green', alpha=0.25)
    except Exception:
        pass

    # Set axis limits based on system extent
    z_min = z_pos[0] - 5
    z_max = z_pos[-1] + 20
    h_extent = max(beam_sd) * 1.3
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(-h_extent, h_extent)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('height (mm)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'EFL={efl:.1f}mm  f/{f_number}  field=+/-{field_angle_deg} deg')

    plt.tight_layout()
    return fig


def load_from_json(filepath):
    """Load system from optimizer JSON output."""
    with open(filepath) as f:
        data = json.load(f)

    n = data['n_elements']
    surfaces = []
    gaps = []
    ca_list = []

    for i, elem in enumerate(data['elements']):
        nd = elem['n_d']
        surfaces.append((elem['c_front'], 1.0, nd))
        surfaces.append((elem['c_back'], nd, 1.0))
        gaps.append((elem['d_glass'], nd))
        if i < n - 1:
            gaps.append((data['air_gaps'][i], 1.0))
        # Estimate CA from edge thickness
        from generate import _thin_lens_ca
        ca = _thin_lens_ca(elem['c_front'], elem['c_back'],
                           elem['d_glass'], nd)
        ca_list.append(ca)

    return surfaces, gaps, data['stop_idx'], data['f_number'], \
           data['field_angle_deg'], ca_list, data


def load_from_sweep(sweep_result):
    """Load system from sweep result dict."""
    from sweep_layouts import _build
    n = len(sweep_result['signs'])
    surfaces, gaps = _build(
        [c for pair in zip(sweep_result['c1_list'], sweep_result['c2_list'])
         for c in pair],
        sweep_result['d_glass'],
        sweep_result['d_air'],
        sweep_result['n_d_list'],
    )
    return surfaces, gaps, sweep_result.get('ca_list')


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        surfs, gaps, stop_idx, fno, field, ca_list, data = \
            load_from_json(sys.argv[1])
        r = data.get('result', {})
        title = (f"EFL={r.get('efl', 0):.1f}mm  f/{fno}  "
                 f"field=+/-{field} deg  "
                 f"BFL={r.get('bfl', 0):.1f}  TTL={r.get('ttl', 0):.1f}  "
                 f"RI={r.get('ri', 0):.3f}")
    else:
        # Default: load the latest optimizer result
        json_files = [f for f in os.listdir(os.path.dirname(__file__))
                      if f.startswith('optimized_') and f.endswith('.json')]
        if not json_files:
            print("No JSON files found. Pass a .json file as argument.")
            return
        filepath = os.path.join(os.path.dirname(__file__), sorted(json_files)[-1])
        print(f"Loading: {filepath}")
        surfs, gaps, stop_idx, fno, field, ca_list, data = \
            load_from_json(filepath)
        r = data.get('result', {})
        title = (f"EFL={r.get('efl', 0):.1f}mm  f/{fno}  "
                 f"field=+/-{field} deg  "
                 f"BFL={r.get('bfl', 0):.1f}  TTL={r.get('ttl', 0):.1f}  "
                 f"RI={r.get('ri', 0):.3f}")

    fig = visualize_system(surfs, gaps, stop_idx, fno, field,
                           ca_list=ca_list, title=title)

    outpath = os.path.join(os.path.dirname(__file__), 'lens_layout.png')
    fig.savefig(outpath, dpi=150)
    print(f"Saved: {outpath}")
    plt.show()


if __name__ == "__main__":
    main()
