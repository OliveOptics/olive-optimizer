"""
Compare our y-nu trace + Seidel results against Zemax OpticStudio via ZOS-API.

Usage:
  1. Standalone (launches OpticStudio): python compare_with_zemax.py
  2. Extension (attach to running OpticStudio): python compare_with_zemax.py --extension

Requires: zospy, numpy, Zemax OpticStudio installed.
"""

import argparse
import sys
import os
import tempfile
import re
import numpy as np
import zospy as zp

# Fix encoding issues on Windows with non-ASCII characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from parse_zmx import parse_zmx, build_system_from_zmx, ynu_trace, find_chief_ray_initial, seidel_coefficients


ZMX_FILE = "sc_dbga1_opt2.zmx"


# ── ZOS-API helpers ──────────────────────────────────────────────────────────

def connect_to_zemax(mode="standalone"):
    zos = zp.ZOS()
    oss = zos.connect(mode=mode)
    return zos, oss


def load_zmx_in_zemax(oss, filepath):
    abspath = os.path.abspath(filepath)
    oss.load(abspath)
    return oss


def get_seidel_from_zemax(oss):
    """Run Seidel analysis and return decoded text."""
    seidel_id = oss.ZOS.ZOSAPI.Analysis.AnalysisIDM.SeidelCoefficients
    analysis = zp.analyses.new_analysis(oss, seidel_id)
    analysis.ApplyAndWaitForCompletion()
    results = analysis.GetResults()

    tmp = os.path.join(tempfile.gettempdir(), "zemax_seidel.txt")
    results.GetTextFile(tmp)

    # Zemax writes UTF-16 with spaced characters
    try:
        with open(tmp, "r", encoding="utf-16") as f:
            raw = f.read()
    except (UnicodeError, UnicodeDecodeError):
        with open(tmp, "r", encoding="utf-8") as f:
            raw = f.read()

    analysis.Close()
    return raw


def decode_zemax_spaced(text):
    """
    Zemax UTF-16 files have spaces between every character.
    Remove extra spaces: 'S u r f' -> 'Surf', '0 . 0 7' -> '0.07'
    """
    lines = text.split("\n")
    decoded = []
    for line in lines:
        # Remove the space-between-chars pattern
        # Check if line has the spaced pattern (mostly alternating char-space)
        stripped = line.rstrip()
        if not stripped:
            decoded.append("")
            continue

        # Heuristic: if more than half the chars are spaces, it's spaced
        space_count = stripped.count(" ")
        if len(stripped) > 0 and space_count > len(stripped) * 0.3:
            # Remove every other space (the inserted ones)
            # Simple approach: remove spaces between non-space chars
            result = []
            i = 0
            while i < len(stripped):
                result.append(stripped[i])
                # If next char is space and char after that is not space, skip the space
                if (i + 1 < len(stripped) and stripped[i + 1] == " " and
                    i + 2 < len(stripped) and stripped[i + 2] != " "):
                    i += 2  # skip the inserted space
                else:
                    i += 1
            decoded.append("".join(result))
        else:
            decoded.append(stripped)

    return "\n".join(decoded)


def parse_seidel_text(raw_text):
    """
    Parse the Seidel Coefficients text output from OpticStudio.
    Handles the UTF-16 spaced format.
    """
    text = decode_zemax_spaced(raw_text)
    lines = text.split("\n")

    seidel_data = {
        "surfaces": [],
        "totals": {},
        "header_info": {},
        "decoded_text": text,
    }

    # Extract header info
    for line in lines:
        if "Wavelength" in line and ":" in line:
            m = re.search(r":\s+([\d.]+)", line)
            if m:
                seidel_data["header_info"]["wavelength"] = float(m.group(1))
        if "Chief Ray Slope, Object" in line:
            m = re.search(r":\s+([\d.+-]+)", line)
            if m:
                seidel_data["header_info"]["chief_slope_obj"] = float(m.group(1))
        if "Marginal Ray Slope, Image" in line:
            m = re.search(r":\s+([\d.+-]+)", line)
            if m:
                seidel_data["header_info"]["marginal_slope_img"] = float(m.group(1))
        if "Optical Invariant" in line:
            m = re.search(r":\s+([\d.+-]+)", line)
            if m:
                seidel_data["header_info"]["invariant"] = float(m.group(1))
        if "Petzval radius" in line:
            m = re.search(r":\s+([\d.+-]+)", line)
            if m:
                seidel_data["header_info"]["petzval_radius"] = float(m.group(1))

    # Find the first "Seidel Aberration Coefficients:" table (not "in Waves")
    in_seidel_table = False
    for line in lines:
        line = line.strip()

        # Start of table: "Surf   SPHA S1   COMA S2   ..."
        if line.startswith("Surf") and "SPHA" in line:
            in_seidel_table = True
            continue

        # End of table: blank line or next section
        if in_seidel_table and (not line or "Seidel" in line or "Transverse" in line or "Longitudinal" in line):
            if seidel_data["surfaces"]:  # we already got data
                break
            continue

        if not in_seidel_table:
            continue

        # Parse data row: "1\t0.070007\t-0.006304\t..."
        # or "STO\t-0.000000\t..." or "TOT\t0.017459\t..."
        parts = line.split()
        if len(parts) >= 6:
            try:
                label = parts[0]
                s1 = float(parts[1])
                s2 = float(parts[2])
                s3 = float(parts[3])
                s4 = float(parts[4])
                s5 = float(parts[5])

                if label in ("TOT", "Tot", "Total"):
                    seidel_data["totals"] = {
                        "S_I": s1, "S_II": s2, "S_III": s3,
                        "S_IV": s4, "S_V": s5,
                    }
                elif label in ("IMA", "Ima"):
                    pass  # skip image surface
                elif label in ("STO", "Sto"):
                    seidel_data["surfaces"].append({
                        "label": "STO", "num": None,
                        "S_I": s1, "S_II": s2, "S_III": s3,
                        "S_IV": s4, "S_V": s5,
                    })
                else:
                    surf_num = int(label)
                    seidel_data["surfaces"].append({
                        "label": label, "num": surf_num,
                        "S_I": s1, "S_II": s2, "S_III": s3,
                        "S_IV": s4, "S_V": s5,
                    })
            except (ValueError, IndexError):
                pass

    return seidel_data


def get_paraxial_ray_trace_from_zemax(oss):
    """Trace marginal and chief rays via SingleRayTrace."""
    from zospy.analyses.raysandspots import SingleRayTrace
    results = {}

    # Marginal ray: on-axis, edge of pupil
    marginal = SingleRayTrace(hx=0, hy=0, px=0, py=1, wavelength=1,
                              raytrace_type="DirectionCosines")
    results["marginal"] = marginal.run(oss)

    # Chief ray: max field, center of pupil
    num_fields = oss.SystemData.Fields.NumberOfFields
    chief = SingleRayTrace(hx=0, hy=1, px=0, py=0, wavelength=1,
                           field=num_fields,
                           raytrace_type="DirectionCosines")
    results["chief"] = chief.run(oss)

    return results


def extract_paraxial_heights(ray_result):
    """Extract Y-coordinate from paraxial ray trace DataFrame, skip OBJ."""
    if ray_result is None or ray_result.data is None:
        return None
    paraxial = ray_result.data.paraxial_ray_trace_data
    if paraxial is None:
        return None
    return paraxial["Y-coordinate"].values[1:]  # skip OBJ


def get_cardinal_points_from_zemax(oss):
    from zospy.analyses.reports import CardinalPoints
    return CardinalPoints().run(oss)


def get_refractive_indices_from_zemax(oss):
    """
    Pull actual refractive indices at primary wavelength from Zemax
    using the SurfaceData analysis for each glass surface.
    Returns dict mapping surface number -> n at primary wavelength.
    """
    from zospy.analyses.reports import SurfaceData
    lde = oss.LDE
    num_surf = lde.NumberOfSurfaces
    indices = {}  # surf_num -> {"material": str, "n": float}
    for i in range(num_surf):
        row = lde.GetSurfaceAt(i)
        material = row.Material if row.Material else ""
        if material:
            try:
                sd = SurfaceData(surface=i)
                result = sd.run(oss)
                # Use wavelength #2 (d-line, 0.5876 um) which matches n_d
                # The indices list is ordered by wavelength number
                for ri in result.data.material.indices:
                    if abs(ri.wavelength - 0.5876) < 0.001:
                        indices[i] = {"material": material, "n": ri.index}
                        break
                # Fallback: use second index if available (usually d-line)
                if i not in indices and len(result.data.material.indices) >= 2:
                    ri = result.data.material.indices[1]
                    indices[i] = {"material": material, "n": ri.index}
            except Exception:
                pass
    return indices


# ── Our calculation ──────────────────────────────────────────────────────────

def run_our_calculation(zmx_file, index_overrides=None):
    """
    Run our y-nu trace + Seidel pipeline.
    index_overrides: optional dict {zmx_surf_num: {"material": str, "n": float}}
        to replace parsed n_d values with actual glass catalog values.
    """
    zmx_data = parse_zmx(zmx_file)

    # Apply index overrides if provided
    if index_overrides:
        print("  Applying refractive index overrides from Zemax:")
        for s in zmx_data["surfaces"]:
            snum = s["num"]
            if snum in index_overrides:
                old_n = s["n_d"]
                new_n = index_overrides[snum]["n"]
                mat = index_overrides[snum]["material"]
                print(f"    Surf {snum} ({mat}): n_d {old_n:.6f} -> {new_n:.10f}")
                s["n_d"] = new_n

    surfaces, gaps, stop_info = build_system_from_zmx(zmx_data)
    K = len(surfaces)

    semi_aperture = zmx_data["enpd"] / 2.0
    h_m, nu_m = ynu_trace(surfaces, gaps, semi_aperture, 0.0)
    u_final = nu_m[-1] / 1.0
    efl = -semi_aperture / u_final
    bfl = -h_m[-1] / u_final

    max_field_value = max(zmx_data["fields_y"])
    field_type = zmx_data.get("field_type", 0)

    if field_type == 0:
        max_field_angle = max_field_value
    elif field_type == 2:
        # Paraxial image height -> angle via EFL
        max_field_angle = np.degrees(np.arctan(max_field_value / efl))
    else:
        max_field_angle = max_field_value

    h0_chief, nu0_chief = find_chief_ray_initial(surfaces, gaps, stop_info, max_field_angle)
    h_c, nu_c = ynu_trace(surfaces, gaps, h0_chief, nu0_chief)

    S_I, S_II, S_III, S_IV, S_V = seidel_coefficients(
        surfaces, h_m, nu_m, h_c, nu_c, 0.0, nu0_chief
    )

    H = 0.0 * h_c[0] - nu0_chief * h_m[0]

    return {
        "zmx_data": zmx_data, "surfaces": surfaces, "gaps": gaps,
        "stop_info": stop_info, "K": K,
        "h_marginal": h_m, "nu_marginal": nu_m,
        "h_chief": h_c, "nu_chief": nu_c,
        "efl": efl, "bfl": bfl, "max_field": max_field_angle,
        "S_I": S_I, "S_II": S_II, "S_III": S_III, "S_IV": S_IV, "S_V": S_V,
        "lagrange_invariant": H,
        "chief_slope_obj": nu0_chief,
        "marginal_slope_img": nu_m[-1],
    }


# ── Comparison ───────────────────────────────────────────────────────────────

def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def fmt_compare(label, ours, theirs):
    diff = abs(ours - theirs)
    rel = diff / abs(theirs) * 100 if abs(theirs) > 1e-12 else 0
    flag = " <<<" if rel > 1.0 else ""
    return f"  {label:20s} {ours:14.6f} {theirs:14.6f} {diff:14.2e} {rel:10.4f}{flag}"


def build_zmx_to_trace_map(zmx_data):
    """
    Build a mapping from Zemax surface number (1-based, excluding OBJ/IMA)
    to our trace surface index (0-based, dummy surfaces removed).

    Returns a dict {zmx_surf_num: trace_idx} for real surfaces,
    and a set of skipped (dummy) Zemax surface numbers.
    """
    zmx_surfaces = zmx_data['surfaces']
    # Determine medium after each Zemax surface
    n_media = []
    for s in zmx_surfaces:
        if s['glass'] is not None:
            n_media.append(s['n_d'])
        else:
            n_media.append(1.0)

    first_optical = 1
    last_optical = len(zmx_surfaces) - 2

    zmx_to_trace = {}
    skipped = set()
    trace_idx = 0

    for i in range(first_optical, last_optical + 1):
        n_before = n_media[i - 1]
        n_after = n_media[i]
        is_dummy = abs(n_before - n_after) < 1e-10

        if is_dummy:
            skipped.add(i)
        else:
            zmx_to_trace[i] = trace_idx
            trace_idx += 1

    return zmx_to_trace, skipped


def compare_results(our, zemax_seidel_text, zemax_cardinal, zemax_rays):
    print_header("COMPARISON: Our Parser vs Zemax OpticStudio")

    # ── 1. EFL ───────────────────────────────────────────────────────────
    print_header("FOCAL LENGTH")
    print(f"  {'':20s} {'Ours':>14s} {'Zemax':>14s} {'Diff':>14s} {'Rel%':>10s}")
    print(f"  {'':20s} {'----':>14s} {'-----':>14s} {'----':>14s} {'----':>10s}")

    zemax_efl = None
    if zemax_cardinal is not None:
        data = zemax_cardinal.data
        cp = data.cardinal_points
        zemax_efl = cp.focal_length.image
        print(f"  Zemax cardinal pts at wl={data.wavelength} um")

    if zemax_efl is not None:
        print(fmt_compare("EFL", our["efl"], zemax_efl))

    # ── 2. Header info from Seidel ───────────────────────────────────────
    zemax_seidel = parse_seidel_text(zemax_seidel_text)
    hi = zemax_seidel["header_info"]

    if hi:
        print_header("SYSTEM PARAMETERS")
        print(f"  {'':20s} {'Ours':>14s} {'Zemax':>14s}")
        print(f"  {'':20s} {'----':>14s} {'-----':>14s}")
        if "chief_slope_obj" in hi:
            print(f"  {'Chief slope (obj)':20s} {our['chief_slope_obj']:14.6f} {hi['chief_slope_obj']:14.6f}")
        if "marginal_slope_img" in hi:
            print(f"  {'Marginal slope (img)':20s} {our['marginal_slope_img']:14.6f} {hi['marginal_slope_img']:14.6f}")
        if "invariant" in hi:
            print(f"  {'Lagrange invariant':20s} {our['lagrange_invariant']:14.6f} {hi['invariant']:14.6f}")

    # ── 3. Paraxial ray heights ──────────────────────────────────────────
    print_header("PARAXIAL RAY HEIGHTS (Y-coordinate)")

    zmx_data = our["zmx_data"]
    zmx_to_trace, skipped_surfs = build_zmx_to_trace_map(zmx_data)
    zmx_surfaces = zmx_data['surfaces']
    first_optical = 1
    last_optical = len(zmx_surfaces) - 2

    if zemax_rays:
        for ray_name, our_h in [("marginal", our["h_marginal"]),
                                 ("chief", our["h_chief"])]:
            result = zemax_rays.get(ray_name)
            zmx_h = extract_paraxial_heights(result) if result else None

            print(f"\n  {ray_name.upper()} ray:")
            print(f"  {'Surf':>4s} {'Ours':>14s} {'Zemax':>14s} {'Diff':>14s} {'Rel%':>10s}")
            print(f"  {'----':>4s} {'----':>14s} {'-----':>14s} {'----':>14s} {'----':>10s}")

            # zmx_h[j] corresponds to Zemax surface j+1 (skipped OBJ=0)
            for zmx_surf in range(first_optical, last_optical + 1):
                zmx_h_idx = zmx_surf - 1  # index into zmx_h array
                zmx_val = zmx_h[zmx_h_idx] if (zmx_h is not None and zmx_h_idx < len(zmx_h)) else None

                if zmx_surf in skipped_surfs:
                    # Dummy surface (STO) — show Zemax value but no comparison
                    zmx_str = f"{zmx_val:14.6f}" if zmx_val is not None else f"{'N/A':>14s}"
                    print(f"  {zmx_surf:4d} {'(skipped)':>14s} {zmx_str} {'---':>14s} {'STO':>10s}")
                elif zmx_surf in zmx_to_trace:
                    trace_idx = zmx_to_trace[zmx_surf]
                    our_val = our_h[trace_idx]
                    if zmx_val is not None:
                        diff = abs(our_val - zmx_val)
                        rel = diff / abs(zmx_val) * 100 if abs(zmx_val) > 1e-12 else 0
                        flag = " <<<" if rel > 1.0 else ""
                        print(f"  {zmx_surf:4d} {our_val:14.6f} {zmx_val:14.6f} {diff:14.2e} {rel:10.4f}{flag}")
                    else:
                        print(f"  {zmx_surf:4d} {our_val:14.6f} {'N/A':>14s}")

    # ── 4. Seidel coefficients ───────────────────────────────────────────
    print_header("SEIDEL COEFFICIENTS (per surface)")

    names = ["S_I", "S_II", "S_III", "S_IV", "S_V"]
    our_arrays = [our["S_I"], our["S_II"], our["S_III"], our["S_IV"], our["S_V"]]

    if zemax_seidel["surfaces"]:
        print(f"  {'Surf':>4s} {'Coeff':>6s} {'Ours':>14s} {'Zemax':>14s} {'Diff':>14s} {'Rel%':>10s}")
        print(f"  {'----':>4s} {'-----':>6s} {'----':>14s} {'-----':>14s} {'----':>14s} {'----':>10s}")

        for zs in zemax_seidel["surfaces"]:
            if zs["num"] is None:
                # STO — dummy surface, show Zemax values only
                for name in names:
                    zmx_val = zs[name]
                    print(f"  {'STO':>4s} {name:>6s} {'(skipped)':>14s} {zmx_val:14.6f} {'---':>14s} {'---':>10s}")
                continue
            surf_num = zs["num"]
            if surf_num not in zmx_to_trace:
                continue
            trace_idx = zmx_to_trace[surf_num]
            if 0 <= trace_idx < our["K"]:
                for name, arr in zip(names, our_arrays):
                    our_val = arr[trace_idx]
                    zmx_val = zs[name]
                    diff = abs(our_val - zmx_val)
                    rel = diff / abs(zmx_val) * 100 if abs(zmx_val) > 1e-12 else 0
                    flag = " <<<" if rel > 1.0 else ""
                    print(f"  {surf_num:4d} {name:>6s} {our_val:14.6f} {zmx_val:14.6f} {diff:14.2e} {rel:10.4f}{flag}")

    # Totals
    print_header("SEIDEL TOTALS")
    print(f"  {'Coeff':>6s} {'Ours':>14s} {'Zemax':>14s} {'Diff':>14s} {'Rel%':>10s}")
    print(f"  {'-----':>6s} {'----':>14s} {'-----':>14s} {'----':>14s} {'----':>10s}")

    if zemax_seidel["totals"]:
        for name, arr in zip(names, our_arrays):
            our_val = arr.sum()
            zmx_val = zemax_seidel["totals"][name]
            diff = abs(our_val - zmx_val)
            rel = diff / abs(zmx_val) * 100 if abs(zmx_val) > 1e-12 else 0
            flag = " <<<" if rel > 1.0 else ""
            print(f"  {name:>6s} {our_val:14.6f} {zmx_val:14.6f} {diff:14.2e} {rel:10.4f}{flag}")
    else:
        for name, arr in zip(names, our_arrays):
            print(f"  {name:>6s} {our_val:14.6f} {'N/A':>14s}")

    # ── Summary ──────────────────────────────────────────────────────────
    print_header("SUMMARY")
    if zemax_seidel["totals"]:
        max_rel = 0
        worst = ""
        for name, arr in zip(names, our_arrays):
            zmx_val = zemax_seidel["totals"][name]
            if abs(zmx_val) > 1e-12:
                rel = abs(arr.sum() - zmx_val) / abs(zmx_val) * 100
                if rel > max_rel:
                    max_rel = rel
                    worst = name
        if max_rel < 0.1:
            print("  PASS: All Seidel totals match within 0.1%")
        elif max_rel < 5.0:
            print(f"  CLOSE: Max relative error = {max_rel:.2f}% on {worst}")
        else:
            print(f"  MISMATCH: Max relative error = {max_rel:.2f}% on {worst}")
            print("  Possible causes: wavelength diff, sign convention, chief ray normalization")
    else:
        print("  Could not parse Zemax Seidel output.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare y-nu trace vs Zemax OpticStudio")
    parser.add_argument("--extension", action="store_true",
                        help="Connect as extension (OpticStudio must be running)")
    parser.add_argument("--zmx", default=ZMX_FILE, help="ZMX file to compare")
    args = parser.parse_args()

    mode = "extension" if args.extension else "standalone"

    # ── Step 1: Our calculation ──
    print("Running our y-nu trace + Seidel calculation...")
    our = run_our_calculation(args.zmx)
    print(f"  EFL = {our['efl']:.4f} mm, BFL = {our['bfl']:.4f} mm")
    print(f"  Surfaces: {our['K']}, Stop: {our['stop_info']}")

    # ── Step 2: Zemax calculation ──
    print(f"\nConnecting to Zemax OpticStudio ({mode} mode)...")
    zos, oss = connect_to_zemax(mode=mode)
    print("  Connected.")

    print(f"Loading {args.zmx}...")
    load_zmx_in_zemax(oss, args.zmx)
    print("  Loaded.")

    # Pull actual refractive indices from Zemax glass catalog
    print("Pulling refractive indices from Zemax...")
    try:
        zemax_indices = get_refractive_indices_from_zemax(oss)
        print(f"  Got indices for {len(zemax_indices)} glass surfaces.")
    except Exception as e:
        print(f"  Failed to get indices: {e}")
        zemax_indices = None

    # Re-run our calculation with corrected indices
    if zemax_indices:
        print("\nRe-running our calculation with Zemax glass indices...")
        our = run_our_calculation(args.zmx, index_overrides=zemax_indices)
        print(f"  EFL = {our['efl']:.4f} mm, BFL = {our['bfl']:.4f} mm")

    print("Running Zemax Seidel analysis...")
    try:
        seidel_text = get_seidel_from_zemax(oss)
        print("  Got Seidel text output.")
    except Exception as e:
        print(f"  Seidel analysis failed: {e}")
        seidel_text = ""

    print("Getting cardinal points...")
    try:
        cardinal = get_cardinal_points_from_zemax(oss)
        print("  Got cardinal points.")
    except Exception as e:
        print(f"  Cardinal points failed: {e}")
        cardinal = None

    print("Running paraxial ray traces...")
    try:
        rays = get_paraxial_ray_trace_from_zemax(oss)
        print("  Got ray trace data.")
    except Exception as e:
        print(f"  Ray trace failed: {e}")
        rays = None

    # ── Step 3: Compare ──
    compare_results(our, seidel_text, cardinal, rays)

    # Cleanup
    zos.disconnect()
    print("\nDone. Zemax disconnected.")


if __name__ == "__main__":
    main()
