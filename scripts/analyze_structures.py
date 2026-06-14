#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X-PIE Structure Analysis Script

Analyzes XPLOR-sampled PDB structures and performs RMSD calculations.

Usage:
    python scripts/analyze_structures.py [--output FILE] [--atom-type backbone|ca|heavy]

Interactive workflow:
    1. Prompt for PDB directory path (default: ./output)
    2. Prompt for number of structures to analyze (default: 5)
    3. Prompt for reference structure (optional)
    4. Execute analysis and output results
"""

import os
import sys
import argparse
import glob
import numpy as np
import re
from statistics import mean, stdev

# ============================================================
# X-PIE Logo
# ============================================================
XPIE_LOGO = r"""

██╗  ██╗      ██████╗ ██╗███████╗
╚██╗██╔╝      ██╔══██╗██║██╔════╝
 ╚███╔╝ █████╗██████╔╝██║█████╗
 ██╔██╗ ╚════╝██╔═══╝ ██║██╔══╝
██╔╝ ██╗      ██║     ██║███████╗
╚═╝  ╚═╝      ╚═╝     ╚═╝╚══════╝

Model Selection and Analysis

"""

# ============================================================
# Defaults
# ============================================================
DEFAULT_PDB_DIR = "./output"
DEFAULT_OUTPUT = "structure_analysis.txt"
DEFAULT_TOP_N = 5
DEFAULT_VDW_THRESHOLD = 500.0
DEFAULT_REF_PDB = "ref.pdb"

ATOM_TYPE_FILTERS = {
    'backbone': {'N', 'CA', 'C', 'O'},
    'ca': {'CA'},
    'heavy': None,  # all non-H
}


# ============================================================
# PDB Header Parsing
# ============================================================

def parse_pdb_remarks(pdb_path):
    """
    Parse REMARK lines from PDB header.
    Returns dict with keys: 'total', 'vdw', 'violations', 'name'
    """
    result = {'name': os.path.basename(pdb_path),
              'total': None,
              'vdw': None,
              'violations': None}

    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith('REMARK'):
                if line.startswith('ATOM') or line.startswith('HETATM') or line.startswith('END'):
                    break
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            if parts[1] == 'summary' and parts[2] == 'total' and len(parts) >= 4:
                try:
                    result['total'] = float(parts[3])
                except ValueError:
                    pass

            if parts[1] == 'summary' and parts[2] == 'VDW' and len(parts) >= 4:
                try:
                    result['vdw'] = float(parts[3])
                except ValueError:
                    pass

            if parts[1] == 'NOE' and parts[2] == 'xlms' and len(parts) >= 6:
                try:
                    result['violations'] = int(parts[5])
                except ValueError:
                    pass

    return result


# ============================================================
# Violation File Parsing
# ============================================================

def extract_violation_details(viol_path):
    """
    Extract content between:
        'Violated NOE restraints in potential term: xlms ...'
    and:
        'simulation: SizeOneEnsemble_XplorSimulation'
    Returns the extracted text or None if markers not found.
    """
    if not os.path.exists(viol_path):
        return None

    with open(viol_path, 'r') as f:
        lines = f.readlines()

    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if "Violated NOE restraints in potential term: xlms" in line:
            start_idx = i
        if start_idx is not None and "simulation: SizeOneEnsemble_XplorSimulation" in line:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        return None

    detail_lines = lines[start_idx + 1:end_idx]
    return ''.join(detail_lines)


# ============================================================
# refine.py Parsing
# ============================================================

def parse_refine_py(refine_path):
    """
    Parse refine.py to extract:
      - segid -> protein base name mapping
      - grouped segids with residue ranges
    Returns (segid_to_protein, group_defs)
    where group_defs is list of (segid, include_ranges, exclude_ranges)
    include_ranges and exclude_ranges are lists of (start, end) tuples.
    """
    segid_to_protein = {}
    group_defs = []
    last_pdb = None

    with open(refine_path, 'r') as f:
        content = f.read()

    # Parse initCoords -> SetProperty('segmentName', 'SEGID')
    # Track the last seen initCoords before each SetProperty
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check for initCoords
        m_init = re.search(r"protocol\.initCoords\(['\"](.+?)['\"]\)", line)
        if m_init:
            last_pdb = m_init.group(1)
        # Check for SetProperty('segmentName', 'SEGID')
        m_seg = re.search(r"SetProperty\(['\"]segmentName['\"],\s*['\"](.+?)['\"]\)", line)
        if m_seg and last_pdb:
            segid = m_seg.group(1)
            # Extract base name: strip input/, -prepare.pdb, .pdb
            basename = os.path.basename(last_pdb)
            basename = basename.replace('-prepare.pdb', '').replace('.pdb', '')
            segid_to_protein[segid] = basename
            last_pdb = None
        i += 1

    # Parse dyn.group(""" ... """) lines
    group_pattern = re.compile(r'dyn\.group\("""\s*(.+?)\s*"""\)', re.DOTALL)
    for m in group_pattern.finditer(content):
        group_sel = m.group(1).strip()
        # Parse segid
        segid_match = re.search(r'segid\s+(\S+)', group_sel)
        if not segid_match:
            continue
        segid = segid_match.group(1)
        include_ranges = []
        exclude_ranges = []
        # Check for resid ranges
        # Pattern: and resid X:Y
        for rm in re.finditer(r'and\s+resid\s+(\d+):(\d+)', group_sel):
            include_ranges.append((int(rm.group(1)), int(rm.group(2))))
        # Pattern: and not (resid X:Y)
        for rm in re.finditer(r'and\s+not\s+\(\s*resid\s+(\d+):(\d+)\s*\)', group_sel):
            exclude_ranges.append((int(rm.group(1)), int(rm.group(2))))
        group_defs.append((segid, include_ranges, exclude_ranges))

    return segid_to_protein, group_defs


def build_display_names(ordered_segids, segid_to_protein):
    """
    Build display names for grouped segids.
    If a base name appears for multiple segids, append numeric suffix.
    """
    base_counts = {}
    for segid in ordered_segids:
        base = segid_to_protein.get(segid, segid)
        base_counts[base] = base_counts.get(base, 0) + 1

    base_counters = {}
    display_names = {}
    for segid in ordered_segids:
        base = segid_to_protein.get(segid, segid)
        if base_counts[base] > 1:
            base_counters[base] = base_counters.get(base, 0) + 1
            display_names[segid] = f"{base}{base_counters[base]}"
        else:
            display_names[segid] = base
    return display_names


# ============================================================
# Structure Selection
# ============================================================

def select_structures(pdb_dir, vdw_threshold=DEFAULT_VDW_THRESHOLD, top_n=DEFAULT_TOP_N):
    """
    Scan PDB files, parse headers, and select structures.
    Selection is based solely on VDW energy <= threshold,
    regardless of cross-link violations.

    Returns a dict:
        {
            'all':          list of all parsed dicts (unsorted),
            'candidates':   list of dicts with vdw<=threshold (sorted by total energy),
            'selected':     top-N selected dicts (by total energy, ascending),
            'has_violations': bool, whether any selected has violations>0
            'min_viol_struct': dict with minimum violations (or None)
        }
    """
    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "Calc_*.pdb")))
    if not pdb_files:
        print(f"[ERROR] No Calc_*.pdb files found in {pdb_dir}")
        sys.exit(1)

    all_data = []
    for pdb_path in pdb_files:
        data = parse_pdb_remarks(pdb_path)
        data['path'] = pdb_path
        all_data.append(data)

    for d in all_data:
        if d['total'] is None or d['vdw'] is None or d['violations'] is None:
            print(f"[WARNING] Incomplete header data for {d['name']} "
                  f"(total={d['total']}, vdw={d['vdw']}, violations={d['violations']})")

    valid = [d for d in all_data if d['total'] is not None and d['vdw'] is not None and d['violations'] is not None]

    candidates = [d for d in valid if d['vdw'] <= vdw_threshold]
    candidates.sort(key=lambda x: x['total'])

    selected = candidates[:top_n]

    has_violations = any(d['violations'] > 0 for d in selected)

    min_viol_struct = None
    if valid:
        min_viol_struct = min(valid, key=lambda x: x['violations'])

    return {
        'all': valid,
        'candidates': candidates,
        'selected': selected,
        'has_violations': has_violations,
        'min_viol_struct': min_viol_struct
    }


# ============================================================
# Output Reporting
# ============================================================

def print_table(selected, out_file=None):
    """Print and save the selected structures table."""
    lines = []
    header = f"{'PDB Name':<20} {'Total Energy':>15} {'VDW Energy':>15} {'Violations':>12}"
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    for d in selected:
        lines.append(f"{d['name']:<20} {d['total']:>15.2f} {d['vdw']:>15.2f} {d['violations']:>12d}")

    text = '\n'.join(lines)
    print("\n" + text)
    if out_file:
        out_file.write(text + '\n')
    return text


def report_no_zero_violation(min_viol_struct, out_file=None):
    """
    Report when no violation-free structure exists.
    Print violated restraint details and suggestions.
    """
    pdb_name = min_viol_struct['name']
    viol_count = min_viol_struct['violations']
    viol_path = min_viol_struct['path'] + ".viols"

    msg = f"\n[!] No structure with zero violations was found.\n"
    msg += f"    The structure with the minimum violations is: {pdb_name}\n"
    msg += f"    Minimum violation count: {viol_count}\n"
    print(msg)
    if out_file:
        out_file.write(msg + '\n')

    details = extract_violation_details(viol_path)
    if details and details.strip():
        detail_msg = (
            f"\n>> Violated NOE restraints in potential term: xlms (details from {os.path.basename(viol_path)})\n"
            + details
        )
        print(detail_msg)
        if out_file:
            out_file.write(detail_msg + '\n')
    else:
        no_detail = "\n>> No detailed violation records found in the .viols file.\n"
        print(no_detail)
        if out_file:
            out_file.write(no_detail + '\n')

    suggest = (
        "\n[Suggestions] The following constraints are not fully satisfied. You may try:\n"
        "  1) Increase the number of interface states (e.g., add more cluster definitions).\n"
        "  2) Expand the flexible residue ranges to enlarge the conformational sampling space\n"
        "     for multi-domain proteins.\n"
        "  3) Relax the distance threshold of cross-link restraints in xlms.tbl moderately.\n"
    )
    print(suggest)
    if out_file:
        out_file.write(suggest + '\n')


# ============================================================
# Manual PDB Atom Parsing & Kabsch Alignment
# ============================================================

def parse_pdb_atoms(pdb_path, atom_filter=None):
    """
    Parse a PDB file manually and return a dict of atom coordinates
    keyed by (chain_id, res_seq, icode, res_name, atom_name).
    atom_filter: set of atom names to keep (e.g. {'N','CA','C','O'}).
                 If None, keep all heavy atoms (non-H).
    """
    atoms = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            if len(line) < 54:
                continue

            atom_name = line[12:16].strip()
            if atom_name.startswith('H'):
                continue
            if atom_filter is not None and atom_name not in atom_filter:
                continue

            res_name = line[17:20].strip()
            try:
                res_seq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26] if len(line) > 26 else ' '
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            # Parse chain ID (column 22) and segid (columns 73-76)
            chain_id = line[21] if len(line) > 21 else ' '
            segid = line[72:76].strip() if len(line) > 76 else ''

            # If chain ID is missing but segid is present, map segid to chain ID.
            # Rule: ALT1 -> A, BLT1 -> B, CLT1 -> C, etc.
            if chain_id == ' ' and segid:
                chain_id = segid[0].upper()

            key = (chain_id, res_seq, icode, res_name, atom_name)
            atoms[key] = np.array([x, y, z])
    return atoms


def parse_pdb_atoms_filtered(pdb_path, segid, include_ranges, exclude_ranges, atom_filter=None):
    """
    Parse a PDB file and return a dict of atom coordinates
    keyed by (chain_id, res_seq, icode, res_name, atom_name),
    but only keep atoms matching the given segid and residue ranges.
    """
    atoms = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            if len(line) < 54:
                continue

            atom_name = line[12:16].strip()
            if atom_name.startswith('H'):
                continue
            if atom_filter is not None and atom_name not in atom_filter:
                continue

            # Parse segid (columns 73-76)
            atom_segid = line[72:76].strip() if len(line) > 76 else ''
            if atom_segid != segid:
                continue

            res_name = line[17:20].strip()
            try:
                res_seq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26] if len(line) > 26 else ' '
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            # Residue filtering
            if include_ranges:
                in_range = any(start <= res_seq <= end for start, end in include_ranges)
                if not in_range:
                    continue
            if exclude_ranges:
                in_excluded = any(start <= res_seq <= end for start, end in exclude_ranges)
                if in_excluded:
                    continue

            chain_id = line[21] if len(line) > 21 else ' '
            if chain_id == ' ' and atom_segid:
                chain_id = atom_segid[0].upper()

            key = (chain_id, res_seq, icode, res_name, atom_name)
            atoms[key] = np.array([x, y, z])
    return atoms


def kabsch_core(mobile_coords, ref_coords):
    """
    Core Kabsch alignment (no truncation).
    Returns (rotation_matrix, translation_vector, rmsd).
    """
    cm = np.mean(mobile_coords, axis=0)
    cr = np.mean(ref_coords, axis=0)

    P = mobile_coords - cm
    Q = ref_coords - cr

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    t = cr - cm @ R.T
    aligned = mobile_coords @ R.T + t
    rmsd = np.sqrt(np.mean(np.sum((aligned - ref_coords) ** 2, axis=1)))
    return R, t, rmsd


def iterative_kabsch_align(mobile_coords, ref_coords, cutoff=10.0, cycles=5):
    """
    Iterative Kabsch alignment similar to PyMOL's align.
    At each cycle, atoms with distance > cutoff are excluded from the next fit.
    Returns (aligned_coords_of_all_atoms, rmsd_of_kept_atoms, kept_count, mask).
    The returned RMSD reflects only the atoms that survived truncation,
    which is consistent with how PyMOL's align reports its value.
    """
    m = mobile_coords.copy()
    r = ref_coords.copy()
    mask = np.ones(len(m), dtype=bool)

    for _ in range(cycles):
        if mask.sum() == 0:
            break

        R, t, _ = kabsch_core(m[mask], r[mask])
        aligned_all = m @ R.T + t
        dists = np.sqrt(np.sum((aligned_all - r) ** 2, axis=1))
        new_mask = dists <= cutoff

        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    # Final alignment using the last kept set
    if mask.sum() == 0:
        return m, np.nan, 0, mask

    R, t, _ = kabsch_core(m[mask], r[mask])
    aligned_all = m @ R.T + t
    # PyMOL-style RMSD: only on the kept (well-aligned) atoms
    rmsd_kept = np.sqrt(np.mean(np.sum((aligned_all[mask] - r[mask]) ** 2, axis=1)))
    return aligned_all, rmsd_kept, int(mask.sum()), mask


def rmsd_to_average_filtered(pdb_paths, segid, include_ranges, exclude_ranges, atom_filter=None, cutoff=None):
    """
    Compute RMSD of each structure to the average structure,
    but only using atoms matching the given segid and residue ranges.
    NO alignment is performed — coordinates are used as-is, because
    grouped domains are treated as rigid bodies and alignment would
    erase the very conformational differences we want to measure.
    Returns dict with 'rmsds', 'min', 'max', 'mean', 'std', 'common_atoms'.
    """
    if not pdb_paths:
        return None

    all_atoms = [parse_pdb_atoms_filtered(p, segid, include_ranges, exclude_ranges, atom_filter) for p in pdb_paths]

    common_keys = set(all_atoms[0].keys())
    for atoms in all_atoms[1:]:
        common_keys &= set(atoms.keys())

    if not common_keys:
        print(f"[WARNING] No common atoms found across structures for segid {segid}.")
        return None

    common_keys = sorted(common_keys)
    coords_list = [np.array([atoms[k] for k in common_keys]) for atoms in all_atoms]

    # Direct average of raw coordinates (no alignment)
    avg = np.mean(coords_list, axis=0)

    # Compute RMSD for each structure vs average directly
    rmsds = []
    for coords in coords_list:
        rmsds.append(np.sqrt(np.mean(np.sum((coords - avg) ** 2, axis=1))))

    return {
        'rmsds': rmsds,
        'min': min(rmsds),
        'max': max(rmsds),
        'mean': mean(rmsds),
        'std': stdev(rmsds) if len(rmsds) > 1 else 0.0,
        'common_atoms': len(common_keys),
        'iterations': 1
    }


def compute_conformational_heterogeneity(pdb_paths, group_defs, display_names, atom_filter=None, cutoff=None):
    """
    Compute conformational heterogeneity RMSD for grouped segids.
    Returns (per_seg_results, combined_rmsds, sorted_indices)
    where per_seg_results is dict of segid -> result dict,
    combined_rmsds is list of combined RMSD per structure,
    sorted_indices is list of structure indices sorted by combined RMSD ascending.
    """
    per_seg_results = {}
    all_rmsd_arrays = []

    for segid, include_ranges, exclude_ranges in group_defs:
        result = rmsd_to_average_filtered(pdb_paths, segid, include_ranges, exclude_ranges, atom_filter=atom_filter, cutoff=cutoff)
        if result is None:
            continue
        per_seg_results[segid] = result
        all_rmsd_arrays.append(np.array(result['rmsds']))

    if not per_seg_results:
        return None, None, None

    n_structs = len(pdb_paths)
    if len(all_rmsd_arrays) == 1:
        combined_rmsds = all_rmsd_arrays[0].tolist()
    else:
        combined_rmsds = np.sum(all_rmsd_arrays, axis=0).tolist()

    sorted_indices = np.argsort(combined_rmsds).tolist()
    return per_seg_results, combined_rmsds, sorted_indices


def rmsd_to_reference(pdb_paths, ref_path, atom_filter=None, cutoff=None):
    """
    Part 2: Align each structure to the reference structure and compute RMSD.
    If strict chain-ID-matching yields no atoms, falls back to matching by
    (res_seq, icode, res_name, atom_name) ignoring chain ID.
    Returns dict with 'rmsds', 'matched_counts', 'min', 'mean', 'std'.
    """
    if not pdb_paths:
        return None

    ref_atoms = parse_pdb_atoms(ref_path, atom_filter)

    rmsds = []
    matched_counts = []
    for pdb_path in pdb_paths:
        atoms = parse_pdb_atoms(pdb_path, atom_filter)
        common_keys = sorted(set(ref_atoms.keys()) & set(atoms.keys()))

        if common_keys:
            ref_coords = np.array([ref_atoms[k] for k in common_keys])
            mobile_coords = np.array([atoms[k] for k in common_keys])
            matched_counts.append(len(common_keys))
            fallback = False
        else:
            # Fallback: ignore chain ID
            ref_by_key = {}
            for (_, rs, ic, rn, an), coord in ref_atoms.items():
                key = (rs, ic, rn, an)
                ref_by_key[key] = coord

            atoms_by_key = {}
            for (_, rs, ic, rn, an), coord in atoms.items():
                key = (rs, ic, rn, an)
                atoms_by_key[key] = coord

            fallback_keys = sorted(set(ref_by_key.keys()) & set(atoms_by_key.keys()))
            if not fallback_keys:
                print(f"[WARNING] No common atoms with reference for {os.path.basename(pdb_path)}. Skipping.")
                rmsds.append(None)
                matched_counts.append(0)
                continue

            ref_coords = np.array([ref_by_key[k] for k in fallback_keys])
            mobile_coords = np.array([atoms_by_key[k] for k in fallback_keys])
            matched_counts.append(len(fallback_keys))
            print(f"[INFO] {os.path.basename(pdb_path)}: chain ID mismatch with reference; "
                  f"using residue-only matching ({len(fallback_keys)} atoms).")
            fallback = True

        if cutoff is not None and cutoff > 0:
            _, rmsd, _, _ = iterative_kabsch_align(mobile_coords, ref_coords, cutoff=cutoff)
        else:
            _, _, rmsd = kabsch_core(mobile_coords, ref_coords)

        rmsds.append(rmsd)

    valid_rmsds = [r for r in rmsds if r is not None and not np.isnan(r)]
    if not valid_rmsds:
        print("[WARNING] No valid RMSD could be calculated against the reference.")
        return None

    return {
        'rmsds': rmsds,
        'matched_counts': matched_counts,
        'min': min(valid_rmsds),
        'max': max(valid_rmsds),
        'mean': mean(valid_rmsds),
        'std': stdev(valid_rmsds) if len(valid_rmsds) > 1 else 0.0
    }


def diagnose_structures(pdb_paths, selected_names, atom_filter=None):
    """Check if all selected structures have identical atom coordinates."""
    if len(pdb_paths) < 2:
        return

    all_atoms = [parse_pdb_atoms(p, atom_filter) for p in pdb_paths]
    common_keys = set(all_atoms[0].keys())
    for atoms in all_atoms[1:]:
        common_keys &= set(atoms.keys())

    if not common_keys:
        return

    identical = True
    for k in common_keys:
        coords = [atoms[k] for atoms in all_atoms]
        for i in range(1, len(coords)):
            if np.any(np.abs(coords[0] - coords[i]) > 1e-6):
                identical = False
                break
        if not identical:
            break

    if identical:
        print("\n[WARNING] All selected structures have IDENTICAL coordinates.")
        print("          RMSD values will all be 0.000 A.")
        print("          This usually means XPLOR sampling did not generate")
        print("          diverse conformations. Please check your sampling")
        print("          parameters (random seed, annealing steps, etc.).")


# ============================================================
# Main Workflow
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze XPLOR-sampled PDB structures and calculate RMSDs"
    )
    parser.add_argument(
        '--output', '-o',
        default=DEFAULT_OUTPUT,
        help=f'Output file for analysis report. Default: {DEFAULT_OUTPUT}'
    )
    parser.add_argument(
        '--atom-type',
        choices=['backbone', 'ca', 'heavy'],
        default='backbone',
        help='Atom type for RMSD calculation: backbone (N,CA,C,O), ca, or heavy. Default: backbone'
    )
    args = parser.parse_args()

    # 1. Print X-PIE Logo
    print(XPIE_LOGO)

    # 2. Prompt for PDB directory
    pdb_dir_input = input("Enter the path to the structure files for analysis [default: ./output]: ").strip()
    pdb_dir = pdb_dir_input if pdb_dir_input else DEFAULT_PDB_DIR
    pdb_dir = os.path.abspath(pdb_dir)

    if not os.path.isdir(pdb_dir):
        print(f"[ERROR] Directory not found: {pdb_dir}")
        sys.exit(1)

    # 3. Prompt for number of structures
    top_n_input = input("Enter the number of structures to compare and analyze [default: 5]: ").strip()
    try:
        top_n = int(top_n_input) if top_n_input else DEFAULT_TOP_N
    except ValueError:
        print("[WARNING] Invalid input, using default value 5")
        top_n = DEFAULT_TOP_N

    # 4. Prompt for reference structure
    ref_structure_path = None
    has_ref = input("\nDo you have a reference structure? (y/n) [n]: ").strip().lower()
    if has_ref in ('y', 'yes'):
        ref_input = input("Enter the name and path of the reference structure: ").strip()
        if ref_input:
            ref_structure_path = os.path.abspath(ref_input)
            if not os.path.exists(ref_structure_path):
                print(f"[WARNING] Reference structure not found: {ref_structure_path}, skipping reference comparison")
                ref_structure_path = None

    # 5. Confirm settings
    out_path = os.path.abspath(args.output)
    print("\n" + "=" * 70)
    print("Confirmation")
    print("=" * 70)
    print(f"  Structure path   : {pdb_dir}")
    print(f"  Number of structures : {top_n}")
    print(f"  Reference structure  : {ref_structure_path if ref_structure_path else 'None'}")
    print(f"  Atom type            : {args.atom_type}")
    print(f"  Output report        : {out_path}")
    print("=" * 70)

    # Execute analysis
    atom_filter = ATOM_TYPE_FILTERS[args.atom_type]
    avg_cutoff = 10.0   # Iterative cutoff for ensemble convergence
    ref_cutoff = None   # Use standard Kabsch for reference comparison
                        # (iterative cutoff would exclude rigid-body motions)

    # Step 1: Selection
    print("\n" + "=" * 70)
    print("X-PIE Structure Analysis")
    print("=" * 70)
    print(f"\nPDB directory : {pdb_dir}")
    print(f"VDW threshold : {DEFAULT_VDW_THRESHOLD}")
    print(f"Top N         : {top_n}")
    print(f"Atom type     : {args.atom_type}")
    print()

    result = select_structures(pdb_dir, vdw_threshold=DEFAULT_VDW_THRESHOLD, top_n=top_n)
    valid = result['all']
    selected = result['selected']
    has_violations = result['has_violations']
    min_viol_struct = result['min_viol_struct']

    print(f"Total structures scanned : {len(valid)}")
    zero_viol_count = sum(1 for d in result['candidates'] if d['violations'] == 0)
    print(f"Zero-violation structures: {zero_viol_count} (out of {len(result['candidates'])} VDW-qualified)")
    print(f"Selected structures      : {len(selected)}")

    with open(out_path, 'w') as out_file:
        out_file.write("=" * 70 + '\n')
        out_file.write("X-PIE Structure Analysis Report\n")
        out_file.write("=" * 70 + '\n')
        out_file.write(f"PDB directory : {pdb_dir}\n")
        out_file.write(f"VDW threshold : {DEFAULT_VDW_THRESHOLD}\n")
        out_file.write(f"Top N         : {top_n}\n")
        out_file.write(f"Atom type     : {args.atom_type}\n")
        out_file.write('\n')

        # Report violation info if any selected structure has violations
        if result['has_violations']:
            if min_viol_struct:
                report_no_zero_violation(min_viol_struct, out_file)
            else:
                print("[ERROR] No valid structures found.")
                sys.exit(1)
        else:
            out_file.write("[OK] Violation-free structures were found.\n\n")
            print("[OK] Violation-free structures were found.")

        header_msg = (
            f"Top {len(selected)} structure(s) with VDW<={DEFAULT_VDW_THRESHOLD} "
            f"(sorted by total energy):\n"
        )
        print(header_msg)
        out_file.write(header_msg + '\n')

        print_table(selected, out_file)
        out_file.write('\n')

    print(f"\n[INFO] Analysis report saved to: {out_path}")

    pdb_paths = [d['path'] for d in selected]
    if not pdb_paths:
        print("[ERROR] No structures available for RMSD calculation.")
        sys.exit(1)

    if ref_structure_path:
        print(f"[INFO] Loaded reference structure: {ref_structure_path}")

    # --- Part 1: Conformational Heterogeneity RMSD ---
    # Check for energy tie among selected top structures
    energy_tie = False
    if len(selected) >= 2 and selected[0]['total'] == selected[-1]['total']:
        energy_tie = True

    # Parse refine.py for grouped segids
    refine_path = os.path.join(os.path.dirname(__file__), '..', 'refine.py')
    refine_path = os.path.abspath(refine_path)
    segid_to_protein = {}
    group_defs = []
    if os.path.exists(refine_path):
        segid_to_protein, group_defs = parse_refine_py(refine_path)
    else:
        print(f"[WARNING] refine.py not found at {refine_path}. Skipping conformational heterogeneity analysis.")

    # If energy tie, reselect from ALL VDW-qualified candidates
    if energy_tie and group_defs:
        print(f"\n[INFO] Energy tie detected among top {len(selected)} structures (total={selected[0]['total']:.2f}).")
        print(f"[INFO] Re-selecting from all {len(result['candidates'])} VDW-qualified candidates using conformational heterogeneity RMSD.")
        candidate_paths = [d['path'] for d in result['candidates']]
        ordered_segids = [g[0] for g in group_defs]
        display_names = build_display_names(ordered_segids, segid_to_protein)
        per_seg_results, combined_rmsds, sorted_indices = compute_conformational_heterogeneity(
            candidate_paths, group_defs, display_names, atom_filter=atom_filter, cutoff=avg_cutoff
        )

        if per_seg_results is not None and sorted_indices is not None:
            # Re-select top N by combined RMSD
            new_selected = [result['candidates'][i] for i in sorted_indices[:top_n]]
            selected = new_selected
            pdb_paths = [d['path'] for d in selected]

            # Re-compute conformational heterogeneity based on selected top-N only
            per_seg_results, combined_rmsds, sorted_indices = compute_conformational_heterogeneity(
                pdb_paths, group_defs, display_names, atom_filter=atom_filter, cutoff=avg_cutoff
            )

            print("\n" + "=" * 70)
            print("RMSD Analysis - Part 1: Conformational Heterogeneity")
            print("=" * 70)

            # Build group info string
            group_info_parts = []
            for segid, include_ranges, exclude_ranges in group_defs:
                dname = display_names.get(segid, segid)
                # Build range description
                range_str = ""
                if include_ranges:
                    range_str = " " + ", ".join(f"{s}:{e}" for s, e in include_ranges)
                elif exclude_ranges:
                    range_str = " all except " + ", ".join(f"{s}:{e}" for s, e in exclude_ranges)
                else:
                    range_str = ""
                group_info_parts.append(f"{dname} ({segid}{range_str})")

            print(f"  Grouped components     : {', '.join(group_info_parts)}")
            print(f"  Number of structures   : {len(pdb_paths)}")

            with open(out_path, 'a') as out_file:
                out_file.write('\n' + '=' * 70 + '\n')
                out_file.write("RMSD Analysis - Part 1: Conformational Heterogeneity\n")
                out_file.write('=' * 70 + '\n')
                out_file.write(f"  Grouped components     : {', '.join(group_info_parts)}\n")
                out_file.write(f"  Number of structures   : {len(pdb_paths)}\n")

                # Per-component RMSD stats
                out_file.write("\n  Per-component RMSD to average:\n")
                print("\n  Per-component RMSD to average:")
                for segid, _, _ in group_defs:
                    if segid not in per_seg_results:
                        continue
                    res = per_seg_results[segid]
                    dname = display_names.get(segid, segid)
                    line = f"    {dname:<20} Min = {res['min']:.3f} A, Max = {res['max']:.3f} A, Mean = {res['mean']:.3f} A"
                    print(line)
                    out_file.write(line + '\n')

                # Combined RMSD stats
                if len(group_defs) > 1:
                    cmin = min(combined_rmsds)
                    cmax = max(combined_rmsds)
                    cmean = mean(combined_rmsds)
                    line = f"\n  Combined RMSD (sum)    : Min = {cmin:.3f} A, Max = {cmax:.3f} A, Mean = {cmean:.3f} A"
                    print(line)
                    out_file.write(line + '\n')

                out_file.write(f"\n  Top {len(selected)} structures by combined RMSD:\n")
                print(f"\n  Top {len(selected)} structures by combined RMSD:")
                for rank, idx in enumerate(sorted_indices[:top_n], 1):
                    struct_name = selected[idx]['name']
                    combined = combined_rmsds[idx]
                    parts = []
                    for segid, _, _ in group_defs:
                        if segid in per_seg_results:
                            dname = display_names.get(segid, segid)
                            parts.append(f"{dname}={per_seg_results[segid]['rmsds'][idx]:.3f}")
                    detail = ", ".join(parts)
                    if len(group_defs) > 1:
                        line = f"    {struct_name:<20} Combined = {combined:.3f} A  [{detail}]"
                    else:
                        line = f"    {struct_name:<20} RMSD = {combined:.3f} A"
                    print(line)
                    out_file.write(line + '\n')
        else:
            print("[WARNING] Could not compute conformational heterogeneity RMSD.")
            # Keep original selection
    elif group_defs:
        # No tie: show standard conformational heterogeneity for selected structures
        ordered_segids = [g[0] for g in group_defs]
        display_names = build_display_names(ordered_segids, segid_to_protein)
        per_seg_results, combined_rmsds, sorted_indices = compute_conformational_heterogeneity(
            pdb_paths, group_defs, display_names, atom_filter=atom_filter, cutoff=avg_cutoff
        )

        print("\n" + "=" * 70)
        print("RMSD Analysis - Part 1: Conformational Heterogeneity")
        print("=" * 70)

        if per_seg_results is not None:
            group_info_parts = []
            for segid, include_ranges, exclude_ranges in group_defs:
                dname = display_names.get(segid, segid)
                range_str = ""
                if include_ranges:
                    range_str = " " + ", ".join(f"{s}:{e}" for s, e in include_ranges)
                elif exclude_ranges:
                    range_str = " all except " + ", ".join(f"{s}:{e}" for s, e in exclude_ranges)
                else:
                    range_str = ""
                group_info_parts.append(f"{dname} ({segid}{range_str})")

            print(f"  Grouped components     : {', '.join(group_info_parts)}")
            print(f"  Number of structures   : {len(pdb_paths)}")

            with open(out_path, 'a') as out_file:
                out_file.write('\n' + '=' * 70 + '\n')
                out_file.write("RMSD Analysis - Part 1: Conformational Heterogeneity\n")
                out_file.write('=' * 70 + '\n')
                out_file.write(f"  Grouped components     : {', '.join(group_info_parts)}\n")
                out_file.write(f"  Number of structures   : {len(pdb_paths)}\n")

                out_file.write("\n  Per-component RMSD to average:\n")
                print("\n  Per-component RMSD to average:")
                for segid, _, _ in group_defs:
                    if segid not in per_seg_results:
                        continue
                    res = per_seg_results[segid]
                    dname = display_names.get(segid, segid)
                    line = f"    {dname:<20} Min = {res['min']:.3f} A, Max = {res['max']:.3f} A, Mean = {res['mean']:.3f} A"
                    print(line)
                    out_file.write(line + '\n')

                if len(group_defs) > 1:
                    cmin = min(combined_rmsds)
                    cmax = max(combined_rmsds)
                    cmean = mean(combined_rmsds)
                    line = f"\n  Combined RMSD (sum)    : Min = {cmin:.3f} A, Max = {cmax:.3f} A, Mean = {cmean:.3f} A"
                    print(line)
                    out_file.write(line + '\n')

                out_file.write(f"\n  Selected structures (sorted by total energy):\n")
                print(f"\n  Selected structures (sorted by total energy):")
                for i, d in enumerate(selected):
                    struct_name = d['name']
                    parts = []
                    for segid, _, _ in group_defs:
                        if segid in per_seg_results:
                            dname = display_names.get(segid, segid)
                            parts.append(f"{dname}={per_seg_results[segid]['rmsds'][i]:.3f}")
                    detail = ", ".join(parts)
                    if len(group_defs) > 1:
                        line = f"    {struct_name:<20} Combined = {combined_rmsds[i]:.3f} A  [{detail}]"
                    else:
                        line = f"    {struct_name:<20} RMSD = {combined_rmsds[i]:.3f} A"
                    print(line)
                    out_file.write(line + '\n')
        else:
            print("[WARNING] Could not compute conformational heterogeneity RMSD.")
    else:
        print("[WARNING] No group definitions found in refine.py. Skipping Part 1.")

    diagnose_structures(pdb_paths, [d['name'] for d in selected], atom_filter=atom_filter)

    # Part 2: RMSD to reference structure
    if ref_structure_path:
        print("\n" + "=" * 70)
        print("RMSD Analysis - Part 2: Comparison with reference structure")
        print("=" * 70)
        rmsd_ref = rmsd_to_reference(pdb_paths, ref_structure_path,
                                      atom_filter=atom_filter, cutoff=ref_cutoff)
        if rmsd_ref:
            print(f"  Reference PDB        : {ref_structure_path}")
            print(f"  Structures processed : {len([r for r in rmsd_ref['rmsds'] if r is not None])}")
            print(f"  Min RMSD             : {rmsd_ref['min']:.3f} A")
            print(f"  Max RMSD             : {rmsd_ref['max']:.3f} A")
            print(f"  Mean RMSD            : {rmsd_ref['mean']:.3f} A")
            print(f"  Std Dev              : {rmsd_ref['std']:.3f} A")
            for i, r in enumerate(rmsd_ref['rmsds']):
                name = selected[i]['name']
                n_match = rmsd_ref['matched_counts'][i]
                if r is not None:
                    print(f"    {name:<20} RMSD = {r:.3f} A  (matched atoms: {n_match})")
                else:
                    print(f"    {name:<20} SKIPPED  (matched atoms: {n_match})")

            with open(out_path, 'a') as out_file:
                out_file.write('\n' + '=' * 70 + '\n')
                out_file.write("RMSD Analysis - Part 2: Comparison with reference structure\n")
                out_file.write('=' * 70 + '\n')
                out_file.write(f"  Reference PDB        : {ref_structure_path}\n")
                out_file.write(f"  Structures processed : {len([r for r in rmsd_ref['rmsds'] if r is not None])}\n")
                out_file.write(f"  Min RMSD             : {rmsd_ref['min']:.3f} A\n")
                out_file.write(f"  Max RMSD             : {rmsd_ref['max']:.3f} A\n")
                out_file.write(f"  Mean RMSD            : {rmsd_ref['mean']:.3f} A\n")
                out_file.write(f"  Std Dev              : {rmsd_ref['std']:.3f} A\n")
                for i, r in enumerate(rmsd_ref['rmsds']):
                    name = selected[i]['name']
                    n_match = rmsd_ref['matched_counts'][i]
                    if r is not None:
                        out_file.write(f"    {name:<20} RMSD = {r:.3f} A  (matched atoms: {n_match})\n")
                    else:
                        out_file.write(f"    {name:<20} SKIPPED  (matched atoms: {n_match})\n")
    else:
        print("\n[INFO] No reference structure provided; skipping Part 2 (reference comparison).")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Report saved to: {out_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
