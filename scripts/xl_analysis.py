#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Cross-linking-based Protein Structure Analysis
Domain partitioning, interaction interface clustering, and visualization
"""

import os
import sys
import math
import warnings
import numpy as np
from collections import defaultdict
from Bio.PDB import PDBParser, DSSP, PDBIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# Suppress Bio.PDB.DSSP mmCIF parsing warning (DSSP v4.x tries mmCIF first then falls back to PDB)
warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")

# ==================== Parameter Configuration ====================
# Working directory (script will chdir here)
WORK_DIR = "."

# Input file paths
LINK_FILE = os.environ.get('XPIE_LINK_FILE', 'link.dat')  # Cross-link information file
PDB_DIR = os.environ.get('XPIE_PDB_DIR', '.')             # PDB file directory

# Output directory
OUTPUT_DIR = "interface-define"

# Cross-linker arm length (Angstrom)
CROSS_LINKER_LENGTH = 15

# Disordered region detection parameters
DISORDERED_MIN_LENGTH = 30      # Minimum disordered region length (consecutive disordered residues)
ORDERED_GAP_TOLERANCE = 10      # Maximum allowed ordered residues inside a disordered region
USE_LOOP_FLEXIBILITY = False    # If True, detect short flexible loops containing XL sites

# DSSP executable path (leave empty to auto-detect mkdssp in PATH)
DSSP_PATH = ""
# ==================== End of Parameter Configuration ====================


# ==================== Utility Functions ====================

def infer_element(atom_name):
    """Infer element symbol from atom name (for fixing PDB element column)"""
    atom_name = atom_name.strip()
    if not atom_name:
        return ''
    # Hydrogen atoms
    if atom_name[0] == 'H':
        return 'H'
    # Common metal ions
    metal_map = {
        'FE': 'FE', 'ZN': 'ZN', 'MG': 'MG', 'MN': 'MN', 'CU': 'CU',
        'CO': 'CO', 'NI': 'NI', 'CD': 'CD', 'HG': 'HG', 'PB': 'PB',
        'NA': 'NA', 'K': 'K', 'CA': 'CA', 'CL': 'CL', 'BR': 'BR',
        'I': 'I', 'SR': 'SR', 'BA': 'BA'
    }
    two = atom_name[:2].upper()
    if two in metal_map:
        return metal_map[two]
    # Common protein atoms: take first character
    return atom_name[0].upper()


def fix_pdb_element_column(input_path, output_path):
    """Fix PDB file by adding/completing element column (cols 76-78)"""
    has_model = False
    has_endmdl = False
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            if line.startswith('MODEL'):
                has_model = True
            if line.startswith('ENDMDL'):
                has_endmdl = True
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Check if element column is empty
                if len(line) < 78 or line[76:78].strip() == '':
                    atom_name = line[12:16].strip()
                    elem = infer_element(atom_name)
                    # Ensure line has sufficient length
                    line = line.rstrip('\n')
                    if len(line) < 78:
                        line = line.ljust(78)
                    # Replace columns 76-78
                    line = line[:76] + elem.rjust(2) + line[78:]
                    line += '\n'
            fout.write(line)
        # DSSP requires ENDMDL when MODEL is present; add it if missing
        if has_model and not has_endmdl:
            fout.write("ENDMDL\n")


def ensure_dir(path):
    """Ensure directory exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def parse_link_file(filepath):
    """
    Parse link.dat file
    Returns: list of tuples [(prot1, res1, prot2, res2), ...]
    """
    links = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                prot1, res1, prot2, res2 = parts[0], int(parts[1]), parts[2], int(parts[3])
                links.append((prot1, res1, prot2, res2))
    return links


def get_structure(pdb_path):
    """Parse PDB file using Bio.PDB, return structure object and residue list"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    return structure


def get_residue_info(structure):
    """
    Get information for all residues in the structure
    Returns: dict {res_id: {'plddt': float, 'atoms': dict}}
    res_id = (chain_id, res_seq, res_name)
    """
    res_info = {}
    for model in structure:
        for chain in model:
            for residue in chain.get_list():
                res_seq = residue.get_id()[1]
                chain_id = chain.id
                res_name = residue.resname
                res_key = (chain_id, res_seq, res_name)
                
                atoms = {}
                plddts = []
                for atom in residue.get_list():
                    coord = atom.get_coord()
                    bfactor = atom.get_bfactor()
                    atoms[atom.get_id()] = {
                        'coord': coord,
                        'bfactor': bfactor
                    }
                    plddts.append(bfactor)
                
                avg_plddt = np.mean(plddts) if plddts else 0.0
                res_info[res_key] = {
                    'plddt': avg_plddt,
                    'atoms': atoms,
                    'res_name': res_name
                }
    return res_info


def get_residue_list(structure):
    """
    Get sequentially ordered residue list
    Returns: list of (chain_id, res_seq, res_name)
    """
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain.get_list():
                res_seq = residue.get_id()[1]
                chain_id = chain.id
                res_name = residue.resname
                residues.append((chain_id, res_seq, res_name))
    return residues


def has_plddt(residue_info):
    """Check if valid pLDDT information exists (not all zeros)"""
    plddts = [info['plddt'] for info in residue_info.values()]
    if not plddts:
        return False
    return any(p > 0 for p in plddts)


def get_dssp_path():
    """Get DSSP executable path"""
    if DSSP_PATH and os.path.exists(DSSP_PATH):
        return DSSP_PATH
    # Try mkdssp in system PATH
    import shutil
    dssp_cmd = shutil.which('mkdssp')
    if dssp_cmd:
        return dssp_cmd
    return None


def calculate_secondary_structure(structure, pdb_path):
    """
    Calculate secondary structure using DSSP
    Returns: dict {(chain_id, res_seq): ss_type}
    ss_type: DSSP secondary structure type character
    """
    dssp_path = get_dssp_path()
    if dssp_path is None:
        raise RuntimeError("DSSP executable (mkdssp) not found. Please install DSSP or set DSSP_PATH.")
    
    # Fix PDB file element column (must use .pdb extension for DSSP to recognize)
    import tempfile
    tmp_dir = tempfile.gettempdir()
    base_name = os.path.splitext(os.path.basename(pdb_path))[0]
    fixed_pdb = os.path.join(tmp_dir, base_name + '_fixed.pdb')
    fix_pdb_element_column(pdb_path, fixed_pdb)
    
    ss_dict = {}
    for model in structure:
        try:
            dssp = DSSP(model, fixed_pdb, dssp=dssp_path)
            for key in dssp.keys():
                chain_id, res_id = key
                res_seq = res_id[1]
                ss = dssp[key][2]  # secondary structure type
                ss_dict[(chain_id, res_seq)] = ss
        except Exception as e:
            print(f"Warning: DSSP calculation failed ({e})，using fallback...")
            # Fallback: mark all residues as unstructured
            for chain in model:
                for residue in chain.get_list():
                    res_seq = residue.get_id()[1]
                    ss_dict[(chain.id, res_seq)] = '-'
    return ss_dict


def is_disordered_by_ss(ss_type):
    """
    Determine disorder from secondary structure type for linker detection.
    In DSSP, '-' means coil (no structure), considered disordered.
    """
    # H=helix, B=bridge, E=strand, G=3-helix, I=5-helix, T=turn, S=bend, -=coil
    return ss_type == '-'


def is_unstructured_by_ss(ss_type):
    """
    Determine unstructured region for loop detection.
    Coil ('-'), turn ('T'), and bend ('S') are considered unstructured.
    """
    return ss_type in ('-', 'T', 'S')


def find_disordered_regions(disordered_flags, min_length=30, gap_tolerance=10):
    """
    Find disordered regions
    disordered_flags: list of bool, True means disordered
    min_length: minimum strictly disordered residue count
    gap_tolerance: maximum consecutive ordered insertions allowed
    Returns: list of (start_idx, end_idx) indices in residue list
    """
    n = len(disordered_flags)
    if n == 0:
        return []
    
    regions = []
    i = 0
    while i < n:
        if not disordered_flags[i]:
            i += 1
            continue
        
        # Start from a disordered residue and expand backward
        start = i
        j = i + 1
        gap = 0
        last_disordered = i
        
        while j < n:
            if disordered_flags[j]:
                gap = 0
                last_disordered = j
            else:
                gap += 1
                if gap > gap_tolerance:
                    break
            j += 1
        
        end = last_disordered
        strict_disordered_count = sum(disordered_flags[start:end + 1])
        
        if strict_disordered_count >= min_length:
            regions.append((start, end))
            i = end + 1
        else:
            i += 1
    
    return regions


def find_loops_around_xl_sites(disordered_flags, residue_list, xl_sites):
    """
    For each XL site that lies in a coil region, find a 5-residue loop window
    centered around the XL site, within a contiguous coil stretch of >=5 residues.
    Returns: list of (start_idx, end_idx), each spanning exactly 5 residues
             (or fewer only if the protein terminus is reached).
    """
    if not xl_sites:
        return []
    
    n = len(disordered_flags)
    loops = []
    
    # Find all strictly contiguous coil regions
    coil_regions = []
    i = 0
    while i < n:
        if not disordered_flags[i]:
            i += 1
            continue
        start = i
        j = i + 1
        while j < n and disordered_flags[j]:
            j += 1
        coil_regions.append((start, j - 1))
        i = j
    
    for start_idx, end_idx in coil_regions:
        length = end_idx - start_idx + 1
        if length < 5:
            continue
        
        start_seq = residue_list[start_idx][1]
        end_seq = residue_list[end_idx][1]
        # XL sites that fall inside this coil region
        region_xl = sorted([r for r in xl_sites if start_seq <= r <= end_seq])
        
        for xl_res in region_xl:
            # Find the index of xl_res within residue_list
            xl_idx = None
            for idx in range(start_idx, end_idx + 1):
                if residue_list[idx][1] == xl_res:
                    xl_idx = idx
                    break
            if xl_idx is None:
                continue
            
            # Try to center the XL site in a 5-residue window
            window_start = xl_idx - 2
            window_end = xl_idx + 2
            
            if window_start < start_idx:
                window_start = start_idx
                window_end = min(start_idx + 4, end_idx)
            elif window_end > end_idx:
                window_end = end_idx
                window_start = max(end_idx - 4, start_idx)
            
            # Represent loop by residue sequence range for deduplication
            loop_seq_range = (residue_list[window_start][1], residue_list[window_end][1])
            if loop_seq_range not in [(residue_list[s][1], residue_list[e][1]) for s, e in loops]:
                loops.append((window_start, window_end))
    
    return loops


def classify_domains(residue_list, disordered_regions):
    """
    Partition domains based on disordered regions
    Returns: list of (start_res_seq, end_res_seq)
    """
    if not residue_list:
        return []
    
    # Get all residue sequence numbers
    all_seqs = [res[1] for res in residue_list]
    min_seq = min(all_seqs)
    max_seq = max(all_seqs)
    
    if not disordered_regions:
        # No disordered regions, entire protein is one domain
        return [(min_seq, max_seq)]
    
    # Convert disordered regions to residue sequence ranges
    disordered_seq_ranges = []
    for start_idx, end_idx in disordered_regions:
        disordered_seq_ranges.append((residue_list[start_idx][1], residue_list[end_idx][1]))
    
    # Sort by sequence number
    disordered_seq_ranges.sort()
    
    # Calculate domain intervals (regions between disordered zones)
    domains = []
    current_start = min_seq
    
    for d_start, d_end in disordered_seq_ranges:
        if current_start < d_start:
            domains.append((current_start, d_start - 1))
        current_start = d_end + 1
    
    if current_start <= max_seq:
        domains.append((current_start, max_seq))
    
    return domains


def compute_radius_of_gyration(structure):
    """
    Calculate protein radius of gyration (Rg)
    Use all heavy atoms (non-hydrogen)
    """
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain.get_list():
                for atom in residue.get_list():
                    # Skip hydrogen atoms
                    if atom.element == 'H' or atom.get_id().startswith('H'):
                        continue
                    coords.append(atom.get_coord())
    
    if not coords:
        return 0.0
    
    coords = np.array(coords)
    center = np.mean(coords, axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))
    return rg


def get_nz_coord(structure, chain_id, res_seq):
    """Get NZ atom coordinate for specified residue"""
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain.get_list():
                if residue.get_id()[1] == res_seq:
                    for atom in residue.get_list():
                        if atom.get_id() == 'NZ':
                            return atom.get_coord()
    return None


def get_nz_coords(structure, res_list):
    """Get NZ atom coordinates for a list of residue numbers.
    If NZ atom is absent, fallback to backbone N atom regardless of residue type or position."""
    coords = {}
    for res_seq in res_list:
        for model in structure:
            for chain in model:
                for residue in chain.get_list():
                    if residue.get_id()[1] == res_seq:
                        # First try NZ atom (standard Lys side chain)
                        nz_coord = None
                        n_coord = None
                        for atom in residue.get_list():
                            if atom.get_id() == 'NZ':
                                nz_coord = atom.get_coord()
                            elif atom.get_id() == 'N':
                                n_coord = atom.get_coord()
                        if nz_coord is not None:
                            coords[res_seq] = nz_coord
                        elif n_coord is not None:
                            coords[res_seq] = n_coord
                        break
    return coords


def compute_xl_site_stats(prot, links, protein_data):
    """
    计算蛋白在交联中的统计信息，用于选择参考/锚点蛋白。
    返回: (参与交联的不同氨基酸残基数, 交联位点间最大距离)
    """
    res_set = set()
    for p1, r1, p2, r2 in links:
        if p1 == prot:
            res_set.add(r1)
        elif p2 == prot:
            res_set.add(r2)

    unique_count = len(res_set)

    max_dist = 0.0
    if unique_count >= 2:
        struct = protein_data[prot]['structure']
        coords_dict = get_nz_coords(struct, sorted(res_set))
        coords = list(coords_dict.values())
        n = len(coords)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist > max_dist:
                    max_dist = dist

    return unique_count, max_dist


def cluster_sites_by_distance(site_coords, threshold, loop_region_lengths=None):
    """
    Cluster sites based on distance threshold
    site_coords: dict {site_id: coord_array}
    threshold: distance threshold
    loop_region_lengths: dict {site_id: loop_region_length}. If a site is in a
                         loop region, the effective threshold for pairs involving
                         this site is increased by loop_region_length * 4.
    Returns: dict {site_id: class_id}
    """
    sites = list(site_coords.keys())
    n = len(sites)
    if n == 0:
        return {}
    if n == 1:
        return {sites[0]: 1}
    
    # Union-Find
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    
    # Compute pairwise distances, merge if below effective threshold
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(site_coords[sites[i]] - site_coords[sites[j]])
            effective_threshold = threshold
            if loop_region_lengths:
                if sites[i] in loop_region_lengths:
                    effective_threshold = max(effective_threshold,
                                              threshold + loop_region_lengths[sites[i]] * 4)
                if sites[j] in loop_region_lengths:
                    effective_threshold = max(effective_threshold,
                                              threshold + loop_region_lengths[sites[j]] * 4)
            if dist < effective_threshold:
                union(i, j)
    
    # Assign class IDs
    class_map = {}
    class_id_map = {}
    next_class_id = 1
    for i in range(n):
        root = find(i)
        if root not in class_id_map:
            class_id_map[root] = next_class_id
            next_class_id += 1
        class_map[sites[i]] = class_id_map[root]
    
    return class_map


def compute_corrected_avg_distance(coords, target_avg=None, loop_region_lengths=None, max_correction_per_flex_site=4.0):
    """
    Compute average pairwise distance with flexible correction.
    
    If target_avg is provided and loop_region_lengths is not None:
        For each pair involving flexible sites, we can add OR subtract up to
        (flex_count * max_correction_per_flex_site) to minimize |avg_B - target_avg|.
        The total correction across all pairs is chosen optimally within the
        feasible range to get as close to target_avg as possible.
    
    Otherwise, return raw average distance.
    """
    sites = list(coords.keys())
    n = len(sites)
    if n < 2:
        return 0.0

    total_raw = 0.0
    count = 0
    min_total_shift = 0.0   # most negative total correction
    max_total_shift = 0.0   # most positive total correction
    
    for i in range(n):
        for j in range(i + 1, n):
            raw_dist = np.linalg.norm(coords[sites[i]] - coords[sites[j]])
            total_raw += raw_dist
            count += 1
            
            if loop_region_lengths:
                flex_count = 0
                if sites[i] in loop_region_lengths:
                    flex_count += loop_region_lengths[sites[i]]
                if sites[j] in loop_region_lengths:
                    flex_count += loop_region_lengths[sites[j]]
                max_corr = flex_count * max_correction_per_flex_site
                min_total_shift -= max_corr
                max_total_shift += max_corr
    
    avg_raw = total_raw / count if count > 0 else 0.0
    
    if target_avg is None or count == 0 or not loop_region_lengths:
        return avg_raw
    
    # We want avg_raw + total_shift/count ≈ target_avg
    # => total_shift ≈ count * (target_avg - avg_raw)
    desired_shift = count * (target_avg - avg_raw)
    
    # Clamp to feasible range [min_total_shift, max_total_shift]
    actual_shift = max(min_total_shift, min(max_total_shift, desired_shift))
    
    return avg_raw + actual_shift / count


def find_outliers_to_match(b_sites, b_coords, target_avg, loop_region_lengths=None, diff_threshold=10.0):
    """
    Iteratively remove B site outliers until |avg_B - target_avg| <= diff_threshold.
    At each step, remove the site whose removal gives the largest reduction in diff.
    Returns: (remaining_sites, list_of_outlier_sites)
    """
    current_sites = list(b_sites)
    outliers = []

    while len(current_sites) >= 3:
        current_coords = {s: b_coords[s] for s in current_sites if s in b_coords}
        if len(current_coords) < 2:
            break

        current_avg = compute_corrected_avg_distance(current_coords, target_avg=target_avg, loop_region_lengths=loop_region_lengths)
        current_diff = abs(current_avg - target_avg)

        if current_diff <= diff_threshold:
            break

        # Try removing each site to find the best improvement
        best_diff = current_diff
        best_site = None

        for s in current_sites:
            test_sites = [x for x in current_sites if x != s]
            if len(test_sites) < 2:
                continue
            test_coords = {x: b_coords[x] for x in test_sites if x in b_coords}
            if len(test_coords) < 2:
                continue
            test_avg = compute_corrected_avg_distance(test_coords, target_avg=target_avg, loop_region_lengths=loop_region_lengths)
            test_diff = abs(test_avg - target_avg)
            if test_diff < best_diff:
                best_diff = test_diff
                best_site = s

        if best_site is None:
            # No single removal can improve the match
            break

        outliers.append(best_site)
        current_sites.remove(best_site)

        if best_diff <= diff_threshold:
            break

    return current_sites, outliers


def cluster_b_by_avg_distance(a_coords_in_class, b_sites_in_class, b_coords,
                               loop_region_lengths=None, diff_threshold=10.0):
    """
    Cluster B sites by comparing average pairwise distance with A cluster.
    
    1. Compute avg pairwise distance among A sites
    2. Compute avg pairwise distance among B sites (with flexibility correction)
    3. If |avg_A - avg_B| <= diff_threshold: all B sites -> 1 cluster
    4. If > diff_threshold: iteratively remove outliers until diff <= threshold
    
    Returns: list of B-site sets, each set representing one cluster.
    """
    b_coords_in_class = {b: b_coords[b] for b in b_sites_in_class if b in b_coords}
    if not b_coords_in_class:
        return []
    
    a_sites = list(a_coords_in_class.keys())
    b_sites = list(b_coords_in_class.keys())
    
    # If either side has only 0 or 1 site, no meaningful distance comparison
    if len(a_sites) <= 1 or len(b_sites) <= 1:
        return [set(b_sites_in_class)]
    
    avg_dist_a = compute_corrected_avg_distance(a_coords_in_class, target_avg=None, loop_region_lengths=None)
    avg_dist_b = compute_corrected_avg_distance(b_coords_in_class, target_avg=avg_dist_a, loop_region_lengths=loop_region_lengths)
    diff = abs(avg_dist_a - avg_dist_b)
    
    if diff <= diff_threshold:
        return [set(b_sites_in_class)]
    else:
        remaining, outliers = find_outliers_to_match(
            b_sites_in_class, b_coords, avg_dist_a,
            loop_region_lengths=loop_region_lengths, diff_threshold=diff_threshold
        )
        clusters = []
        if remaining:
            clusters.append(set(remaining))
        for outlier in outliers:
            clusters.append({outlier})
        return clusters


def cluster_b_with_singleton_merge(a_class_to_b_sites, a_class_map, a_coords, b_coords,
                                    loop_region_lengths, diff_threshold=10.0, label_prefix="A-class"):
    """
    Cluster B-protein sites with cross-A-class merging.
    
    When multiple A-classes each produce exactly 1 B-cluster, merge them and
    evaluate if the merged avg distances match within threshold. If so, assign
    all their B-sites to the same B-state.
    
    The rationale: if A-sites are spatially separated into multiple clusters
    (possible interaction interfaces), but each cluster's corresponding B-sites
    can be coherently explained by a single binding mode, then all B-sites
    should be assigned to the same state.
    
    Returns: list of (a_cls_list, b_cluster_set)
    """
    # First, compute B-clusters for each A-class individually
    class_results = {}  # {a_cls: list of b_cluster_sets}
    
    for a_cls, b_list in a_class_to_b_sites.items():
        b_sites_in_cls = sorted(set(b_list))
        a_sites = [a_res for a_res, cls in a_class_map.items() if cls == a_cls]
        a_coords_in_cls = {a: a_coords[a] for a in a_sites if a in a_coords}
        
        b_clusters = cluster_b_by_avg_distance(
            a_coords_in_cls, b_sites_in_cls, b_coords,
            loop_region_lengths=loop_region_lengths, diff_threshold=diff_threshold
        )
        class_results[a_cls] = b_clusters
    
    # Identify A-classes that produce exactly 1 B-cluster (merge candidates)
    single_cluster_classes = [a_cls for a_cls, clusters in class_results.items() if len(clusters) == 1]
    multi_cluster_classes = [a_cls for a_cls, clusters in class_results.items() if len(clusters) != 1]
    
    results = []
    
    # Try merging all single-cluster A-classes
    if len(single_cluster_classes) >= 2:
        merged_b_sites = []
        for a_cls in single_cluster_classes:
            unique_b = list(set(a_class_to_b_sites[a_cls]))
            merged_b_sites.extend(unique_b)
        
        merged_b_coords = {b: b_coords[b] for b in merged_b_sites if b in b_coords}
        
        # Compute A-class centroids: each cluster is represented by the mean NZ position
        centroids = {}
        for a_cls in single_cluster_classes:
            a_sites = [a_res for a_res, cls in a_class_map.items() if cls == a_cls]
            coords = [a_coords[s] for s in a_sites if s in a_coords]
            if len(coords) == 0:
                continue
            elif len(coords) == 1:
                centroids[a_cls] = coords[0]
            else:
                centroids[a_cls] = np.mean(coords, axis=0)
        
        # Compute avg_A from centroid-centroid pairwise distances
        n_cls = len(centroids)
        avg_A = 0.0
        if n_cls >= 2:
            total = 0.0
            count = 0
            cls_list = list(centroids.keys())
            for i in range(n_cls):
                for j in range(i + 1, n_cls):
                    dist = np.linalg.norm(centroids[cls_list[i]] - centroids[cls_list[j]])
                    total += dist
                    count += 1
            avg_A = total / count if count > 0 else 0.0
        
        can_merge = False
        unique_b_count = len(set(merged_b_sites))
        if n_cls >= 2 and unique_b_count >= 2:
            avg_B = compute_corrected_avg_distance(merged_b_coords, target_avg=avg_A, loop_region_lengths=loop_region_lengths)
            diff = abs(avg_A - avg_B)
            print(f"    {label_prefix} cross-class merge test {single_cluster_classes}: avg_A={avg_A:.2f}, avg_B={avg_B:.2f}, diff={diff:.2f}")
            if diff <= diff_threshold:
                can_merge = True
        else:
            # Either too few A-clusters or all point to same B-site -> merge by default
            can_merge = True
        
        if can_merge:
            # All single-cluster classes share the same B-state
            combined_b_sites = set()
            for a_cls in single_cluster_classes:
                combined_b_sites.update(class_results[a_cls][0])
            results.append((single_cluster_classes, combined_b_sites))
        else:
            for a_cls in single_cluster_classes:
                results.append(([a_cls], class_results[a_cls][0]))
    elif len(single_cluster_classes) == 1:
        a_cls = single_cluster_classes[0]
        results.append(([a_cls], class_results[a_cls][0]))
    
    # Handle multi-cluster classes (keep as-is)
    for a_cls in multi_cluster_classes:
        for b_cluster in class_results[a_cls]:
            results.append(([a_cls], b_cluster))
    
    return results


def analyze_protein(prot_name, pdb_dir, output_dir, xl_sites=None):
    """
    Analyze domains for a single protein.
    Linkers (disordered >= 30) are used for domain splitting.
    Loops (disordered >= 5 and < 30 containing XL sites) are output separately
    and do not split domains.
    Returns: domains, linker_regions, loop_regions, residue_list, structure, ss_dict
    """
    pdb_path = os.path.join(pdb_dir, f"{prot_name}.pdb")
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    
    structure = get_structure(pdb_path)
    residue_list = get_residue_list(structure)
    res_info = get_residue_info(structure)
    
    # Determine disordered/unstructured residues using DSSP secondary structure
    disordered_flags = []
    unstructured_flags = []
    print(f"  [{prot_name}] Using DSSP secondary structure for disorder detection")
    ss_dict = calculate_secondary_structure(structure, pdb_path)
    for res in residue_list:
        chain_id, res_seq, res_name = res
        ss = ss_dict.get((chain_id, res_seq), '-')
        disordered_flags.append(is_disordered_by_ss(ss))
        unstructured_flags.append(is_unstructured_by_ss(ss))
    
    # Find linker regions (strictly disordered >= 30) for domain splitting
    linker_regions = find_disordered_regions(
        disordered_flags,
        min_length=DISORDERED_MIN_LENGTH,
        gap_tolerance=ORDERED_GAP_TOLERANCE
    )
    
    # Find loop regions only when loop flexibility is enabled
    loop_regions = []
    if USE_LOOP_FLEXIBILITY:
        raw_loop_regions = find_loops_around_xl_sites(
            unstructured_flags, residue_list, xl_sites
        )
        
        # 1. Exclude loops that fall completely inside any linker region
        filtered = []
        for ls, le in raw_loop_regions:
            inside_linker = False
            for lks, lke in linker_regions:
                if lks <= ls and le <= lke:
                    inside_linker = True
                    break
            if not inside_linker:
                filtered.append((ls, le))
        
        # 2. Merge overlapping loops
        if filtered:
            seq_ranges = [(residue_list[s][1], residue_list[e][1]) for s, e in filtered]
            seq_ranges.sort()
            merged = [list(seq_ranges[0])]
            for s, e in seq_ranges[1:]:
                if s <= merged[-1][1]:  # overlap or adjacent
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            # Convert back to indices
            seq_to_idx = {residue_list[i][1]: i for i in range(len(residue_list))}
            for s, e in merged:
                if s in seq_to_idx and e in seq_to_idx:
                    loop_regions.append((seq_to_idx[s], seq_to_idx[e]))
    
    # Partition domains based on linker regions only
    domains = classify_domains(residue_list, linker_regions)
    
    # Output domain information
    domain_file = os.path.join(output_dir, f"{prot_name}_domains.txt")
    with open(domain_file, 'w') as f:
        if domains:
            ranges = " ".join([f"{s}-{e}" for s, e in domains])
            f.write(f"{prot_name}:{ranges}\n")
        else:
            f.write(f"{prot_name}:\n")
    
    # Output linker region information
    linker_file = os.path.join(output_dir, f"{prot_name}_linker.txt")
    with open(linker_file, 'w') as f:
        if linker_regions:
            ranges = " ".join([f"{residue_list[s][1]}-{residue_list[e][1]}" for s, e in linker_regions])
            f.write(f"{prot_name}:{ranges}\n")
        else:
            f.write(f"{prot_name}:\n")
    
    # Output loop region information
    loop_file = os.path.join(output_dir, f"{prot_name}_loop.txt")
    with open(loop_file, 'w') as f:
        if loop_regions:
            ranges = " ".join([f"{residue_list[s][1]}-{residue_list[e][1]}" for s, e in loop_regions])
            f.write(f"{prot_name}:{ranges}\n")
        else:
            f.write(f"{prot_name}:\n")
    
    return domains, linker_regions, loop_regions, residue_list, structure, ss_dict


def analyze_interaction(prot_a, prot_b, links, pdb_dir, output_dir, cross_linker_length, loop_region_dict=None):
    """
    Analyze interaction interface clustering for a protein pair
    prot_a: larger protein name
    prot_b: smaller protein name
    links: list of (a_res, b_res)
    loop_region_dict: {prot: {res_seq: loop_region_length}} for flexibility correction
    """
    pdb_a = os.path.join(pdb_dir, f"{prot_a}.pdb")
    pdb_b = os.path.join(pdb_dir, f"{prot_b}.pdb")
    
    struct_a = get_structure(pdb_a)
    struct_b = get_structure(pdb_b)
    
    # Calculate B protein radius of gyration
    rg_b = compute_radius_of_gyration(struct_b)
    print(f"  [{prot_a}-{prot_b}] Radius of gyration of {prot_b}: {rg_b:.2f} A")
    
    # A protein cross-link site NZ coordinates
    a_sites = sorted(set([a_res for a_res, b_res in links]))
    a_coords = {}
    for a_res in a_sites:
        # Find chain containing this residue
        coord = None
        for model in struct_a:
            for chain in model:
                for residue in chain.get_list():
                    if residue.get_id()[1] == a_res:
                        for atom in residue.get_list():
                            if atom.get_id() == 'NZ':
                                coord = atom.get_coord()
                                break
                    if coord is not None:
                        break
                if coord is not None:
                    break
            if coord is not None:
                break
        if coord is not None:
            a_coords[a_res] = coord
        else:
            print(f"  Warning: NZ atom not found in {prot_a} residue {a_res}")
    
    # B protein cross-link site NZ coordinates
    b_sites = sorted(set([b_res for a_res, b_res in links]))
    b_coords = {}
    for b_res in b_sites:
        coord = None
        for model in struct_b:
            for chain in model:
                for residue in chain.get_list():
                    if residue.get_id()[1] == b_res:
                        for atom in residue.get_list():
                            if atom.get_id() == 'NZ':
                                coord = atom.get_coord()
                                break
                    if coord is not None:
                        break
                if coord is not None:
                    break
            if coord is not None:
                break
        if coord is not None:
            b_coords[b_res] = coord
        else:
            print(f"  Warning: NZ atom not found in {prot_b} residue {b_res}")
    
    # Cluster A protein: distance < 2 * Rg_B (unchanged)
    a_class_map = cluster_sites_by_distance(a_coords, rg_b * 2)
    print(f"  [{prot_a}-{prot_b}] Protein {prot_a} interface clustered into {len(set(a_class_map.values()))} states")
    
    # Build mapping from A class -> B site list
    a_class_to_b_sites = defaultdict(list)
    for a_res, b_res in links:
        if a_res in a_class_map:
            a_class_to_b_sites[a_class_map[a_res]].append(b_res)
    
    # Check cross-class merge prerequisites
    a_length = len(get_residue_list(struct_a))
    b_length = len(get_residue_list(struct_b))
    rg_a = compute_radius_of_gyration(struct_a)
    can_cross_merge = (abs(rg_a - rg_b) <= 5.0) and (abs(a_length - b_length) <= 50)
    
    # New B clustering logic based on average distance comparison with A cluster
    prot_b_loop_lengths = loop_region_dict.get(prot_b, {}) if loop_region_dict else {}
    b_class_map = {}  # {(a_class_id, b_res): global_b_cls}
    b_class_sites = {}  # {global_b_cls: set(b_res)}
    b_next_global_class = 1
    
    if can_cross_merge:
        print(f"  [{prot_a}-{prot_b}] Cross-class merge enabled (Rg diff={abs(rg_a-rg_b):.2f} A, length diff={abs(a_length-b_length)} aa)")
        merged_results = cluster_b_with_singleton_merge(
            a_class_to_b_sites, a_class_map, a_coords, b_coords,
            loop_region_lengths=prot_b_loop_lengths, diff_threshold=5.0, label_prefix="A-class"
        )
    else:
        print(f"  [{prot_a}-{prot_b}] Cross-class merge disabled (Rg diff={abs(rg_a-rg_b):.2f} A, length diff={abs(a_length-b_length)} aa)")
        merged_results = []
        for a_class_id in sorted(a_class_to_b_sites.keys()):
            b_sites_in_class = sorted(set(a_class_to_b_sites[a_class_id]))
            a_sites_in_class = [a_res for a_res, cls in a_class_map.items() if cls == a_class_id]
            a_coords_in_class = {a: a_coords[a] for a in a_sites_in_class if a in a_coords}
            b_clusters = cluster_b_by_avg_distance(
                a_coords_in_class, b_sites_in_class, b_coords,
                loop_region_lengths=prot_b_loop_lengths, diff_threshold=5.0
            )
            for b_cluster in b_clusters:
                merged_results.append(([a_class_id], b_cluster))
    
    for a_cls_list, b_cluster in merged_results:
        if not b_cluster:
            continue
        
        # Compute metrics for reporting
        if len(a_cls_list) >= 2 and can_cross_merge:
            # Centroid-based avg_A for cross-class merge
            centroids = {}
            for a_cls in a_cls_list:
                a_sites = [a_res for a_res, cls in a_class_map.items() if cls == a_cls]
                coords = [a_coords[s] for s in a_sites if s in a_coords]
                if len(coords) == 1:
                    centroids[a_cls] = coords[0]
                else:
                    centroids[a_cls] = np.mean(coords, axis=0)
            n = len(centroids)
            total = 0.0
            count = 0
            c_list = list(centroids.keys())
            for i in range(n):
                for j in range(i + 1, n):
                    total += np.linalg.norm(centroids[c_list[i]] - centroids[c_list[j]])
                    count += 1
            avg_dist_a = total / count if count > 0 else 0.0
        else:
            # Standard pairwise avg_A within single class
            merged_a_sites = []
            for a_cls in a_cls_list:
                merged_a_sites.extend([a_res for a_res, cls in a_class_map.items() if cls == a_cls])
            merged_a_coords = {a: a_coords[a] for a in merged_a_sites if a in a_coords}
            avg_dist_a = compute_corrected_avg_distance(merged_a_coords, target_avg=None, loop_region_lengths=None)
        
        merged_b_coords = {b: b_coords[b] for b in b_cluster if b in b_coords}
        avg_dist_b = compute_corrected_avg_distance(merged_b_coords, target_avg=avg_dist_a, loop_region_lengths=prot_b_loop_lengths)
        diff = abs(avg_dist_a - avg_dist_b)
        print(f"    A-classes {a_cls_list}: avg_A={avg_dist_a:.2f}, avg_B={avg_dist_b:.2f}, diff={diff:.2f} -> 1 B-cluster")
        
        # Assign global class IDs
        b_class_sites[b_next_global_class] = set(b_cluster)
        for b_site in b_cluster:
            for a_cls in a_cls_list:
                b_class_map[(a_cls, b_site)] = b_next_global_class
        b_next_global_class += 1
    
    total_b_classes = len(set(b_class_map.values()))
    print(f"  [{prot_a}-{prot_b}] Protein {prot_b} interface clustered into {total_b_classes} states")
    
    # Output clustering results
    cluster_file = os.path.join(output_dir, f"{prot_a}_{prot_b}_clusters.txt")
    with open(cluster_file, 'w') as f:
        f.write(f"# Protein_A\tSite_A\tState_A\tProtein_B\tSite_B\tState_B\n")
        for a_res, b_res in links:
            a_cls = a_class_map.get(a_res, -1)
            b_cls = b_class_map.get((a_cls, b_res), -1)
            # State_A column is fixed to 1 (protein count ID), while State_B retains clustering ID
            f.write(f"{prot_a}\t{a_res}\t1\t{prot_b}\t{b_res}\t{b_cls}\n")
    
    return a_class_map, b_class_map, b_class_sites, a_coords, b_coords, rg_b


def analyze_anchor_system(anchor_prot, all_links, protein_data, cross_linker_length, loop_region_dict=None, disordered_region_dict=None):
    """
    Analyze interactions using anchor protein as reference.
    Anchor protein cross-link sites are clustered first (threshold = max Rg of other proteins).
    Each other protein is then clustered separately based on anchor interface classes
    (threshold = 2 * cross_linker_length).
    
    Returns: anchor_class_map, prot_class_maps, rg_map, anchor_links, other_links
    """
    if loop_region_dict is None:
        loop_region_dict = {}
    if disordered_region_dict is None:
        disordered_region_dict = {}
    # Separate links involving anchor vs not involving anchor
    anchor_links = []      # list of (anchor_res, other_prot, other_res)
    other_links = []       # list of (prot1, res1, prot2, res2) not involving anchor
    
    for prot1, res1, prot2, res2 in all_links:
        if prot1 == anchor_prot:
            anchor_links.append((res1, prot2, res2))
        elif prot2 == anchor_prot:
            anchor_links.append((res2, prot1, res1))
        else:
            other_links.append((prot1, res1, prot2, res2))
    
    # Get anchor protein NZ coordinates
    anchor_struct = protein_data[anchor_prot]['structure']
    anchor_sites = sorted(set([res for res, _, _ in anchor_links]))
    anchor_coords = get_nz_coords(anchor_struct, anchor_sites)
    
    # Compute Rg of all other proteins, use max as threshold for anchor clustering
    other_prots = sorted(set([prot for _, prot, _ in anchor_links]))
    max_rg = 0.0
    rg_map = {}
    for prot in other_prots:
        rg = compute_radius_of_gyration(protein_data[prot]['structure'])
        rg_map[prot] = rg
        max_rg = max(max_rg, rg)
    max_rg *= 2  # anchor clustering threshold = 2 * max partner Rg
    
    # Cluster anchor protein sites
    anchor_class_map = cluster_sites_by_distance(anchor_coords, max_rg,
                                                  loop_region_lengths=loop_region_dict.get(anchor_prot, {}))
    print(f"  Anchor {anchor_prot} interface clustered into {len(set(anchor_class_map.values()))} states (threshold: {max_rg:.2f} A)")
    
    # Cluster each other protein based on anchor interface classes
    prot_class_maps = {}  # {prot: {res: global_class}}
    
    for prot in other_prots:
        # Collect this protein's sites linked to anchor
        prot_anchor_links = [(a_res, o_res) for a_res, o_prot, o_res in anchor_links if o_prot == prot]
        prot_sites = sorted(set([o_res for _, o_res in prot_anchor_links]))
        
        # Get coordinates
        prot_struct = protein_data[prot]['structure']
        prot_coords = get_nz_coords(prot_struct, prot_sites)
        
        # Map anchor class -> this protein's sites
        anchor_class_to_prot_sites = defaultdict(list)
        for a_res, o_res in prot_anchor_links:
            if a_res in anchor_class_map:
                anchor_class_to_prot_sites[anchor_class_map[a_res]].append(o_res)
        
        # Check cross-class merge prerequisites
        anchor_length = len(protein_data[anchor_prot]['residues'])
        prot_length = len(protein_data[prot]['residues'])
        rg_anchor = compute_radius_of_gyration(anchor_struct)
        rg_prot = rg_map[prot]
        can_cross_merge = (abs(rg_anchor - rg_prot) <= 5.0) and (abs(anchor_length - prot_length) <= 50)
        
        # Cluster this protein's sites within each anchor class (new avg-distance logic)
        prot_loop_lengths = loop_region_dict.get(prot, {}) if loop_region_dict else {}
        prot_class_map = {}
        next_global_class = 1
        
        if can_cross_merge:
            print(f"    Cross-class merge enabled for {prot} (Rg diff={abs(rg_anchor-rg_prot):.2f} A, length diff={abs(anchor_length-prot_length)} aa)")
            merged_results = cluster_b_with_singleton_merge(
                anchor_class_to_prot_sites, anchor_class_map, anchor_coords, prot_coords,
                loop_region_lengths=prot_loop_lengths, diff_threshold=5.0, label_prefix="Anchor-class"
            )
        else:
            print(f"    Cross-class merge disabled for {prot} (Rg diff={abs(rg_anchor-rg_prot):.2f} A, length diff={abs(anchor_length-prot_length)} aa)")
            merged_results = []
            for a_class_id in sorted(anchor_class_to_prot_sites.keys()):
                sites_in_class = sorted(set(anchor_class_to_prot_sites[a_class_id]))
                a_sites_in_class = [a_res for a_res, cls in anchor_class_map.items() if cls == a_class_id]
                a_coords_in_class = {a: anchor_coords[a] for a in a_sites_in_class if a in anchor_coords}
                b_clusters = cluster_b_by_avg_distance(
                    a_coords_in_class, sites_in_class, prot_coords,
                    loop_region_lengths=prot_loop_lengths, diff_threshold=5.0
                )
                for b_cluster in b_clusters:
                    merged_results.append(([a_class_id], b_cluster))
        
        for a_cls_list, b_cluster in merged_results:
            if not b_cluster:
                continue
            
            # Compute metrics for reporting
            if len(a_cls_list) >= 2 and can_cross_merge:
                # Centroid-based avg_A for cross-class merge
                centroids = {}
                for a_cls in a_cls_list:
                    a_sites = [a_res for a_res, cls in anchor_class_map.items() if cls == a_cls]
                    coords = [anchor_coords[s] for s in a_sites if s in anchor_coords]
                    if len(coords) == 1:
                        centroids[a_cls] = coords[0]
                    else:
                        centroids[a_cls] = np.mean(coords, axis=0)
                n = len(centroids)
                total = 0.0
                count = 0
                c_list = list(centroids.keys())
                for i in range(n):
                    for j in range(i + 1, n):
                        total += np.linalg.norm(centroids[c_list[i]] - centroids[c_list[j]])
                        count += 1
                avg_dist_a = total / count if count > 0 else 0.0
            else:
                # Standard pairwise avg_A within single class
                merged_a_sites = []
                for a_cls in a_cls_list:
                    merged_a_sites.extend([a_res for a_res, cls in anchor_class_map.items() if cls == a_cls])
                merged_a_coords = {a: anchor_coords[a] for a in merged_a_sites if a in anchor_coords}
                avg_dist_a = compute_corrected_avg_distance(merged_a_coords, target_avg=None, loop_region_lengths=None)
            
            merged_b_coords = {b: prot_coords[b] for b in b_cluster if b in prot_coords}
            avg_dist_b = compute_corrected_avg_distance(merged_b_coords, target_avg=avg_dist_a, loop_region_lengths=prot_loop_lengths)
            diff = abs(avg_dist_a - avg_dist_b)
            print(f"    Anchor-classes {a_cls_list} -> {prot}: avg_A={avg_dist_a:.2f}, avg_B={avg_dist_b:.2f}, diff={diff:.2f} -> 1 cluster(s)")
            
            # Assign global class IDs
            for site in b_cluster:
                prot_class_map[site] = next_global_class
            next_global_class += 1
        
        prot_class_maps[prot] = prot_class_map
        print(f"  Protein {prot} clustered into {len(set(prot_class_map.values()))} states")
    
    return anchor_class_map, prot_class_maps, rg_map, anchor_links, other_links


def analyze_secondary_anchor(sec_anchor, sec_links, protein_data, cross_linker_length, loop_region_dict=None):
    """
    Analyze interactions for a secondary anchor protein.
    Similar to analyze_anchor_system but for a non-main anchor.
    
    sec_links: list of (sec_anchor_res, partner_prot, partner_res)
    Returns: sec_class_map, partner_class_maps
    """
    if loop_region_dict is None:
        loop_region_dict = {}
    
    # Get secondary anchor NZ coordinates
    sec_struct = protein_data[sec_anchor]['structure']
    sec_sites = sorted(set([res for res, _, _ in sec_links]))
    sec_coords = get_nz_coords(sec_struct, sec_sites)
    
    # Compute max Rg of partners
    partner_prots = sorted(set([prot for _, prot, _ in sec_links]))
    max_rg = 0.0
    for prot in partner_prots:
        rg = compute_radius_of_gyration(protein_data[prot]['structure'])
        max_rg = max(max_rg, rg)
    max_rg *= 2  # secondary anchor clustering threshold = 2 * max partner Rg
    
    # Cluster secondary anchor sites
    sec_class_map = cluster_sites_by_distance(sec_coords, max_rg,
                                               loop_region_lengths=loop_region_dict.get(sec_anchor, {}))
    
    # Cluster each partner based on secondary anchor interface classes
    partner_class_maps = {}
    
    for prot in partner_prots:
        prot_sec_links = [(s_res, p_res) for s_res, p_prot, p_res in sec_links if p_prot == prot]
        prot_sites = sorted(set([p_res for _, p_res in prot_sec_links]))
        prot_struct = protein_data[prot]['structure']
        prot_coords = get_nz_coords(prot_struct, prot_sites)
        
        sec_class_to_prot_sites = defaultdict(list)
        for s_res, p_res in prot_sec_links:
            if s_res in sec_class_map:
                sec_class_to_prot_sites[sec_class_map[s_res]].append(p_res)
        
        # Check cross-class merge prerequisites
        sec_length = len(protein_data[sec_anchor]['residues'])
        prot_length = len(protein_data[prot]['residues'])
        rg_sec = compute_radius_of_gyration(sec_struct)
        rg_prot = compute_radius_of_gyration(protein_data[prot]['structure'])
        can_cross_merge = (abs(rg_sec - rg_prot) <= 5.0) and (abs(sec_length - prot_length) <= 50)
        
        prot_loop_lengths = loop_region_dict.get(prot, {}) if loop_region_dict else {}
        prot_class_map = {}
        next_global_class = 1
        
        if can_cross_merge:
            print(f"    Cross-class merge enabled for {prot} via {sec_anchor} (Rg diff={abs(rg_sec-rg_prot):.2f} A, length diff={abs(sec_length-prot_length)} aa)")
            merged_results = cluster_b_with_singleton_merge(
                sec_class_to_prot_sites, sec_class_map, sec_coords, prot_coords,
                loop_region_lengths=prot_loop_lengths, diff_threshold=5.0, label_prefix="Sec-class"
            )
        else:
            print(f"    Cross-class merge disabled for {prot} via {sec_anchor} (Rg diff={abs(rg_sec-rg_prot):.2f} A, length diff={abs(sec_length-prot_length)} aa)")
            merged_results = []
            for s_class_id in sorted(sec_class_to_prot_sites.keys()):
                sites_in_class = sorted(set(sec_class_to_prot_sites[s_class_id]))
                sec_sites_in_class = [s_res for s_res, cls in sec_class_map.items() if cls == s_class_id]
                sec_coords_in_class = {s: sec_coords[s] for s in sec_sites_in_class if s in sec_coords}
                b_clusters = cluster_b_by_avg_distance(
                    sec_coords_in_class, sites_in_class, prot_coords,
                    loop_region_lengths=prot_loop_lengths, diff_threshold=5.0
                )
                for b_cluster in b_clusters:
                    merged_results.append(([s_class_id], b_cluster))
        
        for a_cls_list, b_cluster in merged_results:
            if not b_cluster:
                continue
            
            # Compute metrics for reporting
            if len(a_cls_list) >= 2 and can_cross_merge:
                # Centroid-based avg_A for cross-class merge
                centroids = {}
                for a_cls in a_cls_list:
                    a_sites = [s_res for s_res, cls in sec_class_map.items() if cls == a_cls]
                    coords = [sec_coords[s] for s in a_sites if s in sec_coords]
                    if len(coords) == 1:
                        centroids[a_cls] = coords[0]
                    else:
                        centroids[a_cls] = np.mean(coords, axis=0)
                n = len(centroids)
                total = 0.0
                count = 0
                c_list = list(centroids.keys())
                for i in range(n):
                    for j in range(i + 1, n):
                        total += np.linalg.norm(centroids[c_list[i]] - centroids[c_list[j]])
                        count += 1
                avg_dist_sec = total / count if count > 0 else 0.0
            else:
                # Standard pairwise avg_A within single class
                merged_sec_sites = []
                for a_cls in a_cls_list:
                    merged_sec_sites.extend([s_res for s_res, cls in sec_class_map.items() if cls == a_cls])
                merged_sec_coords = {s: sec_coords[s] for s in merged_sec_sites if s in sec_coords}
                avg_dist_sec = compute_corrected_avg_distance(merged_sec_coords, target_avg=None, loop_region_lengths=None)
            
            merged_b_coords = {b: prot_coords[b] for b in b_cluster if b in prot_coords}
            avg_dist_b = compute_corrected_avg_distance(merged_b_coords, target_avg=avg_dist_sec, loop_region_lengths=prot_loop_lengths)
            diff = abs(avg_dist_sec - avg_dist_b)
            print(f"    Sec-classes {a_cls_list} -> {prot}: avg_A={avg_dist_sec:.2f}, avg_B={avg_dist_b:.2f}, diff={diff:.2f} -> 1 cluster(s)")
            
            # Assign global class IDs
            for site in b_cluster:
                prot_class_map[site] = next_global_class
            next_global_class += 1
        
        partner_class_maps[prot] = prot_class_map
    
    return sec_class_map, partner_class_maps


def draw_schematic(prot_a, prot_b, links, 
                   a_domains, a_disordered, a_residue_list,
                   b_domains, b_disordered, b_residue_list,
                   a_class_map, b_class_map, b_class_sites,
                   output_dir):
    """
    Draw protein cross-linking schematic
    b_class_map: dict {(a_cls, b_res): global_b_cls}
    b_class_sites: dict {global_b_cls: set(b_res)}
    """
    # Get actual residue ranges
    a_seqs = [res[1] for res in a_residue_list] if a_residue_list else [1]
    b_seqs = [res[1] for res in b_residue_list] if b_residue_list else [1]
    a_min, a_max = min(a_seqs), max(a_seqs)
    b_min, b_max = min(b_seqs), max(b_seqs)
    a_length_real = a_max - a_min + 1
    b_length_real = b_max - b_min + 1
    
    # Determine number of B protein clusters (global B classes)
    b_classes = sorted(set(b_class_map.values())) if b_class_map else []
    
    # Compute average A-site position for each B class
    b_class_avg_a_pos = {}
    for b_cls in b_classes:
        a_positions = []
        for (a_cls, b_res), global_cls in b_class_map.items():
            if global_cls == b_cls:
                for a_res, cls_id in a_class_map.items():
                    if cls_id == a_cls:
                        a_positions.append(a_res)
        b_class_avg_a_pos[b_cls] = np.mean(a_positions) if a_positions else a_max / 2
    
    # Sort B classes by avg A position (N->C)
    sorted_b = sorted(b_classes, key=lambda c: b_class_avg_a_pos[c])
    
    # Build layers from center outward: middle classes near A, terminal classes below
    layers = []
    remaining = list(sorted_b)
    while remaining:
        mid = len(remaining) // 2
        if len(remaining) >= 2 and len(remaining) % 2 == 0:
            layer = [remaining[mid - 1], remaining[mid]]
        else:
            layer = [remaining[mid]]
        layers.append(layer)
        for item in layer:
            remaining.remove(item)
    
    # Assign colors for each A class
    a_classes = sorted(set(a_class_map.values())) if a_class_map else []
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(a_classes), len(b_classes), 1)))
    a_color_map = {cls: colors[i % len(colors)] for i, cls in enumerate(a_classes)}
    
    # Layout parameters
    a_y = 2.0
    rect_height = 0.20  # narrower rectangles
    b_spacing_y = 0.55
    scale = 10.0 / max(a_max, b_max)  # Scaling factor based on actual max residues
    
    # Compute figure size
    max_layer_width = max(sum(b_length_real * scale for _ in layer) + (len(layer) - 1) * 0.5 for layer in layers) if layers else b_length_real * scale
    fig_width = max(12, max_layer_width / scale * 1.2 + 4)
    fig_height = 3 + len(layers) * 1.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Draw A protein rectangle (actual residue range)
    a_x_start = (a_min - 1) * scale
    a_rect = Rectangle((a_x_start, a_y - rect_height/2), a_length_real * scale, rect_height,
                       linewidth=1.5, edgecolor='black', facecolor='#CCE5FF', alpha=0.6)
    ax.add_patch(a_rect)
    
    # A protein domain labels
    if a_domains:
        for i, (d_start, d_end) in enumerate(a_domains):
            d_width = (d_end - d_start + 1) * scale
            d_x = (d_start - 1) * scale
            domain_color = plt.cm.Pastel1(i % 8)
            d_rect = Rectangle((d_x, a_y - rect_height/2), d_width, rect_height,
                              linewidth=0, facecolor=domain_color, alpha=0.5)
            ax.add_patch(d_rect)
    
    ax.text(a_x_start - 0.5, a_y, prot_a, ha='right', va='center', fontsize=16, fontweight='bold')
    
    # A protein disordered region markers (dashed lines)
    if a_disordered:
        for d_start_idx, d_end_idx in a_disordered:
            start_seq = a_residue_list[d_start_idx][1]
            end_seq = a_residue_list[d_end_idx][1]
            d_x = (start_seq - 1) * scale
            d_width = (end_seq - start_seq + 1) * scale
            ax.plot([d_x, d_x + d_width], [a_y + rect_height/2 + 0.1, a_y + rect_height/2 + 0.1],
                   'k--', linewidth=1)
            ax.text(d_x + d_width/2, a_y + rect_height/2 + 0.25, 'disordered',
                   ha='center', va='bottom', fontsize=7, color='red')
    
    # Draw B protein rectangles with 2D spatial layout
    b_rect_info = {}  # {global_b_cls: (x_left, y_center)}
    
    for layer_idx, layer in enumerate(layers):
        b_y = a_y - 0.6 - layer_idx * b_spacing_y
        
        # Compute x positions for this layer based on avg A positions
        layer_info = []
        for b_cls in layer:
            avg_a = b_class_avg_a_pos[b_cls]
            target_x = (avg_a - 1) * scale - (b_length_real * scale) / 2
            # Clamp within A rectangle bounds
            target_x = max(a_x_start, min(target_x, a_x_start + a_length_real * scale - b_length_real * scale))
            layer_info.append((b_cls, target_x, avg_a))
        
        # Sort within layer by avg_a to avoid crossing
        layer_info.sort(key=lambda t: t[2])
        
        # If multiple in layer, spread them out evenly if they would overlap
        n = len(layer_info)
        if n > 1:
            total_width = n * b_length_real * scale + (n - 1) * 0.3
            # Try to center the group around the middle target
            mid_target = np.mean([t[1] + b_length_real * scale / 2 for t in layer_info])
            start_x = max(a_x_start, min(mid_target - total_width / 2, a_x_start + a_length_real * scale - total_width))
            for idx, (b_cls, _, _) in enumerate(layer_info):
                x_left = start_x + idx * (b_length_real * scale + 0.3)
                b_rect_info[b_cls] = (x_left, b_y)
        else:
            b_cls, target_x, _ = layer_info[0]
            b_rect_info[b_cls] = (target_x, b_y)
    
    # If no B classes, draw a default B rectangle
    if not b_classes:
        b_y = a_y - 0.6
        x_left = 0
        b_rect_info[1] = (x_left, b_y)
        b_rect = Rectangle((x_left, b_y - rect_height/2), b_length_real * scale, rect_height,
                          linewidth=1.5, edgecolor='black', facecolor='#FFE5CC', alpha=0.6)
        ax.add_patch(b_rect)
        if b_domains:
            for j, (d_start, d_end) in enumerate(b_domains):
                d_width = (d_end - d_start + 1) * scale
                d_x = x_left + (d_start - b_min) * scale
                domain_color = plt.cm.Pastel1(j % 8)
                d_rect = Rectangle((d_x, b_y - rect_height/2), d_width, rect_height,
                                  linewidth=0, facecolor=domain_color, alpha=0.5)
                ax.add_patch(d_rect)
        ax.text(-0.5, b_y, prot_b, ha='right', va='center', fontsize=16, fontweight='bold')
    
    # Draw B rectangles and labels
    for b_cls, (x_left, b_y) in b_rect_info.items():
        b_rect = Rectangle((x_left, b_y - rect_height/2), b_length_real * scale, rect_height,
                          linewidth=1.5, edgecolor='black', facecolor='#FFE5CC', alpha=0.6)
        ax.add_patch(b_rect)
        
        if b_domains:
            for j, (d_start, d_end) in enumerate(b_domains):
                d_width = (d_end - d_start + 1) * scale
                d_x = x_left + (d_start - b_min) * scale
                domain_color = plt.cm.Pastel1(j % 8)
                d_rect = Rectangle((d_x, b_y - rect_height/2), d_width, rect_height,
                                  linewidth=0, facecolor=domain_color, alpha=0.5)
                ax.add_patch(d_rect)
        
        ax.text(x_left + b_length_real * scale / 2, b_y + rect_height/2 + 0.15,
                f"{prot_b} (state {b_cls})", ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Draw cross-link lines
    for a_res, b_res in links:
        a_cls = a_class_map.get(a_res, 1)
        b_cls = b_class_map.get((a_cls, b_res), 1)
        
        a_x = (a_res - 1) * scale
        
        x_left, b_y = b_rect_info.get(b_cls, (a_x_start, a_y - 0.6))
        b_x_abs = x_left + (b_res - b_min) * scale  # relative to B rectangle start
        
        color = a_color_map.get(a_cls, 'gray')
        
        # Draw cross-link line
        ax.plot([a_x, b_x_abs], [a_y - rect_height/2, b_y + rect_height/2],
               color=color, linewidth=1.5, alpha=0.7, zorder=2)
        
        # Draw dots at cross-link sites
        ax.plot(a_x, a_y - rect_height/2, 'o', color=color, markersize=5, zorder=3)
        ax.plot(b_x_abs, b_y + rect_height/2, 'o', color=color, markersize=5, zorder=3)
    
    # Set figure limits
    all_x = [a_x_start, a_x_start + a_length_real * scale]
    for x_left, b_y in b_rect_info.values():
        all_x.extend([x_left, x_left + b_length_real * scale])
    ax.set_xlim(min(all_x) - 2, max(all_x) + 2)
    ax.set_ylim(b_y - 1.0, a_y + 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.set_title(f"Cross-linking Map: {prot_a} - {prot_b}", fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{prot_a}_{prot_b}_schematic.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Schematic saved: {fig_path}")


def draw_anchor_schematic(anchor_prot, all_links, anchor_class_map, prot_class_maps,
                            sec_results, protein_data, output_dir):
    """
    Draw a multi-layer schematic with main anchor on top, secondary anchors in middle,
    and other proteins at bottom. All connections are drawn.
    """
    # Helper to get actual residue range
    def get_res_range(residue_list):
        if not residue_list:
            return 1, 1
        seqs = [res[1] for res in residue_list]
        return min(seqs), max(seqs)
    
    # Main anchor info
    a_residue_list = protein_data[anchor_prot]['residues']
    a_min, a_max = get_res_range(a_residue_list)
    a_length_real = a_max - a_min + 1
    a_domains = protein_data[anchor_prot]['domains']
    a_disordered = protein_data[anchor_prot]['disordered']
    
    sec_anchors = sorted(sec_results.keys())
    all_partners = sorted(prot_class_maps.keys())
    
    # Layout parameters
    a_y = 4.0
    sec_y = 2.2
    leaf_y = 0.4
    rect_height = 0.22
    scale = 10.0 / a_max  # scale based on main anchor max residue
    
    # Figure size
    fig_width = max(14, a_max * scale * 1.5)
    n_layers = 1 + (1 if sec_anchors else 0) + 1
    fig_height = 2 + n_layers * 1.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # ===== Draw main anchor (Layer 0) =====
    a_x_start = (a_min - 1) * scale
    a_width = a_length_real * scale
    a_rect = Rectangle((a_x_start, a_y - rect_height/2), a_width, rect_height,
                       linewidth=1.5, edgecolor='black', facecolor='#CCE5FF', alpha=0.6)
    ax.add_patch(a_rect)
    
    if a_domains:
        for i, (d_start, d_end) in enumerate(a_domains):
            d_width = (d_end - d_start + 1) * scale
            d_x = (d_start - 1) * scale
            domain_color = plt.cm.Pastel1(i % 8)
            d_rect = Rectangle((d_x, a_y - rect_height/2), d_width, rect_height,
                              linewidth=0, facecolor=domain_color, alpha=0.5)
            ax.add_patch(d_rect)
    
    if a_disordered:
        for d_start_idx, d_end_idx in a_disordered:
            start_seq = a_residue_list[d_start_idx][1]
            end_seq = a_residue_list[d_end_idx][1]
            d_x = (start_seq - 1) * scale
            d_width = (end_seq - start_seq + 1) * scale
            ax.plot([d_x, d_x + d_width], [a_y + rect_height/2 + 0.1, a_y + rect_height/2 + 0.1],
                   'k--', linewidth=1)
            ax.text(d_x + d_width/2, a_y + rect_height/2 + 0.25, 'disordered',
                   ha='center', va='bottom', fontsize=7, color='red')
    
    ax.text(a_x_start - 0.5, a_y, anchor_prot, ha='right', va='center', fontsize=16, fontweight='bold')
    
    # ===== Draw secondary anchors (Layer 1) =====
    sec_rect_info = {}  # {sec_anchor: (x_start, y_center, min_res, max_res)}
    for sec_anchor in sec_anchors:
        sec_residue_list = protein_data[sec_anchor]['residues']
        sec_min, sec_max = get_res_range(sec_residue_list)
        sec_length_real = sec_max - sec_min + 1
        sec_domains = protein_data[sec_anchor]['domains']
        
        # Place based on average contact position with main anchor
        anchor_positions = []
        for p1, r1, p2, r2 in all_links:
            if (p1 == anchor_prot and p2 == sec_anchor) or (p2 == anchor_prot and p1 == sec_anchor):
                anchor_positions.append(r1 if p1 == anchor_prot else r2)
        avg_anchor = np.mean(anchor_positions) if anchor_positions else a_max / 2
        sec_x_start = (avg_anchor - 1) * scale - (sec_length_real * scale) / 2
        sec_x_start = max(0, min(sec_x_start, a_max * scale - sec_length_real * scale))
        sec_width = sec_length_real * scale
        
        sec_rect = Rectangle((sec_x_start, sec_y - rect_height/2), sec_width, rect_height,
                            linewidth=1.5, edgecolor='black', facecolor='#D4EDDA', alpha=0.6)
        ax.add_patch(sec_rect)
        
        if sec_domains:
            for j, (d_start, d_end) in enumerate(sec_domains):
                d_width = (d_end - d_start + 1) * scale
                d_x = sec_x_start + (d_start - sec_min) * scale
                domain_color = plt.cm.Pastel1(j % 8)
                d_rect = Rectangle((d_x, sec_y - rect_height/2), d_width, rect_height,
                                  linewidth=0, facecolor=domain_color, alpha=0.5)
                ax.add_patch(d_rect)
        
        ax.text(sec_x_start - 0.5, sec_y, sec_anchor, ha='right', va='center', fontsize=14, fontweight='bold')
        sec_rect_info[sec_anchor] = (sec_x_start, sec_y, sec_min, sec_max)
    
    # ===== Draw leaves (Layer 2) =====
    leaf_prots = [p for p in all_partners if p not in sec_anchors]
    leaf_info = []
    for prot in leaf_prots:
        b_residue_list = protein_data[prot]['residues']
        b_min, b_max = get_res_range(b_residue_list)
        b_length_real = b_max - b_min + 1
        b_domains = protein_data[prot]['domains']
        
        anchor_positions = []
        for p1, r1, p2, r2 in all_links:
            if (p1 == anchor_prot and p2 == prot) or (p2 == anchor_prot and p1 == prot):
                anchor_positions.append(r1 if p1 == anchor_prot else r2)
        avg_anchor = np.mean(anchor_positions) if anchor_positions else a_max / 2
        
        leaf_info.append({
            'prot': prot,
            'min': b_min,
            'max': b_max,
            'length_real': b_length_real,
            'domains': b_domains,
            'avg_anchor': avg_anchor
        })
    
    leaf_info.sort(key=lambda x: x['avg_anchor'])
    
    n_leaf = len(leaf_info)
    leaf_rect_info = {}  # {prot: (x_start, y_center, min_res, max_res)}
    if n_leaf > 0:
        total_leaf_width = sum(info['length_real'] * scale for info in leaf_info) + (n_leaf - 1) * 1.5
        start_x = max(0, (a_max * scale - total_leaf_width) / 2)
        
        for i, info in enumerate(leaf_info):
            prot = info['prot']
            b_min = info['min']
            b_max = info['max']
            b_length_real = info['length_real']
            b_width = b_length_real * scale
            b_x = start_x + sum(leaf_info[j]['length_real'] * scale for j in range(i)) + i * 1.5
            b_y = leaf_y
            
            b_rect = Rectangle((b_x, b_y - rect_height/2), b_width, rect_height,
                              linewidth=1.5, edgecolor='black', facecolor='#FFE5CC', alpha=0.6)
            ax.add_patch(b_rect)
            
            if info['domains']:
                for j, (d_start, d_end) in enumerate(info['domains']):
                    d_width = (d_end - d_start + 1) * scale
                    d_x = b_x + (d_start - b_min) * scale
                    domain_color = plt.cm.Pastel1(j % 8)
                    d_rect = Rectangle((d_x, b_y - rect_height/2), d_width, rect_height,
                                      linewidth=0, facecolor=domain_color, alpha=0.5)
                    ax.add_patch(d_rect)
            
            ax.text(b_x + b_width / 2, b_y + rect_height/2 + 0.15,
                   prot, ha='center', va='bottom', fontsize=12, fontweight='bold')
            leaf_rect_info[prot] = (b_x, b_y, b_min, b_max)
    
    # ===== Draw all connections =====
    # A color map for main anchor classes
    a_classes = sorted(set(anchor_class_map.values())) if anchor_class_map else []
    a_colors = plt.cm.tab10(np.linspace(0, 1, max(len(a_classes), 1)))
    a_color_map = {cls: a_colors[i % len(a_colors)] for i, cls in enumerate(a_classes)}
    
    # 1. Main anchor <-> secondary anchor connections
    for sec_anchor in sec_anchors:
        sec_x_start, sec_y_c, sec_min, sec_max = sec_rect_info[sec_anchor]
        for p1, r1, p2, r2 in all_links:
            if (p1 == anchor_prot and p2 == sec_anchor) or (p2 == anchor_prot and p1 == sec_anchor):
                anchor_res = r1 if p1 == anchor_prot else r2
                sec_res = r2 if p1 == anchor_prot else r1
                
                a_x = (anchor_res - 1) * scale
                sec_x_abs = sec_x_start + (sec_res - sec_min) * scale
                
                a_cls = anchor_class_map.get(anchor_res, 1)
                color = a_color_map.get(a_cls, 'gray')
                
                ax.plot([a_x, sec_x_abs], [a_y - rect_height/2, sec_y_c + rect_height/2],
                       color=color, linewidth=1.5, alpha=0.7, zorder=2)
                ax.plot(a_x, a_y - rect_height/2, 'o', color=color, markersize=5, zorder=3)
                ax.plot(sec_x_abs, sec_y_c + rect_height/2, 'o', color=color, markersize=5, zorder=3)
    
    # 2. Main anchor <-> leaf connections
    for prot in leaf_prots:
        if prot not in leaf_rect_info:
            continue
        b_x, b_y_c, b_min, b_max = leaf_rect_info[prot]
        for p1, r1, p2, r2 in all_links:
            if (p1 == anchor_prot and p2 == prot) or (p2 == anchor_prot and p1 == prot):
                anchor_res = r1 if p1 == anchor_prot else r2
                other_res = r2 if p1 == anchor_prot else r1
                
                a_x = (anchor_res - 1) * scale
                b_x_abs = b_x + (other_res - b_min) * scale
                
                b_cls = prot_class_maps.get(prot, {}).get(other_res, 1)
                n_classes = len(set(prot_class_maps.get(prot, {}).values())) if prot in prot_class_maps else 1
                colors = plt.cm.tab10(np.linspace(0, 1, max(n_classes, 1)))
                color = colors[(b_cls - 1) % len(colors)] if b_cls > 0 else 'gray'
                
                ax.plot([a_x, b_x_abs], [a_y - rect_height/2, b_y_c + rect_height/2],
                       color=color, linewidth=1.5, alpha=0.7, zorder=2)
                ax.plot(a_x, a_y - rect_height/2, 'o', color=color, markersize=5, zorder=3)
                ax.plot(b_x_abs, b_y_c + rect_height/2, 'o', color=color, markersize=5, zorder=3)
    
    # 3. Secondary anchor <-> leaf connections
    for sec_anchor in sec_anchors:
        sec_x_start, sec_y_c, sec_min, sec_max = sec_rect_info[sec_anchor]
        sec_class_map, partner_class_maps, sec_links = sec_results[sec_anchor]
        
        for p1, r1, p2, r2 in all_links:
            if (p1 == sec_anchor and p2 in leaf_rect_info) or (p2 == sec_anchor and p1 in leaf_rect_info):
                sec_res = r1 if p1 == sec_anchor else r2
                leaf_prot = p2 if p1 == sec_anchor else p1
                leaf_res = r2 if p1 == sec_anchor else r1
                
                if leaf_prot not in leaf_rect_info:
                    continue
                b_x, b_y_c, b_min, b_max = leaf_rect_info[leaf_prot]
                
                sec_x_abs = sec_x_start + (sec_res - sec_min) * scale
                b_x_abs = b_x + (leaf_res - b_min) * scale
                
                b_cls = partner_class_maps.get(leaf_prot, {}).get(leaf_res, 1)
                n_classes = len(set(partner_class_maps.get(leaf_prot, {}).values())) if leaf_prot in partner_class_maps else 1
                colors = plt.cm.tab10(np.linspace(0, 1, max(n_classes, 1)))
                color = colors[(b_cls - 1) % len(colors)] if b_cls > 0 else 'gray'
                
                ax.plot([sec_x_abs, b_x_abs], [sec_y_c + rect_height/2, b_y_c + rect_height/2],
                       color=color, linewidth=1.5, alpha=0.7, zorder=2)
                ax.plot(sec_x_abs, sec_y_c + rect_height/2, 'o', color=color, markersize=5, zorder=3)
                ax.plot(b_x_abs, b_y_c + rect_height/2, 'o', color=color, markersize=5, zorder=3)
    
    ax.set_xlim(-2, a_max * scale + 2)
    ax.set_ylim(-0.5, a_y + 1.5)
    ax.axis('off')
    title_extra = f" + Secondary: {', '.join(sec_anchors)}" if sec_anchors else ""
    ax.set_title(f"Cross-linking Map: {anchor_prot} as Anchor{title_extra}",
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{anchor_prot}_anchor_schematic.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Anchor schematic saved: {fig_path}")


def main():
    """Main entry point"""
    os.chdir(WORK_DIR)
    ensure_dir(OUTPUT_DIR)
    
    print("=" * 60)
    print("Protein Cross-linking Structure Analysis")
    print("=" * 60)
    
    # Read cross-link information
    print(f"\n[1/4] Reading cross-link file: {LINK_FILE}")
    all_links = parse_link_file(LINK_FILE)
    print(f"      Total {len(all_links)} cross-link records read")
    
    # Collect all involved proteins
    all_proteins = set()
    for prot1, res1, prot2, res2 in all_links:
        all_proteins.add(prot1)
        all_proteins.add(prot2)
    print(f"      Proteins involved: {', '.join(sorted(all_proteins))}")
    
    # Build cross-link site sets per protein for loop detection (only when needed)
    xl_sites_per_prot = defaultdict(set)
    if USE_LOOP_FLEXIBILITY:
        for prot1, res1, prot2, res2 in all_links:
            xl_sites_per_prot[prot1].add(res1)
            xl_sites_per_prot[prot2].add(res2)
    
    # Pre-load structures to find anchor protein
    # 根据交联位点统计（不同氨基酸数、最大距离）选择锚点蛋白
    anchor_prot = None
    best_count = -1
    best_max_dist = -1.0
    protein_data = {}
    for prot in sorted(all_proteins):
        pdb_path = os.path.join(PDB_DIR, f"{prot}.pdb")
        if not os.path.exists(pdb_path):
            print(f"Warning: PDB file not found: {pdb_path}")
            continue
        structure = get_structure(pdb_path)
        residue_list = get_residue_list(structure)
        protein_data[prot] = {'structure': structure, 'residues': residue_list}
        count, max_dist = compute_xl_site_stats(prot, all_links, protein_data)
        print(f"      {prot}: XL unique sites = {count}, max pairwise distance = {max_dist:.2f} A")
        if count > best_count or (count == best_count and max_dist > best_max_dist):
            best_count = count
            best_max_dist = max_dist
            anchor_prot = prot
    
    print(f"\n      Pre-selected anchor protein (by XL stats): {anchor_prot} ({best_count} unique sites, max dist {best_max_dist:.2f} A)")
    
    # Analyze domains for each protein
    print(f"\n[2/4] Protein domain partitioning")
    for prot in sorted(all_proteins):
        print(f"      Analyzing protein {prot} ...")
        domains, linkers, loops, residue_list, structure, ss_dict = analyze_protein(
            prot, PDB_DIR, OUTPUT_DIR, xl_sites=xl_sites_per_prot.get(prot, set())
        )
        protein_data[prot]['domains'] = domains
        protein_data[prot]['linkers'] = linkers
        protein_data[prot]['loops'] = loops
        protein_data[prot]['disordered'] = linkers  # backward compatibility for schematic drawing
        protein_data[prot]['residues'] = residue_list
        protein_data[prot]['structure'] = structure
        protein_data[prot]['ss_dict'] = ss_dict
        if domains:
            domain_str = "  ".join([f"{s}-{e}" for s, e in domains])
            print(f"            Domains: {domain_str}")
        else:
            print(f"            Domains: (none)")
        if linkers:
            linker_str = "  ".join([f"{residue_list[s][1]}-{residue_list[e][1]}" for s, e in linkers])
            print(f"            Linkers: {linker_str}")
        else:
            print(f"            Linkers: none")
        if loops:
            loop_str = "  ".join([f"{residue_list[s][1]}-{residue_list[e][1]}" for s, e in loops])
            print(f"            Loops: {loop_str}")
        else:
            print(f"            Loops: none")
    
    # Re-select anchor protein based on cross-linking statistics
    best_anchor = None
    best_count = -1
    best_max_dist = -1.0
    for prot in protein_data:
        count, max_dist = compute_xl_site_stats(prot, all_links, protein_data)
        print(f"      {prot}: XL unique sites = {count}, max pairwise distance = {max_dist:.2f} A")
        if count > best_count or (count == best_count and max_dist > best_max_dist):
            best_count = count
            best_max_dist = max_dist
            best_anchor = prot
    
    anchor_prot = best_anchor
    print(f"\n      Final anchor protein (by XL stats): {anchor_prot} ({best_count} unique sites, max dist {best_max_dist:.2f} A)")
    
    # Build loop region length dict for flexibility correction (used in both modes)
    loop_region_dict = {}
    for prot in all_proteins:
        ss_dict = protein_data[prot].get('ss_dict', {})
        # Find all loop residues (coil '-', turn 'T', or bend 'S')
        loop_residues = set()
        for (chain_id, res_seq), ss in ss_dict.items():
            if ss in ('-', 'T', 'S'):  # coil, turn, bend = loop region
                loop_residues.add(res_seq)
        # Find continuous loop regions and map each residue to its region length
        prot_loop_dict = {}
        if loop_residues:
            sorted_loop = sorted(loop_residues)
            region_start = sorted_loop[0]
            region_end = sorted_loop[0]
            regions = []
            for res in sorted_loop[1:]:
                if res == region_end + 1:
                    region_end = res
                else:
                    regions.append((region_start, region_end))
                    region_start = res
                    region_end = res
            regions.append((region_start, region_end))
            for start, end in regions:
                length = end - start + 1
                for res in range(start, end + 1):
                    prot_loop_dict[res] = length
        loop_region_dict[prot] = prot_loop_dict
    
    # Choose analysis mode based on number of proteins
    n_proteins = len(all_proteins)
    
    if n_proteins == 2:
        # Pairwise mode: select reference protein by cross-linking statistics
        prots = sorted(all_proteins)
        prot_1, prot_2 = prots[0], prots[1]
        count_1, max_dist_1 = compute_xl_site_stats(prot_1, all_links, protein_data)
        count_2, max_dist_2 = compute_xl_site_stats(prot_2, all_links, protein_data)
        if count_1 > count_2:
            prot_a, prot_b = prot_1, prot_2
        elif count_2 > count_1:
            prot_a, prot_b = prot_2, prot_1
        elif max_dist_1 > max_dist_2:
            prot_a, prot_b = prot_1, prot_2
        else:
            prot_a, prot_b = prot_2, prot_1
        
        # Extract links for this pair
        pair_links = []
        for p1, r1, p2, r2 in all_links:
            if (p1 == prot_a and p2 == prot_b) or (p2 == prot_a and p1 == prot_b):
                a_res = r1 if p1 == prot_a else r2
                b_res = r2 if p1 == prot_a else r1
                pair_links.append((a_res, b_res))
        
        print(f"\n[3/4] Cross-link site clustering analysis (pairwise: {prot_a} - {prot_b})")
        a_class_map, b_class_map, b_class_sites, a_coords, b_coords, rg_b = analyze_interaction(
            prot_a, prot_b, pair_links, PDB_DIR, OUTPUT_DIR, CROSS_LINKER_LENGTH, loop_region_dict
        )
        
        print(f"\n[4/4] Drawing pairwise schematic")
        draw_schematic(prot_a, prot_b, pair_links,
                       protein_data[prot_a]['domains'], protein_data[prot_a]['disordered'], protein_data[prot_a]['residues'],
                       protein_data[prot_b]['domains'], protein_data[prot_b]['disordered'], protein_data[prot_b]['residues'],
                       a_class_map, b_class_map, b_class_sites,
                       OUTPUT_DIR)
    else:
        # Anchor mode: for multi-protein systems (>2)
        print(f"\n[3/4] Cross-link site clustering analysis (anchor: {anchor_prot})")
        
        # Main anchor analysis
        anchor_class_map, prot_class_maps, rg_map, anchor_interface_links, other_links = analyze_anchor_system(
            anchor_prot, all_links, protein_data, CROSS_LINKER_LENGTH, loop_region_dict
        )
        
        # Identify secondary anchors from other_links
        # A secondary anchor must have >= 2 distinct partners in other_links
        sec_anchor_candidates = set()
        link_degree = defaultdict(int)
        partners = defaultdict(set)
        for p1, r1, p2, r2 in other_links:
            sec_anchor_candidates.add(p1)
            sec_anchor_candidates.add(p2)
            link_degree[p1] += 1
            link_degree[p2] += 1
            partners[p1].add(p2)
            partners[p2].add(p1)
        
        sec_anchors = [p for p in sec_anchor_candidates if len(partners[p]) >= 2]
        sec_anchors.sort()
        
        # Analyze each secondary anchor
        sec_results = {}  # {sec_anchor: (sec_class_map, partner_class_maps, sec_links)}
        for sec_anchor in sec_anchors:
            sec_links = []
            for p1, r1, p2, r2 in other_links:
                if p1 == sec_anchor:
                    sec_links.append((r1, p2, r2))
                elif p2 == sec_anchor:
                    sec_links.append((r2, p1, r1))
            
            sec_class_map, partner_class_maps = analyze_secondary_anchor(
                sec_anchor, sec_links, protein_data, CROSS_LINKER_LENGTH, loop_region_dict
            )
            sec_results[sec_anchor] = (sec_class_map, partner_class_maps, sec_links)
            print(f"  Secondary anchor {sec_anchor} interface clustered into {len(set(sec_class_map.values()))} states")
            for prot, cmap in partner_class_maps.items():
                print(f"    Protein {prot} (via {sec_anchor}) clustered into {len(set(cmap.values()))} states")
        
        # Output single merged cluster file
        cluster_file = os.path.join(OUTPUT_DIR, f"{anchor_prot}_anchor_clusters.txt")
        with open(cluster_file, 'w') as f:
            f.write(f"# Anchor_Protein: {anchor_prot}\n")
            if sec_anchors:
                f.write(f"# Secondary_Anchors: {', '.join(sec_anchors)}\n")
            f.write(f"# Protein_A is the anchor/reference, State_A = 1 (protein count ID)\n")
            f.write(f"Protein_A\tSite_A\tState_A\tProtein_B\tSite_B\tState_B\n")
            # Main anchor links
            for anchor_res, other_prot, other_res in anchor_interface_links:
                b_cls = prot_class_maps.get(other_prot, {}).get(other_res, -1)
                f.write(f"{anchor_prot}\t{anchor_res}\t1\t{other_prot}\t{other_res}\t{b_cls}\n")
            # Secondary anchor links
            for sec_anchor, (sec_class_map, partner_class_maps, sec_links) in sec_results.items():
                for s_res, p_prot, p_res in sec_links:
                    b_cls = partner_class_maps.get(p_prot, {}).get(p_res, -1)
                    f.write(f"{sec_anchor}\t{s_res}\t1\t{p_prot}\t{p_res}\t{b_cls}\n")
        
        print(f"\n      Cluster file saved: {cluster_file}")
        
        # Draw anchor schematic with secondary anchors
        print(f"\n[4/4] Drawing anchor schematic")
        draw_anchor_schematic(anchor_prot, all_links, anchor_class_map, prot_class_maps,
                              sec_results, protein_data, OUTPUT_DIR)
    
    print(f"\n{'=' * 60}")
    print("Analysis complete!")
    print(f"Output files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
