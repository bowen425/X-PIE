#!/usr/bin/env python3
"""
Generate XPLOR input files (xlms.tbl and refine.py) from X-PIE clustering results.

Usage:
    python scripts/generate-script.py --output-dir ../interface-define --linker-length 15 --num-structures 1

The script reads:
    - *_clusters.txt   : cross-link clustering results
    - *_domains.txt    : domain definitions for each protein
    - *_linker.txt     : linker (long disordered >=30) region definitions
    - *_loop.txt       : loop (short disordered 5-29 with XL sites) definitions

And generates:
    - xlms.tbl  : distance restraint file for cross-links
    - refine.py : XPLOR sampling control script
"""

import os
import sys
import argparse
import glob

# ============================================================
# Configuration
# ============================================================

# Default cross-linker length in Angstroms
DEFAULT_LINKER_LENGTH = 15

# Default number of structures to calculate
DEFAULT_NUM_STRUCTURES = 1

# Segid prefix letters for proteins (A=anchor, B=2nd, C=3rd, ...)
SEGID_PREFIXES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


def parse_cluster_file(filepath):
    """
    Parse a *_clusters.txt file.
    Returns list of tuples: (prot_a, site_a, state_a, prot_b, site_b, state_b)
    """
    links = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 6:
                try:
                    links.append((
                        parts[0], int(parts[1]), int(parts[2]),
                        parts[3], int(parts[4]), int(parts[5])
                    ))
                except ValueError:
                    # Skip header lines or malformed rows
                    continue
    return links


def parse_domain_linker_loop(output_dir):
    """
    Parse *_domains.txt, *_linker.txt (or *_disordered.txt fallback),
    and *_loop.txt files in output_dir.
    Returns: (domains_dict, linkers_dict, loops_dict)
        domains_dict: {prot_name: [(start, end), ...]}
        linkers_dict: {prot_name: [(start, end), ...]}
        loops_dict:   {prot_name: [(start, end), ...]}
    """
    domains = {}
    linkers = {}
    loops = {}

    # Identify proteins that already have new-style linker files
    linker_files = {f[:-len('_linker.txt')] for f in os.listdir(output_dir) if f.endswith('_linker.txt')}

    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if fname.endswith('_domains.txt'):
            prot = fname[:-len('_domains.txt')]
            with open(fpath, 'r') as f:
                content = f.read().strip()
            if content:
                dom_part = content.split(':')[-1].strip()
                if dom_part:
                    domains[prot] = []
                    for d in dom_part.split():
                        s, e = map(int, d.split('-'))
                        domains[prot].append((s, e))

        elif fname.endswith('_linker.txt'):
            prot = fname[:-len('_linker.txt')]
            with open(fpath, 'r') as f:
                content = f.read().strip()
            if content:
                link_part = content.split(':')[-1].strip()
                if link_part:
                    linkers[prot] = []
                    for d in link_part.split():
                        s, e = map(int, d.split('-'))
                        linkers[prot].append((s, e))

        elif fname.endswith('_disordered.txt'):
            # Fallback for backward compatibility only if no linker file exists
            prot = fname[:-len('_disordered.txt')]
            if prot not in linker_files:
                with open(fpath, 'r') as f:
                    content = f.read().strip()
                if content:
                    dis_part = content.split(':')[-1].strip()
                    if dis_part:
                        linkers[prot] = []
                        for d in dis_part.split():
                            s, e = map(int, d.split('-'))
                            linkers[prot].append((s, e))

        elif fname.endswith('_loop.txt'):
            prot = fname[:-len('_loop.txt')]
            with open(fpath, 'r') as f:
                content = f.read().strip()
            if content:
                loop_part = content.split(':')[-1].strip()
                if loop_part:
                    loops[prot] = []
                    for d in loop_part.split():
                        s, e = map(int, d.split('-'))
                        loops[prot].append((s, e))

    return domains, linkers, loops


def build_protein_info(links_list):
    """
    Build protein ordering, state counts, and segid mapping.

    Protein ordering:
        1. Anchor protein (first Protein_A appearing in cluster files)
        2. Other proteins sorted alphabetically

    Segid assignment:
        - Protein #1 (anchor): ALT1, ALT2, ...
        - Protein #2: BLT1, BLT2, ...
        - Protein #3: CLT1, CLT2, ...
        - etc.
    """
    all_proteins = set()
    for links in links_list:
        for prot_a, site_a, state_a, prot_b, site_b, state_b in links:
            all_proteins.add(prot_a)
            all_proteins.add(prot_b)

    # Anchor = first protein appearing as Protein_A
    anchor = None
    for links in links_list:
        if links:
            anchor = links[0][0]
            break

    if anchor is None:
        raise ValueError("No cross-link data found in cluster files.")

    other_proteins = sorted([p for p in all_proteins if p != anchor])
    protein_order = [anchor] + other_proteins

    # Determine max state count for each protein
    max_states = {prot: 1 for prot in protein_order}
    for links in links_list:
        for prot_a, site_a, state_a, prot_b, site_b, state_b in links:
            max_states[prot_a] = max(max_states[prot_a], state_a)
            max_states[prot_b] = max(max_states[prot_b], state_b)

    # Assign segids
    segid_map = {}  # {prot: {state: segid}}
    for idx, prot in enumerate(protein_order):
        prefix = SEGID_PREFIXES[idx]
        segid_map[prot] = {}
        for state in range(1, max_states[prot] + 1):
            segid_map[prot][state] = f"{prefix}LT{state}"

    return protein_order, segid_map, max_states


def build_n_fallback_map(pdb_dir, xl_sites_map):
    """
    Check PDB files for cross-link sites that lack NZ atom.
    For these residues, distance restraints should use the backbone N atom.
    xl_sites_map: {protein: set_of_residue_numbers}
    Returns: {protein: set_of_residue_numbers}
    """
    n_fallback = {}
    for prot, sites in xl_sites_map.items():
        pdb_path = os.path.join(pdb_dir, f"{prot}.pdb")
        if not os.path.exists(pdb_path):
            continue

        res_atoms = {}
        with open(pdb_path, 'r') as f:
            for line in f:
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                atom_name = line[12:16].strip()
                res_seq_str = line[22:26].strip()
                try:
                    res_seq = int(res_seq_str)
                except ValueError:
                    continue
                if res_seq not in sites:
                    continue
                res_atoms.setdefault(res_seq, set()).add(atom_name)

        fallback = set()
        for res_seq, atoms in res_atoms.items():
            if 'NZ' not in atoms and 'N' in atoms:
                fallback.add(res_seq)

        if fallback:
            n_fallback[prot] = fallback

    return n_fallback


def generate_xlms_tbl(links_list, segid_map, linker_length, output_path, n_fallback_map=None):
    """Generate xlms.tbl distance restraint file."""
    if n_fallback_map is None:
        n_fallback_map = {}
    third_num = linker_length - 4.0

    with open(output_path, 'w') as f:
        f.write(f"# Cross-linker length: {linker_length} A\n")
        f.write(f"# Format: assign (segid X and resi N and name <atom>) "
                f"(segid Y and resid M and name <atom>) 10.0 6.0 {third_num:.1f}\n\n")

        for links in links_list:
            for prot_a, site_a, state_a, prot_b, site_b, state_b in links:
                segid_a = segid_map[prot_a][state_a]
                segid_b = segid_map[prot_b][state_b]
                atom_a = 'n' if site_a in n_fallback_map.get(prot_a, set()) else 'nz'
                atom_b = 'n' if site_b in n_fallback_map.get(prot_b, set()) else 'nz'
                f.write(
                    f"assign (segid {segid_a} and resi {site_a}  and name {atom_a}) "
                    f"(segid {segid_b} and resid {site_b}  and name {atom_b}) "
                    f"10.0 6.0 {third_num:.1f}\n"
                )
        f.write("\n")

    print(f"  Generated: {output_path}")


def count_anchor_xl_per_domain(anchor, links_list, domains):
    """
    Count cross-link sites per domain for the anchor protein.
    Sites falling in gaps between domains are assigned to the nearest domain
    by sequence distance.
    Returns list of (domain_idx, count, start, end) sorted by count descending.
    """
    prot_domains = domains.get(anchor, [])
    if len(prot_domains) <= 1:
        return []

    xl_sites = set()
    for links in links_list:
        for prot_a, site_a, state_a, prot_b, site_b, state_b in links:
            if prot_a == anchor:
                xl_sites.add(site_a)
            elif prot_b == anchor:
                xl_sites.add(site_b)

    # Count sites strictly inside each domain
    domain_counts = []
    for idx, (d_start, d_end) in enumerate(prot_domains):
        count = sum(1 for site in xl_sites if d_start <= site <= d_end)
        domain_counts.append([idx, count, d_start, d_end])

    # Assign orphan sites (in gaps) to nearest domain by sequence distance
    covered_ranges = [(d_start, d_end) for _, _, d_start, d_end in domain_counts]
    for site in xl_sites:
        in_any = any(d_start <= site <= d_end for d_start, d_end in covered_ranges)
        if not in_any:
            min_dist = None
            nearest_idx = 0
            for idx, (d_start, d_end) in enumerate(prot_domains):
                if site < d_start:
                    dist = d_start - site
                elif site > d_end:
                    dist = site - d_end
                else:
                    dist = 0
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx
            domain_counts[nearest_idx][1] += 1

    # Convert back to tuples and sort
    result = [(idx, count, d_start, d_end) for idx, count, d_start, d_end in domain_counts]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def generate_refine_py(protein_order, segid_map, max_states,
                       domains, linkers, loops, links_list, num_structures, output_path,
                       xplor_home=None):
    """Generate refine.py XPLOR sampling control script."""

    # ============================================================
    # step1: structure define
    # ============================================================
    structure_lines = []
    structure_lines.append("# step1 structure define")
    if xplor_home:
        params_path = os.path.join(xplor_home, 'parallhdg_new.pro')
    else:
        params_path = './parallhdg_new.pro'
    structure_lines.append(f"protocol.initParams('{params_path}')")

    assigned_segids = []
    for prot in protein_order:
        psf_file = f"input/{prot}-prepare.psf"
        pdb_file = f"input/{prot}-prepare.pdb"
        for state in range(1, max_states[prot] + 1):
            segid = segid_map[prot][state]
            structure_lines.append(f"protocol.initStruct('{psf_file}',erase=False)")
            structure_lines.append(f"protocol.initCoords('{pdb_file}')")

            if not assigned_segids:
                # First segid: select all atoms
                structure_lines.append(
                    f'AtomSel("all").apply( SetProperty(\'segmentName\', \'{segid}\') )'
                )
            else:
                # Exclude already assigned segids
                exclude = " or ".join([f"segid {s}" for s in assigned_segids])
                structure_lines.append(
                    f'AtomSel("all and not ({exclude})").apply( SetProperty(\'segmentName\', \'{segid}\') )'
                )
            assigned_segids.append(segid)

    structure_lines.append("protocol.initNBond(repel=1.2)")

    # ============================================================
    # step2: interaction define
    # ============================================================
    interaction_lines = []
    interaction_lines.append("# step2 interaction define")
    interaction_lines.append('command("""')
    interaction_lines.append("")
    interaction_lines.append("    constraints")
    interaction_lines.append("")

    all_segids = []
    for prot in protein_order:
        for state in range(1, max_states[prot] + 1):
            all_segids.append(segid_map[prot][state])

    # 1. Inter-protein interactions (all pairs of different proteins)
    for i, prot_i in enumerate(protein_order):
        segids_i = [segid_map[prot_i][s] for s in range(1, max_states[prot_i] + 1)]
        for j in range(i + 1, len(protein_order)):
            prot_j = protein_order[j]
            segids_j = [segid_map[prot_j][s] for s in range(1, max_states[prot_j] + 1)]
            for segid_i in segids_i:
                for segid_j in segids_j:
                    interaction_lines.append(f"    inter = (segid {segid_i})(segid {segid_j})")

    # 2. Intra-protein interactions (domain-domain, linker, and loop)
    for prot in protein_order:
        prot_domains = domains.get(prot, [])
        prot_linkers = linkers.get(prot, [])
        prot_loops = loops.get(prot, [])
        segids = [segid_map[prot][s] for s in range(1, max_states[prot] + 1)]

        if len(prot_domains) > 1:
            for segid in segids:
                # Domain-domain interactions within same segid
                for d1_idx in range(len(prot_domains)):
                    for d2_idx in range(d1_idx + 1, len(prot_domains)):
                        s1, e1 = prot_domains[d1_idx]
                        s2, e2 = prot_domains[d2_idx]
                        interaction_lines.append(
                            f"    inter = (segid {segid} and resid {s1}:{e1})(segid {segid} and resid {s2}:{e2})"
                        )

        # Linker region interactions with all
        for segid in segids:
            for ds, de in prot_linkers:
                interaction_lines.append(
                    f"    inter = (segid {segid} and resid {ds}:{de})(all)"
                )

        # Loop region interactions with all
        for segid in segids:
            for ls, le in prot_loops:
                interaction_lines.append(
                    f"    inter = (segid {segid} and resid {ls}:{le})(all)"
                )

    interaction_lines.append("    weights * 1 end end")
    interaction_lines.append("")
    interaction_lines.append('    """)')
    interaction_lines.append("")
    interaction_lines.append('if xplor.p_processID==0:')
    interaction_lines.append('  command("write psf output=complex.psf end")')

    # ============================================================
    # step3: annealing settings
    # ============================================================
    annealing_lines = """
# step3 annealing settings

init_t  = 3000
final_t = 25

cool_steps = 12000

from simulationTools import MultRamp, StaticRamp, InitialParams
rampedParams=[]

potList = PotList()
potList.add( XplorPot("BOND") )

potList.add( XplorPot("ANGL") )
rampedParams.append( MultRamp(0.4,1,"potList['ANGL'].setScale(VALUE)") )

potList.add( XplorPot("IMPR") )
rampedParams.append( MultRamp(0.4,1,"potList['IMPR'].setScale(VALUE)") )

potList.add( XplorPot("VDW") )
rampedParams.append( MultRamp(1.2,0.75,
                              "command('param nbonds repel VALUE end end')") )
rampedParams.append( MultRamp(.004,4,
                              "command('param nbonds rcon VALUE end end')") )
""".strip('\n')

    # ============================================================
    # step4: restraints define
    # ============================================================
    restraint_lines = """
# step4 restraints define
noe=PotList('noe')
potList.append(noe)
from noePotTools import create_NOEPot
pot = create_NOEPot('xlms',"./input/xlms.tbl")
pot.setPotType("hard")
pot.setScale(2)       
pot.setAveType("sum")
noe.append(pot)
rampedParams.append( MultRamp(2,30, "noe.setScale( VALUE )") )
""".strip('\n')

    # ============================================================
    # step5: IVM setup
    # ============================================================
    ivm_lines = []
    ivm_lines.append("# step5 IVM setup")
    ivm_lines.append("")
    ivm_lines.append("dyn  = IVM()")
    ivm_lines.append("")

    # Anchor protein: fix the domain with most cross-links, group others
    anchor = protein_order[0]
    anchor_segids = [segid_map[anchor][s] for s in range(1, max_states[anchor] + 1)]
    anchor_domains = domains.get(anchor, [])

    anchor_loops = loops.get(anchor, [])
    if len(anchor_domains) > 1:
        # Multi-domain anchor: decide fix/group by cross-link site count per domain
        domain_counts = count_anchor_xl_per_domain(anchor, links_list, domains)
        max_count = domain_counts[0][1] if domain_counts else 0
        if max_count == 0:
            # Fallback: fix entire segid if no XL sites fall in any domain
            for segid in anchor_segids:
                if anchor_loops:
                    loop_exclude = " or ".join([f"resid {ls}:{le}" for ls, le in anchor_loops])
                    ivm_lines.append(f'dyn.fix(""" segid {segid} and not ({loop_exclude}) """)')
                else:
                    ivm_lines.append(f'dyn.fix("""segid {segid} """)')
        else:
            for segid in anchor_segids:
                fixed_one = False
                for idx, count, d_start, d_end in domain_counts:
                    domain_loops = [(ls, le) for ls, le in anchor_loops if d_start <= ls and le <= d_end]
                    if count == max_count and not fixed_one:
                        if domain_loops:
                            loop_exclude = " or ".join([f"resid {ls}:{le}" for ls, le in domain_loops])
                            ivm_lines.append(f'dyn.fix(""" segid {segid} and resid {d_start}:{d_end} and not ({loop_exclude}) """)')
                        else:
                            ivm_lines.append(f'dyn.fix(""" segid {segid} and resid {d_start}:{d_end} """)')
                        fixed_one = True
                    else:
                        if domain_loops:
                            loop_exclude = " or ".join([f"resid {ls}:{le}" for ls, le in domain_loops])
                            ivm_lines.append(f'dyn.group(""" segid {segid} and resid {d_start}:{d_end} and not ({loop_exclude}) """)')
                        else:
                            ivm_lines.append(f'dyn.group(""" segid {segid} and resid {d_start}:{d_end} """)')
    else:
        # Single domain anchor: fix entire segid, excluding loops if any
        for segid in anchor_segids:
            if anchor_loops:
                loop_exclude = " or ".join([f"resid {ls}:{le}" for ls, le in anchor_loops])
                ivm_lines.append(f'dyn.fix(""" segid {segid} and not ({loop_exclude}) """)')
            else:
                ivm_lines.append(f'dyn.fix("""segid {segid} """)')

    # Other proteins = group (by domain if multi-domain, otherwise by segid)
    for prot in protein_order[1:]:
        prot_domains = domains.get(prot, [])
        prot_linkers = linkers.get(prot, [])
        prot_loops = loops.get(prot, [])
        segids = [segid_map[prot][s] for s in range(1, max_states[prot] + 1)]

        if len(prot_domains) > 1:
            # Multi-domain: group each domain separately, excluding loops inside the domain
            for segid in segids:
                for d_start, d_end in prot_domains:
                    domain_loops = [(ls, le) for ls, le in prot_loops if d_start <= ls and le <= d_end]
                    if domain_loops:
                        loop_exclude = " or ".join([f"resid {ls}:{le}" for ls, le in domain_loops])
                        ivm_lines.append(
                            f'dyn.group(""" segid {segid} and resid {d_start}:{d_end} and not ({loop_exclude}) """)'
                        )
                    else:
                        ivm_lines.append(
                            f'dyn.group(""" segid {segid} and resid {d_start}:{d_end} """)'
                        )
        else:
            # Single domain: group entire segid, excluding linker and loop regions
            for segid in segids:
                excludes = []
                if prot_linkers:
                    excludes.extend([f"resid {ls}:{le}" for ls, le in prot_linkers])
                if prot_loops:
                    excludes.extend([f"resid {ls}:{le}" for ls, le in prot_loops])
                if excludes:
                    exclude_str = " or ".join(excludes)
                    ivm_lines.append(
                        f'dyn.group(""" segid {segid} and not ({exclude_str}) """)'
                    )
                else:
                    ivm_lines.append(f'dyn.group(""" segid {segid} """)')

    # ============================================================
    # step6: sampling and output
    # ============================================================
    sampling_lines = f"""
# step6 sampling and output

def structLoopAction(loopInfo):

    protocol.initMinimize(dyn, potList=potList)
    InitialParams( rampedParams )
    dyn.run()

    # high temp dynamics

    ini_timestep = 0.010
    potList["VDW"].setScale(0)
    protocol.initDynamics(dyn,
                          potList=potList,
                          bathTemp=init_t,
                          initVelocities=True,
                          stepsize=ini_timestep,
                          finalTime=10,
                          printInterval=100)
    dyn.run()

    # cooling

    timestep=ini_timestep
    potList["VDW"].setScale(1)

    protocol.initDynamics(dyn,
                          potList=potList,
                          bathTemp=init_t,
                          initVelocities=True,
                          stepsize=timestep,
                          finalTime=0.5,
                          printInterval=100)

    dyn.setResetCMInterval( 100 )

    AnnealIVM(initTemp =init_t,
              finalTemp=final_t,
              numSteps = 50,
              ivm=dyn,
              rampedParams = rampedParams).run()
    # final Powell minimization
    protocol.initMinimize(dyn)
    dyn.run()

    loopInfo.writeStructure(potList)

    pass

StructureLoop(numStructures={num_structures},
              pdbTemplate='./output/Calc_STRUCTURE.pdb',
              structLoopAction=structLoopAction).run()
""".strip('\n')

    # ============================================================
    # Assemble full refine.py
    # ============================================================
    lines = []
    lines.append("# import section")
    lines.append("command = xplor.command")
    lines.append("from jCoupPot import JCoupPot")
    lines.append("from noePot import NOEPot")
    lines.append("import prePot")
    lines.append("from xplor import select")
    lines.append("from xplorPot import XplorPot")
    lines.append("from rdcPotTools import *")
    lines.append("from pdbTool import *")
    lines.append("from atomAction import *")
    lines.append("from selectTools import *")
    lines.append("from simulationTools import *")
    lines.append("from ivm import IVM")
    lines.append("import protocol")
    lines.append("import monteCarlo")
    lines.append("protocol.initRandomSeed()   #set random seed - by time")
    lines.append("")
    lines.extend(structure_lines)
    lines.append("")
    lines.extend(interaction_lines)
    lines.append("")
    lines.append(annealing_lines)
    lines.append("")
    lines.append(restraint_lines)
    lines.append("")
    lines.extend(ivm_lines)
    lines.append("")
    lines.append(sampling_lines)

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"  Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate XPLOR input files from X-PIE clustering results"
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='../interface-define',
        help='Directory containing X-PIE output files (clusters, domains, linkers, loops). Default: ../interface-define'
    )
    parser.add_argument(
        '--xplor-dir', '-x',
        default='..',
        help='Directory to write generated XPLOR files. Default: parent directory'
    )
    parser.add_argument(
        '--linker-length', '-l',
        type=float,
        default=DEFAULT_LINKER_LENGTH,
        help=f'Cross-linker arm length in Angstroms. Default: {DEFAULT_LINKER_LENGTH}'
    )
    parser.add_argument(
        '--num-structures', '-n',
        type=int,
        default=DEFAULT_NUM_STRUCTURES,
        help=f'Number of structures to calculate. Default: {DEFAULT_NUM_STRUCTURES}'
    )
    parser.add_argument(
        '--xplor-home',
        default=None,
        help='XPLOR-NIH toppar directory. If provided, refine.py will use the absolute path to parameter files.'
    )
    parser.add_argument(
        '--use-loop-flexibility',
        action='store_true',
        default=False,
        help='Enable separate interaction/group handling for short flexible loops containing cross-link sites.'
    )

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    xplor_dir = os.path.abspath(args.xplor_dir)
    xplor_home = os.path.abspath(args.xplor_home) if args.xplor_home else None

    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)

    os.makedirs(xplor_dir, exist_ok=True)

    print(f"Reading clustering results from: {output_dir}")
    print(f"Writing XPLOR files to: {xplor_dir}")
    print(f"Cross-linker length: {args.linker_length} A")
    print(f"Number of structures: {args.num_structures}")
    print()

    # Find all cluster files
    cluster_files = sorted(glob.glob(os.path.join(output_dir, '*_clusters.txt')))
    if not cluster_files:
        print(f"Error: no *_clusters.txt files found in {output_dir}")
        sys.exit(1)

    print(f"Found cluster files: {[os.path.basename(f) for f in cluster_files]}")

    # Parse cluster files
    links_list = []
    for cf in cluster_files:
        links = parse_cluster_file(cf)
        links_list.append(links)
        print(f"  {os.path.basename(cf)}: {len(links)} cross-link(s)")

    # Parse domains, linker, and loop regions
    domains, linkers, loops = parse_domain_linker_loop(output_dir)
    if not args.use_loop_flexibility:
        loops = {}
    print(f"\nFound domain info for: {list(domains.keys())}")
    print(f"Found linker info for: {list(linkers.keys())}")
    if args.use_loop_flexibility:
        print(f"Found loop info for: {list(loops.keys())}")

    # Build protein ordering and segid mapping
    protein_order, segid_map, max_states = build_protein_info(links_list)

    print(f"\nProtein ordering and segid assignment:")
    for prot in protein_order:
        segids = [segid_map[prot][s] for s in range(1, max_states[prot] + 1)]
        state_str = f"{max_states[prot]} state(s)"
        dom_str = ""
        if prot in domains:
            dom_str = f", domains: {domains[prot]}"
        link_str = ""
        if prot in linkers:
            link_str = f", linkers: {linkers[prot]}"
        loop_str = ""
        if prot in loops:
            loop_str = f", loops: {loops[prot]}"
        print(f"  {prot}: {state_str} -> segid(s): {', '.join(segids)}{dom_str}{link_str}{loop_str}")

    # Generate files
    print()
    input_dir = os.path.join(xplor_dir, 'input')
    os.makedirs(input_dir, exist_ok=True)
    # Collect all cross-link sites from cluster files
    xl_sites_map = {}
    for links in links_list:
        for prot_a, site_a, state_a, prot_b, site_b, state_b in links:
            xl_sites_map.setdefault(prot_a, set()).add(site_a)
            xl_sites_map.setdefault(prot_b, set()).add(site_b)

    tbl_path = os.path.join(input_dir, 'xlms.tbl')
    n_fallback_map = build_n_fallback_map(xplor_dir, xl_sites_map)
    if n_fallback_map:
        print(f"  N-atom fallback for non-Lys sites: {n_fallback_map}")
    generate_xlms_tbl(links_list, segid_map, args.linker_length, tbl_path, n_fallback_map)

    py_path = os.path.join(xplor_dir, 'refine.py')
    generate_refine_py(
        protein_order, segid_map, max_states,
        domains, linkers, loops, links_list,
        args.num_structures, py_path,
        xplor_home=xplor_home
    )

    # Create output directory for XPLOR sampling structures
    out_dir = os.path.join(xplor_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Created: {out_dir}")

    print("\nDone!")


if __name__ == '__main__':
    main()
