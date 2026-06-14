#!/usr/bin/env python3
"""
Prepare PDB and PSF files for XPLOR structure calculation.

This script processes all PDB files in the input directory and generates
cleaned PDBs, topology files (.psf), and hydrogen-rebuilt structures
(-prepare.pdb / -prepare.psf) compatible with XPLOR-NIH.

Usage:
    python scripts/pdbpsf-prepare.py --pdb-dir .. --output-dir ../input

Steps for each PDB file:
    1. Clean PDB: remove H atoms, remove OT2, rename OT1 -> O
    2. Generate PSF via XPLOR topology building
    3. Rebuild H atoms via XPLOR hbuild, minimize, output prepare files
"""

import os
import sys
import argparse
import glob
import shutil
import subprocess

# ============================================================
# Default configuration
# ============================================================

DEFAULT_XPLOR_HOME = "/home/gz/opt/xplor-nih-3.10/toppar"
DEFAULT_XPLOR_BIN = "/home/gz/opt/xplor-nih-3.10/bin/xplor"


def clean_pdb(input_path, output_path):
    """
    Clean a PDB file for XPLOR topology building:
      - Keep only ATOM records
      - Remove hydrogen atoms (col 13 or 14 is 'H')
      - Remove OT2 atoms
      - Rename OT1 -> O (pad to 4 chars)
    """
    # Read all lines first to handle in-place rewriting safely
    with open(input_path, 'r') as fin:
        lines = fin.readlines()

    cleaned = []
    for line in lines:
        if not line.startswith("ATOM"):
            continue

        # PDB atom name is in columns 13-16 (0-indexed: 12-15)
        atom_name = line[12:16]

        # Skip hydrogen atoms
        if atom_name[0] == 'H' or atom_name[1] == 'H':
            continue

        # Skip OT2
        if atom_name.strip() == 'OT2':
            continue

        # Rename OT1 -> O (keep 4-char width, left-aligned)
        if atom_name.strip() == 'OT1':
            line = line[:12] + "O   " + line[16:]

        cleaned.append(line)

    with open(output_path, 'w') as fout:
        fout.writelines(cleaned)


def generate_psf_input(pdbname, xplor_home, output_dir):
    """Generate XPLOR input script for PSF/topology generation."""
    inp_path = os.path.join(output_dir, f"{pdbname}_psf.inp")
    pdb_path = os.path.join(output_dir, f"{pdbname}.pdb")
    psf_path = os.path.join(output_dir, f"{pdbname}.psf")

    content = f"""rtf @{xplor_home}/topallhdg_new.pro end

parameter @{xplor_home}/parallhdg_new.pro end


segment
   name=" "
   SETUP=TRUE
   chain
      @{xplor_home}/toph11.pep

coor @{pdb_path}
end
end
end
delete select (name OT1 or name OT2) end
write psf output={psf_path} end

stop
"""
    with open(inp_path, 'w') as f:
        f.write(content)
    return inp_path


def generate_hbuild_input(pdbname, xplor_home, output_dir):
    """Generate XPLOR input script for hydrogen building and minimization."""
    inp_path = os.path.join(output_dir, f"{pdbname}_hbuild.inp")
    psf_path = os.path.join(output_dir, f"{pdbname}.psf")
    pdb_path = os.path.join(output_dir, f"{pdbname}.pdb")
    prepare_pdb = os.path.join(output_dir, f"{pdbname}-prepare.pdb")
    prepare_psf = os.path.join(output_dir, f"{pdbname}-prepare.psf")

    content = f"""rtf @{xplor_home}/topallhdg_new.pro
end

parameter @{xplor_home}/parallhdg_new.pro
end


structure @{psf_path} end

coor @{pdb_path}

delete  select (name H*) end


hbuild select=(name H*) phistep=360 end
hbuild select=(name H*) phistep=5 end

flags exclude * include bonds angle impr end !
constraint fix (not name H*) end
mini powell nstep 10 end

write coor output={prepare_pdb} end
write psf output={prepare_psf} end
stop
"""
    with open(inp_path, 'w') as f:
        f.write(content)
    return inp_path


def run_xplor(inp_path, xplor_bin=DEFAULT_XPLOR_BIN):
    """Run XPLOR with the given input file."""
    print(f"    Running XPLOR: {xplor_bin} -in {inp_path}")
    try:
        result = subprocess.run(
            [xplor_bin, "-in", inp_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"    XPLOR completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: XPLOR failed with exit code {e.returncode}")
        if e.stdout:
            print(f"    stdout: {e.stdout[:500]}")
        if e.stderr:
            print(f"    stderr: {e.stderr[:500]}")
        return False
    except FileNotFoundError:
        print(f"    ERROR: XPLOR executable not found: {xplor_bin}")
        print(f"    Please ensure XPLOR-NIH is installed and in PATH,")
        print(f"    or specify --xplor-bin explicitly.")
        return False


def process_pdb(pdb_path, xplor_home, xplor_bin, output_dir):
    """Process a single PDB file through the full preparation pipeline."""
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    print(f"\n  Processing: {pdb_name}")

    # Step 1: Clean PDB
    cleaned_pdb = os.path.join(output_dir, f"{pdb_name}.pdb")
    # If output_dir is the same as input dir, backup original first
    if os.path.abspath(pdb_path) == os.path.abspath(cleaned_pdb):
        backup_pdb = os.path.join(output_dir, f"{pdb_name}_raw.pdb")
        shutil.copy2(pdb_path, backup_pdb)
        print(f"    Backed up original PDB -> {backup_pdb}")
    print(f"    Cleaning PDB -> {cleaned_pdb}")
    clean_pdb(pdb_path, cleaned_pdb)

    # Step 2: Generate PSF
    psf_inp = generate_psf_input(pdb_name, xplor_home, output_dir)
    print(f"    Generating PSF input -> {psf_inp}")
    if not run_xplor(psf_inp, xplor_bin):
        print(f"    Skipping H-build for {pdb_name} due to PSF failure")
        return False

    # Step 3: H-build and minimize
    hbuild_inp = generate_hbuild_input(pdb_name, xplor_home, output_dir)
    print(f"    Generating H-build input -> {hbuild_inp}")
    if not run_xplor(hbuild_inp, xplor_bin):
        print(f"    H-build failed for {pdb_name}")
        return False

    print(f"    Success: {pdb_name}-prepare.pdb / {pdb_name}-prepare.psf")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PDB and PSF files for XPLOR structure calculation"
    )
    parser.add_argument(
        '--pdb-dir', '-p',
        default='..',
        help='Directory containing raw PDB files. Default: parent directory'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='../input',
        help='Directory for output files (cleaned PDBs, PSFs, prepare files). Default: ../input'
    )
    parser.add_argument(
        '--xplor-home',
        default=os.environ.get('XPLOR_HOME', DEFAULT_XPLOR_HOME),
        help=f'XPLOR-NIH toppar directory. Default: {DEFAULT_XPLOR_HOME}'
    )
    parser.add_argument(
        '--xplor-bin',
        default=DEFAULT_XPLOR_BIN,
        help=f'XPLOR executable. Default: {DEFAULT_XPLOR_BIN}'
    )
    parser.add_argument(
        '--pattern',
        default='*.pdb',
        help='Glob pattern for PDB files. Default: *.pdb'
    )

    args = parser.parse_args()

    pdb_dir = os.path.abspath(args.pdb_dir)
    output_dir = os.path.abspath(args.output_dir)
    xplor_home = args.xplor_home
    xplor_bin = args.xplor_bin

    # Validate XPLOR home
    if not os.path.isdir(xplor_home):
        print(f"Error: XPLOR home directory not found: {xplor_home}")
        print(f"Please set --xplor-home or environment variable XPLOR_HOME")
        sys.exit(1)

    # Check required topology files
    required_files = [
        os.path.join(xplor_home, 'topallhdg_new.pro'),
        os.path.join(xplor_home, 'parallhdg_new.pro'),
        os.path.join(xplor_home, 'toph11.pep'),
    ]
    for rf in required_files:
        if not os.path.exists(rf):
            print(f"Warning: Required XPLOR file not found: {rf}")

    # Find PDB files
    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, args.pattern)))
    # Exclude already-prepared files, raw backups, and files in output dir (if different from pdb_dir)
    pdb_files = [f for f in pdb_files
                 if not os.path.basename(f).endswith('-prepare.pdb')
                 and not os.path.basename(f).endswith('_raw.pdb')]
    if os.path.abspath(output_dir) != os.path.abspath(pdb_dir):
        pdb_files = [f for f in pdb_files
                     if not f.startswith(os.path.abspath(output_dir) + os.sep)]

    if not pdb_files:
        print(f"No PDB files found in {pdb_dir} matching '{args.pattern}'")
        sys.exit(1)

    print(f"Found {len(pdb_files)} PDB file(s) to process:")
    for f in pdb_files:
        print(f"  - {os.path.basename(f)}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nXPLOR home: {xplor_home}")
    print(f"XPLOR bin:  {xplor_bin}")
    print(f"Output dir: {output_dir}")

    success_count = 0
    for pdb_path in pdb_files:
        if process_pdb(pdb_path, xplor_home, xplor_bin, output_dir):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(pdb_files)} PDB file(s) processed successfully")
    print(f"Output files in: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
