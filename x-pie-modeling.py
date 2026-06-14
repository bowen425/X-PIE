#!/usr/bin/env python3
"""
X-PIE: Unified framework for cross-linking based protein complex modeling.

This script orchestrates the complete workflow:
    1. xl_analysis.py       : analyze cross-links, cluster interfaces
    2. generate-script.py   : generate XPLOR input files (xlms.tbl, refine.py)
    3. pdbpsf-prepare.py    : prepare PDB/PSF files for XPLOR

Usage (interactive mode):
    python x-pie.py

Usage (batch mode with config file):
    python x-pie.py --config x-pie.cfg
"""

import os
import sys
import argparse
import configparser
import subprocess
import shutil
import glob
import time
from datetime import datetime

# ============================================================
# ASCII Art
# ============================================================
XPIE_LOGO = r"""

██╗  ██╗      ██████╗ ██╗███████╗
╚██╗██╔╝      ██╔══██╗██║██╔════╝
 ╚███╔╝ █████╗██████╔╝██║█████╗
 ██╔██╗ ╚════╝██╔═══╝ ██║██╔══╝
██╔╝ ██╗      ██║     ██║███████╗
╚═╝  ╚═╝      ╚═╝     ╚═╝╚══════╝

Cross-linking Guided Protein Interaction Modeling

"""

# ============================================================
# Utility functions
# ============================================================

def log_message(msg, log_file=None):
    """Print to screen and optionally write to log file."""
    print(msg)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(msg + '\n')


def run_command(cmd, log_file=None, cwd=None, env=None):
    """Run a shell command and capture output."""
    log_message(f"\n[CMD] {' '.join(cmd)}", log_file)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd, check=False, env=env
        )
        if result.stdout:
            log_message(result.stdout, log_file)
        if result.returncode != 0:
            log_message(f"[ERROR] Exit code {result.returncode}", log_file)
            if result.stderr:
                log_message(result.stderr, log_file)
            return False
        return True
    except FileNotFoundError:
        log_message(f"[ERROR] Command not found: {cmd[0]}", log_file)
        return False


def prompt_input(prompt_text, default_value):
    """Prompt user for input with a default value."""
    user_input = input(f"{prompt_text} [{default_value}]: ").strip()
    return user_input if user_input else str(default_value)


def validate_pdb_files(link_file, pdb_dir):
    """
    Parse cross-link file and verify that every protein has a corresponding .pdb file.
    Returns: (is_valid, missing_proteins, all_proteins)
    """
    proteins = set()
    if not os.path.exists(link_file):
        return False, [f"link file not found: {link_file}"], []
    
    with open(link_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                proteins.add(parts[0])
                proteins.add(parts[2])
    
    missing = []
    for prot in sorted(proteins):
        pdb_path = os.path.join(pdb_dir, f"{prot}.pdb")
        if not os.path.exists(pdb_path):
            missing.append(prot)
    
    return len(missing) == 0, missing, sorted(proteins)


def pause_for_edit(message, timeout=None):
    """
    Pause to allow user to edit generated files.
    If timeout is set (batch mode), auto-continue after timeout seconds.
    """
    if timeout and timeout > 0:
        print(f"\n{message}")
        print(f"  Auto-continuing in {timeout} seconds...")
        try:
            import threading
            import signal

            def alarm_handler(signum, frame):
                raise TimeoutError

            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(timeout)
            input("  Press Enter to continue immediately... ")
            signal.alarm(0)
        except TimeoutError:
            print("  Timeout reached, continuing...")
        except Exception:
            # Fallback for Windows or other platforms
            time.sleep(timeout)
            print("  Timeout reached, continuing...")
    else:
        input(f"\n{message}\n  Press Enter when ready to continue... ")


# ============================================================
# Step 0: Dependency Check
# ============================================================

def check_dependencies(cfg, log_file=None):
    """Check that all required components are installed."""
    log_message("\n" + "=" * 60, log_file)
    log_message("Step 0: Checking dependencies", log_file)
    log_message("=" * 60, log_file)

    missing = []

    # 1. Python packages
    try:
        import Bio.PDB
        log_message("  [OK] Bio.PDB (Biopython) installed", log_file)
    except ImportError:
        log_message("  [MISSING] Bio.PDB (Biopython)", log_file)
        missing.append("biopython")
        log_message("      -> Install: pip install biopython  or  conda install -c conda-forge biopython", log_file)

    # 2. DSSP (mkdssp)
    dssp_cmd = shutil.which('mkdssp')
    if dssp_cmd:
        log_message(f"  [OK] DSSP found: {dssp_cmd}", log_file)
    else:
        log_message("  [MISSING] DSSP (mkdssp)", log_file)
        missing.append("DSSP / mkdssp")
        log_message("      -> Install: conda install -c salilab dssp  or  conda install -c bioconda dssp", log_file)
        log_message("         (On Ubuntu/Debian: sudo apt-get install dssp)", log_file)

    # 3. XPLOR
    xplor_bin = cfg.get('paths', 'xplor_bin', fallback=None)
    if not xplor_bin or not os.path.exists(xplor_bin):
        # Try default
        xplor_bin = shutil.which('xplor') or '/home/gz/opt/xplor-nih-3.10/bin/xplor'
    if os.path.exists(xplor_bin):
        log_message(f"  [OK] XPLOR found: {xplor_bin}", log_file)
    else:
        log_message("  [MISSING] XPLOR-NIH", log_file)
        missing.append("XPLOR-NIH")
        log_message("      -> Please download and install XPLOR-NIH from the official website:", log_file)
        log_message("         https://nmr.cit.nih.gov/xplor-nih/", log_file)
        log_message("         After installation, set the path in x-pie.cfg or ensure 'xplor' is in your PATH.", log_file)

    # 4. NumPy, matplotlib
    try:
        import numpy
        log_message("  [OK] NumPy installed", log_file)
    except ImportError:
        log_message("  [MISSING] NumPy", log_file)
        missing.append("numpy")
        log_message("      -> Install: pip install numpy  or  conda install numpy", log_file)

    try:
        import matplotlib
        log_message("  [OK] Matplotlib installed", log_file)
    except ImportError:
        log_message("  [MISSING] Matplotlib", log_file)
        missing.append("matplotlib")
        log_message("      -> Install: pip install matplotlib  or  conda install matplotlib", log_file)

    if missing:
        log_message("\n[!] Missing dependencies:", log_file)
        for m in missing:
            log_message(f"    - {m}", log_file)
        log_message("\nPlease install the missing components and rerun.", log_file)
        sys.exit(1)
    else:
        log_message("\n[OK] All dependencies satisfied.", log_file)

    return xplor_bin


# ============================================================
# Step 1: xl_analysis.py
# ============================================================

def run_xl_analysis(linker_length, use_loop_flexibility, link_file, pdb_dir, log_file=None):
    """Run xl_analysis.py with the given linker length and loop flexibility option."""
    log_message("\n" + "=" * 60, log_file)
    log_message("Step 1: Running xl_analysis.py", log_file)
    log_message("=" * 60, log_file)

    cmd = [sys.executable, 'scripts/xl_analysis.py']
    env = os.environ.copy()
    # Pass link file and pdb dir via environment variables
    env['XPIE_LINK_FILE'] = link_file
    env['XPIE_PDB_DIR'] = pdb_dir

    # Read xl_analysis.py and replace constants
    with open('scripts/xl_analysis.py', 'r') as f:
        content = f.read()

    # Backup original
    if not os.path.exists('scripts/xl_analysis.py.bak'):
        with open('scripts/xl_analysis.py.bak', 'w') as f:
            f.write(content)

    # Replace constants
    import re
    new_content = re.sub(
        r'CROSS_LINKER_LENGTH\s*=\s*\d+',
        f'CROSS_LINKER_LENGTH = {linker_length}',
        content
    )
    new_content = re.sub(
        r'USE_LOOP_FLEXIBILITY\s*=\s*(True|False)',
        f'USE_LOOP_FLEXIBILITY = {use_loop_flexibility}',
        new_content
    )
    with open('scripts/xl_analysis.py', 'w') as f:
        f.write(new_content)

    success = run_command(cmd, log_file=log_file, env=env)

    # Restore original if failed
    if not success:
        with open('scripts/xl_analysis.py.bak', 'r') as f:
            original = f.read()
        with open('scripts/xl_analysis.py', 'w') as f:
            f.write(original)
        return False

    return True


def parse_analysis_results(output_dir='./interface-define', log_file=None):
    """Parse xl_analysis.py output and generate summary."""
    summary_lines = []
    summary_lines.append("\n" + "=" * 60)
    summary_lines.append("Analysis Results Summary")
    summary_lines.append("=" * 60)

    # Find cluster files
    cluster_files = sorted(glob.glob(os.path.join(output_dir, '*_clusters.txt')))
    if not cluster_files:
        summary_lines.append("  No cluster files found.")
        return '\n'.join(summary_lines)

    # Parse domains, linkers, and loops
    domains = {}
    linkers = {}
    loops = {}
    linker_files = {f[:-len('_linker.txt')] for f in os.listdir(output_dir) if f.endswith('_linker.txt')}
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if fname.endswith('_domains.txt'):
            prot = fname[:-len('_domains.txt')]
            with open(fpath, 'r') as f:
                content = f.read().strip()
            domains[prot] = content.split(':')[-1].strip() if content else 'none'
        elif fname.endswith('_linker.txt'):
            prot = fname[:-len('_linker.txt')]
            with open(fpath, 'r') as f:
                content = f.read().strip()
            linkers[prot] = content.split(':')[-1].strip() if content else 'none'
        elif fname.endswith('_disordered.txt'):
            prot = fname[:-len('_disordered.txt')]
            if prot not in linker_files:
                with open(fpath, 'r') as f:
                    content = f.read().strip()
                linkers[prot] = content.split(':')[-1].strip() if content else 'none'
        elif fname.endswith('_loop.txt'):
            prot = fname[:-len('_loop.txt')]
            with open(fpath, 'r') as f:
                content = f.read().strip()
            loops[prot] = content.split(':')[-1].strip() if content else 'none'

    for cf in cluster_files:
        prot_a_states = {}
        prot_b_states = {}
        with open(cf, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    try:
                        pa, sa, sta_a, pb, sb, sta_b = parts[0], int(parts[1]), int(parts[2]), parts[3], int(parts[4]), int(parts[5])
                        prot_a_states[pa] = max(prot_a_states.get(pa, 0), sta_a)
                        prot_b_states[pb] = max(prot_b_states.get(pb, 0), sta_b)
                    except ValueError:
                        continue

        summary_lines.append(f"\n  Cluster file: {os.path.basename(cf)}")
        all_prots = sorted(set(list(prot_a_states.keys()) + list(prot_b_states.keys())))
        for prot in all_prots:
            n_states = max(prot_a_states.get(prot, 0), prot_b_states.get(prot, 0))
            dom = domains.get(prot, 'none')
            link = linkers.get(prot, 'none')
            loop = loops.get(prot, 'none')
            summary_lines.append(f"    {prot}:")
            summary_lines.append(f"      Domains     : {dom}")
            summary_lines.append(f"      Linkers     : {link}")
            summary_lines.append(f"      Loops       : {loop}")
            summary_lines.append(f"      Interface states: {n_states}")

    summary = '\n'.join(summary_lines)
    log_message(summary, log_file)
    return summary


# ============================================================
# Step 2: generate-script.py
# ============================================================

def run_generate_script(linker_length, num_structures, xplor_home, use_loop_flexibility, log_file=None):
    """Run generate-script.py to create XPLOR input files."""
    log_message("\n" + "=" * 60, log_file)
    log_message("Step 2: Running generate-script.py", log_file)
    log_message("=" * 60, log_file)

    cmd = [
        sys.executable, 'scripts/generate-script.py',
        '--output-dir', './interface-define',
        '--xplor-dir', '.',
        '--linker-length', str(linker_length),
        '--num-structures', str(num_structures)
    ]
    if xplor_home:
        cmd.extend(['--xplor-home', xplor_home])
    if use_loop_flexibility:
        cmd.append('--use-loop-flexibility')
    return run_command(cmd, log_file=log_file)


# ============================================================
# Step 3: pdbpsf-prepare.py
# ============================================================

def run_pdbpsf_prepare(xplor_home, xplor_bin, pdb_dir, log_file=None):
    """Run pdbpsf-prepare.py to prepare PDB/PSF files."""
    log_message("\n" + "=" * 60, log_file)
    log_message("Step 3: Running pdbpsf-prepare.py", log_file)
    log_message("=" * 60, log_file)

    cmd = [
        sys.executable, 'scripts/pdbpsf-prepare.py',
        '--pdb-dir', pdb_dir,
        '--output-dir', './input',
        '--xplor-home', xplor_home,
        '--xplor-bin', xplor_bin
    ]
    return run_command(cmd, log_file=log_file)


# ============================================================
# Step 4: Launch sampling
# ============================================================

def launch_sampling(xplor_bin, num_cores, log_file=None):
    """Launch XPLOR sampling in background."""
    log_message("\n" + "=" * 60, log_file)
    log_message("Step 4: Launching XPLOR sampling", log_file)
    log_message("=" * 60, log_file)

    log_file_out = "xplor_sampling.log"
    cmd = f"nohup {xplor_bin} -py refine.py -smp {num_cores} > {log_file_out} 2>&1 &"

    log_message(f"  Command: {cmd}", log_file)
    log_message(f"  Output redirected to: {log_file_out}", log_file)
    log_message("  Sampling is running in background.", log_file)

    os.system(cmd)
    return True


# ============================================================
# Config file handling
# ============================================================

def load_config(config_path):
    """Load configuration from .cfg file."""
    cfg = configparser.ConfigParser()
    if config_path and os.path.exists(config_path):
        cfg.read(config_path)
        print(f"[INFO] Loaded config: {config_path}")
    else:
        # Default config
        cfg['data'] = {
            'link_file': './link.dat',
            'pdb_dir': '.'
        }
        cfg['paths'] = {
            'xplor_home': '/home/gz/opt/xplor-nih-3.10/toppar',
            'xplor_bin': '/home/gz/opt/xplor-nih-3.10/bin/xplor'
        }
        cfg['parameters'] = {
            'linker_length': '15',
            'num_structures': '24',
            'num_cores': '8',
            'use_loop_flexibility': 'false'
        }
        cfg['workflow'] = {
            'interactive': 'true',
            'pause_after_analysis': 'true',
            'pause_after_generate': 'true',
            'auto_run_sampling': 'false',
            'edit_timeout_seconds': '0'
        }
    return cfg


def write_default_config(path='x-pie.cfg'):
    """Write a default config file for user reference."""
    cfg = configparser.ConfigParser()
    cfg['data'] = {
        'link_file': './link.dat',
        'pdb_dir': '.'
    }
    cfg['paths'] = {
        'xplor_home': '/home/gz/opt/xplor-nih-3.10/toppar',
        'xplor_bin': '/home/gz/opt/xplor-nih-3.10/bin/xplor'
    }
    cfg['parameters'] = {
        'linker_length': '15',
        'num_structures': '24',
        'num_cores': '8',
        'use_loop_flexibility': 'false'
    }
    cfg['workflow'] = {
        'interactive': 'true',
        'pause_after_analysis': 'true',
        'pause_after_generate': 'true',
        'auto_run_sampling': 'false',
        'edit_timeout_seconds': '0'
    }
    with open(path, 'w') as f:
        cfg.write(f)
    print(f"[INFO] Default config written to: {path}")


# ============================================================
# Main workflow
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='X-PIE: Unified framework for XL-MS guided modeling'
    )
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration file (.cfg). If not provided, defaults to x-pie.cfg.'
    )
    parser.add_argument(
        '--generate-config',
        action='store_true',
        help='Generate a default config file (x-pie.cfg) and exit.'
    )
    args = parser.parse_args()

    if args.generate_config:
        write_default_config()
        return

    # Determine config path: if not specified, default to x-pie.cfg
    config_path = args.config if args.config else 'x-pie.cfg'
    
    # If config file does not exist, create it and ask user to edit before proceeding
    if not os.path.exists(config_path):
        write_default_config(config_path)
        print(f"\n[INFO] Config file '{config_path}' not found.")
        print("       A default config file has been generated.")
        print("       Please edit the XPLOR paths in this file before running again.")
        print("\n       Key fields to edit under [paths]:")
        print("         xplor_home = /path/to/your/xplor-nih/toppar")
        print("         xplor_bin  = /path/to/your/xplor-nih/bin/xplor")
        print(f"\n       After editing, run: python x-pie-modeling.py")
        if args.config:
            print(f"                  or: python x-pie-modeling.py --config {args.config}")
        return

    # Print logo
    print(XPIE_LOGO)
    time.sleep(0.5)

    # Load config
    cfg = load_config(config_path)
    
    # Default mode is interactive. Only use batch/auto mode when --config is explicitly provided.
    if args.config is None:
        is_interactive = True
        pause_after_analysis = True
        pause_after_generate = True
        auto_run_sampling = False
        edit_timeout = 0
    else:
        is_interactive = cfg.getboolean('workflow', 'interactive', fallback=False)
        pause_after_analysis = cfg.getboolean('workflow', 'pause_after_analysis', fallback=True)
        pause_after_generate = cfg.getboolean('workflow', 'pause_after_generate', fallback=True)
        auto_run_sampling = cfg.getboolean('workflow', 'auto_run_sampling', fallback=False)
        edit_timeout = cfg.getint('workflow', 'edit_timeout_seconds', fallback=0)

    # XPLOR paths: always read from config first, allow interactive override
    default_xplor_home = cfg.get('paths', 'xplor_home', fallback='/home/gz/opt/xplor-nih-3.10/toppar')
    default_xplor_bin = cfg.get('paths', 'xplor_bin', fallback='/home/gz/opt/xplor-nih-3.10/bin/xplor')
    
    if is_interactive:
        xplor_home = prompt_input("Enter XPLOR toppar directory", default_xplor_home)
        xplor_bin = prompt_input("Enter XPLOR executable path", default_xplor_bin)
    else:
        xplor_home = default_xplor_home
        xplor_bin = default_xplor_bin
    
    cfg.set('paths', 'xplor_home', xplor_home)
    cfg.set('paths', 'xplor_bin', xplor_bin)

    default_linker = cfg.getint('parameters', 'linker_length', fallback=15)
    default_structures = cfg.getint('parameters', 'num_structures', fallback=24)
    default_cores = cfg.getint('parameters', 'num_cores', fallback=8)
    default_use_loop_flex = cfg.getboolean('parameters', 'use_loop_flexibility', fallback=False)

    # Log file
    log_file = f"x-pie_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, 'w') as f:
        f.write(f"X-PIE Workflow Log\nStarted: {datetime.now()}\n\n")

    log_message(f"\nLog file: {log_file}", log_file)

    # Step 0: Check dependencies
    xplor_bin = check_dependencies(cfg, log_file)
    cfg.set('paths', 'xplor_bin', xplor_bin)

    # Get cross-link data file and PDB directory
    default_link_file = cfg.get('data', 'link_file', fallback='./link.dat')
    default_pdb_dir = cfg.get('data', 'pdb_dir', fallback='.')
    
    if is_interactive:
        link_file = prompt_input("Enter cross-link data file path", default_link_file)
        pdb_dir = prompt_input("Enter PDB file directory", default_pdb_dir)
    else:
        link_file = default_link_file
        pdb_dir = default_pdb_dir
        log_message(f"\n[CFG] Using link_file = {link_file}", log_file)
        log_message(f"[CFG] Using pdb_dir = {pdb_dir}", log_file)
    
    # Validate that every protein in link file has a corresponding .pdb file
    is_valid, missing, all_prots = validate_pdb_files(link_file, pdb_dir)
    if not is_valid:
        log_message(f"\n[ERROR] The following proteins are listed in {link_file} but have no corresponding .pdb file in {pdb_dir}/:", log_file)
        for prot in missing:
            log_message(f"    - {prot}.pdb (missing)", log_file)
        log_message("\nPlease check your PDB files and try again.", log_file)
        sys.exit(1)
    else:
        log_message(f"\n[OK] All {len(all_prots)} proteins from {link_file} have corresponding .pdb files in {pdb_dir}/", log_file)
    
    # Update config with user-provided values
    if not cfg.has_section('data'):
        cfg.add_section('data')
    cfg.set('data', 'link_file', link_file)
    cfg.set('data', 'pdb_dir', pdb_dir)

    # Step 1: xl_analysis.py
    if is_interactive:
        linker_length = int(prompt_input(
            "Enter cross-linker arm length (Angstroms)", default_linker
        ))
        use_loop_flexibility = prompt_input(
            "Enable flexible loop handling for intra-domain cross-link sites? (y/n)",
            'y' if default_use_loop_flex else 'n'
        ).strip().lower() in ('y', 'yes')
    else:
        linker_length = default_linker
        use_loop_flexibility = default_use_loop_flex
        log_message(f"\n[CFG] Using linker_length = {linker_length}", log_file)
        log_message(f"[CFG] Using use_loop_flexibility = {use_loop_flexibility}", log_file)

    success = run_xl_analysis(linker_length, use_loop_flexibility, link_file, pdb_dir, log_file)

    # Restore original xl_analysis.py regardless of success
    if os.path.exists('scripts/xl_analysis.py.bak'):
        with open('scripts/xl_analysis.py.bak', 'r') as f:
            original = f.read()
        with open('scripts/xl_analysis.py', 'w') as f:
            f.write(original)
        os.remove('scripts/xl_analysis.py.bak')

    if not success:
        log_message("\n[ABORT] xl_analysis.py failed. Check log for details.", log_file)
        sys.exit(1)

    # Parse and summarize results
    parse_analysis_results('./interface-define', log_file)

    # Pause for user to edit if needed
    if is_interactive and pause_after_analysis:
        pause_for_edit(
            "You may now review/edit the output files in ./interface-define/ "
            "(e.g., domain definitions, cluster results).",
            timeout=edit_timeout if not is_interactive else None
        )
    elif not is_interactive and pause_after_analysis and edit_timeout > 0:
        pause_for_edit(
            "Auto-pausing to allow file review/edit.",
            timeout=edit_timeout
        )

    # Step 2: generate-script.py
    if is_interactive:
        num_structures = int(prompt_input(
            "Enter number of structures to calculate", default_structures
        ))
    else:
        num_structures = default_structures
        log_message(f"\n[CFG] Using num_structures = {num_structures}", log_file)

    if not run_generate_script(linker_length, num_structures, xplor_home, use_loop_flexibility, log_file):
        log_message("\n[ABORT] generate-script.py failed.", log_file)
        sys.exit(1)

    if is_interactive and pause_after_generate:
        pause_for_edit(
            "You may now review/edit xlms.tbl and refine.py before sampling.",
            timeout=edit_timeout if not is_interactive else None
        )
    elif not is_interactive and pause_after_generate and edit_timeout > 0:
        pause_for_edit(
            "Auto-pausing to allow review of xlms.tbl and refine.py.",
            timeout=edit_timeout
        )

    # Step 3: pdbpsf-prepare.py
    log_message("\n[!] Note: PDB/PSF preparation may take several minutes per protein.", log_file)
    if is_interactive:
        input("  Press Enter to start PDB/PSF preparation... ")

    if not run_pdbpsf_prepare(xplor_home, xplor_bin, pdb_dir, log_file):
        log_message("\n[ABORT] pdbpsf-prepare.py failed.", log_file)
        sys.exit(1)

    # Step 4: Sampling
    if is_interactive:
        response = input("\nLaunch XPLOR sampling now? (y/n) [n]: ").strip().lower()
        run_sampling = response in ('y', 'yes')
    else:
        run_sampling = auto_run_sampling
        if run_sampling:
            log_message("\n[CFG] auto_run_sampling = true, launching sampling.", log_file)

    if run_sampling:
        if is_interactive:
            num_cores = int(prompt_input("Enter number of CPU cores to use", default_cores))
        else:
            num_cores = default_cores
        launch_sampling(xplor_bin, num_cores, log_file)
    else:
        log_message("\n[INFO] Sampling not launched.", log_file)
        log_message("  To run sampling manually:", log_file)
        log_message(f"    {xplor_bin} -py refine.py -smp {default_cores}", log_file)

    log_message("\n" + "=" * 60, log_file)
    log_message("X-PIE workflow complete!", log_file)
    log_message("=" * 60, log_file)
    log_message(f"Log saved to: {log_file}", log_file)


if __name__ == '__main__':
    main()
