#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from scripts.pdb_homology import run_pdb_homology_workflow
from scripts.psm_processor import PSMProcessor
from scripts.string_api import StringApiClient, detect_species_from_accessions


APP_BANNER = r"""

██╗  ██╗      ██████╗ ██╗███████╗
╚██╗██╔╝      ██╔══██╗██║██╔════╝
 ╚███╔╝ █████╗██████╔╝██║█████╗
 ██╔██╗ ╚════╝██╔═══╝ ██║██╔══╝
██╔╝ ██╗      ██║     ██║███████╗
╚═╝  ╚═╝      ╚═╝     ╚═╝╚══════╝

End-to-end XLMS PPI evaluation workflow
"""

DEFAULT_INPUT_DIR = Path('./input')
DEFAULT_OUTPUT_DIR = Path('./output')
DEFAULT_FDR_PERCENT = 1.0
DEFAULT_MIN_SITE_PAIRS = 1
DEFAULT_IDENTITY_THRESHOLD = 30.0
DEFAULT_STRING_SCORE_THRESHOLD = 0.0
DEFAULT_NETWORK_TYPE = 'functional'
DEFAULT_NETWORK_FAILURE_MODE = 'exit'

NETWORK_CHECK_TARGETS = [
    ('UniProt', 'https://rest.uniprot.org/uniprotkb/P69905.json'),
    ('STRING', 'https://version-12-0.string-db.org'),
    ('RCSB PDB', 'https://data.rcsb.org'),
]

REPORTED_COLUMNS = ['Protein1', 'Protein2', 'SitePairCount', 'STRING_CombinedScore']
PUTATIVE_COLUMNS = ['Protein1', 'Protein2', 'SitePairCount', 'STRING_CombinedScore', 'HasHomologousPDB']
PDB_RESULT_COLUMNS = [
    'Protein1',
    'Protein2',
    'PDB_ID',
    'Protein1_Homologue_UniProt',
    'Protein1_IdentityPct',
    'Protein1_Chain',
    'Protein2_Homologue_UniProt',
    'Protein2_IdentityPct',
    'Protein2_Chain',
]

REQUIRED_COLUMNS = [
    'Peptide_Type',
    'Protein_Type',
    'Score',
    'Target_Decoy',
    'Q-value',
    'Proteins',
]

PROTEIN_PATTERNS = [
    re.compile(r"sp\|([^|]+)\|.+?\((\d+)\)-sp\|([^|]+)\|.+?\((\d+)\)"),
    re.compile(r"(.+?)\s*\((\d+)\)-(.+?)\s*\((\d+)\)"),
]


def log_message(message: str, log_file: Path | None = None) -> None:
    print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as handle:
            handle.write(message + '\n')


def prompt_input(prompt_text: str, default_value: str | int | float) -> str:
    user_input = input(f'{prompt_text} [{default_value}]: ').strip()
    return user_input if user_input else str(default_value)


def prompt_choice(prompt_text: str, choices: dict[str, str], default_key: str) -> str:
    choice_text = '/'.join(f'{key}={label}' for key, label in choices.items())
    while True:
        user_input = input(f'{prompt_text} [{choice_text}, default {default_key}]: ').strip().lower()
        if not user_input:
            return default_key
        if user_input in choices:
            return user_input
        print(f'Please choose one of: {", ".join(choices)}')


def parse_percent(value: str | int | float) -> float:
    percent = float(value)
    if percent < 0 or percent > 100:
        raise ValueError('FDR percent must be between 0 and 100.')
    return percent / 100.0


def parse_score_threshold(value: str | int | float) -> float:
    threshold = float(value)
    if threshold < 0 or threshold > 1:
        raise ValueError('STRING score threshold must be between 0 and 1.')
    return threshold


def normalize_peptide_type(value) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    mapping = {
        'cross-link': 3,
        'crosslink': 3,
    }
    return mapping.get(text.lower())


def normalize_protein_type(value) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    mapping = {
        'intra-protein': 1,
        'inter-protein': 2,
        'none': 1,
    }
    return mapping.get(text.lower())


def check_dependencies(log_file: Path | None = None) -> None:
    log_message('\n' + '=' * 60, log_file)
    log_message('Step 0: Checking dependencies', log_file)
    log_message('=' * 60, log_file)
    missing = []
    for package_name in ('pandas', 'numpy', 'requests', 'Bio'):
        try:
            __import__(package_name)
            log_message(f'  [OK] {package_name} installed', log_file)
        except ImportError:
            missing.append(package_name)
            log_message(f'  [MISSING] {package_name}', log_file)
    if missing:
        raise RuntimeError(f"Missing required packages: {', '.join(missing)}")


def check_internet_access(log_file: Path | None = None) -> tuple[bool, list[str]]:
    log_message('\n' + '=' * 60, log_file)
    log_message('Step 0.5: Checking internet access', log_file)
    log_message('=' * 60, log_file)
    unavailable: list[str] = []
    for service_name, url in NETWORK_CHECK_TARGETS:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code >= 500:
                unavailable.append(service_name)
                log_message(f'  [FAILED] {service_name} returned HTTP {response.status_code}', log_file)
            else:
                log_message(f'  [OK] {service_name} reachable', log_file)
        except Exception as exc:
            unavailable.append(service_name)
            log_message(f'  [FAILED] {service_name}: {exc}', log_file)
    return not unavailable, unavailable


def collect_input_files(input_path: str | Path) -> list[str]:
    abs_path = os.path.abspath(str(input_path))
    if os.path.isfile(abs_path):
        if abs_path.lower().endswith('.csv'):
            return [abs_path]
        raise ValueError(f'Input file is not a CSV: {abs_path}')
    if not os.path.isdir(abs_path):
        raise ValueError(f'Input folder not found: {abs_path}')
    files = sorted(glob.glob(os.path.join(abs_path, '*.csv')))
    if not files:
        raise ValueError(f'No CSV files found in: {abs_path}')
    return files


def load_plink_results(input_dir: str | Path, log_file: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    log_message('\n' + '=' * 60, log_file)
    log_message('Step 1: Loading pLink result files', log_file)
    log_message('=' * 60, log_file)
    csv_files = collect_input_files(input_dir)
    log_message(f'  Found {len(csv_files)} CSV file(s).', log_file)
    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(csv_path)} is missing required columns: {missing}")
        working = df.copy()
        working['Source_File'] = os.path.basename(csv_path)
        frames.append(working)
        log_message(f'  Loaded: {os.path.basename(csv_path)} ({len(df)} rows)', log_file)
    merged = pd.concat(frames, ignore_index=True)
    log_message(f'  Total rows loaded: {len(merged)}', log_file)
    return merged, csv_files


def filter_crosslink_psms(df: pd.DataFrame, fdr_threshold: float, log_file: Path | None = None) -> pd.DataFrame:
    log_message('\n' + '=' * 60, log_file)
    log_message('Step 2: Filtering inter-protein cross-link PSMs', log_file)
    log_message('=' * 60, log_file)
    working = df.copy()
    working['Peptide_Type_norm'] = working['Peptide_Type'].apply(normalize_peptide_type)
    working['Protein_Type_norm'] = working['Protein_Type'].apply(normalize_protein_type)
    crosslink_df = working[working['Peptide_Type_norm'] == 3].copy()
    inter_df = crosslink_df[crosslink_df['Protein_Type_norm'] == 2].copy()
    log_message(f'  Cross-link PSMs: {len(crosslink_df)}', log_file)
    log_message(f'  Inter-protein cross-link PSMs: {len(inter_df)}', log_file)
    filtered = PSMProcessor.filter_psms(inter_df, fdr_threshold).copy()
    log_message(f'  Retained TT PSMs at FDR <= {fdr_threshold * 100:.2f}%: {len(filtered)}', log_file)
    if filtered.empty:
        raise ValueError('No inter-protein TT PSMs remain after FDR filtering.')
    return filtered


def parse_protein_candidates(protein_field) -> list[tuple[str, int, str, int]]:
    candidates: list[tuple[str, int, str, int]] = []
    if pd.isna(protein_field):
        return candidates
    for token in str(protein_field).split('/'):
        entry = token.strip()
        if not entry:
            continue
        for pattern in PROTEIN_PATTERNS:
            match = pattern.fullmatch(entry)
            if not match:
                continue
            protein1, site1, protein2, site2 = match.groups()
            site1 = int(site1)
            site2 = int(site2)
            protein1 = protein1.strip()
            protein2 = protein2.strip()
            if (protein1, site1) > (protein2, site2):
                protein1, protein2 = protein2, protein1
                site1, site2 = site2, site1
            candidates.append((protein1, site1, protein2, site2))
            break
    return candidates


def resolve_site_pairs(filtered_df: pd.DataFrame, log_file: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_message('\n' + '=' * 60, log_file)
    log_message('Step 3: Resolving PPIs and XL site pairs', log_file)
    log_message('=' * 60, log_file)
    pair_frequency: Counter[tuple[str, str]] = Counter()
    parsed_candidates: list[list[tuple[str, int, str, int]]] = []
    for _, row in filtered_df.iterrows():
        candidates = parse_protein_candidates(row['Proteins'])
        parsed_candidates.append(candidates)
        for protein1, _, protein2, _ in candidates:
            pair_frequency[(protein1, protein2)] += 1

    resolved_rows: list[dict[str, object]] = []
    skipped = 0
    for (_, row), candidates in zip(filtered_df.iterrows(), parsed_candidates):
        if not candidates:
            skipped += 1
            continue
        chosen = max(
            candidates,
            key=lambda item: (
                pair_frequency[(item[0], item[2])],
                item[0],
                item[2],
                item[1],
                item[3],
            ),
        )
        protein1, site1, protein2, site2 = chosen
        resolved_rows.append(
            {
                'Protein1': protein1,
                'Site1': site1,
                'Protein2': protein2,
                'Site2': site2,
                'Source_File': row.get('Source_File', ''),
                'Score': row.get('Score'),
                'Q-value': row.get('Q-value'),
            }
        )

    resolved_df = pd.DataFrame(resolved_rows)
    if resolved_df.empty:
        raise ValueError('No valid protein/site pairs could be resolved from the filtered PSMs.')
    if skipped:
        log_message(f'  Skipped rows without parseable Proteins annotations: {skipped}', log_file)

    site_pairs = resolved_df[['Protein1', 'Site1', 'Protein2', 'Site2']].drop_duplicates()
    self_pair_count = int((site_pairs['Protein1'] == site_pairs['Protein2']).sum())
    if self_pair_count:
        log_message(f'  Removed self-pair XL site pairs (Protein1 == Protein2): {self_pair_count}', log_file)
        site_pairs = site_pairs[site_pairs['Protein1'] != site_pairs['Protein2']].copy()
    ppi_table = (
        site_pairs.groupby(['Protein1', 'Protein2'])
        .size()
        .reset_index(name='SitePairCount')
        .sort_values(['SitePairCount', 'Protein1', 'Protein2'], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    log_message(f'  Unique resolved XL site pairs: {len(site_pairs)}', log_file)
    log_message(f'  Unique PPIs before thresholding: {len(ppi_table)}', log_file)
    return site_pairs, ppi_table


def apply_site_pair_threshold(site_pairs: pd.DataFrame, ppi_table: pd.DataFrame, min_site_pairs: int, log_file: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_message('\n' + '=' * 60, log_file)
    log_message('Step 4: Applying XL site-pair threshold', log_file)
    log_message('=' * 60, log_file)
    retained_ppi = ppi_table[ppi_table['SitePairCount'] >= min_site_pairs].copy()
    retained_sites = site_pairs.merge(retained_ppi[['Protein1', 'Protein2']], on=['Protein1', 'Protein2'], how='inner')
    retained_sites = retained_sites.sort_values(['Protein1', 'Protein2', 'Site1', 'Site2']).reset_index(drop=True)
    log_message(f'  Minimum site-pair count: {min_site_pairs}', log_file)
    log_message(f'  Retained PPIs: {len(retained_ppi)}', log_file)
    log_message(f'  Retained XL site pairs: {len(retained_sites)}', log_file)
    return retained_ppi, retained_sites


def write_dat(df: pd.DataFrame, path: Path, columns: list[str], na_rep: str = '') -> Path:
    output = df.copy()
    if output.empty:
        output = pd.DataFrame(columns=columns)
    else:
        output = output.reindex(columns=columns)
    output.to_csv(path, sep=' ', index=False, na_rep=na_rep)
    return path


def write_xlms_outputs(output_dir: Path, ppi_table: pd.DataFrame, site_pairs: pd.DataFrame, log_file: Path | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ppi_path = write_dat(ppi_table, output_dir / 'PPI.dat', ['Protein1', 'Protein2', 'SitePairCount'])
    site_path = write_dat(site_pairs, output_dir / 'PPI_XL_Sites.dat', ['Protein1', 'Site1', 'Protein2', 'Site2'])
    log_message('\n' + '=' * 60, log_file)
    log_message('Step 5: Wrote XLMS summary files', log_file)
    log_message('=' * 60, log_file)
    log_message(f'  PPI.dat: {ppi_path}', log_file)
    log_message(f'  PPI_XL_Sites.dat: {site_path}', log_file)
    return {'ppi': ppi_path, 'sites': site_path}


def build_edge_lookup(network_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for _, row in network_df.iterrows():
        key = tuple(sorted((str(row['stringId_A']), str(row['stringId_B']))))
        lookup[key] = row.to_dict()
    return lookup


def evaluate_ppis_with_string(
    ppi_df: pd.DataFrame,
    species: int,
    score_threshold: float,
    log_file: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_message('\n' + '=' * 60, log_file)
    log_message('Step 6: Evaluating PPIs against STRING', log_file)
    log_message('=' * 60, log_file)
    client = StringApiClient()
    unique_accessions = sorted(set(ppi_df['Protein1']).union(ppi_df['Protein2']))
    mapping = client.map_identifiers(unique_accessions, species)
    mapped_string_ids = [item.string_id for item in mapping.values() if item.string_id]
    network_df = client.fetch_network(
        mapped_string_ids,
        species=species,
        required_score=0,
        network_type=DEFAULT_NETWORK_TYPE,
    )
    edge_lookup = build_edge_lookup(network_df)

    reported_rows: list[dict[str, object]] = []
    putative_rows: list[dict[str, object]] = []
    for _, row in ppi_df.iterrows():
        protein1 = row['Protein1']
        protein2 = row['Protein2']
        map1 = mapping[protein1]
        map2 = mapping[protein2]
        base = {
            'Protein1': protein1,
            'Protein2': protein2,
            'SitePairCount': row['SitePairCount'],
        }
        if not map1.string_id or not map2.string_id:
            putative_rows.append({**base, 'STRING_CombinedScore': pd.NA})
            continue
        edge_key = tuple(sorted((map1.string_id, map2.string_id)))
        edge = edge_lookup.get(edge_key)
        if not edge:
            putative_rows.append({**base, 'STRING_CombinedScore': pd.NA})
            continue
        score = edge.get('score')
        score_value = round(float(score), 3) if pd.notna(score) else pd.NA
        if pd.notna(score_value) and float(score_value) >= score_threshold:
            reported_rows.append(
                {
                    **base,
                    'STRING_CombinedScore': score_value,
                }
            )
        else:
            putative_rows.append(
                {
                    **base,
                    'STRING_CombinedScore': score_value,
                }
            )

    reported_df = pd.DataFrame(reported_rows)
    putative_df = pd.DataFrame(putative_rows)
    if reported_df.empty:
        reported_df = pd.DataFrame(columns=REPORTED_COLUMNS)
    else:
        reported_df = reported_df.reindex(columns=REPORTED_COLUMNS)
    if putative_df.empty:
        putative_df = pd.DataFrame(columns=['Protein1', 'Protein2', 'SitePairCount', 'STRING_CombinedScore'])
    else:
        putative_df = putative_df.reindex(columns=['Protein1', 'Protein2', 'SitePairCount', 'STRING_CombinedScore'])
    log_message(f'  STRING score threshold for Reported_PPI.dat: {score_threshold:.3f}', log_file)
    log_message(f'  Reported PPIs above threshold: {len(reported_df)}', log_file)
    log_message(f'  Putative PPIs for PDB follow-up: {len(putative_df)}', log_file)
    return reported_df, putative_df


def run_pdb_step(output_dir: Path, putative_df: pd.DataFrame, identity_threshold: float, log_file: Path | None = None) -> dict[str, object]:
    log_message('\n' + '=' * 60, log_file)
    log_message('Step 7: Searching homologous PDB structures for putative PPIs', log_file)
    log_message('=' * 60, log_file)
    result = run_pdb_homology_workflow(
        ppi_df=putative_df[['Protein1', 'Protein2', 'SitePairCount']].copy(),
        output_dir=output_dir,
        identity_threshold_percent=identity_threshold,
        logger=lambda message: log_message(message, log_file),
    )
    log_message(f"  Putative PPIs with homologous PDB support: {len(result['result_df'])}", log_file)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run pLink XLMS filtering, STRING evaluation, and homologous PDB evaluation in one workflow.'
    )
    parser.add_argument('--input-dir', default=str(DEFAULT_INPUT_DIR), help='Input CSV file or folder containing pLink result CSV files.')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='Output directory.')
    parser.add_argument('--fdr', type=float, default=DEFAULT_FDR_PERCENT, help='Cross-link PSM FDR threshold in percent.')
    parser.add_argument('--min-site-pairs', type=int, default=DEFAULT_MIN_SITE_PAIRS, help='Minimum unique XL site pairs required to keep a PPI.')
    parser.add_argument(
        '--string-score-threshold',
        type=float,
        default=DEFAULT_STRING_SCORE_THRESHOLD,
        help='STRING combined score threshold in the 0-1 range for Reported_PPI.dat.',
    )
    parser.add_argument('--identity-threshold', type=float, default=DEFAULT_IDENTITY_THRESHOLD, help='Homologous sequence identity threshold in percent for strict PDB support.')
    parser.add_argument(
        '--network-failure-mode',
        choices=['exit', 'local-only'],
        default=DEFAULT_NETWORK_FAILURE_MODE,
        help='Behavior when UniProt/STRING/RCSB cannot be reached.',
    )
    parser.add_argument('--non-interactive', action='store_true', help='Run without prompts and use CLI arguments/default values.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(APP_BANNER)
    script_dir = Path(__file__).resolve().parent
    log_file = script_dir / f"xlms_ppi_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, 'w', encoding='utf-8') as handle:
        handle.write(f'XLMS-PPI-Evaluation Workflow Log\nStarted: {datetime.now()}\n\n')

    try:
        log_message(f'Log file: {log_file}', log_file)
        check_dependencies(log_file)
        internet_ok, unavailable_services = check_internet_access(log_file)

        skip_online_steps = False
        if not internet_ok:
            message = 'Internet connection is required for STRING and PDB annotation'
            log_message(message, log_file)
            if unavailable_services:
                log_message(f'Unavailable services: {", ".join(unavailable_services)}', log_file)
            if args.non_interactive:
                if args.network_failure_mode == 'local-only':
                    skip_online_steps = True
                    log_message('Network failure mode: local-only. The workflow will continue with local XL-MS filtering only.', log_file)
                else:
                    raise RuntimeError(message)
            else:
                action = prompt_choice(
                    'Network-dependent annotation is unavailable. Choose how to continue',
                    {'exit': 'exit', 'local': 'local-only'},
                    'exit',
                )
                if action == 'exit':
                    log_message('Workflow stopped by user because internet access is unavailable.', log_file)
                    return
                skip_online_steps = True
                log_message('User selected local-only execution. STRING and PDB annotation will be skipped.', log_file)

        if args.non_interactive:
            input_dir = Path(args.input_dir)
            output_dir = Path(args.output_dir)
            fdr_threshold = parse_percent(args.fdr)
            min_site_pairs = args.min_site_pairs
            string_score_threshold = parse_score_threshold(args.string_score_threshold)
            identity_threshold = args.identity_threshold
        else:
            input_dir = Path(prompt_input('Enter the input CSV file or folder containing pLink CSV files', args.input_dir))
            output_dir = Path(prompt_input('Enter the output directory', args.output_dir))
            fdr_threshold = parse_percent(prompt_input('Enter the cross-link PSM FDR threshold (%)', args.fdr))
            min_site_pairs = int(prompt_input('Enter the minimum XL site-pair count required to keep a PPI', args.min_site_pairs))
            if skip_online_steps:
                string_score_threshold = parse_score_threshold(args.string_score_threshold)
                identity_threshold = args.identity_threshold
            else:
                string_score_threshold = parse_score_threshold(
                    prompt_input('Enter the STRING score threshold for Reported_PPI.dat (0-1)', args.string_score_threshold)
                )
                identity_threshold = float(prompt_input('Enter the homologous PDB identity threshold (%)', args.identity_threshold))

        merged_df, csv_files = load_plink_results(input_dir, log_file)
        filtered_df = filter_crosslink_psms(merged_df, fdr_threshold, log_file)
        site_pairs, ppi_table = resolve_site_pairs(filtered_df, log_file)
        retained_ppi, retained_sites = apply_site_pair_threshold(site_pairs, ppi_table, min_site_pairs, log_file)
        xlms_outputs = write_xlms_outputs(output_dir, retained_ppi, retained_sites, log_file)

        if skip_online_steps:
            log_message('\n' + '=' * 60, log_file)
            log_message('Workflow complete (local XL-MS filtering only)', log_file)
            log_message('=' * 60, log_file)
            log_message(f'  Input CSV files processed: {len(csv_files)}', log_file)
            log_message(f'  PPI.dat: {xlms_outputs["ppi"]}', log_file)
            log_message(f'  PPI_XL_Sites.dat: {xlms_outputs["sites"]}', log_file)
            log_message('  STRING and PDB annotation were skipped because internet access was unavailable.', log_file)
        else:
            accessions = sorted(set(retained_ppi['Protein1']).union(retained_ppi['Protein2']))
            species = detect_species_from_accessions(accessions)
            log_message('\n' + '=' * 60, log_file)
            log_message('Detected species from UniProt accessions', log_file)
            log_message('=' * 60, log_file)
            log_message(f'  Selected species: {species.scientific_name} ({species.taxon_id})', log_file)
            if species.mixed_taxa:
                mixed_text = ', '.join(f'{taxon}:{count}' for taxon, count in sorted(species.mixed_taxa.items()))
                log_message(f'  Warning: multiple taxa detected; using the dominant species. Other taxa counts: {mixed_text}', log_file)
            if species.unresolved_accessions:
                unresolved = ', '.join(species.unresolved_accessions[:10])
                more = ' ...' if len(species.unresolved_accessions) > 10 else ''
                log_message(f'  Warning: failed to resolve taxon for: {unresolved}{more}', log_file)

            reported_df, putative_df = evaluate_ppis_with_string(retained_ppi, species.taxon_id, string_score_threshold, log_file)
            pdb_result = run_pdb_step(output_dir, putative_df, identity_threshold, log_file)
            pdb_df = pdb_result['result_df']

            has_pdb_keys = set(zip(pdb_df['Protein1'], pdb_df['Protein2'])) if not pdb_df.empty else set()
            putative_output = putative_df.copy()
            if putative_output.empty:
                putative_output = pd.DataFrame(columns=PUTATIVE_COLUMNS)
            else:
                putative_output['HasHomologousPDB'] = [
                    'Yes' if (row.Protein1, row.Protein2) in has_pdb_keys else 'No'
                    for row in putative_output.itertuples(index=False)
                ]
                putative_output = putative_output.reindex(columns=PUTATIVE_COLUMNS)

            output_dir.mkdir(parents=True, exist_ok=True)
            reported_path = write_dat(reported_df, output_dir / 'Reported_PPI.dat', REPORTED_COLUMNS)
            putative_path = write_dat(putative_output, output_dir / 'Putative_PPI.dat', PUTATIVE_COLUMNS, na_rep='NA')
            pdb_path = write_dat(pdb_df, output_dir / 'PDB_homology_results.dat', PDB_RESULT_COLUMNS)

            log_message(f'  Reported_PPI.dat: {reported_path}', log_file)
            log_message(f'  Putative_PPI.dat: {putative_path}', log_file)
            log_message(f'  PDB_homology_results.dat: {pdb_path}', log_file)
            log_message(f'  Reported PPIs: {len(reported_df)}', log_file)
            log_message(f'  Putative PPIs: {len(putative_output)}', log_file)
            log_message(f'  Putative PPIs with homologous PDB support: {len(pdb_df)}', log_file)
            log_message('\n' + '=' * 60, log_file)
            log_message('Workflow complete', log_file)
            log_message('=' * 60, log_file)
            log_message(f'  Input CSV files processed: {len(csv_files)}', log_file)
            log_message(f'  PPI.dat: {xlms_outputs["ppi"]}', log_file)
            log_message(f'  PPI_XL_Sites.dat: {xlms_outputs["sites"]}', log_file)
            log_message(f'  Reported_PPI.dat: {reported_path}', log_file)
            log_message(f'  Putative_PPI.dat: {putative_path}', log_file)
            log_message(f'  PDB_homology_results.dat: {pdb_path}', log_file)
            log_message(f'  Reported PPIs: {len(reported_df)}', log_file)
            log_message(f'  Putative PPIs: {len(putative_output)}', log_file)
            log_message(f'  Putative PPIs with homologous PDB support: {len(pdb_df)}', log_file)
    except Exception as exc:
        log_message(f'\n[ERROR] {exc}', log_file)
        raise


if __name__ == '__main__':
    main()
