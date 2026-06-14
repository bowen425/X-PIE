from __future__ import annotations

import json
import math
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from Bio import BiopythonDeprecationWarning

warnings.simplefilter('ignore', BiopythonDeprecationWarning)

from Bio import pairwise2


RCSB_SEARCH_URL = 'https://search.rcsb.org/rcsbsearch/v2/query'
RCSB_SEARCH_FALLBACK_URL = 'https://search-west.rcsb.org/rcsbsearch/v2/query'
RCSB_GRAPHQL_URL = 'https://data.rcsb.org/graphql'
UNIPROT_URL = 'https://rest.uniprot.org/uniprotkb/{accession}.json'

SEARCH_PAGE_SIZE = 1000
GRAPHQL_CHUNK = 100
EVALUE_CUTOFF = 0.1
REQUEST_TIMEOUT = 90
MAX_RETRIES = 6
RESULT_COLUMNS = [
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


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def sleep_backoff(attempt: int) -> None:
    time.sleep(min(2 ** attempt, 20))


class ApiClient:
    def __init__(self) -> None:
        self._new_session()

    def _new_session(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                'User-Agent': 'Trae-XLMS-PPI-Evaluation-PDB/1.0',
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'Connection': 'close',
            }
        )

    def _decode_json(self, response: requests.Response) -> dict[str, Any]:
        text = response.text.strip()
        if not text:
            raise ValueError('Empty response body')
        if text[0] not in '{[':
            raise ValueError(f'Non-JSON response: {text[:120]}')
        return json.loads(text)

    def get_json(self, url: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT)
                if response.status_code == 204:
                    return {}
                response.raise_for_status()
                return self._decode_json(response)
            except Exception as exc:
                last_error = exc
                self.session.close()
                self._new_session()
                sleep_backoff(attempt)
        raise RuntimeError(f'GET failed for {url}: {last_error}')

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        candidate_urls = [url]
        if url == RCSB_SEARCH_URL:
            candidate_urls.append(RCSB_SEARCH_FALLBACK_URL)
        for attempt in range(MAX_RETRIES):
            for candidate_url in candidate_urls:
                try:
                    response = self.session.post(candidate_url, json=payload, timeout=REQUEST_TIMEOUT)
                    if response.status_code == 204:
                        return {}
                    response.raise_for_status()
                    return self._decode_json(response)
                except Exception as exc:
                    last_error = exc
                    self.session.close()
                    self._new_session()
            sleep_backoff(attempt)
        raise RuntimeError(f'POST failed for {url}: {last_error}')


def best_uniprot_name(record: dict[str, Any]) -> str:
    desc = record.get('proteinDescription', {})
    for key in ('recommendedName', 'submissionNames', 'alternativeNames'):
        if key not in desc:
            continue
        value = desc[key]
        if isinstance(value, list):
            for item in value:
                if item.get('fullName', {}).get('value'):
                    return item['fullName']['value']
        elif isinstance(value, dict):
            full_name = value.get('fullName', {}).get('value')
            if full_name:
                return full_name
    return record.get('uniProtkbId', record.get('primaryAccession', ''))


def make_sequence_query(sequence: str, identity_cutoff: float) -> dict[str, Any]:
    return {
        'type': 'terminal',
        'service': 'sequence',
        'parameters': {
            'evalue_cutoff': EVALUE_CUTOFF,
            'identity_cutoff': identity_cutoff,
            'sequence_type': 'protein',
            'value': sequence,
        },
    }


def entry_filter_query(entry_ids: list[str]) -> dict[str, Any]:
    return {
        'type': 'terminal',
        'service': 'text',
        'parameters': {
            'attribute': 'rcsb_entry_container_identifiers.entry_id',
            'operator': 'in',
            'value': entry_ids,
        },
    }


def search_all_ids(client: ApiClient, query: dict[str, Any], return_type: str) -> list[str]:
    identifiers: list[str] = []
    start = 0
    while True:
        payload = {
            'query': query,
            'return_type': return_type,
            'request_options': {
                'results_content_type': ['experimental'],
                'paginate': {'start': start, 'rows': SEARCH_PAGE_SIZE},
            },
        }
        result = client.post_json(RCSB_SEARCH_URL, payload)
        page = [item['identifier'] for item in result.get('result_set', [])]
        identifiers.extend(page)
        total = int(result.get('total_count', len(page)))
        time.sleep(0.2)
        if not page or len(identifiers) >= total or len(page) < SEARCH_PAGE_SIZE:
            break
        start += SEARCH_PAGE_SIZE
    return identifiers


def local_identity_metrics(query_seq: str, target_seq: str) -> dict[str, float | int]:
    alignments = pairwise2.align.localms(
        query_seq,
        target_seq,
        2,
        -1,
        -5,
        -0.5,
        one_alignment_only=True,
    )
    if not alignments:
        return {
            'identity_pct': 0.0,
            'aligned_length': 0,
            'matches': 0,
            'query_coverage_pct': 0.0,
            'target_coverage_pct': 0.0,
        }
    aligned_query, aligned_target, _, _, _ = alignments[0]
    shared_positions = [
        idx
        for idx, (qa, ta) in enumerate(zip(aligned_query, aligned_target))
        if qa != '-' and ta != '-'
    ]
    if not shared_positions:
        return {
            'identity_pct': 0.0,
            'aligned_length': 0,
            'matches': 0,
            'query_coverage_pct': 0.0,
            'target_coverage_pct': 0.0,
        }
    left = shared_positions[0]
    right = shared_positions[-1] + 1
    trimmed_query = aligned_query[left:right]
    trimmed_target = aligned_target[left:right]
    matches = sum(1 for qa, ta in zip(trimmed_query, trimmed_target) if qa == ta and qa != '-' and ta != '-')
    aligned_length = len(trimmed_query)
    query_residues = sum(1 for aa in trimmed_query if aa != '-')
    target_residues = sum(1 for aa in trimmed_target if aa != '-')
    return {
        'identity_pct': matches / aligned_length * 100,
        'aligned_length': aligned_length,
        'matches': matches,
        'query_coverage_pct': query_residues / len(query_seq) * 100,
        'target_coverage_pct': target_residues / len(target_seq) * 100,
    }


def fetch_protein_records(client: ApiClient, accessions: list[str], logger=None) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for idx, accession in enumerate(accessions, 1):
        obj = client.get_json(UNIPROT_URL.format(accession=accession))
        records[accession] = {
            'accession': accession,
            'name': best_uniprot_name(obj),
            'sequence': obj['sequence']['value'],
            'length': obj['sequence']['length'],
            'organism': obj.get('organism', {}).get('scientificName', ''),
        }
        if logger:
            logger(f'[UniProt] {idx}/{len(accessions)} {accession}')
    return records


def fetch_entity_metadata(client: ApiClient, entity_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not entity_ids:
        return {}
    query = """
    query($entity_ids:[String!]!) {
      polymer_entities(entity_ids:$entity_ids) {
        rcsb_id
        entity_poly {
          pdbx_seq_one_letter_code_can
        }
        rcsb_polymer_entity {
          pdbx_description
        }
        rcsb_polymer_entity_container_identifiers {
          auth_asym_ids
          entry_id
          entity_id
          uniprot_ids
        }
      }
    }
    """
    metadata: dict[str, dict[str, Any]] = {}
    for part in chunked(entity_ids, GRAPHQL_CHUNK):
        result = client.post_json(RCSB_GRAPHQL_URL, {'query': query, 'variables': {'entity_ids': part}})
        for item in result.get('data', {}).get('polymer_entities', []):
            identifiers = item.get('rcsb_polymer_entity_container_identifiers', {})
            metadata[item['rcsb_id']] = {
                'polymer_entity_id': item['rcsb_id'],
                'entry_id': identifiers.get('entry_id', ''),
                'entity_id': identifiers.get('entity_id', ''),
                'chain_ids': identifiers.get('auth_asym_ids', []) or [],
                'sequence': item.get('entity_poly', {}).get('pdbx_seq_one_letter_code_can', ''),
                'pdb_protein_name': item.get('rcsb_polymer_entity', {}).get('pdbx_description', ''),
                'uniprot_ids': identifiers.get('uniprot_ids', []) or [],
            }
    return metadata


def find_positive_entries_for_pair(client: ApiClient, seq_a: str, seq_b: str, identity_cutoff: float) -> list[str]:
    query = {
        'type': 'group',
        'logical_operator': 'and',
        'nodes': [
            make_sequence_query(seq_a, identity_cutoff),
            make_sequence_query(seq_b, identity_cutoff),
        ],
    }
    return search_all_ids(client, query, return_type='entry')


def find_entities_for_sequence_in_entries(client: ApiClient, sequence: str, entry_ids: list[str], identity_cutoff: float) -> list[str]:
    if not entry_ids:
        return []
    query = {
        'type': 'group',
        'logical_operator': 'and',
        'nodes': [make_sequence_query(sequence, identity_cutoff), entry_filter_query(entry_ids)],
    }
    return search_all_ids(client, query, return_type='polymer_entity')


def same_chain_only(a_hits: list[dict[str, Any]], b_hits: list[dict[str, Any]]) -> bool:
    for a_hit in a_hits:
        for b_hit in b_hits:
            for a_chain in a_hit['chain_ids']:
                for b_chain in b_hit['chain_ids']:
                    if not (a_hit['polymer_entity_id'] == b_hit['polymer_entity_id'] and a_chain == b_chain):
                        return False
    return True


def build_result_table(strict_top: pd.DataFrame, chain_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in strict_top.iterrows():
        protein1 = row['Protein1']
        protein2 = row['Protein2']
        pdb_id = row['PDB_ID']
        subset = chain_df[(chain_df['Protein1'] == protein1) & (chain_df['Protein2'] == protein2) & (chain_df['PDB_ID'] == pdb_id)].copy()

        def best_hit(matched_to: str) -> dict[str, Any]:
            hits = subset[subset['MatchedTo'] == matched_to].copy()
            if hits.empty:
                return {}
            hits = hits.sort_values(
                ['IdentityPct', 'QueryCoveragePct', 'EntityCoveragePct', 'ChainID'],
                ascending=[False, False, False, True],
            )
            best = hits.iloc[0]
            same_entity = hits[hits['EntityID'] == best['EntityID']].copy()
            chains = ','.join(sorted(str(x) for x in same_entity['ChainID'].dropna().astype(str).unique()))
            return {
                'uniprot': best['HomologueUniProt'],
                'identity': float(best['IdentityPct']),
                'chain': chains,
            }

        hit1 = best_hit('Protein1')
        hit2 = best_hit('Protein2')
        rows.append(
            {
                'Protein1': protein1,
                'Protein2': protein2,
                'PDB_ID': pdb_id,
                'Protein1_Homologue_UniProt': hit1.get('uniprot', ''),
                'Protein1_IdentityPct': round(hit1.get('identity', float('nan')), 2) if hit1 else None,
                'Protein1_Chain': hit1.get('chain', ''),
                'Protein2_Homologue_UniProt': hit2.get('uniprot', ''),
                'Protein2_IdentityPct': round(hit2.get('identity', float('nan')), 2) if hit2 else None,
                'Protein2_Chain': hit2.get('chain', ''),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(columns=RESULT_COLUMNS)
    else:
        result = result.reindex(columns=RESULT_COLUMNS)
    return result


def run_pdb_homology_workflow(
    ppi_df: pd.DataFrame,
    output_dir: Path,
    identity_threshold_percent: float = 30.0,
    logger=None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'PDB_homology_results.dat'
    identity_cutoff = identity_threshold_percent / 100.0
    client = ApiClient()
    working_df = ppi_df.copy().reset_index(drop=True)
    if working_df.empty:
        empty = pd.DataFrame(columns=RESULT_COLUMNS)
        empty.to_csv(output_path, sep=' ', index=False)
        return {'output_path': output_path, 'result_df': empty}

    accessions = sorted(set(working_df['Protein1']).union(working_df['Protein2']))
    protein_records = fetch_protein_records(client, accessions, logger=logger)

    pair_rows: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []
    all_positive_entry_ids: set[str] = set()
    all_positive_entity_ids: set[str] = set()
    pair_entry_cache: dict[tuple[str, str], list[str]] = {}
    pair_entity_cache: dict[tuple[str, str, tuple[str, ...]], list[str]] = {}

    for idx, row in working_df.iterrows():
        protein1 = row['Protein1']
        protein2 = row['Protein2']
        rec1 = protein_records[protein1]
        rec2 = protein_records[protein2]
        if logger:
            logger(f'[PDB Pair] {idx + 1}/{len(working_df)} {protein1} vs {protein2}')
        positive_entries = find_positive_entries_for_pair(client, rec1['sequence'], rec2['sequence'], identity_cutoff)
        pair_entry_cache[(protein1, protein2)] = positive_entries
        all_positive_entry_ids.update(positive_entries)
        if not positive_entries:
            continue
        key1 = (protein1, 'A', tuple(positive_entries))
        key2 = (protein2, 'B', tuple(positive_entries))
        entities1 = pair_entity_cache.get(key1)
        entities2 = pair_entity_cache.get(key2)
        if entities1 is None:
            entities1 = find_entities_for_sequence_in_entries(client, rec1['sequence'], positive_entries, identity_cutoff)
            pair_entity_cache[key1] = entities1
        if entities2 is None:
            entities2 = find_entities_for_sequence_in_entries(client, rec2['sequence'], positive_entries, identity_cutoff)
            pair_entity_cache[key2] = entities2
        all_positive_entity_ids.update(entities1)
        all_positive_entity_ids.update(entities2)

    entity_meta = fetch_entity_metadata(client, sorted(all_positive_entity_ids))

    for _, row in working_df.iterrows():
        protein1 = row['Protein1']
        protein2 = row['Protein2']
        rec1 = protein_records[protein1]
        rec2 = protein_records[protein2]
        positive_entries = pair_entry_cache[(protein1, protein2)]
        if not positive_entries:
            continue
        entities1 = pair_entity_cache[(protein1, 'A', tuple(positive_entries))]
        entities2 = pair_entity_cache[(protein2, 'B', tuple(positive_entries))]
        hits1_by_entry: dict[str, list[dict[str, Any]]] = defaultdict(list)
        hits2_by_entry: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for entity_id in entities1:
            meta = entity_meta[entity_id]
            metrics = local_identity_metrics(rec1['sequence'], meta['sequence'])
            hit = {**meta, **metrics, 'MatchedTo': 'Protein1', 'QueryAccession': protein1}
            hits1_by_entry[meta['entry_id']].append(hit)

        for entity_id in entities2:
            meta = entity_meta[entity_id]
            metrics = local_identity_metrics(rec2['sequence'], meta['sequence'])
            hit = {**meta, **metrics, 'MatchedTo': 'Protein2', 'QueryAccession': protein2}
            hits2_by_entry[meta['entry_id']].append(hit)

        for entry_id in positive_entries:
            hits1 = hits1_by_entry.get(entry_id, [])
            hits2 = hits2_by_entry.get(entry_id, [])
            if not hits1 or not hits2:
                continue
            if same_chain_only(hits1, hits2):
                continue
            best_identity1 = max(hit['identity_pct'] for hit in hits1)
            best_identity2 = max(hit['identity_pct'] for hit in hits2)
            pair_rows.append(
                {
                    'Protein1': protein1,
                    'Protein2': protein2,
                    'PDB_ID': entry_id,
                    'BestIdentity1': round(best_identity1, 2),
                    'BestIdentity2': round(best_identity2, 2),
                    'PairRankScore': min(best_identity1, best_identity2),
                }
            )
            for hit in hits1 + hits2:
                homologue_uniprot = ','.join(hit['uniprot_ids']) if hit['uniprot_ids'] else ''
                for chain_id in hit['chain_ids']:
                    chain_rows.append(
                        {
                            'Protein1': protein1,
                            'Protein2': protein2,
                            'PDB_ID': entry_id,
                            'MatchedTo': hit['MatchedTo'],
                            'EntityID': hit['entity_id'],
                            'HomologueUniProt': homologue_uniprot,
                            'ChainID': chain_id,
                            'IdentityPct': round(hit['identity_pct'], 2),
                            'QueryCoveragePct': round(hit['query_coverage_pct'], 2),
                            'EntityCoveragePct': round(hit['target_coverage_pct'], 2),
                        }
                    )

    summary_df = pd.DataFrame(pair_rows)
    chain_df = pd.DataFrame(chain_rows)
    if summary_df.empty:
        result_df = pd.DataFrame(columns=RESULT_COLUMNS)
        result_df.to_csv(output_path, sep=' ', index=False)
        return {'output_path': output_path, 'result_df': result_df}

    strict_df = summary_df[(summary_df['BestIdentity1'] > identity_threshold_percent) & (summary_df['BestIdentity2'] > identity_threshold_percent)].copy()
    if strict_df.empty:
        result_df = pd.DataFrame(columns=RESULT_COLUMNS)
        result_df.to_csv(output_path, sep=' ', index=False)
        return {'output_path': output_path, 'result_df': result_df}

    strict_top = (
        strict_df.sort_values(
            ['Protein1', 'Protein2', 'PairRankScore', 'BestIdentity1', 'BestIdentity2'],
            ascending=[True, True, False, False, False],
        )
        .groupby(['Protein1', 'Protein2'], as_index=False)
        .first()
    )
    result_df = build_result_table(strict_top, chain_df)
    result_df.to_csv(output_path, sep=' ', index=False)
    return {'output_path': output_path, 'result_df': result_df}
