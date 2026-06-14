from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


STRING_API_BASE = 'https://version-12-0.string-db.org/api'
UNIPROT_URL = 'https://rest.uniprot.org/uniprotkb/{accession}.json'
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
MIN_CALL_INTERVAL_SECONDS = 1.0
MAP_BATCH_SIZE = 500

MAP_COLUMNS = [
    'query_item',
    'query_index',
    'string_id',
    'ncbi_taxon_id',
    'species_name',
    'preferred_name',
    'annotation',
]

NETWORK_COLUMNS = [
    'stringId_A',
    'stringId_B',
    'preferredName_A',
    'preferredName_B',
    'ncbiTaxonId',
    'score',
    'nscore',
    'fscore',
    'pscore',
    'ascore',
    'escore',
    'dscore',
    'tscore',
]


@dataclass
class StringMapping:
    query_accession: str
    mapped_query: str | None
    string_id: str | None
    preferred_name: str | None
    annotation: str | None
    mapping_source: str


@dataclass
class SpeciesDetection:
    taxon_id: int
    scientific_name: str
    mixed_taxa: dict[int, int]
    unresolved_accessions: list[str]


def candidate_identifiers(accession: str) -> list[str]:
    accession = str(accession).strip()
    candidates = [accession]
    if '-' in accession:
        canonical = accession.split('-', 1)[0]
        if canonical and canonical not in candidates:
            candidates.append(canonical)
    return candidates


def _new_json_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            'User-Agent': user_agent,
            'Accept': 'application/json, text/plain, */*',
            'Connection': 'close',
        }
    )
    return session


def fetch_uniprot_record(accession: str, session: requests.Session | None = None) -> dict[str, Any] | None:
    owned_session = False
    if session is None:
        session = _new_json_session('Trae-XLMS-PPI-Evaluation/1.0')
        owned_session = True
    try:
        for candidate in candidate_identifiers(accession):
            last_error: Exception | None = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = session.get(UNIPROT_URL.format(accession=candidate), timeout=REQUEST_TIMEOUT)
                    if response.status_code == 404:
                        break
                    response.raise_for_status()
                    return response.json()
                except Exception as exc:
                    last_error = exc
                    time.sleep(min(2 ** attempt, 10))
            if last_error and candidate == candidate_identifiers(accession)[-1]:
                raise RuntimeError(f'UniProt request failed for {accession}: {last_error}')
        return None
    finally:
        if owned_session:
            session.close()


def detect_species_from_accessions(accessions: list[str]) -> SpeciesDetection:
    unique_accessions = list(dict.fromkeys(str(x).strip() for x in accessions if str(x).strip()))
    session = _new_json_session('Trae-XLMS-PPI-Evaluation/1.0')
    taxon_counts: dict[int, int] = {}
    taxon_names: dict[int, str] = {}
    unresolved: list[str] = []
    try:
        for accession in unique_accessions:
            record = fetch_uniprot_record(accession, session=session)
            if not record:
                unresolved.append(accession)
                continue
            organism = record.get('organism', {})
            taxon_id = organism.get('taxonId')
            if taxon_id is None:
                unresolved.append(accession)
                continue
            taxon_id = int(taxon_id)
            taxon_counts[taxon_id] = taxon_counts.get(taxon_id, 0) + 1
            taxon_names[taxon_id] = organism.get('scientificName', str(taxon_id))
    finally:
        session.close()

    if not taxon_counts:
        raise RuntimeError('Failed to detect a species from the UniProt accessions in PPI.dat.')

    selected_taxon = max(taxon_counts, key=taxon_counts.get)
    mixed_taxa = {taxon: count for taxon, count in taxon_counts.items() if taxon != selected_taxon}
    return SpeciesDetection(
        taxon_id=selected_taxon,
        scientific_name=taxon_names.get(selected_taxon, str(selected_taxon)),
        mixed_taxa=mixed_taxa,
        unresolved_accessions=unresolved,
    )


class StringApiClient:
    def __init__(self, api_base: str = STRING_API_BASE) -> None:
        self.api_base = api_base.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update(
            {
                'User-Agent': 'Trae-XLMS-PPI-Evaluation/1.0',
                'Accept': 'text/tab-separated-values, text/plain, */*',
                'Connection': 'close',
            }
        )
        self._last_call_time = 0.0

    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self._last_call_time
        if elapsed < MIN_CALL_INTERVAL_SECONDS:
            time.sleep(MIN_CALL_INTERVAL_SECONDS - elapsed)

    def _post_text(self, method: str, data: dict[str, Any], output_format: str = 'tsv-no-header') -> str:
        url = f'{self.api_base}/{output_format}/{method}'
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                self._respect_rate_limit()
                response = self.session.post(url, data=data, timeout=REQUEST_TIMEOUT)
                self._last_call_time = time.time()
                response.raise_for_status()
                return response.text
            except Exception as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f'STRING API request failed for {method}: {last_error}')

    def _post_tsv(self, method: str, data: dict[str, Any], columns: list[str]) -> pd.DataFrame:
        text = self._post_text(method, data, output_format='tsv-no-header')
        if not text.strip():
            return pd.DataFrame(columns=columns)
        return pd.read_csv(io.StringIO(text), sep='\t', header=None, names=columns)

    def map_identifiers(self, accessions: list[str], species: int) -> dict[str, StringMapping]:
        pending = list(dict.fromkeys(str(x).strip() for x in accessions if str(x).strip()))
        resolved: dict[str, StringMapping] = {}

        exact_queries = pending.copy()
        for batch_start in range(0, len(exact_queries), MAP_BATCH_SIZE):
            batch = exact_queries[batch_start : batch_start + MAP_BATCH_SIZE]
            df = self._post_tsv(
                'get_string_ids',
                {
                    'identifiers': '\r'.join(batch),
                    'species': species,
                    'limit': 1,
                    'echo_query': 1,
                },
                MAP_COLUMNS,
            )
            for _, row in df.iterrows():
                query_item = str(row['query_item']).strip()
                if query_item in resolved:
                    continue
                resolved[query_item] = StringMapping(
                    query_accession=query_item,
                    mapped_query=query_item,
                    string_id=str(row['string_id']).strip(),
                    preferred_name=str(row['preferred_name']).strip(),
                    annotation=str(row['annotation']).strip(),
                    mapping_source='exact',
                )

        unresolved = [x for x in pending if x not in resolved]
        fallback_map: dict[str, str] = {}
        fallback_queries: list[str] = []
        for accession in unresolved:
            candidates = candidate_identifiers(accession)
            if len(candidates) > 1:
                fallback = candidates[1]
                fallback_map[fallback] = accession
                fallback_queries.append(fallback)

        fallback_queries = list(dict.fromkeys(fallback_queries))
        for batch_start in range(0, len(fallback_queries), MAP_BATCH_SIZE):
            batch = fallback_queries[batch_start : batch_start + MAP_BATCH_SIZE]
            df = self._post_tsv(
                'get_string_ids',
                {
                    'identifiers': '\r'.join(batch),
                    'species': species,
                    'limit': 1,
                    'echo_query': 1,
                },
                MAP_COLUMNS,
            )
            for _, row in df.iterrows():
                query_item = str(row['query_item']).strip()
                original = fallback_map.get(query_item)
                if not original or original in resolved:
                    continue
                resolved[original] = StringMapping(
                    query_accession=original,
                    mapped_query=query_item,
                    string_id=str(row['string_id']).strip(),
                    preferred_name=str(row['preferred_name']).strip(),
                    annotation=str(row['annotation']).strip(),
                    mapping_source='canonical_fallback',
                )

        for accession in pending:
            if accession not in resolved:
                resolved[accession] = StringMapping(
                    query_accession=accession,
                    mapped_query=None,
                    string_id=None,
                    preferred_name=None,
                    annotation=None,
                    mapping_source='unmapped',
                )
        return resolved

    def fetch_network(
        self,
        string_ids: list[str],
        species: int,
        required_score: int = 0,
        network_type: str = 'functional',
    ) -> pd.DataFrame:
        unique_ids = list(dict.fromkeys(x for x in string_ids if x))
        if len(unique_ids) < 2:
            return pd.DataFrame(columns=NETWORK_COLUMNS)

        text = self._post_text(
            'network',
            {
                'identifiers': '\r'.join(unique_ids),
                'species': species,
                'required_score': required_score,
                'network_type': network_type,
                'add_nodes': 0,
            },
            output_format='tsv',
        )
        if not text.strip():
            return pd.DataFrame(columns=NETWORK_COLUMNS)
        df = pd.read_csv(io.StringIO(text), sep='\t')
        for column in NETWORK_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA
        return df[NETWORK_COLUMNS].copy()
