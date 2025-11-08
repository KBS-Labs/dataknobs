"""Utility functions for Elasticsearch operations and query building.

Provides helper functions for constructing Elasticsearch queries, managing indices,
and processing search results with Pandas integration.
"""

import json
from collections.abc import Generator
from typing import Any, Dict, List, TextIO, Union

# import os
import pandas as pd

import dataknobs_utils.pandas_utils as pd_utils
from dataknobs_utils import requests_utils


def build_field_query_dict(
    fields: Union[str, List[str]], text: str, operator: str | None = None
) -> Dict[str, Any]:
    """Build an Elasticsearch field query to search for text.

    Creates either a match query (single field) or multi_match query (multiple fields).

    Args:
        fields: Field name (str) or list of field names to query.
        text: Text to search for.
        operator: Search operator (e.g., "AND", "OR"). Uses Elasticsearch
            default if None. Defaults to None.

    Returns:
        Dict[str, Any]: Elasticsearch query dictionary.
    """
    rv: Dict[str, Any]
    if isinstance(fields, str) or len(fields) == 1:  # single match
        if not isinstance(fields, str):
            fields = fields[0]
        field_dict = {"query": text}
        if operator:
            field_dict["operator"] = operator
        rv = {
            "query": {
                "match": {
                    fields: field_dict,
                }
            }
        }
    else:  # multi-match
        rv = {
            "query": {
                "multi_match": {
                    "query": text,
                    "fields": fields,
                }
            }
        }
    return rv


def build_phrase_query_dict(field: str, phrase: str, slop: int = 0) -> Dict[str, Any]:
    """Build an Elasticsearch phrase query with slop tolerance.

    Args:
        field: Field name to query.
        phrase: Exact phrase to search for.
        slop: Maximum number of positions between terms. Defaults to 0 (exact match).

    Returns:
        Dict[str, Any]: Elasticsearch match_phrase query dictionary.
    """
    return {
        "query": {
            "match_phrase": {
                field: {
                    "query": phrase,
                    "slop": slop,
                }
            }
        }
    }


def build_hits_dataframe(query_result: Dict[str, Any]) -> pd.DataFrame | None:
    """Extract search hits from Elasticsearch query results as DataFrame.

    Args:
        query_result: Elasticsearch query response dictionary.

    Returns:
        pd.DataFrame | None: DataFrame with _source fields from hits, or None
            if no hits found.
    """
    df = None
    if "hits" in query_result:
        qr_hits = query_result["hits"]
        if "hits" in qr_hits:
            hits = qr_hits["hits"]
            dicts = [[hit["_source"]] for hit in hits]
            df = pd_utils.dicts2df(dicts, item_id=None)
    return df


def build_aggs_dataframe(query_result: Dict[str, Any]) -> pd.DataFrame | None:
    """Extract aggregations from Elasticsearch query results as DataFrame.

    Args:
        query_result: Elasticsearch query response dictionary.

    Returns:
        pd.DataFrame | None: DataFrame with aggregation results (not yet implemented).

    Note:
        This function is a placeholder and currently returns None.
    """
    # TODO: implement this
    return None


def decode_results(query_result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Decode Elasticsearch query results into DataFrames.

    Args:
        query_result: Elasticsearch query response dictionary.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with "hits_df" and/or "aggs_df" keys
            containing result DataFrames (only present if data exists).
    """
    result = {}
    hits_df = build_hits_dataframe(query_result)
    if hits_df is not None:
        result["hits_df"] = hits_df
    aggs_df = build_aggs_dataframe(query_result)
    if aggs_df is not None:
        result["aggs_df"] = aggs_df
    return result


def add_batch_data(
    batchfile: TextIO,
    record_generator: Any,
    idx_name: str,
    source_id_fieldname: str = "id",
    cur_id: int = 1,
) -> int:
    """Write records to Elasticsearch bulk load batch file.

    Generates newline-delimited JSON (NDJSON) format for Elasticsearch bulk API,
    with alternating action/source lines for each record.

    Args:
        batchfile: File handle for writing batch data.
        record_generator: Generator yielding source record dictionaries to index.
        idx_name: Elasticsearch index name for these records.
        source_id_fieldname: If non-empty, adds/updates this field in source
            records with the document ID. Defaults to "id".
        cur_id: Starting document ID. Defaults to 1.

    Returns:
        int: Next available document ID after processing all records.
    """
    for record in record_generator:
        if source_id_fieldname:
            record[source_id_fieldname] = cur_id
        action = {
            "index": {
                "_index": idx_name,
                "_id": cur_id,
            }
        }
        print(json.dumps(action), file=batchfile)
        print(json.dumps(record), file=batchfile, flush=True)
        cur_id += 1
    return cur_id


def batchfile_record_generator(batchfile_path: str) -> Generator[Any, None, None]:
    """Generate records from an Elasticsearch bulk load batch file.

    Parses NDJSON batch files, yielding only source documents (skipping action lines).

    Args:
        batchfile_path: Path to the Elasticsearch batch file.

    Yields:
        Dict[str, Any]: Each source record dictionary from the batch file.
    """
    with open(batchfile_path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith('{"index":'):
                yield json.loads(line)


def collect_batchfile_values(
    batchfile_path: str, fieldname: str, default_value: Any = ""
) -> List[Any]:
    """Collect all values for a specific field from batch file records.

    Args:
        batchfile_path: Path to the Elasticsearch batch file.
        fieldname: Name of the field whose values to collect.
        default_value: Value to use when field doesn't exist in a record.
            Defaults to "".

    Returns:
        List[Any]: List of field values from all records.
    """
    values = []
    for ldata in batchfile_record_generator(batchfile_path):
        values.append(ldata.get(fieldname, default_value))
    return values


def collect_batchfile_records(batchfile_path: str) -> pd.DataFrame:
    """Load all batch file records into a pandas DataFrame.

    Args:
        batchfile_path: Path to the Elasticsearch batch file.

    Returns:
        pd.DataFrame: DataFrame containing all records from the batch file.
    """
    records = []
    for record in batchfile_record_generator(batchfile_path):
        records.append(record)
    return pd.DataFrame(records)


class TableSettings:
    """Configuration container for an Elasticsearch index.

    Attributes:
        name: Index name.
        settings: Index settings (shards, replicas, analyzers, etc.).
        mapping: Field mappings and types.
    """

    def __init__(
        self,
        table_name: str,
        data_settings: Dict[str, Any],
        data_mapping: Dict[str, Any],
    ) -> None:
        """Initialize table settings.

        Args:
            table_name: Name of the Elasticsearch index.
            data_settings: Index settings dictionary.
            data_mapping: Field mappings dictionary.
        """
        self.name = table_name
        self.settings = data_settings
        self.mapping = data_mapping


class ElasticsearchIndex:
    """Wrapper for managing Elasticsearch indices with settings and mappings.

    Handles index creation, initialization, and querying across multiple indices
    with predefined settings and mappings.

    Attributes:
        request_helper: RequestHelper for making HTTP requests.
        tables: List of TableSettings for managed indices.
    """

    def __init__(
        self,
        request_helper: Any | None,
        table_settings: List[TableSettings],
        elasticsearch_ip: str | None = None,
        elasticsearch_port: int = 9200,
        mock_requests: Any | None = None,
    ) -> None:
        """Initialize Elasticsearch index manager.

        Args:
            request_helper: Pre-configured RequestHelper. If None, creates one
                using elasticsearch_ip and elasticsearch_port. Defaults to None.
            table_settings: List of TableSettings for indices to manage.
            elasticsearch_ip: Elasticsearch host IP. Used if request_helper is None.
                Defaults to None (uses "localhost").
            elasticsearch_port: Elasticsearch port. Defaults to 9200.
            mock_requests: Mock requests object for testing. Defaults to None.
        """
        self.request_helper: Any  # Always set, never None
        if request_helper is None:
            # Use localhost as default if no IP is provided
            self.request_helper = requests_utils.RequestHelper(
                elasticsearch_ip or "localhost",
                elasticsearch_port,
                mock_requests=mock_requests,
            )
        else:
            self.request_helper = request_helper
        self.tables = table_settings or []
        self._init_tables()

    def _init_tables(self) -> None:
        """Create and initialize all managed indices.

        For each index that doesn't exist, creates it and applies settings
        and mappings. Settings require the index to be closed temporarily.
        """
        for table in self.tables:
            resp = self._request("get", f"{table.name}/_mapping")
            if not resp.succeeded:
                self._request("put", table.name)  # create table
                # initialize settings (for which index must be closed)
                self._request("post", f"{table.name}/_close", None)
                self._request("put", f"{table.name}/_settings", json.dumps(table.settings))
                self._request("post", f"{table.name}/_open", None)
                # initialize mappings
                self._request("put", f"{table.name}/_mapping", json.dumps(table.mapping))

    def _request(
        self,
        rtype: str,
        path: str,
        payload: str | None = None,
        params: Dict[str, Any] | None = None,
        files: Dict[str, Any] | None = None,
        response_handler: Any | None = None,
        headers: Dict[str, str] | None = None,
        timeout: int = 0,
        verbose: bool = False,
    ) -> Any:
        return self.request_helper.request(
            rtype,
            path,
            payload=payload,
            params=params,
            files=files,
            response_handler=response_handler,
            headers=headers,
            timeout=timeout,
            verbose=verbose,
        )

    def is_up(self) -> bool:
        """Check if the Elasticsearch server is reachable.

        Returns:
            bool: True if server responds to cluster health check.
        """
        resp = self._request("get", "_cluster/health")
        return bool(resp.succeeded if resp else False)

    def get_cluster_health(self, verbose: bool = False) -> Any:
        """Get Elasticsearch cluster health information.

        Args:
            verbose: If True, prints response. Defaults to False.

        Returns:
            ServerResponse: Response with cluster health data.
        """
        return self._request("get", "_cluster/health", verbose=verbose)

    def inspect_indices(self, verbose: bool = False) -> Any:
        """List all indices with their statistics.

        Args:
            verbose: If True, prints response. Defaults to False.

        Returns:
            ServerResponse: Response with indices information.
        """
        return self._request(
            "get",
            "_cat/indices?v&pretty",
            verbose=verbose,
            response_handler=requests_utils.plain_api_response_handler,
        )

    def purge(self, verbose: bool = False) -> Any:
        """Delete and recreate all managed indices.

        Removes all data by deleting indices, then recreates them with
        original settings and mappings.

        Args:
            verbose: If True, prints responses. Defaults to False.

        Returns:
            ServerResponse: Response from the last delete operation.
        """
        resp = None
        for table in self.tables:
            resp = self.delete_table(table.name, verbose=verbose)
        self._init_tables()
        return resp

    def delete_table(self, table_name: str, verbose: bool = False) -> Any:
        """Delete an index.

        Args:
            table_name: Name of the index to delete.
            verbose: If True, prints response. Defaults to False.

        Returns:
            ServerResponse: Delete operation response.
        """
        return self._request("delete", table_name, verbose=verbose)

    ## NOT WORKING
    # def bulk_load(self, batchfilepath, verbose=False):
    #    '''
    #    Bulk load the contents of the batch file.
    #
    #    :param batchfilepath: The path to the batch file
    #    :param verbose: True to print the request response.
    #    :return: a requests_utils.ServerResponse instance
    #    '''
    #    return self._request('post-files', '_bulk', files={
    #        'batchfile': (
    #            os.path.basename(batchfilepath),
    #            open(batchfilepath, 'rb'),
    #            'Content-Type: application/x-ndjson',
    #            {'Expires': '0'}
    #        )
    #    }, verbose=verbose)

    def analyze(
        self,
        text: str,
        analyzer: str,
        verbose: bool = False,
    ) -> Any:
        """Analyze text using a specified analyzer.

        Args:
            text: Text to analyze.
            analyzer: Name of the analyzer to use.
            verbose: If True, prints response. Defaults to False.

        Returns:
            ServerResponse: Response with analysis results (tokens, etc.).
        """
        return self._request(
            "post",
            "_analyze",
            payload=json.dumps(
                {
                    "analyzer": analyzer,
                    "text": text,
                }
            ),
            verbose=verbose,
        )

    def search(
        self,
        query: Dict[str, Any],
        table: str | None = None,
        verbose: bool = False,
    ) -> Any | None:
        """Execute an Elasticsearch search DSL query.

        Args:
            query: Elasticsearch query dictionary, e.g.:
                {"query": {"match": {field: {"query": text, "operator": "AND"}}}}
            table: Index name to search. If None, uses first managed index.
                Defaults to None.
            verbose: If True, prints response. Defaults to False.

        Returns:
            ServerResponse | None: Response with results. If successful,
                resp.extra contains 'hits_df' and/or 'aggs_df' DataFrames.
                Returns None if no table is available.
        """
        if table is None:
            if len(self.tables) > 0:
                table = self.tables[0].name
            else:
                return None
        resp = self._request("post", f"{table}/_search", json.dumps(query), verbose=verbose)
        if resp.succeeded:
            d = decode_results(resp.result)
            for k, df in d.items():
                resp.add_extra(k, df)
        return resp

    def sql(
        self,
        query: str,
        fetch_size: int = 10000,
        columnar: bool = True,
        verbose: bool = False,
    ) -> Any:
        """Execute an Elasticsearch SQL query.

        Automatically handles pagination using cursors to retrieve all results.

        Args:
            query: SQL query string.
            fetch_size: Maximum records per fetch. Defaults to 10000.
            columnar: If True, uses compact columnar format (better for few columns).
                Defaults to True.
            verbose: If True, prints responses. Defaults to False.

        Returns:
            ServerResponse: Response with results. If successful, resp.extra['df']
                contains a DataFrame with all query results.
        """
        df = None
        payload = json.dumps(
            {
                "query": query,
                "fetch_size": fetch_size,
                "columnar": columnar,
            }
        )
        resp = self._request(
            "post",
            "_sql?format=json",
            payload=payload,
            verbose=verbose,
        )
        rcols = resp.result.get("columns", None)
        while resp.succeeded:
            cols = [x["name"] for x in rcols]
            rdf = None
            if "values" in resp.result:  # columnar==True
                rdf = pd.DataFrame(resp.result["values"]).T
                rdf.columns = cols
            elif "rows" in resp.result:  # columnar==False
                rdf = pd.DataFrame(resp.result["rows"], columns=cols)
            if rdf is not None:
                if df is not None:
                    df = pd.concat([df, rdf])
                else:
                    df = rdf
            rjson = resp.result
            if "cursor" in rjson:
                resp = self._request(
                    "post",
                    "_sql?format=json",
                    json.dumps(
                        {
                            "cursor": rjson["cursor"],
                            "columnar": columnar,
                        }
                    ),
                    verbose=verbose,
                )
            else:
                break
        if df is not None:
            resp.add_extra("df", df)
        return resp


class SimplifiedElasticsearchIndex:
    """Simplified Elasticsearch index wrapper for single index operations.
    
    This class provides a simpler API for working with a single Elasticsearch index,
    suitable for use as a database backend.
    """

    def __init__(
        self,
        index_name: str,
        host: str = "localhost",
        port: int = 9200,
        settings: Dict[str, Any] | None = None,
        mappings: Dict[str, Any] | None = None,
    ):
        """Initialize the index wrapper.
        
        Args:
            index_name: Name of the Elasticsearch index
            host: Elasticsearch host
            port: Elasticsearch port
            settings: Optional index settings
            mappings: Optional index mappings
        """
        self.index_name = index_name
        self.host = host
        self.port = port
        self.settings = settings or {"number_of_shards": 1, "number_of_replicas": 0}
        self.mappings = mappings or {}

        # Create RequestHelper for making requests
        self.request_helper = requests_utils.RequestHelper(host, port)

    def _request(
        self,
        method: str,
        path: str,
        body: Any | None = None,
        params: Dict[str, Any] | None = None,
    ) -> Any:
        """Make a request to Elasticsearch."""
        # Build path without leading slash (RequestHelper will add it)
        full_path = f"{self.index_name}/{path}" if path else self.index_name

        # Convert body to JSON if it's a dict
        if isinstance(body, dict):
            body = json.dumps(body)

        return self.request_helper.request(
            method,
            full_path,
            payload=body,
            params=params,
        )

    def exists(self, doc_id: str | None = None) -> bool:
        """Check if index or document exists.
        
        Args:
            doc_id: If provided, check if document exists. Otherwise check if index exists.
            
        Returns:
            True if exists, False otherwise
        """
        if doc_id:
            # Check if document exists
            response = self._request("head", f"_doc/{doc_id}")
            return response.succeeded
        else:
            # Check if index exists
            response = self.request_helper.request("head", self.index_name)
            return response.succeeded

    def create(self) -> bool:
        """Create the index with settings and mappings.
        
        Returns:
            True if created successfully
        """
        body = {}
        if self.settings:
            body["settings"] = self.settings
        if self.mappings:
            body["mappings"] = self.mappings

        response = self.request_helper.request(
            "put",
            self.index_name,
            payload=json.dumps(body) if body else None,
        )
        return response.succeeded

    def delete(self, doc_id: str | None = None) -> bool:
        """Delete the index or a document.
        
        Args:
            doc_id: If provided, delete document. Otherwise delete entire index.
            
        Returns:
            True if deleted successfully
        """
        if doc_id:
            # Delete document
            response = self._request("delete", f"_doc/{doc_id}")
            return response.succeeded
        else:
            # Delete index
            response = self.request_helper.request("delete", self.index_name)
            return response.succeeded

    def index(self, body: Dict[str, Any], doc_id: str | None = None, refresh: bool = False) -> Dict[str, Any]:
        """Index a document.
        
        Args:
            body: Document to index
            doc_id: Optional document ID (will be auto-generated if not provided)
            refresh: Whether to refresh immediately for search visibility
            
        Returns:
            Response with created document ID
        """
        path = f"_doc/{doc_id}" if doc_id else "_doc"
        params = {"refresh": "true"} if refresh else None

        response = self._request("put" if doc_id else "post", path, body, params)

        if response.succeeded and response.json:
            return response.json
        return {"_id": None, "result": "error"}

    def get(self, doc_id: str) -> Dict[str, Any] | None:
        """Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        response = self._request("get", f"_doc/{doc_id}")

        if response.succeeded and response.json:
            return response.json
        return None

    def update(self, doc_id: str, body: Dict[str, Any], refresh: bool = False) -> bool:
        """Update a document.
        
        Args:
            doc_id: Document ID
            body: Update body (should contain "doc" field with partial update)
            refresh: Whether to refresh immediately
            
        Returns:
            True if updated successfully
        """
        params = {"refresh": "true"} if refresh else None
        response = self._request("post", f"_update/{doc_id}", body, params)
        return response.succeeded

    def search(self, body: Dict[str, Any] | None = None) -> Any:
        """Search documents.
        
        Args:
            body: Search query body
            
        Returns:
            ServerResponse object with search results
        """
        response = self._request("post", "_search", body or {})

        # Return the ServerResponse object directly
        # If it failed, set a default response
        if not response.succeeded:
            response.result = {"hits": {"total": {"value": 0}, "hits": []}}

        return response

    def count(self, body: Dict[str, Any] | None = None) -> int:
        """Count documents.
        
        Args:
            body: Optional query to count matching documents
            
        Returns:
            Number of documents
        """
        response = self._request("post", "_count", body or {})

        if response.succeeded and response.json:
            return response.json.get("count", 0)
        return 0

    def delete_by_query(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Delete documents matching a query.
        
        Args:
            body: Query to match documents for deletion
            
        Returns:
            Response with deletion info
        """
        response = self._request("post", "_delete_by_query", body)

        if response.succeeded and response.json:
            return response.json
        return {"deleted": 0}

    def refresh(self) -> bool:
        """Refresh the index to make recent changes searchable.
        
        Returns:
            True if refreshed successfully
        """
        response = self._request("post", "_refresh")
        return response.succeeded
