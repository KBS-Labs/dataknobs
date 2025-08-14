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
    """Build an elasticsearch field query to find the text in the field(s).
    :param fields: The field (str) or fields (list[str]) to query.
    :param text: The text to find.
    :param operator: The operator to use (default if None), e.g., "AND", "OR"
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
    """Build an elasticsearch phrase query to find the phrase in the field.
    :param field: The field to query
    :param phrase: The phrase to find
    :param slop: The slop factor to use
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
    """Build a dataframe from an elasticsearch query result's hits."""
    df = None
    if "hits" in query_result:
        qr_hits = query_result["hits"]
        if "hits" in qr_hits:
            hits = qr_hits["hits"]
            dicts = [[hit["_source"]] for hit in hits]
            df = pd_utils.dicts2df(dicts, item_id=None)
    return df


def build_aggs_dataframe(query_result: Dict[str, Any]) -> pd.DataFrame | None:
    """Build a dataframe from an elasticsearch query result's aggregations."""
    # TODO: implement this
    return None


def decode_results(query_result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Decode elasticsearch query results as "hits_df" and/or "aggs_df"
    dataframes.
    """
    result = dict()
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
    """Add source records from the generator to the batchfile for elasticsearch
    bulk load into the named index with record IDs starting at the given value,
    optionally adding the record ID to the source record if indicated.

    :param batchfile: The file handle to which to write the batch data
    :param record_generator: Generator for the "source" record
        dictionaries to index
    :param idx_name: Name of the elasticsearch index to hold the records
    :param source_id_fieldname: If non-empty, ensure that the source record
        also has the correct index ID in the field with this name
    :param cur_id: The id of the next source record
    :return: The id of the next source record to be added after exhausting
        the generator
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
    """Given the path to an elasticsearch batchfile yield each elasticsearch
    record (dict).
    :param batchfile_path: The path to the elasticdsearch batch file
    :yield: Each elasticsearch record dictionary
    """
    with open(batchfile_path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith('{"index":'):
                yield json.loads(line)


def collect_batchfile_values(
    batchfile_path: str, fieldname: str, default_value: Any = ""
) -> List[Any]:
    """Given the path to an elasticsearch batchfile and a source record fieldname,
    collect all of the values for the named field.

    :param batchfile_path: The path to the elasticsearch batchfile
    :param fieldname: The name of the source record field whose values to collect
    :param default_value: The value to use if the field doesn't exist for a record.
    :return: The list of collected values
    """
    values = []
    for ldata in batchfile_record_generator(batchfile_path):
        values.append(ldata.get(fieldname, default_value))
    return values


def collect_batchfile_records(batchfile_path: str) -> pd.DataFrame:
    """Collect the batchfile records as a pandas DataFrame."""
    records = []
    for record in batchfile_record_generator(batchfile_path):
        records.append(record)
    return pd.DataFrame(records)


class TableSettings:
    """Container for elasticsearch table settings."""

    def __init__(
        self,
        table_name: str,
        data_settings: Dict[str, Any],
        data_mapping: Dict[str, Any],
    ) -> None:
        self.name = table_name
        self.settings = data_settings
        self.mapping = data_mapping


class ElasticsearchIndex:
    """Wrapper for interacting with an elasticsearch index."""

    def __init__(
        self,
        request_helper: Any | None,
        table_settings: List[TableSettings],
        elasticsearch_ip: str | None = None,
        elasticsearch_port: int = 9200,
        mock_requests: Any | None = None,
    ) -> None:
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
        self.tables = table_settings or list()
        self._init_tables()

    def _init_tables(self) -> None:
        """Ensure the tables have been created and initialized."""
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
        """:return: True if the elasticsearch server is up."""
        resp = self._request("get", "_cluster/health")
        return bool(resp.succeeded if resp else False)

    def get_cluster_health(self, verbose: bool = False) -> Any:
        """:return: a requests_utils.ServerResponse instance"""
        return self._request("get", "_cluster/health", verbose=verbose)

    def inspect_indices(self, verbose: bool = False) -> Any:
        """:return: a requests_utils.ServerResponse instance"""
        return self._request(
            "get",
            "_cat/indices?v&pretty",
            verbose=verbose,
            response_handler=requests_utils.plain_api_response_handler,
        )

    def purge(self, verbose: bool = False) -> Any:
        """Purge all data in tables managed by this wrapper."""
        resp = None
        for table in self.tables:
            resp = self.delete_table(table.name, verbose=verbose)
        self._init_tables()
        return resp

    def delete_table(self, table_name: str, verbose: bool = False) -> Any:
        """:return: a requests_utils.ServerResponse instance"""
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
        """:return: a requests_utils.ServerResponse instance"""
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
        """Submit the elasticsearch search DSL query.

        :param query: The elasticsearch search query of the form, e.g.,:
            {"query": {"match": {<field>: {"query": <text>, "operator": "AND"}}}}
        :param table: The name of the table (defaults to first table's name)
        :param verbose: True to print the request response.
        :return: a requests_utils.ServerResponse instance, resp, with
            resp.extra['hits_df'] and/or resp.extra['aggs_df']
            holding dataframe(s) representing the results if successful.
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
        """Submit the elasticsearch sql query.

        :param query: The elasticsearch search sql query
        :param fetch_size: The max number of records to fetch at a time
        :param columnar: True for a more compact response (best when the number
            of columns returned by the query is small)
        :param verbose: True to print the request response.
        :return: a requests_utils.ServerResponse instance, resp, with
            resp.extra['df'] holding dataframe(s) representing the results
            if successful.
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
