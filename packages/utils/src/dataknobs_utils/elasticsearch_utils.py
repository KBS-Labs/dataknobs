import json
#import os
import pandas as pd
import dataknobs.utils.requests_utils as requests_utils
import dataknobs.utils.pandas_utils as pd_utils
from typing import Dict, List


def build_field_query_dict(fields, text, operator=None):
    '''
    Build an elasticsearch field query to find the text in the field(s).
    :param fields: The field (str) or fields (list[str]) to query.
    :param text: The text to find.
    :param operator: The operator to use (default if None), e.g., "AND", "OR"
    '''
    rv = None
    if isinstance(fields, str) or len(fields) == 1:  # single match
        if not isinstance(fields, str):
            fields = fields[0]
        field_dict = {'query': text}
        if operator:
            field_dict['operator'] = operator
        rv = {
            'query': {
                'match': {
                    fields: field_dict,
                }
            }
        }
    else:  # multi-match
        rv = {
            'query': {
                'multi_match': {
                    'query': text,
                    'fields': fields,
                }
            }
        }
    return rv


def build_phrase_query_dict(field, phrase, slop=0):
    '''
    Build an elasticsearch phrase query to find the phrase in the field.
    :param field: The field to query
    :param phrase: The phrase to find
    :param slop: The slop factor to use
    '''
    return {
        'query': {
            'match_phrase': {
                field: {
                    'query': phrase,
                    'slop': slop,
                }
            }
        }
    }


def build_hits_dataframe(query_result) -> pd.DataFrame:
    '''
    Build a dataframe from an elasticsearch query result's hits.
    '''
    df = None
    if 'hits' in query_result:
        qr_hits = query_result['hits']
        if 'hits' in qr_hits:
            hits = qr_hits['hits']
            dicts = [
                [hit['_source']]
                for hit in hits
            ]
            df = pd_utils.dicts2df(dicts, item_id=None)
    return df


def build_aggs_dataframe(query_result) -> pd.DataFrame:
    '''
    Build a dataframe from an elasticsearch query result's aggregations.
    '''
    #TODO: implement this
    return None


def decode_results(query_result) -> Dict[str, pd.DataFrame]:
    '''
    Decode elasticsearch query results as "hits_df" and/or "aggs_df"
    dataframes.
    '''
    result = dict()
    hits_df = build_hits_dataframe(query_result)
    if hits_df is not None:
        result["hits_df"] = hits_df
    aggs_df = build_aggs_dataframe(query_result)
    if aggs_df is not None:
        result["aggs_df"] = aggs_df
    return result


def add_batch_data(
        batchfile,
        record_generator,
        idx_name,
        source_id_fieldname="id",
        cur_id=1,
):
    '''
    Add source records from the generator to the batchfile for elasticsearch
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
    '''
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


def batchfile_record_generator(batchfile_path):
    '''
    Given the path to an elasticsearch batchfile yield each elasticsearch
    record (dict).
    :param batchfile_path: The path to the elasticdsearch batch file
    :yield: Each elasticsearch record dictionary
    '''
    with open(batchfile_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith('{"index":'):
                yield json.loads(line)


def collect_batchfile_values(batchfile_path, fieldname, default_value=''):
    '''
    Given the path to an elasticsearch batchfile and a source record fieldname,
    collect all of the values for the named field.

    :param batchfile_path: The path to the elasticsearch batchfile
    :param fieldname: The name of the source record field whose values to collect
    :param default_value: The value to use if the field doesn't exist for a record.
    :return: The list of collected values
    '''
    values = []
    for ldata in batchfile_record_generator(batchfile_path):
        values.append(ldata.get(fieldname, default_value))
    return values


def collect_batchfile_records(batchfile_path):
    '''
    Collect the batchfile records as a pandas DataFrame.
    '''
    records = []
    for record in batchfile_record_generator(batchfile_path):
        records.append(record)
    return pd.DataFrame(records)


class TableSettings:
    '''
    Container for elasticsearch table settings.
    '''

    def __init__(
            self,
            table_name,
            data_settings,
            data_mapping,
    ):
        self.name = table_name
        self.settings = data_settings
        self.mapping = data_mapping


class ElasticsearchIndex:
    '''
    Wrapper for interacting with an elasticsearch index.
    '''

    def __init__(
            self,
            request_helper,
            table_settings: List[TableSettings],
            elasticsearch_ip=None,
            elasticsearch_port=9200,
            mock_requests=None,
    ):
        self.request_helper = request_helper
        if request_helper is None:
            self.request_helper = requests_utils.RequestHelper(
                elasticsearch_ip, elasticsearch_port,
                mock_requests=mock_requests,
            )
        self.tables = table_settings or list()
        self._init_tables()

    def _init_tables(self):
        '''
        Ensure the tables have been created and initialized.
        '''
        for table in self.tables:
            resp = self._request('get', f'{table.name}/_mapping')
            if not resp.succeeded:
                self._request('put', table.name)  # create table
                # initialize settings (for which index must be closed)
                self._request('post', f'{table.name}/_close', None)
                self._request('put', f'{table.name}/_settings', json.dumps(table.settings))
                self._request('post', f'{table.name}/_open', None)
                # initialize mappings
                self._request('put', f'{table.name}/_mapping', json.dumps(table.mapping))

    def _request(
            self,
            rtype,
            path,
            payload=None,
            params=None,
            files=None,
            response_handler=None,
            headers=None,
            timeout=0,
            verbose=False,
    ):
        return self.request_helper.request(
            rtype, path,
            payload=payload,
            params=params,
            files=files,
            response_handler=response_handler,
            headers=headers,
            timeout=timeout,
            verbose=verbose,
        )

    def is_up(self):
        '''
        :return: True if the elasticsearch server is up.
        '''
        return self._request('get', '_cluster/health').succeeded

    def get_cluster_health(self, verbose=False):
        '''
        :return: a requests_utils.ServerResponse instance
        '''
        return self._request('get', '_cluster/health', verbose=verbose)

    def inspect_indices(self, verbose=False):
        '''
        :return: a requests_utils.ServerResponse instance
        '''
        return self._request(
            'get',
            '_cat/indices?v&pretty',
            verbose=verbose,
            response_handler=requests_utils.plain_api_response_handler,
        )

    def purge(self, verbose=False):
        '''
        Purge all data in tables managed by this wrapper.
        '''
        resp = None
        for table in self.tables:
            resp = self.delete_table(table.name, verbose=verbose)
        self._init_tables()
        return resp

    def delete_table(self, table_name, verbose=False):
        '''
        :return: a requests_utils.ServerResponse instance
        '''
        return self._request('delete', table_name, verbose=verbose)

    ## NOT WORKING
    #def bulk_load(self, batchfilepath, verbose=False):
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
            text,
            analyzer: str,
            verbose=False,
    ):
        '''
        :return: a requests_utils.ServerResponse instance
        '''
        return self._request(
            'post',
            '_analyze',
            payload=json.dumps({
                "analyzer": analyzer,
                "text": text,
            }),
            verbose=verbose
        )

    def search(
            self,
            query: Dict[str, Dict],
            table: str = None,
            verbose: bool = False,
    ) -> requests_utils.ServerResponse:
        '''
        Submit the elasticsearch search DSL query.

        :param query: The elasticsearch search query of the form, e.g.,:
            {"query": {"match": {<field>: {"query": <text>, "operator": "AND"}}}}
        :param table: The name of the table (defaults to first table's name)
        :param verbose: True to print the request response.
        :return: a requests_utils.ServerResponse instance, resp, with
            resp.extra['hits_df'] and/or resp.extra['aggs_df']
            holding dataframe(s) representing the results if successful.
        '''
        if table is None:
            if len(self.tables) > 0:
                table = self.tables[0].name
            else:
                return None
        resp = self._request(
            'post', f'{table}/_search', json.dumps(query), verbose=verbose
        )
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
    ) -> requests_utils.ServerResponse:
        
        '''
        Submit the elasticsearch sql query.

        :param query: The elasticsearch search sql query
        :param fetch_size: The max number of records to fetch at a time
        :param columnar: True for a more compact response (best when the number
            of columns returned by the query is small)
        :param verbose: True to print the request response.
        :return: a requests_utils.ServerResponse instance, resp, with
            resp.extra['df'] holding dataframe(s) representing the results
            if successful.
        '''
        df = None
        payload = json.dumps({
            "query": query,
            "fetch_size": fetch_size,
            "columnar": columnar,
        })
        resp = self._request(
            'post',
            '_sql?format=json',
            payload=payload,
            verbose=verbose,
        )
        rcols = resp.result.get("columns", None)
        while resp.succeeded:
            cols = [
                x["name"]
                for x in rcols
            ]
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
                    'post',
                    '_sql?format=json',
                    json.dumps({
                        "cursor": rjson["cursor"],
                        "columnar": columnar,
                    }),
                    verbose=verbose,
                )
            else:
                break
        if df is not None:
            resp.add_extra("df", df)
        return resp
