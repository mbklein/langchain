from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore

IMPORT_OPENSEARCH_PY_ERROR = (
    "Could not import OpenSearch. Please install it with `pip install opensearch-py`."
)
MATCH_ALL_QUERY = {"match_all": {}}  # type: Dict

def _import_opensearch() -> Any:
    """Import OpenSearch if available, otherwise raise error."""
    try:
        from opensearchpy import OpenSearch
    except ImportError:
        raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
    return OpenSearch


def _import_bulk() -> Any:
    """Import bulk if available, otherwise raise error."""
    try:
        from opensearchpy.helpers import bulk
    except ImportError:
        raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
    return bulk


def _import_not_found_error() -> Any:
    """Import not found error if available, otherwise raise error."""
    try:
        from opensearchpy.exceptions import NotFoundError
    except ImportError:
        raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
    return NotFoundError


def _get_opensearch_client(opensearch_url: str, **kwargs: Any) -> Any:
    """Get OpenSearch client from the opensearch_url, otherwise raise error."""
    try:
        opensearch = _import_opensearch()
        client = opensearch(opensearch_url, **kwargs)
    except ValueError as e:
        raise ImportError(
            f"OpenSearch client string provided is not in proper format. "
            f"Got error: {e} "
        )
    return client

def _create_index(
    client: Any,
    index_name: str,
    model_id: str,
    mapping: Optional[Dict] = None,
    text_field: Optional[str] = "text",
    vector_field: Optional[str] = "vector_field",
    ingest_pipeline: Optional[str] = None
) -> str:
    if not mapping:
        embedding_body = { "text_docs": ["please embed this"], "return_number": True, "target_response": ["sentence_embedding"] }
        embedding_url = f'/_plugins/_ml/_predict/text_embedding/{model_id}'
        response = client.transport.perform_request("POST", embedding_url, body=embedding_body, timeout=30)
        vector_size = response.get('inference_results', [{}])[0].get('output', [{}])[0].get('shape', [0])[0]
        mapping = {
            "properties": {
                vector_field: {
                    "type": "knn_vector",
                    "dimension": vector_size
                }
            }
        }

    if not ingest_pipeline:
        ingest_pipeline = f'{index_name}-ingest-pipeline'
        pipeline_body = {
            "description": f'Embedding pipeline for {index_name}',
            "processors": [
                {
                    "text_embedding": {
                        "field_map": {
                            text_field: vector_field
                        },
                        "ignore_failure": True,
                        "model_id": model_id
                    }
                }
            ]
        }
        client.transport.perform_request("PUT", f'/_ingest/pipeline/{ingest_pipeline}', body=pipeline_body)

    client.indices.create(index=index_name, body={"settings": {"knn": True, "default_pipeline": ingest_pipeline}, "mappings": mapping})
    return index_name


def _bulk_ingest(
    client: Any,
    index_name: str,
    texts: Iterable[str],
    model_id: Optional[str] = None,
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    text_field: str = "text",
    vector_field: str = "vector_field",
    mapping: Optional[Dict] = None,
    max_chunk_bytes: Optional[int] = 1 * 1024 * 1024,
    ingest_pipeline: Optional[str] = None
) -> List[str]:
    """Bulk Ingest into given index. The index should have a default ingest pipeline with a
    text_embedding processor, or the name of an embedding processor can be included in the call."""
    bulk = _import_bulk()
    not_found_error = _import_not_found_error()
    requests = []
    return_ids = []
    mapping = mapping

    try:
        client.indices.get(index=index_name)
    except not_found_error:
        if not model_id:
            raise ValueError("model_id must be supplied if index or ingest_pipeline does not exist")
        _create_index(
            client, 
            index_name, 
            model_id,
            mapping=mapping,
            text_field=text_field,
            vector_field=vector_field,
            ingest_pipeline=ingest_pipeline
        )

    for i, text in enumerate(texts):
        metadata = metadatas[i] if metadatas else {}
        _id = ids[i] if ids else str(uuid.uuid4())
        request = {
            "_op_type": "index",
            "_index": index_name,
            text_field: text,
            "metadata": metadata,
        }
        request["_id"] = _id
        requests.append(request)
        return_ids.append(_id)
    bulk(client, requests, max_chunk_bytes=max_chunk_bytes, pipeline=ingest_pipeline, request_timeout=600)
    client.indices.refresh(index=index_name)
    return return_ids


def _approximate_search_query_with_boolean_filter(
    query: str,
    model_id: str,
    boolean_filter: Dict,
    k: int = 4,
    include_vectors: bool = False,
    vector_field: str = "vector_field",
    subquery_clause: str = "must",
) -> Dict:
    """For Approximate k-NN Search, with Boolean Filter."""
    result = {
        "size": k,
        "query": {
            "bool": {
                "filter": boolean_filter,
                subquery_clause: [
                    {"neural": {vector_field: {"query_text": query, "model_id": model_id, "k": k}}}
                ],
            }
        },
    }
    if not include_vectors:
        result["_source"] = [f"-{vector_field}"]
    return result

def _approximate_search_query_with_efficient_filter(
    query: str,
    model_id: str,
    efficient_filter: Dict,
    k: int = 4,
    vector_field: str = "vector_field",
) -> Dict:
    """For Approximate k-NN Search, with Efficient Filter for Lucene and
    Faiss Engines."""
    search_query = _default_approximate_search_query(
        query, model_id, k=k, vector_field=vector_field
    )
    search_query["query"]["knn"][vector_field]["filter"] = efficient_filter
    return search_query

def _default_approximate_search_query(
    query: str,
    model_id: str,
    k: int = 4,
    include_vectors: bool = False,
    vector_field: str = "vector_field",
) -> Dict:
    """For Approximate k-NN Search, this is the default query."""
    result = {
        "size": k,
        "query": {"neural": {vector_field: {"query_text": query, "model_id": model_id, "k": k}}}
    }
    if not include_vectors:
        result["_source"] = { "excludes": [vector_field] }
    return result


class OpenSearchNeuralSearch(VectorStore):
    """`Amazon OpenSearch Neural Search` vector store.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import OpenSearchNeuralSearch
            opensearch_vector_search = OpenSearchNeuralSearch(
                "http://localhost:9200",
                "embeddings",
                model_id
            )
    """

    def __init__(
        self,
        opensearch_url: str,
        index_name: str,
        model_id: str,
        **kwargs: Any,
    ):
        """Initialize with necessary components."""
        self.model_id = model_id
        self.index_name = index_name
        self.client = _get_opensearch_client(opensearch_url, **kwargs)
        self.engine = kwargs.get("engine")

    def __add(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        index_name = kwargs.get("index_name", self.index_name)
        text_field = kwargs.get("text_field", "text")
        vector_field = kwargs.get("vector_field", "vector_field")
        max_chunk_bytes = kwargs.get("max_chunk_bytes", 1 * 1024 * 1024)

        return _bulk_ingest(
            self.client,
            index_name,
            texts,
            metadatas=metadatas,
            ids=ids,
            text_field=text_field,
            vector_field=vector_field,
            max_chunk_bytes=max_chunk_bytes
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        bulk_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            bulk_size: Bulk API request count; Default: 500

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        return self.__add(
            texts,
            metadatas=metadatas,
            ids=ids,
            bulk_size=bulk_size,
            **kwargs,
        )

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        By default, supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

            metadata_field: Document field that metadata is stored in. Defaults to
            "metadata".
            Can be set to a special value "*" to include the entire document.

        Optional Args for Approximate Search:
            search_type: "approximate_search"; default: "approximate_search"

            boolean_filter: A Boolean filter is a post filter consists of a Boolean
            query that contains a k-NN query and a filter.

            subquery_clause: Query clause on the knn vector field; default: "must"

            lucene_filter: the Lucene algorithm decides whether to perform an exact
            k-NN search with pre-filtering or an approximate search with modified
            post-filtering. (deprecated, use `efficient_filter`)

            efficient_filter: the Lucene Engine or Faiss Engine decides whether to
            perform an exact k-NN search with pre-filtering or an approximate search
            with modified post-filtering.

        Optional Args for Script Scoring Search:
            search_type: "script_scoring"; default: "approximate_search"

            space_type: "l2", "l1", "linf", "cosinesimil", "innerproduct",
            "hammingbit"; default: "l2"

            pre_filter: script_score query to pre-filter documents before identifying
            nearest neighbors; default: {"match_all": {}}

        Optional Args for Painless Scripting Search:
            search_type: "painless_scripting"; default: "approximate_search"

            space_type: "l2Squared", "l1Norm", "cosineSimilarity"; default: "l2Squared"

            pre_filter: script_score query to pre-filter documents before identifying
            nearest neighbors; default: {"match_all": {}}
        """
        docs_with_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc[0] for doc in docs_with_scores]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs and it's scores most similar to query.

        By default, supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents along with its scores most similar to the query.

        Optional Args:
            same as `similarity_search`
        """
        search_type = kwargs.get("search_type", "approximate_search")
        vector_field = kwargs.get("vector_field", "vector_field")
        include_vectors = kwargs.get("include_vectors", False)
        index_name = kwargs.get("index_name", self.index_name)
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")
        filter = kwargs.get("filter", {})

        if search_type == "approximate_search":
            boolean_filter = kwargs.get("boolean_filter", {})
            subquery_clause = kwargs.get("subquery_clause", "must")
            efficient_filter = kwargs.get("efficient_filter", {})

            if boolean_filter != {} and efficient_filter != {}:
                raise ValueError(
                    "Both `boolean_filter` and `efficient_filter` are provided which "
                    "is invalid"
                )

            if (
                efficient_filter == {}
                and boolean_filter == {}
                and filter != {}
            ):
                if self.engine in ["faiss", "lucene"]:
                    efficient_filter = filter
                else:
                    boolean_filter = filter

            if boolean_filter != {}:
                search_query = _approximate_search_query_with_boolean_filter(
                    query,
                    self.model_id,
                    boolean_filter,
                    k=k,
                    include_vectors=include_vectors,
                    vector_field=vector_field,
                    subquery_clause=subquery_clause,
                )
            elif efficient_filter != {}:
                search_query = _approximate_search_query_with_efficient_filter(
                    query, efficient_filter, k=k, vector_field=vector_field
                )
            else:
                search_query = _default_approximate_search_query(
                    query, self.model_id, k=k, include_vectors=include_vectors, vector_field=vector_field
                )
        # elif search_type == SCRIPT_SCORING_SEARCH:
        #     space_type = kwargs.get("space_type", "l2")
        #     pre_filter = kwargs.get("pre_filter", MATCH_ALL_QUERY)
        #     search_query = _default_script_query(
        #         query, k, space_type, pre_filter, vector_field
        #     )
        # elif search_type == PAINLESS_SCRIPTING_SEARCH:
        #     space_type = kwargs.get("space_type", "l2Squared")
        #     pre_filter = kwargs.get("pre_filter", MATCH_ALL_QUERY)
        #     search_query = _default_painless_scripting_query(
        #         query, k, space_type, pre_filter, vector_field
        #     )
        else:
            raise ValueError("Invalid `search_type` provided as an argument")

        response = self.client.search(index=index_name, body=search_query)
        hits = response.get("hits", {}).get("hits", [])
        return [
            (
                Document(
                    page_content=hit["_source"][text_field],
                    metadata=(
                        hit["_source"]
                        if metadata_field == "*" or metadata_field not in hit["_source"]
                        else hit["_source"][metadata_field]
                    ),
                ),
                hit["_score"],
            )
            for hit in hits
        ]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        model_id: Optional[str] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        ingest_pipeline: Optional[str] = None,
        **kwargs: Any,
    ) -> OpenSearchNeuralSearch:
        """Construct OpenSearchNeuralSearch wrapper from raw texts.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import OpenSearchNeuralSearch
                opensearch_vector_search = OpenSearchNeuralSearch.from_texts(
                    texts,
                    model_id,
                    opensearch_url="http://localhost:9200"
                )

        OpenSearch by default supports Approximate Search powered by nmslib, faiss
        and lucene engines recommended for large datasets. Also supports brute force
        search through Script Scoring and Painless Scripting.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

        Optional Keyword Args for Approximate Search:
            engine: "nmslib", "faiss", "lucene"; default: "nmslib"

            space_type: "l2", "l1", "cosinesimil", "linf", "innerproduct"; default: "l2"

            ef_search: Size of the dynamic list used during k-NN searches. Higher values
            lead to more accurate but slower searches; default: 512

            ef_construction: Size of the dynamic list used during k-NN graph creation.
            Higher values lead to more accurate graph but slower indexing speed;
            default: 512

            m: Number of bidirectional links created for each new element. Large impact
            on memory consumption. Between 2 and 100; default: 16

        Keyword Args for Script Scoring or Painless Scripting:
            is_appx_search: False

        """
        opensearch_url = get_from_dict_or_env(
            kwargs, "opensearch_url", "OPENSEARCH_URL"
        )
        kwargs.pop("opensearch_url", None)
        index_name = get_from_dict_or_env(
            kwargs, "index_name", "OPENSEARCH_INDEX_NAME", default=uuid.uuid4().hex
        )
        kwargs.pop("index_name", None)
        vector_field = kwargs.get("vector_field", "vector_field")
        text_field = kwargs.get("text_field", "text")
        vector_field = kwargs.get("vector_field", "vector_field")
        max_chunk_bytes = kwargs.get("max_chunk_bytes", 1 * 1024 * 1024)
        engine = None
        client = _get_opensearch_client(opensearch_url, **kwargs)
        _bulk_ingest(
            client,
            index_name,
            texts,
            ids=ids,
            metadatas=metadatas,
            model_id=model_id,
            ingest_pipeline=ingest_pipeline,
            text_field=text_field,
            vector_field=vector_field,
            max_chunk_bytes=max_chunk_bytes,
        )
        kwargs["engine"] = engine
        return cls(opensearch_url, index_name, model_id, **kwargs)
