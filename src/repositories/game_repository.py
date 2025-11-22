from __future__ import annotations

from typing import List, Optional, Tuple

from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from src.services.recommendation.pagination import normalize_page_params


class GameRepository:
    """
    Data access layer around the Chroma document store.
    Encapsulates filtering, pagination and indexing concerns.
    """

    def __init__(self, document_store: ChromaDocumentStore) -> None:
        self._document_store = document_store

    def get_by_mysql_id(self, mysql_id: int) -> Optional[Document]:
        filters = [{"field": "mysql_id", "operator": "==", "value": mysql_id}]
        documents = self._document_store.filter_documents(filters=filters)
        if documents:
            return documents[0]
        return None

    def find_by_embedding_paginated(
        self,
        query_embedding: List[float],
        page: int,
        per_page: int,
    ) -> Tuple[List[Document], int, int]:
        page, per_page = normalize_page_params(page, per_page)
        # Fetch a larger pool of candidates to paginate in memory
        candidate_pool_size = page * per_page

        collection = getattr(self._document_store, "_collection", None)
        if collection is None:
            return [], page, per_page

        # ChromaDB's query returns more than just documents, so we handle the raw response
        query_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_pool_size, # Fetch enough for all pages up to the current one
            include=["metadatas", "documents", "distances"],
        )

        ids = query_result.get("ids", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        documents_content = query_result.get("documents", [[]])[0]
        distances = query_result.get("distances", [[]])[0]

        reconstructed_docs: List[Document] = []
        for doc_id, meta, content, dist in zip(ids, metadatas, documents_content, distances):
            # The score is the similarity, which is 1 - distance
            score = 1 - dist
            doc = Document(id=doc_id, content=content, meta=meta, score=score)
            reconstructed_docs.append(doc)
        
        # Now, perform pagination on the retrieved list
        offset = (page - 1) * per_page
        paginated_docs = reconstructed_docs[offset : offset + per_page]

        return paginated_docs, page, per_page

    def list_paginated(
        self, page: int, per_page: int
    ) -> Tuple[List[Document], int, int, int]:
        page, per_page = normalize_page_params(page, per_page)
        offset = (page - 1) * per_page

        collection = getattr(self._document_store, "_collection", None)
        if collection is None:
            return [], 0, page, per_page

        total = collection.count()
        raw_page = collection.get(
            limit=per_page,
            offset=offset,
            include=["metadatas", "documents", "ids"],
        )

        ids = raw_page.get("ids", [])
        metadatas = raw_page.get("metadatas", [])
        documents = raw_page.get("documents", [])

        reconstructed_docs: List[Document] = []
        for doc_id, meta, content in zip(ids, metadatas, documents):
            reconstructed_docs.append(Document(id=doc_id, content=content, meta=meta))

        return reconstructed_docs, total, page, per_page

    def index_documents(
        self,
        documents: List[Document],
        embedder: OllamaDocumentEmbedder,
        duplicate_policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
        batch_size: int = 64,
    ) -> None:
        """
        Embed and upsert the provided documents into Chroma.
        """
        if not documents:
            return

        writer = DocumentWriter(document_store=self._document_store, policy=duplicate_policy)
        pipeline = Pipeline()
        pipeline.add_component("embedder", embedder)
        pipeline.add_component("writer", writer)
        pipeline.connect("embedder.documents", "writer.documents")

        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            pipeline.run({"embedder": {"documents": batch}})
