from __future__ import annotations

import os

from haystack import Pipeline
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore


def build_query_pipeline(document_store: ChromaDocumentStore) -> Pipeline:
    """
    Assemble the Haystack pipeline used for semantic search queries.
    """
    ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    text_embedder = OllamaTextEmbedder(model=ollama_embed_model, url=ollama_url)
    retriever = ChromaEmbeddingRetriever(document_store=document_store)

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", text_embedder)
    query_pipeline.add_component("embedding_retriever", retriever)
    query_pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")

    return query_pipeline
