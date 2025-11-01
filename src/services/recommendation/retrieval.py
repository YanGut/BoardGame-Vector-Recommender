from __future__ import annotations

from typing import List

from haystack import Document, Pipeline


def run_text_retrieval(pipeline: Pipeline, query_text: str, top_k: int) -> List[Document]:
    """
    Execute the configured query pipeline and return the retrieved documents.
    """
    results = pipeline.run(
        {
            "text_embedder": {"text": query_text},
            "embedding_retriever": {"top_k": top_k},
        }
    )

    retriever_output = results.get("embedding_retriever", {})
    documents = retriever_output.get("documents", []) if isinstance(retriever_output, dict) else []
    return documents
