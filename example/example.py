import os
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter

from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

# Chroma is used in-memory so we use the same instances in the two pipelines below
document_store = ChromaDocumentStore()

HERE = Path(__file__).resolve().parent
file_paths = [HERE / "data" / Path(name) for name in os.listdir(HERE / "data")]


indexing = Pipeline()
indexing.add_component("converter", TextFileToDocument())
indexing.add_component("writer", DocumentWriter(document_store))
indexing.connect("converter", "writer")
indexing.run({"converter": {"sources": file_paths}})

querying = Pipeline()
querying.add_component("retriever", ChromaQueryTextRetriever(document_store))
results = querying.run({"retriever": {"query": "Variable declarations", "top_k": 3}})

for d in results["retriever"]["documents"]:
    print(d.meta, d.score)