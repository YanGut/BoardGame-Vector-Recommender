from haystack.document_stores.errors import DocumentStoreError
from haystack.errors import FilterError


class ChromaDocumentStoreError(DocumentStoreError):
    """Parent class for all ChromaDocumentStore exceptions."""

    pass


class ChromaDocumentStoreFilterError(FilterError, ValueError):
    """Raised when a filter is not valid for a ChromaDocumentStore."""

    pass


class ChromaDocumentStoreConfigError(ChromaDocumentStoreError):
    """Raised when a configuration is not valid for a ChromaDocumentStore."""

    pass
