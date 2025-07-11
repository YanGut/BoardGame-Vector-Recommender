import pytest
from haystack.testing.document_store import DocumentStoreBaseTests
from haystack_integrations.document_stores.example_store import ExampleDocumentStore


@pytest.mark.skip("This is an example Document Store")
class TestDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def docstore(self) -> ExampleDocumentStore:
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        return ExampleDocumentStore()
