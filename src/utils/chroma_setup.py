import os
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

def get_chroma_store(
    persistence_directory: str = os.getenv("CHROMA_PERSISTENCE_DIR", "../../data/vector_databases/chroma_db_raw_boardgames"),
    collection_name: str = os.getenv("CHROMA_COLLECTION", "raw_boardgames"),
    ollama_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    ollama_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
) -> ChromaDocumentStore:
    """
    Cria uma conexão com uma instância ChromaDB com uma função de embedding Ollama especificada.
    """
    # A URL para OllamaEmbeddingFunction deve ser a URL base do servidor Ollama.
    # Ela internamente chamará o endpoint /api/embeddings.

    print(f"Conectando ao ChromaDB na coleção '{collection_name}' com modelo de embedding '{ollama_model}' e URL '{ollama_url}'")

    document_store = ChromaDocumentStore(
        collection_name=collection_name,
        persist_path=persistence_directory,
    )
    
    print(f"Inicializando ChromaDocumentStore com persistência em '{persistence_directory}' e coleção '{collection_name}'")
    print("\n")
    
    try:
        print(f"Conectado à coleção '{document_store._collection_name}'. Documentos existentes: {document_store.count_documents()}")
    except Exception as e:
        print(f"A coleção '{document_store._collection_name}' pode não existir ainda ou erro ao conectar: {e}")
        
    return document_store

if __name__ == "__main__":
    try:
        # Teste a conexão e configuração do embedding
        print("Tentando conectar ao ChromaDB com Ollama embedding function...")
        store = get_chroma_store()
        print(f"Conectado com sucesso ao ChromaDB em {store.host}:{store.port}")
        print(f"Nome da coleção: {store._collection_name}")
        print(f"Função de embedding configurada no store: {store.embedding_function}")
        print(f"Tipo da função de embedding configurada: {type(store.embedding_function)}")
        
        # Teste simples de embedding (opcional)
        # print("Testando embedding de uma frase de exemplo...")
        # example_embedding = store.embedding_function(["Olá mundo!"])
        # print(f"Embedding de exemplo gerado (primeiros 5 valores): {example_embedding[0][:5]}")
        # print(f"Contagem de documentos na coleção: {store.count_documents()}")

    except Exception as e:
        print(f"Falha ao conectar/configurar o ChromaDB: {e}")