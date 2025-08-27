import os

from tqdm import tqdm

from haystack import Pipeline, Document

from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from typing import List

from src.utils.chroma_setup import get_chroma_store
from src.utils.prepare_haystack_docs import prepare_haystack_documents

class RecommendationService:    
    def __init__(self) -> None:
        self.document_store: ChromaDocumentStore = get_chroma_store()
        self.query_pipeline: Pipeline = self._create_query_pipeline()
        self.ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.ollama_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def recommend_games(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Busca por jogos baseados em uma query textual."""
        try:
            results = self.query_pipeline.run({
                "text_embedder": {"text": query_text},
                "embedding_retriever": {"top_k": top_k}
            })

            recommendations: list[dict] = []
            
            print(f"Results: {results}")
            print(f"Results type: {type(results)}")

            if results and "embedding_retriever" in results and "documents" in results["embedding_retriever"]:
                for doc in results["embedding_retriever"]["documents"]:
                    recommendations.append({
                        "id_mysql": doc.meta.get("mysql_id"), # ID original do MySQL
                        "id_chroma": doc.id, # ID do documento no ChromaDB
                        "description": doc.content, # Conteúdo que foi embeddado
                        "nmJogo": doc.meta.get("title"),
                        "thumb": doc.meta.get("thumbnail"),
                        "idadeMinima": doc.meta.get("min_age"),
                        "qtJogadoresMin": doc.meta.get("min_players"),
                        "qtJogadoresMax": doc.meta.get("max_players"),
                        "vlTempoJogo": doc.meta.get("play_time_minutes"),
                        "anoPublicacao": doc.meta.get("ano_publicacao", 0),
                        "anoNacional": doc.meta.get("ano_nacional", 0),
                        "tpJogo": doc.meta.get("game_type"),
                        "artistas": doc.meta.get("artists_list", []),
                        "designers": doc.meta.get("designers_list", []),
                        "categorias": doc.meta.get("categories_list", []),
                        "mecanicas": doc.meta.get("mechanics_list", []),
                        "temas": doc.meta.get("themes_list", []),
                        "score": doc.score
                    })
            return recommendations
        except Exception as e:
            print(f"Erro ao executar pipeline de recomendação: {e}")
            # Considere lançar uma exceção customizada ou retornar um erro
            return []

    def get_game_by_mysql_id(self, game_mysql_id: int) -> dict | None:
        """Busca um jogo pelo seu ID original do MySQL."""
        # ChromaDB permite filtrar por metadados.
        # Certifique-se que 'id_mysql' está no campo meta dos seus documentos.
        try:
            # O método filter_documents retorna uma lista de documentos
            # Não há garantia de que o ID é único se não for o ID primário do Chroma
            filtered_docs = self.document_store.filter_documents(filters={"id_mysql": game_mysql_id})
            
            if filtered_docs:
                doc = filtered_docs[0] # Assume que o primeiro é o correto se houver múltiplos
                return {
                    "id_mysql": doc.meta.get("id_mysql"),
                    "id_chroma": doc.id,
                    "name": doc.meta.get("name"),
                    "description": doc.content,
                    # Adicione outros campos meta
                }
            return None
        except Exception as e:
            print(f"Erro ao buscar jogo por ID MySQL {game_mysql_id}: {e}")
            return None

    def list_all_games(self, page: int = 1, per_page: int = 20):
        """
        Lista jogos de forma paginada.
        NOTA: A paginação eficiente em bancos vetoriais pode ser complexa.
        ChromaDB oferece `offset` e `limit` na query direta, mas via Haystack
        `filter_documents` ou `get_all_documents` pode ser menos direto para paginação eficiente.
        Esta é uma implementação simples e pode não ser performática para datasets muito grandes.
        """
        try:
            # Haystack DocumentStore não tem um método de paginação direto fácil
            # A abordagem de pegar muitos e fatiar é o que você tinha,
            # mas ChromaDB em si suporta offset/limit em queries nativas.
            # Para uma solução mais robusta, você poderia:
            # 1. Usar client ChromaDB diretamente para queries com offset/limit (fora do pipeline Haystack para esta rota).
            # 2. Se o número de jogos não for gigantesco, a abordagem atual pode ser aceitável.

            # Tentativa com `get_all_documents` e fatiamento manual (pode ser ineficiente)
            # Esta é uma limitação comum; bancos vetoriais são otimizados para busca por similaridade.
            # Para listagem geral, um banco de dados tradicional ainda pode ser mais adequado
            # ou usar filtros se a busca por "todos" não for realmente "todos" mas "todos que correspondem a X".

            # Para simplificar, vamos manter a lógica de buscar muitos e fatiar,
            # ciente de suas limitações de performance.
            # A query "game" que você usava é uma forma de tentar pegar todos.
            # Uma alternativa seria usar `document_store.get_all_documents()` mas pode ser pesado.
            
            # Usando a mesma lógica de antes, mas com o pipeline do serviço:
            # A query vazia ou genérica pode não ser a melhor forma.
            # Se ChromaDB/Haystack tiver um `get_all_documents_paginated` seria ideal.
            # Por ora, vamos adaptar sua lógica anterior:
            
            # Simula uma busca genérica para obter documentos. Não é o ideal para "listar todos".
            # Você pode querer um endpoint que aceite filtros em vez de "listar tudo".
            results_for_pagination = self.query_pipeline.run({"retriever": {"query": "", "top_k": 10000}}) # Tentar pegar um número grande
            
            all_docs = []
            if results_for_pagination and "retriever" in results_for_pagination and "documents" in results_for_pagination["retriever"]:
                all_docs = results_for_pagination["retriever"]["documents"]

            total_games = len(all_docs)
            start_index = (page - 1) * per_page
            end_index = start_index + per_page
            paginated_docs = all_docs[start_index:end_index]

            games_list = []
            for doc in paginated_docs:
                games_list.append({
                    "id_mysql": doc.meta.get("id_mysql"),
                    "id_chroma": doc.id,
                    "name": doc.meta.get("name"),
                    # Adicione outros campos meta
                })
            
            return {
                "page": page,
                "per_page": per_page,
                "total": total_games,
                "games": games_list
            }
        except Exception as e:
            print(f"Erro ao listar jogos: {e}")
            return {"page": page, "per_page": per_page, "total": 0, "games": []}
    
    def insert_game(self, game_data: dict) -> bool:
        """
        Insere um novo jogo no ChromaDB.
        game_data deve conter os campos necessários para criar um documento.
        """
        try:
            boardgames_data: list[dict] = [game_data]
            
            haystack_docs: list[Document] = prepare_haystack_documents(boardgames_data=boardgames_data)
            
            self.ollama_embed_model.run(haystack_docs)
            
            writer: DocumentWriter = DocumentWriter(
                document_store=self.document_store,
                policy=DuplicatePolicy.OVERWRITE
            )
            
            indexing_pipeline: Pipeline = Pipeline()
            indexing_pipeline.add_component("embedder", self.ollama_embed_model)
            indexing_pipeline.add_component("writer", writer)
            indexing_pipeline.connect("embedder.documents", "writer.documents")
            
            batch_size: int = 64
            
            for i in tqdm(range(0, len(haystack_docs), batch_size), desc="Indexando documentos em lotes"):
                batch_docs: List[Document] = haystack_docs[i:i + batch_size]
                indexing_pipeline.run({
                    "embedder": {"documents": batch_docs},
                })

            return True
        except Exception as e:
            print(f"Erro ao inserir jogo: {e}")
            return False

    def _create_query_pipeline(self) -> Pipeline:
        """
        Create an Haystack query pipeline for retrieving game recommendations.
        This pipeline uses a ChromaQueryTextRetriever to fetch documents based on text queries.
        The embedding function is expected to be set in the document store.
        The pipeline is initialized with the document store's embedding function.
        This function raises a ValueError if the document store does not have an embedding function configured.
        
        Returns:
            Pipeline: An instance of Haystack's Pipeline configured with a retriever.

        Raises:
            ValueError: If the document store does not have an embedding function configured.
        """
        ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        ollama_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        embedding_function: OllamaTextEmbedder = OllamaTextEmbedder(
            model=ollama_embed_model,
            url=ollama_url,
        )
        if not embedding_function:
            raise ValueError("The document store does not have an embedding function configured.")
        
        embedding_retriever: ChromaEmbeddingRetriever = ChromaEmbeddingRetriever(
            document_store=self.document_store,
        )
        
        query_pipeline: Pipeline = Pipeline()
        query_pipeline.add_component("text_embedder", embedding_function)
        query_pipeline.add_component("embedding_retriever", embedding_retriever)
        
        query_pipeline.connect('text_embedder.embedding', 'embedding_retriever.query_embedding')
        
        return query_pipeline

recommendation_service_instance = RecommendationService()