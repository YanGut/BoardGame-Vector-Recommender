import os

from haystack import Pipeline, Document
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from src.repositories.game_repository import GameRepository
from src.utils.chroma_setup import get_chroma_store
from src.utils.prepare_haystack_docs import prepare_haystack_documents
from src.services.recommendation import (
    build_query_pipeline,
    document_to_game_dict,
    document_to_hybrid_game_dict,
    hybrid_rank,
    run_text_retrieval,
)

class RecommendationService:    
    def __init__(self) -> None:
        self.document_store: ChromaDocumentStore = get_chroma_store()
        self.repository: GameRepository = GameRepository(self.document_store)

        embed_model_name = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.document_embedder = OllamaDocumentEmbedder(
            model=embed_model_name,
            url=ollama_url,
        )

        self.query_pipeline: Pipeline = build_query_pipeline(self.document_store)

    def recommend_games(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Busca por jogos baseados em uma query textual."""
        try:
            documents = run_text_retrieval(self.query_pipeline, query_text, top_k)
            recommendations: list[dict] = []
            
            print(f"Retrieved {len(documents)} documents for query '{query_text}'.")

            for doc in documents:
                recommendations.append(document_to_game_dict(doc))
            return recommendations
        except Exception as e:
            print(f"Erro ao executar pipeline de recomendação: {e}")
            # Considere lançar uma exceção customizada ou retornar um erro
            return []

    def recommend_games_hybrid(
        self, 
        query_text: str, 
        top_k: int = 10,
        candidate_pool_size: int = 100,
        semantic_weight: float = 0.7,
        popularity_weight: float = 0.3
    ) -> list[dict]:
        """
        Busca por jogos usando uma abordagem híbrida de re-ranking.
        Combina a busca semântica com um score de popularidade pré-calculado.
        """
        try:
            print(f"\n[Hybrid Search] Iniciando busca para query: '{query_text}'")
            # 1. Aumentar o pool de candidatos
            documents = run_text_retrieval(self.query_pipeline, query_text, candidate_pool_size)

            if not documents:
                print("[Hybrid Search] Nenhum candidato inicial encontrado.")
                return []

            initial_candidates = documents
            print(f"[Hybrid Search] {len(initial_candidates)} candidatos iniciais recuperados.")
            
            ranked_results = hybrid_rank(
                documents=initial_candidates,
                semantic_weight=semantic_weight,
                popularity_weight=popularity_weight,
                top_k=top_k,
            )

            final_recommendations = [
                document_to_hybrid_game_dict(doc, final_score=score)
                for doc, score in ranked_results
            ]

            print(f"[Hybrid Search] Retornando {len(final_recommendations)} recomendações finais.")
            return final_recommendations

        except Exception as e:
            print(f"Erro ao executar pipeline de recomendação híbrida: {e}")
            return []

    def get_game_by_mysql_id(self, game_mysql_id: int) -> dict | None:
        """Busca um jogo pelo seu ID original do MySQL."""
        try:
            doc = self.repository.get_by_mysql_id(game_mysql_id)
            if not doc:
                return None

            game_payload = document_to_game_dict(doc)
            # Preserva campo legacy 'name' usado anteriormente no endpoint.
            game_payload["name"] = doc.meta.get("name")
            return game_payload
        except Exception as e:
            print(f"Erro ao buscar jogo por ID MySQL {game_mysql_id}: {e}")
            return None

    def list_all_games(self, page: int = 1, per_page: int = 20):
        """
        Lists games with efficient, database-side pagination.
        This implementation directly uses the underlying ChromaDB client to fetch
        only the required page of documents, ensuring scalability.
        """
        try:
            documents, total_games, normalized_page, normalized_per_page = self.repository.list_paginated(page, per_page)
            games_list = [document_to_game_dict(doc) for doc in documents]
            
            return {
                "page": normalized_page,
                "per_page": normalized_per_page,
                "total": total_games,
                "games": games_list
            }
        except Exception as e:
            # It's good practice to log the specific error.
            print(f"Error listing games with direct pagination: {e}")
            return {"page": page, "per_page": per_page, "total": 0, "games": []}
    
    def insert_game(self, game_data: dict) -> bool:
        """
        Insere um novo jogo no ChromaDB.
        game_data deve conter os campos necessários para criar um documento.
        """
        try:
            boardgames_data: list[dict] = [game_data]
            
            haystack_docs: list[Document] = prepare_haystack_documents(boardgames_data=boardgames_data)

            self.repository.index_documents(
                documents=haystack_docs,
                embedder=self.document_embedder,
            )

            return True
        except Exception as e:
            print(f"Erro ao inserir jogo: {e}")
            return False

recommendation_service_instance = RecommendationService()
