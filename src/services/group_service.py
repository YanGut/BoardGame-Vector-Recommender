from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from src.dtos.group_dtos import (
    CriarGruposRequest,
    CriarGruposResponse,
    JogadorDTO,
    JogoRecomendadoDTO,
    MesaDTO,
    PerfilMesaDTO,
)
from src.services.embedding_service import EmbeddingService
from src.services.recommendation_service import recommendation_service_instance

if TYPE_CHECKING:
    from src.services.recommendation_service import RecommendationService


DEFAULT_TOP_K_RECOMMENDATIONS = 5
MAX_REBALANCE_ITERATIONS = 1000


class GroupService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        recommendation_service: "RecommendationService",
    ) -> None:
        self.embedding_service = embedding_service
        self.recommendation_service = recommendation_service

    def create_groups(self, request_data: CriarGruposRequest) -> CriarGruposResponse:
        jogadores = list(request_data.jogadores)
        if not jogadores:
            raise ValueError("Nenhum jogador informado para criação de grupos.")

        quantidade_mesas = request_data.quantidadeMesas
        if quantidade_mesas <= 0:
            raise ValueError("A quantidade de mesas deve ser maior que zero.")
        if quantidade_mesas > len(jogadores):
            raise ValueError("Quantidade de mesas maior que o número de jogadores disponível.")

        restricoes = request_data.restricoes
        minimo = restricoes.tamanhoMinimoMesa if restricoes and restricoes.tamanhoMinimoMesa else 1
        maximo = restricoes.tamanhoMaximoMesa if restricoes and restricoes.tamanhoMaximoMesa else max(len(jogadores), minimo)
        maximo = max(maximo, minimo)
        limite_duracao = restricoes.duracaoMaxima if restricoes else None

        if len(jogadores) < minimo * quantidade_mesas:
            raise ValueError("Número de jogadores insuficiente para cumprir o tamanho mínimo das mesas.")
        if len(jogadores) > maximo * quantidade_mesas:
            raise ValueError("Número de jogadores excede a capacidade máxima configurada para as mesas.")

        perfis_textuais = [self._generate_player_profile_string(j) for j in jogadores]
        embeddings = self.embedding_service.get_embeddings(perfis_textuais)
        if len(embeddings) != len(jogadores):
            raise ValueError("Falha ao gerar embeddings para todos os jogadores.")

        embedding_matrix = np.array(embeddings)

        kmeans = KMeans(n_clusters=quantidade_mesas, random_state=42, n_init=10)
        kmeans.fit(embedding_matrix)

        initial_tables = self._build_initial_tables(kmeans.labels_, len(jogadores), quantidade_mesas)
        similarity_matrix = cosine_similarity(embedding_matrix, kmeans.cluster_centers_)

        rebalanced_tables = self._rebalance_tables(
            tables=initial_tables,
            similarity_matrix=similarity_matrix,
            min_size=minimo,
            max_size=maximo,
        )

        mesas_response: List[MesaDTO] = []
        for mesa_id, player_indices in enumerate(rebalanced_tables, start=1):
            mesa_jogadores = [jogadores[idx] for idx in player_indices]
            perfil = self._create_table_profile(mesa_jogadores)
            query = self._generate_table_profile_string(perfil, limite_duracao)

            recomendacoes = self.recommendation_service.recommend_games_hybrid(
                query_text=query,
                top_k=DEFAULT_TOP_K_RECOMMENDATIONS,
                candidate_pool_size=100,
                semantic_weight=0.6,
                popularity_weight=0.4,
            )

            jogos_recomendados = self._map_recommendations_to_dto(recomendacoes, limite_duracao)

            mesas_response.append(
                MesaDTO(
                    mesaId=mesa_id,
                    jogadores=mesa_jogadores,
                    perfilMesa=perfil,
                    jogosRecomendados=jogos_recomendados,
                )
            )

        return CriarGruposResponse(
            eventoId=request_data.eventoId,
            mesas=mesas_response,
        )

    def _build_initial_tables(
        self,
        labels: Sequence[int],
        player_count: int,
        table_count: int,
    ) -> List[List[int]]:
        tables: List[List[int]] = [[] for _ in range(table_count)]
        for player_idx in range(player_count):
            label = labels[player_idx]
            tables[label].append(player_idx)
        return tables

    def _rebalance_tables(
        self,
        tables: List[List[int]],
        similarity_matrix: np.ndarray,
        min_size: int,
        max_size: int,
    ) -> List[List[int]]:
        tables = [list(table) for table in tables]
        table_count = len(tables)

        for _ in range(MAX_REBALANCE_ITERATIONS):
            changed = False
            # Reduce oversized tables
            for table_idx in range(table_count):
                while len(tables[table_idx]) > max_size:
                    player_idx = min(
                        tables[table_idx],
                        key=lambda idx: similarity_matrix[idx][table_idx],
                    )
                    target_tables = self._sorted_table_preferences(similarity_matrix[player_idx])
                    moved = False
                    for target_idx in target_tables:
                        if target_idx == table_idx:
                            continue
                        if len(tables[target_idx]) < max_size:
                            tables[table_idx].remove(player_idx)
                            tables[target_idx].append(player_idx)
                            changed = True
                            moved = True
                            break
                    if not moved:
                        break

            # Fill undersized tables
            for table_idx in range(table_count):
                while len(tables[table_idx]) < min_size:
                    candidate = self._find_best_player_to_move(
                        destination_table=table_idx,
                        tables=tables,
                        similarity_matrix=similarity_matrix,
                        min_size=min_size,
                    )
                    if candidate is None:
                        break
                    player_idx, source_table = candidate
                    tables[source_table].remove(player_idx)
                    tables[table_idx].append(player_idx)
                    changed = True

            if not changed:
                break

        if not all(min_size <= len(table) <= max_size for table in tables):
            raise ValueError("Não foi possível balancear as mesas respeitando os limites estabelecidos.")

        return tables

    def _find_best_player_to_move(
        self,
        destination_table: int,
        tables: List[List[int]],
        similarity_matrix: np.ndarray,
        min_size: int,
    ) -> Optional[Tuple[int, int]]:
        best_gain = None
        best_candidate: Optional[Tuple[int, int]] = None

        for source_idx, table in enumerate(tables):
            if source_idx == destination_table or len(table) <= min_size:
                continue

            for player_idx in table:
                current_similarity = similarity_matrix[player_idx][source_idx]
                new_similarity = similarity_matrix[player_idx][destination_table]
                gain = new_similarity - current_similarity
                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best_candidate = (player_idx, source_idx)

        return best_candidate

    def _sorted_table_preferences(self, similarities: Iterable[float]) -> List[int]:
        indexed = list(enumerate(similarities))
        indexed.sort(key=lambda item: item[1], reverse=True)
        return [idx for idx, _ in indexed]

    def _generate_player_profile_string(self, jogador: JogadorDTO) -> str:
        parts: List[str] = []
        nivel = self._normalize_experience_level(jogador.nivelExperiencia)
        parts.append(f"Player has {nivel} experience.")

        preferencias = jogador.preferencias
        if preferencias:
            if preferencias.mecanicasFavoritas:
                mecanicas = ", ".join(preferencias.mecanicasFavoritas)
                parts.append(f"Enjoys mechanics like {mecanicas}.")
            if preferencias.temasFavoritos:
                temas = ", ".join(preferencias.temasFavoritos)
                parts.append(f"Prefers themes such as {temas}.")
            if preferencias.tempoDisponivel:
                parts.append(f"Available for sessions up to {preferencias.tempoDisponivel} minutes.")

        return " ".join(parts)

    def _create_table_profile(self, jogadores: Sequence[JogadorDTO]) -> PerfilMesaDTO:
        niveis = [self._normalize_experience_level(j.nivelExperiencia) for j in jogadores if j.nivelExperiencia]
        nivel_predominante = self._most_common(niveis, default="intermediario")

        mecanicas = [
            mecanica
            for jogador in jogadores
            if jogador.preferencias and jogador.preferencias.mecanicasFavoritas
            for mecanica in jogador.preferencias.mecanicasFavoritas
        ]
        temas = [
            tema
            for jogador in jogadores
            if jogador.preferencias and jogador.preferencias.temasFavoritos
            for tema in jogador.preferencias.temasFavoritos
        ]
        tempos = [
            jogador.preferencias.tempoDisponivel
            for jogador in jogadores
            if jogador.preferencias and jogador.preferencias.tempoDisponivel
        ]

        mecanicas_predominantes = self._top_n(mecanicas, limit=3)
        temas_predominantes = self._top_n(temas, limit=3)
        tempo_medio = int(mean(tempos)) if tempos else None

        return PerfilMesaDTO(
            nivelPredominante=nivel_predominante,
            mecanicasPredominantes=mecanicas_predominantes,
            temasPredominantes=temas_predominantes,
            tempoMedioDisponivel=tempo_medio,
        )

    def _generate_table_profile_string(
        self,
        perfil: PerfilMesaDTO,
        limite_duracao: Optional[int],
    ) -> str:
        parts = [
            f"A group of {perfil.nivelPredominante} board game players.",
        ]
        if perfil.mecanicasPredominantes:
            parts.append(f"They enjoy mechanics such as {', '.join(perfil.mecanicasPredominantes)}.")
        if perfil.temasPredominantes:
            parts.append(f"They prefer themes like {', '.join(perfil.temasPredominantes)}.")
        if perfil.tempoMedioDisponivel:
            parts.append(f"The average available play time is {perfil.tempoMedioDisponivel} minutes.")
        if limite_duracao:
            parts.append(f"Sessions should not exceed {limite_duracao} minutes.")

        return " ".join(parts)

    def _map_recommendations_to_dto(
        self,
        recommendations: Sequence[dict],
        limite_duracao: Optional[int],
    ) -> List[JogoRecomendadoDTO]:
        filtered = recommendations
        if limite_duracao is not None:
            filtered = [
                reco
                for reco in recommendations
                if reco.get("vlTempoJogo") is None or reco.get("vlTempoJogo") <= limite_duracao
            ]
            if not filtered:
                filtered = recommendations

        top_results = filtered[:DEFAULT_TOP_K_RECOMMENDATIONS]
        jogos: List[JogoRecomendadoDTO] = []
        for reco in top_results:
            jogos.append(
                JogoRecomendadoDTO(
                    idJogo=reco.get("id_mysql") or reco.get("id_chroma"),
                    nome=reco.get("nmJogo", ""),
                    similaridade=float(reco.get("final_score") or reco.get("score") or 0.0),
                    thumbnail=reco.get("thumb"),
                )
            )
        return jogos

    def _top_n(self, items: Iterable[str], limit: int) -> List[str]:
        counter = Counter([item for item in items if item])
        most_common = counter.most_common(limit)
        return [value for value, _ in most_common]

    def _most_common(self, values: Sequence[str], default: str) -> str:
        if not values:
            return default
        counter = Counter(values)
        return counter.most_common(1)[0][0]

    def _normalize_experience_level(self, nivel: str) -> str:
        if not nivel:
            return "intermediario"
        nivel_lower = nivel.strip().lower()
        if "inic" in nivel_lower:
            return "iniciante"
        if "av" in nivel_lower:
            return "avancado"
        return "intermediario"


embedding_service_instance = EmbeddingService()
group_service_instance = GroupService(
    embedding_service=embedding_service_instance,
    recommendation_service=recommendation_service_instance,
)
