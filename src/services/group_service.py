from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import normalize

from src.dtos.group_dtos import (
    CriarGruposRequest,
    CriarGruposResponse,
    JogadorDTO,
    JogoRecomendadoDTO,
    MesaDTO,
    PerfilMesaDTO,
    AssignPlayerRequest,
    MesaExistenteDTO,
    RestricoesDTO,
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

        restricoes = request_data.restricoes
        minimo = restricoes.tamanhoMinimoMesa if restricoes and restricoes.tamanhoMinimoMesa else 1
        maximo = restricoes.tamanhoMaximoMesa if restricoes and restricoes.tamanhoMaximoMesa else max(len(jogadores), minimo)
        maximo = max(maximo, minimo)

        perfis_textuais = [self._generate_player_profile_string(j) for j in jogadores]
        embeddings = self.embedding_service.get_embeddings(perfis_textuais)
        if len(embeddings) != len(jogadores):
            raise ValueError("Falha ao gerar embeddings para todos os jogadores.")

        embedding_matrix = normalize(np.array(embeddings))

        eps = restricoes.dbscanEps if restricoes and restricoes.dbscanEps else 0.35
        min_samples = restricoes.dbscanMinSamples if restricoes and restricoes.dbscanMinSamples else 2

        distance_matrix = cosine_distances(embedding_matrix)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        clustering.fit(distance_matrix)

        labels = clustering.labels_
        tables, outliers = self._build_tables_from_labels(labels)
        
        print("\n--- [DEBUG] INÍCIO DO PROCESSO DE GRUPO ---")
        print(f"[DEBUG] Jogadores: {len(jogadores)}, Min/Max: {minimo}/{maximo}, EPS: {eps}, MinSamples: {min_samples}")
        print(f"[DEBUG] DBSCAN Labels: {labels}")
        print(f"[DEBUG] Mesas iniciais (Clusters): {tables}")
        print(f"[DEBUG] Outliers: {outliers}")

        self._assign_outliers(tables, outliers, embedding_matrix, maximo)
        
        print(f"[DEBUG] Mesas após _assign_outliers: {tables}")
        print(f"[DEBUG] Tamanhos: {[len(t) for t in tables]}")
        print("--- [DEBUG] INICIANDO _enforce_size_bounds ---")
        
        self._enforce_size_bounds(tables, embedding_matrix, min_size=minimo, max_size=maximo)

        if not tables:
            raise ValueError("Nenhum agrupamento pôde ser formado com os parâmetros fornecidos.")

        mesas_response: List[MesaDTO] = []
        for mesa_id, player_indices in enumerate(tables, start=1):
            mesa_jogadores = [jogadores[idx] for idx in player_indices]
            perfil = self._create_table_profile(mesa_jogadores)
            query = self._generate_table_profile_string(perfil)

            recomendacoes = self.recommendation_service.recommend_games_hybrid(
                query_text=query,
                top_k=DEFAULT_TOP_K_RECOMMENDATIONS,
                candidate_pool_size=100,
                semantic_weight=0.6,
                popularity_weight=0.4,
            )

            jogos_recomendados = self._map_recommendations_to_dto(recomendacoes)

            mesas_response.append(
                MesaDTO(
                    mesaId=mesa_id,
                    jogadores=mesa_jogadores,
                    perfilMesa=perfil,
                    jogosRecomendados=jogos_recomendados,
                )
            )

        return CriarGruposResponse(
            mesas=mesas_response,
        )

    def assign_player(self, request_data: AssignPlayerRequest) -> CriarGruposResponse:
        """
        Stateless, incremental player assignment.
        Decides whether to add the new player to an existing table or create a new one,
        then returns the full, updated state (including fresh recommendations for every table).
        """
        novo_jogador: JogadorDTO = request_data.novoJogador
        mesas_existentes: List[MesaExistenteDTO] = list(request_data.mesasExistentes or [])
        restricoes = request_data.restricoes or RestricoesDTO()

        max_size = restricoes.tamanhoMaximoMesa if restricoes.tamanhoMaximoMesa else 6
        threshold = restricoes.similarityThreshold if restricoes.similarityThreshold is not None else 0.5

        # Compute normalized embedding for new player
        novo_profile_text = self._generate_player_profile_string(novo_jogador)
        novo_embedding = self.embedding_service.get_embeddings([novo_profile_text])
        if not novo_embedding or not novo_embedding[0]:
            raise ValueError("Falha ao gerar embedding para o novo jogador.")
        novo_vec = normalize(np.array(novo_embedding))[0]

        # Find best-fit existing table by centroid cosine similarity
        best_idx: Optional[int] = None
        best_similarity: Optional[float] = None

        for idx, mesa in enumerate(mesas_existentes):
            jogadores_mesa = list(mesa.jogadores or [])
            if len(jogadores_mesa) == 0:
                # Skip empty tables in selection; they offer no signal
                continue
            if len(jogadores_mesa) >= max_size:
                continue

            textos_mesa = [self._generate_player_profile_string(j) for j in jogadores_mesa]
            mesa_embeddings = self.embedding_service.get_embeddings(textos_mesa)
            if not mesa_embeddings:
                continue
            mesa_matrix = normalize(np.array(mesa_embeddings))
            centroid = self._compute_centroid(list(range(mesa_matrix.shape[0])), mesa_matrix)

            similarity = self._cosine_similarity(novo_vec, centroid)

            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx

        best_fit_found = best_idx is not None and (best_similarity or 0.0) >= threshold

        if best_fit_found:
            # Add to the best existing table in-memory
            mesas_existentes[best_idx].jogadores.append(novo_jogador)
        else:
            # Create a new temporary table and append it
            next_id = 1
            if mesas_existentes:
                try:
                    next_id = max(m.mesaId for m in mesas_existentes) + 1
                except ValueError:
                    next_id = 1
            nova_mesa = MesaExistenteDTO(mesaId=next_id, jogadores=[novo_jogador])
            mesas_existentes.append(nova_mesa)

        # Build full response with updated recommendations for all tables
        mesas_response: List[MesaDTO] = []
        for mesa_ex in mesas_existentes:
            mesa_jogadores = list(mesa_ex.jogadores or [])
            perfil = self._create_table_profile(mesa_jogadores)
            query = self._generate_table_profile_string(perfil)

            recomendacoes = self.recommendation_service.recommend_games_hybrid(
                query_text=query,
                top_k=DEFAULT_TOP_K_RECOMMENDATIONS,
                candidate_pool_size=100,
                semantic_weight=0.6,
                popularity_weight=0.4,
            )

            jogos_recomendados = self._map_recommendations_to_dto(recomendacoes)

            mesas_response.append(
                MesaDTO(
                    mesaId=mesa_ex.mesaId,
                    jogadores=mesa_jogadores,
                    perfilMesa=perfil,
                    jogosRecomendados=jogos_recomendados,
                )
            )

        return CriarGruposResponse(mesas=mesas_response)

    def _build_tables_from_labels(
        self,
        labels: Sequence[int],
    ) -> Tuple[List[List[int]], List[int]]:
        tables: dict[int, List[int]] = {}
        outliers: List[int] = []
        for idx, label in enumerate(labels):
            if label == -1:
                outliers.append(idx)
            else:
                tables.setdefault(label, []).append(idx)
        # Sort tables by label to ensure deterministic ordering
        sorted_tables = [tables[label] for label in sorted(tables.keys())]
        return sorted_tables, outliers

    def _assign_outliers(
        self,
        tables: List[List[int]],
        outliers: List[int],
        embedding_matrix: np.ndarray,
        max_size: int,
    ) -> None:
        if not outliers:
            return
        if not tables:
            # Seed first table entirely from outliers if no clusters formed
            while outliers:
                tables.append([outliers.pop(0)])
            return

        centroids = self._compute_centroids(tables, embedding_matrix)
        for player_idx in outliers:
            similarities = [
                self._cosine_similarity(embedding_matrix[player_idx], centroid)
                for centroid in centroids
            ]
            best_table = None
            best_similarity = None
            for table_idx, similarity in enumerate(similarities):
                if len(tables[table_idx]) >= max_size:
                    continue
                if best_similarity is None or similarity > best_similarity:
                    best_similarity = similarity
                    best_table = table_idx
            if best_table is None:
                tables.append([player_idx])
                centroids.append(embedding_matrix[player_idx])
            else:
                tables[best_table].append(player_idx)
                centroids[best_table] = self._compute_centroid(tables[best_table], embedding_matrix)

    def _enforce_size_bounds(
        self,
        tables: List[List[int]],
        embedding_matrix: np.ndarray,
        min_size: int,
        max_size: int,
    ) -> None:
        if not tables:
            return

        for _ in range(MAX_REBALANCE_ITERATIONS):
            print(f"\n[DEBUG] Iteração de Balanceamento #{_}")
            print(f"[DEBUG] Estado atual das mesas: {tables}")
            print(f"[DEBUG] Tamanhos: {[len(t) for t in tables]}")
            
            changed = False
            centroids = self._compute_centroids(tables, embedding_matrix)

            for idx, table in enumerate(list(tables)):
                if len(table) == 0:
                    print(f"[DEBUG] Mesa {idx} está vazia. Removendo.")
                    tables.pop(idx)
                    centroids.pop(idx)
                    changed = True
                    continue
                if len(table) > max_size:
                    print(f"[DEBUG] Mesa {idx} (tamanho {len(table)}) > max_size ({max_size}). Encolhendo...")
                    self._shrink_table(tables, idx, centroids, embedding_matrix, max_size)
                    changed = True
                elif len(table) < min_size:
                    print(f"[DEBUG] Mesa {idx} (tamanho {len(table)}) < min_size ({min_size}). Tentando preencher...")
                    if not self._fill_table(tables, idx, centroids, embedding_matrix, min_size):
                        print(f"[DEBUG] _fill_table falhou para mesa {idx}. Tentando _merge_with_best_table...")
                        if self._merge_with_best_table(
                            tables,
                            idx,
                            centroids,
                            embedding_matrix,
                            max_size,
                        ):
                            print(f"[DEBUG] _merge_with_best_table SUCESSO para mesa {idx}.")
                            changed = True
                            continue
                        else:
                            print(f"[DEBUG] _merge_with_best_table FALHOU para mesa {idx}. Impossível balancear.")
                            raise ValueError("Não foi possível balancear as mesas respeitando os limites estabelecidos.")
                    print(f"[DEBUG] _fill_table SUCESSO para mesa {idx}.")
                    changed = True
            if not changed:
                print("[DEBUG] Balanceamento estável. Saindo do loop.")
                break
        print("[DEBUG] Verificação final de balanceamento...")
        if not all(min_size <= len(table) <= max_size for table in tables):
            print(f"[DEBUG] FALHA NA VERIFICAÇÃO FINAL. Tamanhos: {[len(t) for t in tables]}")
            raise ValueError("Não foi possível balancear as mesas respeitando os limites estabelecidos.")
        print(f"[DEBUG] SUCESSO. Tamanhos finais: {[len(t) for t in tables]}")

    def _shrink_table(
        self,
        tables: List[List[int]],
        table_idx: int,
        centroids: List[np.ndarray],
        embedding_matrix: np.ndarray,
        max_size: int,
    ) -> None:
        # Adicione este log para ver o início da função
        print(f"    [shrink_table] Encolhendo mesa {table_idx} (tamanho {len(tables[table_idx])}) para max {max_size}")

        while len(tables[table_idx]) > max_size:
            # Encontra o jogador menos similar ao centro da mesa atual
            player_idx = min(
                tables[table_idx],
                key=lambda idx: self._cosine_similarity(
                    embedding_matrix[idx],
                    centroids[table_idx],
                ),
            )
            
            # Tenta encontrar a melhor *outra* mesa para este jogador
            target_idx = self._best_table_for_player(
                player_idx,
                current_table=table_idx,
                tables=tables,
                centroids=centroids,
                embedding_matrix=embedding_matrix,
                max_size=max_size,
            )

            # --- ESTA É A LÓGICA DE CORREÇÃO ---
            if target_idx is None:
                # Se NENHUMA outra mesa puder aceitar o jogador (porque não existem ou estão cheias),
                # nós DEVEMOS criar uma nova mesa para ele.
                print(f"    [shrink_table] Nenhuma mesa de destino encontrada para {player_idx}. Criando nova mesa...")
                
                # 1. Remover o jogador da mesa atual (que está sendo encolhida)
                tables[table_idx].remove(player_idx)
                
                # 2. Criar uma nova mesa com este jogador
                tables.append([player_idx])
                
                # 3. ATUALIZAR A LISTA DE CENTROIDS (CRÍTICO!)
                # Recalcular o centroide da mesa antiga
                centroids[table_idx] = self._compute_centroid(tables[table_idx], embedding_matrix)
                # Calcular e adicionar o centroide da nova mesa
                centroids.append(self._compute_centroid(tables[-1], embedding_matrix))
                
            else:
                # Comportamento normal: mover o jogador para a melhor mesa encontrada
                print(f"    [shrink_table] Movendo jogador {player_idx} da mesa {table_idx} para {target_idx}")
                tables[table_idx].remove(player_idx)
                tables[target_idx].append(player_idx)
                
                # Recalcular centroides de ambas as mesas
                centroids[table_idx] = self._compute_centroid(tables[table_idx], embedding_matrix)
                centroids[target_idx] = self._compute_centroid(tables[target_idx], embedding_matrix)
            
            # Log para ver o progresso dentro do while
            print(f"    [shrink_table] ...tamanhos atuais: {[len(t) for t in tables]}")

    def _fill_table(
        self,
        tables: List[List[int]],
        table_idx: int,
        centroids: List[np.ndarray],
        embedding_matrix: np.ndarray,
        min_size: int,
    ) -> bool:
        print(f"  [fill_table] Tentando preencher mesa {table_idx}...")
        for source_idx, table in enumerate(tables):
            if source_idx == table_idx or len(table) <= min_size:
                print(f"    [fill_table] Pulando doador {source_idx} (tamanho {len(table)} <= min_size {min_size})")
                continue
            player_idx = max(
                table,
                key=lambda idx: self._cosine_similarity(
                    embedding_matrix[idx],
                    centroids[table_idx],
                )
                - self._cosine_similarity(
                    embedding_matrix[idx],
                    centroids[source_idx],
                ),
            )
            tables[source_idx].remove(player_idx)
            tables[table_idx].append(player_idx)
            centroids[source_idx] = self._compute_centroid(tables[source_idx], embedding_matrix)
            centroids[table_idx] = self._compute_centroid(tables[table_idx], embedding_matrix)
            print(f"    [fill_table] Jogador movido da mesa {source_idx} para {table_idx}.")
            if len(tables[table_idx]) >= min_size:
                return True
        print(f"  [fill_table] NÃO FOI POSSÍVEL PREENCHER mesa {table_idx}. Nenhum doador encontrado.")
        return False

    def _merge_with_best_table(
        self,
        tables: List[List[int]],
        table_idx: int,
        centroids: List[np.ndarray],
        embedding_matrix: np.ndarray,
        max_size: int,
    ) -> bool:
        print(f"  [merge_table] Tentando fundir mesa {table_idx}...")
        best_idx = None
        best_similarity = None
        for idx, table in enumerate(tables):
            if idx == table_idx:
                continue
            potential_size = len(tables[table_idx]) + len(table)
            if potential_size > max_size:
                print(f"    [merge_table] Não é possível fundir com mesa {idx}. Tamanho {potential_size} > max_size {max_size}")
                continue
            similarity = self._cosine_similarity(centroids[table_idx], centroids[idx])
            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx

        if best_idx is None:
            print(f"  [merge_table] NÃO FOI POSSÍVEL FUNDIR mesa {table_idx}. Nenhum alvo de fusão encontrado.")
            return False
        print(f"  [merge_table] Fundindo mesa {table_idx} com mesa {best_idx}.")
        tables[table_idx].extend(tables[best_idx])
        tables.pop(best_idx)
        centroids[table_idx] = self._compute_centroid(tables[table_idx], embedding_matrix)
        centroids.pop(best_idx)
        return True

    def _best_table_for_player(
        self,
        player_idx: int,
        current_table: int,
        tables: List[List[int]],
        centroids: List[np.ndarray],
        embedding_matrix: np.ndarray,
        max_size: int,
    ) -> Optional[int]:
        best_table = None
        best_similarity = None

        for idx, table in enumerate(tables):
            if idx == current_table or len(table) >= max_size:
                continue
            similarity = self._cosine_similarity(embedding_matrix[player_idx], centroids[idx])
            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_table = idx

        return best_table

    def _compute_centroids(
        self,
        tables: List[List[int]],
        embedding_matrix: np.ndarray,
    ) -> List[np.ndarray]:
        return [self._compute_centroid(table, embedding_matrix) for table in tables]

    def _compute_centroid(
        self,
        table: List[int],
        embedding_matrix: np.ndarray,
    ) -> np.ndarray:
        if not table:
            return np.zeros(embedding_matrix.shape[1])
        centroid = np.mean(embedding_matrix[table], axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return centroid
        return centroid / norm

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9))

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

        mecanicas_predominantes = self._top_n(mecanicas, limit=3)
        temas_predominantes = self._top_n(temas, limit=3)
        return PerfilMesaDTO(
            nivelPredominante=nivel_predominante,
            mecanicasPredominantes=mecanicas_predominantes,
            temasPredominantes=temas_predominantes,
        )

    def _generate_table_profile_string(
        self,
        perfil: PerfilMesaDTO,
    ) -> str:
        parts = [
            f"A group of {perfil.nivelPredominante} board game players.",
        ]
        if perfil.mecanicasPredominantes:
            parts.append(f"They enjoy mechanics such as {', '.join(perfil.mecanicasPredominantes)}.")
        if perfil.temasPredominantes:
            parts.append(f"They prefer themes like {', '.join(perfil.temasPredominantes)}.")

        return " ".join(parts)

    def _map_recommendations_to_dto(
        self,
        recommendations: Sequence[dict],
    ) -> List[JogoRecomendadoDTO]:
        top_results = recommendations[:DEFAULT_TOP_K_RECOMMENDATIONS]
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
