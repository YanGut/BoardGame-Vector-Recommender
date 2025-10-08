from typing import List
from src.dtos.group_dtos import (
    CriarGruposRequest,
    CriarGruposResponse,
    MesaDTO,
    JogadorDTO,
    PerfilMesaDTO,
    JogoRecomendadoDTO,
    PreferenciasDTO
)


class GroupService:
    def create_groups(self, request_data: CriarGruposRequest) -> CriarGruposResponse:
        """
        Mocks the creation of player groups based on the request data.
        In a real implementation, this would involve clustering algorithms,
        player preference matching, and game recommendations.
        """
        mesas: List[MesaDTO] = []
        players_per_table = len(request_data.jogadores) // request_data.quantidadeMesas
        
        # Simple distribution of players for mocking purposes
        current_player_index = 0
        for i in range(request_data.quantidadeMesas):
            assigned_players: List[JogadorDTO] = []
            for _ in range(players_per_table):
                if current_player_index < len(request_data.jogadores):
                    assigned_players.append(request_data.jogadores[current_player_index])
                    current_player_index += 1
            
            # Add remaining players to the last table if any
            if i == request_data.quantidadeMesas - 1 and current_player_index < len(request_data.jogadores):
                while current_player_index < len(request_data.jogadores):
                    assigned_players.append(request_data.jogadores[current_player_index])
                    current_player_index += 1

            # Mock PerfilMesaDTO
            mock_perfil = PerfilMesaDTO(
                nivelPredominante="intermediario",
                mecanicasPredominantes=["eurogame", "deck-building"],
                temasPredominantes=["fantasia", "ficcao-cientifica"],
                tempoMedioDisponivel=120
            )

            # Mock JogoRecomendadoDTO
            mock_jogos_recomendados = [
                JogoRecomendadoDTO(
                    idJogo=101,
                    nome="Catan",
                    similaridade=0.85,
                    thumbnail="https://example.com/catan.jpg"
                ),
                JogoRecomendadoDTO(
                    idJogo=102,
                    nome="Ticket to Ride",
                    similaridade=0.78,
                    thumbnail="https://example.com/ticket_to_ride.jpg"
                )
            ]

            mesas.append(
                MesaDTO(
                    mesaId=i + 1,
                    jogadores=assigned_players,
                    perfilMesa=mock_perfil,
                    jogosRecomendados=mock_jogos_recomendados
                )
            )

        return CriarGruposResponse(
            eventoId=request_data.eventoId,
            mesas=mesas
        )


group_service_instance = GroupService()
