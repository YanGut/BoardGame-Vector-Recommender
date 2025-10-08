from typing import List, Optional
from pydantic import BaseModel


class PreferenciasDTO(BaseModel):
    """
    Player's declared preferences.
    These values come from the event registration form.
    """
    mecanicasFavoritas: List[str]
    temasFavoritos: List[str]
    tempoDisponivel: Optional[int]  # in minutes


class JogadorDTO(BaseModel):
    """
    Represents a player participating in the event.
    Contains both identification and declared preferences.
    """
    idUsuario: int
    nome: str
    nivelExperiencia: str  # e.g., "iniciante", "intermediario", "avancado"
    preferencias: Optional[PreferenciasDTO]


class RestricoesDTO(BaseModel):
    """
    Constraints defined by the organizer when creating groups.
    They control the minimum/maximum table size and other rules.
    """
    tamanhoMinimoMesa: int
    tamanhoMaximoMesa: int
    duracaoMaxima: Optional[int]  # in minutes


class CriarGruposRequest(BaseModel):
    """
    Input DTO for the endpoint that generates groups (tables).
    The organizer specifies the event, number of tables, constraints,
    and the list of players who registered.
    """
    eventoId: int
    quantidadeMesas: int
    restricoes: Optional[RestricoesDTO]
    jogadores: List[JogadorDTO]


class JogoRecomendadoDTO(BaseModel):
    """
    Represents a recommended board game for a group (table).
    Includes metadata from ChromaDB/ALS pipelines.
    """
    idJogo: int
    nome: str
    similaridade: float  # similarity score between 0 and 1
    thumbnail: Optional[str]


class PerfilMesaDTO(BaseModel):
    """
    Summary profile of a group of players.
    Useful for the organizer to understand the characteristics
    of the generated table.
    """
    nivelPredominante: str
    mecanicasPredominantes: List[str]
    temasPredominantes: List[str]
    tempoMedioDisponivel: Optional[int]


class MesaDTO(BaseModel):
    """
    A single table (group of players).
    Contains the assigned players, the summarized profile,
    and the recommended games for the group.
    """
    mesaId: int
    jogadores: List[JogadorDTO]
    perfilMesa: PerfilMesaDTO
    jogosRecomendados: List[JogoRecomendadoDTO]


class CriarGruposResponse(BaseModel):
    """
    Output DTO for the group creation process.
    The response includes the event ID and the list of generated tables.
    """
    eventoId: int
    mesas: List[MesaDTO]
