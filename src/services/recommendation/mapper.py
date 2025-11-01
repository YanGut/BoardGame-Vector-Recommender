from __future__ import annotations

from typing import Any, Dict

from haystack import Document


def _base_game_payload(doc: Document) -> Dict[str, Any]:
    """
    Extract the common payload used across recommendation responses.
    """
    meta = doc.meta or {}
    return {
        "id_mysql": meta.get("mysql_id"),
        "id_chroma": doc.id,
        "description": doc.content,
        "nmJogo": meta.get("title"),
        "thumb": meta.get("thumbnail"),
        "idadeMinima": meta.get("min_age"),
        "qtJogadoresMin": meta.get("min_players"),
        "qtJogadoresMax": meta.get("max_players"),
        "vlTempoJogo": meta.get("play_time_minutes"),
        "anoPublicacao": meta.get("ano_publicacao", 0),
        "anoNacional": meta.get("ano_nacional", 0),
        "tpJogo": meta.get("game_type"),
        "artistas": meta.get("artists_list", []),
        "designers": meta.get("designers_list", []),
        "categorias": meta.get("categories_list", []),
        "mecanicas": meta.get("mechanics_list", []),
        "temas": meta.get("themes_list", []),
        "popularity_score": meta.get("popularity_score", 0),
    }


def document_to_game_dict(doc: Document) -> Dict[str, Any]:
    """
    Map a Haystack Document to the public game representation.
    """
    payload = _base_game_payload(doc)
    payload["score"] = getattr(doc, "score", None)
    return payload


def document_to_hybrid_game_dict(doc: Document, final_score: float) -> Dict[str, Any]:
    """
    Map a Haystack Document to the hybrid recommendation representation.
    """
    payload = _base_game_payload(doc)
    payload["semantic_score"] = getattr(doc, "score", None)
    payload["final_score"] = final_score
    return payload
