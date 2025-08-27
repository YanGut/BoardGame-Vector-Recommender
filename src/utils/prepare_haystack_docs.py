from tqdm import tqdm # Para barras de progresso

from typing import Optional, Union, Dict

from haystack import Document

def prepare_haystack_documents(boardgames_data: list[dict]) -> list[Document]:
    """
    Prepara uma lista de Documentos Haystack a partir dos dados dos jogos de tabuleiro.
    Cada Documento contém informações relevantes sobre o jogo, como nome, tipo, idade mínima,
    número de jogadores, tempo de jogo, categorias, temas, mecânicas e artistas.
    
    Args:
        boardgames_data (list[dict]): Lista de dicionários contendo os dados dos jogos de tabuleiro.
    
    Returns:
        list[Document]: Lista de Documentos Haystack preparados.
    """
    
    print("Preparando Documentos Haystack...")
    haystack_docs: list[Document] = []

    for game in tqdm(boardgames_data, desc="Convertendo dados para Documentos"):
        content_parts: list[str] = [
            f"Nome do jogo: {game.get('nm_jogo', 'N/A')}.",
            f"Tipo: {game.get('tp_jogo', 'N/A')}.",
            f"Adequado para maiores de {game.get('idade_minima', 'N/A')} anos.",
            f"Pode ser jogado por {game.get('qt_jogadores_min', 'N/A')} a {game.get('qt_jogadores_max', 'N/A')} jogadores.",
            f"Tempo médio de jogo: {game.get('vl_tempo_jogo', 'N/A')} minutos.",
            f"Descrição: {game.get('descricao', 'N/A')}." if game.get('descricao') else "Descrição não disponível.",
            f"O jogo se baseia nas seguintes categorias: {game.get('categorias', 'N/A')}." if game.get('categorias') else "Categorias não disponíveis.",
            f"O jogo aborda os seguintes temas: {game.get('temas', 'N/A')}." if game.get('temas') else "Temas não disponíveis.",
            f"O jogo utiliza as seguintes mecânicas: {game.get('mecanicas', 'N/A')}." if game.get('mecanicas') else "Mecânicas não disponíveis.",
        ]
        
        if game.get('categorias'):
            content_parts.append(f"Categorias: {game['categorias']}.")
        if game.get('temas'):
            content_parts.append(f"Temas: {game['temas']}.")
        if game.get('mecanicas'):
            content_parts.append(f"Mecânicas: {game['mecanicas']}.")
        if game.get('artistas'):
            content_parts.append(f"Artistas: {game['artistas']}.")

        content: str = " ".join(content_parts)

        meta: Dict[str, Optional[Union[str, int]]] = {
            "mysql_id": game.get('id'),
            "title": game.get('nm_jogo', 'N/A'), # 'title' é um campo comum para meta
            "min_age": game.get('idade_minima'),
            "max_players": game.get('qt_jogadores_max'),
            "min_players": game.get('qt_jogadores_min'),
            "play_time_minutes": game.get('vl_tempo_jogo'),
            "thumbnail": game.get('thumb'),
            "game_type": game.get('tp_jogo'),
            "categories_list": game.get('categorias', '').split(', ') if game.get('categorias') else [],
            "themes_list": game.get('temas', '').split(', ') if game.get('temas') else [],
            "mechanics_list": game.get('mecanicas', '').split(', ') if game.get('mecanicas') else [],
            "artists_list": game.get('artistas', '').split(', ') if game.get('artistas') else [],
            "favorite_count": game.get('qt_favorito'),
            "played_count": game.get('qt_jogou'),
            "want_count": game.get('qt_quer'),
            "have_count": game.get('qt_tem'),
            "had_count": game.get('qt_teve')
        }
        
        # Remove chaves com valor None dos metadados para evitar problemas com alguns DocumentStores
        meta_cleaned: Dict[str, str | int] = {k: v for k, v in meta.items() if v is not None}

        haystack_docs.append(Document(content=content, meta=meta_cleaned))

    print(f"Total de {len(haystack_docs)} Documentos Haystack preparados.")
    if haystack_docs:
        print("\nAmostra do primeiro Documento Haystack:")
        print(f"Content: {haystack_docs[0].content[:500]}...")
        print(f"Meta: {haystack_docs[0].meta}")
    return haystack_docs