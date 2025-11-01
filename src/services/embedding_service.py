from __future__ import annotations

import os
from typing import Iterable, List

from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder


class EmbeddingService:
    """
    Simple wrapper around Ollama's text embedder used for player profiles.
    """

    def __init__(self) -> None:
        model_name = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._embedder = OllamaTextEmbedder(model=model_name, url=base_url)

    def get_embeddings(self, texts: Iterable[str]) -> List[List[float]]:
        """
        Generate embeddings for the provided iterable of strings.
        """
        texts = list(texts)
        embeddings: List[List[float]] = []
        for text in texts:
            result = self._embedder.run(text=text)
            embedding = result.get("embedding")
            if embedding is None:
                raise ValueError("Falha ao gerar embedding para o texto informado.")
            embeddings.append(embedding)
        return embeddings
