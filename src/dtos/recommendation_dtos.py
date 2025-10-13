from typing import Optional
from pydantic import BaseModel, Field

class HybridRecommendationRequest(BaseModel):
    """
    Input DTO for the hybrid recommendation endpoint.
    """
    query: str = Field(..., description="The text query to search for recommendations.")
    top_k: int = Field(10, description="The final number of recommendations to return after re-ranking.")
    candidate_pool_size: int = Field(100, description="The number of initial candidates to retrieve for re-ranking.")
    semantic_weight: float = Field(0.7, ge=0, le=1, description="The weight for the semantic similarity score (between 0 and 1).")
    popularity_weight: float = Field(0.3, ge=0, le=1, description="The weight for the popularity score (between 0 and 1).")
