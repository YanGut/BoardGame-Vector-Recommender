from __future__ import annotations

from typing import Tuple


def normalize_page_params(page: int, per_page: int) -> Tuple[int, int]:
    """
    Ensure pagination parameters are within sensible bounds.
    """
    normalized_page = max(page, 1)
    normalized_per_page = max(per_page, 1)
    return normalized_page, normalized_per_page
