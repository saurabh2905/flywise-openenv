# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Deterministic episode graders for FlyWise hackathon tasks.

Each grader returns a float in ``[0.0, 1.0]`` given the final trajectory summary
(same ground truth as the environment: SQLite + hub-adjusted leg prices).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

try:
    from FlyWise.load_data import ShortestPathCache, default_db_path
except ImportError:
    from load_data import ShortestPathCache, default_db_path  # type: ignore


def compute_route_grader_score(
    *,
    start_city: str,
    target_city: str,
    visited_cities: Sequence[str],
    total_path_cost: float,
    claimed_price: Optional[float],
    db_path: Optional[Path | str] = None,
    abs_tol: float = 1e-2,
    cache: Optional[ShortestPathCache] = None,
) -> float:
    """
    Deterministic score in ``[0.0, 1.0]``.

    - ``1.0``: at destination, path cost equals global minimum, and claimed price matches both.
    - Partial credit: suboptimal routes score lower; dishonest claims (claim vs flown sum) reduce score.
    """
    path = Path(db_path) if db_path is not None else default_db_path()
    c = cache or ShortestPathCache(path)
    opt = c.get_cheapest_cost(start_city, target_city)

    if not visited_cities:
        return 0.0
    at_target = str(visited_cities[-1]).upper() == str(target_city).upper()
    if not at_target:
        return 0.0
    if not math.isfinite(opt):
        return 0.0

    path_opt = math.isclose(float(total_path_cost), float(opt), rel_tol=0.0, abs_tol=abs_tol)
    claim = claimed_price
    claim_path = (
        claim is not None
        and math.isclose(float(claim), float(total_path_cost), rel_tol=0.0, abs_tol=abs_tol)
    )
    claim_opt = claim is not None and math.isclose(
        float(claim), float(opt), rel_tol=0.0, abs_tol=abs_tol
    )

    if path_opt and claim_path and claim_opt:
        return 1.0
    if path_opt and not claim_path:
        return 0.35
    # Suboptimal path: scale by cost regret; reward honest reporting of flown sum.
    regret = max(0.0, float(total_path_cost) - float(opt))
    denom = max(float(opt), 1.0)
    quality = max(0.0, 1.0 - min(1.0, regret / denom))
    score = 0.55 * quality
    if claim_path:
        score = min(1.0, score + 0.15)
    else:
        score *= 0.8
    return float(max(0.0, min(1.0, score)))
