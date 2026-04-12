# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Deterministic episode graders for FlyWise hackathon tasks.

Scores are computed on a closed ``[0, 1]`` scale, then mapped to the **open**
interval ``(0, 1)`` (endpoints excluded) so validators that reject exactly
``0.0`` / ``1.0`` still accept deterministic, ordered grades.

Override margin with env ``FLYWISE_GRADER_OPEN_EPS`` (default ``0.005``): returned
value is ``eps + (1 - 2*eps) * raw`` for ``raw`` in ``[0, 1]``.

Set ``FLYWISE_GRADER_DEBUG=1`` to log each ``raw -> mapped`` pair to **stderr** (never stdout).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

try:
    from FlyWise.load_data import ShortestPathCache, default_db_path
except ImportError:
    from load_data import ShortestPathCache, default_db_path  # type: ignore


def _open_interval_eps() -> float:
    # Default 0.005 gives more margin vs validators that treat 2-decimal floats as endpoints.
    return max(1e-6, min(0.49, float(os.environ.get("FLYWISE_GRADER_OPEN_EPS", "0.005"))))


def map_closed_score_to_open_interval(raw: float) -> float:
    """
    Map a closed-interval score ``raw`` in ``[0, 1]`` into ``(0, 1)`` strictly.

    Perfect (raw=1) → ``1 - eps``; failure (raw=0) → ``eps``.
    """
    eps = _open_interval_eps()
    s = max(0.0, min(1.0, float(raw)))
    mapped = eps + (1.0 - 2.0 * eps) * s
    if os.environ.get("FLYWISE_GRADER_DEBUG", "").lower() in ("1", "true", "yes"):
        print(
            f"[FlyWise grader] map_closed_score_to_open_interval raw={s!r} eps={eps!r} mapped={mapped!r}",
            file=sys.stderr,
            flush=True,
        )
    return float(mapped)


def compute_route_grader_score(
    *,
    start_city: str,
    target_city: str,
    visited_cities: Sequence[str],
    total_path_cost: float,
    claimed_price: Optional[float],
    db_path: Optional[Path | str] = None,
    abs_tol: Optional[float] = None,
    cache: Optional[ShortestPathCache] = None,
) -> float:
    """
    Deterministic score in ``(0, 1)`` (strict), derived from a closed ``[0, 1]`` rubric.

    - Best outcome maps near ``1 - eps`` (not exactly ``1.0``).
    - Worst outcome maps near ``eps`` (not exactly ``0.0``).
    """
    tol = (
        float(abs_tol)
        if abs_tol is not None
        else float(os.environ.get("FLYWISE_GRADER_ABS_TOL", "1e-2"))
    )
    path = Path(db_path) if db_path is not None else default_db_path()
    c = cache or ShortestPathCache(path)
    opt = c.get_cheapest_cost(start_city, target_city)

    raw = 0.0
    if not visited_cities:
        raw = 0.0
    else:
        at_target = str(visited_cities[-1]).upper() == str(target_city).upper()
        if not at_target:
            raw = 0.0
        elif not math.isfinite(opt):
            raw = 0.0
        else:
            path_opt = math.isclose(
                float(total_path_cost), float(opt), rel_tol=0.0, abs_tol=tol
            )
            claim = claimed_price
            claim_path = (
                claim is not None
                and math.isclose(
                    float(claim), float(total_path_cost), rel_tol=0.0, abs_tol=tol
                )
            )
            claim_opt = claim is not None and math.isclose(
                float(claim), float(opt), rel_tol=0.0, abs_tol=tol
            )

            if path_opt and claim_path and claim_opt:
                raw = 1.0
            elif path_opt and not claim_path:
                raw = 0.35
            else:
                regret = max(0.0, float(total_path_cost) - float(opt))
                denom = max(float(opt), 1.0)
                quality = max(0.0, 1.0 - min(1.0, regret / denom))
                score = 0.55 * quality
                if claim_path:
                    score = min(1.0, score + 0.15)
                else:
                    score *= 0.8
                raw = float(max(0.0, min(1.0, score)))

    return float(map_closed_score_to_open_interval(raw))
