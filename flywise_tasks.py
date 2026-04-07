# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Named FlyWise hackathon tasks (easy → hard) over the bundled SQLite graph.

Pairs are fixed for the default ``flywise_flights.db`` (synthetic seed-42 graph or
regenerated CSV build). Difficulty follows human-facing routing challenge, not a
different reward function: the same :mod:`graders` apply to all tasks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Tuple

try:
    from FlyWise.load_data import ShortestPathCache, default_db_path, get_leg_price
except ImportError:
    from load_data import ShortestPathCache, default_db_path, get_leg_price  # type: ignore

Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class FlywiseTaskSpec:
    """A single evaluation episode specification."""

    task_id: str
    name: str
    difficulty: Difficulty
    description: str
    source_city: str
    target_city: str


# Default triple for synthetic DB (see repo README / tests). Optimal hops: 1, 2, 2 with a
# very expensive direct edge on the hard pair (MAA→CCU direct >> MAA→BOM→CCU).
DEFAULT_TASK_BLR_BOM_EASY = FlywiseTaskSpec(
    task_id="flywise_route_easy",
    name="Direct cheapest hop",
    difficulty="easy",
    description="BLR→BOM where the cheapest route is a single direct flight.",
    source_city="BLR",
    target_city="BOM",
)
DEFAULT_TASK_BLR_HYD_MEDIUM = FlywiseTaskSpec(
    task_id="flywise_route_medium",
    name="Two-hop metro routing",
    difficulty="medium",
    description="BLR→HYD where the cheapest route uses two legs (via DEL on the default graph).",
    source_city="BLR",
    target_city="HYD",
)
DEFAULT_TASK_MAA_CCU_HARD = FlywiseTaskSpec(
    task_id="flywise_route_hard",
    name="Misleading direct leg",
    difficulty="hard",
    description="MAA→CCU where a direct flight exists but is far more expensive than a two-hop path.",
    source_city="MAA",
    target_city="CCU",
)

DEFAULT_HACKATHON_TASKS: Tuple[FlywiseTaskSpec, ...] = (
    DEFAULT_TASK_BLR_BOM_EASY,
    DEFAULT_TASK_BLR_HYD_MEDIUM,
    DEFAULT_TASK_MAA_CCU_HARD,
)


def task_by_id(task_id: str) -> FlywiseTaskSpec:
    tid = task_id.strip().lower()
    for t in DEFAULT_HACKATHON_TASKS:
        if t.task_id.lower() == tid:
            return t
    raise KeyError(f"Unknown task_id={task_id!r}; known: {[x.task_id for x in DEFAULT_HACKATHON_TASKS]}")


def tasks_by_difficulty(difficulty: Difficulty) -> Tuple[FlywiseTaskSpec, ...]:
    return tuple(t for t in DEFAULT_HACKATHON_TASKS if t.difficulty == difficulty)


def _opt_summary(cache: ShortestPathCache, src: str, dest: str):
    return cache.optimal_route_summary(src, dest)


def pick_tasks_for_database(
    db_path=None,
    *,
    cache: Optional[ShortestPathCache] = None,
) -> Tuple[FlywiseTaskSpec, FlywiseTaskSpec, FlywiseTaskSpec]:
    """
    Derive (easy, medium, hard) task specs from an arbitrary SQLite graph.

    - easy: lexicographically first pair whose optimal route is a single leg
    - medium: first pair whose optimal route has exactly two legs
    - hard: among two-hop-optimal pairs, maximize (direct_leg_price / optimal_cost)
    """
    try:
        from FlyWise.load_data import METROS as metros
    except ImportError:
        from load_data import METROS as metros  # type: ignore

    c = cache or ShortestPathCache(db_path or default_db_path())
    dbp: Path = getattr(c, "_db_path", None) or default_db_path()

    easy_pair: Optional[Tuple[str, str]] = None
    medium_pair: Optional[Tuple[str, str]] = None
    hard_pair: Optional[Tuple[str, str]] = None
    hard_ratio = -1.0

    pairs: Iterable[Tuple[str, str]] = (
        (a, b) for a in metros for b in metros if a != b
    )
    for a, b in sorted(pairs):
        s = _opt_summary(c, a, b)
        if not math.isfinite(s.cheapest_cost):
            continue
        if s.num_legs == 1 and easy_pair is None:
            easy_pair = (a, b)
        if s.num_legs == 2 and medium_pair is None:
            medium_pair = (a, b)
        if s.num_legs == 2:
            leg = get_leg_price(dbp, a, b)
            if leg is not None and s.cheapest_cost > 0:
                ratio = float(leg) / float(s.cheapest_cost)
                if ratio > hard_ratio:
                    hard_ratio = ratio
                    hard_pair = (a, b)

    if easy_pair is None or medium_pair is None:
        return DEFAULT_HACKATHON_TASKS

    if hard_pair is None:
        hard_pair = medium_pair

    easy = FlywiseTaskSpec(
        task_id="flywise_route_easy",
        name="Direct cheapest hop",
        difficulty="easy",
        description=f"{easy_pair[0]}→{easy_pair[1]} with a one-leg optimal route on this graph.",
        source_city=easy_pair[0],
        target_city=easy_pair[1],
    )
    medium = FlywiseTaskSpec(
        task_id="flywise_route_medium",
        name="Two-hop metro routing",
        difficulty="medium",
        description=f"{medium_pair[0]}→{medium_pair[1]} with a two-leg optimal route on this graph.",
        source_city=medium_pair[0],
        target_city=medium_pair[1],
    )
    hd, hr = hard_pair
    hard = FlywiseTaskSpec(
        task_id="flywise_route_hard",
        name="Misleading direct leg",
        difficulty="hard",
        description=(
            f"{hd}→{hr} where a direct leg exists; on this graph its price/optimal-cost ratio is "
            f"{hard_ratio:.3f} (highest among two-hop-optimal pairs)."
        ),
        source_city=hd,
        target_city=hr,
    )
    return (easy, medium, hard)


def task_catalog_dict() -> Dict[str, FlywiseTaskSpec]:
    return {t.task_id: t for t in DEFAULT_HACKATHON_TASKS}
