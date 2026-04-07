# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import math

import pytest

from FlyWise.flywise_tasks import DEFAULT_HACKATHON_TASKS
from FlyWise.graders import compute_route_grader_score
from FlyWise.load_data import ShortestPathCache, default_db_path, get_leg_price


@pytest.fixture(scope="module")
def cache():
    return ShortestPathCache(default_db_path())


def test_grader_perfect_score(cache: ShortestPathCache):
    db = default_db_path()
    for t in DEFAULT_HACKATHON_TASKS:
        opt = cache.get_cheapest_cost(t.source_city, t.target_city)
        assert math.isfinite(opt)
        path = cache.reconstruct_one_shortest_path(t.source_city, t.target_city)
        assert path[0] == t.source_city and path[-1] == t.target_city
        total = 0.0
        for i in range(len(path) - 1):
            p = get_leg_price(db, path[i], path[i + 1])
            assert p is not None
            total += float(p)
        assert math.isclose(total, opt, rel_tol=0.0, abs_tol=1e-2)
        g = compute_route_grader_score(
            start_city=t.source_city,
            target_city=t.target_city,
            visited_cities=path,
            total_path_cost=total,
            claimed_price=opt,
            cache=cache,
        )
        assert g == 1.0


def test_grader_not_at_target(cache: ShortestPathCache):
    t = DEFAULT_HACKATHON_TASKS[0]
    g = compute_route_grader_score(
        start_city=t.source_city,
        target_city=t.target_city,
        visited_cities=[t.source_city],
        total_path_cost=0.0,
        claimed_price=None,
        cache=cache,
    )
    assert g == 0.0


def test_grader_scores_in_unit_interval(cache: ShortestPathCache):
    t = DEFAULT_HACKATHON_TASKS[2]
    opt = cache.get_cheapest_cost(t.source_city, t.target_city)
    path = cache.reconstruct_one_shortest_path(t.source_city, t.target_city)
    db = default_db_path()
    total_bad = opt + 5000.0
    g = compute_route_grader_score(
        start_city=t.source_city,
        target_city=t.target_city,
        visited_cities=path,
        total_path_cost=total_bad,
        claimed_price=total_bad,
        cache=cache,
    )
    assert 0.0 <= g <= 1.0
