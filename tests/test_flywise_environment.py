# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import json
import math

import pytest

from FlyWise.flywise_tasks import DEFAULT_HACKATHON_TASKS
from FlyWise.graders import map_closed_score_to_open_interval
from FlyWise.load_data import ShortestPathCache, default_db_path, get_leg_price
from FlyWise.models import FlywiseAction
from FlyWise.server.FlyWise_environment import FlywiseEnvironment


@pytest.fixture()
def env():
    return FlywiseEnvironment()


def test_reset_task_id_and_observation_json(env: FlywiseEnvironment):
    t = DEFAULT_HACKATHON_TASKS[0]
    obs = env.reset(
        source_city=t.source_city,
        destination_city=t.target_city,
        task_id=t.task_id,
    )
    data = json.loads(obs.observation_json)
    assert data["task_id"] == t.task_id
    assert obs.metadata.get("task_id") == t.task_id


def test_terminal_grader_in_observation_json(env: FlywiseEnvironment):
    cache = ShortestPathCache(default_db_path())
    t = DEFAULT_HACKATHON_TASKS[0]
    path = cache.reconstruct_one_shortest_path(t.source_city, t.target_city)
    env.reset(source_city=t.source_city, destination_city=t.target_city, task_id=t.task_id)
    db = default_db_path()
    for i in range(len(path) - 1):
        dest = path[i + 1]
        obs = env.step(FlywiseAction(command=f"MOVE_TO({dest})"))
        assert not obs.done
    opt = cache.get_cheapest_cost(t.source_city, t.target_city)
    total = sum(
        float(get_leg_price(db, path[j], path[j + 1]) or 0.0)
        for j in range(len(path) - 1)
    )
    assert math.isclose(total, opt, rel_tol=0.0, abs_tol=1e-2)
    final = env.step(FlywiseAction(command=f"FINAL_ANSWER({opt})"))
    assert final.done
    payload = json.loads(final.observation_json)
    expected = map_closed_score_to_open_interval(1.0)
    assert math.isclose(payload.get("grader_score"), expected, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(payload.get("score"), expected, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(final.metadata.get("grader_score"), expected, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(final.metadata.get("score"), expected, rel_tol=0.0, abs_tol=1e-9)
