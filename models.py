# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the FlyWise multi-hop flight environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class FlywiseAction(Action):
    """Single string command: FETCH_FLIGHTS, MOVE_TO(CITY), FINAL_ANSWER(price)."""

    command: str = Field(
        ...,
        description='Command string, e.g. "FETCH_FLIGHTS", "MOVE_TO(DEL)", "FINAL_ANSWER(4521.0)"',
    )


class FlywiseObservation(Observation):
    """JSON snapshot of city, target, available flights, cost, and visit history."""

    observation_json: str = Field(
        default="{}",
        description=(
            "JSON: current_city, target_city, available_flights, total_cost, visited_cities, message; "
            "optional task_id; on terminal steps after FINAL_ANSWER, grader_score and alias score (same value) strictly in (0,1)"
        ),
    )
