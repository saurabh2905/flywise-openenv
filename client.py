# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flywise Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import FlywiseAction, FlywiseObservation
except ImportError:
    from models import FlywiseAction, FlywiseObservation  # type: ignore


class FlywiseEnv(
    EnvClient[FlywiseAction, FlywiseObservation, State]
):
    """
    Client for the Flywise Environment (multi-hop flight routing).

    Maintains a WebSocket session to the server for low-latency stepping.

    Example:
        >>> with FlywiseEnv(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset(source_city="DEL", destination_city="BLR")
        ...     result = client.step(FlywiseAction(command="FETCH_FLIGHTS"))
    """

    def _step_payload(self, action: FlywiseAction) -> Dict:
        return {"command": action.command}

    def _parse_result(self, payload: Dict) -> StepResult[FlywiseObservation]:
        obs_data = payload.get("observation", {})
        observation = FlywiseObservation(
            observation_json=obs_data.get("observation_json", "{}"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
