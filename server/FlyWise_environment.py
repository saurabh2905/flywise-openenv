# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FlyWise environment: multi-hop routing over Indian metros with hub-discounted SQLite prices.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from typing import Any, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from FlyWise.graders import compute_route_grader_score
    from FlyWise.load_data import (
        METROS,
        ShortestPathCache,
        default_db_path,
        get_leg_price,
        query_outbound_flights,
    )
    from FlyWise.models import FlywiseAction, FlywiseObservation
except ImportError:
    from graders import compute_route_grader_score  # type: ignore
    from load_data import (  # type: ignore
        METROS,
        ShortestPathCache,
        default_db_path,
        get_leg_price,
        query_outbound_flights,
    )
    from models import FlywiseAction, FlywiseObservation  # type: ignore


def _parse_command(raw: str) -> Tuple[str, Optional[str], Optional[float]]:
    """
    Returns (kind, city_or_none, price_or_none).
    kind in: FETCH_FLIGHTS, MOVE_TO, FINAL_ANSWER, UNKNOWN
    """
    s = raw.strip()
    if not s:
        return "UNKNOWN", None, None
    upper = s.upper()

    if upper == "FETCH_FLIGHTS" or upper.startswith("FETCH_FLIGHTS"):
        return "FETCH_FLIGHTS", None, None

    m = re.match(r"^MOVE_TO\s*\(\s*([A-Za-z]{3})\s*\)\s*$", s, re.IGNORECASE)
    if m:
        return "MOVE_TO", m.group(1).upper(), None
    m = re.match(r"^MOVE_TO\s+([A-Za-z]{3})\s*$", s, re.IGNORECASE)
    if m:
        return "MOVE_TO", m.group(1).upper(), None

    m = re.match(r"^FINAL_ANSWER\s*\(\s*([0-9]+(?:\.[0-9]*)?)\s*\)\s*$", s, re.IGNORECASE)
    if m:
        return "FINAL_ANSWER", None, float(m.group(1))
    m = re.match(r"^FINAL_ANSWER\s+([0-9]+(?:\.[0-9]*)?)\s*$", s, re.IGNORECASE)
    if m:
        return "FINAL_ANSWER", None, float(m.group(1))

    return "UNKNOWN", None, None


class FlywiseEnvironment(Environment):
    """
    Tracks current_city, target_city, total_cost, visited_cities.
    Uses ShortestPathCache for MOVE_TO shaping and FINAL_ANSWER verification.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = str(db_path or os.environ.get("FLYWISE_DB_PATH") or default_db_path())
        self._cache = ShortestPathCache(self._db_path)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._start_city: str = METROS[0]
        self._target_city: str = METROS[1]
        self._current_city: str = METROS[0]
        self._total_cost: float = 0.0
        self._visited: List[str] = []
        self._pending_start: Optional[str] = None
        self._pending_target: Optional[str] = None
        self._task_id: Optional[str] = None

    def configure_episode(self, start_city: str, target_city: str) -> None:
        """Call before reset() to fix the route (used by training / evaluation scripts)."""
        self._pending_start = start_city
        self._pending_target = target_city

    def _ground_truth_cost(self) -> float:
        return self._cache.get_cheapest_cost(self._start_city, self._target_city)

    def _build_obs_json(
        self,
        available_flights: Optional[List[dict]] = None,
        message: str = "",
        *,
        grader_score: Optional[float] = None,
    ) -> str:
        if available_flights is None:
            available_flights = []
        payload: dict = {
            "current_city": self._current_city,
            "target_city": self._target_city,
            "available_flights": available_flights,
            "total_cost": self._total_cost,
            "visited_cities": list(self._visited),
            "message": message,
        }
        if self._task_id:
            payload["task_id"] = self._task_id
        if grader_score is not None:
            payload["grader_score"] = float(grader_score)
        return json.dumps(payload, ensure_ascii=False)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        source_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        start_city: Optional[str] = None,
        target_city: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FlywiseObservation:
        """
        Start an episode. Optional IATA codes (aliases supported):

        - ``source_city`` / ``start_city``
        - ``destination_city`` / ``target_city``

        Passed through OpenEnv HTTP ``POST /reset`` or WebSocket ``reset`` payloads.
        If omitted, uses ``configure_episode`` or random distinct metros.

        Optional ``task_id`` (e.g. ``flywise_route_easy``) is echoed in ``metadata`` for
        hackathon evaluation scripts; routing still comes from the city pair fields.
        """
        if seed is not None:
            random.seed(seed)

        tid = task_id if task_id is not None else kwargs.get("task_id")
        self._task_id = str(tid).strip() if tid else None

        self._state = State(episode_id=str(uuid4()), step_count=0)
        if episode_id is not None:
            self._state = State(episode_id=str(episode_id), step_count=0)

        src = source_city or start_city
        dst = destination_city or target_city
        if src is not None:
            src = str(src).strip().upper()
        if dst is not None:
            dst = str(dst).strip().upper()

        if src is not None and dst is not None:
            if src not in METROS or dst not in METROS or src == dst:
                raise ValueError(
                    f"Invalid route: source={src!r} destination={dst!r}. "
                    f"Use two distinct IATA codes from {list(METROS)}."
                )
            self._start_city = src
            self._target_city = dst
        elif self._pending_start is not None and self._pending_target is not None:
            self._start_city = self._pending_start
            self._target_city = self._pending_target
            self._pending_start = None
            self._pending_target = None
        else:
            self._start_city, self._target_city = random.sample(list(METROS), 2)

        self._current_city = self._start_city
        self._total_cost = 0.0
        self._visited = [self._start_city]

        msg = (
            f"Start at {self._current_city}, reach {self._target_city}. "
            "Find the cheapest total route (multi-hop allowed). "
            "Use FETCH_FLIGHTS, MOVE_TO along valid legs, then FINAL_ANSWER(price). "
            "Terminal success only if price equals your actual path sum and that sum is the global minimum. "
            "OpenEnv step rewards reflect progress toward the optimal remaining cost."
        )
        meta: dict = {"start_city": self._start_city, "target_city": self._target_city}
        if self._task_id:
            meta["task_id"] = self._task_id
        if os.environ.get("FLYWISE_EXPOSE_GT", "").lower() in ("1", "true", "yes"):
            gt = self._ground_truth_cost()
            meta["ground_truth_cheapest"] = gt if math.isfinite(gt) else None
        return FlywiseObservation(
            observation_json=self._build_obs_json(available_flights=[], message=msg),
            done=False,
            reward=0.0,
            metadata=meta,
        )

    def step(self, action: FlywiseAction) -> FlywiseObservation:  # type: ignore[override]
        self._state.step_count += 1
        kind, city, price = _parse_command(action.command)
        reward = 0.0
        done = False
        flights: List[dict] = []
        message = ""
        grader_score: Optional[float] = None

        if kind == "FETCH_FLIGHTS":
            flights = query_outbound_flights(self._db_path, self._current_city)
            message = f"Listed {len(flights)} outbound flights from {self._current_city}."

        elif kind == "MOVE_TO":
            if city is None:
                reward = -0.1
                message = "MOVE_TO requires a valid 3-letter IATA code."
            else:
                prev = self._current_city
                tgt = self._target_city

                if city not in METROS:
                    reward = -0.5
                    message = f"Unknown airport code: {city}."
                elif city == self._current_city:
                    reward = -0.2
                    message = f"Already at {city}."
                else:
                    leg = get_leg_price(self._db_path, self._current_city, city)
                    if leg is None:
                        reward = -0.5
                        message = f"No flight from {self._current_city} to {city}."
                    else:
                        c_prev = self._cache.get_cheapest_cost(prev, tgt)
                        c_next = self._cache.get_cheapest_cost(city, tgt)
                        if math.isfinite(c_prev) and math.isfinite(c_next):
                            improvement = c_prev - c_next
                            if improvement > 0:
                                reward = 0.2
                                message = f"Moved {prev}→{city}; closer to optimal remaining cost."
                            elif improvement < 0:
                                reward = -0.5
                                message = f"Moved {prev}→{city}; remaining optimal cost increased."
                            else:
                                reward = 0.0
                                message = f"Moved {prev}→{city}; no change in optimal remaining cost."
                        else:
                            reward = 0.0
                            message = f"Moved {prev}→{city}."

                        self._current_city = city
                        self._total_cost += leg
                        self._visited.append(city)
                        flights = query_outbound_flights(self._db_path, self._current_city)

        elif kind == "FINAL_ANSWER":
            done = True
            grader_score = compute_route_grader_score(
                start_city=self._start_city,
                target_city=self._target_city,
                visited_cities=list(self._visited),
                total_path_cost=self._total_cost,
                claimed_price=price,
                db_path=self._db_path,
                cache=self._cache,
            )
            if price is None:
                reward = -0.1
                message = "FINAL_ANSWER requires a numeric price."
            else:
                gt = self._ground_truth_cost()
                at_goal = self._current_city == self._target_city
                path_ok = (
                    at_goal
                    and math.isfinite(gt)
                    and math.isclose(self._total_cost, gt, rel_tol=0.0, abs_tol=1e-2)
                )
                claim_matches_path = math.isclose(
                    price, self._total_cost, rel_tol=0.0, abs_tol=1e-2
                )
                claim_matches_opt = math.isfinite(gt) and math.isclose(
                    price, gt, rel_tol=0.0, abs_tol=1e-2
                )
                if at_goal and path_ok and claim_matches_path and claim_matches_opt:
                    reward = 5.0
                    message = (
                        f"Correct: path total {self._total_cost:.2f} equals cheapest route {gt:.2f}."
                    )
                else:
                    reward = -2.0
                    if at_goal and not path_ok and math.isfinite(gt):
                        message = (
                            f"Suboptimal route: actual path cost {self._total_cost:.2f} vs "
                            f"cheapest possible {gt:.2f}. Claimed {price}."
                        )
                    elif at_goal and not claim_matches_path:
                        message = (
                            f"Claimed {price} but legs flown sum to {self._total_cost:.2f} "
                            f"(optimal is {gt if math.isfinite(gt) else 'N/A'})."
                        )
                    else:
                        message = (
                            f"Wrong answer. At {self._current_city}, target {self._target_city}, "
                            f"claimed {price}, optimal {gt if math.isfinite(gt) else 'N/A'}."
                        )

        else:
            reward = -0.1
            message = f"Unrecognized command: {action.command!r}."

        if not flights and kind != "FINAL_ANSWER":
            flights = query_outbound_flights(self._db_path, self._current_city)

        meta_out: dict = {
            "command_kind": kind,
            "step": self._state.step_count,
        }
        if self._task_id:
            meta_out["task_id"] = self._task_id
        if kind == "FINAL_ANSWER" and grader_score is not None:
            meta_out["grader_score"] = float(grader_score)

        obs_grader = float(grader_score) if kind == "FINAL_ANSWER" and grader_score is not None else None
        return FlywiseObservation(
            observation_json=self._build_obs_json(
                available_flights=flights,
                message=message,
                grader_score=obs_grader,
            ),
            done=done,
            reward=reward,
            metadata=meta_out,
        )

    @property
    def state(self) -> State:
        return self._state
