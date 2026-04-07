# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run a FlyWise routing episode where **returns come from the OpenEnv engine**
(FlywiseEnvironment step rewards: MOVE_TO shaping vs Floyd–Warshall + FINAL_ANSWER).

**No LLM here.** Policies are ``shortest`` (graph oracle + cache) or ``random``.
For an LLM agent, use ``inference.py``; for learning from rewards, use ``train_grpo.py``.

Examples::

    python run_flywise_rl_episode.py --source CCU --dest BLR --policy shortest
    python run_flywise_rl_episode.py --source DEL --dest MAA --policy random
    python run_flywise_rl_episode.py --source DEL --dest MAA --policy shortest --server http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any, List

try:
    from FlyWise.client import FlywiseEnv
    from FlyWise.load_data import OptimalRouteSummary, ShortestPathCache, default_db_path, query_outbound_flights
    from FlyWise.models import FlywiseAction
    from FlyWise.server.FlyWise_environment import FlywiseEnvironment
except ImportError:
    from client import FlywiseEnv  # type: ignore
    from load_data import OptimalRouteSummary, ShortestPathCache, default_db_path, query_outbound_flights  # type: ignore
    from models import FlywiseAction  # type: ignore
    from server.FlyWise_environment import FlywiseEnvironment  # type: ignore


@dataclass
class EpisodeResult:
    """After one episode: OpenEnv return + fare paid vs optimal path from DB."""

    cumulative_reward: float
    total_paid_fare: float
    executed_path: List[str]
    executed_hops: int
    optimal: OptimalRouteSummary


def _step_reward(result: Any) -> float:
    r = getattr(result, "reward", None)
    if r is None and hasattr(result, "observation"):
        r = getattr(result.observation, "reward", None)
    return float(r or 0.0)


def _parse_observation_json(obs: Any) -> dict:
    raw = getattr(obs, "observation_json", None) or "{}"
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)


def run_episode_local(
    source: str,
    destination: str,
    policy: str,
    max_steps: int,
) -> EpisodeResult:
    db = str(default_db_path())
    env = FlywiseEnvironment(db_path=db)
    cache = ShortestPathCache(db)
    optimal = cache.optimal_route_summary(source, destination)
    env.reset(source_city=source, destination_city=destination)
    total_reward = 0.0
    flights: List[dict] = []

    for _ in range(max_steps):
        current = env._current_city
        target = env._target_city
        if current == target:
            gt = cache.get_cheapest_cost(source, destination)
            cmd = f"FINAL_ANSWER({gt})"
        elif not flights:
            cmd = "FETCH_FLIGHTS"
        elif policy == "random":
            opts = [f["destination"] for f in flights]
            cmd = f"MOVE_TO({random.choice(opts)})"
        else:
            legs = [(str(f["destination"]), float(f["price"])) for f in flights]
            nxt = cache.best_next_airport(current, target, legs)
            cmd = "FETCH_FLIGHTS" if nxt is None else f"MOVE_TO({nxt})"

        obs = env.step(FlywiseAction(command=cmd))
        total_reward += float(obs.reward or 0.0)
        flights = query_outbound_flights(db, env._current_city)
        if obs.done:
            break

    paid = float(env._total_cost)
    path = list(env._visited)
    hops = max(0, len(path) - 1)
    return EpisodeResult(
        cumulative_reward=total_reward,
        total_paid_fare=paid,
        executed_path=path,
        executed_hops=hops,
        optimal=optimal,
    )


def run_episode_server(
    base_url: str,
    source: str,
    destination: str,
    policy: str,
    max_steps: int,
) -> EpisodeResult:
    db = str(default_db_path())
    cache = ShortestPathCache(db)
    optimal = cache.optimal_route_summary(source, destination)
    paid = 0.0
    path: List[str] = []

    with FlywiseEnv(base_url=base_url).sync() as client:
        result = client.reset(source_city=source, destination_city=destination)
        total_reward = _step_reward(result)
        obs = result.observation
        payload = _parse_observation_json(obs)
        flights = payload.get("available_flights") or []
        paid = float(payload.get("total_cost", 0.0))
        path = list(payload.get("visited_cities") or [])

        for _ in range(max_steps):
            current = payload.get("current_city", "")
            target = payload.get("target_city", "")
            if current == target:
                gt = cache.get_cheapest_cost(source, destination)
                cmd = f"FINAL_ANSWER({gt})"
            elif not flights:
                cmd = "FETCH_FLIGHTS"
            elif policy == "random":
                opts = [f["destination"] for f in flights]
                cmd = f"MOVE_TO({random.choice(opts)})"
            else:
                legs = [(str(f["destination"]), float(f["price"])) for f in flights]
                nxt = cache.best_next_airport(current, target, legs)
                cmd = "FETCH_FLIGHTS" if nxt is None else f"MOVE_TO({nxt})"

            result = client.step(FlywiseAction(command=cmd))
            total_reward += _step_reward(result)
            obs = result.observation
            payload = _parse_observation_json(obs)
            flights = payload.get("available_flights") or []
            paid = float(payload.get("total_cost", paid))
            path = list(payload.get("visited_cities") or path)
            if result.done or getattr(obs, "done", False):
                break

    hops = max(0, len(path) - 1)
    return EpisodeResult(
        cumulative_reward=total_reward,
        total_paid_fare=paid,
        executed_path=path,
        executed_hops=hops,
        optimal=optimal,
    )


def _format_path(airports: tuple[str, ...] | List[str]) -> str:
    if not airports:
        return "(none)"
    return " → ".join(airports)


def main() -> None:
    p = argparse.ArgumentParser(description="FlyWise RL-style episode (OpenEnv rewards, no LLM)")
    p.add_argument("--source", required=True, help="Origin IATA (e.g. DEL)")
    p.add_argument("--dest", required=True, help="Destination IATA (e.g. BLR)")
    p.add_argument(
        "--policy",
        choices=("shortest", "random"),
        default="shortest",
        help="shortest = graph oracle (no LLM); random = random legal MOVE_TO (no LLM)",
    )
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument(
        "--server",
        default=None,
        help="If set, use OpenEnv server at this URL (e.g. http://localhost:8000); else local env",
    )
    args = p.parse_args()
    args.source = args.source.strip().upper()
    args.dest = args.dest.strip().upper()

    if args.server:
        report = run_episode_server(
            args.server, args.source, args.dest, args.policy, args.max_steps
        )
        mode = f"server {args.server}"
    else:
        report = run_episode_local(args.source, args.dest, args.policy, args.max_steps)
        mode = "local FlywiseEnvironment"

    o = report.optimal
    print(f"[FlyWise RL] mode={mode} route={args.source}->{args.dest} policy={args.policy}")
    print("[FlyWise RL] LLM: no — this script only uses the flight DB + ShortestPathCache + OpenEnv rewards.")
    print(f"[FlyWise RL] cumulative OpenEnv reward = {report.cumulative_reward:.4f}")
    print(
        f"[FlyWise] Optimal cheapest total fare: {o.cheapest_cost:.2f} "
        f"({o.routing}, {o.num_legs} leg(s))  path: {_format_path(o.airports)}"
    )
    print(
        f"[FlyWise] You actually paid (sum of flown legs): {report.total_paid_fare:.2f} "
        f"— executed {report.executed_hops} hop(s)  path: {_format_path(report.executed_path)}"
    )


if __name__ == "__main__":
    main()
