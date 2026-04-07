# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FlyWise data layer: SQLite flight graph, hub-adjusted pricing, Floyd–Warshall cache.
"""

from __future__ import annotations

import math
import os
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

METROS: Tuple[str, ...] = ("DEL", "BOM", "BLR", "CCU", "HYD", "MAA")
HUBS = frozenset({"DEL", "BOM"})

IATA_MAP = {
    "Delhi": "DEL",
    "Mumbai": "BOM",
    "Bangalore": "BLR",
    "Kolkata": "CCU",
    "Hyderabad": "HYD",
    "Chennai": "MAA",
}

DEFAULT_DB_NAME = "flywise_flights.db"


def project_root() -> Path:
    return Path(__file__).resolve().parent


def default_db_path() -> Path:
    """
    Resolve the SQLite path consistently for both layouts:

    - **Dev / repo:** ``load_data.py`` at repo root → DB next to this file.
    - **Docker / HF Space:** ``uv sync`` installs code under ``site-packages/FlyWise/``,
      but ``load_data.py`` during build writes ``/app/env/flywise_flights.db``. Runtime
      must find that file (via ``FLYWISE_DB_PATH`` or ``cwd``), not an empty DB under
      ``site-packages``.
    """
    env = os.environ.get("FLYWISE_DB_PATH")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve().parent
    pkg_candidate = here / DEFAULT_DB_NAME

    cwd = Path.cwd().resolve()
    cwd_candidate = cwd / DEFAULT_DB_NAME

    for candidate in (pkg_candidate, cwd_candidate):
        if candidate.is_file():
            return candidate

    for parent in cwd.parents:
        root_db = parent / DEFAULT_DB_NAME
        if root_db.is_file():
            return root_db
        if (parent / "openenv.yaml").is_file():
            return (parent / DEFAULT_DB_NAME).resolve()

    for parent in here.parents:
        if (parent / "openenv.yaml").is_file():
            return (parent / DEFAULT_DB_NAME).resolve()

    return pkg_candidate


def _hub_adjust_price(src: str, dest: str, base_price: float) -> float:
    if src not in HUBS and dest not in HUBS:
        return float(base_price) * 1.5
    return float(base_price) * 0.7


def setup_flywise_database(csv_path: str | Path, db_path: str | Path | None = None) -> Path:
    """
    Load Kaggle-style Clean_Dataset.csv, apply hub discount logic, write SQLite.
    """
    db_path = Path(db_path) if db_path else default_db_path()
    df = pd.read_csv(csv_path)
    df["src"] = df["source_city"].map(IATA_MAP)
    df["dest"] = df["destination_city"].map(IATA_MAP)
    df = df.dropna(subset=["src", "dest"])

    flights = df[df["class"] == "Economy"].groupby(["src", "dest"], as_index=False)["price"].min()

    flights["price"] = flights.apply(
        lambda row: _hub_adjust_price(str(row["src"]), str(row["dest"]), float(row["price"])),
        axis=1,
    )

    conn = sqlite3.connect(str(db_path))
    flights.to_sql("flights", conn, if_exists="replace", index=False)
    conn.close()
    print(f"FlyWise database initialized: {db_path}")
    return db_path


def seed_synthetic_flywise_database(db_path: str | Path | None = None) -> Path:
    """
    Build a minimal flight graph over METROS when no CSV is available (CI / Docker).
    """
    db_path = Path(db_path) if db_path else default_db_path()
    rows: List[Dict[str, object]] = []
    import random

    rng = random.Random(42)
    for a in METROS:
        for b in METROS:
            if a == b:
                continue
            base = 2000.0 + rng.random() * 4000.0
            price = _hub_adjust_price(a, b, base)
            rows.append({"src": a, "dest": b, "price": round(price, 2)})

    df = pd.DataFrame(rows)
    conn = sqlite3.connect(str(db_path))
    df.to_sql("flights", conn, if_exists="replace", index=False)
    conn.close()
    print(f"FlyWise synthetic database initialized: {db_path}")
    return db_path


@dataclass(frozen=True)
class OptimalRouteSummary:
    """Cheapest path from SQLite + hub-adjusted prices (same ground truth as the env)."""

    cheapest_cost: float
    airports: Tuple[str, ...]
    num_legs: int
    is_direct: bool
    routing: str


def load_edges_from_sqlite(db_path: str | Path) -> Dict[Tuple[str, str], float]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("SELECT src, dest, MIN(price) FROM flights GROUP BY src, dest")
    edges: Dict[Tuple[str, str], float] = {}
    for src, dest, price in cur.fetchall():
        k = (str(src), str(dest))
        edges[k] = min(edges.get(k, math.inf), float(price))
    conn.close()
    return edges


class ShortestPathCache:
    """
    All-pairs shortest path (Floyd–Warshall) over directed flight edges from SQLite.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path) if db_path else default_db_path()
        self._index: Dict[str, int] = {c: i for i, c in enumerate(METROS)}
        n = len(METROS)
        dist = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0.0

        edges = load_edges_from_sqlite(self._db_path)
        for (src, dest), price in edges.items():
            if src in self._index and dest in self._index:
                i, j = self._index[src], self._index[dest]
                dist[i][j] = min(dist[i][j], price)

        for k in range(n):
            for i in range(n):
                if dist[i][k] == math.inf:
                    continue
                for j in range(n):
                    if dist[k][j] == math.inf:
                        continue
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        self._dist = dist
        self._edges: Dict[Tuple[str, str], float] = {
            k: v for k, v in edges.items() if k[0] in self._index and k[1] in self._index
        }

    def _outbound_leg_prices(self, u: str) -> List[Tuple[str, float]]:
        return [(b, w) for (a, b), w in self._edges.items() if a == u]

    def get_cheapest_cost(self, src: str, dest: str) -> float:
        """Ground-truth cheapest path cost (sum of leg prices) from src to dest."""
        if src not in self._index or dest not in self._index:
            return math.inf
        i, j = self._index[src], self._index[dest]
        return self._dist[i][j]

    def best_next_airport(
        self, current: str, target: str, leg_options: List[Tuple[str, float]]
    ) -> Optional[str]:
        """
        One step toward a minimum-cost path: minimize leg_price + cheapest_cost(dest, target).
        """
        best_dest: Optional[str] = None
        best_score = math.inf
        for dest, leg_price in leg_options:
            rest = self.get_cheapest_cost(dest, target)
            if not math.isfinite(rest):
                continue
            score = float(leg_price) + rest
            if score < best_score:
                best_score = score
                best_dest = dest
        return best_dest

    def reconstruct_one_shortest_path(self, src: str, dest: str) -> List[str]:
        """
        Recover one minimum-cost route (airport sequence) consistent with Floyd–Warshall costs.
        Tie-break: lexicographically smallest next airport.
        """
        if src not in self._index or dest not in self._index:
            return []
        if src == dest:
            return [src]
        if not math.isfinite(self.get_cheapest_cost(src, dest)):
            return []
        path: List[str] = [src]
        for _ in range(len(METROS) + 2):
            u = path[-1]
            if u == dest:
                break
            du_dest = self.get_cheapest_cost(u, dest)
            candidates: List[str] = []
            for v, w in self._outbound_leg_prices(u):
                rest = self.get_cheapest_cost(v, dest)
                if math.isfinite(rest) and math.isclose(
                    w + rest, du_dest, rel_tol=0.0, abs_tol=1e-2
                ):
                    candidates.append(v)
            if not candidates:
                break
            path.append(min(candidates))
        return path

    def optimal_route_summary(self, src: str, dest: str) -> OptimalRouteSummary:
        path = self.reconstruct_one_shortest_path(src, dest)
        cost = self.get_cheapest_cost(src, dest)
        legs = max(0, len(path) - 1) if path else 0
        ok = bool(path) and path[-1] == dest
        if not ok or not math.isfinite(cost):
            return OptimalRouteSummary(
                cheapest_cost=cost,
                airports=tuple(path),
                num_legs=0,
                is_direct=False,
                routing="unreachable",
            )
        return OptimalRouteSummary(
            cheapest_cost=cost,
            airports=tuple(path),
            num_legs=legs,
            is_direct=(legs == 1),
            routing="direct" if legs == 1 else "multi_hop",
        )


_cache: Optional[ShortestPathCache] = None


def get_default_cache(db_path: str | Path | None = None) -> ShortestPathCache:
    global _cache
    if _cache is None:
        _cache = ShortestPathCache(db_path)
    return _cache


def reset_default_cache(db_path: str | Path | None = None) -> ShortestPathCache:
    global _cache
    _cache = ShortestPathCache(db_path)
    return _cache


def query_outbound_flights(db_path: str | Path, city: str) -> List[Dict[str, float]]:
    """All direct destinations from city with minimum economy (stored) price."""
    path = str(db_path)
    conn = sqlite3.connect(path)
    cur = conn.execute(
        "SELECT dest, MIN(price) as price FROM flights WHERE src = ? GROUP BY dest ORDER BY dest",
        (city,),
    )
    rows = [{"destination": str(r[0]), "price": float(r[1])} for r in cur.fetchall()]
    conn.close()
    return rows


def get_leg_price(db_path: str | Path, src: str, dest: str) -> Optional[float]:
    path = str(db_path)
    conn = sqlite3.connect(path)
    cur = conn.execute(
        "SELECT MIN(price) FROM flights WHERE src = ? AND dest = ?",
        (src, dest),
    )
    row = cur.fetchone()
    conn.close()
    if row is None or row[0] is None:
        return None
    return float(row[0])


if __name__ == "__main__":
    import sys

    root = project_root()
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "Clean_Dataset.csv"
    if csv.is_file():
        setup_flywise_database(csv)
    else:
        print(f"No CSV at {csv}; generating synthetic graph.")
        seed_synthetic_flywise_database()
