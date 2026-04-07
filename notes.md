# FlyWise — OpenEnv base project (notes)

## What OpenEnv is doing here

**OpenEnv** (Meta’s `openenv-core`) is the **plumbing** for a **remote RL-style environment**: same process ideas as Gym (reset, step, observation, reward, done), but the env can run **in another process or container** and you talk to it over **HTTP + WebSocket**.

You implement three things; the framework wires the server and client protocol:

| Piece | Role |
|--------|------|
| **`models.py`** | Pydantic **`Action`** / **`Observation`** subclasses (`FlywiseAction`, `FlywiseObservation`) — schema + JSON for the API. |
| **`server/FlyWise_environment.py`** | **`Environment`** subclass: real logic in **`reset()`**, **`step(action)`**, and **`state`**. |
| **`client.py`** | **`EnvClient`** subclass: turns actions into wire format and parses **`StepResult`** / **`State`** from the server. |
| **`server/app.py`** | **`create_app(...)`** — builds the FastAPI app with **`POST /reset`**, **`POST /step`**, **`GET /state`**, **`GET /schema`**, **`WS /ws`**. |

**Package layout:** `pyproject.toml` maps the repo root to the **`FlyWise`** package and **`server/`** to **`FlyWise.server`**, with a **`server`** console script pointing at **`Flywise.server.app:main`**.

**Dependencies:** `openenv-core[core]>=0.2.1` (FastAPI server, client types, etc.).

**Docker:** `Dockerfile` targets **`ghcr.io/meta-pytorch/openenv-base`** and copies this env into `/app/env` so judges/deployments can run the same image the OpenEnv tooling expects.

---

## Default “toy” environment (what you replace for the hackathon)

`FlywiseEnvironment` is intentionally minimal:

- **`reset`** → fixed “ready” message, new `episode_id`, `step_count` 0.
- **`step`** → echoes `action.message`, reward `0.1 * len(message)`, `done=False`, small **metadata** (original message, step index).
- **`SUPPORTS_CONCURRENT_SESSIONS = True`** — compatible with multiple WebSocket sessions when the server is configured for per-session env instances.

So the **base project** proves **server + client + Docker** work; your hackathon story is to **swap this logic** (e.g. flight search, DB, multi-hop pricing) while keeping the same OpenEnv contract.

---

## Public Python API

`FlyWise/__init__.py` exports **`FlywiseEnv`**, **`FlywiseAction`**, **`FlywiseObservation`** — typical usage: connect with **`FlywiseEnv(base_url=...)`** (or Docker helper if you use it), then **`reset()`** / **`step(FlywiseAction(...))`**.

---

## Not from the OpenEnv scaffold

**`data_generation.py`** — loads a CSV, maps cities to IATA, builds hub-style prices, writes **`vayu_flights.db`**. It is **project-specific** data prep toward a real FlyWise domain; it may not yet be wired into `FlywiseEnvironment` or `app.py`.

---

## Quick “where to edit” for the hackathon

1. **Domain logic** → `server/FlyWise_environment.py` (`reset` / `step`, state).
2. **Action/observation shape** → `models.py` (+ sync **`client.py`** parsers and any server validation if you add fields).
3. **Concurrency** → `max_concurrent_envs` in `server/app.py` if you need more than one simultaneous session.
4. **Extra deps** (pandas, sqlite drivers, etc.) → `pyproject.toml` / `server/requirements.txt` so Docker and local installs stay aligned.
