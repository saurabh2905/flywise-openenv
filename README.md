---
title: FlyWise — OpenEnv flight routing
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# FlyWise

FlyWise is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment for **multi-hop flight routing** over six Indian metro airports (DEL, BOM, BLR, CCU, HYD, MAA). Leg prices come from a local SQLite graph with hub-style adjustments (same rules as `load_data.py`). The agent must discover the **cheapest total route** from a source to a destination, then report it with `FINAL_ANSWER(price)`.

This repo is structured for **Hugging Face Spaces** (Docker + `openenv` tag), hackathon-style **deterministic graders** in the **open** interval `(0, 1)` (endpoints excluded; see `graders.py`), and a **baseline LLM** path using the official OpenAI API when `OPENAI_API_KEY` is set.

## Try it on the Space (Gradio UI at `/web`)

The left **Quick Start** snippet is a generic OpenEnv template and shows `message="..."` — **FlyWise does not use that.** Use the **`Command`** text box with one line per step:

1. Click **Reset** (starts a new route; read **`target_city`** in the raw JSON).
2. If **`available_flights`** is empty, type **`FETCH_FLIGHTS`** and click **Step**.
3. Type **`MOVE_TO(XXX)`** where `XXX` is a **destination** from `available_flights` (e.g. `MOVE_TO(BOM)`). Click **Step**; repeat until **`current_city`** equals **`target_city`**.
4. Type **`FINAL_ANSWER(<total_cost>)`** using the **`total_cost`** from the observation when you are at the target (must match the cheapest route for full success). Click **Step**.

Watch the **Raw JSON** panel for `observation_json` (it holds cities, flights, costs, and messages). If the UI misbehaves, use **API docs** at `/docs` → `POST /reset` and `POST /step` with body `{"command": "FETCH_FLIGHTS"}`.

## Actions and observations

- **Action** (`FlywiseAction`): one string `command`:
  - `FETCH_FLIGHTS` — list outbound legs from `current_city`
  - `MOVE_TO(XXX)` — fly to a valid neighbor `XXX` (IATA); cost accumulates
  - `FINAL_ANSWER(n)` — end the episode; terminal checks compare `n` to flown sum and global optimum

- **Observation** (`FlywiseObservation`): `observation_json` encodes:

  - `current_city`, `target_city`, `available_flights`, `total_cost`, `visited_cities`, `message`
  - `task_id` when the episode was reset with a task id
  - `grader_score` on the **terminal** observation after `FINAL_ANSWER` (deterministic score strictly in `(0, 1)`)

Step **rewards** from the environment are for RL shaping (not normalized to `[0,1]`). **Official evaluation** for the three tasks uses `grader_score` (and the same function in `graders.py`).

## Tasks (easy → hard)

Three fixed tasks are defined in `flywise_tasks.py` for the default `flywise_flights.db`:

| Task id | Difficulty | Route | Idea |
|--------|------------|-------|------|
| `flywise_route_easy` | easy | BLR → BOM | Optimal route is a **direct** leg |
| `flywise_route_medium` | medium | BLR → HYD | Optimal route uses **two** legs (via DEL on the default synthetic graph) |
| `flywise_route_hard` | hard | MAA → CCU | A **direct** flight exists but is much more expensive than a two-hop path |

If you regenerate the database from CSV, you can derive analogous triples with `pick_tasks_for_database()` (same grader API).

## Deterministic grader

`graders.compute_route_grader_score(...)` returns a float strictly in `(0, 1)` from:

- start/target cities, `visited_cities`, actual path cost, and claimed `FINAL_ANSWER` price  
- ground-truth cheapest cost from the same SQLite graph as the env (`ShortestPathCache`)

The **best** outcome maps near `1 - eps` (default `eps = 0.001` via `FLYWISE_GRADER_OPEN_EPS`), not exactly `1.0`, so submission validators that reject closed-endpoint scores still pass. The server writes the value into the **last** `observation_json` as `grader_score` (OpenEnv’s HTTP serializer omits top-level observation `metadata`, so the JSON field is the portable signal for clients).

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,inference]"
# Ensure DB exists (CSV if present, else synthetic):
python load_data.py
```

Run the server (same Python as the venv):

```bash
python -m uvicorn FlyWise.server.app:app --host 0.0.0.0 --port 8000
# or: server
```

Open `/web` on the server URL for the bundled UI, `/docs` for OpenAPI.

## Baseline inference (`inference.py`)

Remote LLM calls use the **`openai.OpenAI`** client. Env vars align with the hackathon sample:

- `API_BASE_URL` — if unset: `https://api.openai.com/v1` when `OPENAI_API_KEY` is set, else default `https://router.huggingface.co/v1`
- `MODEL_NAME` (or `MODEL`) — model id passed to the API
- `HF_TOKEN` / `API_KEY` / `OPENAI_API_KEY` — API key
- `IMAGE_NAME` or `LOCAL_IMAGE_NAME`, or `--docker-image` — optional; uses `FlywiseEnv.from_docker_image()` instead of `ENV_SERVER_URL`

With a server on `http://localhost:8000`:

```bash
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini
export ENV_SERVER_URL=http://localhost:8000
python inference.py --tasks all
```

**Stdout** (for autograding) uses only: `[START] task=… env=… model=…`, then one `[STEP] …` per `env.step()`, then `[END] success=… steps=… rewards=…` per episode. Debug lines go to **stderr**. Success uses terminal `grader_score >= FLYWISE_SUCCESS_GRADER_THRESHOLD` (default `0.99`, so a near-perfect grader still counts as success).

- `--tasks all` (default): runs the full 3-task hackathon suite (each gets its own START/STEP/END block)
- `--tasks single|easy|medium|hard`: select a subset; for `single`, optional `--source` / `--dest` or `FLYWISE_SOURCE` / `FLYWISE_DEST`
- `FLYWISE_GUIDE_HOPS=0` disables local shortest-path nudging for a purer LLM baseline

Multi-task summaries are printed on stderr, e.g. `[SUMMARY] flywise_route_easy | 1.0000 | 5.200 | True`.

## Docker

From the repo root:

```bash
docker build -t flywise-openenv:latest .
docker run --rm -p 8000:8000 flywise-openenv:latest
```

## Hugging Face Space

From this directory (with `openenv.yaml`):

```bash
openenv validate   # should report ready for multi-mode deployment
openenv push       # after `huggingface-cli login`
```

The Space should stay tagged **`openenv`** (see frontmatter above).

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

## Project layout

```
FlyWise/
├── openenv.yaml
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── flywise_tasks.py      # named tasks
├── graders.py            # [0,1] scores
├── load_data.py          # SQLite + ShortestPathCache
├── inference.py          # LLM + EnvClient loop
├── client.py
├── models.py
├── server/
│   ├── app.py
│   └── FlyWise_environment.py
└── tests/
```

## License

BSD-style — see `LICENSE`.
