# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GRPO training for FlyWise using TRL (Group Relative Policy Optimization).

Install training deps: pip install -e ".[train]"
Run (single GPU):  accelerate launch train_grpo.py
Or:              python train_grpo.py   # may OOM on large models without accelerate

By default this trains **LoRA adapters** only (base model weights stay frozen). Use ``--full-finetune``
to update all parameters. Output dir contains adapter weights + config; load with PEFT on top of the base model id.

Apple Silicon (M1/M2/M3/M4): PyTorch uses the Metal backend (``mps``), not NVIDIA CUDA.
This script enables MPS when available so training uses the Mac GPU.

Verbose logging: ``python train_grpo.py --verbose`` (or ``-v``) prints each reward evaluation:
generated text, parsed commands, per-step env reward and message, format + scaled combined score,
and GRPO trainer metrics each optimizer step.

Database: leg prices and env dynamics come from SQLite (default ``flywise_flights.db`` next to
``load_data.py``). Override with ``--db-path`` or env ``FLYWISE_DB_PATH``. Startup logs print the
resolved path and ``flights`` row count.

Learning signal: each optimizer step also logs ``mean_reward`` for the GRPO sample group. After
training, a **run summary** block prints config, wall time, reward first/last/delta, artifact path,
and suggested next steps.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import textwrap
import time
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from transformers.utils import is_torch_bf16_gpu_available

try:
    from FlyWise.load_data import METROS, default_db_path, seed_synthetic_flywise_database
    from FlyWise.models import FlywiseAction
    from FlyWise.server.FlyWise_environment import FlywiseEnvironment
except ImportError:
    from load_data import METROS, default_db_path, seed_synthetic_flywise_database  # type: ignore
    from models import FlywiseAction  # type: ignore
    from server.FlyWise_environment import FlywiseEnvironment  # type: ignore

THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.IGNORECASE | re.DOTALL)
ACTION_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Set True via ``--verbose`` in ``main()``: detailed reward rollout logs.
_TRAIN_VERBOSE: bool = False
_REWARD_BATCH_SEQ: int = 0
# Set in ``main()`` so reward rollouts and ``make_dataset`` use the same SQLite file as the server.
_TRAIN_DB_PATH: str | None = None


def _db_path_for_training() -> str:
    if _TRAIN_DB_PATH is not None:
        return _TRAIN_DB_PATH
    return str(default_db_path().resolve())


def _resolve_train_db_path(cli_path: str | None) -> str:
    """Prefer ``--db-path``, then ``FLYWISE_DB_PATH``, then ``load_data.default_db_path()``."""
    if cli_path:
        return str(Path(cli_path).expanduser().resolve())
    envp = os.environ.get("FLYWISE_DB_PATH")
    if envp:
        return str(Path(envp).expanduser().resolve())
    return str(default_db_path().resolve())


def _log_database_proof(db_path: str) -> None:
    """Print absolute path and row stats so logs prove training uses your ``flywise_flights.db``."""
    p = Path(db_path)
    print(
        "[train_grpo] SQLite: all FlyWiseEnvironment rollouts (leg prices, FETCH, MOVE_TO) "
        "read this database:",
        flush=True,
    )
    print(f"         path: {p.resolve()}", flush=True)
    if not p.is_file():
        print(
            "         (file missing — run load_data / seed, or point --db-path to your DB)",
            flush=True,
        )
        return
    try:
        conn = sqlite3.connect(str(p))
        try:
            n = int(conn.execute("SELECT COUNT(*) FROM flights").fetchone()[0])
            row = conn.execute("SELECT MIN(price), MAX(price) FROM flights").fetchone()
            lo, hi = float(row[0]), float(row[1])
            print(f"         table flights: {n} rows | price min/max = {lo:.2f} / {hi:.2f}", flush=True)
        finally:
            conn.close()
    except Exception as e:
        print(f"         (could not query DB: {e})", flush=True)


def _vlog(msg: str) -> None:
    if _TRAIN_VERBOSE:
        print(msg, flush=True)


def build_prompt(start: str, target: str, observation_json: str) -> str:
    return (
        f"You route flights between Indian metros (IATA): {', '.join(METROS)}. "
        "Goal: reach the destination spending as little total fare as the task allows; "
        "use the observation (prices, cities, messages) and learn what works from experience.\n\n"
        f"Origin: {start}. Destination: {target}.\n"
        f"Current observation (JSON): {observation_json}\n\n"
        "Reply with exactly one <thought>...</thought> block, then one <action>...</action> block. "
        "Inside <action>, one line only — a valid command:\n"
        "  FETCH_FLIGHTS\n"
        "  MOVE_TO(XXX)\n"
        "  FINAL_ANSWER(<number>)\n"
    )


def make_dataset(num_duplicates: int = 3) -> Dataset:
    rows: List[dict] = []
    db = _db_path_for_training()
    if not os.path.isfile(db):
        seed_synthetic_flywise_database(db)

    for _ in range(num_duplicates):
        for a in METROS:
            for b in METROS:
                if a == b:
                    continue
                env = FlywiseEnvironment(db_path=db)
                obs = env.reset(source_city=a, destination_city=b)
                rows.append(
                    {
                        "prompt": build_prompt(a, b, obs.observation_json),
                        "start_city": a,
                        "target_city": b,
                    }
                )
    return Dataset.from_list(rows)


def _parse_tags(completion: str) -> tuple[str, str]:
    t = THOUGHT_RE.search(completion) or None
    a = ACTION_RE.search(completion) or None
    thought = t.group(1).strip() if t else ""
    action = a.group(1).strip() if a else ""
    return thought, action


def _format_score(completion: str) -> float:
    thought, action = _parse_tags(completion)
    return 0.5 if (thought and action) else 0.0


def _env_rollout(
    completion: str,
    start: str,
    target: str,
    db: str,
) -> tuple[float, float, list[tuple[str, float, str]]]:
    """
    Roll out parsed <action> lines in the FlyWise env.

    Returns (raw_sum_of_step_rewards, raw_sum / 5.0, per_step (command, reward, message_snip)).
    """
    env = FlywiseEnvironment(db_path=db)
    env.reset(source_city=start, destination_city=target)
    _, actions_blob = _parse_tags(completion)
    raw_total = 0.0
    step_logs: list[tuple[str, float, str]] = []
    for line in actions_blob.splitlines():
        cmd = line.strip()
        if not cmd:
            continue
        obs = env.step(FlywiseAction(command=cmd))
        r = float(obs.reward or 0.0)
        raw_total += r
        msg_snip = ""
        try:
            payload = json.loads(obs.observation_json)
            msg_snip = (payload.get("message") or "")[:200]
        except (json.JSONDecodeError, TypeError):
            pass
        step_logs.append((cmd, r, msg_snip))
        if obs.done:
            break
    scaled = raw_total / 5.0
    return raw_total, scaled, step_logs


def format_reward(prompts, completions, **kwargs) -> List[float]:
    return [_format_score(c) for c in completions]


def environment_reward(prompts, completions, start_city, target_city, **kwargs) -> List[float]:
    db = _db_path_for_training()
    out: List[float] = []
    for c, s, t in zip(completions, start_city, target_city):
        _, scaled, _ = _env_rollout(c, s, t, db)
        out.append(scaled)
    return out


def combined_flywise_reward(prompts, completions, start_city, target_city, **kwargs) -> List[float]:
    """
    Format score (parsable <thought>/<action>) plus scaled env return (sum of env step rewards / 5).
    When verbose, logs each generated sample: route, format reward, each env command + reward + message.
    """
    global _REWARD_BATCH_SEQ
    _REWARD_BATCH_SEQ += 1
    batch_id = _REWARD_BATCH_SEQ
    db = _db_path_for_training()
    out: List[float] = []

    for i, (c, s, t) in enumerate(zip(completions, start_city, target_city)):
        fmt = _format_score(c)
        thought, action_blob = _parse_tags(c)
        raw_total, scaled, step_logs = _env_rollout(c, s, t, db)
        combined = fmt + scaled
        out.append(combined)

        if _TRAIN_VERBOSE:
            thought_preview = textwrap.shorten(
                thought.replace("\n", " "), width=220, placeholder="…"
            )
            comp_preview = textwrap.shorten(c.replace("\n", " "), width=320, placeholder="…")
            _vlog("")
            _vlog(
                f"[reward] batch={batch_id} sample={i + 1}/{len(completions)} "
                f"route={s}->{t}"
            )
            _vlog(f"  completion (trunc): {comp_preview}")
            _vlog(
                f"  parsed: has_thought={bool(thought)} has_action_block={bool(action_blob)} "
                f"-> format_reward={fmt:.3f} (0.5 if both tags present)"
            )
            if thought_preview:
                _vlog(f"  thought (trunc): {thought_preview}")
            if not step_logs:
                if action_blob:
                    _vlog("  env: no non-empty command lines in <action> — no env steps.")
                else:
                    _vlog("  env: no <action> block (or empty) — no env steps.")
            for si, (cmd, r_step, msg_snip) in enumerate(step_logs):
                _vlog(f"  env step {si}: cmd={cmd!r} -> step_reward={r_step:.3f}")
                if msg_snip:
                    _vlog(f"           message: {msg_snip}")
            _vlog(
                f"  env: raw_sum(step_rewards)={raw_total:.3f}  scaled_env=raw/5={scaled:.3f}  "
                f"combined={combined:.3f} (= format + scaled_env)"
            )

    return out


def _make_verbose_callback():
    from transformers import TrainerCallback

    class FlyWiseTrainLogCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            _vlog(
                f"[GRPO] training started: max_steps={args.max_steps} "
                f"logging_steps={args.logging_steps} output_dir={args.output_dir}"
            )

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not _TRAIN_VERBOSE or not logs:
                return
            # Trainer / GRPO logs vary by TRL version; print the dict for transparency.
            _vlog(f"[GRPO] optimizer_step={state.global_step} epoch={state.epoch:.4f} metrics={logs}")

        def on_train_end(self, args, state, control, **kwargs):
            _vlog(
                f"[GRPO] training finished: global_step={state.global_step} "
                f"total_epochs={state.epoch:.4f}"
            )

    return FlyWiseTrainLogCallback()


def _extract_mean_reward_from_logs(logs: dict | None) -> float | None:
    if not logs:
        return None
    for key in ("reward", "rewards/combined_flywise_reward/mean"):
        if key in logs:
            try:
                return float(logs[key])
            except (TypeError, ValueError):
                continue
    return None


def _make_reward_trend_callback():
    """Logs mean GRPO group reward each step; full digest is printed in ``_print_run_summary``."""
    from transformers import TrainerCallback

    class RewardTrendCallback(TrainerCallback):
        def __init__(self) -> None:
            self.points: list[tuple[int, float]] = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            r = _extract_mean_reward_from_logs(logs)
            if r is None:
                return
            self.points.append((int(state.global_step), r))
            print(
                f"[train_grpo] optimizer_step={state.global_step}  "
                f"mean_reward(group of {getattr(args, 'num_generations', 4)} samples)={r:.4f}  "
                f"(↑ usually means better completions vs GRPO baseline for that batch)",
                flush=True,
            )

        def on_train_end(self, args, state, control, **kwargs):
            # Detailed reward interpretation lives in ``_print_run_summary`` after ``trainer.train()``.
            pass

    return RewardTrendCallback()


def _print_run_summary(
    *,
    args: argparse.Namespace,
    trainer,
    dataset_len: int,
    db_path: str,
    elapsed_s: float,
    reward_cb,
    device_label: str,
    used_lora: bool,
) -> None:
    """Single end-of-run block: config, timing, rewards, artifacts, next steps."""
    sep = "=" * 72
    print("", flush=True)
    print(sep, flush=True)
    print("  FlyWise GRPO — run summary", flush=True)
    print(sep, flush=True)

    print("  Configuration", flush=True)
    print(f"    SQLite DB          : {db_path}", flush=True)
    print(f"    Base model         : {args.model}", flush=True)
    print(f"    Training mode      : {'LoRA adapters' if used_lora else 'full fine-tune'}", flush=True)
    if used_lora:
        print(
            f"    LoRA               : r={args.lora_r}  alpha={args.lora_alpha}  dropout={args.lora_dropout}",
            flush=True,
        )
    print(f"    Device             : {device_label}", flush=True)
    print(f"    Prompt rows        : {dataset_len}", flush=True)
    print(f"    max_steps          : {args.max_steps}  (optimizer updates)", flush=True)
    print(f"    num_generations    : 4  (completions per prompt when scoring reward)", flush=True)
    print(f"    learning_rate      : 5e-6  |  output_dir : {args.output_dir}", flush=True)

    state = getattr(trainer, "state", None)
    gstep = getattr(state, "global_step", None) if state is not None else None
    epoch = getattr(state, "epoch", None) if state is not None else None
    print("", flush=True)
    print("  Training run", flush=True)
    print(f"    Wall time          : {elapsed_s:.1f}s ({elapsed_s / 60.0:.2f} min)", flush=True)
    if gstep is not None:
        print(f"    global_step        : {gstep}", flush=True)
    if epoch is not None:
        print(f"    epoch (partial)    : {epoch:.4f}", flush=True)

    points = getattr(reward_cb, "points", []) or []
    print("", flush=True)
    print("  Reward (combined = format + env/5)", flush=True)
    if len(points) < 1:
        print("    (no mean_reward logged — check TRL logs)", flush=True)
    elif len(points) < 2:
        _s, _r = points[0]
        print(f"    step {_s} mean_reward : {_r:.4f}  (need ≥2 steps to compare first vs last)", flush=True)
    else:
        first_s, first_r = points[0]
        last_s, last_r = points[-1]
        avg = sum(p[1] for p in points) / len(points)
        best_s, best_r = max(points, key=lambda x: x[1])
        delta = last_r - first_r
        print(f"    first step {first_s:>4}  mean_reward = {first_r:.4f}", flush=True)
        print(f"    last  step {last_s:>4}  mean_reward = {last_r:.4f}", flush=True)
        print(f"    delta (last − first)     = {delta:+.4f}", flush=True)
        print(f"    mean over logged steps   = {avg:.4f}", flush=True)
        print(f"    best at step {best_s}     = {best_r:.4f}", flush=True)
        if delta > 1e-6:
            print("    note: last > first suggests improvement vs start of run (no formal guarantee).", flush=True)
        elif delta < -1e-6:
            print("    note: last < first — noise or need more steps / tuning.", flush=True)
        else:
            print("    note: flat — try more --max-steps or stronger supervision.", flush=True)

    print("", flush=True)
    print("  Artifacts", flush=True)
    print(f"    Saved under          : {Path(args.output_dir).resolve()}", flush=True)
    if used_lora:
        print("    Contents             : LoRA adapter weights + PEFT config (not full base weights).", flush=True)
        print("    Load example         : PeftModel.from_pretrained(base_model, output_dir)", flush=True)
    else:
        print("    Contents             : full updated model checkpoint.", flush=True)

    print("", flush=True)
    print("  Next steps", flush=True)
    print("    • Point inference / eval at the saved path (merge LoRA or load PeftModel).", flush=True)
    print("    • Keep using the same SQLite DB so routes match training.", flush=True)
    print(sep, flush=True)
    print("", flush=True)


def _build_lora_config(
    r: int,
    alpha: int,
    dropout: float,
):
    """LoRA for Qwen2 / Llama-style attention + MLP linears."""
    try:
        from peft import LoraConfig, TaskType
    except ImportError as e:
        raise ImportError(
            "LoRA requires the `peft` package. Install training deps: pip install -e \".[train]\""
        ) from e
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("FLYWISE_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to flywise_flights.db (default: project flywise_flights.db or FLYWISE_DB_PATH env).",
    )
    parser.add_argument("--output-dir", default="flywise-grpo-out")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Train all base model weights instead of LoRA adapters (more VRAM / memory).",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (ignored with --full-finetune).")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (ignored with --full-finetune).")
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (ignored with --full-finetune).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log each reward batch: model completions, parsed actions, per-step env rewards, and GRPO metrics.",
    )
    args = parser.parse_args()

    global _TRAIN_VERBOSE, _TRAIN_DB_PATH
    _TRAIN_VERBOSE = args.verbose
    _TRAIN_DB_PATH = _resolve_train_db_path(args.db_path)

    from trl import GRPOConfig, GRPOTrainer

    if not os.path.isfile(_db_path_for_training()):
        seed_synthetic_flywise_database(_db_path_for_training())

    _log_database_proof(_db_path_for_training())

    dataset = make_dataset(num_duplicates=2)
    print(
        f"[train_grpo] Dataset: {len(dataset)} prompt rows | max_steps={args.max_steps} | "
        f"num_generations=4 (samples per prompt for GRPO) | verbose={'on' if args.verbose else 'off'}",
        flush=True,
    )

    # GRPOConfig defaults bf16=True. Use CPU only when neither CUDA nor Apple Metal (MPS) is available.
    _has_cuda = torch.cuda.is_available()
    _has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    use_cpu = not _has_cuda and not _has_mps
    bf16 = (not use_cpu) and is_torch_bf16_gpu_available()
    if _has_mps:
        device_label = "Apple Metal (mps)"
        print("[train_grpo] Using Apple Metal (mps) — Mac GPU.", flush=True)
    elif _has_cuda:
        device_label = "CUDA"
        print("[train_grpo] Using CUDA.", flush=True)
    else:
        device_label = "CPU"
        print("[train_grpo] Using CPU only (no CUDA/MPS).", flush=True)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=5e-6,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=384,
        logging_steps=1,
        report_to="none",
        use_cpu=use_cpu,
        bf16=bf16,
    )

    peft_config = None if args.full_finetune else _build_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    if peft_config is not None:
        print(
            f"[train_grpo] LoRA: r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} "
            f"(base model frozen: {args.model!r})",
            flush=True,
        )
    else:
        print("[train_grpo] Full fine-tune: all parameters trainable.", flush=True)

    reward_trend_cb = _make_reward_trend_callback()
    train_callbacks = [reward_trend_cb]
    if args.verbose:
        train_callbacks.append(_make_verbose_callback())
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=combined_flywise_reward,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=train_callbacks,
    )
    print("[train_grpo] Starting trainer.train() — generating completions, scoring rewards, optimizer steps.", flush=True)
    _t0 = time.perf_counter()
    trainer.train()
    _elapsed = time.perf_counter() - _t0
    trainer.save_model(args.output_dir)
    _print_run_summary(
        args=args,
        trainer=trainer,
        dataset_len=len(dataset),
        db_path=_db_path_for_training(),
        elapsed_s=_elapsed,
        reward_cb=reward_trend_cb,
        device_label=device_label,
        used_lora=peft_config is not None,
    )


if __name__ == "__main__":
    main()
