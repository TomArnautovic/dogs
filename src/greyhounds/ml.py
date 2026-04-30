from __future__ import annotations

import atexit
import copy
import itertools
import json
import math
import os
import platform
import statistics
import sys
import uuid
from collections.abc import Iterator
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from torch import nn

from .config import Settings
from .db import (
    Dog,
    PredictionEntry,
    PredictionRun,
    Race,
    RaceEntry,
    Track,
    TrainingRun,
    init_database,
    latest_usable_training_run,
    session_scope,
    slugify,
)


FIXED_RUNNER_COUNT = 6
FORM_WINDOW = 6
FORM_MAX_POSITION = 8
FORM_TREND_THRESHOLD = 0.25
FORM_SPECIAL_SYMBOLS = frozenset({"F", "U", "R", "T", "P", "D", "S", "B", "C", "I", "L", "A"})
MODEL_LAYOUT_VERSION = "matlab-common-plus-six-dogs-v6-listwise"
HistoryKey = int | str

COMMON_FEATURE_NAMES = [
    "track_index_norm",
    "distance_norm",
    "grade_number",
    "purse_norm",
    "going_fast_flag",
    "going_slow_flag",
]

DOG_BASE_FEATURE_NAMES = [
    "trap_norm",
    "inside_box",
    "outside_box",
    "weight_norm",
    "age_norm",
    "is_male",
    "is_female",
    "trainer_runs_norm",
    "trainer_win_rate",
    "owner_runs_norm",
    "owner_win_rate",
    "career_runs_norm",
    "career_win_rate",
    "career_place_rate",
    "same_track_runs_norm",
    "same_track_win_rate",
    "same_distance_runs_norm",
    "same_distance_win_rate",
    "days_since_last_run",
    "vacant_flag",
    "scratched_flag",
    "form_last_position_norm",
    "form_avg_position_norm",
    "form_median_position_norm",
    "form_best_position_norm",
    "form_worst_position_norm",
    "form_wins_count_norm",
    "form_top3_count_norm",
    "form_runs_count_norm",
    "form_recent_avg_norm",
    "form_older_avg_norm",
    "form_trend_delta",
    "form_improving_flag",
    "form_declining_flag",
    "form_variance_norm",
    "form_std_dev_norm",
    "form_consistency_score",
    "form_longest_win_streak_norm",
    "form_longest_loss_streak_norm",
    "form_has_fall",
    "form_fall_rate",
    "form_poor_finish_rate",
    "trap_win_rate",
    "trap_bias_score",
    "avg_position_at_distance_norm",
    "avg_position_at_track_norm",
    "win_rate_at_track",
    "win_rate_at_distance",
    "avg_split_time_norm",
    "avg_speed_norm",
    "implied_probability",
    "favourite_flag",
    "relative_avg_position",
    "relative_win_rate",
    "relative_consistency",
    "relative_implied_probability",
    "rank_by_avg_position_norm",
    "rank_by_odds_norm",
    "early_speed_rank_norm",
]

FORM_SLOT_FEATURE_NAMES = [
    feature_name
    for feature_group in (
        "parsed_form_position_norm",
        "parsed_form_special_flag",
        "parsed_form_fall_flag",
    )
    for slot in range(1, FORM_WINDOW + 1)
    for feature_name in [f"{feature_group}_{slot}"]
]

RECENT_HISTORY_FEATURE_NAMES = [
    feature_name
    for feature_group in (
        "recent_finish_norm",
        "recent_speed_norm",
        "recent_sectional_norm",
        "recent_sp_norm",
        "recent_gap_days_norm",
        "recent_same_track_flag",
        "recent_same_distance_flag",
    )
    for slot in range(1, FORM_WINDOW + 1)
    for feature_name in [f"{feature_group}_{slot}"]
]

DOG_FEATURE_NAMES = DOG_BASE_FEATURE_NAMES + FORM_SLOT_FEATURE_NAMES + RECENT_HISTORY_FEATURE_NAMES


@dataclass(slots=True)
class TrainingConfig:
    model_type: str = "permutation"
    resume_from_artifact: str | None = None
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_size_1: int = 128
    hidden_size_2: int = 64
    dropout: float = 0.15
    validation_fraction: float = 0.2
    weight_decay: float = 1e-5
    seed: int = 42
    max_runners: int = FIXED_RUNNER_COUNT
    min_completed_races: int = 25
    permutations_per_race: int = 24
    permutation_runner_limit: int = FIXED_RUNNER_COUNT
    early_stopping_patience: int = 20


@dataclass(slots=True)
class RaceExample:
    race_id: int
    race_key: str
    track_name: str
    scheduled_start: datetime
    common_features: np.ndarray
    dog_features: np.ndarray
    mask: np.ndarray
    targets: np.ndarray
    dog_ids: np.ndarray
    race_entry_ids: np.ndarray
    dog_names: list[str]
    has_target: bool


@dataclass(slots=True)
class Scaler:
    common_mean: np.ndarray
    common_std: np.ndarray
    dog_mean: np.ndarray
    dog_std: np.ndarray


class _TrainingRunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
            handle.flush()


class TrainingStopRequested(Exception):
    def __init__(self, message: str, *, recovery_artifact_path: str | None = None) -> None:
        super().__init__(message)
        self.recovery_artifact_path = recovery_artifact_path


def _format_stop_request_reason(
    stop_request: dict[str, Any] | None,
    *,
    stage: str,
    epoch: int,
    recovery_artifact_path: Path,
) -> str:
    if not isinstance(stop_request, dict):
        return (
            f"Training stopped during {stage} at epoch {epoch}. "
            f"Saved recovery checkpoint to {recovery_artifact_path}."
        )

    requested_by = str(stop_request.get("requested_by") or "unknown")
    requested_at = stop_request.get("requested_at")
    requested_at_text = str(requested_at) if requested_at else "an unknown time"
    requested_reason = str(stop_request.get("reason") or "No reason was provided.")
    return (
        f"Training stopped during {stage} at epoch {epoch} after a request from {requested_by} "
        f"at {requested_at_text}. Reason: {requested_reason} "
        f"Saved recovery checkpoint to {recovery_artifact_path}."
    )


def _logs_dir(settings: Settings) -> Path:
    logs_path = getattr(settings, "logs_dir", settings.artifacts_dir / "logs")
    resolved = Path(logs_path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _control_dir(settings: Settings) -> Path:
    resolved = settings.artifacts_dir / "control"
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _weights_dir(settings: Settings) -> Path:
    resolved = settings.artifacts_dir / "weights"
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def training_weight_snapshot_path(settings: Settings, run_key: str) -> Path:
    return _weights_dir(settings) / f"{run_key}-latest.pt"


def training_stop_request_path(settings: Settings, run_key: str) -> Path:
    return _control_dir(settings) / f"{run_key}-stop.json"


def read_training_stop_request(settings: Settings, run_key: str) -> dict[str, Any] | None:
    path = training_stop_request_path(settings, run_key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"requested_at": None, "requested_by": "unknown", "reason": "Stop requested."}
    return payload if isinstance(payload, dict) else {"requested_at": None, "requested_by": "unknown", "reason": "Stop requested."}


def request_training_stop(
    settings: Settings,
    run_key: str,
    *,
    requested_by: str = "dashboard",
    reason: str = "Stop requested from the dashboard.",
) -> Path:
    path = training_stop_request_path(settings, run_key)
    payload = {
        "run_key": run_key,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "requested_by": requested_by,
        "reason": reason,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def clear_training_stop_request(settings: Settings, run_key: str) -> None:
    path = training_stop_request_path(settings, run_key)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _clip_signed(value: float, scale: float = 1.0) -> float:
    if scale <= 0:
        return 0.0
    return max(-1.0, min(value / scale, 1.0))


def _safe_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _safe_rate(hit_count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return hit_count / total


def _grade_number(grade: str | None) -> float:
    if not grade:
        return 0.0
    digits = "".join(ch for ch in grade if ch.isdigit())
    if not digits:
        return 0.0 if grade.upper() != "OR" else 1.0
    return _clip(int(digits) / 20.0)


def _going_flags(going: str | None) -> tuple[float, float]:
    if not going:
        return 0.0, 0.0
    text = going.lower()
    fast = 1.0 if any(token in text for token in ("fast", "firm", "quick")) else 0.0
    slow = 1.0 if any(token in text for token in ("slow", "heavy", "soft")) else 0.0
    return fast, slow


def _race_common_vector(race: Race, track_index_by_id: dict[int, int], track_count: int) -> list[float]:
    going_fast_flag, going_slow_flag = _going_flags(race.going)
    normalized_track_index = 0.0
    if track_count > 0:
        track_index = track_index_by_id.get(int(race.track_id), 0)
        normalized_track_index = _clip(track_index / track_count)
    return [
        normalized_track_index,
        _clip((race.distance_m or 0) / 1000.0, 0.0, 2.0),
        _grade_number(race.grade),
        _clip(float(race.purse or 0.0) / 5000.0, 0.0, 5.0),
        going_fast_flag,
        going_slow_flag,
    ]


def _new_summary_stats() -> dict[str, float]:
    return {
        "runs": 0.0,
        "finish_total": 0.0,
        "wins": 0.0,
        "top3": 0.0,
        "sectional_total": 0.0,
        "sectional_count": 0.0,
        "speed_total": 0.0,
        "speed_count": 0.0,
    }


def _update_summary_stats(stats: dict[str, float], event: dict[str, Any]) -> None:
    finish = float(event["finish"] or 0.0)
    if finish <= 0:
        return
    stats["runs"] += 1.0
    stats["finish_total"] += finish
    if finish == 1.0:
        stats["wins"] += 1.0
    if finish <= 3.0:
        stats["top3"] += 1.0
    sectional = float(event.get("sectional") or 0.0)
    if sectional > 0:
        stats["sectional_total"] += sectional
        stats["sectional_count"] += 1.0
    speed = float(event.get("speed") or 0.0)
    if speed > 0:
        stats["speed_total"] += speed
        stats["speed_count"] += 1.0


def _dog_identity_history_key(dog_name: str | None, trainer_id: int | None) -> str | None:
    if not dog_name or not trainer_id:
        return None
    name_slug = slugify(dog_name)
    if not name_slug or name_slug == "unknown":
        return None
    return f"dog-name-trainer:{name_slug}:{int(trainer_id)}"


def _dog_history_for_entry(dog: Dog | None, dog_history: dict[HistoryKey, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if dog is None:
        return []

    keys: list[HistoryKey] = [int(dog.id)]
    identity_key = _dog_identity_history_key(dog.name, dog.trainer_id)
    if identity_key is not None:
        keys.append(identity_key)

    events: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for key in keys:
        for event in dog_history.get(key, []):
            event_key = (
                event.get("race_entry_id") or 0,
                event["when"],
                event["track_id"],
                event["trap_number"],
                event["finish"],
            )
            if event_key in seen:
                continue
            seen.add(event_key)
            events.append(event)

    events.sort(key=lambda item: (item["when"], int(item.get("race_entry_id") or 0)))
    return events


def _summary_mean_position(stats: dict[str, float], fallback: float) -> float:
    runs = int(stats.get("runs", 0.0))
    return (float(stats["finish_total"]) / runs) if runs > 0 else fallback


def _summary_win_rate(stats: dict[str, float], fallback: float) -> float:
    runs = int(stats.get("runs", 0.0))
    return (float(stats["wins"]) / runs) if runs > 0 else fallback


def _summary_avg_sectional(stats: dict[str, float], fallback: float) -> float:
    count = int(stats.get("sectional_count", 0.0))
    return (float(stats["sectional_total"]) / count) if count > 0 else fallback


def _summary_avg_speed(stats: dict[str, float], fallback: float) -> float:
    count = int(stats.get("speed_count", 0.0))
    return (float(stats["speed_total"]) / count) if count > 0 else fallback


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if numeric > 0 else None
    if isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return None
        return numeric if numeric > 0 else None
    if isinstance(value, list):
        for item in reversed(value):
            numeric = _coerce_float(item)
            if numeric is not None:
                return numeric
        return None
    if isinstance(value, dict):
        for key in ("decimal", "odds", "price", "value"):
            if key in value:
                numeric = _coerce_float(value[key])
                if numeric is not None:
                    return numeric
        return None
    return None


def _entry_odds_decimal(entry: RaceEntry) -> float | None:
    if entry.sp_decimal and entry.sp_decimal > 0:
        return float(entry.sp_decimal)
    metadata = entry.metadata_json if isinstance(entry.metadata_json, dict) else {}
    for key in ("odds", "sp", "price"):
        numeric = _coerce_float(metadata.get(key))
        if numeric is not None:
            return numeric
    return None


def _entry_implied_probability(entry: RaceEntry) -> float:
    decimal_odds = _entry_odds_decimal(entry)
    if decimal_odds is None or decimal_odds <= 0:
        return 1.0 / FIXED_RUNNER_COUNT
    return 1.0 / decimal_odds


def _parse_form_runs(raw_form: str | None, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    if raw_form:
        for symbol in raw_form.upper():
            if symbol in {"-", " ", "/", "."}:
                continue
            if symbol.isdigit():
                parsed.append(
                    {
                        "symbol": symbol,
                        "finish": max(1, min(int(symbol), FORM_MAX_POSITION)),
                        "is_special": 0.0,
                        "is_fall": 0.0,
                    }
                )
            elif symbol.isalpha():
                parsed.append(
                    {
                        "symbol": symbol,
                        "finish": FORM_MAX_POSITION if symbol in FORM_SPECIAL_SYMBOLS else FORM_MAX_POSITION,
                        "is_special": 1.0,
                        "is_fall": 1.0 if symbol == "F" else 0.0,
                    }
                )
        parsed = parsed[-FORM_WINDOW:]
        if parsed:
            return parsed

    fallback = []
    for item in history[-FORM_WINDOW:]:
        finish = int(item["finish"] or 0)
        fallback.append(
            {
                "symbol": str(finish) if finish > 0 else "",
                "finish": max(1, min(finish, FORM_MAX_POSITION)) if finish > 0 else FORM_MAX_POSITION,
                "is_special": 0.0,
                "is_fall": 0.0,
            }
        )
    return fallback


def _longest_streak(form_runs: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> int:
    longest = 0
    current = 0
    for run in form_runs:
        if predicate(run):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _form_feature_snapshot(raw_form: str | None, history: list[dict[str, Any]]) -> dict[str, Any]:
    form_runs = _parse_form_runs(raw_form, history)
    positions = [int(run["finish"]) for run in form_runs]
    runs_count = len(positions)

    if not positions:
        positions = [FORM_MAX_POSITION]
        runs_count = 0

    last_position = positions[-1]
    avg_position = _safe_mean([float(value) for value in positions])
    median_position = float(statistics.median(positions))
    best_position = float(min(positions))
    worst_position = float(max(positions))
    wins_count = sum(1 for value in positions if value == 1)
    top3_count = sum(1 for value in positions if value <= 3)
    recent_window = positions[-3:] if positions else []
    older_window = positions[: min(3, len(positions))] if positions else []
    recent_avg = _safe_mean([float(value) for value in recent_window]) if recent_window else avg_position
    older_avg = _safe_mean([float(value) for value in older_window]) if older_window else avg_position
    trend_delta = older_avg - recent_avg
    variance = float(np.var(positions)) if len(positions) > 1 else 0.0
    std_dev = math.sqrt(variance)
    consistency_score = 1.0 / (1.0 + variance)
    longest_win_streak = _longest_streak(form_runs, lambda run: int(run["finish"]) == 1)
    longest_loss_streak = _longest_streak(form_runs, lambda run: int(run["finish"]) != 1)
    has_fall = 1.0 if any(float(run["is_fall"]) > 0 for run in form_runs) else 0.0
    fall_rate = _safe_rate(sum(1 for run in form_runs if float(run["is_fall"]) > 0), len(form_runs))
    poor_finish_rate = _safe_rate(sum(1 for value in positions if value >= 5), len(positions))

    recent_slots = list(reversed(form_runs))
    parsed_form_positions: list[float] = []
    parsed_form_special_flags: list[float] = []
    parsed_form_fall_flags: list[float] = []
    for slot in range(FORM_WINDOW):
        if slot < len(recent_slots):
            run = recent_slots[slot]
            parsed_form_positions.append(_clip(float(run["finish"]) / FORM_MAX_POSITION))
            parsed_form_special_flags.append(float(run["is_special"]))
            parsed_form_fall_flags.append(float(run["is_fall"]))
        else:
            parsed_form_positions.append(0.0)
            parsed_form_special_flags.append(0.0)
            parsed_form_fall_flags.append(0.0)

    feature_values: dict[str, float] = {
        "form_last_position_norm": _clip(last_position / FORM_MAX_POSITION),
        "form_avg_position_norm": _clip(avg_position / FORM_MAX_POSITION),
        "form_median_position_norm": _clip(median_position / FORM_MAX_POSITION),
        "form_best_position_norm": _clip(best_position / FORM_MAX_POSITION),
        "form_worst_position_norm": _clip(worst_position / FORM_MAX_POSITION),
        "form_wins_count_norm": _clip(wins_count / FORM_WINDOW),
        "form_top3_count_norm": _clip(top3_count / FORM_WINDOW),
        "form_runs_count_norm": _clip(runs_count / FORM_WINDOW),
        "form_recent_avg_norm": _clip(recent_avg / FORM_MAX_POSITION),
        "form_older_avg_norm": _clip(older_avg / FORM_MAX_POSITION),
        "form_trend_delta": _clip_signed(trend_delta, FORM_MAX_POSITION),
        "form_improving_flag": 1.0 if trend_delta > FORM_TREND_THRESHOLD else 0.0,
        "form_declining_flag": 1.0 if trend_delta < -FORM_TREND_THRESHOLD else 0.0,
        "form_variance_norm": _clip(variance / (FORM_MAX_POSITION**2)),
        "form_std_dev_norm": _clip(std_dev / FORM_MAX_POSITION),
        "form_consistency_score": _clip(consistency_score),
        "form_longest_win_streak_norm": _clip(longest_win_streak / FORM_WINDOW),
        "form_longest_loss_streak_norm": _clip(longest_loss_streak / FORM_WINDOW),
        "form_has_fall": has_fall,
        "form_fall_rate": _clip(fall_rate),
        "form_poor_finish_rate": _clip(poor_finish_rate),
    }
    for slot in range(FORM_WINDOW):
        feature_values[f"parsed_form_position_norm_{slot + 1}"] = parsed_form_positions[slot]
        feature_values[f"parsed_form_special_flag_{slot + 1}"] = parsed_form_special_flags[slot]
        feature_values[f"parsed_form_fall_flag_{slot + 1}"] = parsed_form_fall_flags[slot]

    return {
        "feature_values": feature_values,
        "avg_position_raw": avg_position,
        "win_rate_raw": _safe_rate(wins_count, len(positions)),
        "consistency_raw": consistency_score,
    }


def _recent_history_feature_values(history: list[dict[str, Any]], race: Race) -> dict[str, float]:
    recent = history[-FORM_WINDOW:]
    recent_reverse = list(reversed(recent))
    values: dict[str, float] = {}

    for slot in range(FORM_WINDOW):
        if slot < len(recent_reverse):
            item = recent_reverse[slot]
            gap_days = max((race.scheduled_start - item["when"]).total_seconds() / 86400.0, 0.0)
            values[f"recent_finish_norm_{slot + 1}"] = _clip((item["finish"] or 0) / FORM_MAX_POSITION)
            values[f"recent_speed_norm_{slot + 1}"] = _clip((item["speed"] or 0.0) / 20.0, 0.0, 2.0)
            values[f"recent_sectional_norm_{slot + 1}"] = _clip((item["sectional"] or 0.0) / 10.0, 0.0, 2.0)
            values[f"recent_sp_norm_{slot + 1}"] = _clip((item["sp_decimal"] or 0.0) / 20.0, 0.0, 5.0)
            values[f"recent_gap_days_norm_{slot + 1}"] = _clip(gap_days / 60.0, 0.0, 3.0)
            values[f"recent_same_track_flag_{slot + 1}"] = 1.0 if item["track_id"] == race.track_id else 0.0
            values[f"recent_same_distance_flag_{slot + 1}"] = 1.0 if item["distance_m"] == race.distance_m else 0.0
        else:
            values[f"recent_finish_norm_{slot + 1}"] = 0.0
            values[f"recent_speed_norm_{slot + 1}"] = 0.0
            values[f"recent_sectional_norm_{slot + 1}"] = 0.0
            values[f"recent_sp_norm_{slot + 1}"] = 0.0
            values[f"recent_gap_days_norm_{slot + 1}"] = 0.0
            values[f"recent_same_track_flag_{slot + 1}"] = 0.0
            values[f"recent_same_distance_flag_{slot + 1}"] = 0.0
    return values


def _rank_values(values: list[float], *, reverse: bool) -> list[float]:
    ordered = sorted(
        range(len(values)),
        key=lambda index: ((-values[index]) if reverse else values[index], index),
    )
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        value = values[ordered[start]]
        while end < len(ordered) and values[ordered[end]] == value:
            end += 1
        # Equal values should be neutral, not ranked by trap/order position.
        average_rank = (start + 1 + end) / 2.0
        for ordered_index in range(start, end):
            ranks[ordered[ordered_index]] = average_rank
        start = end
    return ranks


def _build_dog_feature_snapshot(
    entry: RaceEntry,
    race: Race,
    dog_history: dict[HistoryKey, list[dict[str, Any]]],
    trainer_history: dict[int, list[dict[str, Any]]],
    owner_history: dict[int, list[dict[str, Any]]],
    global_summary: dict[str, float],
    track_summaries: dict[int, dict[str, float]],
    distance_summaries: dict[int, dict[str, float]],
    trap_summaries: dict[int, dict[str, float]],
) -> dict[str, Any]:
    dog = entry.dog
    history = _dog_history_for_entry(dog, dog_history)
    trainer_runs = trainer_history.get(dog.trainer_id, []) if dog and dog.trainer_id else []
    owner_runs = owner_history.get(dog.owner_id, []) if dog and dog.owner_id else []
    same_track_history = [item for item in history if item["track_id"] == race.track_id]
    same_distance_history = [item for item in history if item["distance_m"] == int(race.distance_m or 0)]
    same_trap_history = [item for item in history if item["trap_number"] == int(entry.trap_number)]
    raw_form = None
    if isinstance(entry.metadata_json, dict):
        raw_form = entry.metadata_json.get("form")

    global_avg_position = _summary_mean_position(global_summary, (FIXED_RUNNER_COUNT + 1) / 2.0)
    global_win_rate = _summary_win_rate(global_summary, 1.0 / FIXED_RUNNER_COUNT)
    global_avg_sectional = _summary_avg_sectional(global_summary, 10.0)
    global_avg_speed = _summary_avg_speed(global_summary, 0.0)
    track_summary = track_summaries.get(int(race.track_id), _new_summary_stats())
    distance_key = int(race.distance_m or 0)
    distance_summary = distance_summaries.get(distance_key, _new_summary_stats())
    trap_summary = trap_summaries.get(int(entry.trap_number), _new_summary_stats())

    form_snapshot = _form_feature_snapshot(str(raw_form) if raw_form is not None else None, history)
    recent_history_values = _recent_history_feature_values(history, race)

    career_runs = len(history)
    career_wins = sum(1 for item in history if item["finish"] == 1)
    career_places = sum(1 for item in history if item["finish"] and item["finish"] <= 3)
    same_track_wins = sum(1 for item in same_track_history if item["finish"] == 1)
    same_distance_wins = sum(1 for item in same_distance_history if item["finish"] == 1)
    trap_wins = sum(1 for item in same_trap_history if item["finish"] == 1)
    avg_position_at_track = (
        _safe_mean([float(item["finish"]) for item in same_track_history])
        if same_track_history
        else _summary_mean_position(track_summary, global_avg_position)
    )
    avg_position_at_distance = (
        _safe_mean([float(item["finish"]) for item in same_distance_history])
        if same_distance_history
        else _summary_mean_position(distance_summary, global_avg_position)
    )
    win_rate_at_track = (
        _safe_rate(same_track_wins, len(same_track_history))
        if same_track_history
        else _summary_win_rate(track_summary, global_win_rate)
    )
    win_rate_at_distance = (
        _safe_rate(same_distance_wins, len(same_distance_history))
        if same_distance_history
        else _summary_win_rate(distance_summary, global_win_rate)
    )
    avg_split_time = (
        _safe_mean([float(item["sectional"]) for item in history if float(item["sectional"]) > 0])
        if any(float(item["sectional"]) > 0 for item in history)
        else global_avg_sectional
    )
    avg_speed = (
        _safe_mean([float(item["speed"]) for item in history if float(item["speed"]) > 0])
        if any(float(item["speed"]) > 0 for item in history)
        else global_avg_speed
    )

    last = history[-1] if history else None
    days_since_last_run = 0.0
    if last is not None:
        gap_days = (race.scheduled_start - last["when"]).total_seconds() / 86400.0
        days_since_last_run = _clip(gap_days / 60.0, 0.0, 3.0)

    sex = (dog.sex or "").strip().lower() if dog else ""
    age_norm = 0.0
    if dog and dog.date_of_birth:
        age_days = (race.scheduled_start.date() - dog.date_of_birth).days
        age_norm = _clip(age_days / 2000.0, 0.0, 2.0)

    implied_probability = _entry_implied_probability(entry)
    feature_values: dict[str, float] = {
        "trap_norm": _clip(entry.trap_number / 8.0),
        "inside_box": 1.0 if entry.trap_number <= 2 else 0.0,
        "outside_box": 1.0 if entry.trap_number >= 5 else 0.0,
        "weight_norm": _clip((entry.weight_kg or 0.0) / 40.0, 0.0, 2.0),
        "age_norm": age_norm,
        "is_male": 1.0 if sex in {"m", "male", "dog"} else 0.0,
        "is_female": 1.0 if sex in {"f", "female", "bitch"} else 0.0,
        "trainer_runs_norm": _clip(len(trainer_runs) / 500.0, 0.0, 5.0),
        "trainer_win_rate": _safe_rate(sum(1 for item in trainer_runs if item["finish"] == 1), len(trainer_runs)),
        "owner_runs_norm": _clip(len(owner_runs) / 500.0, 0.0, 5.0),
        "owner_win_rate": _safe_rate(sum(1 for item in owner_runs if item["finish"] == 1), len(owner_runs)),
        "career_runs_norm": _clip(career_runs / 100.0, 0.0, 5.0),
        "career_win_rate": _safe_rate(career_wins, career_runs),
        "career_place_rate": _safe_rate(career_places, career_runs),
        "same_track_runs_norm": _clip(len(same_track_history) / 40.0, 0.0, 5.0),
        "same_track_win_rate": _safe_rate(same_track_wins, len(same_track_history)),
        "same_distance_runs_norm": _clip(len(same_distance_history) / 40.0, 0.0, 5.0),
        "same_distance_win_rate": _safe_rate(same_distance_wins, len(same_distance_history)),
        "days_since_last_run": days_since_last_run,
        "vacant_flag": 1.0 if entry.vacant else 0.0,
        "scratched_flag": 1.0 if entry.scratched else 0.0,
        "trap_win_rate": _safe_rate(trap_wins, len(same_trap_history)),
        "trap_bias_score": _summary_win_rate(trap_summary, global_win_rate),
        "avg_position_at_distance_norm": _clip(avg_position_at_distance / FORM_MAX_POSITION),
        "avg_position_at_track_norm": _clip(avg_position_at_track / FORM_MAX_POSITION),
        "win_rate_at_track": win_rate_at_track,
        "win_rate_at_distance": win_rate_at_distance,
        "avg_split_time_norm": _clip(avg_split_time / 10.0, 0.0, 2.0),
        "avg_speed_norm": _clip(avg_speed / 20.0, 0.0, 2.0),
        "implied_probability": _clip(implied_probability, 0.0, 1.0),
        "favourite_flag": 0.0,
        "relative_avg_position": 0.0,
        "relative_win_rate": 0.0,
        "relative_consistency": 0.0,
        "relative_implied_probability": 0.0,
        "rank_by_avg_position_norm": 0.0,
        "rank_by_odds_norm": 0.0,
        "early_speed_rank_norm": 0.0,
    }
    feature_values.update(form_snapshot["feature_values"])
    feature_values.update(recent_history_values)

    return {
        "trap_number": int(entry.trap_number),
        "avg_position_raw": float(form_snapshot["avg_position_raw"]),
        "win_rate_raw": float(form_snapshot["win_rate_raw"]),
        "consistency_raw": float(form_snapshot["consistency_raw"]),
        "implied_probability_raw": float(implied_probability),
        "avg_split_time_raw": float(avg_split_time),
        "feature_values": feature_values,
    }


def _apply_relative_features(snapshots: list[dict[str, Any]]) -> None:
    if not snapshots:
        return

    avg_positions = [float(snapshot["avg_position_raw"]) for snapshot in snapshots]
    win_rates = [float(snapshot["win_rate_raw"]) for snapshot in snapshots]
    consistencies = [float(snapshot["consistency_raw"]) for snapshot in snapshots]
    implied_probabilities = [float(snapshot["implied_probability_raw"]) for snapshot in snapshots]
    split_times = [float(snapshot["avg_split_time_raw"]) for snapshot in snapshots]

    race_avg_position = _safe_mean(avg_positions)
    race_avg_win_rate = _safe_mean(win_rates)
    race_avg_consistency = _safe_mean(consistencies)
    race_avg_implied_probability = _safe_mean(implied_probabilities)

    avg_position_ranks = _rank_values(avg_positions, reverse=False)
    odds_ranks = _rank_values(implied_probabilities, reverse=True)
    early_speed_ranks = _rank_values(split_times, reverse=False)

    for index, snapshot in enumerate(snapshots):
        features = snapshot["feature_values"]
        features["relative_avg_position"] = _clip_signed(
            float(snapshot["avg_position_raw"]) - race_avg_position,
            FORM_MAX_POSITION,
        )
        features["relative_win_rate"] = _clip_signed(
            float(snapshot["win_rate_raw"]) - race_avg_win_rate,
            1.0,
        )
        features["relative_consistency"] = _clip_signed(
            float(snapshot["consistency_raw"]) - race_avg_consistency,
            1.0,
        )
        features["relative_implied_probability"] = _clip_signed(
            float(snapshot["implied_probability_raw"]) - race_avg_implied_probability,
            1.0,
        )
        features["rank_by_avg_position_norm"] = _clip(avg_position_ranks[index] / FIXED_RUNNER_COUNT)
        features["rank_by_odds_norm"] = _clip(odds_ranks[index] / FIXED_RUNNER_COUNT)
        features["early_speed_rank_norm"] = _clip(early_speed_ranks[index] / FIXED_RUNNER_COUNT)
        features["favourite_flag"] = 1.0 if odds_ranks[index] == 1 else 0.0


def _feature_vector_from_snapshot(snapshot: dict[str, Any]) -> list[float]:
    feature_values = snapshot["feature_values"]
    return [float(feature_values.get(feature_name, 0.0)) for feature_name in DOG_FEATURE_NAMES]


def _active_runner_indices(example: RaceExample) -> list[int]:
    return [index for index, present in enumerate(example.mask.tolist()) if present]


def _finish_positions_are_complete(position_values: list[int | None], runner_count: int) -> bool:
    if runner_count <= 0 or len(position_values) != runner_count:
        return False

    positions: list[int] = []
    for value in position_values:
        if value is None:
            return False
        try:
            position = int(value)
        except (TypeError, ValueError):
            return False
        if position < 1 or position > runner_count:
            return False
        positions.append(position)
    return sorted(positions) == list(range(1, runner_count + 1))


def _entries_have_complete_finish_order(entries: list[Any]) -> bool:
    return _finish_positions_are_complete(
        [getattr(entry, "finish_position", None) for entry in entries],
        len(entries),
    )


def _actual_order_indices(example: RaceExample) -> list[int]:
    actual_pairs = [
        (int(example.targets[index]), index)
        for index in _active_runner_indices(example)
        if int(example.targets[index]) > 0
    ]
    actual_pairs.sort(key=lambda item: item[0])
    return [index for _, index in actual_pairs]


def _example_has_complete_target_order(example: RaceExample) -> bool:
    active = _active_runner_indices(example)
    return bool(example.has_target) and _finish_positions_are_complete(
        [int(example.targets[index]) for index in active],
        len(active),
    )


def _scaled_common_vector(example: RaceExample, scaler: Scaler) -> np.ndarray:
    return ((example.common_features - scaler.common_mean) / scaler.common_std).astype(np.float32)


def _scaled_runner_matrix(example: RaceExample, scaler: Scaler) -> np.ndarray:
    scaled = np.zeros_like(example.dog_features, dtype=np.float32)
    valid = example.mask.astype(bool)
    scaled[valid] = (example.dog_features[valid] - scaler.dog_mean) / scaler.dog_std
    return scaled


def _flatten_ordered_feature_matrix(
    example: RaceExample,
    orders: list[tuple[int, ...]],
    scaler: Scaler,
) -> np.ndarray:
    common = _scaled_common_vector(example, scaler)
    scaled = _scaled_runner_matrix(example, scaler)
    order_indices = np.asarray(orders, dtype=np.int64)
    ordered = scaled[order_indices]
    common_batch = np.broadcast_to(common, (len(orders), common.shape[0]))
    return np.concatenate(
        [common_batch, ordered.reshape(len(orders), -1)],
        axis=1,
    ).astype(np.float32, copy=False)


def _sample_training_candidate_orders(
    example: RaceExample,
    rng: np.random.Generator,
    sampled_negative_orders: int,
    permutation_runner_limit: int,
) -> list[tuple[int, ...]]:
    active = _active_runner_indices(example)
    if len(active) != permutation_runner_limit or not _example_has_complete_target_order(example):
        return []

    actual_order = tuple(_actual_order_indices(example))
    if not actual_order:
        return []

    max_orders = math.factorial(len(active))
    target_count = min(max_orders, max(2, sampled_negative_orders + 1))

    chosen_orders: list[tuple[int, ...]] = [actual_order]
    seen: set[tuple[int, ...]] = {actual_order}

    deterministic_candidates: list[tuple[int, ...]] = [
        tuple(active),
        tuple(reversed(actual_order)),
    ]
    for index in range(len(actual_order) - 1):
        swapped = list(actual_order)
        swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
        deterministic_candidates.append(tuple(swapped))

    for candidate in deterministic_candidates:
        if len(chosen_orders) >= target_count:
            break
        if candidate in seen:
            continue
        seen.add(candidate)
        chosen_orders.append(candidate)

    while len(chosen_orders) < target_count:
        candidate = tuple(int(value) for value in rng.permutation(active).tolist())
        if candidate in seen:
            continue
        seen.add(candidate)
        chosen_orders.append(candidate)

    return chosen_orders


def _iter_training_order_batches(
    examples: list[RaceExample],
    scaler: Scaler,
    config: TrainingConfig,
    seed: int,
) -> Iterator[tuple[int, int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(examples)).tolist()
    batch: list[tuple[RaceExample, list[tuple[int, ...]]]] = []

    for example_index in shuffled_indices:
        example = examples[int(example_index)]
        orders = _sample_training_candidate_orders(
            example,
            rng=rng,
            sampled_negative_orders=config.permutations_per_race,
            permutation_runner_limit=config.permutation_runner_limit,
        )
        if not orders:
            continue
        batch.append((example, orders))
        if len(batch) >= config.batch_size:
            yield _build_training_order_batch(batch, scaler)
            batch = []

    if batch:
        yield _build_training_order_batch(batch, scaler)


def _build_training_order_batch(
    batch: list[tuple[RaceExample, list[tuple[int, ...]]]],
    scaler: Scaler,
) -> tuple[int, int, np.ndarray]:
    order_count = len(batch[0][1])
    feature_blocks = []
    for example, orders in batch:
        if len(orders) != order_count:
            raise ValueError("Training order batches require a consistent candidate count.")
        feature_blocks.append(_flatten_ordered_feature_matrix(example, orders, scaler))
    features = np.stack(feature_blocks, axis=0)
    race_count = len(batch)
    return race_count, order_count, features.reshape(race_count * order_count, -1)


def _eligible_examples(examples: list[RaceExample], runner_limit: int, require_target: bool) -> list[RaceExample]:
    filtered = []
    for example in examples:
        active_runner_count = len(_active_runner_indices(example))
        if active_runner_count != runner_limit:
            continue
        if require_target and not _example_has_complete_target_order(example):
            continue
        filtered.append(example)
    return filtered


def _training_data_summary(examples: list[RaceExample], runner_limit: int) -> dict[str, int]:
    matching_runner_count = 0
    complete_finish_order = 0
    missing_or_invalid_finish_order = 0
    wrong_runner_count = 0

    for example in examples:
        active_runner_count = len(_active_runner_indices(example))
        if active_runner_count != runner_limit:
            wrong_runner_count += 1
            continue
        matching_runner_count += 1
        if _example_has_complete_target_order(example):
            complete_finish_order += 1
        else:
            missing_or_invalid_finish_order += 1

    return {
        "total_race_examples": len(examples),
        "runner_count_matching_races": matching_runner_count,
        "complete_finish_order_races": complete_finish_order,
        "missing_or_invalid_finish_order_races": missing_or_invalid_finish_order,
        "wrong_runner_count_races": wrong_runner_count,
    }


def _build_race_example(
    race: Race,
    entries: list[RaceEntry],
    max_runners: int,
    track_index_by_id: dict[int, int],
    track_count: int,
    dog_history: dict[HistoryKey, list[dict[str, Any]]],
    trainer_history: dict[int, list[dict[str, Any]]],
    owner_history: dict[int, list[dict[str, Any]]],
    global_summary: dict[str, float],
    track_summaries: dict[int, dict[str, float]],
    distance_summaries: dict[int, dict[str, float]],
    trap_summaries: dict[int, dict[str, float]],
    has_target: bool,
) -> RaceExample:
    common_features = np.array(
        _race_common_vector(race, track_index_by_id, track_count),
        dtype=np.float32,
    )
    runner_features = np.zeros((max_runners, len(DOG_FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros(max_runners, dtype=np.float32)
    targets = np.zeros(max_runners, dtype=np.int64)
    dog_ids = np.zeros(max_runners, dtype=np.int64)
    race_entry_ids = np.zeros(max_runners, dtype=np.int64)
    dog_names = [""] * max_runners
    snapshots = [
        _build_dog_feature_snapshot(
            entry,
            race,
            dog_history,
            trainer_history,
            owner_history,
            global_summary,
            track_summaries,
            distance_summaries,
            trap_summaries,
        )
        for entry in entries[:max_runners]
    ]
    _apply_relative_features(snapshots)

    for idx, entry in enumerate(entries[:max_runners]):
        runner_features[idx] = np.array(
            _feature_vector_from_snapshot(snapshots[idx]),
            dtype=np.float32,
        )
        mask[idx] = 1.0
        targets[idx] = int(entry.finish_position or 0)
        dog_ids[idx] = int(entry.dog_id or 0)
        race_entry_ids[idx] = int(entry.id or 0)
        dog_names[idx] = entry.dog.name if entry.dog else f"Trap {entry.trap_number}"

    return RaceExample(
        race_id=race.id,
        race_key=race.race_key,
        track_name=race.track.name,
        scheduled_start=race.scheduled_start,
        common_features=common_features,
        dog_features=runner_features,
        mask=mask,
        targets=targets,
        dog_ids=dog_ids,
        race_entry_ids=race_entry_ids,
        dog_names=dog_names,
        has_target=has_target,
    )


def _record_entry_history(
    *,
    race_entry_id: int | None,
    when: datetime,
    distance_m: int | None,
    track_id: int,
    trap_number: int,
    finish_position: int | None,
    official_time_s: float | None,
    sectional_s: float | None,
    sp_decimal: float | None,
    dog_id: int | None,
    dog_name: str | None,
    trainer_id: int | None,
    owner_id: int | None,
    dog_history: dict[HistoryKey, list[dict[str, Any]]],
    trainer_history: dict[int, list[dict[str, Any]]],
    owner_history: dict[int, list[dict[str, Any]]],
    global_summary: dict[str, float],
    track_summaries: dict[int, dict[str, float]],
    distance_summaries: dict[int, dict[str, float]],
    trap_summaries: dict[int, dict[str, float]],
) -> None:
    if not finish_position:
        return
    dog_history_keys: list[HistoryKey] = []
    if dog_id:
        dog_history_keys.append(int(dog_id))
    identity_key = _dog_identity_history_key(dog_name, trainer_id)
    if identity_key is not None:
        dog_history_keys.append(identity_key)
    if not dog_history_keys:
        return
    distance_value = float(distance_m or 0)
    official_time = float(official_time_s or 0.0)
    speed = (distance_value / official_time) if distance_value > 0 and official_time > 0 else 0.0
    event = {
        "race_entry_id": int(race_entry_id or 0),
        "when": when,
        "finish": int(finish_position),
        "distance_m": int(distance_m or 0),
        "track_id": int(track_id),
        "trap_number": int(trap_number),
        "speed": speed,
        "sectional": float(sectional_s or 0.0),
        "sp_decimal": float(sp_decimal or 0.0),
    }
    for dog_history_key in dog_history_keys:
        dog_history[dog_history_key].append(event)
    if trainer_id:
        trainer_history[int(trainer_id)].append(event)
    if owner_id:
        owner_history[int(owner_id)].append(event)
    _update_summary_stats(global_summary, event)
    _update_summary_stats(track_summaries[int(track_id)], event)
    if int(distance_m or 0) > 0:
        _update_summary_stats(distance_summaries[int(distance_m or 0)], event)
    _update_summary_stats(trap_summaries[int(trap_number)], event)


def _build_targeted_race_examples(
    settings: Settings,
    max_runners: int,
    target_race_ids: set[int],
) -> list[RaceExample]:
    dog_history: dict[HistoryKey, list[dict[str, Any]]] = defaultdict(list)
    trainer_history: dict[int, list[dict[str, Any]]] = defaultdict(list)
    owner_history: dict[int, list[dict[str, Any]]] = defaultdict(list)
    global_summary = _new_summary_stats()
    track_summaries: dict[int, dict[str, float]] = defaultdict(_new_summary_stats)
    distance_summaries: dict[int, dict[str, float]] = defaultdict(_new_summary_stats)
    trap_summaries: dict[int, dict[str, float]] = defaultdict(_new_summary_stats)
    examples: list[RaceExample] = []

    def apply_history_group(group: dict[str, Any]) -> None:
        if (group["status"] or "").strip().lower() in {"canceled", "abandoned"}:
            return
        active_rows = [
            row for row in group["entries"] if not bool(row.vacant) and not bool(row.scratched)
        ]
        if not _entries_have_complete_finish_order(active_rows):
            return
        for row in active_rows:
            _record_entry_history(
                race_entry_id=row.race_entry_id,
                when=group["scheduled_start"],
                distance_m=group["distance_m"],
                track_id=group["track_id"],
                trap_number=row.trap_number,
                finish_position=row.finish_position,
                official_time_s=row.official_time_s,
                sectional_s=row.sectional_s,
                sp_decimal=row.sp_decimal,
                dog_id=row.dog_id,
                dog_name=row.dog_name,
                trainer_id=row.trainer_id,
                owner_id=row.owner_id,
                dog_history=dog_history,
                trainer_history=trainer_history,
                owner_history=owner_history,
                global_summary=global_summary,
                track_summaries=track_summaries,
                distance_summaries=distance_summaries,
                trap_summaries=trap_summaries,
            )

    with session_scope(settings) as session:
        target_statement = (
            select(Race)
            .options(
                selectinload(Race.track),
                selectinload(Race.entries).selectinload(RaceEntry.dog),
            )
            .where(Race.id.in_(sorted(target_race_ids)))
            .order_by(Race.scheduled_start.asc(), Race.id.asc())
        )
        target_races = list(session.scalars(target_statement).unique())
        if not target_races:
            return []

        track_ids = [
            int(track_id)
            for (track_id,) in session.execute(
                select(Race.track_id)
                .where(Race.track_id.is_not(None))
                .distinct()
                .order_by(Race.track_id.asc())
            )
        ]
        track_index_by_id = {
            track_id: index
            for index, track_id in enumerate(track_ids, start=1)
        }
        track_count = len(track_ids)
        max_target_start = max(race.scheduled_start for race in target_races)

        history_statement = (
            select(
                Race.id.label("race_id"),
                Race.scheduled_start.label("scheduled_start"),
                Race.distance_m.label("distance_m"),
                Race.track_id.label("track_id"),
                Race.status.label("status"),
                RaceEntry.id.label("race_entry_id"),
                RaceEntry.dog_id.label("dog_id"),
                RaceEntry.trap_number.label("trap_number"),
                RaceEntry.finish_position.label("finish_position"),
                RaceEntry.official_time_s.label("official_time_s"),
                RaceEntry.sectional_s.label("sectional_s"),
                RaceEntry.sp_decimal.label("sp_decimal"),
                RaceEntry.vacant.label("vacant"),
                RaceEntry.scratched.label("scratched"),
                Dog.name.label("dog_name"),
                Dog.trainer_id.label("trainer_id"),
                Dog.owner_id.label("owner_id"),
            )
            .join(RaceEntry, RaceEntry.race_id == Race.id)
            .outerjoin(Dog, Dog.id == RaceEntry.dog_id)
            .where(Race.scheduled_start <= max_target_start)
            .order_by(
                Race.scheduled_start.asc(),
                Race.id.asc(),
                RaceEntry.trap_number.asc(),
            )
        )
        history_groups: list[dict[str, Any]] = []
        current_group: dict[str, Any] | None = None
        for row in session.execute(history_statement):
            race_id = int(row.race_id)
            if current_group is None or int(current_group["race_id"]) != race_id:
                current_group = {
                    "race_id": race_id,
                    "scheduled_start": row.scheduled_start,
                    "distance_m": row.distance_m,
                    "track_id": row.track_id,
                    "status": row.status,
                    "entries": [],
                }
                history_groups.append(current_group)
            current_group["entries"].append(row)

        history_index = 0
        for race in target_races:
            race_sort_key = (race.scheduled_start, int(race.id))
            while history_index < len(history_groups):
                history_group = history_groups[history_index]
                history_sort_key = (
                    history_group["scheduled_start"],
                    int(history_group["race_id"]),
                )
                if history_sort_key >= race_sort_key:
                    break
                apply_history_group(history_group)
                history_index += 1

            if (race.status or "").strip().lower() in {"canceled", "abandoned"}:
                continue
            active_entries = [entry for entry in race.entries if not entry.vacant and not entry.scratched]
            if not active_entries:
                continue
            entries = sorted(active_entries, key=lambda item: item.trap_number)
            has_target = _entries_have_complete_finish_order(entries)
            if len(entries) != FIXED_RUNNER_COUNT:
                continue
            examples.append(
                _build_race_example(
                    race,
                    entries,
                    max_runners,
                    track_index_by_id,
                    track_count,
                    dog_history,
                    trainer_history,
                    owner_history,
                    global_summary,
                    track_summaries,
                    distance_summaries,
                    trap_summaries,
                    has_target,
                )
            )

    return examples


def build_race_examples(
    settings: Settings,
    max_runners: int | None = None,
    *,
    example_race_ids: set[int] | None = None,
) -> list[RaceExample]:
    max_runners = FIXED_RUNNER_COUNT
    init_database(settings)
    target_race_ids = {int(race_id) for race_id in example_race_ids} if example_race_ids is not None else None
    if target_race_ids is not None and not target_race_ids:
        return []
    if target_race_ids is not None:
        return _build_targeted_race_examples(settings, max_runners, target_race_ids)

    dog_history: dict[HistoryKey, list[dict[str, Any]]] = defaultdict(list)
    trainer_history: dict[int, list[dict[str, Any]]] = defaultdict(list)
    owner_history: dict[int, list[dict[str, Any]]] = defaultdict(list)
    global_summary = _new_summary_stats()
    track_summaries: dict[int, dict[str, float]] = defaultdict(_new_summary_stats)
    distance_summaries: dict[int, dict[str, float]] = defaultdict(_new_summary_stats)
    trap_summaries: dict[int, dict[str, float]] = defaultdict(_new_summary_stats)
    examples: list[RaceExample] = []

    with session_scope(settings) as session:
        statement = (
            select(Race)
            .options(
                selectinload(Race.track),
                selectinload(Race.entries).selectinload(RaceEntry.dog).selectinload(Dog.owner),
                selectinload(Race.entries).selectinload(RaceEntry.dog).selectinload(Dog.trainer),
            )
            .order_by(Race.scheduled_start.asc(), Race.id.asc())
        )
        races = list(session.scalars(statement).unique())
        track_ids = sorted(
            {
                int(race.track_id)
                for race in races
                if race.track_id is not None
            }
        )
        track_index_by_id = {
            track_id: index
            for index, track_id in enumerate(track_ids, start=1)
        }
        track_count = len(track_ids)

        for race in races:
            if (race.status or "").strip().lower() in {"canceled", "abandoned"}:
                continue
            active_entries = [entry for entry in race.entries if not entry.vacant and not entry.scratched]
            if not active_entries:
                continue
            entries = sorted(active_entries, key=lambda item: item.trap_number)
            has_target = _entries_have_complete_finish_order(entries)

            if len(entries) == FIXED_RUNNER_COUNT:
                examples.append(
                    _build_race_example(
                        race,
                        entries,
                        max_runners,
                        track_index_by_id,
                        track_count,
                        dog_history,
                        trainer_history,
                        owner_history,
                        global_summary,
                        track_summaries,
                        distance_summaries,
                        trap_summaries,
                        has_target,
                    )
                )

            if has_target:
                for entry in entries:
                    _record_entry_history(
                        race_entry_id=entry.id,
                        when=race.scheduled_start,
                        distance_m=race.distance_m,
                        track_id=race.track_id,
                        trap_number=entry.trap_number,
                        finish_position=entry.finish_position,
                        official_time_s=entry.official_time_s,
                        sectional_s=entry.sectional_s,
                        sp_decimal=entry.sp_decimal,
                        dog_id=entry.dog_id,
                        dog_name=entry.dog.name if entry.dog else None,
                        trainer_id=entry.dog.trainer_id if entry.dog else None,
                        owner_id=entry.dog.owner_id if entry.dog else None,
                        dog_history=dog_history,
                        trainer_history=trainer_history,
                        owner_history=owner_history,
                        global_summary=global_summary,
                        track_summaries=track_summaries,
                        distance_summaries=distance_summaries,
                        trap_summaries=trap_summaries,
                    )

    return examples


class PermutationScoringANN(nn.Module):
    def __init__(self, input_dim: int, hidden_size_1: int, hidden_size_2: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _model_state_dict_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _parameter_summaries(state_dict: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for name, tensor in state_dict.items():
        values = tensor.detach().cpu().float().reshape(-1)
        count = int(values.numel())
        if count:
            mean = float(values.mean().item())
            std = float(values.std(unbiased=False).item())
            minimum = float(values.min().item())
            maximum = float(values.max().item())
            abs_max = float(values.abs().max().item())
        else:
            mean = std = minimum = maximum = abs_max = 0.0
        summaries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "count": count,
                "mean": mean,
                "std": std,
                "min": minimum,
                "max": maximum,
                "abs_max": abs_max,
            }
        )
    return summaries


def _compute_scaler(examples: list[RaceExample]) -> Scaler:
    if not examples:
        raise ValueError("No race examples were available for scaling.")
    common_matrix = np.stack([example.common_features for example in examples])
    dog_rows = np.concatenate(
        [example.dog_features[example.mask.astype(bool)] for example in examples],
        axis=0,
    )
    common_mean = common_matrix.mean(axis=0)
    common_std = common_matrix.std(axis=0)
    common_std[common_std < 1e-6] = 1.0
    dog_mean = dog_rows.mean(axis=0)
    dog_std = dog_rows.std(axis=0)
    dog_std[dog_std < 1e-6] = 1.0
    return Scaler(
        common_mean=common_mean.astype(np.float32),
        common_std=common_std.astype(np.float32),
        dog_mean=dog_mean.astype(np.float32),
        dog_std=dog_std.astype(np.float32),
    )


def _split_examples(examples: list[RaceExample], validation_fraction: float) -> tuple[list[RaceExample], list[RaceExample]]:
    if len(examples) < 2:
        return examples, []
    cutoff = max(1, int(len(examples) * (1.0 - validation_fraction)))
    cutoff = min(cutoff, len(examples) - 1)
    return examples[:cutoff], examples[cutoff:]


def _candidate_orders(example: RaceExample, permutation_runner_limit: int) -> list[tuple[int, ...]]:
    active = _active_runner_indices(example)
    if len(active) != permutation_runner_limit:
        return []
    return [tuple(order) for order in itertools.permutations(active)]


def _score_candidate_orders(
    model: PermutationScoringANN,
    example: RaceExample,
    orders: list[tuple[int, ...]],
    scaler: Scaler,
    device: torch.device,
    batch_size: int,
    should_stop: Callable[[], None] | None = None,
) -> list[float]:
    if not orders:
        return []

    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(orders), batch_size):
            if should_stop is not None:
                should_stop()
            batch_orders = orders[start : start + batch_size]
            batch_features = _flatten_ordered_feature_matrix(example, batch_orders, scaler)
            batch_tensor = torch.tensor(batch_features, dtype=torch.float32, device=device)
            batch_scores = model(batch_tensor).detach().cpu().tolist()
            scores.extend(float(score) for score in batch_scores)
    return scores


def _permutation_race_metrics(
    predicted_order: tuple[int, ...],
    actual_order: list[int],
) -> dict[str, float]:
    predicted_rank = {runner_index: rank for rank, runner_index in enumerate(predicted_order, start=1)}
    actual_rank = {runner_index: rank for rank, runner_index in enumerate(actual_order, start=1)}

    return {
        "winner_accuracy": 1.0 if predicted_order[0] == actual_order[0] else 0.0,
        "top3_set_accuracy": 1.0 if set(predicted_order[:3]) == set(actual_order[:3]) else 0.0,
        "exact_order_accuracy": 1.0 if list(predicted_order) == actual_order else 0.0,
        "mean_abs_rank_error": statistics.fmean(
            abs(predicted_rank[runner_index] - actual_rank[runner_index])
            for runner_index in actual_order
        ),
    }


def _runner_trap_number(example: RaceExample, runner_index: int) -> int:
    if runner_index < 0 or runner_index >= example.dog_features.shape[0]:
        return 0
    trap_norm = float(example.dog_features[runner_index][0])
    trap_number = int(round(trap_norm * 8.0))
    if 1 <= trap_number <= FIXED_RUNNER_COUNT:
        return trap_number
    return 0


def _empty_trap_counts() -> dict[str, int]:
    return {str(trap_number): 0 for trap_number in range(1, FIXED_RUNNER_COUNT + 1)}


def _evaluate_permutation_model(
    model: PermutationScoringANN,
    examples: list[RaceExample],
    scaler: Scaler,
    config: TrainingConfig,
    device: torch.device,
    should_stop: Callable[[], None] | None = None,
) -> dict[str, Any]:
    model.eval()
    losses = []
    metrics = []
    actual_winner_counts = _empty_trap_counts()
    predicted_winner_counts = _empty_trap_counts()
    correct_winner_counts = _empty_trap_counts()

    for example in examples:
        if should_stop is not None:
            should_stop()
        if not _example_has_complete_target_order(example):
            continue
        actual_order = _actual_order_indices(example)
        if not actual_order:
            continue

        orders = _candidate_orders(example, config.permutation_runner_limit)
        if not orders:
            continue

        scores = _score_candidate_orders(
            model,
            example,
            orders,
            scaler,
            device,
            batch_size=max(64, config.batch_size),
            should_stop=should_stop,
        )
        if not scores:
            continue

        actual_order_tuple = tuple(actual_order)
        try:
            actual_order_index = orders.index(actual_order_tuple)
        except ValueError:
            continue
        logits = -torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor([actual_order_index], dtype=torch.long)
        losses.append(float(nn.functional.cross_entropy(logits, target_tensor).item()))

        best_index = min(range(len(scores)), key=lambda index: scores[index])
        predicted_order = orders[best_index]
        metrics.append(_permutation_race_metrics(predicted_order, actual_order))

        actual_winner_trap = _runner_trap_number(example, actual_order[0])
        predicted_winner_trap = _runner_trap_number(example, predicted_order[0])
        if actual_winner_trap:
            actual_winner_counts[str(actual_winner_trap)] += 1
        if predicted_winner_trap:
            predicted_winner_counts[str(predicted_winner_trap)] += 1
        if predicted_order[0] == actual_order[0] and actual_winner_trap:
            correct_winner_counts[str(actual_winner_trap)] += 1

    if not losses or not metrics:
        return {
            "loss": 0.0,
            "winner_accuracy": 0.0,
            "top3_set_accuracy": 0.0,
            "exact_order_accuracy": 0.0,
            "mean_abs_rank_error": 0.0,
            "winner_trap_diagnostic": {
                "actual_winner_counts": _empty_trap_counts(),
                "predicted_winner_counts": _empty_trap_counts(),
                "winner_accuracy_by_actual_trap": {
                    str(trap_number): None for trap_number in range(1, FIXED_RUNNER_COUNT + 1)
                },
            },
        }

    winner_accuracy_by_actual_trap = {
        trap_key: (
            correct_winner_counts[trap_key] / actual_winner_counts[trap_key]
            if actual_winner_counts[trap_key] > 0
            else None
        )
        for trap_key in actual_winner_counts
    }

    return {
        "loss": statistics.fmean(losses),
        "winner_accuracy": statistics.fmean(metric["winner_accuracy"] for metric in metrics),
        "top3_set_accuracy": statistics.fmean(metric["top3_set_accuracy"] for metric in metrics),
        "exact_order_accuracy": statistics.fmean(metric["exact_order_accuracy"] for metric in metrics),
        "mean_abs_rank_error": statistics.fmean(metric["mean_abs_rank_error"] for metric in metrics),
        "winner_trap_diagnostic": {
            "actual_winner_counts": actual_winner_counts,
            "predicted_winner_counts": predicted_winner_counts,
            "winner_accuracy_by_actual_trap": winner_accuracy_by_actual_trap,
        },
    }


def train_model(
    settings: Settings,
    config: TrainingConfig,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    settings.ensure_directories()
    init_database(settings)
    with session_scope(settings) as session:
        existing_run = session.scalar(
            select(TrainingRun)
            .where(TrainingRun.status == "running", TrainingRun.finished_at.is_(None))
            .order_by(TrainingRun.started_at.desc())
            .limit(1)
        )
    if existing_run is not None:
        started_at_text = existing_run.started_at.isoformat() if existing_run.started_at else "an unknown time"
        raise ValueError(
            "Training is already running "
            f"(run key {existing_run.run_key}, started {started_at_text}). "
            "Wait for it to finish or mark it interrupted before starting another run."
        )
    config.max_runners = FIXED_RUNNER_COUNT
    config.permutation_runner_limit = FIXED_RUNNER_COUNT
    if config.model_type != "permutation":
        raise ValueError(
            f"Unsupported model_type '{config.model_type}'. Only 'permutation' is supported right now."
        )

    run_key = f"train-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    training_log_path = _logs_dir(settings) / f"{run_key}.jsonl"
    logger = _TrainingRunLogger(training_log_path)
    started_at = datetime.now(timezone.utc)
    progress_error: str | None = None

    with session_scope(settings) as session:
        training_run = TrainingRun(
            run_key=run_key,
            status="running",
            config_json=asdict(config),
        )
        session.add(training_run)
        session.flush()
        training_run_id = training_run.id

    def emit_progress(event: dict[str, Any]) -> None:
        nonlocal progress, progress_error
        if progress is None:
            return
        try:
            progress(event)
        except Exception as exc:
            progress_error = str(exc)
            logger.log(
                "progress_callback_failed",
                run_key=run_key,
                error=progress_error,
            )
            progress = None

    logger.log(
        "training_run_row_created",
        run_key=run_key,
        training_run_id=training_run_id,
    )
    completion_note: str | None = None

    def mark_run_interrupted() -> None:
        elapsed_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
        try:
            logger.log(
                "process_exit",
                run_key=run_key,
                elapsed_seconds=elapsed_seconds,
            )
        except OSError:
            pass
        try:
            with session_scope(settings) as session:
                training_run = session.scalar(select(TrainingRun).where(TrainingRun.id == training_run_id))
                if training_run is None or training_run.status != "running" or training_run.finished_at is not None:
                    return
                training_run.status = "interrupted"
                training_run.finished_at = datetime.now(timezone.utc)
                training_run.error_text = "Training process exited before final artifacts were written."
            logger.log(
                "training_run_interrupted",
                run_key=run_key,
                training_run_id=training_run_id,
                elapsed_seconds=elapsed_seconds,
            )
        except Exception as exc:
            logger.log(
                "training_run_interrupt_persist_failed",
                run_key=run_key,
                error=str(exc),
                elapsed_seconds=elapsed_seconds,
            )

    def persist_training_failure(exc: Exception) -> None:
        interrupted = isinstance(exc, TrainingStopRequested)
        try:
            with session_scope(settings) as session:
                training_run = session.scalar(select(TrainingRun).where(TrainingRun.id == training_run_id))
                if training_run is not None:
                    training_run.status = "interrupted" if interrupted else "failed"
                    training_run.finished_at = datetime.now(timezone.utc)
                    training_run.error_text = str(exc)
                    if interrupted and getattr(exc, "recovery_artifact_path", None):
                        training_run.artifact_path = str(exc.recovery_artifact_path)
        except Exception as save_exc:
            exc.add_note(f"Training run status could not be persisted: {save_exc}")
            logger.log(
                "training_run_status_persist_failed",
                run_key=run_key,
                error=str(save_exc),
            )
        logger.log(
            "training_interrupted" if interrupted else "training_failed",
            run_key=run_key,
            error=str(exc),
            elapsed_seconds=(datetime.now(timezone.utc) - started_at).total_seconds(),
        )
        emit_progress(
            {
                "event": "interrupted" if interrupted else "failed",
                "run_key": run_key,
                "error": str(exc),
                "elapsed_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
            }
        )

    atexit.register(mark_run_interrupted)

    try:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        examples = build_race_examples(settings, max_runners=config.max_runners)
        completed = _eligible_examples(
            examples,
            runner_limit=config.permutation_runner_limit,
            require_target=True,
        )
        data_summary = {
            **_training_data_summary(examples, config.permutation_runner_limit),
            "eligible_completed_race_count": len(completed),
        }
        logger.log(
            "training_data_prepared",
            run_key=run_key,
            **data_summary,
        )
        if len(completed) < config.min_completed_races:
            skipped = data_summary["missing_or_invalid_finish_order_races"]
            raise ValueError(
                f"Need at least {config.min_completed_races} completed races with complete finish orders "
                f"to train, found {len(completed)}. "
                f"Skipped {skipped} race(s) with missing or invalid finish positions."
            )

        train_examples, validation_examples = _split_examples(completed, config.validation_fraction)
        with session_scope(settings) as session:
            training_run = session.scalar(select(TrainingRun).where(TrainingRun.id == training_run_id))
            if training_run is not None:
                training_run.train_race_count = len(train_examples)
                training_run.validation_race_count = len(validation_examples)
        scaler = _compute_scaler(train_examples)
        logger.log(
            "training_split_prepared",
            run_key=run_key,
            train_race_count=len(train_examples),
            validation_race_count=len(validation_examples),
        )

        resume_artifact: dict[str, Any] | None = None
        resumed_checkpoint_epoch = 0
        saved_last_epoch_summary: dict[str, Any] | None = None
        saved_best_epoch_summary: dict[str, Any] | None = None
        if config.resume_from_artifact:
            resume_artifact = _load_artifact(config.resume_from_artifact)
            if resume_artifact.get("model_type") != "permutation":
                raise ValueError("Only permutation artifacts can be used for resume training.")
            if resume_artifact.get("layout_version") != MODEL_LAYOUT_VERSION:
                raise ValueError(
                    "This artifact uses the legacy permutation-regression layout. "
                    "Start a fresh listwise training run with resume disabled."
                )
            saved_last_epoch_summary = (
                resume_artifact.get("last_epoch")
                if isinstance(resume_artifact.get("last_epoch"), dict)
                else None
            )
            saved_best_epoch_summary = (
                resume_artifact.get("best_epoch")
                if isinstance(resume_artifact.get("best_epoch"), dict)
                else None
            )
            resumed_checkpoint_epoch = int(
                (
                    saved_best_epoch_summary.get("epoch")
                    if saved_best_epoch_summary is not None
                    and (
                        resume_artifact.get("best_state_dict") is not None
                        or resume_artifact.get("artifact_state") == "best_validation"
                    )
                    else None
                )
                or resume_artifact.get("checkpoint_epoch")
                or (saved_last_epoch_summary.get("epoch") if saved_last_epoch_summary is not None else 0)
                or 0
            )
            if resumed_checkpoint_epoch >= config.epochs:
                raise ValueError(
                    f"Resume checkpoint is already at epoch {resumed_checkpoint_epoch}, which meets or exceeds the requested total of {config.epochs} epochs."
                )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PermutationScoringANN(
            input_dim=len(COMMON_FEATURE_NAMES) + (FIXED_RUNNER_COUNT * len(DOG_FEATURE_NAMES)),
            hidden_size_1=config.hidden_size_1,
            hidden_size_2=config.hidden_size_2,
            dropout=config.dropout,
        ).to(device)
        resume_uses_best_state = False
        if resume_artifact is not None:
            if resume_artifact.get("best_state_dict") is not None:
                resume_state_dict = resume_artifact.get("best_state_dict")
                resume_uses_best_state = True
            elif resume_artifact.get("artifact_state") == "best_validation" and resume_artifact.get("state_dict") is not None:
                resume_state_dict = resume_artifact.get("state_dict")
                resume_uses_best_state = True
            else:
                resume_state_dict = resume_artifact.get("state_dict") or resume_artifact.get("latest_state_dict")
            if resume_state_dict is None:
                raise ValueError("Resume artifact does not contain model weights.")
            model.load_state_dict(resume_state_dict)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if resume_artifact is not None:
            if resume_uses_best_state and resume_artifact.get("best_optimizer_state_dict"):
                optimizer.load_state_dict(resume_artifact["best_optimizer_state_dict"])
            elif not resume_uses_best_state and resume_artifact.get("optimizer_state_dict"):
                optimizer.load_state_dict(resume_artifact["optimizer_state_dict"])

        history: list[dict[str, Any]] = []
        if resume_artifact is not None:
            saved_history = resume_artifact.get("history")
            if isinstance(saved_history, list):
                history = [
                    item
                    for item in saved_history
                    if isinstance(item, dict) and int(item.get("epoch") or 0) <= resumed_checkpoint_epoch
                ]
            elif saved_last_epoch_summary is not None and int(saved_last_epoch_summary.get("epoch") or 0) <= resumed_checkpoint_epoch:
                history = [saved_last_epoch_summary]
        best_validation_loss = math.inf
        best_state = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        epochs_without_validation_improvement = 0
        if resume_artifact is not None:
            saved_best_state = resume_artifact.get("best_state_dict")
            if saved_best_state is not None:
                best_state = copy.deepcopy(saved_best_state)
            saved_best_optimizer_state = resume_artifact.get("best_optimizer_state_dict")
            if resume_uses_best_state and saved_best_optimizer_state is not None:
                best_optimizer_state = copy.deepcopy(saved_best_optimizer_state)
            elif not resume_uses_best_state:
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            if history:
                best_validation_loss = min(float(item["validation_loss"]) for item in history)
                best_history_index = max(
                    index
                    for index, item in enumerate(history)
                    if float(item["validation_loss"]) <= best_validation_loss
                )
                epochs_without_validation_improvement = len(history) - 1 - best_history_index
            elif saved_best_epoch_summary is not None:
                best_validation_loss = float(saved_best_epoch_summary["validation_loss"])
            elif saved_last_epoch_summary is not None:
                best_validation_loss = float(saved_last_epoch_summary["validation_loss"])
            elif saved_best_state is None:
                # Older recovery checkpoints only stored the latest weights, so treat the resumed weights as the current best.
                best_validation_loss = math.inf
        recovery_artifact_path = settings.models_dir / f"{run_key}-recovery.pt"
        live_weight_snapshot_path = training_weight_snapshot_path(settings, run_key)

        logger.log(
            "training_started",
            run_key=run_key,
            pid=os.getpid(),
            python=sys.version.split()[0],
            platform=platform.platform(),
            cwd=str(Path.cwd()),
            config=asdict(config),
            train_race_count=len(train_examples),
            validation_race_count=len(validation_examples),
            data_summary=data_summary,
            resume_from_artifact=config.resume_from_artifact,
            resumed_checkpoint_epoch=resumed_checkpoint_epoch,
            weight_snapshot_path=str(live_weight_snapshot_path),
        )

        def save_live_weight_snapshot(
            *,
            stage: str,
            epoch: int | None = None,
            batch_index: int | None = None,
            total_batches: int | None = None,
            batch_loss: float | None = None,
            rolling_batch_loss: float | None = None,
            validation_loss: float | None = None,
        ) -> str | None:
            try:
                state_snapshot = _model_state_dict_snapshot(model)
                parameter_summaries = _parameter_summaries(state_snapshot)
                payload = {
                    "run_key": run_key,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "stage": stage,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "batch_loss": batch_loss,
                    "rolling_batch_loss": rolling_batch_loss,
                    "validation_loss": validation_loss,
                    "model_type": config.model_type,
                    "layout_version": MODEL_LAYOUT_VERSION,
                    "common_feature_names": COMMON_FEATURE_NAMES,
                    "dog_feature_names": DOG_FEATURE_NAMES,
                    "config": asdict(config),
                    "state_dict": state_snapshot,
                    "parameter_summaries": parameter_summaries,
                    "total_parameters": sum(int(item["count"]) for item in parameter_summaries),
                }
                temporary_path = live_weight_snapshot_path.with_suffix(".tmp")
                torch.save(payload, temporary_path)
                temporary_path.replace(live_weight_snapshot_path)
                return str(live_weight_snapshot_path)
            except Exception as exc:
                logger.log(
                    "weight_snapshot_failed",
                    run_key=run_key,
                    stage=stage,
                    epoch=epoch,
                    batch_index=batch_index,
                    error=str(exc),
                )
                return None

        def save_recovery_checkpoint(
            *,
            epoch: int,
            summary: dict[str, Any],
            best_epoch: dict[str, Any] | None,
            last_epoch: dict[str, Any] | None,
        ) -> None:
            latest_state = copy.deepcopy(model.state_dict())
            primary_state = best_state if best_epoch is not None else latest_state
            checkpoint_epoch = int(best_epoch["epoch"]) if isinstance(best_epoch, dict) and "epoch" in best_epoch else epoch
            torch.save(
                {
                    "run_key": run_key,
                    "checkpoint_epoch": checkpoint_epoch,
                    "is_recovery_checkpoint": True,
                    "model_type": config.model_type,
                    "layout_version": MODEL_LAYOUT_VERSION,
                    "common_feature_names": COMMON_FEATURE_NAMES,
                    "dog_feature_names": DOG_FEATURE_NAMES,
                    "config": asdict(config),
                    "state_dict": primary_state,
                    "latest_state_dict": latest_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_optimizer_state_dict": best_optimizer_state if best_epoch is not None else None,
                    "scaler_common_mean": scaler.common_mean.tolist(),
                    "scaler_common_std": scaler.common_std.tolist(),
                    "scaler_dog_mean": scaler.dog_mean.tolist(),
                    "scaler_dog_std": scaler.dog_std.tolist(),
                    "summary": summary,
                    "best_state_dict": best_state if best_epoch is not None else None,
                    "artifact_state": "best_validation" if best_epoch is not None else "latest_no_validation",
                    "best_epoch": best_epoch,
                    "last_epoch": last_epoch,
                    "data_summary": data_summary,
                    "history": history,
                },
                recovery_artifact_path,
            )
            logger.log(
                "recovery_checkpoint_saved",
                run_key=run_key,
                epoch=epoch,
                recovery_artifact_path=str(recovery_artifact_path),
                summary_loss=summary.get("loss"),
            )

        def raise_if_stop_requested(
            *,
            epoch: int,
            stage: str,
            batch_index: int | None = None,
            epoch_samples_seen: int | None = None,
            epoch_samples_total: int | None = None,
            summary_loss: float | None = None,
        ) -> None:
            stop_request = read_training_stop_request(settings, run_key)
            if stop_request is None:
                return
            stop_reason = _format_stop_request_reason(
                stop_request,
                stage=stage,
                epoch=epoch,
                recovery_artifact_path=recovery_artifact_path,
            )
            best_epoch_summary = (
                min(history, key=lambda item: float(item["validation_loss"]))
                if history
                else None
            )
            last_epoch_summary = history[-1] if history else None
            save_recovery_checkpoint(
                epoch=max(epoch - 1, resumed_checkpoint_epoch),
                summary={
                    "loss": float(summary_loss or 0.0),
                    "stop_stage": stage,
                    "stopped_during_epoch": epoch,
                    "stopped_after_batch": batch_index,
                    "epoch_samples_seen": epoch_samples_seen,
                    "epoch_samples_total": epoch_samples_total,
                },
                best_epoch=best_epoch_summary,
                last_epoch=last_epoch_summary,
            )
            logger.log(
                "stop_requested_detected",
                run_key=run_key,
                epoch=epoch,
                batch_index=batch_index,
                stage=stage,
                requested_at=stop_request.get("requested_at"),
                requested_by=stop_request.get("requested_by"),
                reason=stop_request.get("reason"),
                stop_reason=stop_reason,
                recovery_artifact_path=str(recovery_artifact_path),
            )
            clear_training_stop_request(settings, run_key)
            raise TrainingStopRequested(
                stop_reason,
                recovery_artifact_path=str(recovery_artifact_path),
            )

        initial_weight_snapshot_path = save_live_weight_snapshot(
            stage="training_started",
            epoch=resumed_checkpoint_epoch,
        )
        emit_progress(
            {
                "event": "start",
                "run_key": run_key,
                "model_type": config.model_type,
                "total_epochs": config.epochs,
                "train_race_count": len(train_examples),
                "validation_race_count": len(validation_examples),
                "data_summary": data_summary,
                "resume_from_artifact": config.resume_from_artifact,
                "weight_snapshot_path": initial_weight_snapshot_path,
            }
        )
    except Exception as exc:
        persist_training_failure(exc)
        raise

    try:
        permutations_seen_total = 0
        start_epoch = resumed_checkpoint_epoch + 1
        for epoch in range(start_epoch, config.epochs + 1):
            logger.log(
                "epoch_started",
                run_key=run_key,
                epoch=epoch,
                total_epochs=config.epochs,
            )
            model.train()
            candidate_count = min(
                math.factorial(config.permutation_runner_limit),
                max(2, config.permutations_per_race + 1),
            )

            batch_losses = []
            epoch_samples_seen = 0
            epoch_samples_total = len(train_examples) * candidate_count
            emit_interval = max(8192, config.batch_size * 64)
            next_emit_at = emit_interval
            total_batches = math.ceil(len(train_examples) / config.batch_size)
            for batch_index, (race_count, order_count, batch_features) in enumerate(
                _iter_training_order_batches(
                    train_examples,
                    scaler,
                    config,
                    seed=config.seed + epoch,
                ),
                start=1,
            ):
                features = torch.tensor(batch_features, dtype=torch.float32, device=device)
                targets = torch.zeros(race_count, dtype=torch.long, device=device)

                optimizer.zero_grad()
                scores = model(features).reshape(race_count, order_count)
                loss = nn.functional.cross_entropy(-scores, targets)
                loss.backward()
                optimizer.step()
                batch_loss = float(loss.item())
                batch_losses.append(batch_loss)

                batch_sample_count = int(race_count * order_count)
                epoch_samples_seen += batch_sample_count
                permutations_seen_total += batch_sample_count

                if progress is not None and (epoch_samples_seen >= next_emit_at or batch_index == total_batches):
                    rolling_window = batch_losses[-20:]
                    rolling_batch_loss = statistics.fmean(rolling_window)
                    elapsed_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
                    weight_snapshot_path = save_live_weight_snapshot(
                        stage="batch",
                        epoch=epoch,
                        batch_index=batch_index,
                        total_batches=total_batches,
                        batch_loss=batch_loss,
                        rolling_batch_loss=rolling_batch_loss,
                    )
                    logger.log(
                        "batch_heartbeat",
                        run_key=run_key,
                        epoch=epoch,
                        batch_index=batch_index,
                        total_batches=total_batches,
                        batch_loss=batch_loss,
                        rolling_batch_loss=rolling_batch_loss,
                        epoch_samples_seen=epoch_samples_seen,
                        epoch_samples_total=epoch_samples_total,
                        samples_seen_total=permutations_seen_total,
                        elapsed_seconds=elapsed_seconds,
                        weight_snapshot_path=weight_snapshot_path,
                    )
                    emit_progress(
                        {
                            "event": "batch",
                            "run_key": run_key,
                            "epoch": epoch,
                            "total_epochs": config.epochs,
                            "batch_index": batch_index,
                            "total_batches": total_batches,
                            "batch_loss": batch_loss,
                            "rolling_batch_loss": rolling_batch_loss,
                            "epoch_samples_seen": epoch_samples_seen,
                            "epoch_samples_total": epoch_samples_total,
                            "samples_seen_total": permutations_seen_total,
                            "elapsed_seconds": elapsed_seconds,
                            "weight_snapshot_path": weight_snapshot_path,
                        }
                    )
                    next_emit_at += emit_interval

                raise_if_stop_requested(
                    epoch=epoch,
                    stage="training_batch",
                    batch_index=batch_index,
                    epoch_samples_seen=epoch_samples_seen,
                    epoch_samples_total=epoch_samples_total,
                    summary_loss=statistics.fmean(batch_losses) if batch_losses else 0.0,
                )

            if not batch_losses:
                raise ValueError(
                    "No permutation training batches were generated. Check runner counts and completed race data."
                )

            train_loss = statistics.fmean(batch_losses) if batch_losses else 0.0
            train_metrics = _evaluate_permutation_model(
                model,
                train_examples,
                scaler,
                config,
                device,
                should_stop=lambda: raise_if_stop_requested(
                    epoch=epoch,
                    stage="train_evaluation",
                    epoch_samples_seen=epoch_samples_seen,
                    epoch_samples_total=epoch_samples_total,
                    summary_loss=train_loss,
                ),
            )
            validation_metrics = (
                _evaluate_permutation_model(
                    model,
                    validation_examples,
                    scaler,
                    config,
                    device,
                    should_stop=lambda: raise_if_stop_requested(
                        epoch=epoch,
                        stage="validation_evaluation",
                        epoch_samples_seen=epoch_samples_seen,
                        epoch_samples_total=epoch_samples_total,
                        summary_loss=train_loss,
                    ),
                )
                if validation_examples
                else train_metrics
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_winner_accuracy": train_metrics["winner_accuracy"],
                    "train_exact_order_accuracy": train_metrics["exact_order_accuracy"],
                    "validation_loss": validation_metrics["loss"],
                    "validation_winner_accuracy": validation_metrics["winner_accuracy"],
                    "validation_top3_set_accuracy": validation_metrics["top3_set_accuracy"],
                    "validation_exact_order_accuracy": validation_metrics["exact_order_accuracy"],
                    "validation_mean_abs_rank_error": validation_metrics["mean_abs_rank_error"],
                    "winner_trap_diagnostic": validation_metrics["winner_trap_diagnostic"],
                }
            )
            validation_loss = float(validation_metrics["loss"])
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                epochs_without_validation_improvement = 0
            else:
                epochs_without_validation_improvement += 1

            best_epoch_summary = min(history, key=lambda item: float(item["validation_loss"]))
            last_epoch_summary = history[-1]
            save_recovery_checkpoint(
                epoch=epoch,
                summary=validation_metrics,
                best_epoch=best_epoch_summary,
                last_epoch=last_epoch_summary,
            )
            elapsed_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
            weight_snapshot_path = save_live_weight_snapshot(
                stage="epoch",
                epoch=epoch,
                validation_loss=float(validation_metrics["loss"]),
            )
            logger.log(
                "epoch_completed",
                run_key=run_key,
                epoch=epoch,
                total_epochs=config.epochs,
                samples=epoch_samples_total,
                train_loss=train_loss,
                validation_loss=validation_metrics["loss"],
                validation_winner_accuracy=validation_metrics["winner_accuracy"],
                winner_trap_diagnostic=validation_metrics["winner_trap_diagnostic"],
                best_validation_loss=best_validation_loss,
                elapsed_seconds=elapsed_seconds,
                weight_snapshot_path=weight_snapshot_path,
            )
            emit_progress(
                {
                    "event": "epoch",
                    "run_key": run_key,
                    "epoch": epoch,
                    "total_epochs": config.epochs,
                    "samples": epoch_samples_total,
                    "train_loss": train_loss,
                    "train_winner_accuracy": train_metrics["winner_accuracy"],
                    "train_exact_order_accuracy": train_metrics["exact_order_accuracy"],
                    "validation_loss": validation_metrics["loss"],
                    "validation_winner_accuracy": validation_metrics["winner_accuracy"],
                    "validation_top3_set_accuracy": validation_metrics["top3_set_accuracy"],
                    "validation_exact_order_accuracy": validation_metrics["exact_order_accuracy"],
                    "validation_mean_abs_rank_error": validation_metrics["mean_abs_rank_error"],
                    "winner_trap_diagnostic": validation_metrics["winner_trap_diagnostic"],
                    "best_validation_loss": best_validation_loss,
                    "elapsed_seconds": elapsed_seconds,
                    "weight_snapshot_path": weight_snapshot_path,
                }
            )

            if (
                config.early_stopping_patience > 0
                and epochs_without_validation_improvement >= config.early_stopping_patience
            ):
                completion_note = (
                    f"Training stopped after epoch {epoch} of {config.epochs} because validation loss "
                    f"did not improve for {epochs_without_validation_improvement} consecutive epoch(s) "
                    f"(patience {config.early_stopping_patience})."
                )
                logger.log(
                    "early_stopping_triggered",
                    run_key=run_key,
                    epoch=epoch,
                    patience=config.early_stopping_patience,
                    best_validation_loss=best_validation_loss,
                    epochs_without_validation_improvement=epochs_without_validation_improvement,
                    stop_reason=completion_note,
                )
                emit_progress(
                    {
                        "event": "early_stopping",
                        "run_key": run_key,
                        "epoch": epoch,
                        "patience": config.early_stopping_patience,
                        "best_validation_loss": best_validation_loss,
                        "epochs_without_validation_improvement": epochs_without_validation_improvement,
                        "stop_reason": completion_note,
                    }
                )
                break

        latest_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        final_validation_metrics = (
            _evaluate_permutation_model(
                model,
                validation_examples,
                scaler,
                config,
                device,
            )
            if validation_examples
            else _evaluate_permutation_model(
                model,
                train_examples,
                scaler,
                config,
                device,
            )
        )
        best_epoch_summary = (
            min(history, key=lambda item: float(item["validation_loss"]))
            if history
            else None
        )
        last_epoch_summary = history[-1] if history else None
        final_weight_snapshot_path = save_live_weight_snapshot(
            stage="complete",
            epoch=int(last_epoch_summary["epoch"]) if isinstance(last_epoch_summary, dict) and "epoch" in last_epoch_summary else None,
            validation_loss=float(final_validation_metrics["loss"]),
        )

        artifact_path = settings.models_dir / f"{run_key}.pt"
        report_path = settings.reports_dir / f"{run_key}.json"
        logger.log(
            "saving_final_artifacts",
            run_key=run_key,
            artifact_path=str(artifact_path),
            report_path=str(report_path),
        )
        torch.save(
            {
                "run_key": run_key,
                "checkpoint_epoch": int(best_epoch_summary["epoch"]) if isinstance(best_epoch_summary, dict) and "epoch" in best_epoch_summary else 0,
                "model_type": config.model_type,
                "layout_version": MODEL_LAYOUT_VERSION,
                "common_feature_names": COMMON_FEATURE_NAMES,
                "dog_feature_names": DOG_FEATURE_NAMES,
                "config": asdict(config),
                "state_dict": model.state_dict(),
                "latest_state_dict": latest_state,
                "best_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_optimizer_state_dict": best_optimizer_state,
                "artifact_state": "best_validation",
                "scaler_common_mean": scaler.common_mean.tolist(),
                "scaler_common_std": scaler.common_std.tolist(),
                "scaler_dog_mean": scaler.dog_mean.tolist(),
                "scaler_dog_std": scaler.dog_std.tolist(),
                "summary": final_validation_metrics,
                "best_epoch": best_epoch_summary,
                "last_epoch": last_epoch_summary,
                "history": history,
                "data_summary": data_summary,
                "training_log_path": str(training_log_path),
                "weight_snapshot_path": final_weight_snapshot_path,
            },
            artifact_path,
        )
        report = {
            "run_key": run_key,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "history": history,
            "summary": final_validation_metrics,
            "best_epoch": best_epoch_summary,
            "last_epoch": last_epoch_summary,
            "config": asdict(config),
            "data_summary": data_summary,
            "stop_reason": completion_note,
            "progress_error": progress_error,
            "training_log_path": str(training_log_path),
            "weight_snapshot_path": final_weight_snapshot_path,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.log(
            "final_artifacts_saved",
            run_key=run_key,
            artifact_path=str(artifact_path),
            report_path=str(report_path),
            weight_snapshot_path=final_weight_snapshot_path,
        )

        with session_scope(settings) as session:
            training_run = session.scalar(select(TrainingRun).where(TrainingRun.id == training_run_id))
            if training_run is None:
                raise ValueError(f"Training run {training_run_id} was not found when saving completion metadata.")
            training_run.status = "completed"
            training_run.finished_at = datetime.now(timezone.utc)
            training_run.metrics_json = final_validation_metrics
            training_run.artifact_path = str(artifact_path)
            training_run.report_path = str(report_path)
            training_run.error_text = completion_note
        logger.log(
            "training_run_completed",
            run_key=run_key,
            training_run_id=training_run_id,
            artifact_path=str(artifact_path),
            report_path=str(report_path),
            training_log_path=str(training_log_path),
            weight_snapshot_path=final_weight_snapshot_path,
            elapsed_seconds=(datetime.now(timezone.utc) - started_at).total_seconds(),
        )

        emit_progress(
            {
                "event": "complete",
                "run_key": run_key,
                "summary": final_validation_metrics,
                "best_epoch": best_epoch_summary,
                "last_epoch": last_epoch_summary,
                "stop_reason": completion_note,
                "artifact_path": str(artifact_path),
                "report_path": str(report_path),
                "training_log_path": str(training_log_path),
                "weight_snapshot_path": final_weight_snapshot_path,
                "elapsed_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
            }
        )

        return {
            "run_key": run_key,
            "artifact_path": str(artifact_path),
            "recovery_artifact_path": str(recovery_artifact_path),
            "report_path": str(report_path),
            "training_log_path": str(training_log_path),
            "weight_snapshot_path": final_weight_snapshot_path,
            "summary": final_validation_metrics,
            "best_epoch": best_epoch_summary,
            "last_epoch": last_epoch_summary,
            "history": history,
            "data_summary": data_summary,
            "stop_reason": completion_note,
            "progress_error": progress_error,
        }
    except Exception as exc:
        persist_training_failure(exc)
        raise


def _load_artifact(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def predict_upcoming_races(
    settings: Settings,
    artifact_path: str | Path | None = None,
    race_keys: list[str] | None = None,
    track_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    settings.ensure_directories()
    init_database(settings)

    with session_scope(settings) as session:
        training_run = None
        if artifact_path is None:
            training_run = latest_usable_training_run(session)
            if training_run is None or not training_run.artifact_path:
                raise ValueError("No usable training run with an artifact was found.")
            artifact_path = training_run.artifact_path
        artifact = _load_artifact(artifact_path)
        config = TrainingConfig(**artifact["config"])
        config.max_runners = FIXED_RUNNER_COUNT
        config.permutation_runner_limit = FIXED_RUNNER_COUNT
        if artifact.get("model_type") != "permutation":
            raise ValueError("This prediction path currently supports permutation artifacts only.")
        if artifact.get("layout_version") != MODEL_LAYOUT_VERSION:
            raise ValueError(
                "The latest saved artifact uses the legacy permutation-regression layout. "
                "Train a fresh listwise six-dog model before predicting."
            )
        scaler = Scaler(
            common_mean=np.array(artifact["scaler_common_mean"], dtype=np.float32),
            common_std=np.array(artifact["scaler_common_std"], dtype=np.float32),
            dog_mean=np.array(artifact["scaler_dog_mean"], dtype=np.float32),
            dog_std=np.array(artifact["scaler_dog_std"], dtype=np.float32),
        )
        model = PermutationScoringANN(
            input_dim=len(artifact["common_feature_names"]) + (FIXED_RUNNER_COUNT * len(artifact["dog_feature_names"])),
            hidden_size_1=config.hidden_size_1,
            hidden_size_2=config.hidden_size_2,
            dropout=config.dropout,
        )
        prediction_state_dict = artifact.get("best_state_dict") or artifact["state_dict"]
        model.load_state_dict(prediction_state_dict)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        prediction_cutoff = datetime.now(timezone.utc)

        eligible_rows = [
            {
                "race_id": int(race_id),
                "race_key": str(race_key),
                "track_name": str(track_name),
            }
            for race_id, race_key, track_name in session.execute(
                select(Race.id, Race.race_key, Track.name).join(Race.track).where(
                    Race.is_completed.is_(False),
                    Race.scheduled_start >= prediction_cutoff,
                    Race.status.not_in(["canceled", "abandoned"]),
                )
            )
        ]
        if track_names:
            track_name_set = {
                track_name.strip().lower()
                for track_name in track_names
                if track_name.strip()
            }
            eligible_rows = [
                row
                for row in eligible_rows
                if row["track_name"].strip().lower() in track_name_set
            ]
        if race_keys:
            race_key_set = set(race_keys)
            eligible_rows = [row for row in eligible_rows if row["race_key"] in race_key_set]

        eligible_race_ids = {int(row["race_id"]) for row in eligible_rows}
        if not eligible_race_ids:
            return []

        examples = build_race_examples(
            settings,
            max_runners=config.max_runners,
            example_race_ids=eligible_race_ids,
        )
        upcoming = _eligible_examples(
            [example for example in examples if not example.has_target],
            runner_limit=config.permutation_runner_limit,
            require_target=False,
        )

        predictions: list[dict[str, Any]] = []
        for example in upcoming:
            orders = _candidate_orders(example, config.permutation_runner_limit)
            if not orders:
                continue
            scores = _score_candidate_orders(
                model,
                example,
                orders,
                scaler,
                device,
                batch_size=max(64, config.batch_size),
            )
            if not scores:
                continue

            ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index])
            best_index = ranked_indices[0]
            second_best_index = ranked_indices[1] if len(ranked_indices) > 1 else best_index
            best_order = orders[best_index]

            negative_scores = torch.tensor([-score for score in scores], dtype=torch.float32)
            permutation_probabilities = torch.softmax(negative_scores, dim=0).tolist()
            confidence = float(permutation_probabilities[best_index])
            confidence_gap = float(
                confidence - permutation_probabilities[second_best_index]
            ) if len(ranked_indices) > 1 else confidence
            score_margin = float(
                scores[second_best_index] - scores[best_index]
            ) if len(ranked_indices) > 1 else 0.0

            win_probability_by_runner: dict[int, float] = defaultdict(float)
            expected_rank_by_runner: dict[int, float] = defaultdict(float)
            for order, probability in zip(orders, permutation_probabilities):
                for rank_index, runner_index in enumerate(order, start=1):
                    expected_rank_by_runner[runner_index] += rank_index * probability
                    if rank_index == 1:
                        win_probability_by_runner[runner_index] += probability

            race = session.get(Race, example.race_id)
            if race is None:
                continue
            run_key = f"predict-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
            prediction_run = PredictionRun(
                run_key=run_key,
                race=race,
                training_run_id=training_run.id if training_run else None,
                predicted_order_json=[],
                confidence=confidence,
                metadata_json={
                    "artifact_path": str(artifact_path),
                    "confidence_gap": confidence_gap,
                    "score_margin": score_margin,
                },
            )
            session.add(prediction_run)
            session.flush()

            ordered_rows = []
            for predicted_rank, runner_index in enumerate(best_order, start=1):
                dog_name = example.dog_names[runner_index]
                race_entry_id = int(example.race_entry_ids[runner_index]) or None
                dog_id = int(example.dog_ids[runner_index]) or None
                trap_number = _runner_trap_number(example, runner_index)
                expected_rank = max(expected_rank_by_runner.get(runner_index, float(predicted_rank)), 1e-6)
                score = 1.0 / expected_rank
                win_probability = float(win_probability_by_runner.get(runner_index, 0.0))
                ordered_rows.append(
                    {
                        "predicted_rank": predicted_rank,
                        "trap_number": trap_number,
                        "dog_name": dog_name,
                        "score": score,
                        "win_probability": win_probability,
                    }
                )
                session.add(
                    PredictionEntry(
                        prediction_run=prediction_run,
                        race_entry_id=race_entry_id,
                        dog_id=dog_id,
                        predicted_rank=predicted_rank,
                        score=score,
                        win_probability=win_probability,
                    )
                )

            prediction_run.predicted_order_json = ordered_rows
            predictions.append(
                {
                    "run_key": run_key,
                    "race_key": example.race_key,
                    "track_name": example.track_name,
                    "scheduled_start": example.scheduled_start.isoformat(),
                    "confidence": confidence,
                    "confidence_gap": confidence_gap,
                    "score_margin": score_margin,
                    "predicted_order": ordered_rows,
                }
            )

        predictions.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        return predictions
