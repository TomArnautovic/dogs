from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import torch
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

from greyhounds.config import Settings
from greyhounds.db import (
    Dog,
    Race,
    RaceEntry,
    TrainingRun,
    Track,
    latest_completed_training_run,
    latest_training_run,
    recent_prediction_runs,
    recent_training_runs,
    session_scope,
)
from greyhounds.ingest import (
    GbgbApiError,
    fetch_rapidapi_racecards_with_quota,
    filter_rapidapi_racecards,
    ingest_gbgb_range,
    ingest_rapidapi_racecards,
    rapidapi_race_status,
    rapidapi_track_name,
)
from greyhounds.ml import (
    MODEL_LAYOUT_VERSION,
    TrainingConfig,
    predict_upcoming_races,
    read_training_stop_request,
    request_training_stop,
    train_model,
    training_weight_snapshot_path,
)


st.set_page_config(page_title="Greyhounds ANN Monitor", page_icon="🏁", layout="wide")

settings = Settings.from_env()
settings.ensure_directories()

TRAINING_LAUNCH_FEEDBACK_KEY = "training_launch_feedback"
MAIN_PAGE_TABS = ("Import", "Search", "Training", "Weights", "Prediction")
MAIN_PAGE_TAB_QUERY_PARAM = "tab"
MAIN_PAGE_TAB_STATE_KEY = "main_page_tab"


def _training_config_draft_path(_settings: Settings) -> Path:
    return _settings.artifacts_dir / "training_config_draft.json"


def _load_training_config_draft(_settings: Settings) -> dict[str, object]:
    draft_path = _training_config_draft_path(_settings)
    if not draft_path.exists():
        return {}
    try:
        payload = json.loads(draft_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_training_config_draft(_settings: Settings, config: dict[str, object]) -> None:
    draft_path = _training_config_draft_path(_settings)
    draft_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _has_streamlit_run_context() -> bool:
    return get_script_run_ctx() is not None


def _normalize_main_page_tab(value: object) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, str) and value in MAIN_PAGE_TABS:
        return value
    return MAIN_PAGE_TABS[0]


def _main_page_tab_default() -> str:
    session_value = st.session_state.get(MAIN_PAGE_TAB_STATE_KEY)
    if isinstance(session_value, str) and session_value in MAIN_PAGE_TABS:
        return session_value
    if not _has_streamlit_run_context():
        return MAIN_PAGE_TABS[0]
    try:
        query_value = st.query_params.get(MAIN_PAGE_TAB_QUERY_PARAM)
    except Exception:
        return MAIN_PAGE_TABS[0]
    return _normalize_main_page_tab(query_value)


def _sync_main_page_tab_query_param() -> None:
    active_tab = _normalize_main_page_tab(st.session_state.get(MAIN_PAGE_TAB_STATE_KEY))
    st.session_state[MAIN_PAGE_TAB_STATE_KEY] = active_tab
    if not _has_streamlit_run_context():
        return
    try:
        if active_tab == MAIN_PAGE_TABS[0]:
            st.query_params.pop(MAIN_PAGE_TAB_QUERY_PARAM, None)
        else:
            st.query_params[MAIN_PAGE_TAB_QUERY_PARAM] = active_tab
    except Exception:
        return


def _training_runtime_profiles_path(_settings: Settings) -> Path:
    return _settings.artifacts_dir / "training_runtime_profiles.json"


def _load_training_runtime_profiles(_settings: Settings) -> dict[str, dict[str, object]]:
    profile_path = _training_runtime_profiles_path(_settings)
    if not profile_path.exists():
        return {}
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): value
        for key, value in payload.items()
        if isinstance(value, dict)
    }


def _save_training_runtime_profiles(_settings: Settings, profiles: dict[str, dict[str, object]]) -> None:
    profile_path = _training_runtime_profiles_path(_settings)
    profile_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")


def _runtime_affecting_config(config: dict[str, object]) -> dict[str, object]:
    return {
        "layout_version": MODEL_LAYOUT_VERSION,
        "batch_size": int(config.get("batch_size", 32)),
        "hidden_size_1": int(config.get("hidden_size_1", 128)),
        "hidden_size_2": int(config.get("hidden_size_2", 64)),
        "validation_fraction": float(config.get("validation_fraction", 0.2)),
        "permutations_per_race": int(config.get("permutations_per_race", 24)),
        "permutation_runner_limit": int(config.get("permutation_runner_limit", 6)),
    }


def _runtime_profile_key(config: dict[str, object]) -> str:
    return json.dumps(_runtime_affecting_config(config), sort_keys=True)


def _record_runtime_profile(
    _settings: Settings,
    config: dict[str, object],
    *,
    seconds_per_epoch: float,
    observed_epochs: int,
) -> None:
    if seconds_per_epoch <= 0:
        return
    profiles = _load_training_runtime_profiles(_settings)
    key = _runtime_profile_key(config)
    profiles[key] = {
        "config": _runtime_affecting_config(config),
        "seconds_per_epoch": float(seconds_per_epoch),
        "observed_epochs": int(observed_epochs),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_training_runtime_profiles(_settings, profiles)


def _runtime_profile_for_config(_settings: Settings, config: dict[str, object]) -> dict[str, object] | None:
    return _load_training_runtime_profiles(_settings).get(_runtime_profile_key(config))


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_finish_time(seconds_from_now: float) -> str:
    finish_time = datetime.now().astimezone() + timedelta(seconds=max(seconds_from_now, 0.0))
    local_now = datetime.now().astimezone()
    if finish_time.date() == local_now.date():
        return finish_time.strftime("%H:%M")
    return finish_time.strftime("%a %H:%M")


def _format_display_datetime(value: object) -> str | None:
    if value in (None, ""):
        return None

    dt_value: datetime | None = None
    if isinstance(value, datetime):
        dt_value = value
    elif isinstance(value, str):
        try:
            dt_value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return str(value)
    else:
        return str(value)

    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=timezone.utc)
    local_dt = dt_value.astimezone()
    return local_dt.strftime("%d %b %Y %H:%M")


def _format_display_date(value: object) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value.strftime("%d %b %Y")
    return str(value)


def _format_decimal(value: object, decimals: int = 2) -> str | None:
    if value in (None, ""):
        return None
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _search_pattern(value: str) -> str:
    return f"%{value.strip().lower()}%"


def _track_timezone(track: Track | None) -> ZoneInfo:
    timezone_name = track.timezone_name if track is not None else None
    try:
        return ZoneInfo(timezone_name or "Europe/London")
    except Exception:
        return ZoneInfo("Europe/London")


def _race_local_datetime(race: Race) -> datetime:
    scheduled_start = race.scheduled_start
    if scheduled_start.tzinfo is None:
        scheduled_start = scheduled_start.replace(tzinfo=timezone.utc)
    return scheduled_start.astimezone(_track_timezone(race.track))


def _race_result_sort_key(entry: RaceEntry) -> tuple[int, int, int]:
    if entry.finish_position is not None:
        return (0, int(entry.finish_position), int(entry.trap_number))
    return (1, int(entry.trap_number), int(entry.id or 0))


def _entry_status(entry: RaceEntry, race: Race | None = None) -> str:
    if entry.vacant:
        return "Vacant"
    if entry.scratched:
        return "Scratched"
    if entry.finish_position is not None:
        return "Resulted"
    if race is not None and race.status:
        return str(race.status).title()
    return "Scheduled"


def _dog_search_rows(session: Session, query: str, limit: int = 25) -> list[dict[str, object]]:
    cleaned_query = query.strip()
    if len(cleaned_query) < 2:
        return []

    dogs = list(
        session.scalars(
            select(Dog)
            .options(selectinload(Dog.owner), selectinload(Dog.trainer))
            .where(func.lower(Dog.name).like(_search_pattern(cleaned_query)))
            .order_by(Dog.name.asc(), Dog.id.asc())
            .limit(limit)
        )
    )
    if not dogs:
        return []

    dog_ids = [dog.id for dog in dogs]
    stats_by_dog_id: dict[int, dict[str, object]] = {
        dog.id: {"starts": 0, "last_start": None}
        for dog in dogs
    }
    stats_statement = (
        select(
            RaceEntry.dog_id,
            func.count(RaceEntry.id),
            func.max(Race.scheduled_start),
        )
        .join(Race, Race.id == RaceEntry.race_id)
        .where(RaceEntry.dog_id.in_(dog_ids))
        .group_by(RaceEntry.dog_id)
    )
    for dog_id, starts, last_start in session.execute(stats_statement):
        if dog_id is not None:
            stats_by_dog_id[int(dog_id)] = {
                "starts": int(starts or 0),
                "last_start": last_start,
            }

    return [
        {
            "id": dog.id,
            "name": dog.name,
            "trainer": dog.trainer.name if dog.trainer else None,
            "owner": dog.owner.name if dog.owner else None,
            "date_of_birth": _format_display_date(dog.date_of_birth),
            "starts": stats_by_dog_id[dog.id]["starts"],
            "last_start": _format_display_datetime(stats_by_dog_id[dog.id]["last_start"]),
        }
        for dog in dogs
    ]


def _dog_profile(session: Session, dog_id: int, recent_limit: int = 12) -> dict[str, object] | None:
    dog = session.scalar(
        select(Dog)
        .options(selectinload(Dog.owner), selectinload(Dog.trainer))
        .where(Dog.id == dog_id)
    )
    if dog is None:
        return None

    entries = list(
        session.scalars(
            select(RaceEntry)
            .options(selectinload(RaceEntry.race).selectinload(Race.track))
            .join(Race, Race.id == RaceEntry.race_id)
            .where(RaceEntry.dog_id == dog.id)
            .order_by(Race.scheduled_start.desc(), RaceEntry.id.desc())
        )
    )
    completed_entries = [
        entry
        for entry in entries
        if entry.finish_position is not None and not entry.scratched and not entry.vacant
    ]
    starts = len([entry for entry in entries if not entry.scratched and not entry.vacant])
    wins = sum(1 for entry in completed_entries if entry.finish_position == 1)
    places = sum(1 for entry in completed_entries if entry.finish_position is not None and entry.finish_position <= 3)
    official_times = [entry.official_time_s for entry in completed_entries if entry.official_time_s is not None]
    recent_completed = completed_entries[:6]
    recent_form = "-".join(str(entry.finish_position) for entry in recent_completed) or "n/a"

    form_rows: list[dict[str, object]] = []
    for entry in entries[:recent_limit]:
        race = entry.race
        race_time = _race_local_datetime(race)
        form_rows.append(
            {
                "date": race_time.strftime("%d %b %Y"),
                "time": race_time.strftime("%H:%M"),
                "track": race.track.name if race.track else None,
                "race": race.race_name or (f"Race {race.race_number}" if race.race_number else None),
                "trap": entry.trap_number,
                "finish": entry.finish_position,
                "distance_m": race.distance_m,
                "grade": race.grade,
                "official_time_s": _format_decimal(entry.official_time_s, 2),
                "sectional_s": _format_decimal(entry.sectional_s, 2),
                "beaten_distance": entry.beaten_distance,
                "sp": entry.sp_text or _format_decimal(entry.sp_decimal, 2),
                "status": _entry_status(entry, race),
                "comment": entry.comment,
            }
        )

    return {
        "dog": {
            "id": dog.id,
            "name": dog.name,
            "sex": dog.sex,
            "date_of_birth": _format_display_date(dog.date_of_birth),
            "sire": dog.sire_name,
            "dam": dog.dam_name,
            "trainer": dog.trainer.name if dog.trainer else None,
            "owner": dog.owner.name if dog.owner else None,
            "provider": dog.provider,
            "provider_dog_id": dog.provider_dog_id,
        },
        "stats": {
            "starts": starts,
            "wins": wins,
            "places": places,
            "recent_form": recent_form,
            "win_rate": (wins / starts) if starts else None,
            "place_rate": (places / starts) if starts else None,
            "best_time_s": min(official_times) if official_times else None,
            "latest_start": _format_display_datetime(entries[0].race.scheduled_start) if entries else None,
        },
        "form_rows": form_rows,
    }


def _track_search_rows(session: Session, query: str, limit: int = 40) -> list[dict[str, object]]:
    cleaned_query = query.strip()
    statement = select(Track).order_by(Track.name.asc(), Track.id.asc()).limit(limit)
    if cleaned_query:
        statement = (
            select(Track)
            .where(func.lower(Track.name).like(_search_pattern(cleaned_query)))
            .order_by(Track.name.asc(), Track.id.asc())
            .limit(limit)
        )
    tracks = list(session.scalars(statement))
    if not tracks:
        return []

    track_ids = [track.id for track in tracks]
    race_counts = {
        int(track_id): int(race_count or 0)
        for track_id, race_count in session.execute(
            select(Race.track_id, func.count(Race.id))
            .where(Race.track_id.in_(track_ids))
            .group_by(Race.track_id)
        )
    }
    return [
        {
            "id": track.id,
            "name": track.name,
            "country_code": track.country_code,
            "timezone": track.timezone_name,
            "race_count": race_counts.get(track.id, 0),
        }
        for track in tracks
    ]


def _race_option_rows(session: Session, track_id: int, race_date: date) -> list[dict[str, object]]:
    track = session.get(Track, track_id)
    if track is None:
        return []

    track_timezone = _track_timezone(track)
    start_local = datetime.combine(race_date, time.min, tzinfo=track_timezone)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)

    races = list(
        session.scalars(
            select(Race)
            .options(selectinload(Race.track), selectinload(Race.entries))
            .where(
                Race.track_id == track.id,
                Race.scheduled_start >= start_utc,
                Race.scheduled_start < end_utc,
            )
            .order_by(Race.scheduled_start.asc(), Race.race_number.asc(), Race.id.asc())
        )
    )

    rows: list[dict[str, object]] = []
    for race in races:
        local_start = _race_local_datetime(race)
        label_parts = [
            local_start.strftime("%H:%M"),
            f"Race {race.race_number}" if race.race_number else None,
            race.race_name,
            f"{race.distance_m}m" if race.distance_m else None,
            race.grade,
        ]
        rows.append(
            {
                "id": race.id,
                "label": " | ".join(part for part in label_parts if part),
                "scheduled_start": local_start.strftime("%d %b %Y %H:%M"),
                "race_number": race.race_number,
                "race_name": race.race_name,
                "distance_m": race.distance_m,
                "grade": race.grade,
                "status": race.status,
                "is_completed": race.is_completed,
                "runner_count": len(race.entries),
            }
        )
    return rows


def _race_result(session: Session, race_id: int) -> dict[str, object] | None:
    race = session.scalar(
        select(Race)
        .options(
            selectinload(Race.track),
            selectinload(Race.entries).selectinload(RaceEntry.dog).selectinload(Dog.trainer),
            selectinload(Race.entries).selectinload(RaceEntry.dog).selectinload(Dog.owner),
        )
        .where(Race.id == race_id)
    )
    if race is None:
        return None

    local_start = _race_local_datetime(race)
    has_result_order = any(entry.finish_position is not None for entry in race.entries)
    sorted_entries = (
        sorted(race.entries, key=_race_result_sort_key)
        if has_result_order
        else sorted(race.entries, key=lambda entry: (entry.trap_number, entry.id or 0))
    )
    rows: list[dict[str, object]] = []
    for display_index, entry in enumerate(sorted_entries, start=1):
        dog = entry.dog
        rows.append(
            {
                "order": entry.finish_position if has_result_order else display_index,
                "trap": entry.trap_number,
                "dog": dog.name if dog else None,
                "trainer": dog.trainer.name if dog and dog.trainer else None,
                "owner": dog.owner.name if dog and dog.owner else None,
                "finish_position": entry.finish_position,
                "official_time_s": _format_decimal(entry.official_time_s, 2),
                "sectional_s": _format_decimal(entry.sectional_s, 2),
                "beaten_distance": entry.beaten_distance,
                "sp": entry.sp_text or _format_decimal(entry.sp_decimal, 2),
                "weight_kg": _format_decimal(entry.weight_kg, 2),
                "status": _entry_status(entry, race),
                "comment": entry.comment,
            }
        )

    return {
        "race": {
            "id": race.id,
            "track": race.track.name if race.track else None,
            "scheduled_start": local_start.strftime("%d %b %Y %H:%M"),
            "race_number": race.race_number,
            "race_name": race.race_name,
            "distance_m": race.distance_m,
            "grade": race.grade,
            "going": race.going,
            "status": race.status,
            "is_completed": race.is_completed,
            "provider_race_id": race.provider_race_id,
        },
        "rows": rows,
    }


def _friendly_db_error_message(exc: Exception) -> str:
    message = str(exc)
    if "failed to resolve host" in message and ".supabase.co" in message:
        return (
            "Could not resolve the Supabase database host. This usually means `DATABASE_URL` is still "
            "using the direct `db.<project-ref>.supabase.co` host on an IPv4-only network. "
            "Switch to the Supabase `Session Pooler` connection string and restart Streamlit."
        )
    return message


def _safe_session_query(
    _settings: Settings,
    loader,
    default,
) -> tuple[object, str | None]:
    try:
        with session_scope(_settings) as session:
            return loader(session), None
    except Exception as exc:
        return default, _friendly_db_error_message(exc)


def _redacted_database_url(database_url: str) -> str:
    parsed = urlsplit(database_url)
    if not parsed.scheme:
        return database_url

    netloc = parsed.netloc
    if "@" in netloc:
        userinfo, hostinfo = netloc.rsplit("@", 1)
        username = userinfo.split(":", 1)[0] if userinfo else ""
        redacted_userinfo = f"{username}:***" if username else "***"
        netloc = f"{redacted_userinfo}@{hostinfo}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _latest_local_resumable_artifact_path(_settings: Settings) -> str | None:
    candidates = sorted(
        _settings.models_dir.glob("train-*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _logs_dir(_settings: Settings) -> Path:
    logs_path = getattr(_settings, "logs_dir", _settings.artifacts_dir / "logs")
    resolved = Path(logs_path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _training_log_path_for_run(_settings: Settings, run_key: str | None) -> Path | None:
    if not run_key:
        return None
    return _logs_dir(_settings) / f"{run_key}.jsonl"


def _read_last_training_log_event(log_path: Path | None) -> dict[str, object] | None:
    if log_path is None or not log_path.exists():
        return None
    try:
        lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return None
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _read_training_log_events(log_path: Path | None) -> list[dict[str, object]]:
    if log_path is None or not log_path.exists():
        return []
    events: list[dict[str, object]] = []
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    except OSError:
        return []
    return events


def _event_age_seconds(event: dict[str, object] | None) -> float | None:
    if not isinstance(event, dict):
        return None
    timestamp = event.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        return None
    try:
        event_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    return max((datetime.now(timezone.utc) - event_time).total_seconds(), 0.0)


def _maybe_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mark_training_run_interrupted(_settings: Settings, run_key: str) -> tuple[bool, str]:
    try:
        with session_scope(_settings) as session:
            training_run = session.scalar(select(TrainingRun).where(TrainingRun.run_key == run_key))
            if training_run is None:
                return False, f"Training run {run_key} was not found."
            if training_run.status != "running" or training_run.finished_at is not None:
                return False, f"Training run {run_key} is already marked {training_run.status}."
            training_run.status = "interrupted"
            training_run.finished_at = datetime.now(timezone.utc)
            training_run.error_text = "Marked interrupted from the dashboard after the live training session stopped updating."
        return True, f"Marked training run {run_key} as interrupted."
    except Exception as exc:
        return False, _friendly_db_error_message(exc)


def _resume_compatibility_inputs(config: dict[str, object]) -> dict[str, object]:
    return {
        "layout_version": MODEL_LAYOUT_VERSION,
        "hidden_size_1": int(config.get("hidden_size_1", 128)),
        "hidden_size_2": int(config.get("hidden_size_2", 64)),
        "permutation_runner_limit": int(config.get("permutation_runner_limit", 6)),
    }


@st.cache_data(show_spinner=False)
def _artifact_resume_metadata(artifact_path: str) -> dict[str, object]:
    try:
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if not isinstance(artifact, dict):
        return {}
    config = artifact.get("config")
    if not isinstance(config, dict):
        config = {}
    return {
        "layout_version": artifact.get("layout_version"),
        "hidden_size_1": int(config.get("hidden_size_1", 128)),
        "hidden_size_2": int(config.get("hidden_size_2", 64)),
        "permutation_runner_limit": int(config.get("permutation_runner_limit", 6)),
    }


def _resume_compatibility(
    artifact_path: str | None,
    current_config: dict[str, object],
) -> tuple[bool, str | None]:
    if not artifact_path:
        return False, "No saved artifact is currently available to resume from."

    metadata = _artifact_resume_metadata(artifact_path)
    if not metadata:
        return False, "The latest artifact could not be read, so resume has been left off."

    current_inputs = _resume_compatibility_inputs(current_config)
    if metadata.get("layout_version") != MODEL_LAYOUT_VERSION:
        return False, "The latest artifact was trained on an older input layout, so a fresh run is safer."
    if metadata.get("hidden_size_1") != current_inputs["hidden_size_1"]:
        return False, "Hidden size 1 has changed, which would make the saved weight shapes incompatible."
    if metadata.get("hidden_size_2") != current_inputs["hidden_size_2"]:
        return False, "Hidden size 2 has changed, which would make the saved weight shapes incompatible."
    if metadata.get("permutation_runner_limit") != current_inputs["permutation_runner_limit"]:
        return False, "The runner-limit input layout has changed, so resume has been left off."
    return True, None


TRAINING_WIDGET_KEYS = {
    "epochs": "training_epochs",
    "batch_size": "training_batch_size",
    "learning_rate": "training_learning_rate",
    "hidden_size_1": "training_hidden_size_1",
    "hidden_size_2": "training_hidden_size_2",
    "dropout": "training_dropout",
    "validation_fraction": "training_validation_fraction",
    "weight_decay": "training_weight_decay",
    "min_completed_races": "training_min_completed_races",
    "permutations_per_race": "training_permutations_per_race",
    "permutation_runner_limit": "training_permutation_runner_limit",
    "early_stopping_patience": "training_early_stopping_patience",
    "resume_previous": "training_resume_previous",
}
TRAINING_RESUME_AUTO_META_KEY = "training_resume_auto_meta"


def _ensure_training_widget_defaults(initial_config: dict[str, object]) -> None:
    defaults: dict[str, object] = {
        "epochs": int(initial_config.get("epochs", 60)),
        "batch_size": int(initial_config.get("batch_size", 32)),
        "learning_rate": float(initial_config.get("learning_rate", 0.001)),
        "hidden_size_1": int(initial_config.get("hidden_size_1", 128)),
        "hidden_size_2": int(initial_config.get("hidden_size_2", 64)),
        "dropout": float(initial_config.get("dropout", 0.15)),
        "validation_fraction": float(initial_config.get("validation_fraction", 0.2)),
        "weight_decay": float(initial_config.get("weight_decay", 0.00001)),
        "min_completed_races": int(initial_config.get("min_completed_races", 25)),
        "permutations_per_race": int(initial_config.get("permutations_per_race", 24)),
        "permutation_runner_limit": 6,
        "early_stopping_patience": int(initial_config.get("early_stopping_patience", 20)),
        "resume_previous": False,
    }
    for config_key, default_value in defaults.items():
        widget_key = TRAINING_WIDGET_KEYS[config_key]
        if widget_key not in st.session_state:
            st.session_state[widget_key] = default_value


def _load_racecard_snapshot(
    _settings: Settings,
    race_date_value: date,
) -> tuple[list[dict[str, object]], dict[str, int | None], Exception | None]:
    try:
        racecards, quota = _cached_rapidapi_racecards_with_quota(_settings, race_date_value.isoformat())
        return racecards, quota, None
    except Exception as exc:
        return [], {}, exc


def _format_prediction_runner(row: dict[str, object]) -> str | None:
    dog_name = row.get("dog_name")
    if dog_name is None:
        return None
    trap_number = row.get("trap_number")
    if trap_number is None:
        return str(dog_name)
    return f"T{trap_number} {dog_name}"


def _predicted_dog_name(prediction: dict[str, object], rank_index: int) -> str | None:
    ordered_rows = prediction.get("predicted_order")
    if not isinstance(ordered_rows, list) or rank_index >= len(ordered_rows):
        return None
    row = ordered_rows[rank_index]
    if not isinstance(row, dict):
        return None
    return _format_prediction_runner(row)


def _upcoming_prediction_track_options(_settings: Settings) -> list[str]:
    cutoff = datetime.now(timezone.utc)
    with session_scope(_settings) as session:
        statement = (
            select(Track.name)
            .join(Race, Race.track_id == Track.id)
            .where(
                Race.is_completed.is_(False),
                Race.scheduled_start >= cutoff,
                Race.status.not_in(["canceled", "abandoned"]),
            )
            .distinct()
            .order_by(Track.name.asc())
        )
        return [track_name for track_name in session.scalars(statement) if track_name]


last_run = None
last_completed_run = None
current_active_run = None
last_config: dict[str, object] = {}
dashboard_db_error: str | None = None
last_run_result, last_run_error = _safe_session_query(
    settings,
    lambda session: latest_training_run(session),
    None,
)
last_completed_run_result, last_completed_run_error = _safe_session_query(
    settings,
    lambda session: latest_completed_training_run(session),
    None,
)
active_run_result, active_run_error = _safe_session_query(
    settings,
    lambda session: session.scalar(
        select(TrainingRun)
        .where(TrainingRun.status == "running", TrainingRun.finished_at.is_(None))
        .order_by(TrainingRun.started_at.desc())
        .limit(1)
    ),
    None,
)
last_run = last_run_result
last_completed_run = last_completed_run_result
current_active_run = active_run_result
if last_run_error:
    dashboard_db_error = last_run_error
elif last_completed_run_error:
    dashboard_db_error = last_completed_run_error
elif active_run_error:
    dashboard_db_error = active_run_error

config_source_run = last_run if last_run and last_run.config_json else last_completed_run
if config_source_run and config_source_run.config_json:
    last_config = dict(config_source_run.config_json)

resume_artifact_path = None
if last_completed_run and last_completed_run.artifact_path and Path(last_completed_run.artifact_path).exists():
    resume_artifact_path = last_completed_run.artifact_path
else:
    resume_artifact_path = _latest_local_resumable_artifact_path(settings)

interrupted_run_message: str | None = None
interrupted_run_log_hint: str | None = None
interrupted_run_log_path: Path | None = None
interrupted_run_last_log_event: dict[str, object] | None = None
if current_active_run and current_active_run.status == "running" and not current_active_run.finished_at:
    started_text = _format_display_datetime(current_active_run.started_at) or "an unknown time"
    interrupted_run_message = (
        f"A training run is already active (run key {current_active_run.run_key}, started at {started_text}). "
        "Starting another run is disabled until that one finishes or is marked interrupted."
    )
    interrupted_run_log_path = _training_log_path_for_run(settings, current_active_run.run_key)
    interrupted_run_last_log_event = _read_last_training_log_event(interrupted_run_log_path)
    if interrupted_run_log_path and interrupted_run_log_path.exists():
        interrupted_run_log_hint = f"Training log: {interrupted_run_log_path}"
        if interrupted_run_last_log_event:
            event_name = str(interrupted_run_last_log_event.get("event", "unknown"))
            event_time = _format_display_datetime(interrupted_run_last_log_event.get("timestamp"))
            interrupted_run_log_hint += f" | Last recorded event: {event_name}"
            if event_time:
                interrupted_run_log_hint += f" at {event_time}"
elif last_run and last_run.status == "running" and not last_run.finished_at:
    started_text = _format_display_datetime(last_run.started_at) or "an unknown time"
    interrupted_run_message = (
        f"The most recent training run started at {started_text} but never recorded completion. "
        "That usually means the Streamlit session or process restarted before the final artifact/report was written."
    )
    interrupted_log_path = _training_log_path_for_run(settings, last_run.run_key)
    last_log_event = _read_last_training_log_event(interrupted_log_path)
    if interrupted_log_path and interrupted_log_path.exists():
        interrupted_run_log_hint = f"Training log: {interrupted_log_path}"
        if last_log_event:
            event_name = str(last_log_event.get("event", "unknown"))
            event_time = _format_display_datetime(last_log_event.get("timestamp"))
            interrupted_run_log_hint += f" | Last recorded event: {event_name}"
            if event_time:
                interrupted_run_log_hint += f" at {event_time}"
elif last_run and last_run.status == "failed":
    started_text = _format_display_datetime(last_run.started_at) or "an unknown time"
    interrupted_run_message = (
        f"The most recent training run from {started_text} is marked failed."
    )
elif last_run and last_run.status == "interrupted":
    started_text = _format_display_datetime(last_run.started_at) or "an unknown time"
    finished_text = _format_display_datetime(last_run.finished_at) or "an unknown time"
    interrupted_run_message = (
        f"The most recent training run started at {started_text} and was interrupted around {finished_text} "
        "before the final artifact/report was written."
    )

draft_config = _load_training_config_draft(settings)
initial_training_config = {**last_config, **draft_config}
_ensure_training_widget_defaults(initial_training_config)


@st.cache_data(show_spinner=False, ttl=300)
def _cached_rapidapi_racecards_with_quota(
    _settings: Settings,
    race_date_iso: str,
) -> tuple[list[dict[str, object]], dict[str, int | None]]:
    racecards, quota = fetch_rapidapi_racecards_with_quota(_settings, date.fromisoformat(race_date_iso))
    return racecards, {
        "daily_limit": quota.daily_limit,
        "daily_remaining": quota.daily_remaining,
        "minute_limit": quota.minute_limit,
        "minute_remaining": quota.minute_remaining,
    }


def _epoch_snapshots(report: dict[str, object]) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    best_epoch = report.get("best_epoch") if isinstance(report.get("best_epoch"), dict) else None
    last_epoch = report.get("last_epoch") if isinstance(report.get("last_epoch"), dict) else None
    if best_epoch and last_epoch:
        return best_epoch, last_epoch

    history = report.get("history")
    if not isinstance(history, list) or not history:
        return best_epoch, last_epoch

    normalized_history = [item for item in history if isinstance(item, dict)]
    if not normalized_history:
        return best_epoch, last_epoch

    if best_epoch is None:
        best_epoch = min(
            normalized_history,
            key=lambda item: float(item.get("validation_loss", float("inf"))),
        )
    if last_epoch is None:
        last_epoch = normalized_history[-1]
    return best_epoch, last_epoch


def _render_epoch_snapshot_card(title: str, epoch_summary: dict[str, object] | None, *, best_checkpoint: bool) -> None:
    st.markdown(f"**{title}**")
    if not epoch_summary:
        st.caption("No epoch summary available.")
        return

    label_prefix = "Best" if best_checkpoint else "Last"
    top_row = st.columns(3)
    bottom_row = st.columns(3)

    top_row[0].metric("Epoch", str(epoch_summary.get("epoch", "?")))
    top_row[1].metric("Val loss", f"{float(epoch_summary.get('validation_loss', 0.0)):.4f}")
    top_row[2].metric("Winner", f"{float(epoch_summary.get('validation_winner_accuracy', 0.0)):.3f}")

    bottom_row[0].metric("Exact", f"{float(epoch_summary.get('validation_exact_order_accuracy', 0.0)):.3f}")
    bottom_row[1].metric("Top3", f"{float(epoch_summary.get('validation_top3_set_accuracy', 0.0)):.3f}")
    bottom_row[2].metric(
        "Rank err",
        f"{float(epoch_summary.get('validation_mean_abs_rank_error', 0.0)):.3f}",
        help=f"{label_prefix} validation mean absolute rank error.",
    )


def _winner_trap_diagnostic_rows(epoch_summary: dict[str, object] | None) -> list[dict[str, object]]:
    if not epoch_summary:
        return []
    diagnostic = epoch_summary.get("winner_trap_diagnostic")
    if not isinstance(diagnostic, dict):
        return []

    actual_counts = diagnostic.get("actual_winner_counts")
    predicted_counts = diagnostic.get("predicted_winner_counts")
    accuracy_by_actual = diagnostic.get("winner_accuracy_by_actual_trap")
    if not isinstance(actual_counts, dict) or not isinstance(predicted_counts, dict) or not isinstance(accuracy_by_actual, dict):
        return []

    rows: list[dict[str, object]] = []
    for trap_number in range(1, 7):
        trap_key = str(trap_number)
        accuracy_value = accuracy_by_actual.get(trap_key)
        rows.append(
            {
                "trap": trap_number,
                "actual winners": int(actual_counts.get(trap_key, 0) or 0),
                "predicted winners": int(predicted_counts.get(trap_key, 0) or 0),
                "winner acc when actual": (
                    f"{float(accuracy_value):.3f}" if accuracy_value is not None else "n/a"
                ),
            }
        )
    return rows


def _render_winner_trap_diagnostic(title: str, epoch_summary: dict[str, object] | None) -> None:
    st.markdown(f"**{title}**")
    rows = _winner_trap_diagnostic_rows(epoch_summary)
    if not rows:
        st.caption("No winner-trap diagnostic available.")
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_training_log_progress(title: str, log_path: Path | None) -> None:
    st.markdown(f"**{title}**")
    if log_path is None or not log_path.exists():
        st.caption("No training log is available yet for this run.")
        return

    events = _read_training_log_events(log_path)
    if not events:
        st.caption(f"Training log exists but no readable events have been written yet: {log_path}")
        return

    start_event = next((event for event in events if event.get("event") == "training_started"), None)
    last_event = events[-1]
    total_epochs = None
    if isinstance(start_event, dict):
        config = start_event.get("config")
        if isinstance(config, dict):
            total_epochs = _maybe_int(config.get("epochs"))

    batch_rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    current_epoch_marker: int | None = None
    for event_index, payload in enumerate(events, start=1):
        event_name = str(payload.get("event", ""))
        if event_name == "epoch_started":
            current_epoch_marker = _maybe_int(payload.get("epoch")) or current_epoch_marker
            total_epochs = total_epochs or _maybe_int(payload.get("total_epochs"))
        elif event_name == "batch_heartbeat":
            current_epoch_marker = _maybe_int(payload.get("epoch")) or current_epoch_marker
            batch_rows.append(
                {
                    "heartbeat_index": event_index,
                    "epoch": _maybe_int(payload.get("epoch")),
                    "batch_index": _maybe_int(payload.get("batch_index")),
                    "total_batches": _maybe_int(payload.get("total_batches")),
                    "batch_loss": _maybe_float(payload.get("batch_loss")),
                    "rolling_batch_loss": _maybe_float(payload.get("rolling_batch_loss")),
                    "samples_seen_total": _maybe_int(payload.get("samples_seen_total")),
                    "elapsed_seconds": _maybe_float(payload.get("elapsed_seconds")),
                }
            )
        elif event_name == "epoch_completed":
            current_epoch_marker = _maybe_int(payload.get("epoch")) or current_epoch_marker
            total_epochs = total_epochs or _maybe_int(payload.get("total_epochs"))
            epoch_rows.append(
                {
                    "epoch": _maybe_int(payload.get("epoch")),
                    "total_epochs": _maybe_int(payload.get("total_epochs")),
                    "train_loss": _maybe_float(payload.get("train_loss")),
                    "validation_loss": _maybe_float(payload.get("validation_loss")),
                    "validation_winner_accuracy": _maybe_float(payload.get("validation_winner_accuracy")),
                    "best_validation_loss": _maybe_float(payload.get("best_validation_loss")),
                    "elapsed_seconds": _maybe_float(payload.get("elapsed_seconds")),
                }
            )

    batch_df = pd.DataFrame(batch_rows)
    epoch_df = pd.DataFrame(epoch_rows)
    latest_batch = batch_rows[-1] if batch_rows else None
    latest_epoch = epoch_rows[-1] if epoch_rows else None
    current_epoch = current_epoch_marker or (
        _maybe_int(latest_batch.get("epoch")) if latest_batch else None
    ) or (
        _maybe_int(latest_epoch.get("epoch")) if latest_epoch else None
    )
    current_batch = _maybe_int(latest_batch.get("batch_index")) if latest_batch else None
    total_batches = _maybe_int(latest_batch.get("total_batches")) if latest_batch else None
    elapsed_seconds = _maybe_float(last_event.get("elapsed_seconds"))
    if elapsed_seconds is None and latest_epoch:
        elapsed_seconds = _maybe_float(latest_epoch.get("elapsed_seconds"))
    if elapsed_seconds is None and latest_batch:
        elapsed_seconds = _maybe_float(latest_batch.get("elapsed_seconds"))

    progress_fraction = None
    if total_epochs and current_epoch:
        if current_batch and total_batches:
            progress_fraction = ((current_epoch - 1) + (current_batch / total_batches)) / total_epochs
        else:
            progress_fraction = current_epoch / total_epochs
        progress_fraction = min(max(progress_fraction, 0.0), 1.0)

    summary_cols = st.columns(4, gap="medium")
    summary_cols[0].metric("Epoch", f"{current_epoch or '?'} / {total_epochs or '?'}")
    summary_cols[1].metric("Batch", f"{current_batch or '-'} / {total_batches or '-'}")
    summary_cols[2].metric("Elapsed", _format_duration(elapsed_seconds) if elapsed_seconds is not None else "n/a")
    summary_cols[3].metric("Last update", _format_display_datetime(last_event.get("timestamp")) or "unknown")

    train_race_count = _maybe_int(start_event.get("train_race_count")) if isinstance(start_event, dict) else None
    validation_race_count = _maybe_int(start_event.get("validation_race_count")) if isinstance(start_event, dict) else None
    st.caption(
        " | ".join(
            item
            for item in [
                f"Log: {log_path}",
                f"Train races: {train_race_count}" if train_race_count is not None else None,
                f"Validation races: {validation_race_count}" if validation_race_count is not None else None,
            ]
            if item
        )
    )

    if progress_fraction is not None:
        st.progress(progress_fraction)

    if not batch_df.empty:
        batch_plot_df = batch_df.copy()
        batch_plot_df["plot_step"] = batch_plot_df["samples_seen_total"].fillna(batch_plot_df["heartbeat_index"])
        st.caption("Batch loss from the active JSONL log")
        st.line_chart(
            batch_plot_df.set_index("plot_step")[["rolling_batch_loss", "batch_loss"]],
            width="stretch",
        )

    if not epoch_df.empty:
        st.caption("Epoch summary from the active JSONL log")
        st.line_chart(
            epoch_df.set_index("epoch")[["train_loss", "validation_loss", "best_validation_loss"]],
            width="stretch",
        )
        st.line_chart(
            epoch_df.set_index("epoch")[["validation_winner_accuracy"]],
            width="stretch",
        )

    with st.expander("Latest log event"):
        st.code(json.dumps(last_event, indent=2, default=str), language="json")


def _load_torch_payload(path: Path) -> dict[str, object] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _state_dict_from_payload(payload: dict[str, object] | None) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        return {}
    raw_state = payload.get("state_dict")
    if not isinstance(raw_state, dict):
        return {}
    return {
        str(name): tensor.detach().cpu()
        for name, tensor in raw_state.items()
        if torch.is_tensor(tensor)
    }


def _parameter_summaries_from_state_dict(state_dict: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, tensor in state_dict.items():
        values = tensor.float().reshape(-1)
        count = int(values.numel())
        if count:
            row = {
                "parameter": name,
                "shape": "x".join(str(value) for value in tensor.shape) or "scalar",
                "count": count,
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "abs_max": float(values.abs().max().item()),
            }
        else:
            row = {
                "parameter": name,
                "shape": "x".join(str(value) for value in tensor.shape) or "scalar",
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "abs_max": 0.0,
            }
        rows.append(row)
    return rows


def _parameter_summaries_from_payload(payload: dict[str, object], state_dict: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    raw_summaries = payload.get("parameter_summaries")
    if isinstance(raw_summaries, list):
        rows = []
        for item in raw_summaries:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "parameter": item.get("name"),
                    "shape": "x".join(str(value) for value in item.get("shape", [])) or "scalar",
                    "count": item.get("count"),
                    "mean": item.get("mean"),
                    "std": item.get("std"),
                    "min": item.get("min"),
                    "max": item.get("max"),
                    "abs_max": item.get("abs_max"),
                }
            )
        if rows:
            return rows
    return _parameter_summaries_from_state_dict(state_dict)


def _weight_snapshot_path_for_run(_settings: Settings, run_key: str | None) -> Path | None:
    if not run_key:
        return None
    snapshot_path = training_weight_snapshot_path(_settings, run_key)
    return snapshot_path if snapshot_path.exists() else None


def _latest_weight_payload_source(
    _settings: Settings,
    active_run: TrainingRun | None,
    completed_run: TrainingRun | None,
    latest_run: TrainingRun | None,
) -> tuple[dict[str, object] | None, Path | None, str]:
    for run, label in (
        (active_run, "live snapshot"),
        (completed_run, "saved snapshot"),
        (latest_run, "latest run snapshot"),
    ):
        if not run:
            continue
        snapshot_path = _weight_snapshot_path_for_run(_settings, run.run_key)
        if snapshot_path is not None:
            payload = _load_torch_payload(snapshot_path)
            if payload is not None:
                return payload, snapshot_path, label
        if run.artifact_path and Path(run.artifact_path).exists():
            payload = _load_torch_payload(Path(run.artifact_path))
            if payload is not None:
                return payload, Path(run.artifact_path), "model artifact"

    local_artifact = _latest_local_resumable_artifact_path(_settings)
    if local_artifact:
        artifact_path = Path(local_artifact)
        payload = _load_torch_payload(artifact_path)
        if payload is not None:
            return payload, artifact_path, "local model artifact"
    return None, None, ""


def _weight_value_rows(
    state_dict: dict[str, torch.Tensor],
    parameter_name: str,
    max_rows: int,
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    total_values = 0
    for name, tensor in state_dict.items():
        if parameter_name != "All parameters" and name != parameter_name:
            continue
        flat_values = tensor.reshape(-1)
        total_values += int(flat_values.numel())
        remaining = max_rows - len(rows)
        if remaining <= 0:
            continue
        for index, value in enumerate(flat_values[:remaining].tolist()):
            rows.append(
                {
                    "parameter": name,
                    "index": index,
                    "value": float(value),
                }
            )
    return rows, total_values


def _weight_payload_epoch_label(payload: dict[str, object]) -> str:
    epoch = payload.get("epoch")
    if epoch not in (None, ""):
        return str(epoch)
    last_epoch = payload.get("last_epoch")
    if isinstance(last_epoch, dict) and last_epoch.get("epoch") not in (None, ""):
        return str(last_epoch.get("epoch"))
    return "n/a"


def _render_ann_weights_panel(
    _settings: Settings,
    active_run: TrainingRun | None,
    completed_run: TrainingRun | None,
    latest_run: TrainingRun | None,
    *,
    payload_override: dict[str, object] | None = None,
    source_path_override: Path | None = None,
    source_label_override: str | None = None,
    compact: bool = False,
) -> None:
    if payload_override is not None:
        payload = payload_override
        source_path = source_path_override
        source_label = source_label_override or "live snapshot"
    else:
        payload, source_path, source_label = _latest_weight_payload_source(
            _settings,
            active_run,
            completed_run,
            latest_run,
        )

    if payload is None:
        st.info("No ANN weights are available yet. Start a training run or load a saved model artifact.")
        return

    state_dict = _state_dict_from_payload(payload)
    if not state_dict:
        st.info("The selected weight source did not contain a readable ANN state dictionary.")
        return

    summaries = _parameter_summaries_from_payload(payload, state_dict)
    total_parameters = int(payload.get("total_parameters") or sum(int(row.get("count") or 0) for row in summaries))
    metadata_cols = st.columns(5, gap="medium")
    metadata_cols[0].metric("Run", str(payload.get("run_key") or "n/a"))
    metadata_cols[1].metric("Stage", str(payload.get("stage") or payload.get("artifact_state") or source_label or "n/a"))
    metadata_cols[2].metric("Epoch", _weight_payload_epoch_label(payload))
    metadata_cols[3].metric("Batch", str(payload.get("batch_index") or "-"))
    metadata_cols[4].metric("Parameters", f"{total_parameters:,}")

    details = [
        f"Source: {source_label}" if source_label else None,
        f"Path: {source_path}" if source_path else None,
        f"Updated: {_format_display_datetime(payload.get('updated_at'))}" if payload.get("updated_at") else None,
    ]
    st.caption(" | ".join(item for item in details if item))

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        st.dataframe(summary_df, width="stretch", hide_index=True)
        chart_df = summary_df[["parameter", "abs_max"]].dropna()
        if not chart_df.empty:
            st.bar_chart(chart_df.set_index("parameter"), width="stretch")

    if compact:
        if bool(st.session_state.get("ann_weights_live_values", False)):
            max_live_rows = int(st.session_state.get("ann_weights_live_max_rows", 10_000))
            value_rows, total_values = _weight_value_rows(state_dict, "All parameters", max_live_rows)
            if total_values > len(value_rows):
                st.caption(f"Live table showing {len(value_rows):,} of {total_values:,} weight values.")
            else:
                st.caption(f"Live table showing all {total_values:,} weight values.")
            if value_rows:
                st.dataframe(pd.DataFrame(value_rows), width="stretch", hide_index=True)
        return

    parameter_names = ["All parameters", *state_dict.keys()]
    selected_parameter = st.selectbox("Parameter values", options=parameter_names, key="ann_weights_parameter_filter")
    max_rows = int(
        st.number_input(
            "Maximum weight rows",
            min_value=100,
            max_value=1_000_000,
            value=200_000,
            step=1000,
            key="ann_weights_max_rows",
        )
    )
    value_rows, total_values = _weight_value_rows(state_dict, selected_parameter, max_rows)
    if total_values > len(value_rows):
        st.caption(f"Showing {len(value_rows):,} of {total_values:,} weight values for the selected parameter set.")
    else:
        st.caption(f"Showing all {total_values:,} weight values for the selected parameter set.")
    if value_rows:
        st.dataframe(pd.DataFrame(value_rows), width="stretch", hide_index=True)


def _render_inline_training_event(event: dict[str, object], config: TrainingConfig) -> None:
    event_name = str(event.get("event") or "training")
    epoch = _maybe_int(event.get("epoch"))
    total_epochs = _maybe_int(event.get("total_epochs")) or int(config.epochs)
    batch_index = _maybe_int(event.get("batch_index"))
    total_batches = _maybe_int(event.get("total_batches"))
    elapsed_seconds = _maybe_float(event.get("elapsed_seconds"))

    cols = st.columns(4, gap="medium")
    cols[0].metric("Event", event_name)
    cols[1].metric("Epoch", f"{epoch or 0} / {total_epochs}")
    cols[2].metric("Batch", f"{batch_index or '-'} / {total_batches or '-'}")
    cols[3].metric("Elapsed", _format_duration(elapsed_seconds) if elapsed_seconds is not None else "n/a")

    if epoch and total_epochs:
        if batch_index and total_batches:
            progress_fraction = ((epoch - 1) + (batch_index / total_batches)) / total_epochs
        else:
            progress_fraction = epoch / total_epochs
        st.progress(min(max(progress_fraction, 0.0), 1.0))


def _run_inline_training(
    _settings: Settings,
    config: TrainingConfig,
    current_config: dict[str, object],
    progress_slot,
    weights_slot,
) -> tuple[bool, str, dict[str, object] | None]:
    last_event: dict[str, object] | None = None
    observed_epochs = 0

    def progress_callback(event: dict[str, object]) -> None:
        nonlocal last_event, observed_epochs
        last_event = event
        if event.get("event") == "epoch":
            observed_epochs = max(observed_epochs, _maybe_int(event.get("epoch")) or 0)

        with progress_slot.container():
            _render_inline_training_event(event, config)
            run_key = str(event.get("run_key") or "")
            log_path = _training_log_path_for_run(_settings, run_key)
            if log_path and log_path.exists():
                _render_training_log_progress("Live Training Progress", log_path)
            else:
                with st.expander("Latest training event"):
                    st.code(json.dumps(event, indent=2, default=str), language="json")

        snapshot_path_value = event.get("weight_snapshot_path")
        if isinstance(snapshot_path_value, str) and snapshot_path_value:
            snapshot_path = Path(snapshot_path_value)
            payload = _load_torch_payload(snapshot_path)
            if payload is not None:
                with weights_slot.container():
                    _render_ann_weights_panel(
                        _settings,
                        None,
                        None,
                        None,
                        payload_override=payload,
                        source_path_override=snapshot_path,
                        source_label_override="live in-process snapshot",
                        compact=True,
                    )

    try:
        summary = train_model(_settings, config, progress=progress_callback)
    except Exception as exc:
        with progress_slot.container():
            if last_event:
                _render_inline_training_event(last_event, config)
            st.error(_friendly_db_error_message(exc))
        return False, _friendly_db_error_message(exc), None

    elapsed_seconds = _maybe_float(summary.get("summary", {}).get("elapsed_seconds") if isinstance(summary.get("summary"), dict) else None)
    if elapsed_seconds is None and last_event:
        elapsed_seconds = _maybe_float(last_event.get("elapsed_seconds"))
    if elapsed_seconds and observed_epochs:
        _record_runtime_profile(
            _settings,
            current_config,
            seconds_per_epoch=elapsed_seconds / observed_epochs,
            observed_epochs=observed_epochs,
        )

    return (
        True,
        f"Training completed in-process for {summary.get('run_key')}.",
        summary,
    )


def _render_recent_training_runs(_settings: Settings) -> None:
    st.subheader("Recent Training Runs")
    training_runs_result, training_runs_error = _safe_session_query(
        _settings,
        lambda session: recent_training_runs(session, limit=20),
        [],
    )
    training_runs = training_runs_result if isinstance(training_runs_result, list) else []

    if training_runs_error:
        st.info(training_runs_error)
        return
    if not training_runs:
        st.info("No training runs have been recorded yet.")
        return

    training_df = pd.DataFrame(
        [
            {
                "run_key": run.run_key,
                "status": run.status,
                "started_at": _format_display_datetime(run.started_at),
                "finished_at": _format_display_datetime(run.finished_at),
                "winner_accuracy": (run.metrics_json or {}).get("winner_accuracy"),
                "top3_set_accuracy": (run.metrics_json or {}).get("top3_set_accuracy"),
                "exact_order_accuracy": (run.metrics_json or {}).get("exact_order_accuracy"),
                "mean_abs_rank_error": (run.metrics_json or {}).get("mean_abs_rank_error"),
                "artifact_path": run.artifact_path,
            }
            for run in training_runs
        ]
    )
    st.dataframe(training_df, width="stretch", hide_index=True)

    latest_report_run = next(
        (
            run for run in training_runs
            if run.report_path and Path(run.report_path).exists()
        ),
        None,
    )
    if latest_report_run is None:
        st.caption("No saved report is available yet for the recent runs, so only the run table is shown.")
        return

    try:
        report = json.loads(Path(latest_report_run.report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st.caption("The latest saved training report could not be read.")
        return

    best_epoch, last_epoch = _epoch_snapshots(report)
    st.caption(
        f"Historical charts below come from {latest_report_run.run_key}, the latest run with a saved report."
    )
    summary_left, summary_right = st.columns(2, gap="large")
    with summary_left:
        _render_epoch_snapshot_card("Best Checkpoint", best_epoch, best_checkpoint=True)
    with summary_right:
        _render_epoch_snapshot_card("Last Epoch", last_epoch, best_checkpoint=False)
    trap_left, trap_right = st.columns(2, gap="large")
    with trap_left:
        _render_winner_trap_diagnostic("Best Checkpoint Trap Bias", best_epoch)
    with trap_right:
        _render_winner_trap_diagnostic("Last Epoch Trap Bias", last_epoch)

    history = report.get("history")
    history_df = pd.DataFrame(history) if isinstance(history, list) else pd.DataFrame()
    if history_df.empty:
        return
    st.line_chart(
        history_df.set_index("epoch")[["train_loss", "validation_loss"]],
        width="stretch",
    )
    winner_cols = [
        col for col in ["train_winner_accuracy", "validation_winner_accuracy"]
        if col in history_df.columns
    ]
    if winner_cols:
        st.line_chart(
            history_df.set_index("epoch")[winner_cols],
            width="stretch",
        )
    exact_cols = [
        col for col in ["train_exact_order_accuracy", "validation_exact_order_accuracy"]
        if col in history_df.columns
    ]
    if exact_cols:
        st.line_chart(
            history_df.set_index("epoch")[exact_cols],
            width="stretch",
        )


def _render_recent_prediction_runs(_settings: Settings) -> None:
    st.subheader("Recent Prediction Runs")
    prediction_runs_result, prediction_runs_error = _safe_session_query(
        _settings,
        lambda session: recent_prediction_runs(session, limit=20),
        [],
    )
    prediction_runs = prediction_runs_result if isinstance(prediction_runs_result, list) else []

    if prediction_runs_error:
        st.info(prediction_runs_error)
        return
    if not prediction_runs:
        st.info("No prediction runs have been stored yet.")
        return

    prediction_df = pd.DataFrame(
        [
            {
                "race_id": run.race_id,
                "created_at": _format_display_datetime(run.created_at),
                "confidence": run.confidence,
                "gap": (run.metadata_json or {}).get("confidence_gap"),
                "predicted_order": " > ".join(
                    runner
                    for item in run.predicted_order_json[:4]
                    if isinstance(item, dict)
                    for runner in [_format_prediction_runner(item)]
                    if runner is not None
                ),
            }
            for run in prediction_runs
        ]
    )
    st.dataframe(prediction_df, width="stretch", hide_index=True)

st.title("Greyhounds ANN Monitor")
st.caption("Train a MATLAB-style permutation scorer, inspect recent runs, and score upcoming races.")

launch_feedback = st.session_state.pop(TRAINING_LAUNCH_FEEDBACK_KEY, None)
if isinstance(launch_feedback, dict):
    level = str(launch_feedback.get("level", "info"))
    message = str(launch_feedback.get("message", ""))
    if message:
        if level == "success":
            st.success(message)
        elif level == "error":
            st.error(message)
        else:
            st.info(message)

if dashboard_db_error:
    st.warning(dashboard_db_error)

prediction_track_options: list[str] = []
if not dashboard_db_error:
    try:
        prediction_track_options = _upcoming_prediction_track_options(settings)
    except Exception as exc:
        dashboard_db_error = _friendly_db_error_message(exc)
        prediction_track_options = []
        st.warning(dashboard_db_error)
prediction_track_state_key = "prediction_track_selection"
prediction_track_meta_key = "prediction_track_selection_meta"
prediction_track_meta = tuple(prediction_track_options)
if st.session_state.get(prediction_track_meta_key) != prediction_track_meta:
    st.session_state[prediction_track_state_key] = prediction_track_options
    st.session_state[prediction_track_meta_key] = prediction_track_meta
else:
    st.session_state[prediction_track_state_key] = [
        track_name
        for track_name in st.session_state.get(prediction_track_state_key, prediction_track_options)
        if track_name in prediction_track_options
    ]

import_tab, search_tab, training_tab, weights_tab, prediction_tab = st.tabs(
    list(MAIN_PAGE_TABS),
    default=_main_page_tab_default(),
    key=MAIN_PAGE_TAB_STATE_KEY,
    on_change=_sync_main_page_tab_query_param,
)

with import_tab:
    st.subheader("Racecard Import")
    st.caption("Load a day of RapidAPI racecards, choose the tracks you want, and import only those races into the database.")

    if not settings.rapidapi_key:
        st.warning("RAPIDAPI_KEY is not set in `.env`, so racecard import is currently disabled.")
    else:
        import_controls = st.columns([1.2, 0.9, 0.9, 0.9], gap="medium")
        racecard_date = import_controls[0].date_input(
            "Racecard date",
            value=date.today(),
            key="racecard_import_date",
        )
        include_finished_racecards = import_controls[1].checkbox(
            "Include finished",
            value=False,
            help="Keep this off for a clean upcoming-races import. Turn it on later if you want a full-day refresh.",
        )
        include_canceled_racecards = import_controls[2].checkbox(
            "Include canceled",
            value=False,
            help="Normally leave canceled races out so they do not clutter the prediction queue.",
        )
        import_delay_seconds = import_controls[3].number_input(
            "Delay (seconds)",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.1,
            format="%.1f",
        )
        refresh_existing_racecards = st.checkbox(
            "Refresh existing races",
            value=False,
            help="Leave this off to save quota by skipping races already imported from RapidAPI. Turn it on later when you want to update existing races with fresh results.",
        )
        load_racecards_now = st.button("Load/Refresh Racecards", width="stretch")

        snapshot_date_key = "racecard_snapshot_date"
        snapshot_data_key = "racecard_snapshot_data"
        snapshot_quota_key = "racecard_snapshot_quota"
        snapshot_error_key = "racecard_snapshot_error"

        if load_racecards_now:
            racecards, quota, racecard_error = _load_racecard_snapshot(settings, racecard_date)
            st.session_state[snapshot_date_key] = racecard_date.isoformat()
            st.session_state[snapshot_data_key] = racecards
            st.session_state[snapshot_quota_key] = quota
            st.session_state[snapshot_error_key] = str(racecard_error) if racecard_error else None

        snapshot_matches_date = st.session_state.get(snapshot_date_key) == racecard_date.isoformat()
        racecards = st.session_state.get(snapshot_data_key, []) if snapshot_matches_date else []
        quota = st.session_state.get(snapshot_quota_key, {}) if snapshot_matches_date else {}
        racecard_error_text = st.session_state.get(snapshot_error_key) if snapshot_matches_date else None

        if not snapshot_matches_date:
            st.info("Click `Load/Refresh Racecards` to fetch the selected date and see quota/track options.")
        elif racecard_error_text:
            st.error(f"Could not load racecards for {racecard_date.isoformat()}: {racecard_error_text}")
        elif not racecards:
            if quota:
                quota_cols = st.columns(4)
                quota_cols[0].metric("Daily requests left", str(quota.get("daily_remaining") or 0))
                quota_cols[1].metric("Daily limit", str(quota.get("daily_limit") or 0))
                quota_cols[2].metric("Minute requests left", str(quota.get("minute_remaining") or 0))
                quota_cols[3].metric("Minute limit", str(quota.get("minute_limit") or 0))
            st.info(f"No racecards were returned for {racecard_date.isoformat()}.")
        else:
            quota_cols = st.columns(4)
            quota_cols[0].metric(
                "Daily requests left",
                str(quota.get("daily_remaining")) if quota.get("daily_remaining") is not None else "n/a",
            )
            quota_cols[1].metric(
                "Daily limit",
                str(quota.get("daily_limit")) if quota.get("daily_limit") is not None else "n/a",
            )
            quota_cols[2].metric(
                "Minute requests left",
                str(quota.get("minute_remaining")) if quota.get("minute_remaining") is not None else "n/a",
            )
            quota_cols[3].metric(
                "Minute limit",
                str(quota.get("minute_limit")) if quota.get("minute_limit") is not None else "n/a",
            )
            st.caption("Quota is taken from the latest RapidAPI racecards response headers.")

            track_options = sorted({rapidapi_track_name(racecard) for racecard in racecards})
            track_state_key = "racecard_track_selection"
            track_meta_key = "racecard_track_selection_meta"
            track_meta = (racecard_date.isoformat(), tuple(track_options))
            if st.session_state.get(track_meta_key) != track_meta:
                st.session_state[track_state_key] = track_options
                st.session_state[track_meta_key] = track_meta
            else:
                st.session_state[track_state_key] = [
                    track_name
                    for track_name in st.session_state.get(track_state_key, track_options)
                    if track_name in track_options
                ] or track_options

            selected_tracks = st.multiselect(
                "Tracks to import",
                options=track_options,
                key=track_state_key,
                help="Choose one or more tracks from the loaded racecards for this date.",
            )

            filtered_racecards = filter_rapidapi_racecards(
                racecards,
                include_finished=include_finished_racecards,
                include_canceled=include_canceled_racecards,
                track_names=selected_tracks,
            )

            racecard_metric_cols = st.columns(4)
            racecard_metric_cols[0].metric("Racecards returned", len(racecards))
            racecard_metric_cols[1].metric("Tracks selected", len(selected_tracks))
            racecard_metric_cols[2].metric("Races matching filters", len(filtered_racecards))
            racecard_metric_cols[3].metric(
                "Upcoming races",
                sum(1 for racecard in filtered_racecards if rapidapi_race_status(racecard) == "scheduled"),
            )

            preview_df = pd.DataFrame(
                [
                    {
                        "track": rapidapi_track_name(racecard),
                        "scheduled_start": _format_display_datetime(racecard.get("date")),
                        "title": racecard.get("title"),
                        "distance": racecard.get("distance"),
                        "status": rapidapi_race_status(racecard),
                        "race_id": racecard.get("id_race"),
                    }
                    for racecard in filtered_racecards
                ]
            )
            if not preview_df.empty:
                st.dataframe(preview_df, width="stretch", hide_index=True)
            else:
                st.info("No races match the current filters.")

            import_selected_racecards = st.button(
                "Import Selected Tracks",
                type="primary",
                width="stretch",
                disabled=bool(dashboard_db_error),
            )
            if import_selected_racecards:
                if not selected_tracks:
                    st.warning("Select at least one track before importing.")
                else:
                    import_status = st.empty()
                    import_log = st.empty()
                    log_lines: list[str] = []

                    def update_import_progress(message: str) -> None:
                        log_lines.append(message)
                        import_status.info(message)
                        import_log.code("\n".join(log_lines[-12:]), language="text")

                    with st.spinner("Importing selected racecards..."):
                        import_summary = ingest_rapidapi_racecards(
                            settings,
                            race_date=racecard_date,
                            track_names=selected_tracks,
                            refresh_existing=refresh_existing_racecards,
                            include_finished=include_finished_racecards,
                            include_canceled=include_canceled_racecards,
                            delay_seconds=float(import_delay_seconds),
                            progress=update_import_progress,
                        )
                    import_status.success(
                        f"Imported {import_summary['races_touched']} race(s) across {len(import_summary['track_names'])} track(s), "
                        f"skipping {import_summary['races_skipped_existing']} already-imported race(s)."
                    )
                    st.json(import_summary)

    st.divider()
    st.subheader("Historic GBGB Import")
    st.caption("Backfill the last few days of official GBGB results into the database from the UI.")

    historic_import_controls = st.columns([0.9, 1.2, 0.9], gap="medium")
    historic_lookback_days = historic_import_controls[0].number_input(
        "Lookback days",
        min_value=1,
        max_value=30,
        value=3,
        step=1,
        help="Imports from today minus this many days through today.",
    )
    historic_track = historic_import_controls[1].text_input(
        "Track filter (optional)",
        value="",
        help="Leave blank to backfill all GBGB tracks for the chosen window, or enter a track like Crayford.",
    )
    historic_delay_seconds = historic_import_controls[2].number_input(
        "GBGB delay (seconds)",
        min_value=0.0,
        max_value=10.0,
        value=0.5,
        step=0.1,
        format="%.1f",
    )
    run_historic_import = st.button("Import Historic GBGB Results", width="stretch", disabled=bool(dashboard_db_error))

    if run_historic_import:
        historic_status = st.empty()
        historic_log = st.empty()
        historic_log_lines: list[str] = []
        historic_end_date = date.today()
        historic_start_date = historic_end_date - timedelta(days=int(historic_lookback_days) - 1)
        normalized_historic_track = historic_track.strip() or None

        def update_historic_import_progress(message: str) -> None:
            historic_log_lines.append(message)
            historic_status.info(message)
            historic_log.code("\n".join(historic_log_lines[-15:]), language="text")

        historic_summary: dict[str, object] | None = None
        with st.spinner("Importing historic GBGB results..."):
            try:
                historic_summary = ingest_gbgb_range(
                    settings,
                    start_date=historic_start_date,
                    end_date=historic_end_date,
                    track=normalized_historic_track,
                    delay_seconds=float(historic_delay_seconds),
                    progress=update_historic_import_progress,
                )
            except GbgbApiError as exc:
                historic_log_lines.append(f"Stopped: {exc}")
                historic_status.warning(f"GBGB import stopped: {exc}")
                historic_log.code("\n".join(historic_log_lines[-15:]), language="text")
                st.info("Any races imported before the outage remain saved. Run the import again to fill the missing range.")

        if historic_summary is not None:
            historic_status.success(
                f"Imported {historic_summary['races_touched']} GBGB race(s) from {historic_summary['start_date']} "
                f"to {historic_summary['end_date']}, skipping {historic_summary['races_skipped']} existing race(s)."
            )
            st.json(historic_summary)

with search_tab:
    st.subheader("Search")
    st.caption("Find a dog and inspect recent form, or look up the runners and results for a race by track and date.")

    st.markdown("**Dog Search**")
    dog_query = st.text_input(
        "Dog name",
        value="",
        placeholder="Type at least two letters",
        key="dog_search_query",
        disabled=bool(dashboard_db_error),
    )
    if dashboard_db_error:
        st.info("Search is unavailable until the database connection issue above is resolved.")
    elif len(dog_query.strip()) < 2:
        st.info("Enter at least two letters of a dog name to search.")
    else:
        dog_matches_result, dog_matches_error = _safe_session_query(
            settings,
            lambda session: _dog_search_rows(session, dog_query),
            [],
        )
        dog_matches = dog_matches_result if isinstance(dog_matches_result, list) else []
        if dog_matches_error:
            st.warning(dog_matches_error)
        elif not dog_matches:
            st.info("No dogs matched that search.")
        else:
            dog_match_labels = {
                int(row["id"]): (
                    f"{row['name']}"
                    f"{' | Trainer: ' + str(row['trainer']) if row.get('trainer') else ''}"
                    f" | Starts: {row.get('starts', 0)}"
                )
                for row in dog_matches
            }
            dog_option_ids = [int(row["id"]) for row in dog_matches]
            dog_selection_key = "selected_dog_search_result"
            if st.session_state.get(dog_selection_key) not in dog_option_ids:
                st.session_state[dog_selection_key] = dog_option_ids[0]
            selected_dog_id = st.selectbox(
                "Matching dogs",
                options=dog_option_ids,
                format_func=lambda dog_id: dog_match_labels.get(int(dog_id), str(dog_id)),
                key=dog_selection_key,
            )
            profile_result, profile_error = _safe_session_query(
                settings,
                lambda session: _dog_profile(session, int(selected_dog_id)),
                None,
            )
            if profile_error:
                st.warning(profile_error)
            elif not isinstance(profile_result, dict):
                st.info("That dog could not be loaded.")
            else:
                dog = profile_result["dog"]
                stats = profile_result["stats"]
                detail_cols = st.columns(4, gap="medium")
                detail_cols[0].metric("Starts", str(stats.get("starts", 0)))
                detail_cols[1].metric("Wins", str(stats.get("wins", 0)))
                detail_cols[2].metric("Places", str(stats.get("places", 0)))
                detail_cols[3].metric("Recent form", str(stats.get("recent_form") or "n/a"))

                rate_cols = st.columns(4, gap="medium")
                win_rate = stats.get("win_rate")
                place_rate = stats.get("place_rate")
                rate_cols[0].metric(
                    "Win rate",
                    f"{float(win_rate) * 100:.1f}%" if win_rate is not None else "n/a",
                )
                rate_cols[1].metric(
                    "Place rate",
                    f"{float(place_rate) * 100:.1f}%" if place_rate is not None else "n/a",
                )
                rate_cols[2].metric(
                    "Best time",
                    f"{float(stats['best_time_s']):.2f}s" if stats.get("best_time_s") is not None else "n/a",
                )
                rate_cols[3].metric("Latest start", str(stats.get("latest_start") or "n/a"))

                st.markdown(f"**{dog['name']}**")
                dog_detail_df = pd.DataFrame(
                    [
                        {
                            "sex": dog.get("sex"),
                            "date_of_birth": dog.get("date_of_birth"),
                            "sire": dog.get("sire"),
                            "dam": dog.get("dam"),
                            "trainer": dog.get("trainer"),
                            "owner": dog.get("owner"),
                            "provider": dog.get("provider"),
                            "provider_dog_id": dog.get("provider_dog_id"),
                        }
                    ]
                )
                st.dataframe(dog_detail_df, width="stretch", hide_index=True)

                form_rows = profile_result.get("form_rows")
                if isinstance(form_rows, list) and form_rows:
                    st.markdown("**Recent Form**")
                    st.dataframe(pd.DataFrame(form_rows), width="stretch", hide_index=True)
                else:
                    st.info("No race entries are stored for this dog yet.")

    st.divider()
    st.markdown("**Race Search**")
    race_search_cols = st.columns([1.2, 0.8], gap="medium")
    track_query = race_search_cols[0].text_input(
        "Track",
        value="",
        placeholder="Start typing a track name",
        key="race_search_track_query",
        disabled=bool(dashboard_db_error),
    )
    race_search_date = race_search_cols[1].date_input(
        "Race date",
        value=date.today(),
        key="race_search_date",
        disabled=bool(dashboard_db_error),
    )

    if not dashboard_db_error:
        track_matches_result, track_matches_error = _safe_session_query(
            settings,
            lambda session: _track_search_rows(session, track_query),
            [],
        )
        track_matches = track_matches_result if isinstance(track_matches_result, list) else []
        if track_matches_error:
            st.warning(track_matches_error)
        elif not track_matches:
            st.info("No tracks matched that search.")
        else:
            track_labels = {
                int(row["id"]): f"{row['name']} | Races stored: {row.get('race_count', 0)}"
                for row in track_matches
            }
            track_option_ids = [int(row["id"]) for row in track_matches]
            track_selection_key = "selected_race_search_track"
            if st.session_state.get(track_selection_key) not in track_option_ids:
                st.session_state[track_selection_key] = track_option_ids[0]
            selected_track_id = st.selectbox(
                "Matching tracks",
                options=track_option_ids,
                format_func=lambda track_id: track_labels.get(int(track_id), str(track_id)),
                key=track_selection_key,
            )
            race_options_result, race_options_error = _safe_session_query(
                settings,
                lambda session: _race_option_rows(session, int(selected_track_id), race_search_date),
                [],
            )
            race_options = race_options_result if isinstance(race_options_result, list) else []
            if race_options_error:
                st.warning(race_options_error)
            elif not race_options:
                selected_track_label = track_labels.get(int(selected_track_id), "that track").split(" | ", 1)[0]
                st.info(f"No races are stored for {selected_track_label} on {race_search_date.isoformat()}.")
            else:
                race_labels = {
                    int(row["id"]): (
                        f"{row['label']} | Runners: {row.get('runner_count', 0)} | "
                        f"{'Completed' if row.get('is_completed') else str(row.get('status') or 'Scheduled').title()}"
                    )
                    for row in race_options
                }
                race_option_ids = [int(row["id"]) for row in race_options]
                race_selection_key = "selected_race_search_result"
                if st.session_state.get(race_selection_key) not in race_option_ids:
                    st.session_state[race_selection_key] = race_option_ids[0]
                selected_race_id = st.selectbox(
                    "Races on that date",
                    options=race_option_ids,
                    format_func=lambda race_id: race_labels.get(int(race_id), str(race_id)),
                    key=race_selection_key,
                )
                race_result, race_error = _safe_session_query(
                    settings,
                    lambda session: _race_result(session, int(selected_race_id)),
                    None,
                )
                if race_error:
                    st.warning(race_error)
                elif not isinstance(race_result, dict):
                    st.info("That race could not be loaded.")
                else:
                    race = race_result["race"]
                    race_cols = st.columns(5, gap="medium")
                    race_cols[0].metric("Track", str(race.get("track") or "n/a"))
                    race_cols[1].metric("Start", str(race.get("scheduled_start") or "n/a"))
                    race_cols[2].metric("Race", str(race.get("race_number") or "n/a"))
                    race_cols[3].metric("Distance", f"{race['distance_m']}m" if race.get("distance_m") else "n/a")
                    race_cols[4].metric("Grade", str(race.get("grade") or "n/a"))

                    race_caption_parts = [
                        str(race.get("race_name")) if race.get("race_name") else None,
                        f"Going: {race.get('going')}" if race.get("going") else None,
                        f"Status: {race.get('status')}" if race.get("status") else None,
                        f"Provider race ID: {race.get('provider_race_id')}" if race.get("provider_race_id") else None,
                    ]
                    st.caption(" | ".join(part for part in race_caption_parts if part))

                    result_rows = race_result.get("rows")
                    if isinstance(result_rows, list) and result_rows:
                        st.dataframe(pd.DataFrame(result_rows), width="stretch", hide_index=True)
                    else:
                        st.info("No runners are stored for this race yet.")

with weights_tab:
    st.subheader("ANN Weights")
    st.caption("Inspect the current ANN tensors from the live training snapshot or the latest saved model artifact.")
    live_weight_controls = st.columns([1.0, 1.0], gap="medium")
    live_weight_controls[0].checkbox(
        "Update flattened values live",
        value=False,
        key="ann_weights_live_values",
    )
    live_weight_controls[1].number_input(
        "Live value row cap",
        min_value=100,
        max_value=1_000_000,
        value=10_000,
        step=1000,
        key="ann_weights_live_max_rows",
    )
    if st.button("Refresh weights view", width="stretch"):
        st.rerun()
    weights_live_slot = st.empty()
    with weights_live_slot.container():
        _render_ann_weights_panel(settings, current_active_run, last_completed_run, last_run)


with training_tab:
    st.subheader("Training")
    st.caption("Launch training, rebuild live progress from the JSONL log after refreshes, and inspect recent runs.")

    config_row_1 = st.columns(3, gap="medium")
    epochs = config_row_1[0].number_input(
        "Epochs",
        min_value=1,
        max_value=5000,
        key=TRAINING_WIDGET_KEYS["epochs"],
        step=5,
        help="How many full passes the trainer makes over the sampled race permutations.",
    )
    batch_size = config_row_1[1].number_input(
        "Batch size",
        min_value=1,
        max_value=256,
        key=TRAINING_WIDGET_KEYS["batch_size"],
        step=1,
        help="How many races are scored together before each weight update.",
    )
    learning_rate = config_row_1[2].number_input(
        "Learning rate",
        min_value=0.00001,
        max_value=0.1,
        key=TRAINING_WIDGET_KEYS["learning_rate"],
        step=0.0005,
        format="%.5f",
        help="Controls how aggressively the network changes its weights after each batch.",
    )

    config_row_2 = st.columns(3, gap="medium")
    hidden_size_1 = config_row_2[0].number_input(
        "Hidden size 1",
        min_value=8,
        max_value=1024,
        key=TRAINING_WIDGET_KEYS["hidden_size_1"],
        step=8,
        help="Number of neurons in the first hidden layer of the ANN.",
    )
    hidden_size_2 = config_row_2[1].number_input(
        "Hidden size 2",
        min_value=4,
        max_value=1024,
        key=TRAINING_WIDGET_KEYS["hidden_size_2"],
        step=4,
        help="Number of neurons in the second hidden layer of the ANN.",
    )
    dropout = config_row_2[2].slider(
        "Dropout",
        min_value=0.0,
        max_value=0.7,
        key=TRAINING_WIDGET_KEYS["dropout"],
        step=0.05,
        help="Randomly turns off part of the network during training to reduce overfitting.",
    )

    config_row_3 = st.columns(3, gap="medium")
    validation_fraction = config_row_3[0].slider(
        "Validation fraction",
        min_value=0.05,
        max_value=0.5,
        key=TRAINING_WIDGET_KEYS["validation_fraction"],
        step=0.05,
        help="Fraction of completed races held back from training and used only for validation metrics.",
    )
    weight_decay = config_row_3[1].number_input(
        "Weight decay",
        min_value=0.0,
        max_value=0.1,
        key=TRAINING_WIDGET_KEYS["weight_decay"],
        format="%.5f",
        help="A small regularization penalty that discourages overly large weights.",
    )
    min_completed_races = config_row_3[2].number_input(
        "Minimum completed races",
        min_value=5,
        max_value=10000,
        key=TRAINING_WIDGET_KEYS["min_completed_races"],
        step=5,
        help="Training will not start unless at least this many fully resulted six-dog races are available.",
    )

    config_row_4 = st.columns(3, gap="medium")
    permutations_per_race = config_row_4[0].number_input(
        "Permutations per race",
        min_value=1,
        max_value=720,
        key=TRAINING_WIDGET_KEYS["permutations_per_race"],
        step=1,
        help="How many non-winning candidate finish orders are sampled alongside the true order for each race per epoch.",
    )
    permutation_runner_limit = config_row_4[1].number_input(
        "Permutation runner limit",
        min_value=6,
        max_value=6,
        key=TRAINING_WIDGET_KEYS["permutation_runner_limit"],
        step=1,
        disabled=True,
        help="The MATLAB-style mode uses a fixed six-dog exhaustive permutation setup.",
    )
    early_stopping_patience = config_row_4[2].number_input(
        "Early stopping patience",
        min_value=0,
        max_value=5000,
        key=TRAINING_WIDGET_KEYS["early_stopping_patience"],
        step=1,
        help="Stop after this many epochs without validation-loss improvement. Set to 0 to run all requested epochs.",
    )

    current_training_config = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "hidden_size_1": int(hidden_size_1),
        "hidden_size_2": int(hidden_size_2),
        "dropout": float(dropout),
        "validation_fraction": float(validation_fraction),
        "weight_decay": float(weight_decay),
        "min_completed_races": int(min_completed_races),
        "permutations_per_race": int(permutations_per_race),
        "permutation_runner_limit": int(permutation_runner_limit),
        "early_stopping_patience": int(early_stopping_patience),
    }
    if _has_streamlit_run_context() and current_training_config != draft_config:
        _save_training_config_draft(settings, current_training_config)

    resume_compatibility_config = {
        "hidden_size_1": int(hidden_size_1),
        "hidden_size_2": int(hidden_size_2),
        "permutation_runner_limit": int(permutation_runner_limit),
    }
    resume_is_compatible, resume_incompatibility_reason = _resume_compatibility(
        resume_artifact_path,
        resume_compatibility_config,
    )
    resume_auto_meta = {
        "artifact_path": resume_artifact_path or "",
        "compatibility": _resume_compatibility_inputs(resume_compatibility_config),
        "enabled": bool(resume_artifact_path) and resume_is_compatible,
    }
    if st.session_state.get(TRAINING_RESUME_AUTO_META_KEY) != resume_auto_meta:
        st.session_state[TRAINING_WIDGET_KEYS["resume_previous"]] = bool(resume_artifact_path) and resume_is_compatible
        st.session_state[TRAINING_RESUME_AUTO_META_KEY] = resume_auto_meta

    runtime_profile = _runtime_profile_for_config(settings, current_training_config)
    control_cols = st.columns([1.2, 1.0, 0.8], gap="large")
    with control_cols[0]:
        resume_previous = st.checkbox(
            "Resume from latest saved weights",
            key=TRAINING_WIDGET_KEYS["resume_previous"],
            disabled=not bool(resume_artifact_path),
            help="Continue training from the latest saved model artifact instead of starting from fresh random weights.",
        )
        st.caption("Training settings are autosaved locally as you change them.")
        if resume_artifact_path:
            st.caption(f"Latest resumable artifact: {resume_artifact_path}")
            if resume_is_compatible:
                st.caption(
                    "Resume has been auto-enabled because the current layout and hidden layer sizes still match the latest artifact."
                )
            elif resume_incompatibility_reason:
                st.caption(resume_incompatibility_reason)

    with control_cols[1]:
        if runtime_profile:
            estimated_total_seconds = float(runtime_profile.get("seconds_per_epoch", 0.0)) * int(current_training_config["epochs"])
            st.info(
                "Estimated runtime for this config: "
                f"about {_format_duration(estimated_total_seconds)} "
                f"(finish around {_format_finish_time(estimated_total_seconds)})."
            )
            st.caption(
                f"Based on {int(runtime_profile.get('observed_epochs', 0))} observed epoch(s) "
                f"for the current model-shape config."
            )
        else:
            st.caption("No saved timing estimate yet for this model-shape config. After epoch 1 the dashboard will learn one.")

    with control_cols[2]:
        train_now = st.button(
            "Start training",
            type="primary",
            width="stretch",
            disabled=bool(dashboard_db_error) or bool(current_active_run),
        )
        if st.button("Refresh training view", width="stretch"):
            st.rerun()

    inline_progress_slot = st.empty()
    if train_now:
        training_config = TrainingConfig(
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            hidden_size_1=int(hidden_size_1),
            hidden_size_2=int(hidden_size_2),
            dropout=float(dropout),
            validation_fraction=float(validation_fraction),
            weight_decay=float(weight_decay),
            max_runners=settings.max_runners,
            min_completed_races=int(min_completed_races),
            model_type="permutation",
            resume_from_artifact=resume_artifact_path if (resume_previous and resume_artifact_path) else None,
            permutations_per_race=int(permutations_per_race),
            permutation_runner_limit=int(permutation_runner_limit),
            early_stopping_patience=int(early_stopping_patience),
        )
        launch_success, launch_message, _training_summary = _run_inline_training(
            settings,
            training_config,
            current_training_config,
            inline_progress_slot,
            weights_live_slot,
        )
        st.session_state[TRAINING_LAUNCH_FEEDBACK_KEY] = {
            "level": "success" if launch_success else "error",
            "message": launch_message,
        }
        if launch_success:
            st.rerun()

    if interrupted_run_message:
        st.warning(interrupted_run_message)
        if interrupted_run_log_hint:
            st.caption(interrupted_run_log_hint)

    if current_active_run:
        stop_request = read_training_stop_request(settings, current_active_run.run_key)
        if interrupted_run_last_log_event:
            last_event_age_seconds = _event_age_seconds(interrupted_run_last_log_event)
            stale_threshold_seconds = 180.0
            if last_event_age_seconds is not None:
                if last_event_age_seconds <= stale_threshold_seconds:
                    st.info(
                        "The active run still looks live based on its log heartbeat "
                        f"({_format_duration(last_event_age_seconds)} ago)."
                    )
                else:
                    st.warning(
                        "The active run looks stale because its last log heartbeat was "
                        f"{_format_duration(last_event_age_seconds)} ago. "
                        "If you know the job is no longer running, mark it interrupted below."
                    )
            detail_cols = st.columns([1.4, 1.0, 1.0], gap="medium")
            detail_cols[0].metric("Run key", current_active_run.run_key)
            detail_cols[1].metric(
                "Last event",
                str(interrupted_run_last_log_event.get("event", "unknown")),
            )
            detail_cols[2].metric(
                "Last update",
                _format_display_datetime(interrupted_run_last_log_event.get("timestamp")) or "unknown",
            )
        else:
            st.info("Training is active, but no log heartbeat has been recorded yet. Refresh in a few seconds.")

        if stop_request:
            requested_at_text = _format_display_datetime(stop_request.get("requested_at")) or "an unknown time"
            st.info(
                "A graceful stop has already been requested for this run. "
                f"Requested at {requested_at_text}; training will save a recovery checkpoint and stop after the current batch."
            )

        action_cols = st.columns(2, gap="medium")
        if action_cols[0].button("Stop active run and save checkpoint", width="stretch", disabled=bool(stop_request)):
            request_training_stop(
                settings,
                current_active_run.run_key,
                requested_by="dashboard",
                reason="Stop requested from the dashboard; save the latest recovery checkpoint before exiting.",
            )
            st.session_state[TRAINING_LAUNCH_FEEDBACK_KEY] = {
                "level": "success",
                "message": (
                    f"Stop requested for {current_active_run.run_key}. "
                    "Training will save a recovery checkpoint and then stop."
                ),
            }
            st.rerun()
        if action_cols[1].button("Mark stale run interrupted", width="stretch"):
            success, interrupt_message = _mark_training_run_interrupted(settings, current_active_run.run_key)
            if success:
                st.session_state[TRAINING_LAUNCH_FEEDBACK_KEY] = {
                    "level": "success",
                    "message": (
                        f"{interrupt_message} This only updates the dashboard state; "
                        "it does not stop a still-running training call."
                    ),
                }
                st.rerun()
            else:
                st.error(interrupt_message)

        _render_training_log_progress("Live Training Progress", interrupted_run_log_path)
    elif last_run and last_run.run_key:
        latest_log_path = _training_log_path_for_run(settings, last_run.run_key)
        if latest_log_path and latest_log_path.exists():
            _render_training_log_progress("Latest Logged Training Progress", latest_log_path)

    _render_recent_training_runs(settings)

with prediction_tab:
    st.subheader("Prediction")
    st.caption("Score upcoming races with the latest saved model and inspect stored prediction runs.")

    selected_prediction_tracks = st.multiselect(
        "Tracks to predict",
        options=prediction_track_options,
        key=prediction_track_state_key,
        help="Leave all selected to score every upcoming track currently in the database, or narrow prediction to one or more tracks.",
    )
    if prediction_track_options:
        st.caption(
            f"{len(selected_prediction_tracks)} of {len(prediction_track_options)} upcoming track(s) selected."
        )
    else:
        st.caption("No future unresolved races are currently available to predict.")

    predict_now = st.button(
        "Predict upcoming races",
        type="primary",
        width="stretch",
        disabled=bool(dashboard_db_error) or not bool(prediction_track_options),
    )

    if predict_now:
        if not selected_prediction_tracks:
            st.warning("Select at least one track before generating predictions.")
        else:
            with st.spinner("Generating predictions..."):
                try:
                    predictions = predict_upcoming_races(settings, track_names=selected_prediction_tracks)
                except Exception as exc:
                    st.error(str(exc))
                    predictions = None
            if predictions is not None:
                st.success(
                    f"Generated {len(predictions)} prediction run(s) across {len(selected_prediction_tracks)} track(s)."
                )
                prediction_results_df = pd.DataFrame(
                    [
                        {
                            "track": prediction.get("track_name"),
                            "time": _format_display_datetime(prediction.get("scheduled_start")),
                            "confidence": prediction.get("confidence"),
                            "gap": prediction.get("confidence_gap"),
                            "1st": _predicted_dog_name(prediction, 0),
                            "2nd": _predicted_dog_name(prediction, 1),
                            "3rd": _predicted_dog_name(prediction, 2),
                        }
                        for prediction in predictions
                    ]
                )
                if not prediction_results_df.empty:
                    st.dataframe(prediction_results_df, width="stretch", hide_index=True)
                with st.expander("Raw prediction JSON"):
                    st.json(predictions)

    _render_recent_prediction_runs(settings)

st.subheader("Environment")
st.code(
    "\n".join(
        [
            f"DATABASE_URL={_redacted_database_url(settings.database_url)}",
            f"ARTIFACTS_DIR={settings.artifacts_dir}",
            f"GBGB_BASE_URL={settings.gbgb_base_url}",
            f"RAPIDAPI_BASE_URL={settings.rapidapi_base_url}",
            f"RAPIDAPI_HOST={settings.rapidapi_host}",
            f"RAPIDAPI_KEY={'set' if settings.rapidapi_key else 'missing'}",
            f"MAX_RUNNERS={settings.max_runners}",
        ]
    ),
    language="bash",
)
