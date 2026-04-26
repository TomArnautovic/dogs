from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import OperationalError
from urllib3.util import Retry

from .config import Settings
from .db import Dog, IngestionRun, PredictionEntry, Race, RaceEntry, RawFetch, Track, find_or_create_named_entity, init_database, session_scope, slugify
from .db import Owner, Trainer


class RapidApiRateLimitError(RuntimeError):
    def __init__(self, message: str, race_id: str | None = None) -> None:
        super().__init__(message)
        self.race_id = race_id


class GbgbApiError(RuntimeError):
    """Raised when GBGB returns an unrecoverable response for an ingestion request."""


@dataclass(slots=True)
class RapidApiQuota:
    daily_limit: int | None = None
    daily_remaining: int | None = None
    minute_limit: int | None = None
    minute_remaining: int | None = None


def _parse_datetime(value: str | None, timezone_name: str | None = None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = None
    if parsed is None:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        default_timezone = timezone.utc
        if timezone_name:
            try:
                default_timezone = ZoneInfo(timezone_name)
            except Exception:
                default_timezone = timezone.utc
        parsed = parsed.replace(tzinfo=default_timezone)
    return parsed.astimezone(timezone.utc)


def _parse_date(value: str | None):
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ("%b-%Y", "%B-%Y", "%d/%m/%Y"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt in {"%b-%Y", "%B-%Y"}:
                return date(parsed.year, parsed.month, 1)
            return parsed.date()
        except ValueError:
            continue
    return None


def _parse_int(value: str | int | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _parse_float(value: str | float | int | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_purse(value: str | float | int | None) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (float, int)):
        return float(value)

    text = value.strip()
    if not text:
        return None

    race_total_match = re.search(r"Race Total\s*£\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if race_total_match:
        return float(race_total_match.group(1).replace(",", ""))

    sterling_match = re.search(r"£\s*([\d,]+(?:\.\d+)?)", text)
    if sterling_match:
        return float(sterling_match.group(1).replace(",", ""))

    return _parse_float(text)


def _parse_distance_m(value: str | int | None) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return value
    text = _stringify(value)
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def _parse_race_number(value: str | int | None) -> int | None:
    if isinstance(value, int):
        return value
    text = _stringify(value)
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def _parse_sp_decimal(
    sp_text: str | None,
    numerator: str | int | None = None,
    denominator: str | int | None = None,
) -> float | None:
    num = _parse_float(numerator)
    den = _parse_float(denominator)
    if num is not None and den not in (None, 0):
        return 1.0 + (num / den)

    text = _stringify(sp_text).upper()
    if not text:
        return None
    text = re.sub(r"[A-Z]+$", "", text)
    match = re.fullmatch(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    parsed_num = float(match.group(1))
    parsed_den = float(match.group(2))
    if parsed_den == 0:
        return None
    return 1.0 + (parsed_num / parsed_den)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _track_key(provider: str, row: dict[str, Any]) -> str:
    provider_track_id = _stringify(row.get("source_track_id") or row.get("track_code"))
    if provider_track_id:
        return f"{provider}:track:{provider_track_id}"
    return f"{provider}:track:{slugify(row['track_name'])}"


def _canonical_track_key(row: dict[str, Any]) -> str:
    country_code = _stringify(row.get("country_code")).lower() or "gb"
    return f"canonical:track:{country_code}:{slugify(row['track_name'])}"


def _dog_key(provider: str, row: dict[str, Any]) -> str:
    provider_dog_id = _stringify(row.get("source_dog_id"))
    if provider_dog_id:
        return f"{provider}:dog:{provider_dog_id}"
    return f"{provider}:dog:{slugify(row['dog_name'])}"


def _canonical_dog_key(row: dict[str, Any]) -> str:
    name_slug = slugify(row["dog_name"])
    dog_date_of_birth = _parse_date(row.get("dog_date_of_birth"))
    if dog_date_of_birth is not None:
        return f"canonical:dog:{name_slug}:{dog_date_of_birth.isoformat()}"

    sire_slug = slugify(_stringify(row.get("sire_name")))
    dam_slug = slugify(_stringify(row.get("dam_name")))
    if sire_slug or dam_slug:
        return f"canonical:dog:{name_slug}:{sire_slug or 'unknown'}:{dam_slug or 'unknown'}"

    return f"canonical:dog:{name_slug}"


def _race_key(provider: str, row: dict[str, Any]) -> str:
    provider_race_id = _stringify(row.get("source_race_id"))
    if provider_race_id:
        return f"{provider}:race:{provider_race_id}"
    race_number = _stringify(row.get("race_number") or "0")
    scheduled = _stringify(row.get("scheduled_start"))
    track_fragment = slugify(row["track_name"])
    return f"{provider}:race:{track_fragment}:{scheduled}:{race_number}"


def _entry_key(race_key: str, trap_number: int) -> str:
    return f"{race_key}:trap:{trap_number}"


def _merge_tracks(session, primary: Track, duplicates: list[Track]) -> Track:
    for duplicate in duplicates:
        if duplicate.id == primary.id:
            continue
        session.execute(update(Race).where(Race.track_id == duplicate.id).values(track_id=primary.id))
        session.delete(duplicate)
    session.flush()
    return primary


def _normalized_slug(value: Any) -> str:
    text = _stringify(value)
    return slugify(text) if text else ""


def _track_candidates(session, provider: str, row: dict[str, Any]) -> list[Track]:
    provider_key = _track_key(provider, row)
    canonical_key = _canonical_track_key(row)
    normalized_track_name = _stringify(row["track_name"]).lower()
    statement = select(Track).where(
        or_(
            Track.track_key.in_([provider_key, canonical_key]),
            func.lower(Track.name) == normalized_track_name,
        )
    ).order_by(Track.id.asc())
    candidates: list[Track] = []
    seen_track_ids: set[int] = set()
    for track in session.scalars(statement):
        if track.id in seen_track_ids:
            continue
        seen_track_ids.add(track.id)
        candidates.append(track)
    return candidates


def _touch_track(session, provider: str, row: dict[str, Any]) -> Track:
    provider_key = _track_key(provider, row)
    canonical_key = _canonical_track_key(row)
    candidates = _track_candidates(session, provider, row)
    track = next((item for item in candidates if item.track_key == canonical_key), None)
    if track is None:
        track = next((item for item in candidates if item.track_key == provider_key), None)
    if track is None and candidates:
        track = candidates[0]

    if track is None:
        track = Track(
            track_key=canonical_key,
            provider=provider,
            provider_track_id=_stringify(row.get("source_track_id") or row.get("track_code")) or None,
            name=row["track_name"].strip(),
            country_code=(row.get("country_code") or None),
            timezone_name=(row.get("timezone") or "Europe/London"),
        )
        session.add(track)
        session.flush()
    else:
        track = _merge_tracks(session, track, candidates)
        track.track_key = canonical_key
        track.name = row["track_name"].strip()
        track.country_code = row.get("country_code") or track.country_code
        track.timezone_name = row.get("timezone") or track.timezone_name
        track.provider_track_id = _stringify(row.get("source_track_id") or row.get("track_code")) or track.provider_track_id
    return track


def _dog_compatible_with_row(dog: Dog, row: dict[str, Any]) -> bool:
    if _normalized_slug(dog.name) != _normalized_slug(row.get("dog_name")):
        return False

    incoming_dob = _parse_date(row.get("dog_date_of_birth"))
    if incoming_dob is not None and dog.date_of_birth is not None and dog.date_of_birth != incoming_dob:
        return False

    incoming_sire = _normalized_slug(row.get("sire_name"))
    if incoming_sire and dog.sire_name and _normalized_slug(dog.sire_name) != incoming_sire:
        return False

    incoming_dam = _normalized_slug(row.get("dam_name"))
    if incoming_dam and dog.dam_name and _normalized_slug(dog.dam_name) != incoming_dam:
        return False

    return True


def _dog_candidates(session, provider: str, row: dict[str, Any]) -> tuple[list[Dog], bool]:
    provider_key = _dog_key(provider, row)
    canonical_key = _canonical_dog_key(row)
    normalized_dog_name = _stringify(row["dog_name"]).lower()

    statement = select(Dog).where(
        or_(
            Dog.dog_key.in_([provider_key, canonical_key]),
            func.lower(Dog.name) == normalized_dog_name,
        )
    ).order_by(Dog.id.asc())
    raw_candidates: list[Dog] = []
    seen_dog_ids: set[int] = set()
    for dog in session.scalars(statement):
        if dog.id in seen_dog_ids:
            continue
        seen_dog_ids.add(dog.id)
        raw_candidates.append(dog)

    direct_matches = [dog for dog in raw_candidates if dog.dog_key in {provider_key, canonical_key}]
    compatible_candidates = [dog for dog in raw_candidates if _dog_compatible_with_row(dog, row)]

    strong_identity = bool(
        _parse_date(row.get("dog_date_of_birth"))
        or _stringify(row.get("sire_name"))
        or _stringify(row.get("dam_name"))
    )
    if compatible_candidates and (strong_identity or len(compatible_candidates) == 1):
        return compatible_candidates, True

    trainer_slug = _normalized_slug(row.get("trainer_name"))
    if compatible_candidates and trainer_slug:
        trainer_matches = [
            dog
            for dog in compatible_candidates
            if dog.trainer is not None and _normalized_slug(dog.trainer.name) == trainer_slug
        ]
        if len(trainer_matches) == 1:
            return trainer_matches, True
        if len(trainer_matches) > 1 and any(dog in direct_matches for dog in trainer_matches):
            return trainer_matches, True

    if direct_matches:
        return direct_matches, False
    return [], False


def _merge_dogs(session, primary: Dog, duplicates: list[Dog]) -> Dog:
    for duplicate in duplicates:
        if duplicate.id == primary.id:
            continue
        if primary.sex is None:
            primary.sex = duplicate.sex
        if primary.date_of_birth is None:
            primary.date_of_birth = duplicate.date_of_birth
        if primary.sire_name is None:
            primary.sire_name = duplicate.sire_name
        if primary.dam_name is None:
            primary.dam_name = duplicate.dam_name
        if primary.owner is None:
            primary.owner = duplicate.owner
        if primary.trainer is None:
            primary.trainer = duplicate.trainer
        if primary.provider_dog_id is None:
            primary.provider_dog_id = duplicate.provider_dog_id

        session.execute(update(RaceEntry).where(RaceEntry.dog_id == duplicate.id).values(dog_id=primary.id))
        session.execute(update(PredictionEntry).where(PredictionEntry.dog_id == duplicate.id).values(dog_id=primary.id))
        session.delete(duplicate)

    session.flush()
    return primary


def _touch_dog(session, provider: str, row: dict[str, Any]) -> Dog:
    owner = find_or_create_named_entity(session, Owner, row.get("owner_name"))
    trainer = find_or_create_named_entity(session, Trainer, row.get("trainer_name"))
    provider_key = _dog_key(provider, row)
    canonical_key = _canonical_dog_key(row)
    candidates, use_canonical_key = _dog_candidates(session, provider, row)
    dog = next((item for item in candidates if item.dog_key == canonical_key), None)
    if dog is None:
        dog = next((item for item in candidates if item.dog_key == provider_key), None)
    if dog is None and candidates:
        dog = candidates[0]

    if dog is None:
        dog = Dog(
            dog_key=canonical_key if use_canonical_key else provider_key,
            provider=provider,
            provider_dog_id=_stringify(row.get("source_dog_id")) or None,
            name=row["dog_name"].strip(),
            sex=(row.get("dog_sex") or None),
            date_of_birth=_parse_date(row.get("dog_date_of_birth")),
            sire_name=(row.get("sire_name") or None),
            dam_name=(row.get("dam_name") or None),
            owner=owner,
            trainer=trainer,
        )
        session.add(dog)
        session.flush()
    else:
        dog = _merge_dogs(session, dog, candidates)
        if use_canonical_key:
            dog.dog_key = canonical_key
        dog.name = row["dog_name"].strip()
        dog.sex = row.get("dog_sex") or dog.sex
        dog.date_of_birth = _parse_date(row.get("dog_date_of_birth")) or dog.date_of_birth
        dog.sire_name = row.get("sire_name") or dog.sire_name
        dog.dam_name = row.get("dam_name") or dog.dam_name
        dog.owner = owner or dog.owner
        dog.trainer = trainer or dog.trainer
        dog.provider_dog_id = _stringify(row.get("source_dog_id")) or dog.provider_dog_id
    return dog


def _touch_race(session, provider: str, row: dict[str, Any], track: Track) -> Race:
    race_key = _race_key(provider, row)
    race = session.scalar(select(Race).where(Race.race_key == race_key))
    status = _stringify(row.get("status")).lower() or None
    completed = (
        _parse_int(row.get("finish_position")) is not None
        or status in {"completed", "finished", "resulted"}
    )
    timezone_name = _stringify(row.get("timezone")) or track.timezone_name or "Europe/London"
    if race is None:
        race = Race(
            race_key=race_key,
            provider=provider,
            provider_race_id=_stringify(row.get("source_race_id")) or None,
            meeting_id=_stringify(row.get("meeting_id")) or None,
            track=track,
            scheduled_start=_parse_datetime(row.get("scheduled_start"), timezone_name=timezone_name)
            or datetime.now(timezone.utc),
            off_time=_parse_datetime(row.get("off_time"), timezone_name=timezone_name),
            race_number=_parse_int(row.get("race_number")),
            race_name=(row.get("race_name") or None),
            distance_m=_parse_int(row.get("distance_m")),
            grade=(row.get("grade") or None),
            going=(row.get("going") or None),
            purse=_parse_purse(row.get("purse")),
            status=row.get("status") or ("resulted" if completed else "scheduled"),
            is_completed=completed,
            metadata_json=dict(row.get("race_metadata_json") or {}),
        )
        session.add(race)
        session.flush()
    else:
        race.provider_race_id = _stringify(row.get("source_race_id")) or race.provider_race_id
        race.meeting_id = _stringify(row.get("meeting_id")) or race.meeting_id
        race.track = track
        race.scheduled_start = (
            _parse_datetime(row.get("scheduled_start"), timezone_name=timezone_name) or race.scheduled_start
        )
        race.off_time = _parse_datetime(row.get("off_time"), timezone_name=timezone_name) or race.off_time
        race.race_number = _parse_int(row.get("race_number")) if row.get("race_number") else race.race_number
        race.race_name = row.get("race_name") or race.race_name
        race.distance_m = _parse_int(row.get("distance_m")) if row.get("distance_m") else race.distance_m
        race.grade = row.get("grade") or race.grade
        race.going = row.get("going") or race.going
        race.purse = _parse_purse(row.get("purse")) if row.get("purse") else race.purse
        race.status = row.get("status") or race.status
        race.is_completed = race.is_completed or completed
        if row.get("race_metadata_json"):
            race.metadata_json = {**(race.metadata_json or {}), **row["race_metadata_json"]}
    return race


def _touch_entry(session, row: dict[str, Any], race: Race, dog: Dog) -> RaceEntry:
    trap_number = _parse_int(row.get("trap_number"))
    if trap_number is None:
        raise ValueError("trap_number is required")
    entry_key = _entry_key(race.race_key, trap_number)
    entry = session.scalar(select(RaceEntry).where(RaceEntry.entry_key == entry_key))
    if entry is None:
        entry = RaceEntry(
            entry_key=entry_key,
            race=race,
            dog=dog,
            trap_number=trap_number,
            metadata_json=dict(row.get("entry_metadata_json") or {}),
        )
        session.add(entry)
        session.flush()
    else:
        entry.race = race
        entry.dog = dog

    entry.weight_kg = _parse_float(row.get("weight_kg")) if row.get("weight_kg") else entry.weight_kg
    entry.finish_position = _parse_int(row.get("finish_position")) if row.get("finish_position") else entry.finish_position
    entry.official_time_s = _parse_float(row.get("official_time_s")) if row.get("official_time_s") else entry.official_time_s
    entry.sectional_s = _parse_float(row.get("sectional_s")) if row.get("sectional_s") else entry.sectional_s
    entry.beaten_distance = row.get("beaten_distance") or entry.beaten_distance
    entry.sp_text = row.get("sp_text") or entry.sp_text
    entry.sp_decimal = (
        _parse_float(row.get("sp_decimal"))
        or _parse_sp_decimal(
            row.get("sp_text"),
            row.get("sp_numerator"),
            row.get("sp_denominator"),
        )
        or entry.sp_decimal
    )
    entry.comment = row.get("comment") or entry.comment
    entry.vacant = _parse_bool(row.get("vacant")) or entry.vacant
    entry.scratched = _parse_bool(row.get("scratched")) or entry.scratched
    if row.get("entry_metadata_json"):
        entry.metadata_json = {**(entry.metadata_json or {}), **row["entry_metadata_json"]}
    return entry


def import_runner_csv(settings: Settings, csv_path: str | Path, source: str = "manual") -> dict[str, Any]:
    init_database(settings)
    path = Path(csv_path)
    rows_processed = 0
    races_touched: set[str] = set()

    with session_scope(settings) as session:
        run = IngestionRun(source=source)
        session.add(run)
        session.flush()
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    provider = (row.get("source") or source).strip()
                    track = _touch_track(session, provider, row)
                    dog = _touch_dog(session, provider, row)
                    race = _touch_race(session, provider, row, track)
                    _touch_entry(session, row, race, dog)
                    rows_processed += 1
                    races_touched.add(race.race_key)

            run.status = "completed"
            run.finished_at = datetime.now(timezone.utc)
            run.rows_processed = rows_processed
            run.races_touched = len(races_touched)
            run.notes_json = {"path": str(path.resolve())}
        except Exception as exc:
            run.status = "failed"
            run.finished_at = datetime.now(timezone.utc)
            run.rows_processed = rows_processed
            run.races_touched = len(races_touched)
            run.error_text = str(exc)
            raise

    return {
        "status": "completed",
        "rows_processed": rows_processed,
        "races_touched": len(races_touched),
        "path": str(path.resolve()),
    }


@dataclass(slots=True)
class GbgbClient:
    settings: Settings
    session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.settings.http_user_agent})
        retries = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1.0,
            status=3,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get_json(self, path: str, *, params: dict[str, Any], label: str) -> dict[str, Any]:
        url = f"{self.settings.gbgb_base_url}{path}"
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.settings.request_timeout_seconds,
            )
        except requests.exceptions.RetryError as exc:
            raise GbgbApiError(
                f"GBGB API kept returning retryable failures while fetching {label}. "
                "The service may be temporarily unavailable; try again later."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise GbgbApiError(
                f"GBGB API timed out while fetching {label} after "
                f"{self.settings.request_timeout_seconds}s."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise GbgbApiError(f"Could not connect to the GBGB API while fetching {label}.") from exc

        if response.status_code in {429, 500, 502, 503, 504}:
            raise GbgbApiError(
                f"GBGB API returned HTTP {response.status_code} while fetching {label}. "
                "The service may be temporarily unavailable; try again later."
            )

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise GbgbApiError(
                f"GBGB API returned HTTP {response.status_code} while fetching {label}."
            ) from exc

        return response.json()

    def fetch_results_page(self, race_date: date, page: int = 1, track: str | None = None) -> dict[str, Any]:
        params = {
            "page": page,
            "itemsPerPage": 100,
            "date": race_date.isoformat(),
            "race_type": "race",
        }
        if track:
            params["track"] = track
        return self._get_json(
            "/results",
            params=params,
            label=f"results for {race_date.isoformat()} page {page}",
        )

    def fetch_meeting_detail(self, meeting_id: str, race_id: str) -> dict[str, Any]:
        return self._get_json(
            f"/results/meeting/{meeting_id}",
            params={"raceId": race_id},
            label=f"meeting {meeting_id} race {race_id}",
        )


@dataclass(slots=True)
class RapidApiGreyhoundUkClient:
    settings: Settings
    session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.settings.rapidapi_key:
            raise ValueError("RAPIDAPI_KEY is not set. Add it to .env before using RapidAPI ingestion.")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.settings.http_user_agent,
                "Content-Type": "application/json",
                "x-rapidapi-host": self.settings.rapidapi_host,
                "x-rapidapi-key": self.settings.rapidapi_key,
            }
        )
        retries = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1.0,
            status=2,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET",),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _retry_delay_seconds(self, response: requests.Response | None, fallback_seconds: float) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(float(retry_after), fallback_seconds)
                except ValueError:
                    pass
        return fallback_seconds

    def _quota_from_headers(self, response: requests.Response | None) -> RapidApiQuota:
        def as_int(value: str | None) -> int | None:
            if value is None or value == "":
                return None
            try:
                return int(value)
            except ValueError:
                return None

        if response is None:
            return RapidApiQuota()

        return RapidApiQuota(
            daily_limit=as_int(response.headers.get("x-ratelimit-requests-limit")),
            daily_remaining=as_int(response.headers.get("x-ratelimit-requests-remaining")),
            minute_limit=as_int(response.headers.get("x-ratelimit-limit")),
            minute_remaining=as_int(response.headers.get("x-ratelimit-remaining")),
        )

    def _get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        label: str,
        progress: Callable[[str], None] | None = None,
        max_rate_limit_retries: int = 4,
        base_delay_seconds: float = 1.5,
    ) -> tuple[Any, RapidApiQuota]:
        url = f"{self.settings.rapidapi_base_url}{path}"
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.settings.request_timeout_seconds,
                )
            except requests.exceptions.RetryError as exc:
                if attempt > max_rate_limit_retries:
                    raise RapidApiRateLimitError(
                        f"RapidAPI kept rate limiting {label} after {attempt - 1} retries."
                    ) from exc
                wait_seconds = min(60.0, max(base_delay_seconds * (2**attempt), 5.0))
                if progress is not None:
                    progress(
                        f"RapidAPI rate limit hit while fetching {label}; waiting {wait_seconds:.0f}s before retry {attempt}/{max_rate_limit_retries}."
                    )
                time.sleep(wait_seconds)
                continue

            if response.status_code == 429:
                if attempt > max_rate_limit_retries:
                    raise RapidApiRateLimitError(
                        f"RapidAPI kept rate limiting {label} after {attempt - 1} retries."
                    )
                wait_seconds = min(
                    60.0,
                    self._retry_delay_seconds(
                        response,
                        max(base_delay_seconds * (2**(attempt - 1)), 5.0),
                    ),
                )
                if progress is not None:
                    progress(
                        f"RapidAPI rate limit hit while fetching {label}; waiting {wait_seconds:.0f}s before retry {attempt}/{max_rate_limit_retries}."
                    )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response.json(), self._quota_from_headers(response)

    def fetch_racecards(
        self,
        race_date: date,
        *,
        progress: Callable[[str], None] | None = None,
        base_delay_seconds: float = 1.5,
    ) -> list[dict[str, Any]]:
        payload, _ = self._get_json(
            "/racecards",
            params={"date": race_date.isoformat()},
            label=f"racecards for {race_date.isoformat()}",
            progress=progress,
            base_delay_seconds=base_delay_seconds,
        )
        if not isinstance(payload, list):
            raise ValueError("RapidAPI racecards response was not a list.")
        return [item for item in payload if isinstance(item, dict)]

    def fetch_racecards_with_quota(
        self,
        race_date: date,
        *,
        progress: Callable[[str], None] | None = None,
        base_delay_seconds: float = 1.5,
    ) -> tuple[list[dict[str, Any]], RapidApiQuota]:
        payload, quota = self._get_json(
            "/racecards",
            params={"date": race_date.isoformat()},
            label=f"racecards for {race_date.isoformat()}",
            progress=progress,
            base_delay_seconds=base_delay_seconds,
        )
        if not isinstance(payload, list):
            raise ValueError("RapidAPI racecards response was not a list.")
        return [item for item in payload if isinstance(item, dict)], quota

    def fetch_race(
        self,
        race_id: str,
        *,
        progress: Callable[[str], None] | None = None,
        base_delay_seconds: float = 1.5,
    ) -> dict[str, Any]:
        payload, _ = self._get_json(
            f"/race/{race_id}",
            label=f"race {race_id}",
            progress=progress,
            base_delay_seconds=base_delay_seconds,
        )
        if not isinstance(payload, dict):
            raise ValueError(f"RapidAPI race response for {race_id} was not an object.")
        return payload


def _record_raw_fetch(
    session,
    source: str,
    source_url: str,
    http_status: int,
    payload_text: str,
) -> None:
    session.add(
        RawFetch(
            source=source,
            source_url=source_url,
            http_status=http_status,
            content_sha256=hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
            payload_text=payload_text,
        )
    )


def _rapidapi_race_status(racecard: dict[str, Any], detail: dict[str, Any]) -> str:
    canceled = _parse_bool(detail.get("canceled")) or _parse_bool(racecard.get("canceled"))
    if canceled:
        return "canceled"

    if _parse_bool(detail.get("finished")) or _parse_bool(racecard.get("finished")):
        return "resulted"

    greyhounds = detail.get("greyhounds")
    if isinstance(greyhounds, list) and any(_parse_int(item.get("position")) for item in greyhounds if isinstance(item, dict)):
        return "resulted"

    return "scheduled"


def _rapidapi_track_name(racecard: dict[str, Any]) -> str:
    return _stringify(racecard.get("dogTrack")) or "UNKNOWN"


def _filter_rapidapi_racecards(
    racecards: list[dict[str, Any]],
    include_finished: bool = False,
    include_canceled: bool = False,
    track_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected_tracks = {
        track_name.strip().casefold()
        for track_name in (track_names or [])
        if track_name and track_name.strip()
    }
    filtered: list[dict[str, Any]] = []
    for racecard in racecards:
        status = _rapidapi_race_status(racecard, {})
        if status == "resulted" and not include_finished:
            continue
        if status == "canceled" and not include_canceled:
            continue
        if selected_tracks and _rapidapi_track_name(racecard).casefold() not in selected_tracks:
            continue
        filtered.append(racecard)
    return filtered


def fetch_rapidapi_racecards(settings: Settings, race_date: date) -> list[dict[str, Any]]:
    client = RapidApiGreyhoundUkClient(settings=settings)
    return client.fetch_racecards(race_date)


def fetch_rapidapi_racecards_with_quota(
    settings: Settings,
    race_date: date,
) -> tuple[list[dict[str, Any]], RapidApiQuota]:
    client = RapidApiGreyhoundUkClient(settings=settings)
    return client.fetch_racecards_with_quota(race_date)


def rapidapi_race_status(racecard: dict[str, Any], detail: dict[str, Any] | None = None) -> str:
    return _rapidapi_race_status(racecard, detail or {})


def rapidapi_track_name(racecard: dict[str, Any]) -> str:
    return _rapidapi_track_name(racecard)


def filter_rapidapi_racecards(
    racecards: list[dict[str, Any]],
    include_finished: bool = False,
    include_canceled: bool = False,
    track_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    return _filter_rapidapi_racecards(
        racecards,
        include_finished=include_finished,
        include_canceled=include_canceled,
        track_names=track_names,
    )


def list_rapidapi_tracks(
    settings: Settings,
    race_date: date,
    include_finished: bool = False,
    include_canceled: bool = False,
) -> list[str]:
    racecards = fetch_rapidapi_racecards(settings, race_date)
    filtered = _filter_rapidapi_racecards(
        racecards,
        include_finished=include_finished,
        include_canceled=include_canceled,
    )
    return sorted({_rapidapi_track_name(racecard) for racecard in filtered})


def _rapidapi_to_rows(racecard: dict[str, Any], detail: dict[str, Any]) -> list[dict[str, Any]]:
    provider = "rapidapi_greyhound_uk"
    race_id = _stringify(detail.get("id_race") or racecard.get("id_race"))
    title = _stringify(detail.get("title") or racecard.get("title"))
    status = _rapidapi_race_status(racecard, detail)
    distance_raw = _stringify(detail.get("distance") or racecard.get("distance"))

    common = {
        "source": provider,
        "source_race_id": race_id or None,
        "track_name": _stringify(detail.get("dogTrack") or racecard.get("dogTrack")) or "UNKNOWN",
        "country_code": "GB",
        "timezone": "Europe/London",
        "scheduled_start": _stringify(detail.get("date") or racecard.get("date")) or None,
        "race_number": _parse_race_number(title),
        "distance_m": _parse_distance_m(distance_raw),
        "race_name": title or None,
        "status": status,
        "race_metadata_json": {
            "finished": _stringify(detail.get("finished") or racecard.get("finished")) or None,
            "canceled": _stringify(detail.get("canceled") or racecard.get("canceled")) or None,
            "distance_raw": distance_raw or None,
            "provider_title": title or None,
        },
    }

    greyhounds = detail.get("greyhounds")
    if not isinstance(greyhounds, list):
        return []

    rows: list[dict[str, Any]] = []
    for greyhound in greyhounds:
        if not isinstance(greyhound, dict):
            continue
        trap_number = _parse_int(greyhound.get("number"))
        dog_name = _stringify(greyhound.get("greyhound"))
        if trap_number is None or not dog_name:
            continue
        non_runner = _parse_bool(greyhound.get("non_runner"))
        row = {
            **common,
            "source_dog_id": _stringify(greyhound.get("id_greyhound")) or None,
            "dog_name": dog_name,
            "trainer_name": _stringify(greyhound.get("trainer")) or None,
            "trap_number": trap_number,
            "finish_position": _parse_int(greyhound.get("position")) if status == "resulted" else None,
            "beaten_distance": _stringify(greyhound.get("distance_beaten")) or None,
            "sp_text": _stringify(greyhound.get("sp")) or None,
            "sp_decimal": _parse_float(greyhound.get("sp")),
            "scratched": non_runner,
            "entry_metadata_json": {
                "form": _stringify(greyhound.get("form")) or None,
                "non_runner": _stringify(greyhound.get("non_runner")) or None,
                "odds": greyhound.get("odds") if isinstance(greyhound.get("odds"), list) else [],
            },
        }
        rows.append(row)
    return rows


def _persist_rapidapi_race(
    settings: Settings,
    racecard: dict[str, Any],
    detail_url: str,
    detail: dict[str, Any],
) -> tuple[int, set[str]]:
    detail_payload = json.dumps(detail, ensure_ascii=True)
    rows_processed = 0
    races_touched: set[str] = set()

    with session_scope(settings) as session:
        _record_raw_fetch(session, "rapidapi_greyhound_uk", detail_url, 200, detail_payload)
        for row in _rapidapi_to_rows(racecard, detail):
            provider = row["source"]
            track_row = _touch_track(session, provider, row)
            dog = _touch_dog(session, provider, row)
            race = _touch_race(session, provider, row, track_row)
            _touch_entry(session, row, race, dog)
            rows_processed += 1
            races_touched.add(race.race_key)

    return rows_processed, races_touched


def _gbgb_summary_items(client: GbgbClient, race_date: date, track: str | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = client.fetch_results_page(race_date, page=page, track=track)
        page_items = payload.get("items") or []
        if not page_items:
            break
        items.extend(page_items)
        meta = payload.get("meta") or {}
        if meta.get("page") == meta.get("pageCount"):
            break
        page += 1
    return items


def _gbgb_extract_meeting_and_race(
    detail: dict[str, Any] | list[Any],
    race_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if isinstance(detail, list):
        for meeting in detail:
            if not isinstance(meeting, dict):
                continue
            races = meeting.get("races")
            if not isinstance(races, list):
                continue
            for race in races:
                if not isinstance(race, dict):
                    continue
                candidate_race_id = str(race.get("raceId") or race.get("race_id") or race.get("id") or "")
                if candidate_race_id == race_id:
                    return meeting, race
        return None, None

    if isinstance(detail, dict):
        races = detail.get("races")
        if isinstance(races, list):
            for race in races:
                if not isinstance(race, dict):
                    continue
                candidate_race_id = str(race.get("raceId") or race.get("race_id") or race.get("id") or "")
                if candidate_race_id == race_id:
                    return detail, race
        return detail, detail

    return None, None


def _gbgb_to_rows(summary: dict[str, Any], detail: dict[str, Any]) -> list[dict[str, Any]]:
    provider = "gbgb"
    meeting_id = str(summary.get("meetingId") or summary.get("meeting") or summary.get("meeting_id") or "")
    race_id = str(summary.get("raceId") or summary.get("race_id") or summary.get("id") or "")
    meeting_payload, race_payload = _gbgb_extract_meeting_and_race(detail, race_id)
    race_date = None
    race_time = None
    if isinstance(race_payload, dict):
        race_date = race_payload.get("raceDate")
        race_time = race_payload.get("raceTime")
    scheduled_start = summary.get("dateTime") or summary.get("scheduledStart") or summary.get("startTime")
    if not scheduled_start and race_date and race_time:
        try:
            scheduled_start = datetime.strptime(f"{race_date} {race_time}", "%d/%m/%Y %H:%M:%S").replace(
                tzinfo=timezone.utc
            ).isoformat()
        except ValueError:
            scheduled_start = None
    common = {
        "source": provider,
        "source_race_id": race_id,
        "meeting_id": meeting_id,
        "source_track_id": str(summary.get("trackCode") or summary.get("track_id") or "").strip() or None,
        "track_name": (
            summary.get("trackName")
            or summary.get("track")
            or (meeting_payload.get("trackName") if isinstance(meeting_payload, dict) else None)
            or "UNKNOWN"
        ),
        "country_code": "GB",
        "timezone": "Europe/London",
        "scheduled_start": scheduled_start,
        "race_number": summary.get("raceNumber") or (race_payload.get("raceNumber") if isinstance(race_payload, dict) else None),
        "distance_m": summary.get("distance") or summary.get("distance_m") or (race_payload.get("raceDistance") if isinstance(race_payload, dict) else None),
        "grade": summary.get("grade") or summary.get("className") or (race_payload.get("raceClass") if isinstance(race_payload, dict) else None),
        "going": summary.get("going") or summary.get("goingName") or (race_payload.get("raceGoing") if isinstance(race_payload, dict) else None),
        "race_name": summary.get("raceName") or (race_payload.get("raceTitle") if isinstance(race_payload, dict) else None),
        "purse": race_payload.get("racePrizes") if isinstance(race_payload, dict) else None,
        "status": "resulted",
        "race_metadata_json": {
            "race_type": race_payload.get("raceType") if isinstance(race_payload, dict) else None,
            "race_handicap": race_payload.get("raceHandicap") if isinstance(race_payload, dict) else None,
            "forecast": race_payload.get("raceForecast") if isinstance(race_payload, dict) else None,
            "tricast": race_payload.get("raceTricast") if isinstance(race_payload, dict) else None,
            "meeting_date": meeting_payload.get("meetingDate") if isinstance(meeting_payload, dict) else None,
        },
    }

    runners = []
    if isinstance(race_payload, dict):
        traps = race_payload.get("traps")
        if isinstance(traps, list):
            runners = traps
    if not runners and isinstance(detail, dict):
        for key in ("runners", "dogs", "results", "raceTrapDog", "race_trap_dog"):
            value = detail.get(key)
            if isinstance(value, list):
                runners = value
                break

    rows: list[dict[str, Any]] = []
    for runner in runners:
        trap_number = runner.get("trap") or runner.get("trapNumber") or runner.get("idTrap")
        dog_name = runner.get("dogName") or runner.get("dog") or runner.get("name")
        if trap_number is None or not dog_name:
            continue
        row = {
            **common,
            "source_dog_id": runner.get("dogId") or runner.get("idDog"),
            "dog_name": dog_name,
            "dog_date_of_birth": runner.get("dogBorn") or runner.get("dateOfBirth"),
            "trainer_name": runner.get("trainerName") or runner.get("trainer"),
            "owner_name": runner.get("ownerName") or runner.get("owner"),
            "dog_sex": runner.get("dogSex") or runner.get("sex"),
            "sire_name": runner.get("dogSire") or runner.get("sireName"),
            "dam_name": runner.get("dogDam") or runner.get("damName"),
            "trap_number": trap_number,
            "weight_kg": runner.get("resultDogWeight") or runner.get("dogWeight"),
            "finish_position": runner.get("resultPosition") or runner.get("place") or runner.get("position"),
            "official_time_s": runner.get("resultRunTime") or runner.get("timeSec") or runner.get("time_s"),
            "sectional_s": runner.get("resultSectionalTime") or runner.get("sectional") or runner.get("sectional_s"),
            "beaten_distance": runner.get("resultBtnDistance") or runner.get("beatenDistance") or runner.get("distance"),
            "sp_text": runner.get("SP") or runner.get("startingPrice") or runner.get("sp"),
            "sp_numerator": runner.get("resultPriceNumerator"),
            "sp_denominator": runner.get("resultPriceDenominator"),
            "comment": runner.get("resultComment") or runner.get("comment") or runner.get("remarks"),
            "entry_metadata_json": {
                "trap_handicap": runner.get("trapHandicap"),
                "market_position": runner.get("resultMarketPos"),
                "market_count": runner.get("resultMarketCnt"),
                "adjusted_time": runner.get("resultAdjustedTime"),
                "colour": runner.get("dogColour"),
                "season": runner.get("dogSeason"),
            },
        }
        rows.append(row)
    return rows


def _persist_gbgb_detail(
    settings: Settings,
    summary: dict[str, Any],
    detail_url: str,
    detail: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> tuple[int, set[str]]:
    payload_text = json.dumps(detail, ensure_ascii=True)
    for attempt in range(1, 4):
        rows_processed = 0
        races_touched: set[str] = set()
        try:
            with session_scope(settings) as session:
                session.add(
                    RawFetch(
                        source="gbgb",
                        source_url=detail_url,
                        http_status=200,
                        content_sha256=hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
                        payload_text=payload_text,
                    )
                )
                for row in _gbgb_to_rows(summary, detail):
                    provider = row["source"]
                    track_row = _touch_track(session, provider, row)
                    dog = _touch_dog(session, provider, row)
                    race = _touch_race(session, provider, row, track_row)
                    _touch_entry(session, row, race, dog)
                    rows_processed += 1
                    races_touched.add(race.race_key)
            return rows_processed, races_touched
        except OperationalError as exc:
            if attempt == 3:
                raise
            if progress is not None:
                progress(f"DB write failed for {detail_url}; retrying ({attempt}/3): {exc}")
            time.sleep(min(5, attempt * 2))

    return 0, set()


def _update_ingestion_run(settings: Settings, run_id: int, **fields: Any) -> None:
    with session_scope(settings) as session:
        run = session.get(IngestionRun, run_id)
        if run is None:
            return
        for name, value in fields.items():
            setattr(run, name, value)


def _existing_gbgb_race_ids(settings: Settings, race_ids: list[str]) -> set[str]:
    normalized_ids = sorted({_stringify(race_id) for race_id in race_ids if _stringify(race_id)})
    if not normalized_ids:
        return set()

    with session_scope(settings) as session:
        statement = select(Race.provider_race_id).where(
            Race.provider == "gbgb",
            Race.provider_race_id.in_(normalized_ids),
        )
        return {race_id for race_id in session.scalars(statement) if race_id}


def _existing_provider_race_ids(settings: Settings, provider: str, race_ids: list[str]) -> set[str]:
    normalized_ids = sorted({_stringify(race_id) for race_id in race_ids if _stringify(race_id)})
    if not normalized_ids:
        return set()

    with session_scope(settings) as session:
        statement = select(Race.provider_race_id).where(
            Race.provider == provider,
            Race.provider_race_id.in_(normalized_ids),
        )
        return {race_id for race_id in session.scalars(statement) if race_id}


def ingest_gbgb_range(
    settings: Settings,
    start_date: date,
    end_date: date,
    track: str | None = None,
    delay_seconds: float = 1.0,
    start_race_index: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    init_database(settings)
    client = GbgbClient(settings=settings)
    rows_processed = 0
    races_touched: set[str] = set()
    races_skipped = 0

    def emit(message: str) -> None:
        if progress is not None:
            progress(message)

    with session_scope(settings) as session:
        run = IngestionRun(source="gbgb")
        session.add(run)
        session.flush()
        run_id = run.id

    try:
        current = start_date
        while current <= end_date:
            emit(f"Fetching GBGB summaries for {current.isoformat()}...")
            summaries = _gbgb_summary_items(client, current, track)
            emit(f"{current.isoformat()}: found {len(summaries)} races")
            existing_race_ids = _existing_gbgb_race_ids(
                settings,
                [
                    _stringify(summary.get("raceId") or summary.get("race_id") or summary.get("id"))
                    for summary in summaries
                ],
            )
            for index, summary in enumerate(summaries, start=1):
                if current == start_date and start_race_index is not None and index < start_race_index:
                    races_skipped += 1
                    continue
                meeting_id = summary.get("meetingId") or summary.get("meeting") or summary.get("meeting_id")
                race_id = summary.get("raceId") or summary.get("race_id") or summary.get("id")
                if not meeting_id or not race_id:
                    continue
                normalized_race_id = _stringify(race_id)
                if normalized_race_id in existing_race_ids:
                    races_skipped += 1
                    continue
                emit(f"{current.isoformat()}: fetching race {index}/{len(summaries)}")
                detail_url = f"{settings.gbgb_base_url}/results/meeting/{meeting_id}?raceId={race_id}"
                detail = client.fetch_meeting_detail(str(meeting_id), str(race_id))
                persisted_rows, persisted_races = _persist_gbgb_detail(
                    settings,
                    summary,
                    detail_url,
                    detail,
                    progress=progress,
                )
                rows_processed += persisted_rows
                races_touched.update(persisted_races)
                time.sleep(delay_seconds)
            emit(
                f"{current.isoformat()}: cumulative rows={rows_processed}, races={len(races_touched)}, skipped={races_skipped}"
            )
            current = date.fromordinal(current.toordinal() + 1)

        _update_ingestion_run(
            settings,
            run_id,
            status="completed",
            finished_at=datetime.now(timezone.utc),
            rows_processed=rows_processed,
            races_touched=len(races_touched),
            notes_json={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "track": track,
                "start_race_index": start_race_index,
                "races_skipped": races_skipped,
            },
        )
    except Exception as exc:
        try:
            _update_ingestion_run(
                settings,
                run_id,
                status="failed",
                finished_at=datetime.now(timezone.utc),
                rows_processed=rows_processed,
                races_touched=len(races_touched),
                error_text=str(exc),
            )
        except Exception:
            emit("Failed to update ingestion run status after the primary error.")
        raise

    return {
        "status": "completed",
        "rows_processed": rows_processed,
        "races_touched": len(races_touched),
        "races_skipped": races_skipped,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "start_race_index": start_race_index,
    }


def ingest_rapidapi_racecards(
    settings: Settings,
    race_date: date,
    track_names: list[str] | None = None,
    refresh_existing: bool = False,
    include_finished: bool = False,
    include_canceled: bool = False,
    delay_seconds: float = 0.5,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    init_database(settings)
    client = RapidApiGreyhoundUkClient(settings=settings)
    rows_processed = 0
    races_touched: set[str] = set()
    races_seen = 0
    races_skipped = 0
    races_skipped_existing = 0

    def emit(message: str) -> None:
        if progress is not None:
            progress(message)

    with session_scope(settings) as session:
        run = IngestionRun(source="rapidapi_greyhound_uk")
        session.add(run)
        session.flush()
        run_id = run.id

    try:
        emit(f"Fetching RapidAPI racecards for {race_date.isoformat()}...")
        racecards_url = f"{settings.rapidapi_base_url}/racecards?date={race_date.isoformat()}"
        racecards = client.fetch_racecards(race_date)
        with session_scope(settings) as session:
            _record_raw_fetch(
                session,
                "rapidapi_greyhound_uk",
                racecards_url,
                200,
                json.dumps(racecards, ensure_ascii=True),
            )
        filtered_racecards = _filter_rapidapi_racecards(
            racecards,
            include_finished=include_finished,
            include_canceled=include_canceled,
            track_names=track_names,
        )
        existing_race_ids = (
            set()
            if refresh_existing
            else _existing_provider_race_ids(
                settings,
                "rapidapi_greyhound_uk",
                [_stringify(racecard.get("id_race")) for racecard in filtered_racecards],
            )
        )
        emit(
            f"{race_date.isoformat()}: found {len(racecards)} racecards, "
            f"importing {len(filtered_racecards)} after filters"
        )

        for index, racecard in enumerate(filtered_racecards, start=1):
            races_seen += 1

            race_id = _stringify(racecard.get("id_race"))
            if not race_id:
                races_skipped += 1
                continue
            if race_id in existing_race_ids:
                races_skipped += 1
                races_skipped_existing += 1
                emit(
                    f"{race_date.isoformat()}: skipping race {index}/{len(filtered_racecards)} ({race_id}) because it is already imported."
                )
                continue

            emit(f"{race_date.isoformat()}: fetching race {index}/{len(filtered_racecards)} ({race_id})")
            detail_url = f"{settings.rapidapi_base_url}/race/{race_id}"
            detail = client.fetch_race(race_id)
            detail_status = _rapidapi_race_status(racecard, detail)
            if detail_status == "canceled" and not include_canceled:
                races_skipped += 1
                continue
            if detail_status == "resulted" and not include_finished:
                races_skipped += 1
                continue

            persisted_rows, persisted_races = _persist_rapidapi_race(
                settings,
                racecard=racecard,
                detail_url=detail_url,
                detail=detail,
            )
            rows_processed += persisted_rows
            races_touched.update(persisted_races)
            time.sleep(delay_seconds)

        _update_ingestion_run(
            settings,
            run_id,
            status="completed",
            finished_at=datetime.now(timezone.utc),
            rows_processed=rows_processed,
            races_touched=len(races_touched),
            notes_json={
                "date": race_date.isoformat(),
                "track_names": sorted(track_names or []),
                "refresh_existing": refresh_existing,
                "include_finished": include_finished,
                "include_canceled": include_canceled,
                "races_seen": races_seen,
                "races_skipped": races_skipped,
                "races_skipped_existing": races_skipped_existing,
            },
        )
    except Exception as exc:
        try:
            _update_ingestion_run(
                settings,
                run_id,
                status="failed",
                finished_at=datetime.now(timezone.utc),
                rows_processed=rows_processed,
                races_touched=len(races_touched),
                error_text=str(exc),
            )
        except Exception:
            emit("Failed to update ingestion run status after the primary error.")
        raise

    return {
        "status": "completed",
        "date": race_date.isoformat(),
        "track_names": sorted(track_names or []),
        "refresh_existing": refresh_existing,
        "rows_processed": rows_processed,
        "races_touched": len(races_touched),
        "races_seen": races_seen,
        "races_skipped": races_skipped,
        "races_skipped_existing": races_skipped_existing,
        "include_finished": include_finished,
        "include_canceled": include_canceled,
    }
