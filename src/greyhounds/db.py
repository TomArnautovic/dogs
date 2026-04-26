from __future__ import annotations

import re
from contextlib import contextmanager
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Generator, Iterable, TypeVar

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, LargeBinary, Numeric, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import Settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class Track(TimestampMixin, Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    track_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_track_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    country_code: Mapped[str | None] = mapped_column(String(8), nullable=True)
    timezone_name: Mapped[str | None] = mapped_column("timezone", String(64), nullable=True)

    races: Mapped[list["Race"]] = relationship(back_populates="track")


class Owner(TimestampMixin, Base):
    __tablename__ = "owners"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    dogs: Mapped[list["Dog"]] = relationship(back_populates="owner")


class Trainer(TimestampMixin, Base):
    __tablename__ = "trainers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    dogs: Mapped[list["Dog"]] = relationship(back_populates="trainer")


class Dog(TimestampMixin, Base):
    __tablename__ = "dogs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dog_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_dog_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    sex: Mapped[str | None] = mapped_column(String(16), nullable=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    sire_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    dam_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("owners.id"), nullable=True)
    trainer_id: Mapped[int | None] = mapped_column(ForeignKey("trainers.id"), nullable=True)

    owner: Mapped[Owner | None] = relationship(back_populates="dogs")
    trainer: Mapped[Trainer | None] = relationship(back_populates="dogs")
    entries: Mapped[list["RaceEntry"]] = relationship(back_populates="dog")


class Race(TimestampMixin, Base):
    __tablename__ = "races"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    race_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_race_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    meeting_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"), nullable=False)
    scheduled_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    off_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    race_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    race_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    distance_m: Mapped[int | None] = mapped_column(Integer, nullable=True)
    grade: Mapped[str | None] = mapped_column(String(64), nullable=True)
    going: Mapped[str | None] = mapped_column(String(64), nullable=True)
    purse: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="scheduled", nullable=False)
    is_completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    track: Mapped[Track] = relationship(back_populates="races")
    entries: Mapped[list["RaceEntry"]] = relationship(back_populates="race", cascade="all, delete-orphan", order_by="RaceEntry.trap_number")
    prediction_runs: Mapped[list["PredictionRun"]] = relationship(back_populates="race")


class RaceEntry(TimestampMixin, Base):
    __tablename__ = "race_entries"
    __table_args__ = (
        UniqueConstraint("race_id", "trap_number", name="uq_race_entries_race_trap"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entry_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id"), nullable=False)
    dog_id: Mapped[int | None] = mapped_column(ForeignKey("dogs.id"), nullable=True)
    trap_number: Mapped[int] = mapped_column(Integer, nullable=False)
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    finish_position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    official_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    sectional_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    beaten_distance: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sp_text: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sp_decimal: Mapped[float | None] = mapped_column(Float, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    vacant: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    scratched: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    race: Mapped[Race] = relationship(back_populates="entries")
    dog: Mapped[Dog | None] = relationship(back_populates="entries")
    prediction_entries: Mapped[list["PredictionEntry"]] = relationship(back_populates="race_entry")


class RawFetch(Base):
    __tablename__ = "raw_fetches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_text: Mapped[str] = mapped_column(Text, nullable=False)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running", nullable=False)
    rows_processed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    races_touched: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    notes_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running", nullable=False)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    train_race_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    validation_race_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    prediction_runs: Mapped[list["PredictionRun"]] = relationship(back_populates="training_run")


class PredictionRun(Base):
    __tablename__ = "prediction_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id"), nullable=False)
    training_run_id: Mapped[int | None] = mapped_column(ForeignKey("training_runs.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    predicted_order_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    race: Mapped[Race] = relationship(back_populates="prediction_runs")
    training_run: Mapped[TrainingRun | None] = relationship(back_populates="prediction_runs")
    entries: Mapped[list["PredictionEntry"]] = relationship(back_populates="prediction_run", cascade="all, delete-orphan")


class PredictionEntry(Base):
    __tablename__ = "prediction_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_run_id: Mapped[int] = mapped_column(ForeignKey("prediction_runs.id"), nullable=False)
    race_entry_id: Mapped[int | None] = mapped_column(ForeignKey("race_entries.id"), nullable=True)
    dog_id: Mapped[int | None] = mapped_column(ForeignKey("dogs.id"), nullable=True)
    predicted_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    win_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    prediction_run: Mapped[PredictionRun] = relationship(back_populates="entries")
    race_entry: Mapped[RaceEntry | None] = relationship(back_populates="prediction_entries")
    dog: Mapped[Dog | None] = relationship()


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "unknown"


def _engine_connect_args(database_url: str) -> dict[str, object]:
    if database_url.startswith(("postgresql+psycopg://", "postgresql://", "postgres://")):
        # Supabase poolers/PgBouncer can error on psycopg prepared statements.
        return {"prepare_threshold": None}
    return {}


@lru_cache(maxsize=4)
def _get_engine(database_url: str):
    return create_engine(
        database_url,
        future=True,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args=_engine_connect_args(database_url),
    )


def get_engine(settings: Settings):
    return _get_engine(settings.database_url)


@lru_cache(maxsize=4)
def _get_session_factory(database_url: str):
    return sessionmaker(bind=_get_engine(database_url), expire_on_commit=False, future=True)


def get_session_factory(settings: Settings):
    return _get_session_factory(settings.database_url)


def init_database(settings: Settings) -> None:
    settings.ensure_directories()
    Base.metadata.create_all(get_engine(settings))


@contextmanager
def session_scope(settings: Settings) -> Generator[Session, None, None]:
    session = get_session_factory(settings)()
    try:
        yield session
        session.commit()
    except Exception as exc:
        try:
            session.rollback()
        except Exception as rollback_exc:
            exc.add_note(f"Session rollback failed: {rollback_exc}")
            raise exc.with_traceback(exc.__traceback__)
        raise
    finally:
        session.close()


T = TypeVar("T", Owner, Trainer)


def find_or_create_named_entity(session: Session, model: type[T], name: str | None) -> T | None:
    if not name:
        return None
    slug = slugify(name)
    entity = session.scalar(select(model).where(model.slug == slug))
    if entity is None:
        entity = model(slug=slug, name=name.strip())
        session.add(entity)
        session.flush()
    else:
        entity.name = name.strip()
    return entity


def latest_completed_training_run(session: Session) -> TrainingRun | None:
    statement = (
        select(TrainingRun)
        .where(TrainingRun.status == "completed")
        .order_by(TrainingRun.finished_at.desc(), TrainingRun.started_at.desc())
        .limit(1)
    )
    return session.scalar(statement)


def latest_usable_training_run(session: Session) -> TrainingRun | None:
    statement = (
        select(TrainingRun)
        .where(
            TrainingRun.status.in_(["completed", "interrupted"]),
            TrainingRun.artifact_path.is_not(None),
        )
        .order_by(TrainingRun.finished_at.desc(), TrainingRun.started_at.desc())
        .limit(1)
    )
    return session.scalar(statement)


def recent_training_runs(session: Session, limit: int = 20) -> list[TrainingRun]:
    statement = select(TrainingRun).order_by(TrainingRun.started_at.desc()).limit(limit)
    return list(session.scalars(statement))


def recent_prediction_runs(session: Session, limit: int = 20) -> list[PredictionRun]:
    statement = select(PredictionRun).order_by(PredictionRun.created_at.desc()).limit(limit)
    return list(session.scalars(statement))


def active_training_run(session: Session) -> TrainingRun | None:
    statement = (
        select(TrainingRun)
        .where(TrainingRun.status == "running", TrainingRun.finished_at.is_(None))
        .order_by(TrainingRun.started_at.desc())
        .limit(1)
    )
    return session.scalar(statement)


def latest_training_run(session: Session) -> TrainingRun | None:
    statement = select(TrainingRun).order_by(TrainingRun.started_at.desc()).limit(1)
    return session.scalar(statement)
