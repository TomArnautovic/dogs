from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url


@dataclass(slots=True)
class Settings:
    database_url: str
    artifacts_dir: Path
    gbgb_base_url: str
    rapidapi_base_url: str
    rapidapi_host: str
    rapidapi_key: str | None
    http_user_agent: str
    request_timeout_seconds: int
    max_runners: int

    @classmethod
    def from_env(cls) -> "Settings":
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts")).resolve()
        return cls(
            database_url=_normalize_database_url(os.getenv("DATABASE_URL", "sqlite:///greyhounds.sqlite3")),
            artifacts_dir=artifacts_dir,
            gbgb_base_url=os.getenv("GBGB_BASE_URL", "https://api.gbgb.org.uk/api").rstrip("/"),
            rapidapi_base_url=os.getenv(
                "RAPIDAPI_BASE_URL",
                "https://greyhound-racing-uk.p.rapidapi.com",
            ).rstrip("/"),
            rapidapi_host=os.getenv(
                "RAPIDAPI_HOST",
                "greyhound-racing-uk.p.rapidapi.com",
            ).strip(),
            rapidapi_key=(os.getenv("RAPIDAPI_KEY") or "").strip() or None,
            http_user_agent=os.getenv(
                "HTTP_USER_AGENT",
                "greyhounds-lab/0.1 (+change-me@example.com)",
            ),
            request_timeout_seconds=_int_env("REQUEST_TIMEOUT_SECONDS", 20),
            max_runners=_int_env("MAX_RUNNERS", 8),
        )

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.artifacts_dir / "logs"

    def ensure_directories(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
