from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import typer

from .config import Settings
from .db import init_database
from .ingest import GbgbApiError, import_runner_csv, ingest_gbgb_range, ingest_rapidapi_racecards
from .ml import TrainingConfig, predict_upcoming_races, train_model


app = typer.Typer(help="Greyhound ingestion, ANN training, and prediction tools.")


def _parse_iso_date(value: str, option_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f"{option_name} must be in YYYY-MM-DD format."
        ) from exc


def _cli_progress_callback(event: dict[str, object]) -> None:
    event_type = event.get("event")
    if event_type == "start":
        data_summary = event.get("data_summary")
        skipped = (
            data_summary.get("missing_or_invalid_finish_order_races")
            if isinstance(data_summary, dict)
            else None
        )
        skipped_text = (
            f", {skipped} skipped missing/invalid result order"
            if isinstance(skipped, int) and skipped
            else ""
        )
        typer.echo(
            "Starting permutation training "
            f"({event.get('train_race_count')} train races, "
            f"{event.get('validation_race_count')} validation races, "
            f"{event.get('total_epochs')} epochs"
            f"{skipped_text})"
        )
        return

    if event_type == "epoch":
        epoch = int(event["epoch"])
        total_epochs = int(event["total_epochs"])
        progress_ratio = epoch / total_epochs if total_epochs else 0.0
        bar_width = 24
        filled = min(bar_width, int(round(progress_ratio * bar_width)))
        bar = "#" * filled + "-" * (bar_width - filled)
        line = (
            f"\r[{bar}] {epoch}/{total_epochs} "
            f"train_loss={float(event['train_loss']):.4f} "
            f"val_loss={float(event['validation_loss']):.4f} "
            f"val_win={float(event['validation_winner_accuracy']):.3f} "
            f"val_exact={float(event['validation_exact_order_accuracy']):.3f} "
            f"best_val={float(event['best_validation_loss']):.4f} "
            f"samples={int(event['samples'])} "
            f"elapsed={float(event['elapsed_seconds']):.0f}s"
        )
        sys.stdout.write(line)
        sys.stdout.flush()
        if epoch == total_epochs:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return

    if event_type == "complete":
        typer.echo(
            "Training complete "
            f"(winner_accuracy={float(event['summary']['winner_accuracy']):.3f}, "
            f"exact_order_accuracy={float(event['summary']['exact_order_accuracy']):.3f}, "
            f"elapsed={float(event['elapsed_seconds']):.0f}s)"
        )
        return

    if event_type == "failed":
        typer.echo(
            f"Training failed after {float(event['elapsed_seconds']):.0f}s: {event['error']}",
            err=True,
        )


@app.command("init-db")
def init_db() -> None:
    """Create the database tables for local development."""
    settings = Settings.from_env()
    init_database(settings)
    typer.echo(f"Database initialised for {settings.database_url}")


@app.command("ingest-csv")
def ingest_csv(
    csv_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    source: str = typer.Option("manual", help="Source label to store with the rows."),
) -> None:
    settings = Settings.from_env()
    summary = import_runner_csv(settings, csv_path=csv_path, source=source)
    typer.echo(json.dumps(summary, indent=2))


@app.command("ingest-gbgb")
def ingest_gbgb(
    start_date: str = typer.Option(..., help="Start date in YYYY-MM-DD format."),
    end_date: str = typer.Option(..., help="End date in YYYY-MM-DD format."),
    track: str | None = typer.Option(None, help="Optional track name filter."),
    delay_seconds: float = typer.Option(1.0, min=0.0, help="Delay between requests."),
    start_race_index: int | None = typer.Option(
        None,
        min=1,
        help="Optional 1-based summary index to resume from on the start date, matching the progress output X/N.",
    ),
) -> None:
    settings = Settings.from_env()
    parsed_start_date = _parse_iso_date(start_date, "--start-date")
    parsed_end_date = _parse_iso_date(end_date, "--end-date")
    try:
        summary = ingest_gbgb_range(
            settings,
            start_date=parsed_start_date,
            end_date=parsed_end_date,
            track=track,
            delay_seconds=delay_seconds,
            start_race_index=start_race_index,
            progress=typer.echo,
        )
    except GbgbApiError as exc:
        typer.secho(f"GBGB import stopped: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(summary, indent=2))


@app.command("ingest-rapidapi-racecards")
def ingest_rapidapi(
    race_date: str = typer.Option(..., "--date", help="Racecard date in YYYY-MM-DD format."),
    track: list[str] | None = typer.Option(
        None,
        "--track",
        help="Optional track filter. Pass multiple times to import multiple tracks.",
    ),
    refresh_existing: bool = typer.Option(
        False,
        help="Re-fetch races that already exist in the database instead of skipping them.",
    ),
    include_finished: bool = typer.Option(
        False,
        help="Include races already marked finished by the API.",
    ),
    include_canceled: bool = typer.Option(
        False,
        help="Include races marked canceled by the API.",
    ),
    delay_seconds: float = typer.Option(0.5, min=0.0, help="Delay between race detail requests."),
) -> None:
    settings = Settings.from_env()
    parsed_date = _parse_iso_date(race_date, "--date")
    summary = ingest_rapidapi_racecards(
        settings,
        race_date=parsed_date,
        track_names=track,
        refresh_existing=refresh_existing,
        include_finished=include_finished,
        include_canceled=include_canceled,
        delay_seconds=delay_seconds,
        progress=typer.echo,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("train")
def train(
    resume_from_artifact: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional saved model artifact to continue training from.",
    ),
    epochs: int = typer.Option(50),
    batch_size: int = typer.Option(32),
    learning_rate: float = typer.Option(1e-3),
    hidden_size_1: int = typer.Option(128),
    hidden_size_2: int = typer.Option(64),
    dropout: float = typer.Option(0.15),
    validation_fraction: float = typer.Option(0.2),
    weight_decay: float = typer.Option(1e-5),
    seed: int = typer.Option(42),
    max_runners: int = typer.Option(8),
    min_completed_races: int = typer.Option(25),
    permutations_per_race: int = typer.Option(
        24,
        help="Non-winning candidate finish orders sampled alongside the true order per race per epoch.",
    ),
    permutation_runner_limit: int = typer.Option(6, help="Maximum runners allowed for exhaustive permutation scoring."),
    early_stopping_patience: int = typer.Option(
        20,
        help="Stop after this many epochs without validation-loss improvement. Use 0 to disable.",
    ),
) -> None:
    settings = Settings.from_env()
    config = TrainingConfig(
        model_type="permutation",
        resume_from_artifact=str(resume_from_artifact) if resume_from_artifact else None,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size_1=hidden_size_1,
        hidden_size_2=hidden_size_2,
        dropout=dropout,
        validation_fraction=validation_fraction,
        weight_decay=weight_decay,
        seed=seed,
        max_runners=max_runners,
        min_completed_races=min_completed_races,
        permutations_per_race=permutations_per_race,
        permutation_runner_limit=permutation_runner_limit,
        early_stopping_patience=early_stopping_patience,
    )
    summary = train_model(settings, config, progress=_cli_progress_callback)
    typer.echo(json.dumps(summary["summary"], indent=2))
    typer.echo(f"Artifact: {summary['artifact_path']}")
    typer.echo(f"Report: {summary['report_path']}")


@app.command("predict")
def predict(
    artifact_path: Path | None = typer.Option(None, exists=True, dir_okay=False),
    race_key: list[str] | None = typer.Option(None, help="Optional race_key filter. Can be passed multiple times."),
    track: list[str] | None = typer.Option(None, help="Optional track filter. Can be passed multiple times."),
) -> None:
    settings = Settings.from_env()
    predictions = predict_upcoming_races(
        settings,
        artifact_path=artifact_path,
        race_keys=race_key,
        track_names=track,
    )
    typer.echo(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    app()
