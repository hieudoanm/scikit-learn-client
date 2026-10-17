"""CLI entry point for skl — scikit-learn-client CLI.

`skl run` trains and exports a model from a CSV entirely offline, with every
input (file location, features, label, algorithm, output location) accepted
either as flags or interactively.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd

from .helpers import (
    TASK_TYPES,
    choose_algorithm,
    choose_columns,
    choose_task_type,
    print_error,
    print_info,
    print_success,
)
from .local import (
    algorithms_for_task,
    default_output_path,
    train_and_export,
)

TASK_TYPE_NAMES = [name for name, _ in TASK_TYPES]


@click.group()
def cli() -> None:
    """scikit-learn-client CLI — train and export ONNX from a CSV."""


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    help="Output artefact path (default: <csv stem>.<ext>)",
)
@click.option(
    "--task-type",
    "-t",
    type=click.Choice(TASK_TYPE_NAMES),
    help="Task type (prompted if omitted)",
)
@click.option(
    "--features",
    "-f",
    help="Comma-separated feature columns (prompted if omitted)",
)
@click.option(
    "--label",
    "-l",
    help="Label column (prompted if omitted and required)",
)
@click.option(
    "--algorithm",
    "-a",
    help="Algorithm id (prompted if omitted)",
)
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["onnx", "pkl", "joblib"]),
    default="onnx",
    show_default=True,
    help="Export format",
)
@click.option(
    "--no-interactive",
    is_flag=True,
    help="Fail instead of prompting for missing input",
)
def run(
    csv_path: str,
    output: str | None,
    task_type: str | None,
    features: str | None,
    label: str | None,
    algorithm: str | None,
    export_format: str,
    no_interactive: bool,
) -> None:
    """Train a model from CSV and export an artefact (offline, no backend)."""
    interactive = not no_interactive and sys.stdin.isatty()
    csv_file = Path(csv_path)

    df = pd.read_csv(csv_file)
    columns = [{"name": name} for name in df.columns.tolist()]
    print_info(f"Read {csv_file.name}: {len(df)} rows x {len(columns)} columns")

    # --- task type ---
    if not task_type:
        if interactive:
            task_type = choose_task_type(columns)
        else:
            print_error("--task-type is required in non-interactive mode")
            sys.exit(1)

    # --- features ---
    if not features:
        if interactive:
            feature_columns = choose_columns(columns, "Select feature columns")
        else:
            print_error("--features is required in non-interactive mode")
            sys.exit(1)
    else:
        feature_columns = [name.strip() for name in features.split(",") if name.strip()]
    if not feature_columns:
        print_error("At least one feature column is required")
        sys.exit(1)

    # --- label ---
    label_needs = task_type in ("classification", "regression", "semi_supervised")
    if label_needs and not label:
        if interactive:
            label = choose_columns(columns, "Select label column", multi=False)
        else:
            print_error("--label is required in non-interactive mode")
            sys.exit(1)

    # --- algorithm ---
    if not algorithm:
        if interactive:
            choices = algorithms_for_task(task_type)
            algorithm = choose_algorithm(choices, task_type)["algorithmId"]
        else:
            print_error("--algorithm is required in non-interactive mode")
            sys.exit(1)

    out_path = Path(output) if output else default_output_path(csv_file, export_format)
    print_info(f"Training {task_type} with {algorithm} on {', '.join(feature_columns)}")
    train_and_export(
        csv_path=csv_file,
        output=out_path,
        task_type=task_type,
        feature_columns=feature_columns,
        label_column=label,
        algorithm_id=algorithm,
        export_format=export_format,
    )
    print_success(f"Saved {out_path}")


if __name__ == "__main__":
    cli()
