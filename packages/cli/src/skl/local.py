"""Self-contained training pipeline: read CSV, train, export artefact."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .catalogue import ALGORITHMS, LABEL_REQUIRED_TASKS, TASK_TYPES
from .export import export_joblib, export_onnx, export_pickle
from .pipeline import train_pipeline

EXPORTERS: dict[str, object] = {
    "onnx": export_onnx,
    "pkl": export_pickle,
    "joblib": export_joblib,
}

EXTENSIONS: dict[str, str] = {
    "onnx": ".onnx",
    "pkl": ".pkl",
    "joblib": ".joblib",
}


def algorithms_for_task(task_type: str) -> list[dict]:
    """Catalogue entries for a task type, mirroring the backend algorithm list."""
    if task_type not in TASK_TYPES:
        raise ValueError(f"UNKNOWN_TASK_TYPE: {task_type}")
    return [
        {**entry, "algorithmId": aid}
        for aid, entry in ALGORITHMS.items()
        if entry["task_type"] == task_type
    ]


def default_output_path(csv_path: Path, export_format: str) -> Path:
    """Default artefact path next to the input CSV."""
    return csv_path.with_suffix(EXTENSIONS[export_format])


def train_and_export(
    csv_path: Path,
    output: Path,
    task_type: str,
    feature_columns: list[str],
    label_column: str | None,
    algorithm_id: str,
    export_format: str = "onnx",
) -> Path:
    """Train on csv_path and write the exported artefact to output."""
    df = pd.read_csv(csv_path)

    columns = set(df.columns)
    requested = set(feature_columns)
    if label_column:
        requested.add(label_column)
    missing = sorted(requested - columns)
    if missing:
        raise ValueError(f"columns not found in CSV: {missing}")

    if task_type in LABEL_REQUIRED_TASKS and not label_column:
        raise ValueError(f"label column required for task type {task_type}")

    entry = ALGORITHMS.get(algorithm_id)
    if entry is None or entry["task_type"] != task_type:
        raise ValueError(f"algorithm {algorithm_id!r} is not valid for task type {task_type}")

    pipeline, metrics, _charts = train_pipeline(
        feature_columns=feature_columns,
        label_column=label_column,
        task_type=task_type,
        algorithm_id=algorithm_id,
        hyperparameters=dict(entry["default_hyperparameters"]),
        df=df,
    )

    exporter = EXPORTERS[export_format]
    blob = (
        exporter(pipeline, df[feature_columns]) if export_format == "onnx" else exporter(pipeline)
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)
    print(f"  metrics: {metrics}")
    return output
