"""Interactive prompts and display helpers for skl CLI."""

from __future__ import annotations

import sys

import click
import inquirer
from tabulate import tabulate

TASK_TYPES = [
    ("classification", "Classification — predict a categorical label"),
    ("regression", "Regression — predict a continuous value"),
    ("semi_supervised", "Semi-supervised — classification with some unlabelled rows"),
    ("clustering", "Clustering — group rows without a label"),
    ("anomaly_detection", "Anomaly detection — flag unusual rows"),
    ("reinforcement_learning", "Reinforcement learning — learn from experience columns"),
]


def print_info(msg: str) -> None:
    click.echo(click.style("ℹ ", fg="blue") + msg)


def print_success(msg: str) -> None:
    click.echo(click.style("✔ ", fg="green") + msg)


def print_warning(msg: str) -> None:
    click.echo(click.style("⚠ ", fg="yellow") + msg)


def print_error(msg: str) -> None:
    click.echo(click.style("✖ ", fg="red") + msg, err=True)


def format_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def choose_task_type(columns: list[dict]) -> str:
    """Prompt the user to pick a task type."""
    choices = [f"{name} — {desc}" for name, desc in TASK_TYPES]
    questions = [
        inquirer.List(
            "task_type",
            message="Select a task type",
            choices=choices,
        )
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        print_warning("No selection made.")
        sys.exit(0)
    raw = answers["task_type"]
    return raw.split(" —")[0]


def choose_columns(columns: list[dict], message: str, multi: bool = True) -> list[str] | str:
    """Prompt the user to pick one or more columns."""
    names = [c["name"] for c in columns]
    if multi:
        questions = [
            inquirer.Checkbox(
                "columns",
                message=message,
                choices=names,
            )
        ]
    else:
        questions = [
            inquirer.List(
                "column",
                message=message,
                choices=names,
            )
        ]
    answers = inquirer.prompt(questions)
    if not answers:
        print_warning("No selection made.")
        sys.exit(0)
    key = "columns" if multi else "column"
    return answers[key]


def choose_algorithm(algorithms: list[dict], task_type: str) -> dict:
    """Prompt the user to pick an algorithm."""
    choices = [f"{a['name']} ({a['algorithmId']})" for a in algorithms]
    questions = [
        inquirer.List(
            "algorithm",
            message=f"Select an algorithm for {task_type}",
            choices=choices,
        )
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        print_warning("No selection made.")
        sys.exit(0)
    raw = answers["algorithm"]
    index = choices.index(raw)
    return algorithms[index]


def confirm_export(artefacts: list[dict]) -> bool:
    """Ask the user whether to download artefacts."""
    names = [f"{a['format']} ({a['filename']})" for a in artefacts]
    questions = [
        inquirer.Confirm(
            "export",
            message=f"Export {len(artefacts)} artefact(s): {', '.join(names)}?",
            default=True,
        )
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        return False
    return answers["export"]


def display_job(job: dict) -> None:
    """Pretty-print a single job."""
    click.echo()
    click.echo("── Job ─────────────────────────────────────")
    click.echo(f"  Status:        {job.get('status')}")
    click.echo(f"  Task type:     {job.get('taskType')}")
    click.echo(f"  Algorithm:     {job.get('algorithmId')}")
    click.echo(f"  Features:      {', '.join(job.get('featureColumns', []))}")
    if job.get("labelColumn"):
        click.echo(f"  Label:         {job['labelColumn']}")
    metrics = job.get("metrics") or {}
    if metrics:
        click.echo("  Metrics:")
        for k, v in metrics.items():
            click.echo(f"    {k}: {v}")
    artefacts = job.get("artefacts", [])
    if artefacts:
        click.echo("  Artefacts:")
        for a in artefacts:
            click.echo(f"    {a['format']} — {a['filename']} ({format_size(a['byteSize'])})")
    if job.get("error"):
        click.echo(f"  Error:         {job['error']}")
    click.echo("──────────────────────────────────────────────")
    click.echo()


def display_jobs(jobs: list[dict]) -> None:
    """Pretty-print a list of jobs as a table."""
    if not jobs:
        print_warning("No jobs found.")
        return
    rows = [
        [
            j.get("jobId", "")[:8],
            j.get("status", ""),
            j.get("taskType", ""),
            j.get("algorithmId", ""),
            j.get("name") or "",
            j.get("createdAt", ""),
        ]
        for j in jobs
    ]
    headers = ["ID", "Status", "Task", "Algorithm", "Name", "Created"]
    click.echo(tabulate(rows, headers=headers, tablefmt="github"))
