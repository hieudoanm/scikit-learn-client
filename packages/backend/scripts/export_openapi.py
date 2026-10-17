#!/usr/bin/env python3
"""Export the skl OpenAPI document from the running FastAPI application.

The committed contract lives in ``docs/06-api-specification/backend/openapi.yaml``.
This script regenerates it from the code, so the document can never drift away
from the routes that are actually served.

Usage:

    # Rewrite the committed document.
    uv run python scripts/export_openapi.py

    # Fail (exit code 1) when the committed document is out of date.
    uv run python scripts/export_openapi.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET = REPO_ROOT / 'docs' / '06-api-specification' / 'backend' / 'openapi.yaml'


def build_spec() -> dict[str, Any]:
    """Return the OpenAPI document of the skl application."""
    sys.path.insert(0, str(REPO_ROOT / 'packages' / 'backend'))

    from main import create_app  # noqa: PLC0415 — the app lives outside the package

    return create_app().openapi()


def render(spec: dict[str, Any]) -> str:
    """Render the OpenAPI document as YAML."""
    return yaml.safe_dump(
        spec,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=100,
    )


def main() -> int:
    """Export (or check) the committed OpenAPI document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--check',
        action='store_true',
        help='exit non-zero when the committed document differs from the code',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=TARGET,
        help=f'path of the document to write (default: {TARGET})',
    )
    args = parser.parse_args()

    rendered = render(build_spec())

    if args.check:
        if not args.output.exists():
            print(f'{args.output} does not exist; run the export without --check.')
            return 1
        if args.output.read_text(encoding='utf-8') != rendered:
            print(
                f'{args.output} is out of date with the code.\n'
                'Regenerate it with: uv run python scripts/export_openapi.py'
            )
            return 1
        print(f'{args.output} is up to date.')
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding='utf-8')
    print(f'Wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
