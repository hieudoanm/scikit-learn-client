"""Tests for the committed OpenAPI document (scenario 40).

The committed `docs/06-api-specification/backend/openapi.yaml` must match the
schema the application serves, so a route or a model change cannot drift away
from the documented contract.
"""

import sys
from dataclasses import replace
from pathlib import Path

import yaml
from fastapi.testclient import TestClient
from main import create_app
from src.config import get_settings

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parents[1]
COMMITTED = REPO_ROOT / 'docs' / '06-api-specification' / 'backend' / 'openapi.yaml'


def _app():
    """Build an app that always serves the schema document."""
    return create_app(replace(get_settings(), enable_interactive_docs=True))


def _committed_document() -> dict:
    """Parse the committed OpenAPI document."""
    with open(COMMITTED, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def test_the_committed_document_matches_the_application():
    """The committed YAML is the schema the app generates."""
    assert _committed_document() == _app().openapi()


def test_the_served_schema_matches_the_committed_document():
    """`GET /openapi.json` serves exactly the committed document."""
    client = TestClient(_app())
    response = client.get('/openapi.json')

    assert response.status_code == 200
    assert response.json() == _committed_document()


def test_every_documented_route_is_present():
    """The committed document lists every route the application serves."""
    committed = _committed_document()
    generated = _app().openapi()

    assert set(generated['paths']) == set(committed['paths'])
    assert set(generated['components']['schemas']) == set(committed['components']['schemas'])


def test_the_export_script_reports_the_committed_document_as_current():
    """`scripts/export_openapi.py --check` succeeds on a clean tree."""
    scripts = REPO_ROOT / 'packages' / 'backend' / 'scripts'
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    from export_openapi import build_spec, render  # noqa: PLC0415

    assert render(build_spec()) == COMMITTED.read_text(encoding='utf-8')
