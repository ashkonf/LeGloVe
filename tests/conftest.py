"""Pytest configuration and fixtures for LeGloVe tests."""

import json
import os
import tempfile
from typing import Any, Dict, Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_json_opinion() -> Dict[str, Any]:
    """Sample judicial opinion JSON data for testing."""
    return {
        "html_with_citations": "<p>This is a test opinion with <sup>1</sup> citations.</p><p>Second paragraph.</p>",
        "html_lawbox": None,
        "html": None,
        "html_columbia": None,
        "plain_text": "Fallback plain text",
    }


@pytest.fixture
def sample_json_file(temp_dir: str, sample_json_opinion: Dict[str, Any]) -> str:
    """Create a sample JSON file for testing."""
    file_path = os.path.join(temp_dir, "test_opinion.json")
    with open(file_path, "w") as f:
        json.dump(sample_json_opinion, f)
    return file_path


@pytest.fixture
def sample_corpus_dir(temp_dir: str) -> str:
    """Create a sample corpus directory structure for testing."""
    juris_dir = os.path.join(temp_dir, "scotus")
    os.makedirs(juris_dir)

    opinions = [
        {
            "html_with_citations": "<p>First opinion about legal matters.</p>",
            "html_lawbox": None,
            "html": None,
            "html_columbia": None,
            "plain_text": "First opinion plain text",
        },
        {
            "html_with_citations": "<p>Second opinion discussing court procedures.</p>",
            "html_lawbox": None,
            "html": None,
            "html_columbia": None,
            "plain_text": "Second opinion plain text",
        },
    ]

    for i, opinion in enumerate(opinions):
        file_path = os.path.join(juris_dir, f"opinion_{i}.json")
        with open(file_path, "w") as f:
            json.dump(opinion, f)

    return temp_dir


@pytest.fixture
def sample_plain_text() -> str:
    """Sample plain text for testing tokenization."""
    return "The court held in Smith v. Jones, 123 F.3d 456 (2020), that the defendant's rights were violated."
