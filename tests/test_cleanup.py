"""Tests for the cleanup module."""

import json
import os

from leglove.cleanup import (
    clean_html,
    extract_html,
    extract_text,
    is_well_formatted,
    read_file,
)


class TestReadFile:
    """Tests for the read_file function."""

    def test_read_file_success(self, temp_dir: str) -> None:
        """Test reading a file successfully."""
        content = "Test file content\nSecond line"
        file_path = os.path.join(temp_dir, "test.txt")

        with open(file_path, "w") as f:
            f.write(content)

        result = read_file(file_path)
        assert result == content

    def test_read_file_empty(self, temp_dir: str) -> None:
        """Test reading an empty file."""
        file_path = os.path.join(temp_dir, "empty.txt")

        with open(file_path, "w"):
            pass

        result = read_file(file_path)
        assert result == ""


class TestExtractHtml:
    """Tests for the extract_html function."""

    def test_extract_html_with_citations(self) -> None:
        """Test extracting HTML with citations (highest priority)."""
        json_obj = {
            "html_with_citations": "<p>Citations HTML</p>",
            "html_lawbox": "<p>Lawbox HTML</p>",
            "html": "<p>Regular HTML</p>",
            "html_columbia": "<p>Columbia HTML</p>",
            "plain_text": "Plain text",
        }

        result = extract_html(json_obj)
        assert result == "<p>Citations HTML</p>"

    def test_extract_html_lawbox_fallback(self) -> None:
        """Test extracting HTML lawbox when citations not available."""
        json_obj = {
            "html_with_citations": None,
            "html_lawbox": "<p>Lawbox HTML</p>",
            "html": "<p>Regular HTML</p>",
            "html_columbia": "<p>Columbia HTML</p>",
            "plain_text": "Plain text",
        }

        result = extract_html(json_obj)
        assert result == "<p>Lawbox HTML</p>"

    def test_extract_html_plain_text_fallback(self) -> None:
        """Test extracting plain text when no HTML available."""
        json_obj = {
            "html_with_citations": None,
            "html_lawbox": None,
            "html": None,
            "html_columbia": None,
            "plain_text": "Plain text fallback",
        }

        result = extract_html(json_obj)
        assert result == "Plain text fallback"


class TestIsWellFormatted:
    """Tests for the is_well_formatted function."""

    def test_well_formatted_html(self) -> None:
        """Test well-formatted HTML."""
        html = "<p>This is good HTML</p>"
        assert is_well_formatted(html) is True

    def test_empty_html(self) -> None:
        """Test empty HTML."""
        assert is_well_formatted("") is False
        assert is_well_formatted("   ") is False

    def test_preformatted_html(self) -> None:
        """Test preformatted HTML (should be rejected)."""
        html = "<pre >This is preformatted</pre>"
        assert is_well_formatted(html) is False

    def test_non_html_text(self) -> None:
        """Test non-HTML text."""
        text = "This is just plain text"
        result = is_well_formatted(text)
        assert result is False


class TestCleanHtml:
    """Tests for the clean_html function."""

    def test_clean_html_basic(self) -> None:
        """Test basic HTML cleaning."""
        html = "<p>First paragraph</p><p>Second paragraph</p>"
        result = clean_html(html)
        assert result == "First paragraph\n\nSecond paragraph"

    def test_clean_html_remove_superscript(self) -> None:
        """Test removal of superscript tags."""
        html = "<p>Text with<sup>1</sup> footnote</p>"
        result = clean_html(html)
        assert result == "Text with footnote"

    def test_clean_html_fix_entities(self) -> None:
        """Test fixing HTML entities."""
        html = "<p>Text with &amp; ampersand</p>"
        result = clean_html(html)
        assert result == "Text with & ampersand"

    def test_clean_html_empty(self) -> None:
        """Test cleaning empty HTML."""
        html = "<html><body></body></html>"
        result = clean_html(html)
        assert result == ""


class TestExtractText:
    """Tests for the extract_text function."""

    def test_extract_text_success(self, sample_json_file: str) -> None:
        """Test successful text extraction."""
        result = extract_text(sample_json_file)
        assert "This is a test opinion with" in result
        assert "citations." in result
        assert "Second paragraph." in result

    def test_extract_text_malformed_html(self, temp_dir: str) -> None:
        """Test text extraction with malformed HTML."""
        json_obj = {
            "html_with_citations": "Not HTML at all",
            "html_lawbox": None,
            "html": None,
            "html_columbia": None,
            "plain_text": "Plain text",
        }

        file_path = os.path.join(temp_dir, "malformed.json")
        with open(file_path, "w") as f:
            json.dump(json_obj, f)

        result = extract_text(file_path)
        assert result == ""

    def test_extract_text_preformatted(self, temp_dir: str) -> None:
        """Test text extraction with preformatted HTML."""
        json_obj = {
            "html_with_citations": "<pre>Preformatted text</pre>",
            "html_lawbox": None,
            "html": None,
            "html_columbia": None,
            "plain_text": "Plain text",
        }

        file_path = os.path.join(temp_dir, "preformatted.json")
        with open(file_path, "w") as f:
            json.dump(json_obj, f)

        result = extract_text(file_path)
        assert result == ""
