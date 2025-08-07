"""Tests for the regexes module."""

import re

import pytest

from leglove.regexes import (
    FEDERAL_COURT_REPORTERS,
    FOOTNOTE_REGEX,
    ID_CITATION_REGEX,
    JUDICIAL_OPINION_CITATION_REGEX,
    LAW_CITATION_REGEX,
    REGEX_TOKENS,
    REGEXES,
    STAR_PAGINATION_REGEX,
)


class TestStarPaginationRegex:
    """Tests for star pagination regex."""

    def test_star_pagination_match(self) -> None:
        """Test matching star pagination."""
        text = '<span class="star-pagination">*123</span>'
        match = re.search(STAR_PAGINATION_REGEX, text)
        assert match is not None
        assert match.group() == text

    def test_star_pagination_no_match(self) -> None:
        """Test non-matching text."""
        text = '<span class="other">*123</span>'
        match = re.search(STAR_PAGINATION_REGEX, text)
        assert match is None


class TestLawCitationRegex:
    """Tests for law citation regex."""

    def test_law_citation_basic(self) -> None:
        """Test basic law citation."""
        text = "§ 123.45"
        match = re.search(LAW_CITATION_REGEX, text)
        assert match is not None

    def test_law_citation_complex(self) -> None:
        """Test complex law citation with subsections."""
        text = "§§ 123.45(a)(1)"
        match = re.search(LAW_CITATION_REGEX, text)
        assert match is not None


class TestIdCitationRegex:
    """Tests for Id. citation regex."""

    def test_id_citation_match(self) -> None:
        """Test matching Id. citation."""
        text = "<i>Id.,</i> at 123"
        match = re.search(ID_CITATION_REGEX, text)
        assert match is not None
        assert match.group() == text

    def test_id_citation_with_range(self) -> None:
        """Test Id. citation with page range."""
        text = "<i>Id.,</i> at 123-125"
        match = re.search(ID_CITATION_REGEX, text)
        assert match is not None


class TestFootnoteRegex:
    """Tests for footnote regex."""

    def test_footnote_match(self) -> None:
        """Test matching footnote link."""
        text = '<a class="footnote" href="#fn1" id="ref1">1</a>'
        match = re.search(FOOTNOTE_REGEX, text)
        assert match is not None
        assert match.group(1) == "#fn1"
        assert match.group(2) == "ref1"
        assert match.group(3) == "1"


class TestJudicialOpinionCitationRegex:
    """Tests for judicial opinion citation regex."""

    def test_basic_citation(self) -> None:
        """Test basic judicial opinion citation."""
        text = "123 F.3d 456"
        match = re.search(JUDICIAL_OPINION_CITATION_REGEX, text)
        assert match is not None
        assert match.group(1) == "123"
        assert match.group(3) == "456"

    def test_citation_with_year(self) -> None:
        """Test citation with year."""
        text = "123 F.3d 456 (2020)"
        match = re.search(JUDICIAL_OPINION_CITATION_REGEX, text)
        assert match is not None

    def test_us_citation(self) -> None:
        """Test U.S. Supreme Court citation."""
        text = "123 U.S. 456"
        match = re.search(JUDICIAL_OPINION_CITATION_REGEX, text)
        assert match is not None


class TestFederalCourtReporters:
    """Tests for federal court reporters list."""

    def test_reporters_list_type(self) -> None:
        """Test that FEDERAL_COURT_REPORTERS is a list."""
        assert isinstance(FEDERAL_COURT_REPORTERS, list)
        assert len(FEDERAL_COURT_REPORTERS) > 0

    def test_reporters_contain_common_ones(self) -> None:
        """Test that common reporters are included."""
        assert "U.S." in FEDERAL_COURT_REPORTERS
        assert "F.3d" in FEDERAL_COURT_REPORTERS
        assert "F. Supp." in FEDERAL_COURT_REPORTERS


class TestRegexesAndTokens:
    """Tests for REGEXES and REGEX_TOKENS lists."""

    def test_regexes_list_type(self) -> None:
        """Test that REGEXES is a list of strings."""
        assert isinstance(REGEXES, list)
        assert all(isinstance(regex, str) for regex in REGEXES)

    def test_regex_tokens_list_type(self) -> None:
        """Test that REGEX_TOKENS is a list of strings."""
        assert isinstance(REGEX_TOKENS, list)
        assert all(isinstance(token, str) for token in REGEX_TOKENS)

    def test_regexes_and_tokens_same_length(self) -> None:
        """Test that REGEXES and REGEX_TOKENS have the same length."""
        assert len(REGEXES) == len(REGEX_TOKENS)

    def test_regexes_compile(self) -> None:
        """Test that all regexes compile successfully."""
        for regex in REGEXES:
            try:
                re.compile(regex)
            except re.error:
                pytest.fail(f"Regex failed to compile: {regex}")

    def test_expected_tokens(self) -> None:
        """Test that expected tokens are present."""
        expected_tokens = [
            "JUDICIAL_OPINION_CITATION",
            "FOOTNOTE",
            "ID_CITATION",
            "LAW_CITATION",
            "STAR_PAGINATION",
        ]
        assert REGEX_TOKENS == expected_tokens
