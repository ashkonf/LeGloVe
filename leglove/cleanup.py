import json
import re
from typing import Any, Dict

import bs4


def read_file(file_path: str) -> str:
    """
    Read the contents of a file.

    Args:
        file_path: Path to the file to read.

    Returns:
        The contents of the file as a string.

    Example:
        >>> content = read_file("/path/to/file.txt")
        >>> print(len(content))
        1234
    """
    with open(file_path) as file:
        return file.read()


def extract_html(json_object: Dict[str, Any]) -> str:
    """
    Extract primary HTML content from a judicial opinion JSON object.

    Attempts to extract HTML content from various fields in order of preference:
    html_with_citations, html_lawbox, html, html_columbia, and finally plain_text.

    Args:
        json_object: Dictionary containing judicial opinion data.

    Returns:
        HTML or plain text content from the opinion.

    Example:
        >>> opinion = {"html_with_citations": "<p>Court opinion...</p>"}
        >>> html = extract_html(opinion)
        >>> print(html[:10])
        <p>Court o
    """
    if json_object["html_with_citations"]:
        return json_object["html_with_citations"]
    elif json_object["html_lawbox"]:
        return json_object["html_lawbox"]
    elif json_object["html"]:
        return json_object["html"]
    elif json_object["html_columbia"]:
        return json_object["html_columbia"]
    else:
        return json_object["plain_text"]


def is_well_formatted(html: str) -> bool:
    """
    Check if an opinion's HTML is well formatted.

    Determines if HTML content is properly formatted by checking if it's non-empty,
    starts with an HTML tag, and is not a preformatted text block.

    Args:
        html: HTML content to check.

    Returns:
        True if the HTML is well formatted, False otherwise.

    Example:
        >>> is_well_formatted("<p>Good HTML</p>")
        True
        >>> is_well_formatted("<pre>Bad HTML</pre>")
        False
    """
    return (
        len(html.strip()) > 0
        and bool(re.match("^<", html))
        and not bool(re.match("^<pre ", html))
    )


def clean_html(html: str) -> str:
    """
    Clean an opinion's HTML by removing metadata and extracting text content.

    Removes superscript tags, fixes HTML entities, and extracts paragraph text
    from the HTML content.

    Args:
        html: Raw HTML content to clean.

    Returns:
        Cleaned plain text with paragraphs separated by double newlines.

    Example:
        >>> html = "<p>First paragraph</p><sup>1</sup><p>Second paragraph</p>"
        >>> clean_text = clean_html(html)
        >>> print(clean_text)
        First paragraph

        Second paragraph
    """
    soup = bs4.BeautifulSoup(html, "html5lib")

    for tag in soup.find_all("sup"):
        tag.extract()
    html = str(soup)
    html = re.sub("&amp;", "&", html)
    html = re.sub("\xe2", "'", html)

    soup = bs4.BeautifulSoup(html, "html5lib")
    paragraphs = [paragraph.get_text() for paragraph in soup.find_all("p")]
    return "\n\n".join(paragraphs)


def extract_text(file_path: str) -> str:
    """
    Extract plain text from a judicial opinion JSON file.

    Reads a JSON file containing a judicial opinion, extracts HTML content,
    and returns cleaned plain text if the HTML is well formatted.

    Args:
        file_path: Path to the JSON file containing the judicial opinion.

    Returns:
        Cleaned plain text from the opinion, or empty string if not well formatted.

    Example:
        >>> text = extract_text("/path/to/opinion.json")
        >>> print(text[:50])
        The Supreme Court held that the defendant's rights...
    """
    json_object = json.loads(read_file(file_path))
    raw_html = extract_html(json_object)
    if is_well_formatted(raw_html):
        return clean_html(raw_html)
    else:
        return ""
