import html
import json
import re
from typing import Any, Dict

import bs4


def read_file(file_path: str) -> str:
    """Return the contents of a file as a string."""
    with open(file_path) as file:
        return file.read()


def extract_html(json_object: Dict[str, Any]) -> str:
    """Extract primary HTML content from a judicial opinion JSON object."""
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
    """Return True if HTML is non-empty, starts with a tag, and isn't preformatted."""
    return (
        len(html.strip()) > 0
        and bool(re.match("^<", html))
        and not bool(re.match("^<pre ", html))
    )


def clean_html(html_content: str) -> str:
    """Remove metadata from opinion HTML and extract paragraph text."""
    soup = bs4.BeautifulSoup(html_content, "html5lib")

    for tag in soup.find_all("sup"):
        tag.extract()
    html_text = html.unescape(str(soup))

    soup = bs4.BeautifulSoup(html_text, "html5lib")
    paragraphs = [paragraph.get_text() for paragraph in soup.find_all("p")]
    return "\n\n".join(paragraphs)


def extract_text(file_path: str) -> str:
    """Extract cleaned plain text from a judicial opinion JSON file."""
    json_object = json.loads(read_file(file_path))
    raw_html = extract_html(json_object)
    if is_well_formatted(raw_html):
        return clean_html(raw_html)
    else:
        return ""
