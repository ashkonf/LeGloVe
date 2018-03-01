import os
import sys
import re
import time
import collections
import json

import bs4

# Reads the contents of a file:
def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

# Extracts primary HTML from a raw judicial opinion JSON object:
def extract_html(json_object):
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

# Checks if an opinion's HTML is well formatted:
def is_well_formatted(html):
    return len(html.strip()) > 0 and re.match("^<", html) and not re.match("^<pre ", html)

# Cleans an opinion's HTML, removing some garbage and all HTML meta data:
def clean_html(html):
    soup = bs4.BeautifulSoup(html)
    
    for tag in soup.find_all("sup"):
        tag.extract()
    html = str(soup)
    html = re.sub("&amp;", "&", html)
    html = re.sub("\xe2", "'", html)

    soup = bs4.BeautifulSoup(html)
    paragraphs = [paragraph.get_text() for paragraph in soup.find_all("p")]
    return "\n\n".join(paragraphs)

# Extracts plain text from an opinion, returning None if the opinion isn't well formatted:
def extract_text(file_path):
    json_object = json.loads(read_file(file_path))
    raw_html = extract_html(json_object)
    if is_well_formatted(raw_html):
        return clean_html(raw_html)
    else:
        return ""
