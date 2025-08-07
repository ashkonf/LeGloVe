from typing import List

# Identifying star paginations:
STAR_PAGINATION_REGEX = r"<span class=\"star-pagination\">\*\d+</span>"

# Identifying citations of laws (in contrast to cases):
LAW_CITATION_REGEX = (
    "\u00a7"
    + r"{1,2} ?[\w\.]+(?: ?\([a-z]+\)(?: ?\(\d+\)(?: (?:(?:and)|(?:or)) \(\d+\))?)?)?"
)

# Identifying citations via Id.:
ID_CITATION_REGEX = r"(<i>[Ii]d\.,</i> at \d+(?:-\d+)?)"

# Identifying footnote links:
FOOTNOTE_REGEX = r"<a class=\"footnote\" href=\"([\w\#\?]+)\" id=\"(\w+)\">(\w+)</a>"

# Identifying judicial opinion citations:
FEDERAL_COURT_REPORTERS: List[str] = [
    "U.S.",
    "S.Ct.",
    "L.E.",
    "L.E. 2d",
    "L.Ed.2d",
    "F.",
    "F.2d",
    "F.3d",
    "F. App'x",
    "Fed. Cl.",
    "F. Supp.",
    "F. Supp.2d",
    "F.R.D.",
    "B.R.",
    "T.C.",
    "Vet. App.",
    "M.J.",
]
FEDERAL_COURT_REPORTERS_PATTERN = "|".join(FEDERAL_COURT_REPORTERS)
PAGE_NUMBER_REGEX = r"(?:, ((?:\d+(?:\-\d+)?)(?:(?: &)? n\. \d+)? ?))"
YEAR_REGEX = r"(?: ?(?:, )?(\(\d\d\d\d\)))"
PAGE_SUFFIX_REGEX = r"( ?n\. \d+)?"
JUDICIAL_OPINION_CITATION_REGEX = rf"(\d+) ({FEDERAL_COURT_REPORTERS_PATTERN}) (?:at )?(\d+)(?:{PAGE_NUMBER_REGEX})?(?:{YEAR_REGEX})?({PAGE_SUFFIX_REGEX})?"

# List of all regexes above
REGEXES: List[str] = [
    JUDICIAL_OPINION_CITATION_REGEX,
    FOOTNOTE_REGEX,
    ID_CITATION_REGEX,
    LAW_CITATION_REGEX,
    STAR_PAGINATION_REGEX,
]

# List of corresponding dummy tokens for above regexes/citations
REGEX_TOKENS: List[str] = [
    "JUDICIAL_OPINION_CITATION",
    "FOOTNOTE",
    "ID_CITATION",
    "LAW_CITATION",
    "STAR_PAGINATION",
]
