# Identifying star paginations:
STAR_PAGINATION_REGEX = "<span class=\"star-pagination\">\*\d+</span>"

# Identifying citations of laws (in contrast to cases):
LAW_CITATION_REGEX = u"\u00A7" + "{1,2} ?[\w\.]+(?: ?\([a-z]+\)(?: ?\(\d+\)(?: (?:(?:and)|(?:or)) \(\d+\))?)?)?"

# Identifying citations via Id.:
ID_CITATION_REGEX = "(<i>[Ii]d\.,</i> at \d+(?:-\d+)?)"

# Ientifying footnote links:
FOOTNOTE_REGEX = "<a class=\"footnote\" href=\"([\w\#\?]+)\" id=\"(\w+)\">(\w+)</a>"

# Identifying judicial opinion citations:
FEDERAL_COURT_REPORTERS = ["U.S.", "S.Ct.", "L.E.", "L.E. 2d", "L.Ed.2d", "F.", "F.2d", "F.3d", "F. App'x", "Fed. Cl.", "F. Supp.", "F. Supp.2d", "F.R.D.", "B.R.", "T.C.", "Vet. App.", "M.J."]
# FEDERAL_COURT_REPORTERS_PATTERN = "|".join(FEDERAL_COURT_REPORTERS)
PAGE_NUMBER_REGEX = "(?:, ((?:\d+(?:\-\d+)?)(?:(?: &)? n\. \d+)? ?))"
YEAR_REGEX = "(?: ?(?:, )?(\(\d\d\d\d\)))"
JUDICIAL_OPINION_CITATION_REGEX = "(\d+) (%s) (?:at )?(\d+)(?:%s)?(?:%s)?" % ("U.S.", PAGE_NUMBER_REGEX, YEAR_REGEX)

# List of all regexes above
REGEXES = [STAR_PAGINATION_REGEX, LAW_CITATION_REGEX, ID_CITATION_REGEX, FOOTNOTE_REGEX, FOOTNOTE_REGEX, JUDICIAL_OPINION_CITATION_REGEX]