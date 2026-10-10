"""Audience and editorial rules without optional rendering dependencies."""

import re


def assert_student_content(text, name):
    forbidden = [
        r"(?im)^#{1,6} .*instructor",
        r"(?im)^#{1,6} .*worked answers",
        r"(?im)^#{1,6}\s+(?:\d+[.)]\s+)?(?:reproduce(?:\s|$)|reproduction(?:\s|$)|reproducibility\s+(?:evidence|notes))",
        r"(?i)no previous .*experience is required",
        r"(?i)(?:duration|estimated time|takes|allow)\s*[:=]?\s*\d+(?:\s*[–-]\s*\d+)?\s+minutes\b",
        r"(?i)\*\*(?:Duration|Prerequisites):",
        r"\]\([^)]*instructors/",
        r"\]\(reference\.json\)",
    ]
    for pattern in forbidden:
        if re.search(pattern, text):
            raise ValueError(f"Student material contains excluded content ({pattern}): {name}")
