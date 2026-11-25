"""Problem text normalization and hashing for test-set deduplication (T-023)."""
from __future__ import annotations

import hashlib
import re
import unicodedata

# Matches explicit LaTeX horizontal spacing commands and non-breaking tilde space.
LATEX_SPACE_RE = re.compile(
    r"\\(?:qquad|quad|enspace|thinspace|medspace|thickspace|,|;|:|!|[ \t])"
    r"|\\hspace\*?\{[^}]*\}"
    r"|~"
)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_problem_text(text: str) -> str:
    """Normalize math problem text for leakage-free hash matching.

    Rules (T-023 Stage 1):
    1. Unicode NFKC normalization (unifies full-width and half-width characters).
    2. Strip explicit LaTeX spacing commands (\\,, \\;, \\:, \\!, \\quad, \\qquad,
       \\enspace, \\thinspace, \\medspace, \\thickspace, \\ , \\hspace{...}, ~).
    3. Remove all remaining whitespace characters.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = LATEX_SPACE_RE.sub("", normalized)
    normalized = WHITESPACE_RE.sub("", normalized)
    return normalized


def problem_text_hash(text: str) -> str:
    """Return SHA-256 hex digest of the normalized problem text."""
    return hashlib.sha256(normalize_problem_text(text).encode("utf-8")).hexdigest()
