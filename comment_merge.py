"""
Comment merging utilities for mood tagging.

Mood tags are written into the standard "comment" field of audio files so they
appear in DJ software like Rekordbox, which has no native "Mood" field. To
coexist with other tools that also write to the comment field (notably Mixed
In Key, which prepends a prefix like "Energy 7 - 8A -"), mood data is wrapped
in a bracketed marker that can be reliably found and replaced on re-runs.

Format
------
Plain:
    [MOOD: Happy; Energetic; Uplifting]
With confidence percentages:
    [MOOD: Happy 87%; Energetic 72%; Uplifting 65%]

Behaviour
---------
The marker is appended to the end of the existing comment, separated by a
single space. On re-runs, any prior [MOOD: ...] block in the comment is
stripped before the new one is appended, so the comment never accumulates
stale mood data. The semicolon separator inside the brackets matches the
existing convention used elsewhere in this project for genre lists.

Known limitation
----------------
A user who hand-writes "[MOOD: ...]" as part of their own free-form comment
will have it overwritten on the next analysis run. The marker is reserved.
"""

import re
from typing import List, Optional, Sequence


# Matches the [MOOD: ...] marker plus any surrounding whitespace, so that
# stripping it on a re-run leaves no awkward gaps. Case-insensitive so that
# hand-edited or older variants (e.g. lowercase "mood:") also get cleaned up.
# `[^\]]*` matches the inner contents up to the closing bracket -- the marker
# itself never contains a literal `]`.
MOOD_MARKER_RE = re.compile(r'\s*\[MOOD:[^\]]*\]\s*', re.IGNORECASE)


def build_mood_marker(items: Sequence[str],
                      confidences: Optional[Sequence[float]] = None) -> str:
    """Build a bracketed mood marker string.

    Args:
        items: Pre-formatted mood label strings (e.g. ['Happy', 'Energetic']).
            Formatting (capitalisation, etc.) is the caller's responsibility;
            this function does not transform the labels.
        confidences: Optional parallel sequence of confidences in [0, 1].
            When provided, each label is suffixed with a rounded percentage
            (e.g. 'Happy 87%').

    Returns:
        A string like '[MOOD: Happy; Energetic]' or
        '[MOOD: Happy 87%; Energetic 72%]'. Returns an empty string if
        `items` is empty.

    Raises:
        ValueError: if `confidences` is provided but has a different length
            than `items`.
    """
    if not items:
        return ''
    if confidences is not None:
        if len(confidences) != len(items):
            raise ValueError(
                "items and confidences must be the same length "
                f"(got {len(items)} and {len(confidences)})"
            )
        parts: List[str] = [
            f"{label} {round(conf * 100)}%"
            for label, conf in zip(items, confidences)
        ]
    else:
        parts = list(items)
    return f"[MOOD: {'; '.join(parts)}]"


def merge_mood_into_comment(existing_comment: Optional[str],
                            mood_marker: str) -> str:
    """Strip any prior [MOOD: ...] block from the comment and append a new one.

    The function is safe to call repeatedly: running it twice with the same
    `mood_marker` is idempotent. It also coexists with tools that prepend
    their own data to the comment (like Mixed In Key), because the mood
    marker is always appended as a suffix and only the [MOOD: ...] block
    itself is touched.

    Args:
        existing_comment: The current comment text (may be None or empty).
        mood_marker: The marker to append, typically the output of
            `build_mood_marker()`. If empty, the function just strips any
            prior marker and returns the cleaned comment.

    Returns:
        The comment with any prior mood marker removed and the new marker
        appended. Whitespace is normalised so that adjacent markers or
        unusual spacing don't leave double spaces behind.
    """
    cleaned = MOOD_MARKER_RE.sub(' ', existing_comment or '')
    # Collapse any runs of whitespace produced by the substitution and trim
    # the ends in one pass.
    cleaned = ' '.join(cleaned.split())
    if not mood_marker:
        return cleaned
    if not cleaned:
        return mood_marker
    return f"{cleaned} {mood_marker}"
