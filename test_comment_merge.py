"""
Tests for comment_merge.

Run from the project root:
    python -m unittest test_comment_merge -v
"""

import unittest

from comment_merge import (
    MOOD_MARKER_RE,
    build_mood_marker,
    merge_mood_into_comment,
)


class BuildMoodMarkerTests(unittest.TestCase):
    """build_mood_marker formats the bracketed marker string."""

    def test_empty_items_returns_empty_string(self):
        self.assertEqual(build_mood_marker([]), '')

    def test_single_item(self):
        self.assertEqual(build_mood_marker(['Happy']), '[MOOD: Happy]')

    def test_multiple_items_semicolon_separated(self):
        self.assertEqual(
            build_mood_marker(['Happy', 'Energetic', 'Uplifting']),
            '[MOOD: Happy; Energetic; Uplifting]',
        )

    def test_does_not_transform_label_casing(self):
        # The function takes pre-formatted labels and should pass them
        # through unchanged.
        self.assertEqual(
            build_mood_marker(['happy', 'ENERGETIC']),
            '[MOOD: happy; ENERGETIC]',
        )

    def test_with_confidences_appends_percentages(self):
        result = build_mood_marker(
            ['Happy', 'Energetic'],
            confidences=[0.87, 0.72],
        )
        self.assertEqual(result, '[MOOD: Happy 87%; Energetic 72%]')

    def test_confidence_rounding_half_up(self):
        # 0.875 -> 88, 0.874 -> 87 (banker's rounding may apply in Python --
        # round() uses banker's rounding, so 0.875 actually rounds to nearest
        # even. We assert the Python-standard behaviour rather than half-up.)
        result = build_mood_marker(
            ['A', 'B', 'C'],
            confidences=[0.874, 0.876, 1.0],
        )
        self.assertEqual(result, '[MOOD: A 87%; B 88%; C 100%]')

    def test_confidence_zero_and_one_boundaries(self):
        result = build_mood_marker(['Low', 'High'], confidences=[0.0, 1.0])
        self.assertEqual(result, '[MOOD: Low 0%; High 100%]')

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            build_mood_marker(['A', 'B'], confidences=[0.5])

    def test_empty_items_with_empty_confidences_returns_empty(self):
        # The empty-items short-circuit fires before length validation, which
        # is the expected behaviour: no items means no marker.
        self.assertEqual(build_mood_marker([], confidences=[]), '')


class MergeMoodIntoCommentTests(unittest.TestCase):
    """merge_mood_into_comment handles all the cases the tagger will hit."""

    MARKER = '[MOOD: Happy; Energetic]'
    OTHER_MARKER = '[MOOD: Sad; Mellow]'

    # --- empty / None inputs -------------------------------------------------

    def test_none_existing_comment_returns_just_marker(self):
        self.assertEqual(
            merge_mood_into_comment(None, self.MARKER),
            self.MARKER,
        )

    def test_empty_existing_comment_returns_just_marker(self):
        self.assertEqual(
            merge_mood_into_comment('', self.MARKER),
            self.MARKER,
        )

    def test_whitespace_only_existing_comment_returns_just_marker(self):
        self.assertEqual(
            merge_mood_into_comment('   \t  ', self.MARKER),
            self.MARKER,
        )

    # --- appending to existing content --------------------------------------

    def test_existing_comment_no_prior_marker_appends(self):
        self.assertEqual(
            merge_mood_into_comment('Killer drop at 1:30', self.MARKER),
            'Killer drop at 1:30 [MOOD: Happy; Energetic]',
        )

    def test_mik_prefix_only_is_preserved(self):
        # MIK writes 'Energy X - Yk - ' with a trailing dash and space when
        # there's no other comment text.
        result = merge_mood_into_comment('Energy 7 - 8A - ', self.MARKER)
        self.assertEqual(
            result,
            'Energy 7 - 8A - [MOOD: Happy; Energetic]',
        )

    def test_mik_prefix_plus_user_comment_is_preserved(self):
        result = merge_mood_into_comment(
            'Energy 7 - 8A - Killer drop at 1:30',
            self.MARKER,
        )
        self.assertEqual(
            result,
            'Energy 7 - 8A - Killer drop at 1:30 [MOOD: Happy; Energetic]',
        )

    # --- replacing prior markers -------------------------------------------

    def test_prior_marker_is_replaced_no_duplication(self):
        existing = 'Killer drop [MOOD: Sad; Mellow]'
        result = merge_mood_into_comment(existing, self.MARKER)
        self.assertEqual(result, 'Killer drop [MOOD: Happy; Energetic]')

    def test_idempotent_under_repeat(self):
        # Running the merger twice with the same marker must be a no-op
        # after the first run -- this is critical for re-runs of the tagger.
        first = merge_mood_into_comment('Existing comment', self.MARKER)
        second = merge_mood_into_comment(first, self.MARKER)
        self.assertEqual(first, second)

    def test_idempotent_with_mik_prefix(self):
        existing = 'Energy 7 - 8A - Killer drop'
        first = merge_mood_into_comment(existing, self.MARKER)
        second = merge_mood_into_comment(first, self.MARKER)
        self.assertEqual(first, second)
        # And the MIK prefix is still intact.
        self.assertTrue(second.startswith('Energy 7 - 8A - '))

    def test_marker_only_existing_returns_just_new_marker(self):
        result = merge_mood_into_comment(self.OTHER_MARKER, self.MARKER)
        self.assertEqual(result, self.MARKER)

    def test_marker_at_start_of_existing_comment(self):
        result = merge_mood_into_comment(
            '[MOOD: Sad] Killer drop',
            self.MARKER,
        )
        self.assertEqual(result, 'Killer drop [MOOD: Happy; Energetic]')

    def test_marker_in_middle_of_existing_comment(self):
        result = merge_mood_into_comment(
            'pre [MOOD: Sad] post',
            self.MARKER,
        )
        self.assertEqual(result, 'pre post [MOOD: Happy; Energetic]')

    def test_multiple_prior_markers_are_all_stripped(self):
        # Defensive: if some earlier bug or external tool produced multiple
        # markers, we should still recover cleanly.
        existing = '[MOOD: Old1] middle [MOOD: Old2] tail'
        result = merge_mood_into_comment(existing, self.MARKER)
        self.assertEqual(result, 'middle tail [MOOD: Happy; Energetic]')

    def test_case_insensitive_marker_stripping(self):
        # Lowercase or mixed-case prior markers should be cleaned up too.
        result = merge_mood_into_comment(
            'note [mood: stale]',
            self.MARKER,
        )
        self.assertEqual(result, 'note [MOOD: Happy; Energetic]')

    # --- empty marker (strip-only mode) -------------------------------------

    def test_empty_marker_strips_existing_marker(self):
        result = merge_mood_into_comment('comment [MOOD: Sad]', '')
        self.assertEqual(result, 'comment')

    def test_empty_marker_with_no_existing_marker(self):
        result = merge_mood_into_comment('comment', '')
        self.assertEqual(result, 'comment')

    def test_empty_marker_and_empty_existing(self):
        self.assertEqual(merge_mood_into_comment(None, ''), '')
        self.assertEqual(merge_mood_into_comment('', ''), '')

    # --- whitespace handling ------------------------------------------------

    def test_no_double_spaces_after_strip(self):
        # If the marker had spaces on both sides and we strip it, we mustn't
        # leave a double space behind.
        existing = 'pre  [MOOD: Sad]  post'
        result = merge_mood_into_comment(existing, self.MARKER)
        self.assertEqual(result, 'pre post [MOOD: Happy; Energetic]')

    def test_adjacent_markers_collapse_cleanly(self):
        # Pathological input but should still produce sane output.
        existing = '[MOOD: A][MOOD: B]'
        result = merge_mood_into_comment(existing, self.MARKER)
        self.assertEqual(result, self.MARKER)


class MoodMarkerRegexTests(unittest.TestCase):
    """Sanity checks on the regex itself."""

    def test_matches_basic_marker(self):
        self.assertIsNotNone(MOOD_MARKER_RE.search('[MOOD: x]'))

    def test_matches_with_semicolon_list(self):
        self.assertIsNotNone(
            MOOD_MARKER_RE.search('foo [MOOD: a; b; c] bar')
        )

    def test_matches_with_percentages(self):
        self.assertIsNotNone(
            MOOD_MARKER_RE.search('[MOOD: Happy 87%; Sad 12%]')
        )

    def test_does_not_match_bare_word(self):
        self.assertIsNone(MOOD_MARKER_RE.search('mood happy'))

    def test_does_not_match_unclosed_marker(self):
        # No closing bracket -- shouldn't eat the rest of the comment.
        self.assertIsNone(MOOD_MARKER_RE.search('[MOOD: x'))


if __name__ == '__main__':
    unittest.main()
