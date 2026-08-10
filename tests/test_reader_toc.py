"""Tests for the Reader's table-of-contents helpers (``md_llm.reader``).

These cover the pure text helpers: _toc_entries (heading parsing, fence
skipping), _normalize_heading (markdown markup -> plain text) and _toc_row_key
(per-level button keys). The clickable-widget part (render_toc) and the DOM
jump script are Streamlit / browser behaviour and are not unit-testable here.
"""

import unittest

from md_llm.reader import (
    _normalize_heading,
    _toc_auto_open,
    _toc_depths,
    _toc_entries,
    _toc_row_key,
    _toc_tree,
)


class TocEntriesTests(unittest.TestCase):
    def test_parses_levels_and_titles(self):
        md = "# One\n\n## Two\n\n### Three\n\n###### Six\n\nplain text\n"
        self.assertEqual(
            _toc_entries(md),
            [(1, "One"), (2, "Two"), (3, "Three"), (6, "Six")],
        )

    def test_skips_headings_inside_code_fences(self):
        md = (
            "# Real heading\n\n"
            "```md\n# Fake heading\n```\n\n"
            "## Another real one\n"
        )
        self.assertEqual(
            _toc_entries(md),
            [(1, "Real heading"), (2, "Another real one")],
        )

    def test_strips_setext_style_closing_hashes(self):
        md = "## My section ##\n"
        self.assertEqual(_toc_entries(md), [(2, "My section")])

    def test_heading_inside_blockquote_is_not_a_heading(self):
        md = "> # Not a heading\n\n# Real heading\n"
        self.assertEqual(_toc_entries(md), [(1, "Real heading")])

    def test_bare_hash_is_not_a_heading(self):
        md = "# not closed\n\n####\n\n## Fine\n"
        self.assertEqual(_toc_entries(md), [(1, "not closed"), (2, "Fine")])

    def test_empty_and_headingless_text(self):
        self.assertEqual(_toc_entries(""), [])
        self.assertEqual(_toc_entries("just paragraphs\n\nno headings\n"), [])


class NormalizeHeadingTests(unittest.TestCase):
    def test_strips_bold_italic_code_and_links(self):
        raw = "**Bold** and *it* and `code` and [link](https://x.test)"
        self.assertEqual(
            _normalize_heading(raw),
            "Bold and it and code and link",
        )

    def test_strips_inline_html(self):
        self.assertEqual(
            _normalize_heading("Keep <b>bold</b> text"), "Keep bold text"
        )

    def test_collapses_whitespace(self):
        self.assertEqual(_normalize_heading("a   b\t\n c"), "a b c")

    def test_leaves_plain_text_untouched(self):
        self.assertEqual(_normalize_heading("Plain 1:2 title"), "Plain 1:2 title")


class TocRowKeyTests(unittest.TestCase):
    def test_key_encodes_level_and_index(self):
        self.assertEqual(_toc_row_key(1, 0), "_reader_toc_l1_0")
        self.assertEqual(_toc_row_key(3, 12), "_reader_toc_l3_12")

    def test_deep_levels_share_the_cap_depth(self):
        self.assertEqual(_toc_row_key(6, 4), "_reader_toc_l5_4")
        self.assertEqual(_toc_row_key(4, 1), "_reader_toc_l4_1")

    def test_distinct_headings_get_distinct_keys(self):
        self.assertNotEqual(_toc_row_key(2, 3), _toc_row_key(3, 3))


class TocDepthTests(unittest.TestCase):
    def test_empty_entries(self):
        self.assertEqual(_toc_depths([]), [])

    def test_reroots_at_topmost_level(self):
        entries = [(2, "A"), (3, "B"), (3, "C"), (2, "D")]
        self.assertEqual(_toc_depths(entries), [1, 2, 2, 1])

    def test_level_1_docs_are_untouched(self):
        entries = [(1, "T"), (2, "A"), (2, "B"), (3, "C")]
        self.assertEqual(_toc_depths(entries), [1, 2, 2, 3])

    def test_gap_levels_keep_full_relative_depth(self):
        entries = [(1, "A"), (6, "B"), (7, "C")]
        self.assertEqual(_toc_depths(entries), [1, 6, 7])


class TocTreeTests(unittest.TestCase):
    def test_flat_doc_is_a_list_of_leaves(self):
        roots = _toc_tree([(1, "A"), (1, "B")])
        self.assertEqual([n["title"] for n in roots], ["A", "B"])
        self.assertTrue(all(n["children"] == [] for n in roots))
        self.assertEqual([n["id"] for n in roots], [0, 1])

    def test_nests_children_under_their_parent(self):
        roots = _toc_tree([(1, "A"), (2, "B"), (3, "C"), (1, "D")])
        a, d = roots
        self.assertEqual([n["title"] for n in a["children"]], ["B"])
        self.assertEqual([n["title"] for n in a["children"][0]["children"]], ["C"])
        self.assertEqual([n["title"] for n in d["children"]], [])

    def test_siblings_do_not_cross_into_each_other(self):
        roots = _toc_tree([(1, "A"), (2, "B"), (2, "C"), (1, "D"), (2, "E")])
        a, d = roots
        self.assertEqual(
            [n["title"] for n in a["children"]], ["B", "C"]
        )
        self.assertEqual([n["title"] for n in d["children"]], ["E"])

    def test_level_gaps_still_nest_by_relative_depth(self):
        roots = _toc_tree([(2, "A"), (6, "B"), (2, "C")])
        self.assertEqual([n["title"] for n in roots[0]["children"]], ["B"])

    def test_visit_order_matches_the_document(self):
        entries = [(1, "A"), (2, "B"), (3, "C"), (2, "D"), (1, "E"), (2, "F")]

        def walk(nodes):
            out = []
            for n in nodes:
                out.append(n["title"])
                out.extend(walk(n["children"]))
            return out

        self.assertEqual(walk(_toc_tree(entries)), [t for _, t in entries])


class TocAutoOpenTests(unittest.TestCase):
    def test_single_root_starts_open(self):
        roots = _toc_tree([(1, "T"), (2, "A"), (2, "B")])
        self.assertEqual(_toc_auto_open(roots), {0})

    def test_multi_root_starts_collapsed(self):
        roots = _toc_tree([(1, "A"), (1, "B")])
        self.assertEqual(_toc_auto_open(roots), set())

    def test_empty_doc(self):
        self.assertEqual(_toc_auto_open([]), set())


if __name__ == "__main__":
    unittest.main()