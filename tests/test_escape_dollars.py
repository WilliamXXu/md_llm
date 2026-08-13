"""Tests for ``_escape_currency_dollars`` (``md_llm.state``).

Streamlit's ``st.markdown`` parses ``$...$`` as KaTeX math; a lone ``$`` in
prose (a dollar amount) gets mis-paired into a huge math span that garbles the
text. These cover the currency heuristic + code-span/fence protection.
"""

import unittest

from md_llm.state import _escape_currency_dollars


class EscapeCurrencyDollarsTests(unittest.TestCase):
    def test_escapes_dollar_followed_by_digit(self):
        # The original bug: $2.5B ... $1.5B ... $1B ... $6.6B ... $3.7B
        # were paired into math spans, garbling the prose between them.
        md = (
            "Harvard had to **borrow $2.5B** (Dec 2008) \u2014 $1.5B taxable + "
            "$1B tax-exempt bonds \u2014 to survive (from $6.6B to $3.7B)."
        )
        out = _escape_currency_dollars(md)
        # Every currency $ is escaped; no unescaped $-followed-by-digit remains.
        self.assertEqual(out.count(r"\$"), 5)
        self.assertEqual(_escape_currency_dollars(out), out)  # idempotent

    def test_renders_as_literal_dollar(self):
        # \$ is the markdown escape for a literal $.
        self.assertEqual(_escape_currency_dollars("$5"), r"\$5")

    def test_handles_commas_and_decimals(self):
        self.assertEqual(_escape_currency_dollars("$1,500.00"), r"\$1,500.00")

    def test_does_not_escape_math(self):
        # Genuine LaTeX math does not start with a bare digit.
        for expr in ("$x^2$", r"$\frac{a}{b}$", "$E=mc^2$", "$$x+y$$"):
            self.assertEqual(_escape_currency_dollars(expr), expr)

    def test_does_not_escape_dollar_without_trailing_digit(self):
        # A bare $ (e.g. a stray symbol) is left alone — no false positives.
        for s in ("$ ", "$", "cost $ each", "$word"):
            self.assertEqual(_escape_currency_dollars(s), s)

    def test_does_not_double_escape(self):
        self.assertEqual(_escape_currency_dollars(r"\$5"), r"\$5")

    def test_skips_inline_code_spans(self):
        # Inside `...` content is literal; a backslash would show verbatim.
        md = "see `$2.5B` in code, and $3 in prose"
        self.assertEqual(_escape_currency_dollars(md), "see `$2.5B` in code, and \\$3 in prose")

    def test_skips_fenced_code_blocks(self):
        md = "text $5\n```\n$5 not touched\n```\nmore $6\n"
        self.assertEqual(
            _escape_currency_dollars(md),
            "text \\$5\n```\n$5 not touched\n```\nmore \\$6\n",
        )

    def test_no_dollar_returns_unchanged(self):
        self.assertEqual(_escape_currency_dollars("no money here"), "no money here")

    def test_empty_and_none(self):
        self.assertEqual(_escape_currency_dollars(""), "")

    def test_full_sentence_round_trip(self):
        md = "Revenue was $2.5B in 2008 and $3.7B in 2010."
        out = _escape_currency_dollars(md)
        self.assertEqual(
            out, r"Revenue was \$2.5B in 2008 and \$3.7B in 2010."
        )


if __name__ == "__main__":
    unittest.main()
