"""Tests for shared.utils.text_utils — text normalization + Telegram chunking."""

from shared.utils.text_utils import TELEGRAM_CHAR_LIMIT, chunk_telegram, normalize_whitespace


class TestChunkTelegram:
    def test_short_text_single_chunk_no_prefix(self):
        text = "a" * 3800
        result = chunk_telegram(text)
        assert len(result) == 1
        assert result[0] == text
        assert "(1/" not in result[0]

    def test_long_text_splits_into_multiple_chunks(self):
        text = "a" * 7700
        result = chunk_telegram(text)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= TELEGRAM_CHAR_LIMIT  # including prefix, must be ≤3800

    def test_multi_chunk_has_prefix(self):
        text = "a" * 7700
        result = chunk_telegram(text)
        assert result[0].startswith("(1/")
        assert result[-1].startswith(f"({len(result)}/")

    def test_empty_text(self):
        result = chunk_telegram("")
        assert result == [""]

    def test_exactly_3800_chars(self):
        text = "x" * 3800
        result = chunk_telegram(text)
        assert len(result) == 1

    def test_prefers_paragraph_boundaries(self):
        # 2 paragraphs, each ~2000 chars, total ~4004 (over limit)
        para1 = "a" * 2000
        para2 = "b" * 2000
        text = f"{para1}\n\n{para2}"
        result = chunk_telegram(text)
        assert len(result) == 2
        assert "a" * 100 in result[0]
        assert "b" * 100 in result[1]


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("hello   world") == "hello world"

    def test_collapses_newlines(self):
        assert normalize_whitespace("hello\n\n\nworld") == "hello world"

    def test_strips_edges(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_mixed_whitespace(self):
        assert normalize_whitespace("  a \t b \n c  ") == "a b c"
